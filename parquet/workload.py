import json
import logging
import os
import resource
import tempfile
import time
from typing import Any
import faiss
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

logger = logging.getLogger(__name__)


def _rss_gb() -> float:
    """Return current process RSS in GB.

    Uses /proc/self/status on Linux (current RSS) and resource.getrusage
    on other platforms (peak RSS -- the best available without psutil).
    """
    try:
        with open("/proc/self/status") as _f:
            for _line in _f:
                if _line.startswith("VmRSS:"):
                    return int(_line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    import platform
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024 ** 2 if platform.system() == "Linux" else kb / 1024 ** 3


# Global state shared between indexing parameter generator and query ground truth builder.
# faiss_index is intentionally absent -- vectors are streamed to a memmap file instead of
# held in RAM.  Only sample_queries (1000 dicts) and ground_truth live here.
BENCHMARK_STATE = {
    "sample_queries": [],
    "ground_truth": {},
}

_DEFAULT_QUERIES_FILE = "parquet-vectors-gt-{target_count}.json"


def _format_count(n: int) -> str:
    """Return a compact human-readable label for a vector count.

    Examples: 500_000 -> '500k', 1_000_000 -> '1m', 50_000 -> '50k'.
    """
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}m"
    if n >= 1_000 and n % 1_000 == 0:
        return f"{n // 1_000}k"
    return str(n)


# ============================================================================
# 1. PARAMETER READERS (Generates Data for Operations)
# ============================================================================


class ParquetBulkParamReader:
    """Streams PyArrow Parquet batches, builds the FAISS ground-truth index,

    and yields bulk indexing requests to OpenSearch Benchmark.

    Memory model
    ------------
    Vectors are written row-by-row to a numpy.memmap file on disk during the
    single streaming pass so RAM usage stays O(batch_size) regardless of corpus
    size.  After streaming completes, _compute_ground_truth opens the memmap
    read-only and searches it in fixed-size row-batches -- peak RAM during
    ground-truth computation is one search batch (~100 MB for _SEARCH_BATCH_SIZE
    rows at 1536 dims).  The memmap and doc-id sidecar files are deleted when
    computation finishes, or immediately on any exception / cancellation via the
    try/finally guard in _stream_and_index.
    """

    infinite = False  # finite -- stops after target_vector_count docs

    # Number of vectors per FAISS search batch during ground-truth computation.
    # At 1536 dims / float32 each row is 6 144 bytes.
    # 16 384 rows ~= 100 MB -- comfortably fits alongside the query matrix.
    _SEARCH_BATCH_SIZE   = 16_384
    _PROGRESS_EVERY_DOCS = 100_000  # emit a progress log line every N docs

    def __init__(self, workload, params):
        self.workload = workload
        self._raw_params = params
        self.batch_size = params.get("batch_size", 5000)   # PyArrow rows per read
        self.bulk_size = params.get("bulk_size", 1000)     # docs per bulk request
        self.target_docs = params.get("target_vector_count", 50000)
        self.dimension = params.get("target_index_dimension", 1024)
        self.field_name = params.get("target_field_name", "emb")
        default_index_name = f"parquet-vectors-{_format_count(self.target_docs)}"
        self.index_name = params.get("index", default_index_name)
        self.num_queries = params.get("num_queries", 1000)
        default_queries_file = _DEFAULT_QUERIES_FILE.format(
            target_count=self.target_docs
        )
        self.queries_file = params.get("queries_file", default_queries_file)

        self.fs = HfFileSystem()
        self.repo_id = params.get(
            "hf_repo_id",
            "CohereLabs/wikipedia-2023-11-embed-multilingual-v3",
        )
        self.dataset_dir = params.get(
            "hf_dataset_dir",
            "datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3/en",
        )

        self._partition_index = 0
        self._total_partitions = 1
        self._generator = None  # created lazily after partition() is called

    def partition(self, partition_index, total_partitions):
        """Return a new reader covering only this client's slice of target_docs."""
        p = ParquetBulkParamReader.__new__(ParquetBulkParamReader)
        p.workload = self.workload
        p._raw_params = self._raw_params
        p.batch_size = self.batch_size
        p.bulk_size = self.bulk_size
        p.target_docs = self.target_docs
        p.dimension = self.dimension
        p.field_name = self.field_name
        p.index_name = self.index_name
        p.num_queries = self.num_queries
        p.queries_file = self.queries_file
        p.fs = self.fs
        p.repo_id = self.repo_id
        p.dataset_dir = self.dataset_dir
        p._partition_index = partition_index
        p._total_partitions = total_partitions
        # Divide the total doc count evenly; last partition absorbs the remainder.
        docs_per_partition = self.target_docs // total_partitions
        p._start_doc = partition_index * docs_per_partition
        p._end_doc = (
            p._start_doc + docs_per_partition
            if partition_index < total_partitions - 1
            else self.target_docs
        )
        p._generator = p._stream_and_index()
        return p

    def _get_local_cached_path(self, remote_file_path: str) -> str:
        relative_path = "/".join(remote_file_path.split("/")[-2:])
        return hf_hub_download(
            repo_id=self.repo_id,
            filename=relative_path,
            repo_type="dataset",
        )

    def _stream_and_index(self):
        if os.path.exists(self.queries_file):
            logger.info(
                f"Ground truth file {self.queries_file!r} already exists -- "
                "skipping FAISS index build and ground truth computation."
            )
            gt_exists = True
        else:
            gt_exists = False

        # Only partition 0 builds the FAISS ground-truth index, covering all
        # target_docs so recall is computed against the full indexed corpus.
        # All partitions index their own slice into OpenSearch.
        build_gt = not gt_exists and self._partition_index == 0

        # Reset shared query/ground-truth state for this run so retries do not
        # accumulate duplicate entries from a previous attempt.
        if build_gt:
            BENCHMARK_STATE["sample_queries"].clear()
            BENCHMARK_STATE["ground_truth"].clear()

        remote_files = sorted(self.fs.glob(f"{self.dataset_dir}/*.parquet"))
        # global_doc_idx counts every doc across all files (0-based) so we can
        # honour each partition's [_start_doc, _end_doc) ingest window.
        global_doc_idx = 0
        # Only request columns that actually exist in this parquet file.
        # title and text are optional -- datasets like openai-large-5m have
        # only id and emb, so we skip missing columns rather than crashing.
        first_file_path = self._get_local_cached_path(sorted(self.fs.glob(f"{self.dataset_dir}/*.parquet"))[0])
        available_columns: Any = pq.read_schema(first_file_path).names
        has_title = "title" in available_columns
        has_text = "text" in available_columns
        id_col = "_id" if "_id" in available_columns else "id"
        columns = [id_col] + (["title"] if has_title else []) + (["text"] if has_text else []) + [self.field_name]

        # ------------------------------------------------------------------
        # Ground-truth streaming state (partition 0 only)
        #
        # Vectors are written to a numpy.memmap file instead of an in-memory
        # FAISS index so peak RAM stays O(batch_size) during the ingest pass.
        # The try/finally below guarantees the temp files are removed on any
        # exit path -- normal completion, exception, or generator close/cancel.
        # ------------------------------------------------------------------
        memmap_file = None    # numpy.memmap handle (write mode during ingest)
        memmap_path = None    # str path -- kept for deletion after GT computation
        docid_path  = None    # str path to newline-delimited doc-id sidecar
        docid_file  = None    # open file handle -- kept open for the streaming pass

        if build_gt:
            gt_dir = os.path.dirname(os.path.abspath(self.queries_file)) or "."
            os.makedirs(gt_dir, exist_ok=True)
            tmp = tempfile.NamedTemporaryFile(
                suffix=".bin", delete=False,
                dir=gt_dir,
            )
            memmap_path = tmp.name
            tmp.close()
            docid_path = memmap_path + ".ids"
            # Open the doc-id sidecar once and hold it for the entire streaming
            # pass -- avoids one open()/close() syscall per vector row.
            docid_file = open(docid_path, "w")
            # Pre-allocate the full memmap so the OS can use sparse allocation.
            memmap_file = np.memmap(
                memmap_path, dtype="float32", mode="w+",
                shape=(self.target_docs, self.dimension),
            )
            logger.info(
                f"Allocated memmap for {self.target_docs:,} x {self.dimension}-dim "
                f"vectors at {memmap_path!r} "
                f"({self.target_docs * self.dimension * 4 / 1024**3:.1f} GB on disk)"
            )

        gt_vec_count = 0  # rows written to memmap so far
        _ingest_start  = time.monotonic()
        _last_progress = 0  # global_doc_idx at the last progress log

        try:
            for remote_path in remote_files:
                local_path = self._get_local_cached_path(remote_path)
                parquet_file = pq.ParquetFile(local_path)

                for batch in parquet_file.iter_batches(
                    batch_size=self.batch_size, columns=columns
                ):
                    data = batch.to_pydict()
                    id_list = data.get("_id") or data.get("id")
                    if id_list is None:
                        raise ValueError(
                            f"Parquet file {remote_path!r} has neither '_id' nor 'id' column"
                        )

                    # Accumulate bulk actions for this partition's slice, then
                    # flush in bulk_size chunks so PyArrow IO and OS bulk request
                    # sizes are tuned independently.
                    pending = []

                    for i in range(len(id_list)):
                        # Stop once all target docs have been seen.
                        if global_doc_idx >= self.target_docs:
                            break

                        doc_id = str(id_list[i])
                        title = data["title"][i] if has_title else ""
                        text = data["text"][i] if has_text else ""
                        raw_emb = data[self.field_name][i]

                        if build_gt:
                            # Normalize vector for cosine similarity (IndexFlatIP).
                            vector = np.array(raw_emb, dtype=np.float32)
                            norm = np.linalg.norm(vector)
                            if norm > 0:
                                vector /= norm

                            # Write directly to the pre-allocated memmap row.
                            # No Python list grows -- RAM stays flat.
                            memmap_file[gt_vec_count] = vector
                            gt_vec_count += 1

                            # Append doc id to sidecar (held open for the full pass).
                            docid_file.write(doc_id + "\n")

                            # Sample queries from across the full corpus.
                            if len(BENCHMARK_STATE["sample_queries"]) < self.num_queries:
                                BENCHMARK_STATE["sample_queries"].append(
                                    {
                                        "query_id": doc_id,
                                        "vector": vector.tolist(),
                                        "text": text,
                                        "title": title,
                                    }
                                )

                        # Only accumulate bulk actions for this partition's slice
                        if self._start_doc <= global_doc_idx < self._end_doc:
                            pending.append(
                                {"index": {"_index": self.index_name, "_id": doc_id}}
                            )
                            doc_body = {self.field_name: raw_emb}
                            if has_title:
                                doc_body["title"] = title
                            if has_text:
                                doc_body["text"] = text
                            pending.append(doc_body)

                            # Flush a full bulk_size chunk as soon as it's ready
                            if len(pending) // 2 >= self.bulk_size:
                                yield {
                                    "body": pending,
                                    "bulk-size": len(pending) // 2,
                                    "unit": "docs",
                                    "action-metadata-present": True,
                                    "index": self.index_name,
                                }
                                pending = []

                        global_doc_idx += 1

                        # Emit a progress line every _PROGRESS_EVERY_DOCS docs.
                        if global_doc_idx - _last_progress >= self._PROGRESS_EVERY_DOCS:
                            _last_progress = global_doc_idx
                            _elapsed = time.monotonic() - _ingest_start
                            _pct     = global_doc_idx / self.target_docs * 100
                            _rate    = global_doc_idx / _elapsed if _elapsed > 0 else 0
                            _eta     = (self.target_docs - global_doc_idx) / _rate if _rate > 0 else 0
                            _disk_gb = (
                                os.path.getsize(memmap_path) / 1024 ** 3
                                if memmap_path and os.path.exists(memmap_path)
                                else 0.0
                            )
                            logger.info(
                                f"[ingest] {global_doc_idx:>9,}/{self.target_docs:,} "
                                f"({_pct:5.1f}%) "
                                f"| {_rate:,.0f} docs/s "
                                f"| elapsed {_elapsed/60:,.1f} min "
                                f"| eta {_eta/60:,.1f} min "
                                f"| rss {_rss_gb():.2f} GB "
                                f"| memmap {_disk_gb:.2f} GB"
                            )

                    # Flush any remainder from this Parquet batch
                    if pending:
                        yield {
                            "body": pending,
                            "bulk-size": len(pending) // 2,
                            "unit": "docs",
                            "action-metadata-present": True,
                            "index": self.index_name,
                        }

                    if global_doc_idx >= self.target_docs:
                        if build_gt:
                            # Flush both buffers before searching.
                            memmap_file.flush()
                            docid_file.close()
                            self._compute_ground_truth(
                                memmap_file, memmap_path, docid_path, gt_vec_count
                            )
                        return

            if build_gt:
                memmap_file.flush()
                docid_file.close()
                self._compute_ground_truth(
                    memmap_file, memmap_path, docid_path, gt_vec_count
                )

        finally:
            # Guarantee temp files are removed on every exit path:
            #   Normal completion  -- _compute_ground_truth already deleted them;
            #                         os.path.exists guards make this a no-op.
            #   Exception          -- files are cleaned up before propagating.
            #   GeneratorExit/cancel -- OSB closes the generator; finally still runs.
            if memmap_file is not None:
                try:
                    del memmap_file   # flush dirty pages and release the mmap
                except Exception:
                    pass
            if docid_file is not None and not docid_file.closed:
                try:
                    docid_file.close()
                except Exception:
                    pass
            for _tmp_path in (memmap_path, docid_path):
                if _tmp_path and os.path.exists(_tmp_path):
                    try:
                        os.unlink(_tmp_path)
                        logger.debug(f"Cleaned up temp file {_tmp_path!r}")
                    except OSError as _e:
                        logger.warning(f"Could not remove temp file {_tmp_path!r}: {_e}")

    def _compute_ground_truth(
        self,
        memmap_file: np.memmap,
        memmap_path: str,
        docid_path: str,
        total_vectors: int,
    ) -> None:
        """Compute exact nearest-neighbour ground truth by searching the memmap
        in fixed-size row-batches so peak RAM is O(_SEARCH_BATCH_SIZE) rather
        than O(total_vectors).

        Algorithm
        ---------
        We want exact top-100 neighbours for each of the num_queries query
        vectors against the full corpus of total_vectors indexed vectors.
        A single FAISS IndexFlatIP.search() call over all rows would require
        loading all rows into RAM simultaneously (same as the old approach).

        Instead we maintain a running top-K heap per query:
          1. Open the memmap read-only (OS page-cache handles I/O).
          2. Slice rows in chunks of _SEARCH_BATCH_SIZE.
          3. For each chunk build a temporary IndexFlatIP over just those rows
             and search with k=min(100, chunk_size).
          4. Merge the chunk results into a per-query heap, keeping only the
             global top-100 distances and their original corpus positions.
          5. After all chunks are processed resolve corpus positions to doc IDs
             via the sidecar file, write the queries file, and delete temp files.

        Peak RAM: query_matrix (num_queries x dim x 4 B)
                + one chunk   (_SEARCH_BATCH_SIZE x dim x 4 B)
                + heap arrays (num_queries x 100 x 8 B)
                ~= 25 MB + 100 MB + <1 MB  for the default parameters.
        """
        logger.info(
            f"Computing exact FAISS ground truth for {total_vectors:,} vectors "
            f"in batches of {self._SEARCH_BATCH_SIZE:,} rows ..."
        )
        queries = BENCHMARK_STATE["sample_queries"]
        if not queries:
            return

        k_gt = min(100, total_vectors)
        num_q = len(queries)

        query_vectors = np.array(
            [q["vector"] for q in queries], dtype=np.float32
        )  # shape: (num_q, dim) -- always small, fits in RAM

        # Running top-K accumulators -- one row per query.
        # best_dists: (num_q, k_gt)  largest inner-products seen so far
        # best_pos:   (num_q, k_gt)  corresponding corpus row indices
        # Initialised to -inf / -1 so any real result beats them immediately.
        best_dists = np.full((num_q, k_gt), -np.inf, dtype=np.float64)
        best_pos   = np.full((num_q, k_gt), -1,      dtype=np.int64)

        # Re-open the memmap read-only so the OS can page it in on demand.
        corpus = np.memmap(
            memmap_path, dtype="float32", mode="r",
            shape=(total_vectors, self.dimension),
        )

        chunk_start = 0
        while chunk_start < total_vectors:
            chunk_end  = min(chunk_start + self._SEARCH_BATCH_SIZE, total_vectors)
            chunk_size = chunk_end - chunk_start

            # Load only this chunk into a temporary in-memory array for FAISS.
            chunk = np.array(corpus[chunk_start:chunk_end], dtype=np.float32)

            idx_chunk = faiss.IndexFlatIP(self.dimension)
            idx_chunk.add(chunk)
            del chunk  # free immediately after FAISS copies the data

            k_this = min(k_gt, chunk_size)
            dists, local_pos = idx_chunk.search(query_vectors, k_this)
            # local_pos is relative to chunk_start; translate to corpus-absolute.
            abs_pos = local_pos + chunk_start  # (num_q, k_this)

            # Merge chunk results into the running top-K.
            # Concatenate existing best with chunk results, re-sort, keep top-k_gt.
            merged_dists = np.concatenate([best_dists, dists.astype(np.float64)], axis=1)
            merged_pos   = np.concatenate([best_pos,   abs_pos],                  axis=1)

            # argsort descending by distance for each query row.
            order      = np.argsort(-merged_dists, axis=1)[:, :k_gt]
            best_dists = np.take_along_axis(merged_dists, order, axis=1)
            best_pos   = np.take_along_axis(merged_pos,   order, axis=1)

            chunk_start = chunk_end
            if chunk_start % (self._SEARCH_BATCH_SIZE * 10) == 0 or chunk_start == total_vectors:
                logger.info(
                    f"  ground-truth search: {chunk_start:,}/{total_vectors:,} rows processed"
                )

        del corpus  # release the read-only memmap view

        # Resolve corpus row indices to doc-id strings via the sidecar file.
        # Read the entire sidecar into a list -- it contains at most target_docs
        # short strings (one per line) so it is always small (~50 MB for 5M IDs).
        with open(docid_path) as fid:
            all_doc_ids = [line.rstrip("\n") for line in fid]

        for i, q in enumerate(queries):
            gt_doc_ids = [
                all_doc_ids[pos]
                for pos in best_pos[i]
                if pos != -1 and pos < len(all_doc_ids)
            ]
            BENCHMARK_STATE["ground_truth"][q["query_id"]] = gt_doc_ids

        # Persist to disk so vector-search can load them without re-running ingest.
        payload = {
            "queries": BENCHMARK_STATE["sample_queries"],
            "ground_truth": BENCHMARK_STATE["ground_truth"],
        }
        with open(self.queries_file, "w") as f:
            json.dump(payload, f)
        logger.info(
            f"Saved {len(queries)} queries and ground truth to {self.queries_file}"
        )

        # Release shared state -- no longer needed now the file is written.
        BENCHMARK_STATE["sample_queries"].clear()
        BENCHMARK_STATE["ground_truth"].clear()

        # Delete the temporary memmap and doc-id sidecar files.
        # The try/finally in _stream_and_index provides a second safety net,
        # but deleting here on the happy path means the finally block is a no-op.
        for path in (memmap_path, docid_path):
            try:
                os.unlink(path)
                logger.info(f"Deleted temporary file {path!r}")
            except OSError as exc:
                logger.warning(f"Could not delete temporary file {path!r}: {exc}")

    def params(self):
        """Called by OSB each iteration to get the next bulk request dict."""
        # Lazily initialise the generator for the single-client (no partition()) path.
        if self._generator is None:
            self._start_doc = 0
            self._end_doc = self.target_docs
            self._generator = self._stream_and_index()
        return next(self._generator)


class VectorSearchParamReader:
    """Provides vector search query payloads to search runners.

    When running as a standalone procedure (i.e. without bulk-ingest-data having
    run in the same process), loads queries and ground truth from the file written
    by ParquetBulkParamReader._compute_ground_truth.  The file path must match the
    one used during ingest -- both default to _DEFAULT_QUERIES_FILE and can be
    overridden via the "queries_file" workload parameter.

    Hybrid search mode
    ------------------
    Set search_mode: "hybrid" in workload params to issue a hybrid query
    (BM25 match on the text field combined with knn) via a normalization-processor
    search pipeline instead of a pure kNN query.

    Required additional params:
      search_pipeline   -- name of the search pipeline to use
                           (default: "hybrid-search-pipeline")
      hybrid_text_field -- index field to run the BM25 leg against
                           (default: "text")
      bm25_weight       -- arithmetic-mean weight for the BM25 score  (default: 0.3)
      knn_weight        -- arithmetic-mean weight for the kNN score    (default: 0.7)

    Recall notes
    ------------
    Ground truth is always computed by FAISS against pure vector similarity.
    For hybrid mode this means recall measures how closely the combined
    BM25+kNN ranking agrees with exact kNN -- useful as a quality/tradeoff
    metric but not a true hybrid ground truth.
    """

    infinite = True  # cycles through queries indefinitely for time-period-based search

    def __init__(self, workload, params):
        self._params = params
        target_docs = params.get("target_vector_count", 50000)
        default_index_name = f"parquet-vectors-{_format_count(target_docs)}"
        self.index_name = params.get("index", default_index_name)
        self.k = params.get("k", 10)
        self.ef_search = params.get("ef_search", 64)
        self.field_name = params.get("target_field_name", "emb")
        self.operation_type = params.get("operation-type", "vector-search")
        self.search_mode = params.get("search_mode", "vector")   # "vector" | "hybrid"
        self.search_pipeline = params.get("search_pipeline", "hybrid-search-pipeline")
        self.hybrid_text_field = params.get("hybrid_text_field", "text")
        self._cursor = 0
        self._query_indices = None  # set by partition(); None means round-robin

        # If ingest did not run in this process, load state from disk.
        if not BENCHMARK_STATE["sample_queries"]:
            default_queries_file = _DEFAULT_QUERIES_FILE.format(
                target_count=target_docs
            )
            queries_file = params.get("queries_file", default_queries_file)
            if not os.path.exists(queries_file):
                raise FileNotFoundError(
                    f"No queries found in memory and queries file not found: {queries_file!r}. "
                    "Run bulk-ingest-data first to generate the queries and ground truth file."
                )
            logger.info(f"Loading queries and ground truth from {queries_file}")
            with open(queries_file) as f:
                payload = json.load(f)
            BENCHMARK_STATE["sample_queries"] = payload["queries"]
            BENCHMARK_STATE["ground_truth"] = payload["ground_truth"]
            logger.info(
                f"Loaded {len(BENCHMARK_STATE['sample_queries'])} queries "
                f"and {len(BENCHMARK_STATE['ground_truth'])} ground truth entries"
            )

    def partition(self, partition_index, total_partitions):
        """Return a copy that owns a strided slice of the query list.

        Partition i takes indices i, i+N, i+2N, ... so every query is issued by
        exactly one client and no shuffling or RNG is needed.
        """
        p = VectorSearchParamReader.__new__(VectorSearchParamReader)
        p._params = self._params
        p.index_name = self.index_name
        p.k = self.k
        p.ef_search = self.ef_search
        p.field_name = self.field_name
        p.operation_type = self.operation_type
        p.search_mode = self.search_mode
        p.search_pipeline = self.search_pipeline
        p.hybrid_text_field = self.hybrid_text_field
        queries = BENCHMARK_STATE["sample_queries"]
        p._query_indices = list(range(partition_index, len(queries), total_partitions))
        p._cursor = 0
        return p

    def params(self):
        """Called by OSB each iteration to get the next search request dict."""
        queries = BENCHMARK_STATE["sample_queries"]
        if not queries:
            raise StopIteration("No queries available in state.")

        # Use strided indices when partitioned, otherwise plain round-robin.
        if self._query_indices is not None:
            idx = self._query_indices[self._cursor % len(self._query_indices)]
        else:
            idx = self._cursor % len(queries)
        self._cursor += 1
        query_item = queries[idx]

        if self.search_mode == "hybrid":
            query_payload = self._build_hybrid_query(query_item)
        else:
            query_payload = self._build_vector_query(query_item)

        if self._cursor == 1:
            logger.info(
                "[search-mode] %s | index=%s k=%s ef_search=%s%s",
                self.search_mode,
                self.index_name,
                self.k,
                self.ef_search,
                f" pipeline={self.search_pipeline} text_field={self.hybrid_text_field}"
                if self.search_mode == "hybrid" else "",
            )

        # Ground truth is always the exact FAISS kNN neighbors -- sliced to k.
        # In hybrid mode this measures agreement with pure vector ranking.
        gt_ids = BENCHMARK_STATE["ground_truth"].get(query_item["query_id"], [])
        neighbors = gt_ids[: self.k]

        result = {
            "operation-type": self.operation_type,
            "index": self.index_name,
            "body": query_payload,
            "query_id": query_item["query_id"],
            "k": self.k,
            "neighbors": neighbors,
            "detailed-results": self._params.get("detailed-results", True),
        }
        if self.search_mode == "hybrid":
            result["search_pipeline"] = self.search_pipeline
        return result

    def _build_vector_query(self, query_item):
        """Pure kNN query -- original behaviour."""
        knn_clause = {
            "vector": query_item["vector"],
            "k": self.k,
        }
        if self.ef_search:
            knn_clause["method_parameters"] = {"ef_search": self.ef_search}
        return {
            "size": self.k,
            "query": {"knn": {self.field_name: knn_clause}},
        }

    def _build_hybrid_query(self, query_item):
        """Hybrid BM25 + kNN query routed through a normalization search pipeline.

        The query text is the document's own text field value sampled
        during ingest -- a valid BM25 query because Wikipedia passages are
        self-describing and the text was chosen to be representative.
        """
        query_text = query_item.get("text", query_item.get("title", ""))
        knn_clause = {
            "vector": query_item["vector"],
            "k": self.k,
        }
        if self.ef_search:
            knn_clause["method_parameters"] = {"ef_search": self.ef_search}
        return {
            "size": self.k,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "match": {
                                self.hybrid_text_field: {
                                    "query": query_text
                                }
                            }
                        },
                        {"knn": {self.field_name: knn_clause}},
                    ]
                }
            },
        }


# ============================================================================
# 3. HOOK REGISTRATIONS
# ============================================================================


def register(registry):
    registry.register_param_source(
        "parquet-bulk-reader", ParquetBulkParamReader
    )
    registry.register_param_source(
        "vector-search-reader", VectorSearchParamReader
    )
