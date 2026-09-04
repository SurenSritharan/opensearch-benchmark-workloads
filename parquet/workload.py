import json
import logging
import os
import resource
import subprocess
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
# Only sample_queries (up to num_queries dicts) and ground_truth live here --
# no large vector arrays.  Vectors are never held in RAM across the ingest pass.
BENCHMARK_STATE = {
    "sample_queries": [],
    "ground_truth": {},
}

_DEFAULT_QUERIES_FILE = "parquet-vectors-gt-{target_count}.json"


def _format_count(n: int) -> str:
    """Return a compact human-readable count string, e.g. 50000 -> '50k', 5000000 -> '5m'."""
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}m"
    if n >= 1_000 and n % 1_000 == 0:
        return f"{n // 1_000}k"
    return str(n)


# ============================================================================
# 1. BULK INGEST PARAMETER SOURCE
# ============================================================================


class ParquetBulkParamReader:
    """Streams documents from HuggingFace parquet files into OpenSearch.

    Pass 1 (ingest): vectors are sent to OpenSearch in bulk batches.
      - A sample of up to num_queries vectors is collected in memory for GT.
      - No large arrays are allocated; RSS stays flat throughout ingest.

    Pass 2 (GT): after all target docs are indexed, re-stream the locally
      cached parquet files and compute exact top-K neighbours for each query
      vector.  Each parquet file is processed in row-chunks of _CHUNK_SIZE so
      the peak score matrix is (num_queries x _CHUNK_SIZE) -- ~640 MB at
      10k queries / 16k rows -- constant regardless of corpus size.

    GCS cache: before computing GT, check whether a pre-built GT file exists
      in GCS.  If it does, download it and skip GT computation entirely.
      After a fresh computation, upload the result to GCS so subsequent runs
      are free.  Controlled by the ``gcs_gt_bucket`` workload param.
    """

    infinite = False  # finite -- stops after target_vector_count docs

    # Number of top neighbours to retrieve per query for ground truth.
    _K_GT                = 100
    # iter_batches batch size for the GT pass.
    # Peak RAM per batch = (num_queries + _GT_BATCH) * dim * 4B * 2 (Python + FAISS copy)
    # = (10k + 10k) * 1536 * 4B * 2 = ~240 MB -- well within the 48 GB pod limit.
    _GT_BATCH            = 10_000
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

        # GCS GT cache -- set gcs_gt_bucket in workload params to enable.
        # GT files are stored at gs://<bucket>/gt/<basename of queries_file>.
        self.gcs_gt_bucket = params.get("gcs_gt_bucket", "")

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
        p.gcs_gt_bucket = self.gcs_gt_bucket
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

    def _gcs_gt_object(self) -> str:
        """Return the full GCS URI for this dataset's GT file, or '' if disabled."""
        if not self.gcs_gt_bucket:
            return ""
        basename = os.path.basename(self.queries_file)
        bucket = self.gcs_gt_bucket.rstrip("/")
        return f"gs://{bucket}/gt/{basename}"

    def _download_gt_from_gcs(self) -> bool:
        """Try to download the GT file from GCS.  Returns True on success."""
        uri = self._gcs_gt_object()
        if not uri:
            return False
        try:
            result = subprocess.run(
                ["gsutil", "-q", "stat", uri],
                capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                logger.info(f"GCS GT cache miss: {uri}")
                return False
        except Exception as e:
            logger.warning(f"GCS stat failed ({uri}): {e} -- will recompute GT")
            return False

        logger.info(f"GCS GT cache hit -- downloading {uri}")
        os.makedirs(os.path.dirname(os.path.abspath(self.queries_file)), exist_ok=True)
        try:
            subprocess.run(
                ["gsutil", "-q", "cp", uri, self.queries_file],
                capture_output=True, check=True, timeout=300,
            )
            logger.info(f"Downloaded GT file from GCS to {self.queries_file!r}")
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"GCS download failed: {e.stderr.decode().strip()} -- will recompute GT")
            return False

    def _upload_gt_to_gcs(self) -> None:
        """Upload the freshly computed GT file to GCS for future runs."""
        uri = self._gcs_gt_object()
        if not uri:
            return
        try:
            subprocess.run(
                ["gsutil", "-q", "cp", self.queries_file, uri],
                capture_output=True, check=True, timeout=300,
            )
            logger.info(f"Uploaded GT file to GCS: {uri}")
        except Exception as e:
            logger.warning(f"GCS upload failed ({uri}): {e} -- GT file kept locally only")

    def _stream_and_index(self):
        # ------------------------------------------------------------------
        # GT file existence check -- three possible states:
        #   1. Local file exists (previous run on same PVC)  -> skip GT entirely
        #   2. GCS cache hit                                 -> download, skip GT
        #   3. Neither                                       -> compute after ingest
        # ------------------------------------------------------------------
        if os.path.exists(self.queries_file):
            logger.info(
                f"Ground truth file {self.queries_file!r} already exists -- "
                "skipping GT computation."
            )
            gt_exists = True
        elif self._download_gt_from_gcs():
            gt_exists = True
        else:
            gt_exists = False

        # Only partition 0 builds ground truth, covering the full corpus so
        # recall is computed against every indexed doc.
        build_gt = not gt_exists and self._partition_index == 0

        if build_gt:
            BENCHMARK_STATE["sample_queries"].clear()
            BENCHMARK_STATE["ground_truth"].clear()

        remote_files = sorted(self.fs.glob(f"{self.dataset_dir}/*.parquet"))
        global_doc_idx = 0

        # Detect available columns from the first file.
        first_file_path = self._get_local_cached_path(
            sorted(self.fs.glob(f"{self.dataset_dir}/*.parquet"))[0]
        )
        available_columns: Any = pq.read_schema(first_file_path).names
        has_title = "title" in available_columns
        has_text = "text" in available_columns
        id_col = "_id" if "_id" in available_columns else "id"
        columns = (
            [id_col]
            + (["title"] if has_title else [])
            + (["text"] if has_text else [])
            + [self.field_name]
        )

        _ingest_start  = time.monotonic()
        _last_progress = 0

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

                pending = []

                for i in range(len(id_list)):
                    if global_doc_idx >= self.target_docs:
                        break

                    doc_id = str(id_list[i])
                    title = data["title"][i] if has_title else ""
                    text = data["text"][i] if has_text else ""
                    raw_emb = data[self.field_name][i]

                    # Sample query vectors during ingest -- O(num_queries) RAM only.
                    # No large arrays; the full corpus is never held in memory.
                    if build_gt and len(BENCHMARK_STATE["sample_queries"]) < self.num_queries:
                        vector = np.array(raw_emb, dtype=np.float32)
                        norm = np.linalg.norm(vector)
                        if norm > 0:
                            vector /= norm
                        BENCHMARK_STATE["sample_queries"].append(
                            {
                                "query_id": doc_id,
                                "vector": vector.tolist(),
                                "text": text,
                                "title": title,
                            }
                        )

                    # Accumulate bulk actions for this partition's slice.
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

                    if global_doc_idx - _last_progress >= self._PROGRESS_EVERY_DOCS:
                        _last_progress = global_doc_idx
                        _elapsed = time.monotonic() - _ingest_start
                        _pct  = global_doc_idx / self.target_docs * 100
                        _rate = global_doc_idx / _elapsed if _elapsed > 0 else 0
                        _eta  = (self.target_docs - global_doc_idx) / _rate if _rate > 0 else 0
                        logger.info(
                            f"[ingest] {global_doc_idx:>9,}/{self.target_docs:,} "
                            f"({_pct:5.1f}%) "
                            f"| {_rate:,.0f} docs/s "
                            f"| elapsed {_elapsed/60:,.1f} min "
                            f"| eta {_eta/60:,.1f} min "
                            f"| rss {_rss_gb():.2f} GB"
                        )

                # Flush remainder from this PyArrow batch.
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
                        self._compute_ground_truth()
                    return

        if build_gt:
            self._compute_ground_truth()

    def _compute_ground_truth(self) -> None:
        """Compute exact nearest-neighbour ground truth by re-streaming the
        locally cached parquet files in fixed-size batches using FAISS
        IndexFlatIP for heap-based top-K search.

        Algorithm
        ---------
        Query vectors (num_queries x dim) are already in BENCHMARK_STATE from
        the ingest pass.  We iterate with iter_batches(batch_size=_GT_BATCH)
        so we never materialise more than _GT_BATCH rows at once regardless of
        how large the parquet row-groups are (the real HuggingFace files have
        one row-group of 500k rows -- reading the whole group at once would OOM).

        For each batch:
          1. to_pylist() + np.array() for the emb column (~batch_size rows).
          2. Normalise in-place.
          3. faiss.IndexFlatIP.add(batch_vectors) -- copies into C++ heap.
          4. del batch_vectors to free Python side immediately.
          5. idx.search(query_vectors, K) -- heap-based, O(M*K) extra RAM.
          6. del idx to free C++ heap.
          7. Merge into running top-K accumulators.

        Why iter_batches not read_row_group:
          HuggingFace parquet files have 1 row-group = 500k rows.
          read_row_group(0) would materialise all 500k × 1536 × 4B = 3 GB at
          once via to_pylist(), causing OOM.  iter_batches streams _GT_BATCH
          rows at a time regardless of physical row-group boundaries.

        Peak RAM per batch (_GT_BATCH=10k, dim=1536, 10k queries, K=100):
          query_matrix:   10k × 1536 × 4B =  60 MB  (constant)
          batch_vectors:  10k × 1536 × 4B =  60 MB  (freed before FAISS search)
          FAISS C++ copy: 10k × 1536 × 4B =  60 MB  (freed after search)
          FAISS heap:     10k × 100  × 4B =   4 MB
          Total:                          ~ 184 MB peak
        """
        queries = BENCHMARK_STATE["sample_queries"]
        if not queries:
            logger.warning("No sample queries collected -- skipping GT computation")
            return

        remote_files = sorted(self.fs.glob(f"{self.dataset_dir}/*.parquet"))
        total_files = len(remote_files)

        k_gt  = self._K_GT
        num_q = len(queries)

        query_vectors = np.array(
            [q["vector"] for q in queries], dtype=np.float32
        )  # (num_q, dim) -- always small, fits comfortably in RAM

        # Running top-K accumulators, one row per query.
        # Initialised to -inf / -1 so any real result beats them immediately.
        best_dists = np.full((num_q, k_gt), -np.inf, dtype=np.float32)
        best_pos   = np.full((num_q, k_gt), -1,      dtype=np.int64)

        # doc_ids accumulates one string ID per corpus row in ingest order.
        # ~50 MB for 5M short IDs -- always fits in RAM.
        doc_ids: list = []

        logger.info(
            f"Computing exact ground truth for {num_q:,} queries "
            f"over {self.target_docs:,} corpus vectors "
            f"({total_files} parquet files, FAISS IndexFlatIP, "
            f"batch_size={self._GT_BATCH:,}) ..."
        )
        gt_start = time.monotonic()

        available_columns: Any = pq.read_schema(
            self._get_local_cached_path(remote_files[0])
        ).names
        id_col = "_id" if "_id" in available_columns else "id"

        corpus_row_offset = 0  # absolute index of the next unprocessed corpus row
        batches_done = 0

        for file_idx, remote_path in enumerate(remote_files, 1):
            if corpus_row_offset >= self.target_docs:
                break

            local_path   = self._get_local_cached_path(remote_path)
            parquet_file = pq.ParquetFile(local_path)

            for batch in parquet_file.iter_batches(
                batch_size=self._GT_BATCH, columns=[id_col, self.field_name]
            ):
                if corpus_row_offset >= self.target_docs:
                    break

                batch_rows  = len(batch)
                rows_to_use = min(batch_rows, self.target_docs - corpus_row_offset)

                # Collect doc IDs for this batch.
                ids_col = batch.column(id_col).to_pylist()
                doc_ids.extend(str(x) for x in ids_col[:rows_to_use])

                # Deserialise embedding column into a float32 numpy array.
                # to_pylist() on _GT_BATCH rows is fast and memory-bounded.
                emb_col      = batch.column(self.field_name).to_pylist()
                batch_vectors = np.array(emb_col[:rows_to_use], dtype=np.float32)
                del batch, emb_col, ids_col  # free PyArrow memory immediately

                norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                batch_vectors /= norms  # normalise in-place

                # Build a temporary flat index over this batch.
                # FAISS copies batch_vectors into its C++ heap (~60 MB).
                idx_b = faiss.IndexFlatIP(self.dimension)
                idx_b.add(batch_vectors)
                del batch_vectors  # Python side freed; C++ copy lives in idx_b

                # Heap-based search -- O(num_q * k_gt) extra = 4 MB.
                k_this = min(k_gt, rows_to_use)
                dists, local_pos = idx_b.search(query_vectors, k_this)
                del idx_b  # free C++ heap immediately after search

                # Translate batch-relative positions to absolute corpus indices.
                abs_pos = local_pos.astype(np.int64) + corpus_row_offset
                abs_pos[local_pos == -1] = -1  # FAISS padding sentinel

                # Merge into running global top-K.
                merged_dists = np.concatenate([best_dists, dists],   axis=1)
                merged_pos   = np.concatenate([best_pos,   abs_pos], axis=1)
                order        = np.argsort(-merged_dists, axis=1)[:, :k_gt]
                best_dists   = np.take_along_axis(merged_dists, order, axis=1)
                best_pos     = np.take_along_axis(merged_pos,   order, axis=1)

                corpus_row_offset += rows_to_use
                batches_done += 1

            elapsed = time.monotonic() - gt_start
            rate = corpus_row_offset / elapsed if elapsed > 0 else 0
            eta  = (self.target_docs - corpus_row_offset) / rate if rate > 0 else 0
            logger.info(
                f"  [gt] file {file_idx}/{total_files} -- "
                f"{corpus_row_offset:,}/{self.target_docs:,} rows "
                f"({batches_done} batches) "
                f"| {_rss_gb():.2f} GB rss "
                f"| elapsed {elapsed/60:.1f} min "
                f"| eta {eta/60:.1f} min"
            )

        # Resolve corpus row indices to doc-id strings.
        for i, q in enumerate(queries):
            gt_doc_ids = [
                doc_ids[pos]
                for pos in best_pos[i]
                if pos != -1 and pos < len(doc_ids)
            ]
            BENCHMARK_STATE["ground_truth"][q["query_id"]] = gt_doc_ids

        # Persist to disk.
        os.makedirs(os.path.dirname(os.path.abspath(self.queries_file)), exist_ok=True)
        payload = {
            "queries": BENCHMARK_STATE["sample_queries"],
            "ground_truth": BENCHMARK_STATE["ground_truth"],
        }
        with open(self.queries_file, "w") as f:
            json.dump(payload, f)
        elapsed_total = time.monotonic() - gt_start
        logger.info(
            f"Saved {len(queries):,} queries and ground truth to {self.queries_file!r} "
            f"in {elapsed_total/60:.1f} min"
        )

        # Upload to GCS so future runs skip GT computation entirely.
        self._upload_gt_to_gcs()

        # Release shared state -- no longer needed now the file is written.
        BENCHMARK_STATE["sample_queries"].clear()
        BENCHMARK_STATE["ground_truth"].clear()

    def params(self):
        """Called by OSB each iteration to get the next bulk request dict."""
        if self._generator is None:
            self._start_doc = 0
            self._end_doc = self.target_docs
            self._generator = self._stream_and_index()
        return next(self._generator)


# ============================================================================
# 2. VECTOR SEARCH PARAMETER SOURCE
# ============================================================================


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
    Ground truth is computed by exact numpy nearest-neighbour search against
    the full indexed corpus.  For hybrid mode this means recall measures how
    closely the combined BM25+kNN ranking agrees with exact kNN -- useful as
    a quality/tradeoff metric but not a true hybrid ground truth.
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
        """Return a copy that owns a strided slice of the query list."""
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
        """Pure kNN query."""
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
        """Hybrid BM25 + kNN query routed through a normalization search pipeline."""
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
