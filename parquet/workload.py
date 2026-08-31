import json
import logging
import os
from typing import Any
import faiss
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

logger = logging.getLogger(__name__)

# Global state shared between indexing parameter generator and query ground truth builder
# faiss_index is None until ParquetBulkParamReader initialises it with the correct dimension.
BENCHMARK_STATE = {
    "faiss_index": None,
    "indexed_doc_ids": [],
    "sample_queries": [],
    "ground_truth": {},
}

_DEFAULT_QUERIES_FILE = "parquet-vectors-gt-{target_count}.json"


def _format_count(n: int) -> str:
    """Return a compact human-readable label for a vector count.

    Examples: 500_000 → '500k', 1_000_000 → '1m', 50_000 → '50k'.
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
    """

    infinite = False  # finite — stops after target_vector_count docs

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

        # Initialise (or re-initialise) the FAISS index with the correct dimension.
        # Only partition 0 does this; single-client path initialises here too.
        if BENCHMARK_STATE["faiss_index"] is None:
            BENCHMARK_STATE["faiss_index"] = faiss.IndexFlatIP(self.dimension)

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
                f"Ground truth file {self.queries_file!r} already exists — "
                "skipping FAISS index build and ground truth computation."
            )
            gt_exists = True
        else:
            gt_exists = False

        # Only partition 0 builds the FAISS ground-truth index, covering all
        # target_docs so recall is computed against the full indexed corpus.
        # All partitions index their own slice into OpenSearch.
        build_gt = not gt_exists and self._partition_index == 0

        remote_files = sorted(self.fs.glob(f"{self.dataset_dir}/*.parquet"))
        # global_doc_idx counts every doc across all files (0-based) so we can
        # honour each partition's [_start_doc, _end_doc) ingest window.
        global_doc_idx = 0
        # Only request columns that actually exist in this parquet file.
        # title and text are optional — datasets like openai-large-5m have
        # only id and emb, so we skip missing columns rather than crashing.
        first_file_path = self._get_local_cached_path(sorted(self.fs.glob(f"{self.dataset_dir}/*.parquet"))[0])
        available_columns: Any = pq.read_schema(first_file_path).names
        has_title = "title" in available_columns
        has_text = "text" in available_columns
        id_col = "_id" if "_id" in available_columns else "id"
        columns = [id_col] + (["title"] if has_title else []) + (["text"] if has_text else []) + [self.field_name]

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
                        # Normalize vector for FAISS IndexFlatIP (cosine similarity)
                        vector = np.array(raw_emb, dtype=np.float32)
                        norm = np.linalg.norm(vector)
                        if norm > 0:
                            vector = vector / norm

                        # Build FAISS in-memory index over all indexed documents
                        BENCHMARK_STATE["faiss_index"].add(
                            np.expand_dims(vector, axis=0)
                        )
                        BENCHMARK_STATE["indexed_doc_ids"].append(doc_id)

                        # Sample queries from across the full corpus.
                        # text and title are stored so hybrid search can use
                        # them as the BM25 leg of a hybrid query.
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
                        self._compute_ground_truth()
                    return

        if build_gt:
            self._compute_ground_truth()

    def _compute_ground_truth(self):
        """Runs FAISS search over loaded dataset to establish exact ground-truth nearest neighbors.

        Saves queries and ground truth to self.queries_file so the vector-search
        procedure can load them when it runs in a separate OSB invocation.
        Skips computation if the file already exists (checked by caller).
        """
        logger.info("Computing exact FAISS Ground Truth for evaluation...")
        queries = BENCHMARK_STATE["sample_queries"]
        if not queries:
            return

        query_vectors = np.array(
            [q["vector"] for q in queries], dtype=np.float32
        )
        # Search for top-100 so every k value (10, 64, 100…) is covered by the
        # same ground-truth set — matching the msmarco approach.
        k_gt = min(100, BENCHMARK_STATE["faiss_index"].ntotal)
        distances, indices = BENCHMARK_STATE["faiss_index"].search(
            query_vectors, k_gt
        )

        for i, q in enumerate(queries):
            # Store as a flat list of doc-id strings, ordered by descending
            # similarity — same shape as msmarco's ground_truth_ids list so
            # VectorSearchParamReader can pass it directly as "neighbors".
            gt_doc_ids = [
                BENCHMARK_STATE["indexed_doc_ids"][idx]
                for idx in indices[i]
                if idx != -1
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
    one used during ingest — both default to _DEFAULT_QUERIES_FILE and can be
    overridden via the "queries_file" workload parameter.

    Hybrid search mode
    ------------------
    Set ``search_mode: "hybrid"`` in workload params to issue a hybrid query
    (BM25 ``match`` on the ``text`` field combined with ``knn``) via a
    normalization-processor search pipeline instead of a pure kNN query.

    Required additional params:
      search_pipeline   — name of the search pipeline to use
                          (default: "hybrid-search-pipeline")
      hybrid_text_field — index field to run the BM25 leg against
                          (default: "text")
      bm25_weight       — arithmetic-mean weight for the BM25 score  (default: 0.3)
      knn_weight        — arithmetic-mean weight for the kNN score    (default: 0.7)

    Recall notes
    ------------
    Ground truth is always computed by FAISS against pure vector similarity.
    For hybrid mode this means recall measures how closely the combined
    BM25+kNN ranking agrees with exact kNN — useful as a quality/tradeoff
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

        Partition i takes indices i, i+N, i+2N, … so every query is issued by
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

        # Ground truth is always the exact FAISS kNN neighbors — sliced to k.
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
        """Pure kNN query — original behaviour."""
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

        The query text is the document's own ``text`` field value sampled
        during ingest — a valid BM25 query because Wikipedia passages are
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