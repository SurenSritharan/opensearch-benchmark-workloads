import json
import logging
import os
import faiss
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

logger = logging.getLogger(__name__)

# Global state shared between indexing parameter generator and query ground truth builder
BENCHMARK_STATE = {
    "faiss_index": faiss.IndexFlatIP(1024),
    "indexed_doc_ids": [],
    "sample_queries": [],
    "ground_truth": {},
}

_DEFAULT_QUERIES_FILE = "wikiparquet-queries-gt-{target_count}.json"


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

    def __init__(self, workload, params):
        self.workload = workload
        self.params = params
        self.batch_size = params.get("batch_size", 1000)
        self.target_docs = params.get("target_vector_count", 50000)
        default_index_name = f"cohere-wikipedia-{_format_count(self.target_docs)}"
        self.index_name = params.get("index", default_index_name)
        self.num_queries = params.get("num_queries", 1000)
        default_queries_file = _DEFAULT_QUERIES_FILE.format(
            target_count=self.target_docs
        )
        self.queries_file = params.get("queries_file", default_queries_file)

        self.fs = HfFileSystem()
        self.repo_id = "CohereLabs/wikipedia-2023-11-embed-multilingual-v3"
        self.dataset_dir = (
            "datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3/en"
        )

        self._generator = self._stream_and_index()

    def partition(self, partition_index, total_partitions):
        # OpenSearch Benchmark client partitioning support
        return self

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
            # Still stream and yield bulk actions, but skip FAISS/GT work
            gt_exists = True
        else:
            gt_exists = False

        remote_files = sorted(self.fs.glob(f"{self.dataset_dir}/*.parquet"))
        vector_count = 0
        columns = ["_id", "title", "text", "emb"]

        for file_idx, remote_path in enumerate(remote_files):
            local_path = self._get_local_cached_path(remote_path)
            parquet_file = pq.ParquetFile(local_path)

            for batch in parquet_file.iter_batches(
                batch_size=self.batch_size, columns=columns
            ):
                data = batch.to_pydict()
                id_list = data.get("_id") or data.get("id")
                bulk_actions = []

                for i in range(len(id_list)):
                    doc_id = str(id_list[i])
                    title = data["title"][i]
                    text = data["text"][i]
                    raw_emb = data["emb"][i]

                    if not gt_exists:
                        # Normalize vector for FAISS IndexFlatIP (Cosine Similarity)
                        vector = np.array(raw_emb, dtype=np.float32)
                        norm = np.linalg.norm(vector)
                        if norm > 0:
                            vector = vector / norm

                        # Build FAISS in-memory index for ground truth
                        BENCHMARK_STATE["faiss_index"].add(
                            np.expand_dims(vector, axis=0)
                        )
                        BENCHMARK_STATE["indexed_doc_ids"].append(doc_id)

                        # Sample queries
                        if (
                            len(BENCHMARK_STATE["sample_queries"])
                            < self.num_queries
                        ):
                            BENCHMARK_STATE["sample_queries"].append(
                                {
                                    "query_id": doc_id,
                                    "vector": vector.tolist(),
                                }
                            )

                    # Format bulk action payload
                    bulk_actions.append(
                        {"index": {"_index": self.index_name, "_id": doc_id}}
                    )
                    bulk_actions.append(
                        {"title": title, "text": text, "emb": raw_emb}
                    )

                    vector_count += 1
                    if self.target_docs and vector_count >= self.target_docs:
                        break

                yield {
                    "action": "bulk",
                    "body": bulk_actions,
                    "bulk-size": len(id_list),
                }

                if self.target_docs and vector_count >= self.target_docs:
                    if not gt_exists:
                        self._compute_ground_truth()
                    return

        if not gt_exists:
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

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._generator)


class VectorSearchParamReader:
    """Provides vector search query payloads to search runners.

    When running as a standalone procedure (i.e. without bulk-ingest-data having
    run in the same process), loads queries and ground truth from the file written
    by ParquetBulkParamReader._compute_ground_truth.  The file path must match the
    one used during ingest — both default to _DEFAULT_QUERIES_FILE and can be
    overridden via the "queries_file" workload parameter.
    """

    def __init__(self, workload, params):
        self._params = params
        target_docs = params.get("target_vector_count", 50000)
        default_index_name = f"cohere-wikipedia-parquet-{_format_count(target_docs)}"
        self.index_name = params.get("index", default_index_name)
        self.k = params.get("k", 10)
        self.ef_search = params.get("ef_search", 64)
        self.cursor = 0

        # If ingest did not run in this process, load state from disk.
        if not BENCHMARK_STATE["sample_queries"]:
            target_docs = params.get("target_vector_count", 50000)
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
        return self

    def __iter__(self):
        return self

    def __next__(self):
        queries = BENCHMARK_STATE["sample_queries"]
        if not queries:
            raise StopIteration("No queries available in state.")

        query_item = queries[self.cursor % len(queries)]
        self.cursor += 1

        query_payload = {
            "size": self.k,
            "query": {
                "knn": {
                    "emb": {
                        "vector": query_item["vector"],
                        "k": self.k,
                    }
                }
            },
        }

        if self.ef_search:
            query_payload["ext"] = {
                "knn": {"method_parameters": {"ef_search": self.ef_search}}
            }

        # Ground truth for this query — top-100 neighbor doc-id strings ordered
        # by descending similarity, computed by FAISS during ingest.
        # Sliced to k here so OSB's built-in recall scorer compares the right
        # window, matching exactly how msmarco passes "neighbors".
        gt_ids = BENCHMARK_STATE["ground_truth"].get(query_item["query_id"], [])
        neighbors = gt_ids[: self.k]

        return {
            "index": self.index_name,
            "body": query_payload,
            "query_id": query_item["query_id"],
            # Top-level "k" and "neighbors" let OSB compute Recall@K natively
            # without a custom runner — identical to msmarco's RandomSearchParamSource.
            "k": self.k,
            "neighbors": neighbors,
            "detailed-results": self._params.get("detailed-results", True),
        }


# ============================================================================
# 3. HOOK REGISTRATIONS
# ============================================================================


def register(registry):
    registry.register_param_reader(
        "parquet-bulk-reader", ParquetBulkParamReader
    )
    registry.register_param_reader(
        "vector-search-reader", VectorSearchParamReader
    )