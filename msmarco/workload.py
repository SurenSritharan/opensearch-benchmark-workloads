import os
import struct
import urllib.request
from osbenchmark.workload.params import ParamSource
from .runners import register as register_runners
import numpy as np
import json
import copy
from pathlib import Path

def register(registry):
    register_runners(registry)
    registry.register_param_source("msmarco-fvec-bulk-source", MsMarcoFvecBulkSource)
    registry.register_param_source("random-vector-search-param-source", RandomSearchParamSource)

def _open_fvec(file_path_or_url, byte_offset=0):
    """Open a .fvec source by local path or HTTP(S) URL.

    For HTTP(S) URLs, issues a Range request so only the bytes needed for this
    partition are transferred — no local copy is written.  For local paths the
    file is opened normally and seeked to byte_offset.

    Returns a file-like object positioned at byte_offset.
    """
    if file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://"):
        req = urllib.request.Request(
            file_path_or_url,
            headers={"Range": f"bytes={byte_offset}-"},
        )
        return urllib.request.urlopen(req)
    else:
        f = open(file_path_or_url, "rb")
        if byte_offset:
            f.seek(byte_offset)
        return f


def _ensure_local_file(url_or_path):
    """Return a local file path for *url_or_path*, downloading if necessary.

    If *url_or_path* is already a local path it is returned unchanged.
    If it is an HTTP(S) URL the file is downloaded once to
    $BENCHMARK_HOME/.osb/benchmarks/data/msmarco/<filename> and that path
    is returned on all subsequent calls.
    """
    if not (url_or_path.startswith("http://") or url_or_path.startswith("https://")):
        return url_or_path

    benchmark_home = os.environ.get("BENCHMARK_HOME", "/datasets/opensearch-benchmark")
    cache_dir = Path(benchmark_home) / ".osb" / "benchmarks" / "data" / "msmarco"
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = url_or_path.split("/")[-1]
    local_path = cache_dir / filename

    if local_path.exists():
        print(f"Using cached file: {local_path}")
        return str(local_path)

    print(f"Downloading {filename} ...")
    tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url_or_path, str(tmp_path))
        tmp_path.rename(local_path)
        print(f"Downloaded {filename} ({local_path.stat().st_size / (1024**2):.1f} MB)")
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return str(local_path)


def _fvec_total_docs(file_path_or_url, vector_size_bytes):
    """Return the total number of vectors in a .fvec source.

    For HTTP(S) URLs, reads the Content-Length from a HEAD request.
    For local paths, uses os.path.getsize.
    """
    if file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://"):
        req = urllib.request.Request(file_path_or_url, method="HEAD")
        with urllib.request.urlopen(req) as resp:
            content_length = int(resp.headers.get("Content-Length", 0))
        return content_length // vector_size_bytes
    else:
        return os.path.getsize(file_path_or_url) // vector_size_bytes


class MsMarcoFvecBulkSource:
    def __init__(self, workload, params, **kwargs):
        # file_path accepts either a local path or an HTTP(S) URL — when a URL
        # is supplied each partition streams its byte slice directly from S3/GCS
        # without writing anything to disk.
        self.file_path = params.get("file_path")
        self.bulk_size = params.get("bulk_size", 1000)
        self.index_name = params.get("index")
        self.detailed_results = params.get("detailed-results", False)
        self.request_timeout = params.get("request-timeout", None)

        # Dimension comes from datasets.yaml common_params as "dimension".
        # Falls back to 1024 so existing direct invocations that don't set it still work.
        self.dim = int(params.get("dimension", 1024))
        self.vector_size_bytes = 4 + (self.dim * 4)

        # num_vectors is injected by the loader from corpus_size (e.g. 8m → 8_000_000).
        # Only fall back to a HEAD request if it is genuinely absent (e.g. direct invocation).
        self.total_docs = params.get("num_vectors")

        if self.total_docs is None:
            self.total_docs = _fvec_total_docs(self.file_path, self.vector_size_bytes)
            print(f"Calculated total_docs from source size: {self.total_docs}")
        else:
            print(f"Using num_vectors from params: {self.total_docs}")

    def partition(self, client_index, total_clients):
        # Segmenting file chunks cleanly across multi-client GKE pod deployments
        return MsMarcoFvecPartition(self, client_index, total_clients)

class MsMarcoFvecPartition:
    def __init__(self, source, client_index, total_clients):
        self.source = source
        self.bulk_size = source.bulk_size
        self.index_name = source.index_name
        self.detailed_results = source.detailed_results
        self.request_timeout = source.request_timeout
        self.vector_size_bytes = source.vector_size_bytes
        self.dim = source.dim
        self.infinite = False
        
        # Parallel slice math
        docs_per_client = source.total_docs // total_clients
        self.start_doc = client_index * docs_per_client
        self.end_doc = self.start_doc + docs_per_client if client_index < total_clients - 1 else source.total_docs
        self.current_doc = self.start_doc
        
        # Open the source positioned at this partition's start byte.
        # For HTTP URLs this issues a Range request so only this slice is
        # transferred; for local files it seeks to the correct offset.
        byte_offset = self.current_doc * self.vector_size_bytes
        self.f = _open_fvec(source.file_path, byte_offset)

    def __iter__(self):
        return self

    def __next__(self):
        return self.params()

    @property
    def percent_completed(self):
        total = self.end_doc - self.start_doc
        return 1.0 if total == 0 else (self.current_doc - self.start_doc) / total

    def close(self):
        if hasattr(self, "f") and self.f:
            try:
                if not getattr(self.f, "closed", False):
                    self.f.close()
            except Exception:
                pass

    def __del__(self):
        self.close()

    def params(self):
        if self.current_doc >= self.end_doc:
            self.close()
            raise StopIteration
        
        docs_to_read = min(self.bulk_size, self.end_doc - self.current_doc)
        body = []
        
        for _ in range(docs_to_read):
            length_bytes = self.f.read(4)
            if not length_bytes or len(length_bytes) < 4:
                break
                
            vec_bytes = self.f.read(self.dim * 4)
            if not vec_bytes or len(vec_bytes) < (self.dim * 4):
                break
                
            # Direct float extraction mapping
            vec = struct.unpack(f"{self.dim}f", vec_bytes)
            
            # Action line mapping array (Using string versions of current_doc as _id)
            body.append({"index": {"_index": self.index_name, "_id": str(self.current_doc)}})
            # Data array line mapping
            body.append({
                "vector": list(vec)
            })
            self.current_doc += 1
            
        if not body:
            self.close()
            raise StopIteration
            
        result = {
            "bulk-size": len(body) // 2,
            "unit": "docs",
            "action-metadata-present": True,
            "body": body,
            "index": self.index_name,
            "detailed-results": self.detailed_results
        }
        if self.request_timeout is not None:
            result["request-timeout"] = self.request_timeout
        return result
        
class RandomSearchParamSource(ParamSource):
    def __init__(self, workload, params, **kwargs):
        super().__init__(workload, params, **kwargs)
        print(params)
        
        self._operation_type = params.get('operation-type', "vector-search")
        self._index_name = params.get('index_name', 'target_index')
        self._dims = int(params.get("dimension", params.get("dims", 1024)))
        self._top_k = int(params.get("k", 10))
        self._field = params.get("field", "target_field")
        self._queries_file = params.get("queries_file", "queries.fvec")
        self._ground_truth_file = params.get("ground_truth_file", "ground_truth.ivec")
        self._detailed_results = params.get("detailed-results", True)
        # datasets.yaml exposes this as hnsw_ef_search; fall back to ef_search for
        # direct invocations that use the shorter name.
        self._ef_search = int(params.get("hnsw_ef_search", params.get("ef_search", 128)))
        self._overquery_factor = params.get("overquery_factor")
        self._oversample_factor = params.get("oversample_factor")
        self._filter_type = params.get("filter_type")
        self._filter_body = params.get("filter_body")
        self._space_type = params.get("space_type", "l2")
        self._is_nested = "." in self._field  # mirrors params.py NESTED_FIELD_SEPARATOR
        self._disable_source = params.get("disable-source", False)
        
        # Ensure queries and ground-truth files are present locally.
        # If the params carry an HTTP(S) URL the file is downloaded once and
        # cached under $BENCHMARK_HOME/.osb/benchmarks/data/msmarco/.
        self._queries_file = _ensure_local_file(self._queries_file)
        self._ground_truth_file = _ensure_local_file(self._ground_truth_file)

        # .fvec format: 4 bytes (int32) for dimension + (dims * 4) bytes for float32 data
        self._record_size_bytes = 4 + (self._dims * 4)
        self._data = np.memmap(self._queries_file, dtype='uint8', mode='r')

        self._ground_truth = self._load_ivec_ground_truth()
        
        self._num_queries = len(self._ground_truth)
        
        self._rng = np.random.RandomState(42)
        self._query_body = self._parse_body(params.get("body", {}))
        
        # Shuffle-based query selection (no repeats until all queries used)
        self._query_indices = np.arange(self._num_queries)
        self._rng.shuffle(self._query_indices)
        self._current_idx = 0
        
        # ===== DEBUG: Ground Truth Info =====
        print("\n" + "="*60)
        print("GROUND TRUTH FILE LOADED")
        print("="*60)
        print(f"Ground truth file: {self._ground_truth_file}")
        print(f"Ground truth shape: {self._ground_truth.shape}")
        print(f"Number of queries: {self._num_queries}")
        print(f"Top-k: {self._top_k}")
        print(f"Query selection: Shuffle-based (no repeats until all {self._num_queries} queries used)")
        print(f"\nFirst 3 queries and their ground truth neighbors:")
        for i in range(min(3, self._num_queries)):
            print(f"  Query {i}: {self._ground_truth[i].tolist()}")
        print("="*60 + "\n")
        # ===== END DEBUG =====

    def close(self):
        if hasattr(self, "_data") and self._data is not None:
            if hasattr(self._data, "_mmap") and self._data._mmap is not None:
                try:
                    self._data._mmap.close()
                except Exception:
                    pass
            self._data = None

    def __del__(self):
        self.close()

    def _load_ivec_ground_truth(self):
        """
        Custom parser extension to dynamically consume standard .ivec length prefixes
        to ensure data indices are not bit-shifted.
        """
        gt_list = []
        try:
            with open(self._ground_truth_file, 'rb') as f:
                while True:
                    # Read the 4-byte prefix identifying the current row's length
                    k_bytes = f.read(4)
                    if not k_bytes or len(k_bytes) < 4:
                        break
                    
                    # Unpack row length integer (e.g. k=10)
                    k_length = struct.unpack('i', k_bytes)[0]
                    
                    # Safely consume exactly 'k_length' integer entries for this row
                    row_data = np.fromfile(f, dtype='int32', count=k_length)
                    
                    # Structural integrity verification
                    if len(row_data) == k_length:
                        # Slice or pad to enforce matching dimensions if required
                        gt_list.append(row_data[:self._top_k])
                    else:
                        break
        except Exception as e:
            raise RuntimeError(f"Failed to parse .ivec ground truth using custom client reader: {str(e)}")
            
        return np.array(gt_list, dtype='int32')
    
    def _parse_body(self, body_param):
        if isinstance(body_param, str):
            try:
                return json.loads(body_param)
            except json.JSONDecodeError:
                return {}
        return body_param

    def partition(self, partition_index, total_partitions):
        # Create a deep copy of the partition via super()
        partition = super().partition(partition_index, total_partitions)
        partition._data = self._data
        partition._ground_truth = self._ground_truth
        # Ensure each partition has an isolated RNG state and shuffled indices
        partition._rng = np.random.RandomState(42 + partition_index)
        partition._query_indices = np.arange(self._num_queries)
        partition._rng.shuffle(partition._query_indices)
        partition._current_idx = 0
        partition._overquery_factor = self._overquery_factor
        partition._disable_source = self._disable_source
        return partition

    def params(self):
        # Get next query from shuffled list
        query_idx = self._query_indices[self._current_idx]
        self._current_idx += 1
        
        # When we've used all queries, reshuffle and start over
        if self._current_idx >= self._num_queries:
            self._rng.shuffle(self._query_indices)
            self._current_idx = 0
        
        # # ===== DEBUG: Query Info =====
        # print("\n" + "="*60)
        # print(f"GENERATING QUERY PARAMS (Query Index: {query_idx})")
        # print("="*60)
        # print(f"Expected ground truth neighbors: {self._ground_truth[query_idx].tolist()}")
        # # ===== END DEBUG =====
        
        # Extract raw vector slice
        start_byte = query_idx * self._record_size_bytes + 4
        end_byte = start_byte + (self._dims * 4)
        query_vec = self._data[start_byte : end_byte].view(np.float32).tolist()
        
        # Generate baseline query
        query = self.generate_knn_query(query_vec)
        
        if self._filter_type == "post_filter":
            query["post_filter"] = self._filter_body
        
        # Merge dynamic body overrides if they exist
        if self._query_body:
            # We copy to prevent cross-pollination between iterations
            self._deep_merge(query, copy.deepcopy(self._query_body))

        # Suppress _source when disable-source is set. The recall runner reads
        # the hit's top-level _id directly, so no docvalue_fields needed here
        # since msmarco uses _id as the identifier.
        if self._disable_source:
            query["_source"] = False

        # Convert to string to match opensearch _id
        ground_truth_ids = [str(int(x)) for x in self._ground_truth[query_idx]]
        
        result = {
            "index": self._index_name,
            "size": self._top_k,
            "k": self._top_k,
            "operation-type": self._operation_type,
            "body": query,
            "neighbors": ground_truth_ids, # Convert to list for JSON
            "detailed-results": self._detailed_results,
            "request-params": {"size": self._top_k}
        }
        # print(f"DEBUG: 'k' in params = {'k' in result}")
        # print(f"DEBUG: k value = {result.get('k', 'NOT FOUND')}")
        
        return result

    def _deep_merge(self, base, overrides):
        """
        Recursively merges overrides into base.
        """
        for key, value in overrides.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def generate_knn_query(self, query_vector):
        # efficient filter goes inside the knn body
        efficient_filter = self._filter_body if self._filter_type == "efficient" else None

        knn_body = {
            "vector": query_vector,
            "k": self._top_k,
        }

        if efficient_filter:
            knn_body["filter"] = efficient_filter

        if self._ef_search or self._overquery_factor:
            method_params = {}
            if self._ef_search:
                method_params["ef_search"] = self._ef_search
            if self._overquery_factor:
                method_params["overquery_factor"] = self._overquery_factor
            knn_body["method_parameters"] = method_params

        if self._oversample_factor:
            knn_body["rescore"] = {"oversample_factor": self._oversample_factor}

        knn_search_query = {
            "query": {
                "knn": {
                    self._field: knn_body
                }
            }
        }

        # nested field: wrap in a nested query
        if self._is_nested:
            outer_field = self._field.split(".")[0]
            return {
                "query": {
                    "nested": {
                        "path": outer_field,
                        "query": {"knn": {self._field: knn_body}}
                    }
                }
            }

        # post_filter is handled in params(), not here
        if self._filter_type and not efficient_filter and self._filter_type != "post_filter":
            return self._knn_query_with_filter(query_vector, knn_search_query)

        return knn_search_query

    def _knn_query_with_filter(self, query_vector, knn_query):
        if self._filter_type == "script":
            return {
                "query": {
                    "script_score": {
                        "query": {"bool": {"filter": self._filter_body}},
                        "script": {
                            "source": "knn_score",
                            "lang": "knn",
                            "params": {
                                "field": self._field,
                                "query_value": query_vector,
                                "space_type": self._space_type
                            }
                        }
                    }
                }
            }
        if self._filter_type == "boolean":
            return {
                "query": {
                    "bool": {
                        "filter": self._filter_body,
                        "must": [knn_query["query"]]
                    }
                }
            }
        raise ValueError(f"Unsupported filter_type: {self._filter_type!r}")

