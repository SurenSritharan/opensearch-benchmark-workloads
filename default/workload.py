import os
import struct
import time
import multiprocessing
from osbenchmark.worker_coordinator.runner import Runner
from osbenchmark.workload.params import ParamSource
import logging
from .runners import register as register_runners
import random
import numpy as np
from sklearn.datasets import make_blobs
import json

logger = logging.getLogger(__name__)

class GenericVecBulkSource:
    """
    A generic parameter source capable of reading any standard .fvecs, .ivecs, or .bvecs 
    binary file by dynamically detecting dimensions and native data type formats.
    """
    
    # Map file extensions to (struct_format_char, element_size_in_bytes)
    _FORMAT_MAPPING = {
        '.fvecs': ('f', 4),  # float32
        '.ivecs': ('i', 4),  # int32
        '.bvecs': ('B', 1)   # uint8 (unsigned char)
    }

    def __init__(self, workload, params, **kwargs):
        self.file_path = params.get("file_path")
        self.bulk_size = params.get("bulk_size", 1000)
        self.index_name = params.get("index")
        self.vector_field_name = params.get("vector_field", "vector")
        
        if not self.file_path or not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The specified vector file path does not exist: {self.file_path}")

        # Extract type information based on file extension
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext not in self._FORMAT_MAPPING:
            raise ValueError(f"Unsupported file format: {ext}. Must be .fvecs, .ivecs, or .bvecs")
            
        self.fmt_char, self.element_size = self._FORMAT_MAPPING[ext]

        # Automatically determine vector dimension from the first 4 bytes of the file (always int32)
        with open(self.file_path, "rb") as f:
            dim_bytes = f.read(4)
            if not dim_bytes or len(dim_bytes) < 4:
                raise ValueError(f"Failed to read dimension header from empty or corrupt file: {self.file_path}")
            self.dim = struct.unpack("i", dim_bytes)[0]
            
        # Structural variables: 4 bytes (int for dimension) + (dimension * type element size)
        self.vector_size_bytes = 4 + (self.dim * self.element_size)
        
        self.file_size = os.path.getsize(self.file_path)
        self.total_docs = self.file_size // self.vector_size_bytes
        
        logger.info(f"Initialized GenericVecBulkSource. File: {self.file_path} | Extension: {ext} | Detected Dimension: {self.dim} | Total Docs: {self.total_docs}")

    def partition(self, client_index, total_clients):
        # Segmenting file chunks cleanly across multi-client runners
        return GenericVecPartition(self, client_index, total_clients)


class GenericVecPartition:
    def __init__(self, source, client_index, total_clients):
        self.source = source
        self.bulk_size = source.bulk_size
        self.index_name = source.index_name
        self.vector_field_name = source.vector_field_name
        self.vector_size_bytes = source.vector_size_bytes
        self.dim = source.dim
        self.fmt_char = source.fmt_char
        self.element_size = source.element_size
        self.infinite = False  
        
        # Parallel slice math
        docs_per_client = source.total_docs // total_clients
        self.start_doc = client_index * docs_per_client
        self.end_doc = self.start_doc + docs_per_client if client_index < total_clients - 1 else source.total_docs
        self.current_doc = self.start_doc
        
        # Independent pointer position per active file channel stream
        self.f = open(source.file_path, "rb")
        self.f.seek(self.current_doc * self.vector_size_bytes)

        # Pre-compile the dynamic unpack format string for speed execution
        self.struct_unpack_str = f"{self.dim}{self.fmt_char}"
        self.payload_bytes_size = self.dim * self.element_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.params()

    @property
    def percent_completed(self):
        total = self.end_doc - self.start_doc
        return 1.0 if total == 0 else (self.current_doc - self.start_doc) / total

    def params(self):
        if self.current_doc >= self.end_doc:
            self.f.close()
            raise StopIteration
        
        docs_to_read = min(self.bulk_size, self.end_doc - self.current_doc)
        body = []
        
        for _ in range(docs_to_read):
            length_bytes = self.f.read(4)
            if not length_bytes or len(length_bytes) < 4:
                break
                
            vec_bytes = self.f.read(self.payload_bytes_size)
            if not vec_bytes or len(vec_bytes) < self.payload_bytes_size:
                break
                
            # Dynamic extraction based on file format structure
            vec = struct.unpack(self.struct_unpack_str, vec_bytes)
            
            # Action line mapping array (Using string versions of current_doc as _id)
            body.append({"index": {"_index": self.index_name, "_id": str(self.current_doc)}})
            # Data array line mapping dynamically allocating the custom vector field key
            body.append({
                self.vector_field_name: list(vec)
            })
            self.current_doc += 1
            
        if not body:
            self.f.close()
            raise StopIteration
            
        return {
            "bulk-size": len(body) // 2,
            "unit": "docs",
            "action-metadata-present": True,
            "body": body
        }


# Cache structure for synthetic cluster centers
_cluster_centers = None

def _get_cluster_centers(dims, num_centers, seed=42):
    """Generate and cache cluster centers so all processes use the same centers."""
    global _cluster_centers
    if _cluster_centers is None or _cluster_centers.shape != (num_centers, dims):
        rng = np.random.RandomState(seed)
        _cluster_centers = rng.rand(num_centers, dims).astype('float32') * 100
    return _cluster_centers


class RandomBulkParamSource(ParamSource):
    def __init__(self, workload, params, **kwargs):
        super().__init__(workload, params, **kwargs)
        logger.info("Workload: [%s], params: [%s]", workload, params)
        self._bulk_size = params.get("bulk-size", 100)
        self._index_name = params.get('index_name', 'target_index')
        self._field = params.get("field", "target_field")
        self._dims = params.get("dims", 768)
        self._partitions = params.get("partitions", 1000)
        self._num_centers = params.get("num_centers", 2000)
        self._cluster_std = params.get("cluster_std", 0.5)
        self._centers = _get_cluster_centers(self._dims, self._num_centers)

    def partition(self, partition_index, total_partitions):
        return self

    def params(self):
        bulk_data = []
        vectors, _ = make_blobs(
            n_samples=self._bulk_size,
            n_features=self._dims,
            centers=self._centers,
            cluster_std=self._cluster_std
        )
        for i in range(self._bulk_size):
            partition_id = random.randint(0, self._partitions)
            metadata = {"_index": self._index_name}
            bulk_data.append({"create": metadata})
            bulk_data.append({"partition_id": partition_id, self._field: vectors[i].tolist()})

        return {
            "body": bulk_data,
            "bulk-size": self._bulk_size,
            "action-metadata-present": True,
            "unit": "docs",
            "index": self._index_name,
            "type": "",
        }


class RandomSearchParamSource(ParamSource):
    def __init__(self, workload, params, **kwargs):
        super().__init__(workload, params, **kwargs)
        logger.info("Workload: [%s], params: [%s]", workload, params)
        self._index_name = params.get('index_name', 'target_index')
        self._dims = params.get("dims", 1024)
        self._cache = params.get("cache", False)
        self._top_k = params.get("k", 100)
        self._field = params.get("field", "target_field")
        body_param = params.get("body", {})
        if isinstance(body_param, str):
            try:
                self._query_body = json.loads(body_param) if body_param.strip() else {}
            except json.JSONDecodeError:
                logger.warning("Failed to decode 'body' param as JSON string. Falling back to empty dict.")
                self._query_body = {}
        else:
            self._query_body = body_param
        self._detailed_results = params.get("detailed-results", True)
        self._num_centers = params.get("num_centers", 2000)
        self._cluster_std = params.get("cluster_std", 0.5)
        self._centers = _get_cluster_centers(self._dims, self._num_centers)

    def partition(self, partition_index, total_partitions):
        return self

    def params(self):
        query_vec, _ = make_blobs(
            n_samples=1,
            n_features=self._dims,
            centers=self._centers,
            cluster_std=self._cluster_std
        )
        query_vec = query_vec[0].tolist()
        query = self.generate_knn_query(query_vec)
        query.update(self._query_body)
        return {
            "index": self._index_name, 
            "cache": self._cache, 
            "size": self._top_k, 
            "body": query, 
            "detailed-results": self._detailed_results
        }

    def generate_knn_query(self, query_vector):
        return {
            "query": {
                "knn": {
                    self._field: {
                        "vector": query_vector,
                        "k": self._top_k
                    }
                }
            }
        }


def register(registry):
    register_runners(registry)
    # Renamed to a universally generic parameter source key
    registry.register_param_source("vector-file-bulk-source", GenericVecBulkSource)
    registry.register_param_source("random-vector-search-param-source", RandomSearchParamSource)