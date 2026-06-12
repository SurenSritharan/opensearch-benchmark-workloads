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
import logging
import json

def register(registry):
    register_runners(registry)
    registry.register_param_source("msmarco-fvec-bulk-source", MsMarcoFvecBulkSource)
    registry.register_param_source("random-vector-search-param-source", RandomSearchParamSource)

class MsMarcoFvecBulkSource:
    def __init__(self, workload, params, **kwargs):
        # Configuration properties defined in workload.json
        self.file_path = params.get("file_path")
        self.bulk_size = params.get("bulk_size", 1000)
        self.index_name = params.get("index")
        
        # Fixed MS MARCO Cohere structural variables
        self.dim = 1024  
        self.vector_size_bytes = 4 + (self.dim * 4)
        
        self.file_size = os.path.getsize(self.file_path)
        self.total_docs = self.file_size // self.vector_size_bytes

    def partition(self, client_index, total_clients):
        # Segmenting file chunks cleanly across multi-client GKE pod deployments
        return MsMarcoFvecPartition(self, client_index, total_clients)

class MsMarcoFvecPartition:
    def __init__(self, source, client_index, total_clients):
        self.source = source
        self.bulk_size = source.bulk_size
        self.index_name = source.index_name
        self.vector_size_bytes = source.vector_size_bytes
        self.dim = source.dim
        self.infinite = False
        
        # Parallel slice math
        docs_per_client = source.total_docs // total_clients
        self.start_doc = client_index * docs_per_client
        self.end_doc = self.start_doc + docs_per_client if client_index < total_clients - 1 else source.total_docs
        self.current_doc = self.start_doc
        
        # Independent pointer position per active file channel stream
        self.f = open(source.file_path, "rb")
        self.f.seek(self.current_doc * self.vector_size_bytes)

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
            self.f.close()
            raise StopIteration
            
        return {
            "bulk-size": len(body) // 2,
            "unit": "docs",
            "action-metadata-present": True,
            "body": body
        }

# Why fixed seed (42)?
# - Ensures all parallel processes/clients generate vectors from the same cluster structure
# - Critical when running with multiple indexing clients (e.g., 200 processes for 1B docs)
_cluster_centers = None


def _get_cluster_centers(dims, num_centers, seed=42):
    """Generate and cache cluster centers so all processes use the same centers."""
    global _cluster_centers
    if _cluster_centers is None or _cluster_centers.shape != (num_centers, dims):
        rng = np.random.RandomState(seed)
        _cluster_centers = rng.rand(num_centers, dims).astype('float32') * 100
    return _cluster_centers

class RandomSearchParamSource(ParamSource):
    def __init__(self, workload, params, **kwargs):
        super().__init__(workload, params, **kwargs)
        logging.getLogger(__name__).info("Workload: [%s], params: [%s]", workload, params)
        self._workload = workload
        # self._params = params
        
        self._index_name = params.get('index_name', 'target_index')
        self._dims = int(params.get("dims", 1024))
        self._top_k = int(params.get("k", 10))
        self._field = params.get("field", "target_field")
        self._vector_file = params.get("vector_file", "cohere_msmarco_base.fvec")
        
        # .fvec format: 4 bytes (int32) for dimension + (dims * 4) bytes for float32 data
        self._record_size_bytes = 4 + (self._dims * 4)
        
        # Memory-map as uint8 for raw byte access
        self._data = np.memmap(self._vector_file, dtype='uint8', mode='r')
        
        # Ground truth
        self._ground_truth_file = params.get("ground_truth_file", "ground_truth.ivec")
        self._ground_truth = np.fromfile(self._ground_truth_file, dtype='int32').reshape(-1, 10)
        self._num_queries = len(self._ground_truth)
        
        # RNG and other state
        self._rng = np.random.RandomState(42)
        self._query_body = self._parse_body(params.get("body", {}))
        self._detailed_results = params.get("detailed-results", True)

    def _parse_body(self, body_param):
        if isinstance(body_param, str):
            try:
                return json.loads(body_param) if body_param.strip() else {}
            except json.JSONDecodeError:
                return {}
        return body_param

    def partition(self, partition_index, total_partitions):
        new_source = self.__class__(self._workload, self._params)
        new_source._data = self._data
        new_source._ground_truth = self._ground_truth
        new_source._rng = np.random.RandomState(42 + partition_index)
        return new_source

    def params(self):
        query_idx = self._rng.randint(0, self._num_queries)
        
        # Calculate start: Skip the 4-byte header of the chosen record
        start_byte = query_idx * self._record_size_bytes + 4
        end_byte = start_byte + (self._dims * 4)
        
        # Extract raw bytes and interpret as float32
        raw_vec = self._data[start_byte : end_byte].view(np.float32)
        
        # Force list conversion (essential for JSON serialization)
        query_vec = raw_vec.tolist()
        
        query = self.generate_knn_query(query_vec)
        query.update(self._query_body)
        print(self._query_body)
        
        return {
            "index": self._index_name, 
            "size": self._top_k, 
            "body": query, 
            "detailed-results": self._detailed_results,
            "neighbors": self._ground_truth[query_idx], 
            "detailed-results": self._detailed_results
        }

    def generate_knn_query(self, query_vector):
        return {
            "query": {
                "knn": {
                    self._field: {
                        "vector": query_vector,
                        "k": self._top_k,
                        "method_parameters": {
                           "ef_search": 128
                        }
                    }
                }
            }
        }


