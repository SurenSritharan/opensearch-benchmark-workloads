import os
import struct
from osbenchmark.workload.params import ParamSource
from .runners import register as register_runners
import numpy as np
import json
import copy

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
class RandomSearchParamSource(ParamSource):
    def __init__(self, workload, params, **kwargs):
        super().__init__(workload, params, **kwargs)
        
        self._index_name = params.get('index_name', 'target_index')
        self._dims = int(params.get("dims", 1024))
        self._top_k = int(params.get("k", 10))
        self._field = params.get("field", "target_field")
        self._vector_file = params.get("vector_file", "cohere_msmarco_base.fvec")
        self._ground_truth_file = params.get("ground_truth_file", "ground_truth.ivec")
        self._detailed_results = params.get("detailed-results", True)
        
        # .fvec format: 4 bytes (int32) for dimension + (dims * 4) bytes for float32 data
        self._record_size_bytes = 4 + (self._dims * 4)
        self._data = np.memmap(self._vector_file, dtype='uint8', mode='r')
        
        # Load ground truth and ensure it matches expected shape
        self._ground_truth = np.fromfile(self._ground_truth_file, dtype='int32').reshape(-1, self._top_k)
        self._num_queries = len(self._ground_truth)
        
        self._rng = np.random.RandomState(42)
        self._query_body = self._parse_body(params.get("body", {}))

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
        # Ensure each partition has an isolated RNG state
        partition._rng = np.random.RandomState(42 + partition_index)
        return partition

    def params(self):
        query_idx = self._rng.randint(0, self._num_queries)
        
        # Extract raw vector slice
        start_byte = query_idx * self._record_size_bytes + 4
        end_byte = start_byte + (self._dims * 4)
        query_vec = self._data[start_byte : end_byte].view(np.float32).tolist()
        
        # Generate baseline query
        query = self.generate_knn_query(query_vec)
        
        # Merge dynamic body overrides if they exist
        if self._query_body:
            # We copy to prevent cross-pollination between iterations
            self._deep_merge(query, copy.deepcopy(self._query_body))

        return {
            "index": self._index_name, 
            "size": self._top_k, 
            "body": query, 
            "neighbors": self._ground_truth[query_idx].tolist(), # Convert to list for JSON
            "detailed-results": self._detailed_results
        }

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
        return {
            "query": {
                "knn": {
                    self._field: {
                        "vector": query_vector,
                        "k": self._top_k,
                        "method_parameters": {"ef_search": 128}
                    }
                }
            }
        }

