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


class RandomBulkParamSource(ParamSource):
    def __init__(self, workload, params, **kwargs):
        super().__init__(workload, params, **kwargs)
        logging.getLogger(__name__).info("Workload: [%s], params: [%s]", workload, params)
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
        logging.getLogger(__name__).info("Workload: [%s], params: [%s]", workload, params)
        # 1. Standardized defaults to match your current index configuration safely
        self._index_name = params.get('index', 'cohere-msmarco-1024')
        self._dims = params.get("dims", 1024)
        self._cache = params.get("cache", False)
        self._top_k = params.get("k", 10)
        self._field = params.get("field", "vector") # Matches your mapping field: 'vector'
        self._query_body = params.get("body", {})
        self._detailed_results = params.get("detailed-results", True)
        self._num_centers = params.get("num_centers", 2000)
        self._cluster_std = params.get("cluster_std", 0.5)
        self._centers = _get_cluster_centers(self._dims, self._num_centers)

    def partition(self, partition_index, total_partitions):
        return self

    def params(self):
        # Generate query vector from the same cluster distribution
        query_vec, _ = make_blobs(
            n_samples=1,
            n_features=self._dims,
            centers=self._centers,
            cluster_std=self._cluster_std
        )
        query_vec = query_vec[0].tolist()
        
        # 2. Extract out any custom source or filtering metadata from your configuration body
        source_filter = self._query_body.get("_source", False)
        stored_fields_filter = self._query_body.get("stored_fields", ["_id"])
        
        # 3. Construct the clean final payload OpenSearch expects
        payload = {
            "_source": source_filter,
            "stored_fields": stored_fields_filter,
            "query": {
                "knn": {
                    self._field: {
                        "vector": query_vec,
                        "k": self._top_k
                    }
                }
            }
        }

        # 4. Return as a standard 'search' operation contract payload
        return {
            "index": self._index_name, 
            "cache": self._cache, 
            "body": payload, 
            "detailed-results": self._detailed_results
        }


def register(registry):
    register_runners(registry)
    registry.register_param_source("msmarco-fvcec-bulk-source", MsMarcoFvecBulkSource)
    registry.register_param_source("random-vector-search-param-source", RandomSearchParamSource)
