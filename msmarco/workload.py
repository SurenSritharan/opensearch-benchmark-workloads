import os
import struct
import time

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
            
            # Action line mapping array
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

class MsMarcoSearchSource:
    """Param source for MS MARCO vector search queries with ground truth validation"""
    def __init__(self, workload, params, **kwargs):
        self.index_name = params.get("index")
        self.k = params.get("k", 10)
        self.query_count = params.get("query_count", 10000)
        
        # MS MARCO query files
        self.queries_file = "/datasets/msmarco/cohere_msmarco_base.fvec"  # Using base vectors as queries for now
        self.dim = 1024
        self.vector_size_bytes = 4 + (self.dim * 4)
        
        # Load ground truth if available
        self.ground_truth_indices = None
        self.ground_truth_distances = None
        
        try:
            # Try to load ground truth data
            indices_file = "/datasets/msmarco/cohere_msmarco_indices_d1024_k10_1m.ivec"
            distances_file = "/datasets/msmarco/cohere_msmarco_distances_d1024_k10_1m.fvec"
            
            if os.path.exists(indices_file):
                self.ground_truth_indices = self._load_ivec(indices_file, self.query_count, self.k)
            if os.path.exists(distances_file):
                self.ground_truth_distances = self._load_fvec(distances_file, self.query_count, self.k)
        except Exception as e:
            print(f"Warning: Could not load ground truth data: {e}")
    
    def _load_ivec(self, filepath, num_vectors, dim):
        """Load integer vectors (indices) from .ivec file"""
        data = []
        with open(filepath, 'rb') as f:
            for _ in range(num_vectors):
                length_bytes = f.read(4)
                if not length_bytes:
                    break
                vec_bytes = f.read(dim * 4)
                if not vec_bytes:
                    break
                vec = struct.unpack(f"{dim}i", vec_bytes)
                data.append(list(vec))
        return data
    
    def _load_fvec(self, filepath, num_vectors, dim):
        """Load float vectors (distances) from .fvec file"""
        data = []
        with open(filepath, 'rb') as f:
            for _ in range(num_vectors):
                length_bytes = f.read(4)
                if not length_bytes:
                    break
                vec_bytes = f.read(dim * 4)
                if not vec_bytes:
                    break
                vec = struct.unpack(f"{dim}f", vec_bytes)
                data.append(list(vec))
        return data
    
    def partition(self, client_index, total_clients):
        return MsMarcoSearchPartition(self, client_index, total_clients)

class MsMarcoSearchPartition:
    def __init__(self, source, client_index, total_clients):
        self.source = source
        self.index_name = source.index_name
        self.k = source.k
        self.dim = source.dim
        self.vector_size_bytes = source.vector_size_bytes
        self.infinite = False
        
        # Partition queries across clients
        queries_per_client = source.query_count // total_clients
        self.start_query = client_index * queries_per_client
        self.end_query = self.start_query + queries_per_client if client_index < total_clients - 1 else source.query_count
        self.current_query = self.start_query
        
        # Open query file
        self.f = open(source.queries_file, "rb")
        self.f.seek(self.current_query * self.vector_size_bytes)
        
        self.ground_truth_indices = source.ground_truth_indices
        self.ground_truth_distances = source.ground_truth_distances
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.params()
    
    @property
    def percent_completed(self):
        total = self.end_query - self.start_query
        return 1.0 if total == 0 else (self.current_query - self.start_query) / total
    
    def params(self):
        if self.current_query >= self.end_query:
            self.f.close()
            raise StopIteration
        
        # Read query vector
        length_bytes = self.f.read(4)
        if not length_bytes or len(length_bytes) < 4:
            self.f.close()
            raise StopIteration
        
        vec_bytes = self.f.read(self.dim * 4)
        if not vec_bytes or len(vec_bytes) < (self.dim * 4):
            self.f.close()
            raise StopIteration
        
        query_vector = list(struct.unpack(f"{self.dim}f", vec_bytes))
        
        # Build search query
        query_body = {
            "size": self.k,
            "query": {
                "knn": {
                    "vector": {
                        "vector": query_vector,
                        "k": self.k
                    }
                }
            },
            "_source": False,
            "docvalue_fields": ["_id"]
        }
        
        result = {
            "index": self.index_name,
            "body": query_body,
            "cache": False,
            "k": self.k
        }
        
        # Add ground truth if available
        if self.ground_truth_indices and self.current_query < len(self.ground_truth_indices):
            result["expected_ids"] = self.ground_truth_indices[self.current_query]
        if self.ground_truth_distances and self.current_query < len(self.ground_truth_distances):
            result["expected_distances"] = self.ground_truth_distances[self.current_query]
        
        self.current_query += 1
        return result

class MsMarcoVectorSearchRunner:
    """
    Custom runner that executes queries built by MsMarcoSearchPartition,
    and returns metrics that OpenSearch Benchmark aggregates into the summary table.
    """
    def __init__(self):
        pass

    async def __call__(self, es, params):
        index_name = params["index"]
        body = params["body"]
        
        # 1. Execute query
        start_time = time.perf_counter()
        response = await es.search(index=index_name, body=body)
        duration = time.perf_counter() - start_time
        
        # 2. Extract OpenSearch internal IDs from the response
        returned_ids = []
        if "hits" in response and "hits" in response["hits"]:
            for hit in response["hits"]["hits"]:
                try:
                    returned_ids.append(int(hit["_id"]))
                except (ValueError, TypeError):
                    continue

        # 3. Base benchmark transaction metrics
        metrics = {
            "weight": 1,
            "unit": "ops",
            "success": True
        }

        # Safe extraction of ground truth IDs from the partition payload
        gt_indices = params.get("expected_ids")
        k = params.get("k", len(returned_ids) if returned_ids else 10)

        if gt_indices is not None:
            # Build evaluation sets
            true_set_at_k = set(gt_indices[:k])
            returned_set_at_k = set(returned_ids[:k])
            
            # --- RECALL@K CALCULATION ---
            recall_at_k = 0.0
            if len(true_set_at_k) > 0:
                true_pos_k = len(true_set_at_k.intersection(returned_set_at_k))
                recall_at_k = true_pos_k / len(true_set_at_k)
            
            # --- RECALL@1 CALCULATION ---
            recall_at_1 = 0.0
            if len(gt_indices) > 0 and len(returned_ids) > 0:
                if returned_ids[0] == gt_indices[0]:
                    recall_at_1 = 1.0

            # FIX: Format stats as an array of structured metric blocks 
            # so OSBenchmark can compute min/mean/median/max calculations.
            metrics["stats"] = [
                {"name": "recall@k", "value": recall_at_k, "unit": ""},
                {"name": "recall@1", "value": recall_at_1, "unit": ""}
            ]
            
        return metrics

def register(registry):
    registry.register_param_source("msmarco-fvcec-bulk-source", MsMarcoFvecBulkSource)
    registry.register_param_source("msmarco-search-source", MsMarcoSearchSource)
    registry.register_runner("msmarco-vector-search", MsMarcoVectorSearchRunner(), async_runner=True)