import random
import os
import json
from pathlib import Path

import re
from typing import List
from collections import Counter

from osbenchmark.workload.loader import Downloader
from osbenchmark.workload.loader import Decompressor
from osbenchmark import exceptions

script_dir = os.path.dirname(os.path.realpath(__file__))

def get_by_path(data: dict, path: str, default=None):
    """
    Safely retrieve a value from a nested dictionary using dot-separated path.

    :param data: The dictionary to traverse (e.g., params).
    :param path: Dot-separated path string (e.g., "body.query.neural.passage_embedding").
    :param default: Default value to return if any part of the path is missing.
    :return: The value at the specified path, or default if not found.
    """
    keys = path.split(".")
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            if default is not None:
                return default
            else:
                raise KeyError(f"Key '{key}' not found in {data} in path '{path}'")
    return data

def inject_model_id(params: dict, model_id_file: str = "model_id.json"):
    """
    Injects model_id from file into the given nested path + '.model_id'.

    :param params: The dictionary to modify.
    :param model_id_file: File containing a JSON object with 'model_id'.
    """
    target_file = model_id_file if os.path.exists(model_id_file) else "model_id.json"
    if os.path.exists(target_file):
        with open(target_file, "r") as f:
            model_id = json.load(f).get("model_id")
        params["model_id"] = model_id

def ingest_pipeline_param_source(workload, params, **kwargs):
    model_id_file = "embedding_model_id.json" if os.path.exists("embedding_model_id.json") else "model_id.json"
    supported_processors = ['sparse_encoding', 'text_embedding', 'text_image_embedding']
    processors = params['body'].get('processors', [])

    found_supported_processor = False

    for processor in processors:
        for name in supported_processors:
            if name in processor:
                found_supported_processor = True
                inject_model_id(processor[name], model_id_file=model_id_file)

    if not found_supported_processor:
        raise ValueError(f"No supported processor types found in: {processors}")

    return params

def rerank_search_pipeline_param_source(workload, params, **kwargs):
    model_id_file = "reranker_model_id.json" if os.path.exists("reranker_model_id.json") else "model_id.json"
    with open(model_id_file, "r") as f:
        model_id = json.load(f).get("model_id")

    processors = params.get("body", {}).get("response_processors", [])
    for processor in processors:
        if "rerank" in processor:
            rerank = processor["rerank"]
            if "ml_opensearch" in rerank:
                rerank["ml_opensearch"]["model_id"] = model_id
    return params

class SaveModelId:
    async def __call__(self, opensearch, params):
        import shutil
        source = params.get("source", "model_id.json")
        target = params.get("target", "embedding_model_id.json")
        if os.path.exists(source):
            shutil.copyfile(source, target)
        return {"weight": 1, "unit": "ops", "success": True}

    def __repr__(self, *args, **kwargs):
        return "save-model-id"


class SaveModelIdFromParam:
    """
    Writes a caller-supplied model_id directly to the target file so that
    subsequent steps (ingest-pipeline creation, query param sources) can
    read it without going through register / deploy.
    """
    async def __call__(self, opensearch, params):
        model_id = params.get("model-id")
        if not model_id:
            raise ValueError("save-model-id-from-param requires a 'model-id' parameter")
        target = params.get("target", "embedding_model_id.json")
        with open(target, "w") as f:
            json.dump({"model_id": model_id}, f)
        return {"weight": 1, "unit": "ops", "success": True}

    def __repr__(self, *args, **kwargs):
        return "save-model-id-from-param"

class QueryParamSource:
    def __init__(self, workload, params, **kwargs):
        if len(workload.indices) == 1:
            index = workload.indices[0].name
            if len(workload.indices[0].types) == 1:
                type = workload.indices[0].types[0].name
            else:
                type = None
        else:
            index = "_all"
            type = None

        self._params = params
        self._params['index'] = index
        self._params['type'] = type
        self._params['variable-queries'] = params.get("variable-queries", 0)
        self.infinite = True

        self.dataset_name = self.get_dataset_name()
        self.queries = []
        self.load_queries_file()

    def get_dataset_name(self):
        return self._params.get("corpora_name", self._params.get("corpora", self._params.get("dataset", "quora")))

    def load_queries_file(self):
        if self._params.get('variable-queries', 0) > 0:
            manifest_file = os.path.join(script_dir, "workload_queries.json")
            if not os.path.exists(manifest_file):
                return
            with open(manifest_file, "r") as f:
                manifest = json.load(f)

            entry = next((item for item in manifest if item.get("name") == self.dataset_name), None)
            if not entry:
                # Fallback to first entry if dataset_name is not matched
                entry = manifest[0] if manifest else None
            if not entry:
                return

            source_file = entry["source-file"]
            base_url = entry["base-url"]
            compressed_bytes = entry.get("compressed-bytes")
            uncompressed_bytes = entry.get("uncompressed-bytes")
            target_file = entry.get("target-file", Path(source_file).stem)

            compressed_path = os.path.join(script_dir, source_file)
            uncompressed_path = os.path.join(script_dir, target_file)

            if not os.path.exists(uncompressed_path):
                if not os.path.exists(compressed_path):
                    downloader = Downloader(False, False)
                    downloader.download(base_url, None, compressed_path, compressed_bytes)
                decompressor = Decompressor()
                decompressor.decompress(compressed_path, uncompressed_path, uncompressed_bytes)

            if os.path.exists(uncompressed_path):
                with open(uncompressed_path, "r") as f:
                    self.queries = f.read().splitlines()

    def get_random_query_obj(self):
        if not self.queries:
            return None
        line = random.choice(self.queries)
        return json.loads(line)

    def get_random_query_text(self, field_name: str = "text") -> str:
        obj = self.get_random_query_obj()
        if obj:
            return obj.get(field_name, "")
        return "What are some of the best science projects?"

    def partition(self, partition_index, total_partitions):
        return self


class NeuralQueryParamSource(QueryParamSource):
    def params(self):
        params = self._params
        is_query_semantic_field = params.get('is_query_semantic_field', False)
        if is_query_semantic_field is True:
            embedding_query = get_by_path(params, "body.query.neural.text")
        else:
            search_operation_name = params.get('name', None)
            if search_operation_name in ("semantic-search", "semantic-search-with-reranker"):
                query_name = "neural"
            elif search_operation_name == "sparse-search":
                query_name = "neural_sparse"
            else:
                raise KeyError(
                    f"Unsupported search operation name: '{search_operation_name}' in NeuralQueryParamSource.")

            nested = params.get('nested', 'False')
            if nested == 'True':
                path_to_neural_query = f"body.query.nested.query.{query_name}"
                if query_name == "neural":
                    embedding_field = 'passage_chunk_embedding.knn'
                else:
                    embedding_field = 'passage_chunk_embedding.sparse_encoding'
            else:
                path_to_neural_query = f"body.query.{query_name}"
                embedding_field = "passage_embedding"
            neural_query = get_by_path(params, path_to_neural_query)
            try:
                embedding_query = neural_query[embedding_field]
            except KeyError:
                raise KeyError(f"Key '{embedding_field}' not found in {neural_query}")

            model_id_file = "embedding_model_id.json" if os.path.exists("embedding_model_id.json") else "model_id.json"
            inject_model_id(embedding_query, model_id_file=model_id_file)

        if params.get("variable-queries", 0) > 0:
            embedding_query["query_text"] = self.get_random_query_text("text")

        return params

class NeuralHybridQueryParamSource(QueryParamSource):
    def params(self):
        params = self._params
        hybrid_queries = params['body']['query']['hybrid']['queries']

        model_id_file = "embedding_model_id.json" if os.path.exists("embedding_model_id.json") else "model_id.json"
        with open(model_id_file, 'r') as f:
            d = json.loads(f.read())
            hybrid_queries[1]['neural']['passage_embedding']['model_id'] = d['model_id']

        count = self._params.get("variable-queries", 0)
        if count > 0:
            query_text = self.get_random_query_text("text")
            hybrid_queries[0]['match']['text']['query'] = query_text
            hybrid_queries[1]['neural']['passage_embedding']['query_text'] = query_text
        return params

class NeuralHybridQueryBoolParamSource(QueryParamSource):
    def params(self):
        params = self._params
        bool_should_query = params['body']['query']['bool']['should']

        model_id_file = "embedding_model_id.json" if os.path.exists("embedding_model_id.json") else "model_id.json"
        with open(model_id_file, 'r') as f:
            d = json.loads(f.read())
            bool_should_query[1]['neural']['passage_embedding']['model_id'] = d['model_id']

        count = self._params.get("variable-queries", 0)
        if count > 0:
            query_text = self.get_random_query_text("text")
            bool_should_query[0]['match']['text']['query'] = query_text
            bool_should_query[1]['neural']['passage_embedding']['query_text'] = query_text
        return params


class NeuralHybridQueryComplexParamSource(QueryParamSource):
    def params(self):
        params = self._params
        hybrid_queries = params['body']['query']['hybrid']['queries']

        model_id_file = "embedding_model_id.json" if os.path.exists("embedding_model_id.json") else "model_id.json"
        with open(model_id_file, 'r') as f:
            d = json.loads(f.read())
            hybrid_queries[2]['neural']['passage_embedding']['model_id'] = d['model_id']

        def tokenize_query(query_text: str) -> List[str]:
            clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', query_text.lower())
            tokens = clean_text.split()
            return tokens

        count = self._params.get("variable-queries", 0)
        if count > 0:
            query_text = self.get_random_query_text("text")

            tokens = tokenize_query(query_text)
            word_freq = Counter(tokens)
            phrases = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
            important_terms = [word for word, freq in word_freq.items() if len(word) > 2]

            hybrid_queries[0]['match_phrase']['text']['query'] = query_text
            hybrid_queries[1]['match']['text']['query'] = " ".join(important_terms)
            hybrid_queries[2]['neural']['passage_embedding']['query_text'] = query_text

            current_length = len(hybrid_queries)
            new_phrase_queries = [
                {
                    "match_phrase": {
                        "text": {
                            "query": phrase,
                            "boost": 1.5,
                            "slop": 1
                        }
                    }
                } for phrase in phrases[:2]
            ]

            if current_length == 3:
                hybrid_queries.extend(new_phrase_queries)
            else:
                del hybrid_queries[3:]
                hybrid_queries.extend(new_phrase_queries)
        return params

class NeuralMultimodalQueryParamSource(QueryParamSource):
    def get_dataset_name(self):
        return self._params.get("corpora_name", self._params.get("corpora", self._params.get("dataset", "abo")))

    def params(self):
        params = self._params
        vector_embedding_query = params['body']['query']['neural']['vector_embedding']

        model_id_file = "embedding_model_id.json" if os.path.exists("embedding_model_id.json") else "model_id.json"
        with open(model_id_file, 'r') as f:
            d = json.loads(f.read())
        vector_embedding_query['model_id'] = d['model_id']

        count = self._params.get("variable-queries", 1)

        if count == 0:
            raise exceptions.DataError("variable-queries parameter for multimodal search cannot be 0.")
        if count > 0:
            item = self.get_random_query_obj()
            if item:
                vector_embedding_query['query_text'] = item.get('image_description', "")
                vector_embedding_query['query_image'] = item.get('image_binary', "")
        return params

class CreateIndexWithSemanticFieldParamSource:
    def __init__(self, workload, params, **kwargs):
        self.infinite = True
        self.index_name = workload.indices[0].name
        self.index_body  = workload.indices[0].body

    def partition(self, partition_index, total_partitions):
        return self

    def params(self):
        path_to_semantic_field = "mappings.properties.text"
        semantic_field = get_by_path(self.index_body, path_to_semantic_field)
        inject_model_id(semantic_field)

        return {
            "indices": [
                [self.index_name, self.index_body]
            ]
        }

def register(registry):
    registry.register_runner("save-model-id", SaveModelId(), async_runner=True)
    registry.register_runner("save-model-id-from-param", SaveModelIdFromParam(), async_runner=True)
    registry.register_param_source("neural-search-source", NeuralQueryParamSource)
    registry.register_param_source("neural-hybrid-search-source", NeuralHybridQueryParamSource)
    registry.register_param_source("neural-hybrid-search-bool-source", NeuralHybridQueryBoolParamSource)
    registry.register_param_source("neural-hybrid-search-complex-source", NeuralHybridQueryComplexParamSource)
    registry.register_param_source("neural-multimodal-search-source", NeuralMultimodalQueryParamSource)
    registry.register_param_source("create-ingest-pipeline-source", ingest_pipeline_param_source)
    registry.register_param_source("create-rerank-search-pipeline-source", rerank_search_pipeline_param_source)
    registry.register_param_source("create-index-with-semantic-field-source", CreateIndexWithSemanticFieldParamSource)
