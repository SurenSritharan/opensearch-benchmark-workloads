"""Search runner with NDCG@10 scoring for the parquet workload.

Wraps OSB's built-in vector-search runner so latency, recall@k and recall@1 are
measured exactly as before. When the param source attaches a ``relevance`` entry
(queries with qrels), the same response is scored against the qrels and the
result is added to the request metadata as ``ndcg@10`` and to the summary file.

Registered under its own operation type so the stock "vector-search" runner is
left untouched for every other workload.
"""

import json
import logging

from osbenchmark.worker_coordinator.runner import Query

try:
    from .relevance import dedupe_preserving_order, ndcg_at_k, qrel_doc_id
except ImportError:
    from relevance import dedupe_preserving_order, ndcg_at_k, qrel_doc_id

logger = logging.getLogger(__name__)

OPERATION_TYPE = "vector-search-ndcg"


class VectorSearchWithNdcg(Query):
    """``vector-search`` plus NDCG@10 against qrels when the query has them."""

    async def __call__(self, opensearch, params):
        relevance = params.pop("relevance", None)
        params["operation-type"] = "vector-search"  # dispatch to the recall path
        if relevance is None:
            return await super().__call__(opensearch, params)

        captured = {}
        original = self._raw_search

        async def capturing_raw_search(*args, **kwargs):
            response = await original(*args, **kwargs)
            captured["body"] = response.getvalue()
            return response

        self._raw_search = capturing_raw_search
        try:
            result = await super().__call__(opensearch, params)
        finally:
            self._raw_search = original

        if "body" in captured:
            hits = json.loads(captured["body"]).get("hits", {}).get("hits", [])
            pid_field = relevance["pid_field"]
            ranked = dedupe_preserving_order(
                qrel_doc_id(str(h.get("_source", {}).get(pid_field, h["_id"])), relevance["separator"])
                for h in hits
            )
            score = ndcg_at_k(ranked, relevance["qrels"], 10)
            result["ndcg@10"] = score
            relevance["accumulator"].add(relevance["query_id"], score)
        return result

    def __repr__(self, *args, **kwargs):
        return OPERATION_TYPE


def register(registry):
    registry.register_runner(OPERATION_TYPE, VectorSearchWithNdcg(), async_runner=True)
