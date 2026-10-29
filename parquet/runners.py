"""Search runner with NDCG@10 scoring for the parquet workload.

Wraps OSB's built-in vector-search runner so latency, recall@k and recall@1 are
measured exactly as before. When the param source attaches a ``relevance`` entry
(queries with qrels), the same response is scored against the qrels and the
result is added to the request metadata as ``ndcg@10`` and to the summary file.

Registered under its own operation type so the stock "vector-search" runner is
left untouched for every other workload.
"""

import contextvars
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

    def __init__(self):
        super().__init__()
        self._captured_response = contextvars.ContextVar("ndcg_captured_response", default=None)

    async def _raw_search(self, *args, **kwargs):
        response = await super()._raw_search(*args, **kwargs)
        captured = self._captured_response.get()
        if captured is not None:
            captured["body"] = response.getvalue()
        return response

    async def __call__(self, opensearch, params):
        relevance = params.pop("relevance", None)
        params["operation-type"] = "vector-search"  # dispatch to the recall path
        if relevance is None:
            return await super().__call__(opensearch, params)

        captured = {}
        token = self._captured_response.set(captured)
        try:
            result = await super().__call__(opensearch, params)
        finally:
            self._captured_response.reset(token)

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
