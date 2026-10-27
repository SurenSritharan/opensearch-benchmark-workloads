"""Regression: concurrent NDCG requests must leave the shared runner intact."""

import asyncio
import io
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from osbenchmark.worker_coordinator.runner import request_context_holder
from runners import VectorSearchWithNdcg


class FakeTransport:
    async def perform_request(self, method, path, params=None, body=None, headers=None):
        assert params == {"search_pipeline": "test-pipeline"}
        query_number = int(body["query"]["match"]["text"]["query"])
        await asyncio.sleep(query_number * 0.003)
        hit_id = f"doc-{query_number}"
        return io.BytesIO(json.dumps({
            "hits": {"total": {"value": 1, "relation": "eq"},
                     "hits": [{"_id": hit_id, "_source": {"pid": hit_id}}]},
            "timed_out": False, "took": 1,
        }).encode())


class FakeClient:
    def __init__(self):
        self.transport = FakeTransport()

    def return_raw_response(self):
        pass


class FakeAccumulator:
    def __init__(self):
        self.scores = {}

    def add(self, query_id, score):
        self.scores[query_id] = score


class TestRunnerConcurrency(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_requests_do_not_leave_capture_wrappers(self):
        runner = VectorSearchWithNdcg()
        client = FakeClient()
        scores = FakeAccumulator()

        async def request(i):
            hit_id = f"doc-{i}"
            params = {
                "operation-type": "vector-search-ndcg", "index": "test-index",
                "body": {"query": {"match": {"text": {"query": str(i)}}}},
                "request-params": {"search_pipeline": "test-pipeline"},
                "k": 1, "neighbors": [hit_id], "num_clients": 8, "num_cores": 8,
                "relevance": {"pid_field": "pid", "separator": "", "qrels": {hit_id: 1},
                              "query_id": str(i), "accumulator": scores},
            }
            async with request_context_holder.new_request_context():
                return await runner(client, params)

        results = await asyncio.wait_for(asyncio.gather(*(request(i) for i in range(8))), timeout=2)
        self.assertEqual([r["ndcg@10"] for r in results], [1.0] * 8)
        self.assertEqual(scores.scores, {str(i): 1.0 for i in range(8)})
        self.assertIs(getattr(runner._raw_search, "__func__", None),
                      VectorSearchWithNdcg._raw_search)


if __name__ == "__main__":
    unittest.main()
