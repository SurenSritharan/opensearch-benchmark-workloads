"""
Unit tests for the hybrid / NDCG additions to the parquet workload.

Covers: NDCG@10 math, qrel id mapping, real-query loading, configurable column
names (MS MARCO v2.1 layout), and the query bodies each search_mode produces.
No OpenSearch or HuggingFace access needed.

Usage:
    python3 parquet/test_hybrid_ndcg.py
"""

import json
import math
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import relevance  # noqa: E402
import workload as wl  # noqa: E402

DIM = 8


class TestNdcg(unittest.TestCase):
    def test_perfect_ranking_is_one(self):
        self.assertAlmostEqual(relevance.ndcg_at_k(["a", "b"], {"a": 2, "b": 1}), 1.0)

    def test_single_relevant_doc_at_rank_two(self):
        self.assertAlmostEqual(relevance.ndcg_at_k(["x", "a"], {"a": 1}), 1 / math.log2(3))

    def test_no_relevant_docs_is_zero(self):
        self.assertEqual(relevance.ndcg_at_k(["x", "y"], {"a": 0}), 0.0)

    def test_graded_gain_uses_two_pow_rel_minus_one(self):
        # one grade-3 doc at rank 1 vs ideal of [3, 1]
        got = relevance.ndcg_at_k(["a"], {"a": 3, "b": 1})
        ideal = 7 + 1 / math.log2(3)
        self.assertAlmostEqual(got, 7 / ideal)

    def test_cutoff_ignores_rank_eleven(self):
        ranked = [f"x{i}" for i in range(10)] + ["a"]
        self.assertEqual(relevance.ndcg_at_k(ranked, {"a": 2}), 0.0)

    def test_segment_ids_collapse_to_doc(self):
        hits = ["d1#0_5", "d1#3_9", "d2#1_1"]
        mapped = relevance.dedupe_preserving_order(relevance.qrel_doc_id(h, "#") for h in hits)
        self.assertEqual(mapped, ["d1", "d2"])
        # without dedupe, d1 would be credited twice and NDCG could exceed 1
        self.assertLessEqual(relevance.ndcg_at_k(mapped, {"d1": 3, "d2": 3}), 1.0)

    def test_accumulator_averages_repeats_per_query(self):
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "ndcg.json")
            acc = relevance.RelevanceAccumulator(path, {"search_mode": "vector"})
            for s in (1.0, 0.0):          # q1 seen twice -> 0.5
                acc.add("q1", s)
            acc.add("q2", 1.0)
            acc.flush()
            summary = json.load(open(path))
            self.assertEqual(summary["queries_scored"], 2)
            self.assertAlmostEqual(summary["ndcg@10_mean"], 0.75)
        finally:
            shutil.rmtree(tmp)


def _write_corpus(tmpdir, rows=50):
    """MS MARCO v2.1-shaped parquet: docid / segment / url / emb."""
    rng = np.random.default_rng(1)
    vecs = rng.standard_normal((rows, DIM)).astype(np.float32)
    table = pa.table({
        "docid": [f"msmarco_v2.1_doc_00_{i}#0_{i}" for i in range(rows)],
        "title": [f"title {i}" for i in range(rows)],
        "segment": [f"passage text number {i}" for i in range(rows)],
        "url": [f"https://www.site{i % 3}.com/page/{i}" for i in range(rows)],
        "emb": pa.array([v.tolist() for v in vecs], type=pa.list_(pa.float32())),
    })
    path = os.path.join(tmpdir, "shard_00.parquet")
    pq.write_table(table, path)
    return path, vecs


def _write_queries(tmpdir):
    rng = np.random.default_rng(2)
    table = pa.table({
        "_id": ["10", "20", "30"],
        "text": ["unjudged query", "judged query", "another judged"],
        "emb": pa.array([v.tolist() for v in rng.standard_normal((3, DIM)).astype(np.float32)],
                        type=pa.list_(pa.float32())),
        "qrels": pa.array([[], [["msmarco_v2.1_doc_00_1", "2"]], [["msmarco_v2.1_doc_00_7", "1"]]],
                          type=pa.list_(pa.list_(pa.string()))),
    })
    path = os.path.join(tmpdir, "queries.parquet")
    pq.write_table(table, path)
    return path


class TestRealQueriesAndColumns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.corpus_path, self.vecs = _write_corpus(self.tmp)
        self.queries_path = _write_queries(self.tmp)
        tmp = self.tmp

        class FakeFs:
            def glob(self, pattern):
                return ["datasets/fake/data/shard_00.parquet"]

        wl.HfFileSystem = FakeFs
        wl.hf_hub_download = lambda repo_id, filename, repo_type=None: os.path.join(tmp, os.path.basename(filename))
        wl.BENCHMARK_STATE["sample_queries"].clear()
        wl.BENCHMARK_STATE["ground_truth"].clear()

    def tearDown(self):
        shutil.rmtree(self.tmp)
        wl.BENCHMARK_STATE["sample_queries"].clear()
        wl.BENCHMARK_STATE["ground_truth"].clear()

    def test_judged_queries_load_first(self):
        qs = wl.load_real_queries(self.queries_path, "emb", limit=2)
        self.assertEqual([q["query_id"] for q in qs], ["20", "30"])
        self.assertEqual(qs[0]["qrels"], {"msmarco_v2.1_doc_00_1": 2})
        self.assertAlmostEqual(float(np.linalg.norm(qs[0]["vector"])), 1.0, places=5)

    def test_ingest_maps_columns_and_builds_gt_for_real_queries(self):
        gt_file = os.path.join(self.tmp, "gt.json")
        reader = wl.ParquetBulkParamReader(object(), {
            "target_vector_count": 50, "target_index_dimension": DIM, "target_field_name": "emb",
            "batch_size": 20, "bulk_size": 25, "num_queries": 10, "queries_file": gt_file,
            "hf_repo_id": "fake", "hf_dataset_dir": "datasets/fake/data",
            "id_column": "docid", "text_column": "segment", "extra_columns": ["url"],
            "derived_fields": {"domain": "url"}, "include_pid": True,
            "real_queries_file": self.queries_path,
        })
        reader = reader.partition(0, 1)
        bodies = []
        try:
            while True:
                bodies.extend(reader.params()["body"])
        except StopIteration:
            pass
        docs = bodies[1::2]
        self.assertEqual(len(docs), 50)
        self.assertEqual(docs[3]["text"], "passage text number 3")
        self.assertEqual(docs[3]["pid"], "msmarco_v2.1_doc_00_3#0_3")
        self.assertEqual(docs[3]["domain"], "site0.com")
        self.assertEqual(bodies[0]["index"]["_id"], "msmarco_v2.1_doc_00_0#0_0")

        payload = json.load(open(gt_file))
        self.assertEqual([q["query_id"] for q in payload["queries"]], ["20", "30", "10"])
        self.assertIn("qrels", payload["queries"][0])
        # exact top-1 for query "20" against the corpus
        q = np.array(payload["queries"][0]["vector"], dtype=np.float32)
        norm = self.vecs / np.linalg.norm(self.vecs, axis=1, keepdims=True)
        expected = f"msmarco_v2.1_doc_00_{int(np.argmax(norm @ q))}#0_{int(np.argmax(norm @ q))}"
        self.assertEqual(payload["ground_truth"]["20"][0], expected)


class TestQueryBodies(unittest.TestCase):
    def setUp(self):
        wl.BENCHMARK_STATE["sample_queries"][:] = [{
            "query_id": "20", "vector": [0.1] * DIM, "text": "bone mass age",
            "qrels": {"msmarco_v2.1_doc_00_1": 2},
        }]
        wl.BENCHMARK_STATE["ground_truth"].clear()
        wl.BENCHMARK_STATE["ground_truth"]["20"] = ["a", "b"]
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)
        wl.BENCHMARK_STATE["sample_queries"].clear()
        wl.BENCHMARK_STATE["ground_truth"].clear()

    def _params(self, mode, **extra):
        reader = wl.VectorSearchParamReader(object(), {
            "index": "idx", "k": 10, "target_field_name": "emb", "search_mode": mode,
            "ndcg_summary_file": os.path.join(self.tmp, f"{mode}.json"),
            "qrel_id_separator": "#", **extra,
        })
        return reader.params()

    def test_vector_mode_unchanged_shape(self):
        p = self._params("vector")
        self.assertIn("knn", p["body"]["query"])
        self.assertNotIn("request-params", p)
        self.assertEqual(p["relevance"]["qrels"], {"msmarco_v2.1_doc_00_1": 2})

    def test_hybrid_sends_pipeline_as_request_param(self):
        p = self._params("hybrid", search_pipeline="my-pipe")
        self.assertEqual(p["request-params"], {"search_pipeline": "my-pipe"})
        self.assertIn("hybrid", p["body"]["query"])

    def test_bm25_has_no_knn(self):
        body = json.dumps(self._params("bm25")["body"])
        self.assertNotIn("knn", body)
        self.assertIn("multi_match", body)

    def test_openrag_shape_matches_search_service(self):
        should = self._params("openrag-hybrid")["body"]["query"]["bool"]["should"]
        self.assertEqual(should[0]["dis_max"]["boost"], 0.7)
        self.assertEqual(should[1]["multi_match"]["boost"], 0.3)
        self.assertEqual(should[1]["multi_match"]["fuzziness"], "AUTO:4,7")
        self.assertEqual(should[2]["match_phrase_prefix"]["text"]["boost"], 0.25)

    def test_efficient_filter_goes_inside_knn_and_bool(self):
        f = {"term": {"domain": "site0.com"}}
        knn = self._params("vector", filter_body=f)["body"]["query"]["knn"]["emb"]
        self.assertEqual(knn["filter"], f)
        bm25 = self._params("bm25", filter_body=f)["body"]["query"]["bool"]
        self.assertEqual(bm25["filter"], [f])

    def test_post_filter_mode(self):
        f = {"term": {"domain": "site0.com"}}
        body = self._params("vector", filter_body=f, filter_mode="post")["body"]
        self.assertEqual(body["post_filter"], f)
        self.assertNotIn("filter", body["query"]["knn"]["emb"])

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            self._params("nope")


if __name__ == "__main__":
    unittest.main(verbosity=2)
