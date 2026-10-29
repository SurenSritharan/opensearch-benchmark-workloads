"""Relevance (qrels) helpers for the parquet workload.

Recall@k measures agreement with exact kNN ground truth. NDCG@10 measures agreement
with human relevance judgments (qrels), which is the right quality metric once
BM25 or hybrid scoring can legitimately return different documents than kNN.

OSB only aggregates recall@k / recall@1 into its report, so NDCG is accumulated
here and written to a JSON summary next to the ground-truth file when the search
task finishes. One summary file per (index, search_mode).
"""

import atexit
import json
import logging
import math
import os
import threading

logger = logging.getLogger(__name__)


def qrel_doc_id(passage_id: str, sep: str) -> str:
    """Map a hit's passage id to the id the qrels are keyed on.

    MS MARCO v2.1 qrels are document-level (``msmarco_v2.1_doc_01_1281570012``)
    while passages are segments (``msmarco_v2.1_doc_01_1281570012#3_...``).
    With ``sep="#"`` the segment suffix is dropped; with ``sep=""`` ids pass through.
    """
    if sep and sep in passage_id:
        return passage_id.split(sep, 1)[0]
    return passage_id


def dedupe_preserving_order(ids):
    seen = set()
    out = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def ndcg_at_k(ranked_ids, qrels: dict, k: int = 10) -> float:
    """Graded NDCG@k with the standard 2^rel - 1 gain (as in BEIR / trec_eval).

    ``ranked_ids`` must already be mapped to qrel ids and de-duplicated, so a
    document split into many passages is only credited once.
    """
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k]):
        rel = qrels.get(doc_id, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(rank + 2)
    ideal = sorted((g for g in qrels.values() if g > 0), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(rank + 2) for rank, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


class RelevanceAccumulator:
    """Collects per-query NDCG from the search runner and writes a summary.

    Each search client runs in its own worker process, so every process writes its
    own part file (``<summary>.p<pid>.json``); ``merge_parts`` combines them.
    Repeated queries (time-period runs cycle through the set) are averaged per query
    first so every query carries equal weight.
    """

    def __init__(self, summary_path: str, meta: dict):
        self.summary_path = summary_path
        self.meta = meta
        self._lock = threading.Lock()
        self._per_query = {}
        atexit.register(self.flush)

    def add(self, query_id: str, ndcg: float) -> None:
        with self._lock:
            total, n = self._per_query.get(query_id, (0.0, 0))
            self._per_query[query_id] = (total + ndcg, n + 1)

    def flush(self) -> None:
        with self._lock:
            if not self._per_query:
                return
            per_query = {q: t / n for q, (t, n) in self._per_query.items()}
        part = f"{os.path.splitext(self.summary_path)[0]}.p{os.getpid()}.json"
        os.makedirs(os.path.dirname(os.path.abspath(part)), exist_ok=True)
        with open(part, "w") as f:
            json.dump({**self.meta, "per_query_ndcg@10": per_query}, f)
        merge_parts(self.summary_path)


def merge_parts(summary_path: str) -> dict:
    """Merge every part file for ``summary_path`` into the summary and return it."""
    base = os.path.splitext(summary_path)[0]
    directory = os.path.dirname(os.path.abspath(summary_path)) or "."
    per_query, meta = {}, {}
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if path.startswith(base + ".p") and name.endswith(".json"):
            with open(path) as f:
                part = json.load(f)
            per_query.update(part.pop("per_query_ndcg@10"))
            meta = part
    if not per_query:
        return {}
    scores = sorted(per_query.values())
    summary = {
        **meta,
        "queries_scored": len(scores),
        "ndcg@10_mean": sum(scores) / len(scores),
        "ndcg@10_median": scores[len(scores) // 2],
        "per_query_ndcg@10": per_query,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "[relevance] %s: NDCG@10 mean=%.4f over %d queries -> %s",
        meta.get("search_mode"), summary["ndcg@10_mean"], len(scores), summary_path,
    )
    return summary
