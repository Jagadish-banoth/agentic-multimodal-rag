"""evaluation/metrics.py

Lightweight evaluation metrics used by scripts in `evaluation/`.

Notes:
- Keep this module dependency-light (std lib only) so CI can run it.
- `retrieval/metrics.py` contains richer tracking for production telemetry.
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, Iterable, List, Mapping, Sequence


def recall_at_k(topk: Iterable[str], gold: str) -> float:
    """Return 1.0 if gold in topk else 0.0"""
    return 1.0 if gold in set(topk) else 0.0


def mrr_at_k(topk: Iterable[str], gold: str) -> float:
    """Mean Reciprocal Rank for a single query and a ranked list topk."""
    for rank, cid in enumerate(topk, start=1):
        if cid == gold:
            return 1.0 / rank
    return 0.0


def recall_hit_at_k(retrieved: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Hit-style Recall@K: 1 if any relevant doc appears in top-k, else 0."""
    if k <= 0 or not gold_ids:
        return 0.0
    topk = set(retrieved[:k])
    return 1.0 if any(g in topk for g in gold_ids) else 0.0


def recall_fraction_at_k(retrieved: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Fractional recall@K: |topk ∩ gold| / |gold| (multi-relevant)."""
    if k <= 0 or not gold_ids:
        return 0.0
    gold = set(gold_ids)
    topk = set(retrieved[:k])
    return len(topk & gold) / max(1, len(gold))


def mrr(retrieved: Sequence[str], gold_ids: Sequence[str]) -> float:
    """MRR for multi-relevant: 1 / rank(first relevant)."""
    gold = set(gold_ids)
    if not gold:
        return 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str],
    relevance: Mapping[str, float],
    k: int,
    *,
    log_base: float = 2.0,
) -> float:
    """nDCG@K with graded relevance.

    relevance: dict(doc_id -> rel), where rel is typically in [0..3] or [0..1].
    """
    if k <= 0 or not relevance:
        return 0.0

    def _dcg(ids: Sequence[str]) -> float:
        dcg = 0.0
        for i, doc_id in enumerate(ids[:k], start=1):
            rel = float(relevance.get(doc_id, 0.0))
            denom = math.log(i + 1, log_base)
            dcg += (2.0 ** rel - 1.0) / denom
        return dcg

    dcg = _dcg(retrieved)
    ideal_gains = sorted((float(v) for v in relevance.values()), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_gains[:k], start=1):
        denom = math.log(i + 1, log_base)
        idcg += (2.0 ** rel - 1.0) / denom
    return (dcg / idcg) if idcg > 0 else 0.0


def aggregate(metrics: Iterable[float]) -> float:
    """Compute mean of an iterable of metric values, return 0 for empty."""
    vals = list(metrics)
    return sum(vals) / len(vals) if vals else 0.0


def percentile(values: Iterable[float], p: float) -> float:
    """Compute percentile p in [0,100] using a simple linear interpolation."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return 0.0
    if p <= 0:
        return xs[0]
    if p >= 100:
        return xs[-1]
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


def summarize_latency_ms(latencies_ms: Sequence[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    return {
        "p50": percentile(latencies_ms, 50),
        "p95": percentile(latencies_ms, 95),
        "p99": percentile(latencies_ms, 99),
        "mean": statistics.fmean(latencies_ms),
    }
