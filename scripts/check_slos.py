"""SLO gate checker.

Reads benchmark artifacts and fails (exit code 2) if SLOs are violated.

Typical inputs:
- artifacts/retrieval_benchmark.json (from evaluation/retrieval_benchmark.py)
- artifacts/pipeline_profile.json (from scripts/profile_pipeline.py)
- artifacts/load_test.json (from scripts/load_test_local.py)

Usage:
  python scripts/check_slos.py --slo evaluation/slo.yaml \
      --retrieval artifacts/retrieval_benchmark.json \
      --profile artifacts/pipeline_profile.json \
      --load artifacts/load_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get(d: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    try:
        return float(cur)
    except Exception:
        return default


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Check SLO gates")
    parser.add_argument("--slo", default="evaluation/slo.yaml")
    parser.add_argument("--retrieval", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--load", default=None)
    parser.add_argument("--retriever", default="hybrid_rerank", help="Which retriever to gate on")
    args = parser.parse_args(argv)

    slo = _load_yaml(args.slo)
    violations = []

    if args.retrieval:
        r = _load_json(args.retrieval)
        rr = (r.get("retrievers") or {}).get(args.retriever) or {}
        mean = rr.get("metrics_mean") or {}
        lat = rr.get("latency_ms") or {}

        min_recall_hit_at_5 = float(slo.get("retrieval", {}).get("min_recall_hit_at_5", 0.0))
        min_mrr = float(slo.get("retrieval", {}).get("min_mrr", 0.0))
        min_ndcg_at_10 = float(slo.get("retrieval", {}).get("min_ndcg_at_10", 0.0))
        max_retrieval_p95_ms = float(slo.get("latency", {}).get("max_retrieval_p95_ms", 1e9))

        recall_hit_5 = float(mean.get("recall_hit@5", 0.0))
        mrr = float(mean.get("mrr", 0.0))
        ndcg10 = float(mean.get("ndcg@10", 0.0))
        p95 = float(lat.get("p95", 0.0))

        if recall_hit_5 < min_recall_hit_at_5:
            violations.append(f"retrieval recall_hit@5 {recall_hit_5:.3f} < {min_recall_hit_at_5:.3f}")
        if mrr < min_mrr:
            violations.append(f"retrieval mrr {mrr:.3f} < {min_mrr:.3f}")
        if ndcg10 < min_ndcg_at_10:
            violations.append(f"retrieval ndcg@10 {ndcg10:.3f} < {min_ndcg_at_10:.3f}")
        if p95 > max_retrieval_p95_ms:
            violations.append(f"retrieval p95_ms {p95:.1f} > {max_retrieval_p95_ms:.1f}")

    if args.profile:
        p = _load_json(args.profile)
        max_pipeline_p95_ms = float(slo.get("latency", {}).get("max_pipeline_p95_ms", 1e9))
        p95 = _get(p, "total_ms", "p95", default=0.0)
        if p95 > max_pipeline_p95_ms:
            violations.append(f"pipeline p95_ms {p95:.1f} > {max_pipeline_p95_ms:.1f}")

    if args.load:
        l = _load_json(args.load)
        min_qps = float(slo.get("load", {}).get("min_qps", 0.0))
        qps = float(l.get("qps", 0.0))
        if qps < min_qps:
            violations.append(f"load qps {qps:.3f} < {min_qps:.3f}")

    if violations:
        print("SLO VIOLATIONS:")
        for v in violations:
            print("-", v)
        sys.exit(2)

    print("SLO OK")


if __name__ == "__main__":
    main()
