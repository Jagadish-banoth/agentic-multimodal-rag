"""Drift/regression checker for retrieval metrics.

Compares a new retrieval benchmark report against a baseline and exits non-zero
if metrics regress beyond tolerance.

Usage:
  python scripts/drift_check.py --baseline artifacts/baseline_retrieval.json --current artifacts/retrieval_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional, Sequence


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Check retrieval drift")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--retriever", default="hybrid_rerank")
    parser.add_argument("--max_drop", type=float, default=0.03, help="Max allowed absolute metric drop")
    args = parser.parse_args(argv)

    base = _load(args.baseline)
    cur = _load(args.current)

    b = ((base.get("retrievers") or {}).get(args.retriever) or {}).get("metrics_mean") or {}
    c = ((cur.get("retrievers") or {}).get(args.retriever) or {}).get("metrics_mean") or {}

    keys = ["recall_hit@5", "mrr", "ndcg@10"]
    violations = []
    for k in keys:
        bv = float(b.get(k, 0.0))
        cv = float(c.get(k, 0.0))
        if bv - cv > args.max_drop:
            violations.append(f"{k} dropped {bv:.3f} -> {cv:.3f} (drop={bv-cv:.3f})")

    if violations:
        print("DRIFT DETECTED:")
        for v in violations:
            print("-", v)
        sys.exit(2)

    print("NO DRIFT")


if __name__ == "__main__":
    main()
