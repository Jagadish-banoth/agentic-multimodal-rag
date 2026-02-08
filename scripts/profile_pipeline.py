"""End-to-end pipeline profiler.

Runs ExecutionEngine on a query set and writes latency breakdown + percentiles.

Usage:
  python scripts/profile_pipeline.py --queries test_queries.txt --out artifacts/pipeline_profile.json

Input:
- .txt: one query per line
- .csv: column `query`
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from evaluation.metrics import percentile
from orchestrator.execution_engine import ExecutionEngine, load_config


def load_queries(path: str) -> List[str]:
    p = Path(path)
    if p.suffix.lower() == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    if p.suffix.lower() == ".csv":
        out: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                q = (r.get("query") or "").strip()
                if q:
                    out.append(q)
        return out
    raise ValueError(f"Unsupported query file: {path}")


def summarize_stage(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    xs = [float(r.get("timings", {}).get(key, 0.0)) * 1000.0 for r in rows]
    return {
        "p50_ms": percentile(xs, 50),
        "p95_ms": percentile(xs, 95),
        "p99_ms": percentile(xs, 99),
        "mean_ms": sum(xs) / max(1, len(xs)),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Profile end-to-end pipeline latencies")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--out", default="artifacts/pipeline_profile.json")
    parser.add_argument("--phase1", action="store_true", help="Enable Phase 1 optimizations")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = load_config()
    try:
        engine = ExecutionEngine(cfg, enable_phase1=args.phase1)
    except Exception as e:
        # Common on machines/CI without Ollama/OpenRouter credentials.
        # Fall back to a minimal pipeline that still measures retrieval+fusion.
        from retrieval.dense_retriever import DenseRetriever
        from retrieval.sparse_retriever import SparseRetriever
        from retrieval.reranker import Reranker
        from retrieval.parallel_retriever import ParallelRetriever
        from fusion.context_fusion import ContextFusion

        dense = DenseRetriever(cfg)
        sparse = SparseRetriever(cfg)
        reranker = Reranker(cfg)
        parallel = ParallelRetriever(dense, sparse, reranker, query_expander=None, max_workers=4)
        fusion = ContextFusion(cfg)

        class _FallbackEngine:
            def run(self, query: str) -> Dict[str, Any]:
                timings: Dict[str, float] = {}
                t0 = time.perf_counter()
                results, rt = parallel.retrieve_parallel(
                    query,
                    dense_k=int(cfg.get("retrieval", {}).get("dense_k", 50)),
                    sparse_k=int(cfg.get("retrieval", {}).get("sparse_k", 50)),
                    rerank_k=int(cfg.get("retrieval", {}).get("rerank_k", 10)),
                    use_expansion=False,
                )
                timings.update({k: float(v) for k, v in rt.items() if isinstance(v, (int, float))})
                timings["retrieval_total"] = time.perf_counter() - t0

                t1 = time.perf_counter()
                _ctx, _map = fusion.fuse_with_mapping(results, query=query)
                timings["fusion"] = time.perf_counter() - t1

                timings["generation"] = 0.0
                timings["planning"] = 0.0
                return {"timings": timings, "confidence": 0.0, "verified": False, "from_cache": False}

        engine = _FallbackEngine()

    queries = load_queries(args.queries)
    if args.limit and args.limit > 0:
        queries = queries[: args.limit]

    rows: List[Dict[str, Any]] = []
    for q in queries:
        t0 = time.perf_counter()
        res = engine.run(q)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        rows.append(
            {
                "query": q,
                "wall_ms": wall_ms,
                "timings": res.get("timings", {}),
                "confidence": res.get("confidence"),
                "verified": res.get("verified"),
                "from_cache": res.get("from_cache"),
            }
        )

    totals = [r["wall_ms"] for r in rows]
    report = {
        "num_queries": len(rows),
        "total_ms": {
            "p50": percentile(totals, 50),
            "p95": percentile(totals, 95),
            "p99": percentile(totals, 99),
            "mean": sum(totals) / max(1, len(totals)),
        },
        "stages": {
            "planning": summarize_stage(rows, "planning"),
            "retrieval_total": summarize_stage(rows, "retrieval_total"),
            "reranking": summarize_stage(rows, "reranking"),
            "fusion": summarize_stage(rows, "fusion"),
            "generation": summarize_stage(rows, "generation"),
        },
        "rows": rows,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
