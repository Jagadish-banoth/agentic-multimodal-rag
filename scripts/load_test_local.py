"""Local load/throughput test (no HTTP server required).

Simulates concurrent queries against ExecutionEngine.run().

Usage:
  python scripts/load_test_local.py --queries test_queries.txt --concurrency 4 --requests 50
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from evaluation.metrics import percentile
from orchestrator.execution_engine import ExecutionEngine, load_config


def load_queries(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Concurrent load test for local pipeline")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--phase1", action="store_true")
    parser.add_argument("--out", default="artifacts/load_test.json")
    args = parser.parse_args(argv)

    qs = load_queries(args.queries)
    if not qs:
        raise RuntimeError("No queries")

    cfg = load_config()
    try:
        engine = ExecutionEngine(cfg, enable_phase1=args.phase1)
    except Exception:
        # Fallback for machines without Ollama/OpenRouter configured.
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
                t0 = time.perf_counter()
                results, _rt = parallel.retrieve_parallel(
                    query,
                    dense_k=int(cfg.get("retrieval", {}).get("dense_k", 50)),
                    sparse_k=int(cfg.get("retrieval", {}).get("sparse_k", 50)),
                    rerank_k=int(cfg.get("retrieval", {}).get("rerank_k", 10)),
                    use_expansion=False,
                )
                _ctx, _map = fusion.fuse_with_mapping(results, query=query)
                ms = (time.perf_counter() - t0) * 1000.0
                return {"ok": True, "ms": ms, "from_cache": False}

        engine = _FallbackEngine()

    lat_ms: List[float] = []
    failures = 0

    def one(i: int) -> Dict[str, Any]:
        q = qs[i % len(qs)]
        t0 = time.perf_counter()
        res = engine.run(q)
        ms = (time.perf_counter() - t0) * 1000.0
        if isinstance(res, dict):
            return {"i": i, "ms": ms, "ok": bool(res.get("ok", True)), "from_cache": res.get("from_cache")}
        return {"i": i, "ms": ms, "ok": bool(res), "from_cache": None}

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(one, i) for i in range(args.requests)]
        for fut in as_completed(futs):
            try:
                r = fut.result()
                lat_ms.append(float(r["ms"]))
            except Exception:
                failures += 1

    elapsed = time.perf_counter() - t_start
    qps = (args.requests - failures) / max(1e-9, elapsed)

    report = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "failures": failures,
        "elapsed_s": elapsed,
        "qps": qps,
        "latency_ms": {
            "p50": percentile(lat_ms, 50),
            "p95": percentile(lat_ms, 95),
            "p99": percentile(lat_ms, 99),
            "mean": sum(lat_ms) / max(1, len(lat_ms)),
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
