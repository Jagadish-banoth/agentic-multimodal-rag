"""FAISS/HNSW scaling stress test.

This is a *synthetic* benchmark to approximate large-corpus scaling.
It builds an HNSW index with random vectors and measures query latency.

Usage:
  python scripts/stress_test_faiss_scaling.py --n 200000 --dim 1024 --m 32 --ef 100

For millions of docs, ensure you have enough RAM.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Optional, Sequence

import numpy as np


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Synthetic FAISS scaling benchmark")
    parser.add_argument("--n", type=int, default=200_000, help="Number of vectors (docs)")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--q", type=int, default=1000, help="Number of queries")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m", type=int, default=32, help="HNSW M")
    parser.add_argument("--ef", type=int, default=100, help="HNSW efSearch")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)

    import faiss  # type: ignore

    rng = np.random.default_rng(args.seed)
    xb = rng.standard_normal((args.n, args.dim), dtype=np.float32)
    xq = rng.standard_normal((args.q, args.dim), dtype=np.float32)

    # normalize for cosine-ish similarity using inner product
    faiss.normalize_L2(xb)
    faiss.normalize_L2(xq)

    index = faiss.IndexHNSWFlat(args.dim, args.m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = args.ef

    t0 = time.perf_counter()
    index.add(xb)
    build_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    _D, _I = index.search(xq, args.k)
    search_s = time.perf_counter() - t1

    report = {
        "n": args.n,
        "dim": args.dim,
        "q": args.q,
        "k": args.k,
        "m": args.m,
        "ef": args.ef,
        "build_s": build_s,
        "search_s": search_s,
        "avg_query_ms": (search_s / max(1, args.q)) * 1000.0,
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
