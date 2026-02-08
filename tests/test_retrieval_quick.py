#!/usr/bin/env python3
"""Quick retrieval smoke-test: dense + sparse + rerank

Run: python scripts/test_retrieval_quick.py
"""
from pathlib import Path
import sys
# ensure UTF-8 stdout (Windows console often uses cp1252)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever
from retrieval.reranker import Reranker


def load_config():
    with open(ROOT / "config" / "settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    dense = DenseRetriever(cfg)
    sparse = SparseRetriever(cfg)
    reranker = Reranker(cfg)

    queries = [
        "What is the national AI strategy framework?",
        "What are AI enablers?",
        "How to ensure responsible AI?",
    ]

    for q in queries:
        print("\n=== QUERY:", q)
        d = dense.retrieve(q, top_k=5)
        s = sparse.retrieve(q, top_k=5)

        # merge (dense first, then sparse)
        candidates = []
        seen = set()
        for hit in d + s:
            if hit["chunk_id"] in seen:
                continue
            candidates.append(hit)
            seen.add(hit["chunk_id"])

        reranked = reranker.rerank(q, candidates)[:5]
        for r in reranked:
            score = r.get("rerank_score", r["score"])
            print(f"- score={score:.4f}  src={r['source']}")
            print(r["content"][:400].replace("\n", " "))
            print("---")


if __name__ == "__main__":
    main()
