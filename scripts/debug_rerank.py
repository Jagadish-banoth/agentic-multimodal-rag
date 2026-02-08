"""Simple debug script for reranker validation.

Usage:
  python -m scripts.debug_rerank "your query here"

Prints: top-ranked chunk ids, scores and snippets so you can verify Jina reranker is used.
"""

import yaml
import json
import sys
from pathlib import Path
from retrieval.reranker import Reranker


def load_meta(n=500):
    meta_path = Path("data/index/meta.jsonl")
    if not meta_path.exists():
        print("meta.jsonl not found at data/index/meta.jsonl")
        return []
    out = []
    with open(meta_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            try:
                m = json.loads(line)
                out.append({
                    "chunk_id": m.get("chunk_id"),
                    "content": m.get("snippet", ""),
                    "source": m.get("source"),
                    "page_start": m.get("page_start"),
                    "page_end": m.get("page_end"),
                })
            except Exception:
                continue
    return out


def main():
    query = "popular pre-trained models in computer vision"
    if len(sys.argv) > 1:
        query = sys.argv[1]

    cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))
    reranker = Reranker(cfg)
    candidates = load_meta(500)
    if not candidates:
        print("No candidates loaded.")
        return

    ranked = reranker.rerank(query, candidates)

    print("Jina mode:", getattr(reranker, "jina_mode", False))
    print("Top results:\n")
    for i, c in enumerate(ranked[:10], start=1):
        print(f"{i}. chunk_id={c.get('chunk_id')} score={c.get('rerank_score'):.6f}")
        print(c.get("content", "")[:400].replace('\n',' '))
        print("---\n")


if __name__ == '__main__':
    main()
