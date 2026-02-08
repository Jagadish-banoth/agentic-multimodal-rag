"""Ad-hoc local search demo.

Note: This file lives under tests/ but is intended as a runnable script.
It is safe to import (no side effects) so CI/pytest won't accidentally run it.
"""


def main() -> None:
    from pathlib import Path
    import json

    import faiss
    import numpy as np

    from utils.model_loader import get_embedding_model

    root = Path(__file__).resolve().parents[1]
    idx_dir = root / "data" / "index"

    meta = [json.loads(l) for l in open(idx_dir / "meta.jsonl", encoding="utf-8")]
    index = faiss.read_index(str(idx_dir / "faiss.index"))
    model = get_embedding_model()

    query = "Explain the Self-Attention"
    q_emb = model.encode(query, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb.reshape(1, -1))

    distances, ids = index.search(q_emb.reshape(1, -1), 10)
    print("Top ids:", ids[0].tolist())
    print("Scores:", distances[0].tolist())
    for idx in ids[0]:
        print(meta[idx]["chunk_id"], meta[idx]["snippet"][:200].replace("\n", " "))
        print("---")


if __name__ == "__main__":
    main()
