"""Create a tiny processed chunks and index suitable for quick evaluation/tests.
This does NOT require any heavy model downloads; embeddings are synthetic and deterministic.
Run: python scripts/create_tiny_index.py
"""
from pathlib import Path
import json
import numpy as np
import faiss

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUT_PROC = ROOT / "data" / "processed"
DEFAULT_OUT_IDX = ROOT / "data" / "index"


def create_tiny_index(out_proc: Path | str = None, out_idx: Path | str = None):
    out_proc = Path(out_proc) if out_proc else DEFAULT_OUT_PROC
    out_idx = Path(out_idx) if out_idx else DEFAULT_OUT_IDX
    out_proc.mkdir(parents=True, exist_ok=True)
    out_idx.mkdir(parents=True, exist_ok=True)

    chunks = [
        {
            "chunk_id": "doc1_chunk",
            "content": "AI governance involves policy, regulation and oversight to ensure safe and ethical AI.",
            "source": "doc1.pdf",
            "page_start": 1,
            "page_end": 1,
            "token_count": 30,
            "char_start": 0,
            "char_end": 120,
            "meta": {"filename": "doc1.pdf"},
        },
        {
            "chunk_id": "doc2_chunk",
            "content": "Adopting AI responsibly requires data governance, transparency, and human oversight.",
            "source": "doc2.pdf",
            "page_start": 1,
            "page_end": 1,
            "token_count": 28,
            "char_start": 0,
            "char_end": 110,
            "meta": {"filename": "doc2.pdf"},
        },
        {
            "chunk_id": "doc3_chunk",
            "content": "Technical building blocks: data pipelines, compute infrastructure, model lifecycle.",
            "source": "doc3.pdf",
            "page_start": 1,
            "page_end": 1,
            "token_count": 25,
            "char_start": 0,
            "char_end": 95,
            "meta": {"filename": "doc3.pdf"},
        },
    ]

    # write chunks.jsonl
    with open(out_proc / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # meta jsonl and snippets
    meta = []
    for c in chunks:
        meta.append({
            "chunk_id": c["chunk_id"],
            "source": c["source"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "token_count": c["token_count"],
            "char_start": c["char_start"],
            "char_end": c["char_end"],
            "snippet": c["content"][:250],
        })
    with open(out_idx / "meta.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # synth embeddings (deterministic)
    rng = np.random.RandomState(42)
    emb = rng.rand(len(meta), 32).astype("float32")
    # normalize for cosine-sim
    faiss.normalize_L2(emb)
    np.save(out_idx / "embeddings.npy", emb)

    # faiss index
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    faiss.write_index(index, str(out_idx / "faiss.index"))

    # index_map
    index_map = {i: meta[i]["chunk_id"] for i in range(len(meta))}
    with open(out_idx / "index_map.json", "w", encoding="utf-8") as f:
        json.dump(index_map, f, indent=2, ensure_ascii=False)

    manifest = {"n_chunks": len(meta), "embedding_dim": d}
    with open(out_idx / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Tiny index written to {out_idx}; processed chunks in {out_proc}")


if __name__ == "__main__":
    create_tiny_index()
