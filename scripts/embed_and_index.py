import argparse
import json
import logging
import pickle
from pathlib import Path
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import faiss
import numpy as np
import yaml
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from utils.dual_embedder import create_dual_embedder_from_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = ROOT / "data/processed/chunks.jsonl"
OUT_DIR = ROOT / "data/index"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_chunks(path: Path) -> list:
    """Load chunks from JSONL file."""
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def build_metadata(chunk: dict) -> dict:
    """Build metadata dict preserving all relevant fields."""
    modality = chunk.get("modality", "text")
    
    # Get content based on modality - ALWAYS prioritize 'content' field first
    # The 'content' field contains the combined/fused content ready for embedding
    if modality == "image":
        content = chunk.get("content") or chunk.get("combined_content") or chunk.get("caption") or ""
    elif modality == "video":
        # Video: prioritize 'content' (has visual+audio combined), then audio_transcript, then frame_caption
        content = chunk.get("content") or chunk.get("audio_transcript") or chunk.get("frame_caption") or chunk.get("frame_ocr") or ""
    elif modality == "audio":
        content = chunk.get("content") or chunk.get("transcript") or chunk.get("audio_transcript") or ""
    else:
        content = chunk.get("content", "")
    
    meta = {
        "chunk_id": chunk["chunk_id"],
        "modality": modality,
        "source": chunk.get("source", chunk.get("source_file", "")),
        "source_file": chunk.get("source_file", ""),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "snippet": content[:500],  # Longer snippet for better context
        "content": content,  # CRITICAL: Preserve full content for fusion/generation
    }
    
    # Modality-specific metadata
    if modality == "text":
        meta.update({
            "token_count": chunk.get("token_count"),
            "char_start": chunk.get("char_start"),
            "char_end": chunk.get("char_end"),
        })
    elif modality == "image":
        meta.update({
            "image_path": chunk.get("image_path", ""),
            "ocr_text": chunk.get("ocr_text", ""),
            "caption": chunk.get("caption", ""),
        })
    elif modality == "audio":
        meta.update({
            "transcript": chunk.get("transcript", ""),
            "duration_seconds": chunk.get("duration_seconds"),
            "language": chunk.get("language", "en"),
        })
    elif modality == "video":
        meta.update({
            "frame_path": chunk.get("frame_path", ""),
            "frame_idx": chunk.get("frame_idx"),
            "timestamp_seconds": chunk.get("timestamp_seconds"),
            "timestamp_formatted": chunk.get("timestamp_formatted", ""),
            "frame_caption": chunk.get("frame_caption", ""),
            "frame_ocr": chunk.get("frame_ocr", ""),
            "audio_transcript": chunk.get("audio_transcript", ""),
        })
    
    return meta


def tokenize_corpus(texts: list) -> list:
    """Tokenize corpus for BM25."""
    tokenized = []
    for text in tqdm(texts, desc="Tokenizing for BM25"):
        tokens = text.lower().split()
        tokenized.append(tokens)
    return tokenized


def build_bm25_index(tokenized_corpus: list) -> BM25Okapi:
    """Build BM25 index from tokenized corpus."""
    logger.info(f"Building BM25 index from {len(tokenized_corpus)} documents...")
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info("✓ Built BM25 index")
    return bm25


def build_hnsw_index(
    embeddings: np.ndarray,
    dim: int,
    n_docs: int,
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 100,
) -> faiss.Index:
    """Build HNSW index optimized for dataset size."""
    
    # Auto-tune if small dataset
    if n_docs < 1000:
        M = min(M, 32)
        ef_construction = min(ef_construction, 100)
    
    logger.info(f"Building HNSW index: {n_docs} vectors, {dim}-dim")
    logger.info(f"  M={M}, ef_construction={ef_construction}, ef_search={ef_search}")
    
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(embeddings)
    
    return index


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Embed chunks and build dual FAISS indices (BAAI + CLIP) + BM25"
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "settings.yaml"),
        help="Path to config YAML",
    )
    parser.add_argument(
        "--chunks",
        default=str(CHUNKS_FILE),
        help="Path to chunks.jsonl",
    )
    parser.add_argument(
        "--index-type",
        choices=["hnsw", "ivf", "auto"],
        default="hnsw",
        help="FAISS index type",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    chunks_path = Path(args.chunks)
    
    # ===== LOAD DATA =====
    logger.info("=" * 70)
    logger.info("DUAL INDEX PIPELINE: BAAI (text) + CLIP (image) + BM25")
    logger.info("=" * 70)
    
    logger.info(f"\nLoading chunks from {chunks_path}")
    chunks = load_chunks(chunks_path)
    
    if not chunks:
        raise SystemExit(f"No chunks found at {chunks_path}")
    
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Count modalities
    modality_counts = {}
    for c in chunks:
        mod = c.get("modality", "text")
        modality_counts[mod] = modality_counts.get(mod, 0) + 1
    logger.info(f"Modality distribution: {modality_counts}")
    
    # Extract text content and build metadata
    texts = []
    for c in chunks:
        if c.get("modality") == "image":
            text = c.get("combined_content") or c.get("caption") or ""
        elif c.get("modality") == "audio":
            text = c.get("transcript") or ""
        elif c.get("modality") == "video":
            text = " ".join(filter(None, [
                c.get("frame_caption", ""),
                c.get("frame_ocr", ""),
                c.get("audio_transcript", ""),
            ]))
        else:
            text = c.get("content", "")
        texts.append(text)
    
    meta = [build_metadata(c) for c in chunks]
    
    # ===== LOAD CONFIG =====
    cfg = load_config(args.config)

    indexing_cfg = cfg.get("indexing", {}) or {}
    strategy = (indexing_cfg.get("strategy") or "dual").lower()

    # Only build/require CLIP if we actually have image chunks.
    has_image_chunks = modality_counts.get("image", 0) > 0
    build_image_index = has_image_chunks and strategy != "text_only"

    if not has_image_chunks:
        logger.info("No image chunks detected → skipping CLIP and image index")
    elif strategy == "text_only":
        logger.info("indexing.strategy=text_only → skipping CLIP and image index")

    faiss_cfg = cfg.get("faiss", {}).get("hnsw", {})
    M = faiss_cfg.get("M", 32)
    ef_construction = faiss_cfg.get("ef_construction", 200)
    ef_search = faiss_cfg.get("ef_search", 100)
    
    # ===== PHASE 1: SPARSE INDEX (BM25) =====
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: BUILDING SPARSE INDEX (BM25)")
    logger.info("=" * 70)
    
    t_bm25_start = time.time()
    tokenized_corpus = tokenize_corpus(texts)
    bm25 = build_bm25_index(tokenized_corpus)
    t_bm25 = time.time() - t_bm25_start
    
    # Save BM25 artifacts
    with open(OUT_DIR / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    logger.info(f"✓ Saved BM25 index")
    
    with open(OUT_DIR / "bm25_corpus.pkl", "wb") as f:
        pickle.dump(tokenized_corpus, f)
    logger.info(f"✓ Saved BM25 corpus")
    
    # ===== PHASE 2: LOAD DUAL EMBEDDER =====
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: LOADING DUAL EMBEDDER")
    logger.info("=" * 70)
    
    # Create embedder; enable_image only if we will build an image index.
    # (Text embedding is always required for dense text retrieval.)
    embedder = create_dual_embedder_from_config({
        **cfg,
        "indexing": {**(cfg.get("indexing", {}) or {}), "strategy": ("dual" if build_image_index else "text_only")},
    })
    embedder_info = embedder.get_info()
    
    logger.info(f"✓ Text model:  {embedder_info['text_model']} ({embedder_info['text_dim']}-dim)")
    logger.info(f"✓ Image model: {embedder_info['image_model']} ({embedder_info['image_dim']}-dim)")
    logger.info(f"✓ Device: {embedder_info['device']}")

    # Batch sizes (override in config if needed)
    default_text_bs = 64 if str(embedder_info.get("device", "")).lower().startswith("cuda") else 32
    default_image_bs = 64 if str(embedder_info.get("device", "")).lower().startswith("cuda") else 32
    text_batch_size = int(indexing_cfg.get("text_batch_size", default_text_bs))
    image_text_batch_size = int(indexing_cfg.get("image_text_batch_size", default_image_bs))
    
    # ===== PHASE 3: BUILD TEXT INDEX (BAAI) =====
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: BUILDING TEXT INDEX (BAAI/bge-m3, 1024-dim)")
    logger.info("=" * 70)
    
    t_text_start = time.time()

    # Batch encode all chunk texts with BAAI/bge-m3 (huge speedup vs per-chunk calls)
    logger.info(f"Embedding {len(texts)} chunks with BAAI in batches (batch_size={text_batch_size})")
    text_embeddings = embedder.embed_text(
        texts,
        batch_size=text_batch_size,
        show_progress=True,
    )
    text_embeddings = np.asarray(text_embeddings, dtype=np.float32)
    t_text_embed = time.time() - t_text_start
    
    logger.info(f"Text embeddings shape: {text_embeddings.shape}")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(text_embeddings)
    
    # Build HNSW index
    t_text_index_start = time.time()
    text_index = build_hnsw_index(
        text_embeddings,
        dim=embedder.get_text_dim(),
        n_docs=len(chunks),
        M=M,
        ef_construction=ef_construction,
        ef_search=ef_search,
    )
    t_text_index = time.time() - t_text_index_start
    
    # Save text index
    faiss.write_index(text_index, str(OUT_DIR / "faiss_text.index"))
    np.save(OUT_DIR / "embeddings_text.npy", text_embeddings)
    logger.info(f"✓ Saved BAAI text index ({text_index.ntotal} vectors, {embedder.get_text_dim()}-dim)")
    
    # ===== PHASE 4: BUILD IMAGE INDEX (CLIP) [OPTIONAL] =====
    t_image_embed = 0.0
    t_image_index = 0.0
    image_index = None
    image_embeddings = None
    if build_image_index:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 4: BUILDING IMAGE INDEX (CLIP, 768-dim)")
        logger.info("=" * 70)

        t_image_start = time.time()

        # Batch encode CLIP text embeddings for all chunks (used for cross-modal search).
        logger.info(
            f"Embedding {len(texts)} chunks with CLIP-text in batches (batch_size={image_text_batch_size})"
        )
        image_embeddings = embedder.embed_image_text(
            texts,
            batch_size=image_text_batch_size,
            show_progress=True,
        )
        image_embeddings = np.asarray(image_embeddings, dtype=np.float32)

        # For true image chunks with a valid image file, override CLIP-text embedding with CLIP-image embedding.
        image_paths = []
        image_global_indices = []
        for i, c in enumerate(chunks):
            if c.get("modality") == "image":
                p = c.get("image_path")
                if p:
                    image_paths.append(p)
                    image_global_indices.append(i)

        if image_paths:
            logger.info(f"Embedding {len(image_paths)} actual images with CLIP (override embeddings)")
            img_embs, valid_local = embedder.embed_images(image_paths, show_progress=True)
            # valid_local are indices into image_paths
            for local_idx in valid_local:
                global_idx = image_global_indices[local_idx]
                image_embeddings[global_idx] = img_embs[local_idx]

        t_image_embed = time.time() - t_image_start

        logger.info(f"Image embeddings shape: {image_embeddings.shape}")

        # Normalize for cosine similarity
        faiss.normalize_L2(image_embeddings)

        # Build HNSW index
        t_image_index_start = time.time()
        image_index = build_hnsw_index(
            image_embeddings,
            dim=embedder.get_image_dim(),
            n_docs=len(chunks),
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
        t_image_index = time.time() - t_image_index_start

        # Save image index
        faiss.write_index(image_index, str(OUT_DIR / "faiss_image.index"))
        np.save(OUT_DIR / "embeddings_image.npy", image_embeddings)
        logger.info(
            f"✓ Saved CLIP image index ({image_index.ntotal} vectors, {embedder.get_image_dim()}-dim)"
        )
    
    # ===== PHASE 5: SAVE METADATA =====
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5: SAVING METADATA")
    logger.info("=" * 70)
    
    # Save metadata
    with open(OUT_DIR / "meta.jsonl", "w", encoding="utf-8") as fout:
        for m in meta:
            fout.write(json.dumps(m, ensure_ascii=False) + "\n")
    logger.info(f"✓ Saved {len(meta)} metadata records")
    
    # Save index map
    index_map = {i: meta[i]["chunk_id"] for i in range(len(meta))}
    with open(OUT_DIR / "index_map.json", "w", encoding="utf-8") as fout:
        json.dump(index_map, fout, indent=2, ensure_ascii=False)
    logger.info("✓ Saved index map")
    
    # Save manifest
    total_time = t_bm25 + t_text_embed + t_text_index + t_image_embed + t_image_index
    
    manifest = {
        "n_chunks": len(meta),
        "modality_counts": modality_counts,
        "indices": {
            "text": {
                "model": embedder_info["text_model"],
                "dim": embedder_info["text_dim"],
                "vectors": text_index.ntotal,
                "file": "faiss_text.index",
                "type": "hnsw",
            },
            "sparse": {
                "type": "bm25",
                "file": "bm25_index.pkl",
                "corpus_file": "bm25_corpus.pkl",
            },
        },
        "hnsw_config": {
            "M": M,
            "ef_construction": ef_construction,
            "ef_search": ef_search,
        },
        "timings": {
            "bm25_seconds": round(t_bm25, 2),
            "text_embedding_seconds": round(t_text_embed, 2),
            "text_indexing_seconds": round(t_text_index, 2),
            "image_embedding_seconds": round(t_image_embed, 2),
            "image_indexing_seconds": round(t_image_index, 2),
            "total_seconds": round(total_time, 2),
        },
        "created_at": time.time(),
    }

    if build_image_index and image_index is not None:
        manifest["indices"]["image"] = {
            "model": embedder_info["image_model"],
            "dim": embedder_info["image_dim"],
            "vectors": image_index.ntotal,
            "file": "faiss_image.index",
            "type": "hnsw",
        }
    
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as fout:
        json.dump(manifest, fout, indent=2)
    logger.info("✓ Saved manifest")
    
    # ===== SUMMARY =====
    logger.info("\n" + "=" * 70)
    logger.info("✅ DUAL INDEX BUILD COMPLETE")
    logger.info("=" * 70)
    logger.info(f"")
    logger.info(f"📊 SUMMARY")
    logger.info(f"   Total Chunks:     {len(chunks):,}")
    logger.info(f"   Modalities:       {modality_counts}")
    logger.info(f"")
    logger.info(f"📦 INDICES CREATED:")
    logger.info(f"   ├─ Text Index:    {embedder_info['text_model']}")
    logger.info(f"   │                 {text_index.ntotal} vectors × {embedder_info['text_dim']}-dim")
    logger.info(f"   │                 SOTA for text-to-text retrieval")
    logger.info(f"   │")
    if build_image_index and image_index is not None:
        logger.info(f"   ├─ Image Index:   {embedder_info['image_model']}")
        logger.info(f"   │                 {image_index.ntotal} vectors × {embedder_info['image_dim']}-dim")
        logger.info(f"   │                 SOTA for cross-modal retrieval")
        logger.info(f"   │")
    else:
        logger.info(f"   ├─ Image Index:   (skipped)")
        logger.info(f"   │                 No image chunks / text-only strategy")
        logger.info(f"   │")
    logger.info(f"   └─ Sparse Index:  BM25")
    logger.info(f"                     {len(tokenized_corpus)} documents")
    logger.info(f"")
    logger.info(f"⏱️  TIMINGS:")
    logger.info(f"   BM25 Build:       {t_bm25:.2f}s")
    logger.info(f"   Text Embedding:   {t_text_embed:.2f}s")
    logger.info(f"   Text Indexing:    {t_text_index:.2f}s")
    logger.info(f"   Image Embedding:  {t_image_embed:.2f}s")
    logger.info(f"   Image Indexing:   {t_image_index:.2f}s")
    logger.info(f"   Total:            {total_time:.2f}s")
    logger.info(f"")
    logger.info(f"📁 OUTPUT: {OUT_DIR}")
    logger.info(f"   ├─ faiss_text.index      (BAAI, 1024-dim)")
    logger.info(f"   ├─ embeddings_text.npy   (backup)")
    if build_image_index:
        logger.info(f"   ├─ faiss_image.index     (CLIP, 768-dim)")
        logger.info(f"   ├─ embeddings_image.npy  (backup)")
    logger.info(f"   ├─ bm25_index.pkl        (sparse)")
    logger.info(f"   ├─ bm25_corpus.pkl       (tokenized docs)")
    logger.info(f"   ├─ meta.jsonl            (metadata)")
    logger.info(f"   ├─ index_map.json        (ID mapping)")
    logger.info(f"   └─ manifest.json         (stats)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()