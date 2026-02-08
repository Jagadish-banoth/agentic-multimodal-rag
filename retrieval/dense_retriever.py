"""
Dense retrieval using FAISS indices.

Supports dual-index architecture:
- Text index: BAAI/bge-m3 (1024-dim) for text-to-text retrieval
- Image index: CLIP (768-dim) for cross-modal retrieval
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

# Import embedding cache
try:
    from utils.embedding_cache import EmbeddingCache
    CACHE_AVAILABLE = True
except ImportError:
    EmbeddingCache = None
    CACHE_AVAILABLE = False

import os
log_path = os.path.join(os.path.dirname(__file__), '../logs/dense_retriever.log')
log_path = os.path.abspath(log_path)
logger = logging.getLogger("dense_retriever")
if not logger.hasHandlers():
    handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index"


class DenseRetriever:
    """
    FAISS-based dense retriever with dual-index support.
    
    Architecture:
    - Text queries → BAAI index (1024-dim)
    - Image queries → CLIP index (768-dim)
    - Cross-modal queries → Both indices with fusion
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize dense retriever.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.embedder = None
        self.text_index = None
        self.image_index = None
        self.metadata: List[Dict] = []
        self.is_loaded = False
        self.is_dual_mode = False
        
        # Get paths from config
        index_dir = self.config.get("index_dir", INDEX_DIR)
        if isinstance(index_dir, str):
            index_dir = Path(index_dir)
        
        # Index paths
        indexing_cfg = self.config.get("indexing", {})
        text_cfg = indexing_cfg.get("text_index", {})
        image_cfg = indexing_cfg.get("image_index", {})
        
        self.text_index_path = index_dir / text_cfg.get("file", "faiss_text.index")
        self.image_index_path = index_dir / image_cfg.get("file", "faiss_image.index")
        self.legacy_index_path = index_dir / "faiss.index"  # Backward compatibility
        self.meta_path = index_dir / "meta.jsonl"
        self.chunks_path = ROOT / "data" / "processed" / "chunks.jsonl"  # For full content
        
        # Retrieval settings
        retrieval_cfg = self.config.get("retrieval", {})
        self.default_k = retrieval_cfg.get("dense_k", 30)
        self.text_weight = retrieval_cfg.get("text_weight", 0.6)
        self.image_weight = retrieval_cfg.get("image_weight", 0.4)
        
        # Dimensions
        self.text_dim = text_cfg.get("dim", 1024)
        self.image_dim = image_cfg.get("dim", 768)
        
        # Auto-load
        self._load_indices()
        self._load_embedder()
        
        # Initialize embedding cache
        if CACHE_AVAILABLE:
            self.cache = EmbeddingCache(config or {})
        else:
            self.cache = None
            logger.info("Embedding cache unavailable")
    
    def _load_indices(self) -> bool:
        """Load FAISS indices."""
        try:
            # Prefer new dual-index mode when both are present.
            if self.text_index_path.exists() and self.image_index_path.exists():
                self.text_index = faiss.read_index(str(self.text_index_path))
                self.image_index = faiss.read_index(str(self.image_index_path))
                self.is_dual_mode = True
                logger.info(f"✓ Loaded text index: {self.text_index.ntotal} vectors, {self.text_dim}-dim")
                logger.info(f"✓ Loaded image index: {self.image_index.ntotal} vectors, {self.image_dim}-dim")

            # Text-only index (no CLIP index built)
            elif self.text_index_path.exists():
                self.text_index = faiss.read_index(str(self.text_index_path))
                self.image_index = None
                self.is_dual_mode = False
                logger.info(f"✓ Loaded text index: {self.text_index.ntotal} vectors, {self.text_dim}-dim")
                logger.info("Image index not found → running in text-only mode")
            
            # Fallback to legacy single index
            elif self.legacy_index_path.exists():
                self.text_index = faiss.read_index(str(self.legacy_index_path))
                self.is_dual_mode = False
                logger.info(f"✓ Loaded legacy index: {self.text_index.ntotal} vectors")
            
            else:
                logger.warning("No FAISS index found")
                return False
            
            # Load metadata
            if self.meta_path.exists():
                self.metadata = []
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            self.metadata.append(json.loads(line))
                logger.info(f"✓ Loaded {len(self.metadata)} metadata records")
            
            # Load chunk content (for full text)
            self.chunk_content = {}
            if self.chunks_path.exists():
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            chunk = json.loads(line)
                            chunk_id = chunk.get("chunk_id")
                            if chunk_id:
                                self.chunk_content[chunk_id] = chunk.get("content", "")
                logger.info(f"✓ Loaded {len(self.chunk_content)} chunk contents")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False
    
    def _load_embedder(self):
        """Load the dual embedder for query encoding."""
        try:
            from utils.dual_embedder import create_dual_embedder_from_config

            # If the image index is absent, force a text-only embedder to avoid
            # downloading/initializing CLIP unnecessarily.
            effective_config = self.config
            if self.image_index is None:
                effective_config = {
                    **(self.config or {}),
                    "indexing": {**((self.config or {}).get("indexing", {}) or {}), "strategy": "text_only"},
                }

            self.embedder = create_dual_embedder_from_config(effective_config)
            logger.info("✓ Loaded dual embedder for retrieval")
        except Exception as e:
            logger.warning(f"Failed to load dual embedder: {e}")
            self.embedder = None
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        modality: str = "auto",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using dense search.
        
        Args:
            query: Search query
            top_k: Number of results
            modality: "text", "image", "hybrid", or "auto"
        
        Returns:
            List of results with scores and metadata
        """
        if not self.is_loaded:
            logger.warning("Dense index not loaded")
            return []
        
        k = top_k or self.default_k
        
        if self.is_dual_mode and modality in ("auto", "hybrid"):
            return self._hybrid_retrieve(query, k)
        elif modality == "image" and self.image_index is not None:
            return self._image_retrieve(query, k)
        else:
            return self._text_retrieve(query, k)
    
    def _text_retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Retrieve using text index (BAAI)."""
        if self.embedder is None or self.text_index is None:
            return []
        
        # Try cache first
        query_embedding = None
        if self.cache and self.cache.enabled:
            query_embedding = self.cache.get_query_embedding(query, "text")
        
        if query_embedding is None:
            # Cache miss - compute embedding
            query_embedding = self.embedder.embed_text([query])[0]
            # Cache for future
            if self.cache and self.cache.enabled:
                self.cache.set_query_embedding(query, query_embedding, "text")
        
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        distances, indices = self.text_index.search(query_embedding, k)
        
        return self._format_results(distances[0], indices[0], "text")
    
    def _image_retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Retrieve using image index (CLIP)."""
        if self.embedder is None or self.image_index is None:
            return []
        
        # Embed query with CLIP text encoder
        query_embedding = self.embedder.embed_image_text([query])[0]
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        distances, indices = self.image_index.search(query_embedding, k)
        
        return self._format_results(distances[0], indices[0], "image")
    
    def _hybrid_retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval using both indices.
        
        Combines results from text and image indices with weighted fusion.
        """
        if self.embedder is None:
            return []
        
        # Get text embeddings
        text_emb = self.embedder.embed_text([query])[0].reshape(1, -1).astype(np.float32)
        
        # Get CLIP text embeddings for image search
        clip_emb = self.embedder.embed_image_text([query])[0].reshape(1, -1).astype(np.float32)
        
        results_map = {}
        
        # Text index search
        if self.text_index is not None:
            text_k = int(k * self.text_weight * 2)  # Get more for fusion
            distances, indices = self.text_index.search(text_emb, text_k)
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                idx = int(idx)
                score = 1.0 / (1.0 + float(dist))  # Convert distance to similarity
                
                if idx not in results_map:
                    results_map[idx] = {"text_score": 0, "image_score": 0}
                results_map[idx]["text_score"] = score
        
        # Image index search
        if self.image_index is not None:
            image_k = int(k * self.image_weight * 2)
            distances, indices = self.image_index.search(clip_emb, image_k)
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                idx = int(idx)
                score = 1.0 / (1.0 + float(dist))
                
                if idx not in results_map:
                    results_map[idx] = {"text_score": 0, "image_score": 0}
                results_map[idx]["image_score"] = score
        
        # Fuse scores
        results = []
        for idx, scores in results_map.items():
            combined_score = (
                self.text_weight * scores["text_score"] +
                self.image_weight * scores["image_score"]
            )
            
            # Get metadata
            if idx < len(self.metadata):
                meta = self.metadata[idx].copy()
            else:
                meta = {"index": idx}
            
            results.append({
                "index": idx,
                "score": combined_score,
                "text_score": scores["text_score"],
                "image_score": scores["image_score"],
                "content": self._get_content(meta),
                "modality": meta.get("modality", "text"),
                "source": meta.get("source", "unknown"),
                "chunk_id": meta.get("chunk_id", f"chunk_{idx}"),
                "metadata": meta,
                "index_type": "hybrid",
            })
        
        # Sort by combined score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Count modalities
        text_count = sum(1 for r in results[:k] if r.get("text_score", 0) > r.get("image_score", 0))
        image_count = k - text_count
        
        logger.info(f"Hybrid retrieval: {len(results[:k])} results (text={text_count}, image={image_count}, hybrid={k})")
        
        return results[:k]
    
    def _format_results(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
        index_type: str,
    ) -> List[Dict[str, Any]]:
        """Format search results."""
        results = []
        
        for dist, idx in zip(distances, indices):
            if idx < 0:
                continue
            
            idx = int(idx)
            score = 1.0 / (1.0 + float(dist))  # Convert distance to similarity
            
            # Get metadata
            if idx < len(self.metadata):
                meta = self.metadata[idx].copy()
            else:
                meta = {"index": idx}
            
            results.append({
                "index": idx,
                "score": score,
                "content": self._get_content(meta),
                "modality": meta.get("modality", "text"),
                "source": meta.get("source", "unknown"),
                "chunk_id": meta.get("chunk_id", f"chunk_{idx}"),
                "metadata": meta,
                "index_type": index_type,
            })
        
        logger.info(f"Dense retrieval: {len(results)} results")
        return results
    
    def _get_content(self, meta: Dict) -> str:
        """Get full content for a chunk, preferring chunks.jsonl over metadata."""
        chunk_id = meta.get("chunk_id", "")
        # Try to get from chunks.jsonl first
        if hasattr(self, 'chunk_content') and chunk_id in self.chunk_content:
            return self.chunk_content[chunk_id]
        # Fallback to metadata content or snippet
        return meta.get("content", "") or meta.get("snippet", "")
    
    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        stats = {
            "is_loaded": self.is_loaded,
            "is_dual_mode": self.is_dual_mode,
            "num_metadata": len(self.metadata),
        }
        
        if self.text_index is not None:
            stats["text_index"] = {
                "vectors": self.text_index.ntotal,
                "dim": self.text_dim,
            }
        
        if self.image_index is not None:
            stats["image_index"] = {
                "vectors": self.image_index.ntotal,
                "dim": self.image_dim,
            }
        
        return stats