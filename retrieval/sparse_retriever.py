"""
Sparse retrieval using BM25.

This module provides keyword-based retrieval using the BM25 algorithm,
complementing dense semantic search for hybrid retrieval.
"""
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import os
log_path = os.path.join(os.path.dirname(__file__), '../logs/sparse_retriever.log')
log_path = os.path.abspath(log_path)
logger = logging.getLogger("sparse_retriever")
if not logger.hasHandlers():
    handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index"


class SparseRetriever:
    """
    BM25-based sparse retriever for keyword matching.
    
    Complements dense retrieval by capturing exact term matches
    that semantic embeddings might miss.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize sparse retriever.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.bm25 = None
        self.corpus = None
        self.metadata: List[Dict] = []
        self.is_loaded = False
        
        # Get paths from config or use defaults
        index_dir = self.config.get("index_dir", INDEX_DIR)
        if isinstance(index_dir, str):
            index_dir = Path(index_dir)
        
        self.index_path = index_dir / "bm25_index.pkl"
        self.corpus_path = index_dir / "bm25_corpus.pkl"
        self.meta_path = index_dir / "meta.jsonl"
        
        # Retrieval settings
        retrieval_cfg = self.config.get("retrieval", {})
        self.default_k = retrieval_cfg.get("sparse_k", 30)
        
        # Auto-load if index exists
        if self.index_path.exists():
            self._load_index()
    
    def _load_index(self) -> bool:
        """Load BM25 index and metadata."""
        try:
            # Load BM25 index
            with open(self.index_path, "rb") as f:
                self.bm25 = pickle.load(f)
            logger.info("✓ Loaded BM25 index")
            
            # Load tokenized corpus
            if self.corpus_path.exists():
                with open(self.corpus_path, "rb") as f:
                    self.corpus = pickle.load(f)
                logger.info(f"✓ Loaded tokenized corpus ({len(self.corpus)} documents)")
            
            # Load metadata
            if self.meta_path.exists():
                self.metadata = []
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            self.metadata.append(json.loads(line))
                logger.info(f"✓ Loaded {len(self.metadata)} metadata records")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            self.is_loaded = False
            return False
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of results with scores and metadata
        """
        if not self.is_loaded or self.bm25 is None:
            logger.warning("BM25 index not loaded")
            return []
        
        k = top_k or self.default_k
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            logger.warning("Empty query after tokenization")
            return []
        
        # Get BM25 scores
        try:
            scores = self.bm25.get_scores(query_tokens)
        except Exception as e:
            logger.error(f"BM25 scoring failed: {e}")
            return []
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            idx = int(idx)
            score = float(scores[idx])
            
            if score <= 0:
                continue
            
            # Get metadata
            if idx < len(self.metadata):
                meta = self.metadata[idx].copy()
            else:
                meta = {"index": idx}
                logger.warning(f"Index {idx} out of bounds for metadata")
            
            result = {
                "index": idx,
                "score": score,
                "content": meta.get("content", ""),
                "modality": meta.get("modality", "text"),
                "source": meta.get("source", "unknown"),
                "chunk_id": meta.get("chunk_id", f"chunk_{idx}"),
                "metadata": meta,
            }
            results.append(result)
        
        logger.info(f"Sparse BM25: retrieved {len(results)} results (top-{k})")
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Args:
            text: Input text
        
        Returns:
            List of tokens
        """
        # Simple whitespace tokenization with lowercasing
        # Could be enhanced with stemming, stopword removal, etc.
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        return {
            "is_loaded": self.is_loaded,
            "num_documents": len(self.corpus) if self.corpus else 0,
            "num_metadata": len(self.metadata),
            "index_path": str(self.index_path),
        }