"""
Cross-encoder reranker for improving retrieval quality.

Uses a cross-encoder model to rerank candidate documents
based on query-document relevance.

Features timeout-based fallback to faster BGE reranker.
"""

import logging
import os
import time
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "settings.yaml"

def load_config() -> dict:
    """Load configuration from settings.yaml."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Reranker:
    """
    Cross-encoder reranker using sentence-transformers or transformers.
    
    Reranks retrieved candidates to improve precision at top-k.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize reranker.
        
        Args:
            config: Configuration dictionary (loads from settings.yaml if None)
        """
        if config is None:
            self.config = load_config()
        else:
            self.config = config
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # Get reranker settings from config
        reranker_cfg = self.config.get("reranker", {})
        self.model_name = reranker_cfg.get("model", "BAAI/bge-reranker-base")  # Fast default
        self.fallback_model_name = reranker_cfg.get("fallback_model", "BAAI/bge-reranker-base")
        self.timeout_seconds = reranker_cfg.get("timeout_seconds", 30)
        self.batch_size = reranker_cfg.get("batch_size", 32)
        self.top_n = reranker_cfg.get("top_n", 12)
        
        # Fallback model (loaded lazily if needed)
        self.fallback_model = None
        self.fallback_loaded = False

        # Setup reranker-specific file logger
        reranker_log_path = os.path.join(os.path.dirname(__file__), '../logs/reranker.log')
        reranker_log_path = os.path.abspath(reranker_log_path)
        self.reranker_logger = logging.getLogger("reranker_device")
        if not self.reranker_logger.hasHandlers():
            handler = logging.FileHandler(reranker_log_path, mode='a', encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.reranker_logger.addHandler(handler)
            self.reranker_logger.setLevel(logging.INFO)

        # Device selection - prefer CUDA for speed, fallback to CPU gracefully
        use_gpu = reranker_cfg.get("use_gpu", True)
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            try:
                msg = f"🚀 Reranker will use GPU: {torch.cuda.get_device_name(0)}"
            except (AssertionError, RuntimeError):
                msg = "🚀 Reranker will use GPU"
            logger.info(msg)
            self.reranker_logger.info(msg)
        else:
            self.device = "cpu"
            msg = "⚠️ Reranker will use CPU (slower but works without GPU)"
            logger.warning(msg)
            self.reranker_logger.warning(msg)

        # Try to load model
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load the reranker model."""
        try:
            # Try sentence-transformers CrossEncoder first
            from sentence_transformers import CrossEncoder
            
            self.model = CrossEncoder(
                self.model_name,
                max_length=512,
                device=self.device,
                trust_remote_code=True,
            )
            # Fix padding token for batch processing - must set on both tokenizer AND model config
            if self.model.tokenizer.pad_token is None:
                self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
                self.model.tokenizer.pad_token_id = self.model.tokenizer.eos_token_id
            # Critical: set pad_token_id on model config to avoid batch size error
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'config'):
                self.model.model.config.pad_token_id = self.model.tokenizer.pad_token_id
            
            self.is_loaded = True
            logger.info(f"✓ Loaded reranker: {self.model_name} on {self.device.upper()}")
            return True
            
        except ImportError:
            logger.warning("sentence-transformers not available for reranking")
        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder reranker: {e}")
        
        # Fallback: try transformers AutoModelForSequenceClassification
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            # Set padding token if not defined (required for batch processing)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            self._use_transformers = True
            logger.info(f"✓ Loaded reranker via transformers: {self.model_name}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}")
            self.is_loaded = False
            return False
    
    def _load_fallback_model(self) -> bool:
        """Load the faster BGE fallback reranker model."""
        if self.fallback_loaded:
            return self.fallback_model is not None
        
        try:
            from sentence_transformers import CrossEncoder
            
            self.fallback_model = CrossEncoder(
                self.fallback_model_name,
                max_length=512,
                device=self.device,
            )
            self.fallback_loaded = True
            logger.info(f"✓ Loaded fallback reranker: {self.fallback_model_name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load fallback reranker: {e}")
            self.fallback_loaded = True  # Mark as attempted
            self.fallback_model = None
            return False
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance.
        
        Uses primary model with timeout fallback to faster BGE reranker.
        
        Args:
            query: Search query
            documents: List of document dicts with 'content' field
            top_n: Number of top results to return
        
        Returns:
            Reranked list of documents with updated scores
        """
        if not self.is_loaded or not documents:
            return documents[:top_n] if top_n else documents
        
        n = top_n or self.top_n
        
        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            content = doc.get("content", "") or doc.get("text", "")
            if content:
                max_doc_len = 2000
                pairs.append((query, content[:max_doc_len]))
            else:
                pairs.append((query, ""))
        
        # Fast path: if primary and fallback are the same, skip timeout mechanism
        use_fast_path = (self.model_name == self.fallback_model_name)
        
        try:
            start_time = time.time()
            
            if use_fast_path:
                # Direct execution without timeout overhead
                scores = self._rerank_direct(pairs)
            else:
                # Use timeout mechanism for potentially slow models
                scores = self._rerank_with_timeout(pairs)
                if scores is None:
                    # Timeout occurred, use fallback
                    logger.warning(f"Primary reranker timed out ({self.timeout_seconds}s), using fallback BGE model")
                    scores = self._rerank_with_fallback(pairs)
            
            elapsed = time.time() - start_time
            
            if scores is None:
                logger.error("Reranking failed (no scores returned)")
                return documents[:n]
            
            logger.info(f"Reranker completed in {elapsed:.2f}s for {len(pairs)} pairs")
            
            # Add scores and sort
            scored_docs = []
            for i, doc in enumerate(documents):
                doc_copy = doc.copy()
                doc_copy["rerank_score"] = float(scores[i])
                scored_docs.append(doc_copy)
            
            scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
            logger.info(f"Reranked {len(documents)} → top {n}")
            return scored_docs[:n]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:n]
    
    def _rerank_direct(self, pairs: List[tuple]) -> Optional[List[float]]:
        """Run reranker directly without timeout overhead."""
        try:
            if hasattr(self, '_use_transformers') and self._use_transformers:
                return self._score_transformers(pairs)
            else:
                effective_batch = self.batch_size if self.device == "cuda" else 8
                return self.model.predict(
                    pairs,
                    batch_size=effective_batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
        except Exception as e:
            logger.error(f"Direct reranking failed: {e}")
            return None
    
    def _rerank_with_timeout(self, pairs: List[tuple]) -> Optional[List[float]]:
        """Run primary reranker with timeout. Returns None if timeout."""
        def _do_rerank():
            if hasattr(self, '_use_transformers') and self._use_transformers:
                return self._score_transformers(pairs)
            else:
                effective_batch = 16 if self.device == "cuda" else 4
                return self.model.predict(
                    pairs,
                    batch_size=effective_batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_do_rerank)
            try:
                return future.result(timeout=self.timeout_seconds)
            except concurrent.futures.TimeoutError:
                return None
    
    def _rerank_with_fallback(self, pairs: List[tuple]) -> Optional[List[float]]:
        """Use faster BGE fallback reranker."""
        if not self._load_fallback_model():
            return None
        
        try:
            scores = self.fallback_model.predict(
                pairs,
                batch_size=32,  # BGE can handle larger batches
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return scores
        except Exception as e:
            logger.error(f"Fallback reranker failed: {e}")
            return None
    
    def _score_transformers(self, pairs: List[tuple]) -> List[float]:
        """Score pairs using transformers model."""
        scores = []
        
        for query, doc in pairs:
            inputs = self.tokenizer(
                query,
                doc,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Get logits and convert to score
                logits = outputs.logits
                if logits.shape[-1] == 1:
                    score = logits.squeeze().item()
                else:
                    score = torch.softmax(logits, dim=-1)[0, 1].item()
                scores.append(score)
        
        return scores
    
    def is_available(self) -> bool:
        """Check if reranker is available."""
        return self.is_loaded
