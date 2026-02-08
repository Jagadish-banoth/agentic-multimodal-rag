"""
FAANG-Level Retrieval Module
============================

Production-grade retrieval with:
1. Full chunk content preservation (no truncation)
2. Contextual chunk expansion (retrieve neighboring chunks)
3. Multi-signal scoring (semantic + lexical + structural)
4. Score normalization and calibration
5. Intelligent re-ranking with full context
6. Query understanding and decomposition

This module wraps the existing retrievers and adds FAANG-level enhancements.
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger("faang_retriever")
if not logger.hasHandlers():
    log_path = ROOT / "logs" / "faang_retriever.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class FAANGRetriever:
    """
    FAANG-grade retrieval system with full chunk preservation.
    
    Key Features:
    1. FULL CONTENT RETRIEVAL - Never truncate chunks
    2. CONTEXTUAL EXPANSION - Include neighboring chunks for context
    3. MULTI-SIGNAL FUSION - Combine dense, sparse, and structural signals
    4. SCORE CALIBRATION - Normalize scores across different retrievers
    5. INTELLIGENT DEDUPLICATION - Merge overlapping content smartly
    """
    
    def __init__(
        self,
        dense_retriever: Any,
        sparse_retriever: Any,
        reranker: Any,
        config: Optional[Dict] = None
    ):
        """
        Initialize FAANG retriever.
        
        Args:
            dense_retriever: Dense (FAISS) retriever instance
            sparse_retriever: Sparse (BM25) retriever instance
            reranker: Cross-encoder reranker instance
            config: Configuration dictionary
        """
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.reranker = reranker
        self.config = config or {}
        
        # Load full chunks for context expansion
        self.chunks_by_id: Dict[str, Dict] = {}
        self.chunks_by_source: Dict[str, List[Dict]] = defaultdict(list)
        self._load_full_chunks()
        
        # Retrieval config
        retrieval_cfg = self.config.get("retrieval", {})
        self.dense_k = retrieval_cfg.get("dense_k", 30)
        self.sparse_k = retrieval_cfg.get("sparse_k", 30)
        self.rerank_k = retrieval_cfg.get("rerank_k", 15)
        
        # FAANG enhancements
        self.enable_context_expansion = True
        self.context_window = 1  # Include N chunks before/after
        self.min_content_length = 50  # Skip chunks with too little content
        self.score_calibration = True
        
        logger.info(f"✓ FAANGRetriever initialized with {len(self.chunks_by_id)} chunks")
    
    def _load_full_chunks(self) -> None:
        """Load all chunks with full content for context expansion."""
        chunks_path = ROOT / "data" / "processed" / "chunks.jsonl"
        meta_path = ROOT / "data" / "index" / "meta.jsonl"
        
        # Load from chunks.jsonl (primary source with full content)
        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            chunk_id = chunk.get("chunk_id")
                            if chunk_id:
                                self.chunks_by_id[chunk_id] = chunk
                                source = chunk.get("source", "unknown")
                                self.chunks_by_source[source].append(chunk)
                        except json.JSONDecodeError:
                            continue
            logger.info(f"✓ Loaded {len(self.chunks_by_id)} full chunks from chunks.jsonl")
        
        # Also load from meta.jsonl for any missing chunks
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            meta = json.loads(line)
                            chunk_id = meta.get("chunk_id")
                            if chunk_id and chunk_id not in self.chunks_by_id:
                                self.chunks_by_id[chunk_id] = meta
                                source = meta.get("source", "unknown")
                                self.chunks_by_source[source].append(meta)
                        except json.JSONDecodeError:
                            continue
        
        # Sort chunks by source and position for context expansion
        for source in self.chunks_by_source:
            self.chunks_by_source[source].sort(
                key=lambda c: (c.get("page_start", 0), c.get("char_start", 0))
            )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        expand_context: bool = True,
        use_sparse: bool = True
    ) -> List[Dict[str, Any]]:
        """
        FAANG-level retrieval with full chunk content.
        
        Args:
            query: User query
            top_k: Final number of results
            expand_context: Include neighboring chunks
            use_sparse: Use hybrid (dense + sparse) retrieval
        
        Returns:
            List of chunks with full content and calibrated scores
        """
        k = top_k or self.rerank_k
        
        # Step 1: Multi-signal retrieval
        logger.info(f"FAANG Retrieval: query='{query[:80]}...'")
        
        # Dense retrieval (semantic)
        dense_results = self.dense.retrieve(query, top_k=self.dense_k)
        logger.info(f"  Dense: {len(dense_results)} results")
        
        # Sparse retrieval (lexical)
        sparse_results = []
        if use_sparse and self.sparse:
            try:
                sparse_results = self.sparse.retrieve(query, top_k=self.sparse_k)
                logger.info(f"  Sparse: {len(sparse_results)} results")
            except Exception as e:
                logger.warning(f"  Sparse failed: {e}")
        
        # Step 2: Merge with RRF and score calibration
        merged = self._merge_and_calibrate(dense_results, sparse_results)
        logger.info(f"  Merged: {len(merged)} unique chunks")
        
        # Step 3: Ensure full content for all chunks
        merged = self._ensure_full_content(merged)
        
        # Step 4: Context expansion (include neighboring chunks)
        if expand_context and self.enable_context_expansion:
            merged = self._expand_context(merged)
            logger.info(f"  After context expansion: {len(merged)} chunks")
        
        # Step 5: Filter low-quality chunks
        merged = self._filter_quality(merged)
        
        # Step 6: Rerank with cross-encoder (using full content)
        if self.reranker and getattr(self.reranker, 'is_available', lambda: False)():
            # Prepare documents with full content for reranking
            for doc in merged:
                # Ensure content field has full text
                if not doc.get("content") or len(doc.get("content", "")) < self.min_content_length:
                    doc["content"] = self._get_full_content(doc)
            
            merged = self.reranker.rerank(query, merged, top_n=k * 2)
            logger.info(f"  Reranked: top {len(merged[:k])} of {len(merged)}")
        
        # Step 7: Final selection
        final_results = merged[:k]
        
        # Step 8: Add metadata for downstream use
        for i, result in enumerate(final_results):
            result["rank"] = i + 1
            result["retrieval_method"] = "faang_hybrid"
        
        logger.info(f"  Final: {len(final_results)} chunks returned")
        return final_results
    
    def _merge_and_calibrate(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict]
    ) -> List[Dict]:
        """
        Merge results with RRF and calibrate scores.
        
        Uses Reciprocal Rank Fusion with calibration for fair comparison.
        """
        K = 60  # RRF constant
        
        scores: Dict[str, float] = {}
        results_map: Dict[str, Dict] = {}
        
        # Dense contribution (calibrated)
        if dense_results:
            max_dense = max(r.get("score", 0) for r in dense_results) or 1.0
            for rank, result in enumerate(dense_results, 1):
                chunk_id = result.get("chunk_id")
                if not chunk_id:
                    continue
                
                # RRF score
                rrf = 1.0 / (K + rank)
                
                # Calibrated raw score (normalize to 0-1)
                raw_score = result.get("score", 0) / max_dense
                
                # Combined
                scores[chunk_id] = scores.get(chunk_id, 0) + rrf + (raw_score * 0.1)
                
                if chunk_id not in results_map:
                    results_map[chunk_id] = result.copy()
                    results_map[chunk_id]["dense_rank"] = rank
                    results_map[chunk_id]["dense_score"] = result.get("score", 0)
        
        # Sparse contribution (calibrated)
        if sparse_results:
            max_sparse = max(r.get("score", 0) for r in sparse_results) or 1.0
            for rank, result in enumerate(sparse_results, 1):
                chunk_id = result.get("chunk_id")
                if not chunk_id:
                    continue
                
                # RRF score
                rrf = 1.0 / (K + rank)
                
                # Calibrated raw score
                raw_score = result.get("score", 0) / max_sparse
                
                # Combined
                scores[chunk_id] = scores.get(chunk_id, 0) + rrf + (raw_score * 0.1)
                
                if chunk_id not in results_map:
                    results_map[chunk_id] = result.copy()
                results_map[chunk_id]["sparse_rank"] = rank
                results_map[chunk_id]["sparse_score"] = result.get("score", 0)
        
        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        merged = []
        for chunk_id in sorted_ids:
            result = results_map[chunk_id]
            result["fusion_score"] = scores[chunk_id]
            merged.append(result)
        
        return merged
    
    def _ensure_full_content(self, results: List[Dict]) -> List[Dict]:
        """
        Ensure each result has FULL chunk content (no truncation).
        """
        for result in results:
            chunk_id = result.get("chunk_id")
            
            # Get full content from our loaded chunks
            full_content = self._get_full_content(result)
            
            if full_content:
                result["content"] = full_content
                result["content_length"] = len(full_content)
            else:
                # Fallback to whatever we have
                content = result.get("content", "") or result.get("snippet", "")
                result["content"] = content
                result["content_length"] = len(content)
        
        return results
    
    def _get_full_content(self, result: Dict) -> str:
        """Get full content for a chunk, never truncated."""
        chunk_id = result.get("chunk_id")
        
        # Try chunks_by_id first (most complete)
        if chunk_id and chunk_id in self.chunks_by_id:
            chunk = self.chunks_by_id[chunk_id]
            return chunk.get("content", "") or chunk.get("snippet", "")
        
        # Try metadata content
        meta = result.get("metadata", {})
        content = meta.get("content", "") or meta.get("snippet", "")
        if content:
            return content
        
        # Fallback to result's own content
        return result.get("content", "") or result.get("snippet", "")
    
    def _expand_context(self, results: List[Dict]) -> List[Dict]:
        """
        Expand results with neighboring chunks for better context.
        
        This is crucial for queries that span chunk boundaries.
        """
        expanded_ids: Set[str] = set()
        expanded_results: List[Dict] = []
        
        for result in results:
            chunk_id = result.get("chunk_id")
            if not chunk_id or chunk_id in expanded_ids:
                continue
            
            # Add the original chunk
            expanded_results.append(result)
            expanded_ids.add(chunk_id)
            
            # Find neighboring chunks from same source
            source = result.get("source", "")
            if source and source in self.chunks_by_source:
                source_chunks = self.chunks_by_source[source]
                
                # Find position of current chunk
                current_idx = None
                for i, chunk in enumerate(source_chunks):
                    if chunk.get("chunk_id") == chunk_id:
                        current_idx = i
                        break
                
                if current_idx is not None:
                    # Add preceding chunks
                    for offset in range(1, self.context_window + 1):
                        prev_idx = current_idx - offset
                        if prev_idx >= 0:
                            prev_chunk = source_chunks[prev_idx]
                            prev_id = prev_chunk.get("chunk_id")
                            if prev_id and prev_id not in expanded_ids:
                                expanded_chunk = prev_chunk.copy()
                                expanded_chunk["is_context"] = True
                                expanded_chunk["context_type"] = "preceding"
                                expanded_chunk["fusion_score"] = result.get("fusion_score", 0) * 0.8
                                expanded_results.append(expanded_chunk)
                                expanded_ids.add(prev_id)
                    
                    # Add following chunks
                    for offset in range(1, self.context_window + 1):
                        next_idx = current_idx + offset
                        if next_idx < len(source_chunks):
                            next_chunk = source_chunks[next_idx]
                            next_id = next_chunk.get("chunk_id")
                            if next_id and next_id not in expanded_ids:
                                expanded_chunk = next_chunk.copy()
                                expanded_chunk["is_context"] = True
                                expanded_chunk["context_type"] = "following"
                                expanded_chunk["fusion_score"] = result.get("fusion_score", 0) * 0.8
                                expanded_results.append(expanded_chunk)
                                expanded_ids.add(next_id)
        
        # Re-sort by score
        expanded_results.sort(key=lambda x: x.get("fusion_score", 0), reverse=True)
        
        return expanded_results
    
    def _filter_quality(self, results: List[Dict]) -> List[Dict]:
        """Filter out low-quality chunks."""
        filtered = []
        
        for result in results:
            content = result.get("content", "")
            
            # Skip empty or very short chunks
            if not content or len(content.strip()) < self.min_content_length:
                continue
            
            # Skip chunks that are mostly whitespace or special characters
            alpha_ratio = sum(c.isalnum() for c in content) / max(len(content), 1)
            if alpha_ratio < 0.3:
                continue
            
            filtered.append(result)
        
        return filtered
    
    def retrieve_with_decomposition(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Advanced retrieval with query decomposition.
        
        Decomposes complex queries into sub-queries for better coverage.
        """
        k = top_k or self.rerank_k
        
        # Decompose query into key concepts
        sub_queries = self._decompose_query(query)
        
        all_results = []
        seen_ids: Set[str] = set()
        
        # Retrieve for each sub-query
        for sub_query in sub_queries:
            results = self.retrieve(sub_query, top_k=k, expand_context=False)
            for result in results:
                chunk_id = result.get("chunk_id")
                if chunk_id and chunk_id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(chunk_id)
        
        # Re-rank all results with original query
        if self.reranker and all_results:
            all_results = self.reranker.rerank(query, all_results, top_n=k)
        
        return all_results[:k]
    
    def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose a complex query into sub-queries.
        
        Simple heuristic decomposition (can be enhanced with LLM).
        """
        sub_queries = [query]  # Always include original
        
        # Split on common question connectors
        connectors = [" and ", " or ", ", and ", ", "]
        for conn in connectors:
            if conn in query.lower():
                parts = re.split(re.escape(conn), query, flags=re.IGNORECASE)
                for part in parts:
                    part = part.strip()
                    if len(part) > 10:  # Minimum meaningful length
                        sub_queries.append(part)
        
        return list(set(sub_queries))[:5]  # Limit to 5 sub-queries
    
    def get_stats(self) -> Dict[str, Any]:
        """Return retriever statistics."""
        return {
            "total_chunks": len(self.chunks_by_id),
            "total_sources": len(self.chunks_by_source),
            "context_expansion_enabled": self.enable_context_expansion,
            "context_window": self.context_window,
            "dense_available": self.dense is not None,
            "sparse_available": self.sparse is not None,
            "reranker_available": self.reranker is not None and getattr(self.reranker, 'is_available', lambda: False)(),
        }


def create_faang_retriever(config: Dict) -> FAANGRetriever:
    """
    Factory function to create FAANG retriever with all components.
    """
    from retrieval.dense_retriever import DenseRetriever
    from retrieval.sparse_retriever import SparseRetriever
    from retrieval.reranker import Reranker
    
    dense = DenseRetriever(config)
    sparse = SparseRetriever(config)
    reranker = Reranker(config)
    
    return FAANGRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        reranker=reranker,
        config=config
    )
