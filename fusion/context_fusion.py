"""
Context Fusion module for Agentic Multimodal RAG (SOTA)
--------------------------------------------------------
Responsibilities:
- Deduplicate retrieved chunks
- MMR-style diversity enforcement
- Enforce token budget with smart truncation
- Produce citation-ready structured context with stable IDs
- Return chunk mapping for downstream verification

SOTA Features:
- Stable citation IDs ([filename#chunk_id]) for machine-checkable grounding
- MMR (Maximal Marginal Relevance) for relevance+diversity balance
- Structured output with chunk_map for verification pipeline
- Multimodal-aware formatting

Public API:
    ContextFusion.build_context(query: str, retrieved_chunks: List[Dict]) -> str
    ContextFusion.fuse_with_mapping(query: str, retrieved_chunks: List[Dict]) -> Tuple[str, Dict]
"""

from __future__ import annotations
import logging
import os
import re
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any

# Optional tokenizer (recommended)
try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except Exception:
    AutoTokenizer = None
    HAS_TOKENIZER = False

# Optional sentence-transformers for MMR
try:
    from sentence_transformers import util as st_util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    st_util = None
    HAS_SENTENCE_TRANSFORMERS = False

import os
log_path = os.path.join(os.path.dirname(__file__), '../logs/fusion.log')
log_path = os.path.abspath(log_path)
logger = logging.getLogger("context_fusion")
if not logger.hasHandlers():
    handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# -----------------------------
# Context Fusion
# -----------------------------
class ContextFusion:
    """SOTA Context Fusion with stable citation IDs and MMR diversity."""
    
    def __init__(self, config: Dict):
        self.config = config
        fusion_cfg = config.get("fusion", {})

        # SOTA: Increased defaults for better recall
        self.max_chunks: int = fusion_cfg.get("max_chunks", 15)
        self.max_tokens: int = fusion_cfg.get("max_tokens", 3500)  # SOTA: More context
        self.page_group_span: int = fusion_cfg.get("page_group_span", 2)
        self.use_tokenizer: bool = fusion_cfg.get("use_tokenizer", True)
        self.tokenizer_name: str = fusion_cfg.get(
            "tokenizer_name",
            config.get("models", {}).get("tokenizer_model")
        )
        self.word_token_multiplier: float = fusion_cfg.get(
            "token_count_multiplier", 1.3
        )
        
        # SOTA: MMR diversity settings
        self.mmr_lambda: float = fusion_cfg.get("mmr_lambda", 0.7)  # 0=diversity, 1=relevance
        self.use_mmr: bool = fusion_cfg.get("use_mmr", True)
        
        # Track chunk mapping for verification
        self._last_chunk_map: Dict[str, Dict] = {}

        # Load tokenizer safely with trust_remote_code=True
        self.tokenizer = None
        if self.use_tokenizer and HAS_TOKENIZER and self.tokenizer_name:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name,
                    use_fast=True,
                    trust_remote_code=True
                )
                logger.info(
                    f"ContextFusion: loaded tokenizer {self.tokenizer_name}"
                )
            except Exception:
                logger.exception(
                    "ContextFusion: tokenizer load failed, using fallback."
                )
                self.tokenizer = None

    # --------------------------------------------------
    # PUBLIC API (THIS IS WHAT EXECUTION ENGINE CALLS)
    # --------------------------------------------------
    def fuse(self, retrieved_chunks: List[Dict], query: str = "") -> str:
        """
        Alias for build_context() - used by ExecutionEngine.
        Returns context string. Use fuse_with_mapping() for verification.
        """
        context, _ = self.fuse_with_mapping(retrieved_chunks, query)
        return context
    
    def fuse_with_mapping(
        self,
        retrieved_chunks: List[Dict],
        query: str = ""
    ) -> Tuple[str, Dict[str, Dict]]:
        """
        SOTA: Build context and return chunk mapping for verification.
        
        Returns:
            Tuple of (context_string, chunk_map)
            chunk_map: {"C1": {chunk_data}, "C2": {chunk_data}, ...}
        """
        if not retrieved_chunks:
            self._last_chunk_map = {}
            return "", {}
        
        context = self.build_context(query, retrieved_chunks)
        return context, self._last_chunk_map

    def build_context(
        self,
        query: str,
        retrieved_chunks: List[Dict]
    ) -> str:
        """
        Build the final grounded context string with stable citation IDs.
        """
        if not retrieved_chunks:
            self._last_chunk_map = {}
            return ""

        logger.info("ContextFusion: building SOTA context with stable IDs")

        # 1. Deduplicate
        chunks = self._deduplicate(retrieved_chunks)

        # 2. Sort by score (if present)
        chunks = self._sort_by_score(chunks)

        # 3. SOTA: Apply MMR diversity (if embeddings available) or fallback
        if self.use_mmr:
            chunks = self._apply_mmr_diversity(chunks)
        else:
            chunks = self._enforce_diversity(chunks)

        # 4. Apply token budget
        chunks = self._apply_token_budget(chunks)

        # 5. Hard cap
        chunks = chunks[: self.max_chunks]
        
        # 6. SOTA: Assign stable citation IDs and build chunk map
        chunks, chunk_map = self._assign_citation_ids(chunks)
        self._last_chunk_map = chunk_map

        # 7. Format with stable IDs
        context = self._format_context_sota(query, chunks, chunk_map)

        logger.info(
            f"ContextFusion: selected {len(chunks)} chunks "
            f"within {self.max_tokens} tokens (IDs: {list(chunk_map.keys())})"
        )

        return context
    
    def get_chunk_map(self) -> Dict[str, Dict]:
        """Return the last chunk mapping for verification."""
        return self._last_chunk_map

    # --------------------------------------------------
    # INTERNAL LOGIC
    # --------------------------------------------------
    def _deduplicate(self, chunks: List[Dict]) -> List[Dict]:
        seen: Set[str] = set()
        out: List[Dict] = []
        for c in chunks:
            if not isinstance(c, dict):
                continue  # Skip non-dict elements; add print(c) here if you want to inspect them
            cid = c.get("chunk_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            out.append(c)
        return out

    def _sort_by_score(self, chunks: List[Dict]) -> List[Dict]:
        # Prefer the reranker score if available, fallback to legacy 'score'
        if any("rerank_score" in c for c in chunks):
            return sorted(
                chunks,
                key=lambda x: x.get("rerank_score", 0.0),
                reverse=True
            )
        if any("score" in c for c in chunks):
            return sorted(
                chunks,
                key=lambda x: x.get("score", 0.0),
                reverse=True
            )
        return chunks

    def _page_bucket(self, c: Dict) -> Tuple[int, int]:
        ps = c.get("page_start")
        if ps is None:
            return (-1, -1)
        span = self.page_group_span
        start = (ps // span) * span
        return (start, start + span - 1)

    def _enforce_diversity(self, chunks: List[Dict]) -> List[Dict]:
        """Basic diversity enforcement (fallback when MMR not available)."""
        out: List[Dict] = []
        used_sources: Set[str] = set()
        used_pages: Set[Tuple[int, int]] = set()

        for c in chunks:
            src = c.get("source", "unknown")
            page_key = self._page_bucket(c)

            if src not in used_sources or page_key not in used_pages:
                out.append(c)
                used_sources.add(src)
                used_pages.add(page_key)

            if len(out) >= self.max_chunks * 2:
                break

        # Fallback if diversity pruned too aggressively
        if len(out) < min(3, len(chunks)):
            for c in chunks:
                if c not in out:
                    out.append(c)
                if len(out) >= self.max_chunks:
                    break

        return out
    
    def _apply_mmr_diversity(
        self,
        chunks: List[Dict],
        target_count: Optional[int] = None
    ) -> List[Dict]:
        """
        SOTA: Maximal Marginal Relevance for relevance + diversity balance.
        
        MMR = λ * Sim(doc, query) - (1-λ) * max(Sim(doc, selected_docs))
        """
        if not chunks:
            return []
        
        target = target_count or (self.max_chunks * 2)
        
        # If we don't have embeddings, fall back to score-based diversity
        if not any("embedding" in c or "text_embedding" in c for c in chunks):
            logger.debug("MMR: No embeddings found, using score-based diversity")
            return self._mmr_score_based(chunks, target)
        
        # Extract embeddings
        embeddings = []
        valid_chunks = []
        for c in chunks:
            emb = c.get("embedding") or c.get("text_embedding")
            if emb is not None:
                embeddings.append(np.array(emb))
                valid_chunks.append(c)
        
        if len(valid_chunks) < 2:
            return chunks[:target]
        
        embeddings = np.array(embeddings)
        
        # Get scores (relevance to query)
        scores = np.array([c.get("rerank_score", c.get("score", 0.5)) for c in valid_chunks])
        score_range = scores.max() - scores.min()
        if score_range > 0:
            scores = (scores - scores.min()) / score_range
        else:
            scores = np.ones_like(scores) * 0.5
        
        # MMR selection
        selected_idx = []
        remaining_idx = list(range(len(valid_chunks)))
        
        # Start with highest scoring
        best_idx = int(np.argmax(scores))
        selected_idx.append(best_idx)
        remaining_idx.remove(best_idx)
        
        while len(selected_idx) < target and remaining_idx:
            mmr_scores = []
            
            for idx in remaining_idx:
                # Relevance term
                relevance = scores[idx]
                
                # Diversity term: max similarity to already selected
                selected_embs = embeddings[selected_idx]
                candidate_emb = embeddings[idx]
                norms = np.linalg.norm(selected_embs, axis=1) * np.linalg.norm(candidate_emb) + 1e-8
                similarities = np.dot(selected_embs, candidate_emb) / norms
                max_sim = np.max(similarities)
                
                # MMR score
                mmr = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_sim
                mmr_scores.append((idx, mmr))
            
            # Select best MMR
            best_mmr_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_idx.append(best_mmr_idx)
            remaining_idx.remove(best_mmr_idx)
        
        return [valid_chunks[i] for i in selected_idx]
    
    def _mmr_score_based(self, chunks: List[Dict], target: int) -> List[Dict]:
        """
        Score-based MMR approximation when embeddings unavailable.
        Uses content similarity via term overlap.
        """
        if not chunks:
            return []
        
        selected = [chunks[0]]  # Start with top scorer
        remaining = chunks[1:]
        
        while len(selected) < target and remaining:
            best_chunk = None
            best_mmr = float('-inf')
            
            for c in remaining:
                # Relevance
                relevance = c.get("rerank_score", c.get("score", 0.5))
                
                # Diversity: term overlap with selected
                c_terms = set(self._get_content(c).lower().split())
                max_overlap = 0.0
                for sel in selected:
                    sel_terms = set(self._get_content(sel).lower().split())
                    if c_terms and sel_terms:
                        overlap = len(c_terms & sel_terms) / min(len(c_terms), len(sel_terms) + 1)
                        max_overlap = max(max_overlap, overlap)
                
                mmr = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_overlap
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_chunk = c
            
            if best_chunk:
                selected.append(best_chunk)
                remaining.remove(best_chunk)
            else:
                break
        
        return selected
    
    def _get_content(self, chunk: Dict) -> str:
        """Extract text content from chunk for any modality."""
        modality = chunk.get("modality", "text")
        
        if modality == "video":
            return (chunk.get("content") or 
                   chunk.get("audio_transcript") or 
                   chunk.get("frame_caption") or 
                   chunk.get("snippet") or "").strip()
        elif modality == "audio":
            return (chunk.get("content") or 
                   chunk.get("transcript") or 
                   chunk.get("audio_transcript") or 
                   chunk.get("snippet") or "").strip()
        elif modality == "image":
            return (chunk.get("content") or 
                   chunk.get("combined_content") or 
                   chunk.get("caption") or 
                   chunk.get("snippet") or "").strip()
        else:
            return (chunk.get("content") or chunk.get("snippet") or "").strip()
    
    def _assign_citation_ids(
        self,
        chunks: List[Dict]
    ) -> Tuple[List[Dict], Dict[str, Dict]]:
        """
        SOTA: Assign stable citation IDs using file name + chunk_id.

        Format: [filename#chunk_id]

        Returns:
            Tuple of (chunks_with_cite_id, chunk_map)
        """
        chunk_map: Dict[str, Dict] = {}
        updated_chunks: List[Dict] = []

        for i, chunk in enumerate(chunks):
            source = chunk.get("source", chunk.get("source_file", "unknown"))
            chunk_id = chunk.get("chunk_id", f"unknown_{i}")

            cite_id = self._build_citation_id(source, chunk_id)

            # Ensure uniqueness (in case of collisions)
            if cite_id in chunk_map:
                suffix = 2
                new_id = f"{cite_id}~{suffix}"
                while new_id in chunk_map:
                    suffix += 1
                    new_id = f"{cite_id}~{suffix}"
                cite_id = new_id

            chunk_copy = chunk.copy()
            chunk_copy["cite_id"] = cite_id
            updated_chunks.append(chunk_copy)

            # Build map with essential fields for verification
            chunk_map[cite_id] = {
                "cite_id": cite_id,
                "chunk_id": str(chunk_id),
                "source": source,
                "source_file": source,
                "modality": chunk.get("modality", "text"),
                "content": self._get_content(chunk),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "score": chunk.get("rerank_score", chunk.get("score", 0.0)),
            }

        return updated_chunks, chunk_map

    def _build_citation_id(self, source: str, chunk_id: Any) -> str:
        """Build citation ID from file name + chunk_id."""
        source_name = self._sanitize_citation_part(os.path.basename(str(source)))
        chunk_part = self._sanitize_citation_part(str(chunk_id))
        return f"{source_name}#{chunk_part}"

    def _sanitize_citation_part(self, text: str) -> str:
        """Sanitize citation parts to be safe inside brackets."""
        cleaned = re.sub(r"[\[\]]", "", text)
        cleaned = re.sub(r"\s+", "_", cleaned.strip())
        cleaned = re.sub(r"[^\w\-\.#]+", "_", cleaned)
        return cleaned or "unknown"

    def _estimate_tokens(self, text: str) -> int:
        if self.tokenizer:
            try:
                return len(
                    self.tokenizer.encode(
                        text,
                        add_special_tokens=False
                    )
                )
            except Exception:
                pass
        return max(
            1,
            int(len(text.split()) * self.word_token_multiplier)
        )

    def _apply_token_budget(self, chunks: List[Dict]) -> List[Dict]:
        """Apply token budget with smart truncation."""
        out: List[Dict] = []
        used = 0
        
        # Reserve tokens for formatting overhead
        effective_budget = int(self.max_tokens * 0.92)

        for c in chunks:
            text = self._get_content(c)
            
            if not text:
                continue

            tokens = self._estimate_tokens(text)
            
            # SOTA: Smart truncation for oversized chunks
            if tokens > effective_budget // 2:
                # Truncate large chunks to half budget
                max_chars = (effective_budget // 2) * 4  # ~4 chars/token
                text = text[:max_chars] + "..."
                tokens = self._estimate_tokens(text)
                c = c.copy()
                c["content"] = text
                c["_truncated"] = True
            
            if used + tokens > effective_budget:
                # Try to fit one more with truncation
                remaining = effective_budget - used
                if remaining > 100:  # Enough for meaningful content
                    max_chars = remaining * 4
                    truncated_text = text[:max_chars] + "..."
                    c = c.copy()
                    c["content"] = truncated_text
                    c["_truncated"] = True
                    out.append(c)
                break

            out.append(c)
            used += tokens

        return out
    
    def _format_context_sota(
        self,
        query: str,
        chunks: List[Dict],
        chunk_map: Dict[str, Dict]
    ) -> str:
        """
        SOTA: Format chunks with stable citation IDs for grounded generation.

        Uses [filename#chunk_id] for machine-checkable citations.
        """
        # Group chunks by modality
        text_chunks = []
        image_chunks = []
        video_chunks = []
        audio_chunks = []
        
        for c in chunks:
            modality = c.get("modality", "text")
            if modality == "image":
                image_chunks.append(c)
            elif modality == "video":
                video_chunks.append(c)
            elif modality == "audio":
                audio_chunks.append(c)
            else:
                text_chunks.append(c)
        
        blocks = []
        
        # Format text chunks with stable IDs
        if text_chunks:
            blocks.append("📄 TEXT EVIDENCE:")
            for c in text_chunks:
                cite_id = c.get("cite_id", "unknown#?")
                src = c.get("source", "unknown")
                ps = c.get("page_start", "")
                pe = c.get("page_end", "")
                page = f"pp.{ps}-{pe}" if ps != "" else ""
                
                content = self._get_content(c)
                truncated = " [truncated]" if c.get("_truncated") else ""
                
                block = f"[{cite_id}] {src} {page}{truncated}\n{content}"
                blocks.append(block)
        
        # Format image chunks
        if image_chunks:
            blocks.append("\n🖼️ IMAGE EVIDENCE:")
            for c in image_chunks:
                cite_id = c.get("cite_id", "unknown#?")
                src = c.get("source", c.get("source_file", "image"))
                caption = c.get("frame_caption", c.get("caption", ""))
                ocr_text = c.get("frame_ocr", c.get("ocr_text", ""))
                
                parts = [f"[{cite_id}] {src}"]
                if caption:
                    parts.append(f"Description: {caption}")
                if ocr_text:
                    parts.append(f"Text visible: {ocr_text}")
                
                blocks.append("\n".join(parts))
        
        # Format video chunks
        if video_chunks:
            blocks.append("\n🎬 VIDEO EVIDENCE:")
            for c in video_chunks:
                cite_id = c.get("cite_id", "unknown#?")
                src = c.get("source_file", "video")
                timestamp = c.get("timestamp_formatted", c.get("timestamp_seconds", ""))
                content = self._get_content(c)
                
                parts = [f"[{cite_id}] {src}"]
                if timestamp:
                    parts.append(f"@{timestamp}")
                if content:
                    parts.append(content[:600])
                
                blocks.append(" ".join(parts[:2]) + "\n" + (parts[2] if len(parts) > 2 else ""))
        
        # Format audio chunks
        if audio_chunks:
            blocks.append("\n🎧 AUDIO EVIDENCE:")
            for c in audio_chunks:
                cite_id = c.get("cite_id", "C?")
                src = c.get("source_file", "audio")
                transcript = self._get_content(c)
                
                block = f"[{cite_id}] {src}\nTranscript: {transcript[:500]}"
                blocks.append(block)
        
        # Build citation legend
        cite_legend = "Available citations: " + ", ".join(chunk_map.keys())
        
        # Build final context with SOTA structure
        context = (
            "INSTRUCTIONS: Answer using ONLY the evidence below. "
            "Cite sources using [filename#chunk_id] format. "
            "If information is insufficient, say so clearly.\n\n"
            f"{cite_legend}\n\n"
            "===== EVIDENCE =====\n\n"
            + "\n\n".join(blocks)
            + "\n\n===== END EVIDENCE =====\n\n"
            f"QUESTION: {query}\n\n"
            "ANSWER (cite with [filename#chunk_id]):"
        )
        
        return context

    def _format_context(
        self,
        query: str,
        chunks: List[Dict]
    ) -> str:
        """
        Format chunks into an optimized context string for LLM generation.
        
        Features:
        - Multimodal-aware formatting
        - Structured sections for different content types
        - Citation-ready format
        - Token-efficient representation
        """
        # Group chunks by modality
        text_chunks = []
        image_chunks = []
        video_chunks = []
        audio_chunks = []
        
        for c in chunks:
            modality = c.get("modality", "text")
            if modality == "image":
                image_chunks.append(c)
            elif modality == "video":
                video_chunks.append(c)
            elif modality == "audio":
                audio_chunks.append(c)
            else:
                text_chunks.append(c)
        
        blocks = []
        chunk_idx = 1
        
        # Format text chunks
        if text_chunks:
            blocks.append("📄 TEXT EVIDENCE:")
            for c in text_chunks:
                src = c.get("source", "unknown")
                ps = c.get("page_start", "")
                pe = c.get("page_end", "")
                page = f"pages {ps}-{pe}" if ps != "" else ""
                
                content = c.get('content', '').strip()
                
                block = f"[CHUNK {chunk_idx}] {src} {page}\n{content}"
                blocks.append(block)
                chunk_idx += 1
        
        # Format image chunks
        if image_chunks:
            blocks.append("\n🖼️ IMAGE EVIDENCE:")
            for c in image_chunks:
                src = c.get("source", c.get("source_file", "image"))
                caption = c.get("frame_caption", c.get("caption", ""))
                ocr_text = c.get("frame_ocr", c.get("ocr_text", ""))
                
                parts = [f"[{chunk_idx}] {src}"]
                if caption:
                    parts.append(f"Description: {caption}")
                if ocr_text:
                    parts.append(f"Text visible: {ocr_text}")
                
                blocks.append("\n".join(parts))
                chunk_idx += 1
        
        # Format video chunks
        if video_chunks:
            blocks.append("\n🎬 VIDEO EVIDENCE:")
            for c in video_chunks:
                src = c.get("source_file", "video")
                timestamp = c.get("timestamp_formatted", c.get("timestamp_seconds", ""))
                content = c.get("content", "")
                caption = c.get("frame_caption", "")
                ocr_text = c.get("frame_ocr", "")
                audio_seg = c.get("frame_audio_segment", c.get("frame_audio", ""))
                audio_full = c.get("audio_transcript", "")
                
                parts = [f"[{chunk_idx}] {src}"]
                if timestamp:
                    parts.append(f"Timestamp: {timestamp}")
                
                # If we have the combined 'content' field, use it directly (most complete)
                if content and len(content) > 50:
                    parts.append(content[:800])
                else:
                    # Fallback to individual fields
                    if caption:
                        parts.append(f"Scene: {caption}")
                    if ocr_text:
                        parts.append(f"Text visible: {ocr_text}")
                    if audio_full and len(audio_full) > 50:
                        parts.append(f"Audio: {audio_full[:500]}")
                    elif audio_seg:
                        parts.append(f"Audio: {audio_seg[:300]}")
                
                blocks.append("\n".join(parts))
                chunk_idx += 1
        
        # Format audio chunks
        if audio_chunks:
            blocks.append("\n🎧 AUDIO EVIDENCE:")
            for c in audio_chunks:
                src = c.get("source_file", "audio")
                transcript = c.get("transcript", c.get("content", ""))
                language = c.get("language", "")
                
                parts = [f"[{chunk_idx}] {src}"]
                if language:
                    parts.append(f"Language: {language}")
                if transcript:
                    parts.append(f"Transcript: {transcript[:500]}...")
                
                blocks.append("\n".join(parts))
                chunk_idx += 1
        
        # Build final context
        context = (
            "INSTRUCTIONS: Answer the question using ONLY the evidence below. "
            "Cite sources using [filename#chunk_id] format. If information is insufficient, say so.\n\n"
            "===== EVIDENCE START =====\n\n"
            + "\n\n".join(blocks)
            + "\n\n===== EVIDENCE END =====\n\n"
            f"QUESTION: {query}\n\n"
            "ANSWER:"
        )
        
        return context
    
    def get_context_stats(self, chunks: List[Dict]) -> Dict:
        """Return statistics about the context for debugging."""
        modality_counts = {}
        total_tokens = 0
        
        for c in chunks:
            mod = c.get("modality", "text")
            modality_counts[mod] = modality_counts.get(mod, 0) + 1
            total_tokens += self._estimate_tokens(c.get("content", ""))
        
        return {
            "total_chunks": len(chunks),
            "modality_distribution": modality_counts,
            "estimated_tokens": total_tokens,
            "max_tokens": self.max_tokens,
        }


# --------------------------------------------------
# SELF-TEST
# --------------------------------------------------
if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO)

    cfg = yaml.safe_load(open("config/settings.yaml"))
    fusion = ContextFusion(cfg)

    dummy = [
        {
            "chunk_id": "c1",
            "modality": "text",
            "content": "The Transformer architecture uses self-attention mechanisms.",
            "score": 0.92,
            "source": "attention_paper.pdf",
            "page_start": 1,
            "page_end": 1
        },
        {
            "chunk_id": "c2",
            "modality": "image",
            "content": "Architecture diagram",
            "frame_caption": "A diagram showing the encoder-decoder structure",
            "ocr_text": "Multi-Head Attention, Feed Forward",
            "score": 0.88,
            "source": "attention_paper.pdf",
        },
        {
            "chunk_id": "c3",
            "modality": "video",
            "content": "Video frame",
            "source_file": "lecture.mp4",
            "timestamp_formatted": "00:05:30",
            "frame_caption": "Professor explaining attention mechanism",
            "frame_audio_segment": "The key innovation is the self-attention layer",
            "score": 0.85,
        },
    ]

    ctx = fusion.build_context("What is Transformer Architecture?", dummy)
    print(ctx)
    print("\n" + "="*50 + "\n")
    print(fusion.get_context_stats(dummy))
