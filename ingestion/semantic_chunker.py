"""
SOTA Semantic Chunking
=======================
Advanced chunking strategies beyond fixed token splitting:
- Semantic similarity-based splitting
- Sentence-aware chunking
- Paragraph-preserving chunking
- Topic-based segmentation
- Adaptive chunk sizing

Delivers better chunk boundaries for higher retrieval accuracy.
"""

import logging
from typing import List, Dict, Optional, Literal, Tuple, Any
import numpy as np
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

# Lazy imports
_nltk_available = False
_spacy_available = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    _nltk_available = True
except ImportError:
    pass

try:
    import spacy
    _spacy_available = True
except ImportError:
    pass


@dataclass
class Chunk:
    """Chunk container."""
    chunk_id: str
    content: str
    start_char: int
    end_char: int
    sentences: List[str]
    token_count: int
    metadata: Dict = None


class SemanticChunker:
    """
    Semantic-aware chunking that respects document structure.
    
    Strategies:
    1. **semantic**: Split based on embedding similarity
    2. **sentence**: Sentence-boundary aware
    3. **paragraph**: Preserve paragraph structure
    4. **topic**: Topic modeling-based segmentation
    
    Examples:
        chunker = SemanticChunker(
            strategy="semantic",
            chunk_size=512,
            overlap=128,
            similarity_threshold=0.75
        )
        
        chunks = chunker.chunk(document_text, embedder)
    """
    
    def __init__(
        self,
        strategy: Literal["semantic", "sentence", "paragraph", "topic", "adaptive"] = "semantic",
        chunk_size: int = 512,
        overlap: int = 128,
        similarity_threshold: float = 0.75,
        min_chunk_size: int = 200,
        max_chunk_size: int = 800,
        embedder: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        """
        Args:
            strategy: Chunking strategy
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks
            similarity_threshold: Threshold for semantic splits (0-1)
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
            embedder: Embedding model for semantic chunking
            tokenizer: Tokenizer for counting tokens
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.embedder = embedder
        self.tokenizer = tokenizer
        
        # Initialize sentence tokenizer
        if _nltk_available:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
        
        logger.info(f"Initialized SemanticChunker: strategy={strategy}, size={chunk_size}")
    
    def chunk(self, text: str, doc_id: str = "doc") -> List[Chunk]:
        """
        Chunk document using selected strategy.
        
        Args:
            text: Document text
            doc_id: Document identifier
            
        Returns:
            List of Chunk objects
        """
        if self.strategy == "semantic":
            return self._chunk_semantic(text, doc_id)
        elif self.strategy == "sentence":
            return self._chunk_sentence(text, doc_id)
        elif self.strategy == "paragraph":
            return self._chunk_paragraph(text, doc_id)
        elif self.strategy == "topic":
            return self._chunk_topic(text, doc_id)
        elif self.strategy == "adaptive":
            return self._chunk_adaptive(text, doc_id)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _chunk_semantic(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Semantic chunking: split based on embedding similarity.
        
        Process:
        1. Split into sentences
        2. Embed each sentence
        3. Compute similarity between adjacent sentences
        4. Split where similarity drops below threshold
        5. Group sentences into chunks respecting size limits
        """
        if self.embedder is None:
            logger.warning("Embedder not provided for semantic chunking, falling back to sentence chunking")
            return self._chunk_sentence(text, doc_id)
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 1:
            return [Chunk(
                chunk_id=f"{doc_id}_0",
                content=text,
                start_char=0,
                end_char=len(text),
                sentences=sentences,
                token_count=self._count_tokens(text),
                metadata={}
            )]
        
        # Embed sentences
        logger.debug(f"Embedding {len(sentences)} sentences for semantic chunking")
        embeddings = self.embedder.encode(sentences, mode="document")
        
        # Compute similarities
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1])
            similarities.append(sim)
        
        # Find split points (where similarity drops)
        split_indices = [0]
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                split_indices.append(i + 1)
        split_indices.append(len(sentences))
        
        # Create chunks from split points
        chunks = []
        for i in range(len(split_indices) - 1):
            start_idx = split_indices[i]
            end_idx = split_indices[i + 1]
            
            # Group sentences
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)
            
            # Respect size limits
            token_count = self._count_tokens(chunk_text)
            
            if token_count > self.max_chunk_size:
                # Further split large chunks
                sub_chunks = self._split_by_size(chunk_sentences, doc_id, len(chunks))
                chunks.extend(sub_chunks)
            elif token_count >= self.min_chunk_size:
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_{len(chunks)}",
                    content=chunk_text,
                    start_char=text.find(chunk_text),
                    end_char=text.find(chunk_text) + len(chunk_text),
                    sentences=chunk_sentences,
                    token_count=token_count,
                    metadata={"split_reason": "semantic"}
                ))
        
        return chunks
    
    def _chunk_sentence(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Sentence-aware chunking: respect sentence boundaries.
        """
        sentences = self._split_sentences(text)
        
        chunks = []
        current_sentences = []
        current_tokens = 0
        
        for sentence in sentences:
            sent_tokens = self._count_tokens(sentence)
            
            # Check if adding this sentence exceeds max size
            if current_tokens + sent_tokens > self.chunk_size and current_sentences:
                # Create chunk
                chunk_text = " ".join(current_sentences)
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_{len(chunks)}",
                    content=chunk_text,
                    start_char=text.find(chunk_text),
                    end_char=text.find(chunk_text) + len(chunk_text),
                    sentences=current_sentences.copy(),
                    token_count=current_tokens,
                    metadata={"split_reason": "sentence_boundary"}
                ))
                
                # Handle overlap
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    s_tokens = self._count_tokens(s)
                    if overlap_tokens + s_tokens <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break
                
                current_sentences = overlap_sentences
                current_tokens = overlap_tokens
            
            current_sentences.append(sentence)
            current_tokens += sent_tokens
        
        # Add final chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_{len(chunks)}",
                content=chunk_text,
                start_char=text.find(chunk_text),
                end_char=text.find(chunk_text) + len(chunk_text),
                sentences=current_sentences,
                token_count=current_tokens,
                metadata={"split_reason": "final_chunk"}
            ))
        
        return chunks
    
    def _chunk_paragraph(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Paragraph-aware chunking: preserve paragraph structure.
        """
        # Split by double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_paragraphs = []
        current_tokens = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_tokens = self._count_tokens(para)
            
            if current_tokens + para_tokens > self.chunk_size and current_paragraphs:
                # Create chunk
                chunk_text = "\n\n".join(current_paragraphs)
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_{len(chunks)}",
                    content=chunk_text,
                    start_char=text.find(chunk_text),
                    end_char=text.find(chunk_text) + len(chunk_text),
                    sentences=self._split_sentences(chunk_text),
                    token_count=current_tokens,
                    metadata={"split_reason": "paragraph_boundary"}
                ))
                
                current_paragraphs = []
                current_tokens = 0
            
            current_paragraphs.append(para)
            current_tokens += para_tokens
        
        # Final chunk
        if current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_{len(chunks)}",
                content=chunk_text,
                start_char=text.find(chunk_text),
                end_char=text.find(chunk_text) + len(chunk_text),
                sentences=self._split_sentences(chunk_text),
                token_count=current_tokens,
                metadata={"split_reason": "final_chunk"}
            ))
        
        return chunks
    
    def _chunk_topic(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Topic-based chunking (placeholder - requires topic modeling).
        Falls back to sentence chunking.
        """
        logger.warning("Topic chunking not fully implemented, using sentence chunking")
        return self._chunk_sentence(text, doc_id)
    
    def _chunk_adaptive(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Adaptive chunking: choose best strategy based on document structure.
        """
        # Heuristics
        paragraph_count = len(re.findall(r'\n\s*\n', text))
        sentence_count = len(self._split_sentences(text))
        
        # If many paragraphs, use paragraph chunking
        if paragraph_count > 5:
            return self._chunk_paragraph(text, doc_id)
        
        # If embedder available, use semantic
        elif self.embedder is not None and sentence_count > 10:
            return self._chunk_semantic(text, doc_id)
        
        # Otherwise sentence chunking
        else:
            return self._chunk_sentence(text, doc_id)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if _nltk_available:
            return sent_tokenize(text)
        else:
            # Fallback: simple regex
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _split_by_size(self, sentences: List[str], doc_id: str, start_idx: int) -> List[Chunk]:
        """Split large group of sentences into multiple chunks."""
        chunks = []
        current_sentences = []
        current_tokens = 0
        
        for sentence in sentences:
            sent_tokens = self._count_tokens(sentence)
            
            if current_tokens + sent_tokens > self.chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_{start_idx + len(chunks)}",
                    content=chunk_text,
                    start_char=0,  # Approximate
                    end_char=len(chunk_text),
                    sentences=current_sentences.copy(),
                    token_count=current_tokens,
                    metadata={"split_reason": "size_limit"}
                ))
                current_sentences = []
                current_tokens = 0
            
            current_sentences.append(sentence)
            current_tokens += sent_tokens
        
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_{start_idx + len(chunks)}",
                content=chunk_text,
                start_char=0,
                end_char=len(chunk_text),
                sentences=current_sentences,
                token_count=current_tokens,
                metadata={"split_reason": "final"}
            ))
        
        return chunks
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: approximate with whitespace split
            return len(text.split())


# =================================================================
# CONVENIENCE FUNCTIONS
# =================================================================

def create_chunker_from_config(config: dict, embedder=None, tokenizer=None) -> SemanticChunker:
    """
    Create chunker from config.
    
    Example config:
        ingestion:
          chunking_strategy: "semantic"
          chunk_size: 512
          chunk_overlap: 128
          semantic_threshold: 0.75
    """
    ing_cfg = config.get("ingestion", {})
    
    return SemanticChunker(
        strategy=ing_cfg.get("chunking_strategy", "semantic"),
        chunk_size=ing_cfg.get("chunk_size", 512),
        overlap=ing_cfg.get("chunk_overlap", 128),
        similarity_threshold=ing_cfg.get("semantic_threshold", 0.75),
        min_chunk_size=ing_cfg.get("min_chunk_size", 200),
        max_chunk_size=ing_cfg.get("max_chunk_size", 800),
        embedder=embedder,
        tokenizer=tokenizer
    )
