"""Query preprocessing for optimal retrieval accuracy."""
import re
import logging
from typing import List, Optional

try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False
    AutoTokenizer = None

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Preprocess queries for better retrieval accuracy.
    
    SOTA Best Practices:
    - Remove extra whitespace
    - PRESERVE CASE (important for proper nouns, acronyms like "BGE", "FAISS")
    - Preserve key terms and punctuation
    - Truncate to model limits
    
    Note: DO NOT lowercase - it hurts retrieval for names, acronyms, and codes.
    """
    
    def __init__(self, config: dict):
        """Initialize query processor.
        
        Args:
            config: System configuration dict
        """
        self.config = config
        self.max_query_length = 512
        
        # Load tokenizer for accurate truncation
        self.tokenizer = None
        if HAS_TOKENIZER:
            model_name = config.get("models", {}).get("embedding_model", "BAAI/bge-m3")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    use_fast=True,
                    trust_remote_code=True
                )
                logger.info(f"✓ Query processor loaded tokenizer: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
        
    def preprocess(self, query: str) -> str:
        """Clean and normalize query.
        
        SOTA: Preserves case for proper nouns, acronyms, and technical terms.
        
        Args:
            query: Raw user query
            
        Returns:
            Preprocessed query (case-preserved)
        """
        if not query or not query.strip():
            return ""
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # SOTA: DO NOT lowercase - preserves proper nouns, acronyms, codes
        # Previous: query = query.lower()
        
        # Truncate to model limits if tokenizer available
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(
                    query,
                    max_length=self.max_query_length,
                    truncation=True,
                    add_special_tokens=False
                )
                query = self.tokenizer.decode(tokens, skip_special_tokens=True)
            except Exception as e:
                logger.debug(f"Tokenizer truncation failed: {e}")
        
        return query
    
    def expand_query(self, query: str) -> List[str]:
        """Generate query variations for better recall.
        
        Args:
            query: Preprocessed query
            
        Returns:
            List of query variations [original, ...]
        """
        # For now, return original
        # TODO: Add synonym expansion, query rewriting
        return [query]
