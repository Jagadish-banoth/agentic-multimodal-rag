"""
Query Expansion Module for SOTA Retrieval
==========================================

Implements FAANG-level query expansion techniques:
1. HyDE (Hypothetical Document Embeddings)
2. Multi-query expansion (formal, technical, simplified variants)
3. Query reformulation for better retrieval

Used in parallel with original query to significantly improve recall.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "settings.yaml"

logger = logging.getLogger("query_expansion")
if not logger.hasHandlers():
    log_path = ROOT / "logs" / "query_expansion.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def load_config() -> Dict[str, Any]:
    """Load configuration from settings.yaml."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load config: {e}")
        return {}


class QueryExpander:
    """
    FAANG-grade query expansion for improved retrieval recall.
    
    Techniques:
    - HyDE: Generate hypothetical documents and use them for retrieval
    - Multi-query: Create variants (formal, technical, simplified)
    - Query reformulation: Rewrite for clarity
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, enable_hyde: bool = True):
        """
        Initialize query expander.
        
        Args:
            config: Configuration dictionary
            enable_hyde: Enable HyDE expansion (requires LLM)
        """
        self.config = config or load_config()
        self.enable_hyde = enable_hyde and OLLAMA_AVAILABLE
        
        # LLM config
        gen_cfg = self.config.get("generation", {})
        ollama_cfg = gen_cfg.get("ollama", {})
        self.ollama_host = ollama_cfg.get("host", "http://localhost:11434")
        self.ollama_model = gen_cfg.get("local_model", "llama3:8b")
        self.ollama_timeout = ollama_cfg.get("timeout", 30)
        
        # Cache for expanded queries
        self._cache = {}
        
        if self.enable_hyde:
            logger.info("✓ QueryExpander initialized with HyDE support")
        else:
            logger.info("⚠️ QueryExpander running without HyDE (LLM unavailable)")

    def expand_query(self, query: str, use_hyde: bool = True, use_variants: bool = True) -> Dict[str, List[str]]:
        """
        Expand a query using multiple techniques.
        
        Args:
            query: Original query
            use_hyde: Enable HyDE expansion
            use_variants: Enable multi-query variants
        
        Returns:
            Dict with keys:
            - "original": [original_query]
            - "variants": [variant1, variant2, variant3]  (if use_variants=True)
            - "hyde": [hypothesis1, hypothesis2]  (if use_hyde=True)
        """
        cache_key = f"{query}|hyde={use_hyde}|variants={use_variants}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = {"original": [query]}
        
        try:
            # Multi-query variants (always fast)
            if use_variants:
                variants = self._generate_variants(query)
                result["variants"] = variants
                logger.info(f"Generated {len(variants)} query variants")
            
            # HyDE (optional, uses LLM)
            if use_hyde and self.enable_hyde:
                hypotheses = self._generate_hypotheses(query)
                if hypotheses:
                    result["hyde"] = hypotheses
                    logger.info(f"Generated {len(hypotheses)} hypotheses")
        
        except Exception as e:
            logger.warning(f"Query expansion error (falling back to original): {e}")
            # Graceful degradation - return at least the original
        
        self._cache[cache_key] = result
        return result

    def _generate_variants(self, query: str) -> List[str]:
        """
        Generate query variants without LLM (deterministic).
        
        Returns:
        - Formal variant (replace pronouns, expand abbreviations)
        - Technical variant (add technical terms)
        - Simplified variant (remove jargon)
        """
        variants = []
        
        try:
            # Variant 1: Formal (expand contractions, remove colloquialisms)
            formal = self._make_formal(query)
            if formal != query:
                variants.append(formal)
            
            # Variant 2: Technical (add domain terms)
            technical = self._make_technical(query)
            if technical != query:
                variants.append(technical)
            
            # Variant 3: Simplified (remove jargon, explain acronyms)
            simplified = self._make_simplified(query)
            if simplified != query:
                variants.append(simplified)
        
        except Exception as e:
            logger.warning(f"Variant generation error: {e}")
        
        return variants

    def _generate_hypotheses(self, query: str, num_hypotheses: int = 2) -> List[str]:
        """
        HyDE: Generate hypothetical document passages that would answer the query.
        
        This is inspired by the HyDE paper (Gao et al., 2023).
        
        Args:
            query: User query
            num_hypotheses: Number of hypothetical documents to generate
        
        Returns:
            List of hypothetical passages
        """
        if not OLLAMA_AVAILABLE or not self.enable_hyde:
            return []
        
        hypotheses = []
        
        try:
            prompt = f"""Generate a hypothetical document passage (2-3 sentences) that would answer this question.
Make it informative and relevant.

Question: {query}

Hypothetical passage:"""
            
            for i in range(num_hypotheses):
                try:
                    response = ollama.generate(
                        model=self.ollama_model,
                        prompt=prompt,
                        stream=False,
                        options={
                            "temperature": 0.7 + (i * 0.1),  # Vary temperature for diversity
                            "top_p": 0.9,
                            "num_predict": 100,
                        }
                    )
                    
                    hypothesis = response.get("response", "").strip()
                    if hypothesis and len(hypothesis) > 20:
                        hypotheses.append(hypothesis)
                        logger.debug(f"HyDE hypothesis {i+1}: {hypothesis[:80]}...")
                
                except Exception as e:
                    logger.debug(f"HyDE generation {i+1} failed: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"HyDE expansion failed: {e}")
        
        return hypotheses

    @staticmethod
    def _make_formal(query: str) -> str:
        """Convert query to formal style."""
        formal = query
        
        # Expand contractions
        contractions = {
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "won't": "will not",
            "can't": "cannot",
            "couldn't": "could not",
            "shouldn't": "should not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "what's": "what is",
            "it's": "it is",
            "that's": "that is",
        }
        
        for contraction, expanded in contractions.items():
            formal = formal.replace(contraction, expanded)
            formal = formal.replace(contraction.upper(), expanded.upper())
        
        return formal.strip()

    @staticmethod
    def _make_technical(query: str) -> List[str]:
        """Add technical terminology."""
        technical = query
        
        # Common abbreviation → technical expansion
        expansions = {
            "ai": "artificial intelligence",
            "ml": "machine learning",
            "nlp": "natural language processing",
            "llm": "large language model",
            "rag": "retrieval-augmented generation",
            "embedding": "vector embedding",
            "model": "machine learning model",
            "training": "model training",
            "inference": "model inference",
            "optimization": "hyperparameter optimization",
        }
        
        q_lower = technical.lower()
        for abbr, expansion in expansions.items():
            if abbr in q_lower:
                # Replace with expansion maintaining case
                technical = technical.replace(abbr, expansion)
                technical = technical.replace(abbr.upper(), expansion.upper())
        
        return technical.strip()

    @staticmethod
    def _make_simplified(query: str) -> str:
        """Simplify technical language."""
        simplified = query
        
        # Technical term → simplified
        simplifications = {
            "embedding": "vector representation",
            "model": "system",
            "inference": "prediction",
            "optimization": "improvement",
            "hyperparameter": "setting",
            "architecture": "structure",
            "transformer": "attention-based network",
            "encoder": "processor",
            "decoder": "generator",
        }
        
        for technical, simple in simplifications.items():
            simplified = simplified.replace(technical, simple)
            simplified = simplified.replace(technical.upper(), simple.upper())
        
        return simplified.strip()

    def get_all_queries(self, query: str) -> List[str]:
        """
        Get all expanded queries (original + variants + hypotheses).
        
        Returns a list that can be used for parallel retrieval.
        """
        expansion = self.expand_query(query)
        all_queries = expansion.get("original", [])
        all_queries.extend(expansion.get("variants", []))
        all_queries.extend(expansion.get("hyde", []))
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for q in all_queries:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique.append(q)
        
        return unique[:5]  # Cap at 5 to avoid overwhelming retrieval


# Singleton instance
_expander = None


def get_query_expander(config: Optional[Dict] = None, enable_hyde: bool = True) -> QueryExpander:
    """Get or create query expander singleton."""
    global _expander
    if _expander is None:
        _expander = QueryExpander(config, enable_hyde)
    return _expander
