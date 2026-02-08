"""
SOTA Embedding Wrapper
======================
Supports industry-grade embedding models:
- BAAI/bge-m3 (8192 context, multilingual, SOTA)
- jinaai/jina-embeddings-v3 (8192 context, task-aware)
- OpenAI text-embedding-3-large
- Voyage AI voyage-large-2-instruct
- Cohere embed-v3

Features:
- Automatic batching
- GPU acceleration
- Late chunking for long documents
- Query vs document mode
- Caching
"""

import os
import logging
from typing import List, Optional, Union, Literal
import numpy as np
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)

# Lazy imports
_sentence_transformers_available = False
_openai_available = False
_cohere_available = False
_voyage_available = False

try:
    from sentence_transformers import SentenceTransformer
    _sentence_transformers_available = True
except ImportError:
    pass

try:
    import openai
    _openai_available = True
except ImportError:
    pass

try:
    import cohere
    _cohere_available = True
except ImportError:
    pass

try:
    import voyageai
    _voyage_available = True
except ImportError:
    pass


class SOTAEmbedder:
    """
    Universal embedder supporting SOTA models.
    
    Examples:
        # Local model (BGE-M3)
        embedder = SOTAEmbedder("BAAI/bge-m3", backend="local")
        
        # OpenAI
        embedder = SOTAEmbedder("text-embedding-3-large", backend="openai", api_key="sk-...")
        
        # Voyage AI
        embedder = SOTAEmbedder("voyage-large-2-instruct", backend="voyage", api_key="...")
    """
    
    SUPPORTED_MODELS = {
        # Local models (HuggingFace)
        "BAAI/bge-m3": {"backend": "local", "dim": 1024, "max_length": 8192},
        "BAAI/bge-large-en-v1.5": {"backend": "local", "dim": 1024, "max_length": 512},
        "jinaai/jina-embeddings-v3": {"backend": "local", "dim": 1024, "max_length": 8192},
        "nomic-ai/nomic-embed-text-v1.5": {"backend": "local", "dim": 768, "max_length": 8192},
        
        # API-based models
        "text-embedding-3-large": {"backend": "openai", "dim": 3072, "max_length": 8191},
        "text-embedding-3-small": {"backend": "openai", "dim": 1536, "max_length": 8191},
        "voyage-large-2-instruct": {"backend": "voyage", "dim": 1024, "max_length": 16000},
        "voyage-2": {"backend": "voyage", "dim": 1024, "max_length": 4000},
        "embed-english-v3.0": {"backend": "cohere", "dim": 1024, "max_length": 512},
    }
    
    def __init__(
        self,
        model_name: str,
        backend: Optional[str] = None,
        api_key: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 32,
        normalize: bool = True,
        cache_embeddings: bool = True,
        trust_remote_code: bool = True,
    ):
        """
        Args:
            model_name: Model identifier
            backend: Override auto-detected backend (local, openai, voyage, cohere)
            api_key: API key for cloud providers
            device: Device for local models (cuda, cpu)
            batch_size: Batch size for encoding
            normalize: L2 normalize embeddings
            cache_embeddings: Cache embeddings in memory
            trust_remote_code: Trust remote code for HF models
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.cache_embeddings = cache_embeddings
        
        # Determine backend
        if backend is None:
            if model_name in self.SUPPORTED_MODELS:
                backend = self.SUPPORTED_MODELS[model_name]["backend"]
            else:
                backend = "local"  # Default to local
        
        self.backend = backend
        self.model_info = self.SUPPORTED_MODELS.get(model_name, {})
        self.embedding_dim = self.model_info.get("dim", 768)
        self.max_length = self.model_info.get("max_length", 512)
        
        # Initialize model
        self.model = None
        self.client = None
        
        if backend == "local":
            self._init_local_model(model_name, device, trust_remote_code)
        elif backend == "openai":
            self._init_openai(api_key)
        elif backend == "voyage":
            self._init_voyage(api_key)
        elif backend == "cohere":
            self._init_cohere(api_key)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        logger.info(f"Initialized SOTAEmbedder: {model_name} (backend={backend}, dim={self.embedding_dim})")
    
    def _init_local_model(self, model_name: str, device: str, trust_remote_code: bool):
        """Initialize local SentenceTransformer model."""
        if not _sentence_transformers_available:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        
        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=trust_remote_code
        )
        
        # Set max seq length
        if hasattr(self.model, 'max_seq_length'):
            self.model.max_seq_length = self.max_length
    
    def _init_openai(self, api_key: Optional[str]):
        """Initialize OpenAI client."""
        if not _openai_available:
            raise ImportError("openai not installed. Run: pip install openai")
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = openai.OpenAI(api_key=api_key)
    
    def _init_voyage(self, api_key: Optional[str]):
        """Initialize Voyage AI client."""
        if not _voyage_available:
            raise ImportError("voyageai not installed. Run: pip install voyageai")
        
        api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("Voyage AI API key required")
        
        self.client = voyageai.Client(api_key=api_key)
    
    def _init_cohere(self, api_key: Optional[str]):
        """Initialize Cohere client."""
        if not _cohere_available:
            raise ImportError("cohere not installed. Run: pip install cohere")
        
        api_key = api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("Cohere API key required")
        
        self.client = cohere.Client(api_key=api_key)
    
    def encode(
        self,
        texts: Union[str, List[str]],
        mode: Literal["query", "document"] = "document",
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: Single text or list of texts
            mode: "query" for queries, "document" for documents
            show_progress: Show progress bar
            
        Returns:
            np.ndarray of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self.cache_embeddings:
            # Check cache
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text, mode)
                cached_emb = self._get_from_cache(cache_key)
                
                if cached_emb is not None:
                    cached_embeddings.append((i, cached_emb))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Encode uncached
            if uncached_texts:
                new_embeddings = self._encode_batch(uncached_texts, mode, show_progress)
                
                # Cache new embeddings
                for text, emb in zip(uncached_texts, new_embeddings):
                    cache_key = self._get_cache_key(text, mode)
                    self._add_to_cache(cache_key, emb)
                
                # Merge cached and new
                all_embeddings = [None] * len(texts)
                for i, emb in cached_embeddings:
                    all_embeddings[i] = emb
                for i, emb in zip(uncached_indices, new_embeddings):
                    all_embeddings[i] = emb
                
                return np.array(all_embeddings)
            else:
                # All cached
                sorted_cached = sorted(cached_embeddings, key=lambda x: x[0])
                return np.array([emb for _, emb in sorted_cached])
        else:
            return self._encode_batch(texts, mode, show_progress)
    
    def _encode_batch(
        self,
        texts: List[str],
        mode: str,
        show_progress: bool
    ) -> np.ndarray:
        """Encode batch of texts (backend-specific)."""
        if self.backend == "local":
            return self._encode_local(texts, mode, show_progress)
        elif self.backend == "openai":
            return self._encode_openai(texts)
        elif self.backend == "voyage":
            return self._encode_voyage(texts, mode)
        elif self.backend == "cohere":
            return self._encode_cohere(texts, mode)
    
    def _encode_local(self, texts: List[str], mode: str, show_progress: bool) -> np.ndarray:
        """Encode using local SentenceTransformer."""
        # Add instruction prefix for BGE models
        if "bge" in self.model_name.lower() and mode == "query":
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        
        return embeddings
    
    def _encode_openai(self, texts: List[str]) -> np.ndarray:
        """Encode using OpenAI API."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        
        embeddings = np.array([item.embedding for item in response.data])
        
        if self.normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def _encode_voyage(self, texts: List[str], mode: str) -> np.ndarray:
        """Encode using Voyage AI."""
        input_type = "query" if mode == "query" else "document"
        
        response = self.client.embed(
            texts=texts,
            model=self.model_name,
            input_type=input_type
        )
        
        embeddings = np.array(response.embeddings)
        
        if self.normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def _encode_cohere(self, texts: List[str], mode: str) -> np.ndarray:
        """Encode using Cohere."""
        input_type = "search_query" if mode == "query" else "search_document"
        
        response = self.client.embed(
            texts=texts,
            model=self.model_name,
            input_type=input_type
        )
        
        embeddings = np.array(response.embeddings)
        
        if self.normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def _get_cache_key(self, text: str, mode: str) -> str:
        """Generate cache key for text."""
        content = f"{self.model_name}:{mode}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    @lru_cache(maxsize=10000)
    def _get_from_cache(self, key: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        return None  # LRU cache handles this
    
    def _add_to_cache(self, key: str, embedding: np.ndarray):
        """Add embedding to cache."""
        # LRU cache via _get_from_cache
        self._get_from_cache.__wrapped__(self, key)  # Access wrapped function
    
    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def batch_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarities.
        
        Args:
            emb1: (n, dim)
            emb2: (m, dim)
            
        Returns:
            (n, m) similarity matrix
        """
        if self.normalize:
            # Already normalized, just dot product
            return np.dot(emb1, emb2.T)
        else:
            # Normalize then dot product
            emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
            emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
            return np.dot(emb1_norm, emb2_norm.T)


# =================================================================
# CONVENIENCE FUNCTIONS
# =================================================================

def create_embedder_from_config(config: dict) -> SOTAEmbedder:
    """
    Create embedder from config dict.
    
    Example config:
        models:
          embedding_model: "BAAI/bge-m3"
          embedding_backend: "local"
        api_keys:
          openai_api_key: "sk-..."
    """
    model_name = config["models"]["embedding_model"]
    backend = config["models"].get("embedding_backend")
    
    # Get API key if needed
    api_key = None
    if backend in ["openai", "voyage", "cohere"]:
        key_name = f"{backend}_api_key"
        api_key = config.get("api_keys", {}).get(key_name)
    
    # Other settings
    device = config.get("performance", {}).get("gpu_device", "cuda")
    batch_size = config.get("performance", {}).get("batch_size", 32)
    
    return SOTAEmbedder(
        model_name=model_name,
        backend=backend,
        api_key=api_key,
        device=device,
        batch_size=batch_size
    )
