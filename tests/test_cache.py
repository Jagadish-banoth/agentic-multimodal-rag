"""
Test script for embedding cache.

Usage:
    python scripts/test_cache.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.embedding_cache import EmbeddingCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Test embedding cache functionality."""
    # Load config
    config_path = ROOT / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize cache
    cache = EmbeddingCache(config)
    
    if not cache.enabled:
        logger.error("❌ Cache is disabled - check Redis connection")
        logger.info("To enable cache:")
        logger.info("  1. Install Redis: https://redis.io/download")
        logger.info("  2. Start Redis server: redis-server")
        logger.info("  3. Install Python client: pip install redis hiredis")
        return
    
    logger.info("✓ Cache initialized")
    
    # Test query embeddings
    logger.info("\n=== Testing Query Embeddings ===")
    test_queries = [
        "What is machine learning?",
        "Explain transformers",
        "How does attention work?",
    ]
    
    model_name = "test-model"
    
    # First pass - cache misses
    logger.info("Pass 1: Cache misses expected")
    for query in test_queries:
        embedding = np.random.randn(768).astype(np.float32)
        cache.set_query_embedding(query, embedding, model_name)
        logger.info(f"  Cached: {query}")
    
    # Second pass - cache hits
    logger.info("\nPass 2: Cache hits expected")
    for query in test_queries:
        result = cache.get_query_embedding(query, model_name)
        if result is not None:
            logger.info(f"  ✓ HIT: {query}")
        else:
            logger.error(f"  ✗ MISS: {query}")
    
    # Test document embeddings (batch)
    logger.info("\n=== Testing Document Embeddings (Batch) ===")
    doc_embeddings = {
        f"doc_{i}": np.random.randn(768).astype(np.float32)
        for i in range(10)
    }
    
    cache.set_doc_embeddings_batch(doc_embeddings, model_name)
    logger.info(f"  Cached {len(doc_embeddings)} documents")
    
    # Retrieve batch
    doc_ids = list(doc_embeddings.keys())
    results = cache.get_doc_embeddings_batch(doc_ids, model_name)
    
    hits = sum(1 for v in results.values() if v is not None)
    logger.info(f"  Retrieved: {hits}/{len(doc_ids)} hits")
    
    # Stats
    logger.info("\n=== Cache Statistics ===")
    stats = cache.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Test invalidation
    logger.info("\n=== Testing Cache Invalidation ===")
    cache.invalidate_query_cache(model_name)
    logger.info("  Invalidated query cache")
    
    # Verify invalidation
    result = cache.get_query_embedding(test_queries[0], model_name)
    if result is None:
        logger.info("  ✓ Cache successfully invalidated")
    else:
        logger.error("  ✗ Cache invalidation failed")
    
    logger.info("\n=== Test Complete ===")
    logger.info(f"Final stats: {cache.get_stats()}")


if __name__ == "__main__":
    main()
