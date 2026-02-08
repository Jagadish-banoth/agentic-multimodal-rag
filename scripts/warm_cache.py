"""
Warm embedding cache from existing index.

Usage:
    python scripts/warm_cache.py
"""
import json
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
    """Warm cache with document embeddings from metadata."""
    # Load config
    config_path = ROOT / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize cache
    cache = EmbeddingCache(config)
    
    if not cache.enabled:
        logger.error("Cache is disabled - exiting")
        return
    
    # Load metadata
    meta_path = ROOT / "data" / "index" / "meta.jsonl"
    
    if not meta_path.exists():
        logger.error(f"Metadata not found: {meta_path}")
        return
    
    logger.info(f"Loading metadata from {meta_path}")
    
    metadata = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))
    
    logger.info(f"Loaded {len(metadata)} metadata records")
    
    # Extract doc_id → embedding mapping
    # Note: embeddings are not stored in meta.jsonl by default
    # This would require re-computing or storing them separately
    
    # For demonstration, we'll cache the metadata structure
    # In production, you'd recompute embeddings or load from a separate store
    
    logger.info("Cache warm-up requires embeddings to be pre-computed")
    logger.info("Consider storing embeddings alongside metadata for faster warm-up")
    logger.info("\nAlternatively, embeddings will be cached on first retrieval")
    
    logger.info(f"\nCache stats: {cache.get_stats()}")


if __name__ == "__main__":
    main()
