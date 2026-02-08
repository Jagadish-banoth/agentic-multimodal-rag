"""
Model loading utilities for embedding and generation models.

Supports both unified (single model) and dual (text + image) embedding strategies.
"""
import logging
from pathlib import Path
from typing import Union

import torch
import yaml
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "settings.yaml"


def load_config() -> dict:
    """Load configuration from settings.yaml."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_embedding_model() -> SentenceTransformer:
    """
    Load primary text embedding model from config.
    
    Returns:
        SentenceTransformer model instance
    """
    config = load_config()
    model_name = config["models"]["embedding_model"]
    
    logger.info(f"Loading embedding model: {model_name}")
    
    try:
        model = SentenceTransformer(
            model_name,
            trust_remote_code=config["models"].get("trust_remote_code", True),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        logger.info(f"✓ Loaded {model_name} ({model.get_sentence_embedding_dimension()}-dim on {model.device})")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise


def get_image_embedding_model() -> SentenceTransformer:
    """
    Load image embedding model (CLIP) from config.
    
    Returns:
        SentenceTransformer model instance
    """
    config = load_config()
    
    # Try multiple config locations for image model
    model_name = (
        config.get("models", {}).get("image_embedding_model") or
        config.get("image_processing", {}).get("image_encoder") or
        "openai/clip-vit-large-patch14"
    )
    
    logger.info(f"Loading image embedding model: {model_name}")
    
    try:
        model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        logger.info(f"✓ Loaded {model_name} ({model.get_sentence_embedding_dimension()}-dim on {model.device})")
        return model
    except Exception as e:
        logger.error(f"Failed to load image embedding model: {e}")
        raise


def get_embedding_dim() -> int:
    """Get text embedding dimension from config."""
    config = load_config()
    
    model_name = config["models"]["embedding_model"]
    
    # Known dimensions
    if "bge-m3" in model_name.lower():
        return 1024
    if "clip" in model_name.lower():
        return 768
    if "nomic" in model_name.lower():
        return 768
    
    # Fallback: load model to check
    try:
        model = get_embedding_model()
        return model.get_sentence_embedding_dimension()
    except Exception:
        return 1024  # Default to BAAI dimension


def get_image_embedding_dim() -> int:
    """Get image embedding dimension from config."""
    config = load_config()
    return config.get("image_processing", {}).get("image_encoder_dim", 768)


def get_tokenizer_name() -> str:
    """Get tokenizer name from config."""
    config = load_config()
    return config["models"]["tokenizer_model"]


def get_indexing_strategy() -> str:
    """Get indexing strategy from config."""
    config = load_config()
    return config.get("indexing", {}).get("strategy", "dual")