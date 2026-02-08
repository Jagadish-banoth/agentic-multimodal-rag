"""
wrappers/image_encoder.py
--------------------------
Production-grade CLIP-based image encoder for multimodal RAG.

Features:
- OpenAI CLIP ViT-Large for semantic image embeddings
- Batch processing for efficiency
- GPU acceleration support
- Image preprocessing (normalization, resizing)
- Caching and lazy-loading
- Error handling and graceful fallback

Usage:
    encoder = ImageEncoder(config)
    embeddings = encoder.encode_images(image_paths)  # Returns (N, 768) numpy array
    similarity = encoder.similarity(text_query, image_embeddings)
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Union, Tuple, Optional, Dict
import torch
from PIL import Image
import warnings

logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


class ImageEncoder:
    """
    Industry-grade image encoder using OpenAI CLIP ViT-Large.
    """

    def __init__(self, config: Dict, model_name: str = "openai/clip-vit-large-patch14"):
        """
        Initialize image encoder.
        
        Args:
            config: Configuration dict
            model_name: HuggingFace model identifier
        """
        self.config = config
        self.model_name = model_name
        self.batch_size = config.get("image_processing", {}).get("image_encoder_batch_size", 32)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Lazy-load CLIP model and processor."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            logger.info(f"Loading CLIP model: {self.model_name} on {self.device}")
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()  # Set to eval mode
            logger.info("CLIP model loaded successfully")
        except ImportError as e:
            logger.error(f"transformers library not found: {e}")
            raise RuntimeError("Install transformers: pip install transformers")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

    def encode_images(
        self,
        image_paths: List[Union[str, Path]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode images to embedding vectors.
        
        Args:
            image_paths: List of image file paths
            normalize: Whether to L2-normalize embeddings
        
        Returns:
            (N, 768) numpy array of image embeddings
        """
        if not image_paths:
            return np.array([]).reshape(0, 768)

        embeddings = []
        
        # Process in batches
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    img = Image.open(str(path)).convert("RGB")
                    batch_images.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load image {path}: {e}")
                    # Skip bad images
                    continue
            
            if not batch_images:
                continue
            
            # Encode batch
            with torch.no_grad():
                inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
                image_features = self.model.get_image_features(**inputs)
                
                if normalize:
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                embeddings.append(image_features.cpu().numpy())
        
        if embeddings:
            return np.vstack(embeddings)
        else:
            return np.array([]).reshape(0, 768)

    def encode_text(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode text to embedding vectors (for query matching).
        
        Args:
            texts: List of text strings
            normalize: Whether to L2-normalize embeddings
        
        Returns:
            (N, 768) numpy array of text embeddings
        """
        if not texts:
            return np.array([]).reshape(0, 768)

        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            
            with torch.no_grad():
                inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True).to(self.device)
                text_features = self.model.get_text_features(**inputs)
                
                if normalize:
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                embeddings.append(text_features.cpu().numpy())
        
        if embeddings:
            return np.vstack(embeddings)
        else:
            return np.array([]).reshape(0, 768)

    def similarity(
        self,
        query_text: str,
        image_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between text query and image embeddings.
        
        Args:
            query_text: Query text string
            image_embeddings: (N, 768) array of image embeddings
        
        Returns:
            (N,) array of similarity scores [0, 1]
        """
        if len(image_embeddings) == 0:
            return np.array([])
        
        # Encode query
        query_embedding = self.encode_text([query_text], normalize=True)  # (1, 768)
        
        # Compute cosine similarity
        # For normalized vectors: similarity = dot product
        similarities = np.dot(image_embeddings, query_embedding.T).flatten()  # (N,)
        
        # Scale to [0, 1]
        similarities = (similarities + 1) / 2
        return similarities

    def get_embedding_dim(self) -> int:
        """Return embedding dimension (768 for CLIP ViT-Large)."""
        return 768

    def __repr__(self) -> str:
        return f"ImageEncoder(model={self.model_name}, device={self.device})"
