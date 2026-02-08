"""
Dual embedder for multimodal RAG.

Strategy:
- Text chunks → BAAI/bge-m3 (1024-dim, SOTA text retrieval)
- Image chunks → CLIP (768-dim, SOTA multimodal)
- Audio/Video → Text representation → BAAI/bge-m3

This provides SOTA performance for both text and image retrieval
while avoiding dimension mismatch issues.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import yaml
from PIL import Image
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "settings.yaml"


def load_config() -> dict:
    """Load configuration from settings.yaml."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class DualEmbedder:
    """
    Dual embedding model for multimodal RAG.
    
    Uses two specialized models:
    - BAAI/bge-m3 (1024-dim): SOTA for text-to-text semantic search
    - CLIP (768-dim): SOTA for image-text cross-modal search
    
    This architecture provides best-in-class performance for both modalities
    without compromising on either.
    """
    
    def __init__(
        self,
        text_model: str = "BAAI/bge-m3",
        image_model: str = "openai/clip-vit-large-patch14",
        device: Optional[str] = None,
        enable_image: bool = True,
    ):
        """
        Initialize dual embedder.
        
        Args:
            text_model: Model for text embeddings (default: BAAI/bge-m3)
            image_model: Model for image embeddings (default: CLIP)
            device: Device to use (auto-detect if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize text embedder (BAAI/bge-m3)
        logger.info(f"Loading text embedder: {text_model}")
        self.text_model = SentenceTransformer(
            text_model,
            trust_remote_code=True,
            device=self.device,
        )
        self.text_dim = self.text_model.get_sentence_embedding_dimension()
        logger.info(f"✓ Text embedder loaded: {text_model} ({self.text_dim}-dim)")
        
        # Initialize image embedder (CLIP) only when needed
        self.image_dim = 768
        self.clip_model = None
        self._use_transformers_clip = False
        if enable_image:
            logger.info(f"Loading image embedder: {image_model}")
            self._init_clip_model(image_model)
            logger.info(f"✓ Image embedder loaded: {image_model} ({self.image_dim}-dim)")
        else:
            logger.info("Image embedder disabled (no image data detected)")
        
        # Store model names
        self.text_model_name = text_model
        self.image_model_name = image_model
        self.enable_image = enable_image
    
    def _init_clip_model(self, model_name: str):
        """Initialize CLIP model using transformers library directly."""
        try:
            from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
            
            self.clip_model = CLIPModel.from_pretrained(model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(model_name)
            
            self.clip_model.to(self.device)
            self.clip_model.eval()
            
            # Get embedding dimension from vision config
            self.image_dim = self.clip_model.config.projection_dim
            if self.image_dim is None:
                self.image_dim = self.clip_model.config.vision_config.hidden_size
            
            self._use_transformers_clip = True
            logger.info(f"✓ CLIP loaded via transformers ({self.image_dim}-dim)")
            
        except Exception as e:
            logger.warning(f"Failed to load CLIP via transformers: {e}")
            logger.info("Falling back to sentence-transformers CLIP wrapper...")
            
            # Try sentence-transformers clip model
            try:
                self.clip_model = SentenceTransformer("clip-ViT-L-14")
                self.image_dim = self.clip_model.get_sentence_embedding_dimension()
                self._use_transformers_clip = False
                logger.info(f"✓ CLIP loaded via sentence-transformers ({self.image_dim}-dim)")
            except Exception as e2:
                logger.error(f"Failed to load CLIP model: {e2}")
                # Final fallback - set image_dim to expected CLIP dimension
                self.image_dim = 768
                self._use_transformers_clip = False
                self.clip_model = None
                logger.warning("CLIP model unavailable - image embeddings will be zero vectors")
    
    def embed_text(
        self,
        texts: list,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed text using BAAI/bge-m3 (SOTA for text retrieval).
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
        
        Returns:
            numpy array of shape (n_texts, 1024)
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.text_dim)
        
        embeddings = self.text_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)
    
    def embed_images(
        self,
        image_paths: list,
        batch_size: int = 16,
        show_progress: bool = False,
    ) -> Tuple[np.ndarray, list]:
        """
        Embed images using CLIP.
        
        Args:
            image_paths: List of image file paths
            batch_size: Batch size for encoding
            show_progress: Show progress bar
        
        Returns:
            Tuple of (embeddings array, list of valid indices)
        """
        if not image_paths:
            return np.array([], dtype=np.float32).reshape(0, self.image_dim), []
        
        if self.clip_model is None:
            logger.warning("CLIP model not available, returning zero vectors")
            return np.zeros((len(image_paths), self.image_dim), dtype=np.float32), list(range(len(image_paths)))
        
        embeddings = []
        valid_indices = []
        
        for i, path in enumerate(image_paths):
            try:
                image = Image.open(path).convert("RGB")
                
                if self._use_transformers_clip:
                    inputs = self.clip_processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        image_features = self.clip_model.get_image_features(**inputs)
                        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    
                    embedding = image_features.cpu().numpy().flatten()
                else:
                    embedding = self.clip_model.encode(image, convert_to_numpy=True)
                
                embeddings.append(embedding)
                valid_indices.append(i)
                
            except Exception as e:
                logger.warning(f"Failed to encode image {path}: {e}")
                continue
        
        if not embeddings:
            return np.array([], dtype=np.float32).reshape(0, self.image_dim), []
        
        return np.array(embeddings, dtype=np.float32), valid_indices
    
    def embed_image_text(
        self,
        texts: list,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed text using CLIP for cross-modal search.
        
        Use this when you want to search images with text queries.
        
        Args:
            texts: List of text strings (queries or captions)
            batch_size: Batch size for encoding
            show_progress: Show progress bar
        
        Returns:
            numpy array of shape (n_texts, 768)
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.image_dim)
        
        if self.clip_model is None:
            logger.warning("CLIP model not available, returning zero vectors")
            return np.zeros((len(texts), self.image_dim), dtype=np.float32)
        
        # Fast path: batch encode via transformers CLIP
        if self._use_transformers_clip:
            all_embeddings: List[np.ndarray] = []
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                try:
                    inputs = self.clip_tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=77,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        text_features = self.clip_model.get_text_features(**inputs)
                        text_features = text_features / text_features.norm(
                            p=2, dim=-1, keepdim=True
                        )
                    batch_emb = text_features.detach().cpu().numpy().astype(np.float32)
                    all_embeddings.append(batch_emb)
                except Exception as e:
                    logger.warning(f"Failed to batch-encode CLIP texts: {e}")
                    all_embeddings.append(
                        np.zeros((len(batch_texts), self.image_dim), dtype=np.float32)
                    )
            if not all_embeddings:
                return np.array([], dtype=np.float32).reshape(0, self.image_dim)
            return np.vstack(all_embeddings)

        # sentence-transformers CLIP wrapper supports batched encode
        try:
            embeddings = self.clip_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(embeddings, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to encode text for CLIP via sentence-transformers: {e}")
            # Fallback: per-text loop with robust error handling
            out = []
            for text in texts:
                try:
                    emb = self.clip_model.encode(text, convert_to_numpy=True)
                    out.append(np.asarray(emb, dtype=np.float32).flatten())
                except Exception:
                    out.append(np.zeros(self.image_dim, dtype=np.float32))
            return np.array(out, dtype=np.float32)
    
    def embed_chunk(self, chunk: dict) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Embed a chunk based on its modality.
        
        Returns embeddings for BOTH indices:
        - text_embedding: For BAAI index (1024-dim)
        - image_embedding: For CLIP index (768-dim)
        
        Args:
            chunk: Chunk dict with modality, content, image_path, etc.
        
        Returns:
            Tuple of (text_embedding, image_embedding, modality)
        """
        modality = chunk.get("modality", "text")
        
        # Get text content (all modalities have some text representation)
        if modality == "image":
            text_content = chunk.get("combined_content") or chunk.get("caption") or ""
        elif modality == "audio":
            text_content = chunk.get("transcript") or chunk.get("audio_transcript") or ""
        elif modality == "video":
            text_content = " ".join(filter(None, [
                chunk.get("frame_caption", ""),
                chunk.get("frame_ocr", ""),
                chunk.get("audio_transcript", ""),
            ]))
        else:
            text_content = chunk.get("content", "")
        
        # Always embed text content with BAAI (for text index)
        text_embedding = self.embed_text([text_content])[0] if text_content else np.zeros(self.text_dim, dtype=np.float32)
        
        # For image modality, also embed with CLIP
        if modality == "image":
            image_path = chunk.get("image_path")
            if image_path and Path(image_path).exists():
                # Use actual image embedding
                img_emb, valid = self.embed_images([image_path])
                if len(valid) > 0:
                    image_embedding = img_emb[0]
                else:
                    # Fallback to text embedding via CLIP
                    image_embedding = self.embed_image_text([text_content])[0]
            else:
                # Use text representation via CLIP
                image_embedding = self.embed_image_text([text_content])[0] if text_content else np.zeros(self.image_dim, dtype=np.float32)
        else:
            # For non-image modalities, use text via CLIP
            image_embedding = self.embed_image_text([text_content])[0] if text_content else np.zeros(self.image_dim, dtype=np.float32)
        
        return text_embedding, image_embedding, modality
    
    def get_text_dim(self) -> int:
        """Return text embedding dimension (1024 for BAAI)."""
        return self.text_dim
    
    def get_image_dim(self) -> int:
        """Return image embedding dimension (768 for CLIP)."""
        return self.image_dim
    
    def get_info(self) -> dict:
        """Return embedder information."""
        return {
            "text_model": self.text_model_name,
            "text_dim": self.text_dim,
            "image_model": self.image_model_name,
            "image_dim": self.image_dim,
            "device": self.device,
        }


def create_dual_embedder_from_config(config: Optional[dict] = None) -> DualEmbedder:
    """
    Create dual embedder from config.
    
    Args:
        config: Config dict (loads from settings.yaml if None)
    
    Returns:
        DualEmbedder instance
    """
    if config is None:
        config = load_config()
    
    text_model = config.get("models", {}).get("embedding_model", "BAAI/bge-m3")
    image_model = config.get("models", {}).get("image_embedding_model", "openai/clip-vit-large-patch14")
    
    # Fallback for image model
    if not image_model:
        image_model = config.get("image_processing", {}).get("image_encoder", "openai/clip-vit-large-patch14")
    
    # Respect config strategy: allow forcing text-only.
    strategy = (config.get("indexing", {}) or {}).get("strategy", "dual")
    enable_image = strategy != "text_only"
    return DualEmbedder(text_model=text_model, image_model=image_model, enable_image=enable_image)