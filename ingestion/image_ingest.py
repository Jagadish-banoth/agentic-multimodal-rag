"""
ingestion/image_ingest.py
--------------------------
Production-grade multimodal image ingestion pipeline.

Combines:
  - CLIP image encoder (semantic embeddings)
  - BLIP-2 image captioning (text generation)
  - Tesseract OCR (text extraction)

Flow: PDF/Image → Extract → CLIP encode → BLIP-2 caption → OCR → Fuse → Store metadata + embeddings

Output metadata fields:
  - chunk_id, modality, source_file, source, image_path
  - caption (BLIP-2), ocr_text (Tesseract), combined_content (fused)
  - image_width, image_height
  
Embeddings: Stored separately in data/processed/image_embeddings.npz for FAISS indexing
"""

import json
import logging
import re
import yaml
from pathlib import Path
from typing import List, Dict, Optional

from wrappers.multimodal_image_processor import MultimodalImageProcessor
from ingestion.extract_images import extract_images

logger = logging.getLogger(__name__)

# Config paths
PROCESSED_DIR = Path("data/processed")
CHUNK_FILE = PROCESSED_DIR / "chunks.jsonl"
EMBEDDING_FILE = PROCESSED_DIR / "image_embeddings.npz"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_config(path: str = "config/settings.yaml") -> Dict:
    """Load config from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ingest_images_from_file(
    file_path: Path,
    processor: MultimodalImageProcessor
) -> int:
    """
    Process images from a document file using industrial-grade processor.
    
    Args:
        file_path: Path to source document
        processor: MultimodalImageProcessor instance
    
    Returns:
        Number of images processed
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return 0
    
    logger.info(f"Extracting images from: {file_path.name}")
    
    try:
        image_paths = extract_images(file_path)
    except Exception as e:
        logger.error(f"Image extraction failed for {file_path}: {e}")
        return 0
    
    if not image_paths:
        logger.info(f"No images found in: {file_path.name}")
        return 0
    
    logger.info(f"Found {len(image_paths)} images in {file_path.name}")
    
    # Process batch with CLIP + BLIP-2 + OCR
    chunk_id_prefix = f"{file_path.stem}_img"
    results = processor.process_images_batch(image_paths, file_path, chunk_id_prefix)
    
    if not results:
        logger.warning(f"No images successfully processed from {file_path.name}")
        return 0
    
    # Save chunk records to JSONL
    processor.save_chunk_records(results, CHUNK_FILE)
    
    # Save embeddings for FAISS indexing
    processor.save_embeddings(results, EMBEDDING_FILE)
    
    logger.info(f"✓ Processed {len(results)} images from {file_path.name}")
    return len(results)


def ingest_all_images(raw_dir: Path = Path("data/raw/text")) -> int:
    """
    Process all documents in raw_dir for images using industrial-grade processor.
    
    Returns:
        Total number of images processed
    """
    config = load_config()
    processor = MultimodalImageProcessor(config)
    
    logger.info(f"Image processor: {processor}")
    logger.info(f"Image captioning: {config['image_processing']['image_captioning_model']}")
    logger.info(f"Image encoder: {config['image_processing']['image_encoder']}")
    logger.info(f"OCR enabled: {config['image_processing']['ocr_enabled']}")
    
    supported_exts = [".pdf", ".docx", ".pptx", ".html", ".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    total = 0
    
    for ext in supported_exts:
        for file_path in raw_dir.glob(f"*{ext}"):
            count = ingest_images_from_file(file_path, processor)
            total += count
    
    logger.info(f"\n{'='*70}")
    logger.info(f"✓ Total images ingested: {total}")
    logger.info(f"  Chunks saved to: {CHUNK_FILE}")
    logger.info(f"  Embeddings saved to: {EMBEDDING_FILE}")
    logger.info(f"{'='*70}\n")
    
    return total


# ------------------------------------
# Programmatic API for Streamlit/app integration
# ------------------------------------
def ingest_image(uploaded_file):
    """
    Ingest an image or document file from a file-like object (e.g., Streamlit upload).
    Returns a document ID (filename) or status.
    """
    import tempfile
    from pathlib import Path

    raw_dir = Path("data/raw/text")
    raw_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, dir=raw_dir, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    config = load_config()
    processor = MultimodalImageProcessor(config)
    count = ingest_images_from_file(tmp_path, processor)
    if count == 0:
        logger.warning(f"No images processed for {uploaded_file.name}")
        return None
    return tmp_path.name

# ------------------------------------
# CLI / Test
# ------------------------------------
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
        config = load_config()
        processor = MultimodalImageProcessor(config)
        count = ingest_images_from_file(file_path, processor)
        print(f"\n✓ Processed {count} images from {file_path}")
    else:
        # Process all files in raw directory
        count = ingest_all_images()
        print(f"\n✓ Processed {count} images total")

