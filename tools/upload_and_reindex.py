# tools/upload_and_reindex.py
"""
========================================================================
MULTIMODAL FILE UPLOAD & FULL PIPELINE ORCHESTRATION
========================================================================

Upload files (single file or directory) and orchestrate complete processing:
  1️⃣  Text ingestion (PDFs, DOCX, PPTX, HTML, TXT → chunks.jsonl)
  2️⃣  Image ingestion (CLIP + BLIP-2 + OCR → image embeddings + metadata)
  3️⃣  Audio ingestion (Whisper ASR → transcripts + timestamps)
  4️⃣  Video ingestion (Keyframes + Audio + VLM → frame descriptions + transcripts)
  5️⃣  Embedding & FAISS indexing (unified index across all modalities)

Automatically detects file types and routes to appropriate ingestion pipeline.
Supports batch processing of multiple files with modality auto-detection.

USAGE:
  # Single file
  python -m tools.upload_and_reindex path/to/doc.pdf

  # Directory (auto-detects all modalities)
  python -m tools.upload_and_reindex path/to/mixed_data/

  # Keep raw files (don't delete old documents before copying)
  python -m tools.upload_and_reindex --keep-raw path/to/doc.pdf

  # Only copy raw files, skip ingestion
  python -m tools.upload_and_reindex --no-ingest path/to/docs/

  # Specific modalities only
  python -m tools.upload_and_reindex --text-only path/to/docs/
  python -m tools.upload_and_reindex --skip-audio path/to/mixed/

SUPPORTED FORMATS:
  • Text: .pdf, .txt, .docx, .pptx, .html
  • Image: .jpg, .jpeg, .png, .tif, .tiff, .bmp, .webp
  • Audio: .mp3, .wav, .m4a, .flac, .ogg, .webm, .wma, .aac, .aiff
  • Video: .mp4, .avi, .mkv, .mov, .webm, .flv, .wmv, .m4v
"""

import argparse
import shutil
import subprocess
import sys
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Project root (…/agentic-multimodal-rag)
ROOT = Path(__file__).resolve().parents[1]

# File extensions by modality
TEXT_EXTS = {".pdf", ".txt", ".docx", ".pptx", ".html"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".wma", ".aac", ".aiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".m4v"}
ALL_SUPPORTED_EXTS = TEXT_EXTS | IMAGE_EXTS | AUDIO_EXTS | VIDEO_EXTS


# ============================================================================
# CONFIGURATION & HELPERS
# ============================================================================

def load_config() -> Dict:
    """Load config from YAML."""
    cfg_path = ROOT / "config" / "settings.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return yaml.safe_load(cfg_path.open("r", encoding="utf-8"))


def run_command(cmd: List[str], description: str = None) -> int:
    """Run command and return exit code."""
    if description:
        logger.info(f"▶️  {description}")
    logger.debug(f"RUN: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(ROOT))
    if res.returncode != 0:
        logger.error(f"Command failed: {' '.join(cmd)} (exit code {res.returncode})")
        return res.returncode
    return 0


def detect_modalities(files: List[Path]) -> Dict[str, List[Path]]:
    """Classify files by modality."""
    modalities = {
        "text": [],
        "image": [],
        "audio": [],
        "video": [],
        "unknown": []
    }
    
    for f in files:
        ext = f.suffix.lower()
        if ext in TEXT_EXTS:
            modalities["text"].append(f)
        elif ext in IMAGE_EXTS:
            modalities["image"].append(f)
        elif ext in AUDIO_EXTS:
            modalities["audio"].append(f)
        elif ext in VIDEO_EXTS:
            modalities["video"].append(f)
        else:
            modalities["unknown"].append(f)
    
    return modalities


def collect_files(src: Path) -> List[Path]:
    """Collect all supported files from source."""
    files = []
    
    if src.is_file():
        if src.suffix.lower() in ALL_SUPPORTED_EXTS:
            files.append(src)
        else:
            logger.warning(f"Unsupported file format: {src.suffix}")
    
    elif src.is_dir():
        for ext in ALL_SUPPORTED_EXTS:
            files.extend(src.glob(f"*{ext}"))
            files.extend(src.glob(f"*{ext.upper()}"))
    
    return sorted(files)


def copy_inputs(files: List[Path], modalities: Dict[str, List[Path]], 
                cfg: Dict, keep_raw: bool) -> None:
    """Copy files to appropriate raw directories."""
    raw_base = Path(cfg["ingestion"]["raw_dir"]).parent  # data/raw/
    
    # Clear old raw data if not keeping
    if not keep_raw:
        if raw_base.exists():
            logger.info(f"🧹 Clearing old raw data: {raw_base}")
            shutil.rmtree(raw_base)
    
    raw_base.mkdir(parents=True, exist_ok=True)
    
    # Copy by modality
    if modalities["text"]:
        text_dir = raw_base / "text"
        text_dir.mkdir(parents=True, exist_ok=True)
        for f in modalities["text"]:
            shutil.copy2(f, text_dir / f.name)
            logger.info(f"  ✓ Copied text: {f.name}")
    
    if modalities["image"]:
        image_dir = raw_base / "image"
        image_dir.mkdir(parents=True, exist_ok=True)
        for f in modalities["image"]:
            shutil.copy2(f, image_dir / f.name)
            logger.info(f"  ✓ Copied image: {f.name}")
    
    if modalities["audio"]:
        audio_dir = raw_base / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        for f in modalities["audio"]:
            shutil.copy2(f, audio_dir / f.name)
            logger.info(f"  ✓ Copied audio: {f.name}")
    
    if modalities["video"]:
        video_dir = raw_base / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        for f in modalities["video"]:
            shutil.copy2(f, video_dir / f.name)
            logger.info(f"  ✓ Copied video: {f.name}")
    
    if modalities["unknown"]:
        logger.warning(f"⚠️  Skipped {len(modalities['unknown'])} unsupported files")


def print_summary(modalities: Dict[str, List[Path]]) -> None:
    """Print file processing summary."""
    total = sum(len(v) for v in modalities.values() if v)
    logger.info("")
    logger.info("=" * 70)
    logger.info("📊 FILE SUMMARY")
    logger.info("=" * 70)
    if modalities["text"]:
        logger.info(f"  📝 Text files:     {len(modalities['text'])}")
    if modalities["image"]:
        logger.info(f"  📸 Image files:    {len(modalities['image'])}")
    if modalities["audio"]:
        logger.info(f"  🎵 Audio files:    {len(modalities['audio'])}")
    if modalities["video"]:
        logger.info(f"  🎬 Video files:    {len(modalities['video'])}")
    logger.info(f"  ✅ Total:          {total}")
    logger.info("=" * 70)


# ============================================================================
# INGESTION ORCHESTRATION
# ============================================================================

def run_ingestion_pipeline(cfg: Dict, modalities: Dict[str, List[Path]], 
                           skip_modalities: List[str] = None) -> int:
    """
    Run complete ingestion pipeline for all modalities.
    
    Args:
        cfg: Configuration dict
        modalities: Dict of files by modality type
        skip_modalities: List of modalities to skip (e.g., ["audio", "video"])
    
    Returns:
        0 if successful, non-zero on error
    """
    skip_modalities = skip_modalities or []
    
    # Clear processed data and index ONCE before any ingestion
    processed_dir = Path(cfg["ingestion"]["out_dir"])
    index_dir = ROOT / "data" / "index"
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("🧹 CLEARING OLD DATA")
    logger.info("=" * 70)
    
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        logger.info(f"  ✓ Removed: {processed_dir}")
    
    if index_dir.exists():
        shutil.rmtree(index_dir)
        logger.info(f"  ✓ Removed: {index_dir}")
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # STAGE 1: TEXT INGESTION (base chunks)
    # ================================================================
    if modalities["text"] and "text" not in skip_modalities:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"📝 TEXT INGESTION ({len(modalities['text'])} files)")
        logger.info("=" * 70)
        
        exitcode = run_command(
            [sys.executable, "-m", "ingestion.text_ingest"],
            "Processing text documents (PDF, DOCX, PPTX, HTML, TXT)..."
        )
        if exitcode != 0:
            logger.error("❌ Text ingestion failed")
            return exitcode
        logger.info("✅ Text ingestion complete")
    
    # ================================================================
    # STAGE 2: IMAGE INGESTION (CLIP + BLIP-2 + OCR)
    # ================================================================
    if modalities["image"] and "image" not in skip_modalities:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"📸 IMAGE INGESTION ({len(modalities['image'])} files)")
        logger.info("=" * 70)
        
        exitcode = run_command(
            [sys.executable, "-m", "ingestion.image_ingest"],
            "Processing images (CLIP encoder + BLIP-2 captions + Tesseract OCR)..."
        )
        if exitcode != 0:
            logger.error("❌ Image ingestion failed")
            return exitcode
        logger.info("✅ Image ingestion complete")
    
    # ================================================================
    # STAGE 3: AUDIO INGESTION (Whisper ASR)
    # ================================================================
    if modalities["audio"] and "audio" not in skip_modalities:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"🎵 AUDIO INGESTION ({len(modalities['audio'])} files)")
        logger.info("=" * 70)
        
        exitcode = run_command(
            [sys.executable, "-m", "ingestion.audio_ingest"],
            "Processing audio files (Whisper multilingual transcription)..."
        )
        if exitcode != 0:
            logger.error("❌ Audio ingestion failed")
            return exitcode
        logger.info("✅ Audio ingestion complete")
    
    # ================================================================
    # STAGE 4: VIDEO INGESTION (Keyframes + Audio + VLM)
    # ================================================================
    if modalities["video"] and "video" not in skip_modalities:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"🎬 VIDEO INGESTION ({len(modalities['video'])} files)")
        logger.info("=" * 70)
        
        exitcode = run_command(
            [sys.executable, "-m", "ingestion.video_ingest"],
            "Processing videos (keyframe extraction + Whisper + VLM analysis)..."
        )
        if exitcode != 0:
            logger.error("❌ Video ingestion failed")
            return exitcode
        logger.info("✅ Video ingestion complete")
    
    # ================================================================
    # STAGE 5: EMBEDDING & FAISS INDEXING (unified index)
    # ================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("🔢 EMBEDDING & INDEXING")
    logger.info("=" * 70)
    
    exitcode = run_command(
        [sys.executable, "-m", "scripts.embed_and_index"],
        "Building unified FAISS index across all modalities..."
    )
    if exitcode != 0:
        logger.error("❌ Embedding/indexing failed")
        return exitcode
    logger.info("✅ Embedding & indexing complete")
    
    return 0


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Upload files and run complete multimodal RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Single document
  %(prog)s data/my_paper.pdf

  # Directory with mixed modalities
  %(prog)s data/documents/

  # Keep old raw files (append new files)
  %(prog)s --keep-raw data/new_doc.pdf

  # Skip audio processing
  %(prog)s --skip-audio data/mixed/

  # Only copy raw files, don't ingest
  %(prog)s --no-ingest data/documents/
        """
    )
    
    parser.add_argument("src", help="Source file or directory")
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep old raw files (append instead of replace)"
    )
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Copy files only, skip ingestion and indexing"
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Process only text files"
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip audio ingestion"
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Skip video ingestion"
    )
    parser.add_argument(
        "--skip-image",
        action="store_true",
        help="Skip image ingestion"
    )
    
    args = parser.parse_args()
    
    # ================================================================
    # VALIDATION & SETUP
    # ================================================================
    src = Path(args.src).expanduser().resolve()
    if not src.exists():
        logger.error(f"❌ Source not found: {src}")
        sys.exit(1)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("🚀 MULTIMODAL RAG UPLOAD & INDEXING")
    logger.info("=" * 70)
    logger.info(f"Source: {src}")
    logger.info(f"Root:   {ROOT}")
    
    # Load config
    try:
        cfg = load_config()
        logger.info("✓ Configuration loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        sys.exit(1)
    
    # ================================================================
    # COLLECT & CLASSIFY FILES
    # ================================================================
    files = collect_files(src)
    
    if not files:
        logger.error(f"❌ No supported files found in {src}")
        sys.exit(1)
    
    modalities = detect_modalities(files)
    print_summary(modalities)
    
    # Filter by command-line flags
    if args.text_only:
        modalities["image"].clear()
        modalities["audio"].clear()
        modalities["video"].clear()
        logger.info("⚙️  Text-only mode enabled")
    
    skip_list = []
    if args.skip_audio:
        modalities["audio"].clear()
        skip_list.append("audio")
        logger.info("⚙️  Skipping audio")
    if args.skip_video:
        modalities["video"].clear()
        skip_list.append("video")
        logger.info("⚙️  Skipping video")
    if args.skip_image:
        modalities["image"].clear()
        skip_list.append("image")
        logger.info("⚙️  Skipping image")
    
    # ================================================================
    # COPY RAW FILES
    # ================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("📁 COPYING RAW FILES")
    logger.info("=" * 70)
    
    try:
        copy_inputs(files, modalities, cfg, keep_raw=args.keep_raw)
        logger.info("✅ Files copied successfully")
    except Exception as e:
        logger.error(f"❌ Failed to copy files: {e}")
        sys.exit(1)
    
    # ================================================================
    # OPTIONAL: SKIP INGESTION
    # ================================================================
    if args.no_ingest:
        logger.info("")
        logger.info("⚠️  Ingestion skipped (--no-ingest)")
        logger.info("Next step: python -m tools.upload_and_reindex --no-ingest")
        sys.exit(0)
    
    # ================================================================
    # RUN INGESTION PIPELINE
    # ================================================================
    exitcode = run_ingestion_pipeline(cfg, modalities, skip_modalities=skip_list)
    
    if exitcode != 0:
        logger.error("")
        logger.error("=" * 70)
        logger.error("❌ PIPELINE FAILED")
        logger.error("=" * 70)
        sys.exit(1)
    
    # ================================================================
    # SUCCESS
    # ================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Chat:      python chat.py")
    logger.info("  2. Demo:      python -m scripts.retrieve_demo")
    logger.info("  3. Evaluate:  python -m evaluation.eval_retrieval")
    logger.info("")
    logger.info(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()