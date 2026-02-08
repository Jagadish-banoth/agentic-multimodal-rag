"""
ingestion/audio_ingest.py
--------------------------
SOTA Audio ingestion using OpenAI Whisper for multilingual ASR.

Features:
- Whisper large-v3 for SOTA multilingual transcription
- Supports 100+ languages including Indian languages
- Automatic language detection
- Word-level timestamps
- Pure Python audio handling (no FFmpeg required)

Supports: .mp3, .wav, .m4a, .flac, .ogg, .webm, .wma, .aac

Output metadata fields:
  - chunk_id
  - modality: "audio"
  - source_file
  - source
  - transcript
  - segments (with timestamps)
  - language_detected
  - duration_seconds
  - content (transcript for embedding)
"""

import json
import logging
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any

# Suppress Whisper Triton warnings (requires CUDA toolkit, not needed for CPU fallback)
warnings.filterwarnings("ignore", message=".*Failed to launch Triton kernels.*")
warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
CHUNK_FILE = PROCESSED_DIR / "chunks.jsonl"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------
# Configuration
# ------------------------------------
# Use large-v3 for best multilingual support, medium for faster processing
WHISPER_MODEL_SIZE = "medium"  # Options: tiny, base, small, medium, large, large-v2, large-v3

# Supported languages (subset - Whisper supports 100+)
SUPPORTED_LANGUAGES = {
    "en": "english", "hi": "hindi", "te": "telugu", "ta": "tamil",
    "kn": "kannada", "ml": "malayalam", "bn": "bengali", "mr": "marathi",
    "gu": "gujarati", "pa": "punjabi", "ur": "urdu", "ne": "nepali",
    "zh": "chinese", "ja": "japanese", "ko": "korean", "ar": "arabic",
    "de": "german", "fr": "french", "es": "spanish", "pt": "portuguese",
    "ru": "russian", "it": "italian", "nl": "dutch", "pl": "polish",
}

# ------------------------------------
# Lazy-load Whisper model
# ------------------------------------
_whisper_model = None


def _get_whisper_model(model_size: str = None):
    """Lazy-load Whisper model."""
    global _whisper_model
    
    if model_size is None:
        model_size = WHISPER_MODEL_SIZE
    
    if _whisper_model is None:
        try:
            import whisper
            logger.info(f"Loading Whisper model: {model_size}")
            _whisper_model = whisper.load_model(model_size)
            logger.info("Whisper model loaded successfully")
        except ImportError:
            logger.error("Whisper not installed. Install with: pip install openai-whisper")
            raise
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    return _whisper_model


def _convert_audio_to_wav(audio_path: Path) -> Optional[Path]:
    """
    Convert audio to WAV format if needed.
    
    Note: Whisper's load_audio uses its own FFmpeg bindings internally,
    so many formats work directly without conversion.
    We only convert if absolutely necessary.
    """
    suffix = audio_path.suffix.lower()
    
    # Formats that Whisper handles directly via its FFmpeg bindings
    # (no separate conversion needed)
    whisper_native_formats = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"]
    
    if suffix in whisper_native_formats:
        logger.info(f"Format {suffix} supported natively by Whisper")
        return audio_path
    
    # For other formats, try pydub conversion
    try:
        from pydub import AudioSegment
        
        logger.info(f"Converting {audio_path.name} to WAV using pydub")
        
        # Load based on format
        if suffix == ".mp3":
            audio = AudioSegment.from_mp3(str(audio_path))
        elif suffix == ".m4a":
            audio = AudioSegment.from_file(str(audio_path), format="m4a")
        elif suffix == ".flac":
            audio = AudioSegment.from_file(str(audio_path), format="flac")
        elif suffix == ".ogg":
            audio = AudioSegment.from_ogg(str(audio_path))
        elif suffix == ".webm":
            audio = AudioSegment.from_file(str(audio_path), format="webm")
        elif suffix == ".wma":
            audio = AudioSegment.from_file(str(audio_path), format="wma")
        elif suffix == ".aac":
            audio = AudioSegment.from_file(str(audio_path), format="aac")
        else:
            audio = AudioSegment.from_file(str(audio_path))
        
        # Convert to 16kHz mono WAV (optimal for Whisper)
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        # Save to temp file
        temp_wav = Path(tempfile.gettempdir()) / f"{audio_path.stem}_converted.wav"
        audio.export(str(temp_wav), format="wav")
        
        logger.info(f"Converted to: {temp_wav}")
        return temp_wav
        
    except ImportError:
        logger.warning("pydub not installed. Install with: pip install pydub")
    except Exception as e:
        logger.warning(f"pydub conversion failed: {e}")
    
    # If pydub fails, try direct loading (Whisper can handle some formats)
    return audio_path


def transcribe_audio(
    audio_path: Path,
    language: str = None,
    word_timestamps: bool = True
) -> Dict[str, Any]:
    """
    Transcribe audio file using Whisper with multilingual support.
    
    Args:
        audio_path: Path to audio file
        language: Language code (e.g., 'hi' for Hindi). None for auto-detect.
        word_timestamps: Include word-level timestamps
    
    Returns:
        Dict with:
          - text: full transcript
          - segments: list of segments with timestamps
          - language: detected language code
          - language_name: detected language name
    """
    import whisper
    import numpy as np
    
    model = _get_whisper_model()
    audio_path = Path(audio_path)
    
    logger.info(f"Transcribing: {audio_path.name}")
    
    # Whisper transcription options
    options = {
        "verbose": False,
        "fp16": False,  # Disable FP16 to avoid CUDA warnings on CPU
    }
    
    # Word timestamps can cause Triton warnings without CUDA toolkit
    # Disable if running on CPU to avoid warnings
    if word_timestamps:
        try:
            import torch
            if torch.cuda.is_available():
                options["word_timestamps"] = True
            else:
                logger.info("Running on CPU - word timestamps disabled to avoid warnings")
                options["word_timestamps"] = False
        except ImportError:
            options["word_timestamps"] = False
    
    # If language specified, use it; otherwise auto-detect
    if language:
        if language in SUPPORTED_LANGUAGES:
            options["language"] = language
        else:
            logger.warning(f"Language '{language}' not in supported list, using auto-detect")
    
    # STRATEGY: Try librosa FIRST (pure Python, no FFmpeg dependency)
    # This avoids the Whisper load_audio WinError on systems without FFmpeg
    audio_loaded = False
    
    try:
        import librosa
        logger.debug("Loading audio with librosa (primary method)")
        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        audio = audio.astype(np.float32)
        audio_loaded = True
        
        result = model.transcribe(audio, **options)
    except ImportError:
        logger.debug("librosa not available, trying Whisper's built-in loader")
    except Exception as e:
        logger.warning(f"librosa failed: {e}, trying Whisper's built-in loader")
    
    # FALLBACK: Try Whisper's built-in FFmpeg loader
    if not audio_loaded:
        try:
            audio = whisper.load_audio(str(audio_path))
            result = model.transcribe(audio, **options)
            audio_loaded = True
        except Exception as e1:
            logger.warning(f"Whisper load_audio failed: {e1}")
            
            # LAST RESORT: Try soundfile for WAV files
            try:
                import soundfile as sf
                logger.info("Attempting to load audio with soundfile")
                audio, sr = sf.read(str(audio_path))
                
                # Convert to mono if stereo
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                
                # Resample to 16kHz if needed
                if sr != 16000:
                    try:
                        import scipy.signal
                        audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr))
                    except ImportError:
                        logger.warning("scipy not available for resampling")
                        if sr != 16000:
                            raise RuntimeError(f"Audio is {sr}Hz but scipy not available for resampling to 16kHz")
                
                audio = audio.astype(np.float32)
                result = model.transcribe(audio, **options)
                audio_loaded = True
            except Exception as e3:
                logger.error(f"All audio loading methods failed: {e3}")
                raise RuntimeError(
                    f"Cannot load audio. Solutions:\n"
                    f"  1. Install librosa: pip install librosa\n"
                    f"  2. Install FFmpeg: https://ffmpeg.org/download.html\n"
                    f"  3. Convert to WAV 16kHz mono first\n"
                    f"Original error: {e1}"
                )
    
    # Enhance result with language name
    detected_lang = result.get("language", "en")
    result["language_name"] = SUPPORTED_LANGUAGES.get(detected_lang, detected_lang)
    
    return result


def _get_audio_duration(audio_path: Path) -> Optional[float]:
    """Get audio duration in seconds using mutagen (pure Python)."""
    # Try mutagen first (pure Python, works without FFmpeg)
    try:
        from mutagen import File
        audio = File(str(audio_path))
        if audio and audio.info:
            return audio.info.length
    except ImportError:
        logger.warning("mutagen not installed. Install with: pip install mutagen")
    except Exception as e:
        logger.debug(f"mutagen failed: {e}")
    
    # Try pydub
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(audio_path))
        return len(audio) / 1000.0  # pydub returns milliseconds
    except Exception as e:
        logger.debug(f"pydub duration failed: {e}")
    
    return None


def ingest_audio_file(audio_path: Path) -> int:
    """
    Ingest a single audio file.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Number of chunks created (1 if successful)
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return 0
    
    try:
        result = transcribe_audio(audio_path)
    except Exception as e:
        logger.error(f"Transcription failed for {audio_path}: {e}")
        return 0
    
    transcript = result.get("text", "").strip()
    if not transcript:
        logger.warning(f"Empty transcript for {audio_path}")
        return 0
    
    duration = _get_audio_duration(audio_path)
    
    record = {
        "chunk_id": f"audio_{audio_path.stem}",
        "modality": "audio",
        "source_file": audio_path.name,
        "source": str(audio_path),
        "transcript": transcript,
        "language": result.get("language", "en"),
        "duration_seconds": duration,
        "content": transcript,  # For embedding
    }
    
    # Persist
    with CHUNK_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    logger.info(f"Ingested audio: {audio_path.name} ({len(transcript)} chars)")
    return 1


def ingest_all_audio(raw_dir: Path = Path("data/raw/audio")) -> int:
    """
    Process all audio files in directory.
    
    Returns:
        Total number of files processed
    """
    supported_exts = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"]
    total = 0
    
    if not raw_dir.exists():
        logger.warning(f"Audio directory not found: {raw_dir}")
        return 0
    
    for ext in supported_exts:
        for audio_path in raw_dir.glob(f"*{ext}"):
            count = ingest_audio_file(audio_path)
            total += count
    
    logger.info(f"Total audio files ingested: {total}")
    return total


    return total

# ------------------------------------
# Programmatic API for Streamlit/app integration
# ------------------------------------
def ingest_audio(uploaded_file):
    """
    Ingest an audio file from a file-like object (e.g., Streamlit upload).
    Returns a document ID (filename) or status.
    """
    import tempfile
    from pathlib import Path

    raw_dir = Path("data/raw/audio")
    raw_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, dir=raw_dir, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    try:
        count = ingest_audio_file(tmp_path)
        if count == 0:
            logger.warning(f"No audio processed for {uploaded_file.name}")
            return None
        return tmp_path.name
    except Exception as e:
        logger.warning(f"Audio ingestion failed for {uploaded_file.name}: {e}")
        return None

# ------------------------------------
# CLI / Test
# ------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])
        count = ingest_audio_file(audio_path)
        print(f"Processed {count} audio file(s)")
    else:
        count = ingest_all_audio()
        print(f"Processed {count} audio files total")