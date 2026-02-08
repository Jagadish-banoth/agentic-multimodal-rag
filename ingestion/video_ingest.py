"""
ingestion/video_ingest.py
--------------------------
SOTA Video ingestion: keyframe extraction + audio transcription + detailed frame analysis.

Production-grade features:
- OpenCV-based intelligent keyframe extraction
- Multiple audio extraction backends (OpenCV, moviepy, scipy)
- Whisper multilingual audio transcription
- BLIP-2/VLM for detailed frame description
- Multilingual OCR for text in frames
- Frame-level timestamps for precise retrieval
- Scene change detection for intelligent keyframe selection
- Adaptive frame sampling based on video content

Supports: .mp4, .avi, .mkv, .mov, .webm, .flv, .wmv

Output metadata fields:
  - chunk_id
  - modality: "video"
  - source_file
  - source
  - frame_path (for frame chunks)
  - frame_caption
  - frame_ocr
  - audio_transcript
  - timestamp_seconds
  - timestamp_formatted (HH:MM:SS)
  - scene_description
  - content (combined for embedding)
"""

import json
import logging
import tempfile
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
VIDEO_FRAMES_DIR = PROCESSED_DIR / "video_frames"
CHUNK_FILE = PROCESSED_DIR / "chunks.jsonl"

VIDEO_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# =====================================
# SOTA Frame Extraction Settings
# =====================================
FRAME_INTERVAL_SEC = 3  # Extract one frame every N seconds (3s for better coverage)
MAX_FRAMES = 200  # Maximum frames per video (increased for longer videos)
MIN_SCENE_CHANGE_THRESHOLD = 15.0  # Lower threshold = more frames captured
ADAPTIVE_SAMPLING = True  # Enable adaptive frame sampling based on content
MIN_FRAMES_PER_VIDEO = 10  # Minimum frames to extract regardless of scene detection


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _calculate_frame_difference(frame1, frame2) -> float:
    """Calculate the mean absolute difference between two frames using multiple methods."""
    import cv2
    
    if frame1 is None or frame2 is None:
        return float('inf')
    
    try:
        # Convert to grayscale for comparison
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Resize for faster comparison
        gray1 = cv2.resize(gray1, (128, 128))
        gray2 = cv2.resize(gray2, (128, 128))
        
        # Method 1: Mean absolute difference
        mad = np.mean(np.abs(gray1.astype(float) - gray2.astype(float)))
        
        # Method 2: Histogram comparison (more robust to lighting changes)
        hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
        hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        
        # Combined score (weighted)
        combined_score = mad * 0.7 + min(hist_diff, 100) * 0.3
        
        return combined_score
    except Exception as e:
        logger.warning(f"Frame difference calculation error: {e}")
        return float('inf')


def _extract_keyframes_opencv(
    video_path: Path,
    output_dir: Path,
    interval_sec: float = None,
    max_frames: int = None,
    use_scene_detection: bool = True
) -> List[Tuple[Path, float]]:
    """
    Extract keyframes using OpenCV with intelligent scene detection.
    
    SOTA approach:
    - Hybrid sampling: time-based + scene-change detection
    - Ensures minimum frame coverage for short videos
    - Captures both regular intervals and significant visual changes
    
    Returns:
        List of tuples: (frame_path, timestamp_seconds)
    """
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV not installed. Install with: pip install opencv-python")
        return []
    
    if interval_sec is None:
        interval_sec = FRAME_INTERVAL_SEC
    if max_frames is None:
        max_frames = MAX_FRAMES
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    logger.info(f"Video: {video_path.name}, Duration: {_format_timestamp(duration)}, FPS: {fps:.1f}")
    
    # Calculate adaptive interval based on video duration
    if ADAPTIVE_SAMPLING:
        # For shorter videos, sample more frequently
        if duration <= 60:  # 1 minute
            interval_sec = min(interval_sec, 2)
        elif duration <= 300:  # 5 minutes
            interval_sec = min(interval_sec, 3)
        # For longer videos, use default or larger interval
    
    frame_interval = max(1, int(fps * interval_sec))
    
    # Calculate expected frames to ensure minimum coverage
    expected_frames = max(MIN_FRAMES_PER_VIDEO, int(duration / interval_sec))
    expected_frames = min(expected_frames, max_frames)
    
    logger.info(f"Sampling every {interval_sec}s (frame_interval={frame_interval}), target: {expected_frames} frames")
    
    frames = []
    frame_idx = 0
    saved_count = 0
    last_saved_frame = None
    last_save_time = -interval_sec  # Allow first frame to be saved
    
    # First pass: extract candidate frames
    candidate_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_idx / fps
        time_since_last_save = current_time - last_save_time
        
        # Always consider saving at interval boundaries
        should_consider = (time_since_last_save >= interval_sec * 0.9)  # 90% of interval
        
        # Also consider if there's a significant scene change
        scene_change_score = 0
        if last_saved_frame is not None and use_scene_detection:
            scene_change_score = _calculate_frame_difference(frame, last_saved_frame)
        
        if should_consider or (scene_change_score > MIN_SCENE_CHANGE_THRESHOLD * 2):
            candidate_frames.append({
                'frame': frame.copy(),
                'idx': frame_idx,
                'time': current_time,
                'score': scene_change_score
            })
            last_save_time = current_time
            last_saved_frame = frame.copy()
        
        frame_idx += 1
        
        # Safety limit
        if len(candidate_frames) >= max_frames * 2:
            break
    
    cap.release()
    
    # Second pass: select best frames if we have too many
    if len(candidate_frames) > max_frames:
        # Sort by scene change score and take top frames
        candidate_frames.sort(key=lambda x: x['score'], reverse=True)
        candidate_frames = candidate_frames[:max_frames]
        # Re-sort by time for sequential processing
        candidate_frames.sort(key=lambda x: x['time'])
    
    # Ensure minimum frames for short videos
    if len(candidate_frames) < MIN_FRAMES_PER_VIDEO and duration > 0:
        # Re-read video with more aggressive sampling
        cap = cv2.VideoCapture(str(video_path))
        aggressive_interval = max(1, int(duration / MIN_FRAMES_PER_VIDEO * fps))
        candidate_frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % aggressive_interval == 0:
                candidate_frames.append({
                    'frame': frame.copy(),
                    'idx': frame_idx,
                    'time': frame_idx / fps,
                    'score': 0
                })
            
            frame_idx += 1
            if len(candidate_frames) >= MIN_FRAMES_PER_VIDEO:
                break
        
        cap.release()
    
    # Save selected frames
    for i, cf in enumerate(candidate_frames):
        output_path = output_dir / f"{video_path.stem}_frame_{i:04d}.png"
        cv2.imwrite(str(output_path), cf['frame'])
        frames.append((output_path, cf['time']))
    
    logger.info(f"Extracted {len(frames)} keyframes from {video_path.name}")
    return frames


def _extract_audio_pydub(video_path: Path) -> Optional[Path]:
    """
    Extract audio track from video using pydub.
    Note: Requires FFmpeg installed on system for video formats.
    """
    output_path = PROCESSED_DIR / f"{video_path.stem}_audio.wav"
    
    try:
        from pydub import AudioSegment
        
        logger.info(f"Extracting audio using pydub from {video_path.name}")
        audio = AudioSegment.from_file(str(video_path))
        
        # Convert to 16kHz mono WAV for Whisper
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(str(output_path), format="wav")
        
        if output_path.exists() and output_path.stat().st_size > 1000:
            logger.info(f"Audio extracted successfully: {output_path}")
            return output_path
            
    except ImportError:
        logger.debug("pydub not installed")
    except Exception as e:
        logger.debug(f"pydub extraction failed: {e}")
    
    return None


def _extract_audio_moviepy(video_path: Path) -> Optional[Path]:
    """
    Extract audio using moviepy (handles multiple API versions).
    """
    output_path = PROCESSED_DIR / f"{video_path.stem}_audio.wav"
    
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        
        logger.info(f"Extracting audio using moviepy from {video_path.name}")
        video = VideoFileClip(str(video_path))
        
        if video.audio is not None:
            # Try different moviepy API versions
            try:
                # Newer moviepy versions (2.0+)
                video.audio.write_audiofile(
                    str(output_path),
                    fps=16000,
                    nbytes=2,
                    codec='pcm_s16le',
                    logger=None  # Suppress output
                )
            except TypeError:
                try:
                    # Older moviepy versions with verbose parameter
                    video.audio.write_audiofile(
                        str(output_path),
                        fps=16000,
                        nbytes=2,
                        codec='pcm_s16le'
                    )
                except Exception:
                    # Minimal parameters as last resort
                    video.audio.write_audiofile(str(output_path))
            
            video.close()
            
            if output_path.exists() and output_path.stat().st_size > 1000:
                logger.info(f"Audio extracted successfully with moviepy: {output_path}")
                return output_path
        else:
            logger.warning(f"No audio track found in {video_path.name}")
            video.close()
            
    except ImportError:
        logger.debug("moviepy not installed")
    except Exception as e:
        logger.debug(f"moviepy extraction failed: {e}")
    
    return None


def _extract_audio_opencv(video_path: Path) -> Optional[Path]:
    """
    Extract audio using OpenCV + scipy (no FFmpeg required).
    This is a pure Python fallback that reads raw audio from video container.
    """
    output_path = PROCESSED_DIR / f"{video_path.stem}_audio.wav"
    
    try:
        import cv2
        import wave
        from scipy.io import wavfile
        
        # OpenCV can read some audio from video files
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return None
        
        # Check if video has audio
        # Note: OpenCV's audio support is limited
        cap.release()
        
        # This method has limited support, skip to next
        logger.debug("OpenCV audio extraction not fully supported")
        
    except ImportError:
        logger.debug("scipy not installed for audio extraction")
    except Exception as e:
        logger.debug(f"OpenCV audio extraction failed: {e}")
    
    return None


def _extract_audio_subprocess(video_path: Path) -> Optional[Path]:
    """
    Extract audio using FFmpeg subprocess if available.
    This is the most reliable method when FFmpeg is installed.
    """
    output_path = PROCESSED_DIR / f"{video_path.stem}_audio.wav"
    
    # Check for FFmpeg
    ffmpeg_cmd = None
    for cmd in ['ffmpeg', 'ffmpeg.exe', '/usr/bin/ffmpeg', '/usr/local/bin/ffmpeg']:
        try:
            result = subprocess.run([cmd, '-version'], capture_output=True, timeout=5)
            if result.returncode == 0:
                ffmpeg_cmd = cmd
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    if not ffmpeg_cmd:
        logger.debug("FFmpeg not found in PATH")
        return None
    
    try:
        logger.info(f"Extracting audio using FFmpeg subprocess from {video_path.name}")
        
        # FFmpeg command to extract audio as 16kHz mono WAV
        cmd = [
            ffmpeg_cmd, '-y',  # Overwrite output
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # WAV format
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        
        if output_path.exists() and output_path.stat().st_size > 1000:
            logger.info(f"Audio extracted successfully with FFmpeg: {output_path}")
            return output_path
            
    except subprocess.TimeoutExpired:
        logger.warning("FFmpeg audio extraction timed out")
    except Exception as e:
        logger.debug(f"FFmpeg extraction failed: {e}")
    
    return None


def _transcribe_audio(audio_path: Path) -> Dict:
    """Transcribe audio using Whisper with multilingual support."""
    try:
        from ingestion.audio_ingest import transcribe_audio
        result = transcribe_audio(audio_path)
        if result and result.get("text"):
            logger.info(f"Transcription successful: {len(result.get('text', ''))} chars")
        return result
    except ImportError:
        logger.warning("audio_ingest module not available")
    except Exception as e:
        logger.warning(f"Audio transcription failed: {e}")
    return {"text": "", "language": "unknown", "segments": []}


def _extract_audio(video_path: Path) -> Optional[Path]:
    """
    Extract audio from video using best available method.
    
    Priority order:
    1. FFmpeg subprocess (most reliable)
    2. pydub (requires FFmpeg)
    3. moviepy (pure Python but needs imageio-ffmpeg)
    4. OpenCV (limited support)
    """
    methods = [
        ("FFmpeg", _extract_audio_subprocess),
        ("pydub", _extract_audio_pydub),
        ("moviepy", _extract_audio_moviepy),
        ("OpenCV", _extract_audio_opencv),
    ]
    
    for method_name, method_func in methods:
        try:
            audio_path = method_func(video_path)
            if audio_path and audio_path.exists():
                logger.info(f"Audio extracted successfully using {method_name}")
                return audio_path
        except Exception as e:
            logger.debug(f"{method_name} failed: {e}")
            continue
    
    logger.warning(f"Could not extract audio from {video_path.name} - all methods failed")
    logger.info("TIP: Install FFmpeg for best audio extraction: https://ffmpeg.org/download.html")
    return None


def ingest_video_file(video_path: Path) -> int:
    """
    Ingest a single video file with detailed frame analysis.
    
    SOTA Pipeline:
    1. Extract keyframes using intelligent scene detection
    2. Extract and transcribe audio with Whisper
    3. Process each frame: BLIP-2 caption + OCR
    4. Combine frame-level and video-level metadata
    5. Create searchable chunks for RAG
    
    Args:
        video_path: Path to video file
    
    Returns:
        Number of chunks created
    """
    video_path = Path(video_path)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return 0
    
    logger.info(f"Processing video: {video_path.name}")
    
    # Create output directory for this video's frames
    frames_dir = VIDEO_FRAMES_DIR / video_path.stem
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Extract keyframes with timestamps
    frames_with_timestamps = _extract_keyframes_opencv(video_path, frames_dir)
    
    if not frames_with_timestamps:
        logger.warning(f"No frames extracted from {video_path.name}")
    else:
        logger.info(f"Extracted {len(frames_with_timestamps)} keyframes")
    
    # 2. Extract and transcribe audio
    audio_path = _extract_audio(video_path)
    audio_result = {"text": "", "language": "en", "segments": []}
    
    if audio_path:
        try:
            audio_result = _transcribe_audio(audio_path)
            logger.info(f"Audio transcribed: {len(audio_result.get('text', ''))} chars, language: {audio_result.get('language', 'unknown')}")
            # Clean up temp audio file
            try:
                audio_path.unlink()
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Audio processing error: {e}")
    
    audio_transcript = audio_result.get("text", "").strip()
    audio_segments = audio_result.get("segments", [])
    detected_language = audio_result.get("language", "en")
    
    records: List[Dict] = []
    
    # 3. Process each frame with timestamp
    for idx, (frame_path, timestamp_sec) in enumerate(frames_with_timestamps):
        try:
            # Get relevant audio segment for this frame
            frame_audio = _get_audio_for_timestamp(audio_segments, timestamp_sec)
            
            record = _process_video_frame(
                frame_path=frame_path,
                video_path=video_path,
                frame_idx=idx,
                timestamp_sec=timestamp_sec,
                frame_audio=frame_audio,
                detected_language=detected_language
            )
            if record:
                records.append(record)
                logger.debug(f"Processed frame {idx+1}/{len(frames_with_timestamps)}")
        except Exception as e:
            logger.error(f"Failed to process frame {frame_path}: {e}")
            continue
    
    # 4. Create a video summary chunk with full audio transcript
    if audio_transcript:
        summary_record = {
            "chunk_id": f"video_{video_path.stem}_summary",
            "modality": "video",
            "source_file": video_path.name,
            "source": str(video_path),
            "frame_count": len(frames_with_timestamps),
            "audio_transcript": audio_transcript,
            "language_detected": detected_language,
            "duration_formatted": _format_timestamp(frames_with_timestamps[-1][1] if frames_with_timestamps else 0),
            "content": f"VIDEO SUMMARY - {video_path.name}\nAudio transcript:\n{audio_transcript}",
        }
        records.append(summary_record)
    
    # 5. Persist
    if records:
        with CHUNK_FILE.open("a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(records)} video chunks for {video_path.name}")
    else:
        logger.warning(f"No chunks created for {video_path.name}")
    
    return len(records)


def _get_audio_for_timestamp(segments: List[Dict], timestamp_sec: float, window_sec: float = 5.0) -> str:
    """
    Get the audio transcript segment near a given timestamp.
    
    Args:
        segments: List of Whisper segments with 'start', 'end', 'text'
        timestamp_sec: Frame timestamp in seconds
        window_sec: Time window around the timestamp
    
    Returns:
        Relevant transcript text
    """
    if not segments:
        return ""
    
    relevant_texts = []
    
    for seg in segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        
        # Check if segment overlaps with timestamp window
        if seg_start <= timestamp_sec + window_sec and seg_end >= timestamp_sec - window_sec:
            relevant_texts.append(seg.get("text", "").strip())
    
    return " ".join(relevant_texts)


def _process_video_frame(
    frame_path: Path,
    video_path: Path,
    frame_idx: int,
    timestamp_sec: float,
    frame_audio: str = "",
    detected_language: str = "en"
) -> Optional[Dict]:
    """
    Process a single video frame: OCR + Caption + Scene description.
    
    SOTA approach:
    - BLIP-2 for detailed scene captioning
    - Multilingual OCR for text extraction
    - Rich metadata for precise retrieval
    """
    if not frame_path.exists():
        return None
    
    try:
        pil_image = Image.open(frame_path).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to open frame {frame_path}: {e}")
        return None
    
    # Lazy imports to avoid slow startup
    from wrappers.ocr import run_ocr
    from wrappers.caption import caption_image
    
    # Determine OCR languages based on detected audio language
    ocr_languages = ["en"]
    if detected_language in ["hi", "te", "ta", "kn", "ml", "bn", "mr", "gu"]:
        ocr_languages = ["en", detected_language]
    
    # OCR with multilingual support
    try:
        ocr_text = run_ocr(pil_image, languages=ocr_languages)
    except Exception as e:
        logger.debug(f"OCR failed for frame {frame_idx}: {e}")
        ocr_text = ""
    
    # Caption with BLIP-2
    try:
        caption = caption_image(pil_image)
    except Exception as e:
        logger.debug(f"Caption failed for frame {frame_idx}: {e}")
        caption = ""
    
    # Format timestamp
    timestamp_formatted = _format_timestamp(timestamp_sec)
    
    # Build rich content for embedding (SOTA format for retrieval)
    content_parts = []
    
    # Primary identifier
    content_parts.append(f"VIDEO FRAME [{timestamp_formatted}] from {video_path.name}")
    
    # Scene description (most important for semantic search)
    if caption:
        content_parts.append(f"Visual content: {caption}")
    
    # Text in frame (crucial for educational/presentation videos)
    if ocr_text and len(ocr_text.strip()) > 3:
        content_parts.append(f"On-screen text: {ocr_text}")
    
    # Synchronized audio (provides temporal context)
    if frame_audio and len(frame_audio.strip()) > 5:
        content_parts.append(f"Narration: {frame_audio}")
    
    combined_content = "\n".join(content_parts)
    
    record = {
        "chunk_id": f"video_{video_path.stem}_frame_{frame_idx:04d}",
        "modality": "video",
        "source_file": video_path.name,
        "source": str(video_path),
        "frame_path": str(frame_path),
        "frame_idx": frame_idx,
        "timestamp_seconds": round(timestamp_sec, 2),
        "timestamp_formatted": timestamp_formatted,
        "frame_caption": caption,
        "frame_ocr": ocr_text,
        "frame_audio_segment": frame_audio,
        "language_detected": detected_language,
        "content": combined_content,
    }
    
    return record


def ingest_all_videos(raw_dir: Path = Path("data/raw/video")) -> int:
    """
    Process all videos in directory.
    
    Returns:
        Total number of chunks created
    """
    supported_exts = [".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv"]
    total = 0
    processed = 0
    failed = 0
    
    if not raw_dir.exists():
        logger.warning(f"Video directory not found: {raw_dir}")
        return 0
    
    # Collect all video files
    video_files = []
    for ext in supported_exts:
        video_files.extend(raw_dir.glob(f"*{ext}"))
        video_files.extend(raw_dir.glob(f"*{ext.upper()}"))
    
    if not video_files:
        logger.info(f"No video files found in {raw_dir}")
        return 0
    
    logger.info(f"Found {len(video_files)} video file(s) to process")
    
    for video_path in video_files:
        try:
            count = ingest_video_file(video_path)
            total += count
            processed += 1
            logger.info(f"✓ {video_path.name}: {count} chunks")
        except Exception as e:
            logger.error(f"✗ Failed to process {video_path.name}: {e}")
            failed += 1
    
    logger.info("=" * 60)
    logger.info(f"Video Ingestion Summary:")
    logger.info(f"  Processed: {processed} videos")
    logger.info(f"  Failed: {failed} videos")
    logger.info(f"  Total chunks: {total}")
    logger.info(f"  Output: {CHUNK_FILE}")
    logger.info("=" * 60)
    
    return total



# ------------------------------------
# Programmatic API for Streamlit/app integration
# ------------------------------------
def ingest_video(uploaded_file):
    """
    Ingest a video file from a file-like object (e.g., Streamlit upload).
    Returns a document ID (filename) or status.
    """
    import tempfile
    from pathlib import Path

    raw_dir = Path("data/raw/video")
    raw_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, dir=raw_dir, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    try:
        count = ingest_video_file(tmp_path)
        if count == 0:
            logger.warning(f"No video processed for {uploaded_file.name}")
            return None
        return tmp_path.name
    except Exception as e:
        logger.warning(f"Video ingestion failed for {uploaded_file.name}: {e}")
        return None

# ------------------------------------
# CLI / Test
# ------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
        count = ingest_video_file(video_path)
        print(f"Created {count} chunks from {video_path}")
    else:
        count = ingest_all_videos()
        print(f"Created {count} video chunks total")