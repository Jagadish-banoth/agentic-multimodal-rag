"""
wrappers/ocr.py
---------------
SOTA Multilingual OCR wrapper with multiple backends:

Priority order:
1. EasyOCR (multilingual - supports 80+ languages including Indian languages)
2. PaddleOCR (high accuracy for printed text)
3. Tesseract (fallback)

Supported languages:
- English, Hindi, Telugu, Tamil, Kannada, Malayalam, Bengali, Marathi, Gujarati
- Chinese, Japanese, Korean, Arabic, and 70+ more

Accepts:
  - PIL.Image.Image
  - np.ndarray
  - str or Path (file path)

Returns:
  - Extracted text string with confidence scores
"""

import logging
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ------------------------------------
# Language mappings
# ------------------------------------
LANGUAGE_MAPPINGS = {
    # EasyOCR language codes
    "easyocr": {
        "english": "en", "hindi": "hi", "telugu": "te", "tamil": "ta",
        "kannada": "kn", "malayalam": "ml", "bengali": "bn", "marathi": "mr",
        "gujarati": "gu", "punjabi": "pa", "urdu": "ur", "nepali": "ne",
        "chinese_simplified": "ch_sim", "chinese_traditional": "ch_tra",
        "japanese": "ja", "korean": "ko", "arabic": "ar", "german": "de",
        "french": "fr", "spanish": "es", "portuguese": "pt", "russian": "ru",
    },
    # PaddleOCR language codes
    "paddleocr": {
        "english": "en", "chinese": "ch", "hindi": "hi", "telugu": "te",
        "tamil": "ta", "kannada": "ka", "arabic": "ar", "german": "german",
        "french": "french", "japanese": "japan", "korean": "korean",
    }
}

# Default languages for multilingual OCR
# Note: EasyOCR has language compatibility constraints
# Tamil (ta) only works with English (en)
# Devanagari-based: en, hi, mr, ne can work together
# Dravidian: te, kn, ml can work together (but NOT with ta)
DEFAULT_LANGUAGES = ["en"]  # Safe default - English only

# EasyOCR compatible language groups (cannot mix groups)
EASYOCR_LANG_GROUPS = {
    "latin": ["en", "de", "fr", "es", "pt", "it"],
    "devanagari": ["en", "hi", "mr", "ne"],
    "dravidian_te_kn": ["en", "te", "kn"],
    "tamil_only": ["en", "ta"],  # Tamil ONLY works with English
    "malayalam_only": ["en", "ml"],
    "cjk": ["en", "ch_sim", "ja", "ko"],
    "arabic": ["en", "ar"],
}

# ------------------------------------
# Lazy-load OCR engines
# ------------------------------------
_easyocr_reader = None
_paddle_ocr = None
_tesseract_available = None


def _get_compatible_languages(requested_langs: List[str]) -> List[str]:
    """Get EasyOCR-compatible language subset.
    
    EasyOCR has strict language compatibility rules:
    - Tamil (ta) ONLY works with English
    - Other Dravidian languages have their own constraints
    - Returns the best compatible subset
    """
    if not requested_langs:
        return ["en"]
    
    # Always include English as base
    result = ["en"] if "en" not in requested_langs else []
    
    # Check for Tamil - it's exclusive
    if "ta" in requested_langs:
        return ["en", "ta"]
    
    # Check for Malayalam - it's exclusive  
    if "ml" in requested_langs:
        return ["en", "ml"]
    
    # Telugu and Kannada can work together
    dravidian = [l for l in requested_langs if l in ["te", "kn"]]
    if dravidian:
        return ["en"] + dravidian
    
    # Devanagari languages can work together
    devanagari = [l for l in requested_langs if l in ["hi", "mr", "ne"]]
    if devanagari:
        return ["en"] + devanagari
    
    # Default to English only for safety
    return ["en"]


def _get_easyocr_reader(languages: List[str] = None):
    """Lazy-load EasyOCR reader with multilingual support."""
    global _easyocr_reader
    
    if languages is None:
        languages = DEFAULT_LANGUAGES
    
    # Normalize language codes
    lang_codes = []
    for lang in languages:
        lang_lower = lang.lower()
        if lang_lower in LANGUAGE_MAPPINGS["easyocr"]:
            lang_codes.append(LANGUAGE_MAPPINGS["easyocr"][lang_lower])
        elif len(lang) <= 3:  # Already a code
            lang_codes.append(lang)
    
    if not lang_codes:
        lang_codes = ["en"]
    
    # Get compatible language subset to avoid EasyOCR errors
    lang_codes = _get_compatible_languages(lang_codes)
    
    if _easyocr_reader is None:
        try:
            import easyocr
            logger.info(f"Loading EasyOCR with compatible languages: {lang_codes}")
            _easyocr_reader = easyocr.Reader(
                lang_codes,
                gpu=True,  # Use GPU if available
                model_storage_directory=None,  # Use default
                download_enabled=True
            )
            logger.info("EasyOCR initialized successfully")
        except ImportError:
            logger.warning("EasyOCR not installed. Install with: pip install easyocr")
            _easyocr_reader = False
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {e}")
            _easyocr_reader = False
    
    return _easyocr_reader if _easyocr_reader else None


def _get_paddle_ocr(lang: str = "en"):
    """Lazy-load PaddleOCR with compatibility handling for all versions."""
    global _paddle_ocr
    
    if _paddle_ocr is None:
        try:
            from paddleocr import PaddleOCR
            import paddle
            paddle_version = paddle.__version__
            logger.info(f"PaddlePaddle version: {paddle_version}")
            
            # PaddleOCR API changed significantly across versions:
            # - PaddlePaddle 2.x: use_gpu=True/False, show_log=False
            # - PaddlePaddle 3.x: removed use_gpu, device auto-detected
            # We try multiple initialization strategies
            
            init_configs = [
                # Config 1: Minimal (PaddlePaddle 3.x compatible)
                {"use_angle_cls": True, "lang": lang},
                # Config 2: With show_log disabled (some versions)
                {"use_angle_cls": True, "lang": lang, "show_log": False},
                # Config 3: Legacy with use_gpu (PaddlePaddle 2.x)
                {"use_angle_cls": True, "lang": lang, "use_gpu": False, "show_log": False},
            ]
            
            for i, config in enumerate(init_configs):
                try:
                    _paddle_ocr = PaddleOCR(**config)
                    logger.info(f"PaddleOCR initialized successfully (config {i+1})")
                    break
                except TypeError as te:
                    logger.debug(f"PaddleOCR config {i+1} failed: {te}")
                    continue
            else:
                logger.warning("All PaddleOCR configurations failed")
                _paddle_ocr = False
                
        except ImportError:
            logger.warning("PaddleOCR not installed. Install with: pip install paddleocr paddlepaddle")
            _paddle_ocr = False
        except Exception as e:
            logger.warning(f"PaddleOCR initialization failed: {e}")
            _paddle_ocr = False
    
    return _paddle_ocr if _paddle_ocr else None


def _check_tesseract() -> bool:
    """Check if Tesseract is available."""
    global _tesseract_available
    
    if _tesseract_available is not None:
        return _tesseract_available
    
    try:
        import pytesseract
        import shutil
        
        if shutil.which("tesseract"):
            _tesseract_available = True
        else:
            # Try common Windows paths
            common_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"D:\tools\tesseract\tesseract.exe",
            ]
            for p in common_paths:
                if Path(p).exists():
                    pytesseract.pytesseract.tesseract_cmd = p
                    _tesseract_available = True
                    break
            else:
                _tesseract_available = False
        
        return _tesseract_available
    except ImportError:
        _tesseract_available = False
        return False


def _load_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    return_numpy: bool = False,
) -> Union[Image.Image, Tuple[Image.Image, np.ndarray]]:
    """Normalize image input.

    Returns a PIL.Image by default to keep call sites simple; set
    return_numpy=True to also receive a normalized numpy array.
    """
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        pil_img = Image.open(path).convert("RGB")
        np_img = np.array(pil_img)
        return (pil_img, np_img) if return_numpy else pil_img
    
    if isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input).convert("RGB")
        np_img = np.array(pil_img)
        return (pil_img, np_img) if return_numpy else pil_img
    
    if isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
        np_img = np.array(pil_img)
        return (pil_img, np_img) if return_numpy else pil_img
    
    raise TypeError(f"Unsupported image input type: {type(image_input)}")


# ------------------------------------
# OCR Backends
# ------------------------------------

def _ocr_with_easyocr(np_image: np.ndarray, languages: List[str] = None) -> Tuple[str, List[Dict]]:
    """
    Run EasyOCR on image.
    
    Returns:
        Tuple of (text, list of detections with bboxes and confidence)
    """
    reader = _get_easyocr_reader(languages)
    if reader is None:
        return "", []
    
    try:
        results = reader.readtext(np_image)
        
        detections = []
        text_parts = []
        
        for bbox, text, confidence in results:
            if confidence >= 0.3:  # Minimum confidence threshold
                text_parts.append(text)
                detections.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": bbox  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                })
        
        return " ".join(text_parts), detections
    except Exception as e:
        logger.warning(f"EasyOCR failed: {e}")
        return "", []


def _ocr_with_paddle(np_image: np.ndarray, lang: str = "en") -> Tuple[str, List[Dict]]:
    """
    Run PaddleOCR on image.
    
    Returns:
        Tuple of (text, list of detections)
    """
    ocr = _get_paddle_ocr(lang)
    if ocr is None:
        return "", []
    
    try:
        # Call without cls parameter (deprecated in newer versions)
        result = ocr.ocr(np_image)
        
        if not result or not result[0]:
            logger.warning("PaddleOCR returned empty result")
            return "", []
        
        detections = []
        text_parts = []
        
        for line in result[0]:
            bbox, (text, confidence) = line
            if confidence >= 0.3:
                text_parts.append(text)
                detections.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": bbox
                })
        
        return " ".join(text_parts), detections
    except Exception as e:
        logger.warning(f"PaddleOCR failed: {e}")
        return "", []


def _ocr_with_tesseract(pil_image: Image.Image, lang: str = "eng") -> Tuple[str, List[Dict]]:
    """
    Run Tesseract OCR on image.
    
    Returns:
        Tuple of (text, list of detections)
    """
    if not _check_tesseract():
        return "", []
    
    try:
        import pytesseract
        
        # Get detailed data with confidence
        data = pytesseract.image_to_data(pil_image, lang=lang, output_type=pytesseract.Output.DICT)
        
        detections = []
        text_parts = []
        
        for i in range(len(data['text'])):
            conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
            text = data['text'][i].strip()
            
            if conf >= 30 and text:
                text_parts.append(text)
                detections.append({
                    "text": text,
                    "confidence": conf / 100.0,
                    "bbox": [
                        data['left'][i], data['top'][i],
                        data['left'][i] + data['width'][i],
                        data['top'][i] + data['height'][i]
                    ]
                })
        
        return " ".join(text_parts), detections
    except Exception as e:
        logger.warning(f"Tesseract OCR failed: {e}")
        return "", []


# ------------------------------------
# Main OCR Function
# ------------------------------------

def run_ocr(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    languages: List[str] = None,
    backend: str = "auto",
    return_details: bool = False
) -> Union[str, Tuple[str, List[Dict]]]:
    """
    Run OCR on an image using the best available backend.
    
    Args:
        image_input: PIL.Image, numpy array, or file path
        languages: List of language codes (e.g., ["en", "hi", "te"])
                  If None, uses DEFAULT_LANGUAGES
        backend: "auto", "easyocr", "paddleocr", or "tesseract"
        return_details: If True, return (text, detections) with bboxes
    
    Returns:
        If return_details=False: extracted text string
        If return_details=True: tuple of (text, list of detection dicts)
    """
    if languages is None:
        languages = DEFAULT_LANGUAGES
    
    try:
        pil_image, np_image = _load_image(image_input, return_numpy=True)
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return ("", []) if return_details else ""
    
    text = ""
    detections = []
    
    # Try backends in order
    # Priority: EasyOCR (best multilingual) > Tesseract (reliable) > PaddleOCR (has PIR issues)
    if backend == "auto":
        backends_to_try = ["easyocr", "tesseract", "paddleocr"]
    else:
        backends_to_try = [backend]
    
    for ocr_backend in backends_to_try:
        try:
            if ocr_backend == "easyocr":
                text, detections = _ocr_with_easyocr(np_image, languages)
            elif ocr_backend == "paddleocr":
                text, detections = _ocr_with_paddle(np_image, languages[0] if languages else "en")
            elif ocr_backend == "tesseract":
                # Map to Tesseract language codes
                tess_lang = "+".join(["eng" if l == "en" else l for l in languages[:3]])
                text, detections = _ocr_with_tesseract(pil_image, tess_lang)
            
            if text.strip():
                logger.debug(f"OCR successful with {ocr_backend}: {len(text)} chars")
                break
        except Exception as e:
            logger.warning(f"OCR backend {ocr_backend} failed: {e}")
            continue
    
    if not text.strip():
        logger.warning("OCR returned empty result")
    
    return (text.strip(), detections) if return_details else text.strip()


def run_ocr_with_regions(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    languages: List[str] = None
) -> Dict[str, Any]:
    """
    Run OCR and return structured output with regions.
    
    Useful for detailed document analysis, table detection, etc.
    
    Returns:
        Dict with:
          - text: full extracted text
          - regions: list of text regions with coordinates
          - language_detected: primary language detected
          - confidence_avg: average confidence score
    """
    text, detections = run_ocr(image_input, languages, return_details=True)
    
    if not detections:
        return {
            "text": text,
            "regions": [],
            "language_detected": "unknown",
            "confidence_avg": 0.0
        }
    
    avg_conf = sum(d["confidence"] for d in detections) / len(detections)
    
    return {
        "text": text,
        "regions": detections,
        "language_detected": "multilingual",
        "confidence_avg": round(avg_conf, 3)
    }


# ------------------------------------
# Specialized OCR Functions
# ------------------------------------

def extract_table_text(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    languages: List[str] = None
) -> List[List[str]]:
    """
    Extract text from a table image with basic structure preservation.
    
    Returns:
        List of rows, each row is a list of cell texts
    """
    text, detections = run_ocr(image_input, languages, return_details=True)
    
    if not detections:
        return [[text]] if text else []
    
    # Sort by Y coordinate (top to bottom), then X (left to right)
    def get_center_y(d):
        bbox = d.get("bbox", [[0,0]])
        if isinstance(bbox[0], list):
            return (bbox[0][1] + bbox[2][1]) / 2
        return bbox[1]
    
    def get_center_x(d):
        bbox = d.get("bbox", [[0,0]])
        if isinstance(bbox[0], list):
            return (bbox[0][0] + bbox[2][0]) / 2
        return bbox[0]
    
    sorted_detections = sorted(detections, key=lambda d: (get_center_y(d), get_center_x(d)))
    
    # Group by rows (items with similar Y coordinates)
    rows = []
    current_row = []
    last_y = None
    y_threshold = 20  # Pixels
    
    for d in sorted_detections:
        y = get_center_y(d)
        
        if last_y is None or abs(y - last_y) <= y_threshold:
            current_row.append(d)
        else:
            if current_row:
                # Sort row by X coordinate
                current_row.sort(key=get_center_x)
                rows.append([item["text"] for item in current_row])
            current_row = [d]
        
        last_y = y
    
    if current_row:
        current_row.sort(key=get_center_x)
        rows.append([item["text"] for item in current_row])
    
    return rows


def extract_chart_text(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    languages: List[str] = None
) -> Dict[str, Any]:
    """
    Extract text elements from a chart/graph image.
    
    Returns:
        Dict with title, axis_labels, legend, data_labels
    """
    result = run_ocr_with_regions(image_input, languages)
    
    if not result["regions"]:
        return {
            "title": "",
            "axis_labels": [],
            "legend": [],
            "data_labels": [],
            "all_text": result["text"]
        }
    
    # Heuristic categorization based on position
    pil_img = _load_image(image_input)
    img_height, img_width = pil_img.size[1], pil_img.size[0]
    
    title_candidates = []
    axis_labels = []
    legend = []
    data_labels = []
    
    for region in result["regions"]:
        bbox = region.get("bbox", [[0,0]])
        text = region["text"]
        
        if isinstance(bbox[0], list):
            center_y = (bbox[0][1] + bbox[2][1]) / 2
            center_x = (bbox[0][0] + bbox[2][0]) / 2
        else:
            center_y = bbox[1]
            center_x = bbox[0]
        
        # Title: usually at top center
        if center_y < img_height * 0.15 and 0.2 * img_width < center_x < 0.8 * img_width:
            title_candidates.append(text)
        # Y-axis labels: left side
        elif center_x < img_width * 0.15:
            axis_labels.append(text)
        # X-axis labels: bottom
        elif center_y > img_height * 0.85:
            axis_labels.append(text)
        # Legend: often right side
        elif center_x > img_width * 0.8:
            legend.append(text)
        else:
            data_labels.append(text)
    
    return {
        "title": " ".join(title_candidates),
        "axis_labels": axis_labels,
        "legend": legend,
        "data_labels": data_labels,
        "all_text": result["text"]
    }


# ------------------------------------
# CLI / Test
# ------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        langs = sys.argv[2].split(",") if len(sys.argv) > 2 else None
        
        result = run_ocr_with_regions(image_path, langs)
        print(f"Text: {result['text']}")
        print(f"Regions: {len(result['regions'])}")
        print(f"Avg confidence: {result['confidence_avg']}")
    else:
        print("Usage: python -m wrappers.ocr <image_path> [languages]")
        print("Example: python -m wrappers.ocr image.png en,hi,te")
