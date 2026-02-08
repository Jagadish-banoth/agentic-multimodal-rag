"""
wrappers/caption.py
--------------------
Production-grade image captioning using BLIP-2 (SOTA).

Accepts:
  - PIL.Image.Image
  - np.ndarray
  - str or Path (file path)

Returns:
  - Descriptive caption string
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ------------------------------------
# Device & Precision
# ------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# ------------------------------------
# Lazy-loaded model (avoid slow import)
# ------------------------------------
_processor = None
_model = None
MODEL_NAME = "Salesforce/blip2-flan-t5-xl"


def _get_model():
    """Lazy-load BLIP-2 model and processor."""
    global _processor, _model
    
    if _model is None:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        logger.info(f"Loading caption model: {MODEL_NAME}")
        _processor = Blip2Processor.from_pretrained(MODEL_NAME)
        _model = Blip2ForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,  # Correct parameter name
            low_cpu_mem_usage=True
        ).to(device)
        _model.eval()
        logger.info(f"Caption model loaded on {device}")
    
    return _processor, _model


def _load_image(image_input: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    """Normalize input to PIL.Image."""
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        return Image.open(path).convert("RGB")
    
    if isinstance(image_input, np.ndarray):
        return Image.fromarray(image_input).convert("RGB")
    
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    
    raise TypeError(f"Unsupported image input type: {type(image_input)}")


# ------------------------------------
# Caption Function
# ------------------------------------
@torch.inference_mode()
def caption_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    max_tokens: int = 64,
    num_beams: int = 5
) -> str:
    """
    Generate a descriptive caption for an image.
    
    Args:
        image_input: PIL.Image, numpy array, or file path (str/Path)
        max_tokens: Maximum caption length
        num_beams: Beam search width
    
    Returns:
        Caption string describing the image content
    """
    try:
        image = _load_image(image_input)
    except Exception as e:
        logger.error(f"Failed to load image for captioning: {e}")
        return ""
    
    processor, model = _get_model()

    # Use a descriptive prompt to improve caption quality
    prompt = "Question: Describe this image in detail. Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    # Generate caption with better parameters
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_beams=num_beams,
        do_sample=False,
        repetition_penalty=1.5,  # Penalize repetition
        length_penalty=1.0,
        early_stopping=True
    )

    caption = processor.tokenizer.decode(output[0], skip_special_tokens=True)
    return caption.strip()


# ------------------------------------
# CLI / Test
# ------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = r"D:\agentic-multimodal-rag\data\processed\images\Attention_all_you_need_p3_img1_98e23cba.png"
    
    print(f"Captioning: {img_path}")
    caption = caption_image(img_path)
    print("=" * 40)
    print(f"Caption: {caption}")
    print("=" * 40)