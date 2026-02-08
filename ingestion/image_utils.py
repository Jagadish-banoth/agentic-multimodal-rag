from pathlib import Path
from PIL import Image
import uuid

IMAGE_DIR = Path("data/processed/images")



def save_pil_image(image: Image.Image, prefix: str) -> Path:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    """
    Save a PIL image to data/processed/images and return its path.
    """
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    out_path = IMAGE_DIR / filename
    image.save(out_path, format="PNG")
    return out_path