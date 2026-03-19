from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps

def neon_bloom(image: Image.Image) -> Image.Image:
    edges = ImageOps.autocontrast(ImageOps.grayscale(image.filter(ImageFilter.FIND_EDGES)))
    glow = ImageOps.colorize(edges, black="#03050c", white="#4dfcff").convert("RGB")
    base = np.asarray(image, dtype=np.float32)
    overlay = np.asarray(glow.filter(ImageFilter.GaussianBlur(radius=4)), dtype=np.float32)
    mixed = np.clip(base * 0.8 + overlay * 0.6, 0, 255).astype(np.uint8)
    return Image.fromarray(mixed, "RGB")
