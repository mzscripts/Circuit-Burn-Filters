from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image


def load_image_from_bytes(data: bytes) -> Image.Image:
    with Image.open(BytesIO(data)) as image:
        return image.convert("RGB")


def ensure_pil_image(result) -> Image.Image:
    if isinstance(result, Image.Image):
        return result.convert("RGB")
    if isinstance(result, np.ndarray):
        array = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    raise TypeError(f"Unsupported filter result type: {type(result)!r}")


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()
