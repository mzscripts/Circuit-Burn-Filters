from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps


def load_image_from_bytes(data: bytes) -> Image.Image:
    with BytesIO(data) as buffer:
        return load_image_from_stream(buffer)


def load_image_from_stream(stream) -> Image.Image:
    with Image.open(stream) as image:
        image.load()
        normalized = ImageOps.exif_transpose(image)
        if normalized.mode not in {"RGB", "RGBA"}:
            if "A" in normalized.getbands():
                return normalized.convert("RGBA")
            return normalized.convert("RGB")
        return normalized.copy()


def ensure_pil_image(result) -> Image.Image:
    if isinstance(result, Image.Image):
        if result.mode in {"RGB", "RGBA"}:
            return result
        if "A" in result.getbands():
            return result.convert("RGBA")
        return result.convert("RGB")
    if isinstance(result, np.ndarray):
        array = result if result.dtype == np.uint8 else np.clip(result, 0, 255).astype(np.uint8)
        image = Image.fromarray(array)
        if image.mode in {"RGB", "RGBA"}:
            return image
        if "A" in image.getbands():
            return image.convert("RGBA")
        return image.convert("RGB")
    raise TypeError(f"Unsupported filter result type: {type(result)!r}")


def image_has_transparency(image: Image.Image) -> bool:
    if "A" in image.getbands():
        alpha = image.getchannel("A")
        return alpha.getextrema()[0] < 255
    return "transparency" in image.info


def image_to_upload_bytes(image: Image.Image, *, jpeg_quality: int = 89) -> tuple[bytes, str, str]:
    buffer = BytesIO()
    if image_has_transparency(image):
        output_image = image if image.mode == "RGBA" else image.convert("RGBA")
        output_image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue(), "image/png", ".png"

    output_image = image if image.mode == "RGB" else image.convert("RGB")
    output_image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    return buffer.getvalue(), "image/jpeg", ".jpg"
