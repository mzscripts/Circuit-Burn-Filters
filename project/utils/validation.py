from __future__ import annotations

import warnings
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.Image import DecompressionBombError
from PIL import Image as PilImage
from werkzeug.datastructures import FileStorage


class ValidationError(ValueError):
    pass


def allowed_file(filename: str, allowed_extensions: set[str]) -> bool:
    return "." in filename and Path(filename).suffix.lower().lstrip(".") in allowed_extensions


def get_upload_size(file_storage: FileStorage) -> int:
    if file_storage.content_length and file_storage.content_length > 0:
        return file_storage.content_length

    stream = file_storage.stream
    current_position = stream.tell()
    stream.seek(0, 2)
    size = stream.tell()
    stream.seek(current_position)
    return size


def validate_uploaded_file(
    file_storage: FileStorage,
    allowed_extensions: set[str],
    *,
    max_file_size_bytes: int,
) -> int:
    if not file_storage or not file_storage.filename:
        raise ValidationError("Please choose an image to upload.")

    if not allowed_file(file_storage.filename, allowed_extensions):
        raise ValidationError("Only JPG, JPEG, PNG, and WEBP files are supported.")

    file_size = get_upload_size(file_storage)
    if file_size <= 0:
        raise ValidationError("The uploaded file is empty.")
    if file_size > max_file_size_bytes:
        raise ValidationError("Images must be 15MB or smaller.")

    return file_size


def prepare_uploaded_image(
    file_storage: FileStorage,
    allowed_extensions: set[str],
    *,
    max_file_size_bytes: int,
    max_pixels: int,
    max_dimension: int,
) -> tuple[Image.Image, int]:
    file_size = validate_uploaded_file(
        file_storage,
        allowed_extensions,
        max_file_size_bytes=max_file_size_bytes,
    )
    previous_limit = PilImage.MAX_IMAGE_PIXELS

    try:
        file_storage.stream.seek(0)
        PilImage.MAX_IMAGE_PIXELS = max_pixels
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(file_storage.stream) as image:
                image.load()
                normalized = ImageOps.exif_transpose(image)
                if normalized.width * normalized.height > max_pixels:
                    raise ValidationError("That image is too large to process safely.")
                if max(normalized.size) > max_dimension:
                    normalized.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                if normalized.mode not in {"RGB", "RGBA"}:
                    if "A" in normalized.getbands():
                        normalized = normalized.convert("RGBA")
                    else:
                        normalized = normalized.convert("RGB")
                prepared = normalized.copy()
        file_storage.stream.seek(0)
        return prepared, file_size
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValidationError("That file is not a valid image or appears to be corrupted.") from exc
    except (Image.DecompressionBombWarning, DecompressionBombError) as exc:
        raise ValidationError("That image is too large to process safely.") from exc
    finally:
        PilImage.MAX_IMAGE_PIXELS = previous_limit
        try:
            file_storage.stream.seek(0)
        except OSError:
            pass
