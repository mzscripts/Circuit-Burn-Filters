from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError
from werkzeug.datastructures import FileStorage


class ValidationError(ValueError):
    pass


def allowed_file(filename: str, allowed_extensions: set[str]) -> bool:
    return "." in filename and Path(filename).suffix.lower().lstrip(".") in allowed_extensions


def validate_uploaded_file(file_storage: FileStorage, allowed_extensions: set[str]) -> None:
    if not file_storage or not file_storage.filename:
        raise ValidationError("Please choose an image to upload.")

    if not allowed_file(file_storage.filename, allowed_extensions):
        raise ValidationError("Only JPG, JPEG, PNG, and WEBP files are supported.")

    try:
        file_storage.stream.seek(0)
        with Image.open(file_storage.stream) as image:
            image.verify()
        file_storage.stream.seek(0)
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValidationError("That file is not a valid image or appears to be corrupted.") from exc
