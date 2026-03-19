from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem.lower().strip().replace(" ", "_")
    cleaned = "".join(ch for ch in stem if ch.isalnum() or ch in {"_", "-"})
    return cleaned or "image"


def unique_upload_name(filename: str) -> str:
    ext = Path(filename).suffix.lower() or ".png"
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    return f"{safe_stem(filename)}__{timestamp}_{uuid4().hex[:10]}{ext}"


def output_name(original_filename: str, filter_id: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    return f"{safe_stem(original_filename)}__{filter_id}__{timestamp}.png"
