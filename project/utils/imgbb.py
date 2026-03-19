from __future__ import annotations

import base64
from io import BytesIO

import requests
from PIL import Image

from utils.image_io import load_image_from_bytes
from utils.validation import ValidationError


def ensure_imgbb_key(api_key: str) -> None:
    if not api_key:
        raise ValidationError("Set the IMGBB_API_KEY environment variable before uploading images.")


def upload_image_bytes(*, image_bytes: bytes, api_key: str, upload_url: str, name: str) -> dict:
    ensure_imgbb_key(api_key)
    payload = {
        "key": api_key,
        "name": name,
        "image": base64.b64encode(image_bytes).decode("ascii"),
    }
    response = requests.post(upload_url, data=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    if not data.get("success"):
        raise ValidationError("ImgBB rejected the image upload.")
    image_data = data["data"]
    return {
        "url": image_data["url"],
        "display_url": image_data.get("display_url", image_data["url"]),
        "delete_url": image_data.get("delete_url"),
        "size": image_data.get("size"),
        "width": image_data.get("width"),
        "height": image_data.get("height"),
    }


def fetch_remote_image_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def fetch_remote_image(url: str) -> Image.Image:
    return load_image_from_bytes(fetch_remote_image_bytes(url))
