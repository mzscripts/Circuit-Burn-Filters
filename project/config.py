from pathlib import Path
import os


def _load_local_env() -> None:
    candidate_paths = (
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent / ".env",
    )
    for env_path in candidate_paths:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        break


_load_local_env()


class Config:
    BASE_DIR = Path(__file__).resolve().parent
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
    MAX_CONTENT_LENGTH = 15 * 1024 * 1024
    MAX_IMAGE_PIXELS = 20_000_000
    MAX_IMAGE_DIMENSION = 1600
    OUTPUT_JPEG_QUALITY = 89
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
    SESSION_PERMANENT = False
    SESSION_COOKIE_SAMESITE = "Lax"
    IMGBB_API_KEY = os.environ.get("IMGBB_API_KEY", "")
    IMGBB_UPLOAD_URL = "https://api.imgbb.com/1/upload"
