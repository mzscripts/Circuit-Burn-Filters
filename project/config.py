from pathlib import Path
import os


class Config:
    BASE_DIR = Path(__file__).resolve().parent
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
    MAX_CONTENT_LENGTH = 15 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
    SESSION_PERMANENT = False
    SESSION_COOKIE_SAMESITE = "Lax"
    IMGBB_API_KEY = os.environ.get("IMGBB_API_KEY", "")
    IMGBB_UPLOAD_URL = "https://api.imgbb.com/1/upload"
