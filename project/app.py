from __future__ import annotations

import io
import os
import time

from flask import Flask, jsonify, redirect, render_template, request, session, url_for, send_file
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from config import Config
from filters import get_filter, get_filters_grouped_by_category, load_registry
from utils.image_io import image_to_png_bytes
from utils.imgbb import fetch_remote_image, upload_image_bytes
from utils.naming import output_name
from utils.validation import ValidationError, validate_uploaded_file


app = Flask(__name__)
app.config.from_object(Config)
load_registry()


def is_api_request() -> bool:
    return request.path.startswith("/api/") or request.path in {"/apply-multiple", "/reset-session"}


def json_error(message: str, status_code: int = 400):
    response = jsonify({"success": False, "error": message})
    response.status_code = status_code
    return response


def current_upload() -> dict | None:
    return session.get("uploaded_image")


def current_results() -> list[dict]:
    return session.setdefault("generated_results", [])


def save_results(results: list[dict]) -> None:
    session["generated_results"] = results
    session.modified = True


def require_uploaded_image():
    upload = current_upload()
    if not upload:
        return None, redirect(url_for("index"))
    return upload, None


def get_result_by_filter(filter_id: str) -> dict | None:
    for result in reversed(current_results()):
        if result["filter_id"] == filter_id:
            return result
    return None


def upsert_result(result: dict) -> None:
    results = [item for item in current_results() if item["filter_id"] != result["filter_id"]]
    results.append(result)
    results.sort(key=lambda item: item["sequence_number"])
    save_results(results)


def format_file_size(size_bytes: int) -> str:
    size = float(size_bytes)
    units = ["B", "KB", "MB", "GB"]
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"


def format_generation_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


def safe_download_name(result: dict) -> str:
    extension = os.path.splitext(result.get("image_url", ""))[1] or ".png"
    base = result["filter_name"].lower().replace(" ", "_")
    return f"{base}{extension}"


def process_filter(filter_id: str) -> dict:
    metadata = get_filter(filter_id)
    if metadata is None:
        raise ValidationError("The selected filter could not be found.")

    upload = current_upload()
    if not upload:
        raise ValidationError("Upload an image before applying filters.")

    started = time.perf_counter()
    original_image = fetch_remote_image(upload["image_url"])
    output_image = metadata.function(original_image.copy())
    output_bytes = image_to_png_bytes(output_image)
    elapsed = time.perf_counter() - started
    uploaded_output = upload_image_bytes(
        image_bytes=output_bytes,
        api_key=app.config["IMGBB_API_KEY"],
        upload_url=app.config["IMGBB_UPLOAD_URL"],
        name=output_name(upload["original_name"], filter_id),
    )

    result = {
        "filter_id": metadata.id,
        "filter_name": metadata.name,
        "category": metadata.category,
        "sequence_number": metadata.sequence_number,
        "image_url": uploaded_output["url"],
        "file_size_bytes": len(output_bytes),
        "file_size_label": format_file_size(len(output_bytes)),
        "generation_time_seconds": round(elapsed, 3),
        "generation_time_label": format_generation_time(elapsed),
    }
    upsert_result(result)
    return result


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_error):
    message = "Images must be 15MB or smaller."
    if is_api_request():
        return json_error(message, 413)
    return render_template("index.html", error=message), 413


@app.errorhandler(ValidationError)
def handle_validation_error(error):
    if is_api_request():
        return json_error(str(error), 400)
    return render_template("index.html", error=str(error)), 400


@app.get("/")
def index():
    return render_template("index.html", error=None)


@app.post("/upload")
def upload():
    file = request.files.get("image")
    validate_uploaded_file(file, app.config["ALLOWED_EXTENSIONS"])

    original_name = secure_filename(file.filename)
    raw_bytes = file.read()
    uploaded_image = upload_image_bytes(
        image_bytes=raw_bytes,
        api_key=app.config["IMGBB_API_KEY"],
        upload_url=app.config["IMGBB_UPLOAD_URL"],
        name=original_name,
    )

    session["uploaded_image"] = {
        "original_name": original_name,
        "image_url": uploaded_image["url"],
        "display_url": uploaded_image["display_url"],
        "delete_url": uploaded_image.get("delete_url"),
    }
    session["generated_results"] = []
    session.modified = True
    return redirect(url_for("filters_page"))


@app.get("/filters")
def filters_page():
    upload, redirect_response = require_uploaded_image()
    if redirect_response:
        return redirect_response

    return render_template(
        "filters.html",
        uploaded_image=upload,
        grouped_filters=get_filters_grouped_by_category(),
        generated_results=current_results(),
        total_filters=len(load_registry()),
    )


@app.get("/process/<filter_id>")
def process_page(filter_id: str):
    upload, redirect_response = require_uploaded_image()
    if redirect_response:
        return redirect_response

    metadata = get_filter(filter_id)
    if metadata is None:
        return redirect(url_for("filters_page"))

    return render_template(
        "process.html",
        uploaded_image=upload,
        filter_data=metadata.to_public_dict(),
        existing_result=get_result_by_filter(filter_id),
    )


@app.post("/api/process/<filter_id>")
def process_api(filter_id: str):
    try:
        result = process_filter(filter_id)
        return jsonify({"success": True, **result})
    except ValidationError as exc:
        return json_error(str(exc), 400)
    except Exception as exc:
        return json_error(f"Failed to process '{filter_id}': {exc}", 500)


@app.get("/download/<filter_id>")
def download_result(filter_id: str):
    result = get_result_by_filter(filter_id)
    if result is None:
        return redirect(url_for("results_page"))

    image = fetch_remote_image(result["image_url"])
    payload = image_to_png_bytes(image)
    return send_file(
        io.BytesIO(payload),
        mimetype="image/png",
        as_attachment=True,
        download_name=safe_download_name(result),
    )


@app.post("/apply-multiple")
def apply_multiple():
    if not current_upload():
        return json_error("Upload an image before applying filters.", 400)

    payload = request.get_json(silent=True) or {}
    filter_ids = payload.get("filter_ids")
    if not isinstance(filter_ids, list) or not filter_ids:
        return json_error("Provide a non-empty filter_ids array.", 400)

    results = []
    for filter_id in filter_ids:
        try:
            result = process_filter(filter_id)
            results.append({"success": True, **result})
        except Exception as exc:
            results.append({
                "success": False,
                "filter_id": filter_id,
                "error": str(exc),
            })

    return jsonify({"success": True, "results": results})


@app.get("/results")
def results_page():
    upload, redirect_response = require_uploaded_image()
    if redirect_response:
        return redirect_response

    return render_template(
        "results.html",
        uploaded_image=upload,
        generated_results=current_results(),
    )


@app.post("/reset-session")
def reset_session():
    session.clear()
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=app.config["DEBUG"])
