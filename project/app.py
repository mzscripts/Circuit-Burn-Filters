from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import io
import os
import threading
import time
import uuid

from flask import Flask, jsonify, redirect, render_template, request, session, url_for, send_file
from PIL import UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from config import Config
from filters import get_filter, get_filters_grouped_by_category, load_registry
from utils.image_io import image_to_upload_bytes
from utils.imgbb import fetch_remote_image, fetch_remote_image_bytes, upload_image_bytes
from utils.naming import output_name
from utils.validation import ValidationError, prepare_uploaded_image


app = Flask(__name__)
app.config.from_object(Config)
load_registry()

JOB_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="filter-job")
STORE_LOCK = threading.Lock()
JOB_STORE: dict[str, dict] = {}
CLIENT_RESULTS: dict[str, dict[str, dict]] = {}
CLIENT_JOBS: dict[str, set[str]] = {}
CLIENT_UPLOAD_TOKENS: dict[str, str] = {}


def is_api_request() -> bool:
    return request.path.startswith("/api/") or request.path in {"/apply-multiple", "/reset-session"}


def json_error(message: str, status_code: int = 400):
    response = jsonify({"success": False, "error": message})
    response.status_code = status_code
    return response


def current_upload() -> dict | None:
    return session.get("uploaded_image")


def get_client_id() -> str:
    client_id = session.get("client_id")
    if client_id:
        return client_id
    client_id = uuid.uuid4().hex
    session["client_id"] = client_id
    session.modified = True
    return client_id


def current_results() -> list[dict]:
    client_id = get_client_id()
    with STORE_LOCK:
        results = list(CLIENT_RESULTS.get(client_id, {}).values())
    return sorted(results, key=lambda item: item["sequence_number"])


def save_results(results: list[dict]) -> None:
    client_id = get_client_id()
    with STORE_LOCK:
        CLIENT_RESULTS[client_id] = {item["filter_id"]: item for item in results}


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
    client_id = get_client_id()
    store_result_for_client(client_id, result)


def store_result_for_client(client_id: str, result: dict) -> None:
    with STORE_LOCK:
        CLIENT_RESULTS.setdefault(client_id, {})[result["filter_id"]] = result


def current_job_map() -> dict[str, dict]:
    client_id = get_client_id()
    with STORE_LOCK:
        job_ids = list(CLIENT_JOBS.get(client_id, set()))
        return {
            job_id: serialize_job(JOB_STORE[job_id])
            for job_id in job_ids
            if job_id in JOB_STORE
        }


def get_active_job_for_filter(filter_id: str) -> dict | None:
    client_id = get_client_id()
    with STORE_LOCK:
        for job_id in sorted(CLIENT_JOBS.get(client_id, set()), reverse=True):
            job = JOB_STORE.get(job_id)
            if not job or job["filter_id"] != filter_id:
                continue
            if job["status"] in {"queued", "processing"}:
                return serialize_job(job)
    return None


def clear_client_state(client_id: str) -> None:
    with STORE_LOCK:
        CLIENT_RESULTS.pop(client_id, None)
        for job_id in CLIENT_JOBS.pop(client_id, set()):
            job = JOB_STORE.get(job_id)
            if job and job["status"] in {"queued", "processing"}:
                job["status"] = "stale"
                job["completed_at"] = time.time()


def serialize_job(job: dict) -> dict:
    payload = {
        "job_id": job["job_id"],
        "filter_id": job["filter_id"],
        "filter_name": job["filter_name"],
        "category": job["category"],
        "sequence_number": job["sequence_number"],
        "status": job["status"],
        "error": job.get("error"),
        "created_at": job["created_at"],
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
    }
    if job.get("result"):
        payload["result"] = job["result"]
    return payload


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
    extension = result.get("download_extension") or os.path.splitext(result.get("image_url", ""))[1] or ".jpg"
    base = result["filter_name"].lower().replace(" ", "_")
    return f"{base}{extension}"


def process_filter_job(filter_id: str, upload: dict, metadata) -> dict:
    started = time.perf_counter()
    original_image = None
    output_image = None
    output_bytes = None
    uploaded_output = None
    try:
        original_image = fetch_remote_image(upload["image_url"])
        output_image = metadata.function(original_image)
        output_bytes, mime_type, download_extension = image_to_upload_bytes(
            output_image,
            jpeg_quality=app.config["OUTPUT_JPEG_QUALITY"],
        )
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
            "mime_type": mime_type,
            "download_extension": download_extension,
        }
        return result
    finally:
        if original_image is not None:
            original_image.close()
        if output_image is not None and output_image is not original_image:
            output_image.close()
        del original_image
        del output_image
        del output_bytes
        del uploaded_output
        gc.collect()


def run_job(job_id: str) -> None:
    with STORE_LOCK:
        job = JOB_STORE.get(job_id)
        if not job or job["status"] != "queued":
            return
        job["status"] = "processing"
        job["started_at"] = time.time()
        filter_id = job["filter_id"]
        upload = job["upload"]
        upload_token = job["upload_token"]
        client_id = job["client_id"]

    try:
        metadata = get_filter(filter_id)
        if metadata is None:
            raise ValidationError("The selected filter could not be found.")
        result = process_filter_job(filter_id, upload, metadata)
        with STORE_LOCK:
            job = JOB_STORE.get(job_id)
            if not job:
                return
            current_token = CLIENT_UPLOAD_TOKENS.get(client_id)
            if current_token != upload_token:
                job["status"] = "stale"
                job["completed_at"] = time.time()
                job["error"] = "This job belongs to an older upload."
                return
            job["status"] = "completed"
            job["completed_at"] = time.time()
            job["result"] = result
        store_result_for_client(client_id, result)
    except ValidationError as exc:
        with STORE_LOCK:
            job = JOB_STORE.get(job_id)
            if job:
                job["status"] = "failed"
                job["completed_at"] = time.time()
                job["error"] = str(exc)
    except (UnidentifiedImageError, OSError):
        with STORE_LOCK:
            job = JOB_STORE.get(job_id)
            if job:
                job["status"] = "failed"
                job["completed_at"] = time.time()
                job["error"] = "The image could not be decoded safely."
    except Exception as exc:
        with STORE_LOCK:
            job = JOB_STORE.get(job_id)
            if job:
                job["status"] = "failed"
                job["completed_at"] = time.time()
                job["error"] = str(exc)
    finally:
        gc.collect()


def enqueue_filter_job(filter_id: str) -> dict:
    metadata = get_filter(filter_id)
    if metadata is None:
        raise ValidationError("The selected filter could not be found.")

    upload = current_upload()
    if not upload:
        raise ValidationError("Upload an image before applying filters.")

    client_id = get_client_id()
    upload_token = upload.get("upload_token")
    if not upload_token:
        raise ValidationError("Upload an image before applying filters.")

    with STORE_LOCK:
        for existing_job_id in CLIENT_JOBS.get(client_id, set()):
            existing_job = JOB_STORE.get(existing_job_id)
            if (
                existing_job
                and existing_job["filter_id"] == filter_id
                and existing_job["status"] in {"queued", "processing"}
                and existing_job["upload_token"] == upload_token
            ):
                return serialize_job(existing_job)

        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "client_id": client_id,
            "filter_id": metadata.id,
            "filter_name": metadata.name,
            "category": metadata.category,
            "sequence_number": metadata.sequence_number,
            "status": "queued",
            "created_at": time.time(),
            "upload_token": upload_token,
            "upload": dict(upload),
            "error": None,
            "result": None,
        }
        JOB_STORE[job_id] = job
        CLIENT_JOBS.setdefault(client_id, set()).add(job_id)

    JOB_EXECUTOR.submit(run_job, job_id)
    return serialize_job(job)


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
    get_client_id()
    return render_template("index.html", error=None)


@app.post("/upload")
def upload():
    file = request.files.get("image")
    client_id = get_client_id()
    original_name = secure_filename(file.filename if file else "")
    prepared_image = None
    upload_bytes = None
    uploaded_image = None
    try:
        clear_client_state(client_id)
        prepared_image, _file_size = prepare_uploaded_image(
            file,
            app.config["ALLOWED_EXTENSIONS"],
            max_file_size_bytes=app.config["MAX_CONTENT_LENGTH"],
            max_pixels=app.config["MAX_IMAGE_PIXELS"],
            max_dimension=app.config["MAX_IMAGE_DIMENSION"],
        )
        upload_bytes, _mime_type, _download_extension = image_to_upload_bytes(
            prepared_image,
            jpeg_quality=app.config["OUTPUT_JPEG_QUALITY"],
        )
        uploaded_image = upload_image_bytes(
            image_bytes=upload_bytes,
            api_key=app.config["IMGBB_API_KEY"],
            upload_url=app.config["IMGBB_UPLOAD_URL"],
            name=original_name,
        )

        upload_token = uuid.uuid4().hex
        CLIENT_UPLOAD_TOKENS[client_id] = upload_token
        session["uploaded_image"] = {
            "original_name": original_name,
            "image_url": uploaded_image["url"],
            "display_url": uploaded_image["display_url"],
            "delete_url": uploaded_image.get("delete_url"),
            "upload_token": upload_token,
        }
        session.modified = True
        return redirect(url_for("filters_page"))
    finally:
        if prepared_image is not None:
            prepared_image.close()
        del prepared_image
        del upload_bytes
        del uploaded_image
        gc.collect()


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
        active_jobs=current_job_map(),
        total_filters=len(load_registry()),
    )


@app.get("/process/<filter_id>")
def process_page(filter_id: str):
    return redirect(url_for("filters_page"))


@app.post("/api/process/<filter_id>")
def process_api(filter_id: str):
    try:
        job = enqueue_filter_job(filter_id)
        return jsonify({"success": True, "job": job})
    except ValidationError as exc:
        return json_error(str(exc), 400)
    except (UnidentifiedImageError, OSError):
        return json_error(f"Failed to process '{filter_id}': the image could not be decoded safely.", 400)
    except Exception as exc:
        return json_error(f"Failed to process '{filter_id}': {exc}", 500)


@app.get("/api/jobs/<job_id>")
def job_status_api(job_id: str):
    client_id = get_client_id()
    with STORE_LOCK:
        job = JOB_STORE.get(job_id)
        if not job or job["client_id"] != client_id:
            return json_error("That job could not be found.", 404)
        return jsonify({"success": True, "job": serialize_job(job)})


@app.get("/download/<filter_id>")
def download_result(filter_id: str):
    result = get_result_by_filter(filter_id)
    if result is None:
        return redirect(url_for("results_page"))

    payload = fetch_remote_image_bytes(result["image_url"])
    return send_file(
        io.BytesIO(payload),
        mimetype=result.get("mime_type", "application/octet-stream"),
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

    jobs = []
    for filter_id in filter_ids:
        try:
            job = enqueue_filter_job(filter_id)
            jobs.append({"success": True, "job": job})
        except Exception as exc:
            jobs.append({
                "success": False,
                "filter_id": filter_id,
                "error": str(exc),
            })

    return jsonify({"success": True, "jobs": jobs})


@app.get("/results")
def results_page():
    upload, redirect_response = require_uploaded_image()
    if redirect_response:
        return redirect_response

    return render_template(
        "results.html",
        uploaded_image=upload,
        generated_results=current_results(),
        active_jobs=current_job_map(),
    )


@app.post("/reset-session")
def reset_session():
    client_id = session.get("client_id")
    if client_id:
        clear_client_state(client_id)
        CLIENT_UPLOAD_TOKENS.pop(client_id, None)
    session.clear()
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=app.config["DEBUG"])
