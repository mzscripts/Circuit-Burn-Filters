#!/usr/bin/env python3
"""
Fast batch runner for the circuit-burn filter packs.

Design goals:
- Load each source image once
- Build a flat job list for that image
- Parallelize the filter jobs
- Save outputs with numbered filenames
- Print clear progress and timing logs
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

try:
    import circuit_burn_filters_v1 as v1
    import circuit_burn_filters_v2 as v2
    import circuit_burn_filters_v3 as v3
    import circuit_burn_filters_v4 as v4
    import circuit_burn_class_1_filters as class1
except ImportError as exc:
    print(f"Failed to import circuit-burn filter packs: {exc}", file=sys.stderr)
    sys.exit(1)


# Set default paths relative to parent of Scripts folder
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_FOLDER = str(PROJECT_ROOT / "input-IMG")
OUTPUT_FOLDER = str(PROJECT_ROOT / "Scripts" / "combined_output")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


@dataclass(frozen=True)
class FilterJob:
    index: int
    pack_key: str
    filter_name: str
    output_name: str
    output_path: Path
    filter_func: Callable[[Image.Image], Any]


def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("circuit_burn_batch_runner")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(console_handler)

    # Create run_logs directory if it doesn't exist
    run_logs_dir = Path("run_logs")
    run_logs_dir.mkdir(exist_ok=True)
    
    log_name = f"circuit_burn_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = run_logs_dir / log_name
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger


def ensure_pil_image(result: Any) -> Image.Image:
    if isinstance(result, Image.Image):
        return result
    if isinstance(result, np.ndarray):
        arr = result
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    raise TypeError(f"Unsupported filter output type: {type(result)!r}")


def safe_image_copy(img: Image.Image) -> Image.Image:
    try:
        return img.copy()
    except Exception:
        return img


def collect_input_images(input_folder: Path) -> list[Path]:
    if not input_folder.exists():
        return []
    return sorted(
        path for path in input_folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_filter_registry() -> list[tuple[str, dict[str, Callable[[Image.Image], Any]]]]:
    return [
        ("v1", dict(v1.FILTERS)),
        ("v2", dict(v2.FILTERS)),
        ("v3", dict(v3.FILTERS)),
        ("v4", dict(v4.FILTERS)),
        ("class1", dict(class1.FILTER_FUNCTIONS)),
    ]


def build_jobs(image_index: int, output_folder: Path) -> list[FilterJob]:
    jobs: list[FilterJob] = []
    counter = 1

    for pack_key, filters in build_filter_registry():
        for filter_name, filter_func in filters.items():
            output_name = f"{counter:03d}_{pack_key}_{filter_name}_{image_index:03d}.png"
            jobs.append(
                FilterJob(
                    index=counter,
                    pack_key=pack_key,
                    filter_name=filter_name,
                    output_name=output_name,
                    output_path=output_folder / output_name,
                    filter_func=filter_func,
                )
            )
            counter += 1

    return jobs


def load_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as img:
        return img.convert("RGB")


def run_filter_job(base_image: Image.Image, job: FilterJob, overwrite: bool) -> dict[str, Any]:
    started = time.perf_counter()

    if not overwrite and job.output_path.exists():
        return {
            "status": "skipped",
            "job": job,
            "elapsed": time.perf_counter() - started,
            "size": job.output_path.stat().st_size,
        }

    image_copy = safe_image_copy(base_image)
    result = job.filter_func(image_copy)
    output_image = ensure_pil_image(result)
    output_image.save(job.output_path)

    return {
        "status": "ok",
        "job": job,
        "elapsed": time.perf_counter() - started,
        "size": job.output_path.stat().st_size,
        "dimensions": output_image.size,
    }


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"


def process_image(
    image_path: Path,
    image_index: int,
    total_images: int,
    output_folder: Path,
    workers: int,
    overwrite: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    image_started = time.perf_counter()
    jobs = build_jobs(image_index, output_folder)

    logger.info("-" * 72)
    logger.info("[load] image %s/%s | path=%s", image_index, total_images, image_path.name)
    base_image = load_image(image_path)
    logger.info(
        "[ready] %s | size=%sx%s | jobs=%s | workers=%s",
        image_path.name,
        base_image.width,
        base_image.height,
        len(jobs),
        workers,
    )

    summary = {
        "ok": 0,
        "skipped": 0,
        "failed": 0,
        "written_bytes": 0,
        "errors": [],
        "jobs": len(jobs),
        "elapsed": 0.0,
    }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(run_filter_job, base_image, job, overwrite): job
            for job in jobs
        }

        for future in as_completed(future_map):
            job = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                summary["failed"] += 1
                error_text = f"{job.output_name}: {exc}"
                summary["errors"].append(error_text)
                logger.error(
                    "[err] #%03d %s/%s | %s",
                    job.index,
                    job.pack_key,
                    job.filter_name,
                    exc,
                )
                logger.debug(traceback.format_exc())
                continue

            if result["status"] == "skipped":
                summary["skipped"] += 1
                logger.info(
                    "[skip] #%03d %s/%s | %s | time=%.2fs",
                    job.index,
                    job.pack_key,
                    job.filter_name,
                    job.output_name,
                    result["elapsed"],
                )
                continue

            summary["ok"] += 1
            summary["written_bytes"] += result["size"]
            dimensions = result["dimensions"]
            logger.info(
                "[ok] #%03d %s/%s | %s | %sx%s | %s | time=%.2fs",
                job.index,
                job.pack_key,
                job.filter_name,
                job.output_name,
                dimensions[0],
                dimensions[1],
                format_size(result["size"]),
                result["elapsed"],
            )

    summary["elapsed"] = time.perf_counter() - image_started
    logger.info(
        "[image-done] %s | ok=%s | skip=%s | fail=%s | bytes=%s | time=%.2fs",
        image_path.name,
        summary["ok"],
        summary["skipped"],
        summary["failed"],
        format_size(summary["written_bytes"]),
        summary["elapsed"],
    )
    return summary


def run_batch(
    input_folder: Path,
    output_folder: Path,
    workers: int,
    overwrite: bool,
    limit: int | None,
    logger: logging.Logger,
) -> int:
    images = collect_input_images(input_folder)
    if limit is not None:
        images = images[:limit]

    if not images:
        logger.error("No supported images found in %s", input_folder)
        return 1

    output_folder.mkdir(parents=True, exist_ok=True)

    job_count = len(build_jobs(1, output_folder))
    logger.info("=" * 72)
    logger.info("Circuit Bend Batch Runner")
    logger.info("=" * 72)
    logger.info("Input folder: %s", input_folder.resolve())
    logger.info("Output folder: %s", output_folder.resolve())
    logger.info("Images found: %s", len(images))
    logger.info("Filters per image: %s", job_count)
    logger.info("Workers: %s", workers)
    logger.info("Overwrite: %s", overwrite)

    total_started = time.perf_counter()
    overall_ok = 0
    overall_skipped = 0
    overall_failed = 0
    overall_bytes = 0

    total_images = len(images)
    for image_index, image_path in enumerate(images, start=1):
        logger.info(
            "[image-start] %s/%s | %s",
            image_index,
            total_images,
            image_path.name,
        )
        image_summary = process_image(
            image_path=image_path,
            image_index=image_index,
            total_images=total_images,
            output_folder=output_folder,
            workers=workers,
            overwrite=overwrite,
            logger=logger,
        )
        overall_ok += image_summary["ok"]
        overall_skipped += image_summary["skipped"]
        overall_failed += image_summary["failed"]
        overall_bytes += image_summary["written_bytes"]

    elapsed = time.perf_counter() - total_started
    logger.info("=" * 72)
    logger.info("Run complete")
    logger.info("=" * 72)
    logger.info("Images processed: %s", total_images)
    logger.info("Outputs created: %s", overall_ok)
    logger.info("Outputs skipped: %s", overall_skipped)
    logger.info("Outputs failed: %s", overall_failed)
    logger.info("Bytes written: %s", format_size(overall_bytes))
    logger.info("Total time: %.2fs", elapsed)
    logger.info("Average time per image: %.2fs", elapsed / total_images)

    return 0 if overall_failed == 0 else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast runner for circuit_burn filter packs."
    )
    default_input = Path(os.path.abspath(INPUT_FOLDER))
    default_output = Path(os.path.abspath(OUTPUT_FOLDER))
    parser.add_argument("--input", type=Path, default=default_input, help="Input folder")
    parser.add_argument("--output", type=Path, default=default_output, help="Output folder")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, max(4, os.cpu_count() or 4)),
        help="Parallel workers per image",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger(args.verbose)
    return run_batch(
        input_folder=args.input,
        output_folder=args.output,
        workers=max(1, args.workers),
        overwrite=args.overwrite,
        limit=args.limit,
        logger=logger,
    )


if __name__ == "__main__":
    raise SystemExit(main())
