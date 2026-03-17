"""
Unified runner for the active filters in `selected filters`.

This keeps only the filters that are still exposed by those scripts,
preserves their existing settings, and saves outputs in the same flat
style as `circuit_burn_filters_v3.py`:

    circuitburn_class_1/<filter_name>__<original_stem>.png
"""

from __future__ import annotations

import glob
import importlib.util
import os
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

INPUT_FOLDER = "jesse"
OUTPUT_FOLDER = "circuitburn_class_1"
SOURCE_FOLDER = Path(__file__).resolve().parent / "selected filters"
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_images(folder: str) -> list[str]:
    paths: list[str] = []
    for ext in SUPPORTED:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(paths))


def save(image: Image.Image, src_path: str, filter_name: str) -> None:
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    stem = Path(src_path).stem
    dest = Path(OUTPUT_FOLDER) / f"{filter_name}__{stem}.png"
    image.save(dest)
    print(f"  ok  {dest}")


def load_module(module_name: str, file_name: str):
    path = SOURCE_FOLDER / file_name
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    if file_name == "nostalgia.py":
        # This source file still references removed filters in its dispatch table.
        # Seed no-op placeholders so the one remaining real filter can still load.
        missing = [
            "filter_neon_memory",
            "filter_cybernight_ccd",
            "filter_malllight_nostalgia",
            "filter_sodium_dream",
            "filter_midnight_blossom",
            "filter_aqua_magenta_memory",
            "filter_msn_call_haunt",
            "filter_yahoo_messenger_night",
            "filter_bedroom_webcam_2007",
            "filter_blown_window_feed",
            "filter_skype_freezeface",
        ]
        for name in missing:
            module.__dict__[name] = lambda img: img
    spec.loader.exec_module(module)
    return module


def ensure_pil_image(result) -> Image.Image:
    if isinstance(result, Image.Image):
        return result
    if isinstance(result, np.ndarray):
        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)
    raise TypeError(f"Unsupported filter result type: {type(result)!r}")


def resize_to_original(result, original: Image.Image) -> Image.Image:
    image = ensure_pil_image(result)
    if image.size != original.size:
        image = image.resize(original.size, Image.LANCZOS)
    return image


def build_registry() -> dict[str, Callable[[Image.Image], Image.Image]]:
    registry: dict[str, Callable[[Image.Image], Image.Image]] = {}

    module_filters = [
        (
            "camcorder",
            "camcorder.py",
            [
                "minidv_dropframe",
                "nightbus_cmos",
                "tape_head_damage",
                "handycam_ghostwalk",
                "clubcam_redroom",
                "security_stairwell",
                "found_footage_burn",
            ],
        ),
        (
            "corrupt",
            "corrupt.py",
            [
                "decoder_crush",
                "ir_nightwatch",
                "vertical_hold_fail",
                "memory_card_bleed",
                "rolling_shutter_panic",
                "tape_dropout",
                "surveillance_burnin",
                "lowlight_smear",
                "signal_autopsy",
            ],
        ),
        (
            "dirty_glitch",
            "dirty_glithch.py",
            [
                "noisy_portrait",
                "crt_tear",
                "timestamp_cam",
                "datamosh_ghost",
                "scanline_warp",
                "dirty_sky_burn",
                "edge_noise_emboss",
                "rgb_face_break",
            ],
        ),
        (
            "disposable",
            "disposable.py",
            [
                "partyfloor_overkill",
            ],
        ),
        (
            "glitch_filter_pack",
            "glitch_filter_pack.py",
            [
                "broken_ccd",
                "dead_sensor",
                "memory_card_corrupt",
                "vhs_drift",
                "pixel_melt",
                "jpeg_crush",
                "signal_torn",
                "night_vision",
                "infrared_bloom",
                "xray_negative",
            ],
        ),
        (
            "haunted_chroma_pack",
            "haunted_chroma_pack.py",
            [
                "void_stencil",
                "solar_saint",
                "xray_orchid",
                "plasma_ruins",
                "ghost_negative",
                "moonlit_relic",
                "prism_bloom",
                "funeral_sun",
                "hologram_grave",
                "bleached_oracle",
            ],
        ),
        (
            "nostalgia",
            "nostalgia.py",
            [
                "ghost_in_the_lcd",
            ],
        ),
        (
            "pixless",
            "pixless.py",
            [
                "classic_toycam",
                "pixless_plus",
                "sensor_grit",
                "lcd_dream",
                "night_sensor",
                "broken_handheld",
                "v1_simple",
                "v2_pixless_legacy",
                "v3_advanced_legacy",
            ],
        ),
        (
            "spectral",
            "spectral.py",
            [
                "cobalt_mask",
                "toxic_thermal",
                "melted_halo",
                "rainbow_chapel",
                "graveyard_acid",
                "spectral_void",
                "pastel_angel",
                "neon_church",
                "aurora_statue",
            ],
        ),
    ]

    for module_name, file_name, filter_names in module_filters:
        module = load_module(module_name, file_name)
        filter_functions = getattr(module, "FILTER_FUNCTIONS")
        for filter_name in filter_names:
            if filter_name not in filter_functions:
                raise KeyError(f"{filter_name!r} missing from {file_name}")
            func = filter_functions[filter_name]
            if module_name == "pixless":
                registry[filter_name] = (
                    lambda img, fn=func: resize_to_original(fn(img), img)
                )
            else:
                registry[filter_name] = func

    glitch_cam = load_module("glitch_cam", "glitch_cam.py")
    registry["glitch_pipeline"] = lambda img, module=glitch_cam: ensure_pil_image(
        module.apply_glitch_pipeline(img)
    )

    psychedelic = load_module("glitch_psychedelic", "glitch_psychedelic.py")
    registry["psychedelic_circuit_bend"] = lambda img, module=psychedelic: ensure_pil_image(
        module.psychedelic_circuit_bend_filter(img)
    )

    reference_free = load_module("reference_free_glitch", "reference_free_glitch.py")
    for preset_name in getattr(reference_free, "PRESETS"):
        if preset_name in registry:
            continue
        registry[preset_name] = (
            lambda img, preset=preset_name, module=reference_free: ensure_pil_image(
                module.stylize_preset(img, preset)
            )
        )

    return registry


FILTER_FUNCTIONS = build_registry()
FILTERS = list(FILTER_FUNCTIONS.keys())


def main() -> None:
    files = load_images(INPUT_FOLDER)

    if not files:
        print(f"No images found in '{INPUT_FOLDER}'")
        return

    print(f"Found {len(files)} image(s). Processing with {len(FILTERS)} filters...")

    for src_path in files:
        try:
            with Image.open(src_path) as img:
                base = img.convert("RGB")
                print(f"\n>> {Path(src_path).name}")
                for filter_name in FILTERS:
                    out = ensure_pil_image(FILTER_FUNCTIONS[filter_name](base.copy()))
                    save(out, src_path, filter_name)
        except Exception as exc:
            print(f"  !! failed for {src_path}: {exc}")


if __name__ == "__main__":
    main()
