from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
from pathlib import Path
import re
import sys
from typing import Callable

from PIL import Image

from utils.image_io import ensure_pil_image

from .glitch import neon_glitch
from .neon import neon_bloom


FilterCallable = Callable[[Image.Image], object]


@dataclass(frozen=True)
class FilterMetadata:
    id: str
    name: str
    category: str
    group: str
    sequence_number: int
    thumbnail_url: str | None
    function: FilterCallable

    def to_public_dict(self) -> dict:
        data = asdict(self)
        data.pop("function", None)
        return {"display_name": data.pop("name"), **data}


REGISTRY: dict[str, FilterMetadata] = {}


def _humanize_filter_name(filter_id: str) -> str:
    cleaned = re.sub(r"^\d+[_-]?", "", filter_id)
    return cleaned.replace("_", " ").replace("-", " ").title()


def _normalize_category(filter_id: str, name: str) -> str:
    text = f"{filter_id} {name}".lower()
    if any(token in text for token in ("glitch", "vhs", "datamosh", "scan", "signal", "dropframe", "tape", "rgb", "corrupt", "decoder", "shutter", "sensor", "smear", "column", "static")):
        return "Glitch"
    if any(token in text for token in ("neon", "plasma", "prism", "aurora", "hologram", "uv", "infrared", "thermal", "xray", "spectral")):
        return "Neon"
    if any(token in text for token in ("cinematic", "film", "noir", "polaroid", "cassette", "toycam", "cam", "handheld", "nightbus", "clubcam")):
        return "Cinematic"
    if any(token in text for token in ("acid", "psychedelic", "rainbow", "mystiq", "halo", "temple", "church", "void", "oracle", "grave", "funeral")):
        return "Psychedelic"
    if any(token in text for token in ("crystal", "bismuth", "geological", "magnetic", "mycelium", "pollen", "terrain", "satellite", "core")):
        return "Experimental"
    return "Stylized"


def register_filter(
    *,
    filter_id: str,
    name: str,
    category: str,
    sequence_number: int,
    thumbnail_url: str | None,
    function: FilterCallable,
) -> None:
    REGISTRY[filter_id] = FilterMetadata(
        id=filter_id,
        name=name,
        category=category,
        group="Class 1 Filters",
        sequence_number=sequence_number,
        thumbnail_url=thumbnail_url,
        function=function,
    )


def _parse_thumbnail_map() -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], int], dict[str, str]]:
    thumbs_path = Path(__file__).resolve().parents[2] / "thumbs.md"
    if not thumbs_path.exists():
        return {}, {}, {}

    contents = thumbs_path.read_text(encoding="utf-8")
    pattern = re.compile(r'<img\s+src="([^"]+)"\s+alt="([^"]+)"', re.IGNORECASE)
    thumbnail_map: dict[tuple[str, str], str] = {}
    sequence_map: dict[tuple[str, str], int] = {}
    custom_thumbnail_map: dict[str, str] = {}

    for src, alt in pattern.findall(contents):
        parts = alt.split("-")
        if len(parts) >= 4 and parts[0].isdigit():
            sequence_number = int(parts[0])
            pack_key = parts[1].lower()
            if pack_key in {"v1", "v2", "v3", "v4"} and len(parts) >= 5:
                filter_id = f"{parts[2]}_{'_'.join(parts[3:-1])}".lower()
            else:
                filter_id = "_".join(parts[2:-1]).lower()
            thumbnail_map[(pack_key, filter_id)] = src
            sequence_map[(pack_key, filter_id)] = sequence_number
            continue

        lowered_alt = alt.lower()
        if "-neon-glitch-" in lowered_alt:
            custom_thumbnail_map["neon_glitch"] = src
        elif "-neon-bloom-" in lowered_alt:
            custom_thumbnail_map["neon_bloom"] = src

    return thumbnail_map, sequence_map, custom_thumbnail_map


THUMBNAIL_MAP, SEQUENCE_MAP, CUSTOM_THUMBNAIL_MAP = _parse_thumbnail_map()


def _thumbnail_for(pack_key: str, filter_id: str) -> str | None:
    return THUMBNAIL_MAP.get((pack_key.lower(), filter_id.lower()))


def _sequence_for(pack_key: str, filter_id: str) -> int | None:
    return SEQUENCE_MAP.get((pack_key.lower(), filter_id.lower()))


def _wrap_legacy_filter(function: Callable) -> FilterCallable:
    def runner(image: Image.Image):
        result = function(image)
        return ensure_pil_image(result)

    return runner


def _register_custom_filters() -> None:
    register_filter(
        filter_id="neon_glitch",
        name="Neon Glitch",
        category="Glitch",
        sequence_number=187,
        thumbnail_url=CUSTOM_THUMBNAIL_MAP.get("neon_glitch"),
        function=neon_glitch,
    )
    register_filter(
        filter_id="neon_bloom",
        name="Neon Bloom",
        category="Neon",
        sequence_number=188,
        thumbnail_url=CUSTOM_THUMBNAIL_MAP.get("neon_bloom"),
        function=neon_bloom,
    )


def _load_existing_script_filters() -> None:
    scripts_dir = Path(__file__).resolve().parents[2] / "Scripts"
    if not scripts_dir.exists():
        return

    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    pack_specs = [
        ("circuit_burn_filters_v1", "FILTERS", "v1"),
        ("circuit_burn_filters_v2", "FILTERS", "v2"),
        ("circuit_burn_filters_v3", "FILTERS", "v3"),
        ("circuit_burn_filters_v4", "FILTERS", "v4"),
        ("circuit_burn_class_1_filters", "FILTER_FUNCTIONS", "class1"),
    ]

    for module_name, attribute_name, thumb_pack_key in pack_specs:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        mapping = getattr(module, attribute_name)
        for filter_id, function in mapping.items():
            if filter_id in REGISTRY:
                continue
            name = _humanize_filter_name(filter_id)
            register_filter(
                filter_id=filter_id,
                name=name,
                category=_normalize_category(filter_id, name),
                sequence_number=_sequence_for(thumb_pack_key, filter_id) or 9999,
                thumbnail_url=_thumbnail_for(thumb_pack_key, filter_id),
                function=_wrap_legacy_filter(function),
            )


def load_registry() -> dict[str, FilterMetadata]:
    if REGISTRY:
        return REGISTRY
    _load_existing_script_filters()
    _register_custom_filters()
    return REGISTRY


def get_filter(filter_id: str) -> FilterMetadata | None:
    return load_registry().get(filter_id)


def get_all_filters() -> list[dict]:
    filters = [metadata.to_public_dict() for metadata in load_registry().values()]
    return sorted(filters, key=lambda item: item["sequence_number"])


def get_filters_grouped_by_category() -> dict[str, list[dict]]:
    all_filters = get_all_filters()
    return {"": all_filters}
