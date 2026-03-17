import os
import random
import math
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

# =========================================================
# CONFIG
# =========================================================
INPUT_FOLDER = "wallter"
OUTPUT_FOLDER = "reference_free_glitch"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

PRESETS = [
    "spectral_statue",
    "rainbow_church",
    "toxic_thermal",
    "pastel_angel",
    "neon_void",
    "dirty_portrait",
    "crt_ghost",
    "infrared_ruin",
]

SKY_THRESHOLD = 0.55
SUBJECT_THRESHOLD = 0.48

# =========================================================
# UTILS
# =========================================================
def ensure_output_folder():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def list_images(folder):
    if not os.path.exists(folder):
        return []
    return [f for f in os.listdir(folder) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

def load_image(path):
    return Image.open(path).convert("RGB")

def save_image(img, path):
    img.save(path)

def np_img(img):
    return np.array(img).astype(np.uint8)

def pil_img(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def clamp_u8(arr):
    return np.clip(arr, 0, 255).astype(np.uint8)

def grayscale_arr(img_or_arr):
    if isinstance(img_or_arr, Image.Image):
        return np.array(ImageOps.grayscale(img_or_arr))
    else:
        arr = img_or_arr.astype(np.float32)
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        return clamp_u8(gray)

def add_noise(arr, strength=8):
    noise = np.random.randint(-strength, strength + 1, arr.shape, dtype=np.int16)
    out = arr.astype(np.int16) + noise
    return clamp_u8(out)

def add_speckle(arr, amount=0.02):
    out = arr.copy()
    h, w, _ = out.shape
    count = int(h * w * amount)
    ys = np.random.randint(0, h, count)
    xs = np.random.randint(0, w, count)
    vals = np.random.randint(0, 256, (count, 3))
    out[ys, xs] = vals
    return out

def blend(a, b, alpha=0.5):
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    out = af * (1 - alpha) + bf * alpha
    return clamp_u8(out)

def rgb_shift(arr, max_shift=4):
    out = np.zeros_like(arr)
    for c in range(3):
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        out[:, :, c] = np.roll(np.roll(arr[:, :, c], dy, axis=0), dx, axis=1)
    return out

def posterize_np(arr, bits=3):
    shift = 8 - bits
    return ((arr >> shift) << shift).astype(np.uint8)

def box_blur_mask(mask, radius=3):
    m = mask.astype(np.float32)
    for _ in range(radius):
        p = np.pad(m, ((1, 1), (1, 1)), mode='edge')
        m = (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
            p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
            p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
        ) / 9.0
    return np.clip(m, 0, 1)

def add_scanlines(arr, strength=0.12, spacing=2):
    out = arr.astype(np.float32).copy()
    h = out.shape[0]
    for y in range(0, h, spacing):
        out[y, :, :] *= (1.0 - strength)
    return clamp_u8(out)

# =========================================================
# LUT / PALETTE GENERATION
# =========================================================
PRESET_ANCHORS = {
    "spectral_statue": {
        "sky": [
            (15, 0, 80),
            (90, 0, 180),
            (0, 160, 255),
            (120, 255, 200),
            (255, 0, 255),
        ],
        "fg": [
            (0, 0, 0),
            (0, 30, 80),
            (30, 120, 180),
            (180, 220, 255),
            (255, 255, 255),
        ],
        "global": [
            (0, 0, 20),
            (30, 0, 90),
            (0, 120, 255),
            (0, 255, 180),
            (255, 0, 255),
            (255, 255, 255),
        ],
    },
    "rainbow_church": {
        "sky": [
            (0, 0, 255),
            (0, 255, 255),
            (0, 255, 0),
            (255, 255, 0),
            (255, 0, 0),
            (255, 0, 255),
        ],
        "fg": [
            (0, 0, 0),
            (40, 0, 80),
            (0, 100, 255),
            (0, 255, 120),
            (255, 0, 255),
            (255, 255, 0),
        ],
        "global": [
            (0, 0, 40),
            (0, 80, 255),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
        ],
    },
    "toxic_thermal": {
        "sky": [
            (0, 0, 120),
            (0, 180, 255),
            (0, 255, 140),
            (255, 255, 0),
            (255, 100, 0),
            (255, 0, 255),
        ],
        "fg": [
            (0, 0, 100),
            (0, 60, 255),
            (0, 255, 255),
            (0, 255, 0),
            (255, 255, 0),
            (255, 0, 0),
        ],
        "global": [
            (0, 0, 80),
            (0, 120, 255),
            (0, 255, 180),
            (180, 255, 0),
            (255, 60, 220),
            (255, 0, 0),
        ],
    },
    "pastel_angel": {
        "sky": [
            (240, 220, 255),
            (210, 240, 255),
            (255, 240, 220),
            (255, 220, 255),
            (220, 255, 230),
            (255, 255, 255),
        ],
        "fg": [
            (120, 120, 160),
            (180, 200, 240),
            (220, 230, 255),
            (255, 240, 240),
            (255, 255, 255),
        ],
        "global": [
            (180, 180, 220),
            (220, 230, 255),
            (255, 230, 240),
            (255, 255, 255),
        ],
    },
    "neon_void": {
        "sky": [
            (0, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
            (255, 255, 255),
        ],
        "fg": [
            (0, 0, 0),
            (0, 20, 40),
            (0, 120, 180),
            (120, 255, 255),
            (255, 255, 255),
        ],
        "global": [
            (0, 0, 0),
            (0, 50, 80),
            (0, 180, 255),
            (255, 0, 255),
            (255, 255, 255),
        ],
    },
    "dirty_portrait": {
        "sky": [
            (20, 0, 40),
            (80, 0, 120),
            (180, 0, 255),
            (255, 80, 220),
            (255, 180, 255),
        ],
        "fg": [
            (0, 0, 0),
            (50, 20, 80),
            (180, 50, 220),
            (255, 80, 255),
            (255, 255, 255),
        ],
        "global": [
            (0, 0, 0),
            (50, 0, 80),
            (255, 0, 255),
            (0, 255, 180),
            (255, 255, 255),
        ],
    },
    "crt_ghost": {
        "sky": [
            (0, 0, 0),
            (20, 40, 80),
            (40, 160, 180),
            (160, 255, 255),
        ],
        "fg": [
            (0, 0, 0),
            (30, 30, 30),
            (100, 120, 140),
            (220, 240, 255),
        ],
        "global": [
            (0, 0, 0),
            (20, 20, 40),
            (40, 120, 180),
            (200, 255, 255),
        ],
    },
    "infrared_ruin": {
        "sky": [
            (0, 60, 255),
            (0, 200, 255),
            (120, 255, 255),
            (255, 180, 255),
            (255, 255, 255),
        ],
        "fg": [
            (0, 0, 0),
            (60, 20, 80),
            (180, 80, 180),
            (255, 180, 220),
            (255, 255, 255),
        ],
        "global": [
            (0, 0, 20),
            (80, 0, 120),
            (180, 80, 255),
            (255, 200, 255),
            (255, 255, 255),
        ],
    },
}

def make_lut_from_anchors(anchors):
    """
    anchors: list of RGB tuples
    returns 256x3 LUT
    """
    anchors = np.array(anchors, dtype=np.float32)
    n = len(anchors)

    lut = np.zeros((256, 3), dtype=np.float32)

    for i in range(256):
        t = i / 255.0
        scaled = t * (n - 1)
        idx = int(np.floor(scaled))
        frac = scaled - idx

        if idx >= n - 1:
            lut[i] = anchors[-1]
        else:
            lut[i] = anchors[idx] * (1 - frac) + anchors[idx + 1] * frac

    return clamp_u8(lut)

def build_preset_luts(preset_name):
    cfg = PRESET_ANCHORS[preset_name]
    sky_lut = make_lut_from_anchors(cfg["sky"])
    fg_lut = make_lut_from_anchors(cfg["fg"])
    global_lut = make_lut_from_anchors(cfg["global"])
    return sky_lut, fg_lut, global_lut

def apply_lut_to_gray(gray, lut):
    return lut[gray]

# =========================================================
# SKY / SUBJECT MASKS
# =========================================================
def estimate_sky_mask(arr):
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]

    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)

    brightness = arr.mean(axis=2) / 255.0
    blue_dom = np.clip((b - (r + g) / 2.0) / 255.0, 0, 1)
    top_bias = 1.0 - (yy / max(1, h - 1))

    pil = pil_img(arr)
    edges = np.array(pil.filter(ImageFilter.FIND_EDGES).convert("L")) / 255.0
    smoothness = 1.0 - edges

    sky_score = (
        0.35 * top_bias +
        0.25 * blue_dom +
        0.20 * brightness +
        0.20 * smoothness
    )

    sky_mask = (sky_score > SKY_THRESHOLD).astype(np.float32)
    sky_mask = box_blur_mask(sky_mask, radius=4)
    return np.clip(sky_mask, 0, 1)

def estimate_subject_mask(arr, sky_mask):
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]

    pil = pil_img(arr)

    edges = np.array(pil.filter(ImageFilter.FIND_EDGES).convert("L")) / 255.0

    blur = np.array(pil.filter(ImageFilter.GaussianBlur(radius=4)).convert("L")).astype(np.float32)
    gray = grayscale_arr(arr).astype(np.float32)
    local_contrast = np.abs(gray - blur) / 255.0

    center_bias = 1.0 - np.sqrt(
        ((xx - w / 2) / max(1, w / 2)) ** 2 +
        ((yy - h / 2) / max(1, h / 2)) ** 2
    )
    center_bias = np.clip(center_bias, 0, 1)

    bottom_bias = yy / max(1, h - 1)

    fg_score = (
        0.30 * edges +
        0.25 * local_contrast +
        0.25 * center_bias +
        0.20 * bottom_bias
    )

    fg_score *= (1.0 - 0.75 * sky_mask)

    subject_mask = (fg_score > SUBJECT_THRESHOLD).astype(np.float32)
    subject_mask = box_blur_mask(subject_mask, radius=4)

    return np.clip(subject_mask, 0, 1)

# =========================================================
# REGION-AWARE POSTERIZATION
# =========================================================
def preset_poster_bits(preset_name):
    if preset_name in ["rainbow_church", "toxic_thermal"]:
        return (2, 3, 5)  # sky, subject, bg
    elif preset_name in ["pastel_angel"]:
        return (4, 5, 6)
    elif preset_name in ["dirty_portrait", "crt_ghost"]:
        return (3, 4, 5)
    elif preset_name in ["neon_void"]:
        return (3, 4, 5)
    else:
        return (2, 4, 5)

def region_aware_posterize(gray, sky_mask, subject_mask, preset_name):
    sky_bits, subject_bits, bg_bits = preset_poster_bits(preset_name)

    sky_p = posterize_np(gray, bits=sky_bits).astype(np.float32)
    subj_p = posterize_np(gray, bits=subject_bits).astype(np.float32)
    bg_p = posterize_np(gray, bits=bg_bits).astype(np.float32)

    bg_mask = np.clip(1.0 - np.maximum(sky_mask, subject_mask), 0, 1)
    total = sky_mask + subject_mask + bg_mask + 1e-6

    out = (
        sky_p * (sky_mask / total) +
        subj_p * (subject_mask / total) +
        bg_p * (bg_mask / total)
    )

    return clamp_u8(out)

# =========================================================
# MULTI-PASS EDGE STACK
# =========================================================
def threshold_edges(img, threshold):
    e = np.array(img.filter(ImageFilter.FIND_EDGES).convert("L"))
    return np.where(e > threshold, e, 0).astype(np.uint8)

def multi_pass_edge_stack(img):
    coarse_src = img.filter(ImageFilter.GaussianBlur(radius=3))
    coarse = threshold_edges(coarse_src, threshold=random.randint(25, 60)).astype(np.float32)

    medium = threshold_edges(img, threshold=random.randint(20, 50)).astype(np.float32)

    sharp = ImageEnhance.Sharpness(img).enhance(2.5)
    fine = threshold_edges(sharp, threshold=random.randint(12, 35)).astype(np.float32)

    emboss = np.array(img.filter(ImageFilter.EMBOSS).convert("L")).astype(np.float32)

    edge = (
        0.38 * coarse +
        0.32 * medium +
        0.18 * fine +
        0.20 * emboss
    )

    return np.clip(edge, 0, 255).astype(np.uint8)

def apply_adaptive_edges(color_arr, edge_map, sky_mask, subject_mask, preset_name):
    edge = edge_map.astype(np.float32) / 255.0
    bg_mask = np.clip(1.0 - np.maximum(sky_mask, subject_mask), 0, 1)

    # preset-aware contour strengths
    if preset_name in ["pastel_angel"]:
        s_w, b_w, k_w = 0.45, 0.20, 0.05
    elif preset_name in ["dirty_portrait", "crt_ghost"]:
        s_w, b_w, k_w = 0.90, 0.55, 0.12
    else:
        s_w, b_w, k_w = 0.80, 0.45, 0.10

    weight = (
        subject_mask * s_w +
        bg_mask * b_w +
        sky_mask * k_w
    )

    out = color_arr.astype(np.float32)
    edge3 = (edge * weight)[:, :, None]

    # default black contour
    out = out * (1.0 - edge3)

    # pastel angel can use white-ish contour lift
    if preset_name == "pastel_angel":
        out = out * (1 - 0.25 * edge3) + 255 * (0.15 * edge3)

    return clamp_u8(out)

# =========================================================
# SKY OVERLAYS / SPECIAL OVERLAYS
# =========================================================
def make_banded_sky_overlay(h, w, lut):
    yy, xx = np.mgrid[0:h, 0:w]

    t = (yy / max(1, h - 1)) * 0.85 + 0.15 * np.sin(xx / max(1, w) * math.pi * random.uniform(1.5, 4.0))
    t = np.clip(t, 0, 1)

    levels = random.choice([4, 5, 6, 7])
    t = np.floor(t * levels) / levels
    gray = clamp_u8(t * 255)

    return apply_lut_to_gray(gray, lut)

def make_diagonal_aurora_overlay(h, w, lut):
    yy, xx = np.mgrid[0:h, 0:w]
    t = (0.6 * yy / max(1, h - 1)) + (0.4 * xx / max(1, w - 1))
    t += 0.15 * np.sin(xx / max(1, w) * math.pi * random.uniform(1.5, 3.5))
    t = np.clip(t, 0, 1)

    levels = random.choice([5, 6, 7])
    t = np.floor(t * levels) / levels
    gray = clamp_u8(t * 255)

    return apply_lut_to_gray(gray, lut)

def make_radial_halo_overlay(h, w, lut):
    yy, xx = np.mgrid[0:h, 0:w]
    x0 = random.randint(w // 4, 3 * w // 4)
    y0 = random.randint(h // 5, 3 * h // 5)

    dist = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    dist_norm = dist / (dist.max() + 1e-6)

    freq = random.uniform(8.0, 18.0)
    phase = random.uniform(0, math.pi * 2)
    rings = (np.sin(dist_norm * freq * math.pi + phase) + 1.0) / 2.0

    levels = random.choice([4, 5, 6])
    rings = np.floor(rings * levels) / levels
    gray = clamp_u8(rings * 255)

    halo = apply_lut_to_gray(gray, lut)
    return halo, dist_norm

# =========================================================
# PRESET-SPECIFIC FINISHERS
# =========================================================
def apply_preset_finishers(color, arr, sky_mask, subject_mask, preset_name):
    h, w, _ = color.shape

    if preset_name in ["rainbow_church", "spectral_statue", "neon_void"]:
        sky_overlay = make_banded_sky_overlay(h, w, color_lut_for_overlay(preset_name))
        alpha = sky_mask[:, :, None] * random.uniform(0.18, 0.45)
        color = clamp_u8(color.astype(np.float32) * (1 - alpha) + sky_overlay.astype(np.float32) * alpha)

    if preset_name in ["pastel_angel", "infrared_ruin"]:
        aurora = make_diagonal_aurora_overlay(h, w, color_lut_for_overlay(preset_name))
        alpha = sky_mask[:, :, None] * random.uniform(0.12, 0.35)
        color = clamp_u8(color.astype(np.float32) * (1 - alpha) + aurora.astype(np.float32) * alpha)

    if preset_name == "dirty_portrait":
        color = rgb_shift(color, max_shift=random.randint(2, 6))
        color = add_noise(color, strength=random.randint(14, 26))
        color = add_speckle(color, amount=random.uniform(0.01, 0.04))

    if preset_name == "crt_ghost":
        color = add_scanlines(color, strength=random.uniform(0.10, 0.22), spacing=2)
        color = rgb_shift(color, max_shift=random.randint(1, 4))
        color = add_noise(color, strength=random.randint(8, 16))

    if preset_name == "infrared_ruin":
        # brighten foliage-ish / green dominant regions
        r = color[:, :, 0].astype(np.float32)
        g = color[:, :, 1].astype(np.float32)
        b = color[:, :, 2].astype(np.float32)

        foliage = (g > r * 0.9) & (g > b * 1.05)
        color = color.astype(np.float32)
        color[:, :, 0][foliage] = np.clip(color[:, :, 0][foliage] * 1.2 + 30, 0, 255)
        color[:, :, 2][foliage] = np.clip(color[:, :, 2][foliage] * 1.15 + 20, 0, 255)
        color = clamp_u8(color)

    if preset_name == "toxic_thermal":
        color = add_noise(color, strength=random.randint(5, 12))

    return color

def color_lut_for_overlay(preset_name):
    # use sky LUT as overlay basis
    sky_lut, _, _ = build_preset_luts(preset_name)
    return sky_lut

# =========================================================
# MAIN PIPELINE
# =========================================================
def stylize_preset(target_img, preset_name):
    sky_lut, fg_lut, global_lut = build_preset_luts(preset_name)

    # preprocess
    img = target_img.copy()
    img = ImageEnhance.Contrast(img).enhance(random.uniform(1.3, 2.4))
    img = ImageEnhance.Color(img).enhance(random.uniform(1.0, 2.2))
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(1.0, 2.4))

    if preset_name in ["rainbow_church", "spectral_statue", "toxic_thermal"] and random.random() < 0.4:
        img = ImageOps.solarize(img, threshold=random.randint(80, 180))

    arr = np_img(img)
    h, w, _ = arr.shape

    # masks
    sky_mask = estimate_sky_mask(arr)
    subject_mask = estimate_subject_mask(arr, sky_mask)

    # luminance
    gray = grayscale_arr(arr)

    # region-aware posterization
    gray_post = region_aware_posterize(gray, sky_mask, subject_mask, preset_name)

    # region LUT application
    sky_color = apply_lut_to_gray(gray_post, sky_lut).astype(np.float32)
    fg_color = apply_lut_to_gray(gray_post, fg_lut).astype(np.float32)
    global_color = apply_lut_to_gray(gray_post, global_lut).astype(np.float32)

    bg_mask = np.clip(1.0 - np.maximum(sky_mask, subject_mask), 0, 1)
    total = sky_mask + subject_mask + bg_mask + 1e-6

    color = (
        sky_color * (sky_mask[:, :, None] / total[:, :, None]) +
        fg_color * (subject_mask[:, :, None] / total[:, :, None]) +
        global_color * (bg_mask[:, :, None] / total[:, :, None])
    )
    color = clamp_u8(color)

    # edge stack
    edge_map = multi_pass_edge_stack(img)
    color = apply_adaptive_edges(color, edge_map, sky_mask, subject_mask, preset_name)

    # blend a little original detail back
    if preset_name == "pastel_angel":
        color = blend(color, arr, alpha=random.uniform(0.10, 0.22))
    elif preset_name in ["dirty_portrait", "crt_ghost"]:
        color = blend(color, arr, alpha=random.uniform(0.05, 0.16))
    else:
        color = blend(color, arr, alpha=random.uniform(0.08, 0.20))

    # preset-specific finishers
    color = apply_preset_finishers(color, arr, sky_mask, subject_mask, preset_name)

    # mild general finish
    if preset_name not in ["dirty_portrait", "crt_ghost"]:
        if random.random() < 0.45:
            color = rgb_shift(color, max_shift=random.randint(1, 3))
        color = add_noise(color, strength=random.randint(2, 8))

    return pil_img(color)

# =========================================================
# MAIN
# =========================================================
def process_images():
    ensure_output_folder()

    input_files = list_images(INPUT_FOLDER)

    if not input_files:
        print(f"No images found in '{INPUT_FOLDER}'")
        return

    print(f"Found {len(input_files)} image(s). Processing with {len(PRESETS)} presets...")

    for filename in input_files:
        input_path = os.path.join(INPUT_FOLDER, filename)
        base_name, _ = os.path.splitext(filename)

        try:
            img = load_image(input_path)

            for i, preset_name in enumerate(PRESETS, start=1):
                out = stylize_preset(img, preset_name)
                output_filename = f"{base_name}_{preset_name}_v{i}.png"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                save_image(out, output_path)

            print(f"Done: {filename}")

        except Exception as e:
            print(f"Failed: {filename} -> {e}")

    print(f"\nAll done. Saved to '{OUTPUT_FOLDER}'")

if __name__ == "__main__":
    process_images()