import os
import random
import math
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

# =========================================================
# CONFIG
# =========================================================
INPUT_FOLDER = "wallter"
OUTPUT_FOLDER = "haunted_chroma_pack"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

FILTERS = [
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
]

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
    arr = img_or_arr.astype(np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return clamp_u8(gray)

def blend(a, b, alpha=0.5):
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    out = af * (1 - alpha) + bf * alpha
    return clamp_u8(out)

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

def add_scanlines(arr, strength=0.12, spacing=2):
    out = arr.astype(np.float32).copy()
    h = out.shape[0]
    for y in range(0, h, spacing):
        out[y, :, :] *= (1.0 - strength)
    return clamp_u8(out)

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

def contrast_prep(img, contrast=(1.4, 2.6), color=(1.0, 2.4), sharpness=(1.0, 2.6)):
    img = ImageEnhance.Contrast(img).enhance(random.uniform(*contrast))
    img = ImageEnhance.Color(img).enhance(random.uniform(*color))
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(*sharpness))
    return img

# =========================================================
# LUT / PALETTE HELPERS
# =========================================================
PALETTES = {
    "void": [
        (0, 0, 0),
        (0, 30, 50),
        (0, 140, 200),
        (180, 240, 255),
        (255, 255, 255),
    ],
    "solar": [
        (10, 0, 0),
        (90, 10, 0),
        (180, 40, 0),
        (255, 120, 0),
        (255, 220, 60),
        (255, 255, 220),
    ],
    "orchid": [
        (10, 0, 30),
        (60, 0, 100),
        (120, 40, 180),
        (200, 120, 255),
        (255, 220, 255),
    ],
    "plasma": [
        (0, 0, 40),
        (0, 120, 255),
        (0, 255, 255),
        (0, 255, 120),
        (255, 0, 255),
        (255, 255, 0),
    ],
    "ghost": [
        (255, 255, 255),
        (220, 240, 255),
        (180, 220, 255),
        (80, 120, 180),
        (20, 20, 40),
        (0, 0, 0),
    ],
    "moon": [
        (0, 0, 20),
        (20, 40, 80),
        (60, 120, 180),
        (180, 220, 255),
        (255, 255, 255),
    ],
    "funeral": [
        (0, 0, 0),
        (50, 0, 0),
        (120, 20, 0),
        (200, 60, 0),
        (255, 160, 0),
        (255, 240, 120),
    ],
    "holo": [
        (0, 0, 0),
        (0, 80, 120),
        (0, 220, 220),
        (160, 80, 255),
        (255, 180, 255),
        (255, 255, 255),
    ],
    "bleach": [
        (150, 150, 180),
        (200, 210, 240),
        (230, 240, 255),
        (255, 240, 245),
        (255, 255, 255),
    ],
}

def false_color_map(gray_arr, palette):
    h, w = gray_arr.shape
    norm = gray_arr.astype(np.float32) / 255.0
    segments = len(palette) - 1

    scaled = norm * segments
    idx = np.floor(scaled).astype(np.int32)
    frac = scaled - idx
    idx = np.clip(idx, 0, segments - 1)

    p1 = np.array([palette[i] for i in idx.flatten()], dtype=np.float32).reshape(h, w, 3)
    p2 = np.array([palette[i + 1] for i in idx.flatten()], dtype=np.float32).reshape(h, w, 3)

    frac3 = frac[:, :, None]
    out = p1 * (1 - frac3) + p2 * frac3
    return clamp_u8(out)

# =========================================================
# MASKS / EDGES
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

    sky_mask = (sky_score > 0.55).astype(np.float32)
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

    subject_mask = (fg_score > 0.48).astype(np.float32)
    subject_mask = box_blur_mask(subject_mask, radius=4)

    return np.clip(subject_mask, 0, 1)

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

def apply_black_edges(color_arr, edge_map, subject_mask=None, sky_mask=None, s_w=0.85, b_w=0.45, k_w=0.08):
    edge = edge_map.astype(np.float32) / 255.0

    if subject_mask is None or sky_mask is None:
        edge3 = edge[:, :, None] * 0.7
        out = color_arr.astype(np.float32) * (1.0 - edge3)
        return clamp_u8(out)

    bg_mask = np.clip(1.0 - np.maximum(subject_mask, sky_mask), 0, 1)
    weight = subject_mask * s_w + bg_mask * b_w + sky_mask * k_w
    edge3 = (edge * weight)[:, :, None]

    out = color_arr.astype(np.float32)
    out = out * (1.0 - edge3)
    return clamp_u8(out)

def apply_white_edges(color_arr, edge_map, amount=0.35):
    edge = edge_map.astype(np.float32) / 255.0
    edge3 = edge[:, :, None] * amount
    out = color_arr.astype(np.float32)
    out = out * (1.0 - edge3) + 255 * edge3
    return clamp_u8(out)

# =========================================================
# OVERLAYS
# =========================================================
def make_banded_overlay(h, w, palette):
    yy, xx = np.mgrid[0:h, 0:w]
    t = (yy / max(1, h - 1)) * 0.85 + 0.15 * np.sin(xx / max(1, w) * math.pi * random.uniform(1.5, 4.0))
    t = np.clip(t, 0, 1)
    levels = random.choice([4, 5, 6, 7])
    t = np.floor(t * levels) / levels
    gray = clamp_u8(t * 255)
    return false_color_map(gray, palette)

def make_diagonal_overlay(h, w, palette):
    yy, xx = np.mgrid[0:h, 0:w]
    t = (0.6 * yy / max(1, h - 1)) + (0.4 * xx / max(1, w - 1))
    t += 0.15 * np.sin(xx / max(1, w) * math.pi * random.uniform(1.5, 3.5))
    t = np.clip(t, 0, 1)
    levels = random.choice([5, 6, 7])
    t = np.floor(t * levels) / levels
    gray = clamp_u8(t * 255)
    return false_color_map(gray, palette)

def make_radial_halo(h, w, palette):
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

    return false_color_map(gray, palette), dist_norm

def neon_edge_overlay(img, tint=(255, 0, 255)):
    g = ImageOps.grayscale(img)
    e = g.filter(ImageFilter.FIND_EDGES)
    e = ImageEnhance.Contrast(e).enhance(4.0)
    arr = np.array(e).astype(np.float32) / 255.0

    out = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    out[:, :, 0] = np.clip(arr * tint[0], 0, 255)
    out[:, :, 1] = np.clip(arr * tint[1], 0, 255)
    out[:, :, 2] = np.clip(arr * tint[2], 0, 255)
    return out

# =========================================================
# STYLE 1: VOID STENCIL
# Hard black cutout + cyan/white contour
# =========================================================
def filter_void_stencil(img):
    base = contrast_prep(img, contrast=(1.8, 3.2), color=(0.0, 0.2), sharpness=(1.2, 2.8))
    arr = np_img(base)
    gray = grayscale_arr(arr)

    sky_mask = estimate_sky_mask(arr)
    subject_mask = estimate_subject_mask(arr, sky_mask)

    # hard subject silhouette
    thresh = random.randint(70, 140)
    sil = (gray > thresh).astype(np.float32)

    # combine with subject mask for stronger central cutout
    subject_core = np.clip(0.55 * subject_mask + 0.45 * sil, 0, 1)
    subject_core = box_blur_mask(subject_core, radius=2)

    h, w = gray.shape
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:, :, 2] = np.clip(gray * 0.25, 0, 255).astype(np.uint8)
    bg[:, :, 1] = np.clip(gray * 0.15, 0, 255).astype(np.uint8)

    fg = false_color_map(posterize_np(gray, bits=3), PALETTES["void"])
    out = clamp_u8(bg.astype(np.float32) * (1 - subject_core[:, :, None]) + fg.astype(np.float32) * subject_core[:, :, None])

    edge = multi_pass_edge_stack(base)
    out = apply_white_edges(out, edge, amount=random.uniform(0.18, 0.32))
    out = rgb_shift(out, max_shift=random.randint(1, 3))
    out = add_noise(out, strength=random.randint(3, 8))
    return pil_img(out)

# =========================================================
# STYLE 2: SOLAR SAINT
# Gold/orange sacred icon halo
# =========================================================
def filter_solar_saint(img):
    base = contrast_prep(img, contrast=(1.5, 2.8), color=(1.0, 1.8), sharpness=(1.0, 2.0))
    arr = np_img(base)
    gray = grayscale_arr(arr)

    sky_mask = estimate_sky_mask(arr)
    subject_mask = estimate_subject_mask(arr, sky_mask)

    gray = posterize_np(gray, bits=random.choice([3, 4]))
    color = false_color_map(gray, PALETTES["solar"])

    h, w, _ = color.shape

    # sacred radial halo centered around subject-ish area
    halo, dist_norm = make_radial_halo(h, w, PALETTES["solar"])
    alpha = np.clip(0.55 - dist_norm, 0, 0.35)[:, :, None]
    color = clamp_u8(color.astype(np.float32) * (1 - alpha) + halo.astype(np.float32) * alpha)

    # darken edges of frame slightly for icon effect
    yy, xx = np.mgrid[0:h, 0:w]
    vignette = 1.0 - 0.35 * np.clip(
        np.sqrt(((xx - w/2)/(w/2))**2 + ((yy - h/2)/(h/2))**2),
        0, 1
    )
    color = clamp_u8(color.astype(np.float32) * vignette[:, :, None])

    edge = multi_pass_edge_stack(base)
    color = apply_black_edges(color, edge, subject_mask, sky_mask, s_w=0.65, b_w=0.35, k_w=0.05)
    color = blend(color, arr, alpha=random.uniform(0.06, 0.14))
    color = add_noise(color, strength=random.randint(2, 6))
    return pil_img(color)

# =========================================================
# STYLE 3: XRAY ORCHID
# Purple x-ray spectral negative
# =========================================================
def filter_xray_orchid(img):
    base = contrast_prep(img, contrast=(1.4, 2.4), color=(0.0, 0.2), sharpness=(1.0, 2.2))
    arr = np_img(base)
    gray = grayscale_arr(arr)

    # invert luminance for x-ray feel
    inv_gray = 255 - gray
    inv_gray = posterize_np(inv_gray, bits=random.choice([3, 4]))

    color = false_color_map(inv_gray, PALETTES["orchid"])

    # soft blur bloom
    blur = np_img(pil_img(color).filter(ImageFilter.GaussianBlur(radius=random.uniform(1.5, 3.5))))
    color = blend(color, blur, alpha=random.uniform(0.18, 0.35))

    # slight cyan-magenta edge tint
    neon = neon_edge_overlay(base, tint=random.choice([(180, 80, 255), (120, 220, 255)]))
    color = blend(color, neon, alpha=random.uniform(0.08, 0.18))

    edge = multi_pass_edge_stack(base)
    color = apply_white_edges(color, edge, amount=random.uniform(0.10, 0.22))
    color = add_noise(color, strength=random.randint(2, 6))
    return pil_img(color)

# =========================================================
# STYLE 4: PLASMA RUINS
# Electric architecture / ruin energy
# =========================================================
def filter_plasma_ruins(img):
    base = contrast_prep(img, contrast=(1.6, 3.0), color=(1.3, 2.8), sharpness=(1.2, 3.0))
    arr = np_img(base)
    gray = grayscale_arr(arr)

    sky_mask = estimate_sky_mask(arr)
    subject_mask = estimate_subject_mask(arr, sky_mask)

    gray = posterize_np(gray, bits=random.choice([2, 3]))
    color = false_color_map(gray, PALETTES["plasma"])

    h, w, _ = color.shape
    sky_overlay = make_banded_overlay(h, w, PALETTES["plasma"])
    alpha = sky_mask[:, :, None] * random.uniform(0.18, 0.42)
    color = clamp_u8(color.astype(np.float32) * (1 - alpha) + sky_overlay.astype(np.float32) * alpha)

    edge = multi_pass_edge_stack(base)
    color = apply_black_edges(color, edge, subject_mask, sky_mask, s_w=0.92, b_w=0.58, k_w=0.08)

    # electric edge glow
    neon = neon_edge_overlay(base, tint=random.choice([(255, 0, 255), (0, 255, 255), (255, 255, 0)]))
    color = blend(color, neon, alpha=random.uniform(0.08, 0.18))

    color = rgb_shift(color, max_shift=random.randint(1, 4))
    color = add_noise(color, strength=random.randint(4, 10))
    return pil_img(color)

# =========================================================
# STYLE 5: GHOST NEGATIVE
# Inverted spectral photo but cursed / photographic
# =========================================================
def filter_ghost_negative(img):
    base = contrast_prep(img, contrast=(1.2, 2.0), color=(0.8, 1.6), sharpness=(1.0, 2.0))
    arr = np_img(base)

    # partial invert with luminance remap
    inv = 255 - arr
    gray = grayscale_arr(inv)
    gray = posterize_np(gray, bits=random.choice([3, 4, 5]))
    color = false_color_map(gray, PALETTES["ghost"])

    # blend some inverted original back for photo realism
    color = blend(color, inv, alpha=random.uniform(0.10, 0.25))

    # slight dreamy blur
    blur = np_img(pil_img(color).filter(ImageFilter.GaussianBlur(radius=random.uniform(0.8, 2.5))))
    color = blend(color, blur, alpha=random.uniform(0.08, 0.20))

    edge = multi_pass_edge_stack(base)
    color = apply_black_edges(color, edge, s_w=0.75, b_w=0.40, k_w=0.10)
    color = add_noise(color, strength=random.randint(3, 8))
    return pil_img(color)

# =========================================================
# STYLE 6: MOONLIT RELIC
# Cold silver-blue elegant spectral
# =========================================================
def filter_moonlit_relic(img):
    base = contrast_prep(img, contrast=(1.3, 2.4), color=(0.0, 0.25), sharpness=(1.0, 2.2))
    arr = np_img(base)
    gray = grayscale_arr(arr)

    sky_mask = estimate_sky_mask(arr)
    subject_mask = estimate_subject_mask(arr, sky_mask)

    gray = posterize_np(gray, bits=random.choice([4, 5]))
    color = false_color_map(gray, PALETTES["moon"])

    # faint diagonal moon aura in sky
    h, w, _ = color.shape
    diag = make_diagonal_overlay(h, w, PALETTES["moon"])
    alpha = sky_mask[:, :, None] * random.uniform(0.08, 0.20)
    color = clamp_u8(color.astype(np.float32) * (1 - alpha) + diag.astype(np.float32) * alpha)

    # subtle white contour for marble feel
    edge = multi_pass_edge_stack(base)
    color = apply_black_edges(color, edge, subject_mask, sky_mask, s_w=0.55, b_w=0.25, k_w=0.05)
    color = apply_white_edges(color, edge, amount=random.uniform(0.06, 0.14))

    color = blend(color, arr, alpha=random.uniform(0.08, 0.16))
    color = add_noise(color, strength=random.randint(2, 5))
    return pil_img(color)

# =========================================================
# STYLE 7: PRISM BLOOM
# Chromatic bloom / lens-split / spectral flare
# =========================================================
def filter_prism_bloom(img):
    base = contrast_prep(img, contrast=(1.4, 2.4), color=(1.2, 2.4), sharpness=(1.0, 2.2))
    arr = np_img(base)

    # split blur by channels
    r = Image.fromarray(arr[:, :, 0]).filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 3.0)))
    g = Image.fromarray(arr[:, :, 1]).filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 3.0)))
    b = Image.fromarray(arr[:, :, 2]).filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 3.0)))

    r = np.array(r)
    g = np.array(g)
    b = np.array(b)

    split = np.stack([r, g, b], axis=2)
    split = rgb_shift(split, max_shift=random.randint(2, 6))

    gray = grayscale_arr(split)
    gray = posterize_np(gray, bits=random.choice([3, 4]))
    color = false_color_map(gray, PALETTES["plasma"])

    # bloom blend
    blur = np_img(pil_img(color).filter(ImageFilter.GaussianBlur(radius=random.uniform(2.0, 4.5))))
    color = blend(color, blur, alpha=random.uniform(0.18, 0.35))

    # bring original detail a bit
    color = blend(color, arr, alpha=random.uniform(0.08, 0.18))

    edge = multi_pass_edge_stack(base)
    color = apply_white_edges(color, edge, amount=random.uniform(0.08, 0.18))
    color = add_noise(color, strength=random.randint(3, 8))
    return pil_img(color)

# =========================================================
# STYLE 8: FUNERAL SUN
# Black silhouette + blood-orange sky
# =========================================================
def filter_funeral_sun(img):
    base = contrast_prep(img, contrast=(1.7, 3.0), color=(1.0, 1.8), sharpness=(1.0, 2.2))
    arr = np_img(base)
    gray = grayscale_arr(arr)

    sky_mask = estimate_sky_mask(arr)
    subject_mask = estimate_subject_mask(arr, sky_mask)

    # silhouette lower regions / structures
    color = false_color_map(posterize_np(gray, bits=3), PALETTES["funeral"])

    h, w, _ = color.shape
    sky = make_banded_overlay(h, w, PALETTES["funeral"])
    alpha = sky_mask[:, :, None] * random.uniform(0.28, 0.55)
    color = clamp_u8(color.astype(np.float32) * (1 - alpha) + sky.astype(np.float32) * alpha)

    # make subject + bg structure darker, almost silhouette
    bg_mask = np.clip(1.0 - sky_mask, 0, 1)
    darken = bg_mask[:, :, None] * random.uniform(0.45, 0.72)
    color = clamp_u8(color.astype(np.float32) * (1 - darken))

    edge = multi_pass_edge_stack(base)
    color = apply_black_edges(color, edge, subject_mask, sky_mask, s_w=0.95, b_w=0.70, k_w=0.04)

    color = add_noise(color, strength=random.randint(3, 8))
    return pil_img(color)

# =========================================================
# STYLE 9: HOLOGRAM GRAVE
# Teal-purple holographic graveyard energy
# =========================================================
def filter_hologram_grave(img):
    base = contrast_prep(img, contrast=(1.5, 2.6), color=(1.1, 2.2), sharpness=(1.1, 2.4))
    arr = np_img(base)
    gray = grayscale_arr(arr)

    sky_mask = estimate_sky_mask(arr)
    subject_mask = estimate_subject_mask(arr, sky_mask)

    gray = posterize_np(gray, bits=random.choice([3, 4]))
    color = false_color_map(gray, PALETTES["holo"])

    h, w, _ = color.shape
    diag = make_diagonal_overlay(h, w, PALETTES["holo"])
    alpha = sky_mask[:, :, None] * random.uniform(0.12, 0.28)
    color = clamp_u8(color.astype(np.float32) * (1 - alpha) + diag.astype(np.float32) * alpha)

    # holographic edge aura
    neon = neon_edge_overlay(base, tint=random.choice([(0, 255, 255), (180, 80, 255)]))
    blur = np_img(pil_img(neon).filter(ImageFilter.GaussianBlur(radius=random.uniform(1.5, 3.5))))
    aura = blend(neon, blur, alpha=random.uniform(0.25, 0.45))
    color = blend(color, aura, alpha=random.uniform(0.08, 0.18))

    edge = multi_pass_edge_stack(base)
    color = apply_black_edges(color, edge, subject_mask, sky_mask, s_w=0.72, b_w=0.38, k_w=0.06)

    color = rgb_shift(color, max_shift=random.randint(1, 3))
    color = add_noise(color, strength=random.randint(3, 8))
    return pil_img(color)

# =========================================================
# STYLE 10: BLEACHED ORACLE
# Overexposed saintly pastel spectral
# =========================================================
# =========================================================
# STYLE 10: BLEACHED ORACLE
# Premium ethereal "heaven static / ghost alley": readable overexposed 
# with selective bloom, highlight compression, topographic glow
# =========================================================
def filter_bleached_oracle(img):
    """
    Ethereal spiritual filter: preserved overexposed heaven aesthetic without clipping.
    Features:
    - Highlight compression before bloom (prevents blown-out whites)
    - Selective luminance-thresholded bloom only on bright areas
    - Shadow lift + reduced saturation for ethereal feel
    - Icy cyan/silver tinting on highlights instead of pure white
    - Edge-preserving local contrast recovery
    - Soft mist overlays at low opacity
    - Top-heavy glow gradient with darker lower regions for depth
    - Readable architecture edges + ghostly atmosphere
    """
    h_pil, w_pil = img.size[1], img.size[0]
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1. BASE PREP: gentle sharpening, preserve detail
    # ─────────────────────────────────────────────────────────────────────────
    base = ImageEnhance.Sharpness(img).enhance(random.uniform(1.0, 1.3))
    base_arr = np_img(base).astype(np.float32)
    
    h, w = base_arr.shape[:2]
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2. EXTRACT LUMINANCE for analysis & thresholding
    # ─────────────────────────────────────────────────────────────────────────
    gray = grayscale_arr(base_arr.astype(np.uint8))
    gray_norm = gray.astype(np.float32) / 255.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3. HIGHLIGHT COMPRESSION: tone-map bright areas before any effect
    # ─────────────────────────────────────────────────────────────────────────
    # Apply gentle highlight compression using curve
    compress_curve = np.where(
        gray_norm > 0.65,
        0.65 + (gray_norm - 0.65) ** 1.3 * 0.35,  # Compress highlights
        gray_norm
    )
    compress_curve = np.clip(compress_curve, 0, 1)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4. CONTROLLED EXPOSURE LIFT (not aggressive blowout)
    # ─────────────────────────────────────────────────────────────────────────
    arr = base_arr.copy()
    
    # Lift shadows gently
    shadow_lift = random.uniform(2, 8)
    arr = np.clip(arr + shadow_lift, 0, 255)
    
    # Moderate overall lift
    lift_factor = random.uniform(0.96, 1.02)
    arr = np.clip(arr * lift_factor, 0, 255)
    
    # Re-extract and apply compression
    gray_lifted = grayscale_arr(arr.astype(np.uint8))
    gray_lifted_norm = gray_lifted.astype(np.float32) / 255.0
    compress_curve_lifted = np.where(
        gray_lifted_norm > 0.65,
        0.65 + (gray_lifted_norm - 0.65) ** 1.3 * 0.35,
        gray_lifted_norm
    )
    
    # Apply compression curve back to image
    scale_factor = np.where(gray_lifted_norm > 1e-6, compress_curve_lifted / (gray_lifted_norm + 1e-9), 1.0)
    arr[:,:,0] = arr[:,:,0] * scale_factor
    arr[:,:,1] = arr[:,:,1] * scale_factor
    arr[:,:,2] = arr[:,:,2] * scale_factor
    arr = np.clip(arr, 0, 255)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 5. COLOR TINTING: icy cyan/silver on highlights, warm shadows
    # ─────────────────────────────────────────────────────────────────────────
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    gray_final = grayscale_arr(arr.astype(np.uint8))
    gray_final_norm = gray_final.astype(np.float32) / 255.0
    
    # Shadow tinting: warm + slightly desaturated
    shadow_mask = (gray_final_norm < 0.35).astype(np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0] + shadow_mask * 8, 0, 255)   # Warm red
    arr[:,:,1] = np.clip(arr[:,:,1] + shadow_mask * 4, 0, 255)   # Warm green
    arr[:,:,2] = np.clip(arr[:,:,2] * (1 - shadow_mask * 0.05), 0, 255)  # Less blue
    
    # Midtone: neutral pearl
    midtone_mask = ((gray_final_norm >= 0.35) & (gray_final_norm < 0.70)).astype(np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0] * (1 - midtone_mask * 0.02), 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2] * (1 + midtone_mask * 0.08), 0, 255)  # Slight cool tint
    
    # Highlight tinting: icy cyan/silver (cool + desaturated)
    highlight_mask = (gray_final_norm >= 0.70).astype(np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0] * (1 - highlight_mask * 0.08), 0, 255)   # Reduce red
    arr[:,:,2] = np.clip(arr[:,:,2] + highlight_mask * 15, 0, 255)  # Boost cyan/blue
    # Desaturate highlights
    gray_hl = grayscale_arr(arr.astype(np.uint8))
    arr[:,:,0] = np.clip(arr[:,:,0] * (1 - highlight_mask * 0.15) + gray_hl * highlight_mask * 0.15, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1] * (1 - highlight_mask * 0.15) + gray_hl * highlight_mask * 0.15, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2] * (1 - highlight_mask * 0.10) + gray_hl * highlight_mask * 0.10, 0, 255)
    
    arr = np.clip(arr, 0, 255)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 6. LOCAL CONTRAST RECOVERY: preserve edges and architecture
    # ─────────────────────────────────────────────────────────────────────────
    arr_pil = pil_img(arr)
    blurred = np_img(arr_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(4, 6))))
    
    # High-pass filter for local contrast
    high_pass = arr.astype(np.float32) - blurred.astype(np.float32) + 128
    
    # Apply subtle local contrast boost
    contrast_strength = random.uniform(0.15, 0.25)
    arr = np.clip(arr.astype(np.float32) + (high_pass - 128) * contrast_strength * 0.6, 0, 255)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 7. SELECTIVE BLOOM: only on luminance-thresholded highlights
    # ─────────────────────────────────────────────────────────────────────────
    gray_final2 = grayscale_arr(arr.astype(np.uint8))
    bloom_threshold = random.uniform(0.78, 0.88)
    bloom_mask = (gray_final2.astype(np.float32) / 255.0 > bloom_threshold).astype(np.float32)
    bloom_mask = box_blur_mask(bloom_mask, radius=2)
    
    # Limited bloom
    bloomed = np_img(pil_img(arr).filter(ImageFilter.GaussianBlur(radius=random.uniform(2.5, 4.0))))
    bloom_strength = random.uniform(0.03, 0.07)
    arr = np.clip(
        arr * (1 - bloom_mask[:,:,np.newaxis] * bloom_strength) +
        bloomed * (bloom_mask[:,:,np.newaxis] * bloom_strength),
        0, 255
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # 8. SOFT MIST OVERLAY: very low opacity, top-heavy
    # ─────────────────────────────────────────────────────────────────────────
    yy, xx = np.meshgrid(np.arange(w), np.arange(h))
    
    # Top-heavy glow gradient
    top_bias = 1.0 - (yy.astype(np.float32) / max(1, h - 1))
    top_bias = np.power(top_bias, 1.2)  # Make it more concentrated at top
    
    # Bottom stays darker for depth
    bottom_darken = yy.astype(np.float32) / max(1, h - 1)
    bottom_darken = np.clip(bottom_darken, 0, 1)
    
    # Create mist color: pale blue-cyan-white
    mist_color = np.array([240, 245, 255])
    mist_strength = random.uniform(0.01, 0.03)
    mist = mist_color[np.newaxis,np.newaxis,:] * (top_bias[:,:,np.newaxis] * mist_strength)
    arr = np.clip(arr.astype(np.float32) + mist, 0, 255)
    
    # Apply slight darkening to bottom for depth
    bottom_strength = random.uniform(0.10, 0.18)
    arr = np.clip(arr.astype(np.float32) * (1.0 - bottom_darken[:,:,np.newaxis] * bottom_strength), 0, 255)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 9. EDGE DETECTION & FAINT DEFINITION: prevent mushy look
    # ─────────────────────────────────────────────────────────────────────────
    e = threshold_edges(pil_img(arr.astype(np.uint8)).convert("L"), threshold=random.randint(18, 36)) / 255.0
    edge_boost = e[:,:,np.newaxis] * random.uniform(0.04, 0.08)
    arr = np.clip(arr.astype(np.float32) + edge_boost * 20, 0, 255)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 10. DESATURATION for ghostly ethereal feel
    # ─────────────────────────────────────────────────────────────────────────
    gray_desaturate = grayscale_arr(arr.astype(np.uint8))
    desaturate_amount = random.uniform(0.15, 0.25)
    arr[:,:,0] = arr[:,:,0] * (1 - desaturate_amount) + gray_desaturate * desaturate_amount
    arr[:,:,1] = arr[:,:,1] * (1 - desaturate_amount) + gray_desaturate * desaturate_amount
    arr[:,:,2] = arr[:,:,2] * (1 - desaturate_amount) + gray_desaturate * desaturate_amount
    
    # ─────────────────────────────────────────────────────────────────────────
    # 11. FINE GRAIN: subtle texture
    # ─────────────────────────────────────────────────────────────────────────
    grain = np.random.normal(0, random.uniform(2.0, 3.5), arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + grain, 0, 255).astype(np.float32)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 12. FINAL CINEMATIC POLISH
    # ─────────────────────────────────────────────────────────────────────────
    arr = 255.0 - arr
    arr *= np.array([0.92, 0.96, 1.0], dtype=np.float32)
    arr = np.clip(arr * 0.82, 0, 255)

    result = pil_img(arr)
    result = ImageEnhance.Contrast(result).enhance(random.uniform(1.10, 1.22))   # Keep inverted edges readable
    result = ImageEnhance.Color(result).enhance(random.uniform(0.68, 0.82))      # Keep the palette ghostly
    result = ImageEnhance.Brightness(result).enhance(random.uniform(0.86, 0.94)) # Reduce whiteness
    
    return result

# =========================================================
# DISPATCH
# =========================================================
FILTER_FUNCTIONS = {
    "void_stencil": filter_void_stencil,
    "solar_saint": filter_solar_saint,
    "xray_orchid": filter_xray_orchid,
    "plasma_ruins": filter_plasma_ruins,
    "ghost_negative": filter_ghost_negative,
    "moonlit_relic": filter_moonlit_relic,
    "prism_bloom": filter_prism_bloom,
    "funeral_sun": filter_funeral_sun,
    "hologram_grave": filter_hologram_grave,
    "bleached_oracle": filter_bleached_oracle,
}

def apply_filter(img, filter_name):
    return FILTER_FUNCTIONS[filter_name](img)

# =========================================================
# MAIN
# =========================================================
def process_images():
    ensure_output_folder()

    input_files = list_images(INPUT_FOLDER)

    if not input_files:
        print(f"No images found in '{INPUT_FOLDER}'")
        return

    print(f"Found {len(input_files)} image(s). Processing with {len(FILTERS)} styles...")

    for filename in input_files:
        input_path = os.path.join(INPUT_FOLDER, filename)
        base_name, _ = os.path.splitext(filename)

        try:
            img = load_image(input_path)

            for i, filter_name in enumerate(FILTERS, start=1):
                out = apply_filter(img, filter_name)
                output_filename = f"{base_name}_{filter_name}_v{i}.png"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                save_image(out, output_path)

            print(f"Done: {filename}")

        except Exception as e:
            print(f"Failed: {filename} -> {e}")

    print(f"\nAll done. Saved to '{OUTPUT_FOLDER}'")

if __name__ == "__main__":
    process_images()
