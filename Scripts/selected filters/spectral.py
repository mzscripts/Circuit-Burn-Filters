import os
import random
import math
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

# =========================================================
# CONFIG
# =========================================================
INPUT_FOLDER = "wallter"
OUTPUT_FOLDER = "spectral_poster_pack"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

FILTERS = [
    "cobalt_mask",
    "toxic_thermal",
    "melted_halo",
    "rainbow_chapel",
    "graveyard_acid",
    "spectral_void",
    "pastel_angel",
    "neon_church",
    "aurora_statue",
]

# =========================================================
# UTILS
# =========================================================
def ensure_output_folder():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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

def add_noise(arr, strength=10):
    noise = np.random.randint(-strength, strength + 1, arr.shape, dtype=np.int16)
    out = arr.astype(np.int16) + noise
    return clamp_u8(out)

def blend(a, b, alpha=0.5):
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    out = af * (1 - alpha) + bf * alpha
    return clamp_u8(out)

def rgb_shift(arr, max_shift=6):
    out = np.zeros_like(arr)
    for c in range(3):
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        out[:, :, c] = np.roll(np.roll(arr[:, :, c], dy, axis=0), dx, axis=1)
    return out

def posterize_np(arr, bits=3):
    shift = 8 - bits
    return ((arr >> shift) << shift).astype(np.uint8)

def grayscale_arr(img):
    return np.array(ImageOps.grayscale(img))

def add_scanlines(arr, strength=0.14, spacing=2):
    out = arr.astype(np.float32).copy()
    h = out.shape[0]
    for y in range(0, h, spacing):
        out[y, :, :] *= (1.0 - strength)
    return clamp_u8(out)

def contrast_prep(img, contrast=(1.5, 3.0), color=(1.3, 3.0), sharpness=(1.0, 2.8)):
    img = ImageEnhance.Contrast(img).enhance(random.uniform(*contrast))
    img = ImageEnhance.Color(img).enhance(random.uniform(*color))
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(*sharpness))
    return img

# =========================================================
# PALETTES
# =========================================================
PALETTES = {
    "cobalt_mask": [
        (5, 5, 10),
        (20, 30, 120),
        (50, 80, 255),
        (180, 180, 30),
        (240, 240, 80),
    ],
    "toxic_thermal": [
        (0, 0, 120),
        (0, 180, 255),
        (0, 255, 140),
        (255, 255, 0),
        (255, 100, 0),
        (255, 0, 255),
    ],
    "rainbow": [
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 255, 0),
        (255, 0, 0),
        (255, 0, 255),
    ],
    "graveyard": [
        (0, 0, 0),
        (40, 0, 90),
        (0, 120, 255),
        (0, 255, 120),
        (255, 0, 255),
        (255, 255, 0),
    ],
    "pastel": [
        (240, 220, 255),
        (210, 240, 255),
        (255, 240, 220),
        (255, 220, 255),
        (220, 255, 230),
        (255, 255, 255),
    ],
    "neon_dark": [
        (0, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (255, 255, 255),
    ],
    "aurora": [
        (20, 0, 120),
        (120, 0, 255),
        (0, 180, 255),
        (120, 255, 180),
        (255, 0, 255),
        (255, 120, 0),
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
# EDGE / CONTOUR HELPERS
# =========================================================
def edge_mask(img, threshold=70):
    edges = img.filter(ImageFilter.FIND_EDGES).convert("L")
    edges = ImageEnhance.Contrast(edges).enhance(3.0)
    arr = np.array(edges)
    return np.where(arr > threshold, 255, 0).astype(np.uint8)

def apply_black_edges(color_arr, mask, strength=0.8):
    out = color_arr.astype(np.float32)
    mask3 = (mask[:, :, None] / 255.0) * strength
    out = out * (1 - mask3)
    return clamp_u8(out)

def apply_white_edges(color_arr, mask, strength=0.7):
    out = color_arr.astype(np.float32)
    mask3 = (mask[:, :, None] / 255.0) * strength
    out = out * (1 - mask3) + 255 * mask3
    return clamp_u8(out)

def neon_edge_overlay(img):
    g = ImageOps.grayscale(img)
    e = g.filter(ImageFilter.FIND_EDGES)
    e = ImageEnhance.Contrast(e).enhance(3.8)
    arr = np.array(e)

    out = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    out[:, :, 0] = np.clip(arr * random.uniform(0.8, 1.0), 0, 255)
    out[:, :, 1] = np.clip(arr * random.uniform(0.2, 0.9), 0, 255)
    out[:, :, 2] = np.clip(arr * random.uniform(0.8, 1.0), 0, 255)
    return out

# =========================================================
# SKY / HALO / AURORA HELPERS
# =========================================================
def top_region_gradient(h, w, palette, strength=0.65):
    """
    Creates a banded horizontal rainbow/aurora gradient for the upper region.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    # horizontal banding from y + slight x modulation
    t = (yy / max(1, h - 1)) * 0.85 + 0.15 * np.sin(xx / max(1, w) * math.pi * random.uniform(1.5, 4.0))
    t = np.clip(t, 0, 1)

    # quantize to bands
    levels = random.choice([4, 5, 6, 7])
    t = np.floor(t * levels) / levels
    gray = clamp_u8(t * 255)

    grad = false_color_map(gray, palette)
    return grad

def banded_wave_sky_overlay(h, w, palette):
    """
    Banded sky overlay with horizontal waves like rainbow church.
    Creates quantized levels with sinusoidal edge distortion.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    
    # Base vertical gradient with horizontal wave distortion
    t = (yy / max(1, h - 1)) * 0.85 + 0.15 * np.sin(xx / max(1, w) * math.pi * random.uniform(1.5, 4.0))
    t = np.clip(t, 0, 1)
    
    # Quantize into discrete bands
    levels = random.choice([4, 5, 6, 7])
    t = np.floor(t * levels) / levels
    gray = clamp_u8(t * 255)
    
    return false_color_map(gray, palette)

def wave_sky_overlay(h, w, palette):
    """
    Large smooth complete chapel-like wave arches for the upper sky.
    No stripe repetition, no cutout scallops, no weird mask shapes.
    Only shape generation changes; palette controls colors.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    xn = xx.astype(np.float32) / max(1, w - 1)
    yn = yy.astype(np.float32) / max(1, h - 1)

    # 2 big arches across width (matches your reference better)
    cycles = 2
    phase = random.uniform(-0.08, 0.08)

    # main dome curve across width
    arch1 = 0.10 + 0.22 * (0.5 - 0.5 * np.cos((xn * cycles + phase) * math.pi * 2.0))

    # second smaller dome layer underneath
    arch2 = arch1 + random.uniform(0.12, 0.18)

    # build solid dome masks (not repeated stripes)
    outer = (yn < arch1).astype(np.float32)
    inner = (yn < arch2).astype(np.float32)

    # smooth edges near the dome boundaries
    feather1 = random.uniform(0.025, 0.045)
    feather2 = random.uniform(0.02, 0.04)

    outer_soft = np.clip((arch1 - yn) / feather1, 0, 1)
    inner_soft = np.clip((arch2 - yn) / feather2, 0, 1)

    # keep top sky filled so arches feel embedded, not floating cutouts
    top_fill = np.clip(1.0 - yn / random.uniform(0.65, 0.78), 0, 1)

    # layered field: top sky + big dome + smaller dome
    field = np.maximum(top_fill * 0.35, outer_soft * 0.78)
    field = np.maximum(field, inner_soft * 0.52)

    # quantize slightly for posterized synthetic look
    levels = random.choice([4, 5, 6])
    field = np.floor(field * levels) / levels

    gray = clamp_u8(field * 255)
    return false_color_map(gray, palette)
def radial_halo(h, w, palette):
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

def blend_top_region(base_arr, overlay_arr, split_ratio=(0.35, 0.65), alpha=(0.35, 0.75)):
    h = base_arr.shape[0]
    split = random.randint(int(h * split_ratio[0]), int(h * split_ratio[1]))

    out = base_arr.copy().astype(np.float32)
    a = random.uniform(*alpha)

    out[:split] = out[:split] * (1 - a) + overlay_arr[:split].astype(np.float32) * a
    return clamp_u8(out)

def blend_wave_sky(base_arr, overlay_arr, split_ratio, alpha, feather_ratio=0.12):
    h = base_arr.shape[0]
    split = random.randint(int(h * split_ratio[0]), int(h * split_ratio[1]))
    feather = max(1, int(h * feather_ratio))

    yy = np.arange(h, dtype=np.float32)
    mask = np.zeros(h, dtype=np.float32)
    solid_end = max(0, split - feather)
    mask[:solid_end] = 1.0

    if split > solid_end:
        fade = (split - yy[solid_end:split]) / max(1.0, split - solid_end)
        fade = np.clip(fade, 0, 1)
        fade = fade * fade * (3.0 - 2.0 * fade)
        mask[solid_end:split] = fade

    mask3 = mask[:, None, None] * random.uniform(*alpha)
    out = base_arr.astype(np.float32) * (1 - mask3) + overlay_arr.astype(np.float32) * mask3
    return clamp_u8(out)

# =========================================================
# CORE STYLE BUILDERS
# =========================================================
def build_false_color(img, palette_name, poster_bits=(2, 4), blend_original=(0.1, 0.35), black_edges=True, white_edges=False):
    base = contrast_prep(img)

    if random.random() < 0.55:
        base = ImageOps.solarize(base, threshold=random.randint(70, 180))
    if random.random() < 0.18:
        base = ImageOps.invert(base)

    gray = grayscale_arr(base)
    gray = posterize_np(gray, bits=random.choice(list(range(poster_bits[0], poster_bits[1] + 1))))

    color = false_color_map(gray, PALETTES[palette_name])

    orig = np_img(base)
    color = blend(color, orig, alpha=random.uniform(*blend_original))

    mask = edge_mask(base, threshold=random.randint(55, 110))
    if black_edges:
        color = apply_black_edges(color, mask, strength=random.uniform(0.25, 0.85))
    if white_edges:
        color = apply_white_edges(color, mask, strength=random.uniform(0.2, 0.75))

    return color, base

# =========================================================
# FILTER 1: COBALT MASK
# Blue face + yellow background + contour cut
# =========================================================
def filter_cobalt_mask(img):
    color, base = build_false_color(
        img,
        "cobalt_mask",
        poster_bits=(2, 3),
        blend_original=(0.05, 0.18),
        black_edges=True,
        white_edges=False
    )

    # force stronger blue/yellow split feel
    color = color.astype(np.float32)
    color[:, :, 2] *= random.uniform(1.15, 1.5)   # blue up
    color[:, :, 0] *= random.uniform(0.75, 1.05)
    color[:, :, 1] *= random.uniform(0.85, 1.15)
    color = clamp_u8(color)

    color = rgb_shift(color, max_shift=random.randint(1, 4))
    color = add_noise(color, strength=random.randint(4, 12))

    return pil_img(color)

# =========================================================
# FILTER 2: TOXIC THERMAL
# Like your cat sample
# =========================================================
def filter_toxic_thermal(img):
    color, base = build_false_color(
        img,
        "toxic_thermal",
        poster_bits=(2, 3),
        blend_original=(0.05, 0.22),
        black_edges=True,
        white_edges=False
    )

    # timestamp-ish tiny overlay vibe not actual text, just noisy lower-left glow
    h, w, _ = color.shape
    patch_h = max(8, h // 18)
    patch_w = max(18, w // 6)
    patch = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
    patch[:, :, 0] = random.randint(0, 80)
    patch[:, :, 1] = random.randint(120, 255)
    patch[:, :, 2] = random.randint(120, 255)
    color[h-patch_h:h, :patch_w] = blend(color[h-patch_h:h, :patch_w], patch, alpha=random.uniform(0.1, 0.25))

    color = add_noise(color, strength=random.randint(6, 14))
    return pil_img(color)

# =========================================================
# FILTER 3: MELTED HALO
# Face / portrait with stretched vertical melt and soft glow
# =========================================================
def filter_melted_halo(img):
    base = contrast_prep(img, contrast=(1.4, 2.4), color=(1.1, 2.0), sharpness=(1.0, 2.2))
    arr = np_img(base)
    h, w, _ = arr.shape

    # soft bloom
    blur = np_img(base.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.5, 4.0))))
    out = blend(arr, blur, alpha=random.uniform(0.25, 0.45))

    # vertical melt on bright areas
    gray = grayscale_arr(base)
    threshold = random.randint(150, 220)

    step = max(1, w // 220)
    for x in range(0, w, step):
        col = out[:, x].copy()
        gcol = gray[:, x]

        for y in range(h):
            if gcol[y] > threshold and random.random() < 0.08:
                melt_len = random.randint(h // 20, max(h // 6, 12))
                y2 = min(h, y + melt_len)
                src = col[y].copy()

                for yy in range(y, y2):
                    alpha = 1.0 - ((yy - y) / max(1, (y2 - y)))
                    out[yy, x] = np.clip(out[yy, x].astype(np.float32) * (1 - 0.65 * alpha) + src * (0.65 * alpha), 0, 255)

    # warm highlight shift
    out = out.astype(np.float32)
    out[:, :, 0] *= random.uniform(1.05, 1.25)
    out[:, :, 1] *= random.uniform(0.95, 1.15)
    out[:, :, 2] *= random.uniform(0.9, 1.1)
    out = clamp_u8(out)

    out = rgb_shift(out, max_shift=random.randint(1, 4))
    out = add_noise(out, strength=random.randint(3, 8))
    return pil_img(out)

# =========================================================
# FILTER 4: RAINBOW CHAPEL
# Hard rainbow sky + architecture contours
# =========================================================
def filter_rainbow_chapel(img):
    color, base = build_false_color(
        img,
        "graveyard",
        poster_bits=(2, 4),
        blend_original=(0.08, 0.22),
        black_edges=True,
        white_edges=False
    )

    h, w, _ = color.shape
    # Apply banded wave sky overlay with rainbow church sky colors
    rainbow_church_sky = [
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 255, 0),
        (255, 0, 0),
        (255, 0, 255),
    ]
    sky_grad = banded_wave_sky_overlay(h, w, rainbow_church_sky)
    
    # Create sky mask based on height
    yy = np.arange(h, dtype=np.float32)
    sky_mask = np.clip(1.0 - (yy / max(1, h - 1)) * 1.2, 0, 1)[:, None, None]
    
    # Blend with rainbow church intensity
    alpha = sky_mask * random.uniform(0.18, 0.45)
    out = clamp_u8(color.astype(np.float32) * (1 - alpha) + sky_grad.astype(np.float32) * alpha)

    # neon contour overlay
    if random.random() < 0.75:
        neon = neon_edge_overlay(base)
        out = blend(out, neon, alpha=random.uniform(0.12, 0.28))

    out = rgb_shift(out, max_shift=random.randint(1, 5))
    out = add_noise(out, strength=random.randint(4, 10))
    return pil_img(out)

# =========================================================
# FILTER 5: GRAVEYARD ACID
# Statue / monument toxic saturated false-color
# =========================================================
def filter_graveyard_acid(img):
    color, base = build_false_color(
        img,
        "graveyard",
        poster_bits=(2, 3),
        blend_original=(0.05, 0.2),
        black_edges=True,
        white_edges=False
    )

    h, w, _ = color.shape

    out = color

    # hard channel clip for acid vibe
    out = out.astype(np.int16)
    for c in range(3):
        low = random.randint(0, 25)
        high = random.randint(150, 235)
        ch = out[:, :, c]
        ch = np.where(ch < low, 0, ch)
        ch = np.where(ch > high, 255, ch)
        out[:, :, c] = ch

    out = clamp_u8(out)
    out = add_noise(out, strength=random.randint(6, 14))
    return pil_img(out)

# =========================================================
# FILTER 6: SPECTRAL VOID
# Dark cyan silhouette / eerie dream figure
# =========================================================
def filter_spectral_void(img):
    base = contrast_prep(img, contrast=(1.6, 3.0), color=(0.0, 0.3), sharpness=(0.8, 1.8))
    gray = grayscale_arr(base)

    # dark moody cyan mapping
    palette = [
        (0, 0, 0),
        (5, 10, 30),
        (10, 40, 80),
        (30, 120, 160),
        (120, 220, 220),
    ]
    gray = posterize_np(gray, bits=random.choice([3, 4]))
    color = false_color_map(gray, palette)

    h, w, _ = color.shape

    # soft vertical blur / dream smear
    blur = np_img(pil_img(color).filter(ImageFilter.GaussianBlur(radius=random.uniform(1.2, 3.0))))
    out = blend(color, blur, alpha=random.uniform(0.25, 0.45))

    # subtle horizontal fog bands
    for _ in range(random.randint(3, 8)):
        y = random.randint(0, h - 1)
        bh = random.randint(4, max(8, h // 14))
        y2 = min(h, y + bh)
        band = out[y:y2].copy().astype(np.float32)
        band[:, :, 1] *= random.uniform(0.9, 1.2)
        band[:, :, 2] *= random.uniform(1.05, 1.3)
        out[y:y2] = clamp_u8(band)

    out = add_scanlines(out, strength=random.uniform(0.06, 0.14), spacing=random.choice([2, 3]))
    out = add_noise(out, strength=random.randint(4, 10))
    return pil_img(out)

# =========================================================
# FILTER 7: PASTEL ANGEL
# High-key soft dreamy statue / pastel spectral
# =========================================================
def filter_pastel_angel(img):
    base = contrast_prep(img, contrast=(1.2, 2.0), color=(0.8, 1.4), sharpness=(0.8, 1.6))

    # brighten heavily
    arr = np_img(base).astype(np.float32)
    arr = np.clip(arr * random.uniform(1.15, 1.45) + random.uniform(10, 35), 0, 255).astype(np.uint8)

    gray = np.array(ImageOps.grayscale(pil_img(arr)))
    gray = posterize_np(gray, bits=random.choice([3, 4, 5]))

    pastel = false_color_map(gray, PALETTES["pastel"])
    out = blend(pastel, arr, alpha=random.uniform(0.15, 0.35))

    # soft sky bands
    h, w, _ = out.shape
    sky = top_region_gradient(h, w, PALETTES["pastel"], strength=0.5)
    out = blend_top_region(out, sky, split_ratio=(0.3, 0.55), alpha=(0.15, 0.35))

    # white edges
    mask = edge_mask(base, threshold=random.randint(70, 120))
    out = apply_white_edges(out, mask, strength=random.uniform(0.15, 0.45))

    out = add_noise(out, strength=random.randint(2, 6))
    return pil_img(out)

# =========================================================
# FILTER 8: NEON CHURCH
# Dark silhouette church + magenta/cyan sky + strong black base
# =========================================================
def filter_neon_church(img):
    base = contrast_prep(img, contrast=(1.8, 3.2), color=(1.2, 2.4), sharpness=(1.2, 2.6))
    arr = np_img(base)

    h, w, _ = arr.shape
    out = arr.astype(np.float32)
    out = clamp_u8(out)

    # neon upper sky
    sky = wave_sky_overlay(h, w, PALETTES["aurora"])
    out = blend_wave_sky(out, sky, split_ratio=(0.28, 0.45), alpha=(0.35, 0.65))
    # strong contour
    mask = edge_mask(base, threshold=random.randint(55, 95))
    out = apply_black_edges(out, mask, strength=random.uniform(0.35, 0.85))

    # scanline grain
    out = add_scanlines(out, strength=random.uniform(0.08, 0.18), spacing=2)
    out = add_noise(out, strength=random.randint(6, 14))

    return pil_img(out)

# =========================================================
# FILTER 9: AURORA STATUE
# Smooth rainbow arc sky + statue highlight
# =========================================================
def filter_aurora_statue(img):
    color, base = build_false_color(
        img,
        "aurora",
        poster_bits=(3, 5),
        blend_original=(0.1, 0.28),
        black_edges=False,
        white_edges=True
    )

    h, w, _ = color.shape

    # aurora arcs (diagonal smooth bands)
    yy, xx = np.mgrid[0:h, 0:w]
    t = (0.6 * yy / max(1, h - 1)) + (0.4 * xx / max(1, w - 1))
    t += 0.15 * np.sin(xx / max(1, w) * math.pi * random.uniform(1.5, 3.0))
    t = np.clip(t, 0, 1)

    levels = random.choice([5, 6, 7])
    t = np.floor(t * levels) / levels
    gray = clamp_u8(t * 255)
    aurora = false_color_map(gray, PALETTES["aurora"])

    out = blend(color, aurora, alpha=random.uniform(0.18, 0.42))

    out = rgb_shift(out, max_shift=random.randint(1, 4))
    out = add_noise(out, strength=random.randint(3, 8))
    return pil_img(out)

# =========================================================
# DISPATCH
# =========================================================
FILTER_FUNCTIONS = {
    "cobalt_mask": filter_cobalt_mask,
    "toxic_thermal": filter_toxic_thermal,
    "melted_halo": filter_melted_halo,
    "rainbow_chapel": filter_rainbow_chapel,
    "graveyard_acid": filter_graveyard_acid,
    "spectral_void": filter_spectral_void,
    "pastel_angel": filter_pastel_angel,
    "neon_church": filter_neon_church,
    "aurora_statue": filter_aurora_statue,
}

def apply_filter(img, filter_name):
    return FILTER_FUNCTIONS[filter_name](img)

# =========================================================
# MAIN
# =========================================================
def process_images():
    ensure_output_folder()

    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    if not files:
        print(f"No images found in '{INPUT_FOLDER}'")
        return

    print(f"Found {len(files)} image(s). Processing with {len(FILTERS)} filters...")

    for filename in files:
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
