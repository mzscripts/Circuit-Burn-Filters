import os
import random
import math
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageFont

# =========================================================
# CONFIG
# =========================================================
INPUT_FOLDER = "milton"
OUTPUT_FOLDER = "corrupt_capture_pack"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

FILTERS = [
    "decoder_crush",
    "ir_nightwatch",
    "vertical_hold_fail",
    "memory_card_bleed",
    "rolling_shutter_panic",
    # "deadpixel_bloom",
    "tape_dropout",
    "surveillance_burnin",
    "lowlight_smear",
    "signal_autopsy",
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

def blend(a, b, alpha=0.5):
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    out = af * (1 - alpha) + bf * alpha
    return clamp_u8(out)

def add_noise(arr, strength=20):
    noise = np.random.randint(-strength, strength + 1, arr.shape, dtype=np.int16)
    out = arr.astype(np.int16) + noise
    return clamp_u8(out)

def add_speckle(arr, amount=0.03):
    out = arr.copy()
    h, w, _ = out.shape
    count = int(h * w * amount)
    ys = np.random.randint(0, h, count)
    xs = np.random.randint(0, w, count)
    vals = np.random.randint(0, 256, (count, 3))
    out[ys, xs] = vals
    return out

def rgb_shift(arr, max_shift=8):
    out = np.zeros_like(arr)
    for c in range(3):
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        out[:, :, c] = np.roll(np.roll(arr[:, :, c], dy, axis=0), dx, axis=1)
    return out

def add_scanlines(arr, strength=0.18, spacing=2):
    out = arr.astype(np.float32).copy()
    h = out.shape[0]
    for y in range(0, h, spacing):
        out[y, :, :] *= (1.0 - strength)
    return clamp_u8(out)

def contrast_prep(img, contrast=(1.2, 2.0), color=(0.9, 1.8), sharpness=(1.0, 2.5)):
    img = ImageEnhance.Contrast(img).enhance(random.uniform(*contrast))
    img = ImageEnhance.Color(img).enhance(random.uniform(*color))
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(*sharpness))
    return img

def grayscale3(img):
    g = np.array(ImageOps.grayscale(img))
    return np.stack([g, g, g], axis=2)

# =========================================================
# LOW-LEVEL GLITCH HELPERS
# =========================================================
def horizontal_tears(arr, bands=8, max_shift_ratio=0.18):
    h, w, _ = arr.shape
    out = arr.copy()
    for _ in range(bands):
        y = random.randint(0, h - 1)
        bh = random.randint(2, max(4, h // 20))
        y2 = min(h, y + bh)
        band = out[y:y2].copy()
        shift = random.randint(-int(w * max_shift_ratio), int(w * max_shift_ratio))
        band = np.roll(band, shift, axis=1)
        out[y:y2] = band
    return out

def vertical_tears(arr, bands=4, max_shift_ratio=0.12):
    h, w, _ = arr.shape
    out = arr.copy()
    for _ in range(bands):
        x = random.randint(0, w - 1)
        bw = random.randint(2, max(4, w // 25))
        x2 = min(w, x + bw)
        band = out[:, x:x2].copy()
        shift = random.randint(-int(h * max_shift_ratio), int(h * max_shift_ratio))
        band = np.roll(band, shift, axis=0)
        out[:, x:x2] = band
    return out

def wave_warp(arr, amp=6, freq=0.05, vertical=False):
    h, w, _ = arr.shape
    out = np.zeros_like(arr)
    phase = random.uniform(0, 2 * math.pi)

    if not vertical:
        for y in range(h):
            shift = int(amp * math.sin(y * freq + phase))
            out[y] = np.roll(arr[y], shift, axis=0)
    else:
        for x in range(w):
            shift = int(amp * math.sin(x * freq + phase))
            out[:, x] = np.roll(arr[:, x], shift, axis=0)

    return out

def edge_map(img):
    e = ImageOps.grayscale(img).filter(ImageFilter.FIND_EDGES)
    e = ImageEnhance.Contrast(e).enhance(3.0)
    g = np.array(e)
    return np.stack([g, g, g], axis=2)

def neon_edge_map(img, tint=None):
    if tint is None:
        tint = random.choice([
            (255, 0, 255),
            (0, 255, 255),
            (255, 255, 0),
            (255, 80, 0),
        ])

    g = ImageOps.grayscale(img)
    edges = g.filter(ImageFilter.FIND_EDGES)
    edges = ImageEnhance.Contrast(edges).enhance(3.5)
    e = np.array(edges).astype(np.float32) / 255.0

    out = np.zeros((e.shape[0], e.shape[1], 3), dtype=np.uint8)
    out[:, :, 0] = np.clip(e * tint[0], 0, 255)
    out[:, :, 1] = np.clip(e * tint[1], 0, 255)
    out[:, :, 2] = np.clip(e * tint[2], 0, 255)
    return out

def block_shuffle(arr, block_size=16, amount=0.2):
    h, w, _ = arr.shape
    out = arr.copy()

    bh = max(4, block_size)
    bw = max(4, block_size)

    ys = list(range(0, h - bh, bh))
    xs = list(range(0, w - bw, bw))

    num_blocks = int(len(ys) * len(xs) * amount)
    if num_blocks <= 0:
        return out

    coords = [(y, x) for y in ys for x in xs]
    chosen = random.sample(coords, min(num_blocks, len(coords)))

    for (y, x) in chosen:
        y2 = min(h - bh, max(0, y + random.randint(-bh * 2, bh * 2)))
        x2 = min(w - bw, max(0, x + random.randint(-bw * 2, bw * 2)))

        temp = out[y:y+bh, x:x+bw].copy()
        out[y:y+bh, x:x+bw] = out[y2:y2+bh, x2:x2+bw]
        out[y2:y2+bh, x2:x2+bw] = temp

    return out

def block_channel_crush(arr, block_size=12):
    h, w, _ = arr.shape
    out = arr.copy()

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y2 = min(h, y + block_size)
            x2 = min(w, x + block_size)
            block = out[y:y2, x:x2].astype(np.float32)

            # pick a corruption mode
            mode = random.choice(["avg", "channel_drop", "poster"])
            if mode == "avg":
                mean = block.mean(axis=(0, 1), keepdims=True)
                block = np.ones_like(block) * mean
            elif mode == "channel_drop":
                c = random.randint(0, 2)
                block[:, :, c] = block[:, :, c] * random.uniform(0.0, 0.3)
            elif mode == "poster":
                shift = random.choice([2, 3, 4])
                block = ((block.astype(np.uint8) >> shift) << shift)

            out[y:y2, x:x2] = np.clip(block, 0, 255).astype(np.uint8)

    return out

def make_hot_pixels(arr, amount=0.01):
    out = arr.copy()
    h, w, _ = out.shape
    count = int(h * w * amount)
    ys = np.random.randint(0, h, count)
    xs = np.random.randint(0, w, count)

    colors = [
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 255],
        [255, 0, 255],
        [0, 255, 0],
    ]

    for i in range(count):
        out[ys[i], xs[i]] = random.choice(colors)

    return out

def bloom_highlights(arr, threshold=220, radius=4, alpha=0.35):
    lum = arr.mean(axis=2)
    mask = (lum > threshold).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius=radius))
    m = np.array(mask_img).astype(np.float32) / 255.0

    blur = np_img(pil_img(arr).filter(ImageFilter.GaussianBlur(radius=radius)))
    out = arr.astype(np.float32)
    out = out * (1 - m[:, :, None] * alpha) + blur.astype(np.float32) * (m[:, :, None] * alpha)
    return clamp_u8(out)

def add_monitor_overlay(img):
    out = img.copy()
    draw = ImageDraw.Draw(out)

    try:
        font = ImageFont.load_default()
    except:
        font = None

    # fake CCTV / monitor text
    lines = [
        f"CH-{random.randint(1,16):02d}",
        f"REC {random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}",
        f"CAM {random.randint(100,999)}",
    ]

    color = random.choice([
        (0, 255, 0),
        (255, 255, 255),
        (255, 180, 0),
    ])

    y = random.randint(5, 15)
    for line in lines:
        draw.text((random.randint(6, 14), y), line, fill=color, font=font)
        y += 12

    # corner box / HUD-ish rectangle
    if random.random() < 0.8:
        x1 = out.width - random.randint(60, 120)
        y1 = random.randint(8, 20)
        x2 = x1 + random.randint(40, 90)
        y2 = y1 + random.randint(20, 50)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=1)

    return out

def jpeg_roundtrip_like(arr):
    out = arr.copy()

    # approximate JPEG macroblock corruption
    out = block_channel_crush(out, block_size=random.choice([8, 12, 16]))
    out = block_shuffle(out, block_size=random.choice([8, 16, 24]), amount=random.uniform(0.05, 0.18))

    # hard channel clipping
    for c in range(3):
        low = random.randint(0, 25)
        high = random.randint(180, 245)
        ch = out[:, :, c].astype(np.int16)
        ch = np.where(ch < low, 0, ch)
        ch = np.where(ch > high, 255, ch)
        out[:, :, c] = ch

    return out

# =========================================================
# 1. DECODER CRUSH
# Blocky bad decode / crushed macroblocks
# =========================================================
def filter_decoder_crush(img):
    img = contrast_prep(img, contrast=(1.3, 2.4), color=(1.0, 2.0), sharpness=(1.2, 2.4))
    arr = np_img(img)

    out = jpeg_roundtrip_like(arr)
    out = rgb_shift(out, max_shift=random.randint(1, 5))
    out = horizontal_tears(out, bands=random.randint(3, 8), max_shift_ratio=0.10)
    out = add_noise(out, strength=random.randint(8, 18))

    return pil_img(out)

# =========================================================
# 2. IR NIGHTWATCH
# CCTV / infrared / surveillance glow
# =========================================================
def filter_ir_nightwatch(img):
    base = contrast_prep(img, contrast=(1.4, 2.8), color=(0.0, 0.2), sharpness=(1.0, 2.0))
    gray = np.array(ImageOps.grayscale(base)).astype(np.float32)

    # lift darks aggressively like bad IR sensor
    gray = np.clip((gray ** random.uniform(0.75, 0.95)) * random.uniform(1.0, 1.25), 0, 255)

    mode = random.choice(["green", "cyan", "bluegreen"])
    out = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)

    if mode == "green":
        out[:, :, 1] = np.clip(gray * 1.0, 0, 255)
        out[:, :, 0] = np.clip(gray * 0.12, 0, 255)
        out[:, :, 2] = np.clip(gray * 0.08, 0, 255)
    elif mode == "cyan":
        out[:, :, 1] = np.clip(gray * 0.9, 0, 255)
        out[:, :, 2] = np.clip(gray * 1.0, 0, 255)
        out[:, :, 0] = np.clip(gray * 0.05, 0, 255)
    else:
        out[:, :, 1] = np.clip(gray * 0.75, 0, 255)
        out[:, :, 2] = np.clip(gray * 0.95, 0, 255)
        out[:, :, 0] = np.clip(gray * 0.10, 0, 255)

    out = bloom_highlights(out, threshold=random.randint(170, 220), radius=random.randint(2, 5), alpha=random.uniform(0.20, 0.40))
    out = add_scanlines(out, strength=random.uniform(0.08, 0.20), spacing=2)
    out = add_noise(out, strength=random.randint(10, 22))
    out = add_speckle(out, amount=random.uniform(0.01, 0.03))

    if random.random() < 0.8:
        out = np_img(add_monitor_overlay(pil_img(out)))

    return pil_img(out)

# =========================================================
# 3. VERTICAL HOLD FAIL
# Old monitor vertical sync collapse
# =========================================================
def filter_vertical_hold_fail(img):
    img = contrast_prep(img, contrast=(1.2, 2.2), color=(0.8, 1.8), sharpness=(1.0, 2.0))
    arr = np_img(img)

    # strong vertical waves
    out = wave_warp(
        arr,
        amp=random.randint(5, 18),
        freq=random.uniform(0.03, 0.10),
        vertical=True
    )

    # vertical tears + columns slipping
    out = vertical_tears(out, bands=random.randint(6, 14), max_shift_ratio=0.20)

    # occasional full-frame offset split
    h, w, _ = out.shape
    split = random.randint(w // 4, int(w * 0.8))
    offset = random.randint(-h // 8, h // 8)
    out[:, split:] = np.roll(out[:, split:], offset, axis=0)

    out = add_scanlines(out, strength=random.uniform(0.06, 0.18), spacing=random.choice([2, 3]))
    out = add_noise(out, strength=random.randint(8, 18))

    return pil_img(out)

# =========================================================
# 4. MEMORY CARD BLEED
# SD card corruption / block smears / chunk repeats
# =========================================================
def filter_memory_card_bleed(img):
    img = contrast_prep(img, contrast=(1.3, 2.4), color=(1.0, 2.2), sharpness=(1.1, 2.3))
    arr = np_img(img)

    out = arr.copy()

    # block swaps
    out = block_shuffle(out, block_size=random.choice([12, 16, 24, 32]), amount=random.uniform(0.08, 0.22))

    # smear horizontal chunks
    h, w, _ = out.shape
    for _ in range(random.randint(4, 10)):
        y = random.randint(0, h - 1)
        bh = random.randint(4, max(8, h // 16))
        y2 = min(h, y + bh)

        src_x = random.randint(0, max(0, w - 20))
        chunk_w = random.randint(10, max(20, w // 5))
        src_x2 = min(w, src_x + chunk_w)

        smear = out[y:y2, src_x:src_x2].copy()

        # repeat across row
        for xx in range(0, w, smear.shape[1]):
            x2 = min(w, xx + smear.shape[1])
            out[y:y2, xx:x2] = smear[:, :x2-xx]

    # mild decoder crush after bleed
    out = block_channel_crush(out, block_size=random.choice([8, 12, 16]))
    out = rgb_shift(out, max_shift=random.randint(1, 4))
    out = add_noise(out, strength=random.randint(8, 18))

    return pil_img(out)

# =========================================================
# 5. ROLLING SHUTTER PANIC
# Skew / wobble / motion sensor panic
# =========================================================
def filter_rolling_shutter_panic(img):
    img = contrast_prep(img, contrast=(1.3, 2.3), color=(1.0, 2.0), sharpness=(1.0, 2.0))
    arr = np_img(img)
    h, w, _ = arr.shape

    out = np.zeros_like(arr)

    # per-row rolling offset with drift
    drift = random.uniform(-0.25, 0.25)
    wave_amp = random.uniform(2.0, 10.0)
    wave_freq = random.uniform(0.01, 0.06)
    phase = random.uniform(0, 2 * math.pi)

    for y in range(h):
        shift = int((y - h / 2) * drift + wave_amp * math.sin(y * wave_freq + phase))
        out[y] = np.roll(arr[y], shift, axis=0)

    # split-frame panic
    if random.random() < 0.8:
        cut = random.randint(h // 4, int(h * 0.75))
        extra = random.randint(-w // 12, w // 12)
        out[cut:] = np.roll(out[cut:], extra, axis=1)

    out = rgb_shift(out, max_shift=random.randint(1, 5))
    out = add_noise(out, strength=random.randint(6, 16))

    return pil_img(out)

# =========================================================
# 7. TAPE DROPOUT
# VHS/camcorder streaks + analog signal damage
# =========================================================
def filter_tape_dropout(img):
    img = contrast_prep(img, contrast=(1.1, 2.0), color=(0.8, 1.5), sharpness=(0.8, 1.8))
    arr = np_img(img)

    # slightly soften first
    out = np_img(pil_img(arr).filter(ImageFilter.GaussianBlur(radius=random.uniform(0.6, 1.8))))

    # long horizontal dropout lines
    h, w, _ = out.shape
    for _ in range(random.randint(6, 18)):
        y = random.randint(0, h - 2)
        bh = random.randint(1, max(2, h // 120))
        y2 = min(h, y + bh)

        mode = random.choice(["white", "black", "color"])
        if mode == "white":
            out[y:y2] = 255
        elif mode == "black":
            out[y:y2] = 0
        else:
            color = random.choice([
                [255, 0, 255],
                [0, 255, 255],
                [255, 255, 0],
            ])
            out[y:y2] = blend(out[y:y2], np.full_like(out[y:y2], color), alpha=random.uniform(0.35, 0.75))

    # analog drift
    out = horizontal_tears(out, bands=random.randint(6, 14), max_shift_ratio=0.12)
    out = add_scanlines(out, strength=random.uniform(0.08, 0.20), spacing=random.choice([2, 3]))
    out = add_noise(out, strength=random.randint(6, 16))

    return pil_img(out)

# =========================================================
# 8. SURVEILLANCE BURNIN
# Security monitor ghost / HUD / retained screen
# =========================================================
def filter_surveillance_burnin(img):
    base = contrast_prep(img, contrast=(1.4, 2.5), color=(0.5, 1.4), sharpness=(1.2, 2.3))
    arr = np_img(base)

    # desaturate toward monitor look
    gray = grayscale3(base)
    out = blend(arr, gray, alpha=random.uniform(0.55, 0.85))

    # ghost retained frame
    ghost = np.roll(np.roll(out, random.randint(-8, 8), axis=0), random.randint(-8, 8), axis=1)
    ghost = np_img(pil_img(ghost).filter(ImageFilter.GaussianBlur(radius=random.uniform(2.0, 5.5))))
    out = blend(out, ghost, alpha=random.uniform(0.15, 0.40))

    # overlay HUD
    hud = add_monitor_overlay(pil_img(out))
    out = np_img(hud)

    # scanlines + slight tint
    tint = random.choice([
        np.array([0.85, 1.0, 0.85]),
        np.array([0.90, 1.0, 1.0]),
        np.array([1.0, 0.95, 0.85]),
    ])
    out = clamp_u8(out.astype(np.float32) * tint[None, None, :])

    out = add_scanlines(out, strength=random.uniform(0.12, 0.28), spacing=4)
    out = add_noise(out, strength=random.randint(8, 20))

    return pil_img(out)

# =========================================================
# 9. LOWLIGHT SMEAR
# Muddy high ISO + movement smear + bad night capture
# =========================================================
def filter_lowlight_smear(img):
    base = contrast_prep(img, contrast=(0.9, 1.6), color=(0.6, 1.2), sharpness=(0.6, 1.2))
    arr = np_img(base).astype(np.float32)

    # darken scene
    arr = np.clip(arr * random.uniform(0.45, 0.80), 0, 255).astype(np.uint8)

    # strong directional smear
    blur_radius = random.uniform(1.0, 3.5)
    blur = np_img(pil_img(arr).filter(ImageFilter.GaussianBlur(radius=blur_radius)))
    out = blend(arr, blur, alpha=random.uniform(0.35, 0.65))

    # echo trails
    h, w, _ = out.shape
    accum = np.zeros_like(out, dtype=np.float32)
    copies = random.randint(3, 6)

    for i in range(copies):
        dx = int((i + 1) * random.randint(-12, 12) / copies)
        dy = int((i + 1) * random.randint(-6, 6) / copies)
        shifted = np.roll(np.roll(out, dy, axis=0), dx, axis=1).astype(np.float32)
        alpha = max(0.12, 0.8 - i * 0.12)
        accum += shifted * alpha

    accum = accum / max(1.0, np.max([1.0, accum.max() / 255.0]))
    out = clamp_u8(accum)

    # high ISO dirt
    out = add_noise(out, strength=random.randint(18, 38))
    out = add_speckle(out, amount=random.uniform(0.02, 0.07))

    # mild green/magenta sensor contamination
    if random.random() < 0.8:
        out = out.astype(np.float32)
        out[:, :, 1] *= random.uniform(0.9, 1.2)
        out[:, :, 0] *= random.uniform(0.9, 1.15)
        out[:, :, 2] *= random.uniform(0.85, 1.1)
        out = clamp_u8(out)

    return pil_img(out)

# =========================================================
# 10. SIGNAL AUTOPSY
# Diagnostic / forensic / edge-broken corruption
# =========================================================
def filter_signal_autopsy(img):
    base = contrast_prep(img, contrast=(1.5, 2.8), color=(0.2, 1.0), sharpness=(1.4, 3.0))
    arr = np_img(base)

    edges = edge_map(base)
    neon = neon_edge_map(base)

    # mix with block corruption
    out = block_channel_crush(arr, block_size=random.choice([8, 12, 16]))
    out = blend(out, edges, alpha=random.uniform(0.18, 0.35))
    out = blend(out, neon, alpha=random.uniform(0.12, 0.28))

    # forensic grid-ish vertical/horizontal breaks
    out = horizontal_tears(out, bands=random.randint(4, 10), max_shift_ratio=0.08)
    out = vertical_tears(out, bands=random.randint(3, 8), max_shift_ratio=0.08)

    # partial desaturation for lab vibe
    gray = grayscale3(pil_img(out))
    out = blend(out, gray, alpha=random.uniform(0.18, 0.35))

    # harsh noise
    out = add_noise(out, strength=random.randint(12, 26))

    return pil_img(out)

# =========================================================
# DISPATCH
# =========================================================
FILTER_FUNCTIONS = {
    "decoder_crush": filter_decoder_crush,
    "ir_nightwatch": filter_ir_nightwatch,
    "vertical_hold_fail": filter_vertical_hold_fail,
    "memory_card_bleed": filter_memory_card_bleed,
    "rolling_shutter_panic": filter_rolling_shutter_panic,
    # "deadpixel_bloom": filter_deadpixel_bloom,
    "tape_dropout": filter_tape_dropout,
    "surveillance_burnin": filter_surveillance_burnin,
    "lowlight_smear": filter_lowlight_smear,
    "signal_autopsy": filter_signal_autopsy,
}

def apply_filter(img, filter_name):
    return FILTER_FUNCTIONS[filter_name](img)

# =========================================================
# MAIN
# =========================================================
def process_images():
    ensure_output_folder()

    if not os.path.exists(INPUT_FOLDER):
        print(f"Input folder '{INPUT_FOLDER}' not found")
        return

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