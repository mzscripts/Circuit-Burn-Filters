import os
import random
import math
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageFont

# =========================================================
# CONFIG
# =========================================================
INPUT_FOLDER = "milton"
OUTPUT_FOLDER = "camcorder_nightmare_pack"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

FILTERS = [
    "flashburn_paparazzi",
    "minidv_dropframe",
    "nightbus_cmos",
    "tape_head_damage",
    "handycam_ghostwalk",
    "clubcam_redroom",
    "security_stairwell",
    "autofocus_hunt",
    "ccd_bleed_streak",
    "found_footage_burn",
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

def bloom_highlights(arr, threshold=220, radius=4, alpha=0.35):
    lum = arr.mean(axis=2)
    mask = (lum > threshold).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius=radius))
    m = np.array(mask_img).astype(np.float32) / 255.0

    blur = np_img(pil_img(arr).filter(ImageFilter.GaussianBlur(radius=radius)))
    out = arr.astype(np.float32)
    out = out * (1 - m[:, :, None] * alpha) + blur.astype(np.float32) * (m[:, :, None] * alpha)
    return clamp_u8(out)

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
            (255, 80, 0),
            (255, 255, 255),
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

def motion_echo(arr, copies=4, max_offset=18, fade=True):
    out = np.zeros_like(arr, dtype=np.float32)

    for i in range(copies):
        dx = int((i + 1) * random.randint(-max_offset, max_offset) / max(1, copies))
        dy = int((i + 1) * random.randint(-max_offset // 2, max_offset // 2) / max(1, copies))
        shifted = np.roll(np.roll(arr, dy, axis=0), dx, axis=1).astype(np.float32)

        if fade:
            alpha = max(0.15, 1.0 - (i / max(1, copies)))
        else:
            alpha = 1.0 / copies

        out += shifted * alpha

    out = out / max(1.0, np.max([1.0, out.max() / 255.0]))
    return clamp_u8(out)

def add_camcorder_timestamp(img):
    out = img.copy()
    draw = ImageDraw.Draw(out)

    try:
        font = ImageFont.load_default()
    except:
        font = None

    year = random.choice([1999, 2001, 2003, 2006, 2008, 2010])
    text = f"{random.randint(1,12):02d}/{random.randint(1,28):02d}/{str(year)[2:]}  " \
           f"{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"

    x = random.randint(6, 16)
    y = out.height - random.randint(18, 30)

    color = random.choice([
        (255, 180, 0),
        (255, 255, 255),
        (0, 255, 255),
    ])

    draw.text((x, y), text, fill=color, font=font)
    return out

def add_cctv_overlay(img):
    out = img.copy()
    draw = ImageDraw.Draw(out)

    try:
        font = ImageFont.load_default()
    except:
        font = None

    lines = [
        f"CAM {random.randint(1,12):02d}",
        f"REC {random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}",
    ]

    color = random.choice([
        (0, 255, 0),
        (255, 255, 255),
        (255, 180, 0),
    ])

    y = random.randint(6, 14)
    for line in lines:
        draw.text((random.randint(6, 14), y), line, fill=color, font=font)
        y += 12

    if random.random() < 0.8:
        draw.rectangle(
            [
                out.width - random.randint(60, 110),
                random.randint(8, 18),
                out.width - random.randint(12, 25),
                random.randint(28, 50),
            ],
            outline=color,
            width=1
        )

    return out

def add_vignette(arr, strength=0.5):
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist_norm = dist / (dist.max() + 1e-6)
    vignette = np.clip(1.0 - dist_norm * strength, 0.35, 1.0)
    return clamp_u8(arr.astype(np.float32) * vignette[:, :, None])

def apply_flash_blowout(arr, center_bias=True):
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]

    if center_bias:
        cx = random.randint(w // 3, 2 * w // 3)
        cy = random.randint(h // 4, 2 * h // 3)
    else:
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)

    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist_norm = dist / (dist.max() + 1e-6)

    flash = np.clip(1.0 - dist_norm * random.uniform(1.2, 2.5), 0, 1)
    flash = flash ** random.uniform(1.2, 2.8)

    out = arr.astype(np.float32)
    gain = random.uniform(40, 120)
    out += flash[:, :, None] * gain

    # slight harsh contrast from flash
    out = np.clip((out - 128) * random.uniform(1.05, 1.35) + 128, 0, 255)
    return clamp_u8(out)

def add_ccd_vertical_streaks(arr, threshold=220, strength=0.5, max_width=2):
    out = arr.astype(np.float32).copy()
    h, w, _ = out.shape

    lum = out.mean(axis=2)
    bright_points = np.argwhere(lum > threshold)

    if len(bright_points) == 0:
        return clamp_u8(out)

    # sample only a few brightest points
    sample_count = min(len(bright_points), random.randint(4, 20))
    chosen_idx = np.random.choice(len(bright_points), sample_count, replace=False)
    chosen = bright_points[chosen_idx]

    for y, x in chosen:
        width = random.randint(1, max_width)
        x1 = max(0, x - width // 2)
        x2 = min(w, x1 + width)

        # smear vertically
        col = out[:, x1:x2, :]
        boost = np.zeros_like(col)

        for yy in range(h):
            d = abs(yy - y)
            falloff = max(0.0, 1.0 - d / max(1, h * random.uniform(0.15, 0.45)))
            boost[yy] = falloff * strength * np.array([255, 255, 255], dtype=np.float32)

        out[:, x1:x2, :] = np.clip(col + boost, 0, 255)

    return clamp_u8(out)

# =========================================================
# 2. MINIDV DROPFRAME
# MiniDV frame skips / horizontal digital loss
# =========================================================
def filter_minidv_dropframe(img):
    img = contrast_prep(img, contrast=(1.2, 2.2), color=(0.9, 1.8), sharpness=(1.0, 2.0))
    arr = np_img(img)

    out = arr.copy()

    # horizontal digital band loss
    h, w, _ = out.shape
    for _ in range(random.randint(5, 14)):
        y = random.randint(0, h - 1)
        bh = random.randint(2, max(4, h // 40))
        y2 = min(h, y + bh)

        mode = random.choice(["black", "white", "repeat", "color"])
        if mode == "black":
            out[y:y2] = 0
        elif mode == "white":
            out[y:y2] = 255
        elif mode == "repeat":
            src_y = max(0, min(h - bh, y + random.randint(-30, 30)))
            out[y:y2] = out[src_y:src_y+bh]
        else:
            c = random.choice([
                [255, 0, 255],
                [0, 255, 255],
                [255, 255, 0],
            ])
            out[y:y2] = blend(out[y:y2], np.full_like(out[y:y2], c), alpha=random.uniform(0.25, 0.6))

    out = horizontal_tears(out, bands=random.randint(6, 14), max_shift_ratio=0.12)
    out = block_channel_crush(out, block_size=random.choice([8, 12, 16]))
    out = add_noise(out, strength=random.randint(6, 16))

    if random.random() < 0.6:
        out = np_img(add_camcorder_timestamp(pil_img(out)))

    return pil_img(out)

# =========================================================
# 3. NIGHTBUS CMOS
# Sodium vapor / muddy night transit / cheap sensor
# =========================================================
def filter_nightbus_cmos(img):
    base = contrast_prep(img, contrast=(0.9, 1.6), color=(0.6, 1.4), sharpness=(0.7, 1.5))
    arr = np_img(base).astype(np.float32)

    # darken heavily
    arr *= random.uniform(0.45, 0.75)

    # streetlight tint: sodium / greenish / blue
    tint_mode = random.choice(["sodium", "greenish", "cold_blue"])
    if tint_mode == "sodium":
        tint = np.array([1.20, 0.95, 0.60], dtype=np.float32)
    elif tint_mode == "greenish":
        tint = np.array([0.95, 1.10, 0.85], dtype=np.float32)
    else:
        tint = np.array([0.85, 0.95, 1.15], dtype=np.float32)

    out = np.clip(arr * tint[None, None, :], 0, 255).astype(np.uint8)

    # muddy smear
    blur = np_img(pil_img(out).filter(ImageFilter.GaussianBlur(radius=random.uniform(1.2, 3.0))))
    out = blend(out, blur, alpha=random.uniform(0.30, 0.60))

    # light motion drag
    out = motion_echo(out, copies=random.randint(3, 5), max_offset=random.randint(8, 18), fade=True)

    # dirty high ISO
    out = add_noise(out, strength=random.randint(18, 36))
    out = add_speckle(out, amount=random.uniform(0.02, 0.06))
    out = add_vignette(out, strength=random.uniform(0.35, 0.65))

    return pil_img(out)

# =========================================================
# 4. TAPE HEAD DAMAGE
# VHS/Hi8 tracking scrape / tape wear
# =========================================================
def filter_tape_head_damage(img):
    base = contrast_prep(img, contrast=(1.0, 1.8), color=(0.7, 1.4), sharpness=(0.8, 1.6))
    arr = np_img(base)

    # soften like analog tape
    out = np_img(pil_img(arr).filter(ImageFilter.GaussianBlur(radius=random.uniform(0.8, 2.0))))

    h, w, _ = out.shape

    # tape scrape lines
    for _ in range(random.randint(8, 24)):
        y = random.randint(0, h - 1)
        bh = random.randint(1, max(2, h // 100))
        y2 = min(h, y + bh)

        mode = random.choice(["white", "black", "ghost"])
        if mode == "white":
            out[y:y2] = blend(out[y:y2], np.full_like(out[y:y2], 255), alpha=random.uniform(0.5, 0.95))
        elif mode == "black":
            out[y:y2] = blend(out[y:y2], np.zeros_like(out[y:y2]), alpha=random.uniform(0.4, 0.9))
        else:
            src_y = max(0, min(h - bh, y + random.randint(-20, 20)))
            ghost = out[src_y:src_y+bh].copy()
            out[y:y2] = blend(out[y:y2], ghost, alpha=random.uniform(0.4, 0.8))

    # tracking wobble
    out = horizontal_tears(out, bands=random.randint(8, 18), max_shift_ratio=0.10)
    out = wave_warp(out, amp=random.randint(2, 6), freq=random.uniform(0.04, 0.10), vertical=False)

    # analog scan feel
    out = add_scanlines(out, strength=random.uniform(0.06, 0.16), spacing=random.choice([2, 3]))
    out = add_noise(out, strength=random.randint(6, 14))

    if random.random() < 0.5:
        out = np_img(add_camcorder_timestamp(pil_img(out)))

    return pil_img(out)

# =========================================================
# 5. HANDYCAM GHOSTWALK
# Shaky handheld ghost trails / dim street smear
# =========================================================
def filter_handycam_ghostwalk(img):
    base = contrast_prep(img, contrast=(1.0, 1.8), color=(0.8, 1.4), sharpness=(0.7, 1.4))
    arr = np_img(base)

    # darken slightly
    out = np.clip(arr.astype(np.float32) * random.uniform(0.65, 0.90), 0, 255).astype(np.uint8)

    # handheld drag
    out = motion_echo(out, copies=random.randint(4, 8), max_offset=random.randint(12, 28), fade=True)

    # shake wobble
    out = wave_warp(out, amp=random.randint(2, 7), freq=random.uniform(0.03, 0.08), vertical=False)

    # soft focus
    out = np_img(pil_img(out).filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 2.8))))

    # dirty ISO
    out = add_noise(out, strength=random.randint(12, 26))
    out = add_speckle(out, amount=random.uniform(0.01, 0.04))

    # slight cool/green contamination
    tint = np.array([
        random.uniform(0.90, 1.05),
        random.uniform(0.95, 1.15),
        random.uniform(0.90, 1.08)
    ], dtype=np.float32)
    out = clamp_u8(out.astype(np.float32) * tint[None, None, :])

    return pil_img(out)

# =========================================================
# 6. CLUBCAM REDROOM
# Party camera / red-magenta flash / blown nightlife
# =========================================================
def filter_clubcam_redroom(img):
    base = contrast_prep(img, contrast=(1.3, 2.4), color=(1.4, 2.6), sharpness=(1.0, 2.2))
    arr = np_img(base).astype(np.float32)

    # red / magenta club wash
    tint = random.choice([
        np.array([1.35, 0.65, 1.10], dtype=np.float32),
        np.array([1.20, 0.55, 1.25], dtype=np.float32),
        np.array([1.10, 0.50, 1.35], dtype=np.float32),
    ])
    out = np.clip(arr * tint[None, None, :], 0, 255).astype(np.uint8)

    # flash blowout but off-center
    out = apply_flash_blowout(out, center_bias=False)

    # club motion drag
    out = motion_echo(out, copies=random.randint(3, 6), max_offset=random.randint(8, 20), fade=True)

    # neon edges
    neon = neon_edge_map(pil_img(out), tint=random.choice([(255, 0, 255), (255, 80, 0), (255, 255, 255)]))
    out = blend(out, neon, alpha=random.uniform(0.08, 0.18))

    # gritty flash sensor
    out = add_noise(out, strength=random.randint(10, 24))
    out = add_speckle(out, amount=random.uniform(0.01, 0.03))

    if random.random() < 0.5:
        out = np_img(add_camcorder_timestamp(pil_img(out)))

    return pil_img(out)

# =========================================================
# 7. SECURITY STAIRWELL
# Grim green-white surveillance mood
# =========================================================
def filter_security_stairwell(img):
    base = contrast_prep(img, contrast=(1.3, 2.6), color=(0.0, 0.2), sharpness=(1.0, 2.0))
    gray = np.array(ImageOps.grayscale(base)).astype(np.float32)

    # green-white or cold-cyan surveillance
    mode = random.choice(["green", "white", "cyan"])
    out = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)

    if mode == "green":
        out[:, :, 1] = np.clip(gray * 1.0, 0, 255)
        out[:, :, 0] = np.clip(gray * 0.08, 0, 255)
        out[:, :, 2] = np.clip(gray * 0.06, 0, 255)
    elif mode == "white":
        g = np.clip(gray * 1.0, 0, 255).astype(np.uint8)
        out = np.stack([g, g, g], axis=2)
    else:
        out[:, :, 1] = np.clip(gray * 0.85, 0, 255)
        out[:, :, 2] = np.clip(gray * 1.0, 0, 255)
        out[:, :, 0] = np.clip(gray * 0.06, 0, 255)

    out = add_scanlines(out, strength=random.uniform(0.10, 0.22), spacing=2)
    out = add_noise(out, strength=random.randint(8, 18))
    out = add_speckle(out, amount=random.uniform(0.01, 0.03))

    out = np_img(add_cctv_overlay(pil_img(out)))
    out = add_vignette(out, strength=random.uniform(0.25, 0.55))

    return pil_img(out)



# =========================================================
# 10. FOUND FOOTAGE BURN
# End-of-tape / archival burn / damaged media memory
# =========================================================
def filter_found_footage_burn(img):
    base = contrast_prep(img, contrast=(1.1, 2.0), color=(0.7, 1.4), sharpness=(0.8, 1.8))
    arr = np_img(base)

    # soften like old media
    out = np_img(pil_img(arr).filter(ImageFilter.GaussianBlur(radius=random.uniform(0.8, 2.2))))

    h, w, _ = out.shape
    yy, xx = np.mgrid[0:h, 0:w]

    # burn center or edge
    mode = random.choice(["center", "corner", "edge"])
    if mode == "center":
        cx = random.randint(w // 3, 2 * w // 3)
        cy = random.randint(h // 3, 2 * h // 3)
    elif mode == "corner":
        cx = random.choice([0, w - 1])
        cy = random.choice([0, h - 1])
    else:
        cx = random.choice([0, w - 1, w // 2])
        cy = random.choice([0, h - 1, h // 2])

    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist_norm = dist / (dist.max() + 1e-6)

    burn = np.clip(1.0 - dist_norm * random.uniform(1.0, 2.2), 0, 1)
    burn = burn ** random.uniform(1.2, 3.0)

    # burnt amber / white wash
    burn_color = random.choice([
        np.array([255, 180, 60], dtype=np.float32),
        np.array([255, 220, 140], dtype=np.float32),
        np.array([255, 255, 220], dtype=np.float32),
    ])

    out = out.astype(np.float32)
    alpha = burn[:, :, None] * random.uniform(0.18, 0.42)
    out = out * (1 - alpha) + burn_color[None, None, :] * alpha
    out = clamp_u8(out)

    # tape / frame damage
    out = horizontal_tears(out, bands=random.randint(4, 12), max_shift_ratio=0.08)
    out = add_scanlines(out, strength=random.uniform(0.06, 0.14), spacing=random.choice([2, 3]))
    out = add_noise(out, strength=random.randint(6, 16))
    out = add_speckle(out, amount=random.uniform(0.005, 0.02))

    # archival ghost frame
    if random.random() < 0.7:
        ghost = np.roll(np.roll(out, random.randint(-10, 10), axis=0), random.randint(-10, 10), axis=1)
        ghost = np_img(pil_img(ghost).filter(ImageFilter.GaussianBlur(radius=random.uniform(1.5, 3.5))))
        out = blend(out, ghost, alpha=random.uniform(0.06, 0.16))

    if random.random() < 0.7:
        out = np_img(add_camcorder_timestamp(pil_img(out)))

    return pil_img(out)

# =========================================================
# DISPATCH
# =========================================================
FILTER_FUNCTIONS = {
    # "flashburn_paparazzi": filter_flashburn_paparazzi,
    "minidv_dropframe": filter_minidv_dropframe,
    "nightbus_cmos": filter_nightbus_cmos,
    "tape_head_damage": filter_tape_head_damage,
    "handycam_ghostwalk": filter_handycam_ghostwalk,
    "clubcam_redroom": filter_clubcam_redroom,
    "security_stairwell": filter_security_stairwell,
    # "autofocus_hunt": filter_autofocus_hunt,
    # "ccd_bleed_streak": filter_ccd_bleed_streak,
    "found_footage_burn": filter_found_footage_burn,
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