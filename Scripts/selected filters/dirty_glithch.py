import os
import random
import math
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageFont

# =========================================================
# CONFIG
# =========================================================
INPUT_FOLDER = "wallter"
OUTPUT_FOLDER = "dirty_glitch_pack"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

FILTERS = [
    "noisy_portrait",
    "crt_tear",
    "timestamp_cam",
    "datamosh_ghost",
    "scanline_warp",
    "motion_echo_bw",
    "dirty_sky_burn",
    "edge_noise_emboss",
    "rgb_face_break",
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

def blend(a, b, alpha=0.5):
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    out = af * (1 - alpha) + bf * alpha
    return clamp_u8(out)

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
# HELPERS FOR EFFECTS
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

def edge_emboss_map(img):
    # emboss + edges combined
    emb = img.filter(ImageFilter.EMBOSS)
    edg = img.filter(ImageFilter.FIND_EDGES)
    a = np_img(emb)
    b = np_img(edg)
    return blend(a, b, alpha=0.55)

def neon_edge_map(img):
    g = ImageOps.grayscale(img)
    edges = g.filter(ImageFilter.FIND_EDGES)
    edges = ImageEnhance.Contrast(edges).enhance(3.5)
    e = np.array(edges)

    out = np.zeros((e.shape[0], e.shape[1], 3), dtype=np.uint8)
    # map edges to neon-like colors
    out[:, :, 0] = np.clip(e * random.uniform(0.8, 1.0), 0, 255)
    out[:, :, 1] = np.clip(e * random.uniform(0.2, 0.9), 0, 255)
    out[:, :, 2] = np.clip(e * random.uniform(0.8, 1.0), 0, 255)
    return out

def add_timestamp_overlay(img):
    out = img.copy()
    draw = ImageDraw.Draw(out)

    # fake CCTV timestamp
    text = f"2024/{random.randint(1,12):02d}/{random.randint(1,28):02d} " \
           f"{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"

    # default font fallback
    try:
        font = ImageFont.load_default()
    except:
        font = None

    x = random.randint(5, 15)
    y = out.height - random.randint(18, 28)

    color = random.choice([
        (255, 80, 0),
        (255, 255, 0),
        (0, 255, 0),
        (255, 0, 255),
    ])

    draw.text((x, y), text, fill=color, font=font)
    return out

def motion_echo(arr, copies=4, max_offset=18, fade=True):
    h, w, _ = arr.shape
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

# =========================================================
# 1. NOISY PORTRAIT
# Like grainy digital face with RGB noise
# =========================================================
def filter_noisy_portrait(img):
    img = contrast_prep(img, contrast=(1.5, 2.6), color=(1.2, 2.2), sharpness=(1.3, 2.8))
    arr = np_img(img)

    # aggressive RGB misregistration
    arr = rgb_shift(arr, max_shift=random.randint(3, 8))

    # heavy sensor grain
    arr = add_noise(arr, strength=random.randint(18, 40))
    arr = add_speckle(arr, amount=random.uniform(0.02, 0.06))

    # subtle horizontal tears
    arr = horizontal_tears(arr, bands=random.randint(3, 8), max_shift_ratio=0.08)

    # slight timestamp vibe sometimes
    out = pil_img(arr)
    if random.random() < 0.5:
        out = add_timestamp_overlay(out)

    return out

# =========================================================
# 2. CRT TEAR
# B/W + RGB bars + static + scanlines
# =========================================================
def filter_crt_tear(img):
    base = contrast_prep(img, contrast=(1.4, 2.8), color=(0.0, 0.2), sharpness=(1.0, 2.0))
    arr = grayscale3(base)

    h, w, _ = arr.shape

    # RGB horizontal bars
    for _ in range(random.randint(3, 7)):
        y = random.randint(0, h - 1)
        bh = random.randint(3, max(6, h // 18))
        y2 = min(h, y + bh)
        color = random.choice([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 255],
        ])
        arr[y:y2] = blend(arr[y:y2], np.full_like(arr[y:y2], color, dtype=np.uint8), alpha=random.uniform(0.25, 0.6))

    arr = horizontal_tears(arr, bands=random.randint(8, 16), max_shift_ratio=0.25)
    arr = add_scanlines(arr, strength=random.uniform(0.12, 0.28), spacing=2)
    arr = add_noise(arr, strength=random.randint(10, 24))
    arr = add_speckle(arr, amount=random.uniform(0.01, 0.04))

    return pil_img(arr)

# =========================================================
# 3. TIMESTAMP CAM
# Dirty camera screenshot vibe
# =========================================================
def filter_timestamp_cam(img):
    img = contrast_prep(img, contrast=(1.3, 2.3), color=(1.1, 1.8), sharpness=(1.4, 3.0))
    arr = np_img(img)

    arr = rgb_shift(arr, max_shift=random.randint(2, 6))
    arr = add_noise(arr, strength=random.randint(20, 38))
    arr = add_speckle(arr, amount=random.uniform(0.03, 0.08))

    # hard clip some channels
    for c in range(3):
        low = random.randint(5, 40)
        high = random.randint(180, 250)
        ch = arr[:, :, c].astype(np.int16)
        ch = np.where(ch < low, 0, ch)
        ch = np.where(ch > high, 255, ch)
        arr[:, :, c] = ch

    out = pil_img(arr)
    out = add_timestamp_overlay(out)
    return out

# =========================================================
# 4. DATAMOSH GHOST
# Glitch silhouette / repeated displaced subject
# =========================================================
def filter_datamosh_ghost(img):
    img = contrast_prep(img, contrast=(1.3, 2.4), color=(0.8, 1.6), sharpness=(1.0, 2.2))
    arr = np_img(img)

    # create echo trails
    ghost = motion_echo(arr, copies=random.randint(3, 6), max_offset=random.randint(10, 30), fade=True)

    # darken background feel
    dark = (arr.astype(np.float32) * random.uniform(0.25, 0.55)).astype(np.uint8)

    out = blend(dark, ghost, alpha=random.uniform(0.55, 0.85))
    out = rgb_shift(out, max_shift=random.randint(2, 8))

    # isolate magenta/green vibe
    if random.random() < 0.8:
        r = out[:, :, 0].astype(np.int16)
        g = out[:, :, 1].astype(np.int16)
        b = out[:, :, 2].astype(np.int16)

        out[:, :, 0] = np.clip(r * random.uniform(1.1, 1.6), 0, 255)
        out[:, :, 1] = np.clip(g * random.uniform(0.6, 1.2), 0, 255)
        out[:, :, 2] = np.clip(b * random.uniform(0.7, 1.4), 0, 255)

    out = add_noise(out, strength=random.randint(10, 22))
    return pil_img(out)

# =========================================================
# 5. SCANLINE WARP
# Umbrella / street / CRT bent horizontal ripple
# =========================================================
def filter_scanline_warp(img):
    img = contrast_prep(img, contrast=(1.2, 2.0), color=(1.1, 1.8), sharpness=(1.0, 2.0))
    arr = np_img(img)

    # strong horizontal wave
    arr = wave_warp(
        arr,
        amp=random.randint(4, 12),
        freq=random.uniform(0.03, 0.10),
        vertical=False
    )

    # stronger localized tears
    arr = horizontal_tears(arr, bands=random.randint(8, 18), max_shift_ratio=0.15)

    # RGB split
    arr = rgb_shift(arr, max_shift=random.randint(2, 8))

    # scanlines
    arr = add_scanlines(arr, strength=random.uniform(0.08, 0.22), spacing=random.choice([2, 3]))

    # extra noise
    arr = add_noise(arr, strength=random.randint(8, 18))

    return pil_img(arr)


# =========================================================
# 7. DIRTY SKY BURN
# Strange sky banding / noisy landscape / acid atmosphere
# =========================================================
def filter_dirty_sky_burn(img):
    img = contrast_prep(img, contrast=(1.5, 2.8), color=(1.3, 2.5), sharpness=(1.2, 2.2))
    arr = np_img(img)
    h, w, _ = arr.shape

    # assume upper region is sky-ish and distort it harder
    split = random.randint(h // 3, int(h * 0.6))
    sky = arr[:split].copy()
    land = arr[split:].copy()

    # brutal noise in sky
    sky = add_noise(sky, strength=random.randint(18, 40))
    sky = add_speckle(sky, amount=random.uniform(0.03, 0.08))
    sky = rgb_shift(sky, max_shift=random.randint(2, 7))
    sky = wave_warp(sky, amp=random.randint(3, 10), freq=random.uniform(0.05, 0.14), vertical=False)

    # hard channel clipping for acid-cloud look
    for c in range(3):
        low = random.randint(0, 30)
        high = random.randint(140, 230)
        ch = sky[:, :, c].astype(np.int16)
        ch = np.where(ch < low, 0, ch)
        ch = np.where(ch > high, 255, ch)
        sky[:, :, c] = ch

    out = np.vstack([sky, land])

    # optional contour edges
    if random.random() < 0.7:
        edge = neon_edge_map(img)
        out = blend(out, edge, alpha=random.uniform(0.18, 0.4))

    return pil_img(out)

# =========================================================
# 8. EDGE NOISE EMBOSS
# Emboss + neon edge + speckle like your interior sample
# =========================================================
def filter_edge_noise_emboss(img):
    img = contrast_prep(img, contrast=(1.4, 2.6), color=(1.2, 2.2), sharpness=(1.5, 3.0))

    base = np_img(img)
    emb = edge_emboss_map(img)
    neon = neon_edge_map(img)

    out = blend(base, emb, alpha=random.uniform(0.35, 0.6))
    out = blend(out, neon, alpha=random.uniform(0.25, 0.5))

    # dirty high ISO speckle
    out = add_noise(out, strength=random.randint(16, 34))
    out = add_speckle(out, amount=random.uniform(0.04, 0.10))

    # subtle channel offset
    out = rgb_shift(out, max_shift=random.randint(1, 4))

    return pil_img(out)

# =========================================================
# 9. RGB FACE BREAK
# Harsh contour + color breakup portrait vibe
# =========================================================
def filter_rgb_face_break(img):
    img = contrast_prep(img, contrast=(1.8, 3.0), color=(1.3, 2.4), sharpness=(1.6, 3.2))
    arr = np_img(img)

    # edge emphasis
    edge = neon_edge_map(img)
    out = blend(arr, edge, alpha=random.uniform(0.25, 0.5))

    # heavy RGB offset
    out = rgb_shift(out, max_shift=random.randint(4, 10))

    # horizontal + vertical tears
    out = horizontal_tears(out, bands=random.randint(4, 10), max_shift_ratio=0.10)
    out = vertical_tears(out, bands=random.randint(2, 5), max_shift_ratio=0.08)

    # harsh grain
    out = add_noise(out, strength=random.randint(18, 36))
    out = add_speckle(out, amount=random.uniform(0.02, 0.06))

    # partial hard clip for posterized corruption
    for c in range(3):
        low = random.randint(0, 30)
        high = random.randint(160, 245)
        ch = out[:, :, c].astype(np.int16)
        ch = np.where(ch < low, 0, ch)
        ch = np.where(ch > high, 255, ch)
        out[:, :, c] = ch

    return pil_img(out)

# =========================================================
# DISPATCH
# =========================================================
FILTER_FUNCTIONS = {
    "noisy_portrait": filter_noisy_portrait,
    "crt_tear": filter_crt_tear,
    "timestamp_cam": filter_timestamp_cam,
    "datamosh_ghost": filter_datamosh_ghost,
    "scanline_warp": filter_scanline_warp,
    "dirty_sky_burn": filter_dirty_sky_burn,
    "edge_noise_emboss": filter_edge_noise_emboss,
    "rgb_face_break": filter_rgb_face_break,
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