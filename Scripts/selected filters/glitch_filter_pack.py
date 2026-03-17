import os
import io
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

# =========================================================
# CONFIG
# =========================================================
INPUT_FOLDER = "wallter"
OUTPUT_FOLDER = "glitch_filter_pack"
VARIANTS_PER_IMAGE = 10  # one per filter by default
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

FILTERS = [
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

def jpeg_recompress(img, quality=20):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

# =========================================================
# SHARED HELPERS
# =========================================================
def contrast_prep(img, contrast=(1.2, 2.0), color=(1.0, 1.8), sharpness=(1.0, 2.2)):
    img = ImageEnhance.Contrast(img).enhance(random.uniform(*contrast))
    img = ImageEnhance.Color(img).enhance(random.uniform(*color))
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(*sharpness))
    return img

def add_scanlines(arr, strength=0.15):
    out = arr.astype(np.float32).copy()
    h = out.shape[0]
    for y in range(0, h, 2):
        out[y, :, :] *= (1.0 - strength)
    return clamp_u8(out)

def add_bloom(img, blur_radius=8, amount=0.35):
    base = np_img(img).astype(np.float32)
    blur = np_img(img.filter(ImageFilter.GaussianBlur(radius=blur_radius))).astype(np.float32)
    out = base * (1 - amount) + blur * amount
    return pil_img(clamp_u8(out))

# =========================================================
# FILTER 1: BROKEN CCD
# Sensor readout smear + bloom streaks + row corruption
# =========================================================
def filter_broken_ccd(img):
    img = contrast_prep(img, contrast=(1.3, 2.4), color=(1.1, 2.0))
    arr = np_img(img)
    h, w, _ = arr.shape
    out = arr.copy()

    # bright areas cause horizontal smear (CCD bloom style)
    gray = np.array(ImageOps.grayscale(img))
    bright_mask = gray > random.randint(180, 230)

    for y in range(h):
        if bright_mask[y].any() and random.random() < 0.55:
            smear_len = random.randint(w // 20, max(w // 8, 8))
            row = out[y].copy()
            for x in range(w):
                if bright_mask[y, x]:
                    end = min(w, x + smear_len)
                    row[x:end] = np.maximum(row[x:end], row[x])
            out[y] = row

    # row readout corruption
    for _ in range(random.randint(6, 20)):
        y = random.randint(0, h - 2)
        bh = random.randint(1, max(2, h // 60))
        y2 = min(h, y + bh)
        band = out[y:y2].copy()
        shift = random.randint(-w // 10, w // 10)
        out[y:y2] = np.roll(band, shift, axis=1)

    # subtle channel offset
    out = rgb_shift(out, max_shift=random.randint(1, 4))
    out = add_noise(out, strength=random.randint(4, 10))

    if random.random() < 0.7:
        out = np_img(add_bloom(pil_img(out), blur_radius=random.randint(3, 8), amount=random.uniform(0.15, 0.35)))

    return pil_img(out)

# =========================================================
# FILTER 2: DEAD SENSOR
# Dead pixels + hot pixels + stuck columns / lines
# =========================================================
def filter_dead_sensor(img):
    img = contrast_prep(img, contrast=(1.1, 1.8), color=(0.9, 1.4))
    arr = np_img(img)
    h, w, _ = arr.shape
    out = arr.copy()

    # dead pixels (black)
    dead_count = max(50, (h * w) // 2500)
    ys = np.random.randint(0, h, dead_count)
    xs = np.random.randint(0, w, dead_count)
    out[ys, xs] = [0, 0, 0]

    # hot pixels (bright colored)
    hot_count = max(50, (h * w) // 3000)
    ys = np.random.randint(0, h, hot_count)
    xs = np.random.randint(0, w, hot_count)
    colors = np.random.randint(180, 256, (hot_count, 3))
    out[ys, xs] = colors

    # stuck columns
    for _ in range(random.randint(2, 8)):
        x = random.randint(0, w - 1)
        width = random.randint(1, max(1, w // 300))
        x2 = min(w, x + width)
        mode = random.choice(["black", "white", "color"])
        if mode == "black":
            out[:, x:x2] = [0, 0, 0]
        elif mode == "white":
            out[:, x:x2] = [255, 255, 255]
        else:
            color = np.random.randint(0, 256, 3)
            out[:, x:x2] = color

    # stuck rows
    for _ in range(random.randint(1, 5)):
        y = random.randint(0, h - 1)
        height = random.randint(1, max(1, h // 300))
        y2 = min(h, y + height)
        color = np.random.randint(0, 256, 3)
        out[y:y2] = color

    out = add_noise(out, strength=random.randint(3, 8))
    return pil_img(out)

# =========================================================
# FILTER 3: MEMORY CARD CORRUPT
# Chunk duplication + block displacement + bad zones
# =========================================================
def filter_memory_card_corrupt(img):
    img = contrast_prep(img, contrast=(1.0, 1.6), color=(1.0, 1.6))
    arr = np_img(img)
    h, w, _ = arr.shape
    out = arr.copy()

    # block copy / duplicate
    for _ in range(random.randint(10, 28)):
        bw = random.randint(max(8, w // 30), max(16, w // 8))
        bh = random.randint(max(8, h // 30), max(16, h // 8))

        sx = random.randint(0, max(0, w - bw))
        sy = random.randint(0, max(0, h - bh))
        dx = random.randint(0, max(0, w - bw))
        dy = random.randint(0, max(0, h - bh))

        block = out[sy:sy+bh, sx:sx+bw].copy()

        # sometimes corrupt the block color
        if random.random() < 0.5:
            ch = random.randint(0, 2)
            block[:, :, ch] = np.clip(block[:, :, ch].astype(np.int16) + random.randint(40, 120), 0, 255)

        out[dy:dy+bh, dx:dx+bw] = block

    # bad color zones
    for _ in range(random.randint(2, 8)):
        x1 = random.randint(0, w - 1)
        y1 = random.randint(0, h - 1)
        bw = random.randint(max(10, w // 20), max(20, w // 5))
        bh = random.randint(max(10, h // 20), max(20, h // 5))
        x2 = min(w, x1 + bw)
        y2 = min(h, y1 + bh)

        zone = out[y1:y2, x1:x2].copy()

        if random.random() < 0.5:
            zone = np.roll(zone, random.randint(-20, 20), axis=0)
        if random.random() < 0.5:
            zone = np.roll(zone, random.randint(-20, 20), axis=1)

        if random.random() < 0.7:
            zone = rgb_shift(zone, max_shift=random.randint(1, 5))

        out[y1:y2, x1:x2] = zone

    out = add_noise(out, strength=random.randint(5, 12))
    return pil_img(out)

# =========================================================
# FILTER 4: VHS DRIFT
# Tracking drift + chroma bleed + scanlines + tape warp
# =========================================================
def filter_vhs_drift(img):
    img = contrast_prep(img, contrast=(1.0, 1.5), color=(0.8, 1.2), sharpness=(0.6, 1.3))
    arr = np_img(img)
    h, w, _ = arr.shape
    out = arr.copy()

    # horizontal wave drift
    warped = np.zeros_like(out)
    amp = random.randint(3, max(4, w // 80))
    freq = random.uniform(0.02, 0.08)
    phase = random.uniform(0, 2 * np.pi)

    for y in range(h):
        shift = int(amp * np.sin(y * freq + phase))
        warped[y] = np.roll(out[y], shift, axis=0)

    out = warped

    # chroma bleed / channel misalignment
    out = rgb_shift(out, max_shift=random.randint(2, 8))

    # tape tracking bands
    for _ in range(random.randint(3, 8)):
        y = random.randint(0, h - 1)
        bh = random.randint(4, max(8, h // 18))
        y2 = min(h, y + bh)
        band = out[y:y2].copy()
        band = np.roll(band, random.randint(-w // 12, w // 12), axis=1)
        if random.random() < 0.6:
            band = np.clip(band.astype(np.int16) + random.randint(-35, 35), 0, 255).astype(np.uint8)
        out[y:y2] = band

    out = add_scanlines(out, strength=random.uniform(0.08, 0.22))
    out = add_noise(out, strength=random.randint(6, 14))

    # slight blur for tape softness
    out = np_img(pil_img(out).filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2))))
    return pil_img(out)

# =========================================================
# FILTER 5: PIXEL MELT
# Vertical luminance-based melt / drippy digital collapse
# =========================================================
def filter_pixel_melt(img):
    img = contrast_prep(img, contrast=(1.2, 2.0), color=(1.0, 1.6))
    arr = np_img(img)
    h, w, _ = arr.shape
    out = arr.copy()

    gray = np.array(ImageOps.grayscale(img))

    # choose columns and melt downward based on brightness
    step = max(1, w // 250)
    for x in range(0, w, step):
        col = out[:, x].copy()
        gcol = gray[:, x]

        # find bright runs
        threshold = random.randint(120, 200)
        for y in range(h):
            if gcol[y] > threshold and random.random() < 0.08:
                melt_len = random.randint(h // 30, max(h // 8, 10))
                y2 = min(h, y + melt_len)

                # smear the source pixel downward
                color = col[y].copy()
                for yy in range(y, y2):
                    alpha = 1.0 - ((yy - y) / max(1, (y2 - y)))
                    out[yy, x] = np.clip(out[yy, x].astype(np.float32) * (1 - 0.6 * alpha) + color * (0.6 * alpha), 0, 255)

    # soften into drippy effect
    out = np_img(pil_img(out).filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.9))))
    out = add_noise(out, strength=random.randint(2, 8))
    return pil_img(out)

# =========================================================
# FILTER 6: JPEG CRUSH
# Repeated recompression abuse
# =========================================================
def filter_jpeg_crush(img):
    out = img.copy()

    # multiple recompress rounds
    rounds = random.randint(4, 10)
    for _ in range(rounds):
        q = random.randint(5, 28)
        out = jpeg_recompress(out, quality=q)

        # sometimes resize slightly and restore to worsen artifacts
        if random.random() < 0.5:
            w, h = out.size
            scale = random.uniform(0.82, 0.96)
            sw, sh = max(8, int(w * scale)), max(8, int(h * scale))
            out = out.resize((sw, sh), Image.BILINEAR).resize((w, h), Image.NEAREST)

    # optional channel shift after compression
    arr = np_img(out)
    if random.random() < 0.6:
        arr = rgb_shift(arr, max_shift=random.randint(1, 4))
    arr = add_noise(arr, strength=random.randint(2, 6))

    return pil_img(arr)

# =========================================================
# FILTER 7: SIGNAL TORN
# Hard digital tears / offsets / broken transmission
# =========================================================
def filter_signal_torn(img):
    img = contrast_prep(img, contrast=(1.1, 1.8), color=(1.0, 1.5))
    arr = np_img(img)
    h, w, _ = arr.shape
    out = arr.copy()

    # big horizontal tears
    for _ in range(random.randint(4, 12)):
        y = random.randint(0, h - 1)
        bh = random.randint(3, max(8, h // 20))
        y2 = min(h, y + bh)
        band = out[y:y2].copy()

        shift = random.randint(-w // 5, w // 5)
        band = np.roll(band, shift, axis=1)

        if random.random() < 0.5:
            # severe channel isolation
            ch = random.randint(0, 2)
            temp = np.zeros_like(band)
            temp[:, :, ch] = band[:, :, ch]
            band = temp

        out[y:y2] = band

    # vertical tears
    for _ in range(random.randint(2, 6)):
        x = random.randint(0, w - 1)
        bw = random.randint(2, max(6, w // 30))
        x2 = min(w, x + bw)
        band = out[:, x:x2].copy()
        band = np.roll(band, random.randint(-h // 6, h // 6), axis=0)
        out[:, x:x2] = band

    out = rgb_shift(out, max_shift=random.randint(2, 7))
    out = add_noise(out, strength=random.randint(5, 14))
    return pil_img(out)

# =========================================================
# FILTER 8: NIGHT VISION
# Green surveillance + glow + noise
# =========================================================
def filter_night_vision(img):
    img = contrast_prep(img, contrast=(1.4, 2.4), color=(0.0, 0.2), sharpness=(0.8, 1.5))
    gray = ImageOps.grayscale(img)

    arr = np.array(gray).astype(np.float32)

    # contrast stretch
    arr = (arr - arr.min()) / max(1e-6, (arr.max() - arr.min()))
    arr = np.power(arr, random.uniform(0.7, 1.2))
    arr = clamp_u8(arr * 255)

    # green tint
    out = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    out[:, :, 0] = (arr * random.uniform(0.15, 0.35)).astype(np.uint8)
    out[:, :, 1] = arr
    out[:, :, 2] = (arr * random.uniform(0.05, 0.2)).astype(np.uint8)

    # bloom
    glow = np_img(Image.fromarray(out).filter(ImageFilter.GaussianBlur(radius=random.uniform(2.0, 6.0))))
    out = blend(out, glow, alpha=random.uniform(0.2, 0.45))

    # vignette-ish darkening (simple radial)
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist_norm = dist / dist.max()
    vignette = np.clip(1.15 - dist_norm * random.uniform(0.4, 0.8), 0.5, 1.0)
    out = clamp_u8(out.astype(np.float32) * vignette[:, :, None])

    out = add_scanlines(out, strength=random.uniform(0.06, 0.16))
    out = add_noise(out, strength=random.randint(8, 18))
    return pil_img(out)

# =========================================================
# FILTER 9: INFRARED BLOOM
# Pseudo IR hot foliage / bright glow / surreal outdoor feel
# =========================================================
def filter_infrared_bloom(img):
    img = contrast_prep(img, contrast=(1.3, 2.2), color=(1.0, 1.8))
    arr = np_img(img).astype(np.float32)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # pseudo foliage detection (green dominance)
    foliage = (g > r * 0.9) & (g > b * 1.05)

    out = arr.copy()

    # brighten foliage to pink/white IR style
    out[:, :, 0][foliage] = np.clip(out[:, :, 0][foliage] * random.uniform(1.3, 1.8) + 60, 0, 255)
    out[:, :, 1][foliage] = np.clip(out[:, :, 1][foliage] * random.uniform(0.7, 1.1), 0, 255)
    out[:, :, 2][foliage] = np.clip(out[:, :, 2][foliage] * random.uniform(1.2, 1.8) + 40, 0, 255)

    # sky / blue-ish regions cool down
    sky = (b > r * 1.05) & (b > g * 0.95)
    out[:, :, 2][sky] = np.clip(out[:, :, 2][sky] * random.uniform(1.1, 1.5), 0, 255)
    out[:, :, 1][sky] = np.clip(out[:, :, 1][sky] * random.uniform(0.8, 1.1), 0, 255)

    out = clamp_u8(out)
    ir = pil_img(out)

    # dreamy bloom
    ir = add_bloom(ir, blur_radius=random.randint(5, 12), amount=random.uniform(0.2, 0.45))

    # slight channel shift for surreal optics
    out = np_img(ir)
    out = rgb_shift(out, max_shift=random.randint(1, 5))
    out = add_noise(out, strength=random.randint(3, 8))
    return pil_img(out)

# =========================================================
# FILTER 10: XRAY NEGATIVE
# Negative scan / lab-film look
# =========================================================
def filter_xray_negative(img):
    gray = ImageOps.grayscale(contrast_prep(img, contrast=(1.4, 2.5), color=(0.0, 0.1)))
    arr = np.array(gray)

    # invert
    arr = 255 - arr

    # emphasize edges by mixing with edge map
    edges = np.array(gray.filter(ImageFilter.FIND_EDGES))
    edges = np.clip(edges * random.uniform(1.5, 3.0), 0, 255).astype(np.uint8)

    # combine into xray-like monochrome cyan/blue
    out = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    base = np.clip(arr.astype(np.int16) + (edges // 2).astype(np.int16), 0, 255).astype(np.uint8)

    out[:, :, 0] = (base * random.uniform(0.6, 0.9)).astype(np.uint8)
    out[:, :, 1] = (base * random.uniform(0.85, 1.0)).astype(np.uint8)
    out[:, :, 2] = base

    # optional black cracks / film damage
    h, w = arr.shape
    for _ in range(random.randint(4, 15)):
        x = random.randint(0, w - 1)
        width = random.randint(1, max(1, w // 500))
        x2 = min(w, x + width)
        out[:, x:x2] = np.clip(out[:, x:x2].astype(np.int16) - random.randint(40, 120), 0, 255)

    out = add_noise(out, strength=random.randint(3, 10))
    return pil_img(out)

# =========================================================
# FILTER DISPATCH
# =========================================================
FILTER_FUNCTIONS = {
    "broken_ccd": filter_broken_ccd,
    "dead_sensor": filter_dead_sensor,
    "memory_card_corrupt": filter_memory_card_corrupt,
    "vhs_drift": filter_vhs_drift,
    "pixel_melt": filter_pixel_melt,
    "jpeg_crush": filter_jpeg_crush,
    "signal_torn": filter_signal_torn,
    "night_vision": filter_night_vision,
    "infrared_bloom": filter_infrared_bloom,
    "xray_negative": filter_xray_negative,
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