import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

# =========================
# CONFIG
# =========================
INPUT_FOLDER = "wallter"
OUTPUT_FOLDER = "glitch_psychedelic_pro"
VARIANTS_PER_IMAGE = 4
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# =========================
# HELPERS
# =========================
def ensure_output_folder():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_image(path):
    return Image.open(path).convert("RGB")

def save_image(img, path):
    img.save(path)

def posterize_np(arr, bits=3):
    # bits 1..8 ; lower = harsher bands
    shift = 8 - bits
    return ((arr >> shift) << shift).astype(np.uint8)

def rgb_shift(arr, max_shift=8):
    h, w, _ = arr.shape
    out = np.zeros_like(arr)

    for c in range(3):
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        out[:, :, c] = np.roll(np.roll(arr[:, :, c], dy, axis=0), dx, axis=1)

    return out

def add_noise(arr, strength=12):
    noise = np.random.randint(-strength, strength + 1, arr.shape, dtype=np.int16)
    out = arr.astype(np.int16) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def edge_boost(pil_img, edge_strength=0.6):
    edges = pil_img.filter(ImageFilter.FIND_EDGES).convert("RGB")
    edges = ImageEnhance.Contrast(edges).enhance(2.5)
    base = np.array(pil_img).astype(np.int16)
    e = np.array(edges).astype(np.int16)
    out = base + (e * edge_strength)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

# =========================
# FALSE COLOR PALETTES
# =========================
PALETTES = [
    # blue -> cyan -> green -> yellow -> magenta -> red
    [
        (0, 0, 80),
        (0, 120, 255),
        (0, 255, 180),
        (180, 255, 0),
        (255, 60, 220),
        (255, 0, 0),
    ],
    # deep blue -> cyan -> white -> pink -> purple
    [
        (10, 20, 120),
        (0, 255, 255),
        (255, 255, 255),
        (255, 80, 180),
        (120, 0, 255),
    ],
    # black -> neon green -> yellow -> orange -> magenta
    [
        (0, 0, 0),
        (0, 255, 0),
        (255, 255, 0),
        (255, 140, 0),
        (255, 0, 255),
    ],
    # cyan -> blue -> magenta -> red -> yellow
    [
        (0, 255, 255),
        (0, 50, 255),
        (255, 0, 255),
        (255, 0, 0),
        (255, 255, 0),
    ],
]

def interpolate_palette(palette, t):
    """
    t in [0,1]
    linearly interpolate across palette stops
    """
    if t <= 0:
        return palette[0]
    if t >= 1:
        return palette[-1]

    scaled = t * (len(palette) - 1)
    i = int(np.floor(scaled))
    frac = scaled - i

    c1 = np.array(palette[i], dtype=np.float32)
    c2 = np.array(palette[min(i + 1, len(palette) - 1)], dtype=np.float32)
    c = c1 * (1 - frac) + c2 * frac
    return tuple(c.astype(np.uint8))

def false_color_map(gray_arr, palette):
    """
    gray_arr: HxW uint8
    map brightness to custom palette
    """
    h, w = gray_arr.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    # vectorized approach
    norm = gray_arr.astype(np.float32) / 255.0
    segments = len(palette) - 1

    scaled = norm * segments
    idx = np.floor(scaled).astype(np.int32)
    frac = scaled - idx

    idx = np.clip(idx, 0, segments - 1)

    p1 = np.array([palette[i] for i in idx.flatten()]).reshape(h, w, 3).astype(np.float32)
    p2 = np.array([palette[i + 1] for i in idx.flatten()]).reshape(h, w, 3).astype(np.float32)
    frac3 = frac[:, :, None]

    out = p1 * (1 - frac3) + p2 * frac3
    return np.clip(out, 0, 255).astype(np.uint8)

# =========================
# MAIN STYLE FILTER
# =========================
def psychedelic_circuit_bend_filter(pil_img):
    # 1. Resize-safe working copy
    img = pil_img.copy()

    # 2. Strong contrast + saturation
    img = ImageEnhance.Contrast(img).enhance(random.uniform(1.8, 3.0))
    img = ImageEnhance.Color(img).enhance(random.uniform(1.8, 3.5))
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(1.2, 2.5))

    # 3. Optional solarize/invert flavor
    if random.random() < 0.5:
        threshold = random.randint(80, 180)
        img = ImageOps.solarize(img, threshold=threshold)

    if random.random() < 0.25:
        img = ImageOps.invert(img)

    # 4. Convert to grayscale luminance map
    gray = ImageOps.grayscale(img)
    gray_arr = np.array(gray)

    # 5. Posterize brightness for hard bands
    bits = random.choice([2, 3, 4])
    gray_arr = posterize_np(gray_arr, bits=bits)

    # 6. False color remap using neon palette
    palette = random.choice(PALETTES)
    color_arr = false_color_map(gray_arr, palette)

    # 7. Blend with original image slightly for realism
    orig_arr = np.array(img).astype(np.float32)
    color_arr_f = color_arr.astype(np.float32)
    blend_alpha = random.uniform(0.15, 0.35)
    mixed = (color_arr_f * (1 - blend_alpha) + orig_arr * blend_alpha)
    mixed = np.clip(mixed, 0, 255).astype(np.uint8)

    # 8. RGB channel shift
    mixed = rgb_shift(mixed, max_shift=random.randint(2, 10))

    # 9. Horizontal band corruption
    h, w, _ = mixed.shape
    for _ in range(random.randint(2, 8)):
        y1 = random.randint(0, h - 1)
        band_h = random.randint(4, max(8, h // 12))
        y2 = min(h, y1 + band_h)

        band = mixed[y1:y2].copy()
        shift = random.randint(-w // 8, w // 8)
        band = np.roll(band, shift, axis=1)

        # sometimes recolor one channel heavily
        if random.random() < 0.6:
            ch = random.randint(0, 2)
            band[:, :, ch] = np.clip(
                band[:, :, ch].astype(np.int16) + random.randint(40, 120), 0, 255
            )

        mixed[y1:y2] = band

    # 10. Add noise
    mixed = add_noise(mixed, strength=random.randint(6, 18))

    # 11. Convert back to PIL and boost edges
    out = Image.fromarray(mixed)
    if random.random() < 0.8:
        out = edge_boost(out, edge_strength=random.uniform(0.25, 0.8))

    # 12. Optional final posterize for hard graphic look
    if random.random() < 0.7:
        final_bits = random.choice([3, 4, 5])
        out_arr = np.array(out)
        out_arr = posterize_np(out_arr, bits=final_bits)
        out = Image.fromarray(out_arr)

    return out

# =========================
# PROCESS FOLDER
# =========================
def process_images():
    ensure_output_folder()

    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    if not files:
        print(f"No images found in '{INPUT_FOLDER}'")
        return

    print(f"Found {len(files)} image(s). Processing...")

    for filename in files:
        input_path = os.path.join(INPUT_FOLDER, filename)
        base_name, _ = os.path.splitext(filename)

        try:
            img = load_image(input_path)

            for i in range(1, VARIANTS_PER_IMAGE + 1):
                out = psychedelic_circuit_bend_filter(img)
                output_filename = f"{base_name}_psyglitch_v{i}.png"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                save_image(out, output_path)

            print(f"Done: {filename}")

        except Exception as e:
            print(f"Failed: {filename} -> {e}")

    print(f"\nAll done. Saved to '{OUTPUT_FOLDER}'")

if __name__ == "__main__":
    process_images()