import os
import random
import math
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# =========================================================
# CONFIG
# =========================================================
INPUT_FOLDER  = "wallter"
OUTPUT_FOLDER = "pixless_camera_lab"

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# "Sensor" resolution (low-res simulation step)
SENSOR_WIDTH  = 256
SENSOR_HEIGHT = 128
UPSCALE       = 5

# Available filter styles
FILTERS = [
    "classic_toycam",
    "pixless_plus",
    "sensor_grit",
    "lcd_dream",
    "night_sensor",
    "broken_handheld",
    # You can uncomment these if you want the simpler legacy variants too
    "v1_simple",
    "v2_pixless_legacy",
    "v3_advanced_legacy",
]

# =========================================================
# Directory setup
# =========================================================
def ensure_dirs():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for f in FILTERS:
        os.makedirs(os.path.join(OUTPUT_FOLDER, f), exist_ok=True)


# =========================================================
# Basic utilities
# =========================================================
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


# =========================================================
# Sensor / Lens / Tone simulation
# =========================================================
def apply_lens_softness(img, radius=(0.4, 1.6)):
    if random.random() < 0.92:
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(*radius)))
    return img

def tone_shape(arr, contrast=1.1, lift_blacks=6, highlight_clip=250, gamma=1.0):
    x = arr.astype(np.float32)
    # gamma
    x = np.clip(x / 255.0, 0, 1)
    x = np.power(x, gamma) * 255.0
    # contrast
    x = (x - 128.0) * contrast + 128.0
    # lift blacks
    x += lift_blacks
    # soft highlight roll-off
    x = np.where(x > highlight_clip, highlight_clip + (x - highlight_clip) * 0.25, x)
    return clamp_u8(x)

def apply_channel_gains(arr, gains=(1.0, 1.0, 1.0)):
    out = arr.astype(np.float32)
    out[:, :, 0] *= gains[0]
    out[:, :, 1] *= gains[1]
    out[:, :, 2] *= gains[2]
    return clamp_u8(out)


# =========================================================
# Noise & Sensor Imperfections
# =========================================================
def add_per_channel_noise(arr, r=8, g=6, b=10):
    out = arr.astype(np.int16)
    out[:, :, 0] += np.random.randint(-r, r+1, out[:, :, 0].shape)
    out[:, :, 1] += np.random.randint(-g, g+1, out[:, :, 1].shape)
    out[:, :, 2] += np.random.randint(-b, b+1, out[:, :, 2].shape)
    return clamp_u8(out)

def add_row_noise(arr, strength=8):
    out = arr.astype(np.int16)
    h = out.shape[0]
    row_offsets = np.random.randint(-strength, strength+1, (h, 1, 3))
    return clamp_u8(out + row_offsets)

def add_column_noise(arr, strength=5):
    out = arr.astype(np.int16)
    w = out.shape[1]
    col_offsets = np.random.randint(-strength, strength+1, (1, w, 3))
    return clamp_u8(out + col_offsets)

def add_hot_pixels(arr, count=40, bright_only=True):
    out = arr.copy()
    h, w, _ = out.shape
    for _ in range(count):
        y, x = random.randint(0, h-1), random.randint(0, w-1)
        if bright_only:
            color = random.choice([[255,255,255], [255,200,200], [200,255,255], [255,255,200]])
        else:
            color = [random.randint(0,255) for _ in range(3)]
        out[y, x] = color
    return out

def add_dead_pixels(arr, count=20):
    out = arr.copy()
    h, w, _ = out.shape
    for _ in range(count):
        y, x = random.randint(0, h-1), random.randint(0, w-1)
        mode = random.choice(["black", "stuck_r", "stuck_g", "stuck_b"])
        if mode == "black":       out[y,x] = [0,0,0]
        elif mode == "stuck_r":   out[y,x] = [255,0,0]
        elif mode == "stuck_g":   out[y,x] = [0,255,0]
        else:                     out[y,x] = [0,0,255]
    return out


# =========================================================
# Quantization helpers
# =========================================================
def rgb565_quantization(arr):
    out = arr.copy()
    out[:, :, 0] = (out[:, :, 0] >> 3) << 3
    out[:, :, 1] = (out[:, :, 1] >> 2) << 2
    out[:, :, 2] = (out[:, :, 2] >> 3) << 3
    return out

def rgb444_quantization(arr):
    out = arr.copy()
    out[:, :, 0] = (out[:, :, 0] >> 4) << 4
    out[:, :, 1] = (out[:, :, 1] >> 4) << 4
    out[:, :, 2] = (out[:, :, 2] >> 4) << 4
    return out

def quantize_pil(arr, colors=24):
    img = pil_img(arr)
    q = img.quantize(colors=colors, method=Image.FASTOCTREE)
    return np_img(q.convert("RGB"))


# =========================================================
# Ordered Bayer dither
# =========================================================
BAYER_4X4 = np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5]
], dtype=np.float32) / 16

def ordered_dither(arr, levels=16, strength=0.5):
    out = arr.astype(np.float32).copy()
    h, w, _ = out.shape

    threshold_map = np.tile(BAYER_4X4, (math.ceil(h/4), math.ceil(w/4)))[:h, :w]
    threshold_map = (threshold_map - 0.5) * 255 * strength

    for c in range(3):
        ch = out[:, :, c] + threshold_map
        ch = np.clip(ch, 0, 255)
        step = 255.0 / max(1, levels - 1)
        ch = np.round(ch / step) * step
        out[:, :, c] = ch

    return clamp_u8(out)


# =========================================================
# Display / LCD simulation
# =========================================================
def upscale_nearest(arr, factor=UPSCALE):
    img = pil_img(arr)
    return np_img(img.resize((arr.shape[1]*factor, arr.shape[0]*factor), Image.NEAREST))

def add_lcd_row_shading(arr, strength=0.06):
    out = arr.astype(np.float32)
    for y in range(0, out.shape[0], UPSCALE):
        out[y:y+1, :, :] *= (1.0 - strength)
    return clamp_u8(out)

def add_lcd_grid(arr, line_strength=0.07):
    out = arr.astype(np.float32)
    h, w = out.shape[:2]
    for y in range(0, h, UPSCALE):
        out[y:y+1, :, :] *= (1.0 - line_strength)
    for x in range(0, w, UPSCALE):
        out[:, x:x+1, :] *= (1.0 - line_strength)
    return clamp_u8(out)

def add_subpixel_hint(arr, strength=0.035):
    out = arr.astype(np.float32)
    w = out.shape[1]
    for x in range(w):
        mod = x % UPSCALE
        if mod < UPSCALE//3:
            out[:, x, 0] *= (1.0 + strength)
        elif mod < 2*UPSCALE//3:
            out[:, x, 1] *= (1.0 + strength)
        else:
            out[:, x, 2] *= (1.0 + strength)
    return clamp_u8(out)

def add_panel_vignette(arr, strength=0.16):
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w/2, h/2
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    dist_norm = dist / (dist.max() + 1e-6)
    vignette = np.clip(1.0 - dist_norm * strength, 0.78, 1.0)
    return clamp_u8(arr.astype(np.float32) * vignette[..., None])


# =========================================================
# Core pipeline
# =========================================================
def simulate_low_res_sensor(img,
                           lens_blur=(0.5,1.4),
                           contrast=1.12, lift=8, clip=248, gamma=1.0,
                           gains=(1.0,1.0,1.0)):

    # Keep original aspect — only resize to sensor resolution
    small = img.resize((SENSOR_WIDTH, SENSOR_HEIGHT), Image.LANCZOS)
    arr = np_img(small)

    arr = tone_shape(arr, contrast=contrast, lift_blacks=lift,
                     highlight_clip=clip, gamma=gamma)
    arr = apply_channel_gains(arr, gains=gains)

    return arr


def finalize_display(arr,
                    use_grid=True, use_subpixel=False,
                    row_shade=True, vignette=True):

    out = upscale_nearest(arr)

    if row_shade:
        out = add_lcd_row_shading(out, strength=random.uniform(0.03, 0.08))
    if use_grid:
        out = add_lcd_grid(out, line_strength=random.uniform(0.04, 0.09))
    if use_subpixel:
        out = add_subpixel_hint(out, strength=random.uniform(0.02, 0.05))
    if vignette:
        out = add_panel_vignette(out, strength=random.uniform(0.10, 0.22))

    return pil_img(out)


# =========================================================
# Style variants
# =========================================================

def variant_classic_toycam(img):
    arr = simulate_low_res_sensor(img,
        lens_blur=(0.4,1.1), contrast=random.uniform(1.05,1.22),
        lift=random.randint(4,11), clip=random.randint(245,252),
        gamma=random.uniform(0.94,1.06),
        gains=(random.uniform(0.97,1.04), random.uniform(0.98,1.05), random.uniform(0.96,1.04))
    )
    arr = ordered_dither(arr, levels=random.choice([16,24,32]), strength=random.uniform(0.28,0.48))
    arr = quantize_pil(arr, colors=random.choice([24,28,32]))
    return finalize_display(arr, use_grid=True, use_subpixel=False, row_shade=True, vignette=True)


def variant_pixless_plus(img):
    arr = simulate_low_res_sensor(img,
        lens_blur=(0.6,1.5), contrast=random.uniform(1.14,1.38),
        lift=random.randint(5,14), clip=random.randint(238,248),
        gamma=random.uniform(0.90,1.02),
        gains=(random.uniform(1.00,1.07), random.uniform(1.00,1.09), random.uniform(0.93,1.03))
    )
    arr = rgb565_quantization(arr)
    arr = add_per_channel_noise(arr, r=6, g=5, b=9)
    arr = add_row_noise(arr, strength=4)
    arr = ordered_dither(arr, levels=random.choice([16,20]), strength=random.uniform(0.32,0.58))
    arr = quantize_pil(arr, colors=random.choice([18,20,24]))
    return finalize_display(arr, use_grid=True, use_subpixel=True, row_shade=True, vignette=True)


def variant_sensor_grit(img):
    arr = simulate_low_res_sensor(img,
        lens_blur=(0.8,1.9), contrast=random.uniform(1.12,1.45),
        lift=random.randint(2,12), clip=random.randint(225,245),
        gamma=random.uniform(0.94,1.10),
        gains=(random.uniform(1.00,1.10), random.uniform(1.00,1.12), random.uniform(0.88,1.02))
    )
    arr = rgb565_quantization(arr)
    arr = add_per_channel_noise(arr, r=8, g=7, b=13)
    arr = add_row_noise(arr, 7)
    arr = add_column_noise(arr, 5)
    arr = ordered_dither(arr, levels=random.choice([12,16]), strength=random.uniform(0.38,0.68))
    arr = quantize_pil(arr, colors=random.choice([14,18,22]))
    arr = add_hot_pixels(arr, count=random.randint(20,70))
    if random.random() < 0.6:
        arr = add_dead_pixels(arr, count=random.randint(6,25))
    return finalize_display(arr, use_grid=True, use_subpixel=False, row_shade=True, vignette=True)


def variant_lcd_dream(img):
    arr = simulate_low_res_sensor(img,
        lens_blur=(0.5,1.4), contrast=random.uniform(1.08,1.32),
        lift=random.randint(8,20), clip=random.randint(238,252),
        gamma=random.uniform(0.88,0.99),
        gains=(random.uniform(1.01,1.09), random.uniform(1.00,1.09), random.uniform(1.01,1.11))
    )
    arr = rgb444_quantization(arr)
    arr = ordered_dither(arr, levels=random.choice([16,20]), strength=random.uniform(0.26,0.52))
    arr = quantize_pil(arr, colors=random.choice([20,24,30]))
    arr = apply_channel_gains(arr, gains=(1.02,1.03,1.04))
    return finalize_display(arr, use_grid=True, use_subpixel=True, row_shade=True, vignette=True)


def variant_night_sensor(img):
    arr = simulate_low_res_sensor(img,
        lens_blur=(0.9,2.2), contrast=random.uniform(0.92,1.18),
        lift=random.randint(6,20), clip=random.randint(218,240),
        gamma=random.uniform(1.04,1.18),
        gains=(random.uniform(0.94,1.05), random.uniform(1.00,1.12), random.uniform(0.88,1.02))
    )
    arr = clamp_u8(arr.astype(np.float32) * random.uniform(0.68,0.92))
    arr = apply_channel_gains(arr, gains=(0.96,1.05,1.03))
    arr = rgb565_quantization(arr)
    arr = add_per_channel_noise(arr, r=10, g=9, b=16)
    arr = add_row_noise(arr, 8)
    arr = add_column_noise(arr, 5)
    arr = ordered_dither(arr, levels=random.choice([10,14,18]), strength=random.uniform(0.42,0.78))
    arr = quantize_pil(arr, colors=random.choice([12,16,20]))
    arr = add_hot_pixels(arr, count=random.randint(30,90))
    return finalize_display(arr, use_grid=True, use_subpixel=False, row_shade=True, vignette=True)


def variant_broken_handheld(img):
    arr = simulate_low_res_sensor(img,
        lens_blur=(0.7,2.0), contrast=random.uniform(1.08,1.40),
        lift=random.randint(3,16), clip=random.randint(222,245),
        gamma=random.uniform(0.93,1.10),
        gains=(random.uniform(0.98,1.10), random.uniform(0.96,1.10), random.uniform(0.90,1.04))
    )
    arr = rgb444_quantization(arr)
    arr = add_per_channel_noise(arr, r=10, g=8, b=14)
    arr = add_row_noise(arr, 9)
    arr = add_column_noise(arr, 6)
    arr = ordered_dither(arr, levels=random.choice([8,12,16]), strength=random.uniform(0.48,0.85))
    arr = quantize_pil(arr, colors=random.choice([10,14,18]))
    arr = add_hot_pixels(arr, count=random.randint(30,110), bright_only=False)
    arr = add_dead_pixels(arr, count=random.randint(12,50))
    return finalize_display(arr, use_grid=True, use_subpixel=True, row_shade=True, vignette=True)


# Legacy simpler variants (optional — uncomment in FILTERS if wanted)
def variant_v1_simple(img):
    small = img.resize((SENSOR_WIDTH, SENSOR_HEIGHT), Image.NEAREST)
    small = small.quantize(colors=32)
    up = small.resize((SENSOR_WIDTH*UPSCALE, SENSOR_HEIGHT*UPSCALE), Image.NEAREST)
    return up


def variant_v2_pixless_legacy(img):
    small = img.resize((SENSOR_WIDTH, SENSOR_HEIGHT), Image.NEAREST)
    small = ImageEnhance.Contrast(small).enhance(1.4)
    small = small.quantize(colors=24, method=Image.FASTOCTREE).convert("RGB")
    arr = np_img(small)
    arr = add_per_channel_noise(arr, 8,6,10)
    up = pil_img(arr).resize((SENSOR_WIDTH*UPSCALE, SENSOR_HEIGHT*UPSCALE), Image.NEAREST)
    return up


def variant_v3_advanced_legacy(img):
    small = img.resize((SENSOR_WIDTH, SENSOR_HEIGHT), Image.NEAREST)
    small = ImageEnhance.Contrast(small).enhance(1.5)
    arr = np_img(small)
    arr = rgb565_quantization(arr)
    small = pil_img(arr).quantize(colors=20).convert("RGB")
    arr = np_img(small)
    arr = add_per_channel_noise(arr, 12,9,15)
    up = pil_img(arr).resize((SENSOR_WIDTH*UPSCALE, SENSOR_HEIGHT*UPSCALE), Image.NEAREST)
    return up


# =========================================================
# Dispatcher
# =========================================================
FILTER_FUNCTIONS = {
    "classic_toycam":       variant_classic_toycam,
    "pixless_plus":         variant_pixless_plus,
    "sensor_grit":          variant_sensor_grit,
    "lcd_dream":            variant_lcd_dream,
    "night_sensor":         variant_night_sensor,
    "broken_handheld":      variant_broken_handheld,
    "v1_simple":            variant_v1_simple,
    "v2_pixless_legacy":    variant_v2_pixless_legacy,
    "v3_advanced_legacy":   variant_v3_advanced_legacy,
}


# =========================================================
# Main processor
# =========================================================
def process_images():
    ensure_dirs()

    if not os.path.exists(INPUT_FOLDER):
        print(f"Input folder '{INPUT_FOLDER}' not found.")
        return

    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    if not files:
        print(f"No supported images found in '{INPUT_FOLDER}'")
        return

    print(f"Found {len(files)} image(s). Generating {len(FILTERS)} variants each...")

    for file in files:
        path = os.path.join(INPUT_FOLDER, file)
        name = os.path.splitext(file)[0]

        try:
            img = load_image(path)
            orig_w, orig_h = img.size

            for filter_name in FILTERS:
                func = FILTER_FUNCTIONS.get(filter_name)
                if not func:
                    continue

                result_img = func(img)

                # Resize back to ORIGINAL dimensions (important!)
                result_img = result_img.resize((orig_w, orig_h), Image.LANCZOS)

                out_path = os.path.join(
                    OUTPUT_FOLDER,
                    filter_name,
                    f"{name}_{filter_name}.png"
                )
                save_image(result_img, out_path)

            print(f"Processed {file}")

        except Exception as e:
            print(f"Error processing {file}: {e}")

    print(f"\nDone. Results saved in '{OUTPUT_FOLDER}'")


if __name__ == "__main__":
    process_images()