import os
import random
import math
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw

# =========================================================
# CONFIG
# =========================================================
INPUT_FOLDER = "milton"
OUTPUT_FOLDER = "disposable_flash_pack"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

FILTERS = [
    "cybershot_smear",
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

def add_noise(arr, strength=18):
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

def contrast_prep(img, contrast=(1.1, 1.8), color=(0.9, 1.6), sharpness=(0.9, 1.8)):
    img = ImageEnhance.Contrast(img).enhance(random.uniform(*contrast))
    img = ImageEnhance.Color(img).enhance(random.uniform(*color))
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(*sharpness))
    return img

def add_vignette(arr, strength=0.45):
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist_norm = dist / (dist.max() + 1e-6)
    vignette = np.clip(1.0 - dist_norm * strength, 0.35, 1.0)
    return clamp_u8(arr.astype(np.float32) * vignette[:, :, None])

def grayscale_arr(arr):
    arr = arr.astype(np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return clamp_u8(gray)

# =========================================================
# CORE FLASH / CAMERA HELPERS
# =========================================================
def apply_flash_blowout(arr, center_bias=True, intensity=(45, 120), radius_mult=(1.1, 2.4)):
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

    flash = np.clip(1.0 - dist_norm * random.uniform(*radius_mult), 0, 1)
    flash = flash ** random.uniform(1.2, 2.8)

    out = arr.astype(np.float32)
    out += flash[:, :, None] * random.uniform(*intensity)

    # small harsh flash contrast pop
    out = np.clip((out - 128) * random.uniform(1.03, 1.25) + 128, 0, 255)
    return clamp_u8(out), flash

def crush_background(arr, flash_mask, amount=(0.18, 0.55)):
    out = arr.astype(np.float32)
    bg = 1.0 - flash_mask
    crush = random.uniform(*amount)
    out *= (1.0 - bg[:, :, None] * crush)
    return clamp_u8(out)

def bloom_highlights(arr, threshold=215, radius=4, alpha=0.25):
    lum = arr.mean(axis=2)
    mask = (lum > threshold).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius=radius))
    m = np.array(mask_img).astype(np.float32) / 255.0

    blur = np_img(pil_img(arr).filter(ImageFilter.GaussianBlur(radius=radius)))
    out = arr.astype(np.float32)
    out = out * (1 - m[:, :, None] * alpha) + blur.astype(np.float32) * (m[:, :, None] * alpha)
    return clamp_u8(out)

def add_flash_hotspot(arr, flash_mask, tint=(255, 255, 255), alpha=0.12):
    tint_arr = np.zeros_like(arr, dtype=np.uint8)
    tint_arr[:, :, 0] = tint[0]
    tint_arr[:, :, 1] = tint[1]
    tint_arr[:, :, 2] = tint[2]
    return clamp_u8(arr.astype(np.float32) * (1 - flash_mask[:, :, None] * alpha) + tint_arr.astype(np.float32) * (flash_mask[:, :, None] * alpha))

def cheap_sensor_noise(arr, strength=(10, 26), speckle=(0.005, 0.03)):
    out = add_noise(arr, strength=random.randint(*strength))
    out = add_speckle(out, amount=random.uniform(*speckle))
    return out

def compact_camera_softness(arr, radius=(0.4, 1.8)):
    if random.random() < 0.9:
        return np_img(pil_img(arr).filter(ImageFilter.GaussianBlur(radius=random.uniform(*radius))))
    return arr

def fluorescent_contamination(arr, mode=None):
    out = arr.astype(np.float32)

    if mode is None:
        mode = random.choice(["green_magenta", "cyan_magenta", "warm_green", "retail_white"])

    if mode == "green_magenta":
        tint = np.array([1.05, 1.12, 0.95], dtype=np.float32)
        shadows = np.array([1.08, 0.95, 1.08], dtype=np.float32)
    elif mode == "cyan_magenta":
        tint = np.array([0.98, 1.06, 1.12], dtype=np.float32)
        shadows = np.array([1.10, 0.96, 1.08], dtype=np.float32)
    elif mode == "warm_green":
        tint = np.array([1.10, 1.08, 0.90], dtype=np.float32)
        shadows = np.array([0.95, 1.08, 0.92], dtype=np.float32)
    else:
        tint = np.array([1.03, 1.03, 1.00], dtype=np.float32)
        shadows = np.array([0.98, 1.02, 1.05], dtype=np.float32)

    gray = grayscale_arr(arr) / 255.0
    shadow_mask = (1.0 - gray)[:, :, None]

    out = out * tint[None, None, :]
    out = out * (1 - shadow_mask * 0.18) + out * shadows[None, None, :] * (shadow_mask * 0.18)

    return clamp_u8(out)

def motion_smear(arr, copies=4, max_offset=10):
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(copies):
        dx = int((i + 1) * random.randint(-max_offset, max_offset) / max(1, copies))
        dy = int((i + 1) * random.randint(-max_offset // 2, max_offset // 2) / max(1, copies))
        shifted = np.roll(np.roll(arr, dy, axis=0), dx, axis=1).astype(np.float32)
        alpha = max(0.12, 0.8 - i * 0.12)
        out += shifted * alpha

    out = out / max(1.0, np.max([1.0, out.max() / 255.0]))
    return clamp_u8(out)

def add_small_flash_reflection(arr):
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]

    # small reflection / lens spot
    cx = random.randint(w // 4, 3 * w // 4)
    cy = random.randint(h // 5, 4 * h // 5)

    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist_norm = dist / (dist.max() + 1e-6)
    spot = np.clip(1.0 - dist_norm * random.uniform(4.0, 8.0), 0, 1)
    spot = spot ** random.uniform(1.5, 3.5)

    color = random.choice([
        np.array([255, 255, 255], dtype=np.float32),
        np.array([255, 220, 200], dtype=np.float32),
        np.array([220, 255, 255], dtype=np.float32),
    ])

    out = arr.astype(np.float32)
    alpha = spot[:, :, None] * random.uniform(0.06, 0.18)
    out = out * (1 - alpha) + color[None, None, :] * alpha
    return clamp_u8(out)

def add_fake_redeye(arr):
    # approximate only — center-biased small red glows
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]

    out = arr.astype(np.float32)

    eye_centers = []
    base_y = random.randint(h // 3, h // 2)
    base_x = random.randint(w // 3, 2 * w // 3)
    sep = random.randint(max(12, w // 20), max(20, w // 10))

    eye_centers.append((base_x - sep // 2, base_y))
    eye_centers.append((base_x + sep // 2, base_y + random.randint(-4, 4)))

    for cx, cy in eye_centers:
        rx = random.randint(3, max(5, w // 60))
        ry = random.randint(2, max(4, h // 70))

        mask = (((xx - cx) / max(1, rx)) ** 2 + ((yy - cy) / max(1, ry)) ** 2)
        glow = np.clip(1.0 - mask, 0, 1)
        glow = glow ** random.uniform(1.2, 2.4)

        red = np.zeros_like(out)
        red[:, :, 0] = 255
        red[:, :, 1] = random.randint(10, 40)
        red[:, :, 2] = random.randint(10, 40)

        alpha = glow[:, :, None] * random.uniform(0.10, 0.35)
        out = out * (1 - alpha) + red * alpha

    return clamp_u8(out)

def dark_room_crush(arr, strength=(0.20, 0.55)):
    gray = grayscale_arr(arr) / 255.0
    shadow_mask = (1.0 - gray)[:, :, None]
    out = arr.astype(np.float32)
    out *= (1.0 - shadow_mask * random.uniform(*strength))
    return clamp_u8(out)

def add_compact_timestamp(img):
    out = img.copy()
    draw = ImageDraw.Draw(out)

    text = f"{random.randint(1,12):02d}/{random.randint(1,28):02d}/0{random.randint(2,9)}"

    x = random.randint(8, 18)
    y = out.height - random.randint(18, 30)

    color = random.choice([
        (255, 180, 0),
        (255, 255, 255),
        (255, 220, 120),
    ])

    draw.text((x, y), text, fill=color)
    return out

# =========================================================
# FILTERS
# =========================================================

# 10. partyfloor_overkill
def filter_partyfloor_overkill(img):
    base = contrast_prep(img, contrast=(1.15, 2.0), color=(1.2, 2.2), sharpness=(0.9, 1.8))
    arr = np_img(base)

    # chaotic party color wash
    tint = random.choice([
        np.array([1.18, 0.82, 1.10], dtype=np.float32),
        np.array([1.10, 0.90, 1.20], dtype=np.float32),
        np.array([1.05, 1.00, 1.18], dtype=np.float32),
    ])
    out = clamp_u8(arr.astype(np.float32) * tint[None, None, :])

    out, flash = apply_flash_blowout(out, center_bias=False, intensity=(55, 140), radius_mult=(1.1, 2.8))
    out = motion_smear(out, copies=random.randint(3, 6), max_offset=random.randint(6, 16))
    out = bloom_highlights(out, threshold=random.randint(180, 230), radius=random.randint(2, 5), alpha=random.uniform(0.14, 0.30))

    # noisy blacks + slight channel break
    out = rgb_shift(out, max_shift=random.randint(1, 4))
    out = cheap_sensor_noise(out, strength=(12, 28), speckle=(0.008, 0.03))
    out = dark_room_crush(out, strength=(0.08, 0.22))

    if random.random() < 0.45:
        out = np_img(add_compact_timestamp(pil_img(out)))

    return pil_img(out)

# =========================================================
# DISPATCH
# =========================================================
FILTER_FUNCTIONS = {
    # "bathroom_mirror_flash": filter_bathroom_mirror_flash,
    # "parking_lot_snapshot": filter_parking_lot_snapshot,
    # "myspace_paparazzi": filter_myspace_paparazzi,
    # "elevator_selfie_burn": filter_elevator_selfie_burn,
    # "red_eye_memory": filter_red_eye_memory,
    # "cybershot_smear": filter_cybershot_smear,
    # "mall_kiosk_flash": filter_mall_kiosk_flash,
    # "carseat_nightflash": filter_carseat_nightflash,
    "partyfloor_overkill": filter_partyfloor_overkill,
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
