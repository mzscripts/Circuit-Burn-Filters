import os
import random
import math
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw

# =========================================================
# CONFIG
# =========================================================
INPUT_FOLDER = "wallter"
OUTPUT_FOLDER = "nostalgia_sensor_pack"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

FILTERS = [
    # CCD DREAM (pretty / nostalgic / colorful)
    "neon_memory",
    "cybernight_ccd",
    "malllight_nostalgia",
    "sodium_dream",
    "midnight_blossom",
    "aqua_magenta_memory",

    # WEBCAM HAUNT (creepy / dim / old internet)
    "msn_call_haunt",
    "yahoo_messenger_night",
    "bedroom_webcam_2007",
    "blown_window_feed",
    "skype_freezeface",
    "ghost_in_the_lcd",
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

def add_noise(arr, strength=14):
    noise = np.random.randint(-strength, strength + 1, arr.shape, dtype=np.int16)
    out = arr.astype(np.int16) + noise
    return clamp_u8(out)

def add_speckle(arr, amount=0.015):
    out = arr.copy()
    h, w, _ = out.shape
    count = int(h * w * amount)
    ys = np.random.randint(0, h, count)
    xs = np.random.randint(0, w, count)
    vals = np.random.randint(0, 256, (count, 3))
    out[ys, xs] = vals
    return out

def rgb_shift(arr, max_shift=3):
    out = np.zeros_like(arr)
    for c in range(3):
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        out[:, :, c] = np.roll(np.roll(arr[:, :, c], dy, axis=0), dx, axis=1)
    return out

def contrast_prep(img, contrast=(1.0, 1.7), color=(0.9, 1.6), sharpness=(0.8, 1.8)):
    img = ImageEnhance.Contrast(img).enhance(random.uniform(*contrast))
    img = ImageEnhance.Color(img).enhance(random.uniform(*color))
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(*sharpness))
    return img

def add_vignette(arr, strength=0.35):
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist_norm = dist / (dist.max() + 1e-6)
    vignette = np.clip(1.0 - dist_norm * strength, 0.40, 1.0)
    return clamp_u8(arr.astype(np.float32) * vignette[:, :, None])

def grayscale_arr(arr):
    arr = arr.astype(np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return clamp_u8(gray)

# =========================================================
# CORE HELPERS
# =========================================================
def bloom_highlights(arr, threshold=205, radius=4, alpha=0.28):
    lum = arr.mean(axis=2)
    mask = (lum > threshold).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius=radius))
    m = np.array(mask_img).astype(np.float32) / 255.0

    blur = np_img(pil_img(arr).filter(ImageFilter.GaussianBlur(radius=radius)))
    out = arr.astype(np.float32)
    out = out * (1 - m[:, :, None] * alpha) + blur.astype(np.float32) * (m[:, :, None] * alpha)
    return clamp_u8(out)

def compact_camera_softness(arr, radius=(0.5, 2.0)):
    if random.random() < 0.95:
        return np_img(pil_img(arr).filter(ImageFilter.GaussianBlur(radius=random.uniform(*radius))))
    return arr

def cheap_sensor_noise(arr, strength=(8, 22), speckle=(0.004, 0.02)):
    out = add_noise(arr, strength=random.randint(*strength))
    out = add_speckle(out, amount=random.uniform(*speckle))
    return out

def motion_smear(arr, copies=4, max_offset=8):
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(copies):
        dx = int((i + 1) * random.randint(-max_offset, max_offset) / max(1, copies))
        dy = int((i + 1) * random.randint(-max_offset // 2, max_offset // 2) / max(1, copies))
        shifted = np.roll(np.roll(arr, dy, axis=0), dx, axis=1).astype(np.float32)
        alpha = max(0.12, 0.75 - i * 0.10)
        out += shifted * alpha

    out = out / max(1.0, np.max([1.0, out.max() / 255.0]))
    return clamp_u8(out)

def apply_night_tint(arr, mode=None):
    out = arr.astype(np.float32)

    if mode is None:
        mode = random.choice([
            "blue_magenta",
            "cyan_purple",
            "sodium_blue",
            "mall_white_blue",
            "screen_green_blue",
            "cold_monitor",
        ])

    if mode == "blue_magenta":
        tint = np.array([1.05, 0.92, 1.18], dtype=np.float32)
    elif mode == "cyan_purple":
        tint = np.array([0.95, 1.05, 1.18], dtype=np.float32)
    elif mode == "sodium_blue":
        tint = np.array([1.12, 1.00, 0.88], dtype=np.float32)
    elif mode == "mall_white_blue":
        tint = np.array([1.02, 1.02, 1.08], dtype=np.float32)
    elif mode == "screen_green_blue":
        tint = np.array([0.95, 1.08, 1.05], dtype=np.float32)
    else:
        tint = np.array([0.92, 1.00, 1.10], dtype=np.float32)

    return clamp_u8(out * tint[None, None, :])

def split_tone_shadows_highlights(arr, shadow_tint, highlight_tint, strength=0.22):
    gray = grayscale_arr(arr).astype(np.float32) / 255.0
    shadow_mask = (1.0 - gray)[:, :, None]
    highlight_mask = gray[:, :, None]

    out = arr.astype(np.float32)
    out = out * (1 - shadow_mask * strength) + out * shadow_tint[None, None, :] * (shadow_mask * strength)
    out = out * (1 - highlight_mask * strength) + out * highlight_tint[None, None, :] * (highlight_mask * strength)

    return clamp_u8(out)

def lift_blacks(arr, amount=(8, 28)):
    out = arr.astype(np.int16) + random.randint(*amount)
    return clamp_u8(out)

def crush_shadows(arr, amount=(0.08, 0.25)):
    gray = grayscale_arr(arr).astype(np.float32) / 255.0
    shadow_mask = (1.0 - gray)[:, :, None]
    out = arr.astype(np.float32)
    out *= (1.0 - shadow_mask * random.uniform(*amount))
    return clamp_u8(out)

def apply_screen_glow(arr, side=None, color=None, strength=(18, 55)):
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]

    if side is None:
        side = random.choice(["left", "right", "top", "window"])

    if color is None:
        color = random.choice([
            np.array([180, 220, 255], dtype=np.float32),  # cool monitor
            np.array([140, 255, 220], dtype=np.float32),  # green-blue
            np.array([220, 220, 255], dtype=np.float32),  # pale LCD
        ])

    if side == "left":
        glow = np.clip(1.0 - (xx / (w * random.uniform(0.45, 0.9))), 0, 1)
    elif side == "right":
        glow = np.clip(1.0 - ((w - xx) / (w * random.uniform(0.45, 0.9))), 0, 1)
    elif side == "top":
        glow = np.clip(1.0 - (yy / (h * random.uniform(0.35, 0.75))), 0, 1)
    else:
        # fake bright window/monitor patch
        cx = random.randint(w // 5, 4 * w // 5)
        cy = random.randint(h // 5, h // 2)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        dist_norm = dist / (dist.max() + 1e-6)
        glow = np.clip(1.0 - dist_norm * random.uniform(1.4, 3.0), 0, 1)
        glow = glow ** random.uniform(1.2, 2.5)

    out = arr.astype(np.float32)
    out += glow[:, :, None] * random.uniform(*strength) * (color[None, None, :] / 255.0)
    return clamp_u8(out), glow

def lowres_resize(arr, scale_choices=(0.45, 0.55, 0.65, 0.75)):
    img = pil_img(arr)
    w, h = img.size
    scale = random.choice(scale_choices)
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    back = small.resize((w, h), Image.BILINEAR)
    return np_img(back)

def jpegish_blockiness(arr, block=8, strength=0.15):
    h, w, _ = arr.shape
    out = arr.copy().astype(np.float32)

    for y in range(0, h, block):
        for x in range(0, w, block):
            y2 = min(h, y + block)
            x2 = min(w, x + block)
            patch = out[y:y2, x:x2]
            mean = patch.mean(axis=(0, 1), keepdims=True)
            out[y:y2, x:x2] = patch * (1 - strength) + mean * strength

    return clamp_u8(out)

def add_scanlines(arr, strength=0.08, spacing=2):
    out = arr.astype(np.float32).copy()
    h = out.shape[0]
    for y in range(0, h, spacing):
        out[y, :, :] *= (1.0 - strength)
    return clamp_u8(out)

def edge_soft_glow(arr):
    img = pil_img(arr)
    edges = ImageOps.grayscale(img).filter(ImageFilter.FIND_EDGES)
    edges = ImageEnhance.Contrast(edges).enhance(2.0)
    e = np.array(edges).astype(np.float32) / 255.0

    glow = np.zeros_like(arr, dtype=np.float32)
    glow[:, :, 0] = e * random.uniform(30, 90)
    glow[:, :, 1] = e * random.uniform(20, 80)
    glow[:, :, 2] = e * random.uniform(40, 110)

    out = arr.astype(np.float32) + glow * random.uniform(0.15, 0.35)
    return clamp_u8(out)

def add_fake_webcam_overlay(img):
    out = img.copy()
    draw = ImageDraw.Draw(out)

    # tiny old-call-ish overlay
    if random.random() < 0.8:
        draw.rectangle(
            [8, 8, random.randint(45, 85), random.randint(20, 30)],
            outline=(255, 255, 255),
            width=1
        )

    if random.random() < 0.6:
        draw.text((12, 10), random.choice(["LIVE", "CALL", "WEBCAM", "ONLINE"]), fill=(255, 255, 255))

    return out

# =========================================================
# CCD DREAM FILTERS (PRETTY / BEAUTIFUL / NOSTALGIC)
# =========================================================

# 1. ghost_in_the_lcd
def filter_ghost_in_the_lcd(img):
    base = contrast_prep(img, contrast=(0.95, 1.4), color=(0.5, 1.0), sharpness=(0.6, 1.1))
    arr = np_img(base).astype(np.float32)

    # dark and pale
    arr *= random.uniform(0.55, 0.82)
    out = np.clip(arr, 0, 255).astype(np.uint8)

    # cold monitor glow
    out, _ = apply_screen_glow(out, side=random.choice(["left", "right", "window"]), color=np.array([210, 230, 255], dtype=np.float32), strength=(25, 65))
    out = apply_night_tint(out, mode="cold_monitor")

    # ghost double exposure
    ghost = np.roll(np.roll(out, random.randint(-8, 8), axis=0), random.randint(-8, 8), axis=1)
    ghost = np_img(pil_img(ghost).filter(ImageFilter.GaussianBlur(radius=random.uniform(1.2, 3.0))))
    out = blend(out, ghost, alpha=random.uniform(0.10, 0.22))

    out = lowres_resize(out, scale_choices=(0.40, 0.50, 0.60))
    out = compact_camera_softness(out, radius=(1.0, 2.8))
    out = cheap_sensor_noise(out, strength=(10, 24), speckle=(0.008, 0.03))
    out = lift_blacks(out, amount=(6, 14))
    out = add_vignette(out, strength=random.uniform(0.18, 0.32))

    out = np_img(add_fake_webcam_overlay(pil_img(out)))
    return pil_img(out)

# =========================================================
# DISPATCH
# =========================================================
FILTER_FUNCTIONS = {
    # CCD DREAM
    "neon_memory": filter_neon_memory,
    "cybernight_ccd": filter_cybernight_ccd,
    "malllight_nostalgia": filter_malllight_nostalgia,
    "sodium_dream": filter_sodium_dream,
    "midnight_blossom": filter_midnight_blossom,
    "aqua_magenta_memory": filter_aqua_magenta_memory,

    # WEBCAM HAUNT
    "msn_call_haunt": filter_msn_call_haunt,
    "yahoo_messenger_night": filter_yahoo_messenger_night,
    "bedroom_webcam_2007": filter_bedroom_webcam_2007,
    "blown_window_feed": filter_blown_window_feed,
    "skype_freezeface": filter_skype_freezeface,
    "ghost_in_the_lcd": filter_ghost_in_the_lcd,
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