import os
import random
from PIL import Image
import numpy as np

# =========================
# CONFIG
# =========================
INPUT_FOLDER = "card"
OUTPUT_FOLDER = "glitch"
VARIANTS_PER_IMAGE = 4

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# Glitch intensity settings
ROW_SHIFT_CHANCE = 0.12
BLOCK_GLITCH_COUNT = 8
RGB_SHIFT_MAX = 12
NOISE_STRENGTH = 18

# =========================
# UTILS
# =========================
def ensure_output_folder():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)

def save_image(np_img, path):
    img = Image.fromarray(np.clip(np_img, 0, 255).astype(np.uint8))
    img.save(path)

def random_shift_row(row, max_shift=40):
    shift = random.randint(-max_shift, max_shift)
    return np.roll(row, shift, axis=0)

def to_numpy_rgb(img):
    """
    Accepts either:
    - PIL.Image.Image
    - numpy.ndarray

    Returns HxWx3 uint8 numpy array.
    """
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))

    if isinstance(img, np.ndarray):
        arr = img
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # grayscale -> RGB
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=2)

        # RGBA -> RGB
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]

        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Unsupported ndarray shape for glitch_cam: {arr.shape}")

        return arr

    raise TypeError(f"Unsupported input type for glitch_cam: {type(img)}")

# =========================
# GLITCH EFFECTS
# =========================
def row_shift_glitch(img):
    """Shift random rows left/right."""
    glitched = img.copy()
    h, w, c = glitched.shape

    for y in range(h):
        if random.random() < ROW_SHIFT_CHANCE:
            shift = random.randint(-w // 8, w // 8)
            glitched[y] = np.roll(glitched[y], shift, axis=0)

    return glitched

def block_copy_glitch(img):
    """Copy random blocks from one place to another."""
    glitched = img.copy()
    h, w, c = glitched.shape

    for _ in range(BLOCK_GLITCH_COUNT):
        bw = random.randint(max(10, w // 20), max(20, w // 6))
        bh = random.randint(max(10, h // 20), max(20, h // 6))

        src_x = random.randint(0, max(0, w - bw))
        src_y = random.randint(0, max(0, h - bh))

        dst_x = random.randint(0, max(0, w - bw))
        dst_y = random.randint(0, max(0, h - bh))

        block = glitched[src_y:src_y+bh, src_x:src_x+bw].copy()
        glitched[dst_y:dst_y+bh, dst_x:dst_x+bw] = block

    return glitched

def rgb_channel_shift(img):
    """Shift RGB channels independently for chromatic glitch."""
    h, w, c = img.shape

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    r_shift_x = random.randint(-RGB_SHIFT_MAX, RGB_SHIFT_MAX)
    r_shift_y = random.randint(-RGB_SHIFT_MAX, RGB_SHIFT_MAX)

    g_shift_x = random.randint(-RGB_SHIFT_MAX, RGB_SHIFT_MAX)
    g_shift_y = random.randint(-RGB_SHIFT_MAX, RGB_SHIFT_MAX)

    b_shift_x = random.randint(-RGB_SHIFT_MAX, RGB_SHIFT_MAX)
    b_shift_y = random.randint(-RGB_SHIFT_MAX, RGB_SHIFT_MAX)

    r2 = np.roll(np.roll(r, r_shift_y, axis=0), r_shift_x, axis=1)
    g2 = np.roll(np.roll(g, g_shift_y, axis=0), g_shift_x, axis=1)
    b2 = np.roll(np.roll(b, b_shift_y, axis=0), b_shift_x, axis=1)

    return np.stack([r2, g2, b2], axis=2)

def noise_glitch(img):
    """Add random digital noise."""
    noise = np.random.randint(-NOISE_STRENGTH, NOISE_STRENGTH + 1, img.shape, dtype=np.int16)
    glitched = img.astype(np.int16) + noise
    return np.clip(glitched, 0, 255).astype(np.uint8)

def horizontal_slice_repeat(img):
    """Repeat or duplicate random horizontal slices."""
    glitched = img.copy()
    h, w, c = glitched.shape

    num_slices = random.randint(3, 8)
    for _ in range(num_slices):
        y = random.randint(0, h - 2)
        slice_height = random.randint(2, max(3, h // 30))
        slice_height = min(slice_height, h - y)

        src = glitched[y:y+slice_height].copy()
        dst_y = min(h - slice_height, y + random.randint(-30, 30))
        glitched[dst_y:dst_y+slice_height] = src

    return glitched

def color_band_glitch(img):
    """Randomly boost one channel in horizontal bands."""
    glitched = img.copy()
    h, w, c = glitched.shape

    for _ in range(random.randint(2, 6)):
        y1 = random.randint(0, h - 1)
        y2 = random.randint(y1, min(h, y1 + random.randint(10, max(20, h // 8))))
        channel = random.randint(0, 2)
        boost = random.randint(20, 80)

        band = glitched[y1:y2, :, channel].astype(np.int16) + boost
        glitched[y1:y2, :, channel] = np.clip(band, 0, 255).astype(np.uint8)

    return glitched

# =========================
# GLITCH PIPELINE
# =========================
def apply_glitch_pipeline(img):
    """
    Runner-compatible:
    - accepts PIL Image OR numpy array
    - returns numpy array
    """
    glitched = to_numpy_rgb(img).copy()

    # Randomly apply a combination of effects
    effects = [
        row_shift_glitch,
        block_copy_glitch,
        rgb_channel_shift,
        noise_glitch,
        horizontal_slice_repeat,
        color_band_glitch,
    ]

    random.shuffle(effects)

    # Apply 3 to 5 random effects
    num_effects = random.randint(3, 5)
    for effect in effects[:num_effects]:
        glitched = effect(glitched)

    return glitched

# =========================
# MAIN
# =========================
def process_images():
    ensure_output_folder()

    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    if not files:
        print(f"No images found in '{INPUT_FOLDER}' folder.")
        return

    print(f"Found {len(files)} image(s). Processing...")

    for filename in files:
        input_path = os.path.join(INPUT_FOLDER, filename)
        base_name, _ = os.path.splitext(filename)

        try:
            img = load_image(input_path)

            for i in range(1, VARIANTS_PER_IMAGE + 1):
                glitched = apply_glitch_pipeline(img)
                output_filename = f"{base_name}_glitch_v{i}.png"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                save_image(glitched, output_path)

            print(f"Done: {filename}")

        except Exception as e:
            print(f"Failed: {filename} -> {e}")

    print(f"\nAll done. Glitched images saved in '{OUTPUT_FOLDER}' folder.")

if __name__ == "__main__":
    process_images()