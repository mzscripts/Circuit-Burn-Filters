from __future__ import annotations

import numpy as np
from PIL import Image

def neon_glitch(image: Image.Image) -> Image.Image:
    array = np.asarray(image, dtype=np.uint8)
    output = array.copy()
    output[:, :, 0] = np.roll(output[:, :, 0], 12, axis=1)
    output[:, :, 2] = np.roll(output[:, :, 2], -8, axis=0)

    for row in range(0, output.shape[0], 6):
        output[row:row + 1] = np.clip(output[row:row + 1] * 0.62, 0, 255)

    return Image.fromarray(output, "RGB")
