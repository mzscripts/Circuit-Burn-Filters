"""
circuit_burn_filters.py  –  30 psychedelic / circuit-burn filters
==================================================================
Drop images into a folder called  'wallter'  next to this script.
Results go to  'wallter_output/<filter_name>/'.

Dependencies:  pip install Pillow numpy
"""

import os, sys, glob, math
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

# ── paths ──────────────────────────────────────────────────────────────────────
INPUT_FOLDER  = "jesse"
OUTPUT_FOLDER = "circuitburn_V1"
SUPPORTED     = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ── helpers ────────────────────────────────────────────────────────────────────

def load_images(folder):
    paths = []
    for ext in SUPPORTED:
        paths += glob.glob(os.path.join(folder, f"*{ext}"))
        paths += glob.glob(os.path.join(folder, f"*{ext.upper()}"))
    return sorted(set(paths))

def save(img, src_path, filter_name):
    out_dir = Path(OUTPUT_FOLDER) / filter_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(src_path).stem
    dest = out_dir / f"{stem}__{filter_name}.png"
    img.save(dest)
    print(f"  ✓  {dest}")

def to_arr(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32)

def to_img(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

def saturate(img_pil, factor):
    """Boost saturation – uses PIL's Color enhancer (correct API)."""
    return ImageEnhance.Color(img_pil).enhance(factor)

def build_lut(fn):
    return [int(np.clip(fn(i), 0, 255)) for i in range(256)]


# ══════════════════════════════════════════════════════════════════════════════
#  FILTERS 01 – 30
# ══════════════════════════════════════════════════════════════════════════════

# ── 01. acid_palette ──────────────────────────────────────────────────────────
def acid_palette(img):
    """Neon CMYK channel-swap + multi-point solarisation."""
    r, g, b = img.convert("RGB").split()
    merged = Image.merge("RGB", (ImageOps.invert(b), ImageOps.invert(r), g))
    lut_r = build_lut(lambda x: 255 - x if x > 80  else (x * 2) % 256)
    lut_g = build_lut(lambda x: 255 - x if x > 120 else (x * 3) % 256)
    lut_b = build_lut(lambda x: 255 - x if x > 60  else x)
    out = merged.point(lut_r + lut_g + lut_b)
    return saturate(out, 2.5)

# ── 02. thermal_vision ────────────────────────────────────────────────────────
def thermal_vision(img):
    """Infrared heat-map: black→blue→green→yellow→red→white."""
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape
    stops = [(0.00,(0,0,0)),(0.25,(0,0,200)),(0.50,(0,200,0)),
             (0.70,(255,255,0)),(0.85,(255,80,0)),(1.00,(255,255,255))]
    out = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(len(stops)-1):
        t0, c0 = stops[i]; t1, c1 = stops[i+1]
        mask = (gray >= t0) & (gray < t1)
        t = ((gray - t0) / (t1 - t0 + 1e-9))[mask]
        for ch in range(3):
            out[mask, ch] = c0[ch]*(1-t) + c1[ch]*t
    return to_img(out)

# ── 03. vhs_glitch ────────────────────────────────────────────────────────────
def vhs_glitch(img):
    """Horizontal scan-line tears + RGB channel shift."""
    arr = to_arr(img); h, w = arr.shape[:2]
    rng = np.random.default_rng(42)
    for _ in range(rng.integers(8, 20)):
        y = rng.integers(0, h); band = rng.integers(2, 12); shift = rng.integers(-40, 40)
        arr[y:min(y+band, h)] = np.roll(arr[y:min(y+band, h)], shift, axis=1)
    r = np.roll(arr[:,:,0],  8, axis=1)
    g = arr[:,:,1]
    b = np.roll(arr[:,:,2], -8, axis=1)
    return to_img(np.stack([r, g, b], axis=2))

# ── 04. chromatic_melt ────────────────────────────────────────────────────────
def chromatic_melt(img):
    """Heavy chromatic aberration with colour-burn blend."""
    arr = to_arr(img)
    shifted = np.stack([np.roll(arr[:,:,0], 15, axis=1),
                        arr[:,:,1],
                        np.roll(arr[:,:,2], -15, axis=0)], axis=2)
    burned = 255 - np.clip((255-arr)*255/(shifted+1), 0, 255)
    blend  = arr*0.4 + burned*0.6
    return saturate(to_img(blend), 3.0)

# ── 05. neon_edge_burn ────────────────────────────────────────────────────────
def neon_edge_burn(img):
    """Canny-style edges recoloured with a neon cyan/magenta LUT."""
    edges = ImageEnhance.Contrast(img.convert("L").filter(ImageFilter.FIND_EDGES)).enhance(4.0)
    e = np.asarray(edges, dtype=np.float32) / 255.0
    h, w = e.shape
    neon = np.stack([e*255, (1-e)*e*4*255, (1-e)*255], axis=2)
    return to_img(np.clip(to_arr(img)*0.25 + neon*1.5, 0, 255))

# ── 06. psychedelic_solarise ──────────────────────────────────────────────────
def psychedelic_solarise(img):
    """4-breakpoint solarisation creating vivid neon banding."""
    arr = to_arr(img)
    out = np.zeros_like(arr)
    for ch in range(3):
        d = arr[:,:,ch]
        out[:,:,ch] = np.where(d<64,  d*4,
                      np.where(d<128, (128-d)*4,
                      np.where(d<192, (d-128)*4, (255-d)*4)))
    return saturate(to_img(out), 3.5)

# ── 07. false_color_terrain ───────────────────────────────────────────────────
def false_color_terrain(img):
    """Topographic false-colour in 8 brightness bands."""
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    q = (gray*8).astype(int).clip(0, 7)
    pal = np.array([[0,0,128],[0,80,200],[0,180,80],[120,200,0],
                    [255,220,0],[255,120,0],[220,40,0],[180,0,180]], dtype=np.float32)
    return to_img(pal[q])

# ── 10. electric_sky ──────────────────────────────────────────────────────────
def electric_sky(img):
    """Apply the electric sky hue cycle to the full image."""
    arr = to_arr(img)
    electric = np.stack([arr[:,:,2], 255-arr[:,:,1], arr[:,:,0]], axis=2)
    return saturate(to_img(np.clip(electric * 1.4, 0, 255)), 2.2)

# ── 11. circuit_trace ─────────────────────────────────────────────────────────
def circuit_trace(img):
    """Emboss → green PCB traces on dark background."""
    emboss = np.asarray(img.convert("L").filter(ImageFilter.EMBOSS), dtype=np.float32)
    traces = np.clip((emboss-128)*3, 0, 255)
    h, w = traces.shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    out[:,:,1] = traces; out[:,:,0] = traces*0.1
    return to_img(np.clip(to_arr(img)*0.15 + out, 0, 255))

# ── 12. rainbow_scan ──────────────────────────────────────────────────────────
def rainbow_scan(img):
    """Horizontal HSV ramp scanlines multiplied into the image."""
    arr = to_arr(img); h, w = arr.shape[:2]
    def h2r(p, q, t):
        t = t%1.0
        if t<1/6: return p+(q-p)*6*t
        if t<1/2: return q
        if t<2/3: return p+(q-p)*(2/3-t)*6
        return p
    rainbow = np.array([[h2r(0.1,0.9,v+1/3), h2r(0.1,0.9,v), h2r(0.1,0.9,v-1/3)]
                        for v in np.linspace(0,1,h)], dtype=np.float32)*255
    scan = rainbow[:,np.newaxis,:]
    return to_img(arr*0.55 + scan*0.45)

# ── 13. uv_flood ──────────────────────────────────────────────────────────────
def uv_flood(img):
    """UV blacklight simulation: violet wash + bloom on bright regions."""
    arr = to_arr(img)
    uv = arr.copy()
    uv[:,:,0] = np.clip(arr[:,:,0]*1.2+40,  0, 255)
    uv[:,:,1] = arr[:,:,1]*0.3
    uv[:,:,2] = np.clip(arr[:,:,2]*1.5+60,  0, 255)
    bloom = to_arr(to_img(np.clip(uv*1.5,0,255)).filter(ImageFilter.GaussianBlur(10)))*0.4
    return to_img(np.clip(uv+bloom, 0, 255))

# ── 14. infrared_foliage ──────────────────────────────────────────────────────
def infrared_foliage(img):
    """Aerochrome channel swap – foliage turns bright pink/white."""
    arr = to_arr(img)
    result = np.stack([arr[:,:,1]*1.5, arr[:,:,0], arr[:,:,2]*0.5], axis=2)
    return saturate(to_img(result), 2.0)

# ── 15. glitch_datamosh ───────────────────────────────────────────────────────
def glitch_datamosh(img):
    """Block-level pixel displacement simulating video datamosh artifacts."""
    arr = to_arr(img); h, w = arr.shape[:2]; rng = np.random.default_rng(7)
    out = arr.copy(); bs = 24
    for by in range(0, h-bs, bs):
        for bx in range(0, w-bs, bs):
            if rng.random() < 0.25:
                sy = rng.integers(max(0,by-60), min(h-bs, by+60))
                sx = rng.integers(max(0,bx-60), min(w-bs, bx+60))
                out[by:by+bs, bx:bx+bs] = arr[sy:sy+bs, sx:sx+bs]
    return to_img(out)

# ── 16. negative_burn ─────────────────────────────────────────────────────────
def negative_burn(img):
    """Inversion + hard-light self-blend for deep colour burn."""
    arr = to_arr(img)
    inv = 255 - arr
    result = np.where(inv < 128,
                      2*arr*inv/255.0,
                      255 - 2*(255-arr)*(255-inv)/255.0)
    blended = arr*0.3 + result*0.7
    return saturate(to_img(blended), 2.8)

# ── 17. holographic_sheen ─────────────────────────────────────────────────────
def holographic_sheen(img):
    """Iridescent diagonal gradient × multiply blend."""
    arr = to_arr(img); h, w = arr.shape[:2]
    xv, yv = np.meshgrid(np.linspace(0,1,w), np.linspace(0,1,h))
    t = (xv*0.5 + yv*0.5) % 1.0
    holo = np.stack([
        (np.sin(t*2*math.pi)*0.5+0.5)*255,
        (np.sin(t*2*math.pi+2*math.pi/3)*0.5+0.5)*255,
        (np.sin(t*2*math.pi+4*math.pi/3)*0.5+0.5)*255], axis=2)
    result = arr*holo/255.0
    return saturate(to_img(arr*0.4+result*0.6), 2.0)

# ── 18. neural_dissolve ───────────────────────────────────────────────────────
def neural_dissolve(img):
    """Multi-scale frequency decomposition, each band false-coloured."""
    pil = img.convert("RGB"); arr = to_arr(pil)
    b1  = to_arr(pil.filter(ImageFilter.GaussianBlur(2)))
    b2  = to_arr(pil.filter(ImageFilter.GaussianBlur(8)))
    b3  = to_arr(pil.filter(ImageFilter.GaussianBlur(24)))
    lum = lambda b: np.abs(b).mean(axis=2)
    h, w = arr.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.float32)
    out[:,:,0] = lum(arr-b1)*3
    out[:,:,1] = lum(b1-b2)*3
    out[:,:,2] = lum(b2-b3)*4 + 40
    return saturate(to_img(np.clip(arr*0.2+out, 0, 255)), 3.0)

# ── 19. plasma_interference ───────────────────────────────────────────────────
def plasma_interference(img):
    """Classic demoscene sine-wave plasma screen-blended into the image."""
    arr = to_arr(img); h, w = arr.shape[:2]
    xv, yv = np.meshgrid(np.linspace(0,4*math.pi,w), np.linspace(0,4*math.pi,h))
    p = (np.sin(xv)+np.sin(yv)+np.sin(xv*0.5+yv*0.5)+np.sin(np.sqrt(xv**2+yv**2+1e-9)))/4.0
    p = p*0.5+0.5
    plasma = np.stack([
        (np.sin(p*2*math.pi)*0.5+0.5)*255,
        (np.sin(p*2*math.pi+2*math.pi/3)*0.5+0.5)*255,
        (np.sin(p*2*math.pi+4*math.pi/3)*0.5+0.5)*255], axis=2)
    result = 255 - (255-arr)*(255-plasma)/255
    return saturate(to_img(arr*0.35+result*0.65), 2.4)

# ── 22. cobalt_overload ───────────────────────────────────────────────────────
def cobalt_overload(img):
    """Everything crushed into electric cobalt-blue + white flare blooms."""
    arr = to_arr(img)
    lum = arr.mean(axis=2, keepdims=True)
    out = np.stack([
        np.clip(lum[:,:,0]*0.3,      0, 255),
        np.clip(lum[:,:,0]*0.5,      0, 255),
        np.clip(lum[:,:,0]*1.8+60,   0, 255)], axis=2)
    bloom = to_arr(to_img(out).filter(ImageFilter.GaussianBlur(14)))*0.5
    return to_img(np.clip(out+bloom, 0, 255))

# ── 23. lime_poison ───────────────────────────────────────────────────────────
def lime_poison(img):
    """Toxic neon-green chemical spill aesthetic."""
    arr = to_arr(img)
    out = np.stack([
        np.clip(255-arr[:,:,0],  0, 255),
        np.clip(arr[:,:,1]*2.0,  0, 255),
        np.clip(arr[:,:,2]*0.15, 0, 255)], axis=2)
    g = out[:,:,1]
    out[:,:,1] = np.where(g > 180, 255-g, g)
    return saturate(to_img(out), 3.0)

# ── 24. xray_film ─────────────────────────────────────────────────────────────
def xray_film(img):
    """Classic X-ray: inverted blue-tinted greyscale with edge glow."""
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    inv  = 255 - gray
    r = inv*0.6; g = inv*0.75; b = np.clip(inv*1.1+20, 0, 255)
    xray = np.stack([r, g, b], axis=2)
    edges = np.asarray(img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    xray[:,:,2] = np.clip(xray[:,:,2]+edges*0.8, 0, 255)
    return to_img(xray)

# ── 25. lava_flow ─────────────────────────────────────────────────────────────
def lava_flow(img):
    """Molten rock: black-to-orange-to-white heat gradient + sine distortion."""
    arr = to_arr(img); h, w = arr.shape[:2]
    shift = (np.sin(np.arange(h)*0.05)*12).astype(int)
    warped = np.zeros_like(arr)
    for y in range(h):
        warped[y] = np.roll(arr[y], shift[y], axis=0)
    lum = warped.mean(axis=2)
    r = np.clip(lum*1.5,        0, 255)
    g = np.clip((lum-80)*0.8,   0, 255)
    b = np.clip((lum-180)*0.5,  0, 255)
    return to_img(np.stack([r, g, b], axis=2))

# ── 26. oil_slick ─────────────────────────────────────────────────────────────
def oil_slick(img):
    """Iridescent thin-film interference – rainbow sheen on water."""
    arr = to_arr(img); h, w = arr.shape[:2]
    xv, yv = np.meshgrid(np.linspace(0,6*math.pi,w), np.linspace(0,6*math.pi,h))
    gray  = arr.mean(axis=2)/255.0
    phase = gray*math.pi + xv*0.3 + yv*0.3
    film  = np.stack([
        (np.sin(phase)*0.5+0.5)*255,
        (np.sin(phase+2*math.pi/3)*0.5+0.5)*255,
        (np.sin(phase+4*math.pi/3)*0.5+0.5)*255], axis=2)
    result = 255 - (255-arr)*(255-film)/255
    return saturate(to_img(arr*0.3+result*0.7), 2.5)

# ── 27. deep_sea ──────────────────────────────────────────────────────────────
def deep_sea(img):
    """Bioluminescent deep ocean: dark teal base with cyan/white glow blooms."""
    arr = to_arr(img); h, w = arr.shape[:2]
    out = np.stack([
        np.clip(arr[:,:,0]*0.1,                          0, 255),
        np.clip(arr[:,:,1]*0.6 + arr[:,:,2]*0.2,         0, 255),
        np.clip(arr[:,:,2]*0.8 + 30,                     0, 255)], axis=2)
    bright_mask = (arr.mean(axis=2) > 160).astype(np.float32)
    glow = np.stack([np.zeros((h,w)), bright_mask*200, bright_mask*255], axis=2)
    glow_blur = to_arr(to_img(glow).filter(ImageFilter.GaussianBlur(8)))
    return to_img(np.clip(out + glow_blur*0.8, 0, 255))

# ── 28. candy_chrome ──────────────────────────────────────────────────────────
def candy_chrome(img):
    """Hyper-saturated candy chrome: pastel brights + mirror specular."""
    arr = to_arr(img)
    r = np.clip(np.power(arr[:,:,0]/255, 0.5)*255*1.1, 0, 255)
    g = np.clip(np.power(arr[:,:,1]/255, 0.4)*255*1.2, 0, 255)
    b = np.clip(np.power(arr[:,:,2]/255, 0.6)*255*1.3, 0, 255)
    candy = np.stack([r, g, b], axis=2)
    lum  = arr.mean(axis=2)
    spec = np.clip((lum-180)*4, 0, 255)[:,:,np.newaxis]
    return saturate(to_img(np.clip(candy+spec, 0, 255)), 3.2)

# ── 29. crimson_noir ──────────────────────────────────────────────────────────
def crimson_noir(img):
    """High-contrast monochrome with selective blood-red channel burn."""
    arr = to_arr(img)
    gray = arr.mean(axis=2)
    contrast = np.clip((gray-128)*2.2+128, 0, 255)
    red_mask = ((arr[:,:,0]>arr[:,:,1]+30)&(arr[:,:,0]>arr[:,:,2]+30)).astype(np.float32)
    out = np.stack([
        np.clip(contrast + red_mask*120, 0, 255),
        np.clip(contrast - red_mask*80,  0, 255),
        np.clip(contrast - red_mask*100, 0, 255)], axis=2)
    return to_img(out)

# ── 30. aurora_bleed ──────────────────────────────────────────────────────────
def aurora_bleed(img):
    """Northern-lights vertical colour bands bleeding through the image."""
    arr = to_arr(img); h, w = arr.shape[:2]
    xv = np.linspace(0, 4*math.pi, w)
    aurora = np.stack([
        np.tile((np.sin(xv*0.8+0.5)*0.5+0.5)*60,  (h,1)),
        np.tile((np.sin(xv*1.3+1.0)*0.5+0.5)*200, (h,1)),
        np.tile((np.sin(xv*0.6+2.0)*0.5+0.5)*220, (h,1))], axis=2)
    ys = np.linspace(1, 0, h)[:,np.newaxis,np.newaxis]
    aurora = aurora * ys
    dark   = arr * 0.45
    result = 255 - (255-dark)*(255-aurora)/255
    return saturate(to_img(np.clip(result, 0, 255)), 2.0)


# ══════════════════════════════════════════════════════════════════════════════
#  REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
FILTERS = {
    "01_acid_palette":          acid_palette,
    "02_thermal_vision":        thermal_vision,
    "03_vhs_glitch":            vhs_glitch,
    "04_chromatic_melt":        chromatic_melt,
    "05_neon_edge_burn":        neon_edge_burn,
    "06_psychedelic_solarise":  psychedelic_solarise,
    "07_false_color_terrain":   false_color_terrain,
    "10_electric_sky":          electric_sky,
    "11_circuit_trace":         circuit_trace,
    "12_rainbow_scan":          rainbow_scan,
    "13_uv_flood":              uv_flood,
    "14_infrared_foliage":      infrared_foliage,
    "15_glitch_datamosh":       glitch_datamosh,
    "16_negative_burn":         negative_burn,
    "17_holographic_sheen":     holographic_sheen,
    "18_neural_dissolve":       neural_dissolve,
    "19_plasma_interference":   plasma_interference,
    "22_cobalt_overload":       cobalt_overload,
    "23_lime_poison":           lime_poison,
    "24_xray_film":             xray_film,
    "25_lava_flow":             lava_flow,
    "26_oil_slick":             oil_slick,
    "27_deep_sea":              deep_sea,
    "28_candy_chrome":          candy_chrome,
    "29_crimson_noir":          crimson_noir,
    "30_aurora_bleed":          aurora_bleed,
}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if not os.path.isdir(INPUT_FOLDER):
        print(f"[ERROR] Folder '{INPUT_FOLDER}' not found."); sys.exit(1)
    images = load_images(INPUT_FOLDER)
    if not images:
        print(f"[ERROR] No supported images in '{INPUT_FOLDER}'."); sys.exit(1)

    print(f"Found {len(images)} image(s)  ·  {len(FILTERS)} filters each\n")
    for img_path in images:
        print(f"▶  {img_path}")
        try:
            src = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [SKIP] {e}"); continue
        for name, fn in FILTERS.items():
            try:
                save(fn(src), img_path, name)
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")

    print(f"\nAll done  →  '{OUTPUT_FOLDER}/'")

if __name__ == "__main__":
    main()
