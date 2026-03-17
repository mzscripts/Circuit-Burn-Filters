"""
circuit_burn_filters_v3.py  -  40 MORE experimental colour filters
===================================================================
Drop images into  'wallter/'  next to this script.
Every filtered image is saved flat into  'circuitburn/'
with the filter name as the filename:

    circuitburn/<filtername>__<original_stem>.png

Dependencies:  pip install Pillow numpy
"""

import os, sys, glob, math, cmath
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

INPUT_FOLDER  = "jesse"
OUTPUT_FOLDER = "circuitburn_V3"
SUPPORTED     = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def load_images(folder):
    paths = []
    for ext in SUPPORTED:
        paths += glob.glob(os.path.join(folder, f"*{ext}"))
        paths += glob.glob(os.path.join(folder, f"*{ext.upper()}"))
    return sorted(set(paths))

def save(image, src_path, filter_name):
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    stem = Path(src_path).stem
    dest = Path(OUTPUT_FOLDER) / f"{filter_name}__{stem}.png"
    image.save(dest)
    print(f"  ok  {dest}")

# ── core helpers ───────────────────────────────────────────────────────────────
def a(src):   return np.asarray(src.convert("RGB"), dtype=np.float32)
def im(arr):  return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
def sat(p,f): return ImageEnhance.Color(p).enhance(f)
def bri(p,f): return ImageEnhance.Brightness(p).enhance(f)
def con(p,f): return ImageEnhance.Contrast(p).enhance(f)
def sharp(p,f): return ImageEnhance.Sharpness(p).enhance(f)

def lum(arr):
    return arr[:,:,0]*0.299 + arr[:,:,1]*0.587 + arr[:,:,2]*0.114

def mesh(h, w):
    return np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))

def edg(src):
    return np.asarray(src.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)

def gblur(arr_or_pil, r):
    if isinstance(arr_or_pil, np.ndarray):
        return np.asarray(im(arr_or_pil).filter(ImageFilter.GaussianBlur(r)), dtype=np.float32)
    return np.asarray(arr_or_pil.filter(ImageFilter.GaussianBlur(r)), dtype=np.float32)

def screen(x, y): return 255 - (255-x)*(255-y)/255.0
def multiply(x,y): return x*y/255.0
def dodge(x,y):   return np.clip(x*255.0/(255-y+1), 0, 255)
def burn(x,y):    return np.clip(255 - (255-x)*255.0/(y+1), 0, 255)

def hue_rotate_arr(arr, deg):
    """Rotate hue of float32 RGB array (0-255) by deg degrees."""
    angle = math.radians(deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    u = np.array([1,1,1])/math.sqrt(3)
    R = np.array([
        [cos_a+u[0]**2*(1-cos_a),        u[0]*u[1]*(1-cos_a)-u[2]*sin_a,  u[0]*u[2]*(1-cos_a)+u[1]*sin_a],
        [u[1]*u[0]*(1-cos_a)+u[2]*sin_a, cos_a+u[1]**2*(1-cos_a),         u[1]*u[2]*(1-cos_a)-u[0]*sin_a],
        [u[2]*u[0]*(1-cos_a)-u[1]*sin_a, u[2]*u[1]*(1-cos_a)+u[0]*sin_a,  cos_a+u[2]**2*(1-cos_a)]])
    flat = (arr/255.0).reshape(-1,3)
    return np.clip((flat @ R.T).reshape(arr.shape), 0, 1)*255


# ══════════════════════════════════════════════════════════════════════════════
#  40 FILTERS
# ══════════════════════════════════════════════════════════════════════════════

# ── 01. mandelbrot_dye ────────────────────────────────────────────────────────
def mandelbrot_dye(src):
    """Image luminance drives iteration depth into a Mandelbrot palette."""
    arr = a(src); h, w = arr.shape[:2]
    xv, yv = mesh(h, w)
    # map pixel coords to complex plane around the main cardioid
    cx = xv*3.0 - 2.2
    cy = yv*2.4 - 1.2
    z  = np.zeros((h, w), dtype=np.complex64)
    iters = np.zeros((h, w), dtype=np.float32)
    MAX = 24
    for n in range(MAX):
        mask = np.abs(z) <= 2
        z[mask] = z[mask]**2 + (cx+1j*cy)[mask]
        iters[mask] += 1
    t = iters / MAX
    # smooth palette: deep navy → electric blue → white → gold
    r = np.clip(t**0.5*200 + lum(arr)/255.0*55, 0, 255)
    g = np.clip(t**0.8*180 + lum(arr)/255.0*40, 0, 255)
    b = np.clip((1-t)**0.3*255 + t**2*220,      0, 255)
    # modulate with original image
    result = np.stack([r,g,b], axis=2)*0.65 + arr*0.35
    return sat(im(np.clip(result, 0, 255)), 2.4)

# ── 02. kelvin_shift ──────────────────────────────────────────────────────────
def kelvin_shift(src):
    """Cycle through 5 colour temperatures in horizontal thirds."""
    arr = a(src); h, w = arr.shape[:2]
    # colour temperature matrices (RGB multipliers per band)
    temps = [
        (0.5,  0.7,  2.5),   # 2000K tungsten-cold
        (1.3,  1.1,  0.4),   # 4000K warm studio
        (1.0,  1.0,  1.0),   # 6500K daylight
        (0.6,  1.0,  1.8),   # 9000K ice blue
        (1.5,  0.5,  1.4),   # ultraviolet magenta
    ]
    out = arr.copy()
    band = w // len(temps)
    for idx, (mr, mg, mb) in enumerate(temps):
        x0 = idx*band; x1 = (idx+1)*band if idx < len(temps)-1 else w
        out[:, x0:x1, 0] = np.clip(arr[:, x0:x1, 0]*mr, 0, 255)
        out[:, x0:x1, 1] = np.clip(arr[:, x0:x1, 1]*mg, 0, 255)
        out[:, x0:x1, 2] = np.clip(arr[:, x0:x1, 2]*mb, 0, 255)
    # feather band boundaries
    blur_out = gblur(out, 18)
    mask = np.zeros((h, w), dtype=np.float32)
    for idx in range(1, len(temps)):
        x = idx*band
        mask[:, max(0,x-20):min(w,x+20)] = 1.0
    mask3 = mask[:,:,np.newaxis]
    return sat(im(np.clip(out*(1-mask3) + blur_out*mask3, 0, 255)), 1.8)

# ── 03. liquid_nitrogen ───────────────────────────────────────────────────────
def liquid_nitrogen(src):
    """
    Alien infrared psychedelic forest: selective foliage inversion with surreal false-color.
    Detects green-dominant areas (trees/leaves), partially inverts them, remaps to:
    cyan-white, lavender, pale pink, toxic teal. Keeps trunks in plum/wine/indigo.
    Dark road anchor in deep violet-black. Bright sky gets cyan-white/pink bloom.
    Chromatic aberration, solarization, grain, magenta edge glow.
    """
    arr = a(src)
    h, w = arr.shape[:2]
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1. DETECT GREEN-DOMINANT AREAS (foliage/trees)
    # ─────────────────────────────────────────────────────────────────────────
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    green_excess = g - np.maximum(r, b)  # How much greener than R,B
    green_dominance = np.clip(green_excess / 256.0, 0, 1)  # Normalize to 0-1
    
    # Smooth the mask to avoid hard edges
    green_mask_smooth = np.zeros_like(green_dominance)
    for _ in range(2):  # Dilate slightly
        padded = np.pad(green_dominance, ((1,1),(1,1)), mode='edge')
        green_mask_smooth = np.maximum(green_mask_smooth,
            np.maximum(np.maximum(padded[:-2,:-2], padded[:-2,1:-1]), 
                       np.maximum(padded[:-2,2:], padded[1:-1,:-2])))
        green_dominance = green_mask_smooth
    green_dominance = np.clip(green_dominance, 0, 1)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2. SEPARATE INTO FOLIAGE & NON-FOLIAGE REGIONS
    # ─────────────────────────────────────────────────────────────────────────
    # Foliage mask: where green is dominant (threshold 0.35)
    foliage_mask = (green_dominance > 0.35).astype(np.float32)
    
    # Non-foliage: roads, ground, sky
    non_foliage_mask = 1.0 - foliage_mask
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3. PARTIALLY INVERT FOLIAGE (create surreal starting point)
    # ─────────────────────────────────────────────────────────────────────────
    arr_inverted = 255.0 - arr  # Full inversion
    
    # Blend between original and inverted based on green dominance (softer effect)
    invert_blend = 0.45  # How much inversion to apply to foliage
    arr_foliage = arr * (1.0 - foliage_mask[:,:,np.newaxis] * invert_blend) + \
                  arr_inverted * (foliage_mask[:,:,np.newaxis] * invert_blend)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4. FALSE-COLOR REMAP FOR FOLIAGE: cyan-white, lavender, pink, teal
    # ─────────────────────────────────────────────────────────────────────────
    gray_foliage = lum(arr_foliage) / 255.0
    
    # Create surreal false-color palette for foliage
    # Dark greens → lavender/plum
    # Midtone greens → pale pink / toxic teal
    # Bright greens → cyan-white
    foliage_out = np.zeros_like(arr_foliage)
    
    # Shadows (dark foliage) → deep plum/indigo
    shadow_mask_f = (gray_foliage < 0.35).astype(np.float32)
    foliage_out += shadow_mask_f[:,:,np.newaxis] * np.array([80, 40, 120]) * (gray_foliage < 0.35)[:,:,np.newaxis]
    
    # Midtones → lavender blended with toxic teal
    midtone_mask_f = ((gray_foliage >= 0.35) & (gray_foliage < 0.65)).astype(np.float32)
    t_mid = (gray_foliage - 0.35) / 0.30  # Remap 0.35-0.65 to 0-1
    lavender = np.array([180, 120, 200])
    teal_toxic = np.array([20, 240, 200])
    midtone_color = lavender * (1 - t_mid[:,:,np.newaxis]) + teal_toxic * t_mid[:,:,np.newaxis]
    foliage_out += midtone_mask_f[:,:,np.newaxis] * midtone_color
    
    # Highlights (bright foliage) → cyan-white & pale pink
    highlight_mask_f = (gray_foliage >= 0.65).astype(np.float32)
    t_high = (gray_foliage - 0.65) / 0.35  # Remap 0.65-1.0 to 0-1
    cyan_white = np.array([150, 255, 255])
    pale_pink = np.array([255, 200, 220])
    highlight_color = cyan_white * (1 - t_high[:,:,np.newaxis]) + pale_pink * t_high[:,:,np.newaxis]
    foliage_out += highlight_mask_f[:,:,np.newaxis] * highlight_color
    
    # ─────────────────────────────────────────────────────────────────────────
    # 5. NON-FOLIAGE HANDLING: darken roads/trunks in wine/plum/deep blue
    # ─────────────────────────────────────────────────────────────────────────
    gray_non = lum(arr) / 255.0
    non_foliage_out = arr.copy()
    
    # Trunks & dark areas (0-0.4) → wine/plum
    trunk_mask = (gray_non < 0.4).astype(np.float32)
    trunk_color = np.array([60, 20, 80])  # Wine/plum
    non_foliage_out += trunk_mask[:,:,np.newaxis] * (trunk_color - arr) * 0.6
    
    # Medium (0.4-0.65) → keep somewhat neutral but slightly cool
    mid_mask = ((gray_non >= 0.4) & (gray_non < 0.65)).astype(np.float32)
    cool_shift = np.array([-10, -5, 15])  # Boost blue slightly
    non_foliage_out += mid_mask[:,:,np.newaxis] * cool_shift * 0.3
    
    # Road/ground (0.65+) → deep blue-black or violet-black anchor
    dark_anchor_mask = (gray_non >= 0.65).astype(np.float32)
    anchor_color = np.array([40, 30, 60])  # Violet-black
    non_foliage_out = non_foliage_out * (1 - dark_anchor_mask[:,:,np.newaxis]*0.5) + \
                      anchor_color * (dark_anchor_mask[:,:,np.newaxis]*0.35)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 6. BLEND FOLIAGE & NON-FOLIAGE
    # ─────────────────────────────────────────────────────────────────────────
    arr = np.clip(
        foliage_out * foliage_mask[:,:,np.newaxis] + 
        non_foliage_out * non_foliage_mask[:,:,np.newaxis],
        0, 255
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # 7. BRIGHT SKY GAPS & HIGHLIGHTS: cyan-white / pink bloom
    # ─────────────────────────────────────────────────────────────────────────
    gray_all = lum(arr) / 255.0
    bright_mask = (gray_all > 0.7).astype(np.float32)
    sky_bloom = bright_mask[:,:,np.newaxis] * np.array([180, 255, 255])  # Cyan-white
    arr = np.clip(arr * 0.8 + sky_bloom * 0.2, 0, 255)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 8. MILD HIGHLIGHT SOLARIZATION (surreal pop)
    # ─────────────────────────────────────────────────────────────────────────
    solarize_mask = (gray_all > 0.6).astype(np.float32)
    solarized = 255.0 - arr
    arr = arr * (1 - solarize_mask[:,:,np.newaxis]*0.12) + solarized * (solarize_mask[:,:,np.newaxis]*0.12)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 9. SUBTLE CHROMATIC ABERRATION (edges only)
    # ─────────────────────────────────────────────────────────────────────────
    offset_x, offset_y = 2, 1
    r_ch = np.roll(np.roll(arr[:,:,0], offset_y, axis=0), offset_x, axis=1)
    b_ch = np.roll(np.roll(arr[:,:,2], -offset_y, axis=0), -offset_x, axis=1)
    edge_intensity = np.abs(gray_all - 0.5)
    edge_mask = np.clip(edge_intensity * 1.5, 0, 1)
    arr[:,:,0] = arr[:,:,0] * (1 - edge_mask*0.25) + r_ch * (edge_mask*0.25)
    arr[:,:,2] = arr[:,:,2] * (1 - edge_mask*0.25) + b_ch * (edge_mask*0.25)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 10. MAGENTA EDGE GLOW (subject preservation)
    # ─────────────────────────────────────────────────────────────────────────
    e = edg(src) / 255.0
    magenta_glow = np.zeros_like(arr)
    magenta_glow[:,:,0] = e * 255
    magenta_glow[:,:,1] = e * 80
    magenta_glow[:,:,2] = e * 180
    arr = np.clip(arr + magenta_glow * 0.10, 0, 255)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 11. FINE FILM GRAIN
    # ─────────────────────────────────────────────────────────────────────────
    grain = np.random.normal(0, 6, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + grain, 0, 255).astype(np.float32)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 12. VIGNETTE
    # ─────────────────────────────────────────────────────────────────────────
    yy, xx = np.meshgrid(np.arange(w), np.arange(h))
    cx, cy = w / 2, h / 2
    max_dist = math.sqrt(cx**2 + cy**2)
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2) / max_dist
    vignette = 1.0 - (dist / 1.4) ** 1.6
    vignette = np.clip(vignette, 0, 1)
    arr = np.clip(arr * (1.0 - 0.18 + 0.18 * vignette[:,:,np.newaxis]), 0, 255)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 13. FINAL CINEMATIC POLISH
    # ─────────────────────────────────────────────────────────────────────────
    result = im(arr)
    result = sat(result, 1.40)      # Boost saturation for surreal pop
    result = con(result, 1.20)      # Boost contrast
    result = sharp(result, 1.15)    # Slight sharpening for detail
    result = bri(result, 1.02)      # Slight brightness lift
    
    return result

# ── 04. venetian_fresco ───────────────────────────────────────────────────────
def venetian_fresco(src):
    """Renaissance fresco: earthy terracotta, verdigris, aged gold leaf."""
    arr = a(src); x = lum(arr)/255.0
    # terracotta base
    r = np.clip(x**0.55*210 + 30, 0, 255)
    g = np.clip(x**0.80*130 + 20, 0, 255)
    b = np.clip(x**1.20*70,        0, 255)
    base = np.stack([r,g,b], axis=2)
    # verdigris in shadows
    shadow = (1-x)**2
    base[:,:,0] = np.clip(base[:,:,0] - shadow*80, 0, 255)
    base[:,:,1] = np.clip(base[:,:,1] + shadow*60, 0, 255)
    base[:,:,2] = np.clip(base[:,:,2] + shadow*50, 0, 255)
    # gold-leaf glint on edges
    e = edg(src)/255.0
    base[:,:,0] = np.clip(base[:,:,0] + e*120, 0, 255)
    base[:,:,1] = np.clip(base[:,:,1] + e*100, 0, 255)
    return con(im(base), 1.4)

# ── 06. sepia_overdrive ───────────────────────────────────────────────────────
def sepia_overdrive(src):
    """Classic sepia pushed through a colour-dodge explosion into amber fire."""
    arr = a(src); x = lum(arr)/255.0
    # sepia
    sr = np.clip(x*255*1.08, 0, 255)
    sg = np.clip(x*255*0.85, 0, 255)
    sb = np.clip(x*255*0.66, 0, 255)
    sepia = np.stack([sr,sg,sb], axis=2)
    # dodge original red channel into sepia
    dodged = dodge(sepia, arr*np.array([1,0.5,0.2]))
    return con(sat(im(np.clip(dodged, 0, 255)), 2.0), 1.5)

# ── 07. aurora_column ─────────────────────────────────────────────────────────
def aurora_column(src):
    """Auroral column discharge: vertical curtains of teal+violet plasma."""
    arr = a(src); h, w = arr.shape[:2]
    xv, yv = mesh(h, w)
    rng = np.random.default_rng(77)
    # turbulent curtains
    turbulence = np.zeros((h, w), dtype=np.float32)
    for freq in [3, 7, 14]:
        phase = rng.uniform(0, 2*math.pi)
        turbulence += np.sin(xv*freq*math.pi*2 + phase + yv*freq*0.5)*0.5+0.5
    turbulence /= 3
    curtain_g = np.clip(turbulence * lum(arr)/255.0 * 2.5, 0, 1)*255
    curtain_b = np.clip((1-turbulence) * lum(arr)/255.0 * 2.0, 0, 1)*200
    curtain_r = np.clip(turbulence**3 * 180, 0, 255)
    dark = arr*0.15
    out  = dark + np.stack([curtain_r, curtain_g, curtain_b], axis=2)
    return sat(im(np.clip(out, 0, 255)), 2.8)

# ── 09. polaroid_melt ─────────────────────────────────────────────────────────
def polaroid_melt(src):
    """Polaroid film left in the sun: bleached pinks, cyan halos, bubbling."""
    arr = a(src); h, w = arr.shape[:2]
    # bleach: gamma lift everything
    bleached = np.power(arr/255.0, 0.55)*255
    # cyan-pink shift
    bleached[:,:,0] = np.clip(bleached[:,:,0]*1.1 + 20, 0, 255)
    bleached[:,:,1] = np.clip(bleached[:,:,1]*0.8,       0, 255)
    bleached[:,:,2] = np.clip(bleached[:,:,2]*1.25+ 15,  0, 255)
    # bubble distortion: random small warp blobs
    rng = np.random.default_rng(55)
    n_bubbles = 12
    ys_g, xs_g = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = np.zeros((h,w),dtype=np.float32); dy = np.zeros((h,w),dtype=np.float32)
    for _ in range(n_bubbles):
        bx = rng.uniform(0.1, 0.9)*w; by = rng.uniform(0.1, 0.9)*h
        br = rng.uniform(20, 80)
        d  = np.sqrt((xs_g-bx)**2+(ys_g-by)**2)
        strength = np.clip(1-d/br, 0, 1)*rng.uniform(8,20)
        dx += np.cos(d*0.3)*strength
        dy += np.sin(d*0.3)*strength
    xs_w = np.clip(xs_g+dx, 0, w-1).astype(int)
    ys_w = np.clip(ys_g+dy, 0, h-1).astype(int)
    warped = bleached[ys_w, xs_w]
    return sat(im(np.clip(warped, 0, 255)), 1.6)

# ── 10. deep_ultraviolet ──────────────────────────────────────────────────────
def deep_ultraviolet(src):
    """Germicidal UV lab: everything becomes ghostly purple-white on black."""
    arr = a(src); x = lum(arr)/255.0
    # UV simulation: only high-luminance areas survive, tinted violet
    uv = np.power(x, 1.8)
    r  = np.clip(uv*200 + 30, 0, 255)
    g  = np.clip(uv*80,        0, 255)
    b  = np.clip(uv*255 + 50,  0, 255)
    # flicker: random column brightening
    rng = np.random.default_rng(9)
    h_i, w_i = arr.shape[:2]
    col_flicker = rng.uniform(0.7, 1.3, w_i).astype(np.float32)
    r = np.clip(r * col_flicker, 0, 255)
    b = np.clip(b * col_flicker, 0, 255)
    return im(np.stack([r,g,b], axis=2))

# ── 11. chromakey_ghost ───────────────────────────────────────────────────────
def chromakey_ghost(src):
    """Green-screen ghost: green spill turned invisible, edges phantomised."""
    arr = a(src)
    # green dominance mask
    green_dom = np.clip((arr[:,:,1] - np.maximum(arr[:,:,0],arr[:,:,2]))/255.0, 0, 1)
    # replace green with its complement (magenta-ish)
    comp = 255 - arr
    out  = arr*(1-green_dom[:,:,np.newaxis]) + comp*green_dom[:,:,np.newaxis]
    # edge phantom: edges become translucent white
    e = edg(src)[:,:,np.newaxis]/255.0
    ghost = gblur(out, 6)*e*1.5
    return sat(im(np.clip(out*0.7 + ghost, 0, 255)), 2.2)

# ── 12. thermal_gradient_v2 ───────────────────────────────────────────────────
def thermal_gradient_v2(src):
    """FLIR-style but remapped to an alien violet-white-teal palette."""
    arr = a(src); x = lum(arr)/255.0
    stops = [
        (0.00, (10,  0,  30)),
        (0.20, (80,  0, 120)),
        (0.45, (200, 0, 200)),
        (0.65, (255,255,255)),
        (0.80, (0, 230, 230)),
        (1.00, (0,  80, 160)),
    ]
    h_i, w_i = arr.shape[:2]
    out = np.zeros((h_i,w_i,3), dtype=np.float32)
    for i in range(len(stops)-1):
        t0,c0 = stops[i]; t1,c1 = stops[i+1]
        mask = (x>=t0)&(x<t1)
        t = ((x-t0)/(t1-t0+1e-9))[mask]
        for ch in range(3):
            out[mask,ch] = c0[ch]*(1-t)+c1[ch]*t
    return im(out)

# ── 13. riso_print ────────────────────────────────────────────────────────────
def riso_print(src):
    """Risograph 2-colour misregistered print: fluorescent pink + teal."""
    arr = a(src)
    gray = lum(arr)/255.0
    # channel 1: fluorescent pink
    rng = np.random.default_rng(21)
    h_i, w_i = arr.shape[:2]
    reg_x = rng.integers(-6,6); reg_y = rng.integers(-4,4)
    # halftone dot approximation (ordered dither)
    dither = np.indices((h_i,w_i))[0]%4/4.0 + np.indices((h_i,w_i))[1]%4/16.0
    halftone = (gray > dither).astype(np.float32)
    ch1 = halftone  # pink layer
    # channel 2: teal (shifted)
    ch2 = np.roll(np.roll(halftone, reg_x, axis=1), reg_y, axis=0)
    # compose
    r = np.clip(ch1*240 + (1-ch1-ch2).clip(0,1)*245, 0, 255)
    g = np.clip(ch2*190 + (1-ch1-ch2).clip(0,1)*235, 0, 255)
    b = np.clip(ch2*180 + ch1*30,                     0, 255)
    return im(np.stack([r,g,b],axis=2))

# ── 14. lenticular_shift ──────────────────────────────────────────────────────
def lenticular_shift(src):
    """Lenticular lens card: different hue every 4 pixels like angle-change."""
    arr = a(src); h, w = arr.shape[:2]
    out = arr.copy()
    angles = [0, 60, 120, 180, 240, 300]
    stripe_w = 6
    for col in range(0, w, stripe_w):
        angle = angles[(col // stripe_w) % len(angles)]
        x1 = min(col+stripe_w, w)
        out[:, col:x1] = hue_rotate_arr(arr[:, col:x1], angle)
    return sat(im(out), 2.5)

# ── 15. neutrino_scan ─────────────────────────────────────────────────────────
def neutrino_scan(src):
    """Cherenkov radiation detector: dark water, sparse blue-white tracks."""
    arr = a(src); h, w = arr.shape[:2]
    x = lum(arr)/255.0
    # near-black dark water
    water = np.stack([x*15, x*20, x*35], axis=2)
    # Cherenkov tracks: bright edges as particle tracks
    e = edg(src)/255.0
    cone_angle = np.sin(e*math.pi)
    track_r = np.clip(cone_angle*80,  0, 255)
    track_g = np.clip(cone_angle*180, 0, 255)
    track_b = np.clip(cone_angle*255, 0, 255)
    tracks  = np.stack([track_r,track_g,track_b], axis=2)
    bloom   = gblur(tracks, 5)*0.6
    return im(np.clip(water + tracks + bloom, 0, 255))

# ── 17. neon_sashimi ──────────────────────────────────────────────────────────
def neon_sashimi(src):
    """Japanese izakaya: salmon-pink, electric indigo, wasabi green slices."""
    arr = a(src)
    # slice image into horizontal thirds and tint each
    h, w = arr.shape[:2]
    t1, t2 = h//3, 2*h//3
    out = arr.copy()
    # salmon top
    out[:t1,:,0] = np.clip(arr[:t1,:,0]*1.3+40, 0,255)
    out[:t1,:,1] = np.clip(arr[:t1,:,1]*0.6,    0,255)
    out[:t1,:,2] = np.clip(arr[:t1,:,2]*0.7+20, 0,255)
    # indigo middle
    out[t1:t2,:,0] = np.clip(arr[t1:t2,:,0]*0.4+20,  0,255)
    out[t1:t2,:,1] = np.clip(arr[t1:t2,:,1]*0.3,      0,255)
    out[t1:t2,:,2] = np.clip(arr[t1:t2,:,2]*1.8+60,   0,255)
    # wasabi bottom
    out[t2:,:,0] = np.clip(arr[t2:,:,0]*0.3,     0,255)
    out[t2:,:,1] = np.clip(arr[t2:,:,1]*1.7+50,  0,255)
    out[t2:,:,2] = np.clip(arr[t2:,:,2]*0.2,      0,255)
    # feather boundary seams
    seam_blur = gblur(out, 30)
    for boundary in [t1, t2]:
        mask = np.zeros((h,w,3),dtype=np.float32)
        mask[max(0,boundary-25):boundary+25] = 1.0
        out = out*(1-mask) + seam_blur*mask
    return sat(im(np.clip(out,0,255)), 2.4)

# ── 18. electron_microscope ───────────────────────────────────────────────────
def electron_microscope(src):
    """SEM false-colour: metallic gold-green on pitch black substrate."""
    arr = a(src)
    # scanning electron look: heavy edge emphasis, nearly black base
    e = edg(src)/255.0
    x = lum(arr)/255.0
    # topography from unsharp-mask
    sharp_arr = np.asarray(src.convert("RGB").filter(ImageFilter.UnsharpMask(2,150,3)), dtype=np.float32)
    topo = lum(sharp_arr)/255.0
    r = np.clip(topo**0.5*200 + e*100, 0, 255)
    g = np.clip(topo**0.6*230 + e*60,  0, 255)
    b = np.clip(topo**2.0*60,           0, 255)
    dark_mask = (1 - topo**0.4)[:,:,np.newaxis]
    out = np.stack([r,g,b],axis=2) * (1-dark_mask*0.85)
    return im(np.clip(out, 0, 255))

# ── 19. color_channel_tornado ─────────────────────────────────────────────────
def color_channel_tornado(src):
    """Each colour channel rotated by a different angle around image centre."""
    arr = a(src); h, w = arr.shape[:2]
    cy, cx = h/2, w/2
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.zeros_like(arr)
    for ch, twist_deg in enumerate([15, -10, 20]):
        twist = math.radians(twist_deg)
        # polar coords
        rx = xs - cx; ry = ys - cy
        r  = np.sqrt(rx**2 + ry**2)
        theta = np.arctan2(ry, rx) + twist * (r / (max(cx,cy)+1))
        # sample back
        sx = np.clip((r*np.cos(theta) + cx).astype(int), 0, w-1)
        sy = np.clip((r*np.sin(theta) + cy).astype(int), 0, h-1)
        out[:,:,ch] = arr[sy, sx, ch]
    return sat(im(out), 2.2)

# ── 21. short_circuit ─────────────────────────────────────────────────────────
def short_circuit(src):
    """Electrical arc discharge: white-hot arc paths, char black, ozone blue."""
    arr = a(src); h, w = arr.shape[:2]
    rng = np.random.default_rng(66)
    # arc paths: random walk branching from top
    arc_map = np.zeros((h,w), dtype=np.float32)
    for _ in range(8):
        x_pos = rng.integers(w//4, 3*w//4); y_pos = 0
        for step in range(h):
            x_pos = int(np.clip(x_pos + rng.integers(-3,4), 0, w-1))
            y_pos = int(np.clip(y_pos + 1, 0, h-1))
            arc_map[y_pos, x_pos] = 1.0
            # branch probability
            if rng.random() < 0.08:
                bx,by = x_pos,y_pos
                for _ in range(rng.integers(5,25)):
                    bx = int(np.clip(bx+rng.integers(-3,4),0,w-1))
                    by = int(np.clip(by+rng.integers(-1,3),0,h-1))
                    arc_map[by,bx] = 0.7
    arc_blur = gblur(arc_map[:,:,np.newaxis]*np.ones((1,1,3))*255, 3)
    char = arr*0.05
    ozone = arr*np.array([0.1,0.2,0.6])
    out = char + ozone + arc_blur*1.5
    return im(np.clip(out, 0, 255))

# ── 23. pollen_cloud ──────────────────────────────────────────────────────────
def pollen_cloud(src):
    """Allergenic pollen dispersal: golden-yellow soft clouds with spore dots."""
    arr = a(src); h, w = arr.shape[:2]
    rng = np.random.default_rng(33)
    x = lum(arr)/255.0
    # soft golden base
    r = np.clip(x**0.5*255*1.1+10, 0,255)
    g = np.clip(x**0.6*220,         0,255)
    b = np.clip(x**2.0*60,          0,255)
    base = np.stack([r,g,b],axis=2)
    # spore dots: random bright golden spots
    spores = np.zeros((h,w,3),dtype=np.float32)
    n_spores = 300
    sx = rng.integers(0, w, n_spores)
    sy = rng.integers(0, h, n_spores)
    for px,py in zip(sx,sy):
        y0,y1 = max(0,py-3),min(h,py+4)
        x0,x1 = max(0,px-3),min(w,px+4)
        spores[y0:y1,x0:x1] = [255, 210, 0]
    spore_blur = gblur(spores, 4)*0.8
    bloom = gblur(base, 16)*0.3
    return sat(im(np.clip(base+spore_blur+bloom,0,255)), 2.2)

# ── 24. mycelium_network ──────────────────────────────────────────────────────
def mycelium_network(src):
    """Fungal mycelium: dark earth tones, white thread network, spore glow."""
    arr = a(src)
    e = edg(src)/255.0
    x = lum(arr)/255.0
    # dark earth base: warm brown/black
    r = np.clip(x**1.5*120 + 15, 0,255)
    g = np.clip(x**1.8*80  + 8,  0,255)
    b = np.clip(x**2.5*40,        0,255)
    base = np.stack([r,g,b],axis=2)
    # white mycelium threads follow edges
    threads = e[:,:,np.newaxis]*np.array([230,240,220])
    thread_bloom = gblur(threads, 3)*0.5
    # spore glow: soft orange-yellow bloom on bright spots
    spore_mask = np.clip((x-0.7)*5,0,1)
    spore_glow = gblur(np.stack([spore_mask*255,spore_mask*160,spore_mask*20],axis=2),10)*0.6
    return im(np.clip(base+threads+thread_bloom+spore_glow, 0,255))

# ── 25. polarisation_filter ───────────────────────────────────────────────────
def polarisation_filter(src):
    """Cross-polarised light through birefringent crystal: vivid stress colours."""
    arr = a(src); h, w = arr.shape[:2]
    xv, yv = mesh(h, w)
    x = lum(arr)/255.0
    # birefringence pattern: retardation = thickness * function of position
    retard = x * (np.sin(xv*math.pi*6)*np.cos(yv*math.pi*4)*0.5+0.5)
    r = (np.sin(retard*2*math.pi)*0.5+0.5)*255
    g = (np.sin(retard*2*math.pi+2.09)*0.5+0.5)*255
    b = (np.sin(retard*2*math.pi+4.19)*0.5+0.5)*255
    # cross-pol extinction at 0 and pi: darken those
    extinction = np.sin(retard*math.pi)**2
    out = np.stack([r,g,b],axis=2)*extinction[:,:,np.newaxis]
    return sat(im(np.clip(out,0,255)), 3.2)

# ── 26. oxidised_neon_tube ────────────────────────────────────────────────────
def oxidised_neon_tube(src):
    """Old neon sign with failing gas: patches of red, white, dark gaps."""
    arr = a(src); h, w = arr.shape[:2]
    rng = np.random.default_rng(16)
    x = lum(arr)/255.0
    # base neon red-orange
    r = np.clip(x**0.4*255, 0,255)
    g = np.clip(x**2.0*80,  0,255)
    b = np.clip(x**3.0*20,  0,255)
    base = np.stack([r,g,b],axis=2)
    # failing patches: random rectangles go dark or white
    for _ in range(12):
        ry = rng.integers(0,h-20); rx = rng.integers(0,w-30)
        rh = rng.integers(8,30);   rw = rng.integers(15,60)
        mode = rng.choice(['dark','white','flicker'])
        if mode=='dark':
            base[ry:ry+rh,rx:rx+rw] *= 0.05
        elif mode=='white':
            base[ry:ry+rh,rx:rx+rw] = np.clip(base[ry:ry+rh,rx:rx+rw]*3+100,0,255)
        else:
            base[ry:ry+rh,rx:rx+rw] = rng.choice([base[ry:ry+rh,rx:rx+rw]*0.1,
                                                    base[ry:ry+rh,rx:rx+rw]*2.5+50])
    return im(np.clip(base,0,255))

# ── 27. geological_core ───────────────────────────────────────────────────────
def geological_core(src):
    """Drill core sample: horizontal sediment strata in earthy mineral tones."""
    arr = a(src); h, w = arr.shape[:2]
    rng = np.random.default_rng(7)
    x = lum(arr)/255.0
    # strata: each horizontal band gets a random mineral colour
    n_strata = 18
    boundaries = np.sort(rng.integers(0, h, n_strata))
    mineral_colours = [
        (180,140,90),(210,160,70),(90,120,80),(160,80,60),(200,200,180),
        (120,90,70),(80,100,130),(200,170,100),(60,80,60),(220,180,130),
        (100,60,50),(180,200,160),(150,110,80),(70,90,110),(190,150,80),
        (110,130,90),(200,80,60),(160,160,140),
    ]
    out = arr.copy()
    prev = 0
    for idx, boundary in enumerate(boundaries):
        if boundary <= prev:
            continue
        mr,mg,mb = mineral_colours[idx % len(mineral_colours)]
        band_lum = x[prev:boundary]
        out[prev:boundary,:,0] = np.clip(band_lum*mr, 0,255)
        out[prev:boundary,:,1] = np.clip(band_lum*mg, 0,255)
        out[prev:boundary,:,2] = np.clip(band_lum*mb, 0,255)
        prev = boundary

    if prev < h:
        mr,mg,mb = mineral_colours[len(boundaries) % len(mineral_colours)]
        band_lum = x[prev:h]
        out[prev:h,:,0] = np.clip(band_lum*mr, 0,255)
        out[prev:h,:,1] = np.clip(band_lum*mg, 0,255)
        out[prev:h,:,2] = np.clip(band_lum*mb, 0,255)

    return con(im(np.clip(out,0,255)), 1.4)

# ── 29. gradient_map_noir ─────────────────────────────────────────────────────
def gradient_map_noir(src):
    """Gradient map: black→deep crimson→gold→cream, high contrast film look."""
    arr = a(src); x = lum(arr)/255.0
    stops = [
        (0.00, (5,   0,   0  )),
        (0.25, (100, 0,   20 )),
        (0.55, (180, 60,  0  )),
        (0.75, (220, 160, 20 )),
        (1.00, (255, 245, 220)),
    ]
    h_i,w_i = arr.shape[:2]
    out = np.zeros((h_i,w_i,3),dtype=np.float32)
    for i in range(len(stops)-1):
        t0,c0 = stops[i]; t1,c1 = stops[i+1]
        mask = (x>=t0)&(x<t1)
        t = ((x-t0)/(t1-t0+1e-9))[mask]
        for ch in range(3):
            out[mask,ch] = c0[ch]*(1-t)+c1[ch]*t
    return con(im(out), 1.6)

# ── 30. satellite_false_color ─────────────────────────────────────────────────
def satellite_false_color(src):
    """Landsat false-colour composite: vegetation→red, water→black, urban→cyan."""
    arr = a(src)
    # proxy bands from RGB
    nir  = arr[:,:,1]           # use green as NIR proxy → makes veg red
    red  = arr[:,:,0]
    blue = arr[:,:,2]
    # false colour: R=NIR, G=Red, B=Blue
    r = np.clip(nir*1.3,       0,255)
    g = np.clip(red*0.8,       0,255)
    b = np.clip(blue*1.0+20,   0,255)
    # water mask (dark + blue dominant) → deep black-blue
    water = ((arr[:,:,2]>arr[:,:,0]+20)&(lum(arr)<100)).astype(np.float32)
    b = np.clip(b - water*80, 0,255)
    r = np.clip(r - water*60, 0,255)
    return sat(im(np.stack([r,g,b],axis=2)), 2.0)

# ── 31. glassblower ───────────────────────────────────────────────────────────
def glassblower(src):
    """Molten borosilicate glass: transparent amber, trapped bubble caustics."""
    arr = a(src); h, w = arr.shape[:2]
    x = lum(arr)/255.0
    # amber glass base
    r = np.clip(x**0.5*240+15,  0,255)
    g = np.clip(x**0.7*180+10,  0,255)
    b = np.clip(x**1.5*60,       0,255)
    base = np.stack([r,g,b],axis=2)
    # caustic light: bright spots where glass is thinnest
    caustic = np.asarray(src.convert("RGB").filter(ImageFilter.UnsharpMask(8,200,3)),dtype=np.float32)
    caustic_lum = lum(caustic)/255.0
    bright_caustic = np.clip((caustic_lum-0.7)*5,0,1)
    shine = bright_caustic[:,:,np.newaxis]*np.array([255,255,200])*0.8
    # interior refraction ripple
    ripple_phase = x*math.pi*8
    ripple = (np.sin(ripple_phase)*0.5+0.5)[:,:,np.newaxis]*20
    return sat(im(np.clip(base+shine+ripple,0,255)), 1.8)

# ── 33. refraction_pool ───────────────────────────────────────────────────────
def refraction_pool(src):
    """Underwater caustic refraction: wavy bright lines on teal floor."""
    arr = a(src); h, w = arr.shape[:2]
    xv, yv = mesh(h, w)
    # caustic warp field
    warp_x = np.sin(xv*12*math.pi + yv*7*math.pi)*0.015*w
    warp_y = np.cos(xv*9*math.pi  - yv*5*math.pi)*0.015*h
    ys_g, xs_g = np.mgrid[0:h, 0:w].astype(np.float32)
    xs_w = np.clip(xs_g+warp_x, 0, w-1).astype(int)
    ys_w = np.clip(ys_g+warp_y, 0, h-1).astype(int)
    warped = arr[ys_w, xs_w]
    # teal water tint
    warped[:,:,0] = np.clip(warped[:,:,0]*0.4+10, 0,255)
    warped[:,:,1] = np.clip(warped[:,:,1]*0.9+30, 0,255)
    warped[:,:,2] = np.clip(warped[:,:,2]*1.3+40, 0,255)
    # bright caustic lines
    caustics = (np.sin(xv*25*math.pi+yv*18*math.pi)*0.5+0.5)**6*200
    warped[:,:,1] = np.clip(warped[:,:,1]+caustics,0,255)
    warped[:,:,2] = np.clip(warped[:,:,2]+caustics*0.7,0,255)
    return sat(im(warped),1.8)

# ── 34. neon_rain_japan ───────────────────────────────────────────────────────
def neon_rain_japan(src):
    """Shinjuku night rain: vertical neon reflections, pink kanji glow."""
    arr = a(src); h, w = arr.shape[:2]
    # vertical smear: compress columns by blending with vertically blurred
    vblur = np.asarray(src.convert("RGB").filter(ImageFilter.GaussianBlur(0)), dtype=np.float32)
    # custom vertical-only blur via numpy
    kernel_size = 30
    vblur_v = np.zeros_like(arr)
    for dy in range(-kernel_size, kernel_size+1):
        weight = math.exp(-abs(dy)/8.0)
        vblur_v += np.roll(arr, dy, axis=0)*weight
    vblur_v /= vblur_v.max()/255.0+1e-9
    # blend: lower half gets more vertical smear (puddle reflection)
    alpha = np.linspace(0,1,h)[:,np.newaxis,np.newaxis]**2
    blended = arr*(1-alpha) + vblur_v*alpha
    # neon channel tint: boost pink/cyan
    blended[:,:,0] = np.clip(blended[:,:,0]*1.2+20, 0,255)
    blended[:,:,2] = np.clip(blended[:,:,2]*1.4+30, 0,255)
    blended[:,:,1] = np.clip(blended[:,:,1]*0.7,    0,255)
    return sat(im(np.clip(blended,0,255)), 2.6)

# ── 35. chromatic_spiral ──────────────────────────────────────────────────────
def chromatic_spiral(src):
    """Logarithmic spiral hue rotation: image twisted into colour vortex."""
    arr = a(src); h, w = arr.shape[:2]
    cy, cx = h/2, w/2
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    rx = xs-cx; ry = ys-cy
    r_dist = np.sqrt(rx**2+ry**2)/(max(cx,cy)+1)
    theta = np.arctan2(ry,rx)
    # spiral: hue shift = log(r) * 180deg
    hue_shift = np.log1p(r_dist*3)*90
    out = arr.copy().astype(np.float32)
    # rotate each pixel's colour by its position-dependent hue
    for deg_val in np.unique(hue_shift.astype(int)):
        mask = hue_shift.astype(int) == deg_val
        if mask.any():
            out[mask] = hue_rotate_arr(arr[mask].reshape(1,-1,3), float(deg_val)).reshape(-1,3)
    return sat(im(np.clip(out,0,255)), 2.6)

# ── 36. analogue_xerox ────────────────────────────────────────────────────────
def analogue_xerox(src):
    """80s photocopier: blown-out whites, crushed blacks, cyan toner bleeding."""
    arr = a(src)
    x = lum(arr)/255.0
    # aggressive threshold-like contrast
    harsh = np.where(x>0.5, np.power((x-0.5)*2,0.3), np.power(x*2,3)*0.5)
    r = np.clip(harsh*220 + 10, 0,255)
    g = np.clip(harsh*230 + 15, 0,255)
    b = np.clip(harsh*255 + 30, 0,255)
    base = np.stack([r,g,b],axis=2)
    # cyan toner bleed on dark edges
    e = edg(src)/255.0
    base[:,:,1] = np.clip(base[:,:,1]+e*60, 0,255)
    base[:,:,2] = np.clip(base[:,:,2]+e*90, 0,255)
    base[:,:,0] = np.clip(base[:,:,0]-e*30, 0,255)
    # horizontal streaks (paper dust)
    rng = np.random.default_rng(19)
    h_i,w_i = arr.shape[:2]
    for _ in range(6):
        y = rng.integers(0,h_i); strength = rng.uniform(0.02,0.15)
        base[y:y+2,:] = np.clip(base[y:y+2,:]+strength*255,0,255)
    return im(np.clip(base,0,255))

# ── 37. spectrum_inversion ────────────────────────────────────────────────────
def spectrum_inversion(src):
    """Each pixel's hue inverted (rotated 180°) while keeping luminance."""
    arr = a(src)
    rotated = hue_rotate_arr(arr, 180)
    # preserve original luminance
    orig_lum = lum(arr)
    rot_lum  = lum(rotated)+1e-9
    scale    = (orig_lum/rot_lum)[:,:,np.newaxis]
    result   = rotated*scale
    return sat(im(np.clip(result,0,255)), 2.0)

# ── 38. magnetic_field_lines ──────────────────────────────────────────────────
def magnetic_field_lines(src):
    """Iron-filing field lines around image as dipole: gold on deep indigo."""
    arr = a(src); h, w = arr.shape[:2]
    cy, cx = h/2, w/2
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32) - np.array([cy,cx])[:,np.newaxis,np.newaxis]
    r2 = xs**2+ys**2+1
    # dipole field: Bx, By
    Bx = 3*xs*ys/r2**2.5
    By = (2*ys**2-xs**2)/r2**2.5
    # magnitude and angle
    Bmag = np.sqrt(Bx**2+By**2)
    Bmag = Bmag/Bmag.max()
    Bangle = np.arctan2(By,Bx)/math.pi  # -1..1
    # map to colour: angle→hue, magnitude→brightness
    r = (np.sin(Bangle*math.pi)*0.5+0.5)*Bmag*255*1.5
    g = (np.sin(Bangle*math.pi+2.09)*0.5+0.5)*Bmag*180
    b = Bmag*255*0.8 + 30
    field = np.stack([r,g,b],axis=2)
    dark  = arr*0.2
    return sat(im(np.clip(dark+field,0,255)), 2.8)

# ── 39. expired_slide_film ────────────────────────────────────────────────────
def expired_slide_film(src):
    """Kodachrome left in a hot car: magenta fog, cyan shadows, yellow grain."""
    arr = a(src); h, w = arr.shape[:2]
    rng = np.random.default_rng(58)
    x = lum(arr)/255.0
    # colour fog: add magenta base
    r = np.clip(arr[:,:,0]*0.9 + 40,  0,255)
    g = np.clip(arr[:,:,1]*0.6 - 10,  0,255)
    b = np.clip(arr[:,:,2]*0.8 + 30,  0,255)
    # cyan shadow: where it's dark push cyan
    shadow = (1-x)**2
    g2 = np.clip(g + shadow*60, 0,255)
    b2 = np.clip(b + shadow*80, 0,255)
    # yellow grain
    grain = rng.standard_normal((h,w)).astype(np.float32)*12
    r3 = np.clip(r+grain,     0,255)
    g3 = np.clip(g2+grain*0.8,0,255)
    b3 = np.clip(b2-grain*0.3,0,255)
    # vignette
    cy2,cx2 = h/2,w/2
    ys2,xs2 = np.mgrid[0:h,0:w]
    vign = 1-np.clip(np.sqrt(((xs2-cx2)/cx2)**2+((ys2-cy2)/cy2)**2)*0.7,0,1)
    out  = np.stack([r3,g3,b3],axis=2)*vign[:,:,np.newaxis]
    return sat(im(np.clip(out,0,255)), 1.5)


# ══════════════════════════════════════════════════════════════════════════════
FILTERS = {
    "01_mandelbrot_dye":        mandelbrot_dye,
    "02_kelvin_shift":          kelvin_shift,
    "03_liquid_nitrogen":       liquid_nitrogen,
    "04_venetian_fresco":       venetian_fresco,
    "06_sepia_overdrive":       sepia_overdrive,
    "07_aurora_column":         aurora_column,
    "09_polaroid_melt":         polaroid_melt,
    "10_deep_ultraviolet":      deep_ultraviolet,
    "11_chromakey_ghost":       chromakey_ghost,
    "12_thermal_gradient_v2":   thermal_gradient_v2,
    "13_riso_print":            riso_print,
    "14_lenticular_shift":      lenticular_shift,
    "15_neutrino_scan":         neutrino_scan,
    "17_neon_sashimi":          neon_sashimi,
    "18_electron_microscope":   electron_microscope,
    "19_color_channel_tornado":  color_channel_tornado,
    "21_short_circuit":         short_circuit,
    "23_pollen_cloud":          pollen_cloud,
    "24_mycelium_network":      mycelium_network,
    "25_polarisation_filter":   polarisation_filter,
    "26_oxidised_neon_tube":    oxidised_neon_tube,
    "27_geological_core":       geological_core,
    "29_gradient_map_noir":     gradient_map_noir,
    "30_satellite_false_color": satellite_false_color,
    "31_glassblower":           glassblower,
    "33_refraction_pool":       refraction_pool,
    "34_neon_rain_japan":       neon_rain_japan,
    "35_chromatic_spiral":      chromatic_spiral,
    "36_analogue_xerox":        analogue_xerox,
    "37_spectrum_inversion":    spectrum_inversion,
    "38_magnetic_field_lines":  magnetic_field_lines,
    "39_expired_slide_film":    expired_slide_film,
}


def main():
    if not os.path.isdir(INPUT_FOLDER):
        print(f"[ERROR] Folder '{INPUT_FOLDER}' not found."); sys.exit(1)
    images = load_images(INPUT_FOLDER)
    if not images:
        print(f"[ERROR] No supported images in '{INPUT_FOLDER}'."); sys.exit(1)
    print(f"Found {len(images)} image(s)  x  {len(FILTERS)} filters")
    print(f"Output -> flat folder  '{OUTPUT_FOLDER}/'\n")
    for img_path in images:
        print(f">> {img_path}")
        try:
            src = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [SKIP] {e}"); continue
        for name, fn in FILTERS.items():
            try:
                save(fn(src), img_path, name)
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")
    print(f"\nDone -> '{OUTPUT_FOLDER}/'")

if __name__ == "__main__":
    main()
