"""
circuit_burn_filters_v4.py  -  31 filters reverse-engineered from reference images
====================================================================================
Drop images into  'wallter/'  next to this script.
Saved flat into  'circuitburn/'  as  <filtername>__<stem>.png

Dependencies:  pip install Pillow numpy scipy
               (scipy optional - graceful fallback if missing)
"""

import os, sys, glob, math
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw

INPUT_FOLDER  = "jesse"
OUTPUT_FOLDER = "circuitburn_V4"
SUPPORTED     = {".jpg",".jpeg",".png",".bmp",".tiff",".webp"}

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

def a(src):    return np.asarray(src.convert("RGB"), dtype=np.float32)
def im(arr):   return Image.fromarray(np.clip(arr,0,255).astype(np.uint8),"RGB")
def sat(p,f):  return ImageEnhance.Color(p).enhance(f)
def con(p,f):  return ImageEnhance.Contrast(p).enhance(f)
def bri(p,f):  return ImageEnhance.Brightness(p).enhance(f)

def lum(arr):
    return arr[:,:,0]*0.299 + arr[:,:,1]*0.587 + arr[:,:,2]*0.114

def mesh(h,w):
    return np.meshgrid(np.linspace(0,1,w), np.linspace(0,1,h))

def edg(src):
    return np.asarray(src.convert("L").filter(ImageFilter.FIND_EDGES),dtype=np.float32)

def gblur(x, r):
    if isinstance(x, np.ndarray):
        arr = np.clip(x, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return np.asarray(Image.fromarray(arr, "L").filter(ImageFilter.GaussianBlur(r)), dtype=np.float32)
        return np.asarray(im(arr).filter(ImageFilter.GaussianBlur(r)),dtype=np.float32)
    return np.asarray(x.filter(ImageFilter.GaussianBlur(r)),dtype=np.float32)

def screen(x,y):   return 255-(255-x)*(255-y)/255.0
def overlay(b,t):
    return np.where(b<128, 2*b*t/255.0, 255-2*(255-b)*(255-t)/255.0)

def hue_rot(arr, deg):
    angle = math.radians(deg)
    c,s = math.cos(angle), math.sin(angle)
    u = np.array([1,1,1])/math.sqrt(3)
    R = np.array([
        [c+u[0]**2*(1-c),     u[0]*u[1]*(1-c)-u[2]*s, u[0]*u[2]*(1-c)+u[1]*s],
        [u[1]*u[0]*(1-c)+u[2]*s, c+u[1]**2*(1-c),     u[1]*u[2]*(1-c)-u[0]*s],
        [u[2]*u[0]*(1-c)-u[1]*s, u[2]*u[1]*(1-c)+u[0]*s, c+u[2]**2*(1-c)]])
    flat = (arr/255.0).reshape(-1,3)
    return np.clip((flat@R.T).reshape(arr.shape),0,1)*255


# ------------------------------------------------------------------------------
# 31 FILTERS
# ------------------------------------------------------------------------------

# -- 01. open_sign_neon --------------------------------------------------------
# Ref img 1: OPEN neon sign — hot red/yellow/green shapes, deep black bg,
# flat colour pools like a solarised poster print
def open_sign_neon(src):
    arr = a(src)
    # Posterise into 5 flat colour levels
    x = lum(arr)/255.0
    levels = 5
    q = (x*levels).astype(int).clip(0,levels-1)
    # neon palette: black / deep-red / acid-yellow / hot-green / white
    pal = np.array([
        [5,   0,   5  ],
        [220, 10,  30 ],
        [255, 220, 0  ],
        [0,   255, 60 ],
        [255, 255, 240],
    ], dtype=np.float32)
    flat_col = pal[q]
    # overlay original hue: channel-swap to push colour
    arr2 = arr.copy()
    arr2[:,:,0] = np.clip(arr[:,:,0]*1.4,0,255)
    arr2[:,:,1] = np.clip(arr[:,:,1]*1.0,0,255)
    arr2[:,:,2] = np.clip(arr[:,:,2]*0.4,0,255)
    result = flat_col*0.6 + arr2*0.4
    return sat(im(np.clip(result,0,255)), 3.0)

# -- 02. infrared_foliage_rainbow ----------------------------------------------
# Ref img 2: Trees — vivid neon greens, hot pinks, rainbow sky arc
def infrared_foliage_rainbow(src):
    arr = a(src); h,w = arr.shape[:2]
    xv,yv = mesh(h,w)
    # Aerochrome channel swap
    r = np.clip(arr[:,:,1]*1.8,       0,255)
    g = np.clip(arr[:,:,0]*0.9,       0,255)
    b = np.clip(arr[:,:,2]*0.5+20,    0,255)
    base = np.stack([r,g,b],axis=2)
    # rainbow arc across top third
    arc_y  = yv < 0.45
    hue_x  = (xv*2*math.pi)
    arc_r  = (np.sin(hue_x)*0.5+0.5)*255*arc_y
    arc_g  = (np.sin(hue_x+2.09)*0.5+0.5)*255*arc_y
    arc_b  = (np.sin(hue_x+4.19)*0.5+0.5)*255*arc_y
    arc    = np.stack([arc_r,arc_g,arc_b],axis=2)
    result = screen(base, arc*0.8)
    return sat(im(np.clip(result,0,255)), 2.8)

# -- 03. rainbow_treeline ------------------------------------------------------
# Ref img 3: bare tree silhouette, wide rainbow banding behind it
def rainbow_treeline(src):
    arr = a(src); h,w = arr.shape[:2]
    xv,yv = mesh(h,w)
    # horizontal rainbow bands
    rainbow_r = (np.sin(yv*6*math.pi)*0.5+0.5)*255
    rainbow_g = (np.sin(yv*6*math.pi+2.09)*0.5+0.5)*255
    rainbow_b = (np.sin(yv*6*math.pi+4.19)*0.5+0.5)*255
    rainbow = np.stack([rainbow_r,rainbow_g,rainbow_b],axis=2)
    # silhouette: dark areas stay dark
    sil = (lum(arr)/255.0)[:,:,np.newaxis]
    result = rainbow*sil*0.7 + arr*0.3
    return sat(im(np.clip(result,0,255)), 2.5)

# -- 04. pine_solarise ---------------------------------------------------------
# Ref img 4: pine tree — strong teal/purple solarise, rainbow top strip
def pine_solarise(src):
    arr = a(src); h,w = arr.shape[:2]
    # aggressive solarisation
    sol = np.where(arr<128, arr*2, (255-arr)*2)
    sol[:,:,0] = np.clip(sol[:,:,0]*0.4,0,255)   # kill red ? teal
    sol[:,:,1] = np.clip(sol[:,:,1]*1.2,0,255)   # boost green
    sol[:,:,2] = np.clip(sol[:,:,2]*1.5+40,0,255) # boost blue/purple
    # rainbow strip at top 12%
    strip_h = int(h*0.12)
    xv,_ = mesh(strip_h, w)
    strip = np.stack([
        (np.sin(xv*2*math.pi)*0.5+0.5)*255,
        (np.sin(xv*2*math.pi+2.09)*0.5+0.5)*255,
        (np.sin(xv*2*math.pi+4.19)*0.5+0.5)*255],axis=2)
    sol[:strip_h] = strip
    return sat(im(np.clip(sol,0,255)), 2.4)

# -- 05. scan_lines_ghost ------------------------------------------------------
# Ref img 5: figure with heavy horizontal scan lines, green tint, ghosting
def scan_lines_ghost(src):
    arr = a(src); h,w = arr.shape[:2]
    # green/teal tint
    tinted = arr.copy()
    tinted[:,:,0] = np.clip(arr[:,:,0]*0.3,    0,255)
    tinted[:,:,1] = np.clip(arr[:,:,1]*1.2+20, 0,255)
    tinted[:,:,2] = np.clip(arr[:,:,2]*0.6+10, 0,255)
    # ultra-dense horizontal scan lines (every 2px)
    for y in range(0,h,2):
        tinted[y,:] = tinted[y,:]*0.15
    # chromatic ghost offset
    ghost = np.roll(tinted, 4, axis=1)
    ghost[:,:,0] = np.clip(ghost[:,:,0]*1.5,0,255)
    result = tinted*0.6 + ghost*0.4
    return im(np.clip(result,0,255))

# -- 06. silhouette_static -----------------------------------------------------
# Ref img 6: figure against static TV — strong grain, near monochrome,
# figure as crisp dark silhouette
def silhouette_static(src):
    arr = a(src); h,w = arr.shape[:2]
    rng = np.random.default_rng(6)
    gray = lum(arr)
    # dark silhouette: crush mids, boost contrast
    sig = np.where(gray<100, gray*0.3, np.where(gray>180, gray*1.1+30, gray*0.7))
    sig = np.clip(sig,0,255)
    # TV static noise
    noise = rng.standard_normal((h,w)).astype(np.float32)*40
    # horizontal band noise (static bands)
    band_noise = np.zeros((h,w),dtype=np.float32)
    n_bands = rng.integers(8,20)
    for _ in range(n_bands):
        y = rng.integers(0,h); bh = rng.integers(2,8)
        band_noise[y:y+bh,:] = rng.uniform(50,200)
    mixed = sig + noise*0.6 + band_noise*0.3
    # slight warm tint
    r = np.clip(mixed*1.05,0,255); g = np.clip(mixed*1.0,0,255); b = np.clip(mixed*0.85,0,255)
    return im(np.stack([r,g,b],axis=2))

# -- 07. cityscape_teal_scan ---------------------------------------------------
# Ref img 7: city skyline — teal/blue palette, heavy scan lines, VHS feel
def cityscape_teal_scan(src):
    arr = a(src); h,w = arr.shape[:2]
    # teal colour grade
    out = arr.copy()
    out[:,:,0] = np.clip(arr[:,:,0]*0.3+10,  0,255)
    out[:,:,1] = np.clip(arr[:,:,1]*0.9+30,  0,255)
    out[:,:,2] = np.clip(arr[:,:,2]*1.5+40,  0,255)
    # VHS scan lines (4px period, alternating bright/dark)
    for y in range(0,h,4):
        out[y:y+2,:] = np.clip(out[y:y+2,:]*1.3,0,255)
        if y+2<h:
            out[y+2:y+4,:] = out[y+2:y+4,:]*0.55
    # horizontal chromatic bleed
    out[:,:,0] = np.roll(out[:,:,0], 3,axis=1)
    out[:,:,2] = np.roll(out[:,:,2],-3,axis=1)
    return im(np.clip(out,0,255))

# -- 08. tungsten_body ---------------------------------------------------------
# Ref img 8: backlit figure — amber/gold, pure black background, rim light
def tungsten_body(src):
    arr = a(src)
    x = lum(arr)/255.0
    # very dark base, warm highlight only
    r = np.clip(x**0.6*255*1.1, 0,255)
    g = np.clip(x**1.2*200,     0,255)
    b = np.clip(x**3.5*60,      0,255)
    # crush blacks
    dark_mask = (x<0.2).astype(np.float32)
    r = np.clip(r*(1-dark_mask*0.95),0,255)
    g = np.clip(g*(1-dark_mask*0.95),0,255)
    b = np.clip(b*(1-dark_mask*0.95),0,255)
    # rim light glow: edges of bright?dark transitions
    e = edg(src)/255.0
    rim = np.clip((e-0.3)*3,0,1)
    r2 = np.clip(r+rim*180,0,255); g2 = np.clip(g+rim*80,0,255); b2 = np.clip(b+rim*20,0,255)
    return im(np.stack([r2,g2,b2],axis=2))

# -- 09. lsd_mountains ---------------------------------------------------------
# Ref img 9: mountains — intense red/green/purple, flowing contour psychedelia
def lsd_mountains(src):
    arr = a(src)
    # multi-hue solarise with heavy saturation
    sol_r = np.where(arr[:,:,0]<128, arr[:,:,0]*2, (255-arr[:,:,0])*2)
    sol_g = np.where(arr[:,:,1]<100, arr[:,:,1]*3, (255-arr[:,:,1])*1.5)
    sol_b = np.where(arr[:,:,2]<160, (255-arr[:,:,2])*1.2, arr[:,:,2])
    # channel swap for extra weirdness
    result = np.stack([
        np.clip(sol_r*1.3,0,255),
        np.clip(sol_b*0.8+sol_g*0.4,0,255),
        np.clip(sol_g*1.1,0,255)],axis=2)
    return sat(im(result), 3.5)

# -- 10. vaporwave_figure ------------------------------------------------------
# Ref img 10: person — deep pink/magenta vaporwave, neon outline, dark bg
def vaporwave_figure(src):
    arr = a(src); h,w = arr.shape[:2]
    x = lum(arr)/255.0
    # magenta-pink base
    r = np.clip(x**0.5*255*1.1+20,  0,255)
    g = np.clip(x**2.0*100,          0,255)
    b = np.clip(x**0.8*220+30,       0,255)
    base = np.stack([r,g,b],axis=2)
    # neon cyan outline on edges
    e = edg(src)/255.0
    outline = np.clip((e-0.2)*4,0,1)
    base[:,:,0] = np.clip(base[:,:,0]-outline*100,0,255)
    base[:,:,1] = np.clip(base[:,:,1]+outline*255,0,255)
    base[:,:,2] = np.clip(base[:,:,2]+outline*200,0,255)
    # dark background crush
    base = base*(x**0.4)[:,:,np.newaxis]
    return sat(im(np.clip(base,0,255)), 2.6)

# -- 11. double_exposure_face --------------------------------------------------
# Ref img 13: face double-exposed with cityscape — grey monochrome, blended
def double_exposure_face(src):
    arr = a(src); h,w = arr.shape[:2]
    # base: desaturate to near monochrome
    gray = lum(arr)
    mono = np.stack([gray,gray,gray],axis=2)
    # second exposure: horizontally flipped, blended with screen
    flipped = np.fliplr(arr)
    flipped_gray = np.stack([lum(flipped)]*3,axis=2)
    # screen blend the two exposures
    combined = screen(mono, flipped_gray*0.7)
    # slight cool tint
    combined[:,:,0] = np.clip(combined[:,:,0]*0.9,   0,255)
    combined[:,:,2] = np.clip(combined[:,:,2]*1.1+10,0,255)
    return con(im(np.clip(combined,0,255)), 1.5)

# -- 12. acid_flesh_pink -------------------------------------------------------
# Ref img 14: face/figure — hot magenta pink, lime green, high contrast swirls
def acid_flesh_pink(src):
    arr = a(src)
    # solarise strongly
    sol = np.where(arr<128, arr*2, (255-arr)*2)
    # push red?hot-pink, green?lime
    r = np.clip(sol[:,:,0]*1.6+50,  0,255)
    g = np.clip(sol[:,:,1]*1.8,     0,255)
    b = np.clip(sol[:,:,2]*0.3,     0,255)
    result = np.stack([r,g,b],axis=2)
    # high-frequency edge enhancement
    e = edg(src)[:,:,np.newaxis]/255.0
    result = result*(1+e*0.8)
    return sat(im(np.clip(result,0,255)), 3.2)

# -- 13. church_magenta --------------------------------------------------------
# Ref img 16: church with cross — magenta/pink roof, neon green ground, silhouette
def church_magenta(src):
    arr = a(src); h,w = arr.shape[:2]
    # posterise to 4 levels
    x = lum(arr)/255.0
    q = (x*4).astype(int).clip(0,3)
    pal = np.array([
        [10,  10,  10 ],   # black
        [220, 0,   180],   # magenta
        [0,   200, 30 ],   # neon green
        [255, 240, 250],   # near-white
    ],dtype=np.float32)
    out = pal[q]
    # add original colour hue on top of flat
    hue_boost = hue_rot(arr, 270)
    result = out*0.65 + hue_boost*0.35
    return sat(im(np.clip(result,0,255)), 3.0)

# -- 15. neon_petals -----------------------------------------------------------
# Ref img 19 (bottom row last): flowers/plants — electric cyan, vivid green,
# high key with dark bg
def neon_petals(src):
    arr = a(src); h,w = arr.shape[:2]
    # detect and hyper-boost green/cyan regions
    green_dom = np.clip((arr[:,:,1]-np.maximum(arr[:,:,0],arr[:,:,2]))/255.0,0,1)
    # dark base
    dark = arr*0.15
    # neon: high-green areas ? electric cyan/green
    neon_g = np.clip(arr[:,:,1]*2.5,0,255)
    neon_b = np.clip(arr[:,:,2]*2.0+50,0,255)
    neon_r = np.clip(arr[:,:,0]*0.2,0,255)
    neon = np.stack([neon_r,neon_g,neon_b],axis=2)*green_dom[:,:,np.newaxis]
    # bloom
    bloom = gblur(neon,10)*0.5
    result = dark + neon + bloom
    return sat(im(np.clip(result,0,255)), 3.0)

# -- 16. red_static_wall -------------------------------------------------------
# Ref img 20: red/orange static — heavy noise over red-dominant image
def red_static_wall(src):
    arr = a(src); h,w = arr.shape[:2]
    rng = np.random.default_rng(20)
    # red-orange grade
    r = np.clip(arr[:,:,0]*1.5+50,  0,255)
    g = np.clip(arr[:,:,1]*0.3,     0,255)
    b = np.clip(arr[:,:,2]*0.15,    0,255)
    base = np.stack([r,g,b],axis=2)
    # noise layers: coarse + fine
    n_coarse = rng.standard_normal((h//4,w//4)).astype(np.float32)
    n_coarse = np.array(Image.fromarray(n_coarse).resize((w,h),Image.NEAREST))*50
    n_fine   = rng.standard_normal((h,w)).astype(np.float32)*20
    noise    = n_coarse+n_fine
    base[:,:,0] = np.clip(base[:,:,0]+noise,0,255)
    return im(base)

# -- 17. hue_shift_cascade -----------------------------------------------------
# General across many ref images: progressive hue rotation per row/column
def hue_shift_cascade(src):
    arr = a(src); h,w = arr.shape[:2]
    out = np.zeros_like(arr)
    for row in range(h):
        deg = (row/h)*360
        out[row] = hue_rot(arr[row:row+1], deg)[0]
    return sat(im(out), 2.4)

# -- 18. scanline_portrait -----------------------------------------------------
# Ref imgs with figures: coarse scanline halftone, slight warm tint
def scanline_portrait(src):
    arr = a(src); h,w = arr.shape[:2]
    # warm tint
    warm = arr.copy()
    warm[:,:,0] = np.clip(arr[:,:,0]*1.15+15, 0,255)
    warm[:,:,1] = np.clip(arr[:,:,1]*0.95,    0,255)
    warm[:,:,2] = np.clip(arr[:,:,2]*0.6,     0,255)
    # scanline mask: every 3rd row bright, others dark
    mask = np.ones((h,w),dtype=np.float32)
    mask[::3,:] = 1.5
    mask[1::3,:] = 0.6
    mask[2::3,:] = 0.25
    result = warm*mask[:,:,np.newaxis]
    return im(np.clip(result,0,255))

# -- 19. neon_edge_trace_v2 ----------------------------------------------------
# Strong contour extraction with glowing rainbow outline on black
def neon_edge_trace_v2(src):
    arr = a(src); h,w = arr.shape[:2]
    # multi-scale edge extraction
    e1 = edg(src)
    e2 = np.asarray(src.convert("L").filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.FIND_EDGES),dtype=np.float32)
    combined = np.clip(e1*0.6+e2*0.4,0,255)/255.0
    # assign hue by edge direction (approximate via R-G difference of original)
    phase = (arr[:,:,0]-arr[:,:,2])/255.0*math.pi
    r = (np.sin(phase)*0.5+0.5)*combined*255*2
    g = (np.sin(phase+2.09)*0.5+0.5)*combined*255*2
    b = (np.sin(phase+4.19)*0.5+0.5)*combined*255*2
    # glow bloom
    trace = np.stack([r,g,b],axis=2)
    bloom = gblur(trace,6)*0.6
    dark  = arr*0.05
    return sat(im(np.clip(dark+trace+bloom,0,255)), 2.8)

# -- 20. glitch_rgb_bleed ------------------------------------------------------
# Heavy RGB channel separation with horizontal tearing and colour bleed
def glitch_rgb_bleed(src):
    arr = a(src); h,w = arr.shape[:2]
    rng = np.random.default_rng(24)
    out = arr.copy()
    # tear lines with different offsets per channel
    n_tears = rng.integers(15,35)
    for _ in range(n_tears):
        y = rng.integers(0,h); bh = rng.integers(1,8)
        for ch, shift in enumerate([rng.integers(-60,60) for _ in range(3)]):
            out[y:y+bh,:,ch] = np.roll(arr[y:y+bh,:,ch], shift, axis=1)
    # massive channel offset
    out[:,:,0] = np.roll(out[:,:,0],  20, axis=1)
    out[:,:,2] = np.roll(out[:,:,2], -20, axis=1)
    return sat(im(np.clip(out,0,255)), 2.0)

# -- 21. contour_map_vivid -----------------------------------------------------
# Topographic contour lines in vivid colour — closed loops following luminance
def contour_map_vivid(src):
    arr = a(src)
    x = lum(arr)/255.0
    # 16 contour levels with vivid hue per level
    levels = 16
    q = (x*levels).astype(int).clip(0,levels-1)
    # hue wheel mapped to levels
    hues = np.linspace(0,2*math.pi,levels,endpoint=False)
    pal_r = (np.sin(hues)*0.5+0.5)*255
    pal_g = (np.sin(hues+2.09)*0.5+0.5)*255
    pal_b = (np.sin(hues+4.19)*0.5+0.5)*255
    # find contour edges: where level changes between adjacent pixels
    shift_x = np.roll(q,1,axis=1); shift_y = np.roll(q,1,axis=0)
    contour = ((q!=shift_x)|(q!=shift_y)).astype(np.float32)
    # flat fill + bright contour lines
    flat = np.stack([pal_r[q],pal_g[q],pal_b[q]],axis=2)*0.4
    line_col = np.stack([pal_r[q],pal_g[q],pal_b[q]],axis=2)*contour[:,:,np.newaxis]
    return sat(im(np.clip(flat+line_col*3,0,255)), 3.0)

# -- 22. burnt_chrome ----------------------------------------------------------
# Metallic chrome surface that's been heat-discoloured: blue?purple?gold?clear
def burnt_chrome(src):
    arr = a(src)
    x = lum(arr)/255.0
    # heat tint progression
    stops = [
        (0.00,(20,  20,  80 )),
        (0.25,(80,  20,  120)),
        (0.45,(180, 80,  20 )),
        (0.65,(220, 180, 20 )),
        (0.85,(200, 200, 210)),
        (1.00,(240, 240, 255)),
    ]
    h_i,w_i = arr.shape[:2]
    out = np.zeros((h_i,w_i,3),dtype=np.float32)
    for i in range(len(stops)-1):
        t0,c0 = stops[i]; t1,c1 = stops[i+1]
        mask = (x>=t0)&(x<t1)
        t = ((x-t0)/(t1-t0+1e-9))[mask]
        for ch in range(3): out[mask,ch] = c0[ch]*(1-t)+c1[ch]*t
    # metallic sheen: edge specular
    e = edg(src)/255.0
    out[:,:,0] = np.clip(out[:,:,0]+e*80, 0,255)
    out[:,:,1] = np.clip(out[:,:,1]+e*80, 0,255)
    out[:,:,2] = np.clip(out[:,:,2]+e*90, 0,255)
    return im(out)

# -- 23. warhol_quad -----------------------------------------------------------
# Pop art Warhol-style: flat colour fills, 4 bold hue shifts across image
def warhol_quad(src):
    arr = a(src); h,w = arr.shape[:2]
    # posterise to 3 levels
    x = lum(arr)/255.0
    q = (x*3).astype(int).clip(0,2)
    palettes = [
        [[10,0,60],[255,80,0],[255,255,0]],     # purple/orange/yellow
        [[0,80,0],[255,0,120],[200,255,0]],      # dark-green/pink/lime
        [[60,0,60],[0,200,200],[255,240,200]],   # violet/cyan/cream
        [[0,0,100],[220,0,0],[255,200,150]],     # navy/red/peach
    ]
    half_h,half_w = h//2,w//2
    out = np.zeros_like(arr)
    for qi,(ry,rx) in enumerate([(0,0),(0,half_w),(half_h,0),(half_h,half_w)]):
        ey = ry+half_h if qi>=2 else half_h
        ex = rx+half_w if qi%2==1 else half_w
        pal = np.array(palettes[qi],dtype=np.float32)
        region_q = q[ry:ey, rx:ex]
        out[ry:ey,rx:ex] = pal[region_q]
    return sat(im(np.clip(out,0,255)), 2.8)

# -- 24. thermal_body_scan -----------------------------------------------------
# Medical thermal: vivid red/yellow hot zones, blue cold, like FLIR body scan
def thermal_body_scan(src):
    arr = a(src)
    x = lum(arr)/255.0
    # weighted towards warmer tones in originally-bright regions
    temp = x*0.7 + (arr[:,:,0]/255.0)*0.3  # bias toward red channel
    stops = [
        (0.00,(0,   0,   100)),
        (0.20,(0,   50,  200)),
        (0.40,(0,   200, 200)),
        (0.60,(0,   240, 0  )),
        (0.75,(255, 255, 0  )),
        (0.88,(255, 120, 0  )),
        (1.00,(255, 0,   0  )),
    ]
    h_i,w_i = arr.shape[:2]
    out = np.zeros((h_i,w_i,3),dtype=np.float32)
    for i in range(len(stops)-1):
        t0,c0 = stops[i]; t1,c1 = stops[i+1]
        mask = (temp>=t0)&(temp<t1)
        t = ((temp-t0)/(t1-t0+1e-9))[mask]
        for ch in range(3): out[mask,ch] = c0[ch]*(1-t)+c1[ch]*t
    return im(out)

# -- 25. glitch_column_shift ---------------------------------------------------
# Vertical column-based glitch: blocks of columns randomly shifted up/down
def glitch_column_shift(src):
    arr = a(src); h,w = arr.shape[:2]
    rng = np.random.default_rng(31)
    out = arr.copy()
    col_w = 8
    for cx in range(0,w,col_w):
        if rng.random()<0.3:
            shift = rng.integers(-h//6, h//6)
            out[:,cx:cx+col_w,:] = np.roll(arr[:,cx:cx+col_w,:], shift, axis=0)
    # add strong chromatic aberration
    out[:,:,0] = np.roll(out[:,:,0], 12,axis=1)
    out[:,:,2] = np.roll(out[:,:,2],-12,axis=1)
    return sat(im(np.clip(out,0,255)), 2.0)

# -- 26. cross_process_e6 ------------------------------------------------------
# E6 slide film cross-processed in C41: cyan shadows, orange highlights, crazy
def cross_process_e6(src):
    arr = a(src); x = lum(arr)/255.0
    # cross-process: shadows?cyan, mids?weird, highlights?orange/yellow
    r = np.clip(np.where(x<0.5, x*160, x**0.4*255*1.1),     0,255)
    g = np.clip(np.where(x<0.5, x*80,  x**0.6*200+30),      0,255)
    b = np.clip(np.where(x<0.5, (1-x)*200+80, x**3*80+10),  0,255)
    # add original channel contribution
    r2 = np.clip(r + arr[:,:,0]*0.3, 0,255)
    g2 = np.clip(g + arr[:,:,1]*0.1, 0,255)
    b2 = np.clip(b - arr[:,:,2]*0.2, 0,255)
    return sat(im(np.stack([r2,g2,b2],axis=2)), 2.5)

# -- 27. pixel_sort_column -----------------------------------------------------
# Pixel-sort: sort pixels vertically by brightness in bright regions
def pixel_sort_column(src):
    arr = a(src); h,w = arr.shape[:2]
    out = arr.copy()
    x = lum(arr)/255.0
    for col in range(w):
        # find runs where brightness > threshold
        col_lum = x[:,col]
        threshold = 0.3
        in_run = False; run_start = 0
        for row in range(h):
            if col_lum[row]>threshold and not in_run:
                in_run=True; run_start=row
            elif (col_lum[row]<=threshold or row==h-1) and in_run:
                in_run=False
                run_end = row
                if run_end-run_start>4:
                    segment = arr[run_start:run_end,col,:]   # shape (N, 3)
                    sort_key = segment[:,0]*0.299 + segment[:,1]*0.587 + segment[:,2]*0.114
                    order = np.argsort(sort_key)
                    out[run_start:run_end,col,:] = segment[order]
    return sat(im(np.clip(out,0,255)), 1.8)

# -- 28. neon_jungle -----------------------------------------------------------
# Dark environment, vivid neon greens + electric blues, lush overgrown feel
def neon_jungle(src):
    arr = a(src)
    dark = arr*0.2
    # boost: green?electric, blue?neon
    neon = arr.copy()
    neon[:,:,0] = np.clip(arr[:,:,0]*0.3,      0,255)
    neon[:,:,1] = np.clip(arr[:,:,1]*2.5+40,   0,255)
    neon[:,:,2] = np.clip(arr[:,:,2]*2.0+50,   0,255)
    # where green dominates ? make it pop harder
    green_dom = np.clip((arr[:,:,1]-arr[:,:,0]-arr[:,:,2]/2)/255.0,0,1)
    result = dark + neon*green_dom[:,:,np.newaxis]*1.5 + dark*(1-green_dom[:,:,np.newaxis])
    bloom = gblur(neon,8)*0.3
    return sat(im(np.clip(result+bloom,0,255)), 3.0)

# -- 29. posterise_cmyk --------------------------------------------------------
# Hard CMYK-like poster with visible halftone rosette feel and flat fills
def posterise_cmyk(src):
    arr = a(src); h,w = arr.shape[:2]
    # simulate CMYK separation
    r_n = arr[:,:,0]/255.0; g_n = arr[:,:,1]/255.0; b_n = arr[:,:,2]/255.0
    C = 1-r_n; M = 1-g_n; Y = 1-b_n
    K = np.minimum(C,np.minimum(M,Y))
    denom = 1-K+1e-6
    C2 = (C-K)/denom; M2 = (M-K)/denom; Y2 = (Y-K)/denom
    # quantise each plate to 3 levels
    C3 = (C2*3).astype(int).clip(0,2)/2.0
    M3 = (M2*3).astype(int).clip(0,2)/2.0
    Y3 = (Y2*3).astype(int).clip(0,2)/2.0
    K3 = (K *3).astype(int).clip(0,2)/2.0
    # reconstruct
    r_out = np.clip((1-C3)*(1-K3)*255,0,255)
    g_out = np.clip((1-M3)*(1-K3)*255,0,255)
    b_out = np.clip((1-Y3)*(1-K3)*255,0,255)
    return sat(im(np.stack([r_out,g_out,b_out],axis=2)), 2.5)

# -- 30. laser_etch_red --------------------------------------------------------
# Laser engraving aesthetic: red/orange burn lines on dark material
def laser_etch_red(src):
    arr = a(src)
    e = edg(src)/255.0
    x = lum(arr)/255.0
    # dark substrate
    substrate = arr*0.08
    # etched lines: edges glow red-orange
    etch_r = np.clip(e*255*1.5+x*80,  0,255)
    etch_g = np.clip(e*120+x*30,       0,255)
    etch_b = np.clip(e*20,              0,255)
    etch = np.stack([etch_r,etch_g,etch_b],axis=2)
    bloom = gblur(etch,5)*0.5
    return im(np.clip(substrate+etch+bloom,0,255))

# -- 31. chromatic_displacement ------------------------------------------------
# Each channel displaced by image's own gradient — self-warping colour smear
def chromatic_displacement(src):
    arr = a(src); h,w = arr.shape[:2]
    # use image gradient as displacement map
    gray = np.asarray(src.convert("L"),dtype=np.float32)
    # sobel-like gradient
    gx = np.roll(gray,1,axis=1)-np.roll(gray,-1,axis=1)
    gy = np.roll(gray,1,axis=0)-np.roll(gray,-1,axis=0)
    ys_g,xs_g = np.mgrid[0:h,0:w].astype(np.float32)
    out = np.zeros_like(arr)
    # displace each channel by gradient * scale (different scale per ch)
    for ch,scale in enumerate([0.12, -0.08, 0.15]):
        xs_d = np.clip(xs_g+gx*scale,0,w-1).astype(int)
        ys_d = np.clip(ys_g+gy*scale,0,h-1).astype(int)
        out[:,:,ch] = arr[ys_d,xs_d,ch]
    return sat(im(np.clip(out,0,255)), 2.4)


# ------------------------------------------------------------------------------
FILTERS = {
    "01_open_sign_neon":            open_sign_neon,
    "02_infrared_foliage_rainbow":  infrared_foliage_rainbow,
    "03_rainbow_treeline":          rainbow_treeline,
    "04_pine_solarise":             pine_solarise,
    "05_scan_lines_ghost":          scan_lines_ghost,
    "06_silhouette_static":         silhouette_static,
    "07_cityscape_teal_scan":       cityscape_teal_scan,
    "08_tungsten_body":             tungsten_body,
    "09_lsd_mountains":             lsd_mountains,
    "10_vaporwave_figure":          vaporwave_figure,
    "11_double_exposure_face":      double_exposure_face,
    "12_acid_flesh_pink":           acid_flesh_pink,
    "13_church_magenta":            church_magenta,
    "15_neon_petals":               neon_petals,
    "16_red_static_wall":           red_static_wall,
    "17_hue_shift_cascade":         hue_shift_cascade,
    "18_scanline_portrait":         scanline_portrait,
    "19_neon_edge_trace_v2":        neon_edge_trace_v2,
    "20_glitch_rgb_bleed":          glitch_rgb_bleed,
    "21_contour_map_vivid":         contour_map_vivid,
    "22_burnt_chrome":              burnt_chrome,
    "23_warhol_quad":               warhol_quad,
    "24_thermal_body_scan":         thermal_body_scan,
    "25_glitch_column_shift":       glitch_column_shift,
    "26_cross_process_e6":          cross_process_e6,
    "27_pixel_sort_column":         pixel_sort_column,
    "28_neon_jungle":               neon_jungle,
    "29_posterise_cmyk":            posterise_cmyk,
    "30_laser_etch_red":            laser_etch_red,
    "31_chromatic_displacement":    chromatic_displacement,
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

