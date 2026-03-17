"""
circuit_burn_filters_v2.py  -  30 experimental colour filters
==================================================================
Drop images into  'wallter/'  next to this script.
Every filtered image is saved flat into  'circuitburn/'
with the filter name as the filename:

    circuitburn/<filtername>__<original_stem>.png

Dependencies:  pip install Pillow numpy
"""

import os, sys, glob, math
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

INPUT_FOLDER  = "jesse"
OUTPUT_FOLDER = "circuitburn_V2"
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

def a(src):
    return np.asarray(src.convert("RGB"), dtype=np.float32)

def i(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

def sat(pil, f):
    return ImageEnhance.Color(pil).enhance(f)

def bri(pil, f):
    return ImageEnhance.Brightness(pil).enhance(f)

def con(pil, f):
    return ImageEnhance.Contrast(pil).enhance(f)

def screen(x, y):
    return 255 - (255-x)*(255-y)/255.0

def mesh(h, w):
    return np.meshgrid(np.linspace(0,1,w), np.linspace(0,1,h))

def lum(arr):
    return arr[:,:,0]*0.299 + arr[:,:,1]*0.587 + arr[:,:,2]*0.114

def edges(src):
    return np.asarray(src.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)

def blur(src, r):
    return np.asarray(src.filter(ImageFilter.GaussianBlur(r)), dtype=np.float32)

# ---- 01 bismuth_crystal -------------------------------------------------------
def bismuth_crystal(src):
    arr = a(src); h,w = arr.shape[:2]
    q = (lum(arr)/255.0*12).astype(int).clip(0,11)
    hues = np.linspace(0, 2*math.pi, 12, endpoint=False)
    r_lut = np.sin(hues)*0.5+0.5
    g_lut = np.sin(hues+2.09)*0.5+0.5
    b_lut = np.sin(hues+4.19)*0.5+0.5
    out = np.stack([r_lut[q],g_lut[q],b_lut[q]],axis=2)*255
    e = edges(src)[:,:,np.newaxis]*0.6
    return sat(i(np.clip(out*0.7+e,0,255)), 2.8)

# ---- 02 void_static -----------------------------------------------------------
def void_static(src):
    arr = a(src); h,w = arr.shape[:2]
    rng = np.random.default_rng(13)
    n1 = rng.standard_normal((h,w)).astype(np.float32)*30
    n2 = np.array(Image.fromarray(rng.standard_normal((h//4,w//4)).astype(np.float32))
                  .resize((w,h),Image.NEAREST))*60
    crunch = np.clip((lum(arr)/255.0-0.5)*4+0.5,0,1)
    r = np.clip(crunch*80 +n1+np.clip(crunch*120,0,255),0,255)
    g = np.clip(crunch*20 +n1*0.5,0,255)
    b = np.clip(crunch*255+n2*0.8,0,255)
    return i(np.stack([r,g,b],axis=2))

# ---- 03 gamma_burst -----------------------------------------------------------
def gamma_burst(src):
    x = a(src)/255.0
    r = np.power(x[:,:,0],0.15)*255
    g = np.power(x[:,:,1],2.80)*255
    b = np.power(x[:,:,2],0.40)*255
    return sat(i(np.stack([r,g,b],axis=2)),2.2)

# ---- 04 oxidised_silver -------------------------------------------------------
def oxidised_silver(src):
    arr = a(src); gray = lum(arr)
    r = np.clip(gray*0.85,      0,255)
    g = np.clip(gray*0.90,      0,255)
    b = np.clip(gray*1.05+10,   0,255)
    silver = np.stack([r,g,b],axis=2)
    ox = ((gray>60)&(gray<180)).astype(np.float32)*0.6
    silver[:,:,1] = np.clip(silver[:,:,1]+ox*80, 0,255)
    silver[:,:,2] = np.clip(silver[:,:,2]+ox*60, 0,255)
    silver[:,:,0] = np.clip(silver[:,:,0]-ox*20, 0,255)
    return con(i(silver),1.6)

# ---- 05 tungsten_filament -----------------------------------------------------
def tungsten_filament(src):
    x = lum(a(src))/255.0
    r = np.clip(x**0.5*255*1.3,  0,255)
    g = np.clip(x**1.5*255*0.6,  0,255)
    b = np.clip(x**3.0*255*0.2,  0,255)
    hot = np.clip((x-0.85)*7,0,1)[:,:,np.newaxis]
    out = np.stack([r,g,b],axis=2)*(1-hot)+255*hot
    return i(out)

# ---- 06 absinthe_drip ---------------------------------------------------------
def absinthe_drip(src):
    arr = a(src); h,w = arr.shape[:2]
    xv,yv = mesh(h,w)
    shift = (np.sin(yv*8*math.pi)*0.03*w).astype(int)
    warped = arr.copy()
    for y in range(h):
        warped[y] = np.roll(arr[y], shift[y,0], axis=0)
    r = np.clip(warped[:,:,1]*0.8+warped[:,:,0]*0.1,    0,255)
    g = np.clip(warped[:,:,1]*1.4+30,                    0,255)
    b = np.clip(warped[:,:,2]*0.3+warped[:,:,0]*0.2+20,  0,255)
    return sat(i(np.stack([r,g,b],axis=2)),2.6)

# ---- 07 ghost_frequency -------------------------------------------------------
def ghost_frequency(src):
    arr = a(src)
    g1 = np.roll(arr, 20,axis=1); g2 = np.roll(arr,-20,axis=0)
    r = np.clip(g1[:,:,0]*0.9+g2[:,:,0]*0.1+40,0,255)
    g = np.clip(g1[:,:,1]*0.1+g2[:,:,1]*0.9,    0,255)
    b = np.clip(g2[:,:,2]*1.2+30,                0,255)
    return sat(i(arr*0.3+np.stack([r,g,b],axis=2)*0.7),2.0)

# ---- 08 magma_crust -----------------------------------------------------------
def magma_crust(src):
    x = lum(a(src))/255.0
    crack = np.clip((x-0.4)*3,0,1)
    r = np.clip(crack**0.4*255*1.4,   0,255)
    g = np.clip(crack**1.2*255*0.5,   0,255)
    b = np.clip(crack**3.5*255*0.15,  0,255)
    hot = np.clip((crack-0.85)*6,0,1)[:,:,np.newaxis]
    out = np.stack([r,g,b],axis=2)*(1-hot)+255*hot
    return i(out)

# ---- 09 cerebral_cortex -------------------------------------------------------
def cerebral_cortex(src):
    arr = a(src)
    bl = np.asarray(src.convert("RGB").filter(ImageFilter.GaussianBlur(4)),dtype=np.float32)
    hp = np.clip((arr-bl)*3+128,0,255)
    x  = lum(hp)/255.0
    r = np.clip(x**0.7*220+30, 0,255)
    g = np.clip(x**1.1*180,     0,255)
    b = np.clip(x**0.5*255,     0,255)
    return sat(i(np.stack([r,g,b],axis=2)),1.4)

# ---- 10 retinal_burn ----------------------------------------------------------
def retinal_burn(src):
    arr = a(src); L = lum(arr)
    comp  = 255-arr
    bloom_i = i(np.clip(comp*(L/255.0)[:,:,np.newaxis],0,255))
    bloom_a = np.asarray(bloom_i.filter(ImageFilter.GaussianBlur(16)),dtype=np.float32)
    return sat(i(np.clip(arr*0.5+bloom_a*1.2,0,255)),2.5)

# ---- 11 sulfur_springs --------------------------------------------------------
def sulfur_springs(src):
    x = lum(a(src))/255.0
    r = np.clip(x**0.6*255*1.1+20,  0,255)
    g = np.clip(x**0.7*255*0.95,    0,255)
    b = np.clip(x**2.0*255*0.25,    0,255)
    hi = np.clip((x-0.7)*3,0,1)[:,:,np.newaxis]
    out = np.stack([r,g,b],axis=2)*(1-hi*0.5)+np.array([255,252,230])*hi*0.5
    return i(out)

# ---- 12 neon_autopsy ----------------------------------------------------------
def neon_autopsy(src):
    arr = a(src); L = lum(arr)
    base = np.stack([L*0.15,L*0.9,L*0.7],axis=2)
    anom = np.clip((arr[:,:,0]-arr[:,:,2])/255.0,0,1)
    base[:,:,0] = np.clip(base[:,:,0]+anom*220,0,255)
    base[:,:,1] = np.clip(base[:,:,1]-anom*80, 0,255)
    base[:,:,2] = np.clip(base[:,:,2]+anom*100,0,255)
    return sat(i(base),2.3)

# ---- 13 prism_shatter ---------------------------------------------------------
def prism_shatter(src):
    arr = a(src)
    offsets  = [(-12,0),(-6,0),(0,0),(6,0),(12,0)]
    spectrum = [(255,0,0),(255,165,0),(0,255,0),(0,100,255),(148,0,211)]
    out = np.zeros_like(arr)
    for (dy,dx),(sr,sg,sb) in zip(offsets,spectrum):
        s = np.roll(np.roll(arr,dy,axis=0),dx,axis=1)
        out += s*np.array([sr,sg,sb],dtype=np.float32)/255.0
    return sat(i(np.clip(out/len(offsets)*2.5,0,255)),3.0)

# ---- 14 ferrofluid ------------------------------------------------------------
def ferrofluid(src):
    arr = a(src)
    spikes = np.asarray(src.convert("L").filter(ImageFilter.EMBOSS),dtype=np.float32)
    spikes = np.clip((spikes-128)*4,0,255)
    dark   = arr*0.1
    silver = np.stack([spikes*0.9,spikes*0.95,spikes],axis=2)
    return i(np.clip(dark+silver,0,255))

# ---- 15 dichroic_glass --------------------------------------------------------
def dichroic_glass(src):
    arr = a(src); h,w = arr.shape[:2]
    xv,yv = mesh(h,w)
    angle = xv*math.pi+yv*math.pi
    dich = np.stack([
        (np.sin(angle*2)*0.5+0.5),
        (np.sin(angle*2+2.09)*0.5+0.5),
        (np.sin(angle*2+4.19)*0.5+0.5)],axis=2)*255
    return sat(i(np.clip(arr*0.2+arr*dich/255.0*0.8,0,255)),3.5)

# ---- 16 strontium_fire --------------------------------------------------------
def strontium_fire(src):
    x = lum(a(src))/255.0
    r = np.clip(x**0.3*255,      0,255)
    g = np.clip(x**3.0*80,       0,255)
    b = np.clip(x**5.0*30,       0,255)
    hot = np.clip((x-0.9)*10,0,1)[:,:,np.newaxis]
    out = np.stack([r,g,b],axis=2)*(1-hot)+255*hot
    return i(out)

# ---- 17 quantum_foam ----------------------------------------------------------
def quantum_foam(src):
    arr = a(src); h,w = arr.shape[:2]
    rng = np.random.default_rng(99)
    n1 = rng.standard_normal((h,w)).astype(np.float32)*30
    n2 = np.array(Image.fromarray(rng.standard_normal((h//4,w//4)).astype(np.float32))
                  .resize((w,h),Image.NEAREST))*60
    foam = np.stack([
        np.clip(arr[:,:,0]+n1+n2,0,255),
        np.clip(arr[:,:,1]-n1*0.5,0,255),
        np.clip(arr[:,:,2]+n2*0.8,0,255)],axis=2)
    return sat(i(foam),2.0)

# ---- 18 cassette_dub ----------------------------------------------------------
def cassette_dub(src):
    arr = a(src); h,w = arr.shape[:2]
    wobble = (np.sin(np.arange(h)*0.08)*6).astype(int)
    warped = np.zeros_like(arr)
    for y in range(h):
        warped[y] = np.roll(arr[y],wobble[y],axis=0)
    r = np.clip(warped[:,:,0]*1.1+25,0,255)
    g = np.clip(warped[:,:,1]*0.55,  0,255)
    b = np.clip(warped[:,:,2]*1.0+20,0,255)
    return con(bri(i(np.stack([r,g,b],axis=2)),0.85),0.75)

# ---- 19 nerve_signal ----------------------------------------------------------
def nerve_signal(src):
    arr = a(src)
    gray = lum(arr)
    e = np.clip(edges(src)*2,0,255)
    dark = arr*0.08
    dendrite = np.stack([np.clip(e*1.1,0,255),np.clip(e*1.0,0,255),np.clip(e*0.1,0,255)],axis=2)
    h,w = arr.shape[:2]
    myelin_i = i(np.stack([np.zeros((h,w)),np.zeros((h,w)),np.clip(gray*0.6,0,255)],axis=2))
    myelin = np.asarray(myelin_i.filter(ImageFilter.GaussianBlur(8)),dtype=np.float32)
    return i(np.clip(dark+dendrite+myelin,0,255))

# ---- 20 solar_wind ------------------------------------------------------------
def solar_wind(src):
    arr = a(src); h,w = arr.shape[:2]
    xv,yv = mesh(h,w)
    s1 = np.sin((xv+yv)*20*math.pi)*0.5+0.5
    s2 = np.sin((xv-yv)*15*math.pi+1.0)*0.5+0.5
    x  = lum(arr)/255.0
    r = np.clip(x*200+s1*80+30,  0,255)
    g = np.clip(x*160+s2*40,     0,255)
    b = np.clip(x*80 +s2*140+20, 0,255)
    return sat(i(np.stack([r,g,b],axis=2)),2.1)

# ---- 21 crt_phosphor ----------------------------------------------------------
def crt_phosphor(src):
    arr = a(src); h,w = arr.shape[:2]
    x   = lum(arr)/255.0
    dm  = np.ones((h,w),dtype=np.float32)
    dm[::2,:] *= 0.45; dm[:,::3] *= 0.7
    glow_lum = x*dm
    g_i = i(np.stack([np.zeros((h,w)),glow_lum*255,glow_lum*80],axis=2))
    g_b = np.asarray(g_i.filter(ImageFilter.GaussianBlur(2)),dtype=np.float32)*1.3
    return i(np.clip(g_b,0,255))

# ---- 22 tritium_glow ----------------------------------------------------------
def tritium_glow(src):
    x = lum(a(src))/255.0
    glow = np.power(np.clip(x,0,1),0.4)
    r = np.clip(glow**1.5*180, 0,255)
    g = np.clip(glow**0.4*255, 0,255)
    b = np.clip(glow**4.0*50,  0,255)
    base = np.stack([r,g,b],axis=2)
    bloom = np.asarray(i(base).filter(ImageFilter.GaussianBlur(12)),dtype=np.float32)*0.5
    return i(np.clip(base+bloom,0,255))

# ---- 23 rust_bloom ------------------------------------------------------------
def rust_bloom(src):
    arr = a(src); x = lum(arr)/255.0
    steel = np.stack([x*180,x*175,x*170],axis=2)
    rs = np.clip((arr[:,:,0]-arr[:,:,2])/255.0,0,1)
    rust = np.stack([np.clip(rs*230+40,0,255),np.clip(rs*90,0,255),np.clip(rs*10,0,255)],axis=2)
    alpha = (rs*1.5).clip(0,1)[:,:,np.newaxis]
    return con(i(steel*(1-alpha)+rust*alpha),1.5)

# ---- 24 sodium_vapor ----------------------------------------------------------
def sodium_vapor(src):
    x = lum(a(src))/255.0
    r = np.clip(x**0.6*255*1.05+15,0,255)
    g = np.clip(x**0.8*255*0.70,    0,255)
    b = np.clip(x**2.5*255*0.05,    0,255)
    return i(np.stack([r,g,b],axis=2))

# ---- 25 corrupted_memory -------------------------------------------------------
def corrupted_memory(src):
    arr = a(src); h,w = arr.shape[:2]
    rng = np.random.default_rng(31); out = arr.copy(); bs=16
    for by in range(0,h-bs,bs):
        for bx in range(0,w-bs,bs):
            p = rng.random()
            if p<0.12:
                noise = rng.integers(0,255,(bs,bs,3),dtype=np.uint8).astype(np.float32)
                out[by:by+bs,bx:bx+bs] = np.bitwise_xor(
                    out[by:by+bs,bx:bx+bs].astype(np.uint8),noise.astype(np.uint8)).astype(np.float32)
            elif p<0.25:
                sy=rng.integers(0,h-bs); sx=rng.integers(0,w-bs)
                out[by:by+bs,bx:bx+bs]=arr[sy:sy+bs,sx:sx+bs]
    return sat(i(out),1.8)

# ---- 26 teal_orange_split -----------------------------------------------------
def teal_orange_split(src):
    arr = a(src)
    t = (lum(arr)/255.0)[:,:,np.newaxis]
    out = np.array([0,180,180])*(1-t)+np.array([255,130,0])*t
    return sat(i(np.clip(arr*0.25+out*0.75,0,255)),2.5)

# ---- 27 hypnodisk -------------------------------------------------------------
def hypnodisk(src):
    arr = a(src); h,w = arr.shape[:2]
    cy,cx = h/2,w/2
    ys,xs = np.mgrid[0:h,0:w]
    dist  = np.sqrt((xs-cx)**2+(ys-cy)**2)/math.sqrt(cx**2+cy**2)
    theta = np.arctan2(ys-cy,xs-cx)/math.pi
    hue   = (dist*4+theta)%1.0
    x     = lum(arr)/255.0
    r = (np.sin(hue*2*math.pi)*0.5+0.5)*x*255*1.5
    g = (np.sin(hue*2*math.pi+2.09)*0.5+0.5)*x*255*1.5
    b = (np.sin(hue*2*math.pi+4.19)*0.5+0.5)*x*255*1.5
    return sat(i(np.clip(np.stack([r,g,b],axis=2),0,255)),2.4)

# ---- 28 dna_sequencer ---------------------------------------------------------
def dna_sequencer(src):
    arr = a(src); h,w = arr.shape[:2]
    bw   = max(1,w//4)
    cols = np.array([[255,50,50],[50,255,50],[50,50,255],[255,200,0]],dtype=np.float32)
    cidx = (np.arange(w)//bw).clip(0,3)
    cmap = cols[cidx]
    lumn = (lum(arr)/255.0)[:,:,np.newaxis]
    bands = cmap[np.newaxis,:,:]*lumn*1.8
    dark  = arr*0.05
    out   = np.clip(dark+bands,0,255)
    bloom = np.asarray(i(out).filter(ImageFilter.GaussianBlur(6)),dtype=np.float32)*0.4
    return i(np.clip(out+bloom,0,255))

# ---- 29 acid_rain -------------------------------------------------------------
def acid_rain(src):
    arr = a(src); h,w = arr.shape[:2]
    rng = np.random.default_rng(88)
    streaks = np.zeros((h,w),dtype=np.float32)
    for x in rng.integers(0,w,w//3):
        st = rng.integers(0,h//2); length = rng.integers(h//4,h//2)
        end = min(st+length,h); fade = np.linspace(1,0,length)
        streaks[st:end,x] = fade[:end-st]
    r = np.clip(arr[:,:,0]*0.3+streaks*180,0,255)
    g = np.clip(arr[:,:,1]*0.4+streaks*255,0,255)
    b = np.clip(arr[:,:,2]*0.2+streaks*30, 0,255)
    return sat(i(np.stack([r,g,b],axis=2)),2.2)

# ---- 30 carnival_mirror -------------------------------------------------------
def carnival_mirror(src):
    arr = a(src); h,w = arr.shape[:2]
    cy,cx = h/2.0,w/2.0
    ys,xs = np.mgrid[0:h,0:w].astype(np.float32)
    nx=(xs-cx)/cx; ny=(ys-cy)/cy; r2=nx**2+ny**2; k=0.35
    xd=np.clip((nx*(1+k*r2)*cx+cx),0,w-1).astype(int)
    yd=np.clip((ny*(1+k*r2)*cy+cy),0,h-1).astype(int)
    warped=arr[yd,xd]
    return sat(con(i(warped),1.6),3.5)

# ══════════════════════════════════════════════════════════════════════════════
FILTERS = {
    "01_bismuth_crystal":       bismuth_crystal,
    "02_void_static":           void_static,
    "03_gamma_burst":           gamma_burst,
    "04_oxidised_silver":       oxidised_silver,
    "05_tungsten_filament":     tungsten_filament,
    "06_absinthe_drip":         absinthe_drip,
    "07_ghost_frequency":       ghost_frequency,
    "08_magma_crust":           magma_crust,
    "09_cerebral_cortex":       cerebral_cortex,
    "10_retinal_burn":          retinal_burn,
    "11_sulfur_springs":        sulfur_springs,
    "12_neon_autopsy":          neon_autopsy,
    "13_prism_shatter":         prism_shatter,
    "14_ferrofluid":            ferrofluid,
    "15_dichroic_glass":        dichroic_glass,
    "16_strontium_fire":        strontium_fire,
    "17_quantum_foam":          quantum_foam,
    "18_cassette_dub":          cassette_dub,
    "19_nerve_signal":          nerve_signal,
    "20_solar_wind":            solar_wind,
    "21_crt_phosphor":          crt_phosphor,
    "22_tritium_glow":          tritium_glow,
    "23_rust_bloom":            rust_bloom,
    "24_sodium_vapor":          sodium_vapor,
    "25_corrupted_memory":      corrupted_memory,
    "26_teal_orange_split":     teal_orange_split,
    "27_hypnodisk":             hypnodisk,
    "28_dna_sequencer":         dna_sequencer,
    "29_acid_rain":             acid_rain,
    "30_carnival_mirror":       carnival_mirror,
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
