"""
RecPatGen — Unified Recursive Pattern Generator
Combines all operations from RecPatGen04-10 and RecPatGenVarb.
Requires: pip install Pillow numpy scipy scikit-learn opencv-python
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageFilter, ImageEnhance, ImageDraw
import cv2
import os, random, math, time, threading, queue, colorsys, logging
from datetime import datetime
from scipy.stats import triang
from scipy.ndimage import label, generic_filter
from sklearn.cluster import MiniBatchKMeans

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger("RecPatGen")

# ── colours / fonts ───────────────────────────────────────────────────────────
BG    = "#0d0d14"; PANEL = "#13131e"; CARD  = "#1a1a28"
ACC   = "#a06af7"; ACC2  = "#f76ab4"; FG    = "#e8e6f0"
DIM   = "#5a5870"; BORDER= "#22222f"; GREEN = "#6af7c0"; YELLOW= "#f7e06a"
FT  = ("Courier New", 9);  FM  = ("Courier New", 8)
FB  = ("Courier New", 10, "bold"); FS  = ("Courier New", 7)
FH  = ("Georgia", 11, "bold");    FT2 = ("Georgia", 15, "bold")


# ═════════════════════════════════════════════════════════════════════════════
#  SYMMETRY  (all vectorised numpy, no pixel loops)
# ═════════════════════════════════════════════════════════════════════════════

def apply_symmetry(section, sym_type):
    """Apply symmetry and ALWAYS return array with exactly the same h,w,3 as input."""
    if section.size == 0: return section
    h, w = section.shape[:2]
    s = np.ascontiguousarray(section)
    # ensure 3-channel input
    if s.ndim == 2:
        s = np.stack([s]*3, axis=2)
    elif s.shape[2] != 3:
        s = s[:,:,:3]

    try:
        if sym_type == "vertical":
            mid = h // 2
            r = s.copy()
            r[mid:h] = s[0:h-mid][::-1]
        elif sym_type == "horizontal":
            mid = w // 2
            r = s.copy()
            r[:, mid:w] = s[:, 0:w-mid][:, ::-1]
        elif sym_type in ("quad", "radial"):
            # Mirror top-left quadrant into all four quadrants.
            # For odd dimensions, the centre row/col gets the mirror of itself.
            r = s.copy()
            mh, mw = h//2, w//2
            tl = s[:mh, :mw].copy()
            # top-left stays as-is
            # top-right: flip tl horizontally, fit into remaining columns
            tr = tl[:, ::-1]
            r[:mh, mw:mw+tr.shape[1]] = tr[:, :w-mw]
            # bottom-left: flip tl vertically, fit into remaining rows
            bl = tl[::-1]
            r[mh:mh+bl.shape[0], :mw] = bl[:h-mh]
            # bottom-right: flip both axes
            br = tl[::-1, ::-1]
            r[mh:mh+br.shape[0], mw:mw+br.shape[1]] = br[:h-mh, :w-mw]
        elif sym_type == "kaleidoscope3":
            cx, cy = w//2, h//2
            r = s.copy()
            for ang in [120, 240]:
                M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
                rot = cv2.warpAffine(s, M, (w,h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)
                r = np.maximum(r, rot)
        elif sym_type == "kaleidoscope6":
            cx, cy = w//2, h//2
            r = s.copy()
            for ang in [60,120,180,240,300]:
                M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
                rot = cv2.warpAffine(s, M, (w,h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)
                r = np.maximum(r, rot)
        elif sym_type == "glide":
            r = s.copy()
            mid = w//2
            r[:, mid:w] = s[:, 0:w-mid][:, ::-1]
        elif sym_type == "rotate90":
            r = np.rot90(s)[:h, :w]
            if r.shape != (h,w,3):
                tmp = np.zeros((h,w,3),dtype=np.uint8)
                ch=min(r.shape[0],h); cw=min(r.shape[1],w)
                tmp[:ch,:cw]=r[:ch,:cw]; r=tmp
        elif sym_type == "rotate180":
            r = np.rot90(s,2)[:h,:w]
        elif sym_type == "wave":
            r = np.empty_like(s)
            xs = np.arange(w)
            for y in range(h):
                ox = int(10 * math.sin(2*math.pi*y/max(h,1)*3))
                src = np.clip(xs + ox, 0, w-1)
                r[y] = s[y, src]
        elif sym_type == "spiral":
            cx, cy = w//2, h//2; r = s.copy()
            for i in range(1,5):
                M = cv2.getRotationMatrix2D((cx,cy), i*90, 1.0-i*0.15)
                rot = cv2.warpAffine(s, M, (w,h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)
                r = np.maximum(r, rot)
        elif sym_type == "fractal":
            r = s.copy()
            qw, qh = max(1,w//2), max(1,h//2)
            q = cv2.resize(s, (qw,qh))
            r[:qh, :qw]       = q
            r[:qh, w-qw:w]    = q[:,::-1]
            r[h-qh:h, :qw]    = q[::-1,:]
            r[h-qh:h, w-qw:w] = q[::-1,::-1]
        elif sym_type == "wallpaper":
            r = s.copy()
            mh, mw = h//2, w//2
            tl = s[:mh, :mw].copy()
            r[:mh, mw:mw+tl.shape[1]] = tl[:, ::-1][:, :w-mw]
            bl = tl[::-1]
            r[mh:mh+bl.shape[0], :mw] = bl[:h-mh]
            br = tl[::-1, ::-1]
            r[mh:mh+br.shape[0], mw:mw+br.shape[1]] = br[:h-mh, :w-mw]
        elif sym_type == "reflective_tiling":
            r = s.copy()
            mid = w//2
            r[:, mid:w] = s[:, 0:w-mid][:, ::-1]
            mh = h//2
            r[mh:h] = r[0:h-mh][::-1]
        else:
            r = s.copy()
    except Exception as e:
        log.warning(f"symmetry {sym_type} failed: {e}")
        r = s.copy()

    # Guarantee exact h×w×3 output
    out = np.ascontiguousarray(r)
    if out.shape[0] != h or out.shape[1] != w:
        tmp = np.zeros((h,w,3), dtype=np.uint8)
        ch2=min(out.shape[0],h); cw2=min(out.shape[1],w)
        tmp[:ch2,:cw2] = out[:ch2,:cw2]
        out = tmp
    return np.clip(out,0,255).astype(np.uint8)


def _safe_paste(dst, src, y0, x0):
    """Paste src into dst at (y0,x0), clipping to bounds and matching shapes."""
    if src is None or src.size == 0: return
    if src.ndim == 2: src = np.stack([src]*3, axis=2)
    if src.shape[2] != 3: src = src[:,:,:3]
    H, W = dst.shape[:2]
    ay0, ax0 = max(y0,0), max(x0,0)
    ay1, ax1 = min(y0+src.shape[0], H), min(x0+src.shape[1], W)
    if ay1<=ay0 or ax1<=ax0: return
    sh, sw = ay1-ay0, ax1-ax0
    ty0, tx0 = ay0-y0, ax0-x0
    piece = src[ty0:ty0+sh, tx0:tx0+sw]
    ch, cw = min(piece.shape[0],sh), min(piece.shape[1],sw)
    dst[ay0:ay0+ch, ax0:ax0+cw] = piece[:ch,:cw]

SYMMETRY_NAMES = [
    "vertical","horizontal","quad","radial",
    "kaleidoscope3","kaleidoscope6","glide","rotate90","rotate180",
    "wave","spiral","fractal","wallpaper","reflective_tiling",
]


# ═════════════════════════════════════════════════════════════════════════════
#  SELECTION MASKS  (fully vectorised)
# ═════════════════════════════════════════════════════════════════════════════

def get_mask(arr, method, params):
    p = params or {}
    if method == 'brightness':
        return np.mean(arr[..., :3], axis=2) > p.get('threshold', 128)
    if method == 'red':   return arr[..., 0] > p.get('threshold', 128)
    if method == 'green': return arr[..., 1] > p.get('threshold', 128)
    if method == 'blue':  return arr[..., 2] > p.get('threshold', 128)
    if method == 'color_range':
        return ((arr[...,0]>=p.get('min_r',0)) & (arr[...,0]<=p.get('max_r',255)) &
                (arr[...,1]>=p.get('min_g',0)) & (arr[...,1]<=p.get('max_g',255)) &
                (arr[...,2]>=p.get('min_b',0)) & (arr[...,2]<=p.get('max_b',255)))
    if method == 'edge_detection':
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        return np.sqrt(dx**2 + dy**2) > p.get('threshold', 50)
    if method == 'hsv':
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        h = hsv[...,0]*2; s = hsv[...,1]*(100/255); v = hsv[...,2]*(100/255)
        return ((h>=p.get('min_h',0))&(h<=p.get('max_h',360))&
                (s>=p.get('min_s',0))&(s<=p.get('max_s',100))&
                (v>=p.get('min_v',0))&(v<=p.get('max_v',100)))
    if method == 'texture':
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return generic_filter(gray.astype(float), np.var,
                              size=p.get('neighborhood_size', 5)) > p.get('threshold', 100)
    if method == 'segmentation':
        pixels = arr.reshape(-1, 3).astype(np.float32)
        n = min(p.get('n_clusters', 5), len(pixels))
        km = MiniBatchKMeans(n_clusters=n, random_state=42, n_init='auto')
        km.fit(pixels)
        return km.labels_.reshape(arr.shape[:2]) == p.get('target_cluster', 0)
    if method == 'entropy':
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        def _ent(v):
            h = np.histogram(v, bins=32, range=(0,256))[0]; h = h[h>0]
            return -np.sum(h/h.sum()*np.log2(h/h.sum())) if len(h) else 0
        return generic_filter(gray.astype(float), _ent,
                              size=p.get('neighborhood_size', 9)) > p.get('threshold', 4)
    if method == 'random':
        return np.random.random(arr.shape[:2]) < p.get('coverage', 0.5)
    return np.ones(arr.shape[:2], dtype=bool)


# ═════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ═════════════════════════════════════════════════════════════════════════════

def extract_palette(arr, n=8):
    pixels = arr.reshape(-1, 3).astype(np.float32)
    n = min(n, max(1, len(np.unique(pixels, axis=0))))
    sample = pixels[np.random.choice(len(pixels), min(10000,len(pixels)), replace=False)]
    km = MiniBatchKMeans(n_clusters=n, random_state=42, n_init='auto')
    km.fit(sample)
    return km.cluster_centers_.astype(np.uint8)


# ═════════════════════════════════════════════════════════════════════════════
#  GEOMETRIC SHAPES  (vectorised masks)
# ═════════════════════════════════════════════════════════════════════════════

def draw_shape(section, shape_type, color):
    h, w = section.shape[:2]
    if h < 2 or w < 2: return section
    cy, cx = h//2, w//2
    # Always produce 2D grids, never scalars
    ys = np.arange(h).reshape(h, 1)
    xs = np.arange(w).reshape(1, w)
    r = section.copy()
    dh = max(h/2, 1); dw = max(w/2, 1)
    color = list(color)[:3]
    if shape_type == 'diamond':
        m = (np.abs(ys-cy)/dh + np.abs(xs-cx)/dw) <= 0.9
        r[m] = color
        m2 = (np.abs(ys-cy)/(dh/1.6) + np.abs(xs-cx)/(dw/1.6)) <= 0.9
        r[m2] = [color[2], color[1], color[0]]
    elif shape_type == 'circle':
        r[((ys-cy)**2/dh**2 + (xs-cx)**2/dw**2) <= 0.81] = color
    elif shape_type == 'cross':
        t = max(1, min(h,w)//8)
        m = (np.abs(ys-cy) < t) | (np.abs(xs-cx) < t)
        r[m] = color
    elif shape_type == 'X':
        t = max(1, min(h,w)//10)
        m1 = np.abs((ys-cy)*w - (xs-cx)*h) < t*w
        m2 = np.abs((ys-cy)*w - (w-1-xs)*h) < t*w
        r[m1|m2] = color
    elif shape_type == 'oval':
        r[((ys-cy)**2/max((h/2.2)**2,1) + (xs-cx)**2/max((w/2.8)**2,1)) <= 1] = color
    elif shape_type == 'line':
        t = max(1, min(h,w)//8)
        if random.random() < 0.5: r[max(0,cy-t):cy+t, :] = color
        else:                     r[:, max(0,cx-t):cx+t] = color
    return r


# ═════════════════════════════════════════════════════════════════════════════
#  PROCESSOR
# ═════════════════════════════════════════════════════════════════════════════

class Processor:
    def __init__(self, image_path, max_dim=1500):
        raw = Image.open(image_path)
        self.dpi = raw.info.get('dpi', (300, 300))
        img = raw.convert("RGB")
        W, H = img.size
        if max(W, H) > max_dim:
            sc = max_dim / max(W, H)
            img = img.resize((int(W*sc), int(H*sc)), Image.LANCZOS)
        self.orig  = np.array(img)
        self.arr   = self.orig.copy()
        self.H, self.W = self.arr.shape[:2]
        self.gray  = cv2.cvtColor(self.orig, cv2.COLOR_RGB2GRAY)
        self._pal  = None
        log.info(f"Loaded {self.W}x{self.H}")

    def palette(self, n=8):
        if self._pal is None:
            self._pal = extract_palette(self.orig, n)
        return self._pal

    def rand_color(self):
        return self.palette()[random.randrange(len(self.palette()))].tolist()

    def _place(self, t, y0, x0):
        """Write t into self.arr at (y0,x0). Never raises on shape mismatch."""
        if t is None or t.size == 0: return
        # Ensure t is h×w×3
        if t.ndim == 2: t = np.stack([t]*3, axis=2)
        if t.shape[2] != 3: t = t[:,:,:3]
        th, tw = t.shape[:2]
        ay0, ax0 = max(y0, 0), max(x0, 0)
        ay1, ax1 = min(y0+th, self.H), min(x0+tw, self.W)
        if ay1 <= ay0 or ax1 <= ax0: return
        ty0, tx0 = ay0-y0, ax0-x0
        sh, sw = ay1-ay0, ax1-ax0
        src = t[ty0:ty0+sh, tx0:tx0+sw]
        ch, cw = min(src.shape[0], sh), min(src.shape[1], sw)
        self.arr[ay0:ay0+ch, ax0:ax0+cw] = src[:ch, :cw]

    # ── 1. Fractal mirroring ─────────────────────────────────────────────────
    def fractal_mirroring(self, section_size=60, max_depth=3, probability=0.7,
                          sym_types=None, pq=None):
        if sym_types is None: sym_types = ["quad","radial","vertical","horizontal"]
        step = max(1, section_size//2)
        rows = list(range(0, self.H, step))
        for i, y in enumerate(rows):
            for x in range(0, self.W, step):
                if random.random() < probability:
                    sz = min(random.randint(section_size//2, section_size),
                             self.W-x, self.H-y)
                    if sz >= 8:
                        self._rec_sym(x, y, sz, 0, max_depth, sym_types)
            if pq: pq.put((i+1, len(rows)))
        if pq: pq.put((1,1))

    def _rec_sym(self, x, y, sz, depth, max_depth, sym_types):
        if depth >= max_depth or sz < 8: return
        if x+sz > self.W or y+sz > self.H: return
        s = self.arr[y:y+sz, x:x+sz].copy()
        self._place(apply_symmetry(s, random.choice(sym_types)), y, x)
        nsz = sz//2
        if nsz >= 8:
            for dy, dx in [(0,0),(0,nsz),(nsz,0),(nsz,nsz)]:
                self._rec_sym(x+dx, y+dy, nsz, depth+1, max_depth, sym_types)

    # ── 2. Colour-region symmetry ────────────────────────────────────────────
    def color_region_symmetry(self, selection_method='brightness', sel_params=None,
                               num_areas=10, size_min=50, size_max=150, size_mode=100,
                               shape='square', sym_type='quad', pq=None):
        mask = get_mask(self.orig, selection_method, sel_params or {})
        labeled, nf = label(mask)
        n = min(num_areas, nf)
        c_val = (size_mode - size_min) / max(size_max - size_min, 1)
        dist = triang(c_val, loc=size_min, scale=max(size_max - size_min, 1))
        # Shuffle region indices so we sample from across the whole image,
        # not just the first n regions which are always at the top.
        region_ids = list(range(1, nf+1))
        random.shuffle(region_ids)
        region_ids = region_ids[:n]
        for i, reg_id in enumerate(region_ids):
            ys, xs = np.where(labeled == reg_id)
            if not len(ys): continue
            cy, cx = int(ys.mean()), int(xs.mean())
            r = int(dist.rvs()) // 2
            y0, x0 = max(0,cy-r), max(0,cx-r)
            y1, x1 = min(self.H,cy+r), min(self.W,cx+r)
            if y1<=y0 or x1<=x0: continue
            sec = self.orig[y0:y1, x0:x1].copy()
            t = apply_symmetry(sec, sym_type)
            if shape == 'circle':
                th, tw = t.shape[:2]
                gy, gx = np.ogrid[:th, :tw]
                cmask = (gx-tw//2)**2 + (gy-th//2)**2 <= (min(tw,th)//2)**2
                bg = self.arr[y0:y0+th, x0:x0+tw]
                if bg.shape == t.shape: t[~cmask] = bg[~cmask]
            self._place(t, y0, x0)
            if pq: pq.put((i+1, n))
        if pq: pq.put((1,1))

    # ── 3. Vertical structures ───────────────────────────────────────────────
    def vertical_structures(self, num=5, width_min=5, width_max=30,
                             pat_h_min=10, pat_h_max=50,
                             sym_types=None, h_ext_prob=0.15, pq=None):
        if sym_types is None: sym_types = SYMMETRY_NAMES[:6]
        for i in range(num):
            sw = random.randint(min(width_min,width_max), max(width_min,width_max))
            if self.W-sw <= 0: continue
            xp = random.randint(0, self.W-sw)
            ph = random.randint(min(pat_h_min,pat_h_max), max(pat_h_min,pat_h_max))
            if self.H-ph <= 0: continue
            ys = random.randint(0, self.H-ph)
            pat = apply_symmetry(self.arr[ys:ys+ph, xp:xp+sw].copy(),
                                 random.choice(sym_types))
            ph2 = max(1, pat.shape[0])
            pw2 = min(sw, pat.shape[1])
            for y in range(0, self.H, ph2):
                eh = min(y+ph2, self.H)
                rows = eh-y
                self.arr[y:eh, xp:xp+pw2] = pat[:rows, :pw2]
                if random.random() < h_ext_prob:
                    hw = random.randint(sw, min(self.W-xp, sw*3))
                    hh = random.randint(5, 15)
                    if y+hh < self.H and xp+hw <= self.W:
                        sec = self.arr[y:y+hh, xp:xp+hw].copy()
                        _safe_paste(self.arr, apply_symmetry(sec,"horizontal"), y, xp)
            if pq: pq.put((i+1, num))
        if pq: pq.put((1,1))

    # ── 3b. Horizontal structures ────────────────────────────────────────────
    def horizontal_structures(self, num=5, height_min=5, height_max=30,
                               pat_w_min=10, pat_w_max=50,
                               sym_types=None, v_ext_prob=0.15, pq=None):
        """Horizontal equivalent of vertical_structures: tiling rows across the image."""
        if sym_types is None: sym_types = SYMMETRY_NAMES[:6]
        for i in range(num):
            sh = random.randint(min(height_min,height_max), max(height_min,height_max))
            if self.H-sh <= 0: continue
            yp = random.randint(0, self.H-sh)
            pw = random.randint(min(pat_w_min,pat_w_max), max(pat_w_min,pat_w_max))
            if self.W-pw <= 0: continue
            xs = random.randint(0, self.W-pw)
            pat = apply_symmetry(self.arr[yp:yp+sh, xs:xs+pw].copy(),
                                 random.choice(sym_types))
            ph2 = min(sh, pat.shape[0])
            pw2 = max(1, pat.shape[1])
            for x in range(0, self.W, pw2):
                ew = min(x+pw2, self.W)
                cols = ew-x
                self.arr[yp:yp+ph2, x:x+cols] = pat[:ph2, :cols]
                if random.random() < v_ext_prob:
                    vw = random.randint(5, 15)
                    vh = random.randint(sh, min(self.H-yp, sh*3))
                    if yp+vh <= self.H and x+vw <= self.W:
                        sec = self.arr[yp:yp+vh, x:x+vw].copy()
                        _safe_paste(self.arr, apply_symmetry(sec, "vertical"), yp, x)
            if pq: pq.put((i+1, num))
        if pq: pq.put((1,1))

    # ── 4. Pattern cascade ───────────────────────────────────────────────────────
    def pattern_cascade(self, num=15, pw_min=5, pw_max=30, ph_min=5, ph_max=20,
                         blend_min=0.3, blend_max=0.7,
                         sym_types=None, h_ext_prob=0.1, pq=None):
        if sym_types is None: sym_types = ["vertical","horizontal","quad"]
        for i in range(num):
            pw = random.randint(min(pw_min,pw_max), max(pw_min,pw_max))
            ph = random.randint(min(ph_min,ph_max), max(ph_min,ph_max))
            xc = random.randint(pw//2, max(pw//2+1, self.W-pw//2-1))
            ys = random.randint(0, self.H//4)
            cl = random.randint(int(self.H*0.4), int(self.H*0.9))
            x0 = max(0,xc-pw//2); x1 = min(self.W,xc+pw//2)
            y1 = min(self.H, ys+ph)
            if x1<=x0 or y1<=ys: continue
            pat = apply_symmetry(self.arr[ys:y1, x0:x1].copy(), random.choice(sym_types))
            aph, apw = max(1,pat.shape[0]), max(1,pat.shape[1])
            for y in range(ys, min(ys+cl, self.H), aph):
                xoff = xc + int(math.sin(y/20)*pw)
                xl = max(0,xoff-apw//2); xr = min(self.W,xl+apw)
                yb = min(self.H, y+aph)
                dw, dh = xr-xl, yb-y
                if dw<=0 or dh<=0: continue
                pw2, ph2 = min(dw,apw), min(dh,aph)
                br = max(0.1, random.uniform(blend_min,blend_max)*(1-y/self.H*0.5))
                tgt = self.arr[y:y+ph2, xl:xl+pw2]
                self.arr[y:y+ph2, xl:xl+pw2] = (tgt*(1-br)+pat[:ph2,:pw2]*br).astype(np.uint8)
                if random.random() < h_ext_prob:
                    hw = random.randint(apw, min(self.W,apw*2))
                    hx = max(0,xoff-hw//2); hx2 = min(self.W,hx+hw)
                    if hx2>hx and yb>y:
                        hs = self.arr[y:yb, hx:hx2].copy()
                        _safe_paste(self.arr, apply_symmetry(hs,"horizontal"), y, hx)
                if random.random() < 0.2:
                    pat = apply_symmetry(pat, random.choice(sym_types))
                    aph, apw = max(1,pat.shape[0]), max(1,pat.shape[1])
            if pq: pq.put((i+1, num))
        if pq: pq.put((1,1))

    # ── 5. Diamond / geometric patterns ─────────────────────────────────────
    def diamond_patterns(self, num=20, size_min=20, size_max=100,
                          nest_layers=3, noise_strength=0.3,
                          organic_blend=0.4, sym_types=None, pq=None):
        """Nested multi-layer shapes with turbulence distortion and organic edge blending."""
        ALL_SHAPES = ["diamond","circle","cross","X","oval","line"]
        if sym_types is None: sym_types = ["vertical","horizontal","quad","radial","kaleidoscope3"]
        for i in range(num):
            sz = random.randint(min(size_min,size_max), max(size_min,size_max))
            xc = random.randint(0,self.W-1); yc = random.randint(0,self.H-1)
            x0 = max(0,xc-sz//2); x1 = min(self.W,xc+sz//2)
            y0 = max(0,yc-sz//2); y1 = min(self.H,yc+sz//2)
            if x1-x0<6 or y1-y0<6: continue
            sec = self.arr[y0:y1, x0:x1].copy()
            h, w = sec.shape[:2]

            # sample colors locally from the section itself so they harmonise
            local_pixels = sec.reshape(-1,3).astype(np.float32)
            n_local = min(5, max(1, len(np.unique(local_pixels, axis=0))))
            try:
                km = MiniBatchKMeans(n_clusters=n_local, random_state=42, n_init='auto')
                km.fit(local_pixels[np.random.choice(len(local_pixels),
                                   min(500,len(local_pixels)), replace=False)])
                local_pal = km.cluster_centers_.astype(np.uint8)
            except Exception:
                local_pal = np.array([sec.mean(axis=(0,1))], dtype=np.uint8)

            # build nested layers from large to small
            layer = sec.copy()
            n_layers = max(1, nest_layers)
            for d in range(n_layers):
                scale = 1.0 - d * (0.55 / max(n_layers,1))
                lh = max(4, int(h * scale)); lw = max(4, int(w * scale))
                cy2, cx2 = h//2, w//2
                ly0 = max(0, cy2 - lh//2); ly1 = min(h, ly0 + lh)
                lx0 = max(0, cx2 - lw//2); lx1 = min(w, lx0 + lw)
                if ly1<=ly0 or lx1<=lx0: continue
                patch = layer[ly0:ly1, lx0:lx1].copy()
                color = local_pal[random.randrange(len(local_pal))].tolist()
                # rotate color slightly each layer for variety
                shift = d * 40
                color = [min(255,c+shift) if j==d%3 else max(0,c-shift//2)
                         for j,c in enumerate(color)]
                patch = draw_shape(patch, random.choice(ALL_SHAPES), color)
                patch = apply_symmetry(patch, random.choice(sym_types))
                # blend patch back rather than hard-replace
                blend_w = max(0.0, min(1.0, 0.5 + d * 0.15))
                existing = layer[ly0:ly1, lx0:lx1]
                ph = min(patch.shape[0], existing.shape[0])
                pw = min(patch.shape[1], existing.shape[1])
                layer[ly0:ly0+ph, lx0:lx0+pw] = (
                    existing[:ph,:pw] * (1-blend_w) + patch[:ph,:pw] * blend_w
                ).astype(np.uint8)

            # turbulence / noise distortion
            if noise_strength > 0:
                ns = max(0.0, min(1.0, noise_strength))
                noise_x = (np.random.rand(h, w).astype(np.float32) - 0.5) * ns * w * 0.15
                noise_y = (np.random.rand(h, w).astype(np.float32) - 0.5) * ns * h * 0.15
                # add low-freq warp on top
                freq = random.uniform(2.0, 5.0)
                ys_g = np.linspace(0, freq*math.pi, h)[:,None]
                xs_g = np.linspace(0, freq*math.pi, w)[None,:]
                noise_x += np.sin(ys_g + random.uniform(0,math.pi)) * ns * w * 0.08
                noise_y += np.cos(xs_g + random.uniform(0,math.pi)) * ns * h * 0.08
                grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
                map_x = np.clip(grid_x + noise_x, 0, w-1).astype(np.float32)
                map_y = np.clip(grid_y + noise_y, 0, h-1).astype(np.float32)
                layer = cv2.remap(layer, map_x, map_y, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)

            # organic edge blend — feather the boundary so it grows into surroundings
            if organic_blend > 0:
                ob = max(0.0, min(1.0, organic_blend))
                dist_y = np.minimum(np.arange(h), np.arange(h,0,-1)).reshape(h,1).astype(np.float32)
                dist_x = np.minimum(np.arange(w), np.arange(w,0,-1)).reshape(1,w).astype(np.float32)
                feather = np.minimum(dist_y, dist_x)
                feather_r = max(1, min(h,w) // 4)
                alpha = np.clip(feather / feather_r, 0, 1)[..., None] * ob
                layer = (sec * (1 - alpha) + layer * alpha).astype(np.uint8)

            self._place(layer, y0, x0)
            if pq: pq.put((i+1, num))
        if pq: pq.put((1,1))

    # ── 6. Pixel sorting ─────────────────────────────────────────────────────
    def pixel_sorting(self, num=10, w_min=50, w_max=200, h_min=50, h_max=200,
                       directions=None, sort_by='brightness', pq=None):
        if directions is None: directions = ["horizontal","vertical","both"]
        for i in range(num):
            sw = random.randint(min(w_min,w_max), max(w_min,w_max))
            sh = random.randint(min(h_min,h_max), max(h_min,h_max))
            if self.W-sw<=0 or self.H-sh<=0: continue
            x0 = random.randint(0,self.W-sw); y0 = random.randint(0,self.H-sh)
            sec = self.arr[y0:y0+sh, x0:x0+sw].copy()
            direction = random.choice(directions)
            def key(pixels_2d):
                if sort_by == 'brightness': return np.mean(pixels_2d, axis=1)
                if sort_by == 'red':        return pixels_2d[:,0].astype(float)
                if sort_by == 'green':      return pixels_2d[:,1].astype(float)
                if sort_by == 'hue':
                    return np.array([colorsys.rgb_to_hsv(p[0]/255,p[1]/255,p[2]/255)[0]
                                     for p in pixels_2d])
                return np.mean(pixels_2d, axis=1)
            if direction in ("horizontal","both"):
                for r in range(sec.shape[0]):
                    sec[r] = sec[r, np.argsort(key(sec[r]))]
            if direction in ("vertical","both"):
                for c in range(sec.shape[1]):
                    sec[:,c] = sec[np.argsort(key(sec[:,c])), c]
            br = random.uniform(0.5, 0.9)
            dst = self.arr[y0:y0+sh, x0:x0+sw]
            blended = (dst[:min(sh,sec.shape[0]),:min(sw,sec.shape[1])]*(1-br)+sec[:min(sh,sec.shape[0]),:min(sw,sec.shape[1])]*br).astype(np.uint8)
            _safe_paste(self.arr, blended, y0, x0)
            if pq: pq.put((i+1, num))
        if pq: pq.put((1,1))

    # ── 7. Geometric grid ────────────────────────────────────────────────────
    def geometric_grid(self, grid_size=40, cell_prob=0.7,
                        sub_layers=2, sym_types=None, pq=None):
        """Richer grid: multiple shape layers per cell, colors sampled locally so
        they harmonise with the image, optional rotation, and varied cell sizes."""
        ALL_SHAPES = ["diamond","circle","cross","X","oval","line"]
        if sym_types is None: sym_types = ["vertical","horizontal","quad","radial","kaleidoscope3"]
        rows = list(range(0, self.H, grid_size))
        total = max(1, len(rows) * max(1, (self.W + grid_size - 1) // grid_size))
        done = 0
        for i, y in enumerate(rows):
            for x in range(0, self.W, grid_size):
                done += 1
                if random.random() > cell_prob:
                    if pq: pq.put((done, total))
                    continue
                # slight size jitter so cells don't feel perfectly uniform
                jitter = random.randint(-grid_size//4, grid_size//4)
                gs = max(6, grid_size + jitter)
                xe = min(x+gs, self.W); ye = min(y+gs, self.H)
                if xe<=x or ye<=y:
                    if pq: pq.put((done, total)); continue
                cell = self.arr[y:ye, x:xe].copy()
                ch, cw = cell.shape[:2]

                # sample colors from the local cell so they stay close to the image
                flat = cell.reshape(-1, 3).astype(np.float32)
                n_c = min(4, max(1, len(np.unique(flat, axis=0))))
                try:
                    km = MiniBatchKMeans(n_clusters=n_c, random_state=42, n_init='auto')
                    km.fit(flat[np.random.choice(len(flat), min(300,len(flat)), replace=False)])
                    lpal = km.cluster_centers_.astype(np.uint8)
                except Exception:
                    lpal = np.array([flat.mean(axis=0)], dtype=np.uint8)

                # apply a symmetry to the whole cell first as a base
                cell = apply_symmetry(cell, random.choice(sym_types))

                # draw sub_layers nested shapes at decreasing scales
                n_sub = max(1, sub_layers)
                for d in range(n_sub):
                    scale = 1.0 - d * (0.45 / max(n_sub, 1))
                    sh = max(4, int(ch * scale)); sw2 = max(4, int(cw * scale))
                    sy0 = (ch - sh) // 2; sx0 = (cw - sw2) // 2
                    sy1 = sy0 + sh; sx1 = sx0 + sw2
                    if sy1<=sy0 or sx1<=sx0: continue
                    patch = cell[sy0:sy1, sx0:sx1].copy()
                    color = lpal[random.randrange(len(lpal))].tolist()
                    # shift hue slightly per layer so nested shapes have variety
                    # without straying far from local palette
                    tweak = random.randint(-25, 25)
                    color = [int(np.clip(c + tweak * ((-1)**j), 0, 255))
                             for j, c in enumerate(color)]
                    patch = draw_shape(patch, random.choice(ALL_SHAPES), color)
                    patch = apply_symmetry(patch, random.choice(sym_types))
                    # blend with increasing opacity for inner layers
                    bw = 0.55 + d * 0.2
                    bw = min(1.0, bw)
                    existing = cell[sy0:sy1, sx0:sx1]
                    ph2 = min(patch.shape[0], existing.shape[0])
                    pw2 = min(patch.shape[1], existing.shape[1])
                    cell[sy0:sy0+ph2, sx0:sx0+pw2] = (
                        existing[:ph2,:pw2] * (1-bw) + patch[:ph2,:pw2] * bw
                    ).astype(np.uint8)

                # optional random rotation of the whole composed cell
                if random.random() < 0.3:
                    angle = random.choice([90, 180, 270])
                    k = angle // 90
                    rotated = np.rot90(cell, k)
                    if rotated.shape[:2] != (ch, cw):
                        tmp = cell.copy()
                        rh = min(rotated.shape[0], ch); rw = min(rotated.shape[1], cw)
                        tmp[:rh, :rw] = rotated[:rh, :rw]
                        cell = tmp
                    else:
                        cell = rotated

                _safe_paste(self.arr, cell, y, x)
                if pq: pq.put((done, total))
        if pq: pq.put((1,1))

    # ── 8. Recursive glitch ──────────────────────────────────────────────────
    def recursive_glitch(self, iterations=3, pq=None):
        H, W = self.H, self.W
        for i in range(iterations):
            mode = i % 3
            if mode == 0:  # horizontal band shift
                for _ in range(random.randint(5,15)):
                    bh = H // random.randint(5,15)
                    y0 = random.randint(0, H-bh)
                    sh = random.randint(-W//10, W//10)
                    if sh: self.arr[y0:y0+bh] = np.roll(self.arr[y0:y0+bh], sh, axis=1)
            elif mode == 1:  # selective mirror
                for _ in range(random.randint(3,8)):
                    mw = random.randint(W//10,W//3); mh = random.randint(H//10,H//3)
                    mx = random.randint(0,W-mw);     my = random.randint(0,H-mh)
                    sec = self.arr[my:my+mh, mx:mx+mw].copy()
                    sym = "horizontal" if random.random()<0.5 else "vertical"
                    _safe_paste(self.arr, apply_symmetry(sec, sym), my, mx)
            else:  # pattern duplication
                for _ in range(random.randint(5,15)):
                    sw = random.randint(W//20,W//8); sh_ = random.randint(H//20,H//8)
                    sx = random.randint(0,W-sw);     sy = random.randint(0,H-sh_)
                    dx = random.randint(0,W-sw);     dy = random.randint(0,H-sh_)
                    pat = self.arr[sy:sy+sh_, sx:sx+sw].copy()
                    t = random.randint(0,3)
                    if t==1: pat=pat[:,::-1]
                    elif t==2: pat=pat[::-1,:]
                    elif t==3: pat=np.rot90(pat)[:sh_,:sw]
                    br = random.uniform(0.5,0.9)
                    dst = self.arr[dy:dy+sh_, dx:dx+sw]
                    ph=min(pat.shape[0],dst.shape[0]); pw=min(pat.shape[1],dst.shape[1])
                    _safe_paste(self.arr, (dst[:ph,:pw]*(1-br)+pat[:ph,:pw]*br).astype(np.uint8), dy, dx)
            if pq: pq.put((i+1, iterations))
        if pq: pq.put((1,1))

    # ── 9. Vertical cascade ──────────────────────────────────────────────────
    def vertical_cascade(self, intensity=0.8, bleed=0.4, wave_amp=0.3,
                          plasma_strength=0.4, branch_prob=0.2, pq=None):
        """Rich cascade: multi-column bleeding, sine-wave distortion trails,
        plasma colour smearing, and gravity-drip branching."""
        H, W = self.H, self.W
        vk = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
        vert = np.abs(cv2.filter2D(self.gray.astype(np.float32), -1, vk))
        thresh = np.percentile(vert, 85)
        cols = [x for x in range(W) if np.mean(vert[:,x]) > thresh]
        cols += random.choices(range(W), k=max(1, len(cols)//2))
        random.shuffle(cols)

        # pre-build a plasma colour map for smearing
        plasma = np.zeros((H, W, 3), dtype=np.float32)
        for freq in [0.03, 0.07, 0.013]:
            ys = np.arange(H)[:,None]; xs = np.arange(W)[None,:]
            ph = random.uniform(0, 2*math.pi); pv = random.uniform(0, 2*math.pi)
            plasma[...,0] += np.sin(xs*freq + ph) * 0.5 + 0.5
            plasma[...,1] += np.cos(ys*freq + pv) * 0.5 + 0.5
            plasma[...,2] += np.sin((xs+ys)*freq) * 0.5 + 0.5
        plasma = (plasma / plasma.max() * 255).astype(np.uint8)

        for i, x in enumerate(cols):
            # sample seed color from the image at this column
            seed_col = self.arr[:, x].mean(axis=0).astype(float)
            cl = random.randint(int(H*0.3), int(H*0.95))
            sy = random.randint(0, max(0, H-cl))
            wave_freq = random.uniform(2.0, 8.0)
            wave_phase = random.uniform(0, 2*math.pi)
            branch_width = random.randint(1, max(1, W//30))

            current_col = seed_col.copy()
            for y in range(sy, min(sy+cl, H)):
                progress = (y - sy) / max(cl, 1)
                fade = intensity * (1.0 - progress * 0.7)

                # sine-wave lateral offset for the drip
                ox = int(wave_amp * branch_width * math.sin(wave_freq * progress + wave_phase))
                cx = int(np.clip(x + ox, 0, W-1))

                # plasma colour blend into the drip colour
                if plasma_strength > 0:
                    ps = max(0.0, min(1.0, plasma_strength))
                    pc = plasma[y, cx].astype(float)
                    current_col = current_col * (1 - ps*0.1) + pc * (ps*0.1)

                # write to main column with fade
                new_px = np.clip(self.arr[y,cx]*(1-fade) + current_col*fade, 0, 255).astype(np.uint8)
                self.arr[y, cx] = new_px

                # multi-column bleed: smear colour into neighbours
                bleed_r = max(0.0, min(1.0, bleed))
                if bleed_r > 0:
                    bleed_w = max(1, int(bleed_r * branch_width * 3))
                    for dx in range(1, bleed_w+1):
                        falloff = bleed_r * (1 - dx / (bleed_w+1))
                        for nx in [cx+dx, cx-dx]:
                            if 0 <= nx < W:
                                self.arr[y,nx] = np.clip(
                                    self.arr[y,nx]*(1-falloff) + new_px*falloff, 0,255
                                ).astype(np.uint8)

                # branching: spawn a short horizontal splash
                if random.random() < branch_prob * (1 - progress):
                    branch_len = random.randint(3, max(4, W//15))
                    branch_dir = random.choice([-1, 1])
                    for bx_off in range(1, branch_len+1):
                        bx = int(np.clip(cx + branch_dir*bx_off, 0, W-1))
                        bfade = fade * (1 - bx_off/branch_len)
                        self.arr[y, bx] = np.clip(
                            self.arr[y,bx]*(1-bfade) + current_col*bfade, 0,255
                        ).astype(np.uint8)

                # drift the seed colour slightly
                current_col = np.clip(
                    current_col + np.random.randint(-12, 13, size=3).astype(float), 0, 255
                )

            if pq: pq.put((i+1, len(cols)))
        if pq: pq.put((1,1))

    # ── 10. Fractal draw ─────────────────────────────────────────────────────
    def fractal_draw(self, max_depth=5, blend=0.4, contour_levels=8,
                     ifs_points=15000, pq=None):
        """Three-layer fractal draw:
        1. Recursive subdivision with full color, rotation and richer shapes.
        2. Iso-contour lines following image tones.
        3. IFS (iterated function system) dot cloud blended as scatter."""
        H, W = self.H, self.W
        norm = self.gray.astype(float) / 255
        vk = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
        vedge = np.abs(cv2.filter2D(self.gray.astype(np.float32), -1, vk))
        vedge_n = (vedge / max(vedge.max(), 1))
        interest = np.abs(self.gray.astype(int) - 128) / 128
        pal = self.palette(12)

        # ── Layer 1: recursive colored subdivision ───────────────────────────
        color_canvas = self.arr.copy().astype(np.float32)
        ALL_SHAPES = ['circle','cross','diamond','oval','X','line']

        def _rdraw(x1, y1, x2, y2, depth, shape_hint, rot_hint):
            if x2-x1 < 4 or y2-y1 < 4 or depth >= max_depth: return
            rm  = norm[y1:y2, x1:x2].mean()
            rc  = norm[y1:y2, x1:x2].std()
            il  = interest[y1:y2, x1:x2].mean()
            vs  = vedge_n[y1:y2, x1:x2].mean()
            mx, my = (x1+x2)//2, (y1+y2)//2

            # pick shape from image character if not inherited
            if shape_hint is None:
                if vs > 0.25:   shape_hint = 'cross'
                elif rm < 0.3:  shape_hint = 'diamond'
                elif rm > 0.7:  shape_hint = 'circle'
                elif rc > 0.2:  shape_hint = 'oval'
                else:           shape_hint = random.choice(ALL_SHAPES)

            # color sampled from palette, shifted by depth & local brightness
            base_col = pal[int(rm * (len(pal)-1))].astype(float)
            depth_shift = depth * 18
            color = np.clip(base_col + depth_shift * np.array([1,-0.5,-1]), 0, 255)

            sc = max(0.1, (1 - depth/max_depth*0.4) * (0.4 + il))
            rw, rh = x2-x1, y2-y1
            patch = color_canvas[y1:y2, x1:x2].copy().astype(np.uint8)
            patch = draw_shape(patch, shape_hint, color.tolist())

            # rotation at each level
            if rot_hint and rw > 8 and rh > 8:
                angle_deg = rot_hint
                M = cv2.getRotationMatrix2D((rw//2, rh//2), float(angle_deg), sc)
                patch = cv2.warpAffine(patch, M, (rw, rh),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT)

            alpha = 0.3 + il * 0.5
            color_canvas[y1:y2, x1:x2] = (
                color_canvas[y1:y2, x1:x2] * (1-alpha) + patch * alpha
            )

            nd = depth if il > 0.65 else depth + 1
            next_rot = random.choice([0, 15, 30, 45, 60, 90]) if random.random() < 0.6 else None
            next_shape = shape_hint if random.random() < 0.6 else None
            for ax,ay,bx,by in [(x1,y1,mx,my),(mx,y1,x2,my),(x1,my,mx,y2),(mx,my,x2,y2)]:
                _rdraw(ax, ay, bx, by, nd, next_shape, next_rot)

        divs = 4 if max(W,H) > 1000 else 3
        cw2, ch2 = W//divs, H//divs
        for i in range(divs):
            for j in range(divs):
                x1, y1 = j*cw2, i*ch2
                rm = norm[y1:y1+ch2, x1:x1+cw2].mean()
                vs = vedge_n[y1:y1+ch2, x1:x1+cw2].mean()
                pt = 'cross' if vs>0.25 else ('diamond' if rm<0.3 else ('circle' if rm>0.7 else 'oval'))
                _rdraw(x1, y1, x1+cw2, y1+ch2, 0, pt, random.choice([0,45,90]))
            if pq: pq.put((i+1, divs+2))

        # ── Layer 2: iso-contour lines ───────────────────────────────────────
        contour_canvas = np.zeros((H, W, 3), dtype=np.float32)
        levels = np.linspace(0.05, 0.95, max(2, contour_levels))
        for lv in levels:
            lo, hi = lv - 0.04, lv + 0.04
            band = ((norm >= lo) & (norm < hi)).astype(np.uint8)
            # find contour of the band
            cnts, _ = cv2.findContours(band, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            col_idx = int(lv * (len(pal)-1))
            col = pal[col_idx].tolist()
            tmp = np.zeros((H, W, 3), dtype=np.uint8)
            cv2.drawContours(tmp, cnts, -1, col, 1)
            contour_canvas += tmp.astype(np.float32)
        contour_canvas = np.clip(contour_canvas, 0, 255)
        if pq: pq.put((divs+1, divs+2))

        # ── Layer 3: IFS dot cloud ────────────────────────────────────────────
        ifs_canvas = np.zeros((H, W, 3), dtype=np.float32)
        # use a random barnsley-fern-style IFS with palette colors
        # define 4-6 affine transforms
        n_transforms = random.randint(4, 6)
        transforms = []
        for _ in range(n_transforms):
            a = random.uniform(-0.8, 0.8); b = random.uniform(-0.4, 0.4)
            c = random.uniform(-0.4, 0.4); d = random.uniform(-0.8, 0.8)
            e = random.uniform(-0.3, 0.3); f = random.uniform(-0.3, 0.3)
            transforms.append((a, b, c, d, e, f))
        px2, py2 = 0.0, 0.0
        n_pts = max(1000, ifs_points)
        # warmup
        for _ in range(50):
            a,b,c,d,e,f = random.choice(transforms)
            px2, py2 = a*px2+b*py2+e, c*px2+d*py2+f
        for k in range(n_pts):
            a,b,c,d,e,f = random.choice(transforms)
            px2, py2 = a*px2+b*py2+e, c*px2+d*py2+f
            # map [-1,1] range to image coords
            ix = int((px2 * 0.4 + 0.5) * W)
            iy = int((py2 * 0.4 + 0.5) * H)
            if 0 <= ix < W and 0 <= iy < H:
                col_idx = int((norm[iy,ix]) * (len(pal)-1))
                ifs_canvas[iy, ix] += pal[col_idx].astype(float) * 0.6

        ifs_canvas = np.clip(ifs_canvas, 0, 255)

        # ── Composite all three layers ────────────────────────────────────────
        b1 = max(0.0, min(1.0, blend))
        b2 = b1 * 0.6   # contours a bit lighter
        b3 = b1 * 0.4   # IFS subtle
        out = self.arr.astype(np.float32)
        out = out*(1-b1) + color_canvas*b1
        out = out*(1-b2) + contour_canvas*b2
        out = out*(1-b3) + ifs_canvas*b3
        self.arr = np.clip(out, 0, 255).astype(np.uint8)
        if pq: pq.put((1,1))

    # ── 11. Grid Transform ─────────────────────────────────────────────────
    def grid_transform(self, cell_w=100, cell_h=100, cell_prob=1.0,
                       sym_type='quad', operation='symmetry_only',
                       blend=1.0, pq=None):
        ops_pool = ['symmetry_only','fractal_mirror','pixel_sort_h','pixel_sort_v',
                    'rotate_contents','mirror_h','mirror_v','kaleidoscope',
                    'invert','shuffle_rows','shuffle_cols','reverse_rows',
                    'reverse_cols','channel_shift','diagonal_mirror']
        cw = max(2, cell_w)
        ch = max(2, cell_h)
        rows_list = list(range(0, self.H, ch))
        total = max(1, len(rows_list) * max(1, (self.W + cw - 1) // cw))
        done = 0
        for y0 in rows_list:
            y1 = min(y0 + ch, self.H)
            if y1 <= y0: continue
            for x0 in range(0, self.W, cw):
                x1 = min(x0 + cw, self.W)
                if x1 <= x0: continue
                if random.random() > cell_prob:
                    done += 1; continue
                cell = self.arr[y0:y1, x0:x1].copy()
                op = random.choice(ops_pool) if operation == 'random' else operation
                try:
                    if op == 'symmetry_only':
                        out = apply_symmetry(cell, sym_type)
                    elif op == 'fractal_mirror':
                        out = cell.copy()
                        mh, mw = cell.shape[0]//2, cell.shape[1]//2
                        if mh > 0 and mw > 0:
                            tl = cell[:mh, :mw].copy()
                            out[:mh, :mw] = tl
                            out[:mh, mw:] = tl[:, ::-1][:, :cell.shape[1]-mw]
                            out[mh:, :mw] = tl[::-1][:cell.shape[0]-mh]
                            out[mh:, mw:] = tl[::-1, ::-1][:cell.shape[0]-mh, :cell.shape[1]-mw]
                    elif op == 'pixel_sort_h':
                        out = cell.copy()
                        for r in range(out.shape[0]):
                            out[r] = out[r, np.argsort(np.mean(out[r], axis=1))]
                    elif op == 'pixel_sort_v':
                        out = cell.copy()
                        for c in range(out.shape[1]):
                            out[:, c] = out[np.argsort(np.mean(out[:, c], axis=1)), c]
                    elif op == 'rotate_contents':
                        r90 = np.rot90(cell)
                        if r90.shape[:2] != cell.shape[:2]:
                            from PIL import Image as _PIL
                            r90 = np.array(_PIL.fromarray(r90).resize(
                                (cell.shape[1], cell.shape[0]), _PIL.LANCZOS))
                        out = r90[:cell.shape[0], :cell.shape[1]]
                    elif op == 'mirror_h':
                        out = cell[:, ::-1].copy()
                    elif op == 'mirror_v':
                        out = cell[::-1, :].copy()
                    elif op == 'kaleidoscope':
                        out = apply_symmetry(cell, 'kaleidoscope3')
                    elif op == 'invert':
                        out = (255 - cell).astype(np.uint8)
                    elif op == 'shuffle_rows':
                        idx = list(range(cell.shape[0])); random.shuffle(idx)
                        out = cell[idx]
                    elif op == 'shuffle_cols':
                        idx = list(range(cell.shape[1])); random.shuffle(idx)
                        out = cell[:, idx]
                    elif op == 'reverse_rows':
                        out = cell[:, ::-1].copy()
                    elif op == 'reverse_cols':
                        out = cell[::-1, :].copy()
                    elif op == 'channel_shift':
                        shift = random.randint(1, max(1, min(cell.shape[0], cell.shape[1])//4))
                        out = cell.copy()
                        out[:,:,0] = np.roll(cell[:,:,0],  shift, axis=1)
                        out[:,:,1] = np.roll(cell[:,:,1], -shift, axis=0)
                        out[:,:,2] = np.roll(cell[:,:,2],  shift, axis=0)
                    elif op == 'diagonal_mirror':
                        t = np.transpose(cell, (1,0,2))
                        if t.shape[:2] != cell.shape[:2]:
                            from PIL import Image as _PIL
                            t = np.array(_PIL.fromarray(t).resize(
                                (cell.shape[1], cell.shape[0]), _PIL.LANCZOS))
                        out = t[:cell.shape[0], :cell.shape[1]]
                    else:
                        out = apply_symmetry(cell, sym_type)
                    # Guarantee shape/dtype
                    if out.shape != cell.shape:
                        tmp = cell.copy()
                        ch2 = min(out.shape[0], cell.shape[0])
                        cw2 = min(out.shape[1], cell.shape[1])
                        tmp[:ch2, :cw2] = out[:ch2, :cw2]
                        out = tmp
                    out = np.clip(out, 0, 255).astype(np.uint8)
                    if blend >= 1.0:
                        self.arr[y0:y1, x0:x1] = out
                    else:
                        self.arr[y0:y1, x0:x1] = (cell*(1-blend)+out*blend).astype(np.uint8)
                except Exception as e:
                    log.warning(f"grid_transform cell ({x0},{y0}): {e}")
                done += 1
                if pq: pq.put((done, total))
        if pq: pq.put((1,1))

    # ── 12. Mirroring sections ───────────────────────────────────────────────
    def mirror_sections(self, section_height=None, pq=None):
        if section_height is None: section_height = self.H//3
        section_height = max(2, section_height)
        rows = list(range(0, self.H, section_height))
        for i,y in enumerate(rows):
            ye = min(y+section_height, self.H)
            sec = self.arr[y:ye].copy()
            h = sec.shape[0]
            mid = h//2
            if mid > 0:
                top = sec[:mid]          # first half
                bot = top[::-1]          # flipped copy
                n = h - mid              # rows available below mid
                sec[mid:mid+min(bot.shape[0],n)] = bot[:n]
            _safe_paste(self.arr, sec, y, 0)
            if pq: pq.put((i+1,len(rows)))
        if pq: pq.put((1,1))

    # ── 13. Blend with original ──────────────────────────────────────────────
    def blend_original(self, ratio=0.2, pq=None):
        self.arr = np.clip(self.arr*(1-ratio)+self.orig*ratio,0,255).astype(np.uint8)
        if pq: pq.put((1,1))

    # ── 14. Final post-processing ────────────────────────────────────────────
    def final_effects(self, contrast=1.3, sharpen=True, vignette=False, pq=None):
        img = Image.fromarray(self.arr)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        if sharpen: img = img.filter(ImageFilter.SHARPEN)
        if vignette:
            mask = Image.new('L', img.size, 255)
            d = ImageDraw.Draw(mask)
            w,h = img.size
            for off,op in [(0,255),(20,200),(50,100),(80,0)]:
                d.rectangle([off,off,w-off,h-off], fill=op)
            mask = mask.filter(ImageFilter.GaussianBlur(30))
            img.putalpha(mask); img = img.convert("RGB")
        self.arr = np.array(img)
        if pq: pq.put((1,1))

    def result_image(self):
        return Image.fromarray(self.arr)

    def save(self, path):
        img = self.result_image()
        ext = os.path.splitext(path)[1].lower()
        kw = {}
        if self.dpi: kw['dpi'] = self.dpi
        if ext in ('.jpg','.jpeg'): kw['quality']=95; kw['subsampling']=0
        elif ext=='.png': kw['compress_level']=1
        img.save(path, **kw)
        log.info(f"Saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
#  OPERATION REGISTRY
# ═════════════════════════════════════════════════════════════════════════════

# Individual symmetry type by index (used when _sym_preset >= 3)
def _resolve_sym_preset(v):
    if v == 0: return ["quad","radial"]
    if v == 1: return SYMMETRY_NAMES
    if v == 2: return ["vertical","horizontal"]
    if v == 3: return ["kaleidoscope3","kaleidoscope6"]
    if v == 4: return ["wave","spiral"]
    if v == 5: return ["fractal","wallpaper","reflective_tiling"]
    if v == 6: return ["rotate90","rotate180","glide"]
    # v >= 10: single type by index into SYMMETRY_NAMES
    idx = v - 10
    if 0 <= idx < len(SYMMETRY_NAMES): return [SYMMETRY_NAMES[idx]]
    return ["quad","radial"]

SYM_PRESETS = {i: _resolve_sym_preset(i) for i in range(7)}
# Add individual entries
for _i, _n in enumerate(SYMMETRY_NAMES):
    SYM_PRESETS[10+_i] = [_n]

_SYM_PRESET_OPTIONS = [
    "Quad + Radial",
    "All Types",
    "Vertical + Horizontal",
    "Kaleidoscope 3+6",
    "Wave + Spiral",
    "Fractal / Wallpaper / Tiling",
    "Rotate / Glide",
] + SYMMETRY_NAMES  # individual types start at index 7 in the list, offset 10 in key


def resolve_params(op_key, raw):
    """Translate raw GUI param values into kwargs for Processor methods."""
    kw = {}
    for k,v in raw.items():
        if k=='_prob_pct':        kw['probability']      = v/100
        elif k=='_sym_preset':    kw['sym_types']        = _resolve_sym_preset(v if v < 7 else 10 + (v - 7))
        elif k=='_bmin':          kw['blend_min']        = v/100
        elif k=='_bmax':          kw['blend_max']        = v/100
        elif k=='_cell_prob_pct': kw['cell_prob']        = v/100
        elif k=='_contrast_x10':  kw['contrast']         = v/10
        elif k=='_noise_pct':     kw['noise_strength']   = v/100
        elif k=='_organic_pct':   kw['organic_blend']    = v/100
        elif k=='_bleed_pct':     kw['bleed']            = v/100
        elif k=='_wave_amp_pct':  kw['wave_amp']         = v/100
        elif k=='_plasma_pct':    kw['plasma_strength']  = v/100
        elif k=='_branch_pct':    kw['branch_prob']      = v/100
        else:                     kw[k] = v
    # selection_method is a choice index → name
    SEL_METHODS = ['brightness','red','green','blue','color_range',
                   'edge_detection','hsv','texture','segmentation','entropy','random']
    if 'selection_method' in kw and isinstance(kw['selection_method'], int):
        kw['selection_method'] = SEL_METHODS[kw['selection_method']]
    if 'sym_type' in kw and isinstance(kw['sym_type'], int):
        kw['sym_type'] = SYMMETRY_NAMES[kw['sym_type']]
    if 'sort_by' in kw and isinstance(kw['sort_by'], int):
        kw['sort_by'] = ['brightness','red','green','hue'][kw['sort_by']]
    if 'shape' in kw and isinstance(kw['shape'], int):
        kw['shape'] = ['square','circle'][kw['shape']]
    return kw

OPS = [
  ("Fractal Mirroring",       "fractal_mirroring",    [
      ("Section Size",  dict(min=4,   max=2000, default=60,  key="section_size",   kind="int")),
      ("Max Depth",     dict(min=1,   max=10,   default=3,   key="max_depth",      kind="int")),
      ("Probability %", dict(min=1,   max=100,  default=70,  key="_prob_pct",      kind="int")),
      ("Symmetry",      dict(options=_SYM_PRESET_OPTIONS,
                             default=0, key="_sym_preset", kind="choice")),
  ]),
  ("Colour Region Symmetry",  "color_region_symmetry",[
      ("Selection",     dict(options=['brightness','red','green','blue','color_range',
                                      'edge_detection','hsv','texture','segmentation',
                                      'entropy','random'],
                             default=0, key="selection_method", kind="choice")),
      ("Num Areas",     dict(min=1,   max=10000, default=10,  key="num_areas",  kind="int")),
      ("Size Min",      dict(min=1,   max=3000,  default=50,  key="size_min",   kind="int")),
      ("Size Max",      dict(min=1,   max=5000,  default=150, key="size_max",   kind="int")),
      ("Size Mode",     dict(min=1,   max=4000,  default=100, key="size_mode",  kind="int")),
      ("Shape",         dict(options=["square","circle"], default=0, key="shape",     kind="choice")),
      ("Symmetry",      dict(options=SYMMETRY_NAMES,      default=3, key="sym_type",  kind="choice")),
  ]),
  ("Vertical Structures",     "vertical_structures",  [
      ("Count",         dict(min=1,   max=500,  default=5,   key="num",        kind="int")),
      ("Width Min",     dict(min=1,   max=3000, default=5,   key="width_min",  kind="int")),
      ("Width Max",     dict(min=1,   max=3000, default=30,  key="width_max",  kind="int")),
      ("Pat H Min",     dict(min=1,   max=3000, default=10,  key="pat_h_min",  kind="int")),
      ("Pat H Max",     dict(min=1,   max=3000, default=50,  key="pat_h_max",  kind="int")),
      ("H Ext Prob %",  dict(min=0,   max=100,  default=15,  key="_h_ext",     kind="int")),
      ("Symmetry",      dict(options=_SYM_PRESET_OPTIONS,  default=0, key="_sym_preset", kind="choice")),
  ]),
  ("Horizontal Structures",   "horizontal_structures", [
      ("Count",         dict(min=1,   max=500,  default=5,   key="num",          kind="int")),
      ("Height Min",    dict(min=1,   max=3000, default=5,   key="height_min",   kind="int")),
      ("Height Max",    dict(min=1,   max=3000, default=30,  key="height_max",   kind="int")),
      ("Pat W Min",     dict(min=1,   max=3000, default=10,  key="pat_w_min",    kind="int")),
      ("Pat W Max",     dict(min=1,   max=3000, default=50,  key="pat_w_max",    kind="int")),
      ("V Ext Prob %",  dict(min=0,   max=100,  default=15,  key="_v_ext",       kind="int")),
      ("Symmetry",      dict(options=_SYM_PRESET_OPTIONS, default=0, key="_sym_preset", kind="choice")),
  ]),
  ("Pattern Cascade",         "pattern_cascade",      [
      ("Count",         dict(min=1,   max=2000, default=15,  key="num",        kind="int")),
      ("Width Min",     dict(min=1,   max=3000, default=5,   key="pw_min",     kind="int")),
      ("Width Max",     dict(min=1,   max=3000, default=30,  key="pw_max",     kind="int")),
      ("Blend Min %",   dict(min=0,   max=100,  default=30,  key="_bmin",      kind="int")),
      ("Blend Max %",   dict(min=0,   max=100,  default=70,  key="_bmax",      kind="int")),
      ("H Ext Prob %",  dict(min=0,   max=100,  default=10,  key="_h_ext2",    kind="int")),
      ("Symmetry",      dict(options=_SYM_PRESET_OPTIONS,  default=0, key="_sym_preset", kind="choice")),
  ]),
  ("Diamond / Geo Patterns",  "diamond_patterns",     [
      ("Count",         dict(min=1,   max=10000, default=20,  key="num",              kind="int")),
      ("Size Min",      dict(min=1,   max=3000,  default=20,  key="size_min",         kind="int")),
      ("Size Max",      dict(min=1,   max=5000,  default=100, key="size_max",         kind="int")),
      ("Nest Layers",   dict(min=1,   max=8,     default=3,   key="nest_layers",      kind="int")),
      ("Noise %",       dict(min=0,   max=100,   default=30,  key="_noise_pct",       kind="int")),
      ("Organic Blend %",dict(min=0,  max=100,   default=40,  key="_organic_pct",     kind="int")),
      ("Symmetry",      dict(options=_SYM_PRESET_OPTIONS,  default=0, key="_sym_preset", kind="choice")),
  ]),
  ("Pixel Sorting",           "pixel_sorting",        [
      ("Regions",       dict(min=1,   max=5000, default=10,  key="num",   kind="int")),
      ("W Min",         dict(min=1,   max=3000, default=50,  key="w_min", kind="int")),
      ("W Max",         dict(min=1,   max=3000, default=200, key="w_max", kind="int")),
      ("H Min",         dict(min=1,   max=3000, default=50,  key="h_min", kind="int")),
      ("H Max",         dict(min=1,   max=3000, default=200, key="h_max", kind="int")),
      ("Sort By",       dict(options=["brightness","red","green","hue"],
                             default=0, key="sort_by", kind="choice")),
  ]),
  ("Geometric Grid",          "geometric_grid",       [
      ("Grid Size",     dict(min=2,   max=2000, default=40,  key="grid_size",      kind="int")),
      ("Cell Prob %",   dict(min=0,   max=100,  default=70,  key="_cell_prob_pct", kind="int")),
      ("Sub Layers",    dict(min=1,   max=6,    default=2,   key="sub_layers",     kind="int")),
      ("Symmetry",      dict(options=_SYM_PRESET_OPTIONS,  default=0, key="_sym_preset", kind="choice")),
  ]),
  ("Recursive Glitch",        "recursive_glitch",     [
      ("Iterations",    dict(min=1,   max=50,   default=3,   key="iterations",     kind="int")),
  ]),
  ("Vertical Cascade",        "vertical_cascade",     [
      ("Intensity %",   dict(min=0,   max=100,  default=80,  key="_intensity",     kind="int")),
      ("Bleed %",       dict(min=0,   max=100,  default=40,  key="_bleed_pct",     kind="int")),
      ("Wave Amp %",    dict(min=0,   max=100,  default=30,  key="_wave_amp_pct",  kind="int")),
      ("Plasma %",      dict(min=0,   max=100,  default=40,  key="_plasma_pct",    kind="int")),
      ("Branch Prob %", dict(min=0,   max=100,  default=20,  key="_branch_pct",    kind="int")),
  ]),
  ("Fractal Draw",            "fractal_draw",         [
      ("Max Depth",     dict(min=1,   max=12,   default=5,   key="max_depth",      kind="int")),
      ("Blend %",       dict(min=0,   max=100,  default=40,  key="_blend_pct",     kind="int")),
      ("Contour Levels",dict(min=0,   max=32,   default=8,   key="contour_levels", kind="int")),
      ("IFS Points",    dict(min=0,   max=100000,default=15000,key="ifs_points",   kind="int")),
  ]),
  ("Grid Transform",          "grid_transform",       [
      ("Cell Width",    dict(min=2,   max=3000, default=100, key="_cell_w",        kind="int")),
      ("Cell Height",   dict(min=2,   max=3000, default=100, key="_cell_h",        kind="int")),
      ("Cell Prob %",   dict(min=0,   max=100,  default=100, key="_cell_prob_pct", kind="int")),
      ("Blend %",       dict(min=0,   max=100,  default=100, key="_blend_pct",     kind="int")),
      ("Operation",     dict(options=["symmetry_only","fractal_mirror",
                                      "pixel_sort_h","pixel_sort_v",
                                      "rotate_contents","mirror_h","mirror_v",
                                      "kaleidoscope","invert","shuffle_rows",
                                      "shuffle_cols","reverse_rows","reverse_cols",
                                      "channel_shift","diagonal_mirror","random"],
                             default=0, key="_gt_op",        kind="choice")),
      ("Symmetry",      dict(options=SYMMETRY_NAMES, default=3, key="_gt_sym",     kind="choice")),
  ]),
  ("Mirror Sections",         "mirror_sections",      [
      ("Section H",     dict(min=1,   max=3000, default=0,   key="_sec_h",         kind="int")),
  ]),
  ("Blend With Original",     "blend_original",       [
      ("Ratio %",       dict(min=0,   max=100,  default=20,  key="_ratio_pct",     kind="int")),
  ]),
  ("Final Effects",           "final_effects",        [
      ("Contrast ×10",  dict(min=1,   max=50,   default=13,  key="_contrast_x10",  kind="int")),
      ("Sharpen",       dict(options=["Yes","No"], default=0, key="_sharpen_idx",  kind="choice")),
      ("Vignette",      dict(options=["No","Yes"], default=0, key="_vignette_idx", kind="choice")),
  ]),
]

OP_NAMES   = [o[0] for o in OPS]
OP_BY_NAME = {o[0]: o for o in OPS}

def call_op(proc, op_name, raw_kw, pq):
    """Resolve params and dispatch to Processor method."""
    _, method_name, _ = OP_BY_NAME[op_name]
    kw = resolve_params(op_name, raw_kw)
    # Special key remapping
    if '_h_ext' in kw:    kw['h_ext_prob'] = kw.pop('_h_ext')/100
    if '_v_ext'  in kw:    kw['v_ext_prob'] = kw.pop('_v_ext')/100
    if '_h_ext2' in kw:   kw['h_ext_prob'] = kw.pop('_h_ext2')/100
    if '_intensity' in kw: kw['intensity']  = kw.pop('_intensity')/100
    if '_blend_pct' in kw: kw['blend']      = kw.pop('_blend_pct')/100
    if '_cell_w'  in kw:  kw['cell_w']     = kw.pop('_cell_w')
    if '_cell_h'  in kw:  kw['cell_h']     = kw.pop('_cell_h')
    if '_gt_op'   in kw:
        _gt_ops = ["symmetry_only","fractal_mirror","pixel_sort_h","pixel_sort_v",
                   "rotate_contents","mirror_h","mirror_v","kaleidoscope","invert",
                   "shuffle_rows","shuffle_cols","reverse_rows","reverse_cols",
                   "channel_shift","diagonal_mirror","random"]
        _idx = kw.pop('_gt_op')
        kw['operation'] = _gt_ops[_idx] if 0 <= _idx < len(_gt_ops) else 'symmetry_only'
    if '_gt_sym'  in kw:
        _idx = kw.pop('_gt_sym')
        kw['sym_type'] = SYMMETRY_NAMES[_idx] if 0 <= _idx < len(SYMMETRY_NAMES) else 'quad'
    if '_ratio_pct' in kw: kw['ratio']      = kw.pop('_ratio_pct')/100
    if '_sec_h' in kw:
        v = kw.pop('_sec_h')
        if v>0: kw['section_height']=v
    if '_sharpen_idx' in kw:  kw['sharpen']  = kw.pop('_sharpen_idx')==0
    if '_vignette_idx' in kw: kw['vignette'] = kw.pop('_vignette_idx')==1
    kw['pq'] = pq
    getattr(proc, method_name)(**kw)


# ═════════════════════════════════════════════════════════════════════════════
#  PARAM WIDGETS
# ═════════════════════════════════════════════════════════════════════════════

class IntRow(tk.Frame):
    def __init__(self, parent, label, cfg):
        super().__init__(parent, bg=PANEL)
        self._cfg = cfg
        self._val = cfg["default"]          # authoritative value (no clamping to slider range)
        self._slider_var = tk.IntVar(value=cfg["default"])
        self._entry_var  = tk.StringVar(value=str(cfg["default"]))
        self._updating = False

        tk.Label(self,text=label,font=FT,fg=FG,bg=PANEL,width=14,anchor="w").pack(side="left")

        sc = tk.Scale(self,from_=cfg["min"],to=cfg["max"],orient="horizontal",
                      variable=self._slider_var,
                      bg=PANEL,fg=DIM,troughcolor=CARD,highlightthickness=0,
                      showvalue=False,bd=0,command=self._slider_moved)
        sc.pack(side="left",fill="x",expand=True)

        vc = self.register(lambda v: v=="" or v=="-" or v.lstrip("-").isdigit())
        self._ent = tk.Entry(self,textvariable=self._entry_var,width=6,font=FT,fg=FG,bg=CARD,
                             insertbackground=FG,relief="flat",highlightbackground=BORDER,
                             highlightthickness=1,validate="key",validatecommand=(vc,"%P"))
        self._ent.pack(side="left",padx=(4,0))
        self._ent.bind("<Return>",self._entry_commit)
        self._ent.bind("<FocusOut>",self._entry_commit)

    def _slider_moved(self, val):
        if self._updating: return
        self._updating = True
        self._val = int(float(val))
        self._entry_var.set(str(self._val))
        self._updating = False

    def _entry_commit(self, _=None):
        if self._updating: return
        try:
            v = int(self._entry_var.get())
        except ValueError:
            v = self._cfg["default"]
        self._val = max(self._cfg["min"], v)   # only enforce minimum, not maximum
        self._entry_var.set(str(self._val))
        # move slider to nearest valid position without triggering _slider_moved
        self._updating = True
        self._slider_var.set(min(self._val, self._cfg["max"]))
        self._updating = False

    @property
    def value(self):
        # Prefer whatever is currently typed, fall back to committed value
        try:
            v = int(self._entry_var.get())
            return max(self._cfg["min"], v)
        except ValueError:
            return self._val

class ChoiceRow(tk.Frame):
    def __init__(self, parent, label, cfg):
        super().__init__(parent, bg=PANEL)
        tk.Label(self,text=label,font=FT,fg=FG,bg=PANEL,width=14,anchor="w").pack(side="left")
        self._combo=ttk.Combobox(self,values=cfg["options"],state="readonly",font=FT,width=16)
        self._combo.current(cfg["default"]); self._combo.pack(side="left",padx=(4,0))
    @property
    def value(self): return self._combo.current()


# ═════════════════════════════════════════════════════════════════════════════
#  PIPELINE ITEM  (a row in the pipeline list)
# ═════════════════════════════════════════════════════════════════════════════

class PipelineItem(tk.Frame):
    def __init__(self, parent, op_name, on_remove, on_move_up, on_move_down):
        super().__init__(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        self._op = op_name
        self._rows = {}
        hdr = tk.Frame(self, bg=CARD); hdr.pack(fill="x", padx=6, pady=(4,0))
        tk.Label(hdr, text=op_name, font=FM, fg=ACC2, bg=CARD).pack(side="left")
        tk.Button(hdr, text="✕", font=FS, fg=DIM, bg=CARD, activeforeground=ACC,
                  activebackground=CARD, relief="flat", bd=0, cursor="hand2",
                  command=on_remove).pack(side="right", padx=2)
        tk.Button(hdr, text="↓", font=FS, fg=DIM, bg=CARD, activeforeground=ACC,
                  activebackground=CARD, relief="flat", bd=0, cursor="hand2",
                  command=on_move_down).pack(side="right", padx=1)
        tk.Button(hdr, text="↑", font=FS, fg=DIM, bg=CARD, activeforeground=ACC,
                  activebackground=CARD, relief="flat", bd=0, cursor="hand2",
                  command=on_move_up).pack(side="right", padx=1)
        _, _, params = OP_BY_NAME[op_name]
        pf = tk.Frame(self, bg=CARD); pf.pack(fill="x", padx=6, pady=(2,6))
        for lbl, cfg in params:
            row = IntRow(pf,lbl,cfg) if cfg["kind"]=="int" else ChoiceRow(pf,lbl,cfg)
            row.pack(fill="x", pady=1)
            self._rows[cfg["key"]] = row

    @property
    def op_name(self): return self._op

    def get_params(self):
        return {k: r.value for k,r in self._rows.items()}


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RecPatGen — Recursive Pattern Generator")
        self.configure(bg=BG); self.minsize(1150,700)

        self._proc    = None
        self._src_path= None
        self._res_img = None
        self._busy    = False
        self._pq      = queue.Queue()
        self._t0      = 0.
        self._items   = []   # list of PipelineItem

        self._max_dim_var   = tk.IntVar(value=1500)
        self._op_var        = tk.StringVar(value=OP_NAMES[0])
        self._batch_count   = tk.IntVar(value=10)
        self._batch_folder  = tk.StringVar(value="")
        self._batch_prefix  = tk.StringVar(value="output")
        self._batch_running = False
        self._batch_stop    = False

        self._setup_styles()
        self._build()

    def _setup_styles(self):
        s=ttk.Style(); s.theme_use("clam")
        s.configure("TCombobox",fieldbackground=CARD,background=CARD,foreground=FG,
                    selectbackground=ACC,selectforeground=BG,bordercolor=BORDER,arrowcolor=ACC)
        s.map("TCombobox",fieldbackground=[("readonly",CARD)],foreground=[("readonly",FG)])
        s.configure("bar.Horizontal.TProgressbar",troughcolor=CARD,background=ACC,bordercolor=PANEL)
        s.configure("TScrollbar",background=CARD,troughcolor=PANEL,bordercolor=PANEL,arrowcolor=DIM)

    def _build(self):
        h=tk.Frame(self,bg=BG); h.pack(fill="x",padx=24,pady=(16,0))
        tk.Label(h,text="RECURSIVE PATTERN GENERATOR",font=FT2,fg=ACC,bg=BG).pack(side="left")
        tk.Frame(self,bg=BORDER,height=1).pack(fill="x",padx=24,pady=8)
        body=tk.Frame(self,bg=BG); body.pack(fill="both",expand=True,padx=24,pady=(0,16))

        # Left sidebar — scrollable container
        sb_outer=tk.Frame(body,bg=PANEL,width=340,
                          highlightbackground=BORDER,highlightthickness=1)
        sb_outer.pack(side="left",fill="y",padx=(0,14)); sb_outer.pack_propagate(False)
        sb_canvas=tk.Canvas(sb_outer,bg=PANEL,bd=0,highlightthickness=0,width=320)
        sb_vsb=ttk.Scrollbar(sb_outer,orient="vertical",command=sb_canvas.yview)
        sb_canvas.configure(yscrollcommand=sb_vsb.set)
        sb_vsb.pack(side="right",fill="y")
        sb_canvas.pack(side="left",fill="both",expand=True)
        self._sb=tk.Frame(sb_canvas,bg=PANEL)
        _sb_win=sb_canvas.create_window((0,0),window=self._sb,anchor="nw")
        def _sb_configure(e): sb_canvas.configure(scrollregion=sb_canvas.bbox("all"))
        def _sb_width(e): sb_canvas.itemconfig(_sb_win,width=e.width)
        self._sb.bind("<Configure>",_sb_configure)
        sb_canvas.bind("<Configure>",_sb_width)
        # Mouse wheel scrolling
        def _sb_scroll(e):
            delta = -1*(e.delta//120) if e.delta else (-1 if e.num==4 else 1)
            sb_canvas.yview_scroll(delta,"units")
        for seq in ("<MouseWheel>","<Button-4>","<Button-5>"):
            sb_canvas.bind(seq,_sb_scroll)
            self._sb.bind(seq,_sb_scroll)

        # Right — preview
        pv=tk.Frame(body,bg=BG); pv.pack(side="left",fill="both",expand=True)
        self._build_preview(pv)
        self._build_sidebar()

    def _build_preview(self, pv):
        hr=tk.Frame(pv,bg=BG); hr.pack(fill="x",pady=(0,8))
        tk.Label(hr,text="ORIGINAL",font=FH,fg=DIM,bg=BG).pack(side="left")
        tk.Label(hr,text="  ·  ",fg=BORDER,bg=BG,font=FH).pack(side="left")
        tk.Label(hr,text="RESULT",font=FH,fg=ACC,bg=BG).pack(side="left")
        cf=tk.Frame(pv,bg=BG); cf.pack(fill="both",expand=True)
        cf.columnconfigure(0,weight=1); cf.columnconfigure(1,weight=1); cf.rowconfigure(0,weight=1)
        self._oc=tk.Canvas(cf,bg=CARD,bd=0,highlightbackground=BORDER,highlightthickness=1)
        self._rc=tk.Canvas(cf,bg=CARD,bd=0,highlightbackground=BORDER,highlightthickness=1)
        self._oc.grid(row=0,column=0,sticky="nsew",padx=(0,6))
        self._rc.grid(row=0,column=1,sticky="nsew",padx=(6,0))
        self._ph(self._oc,"Open an image to begin")
        self._ph(self._rc,"Result appears here")

    def _build_sidebar(self):
        sb=self._sb
        # 01 source
        self._lbl(sb,"01 — SOURCE IMAGE")
        rf=tk.Frame(sb,bg=PANEL); rf.pack(fill="x",padx=16,pady=(2,2))
        tk.Label(rf,text="Max dim:",font=FT,fg=DIM,bg=PANEL).pack(side="left")
        vc=self.register(lambda p: p.isdigit() or p=="")
        tk.Entry(rf,textvariable=self._max_dim_var,width=6,font=FT,fg=FG,bg=CARD,
                 insertbackground=FG,relief="flat",highlightbackground=BORDER,
                 highlightthickness=1,validate="key",validatecommand=(vc,"%P")
                 ).pack(side="left",padx=(4,0))
        tk.Label(rf,text="px",font=FT,fg=DIM,bg=PANEL).pack(side="left",padx=(3,0))
        self._btn(sb,"[ OPEN IMAGE ]",self._open_image,ACC).pack(fill="x",padx=16,pady=(4,2))
        self._file_lbl=tk.Label(sb,text="no file loaded",font=FM,fg=DIM,bg=PANEL,
                                wraplength=300,justify="left")
        self._file_lbl.pack(anchor="w",padx=16,pady=(2,4))
        self._div(sb)

        # 02 pipeline builder
        self._lbl(sb,"02 — PIPELINE")
        af=tk.Frame(sb,bg=PANEL); af.pack(fill="x",padx=16,pady=(0,4))
        cb=ttk.Combobox(af,textvariable=self._op_var,values=OP_NAMES,
                        state="readonly",font=FT)
        cb.pack(side="left",fill="x",expand=True,padx=(0,6))
        self._btn(af,"+ ADD",self._add_op,ACC).pack(side="left")

        # Scrollable pipeline list
        pl_wrap=tk.Frame(sb,bg=PANEL,height=260)
        pl_wrap.pack(fill="x",padx=8,pady=(0,4)); pl_wrap.pack_propagate(False)
        self._pl_canvas=tk.Canvas(pl_wrap,bg=PANEL,bd=0,highlightthickness=0)
        pl_vsb=ttk.Scrollbar(pl_wrap,orient="vertical",command=self._pl_canvas.yview)
        self._pl_canvas.configure(yscrollcommand=pl_vsb.set)
        pl_vsb.pack(side="right",fill="y"); self._pl_canvas.pack(side="left",fill="both",expand=True)
        self._pl_inner=tk.Frame(self._pl_canvas,bg=PANEL)
        self._pl_win=self._pl_canvas.create_window((0,0),window=self._pl_inner,anchor="nw")
        self._pl_inner.bind("<Configure>",lambda e: self._pl_canvas.configure(
            scrollregion=self._pl_canvas.bbox("all")))
        self._pl_canvas.bind("<Configure>",lambda e:
            self._pl_canvas.itemconfig(self._pl_win,width=e.width))
        for seq in ("<MouseWheel>","<Button-4>","<Button-5>"):
            self._pl_canvas.bind(seq,lambda e,c=self._pl_canvas:
                c.yview_scroll(int(-e.delta/120) if e.num==0 else (-1 if e.num==4 else 1),"units"))

        self._div(sb)

        # 03 actions
        self._lbl(sb,"03 — ACTIONS")
        bf=tk.Frame(sb,bg=PANEL); bf.pack(fill="x",padx=16,pady=(2,4))
        self._run_btn=self._btn(bf,"[ RUN PIPELINE ]",self._run,GREEN)
        self._run_btn.pack(fill="x",pady=(0,5))
        self._btn(bf,"[ SAVE RESULT ]",self._save,YELLOW).pack(fill="x",pady=(0,5))
        self._btn(bf,"[ RESET IMAGE ]",self._reset,DIM).pack(fill="x")
        self._div(sb)

        # 04 batch
        self._lbl(sb,"04 — BATCH GENERATE")
        bb=tk.Frame(sb,bg=PANEL); bb.pack(fill="x",padx=16,pady=(2,4))

        # count row
        cr=tk.Frame(bb,bg=PANEL); cr.pack(fill="x",pady=(0,3))
        tk.Label(cr,text="Count:",font=FT,fg=FG,bg=PANEL,width=8,anchor="w").pack(side="left")
        vc=self.register(lambda v: v.isdigit() or v=="")
        tk.Entry(cr,textvariable=self._batch_count,width=6,font=FT,fg=FG,bg=CARD,
                 insertbackground=FG,relief="flat",highlightbackground=BORDER,
                 highlightthickness=1,validate="key",validatecommand=(vc,"%P")
                 ).pack(side="left",padx=(4,0))

        # prefix row
        pr=tk.Frame(bb,bg=PANEL); pr.pack(fill="x",pady=(0,3))
        tk.Label(pr,text="Prefix:",font=FT,fg=FG,bg=PANEL,width=8,anchor="w").pack(side="left")
        tk.Entry(pr,textvariable=self._batch_prefix,width=14,font=FT,fg=FG,bg=CARD,
                 insertbackground=FG,relief="flat",highlightbackground=BORDER,
                 highlightthickness=1).pack(side="left",padx=(4,0))

        # folder row
        fr=tk.Frame(bb,bg=PANEL); fr.pack(fill="x",pady=(0,3))
        tk.Label(fr,text="Folder:",font=FT,fg=FG,bg=PANEL,width=8,anchor="w").pack(side="left")
        self._folder_lbl=tk.Label(fr,textvariable=self._batch_folder,font=FM,fg=DIM,
                                   bg=PANEL,anchor="w",wraplength=180)
        self._folder_lbl.pack(side="left",padx=(4,0))
        self._btn(fr,"…",self._pick_folder,ACC2).pack(side="right")

        # batch run / stop buttons
        self._batch_btn=self._btn(bb,"[ RUN BATCH ]",self._run_batch,ACC2)
        self._batch_btn.pack(fill="x",pady=(2,0))
        self._stop_btn=self._btn(bb,"[ STOP ]",self._stop_batch,DIM)
        self._stop_btn.pack(fill="x",pady=(3,0))
        self._stop_btn.config(state="disabled")

        # batch progress label
        self._batch_lbl=tk.Label(bb,text="",font=FM,fg=ACC2,bg=PANEL)
        self._batch_lbl.pack(fill="x",pady=(3,0))
        self._div(sb)

        # progress
        pf=tk.Frame(sb,bg=PANEL); pf.pack(fill="x",padx=16,pady=(4,2))
        tr=tk.Frame(pf,bg=PANEL); tr.pack(fill="x")
        self._eta=tk.Label(tr,text="",font=FM,fg=DIM,bg=PANEL,anchor="w"); self._eta.pack(side="left")
        self._pct=tk.Label(tr,text="",font=FM,fg=ACC,bg=PANEL,anchor="e"); self._pct.pack(side="right")
        self._bar=ttk.Progressbar(pf,mode="determinate",maximum=100,style="bar.Horizontal.TProgressbar")
        self._bar.pack(fill="x",pady=(3,2))
        self._op_lbl=tk.Label(pf,text="",font=FS,fg=DIM,bg=PANEL,anchor="w")
        self._op_lbl.pack(fill="x")
        self._stv=tk.StringVar(value="Ready.")
        tk.Label(sb,textvariable=self._stv,font=FM,fg=DIM,bg=PANEL,
                 wraplength=310,justify="left").pack(anchor="w",padx=16,pady=(4,12))

    # ── pipeline management ───────────────────────────────────────────────────

    def _add_op(self):
        name=self._op_var.get()
        idx=len(self._items)
        item=PipelineItem(self._pl_inner, name,
                          on_remove=lambda i=idx: self._remove_op(i),
                          on_move_up=lambda i=idx: self._move_op(i,-1),
                          on_move_down=lambda i=idx: self._move_op(i,1))
        item.pack(fill="x",pady=(0,4))
        self._items.append(item)
        self._refresh_pl_callbacks()

    def _remove_op(self, idx):
        if 0<=idx<len(self._items):
            self._items[idx].destroy()
            self._items.pop(idx)
            self._refresh_pl_callbacks()

    def _move_op(self, idx, direction):
        ni=idx+direction
        if 0<=ni<len(self._items):
            self._items[idx],self._items[ni]=self._items[ni],self._items[idx]
            for item in self._items: item.pack_forget()
            for item in self._items: item.pack(fill="x",pady=(0,4))
            self._refresh_pl_callbacks()

    def _refresh_pl_callbacks(self):
        for i,item in enumerate(self._items):
            item._on_remove    = lambda i=i: self._remove_op(i)
            item._on_move_up   = lambda i=i: self._move_op(i,-1)
            item._on_move_down = lambda i=i: self._move_op(i,1)

    # ── image I/O ─────────────────────────────────────────────────────────────

    def _open_image(self):
        p=filedialog.askopenfilename(
            filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),("All","*.*")])
        if not p: return
        try:
            self._proc=Processor(p, max_dim=max(100,self._max_dim_var.get()))
            self._src_path=p; self._res_img=None
            self._file_lbl.config(text=os.path.basename(p),fg=GREEN)
            self._show(self._oc, self._proc.result_image(),"_otk")
            self._ph(self._rc,"Run pipeline →")
            self._st(f"Loaded {self._proc.W}×{self._proc.H}")
        except Exception as e:
            messagebox.showerror("Load Error",str(e))

    def _save(self):
        if not self._res_img: messagebox.showwarning("No result","Run the pipeline first."); return
        p=filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("All","*.*")])
        if p: self._proc.arr=np.array(self._res_img); self._proc.save(p)
        self._st(f"Saved → {os.path.basename(p)}")

    def _reset(self):
        if not self._proc: return
        self._proc.arr=self._proc.orig.copy()
        self._res_img=None
        self._show(self._oc, self._proc.result_image(),"_otk")
        self._ph(self._rc,"Run pipeline →")
        self._clrbar(); self._st("Reset.")

    # ── run pipeline ──────────────────────────────────────────────────────────

    def _run(self):
        if not self._proc: messagebox.showwarning("No image","Load an image first."); return
        if not self._items: messagebox.showwarning("Empty pipeline","Add at least one operation."); return
        if self._busy: return
        pipeline=[(item.op_name, item.get_params()) for item in self._items]
        self._busy=True; self._t0=time.monotonic()
        while not self._pq.empty():
            try: self._pq.get_nowait()
            except: pass
        self._clrbar(); self._run_btn.config(state="disabled"); self._st("Running…")
        self._proc.arr=self._proc.orig.copy()

        def worker():
            try:
                for op_name,raw in pipeline:
                    self.after(0,lambda n=op_name: self._op_lbl.config(text=f"→ {n}"))
                    call_op(self._proc, op_name, raw, self._pq)
                result=self._proc.result_image()
                self._res_img=result
                self.after(0,lambda: self._show(self._rc,result,"_rtk"))
                self.after(0,lambda: self._st(f"Done in {time.monotonic()-self._t0:.1f}s"))
            except Exception as e:
                self.after(0,lambda: messagebox.showerror("Error",str(e)))
                log.exception("Pipeline error")
            finally:
                self.after(0,lambda: setattr(self,"_busy",False))
                self.after(0,lambda: self._run_btn.config(state="normal"))
                self.after(0,lambda: self._op_lbl.config(text=""))

        threading.Thread(target=worker,daemon=True).start()
        self._poll()

    def _pick_folder(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self._batch_folder.set(folder)

    def _run_batch(self):
        if not self._proc:
            messagebox.showwarning("No image","Load an image first."); return
        if not self._items:
            messagebox.showwarning("Empty pipeline","Add at least one operation."); return
        if self._busy or self._batch_running: return
        folder = self._batch_folder.get()
        if not folder:
            messagebox.showwarning("No folder","Choose an output folder first."); return
        try:
            count = max(1, int(self._batch_count.get()))
        except ValueError:
            count = 10
        prefix = self._batch_prefix.get() or "output"
        pipeline = [(item.op_name, item.get_params()) for item in self._items]

        self._batch_running = True
        self._batch_stop    = False
        self._batch_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._run_btn.config(state="disabled")
        self._batch_lbl.config(text=f"0 / {count}")

        import os as _os

        def worker():
            for idx in range(1, count+1):
                if self._batch_stop: break
                self.after(0, lambda i=idx, c=count:
                    self._batch_lbl.config(text=f"{i} / {c}  generating…"))
                try:
                    self._proc.arr = self._proc.orig.copy()
                    for op_name, raw in pipeline:
                        if self._batch_stop: break
                        call_op(self._proc, op_name, raw, self._pq)
                    if not self._batch_stop:
                        fname = _os.path.join(folder, f"{prefix}_{idx:04d}.png")
                        self._proc.save(fname)
                        self.after(0, lambda i=idx, c=count, f=fname:
                            self._batch_lbl.config(
                                text=f"{i} / {c}  saved {_os.path.basename(f)}"))
                except Exception as e:
                    log.exception(f"Batch error on image {idx}")
                    self.after(0, lambda e=e:
                        messagebox.showerror("Batch error", str(e)))
                    break

            stopped = self._batch_stop
            def finish():
                self._batch_running = False
                self._batch_stop    = False
                self._batch_btn.config(state="normal")
                self._stop_btn.config(state="disabled")
                self._run_btn.config(state="normal")
                msg = f"Stopped." if stopped else f"Done — {count} images saved."
                self._batch_lbl.config(text=msg)
                self._st(msg)
            self.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def _stop_batch(self):
        self._batch_stop = True
        self._stop_btn.config(state="disabled")
        self._batch_lbl.config(text="Stopping…")

    def _poll(self):
        try:
            while True:
                d,t=self._pq.get_nowait(); pct=int(d/max(t,1)*100)
                self._bar["value"]=pct; self._pct.config(text=f"{pct}%")
                el=time.monotonic()-self._t0
                if d>0 and el>0.2:
                    self._eta.config(text=f"ETA {el/d*(t-d):.1f}s")
        except queue.Empty: pass
        if self._busy: self.after(40,self._poll)

    def _clrbar(self):
        self._bar["value"]=0; self._pct.config(text=""); self._eta.config(text="")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _bind_scroll(self, widget, canvas):
        """Recursively bind mousewheel to widget and all its children."""
        def _scroll(e, c=canvas):
            delta = -1*(e.delta//120) if e.delta else (-1 if e.num==4 else 1)
            c.yview_scroll(delta,"units")
        for seq in ("<MouseWheel>","<Button-4>","<Button-5>"):
            widget.bind(seq, _scroll)
        for child in widget.winfo_children():
            self._bind_scroll(child, canvas)

    def _lbl(self,p,t):
        tk.Label(p,text=t,font=FM,fg=ACC2,bg=PANEL).pack(anchor="w",padx=16,pady=(10,2))
    def _div(self,p):
        tk.Frame(p,bg=BORDER,height=1).pack(fill="x",padx=16,pady=4)
    def _btn(self,p,t,cmd,col=FG):
        return tk.Button(p,text=t,font=FB,fg=col,bg=CARD,
            activeforeground=BG,activebackground=col,
            relief="flat",bd=0,cursor="hand2",command=cmd,padx=8,pady=6)
    def _st(self,msg): self._stv.set(msg)
    def _ph(self,c,t):
        c.update_idletasks(); w=c.winfo_width() or 400; h=c.winfo_height() or 400
        c.delete("all"); c.create_rectangle(0,0,w,h,fill=CARD,outline="")
        c.create_text(w//2,h//2,text=t,fill=DIM,font=FT,justify="center")
    def _show(self,c,img,attr):
        c.update_idletasks()
        cw=max(c.winfo_width() or 460,10); ch=max(c.winfo_height() or 460,10)
        t=img.copy(); t.thumbnail((cw,ch),Image.LANCZOS)
        tk_img=ImageTk.PhotoImage(t); setattr(self,attr,tk_img)
        c.delete("all")
        c.create_image((cw-t.width)//2,(ch-t.height)//2,anchor="nw",image=tk_img)
        c.create_text(cw-6,ch-6,text=f"{img.width}×{img.height}",
                      anchor="se",font=FM,fill=DIM)


if __name__ == "__main__":
    app=App()
    W,H=1200,740
    sw,sh=app.winfo_screenwidth(),app.winfo_screenheight()
    app.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")
    app.mainloop()
