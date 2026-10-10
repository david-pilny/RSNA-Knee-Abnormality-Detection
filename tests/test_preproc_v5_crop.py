"""Checks the v5 knee crop in knee_preproc on synthetic phantoms (no real data).

Usage (from the project root): .venv/Scripts/python tests/test_preproc_v5_crop.py
"""
import re
import sys
import types
from pathlib import Path

import numpy as np

src = Path(__file__).resolve().parents[1].joinpath("notebooks", "rsna_knee_preprocess_cache_v5.py").read_text(encoding="utf-8")
module_src = re.search(r"PREPROC_SRC = r'''(.*?)'''", src, re.S).group(1).replace("__CROP_MM__", "140.0")
kp = types.ModuleType("knee_preproc")
exec(module_src, kp.__dict__)


def phantom(H, W, ps, center_mm, axes_mm, D=20):
    """A bright ellipse ('knee') on black, plus noise. center_mm is relative to the image centre (dy, dx)."""
    yy, xx = np.mgrid[:H, :W]
    y_mm, x_mm = (yy - H / 2) * ps[0], (xx - W / 2) * ps[1]
    inside = ((y_mm - center_mm[0]) / axes_mm[0]) ** 2 + ((x_mm - center_mm[1]) / axes_mm[1]) ** 2 <= 1
    rng = np.random.default_rng(0)
    vol = np.where(inside, 0.6, 0.0)[None] + rng.normal(0, 0.03, (D, H, W))
    return np.clip(vol, 0, 1).astype(np.float32)


def bright_centre(u8):
    m = u8.mean(0) > 60
    ys, xs = np.nonzero(m)
    return ys.mean(), xs.mean()


ok = True
def check(name, cond, detail):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + " | " + detail)

# 1) large field of view (205 mm), knee off-centre by +30 mm in x: the crop must re-centre it
vol = phantom(512, 512, (0.4, 0.4), (0, 30), (50, 45))
out, info = kp.resample(vol, [0.4, 0.4], 24, 256)
cy, cx = bright_centre(out)
check("off-centre knee is centred", abs(cy - 128) < 4 and abs(cx - 128) < 4, f"centre {cy:.1f},{cx:.1f} | {info}")
check("nothing of the knee cut", info["tissue_outside"] < 0.01, f"tissue_outside {info['tissue_outside']}")
check("shape and dtype", out.shape == (24, 256, 256) and out.dtype == np.uint8, f"{out.shape} {out.dtype}")
# scale: ellipse 90 mm wide → 90 / (140 / 256) ≈ 165 px
width = (out.mean(0) > 60).any(0).sum()
check("fixed mm per pixel", abs(width - 90 / (140 / 256)) < 6, f"knee width {width} px, expected ≈ {90 / (140 / 256):.0f}")

# 2) knee at the image edge: window is moved back inside the image (no padding needed)
vol = phantom(400, 400, (0.5, 0.5), (0, 60), (40, 35))
out, info = kp.resample(vol, [0.5, 0.5], 24, 256)
check("window stays inside a large image", out[:, :, -1].mean() > 0 or info["tissue_outside"] < 0.05, f"{info}")

# 3) small field of view (100 mm): padded, knee whole, same mm per pixel
vol = phantom(256, 256, (100 / 256, 100 / 256), (0, 0), (40, 40))
out, info = kp.resample(vol, [100 / 256, 100 / 256], 24, 256)
width = (out.mean(0) > 60).any(0).sum()
check("small FOV padded, not stretched", abs(width - 80 / (140 / 256)) < 6 and out[:, :, :10].max() == 0,
      f"knee width {width} px | {info}")

# 4) non-square pixels: window is square in mm
vol = phantom(300, 600, (0.6, 0.3), (0, 0), (40, 40))
out, info = kp.resample(vol, [0.6, 0.3], 24, 256)
m = out.mean(0) > 60
h, w = m.any(1).sum(), m.any(0).sum()
check("non-square pixels give a round knee", abs(h - w) < 6, f"h {h} w {w}")

# 5) CROP_MM = 0 reproduces the first cache's layout (whole field of view padded to a square)
vol = phantom(200, 400, (0.5, 0.5), (0, 0), (40, 40))
out, info = kp.resample(vol, [0.5, 0.5], 24, 256, crop_mm=0)
check("crop off = whole FOV", out[:, :60].max() == 0 and info["tissue_outside"] == 0, f"{info}")

print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
