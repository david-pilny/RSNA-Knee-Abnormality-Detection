# %% [markdown]
# # RSNA Knee: image preprocessing cache v5 (cropped to the knee, 256 px)
#
# The report labeller produces the labels (**y**). This notebook produces the images (**X**): for every training study, a few
# small, uniform 3D arrays that a model can load in milliseconds.
#
# **What changed against the first cache (224 px, whole field of view):** the menisci did not improve when the labels did,
# so the images are the limit. Each series is now cut to a fixed **140 × 140 mm window centred on the knee** before
# resizing, and stored at **256 px**: 0.55 mm per pixel for every study, instead of 0.7–0.9 mm with the knee drifting in a
# padded frame. Fixed millimetres also make the knee the same size in every hospital's images. Downsizing uses an
# anti-aliased filter (less shimmer on thin structures like menisci and ligaments).
#
# Output dataset name: **`knee-mri-cache-v5`** (a new dataset: the 224 px models still need the old one).
#
# **Second setting (6 Oct): 320 px, 130 mm window, in two parts.** 320 px would exceed Kaggle's 20 GB output limit in one
# run, so the studies are split: run this notebook twice, with `PART = "1/2"` and `PART = "2/2"` (two copies of the
# notebook can run at the same time; CPU only). Save the outputs as two **new** datasets, `knee-mri-cache-320a` and
# `knee-mri-cache-320b`. Training attaches both and joins them. (Result: +0.004 validation AUC, not pursued.)
#
# **Third setting (10 Oct, the current default): the joint region, 90 mm window at 256 px → 0.35 mm per pixel.**
# A second model that sees only the joint line at close to native resolution: menisci, both tibiofemoral compartments,
# the cruciate ligaments in the notch, the collaterals. It deliberately loses the patella, the shafts and the posterior
# soft tissue; those findings (PF OA, Baker's, effusion, synovitis, contusion, fracture) stay with the 140 mm model, and
# the submission blends the two per finding. The window is centred on the tissue box like before, which puts the joint
# line near the centre because MRI technologists centre the field of view on it; section 5's pictures are the check:
# **on every sagittal and coronal sample both menisci and both tibiofemoral compartments must be fully inside.**
# If the joint sits too low or high in several samples, raise `CROP_MM` to 100 and rerun. One part fits in one dataset:
# save the output as **`knee-mri-cache-joint`**.
#
# ```
# DICOM folders (≈100k files, mixed sizes, orientations, compression)
#        │  pick the right series per study  →  load + sort slices  →  canonical orientation
#        │  →  normalize intensities  →  resample to a fixed shape  →  uint8
#        ▼
# cache shards (≈45 files)  +  index.csv  +  knee_preproc.py (the exact code, reused at submission time)
# ```
#
# **Settings:** Accelerator **None** (CPU only, so it costs no GPU hours and can run next to the labeller), Internet **on**
# (only to pip-install the DICOM decoders). Expected runtime: roughly 0.5–2 hours; the notebook measures it on a sample first.

# %% [markdown]
# ## 1. Settings
#
# | Setting | Meaning |
# |---|---|
# | `ROLES` | which series to keep per study (next section explains) |
# | `DEPTH`, `IMG` | output shape per series: `DEPTH` slices of `IMG × IMG` pixels |
# | `SHARD_SIZE` | studies per output file. Kaggle keeps at most **500 files** and **20 GB** of notebook output, so studies are packed into shards |
# | `LIMIT` | process only the first N studies (for a quick test); `None` = all |

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os, sys, io, time, json, glob, subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROLES = {                       # role name: (plane, fluid-sensitive wanted?)
    "sag_fs": ("Sagittal", 1),  # ACL, menisci, effusion, contusion, Baker's cyst
    "cor_fs": ("Coronal", 1),   # MCL, meniscal bodies, medial/lateral OA
    "ax_fs":  ("Axial", 1),     # patellofemoral OA, synovitis, Baker's cyst, effusion
}
# settings per cache: whole knee 256 px / 140 mm / "1/1" (knee-mri-cache, latest) | 320 px / 130 mm / "1/2" + "2/2"
# (knee-mri-cache-320a/b) | joint region 256 px / 90 mm / "1/1" (knee-mri-cache-joint, the current default)
DEPTH, IMG = 24, int(os.environ.get("IMG", 256))
CROP_MM = float(os.environ.get("CROP_MM", 90))    # in-plane window around the knee, millimetres (0 = whole field of view)
PART = os.environ.get("PART", "1/1")   # "k/n": this notebook makes part k of n (every n-th study); n = 2 keeps each part
                                      # under Kaggle's 20 GB output limit at 320 px. "1/1" = everything in one run
SHARD_SIZE = 100
LIMIT = None
SIZE_LIMIT_GB = 18              # stop before exceeding Kaggle's 20 GB output limit
N_JOBS = os.cpu_count() or 2

SLUG = "rsna-knee-abnormality-detection"
ROOT = next(Path(p) for p in [os.environ.get("KNEE_ROOT", ""), f"/kaggle/input/competitions/{SLUG}", f"/kaggle/input/{SLUG}"]
            if p and (Path(p) / "train.csv").exists())
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
print("data:", ROOT, "| output:", OUT_DIR, "| CPUs:", N_JOBS)

# %% [markdown]
# ## 2. DICOM decoders
#
# Some sites store compressed DICOMs (JPEG 2000, JPEG Lossless) that pydicom cannot decode alone. This notebook has internet,
# so a normal `pip install` works here. (The submission notebook will need the offline wheels trick instead.)

# %% [code] {"jupyter":{"outputs_hidden":false}}
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"],
                   capture_output=True, text=True)
print("pip:", "ok" if r.returncode == 0 else r.stderr[-400:])
import pydicom
print("pydicom", pydicom.__version__)

# %% [markdown]
# ## 3. The preprocessing code, as a module
#
# All image processing lives in **one file, `knee_preproc.py`**, written to the output folder. The training notebook and the
# submission notebook will import this exact file from the dataset. That guarantees the hidden test images are prepared
# **identically** to the training images; any difference (another resize method, another orientation rule) silently costs score.
#
# What the module does, per study:
#
# **a) Pick one series per role.** From `train_series.csv` (or `test_series.csv`): the right plane, preferably fluid-sensitive
# and fat-suppressed. If a study has no fluid-sensitive series in that plane, any series in that plane is used and the role is
# marked `fallback`. Several candidates → prefer fat suppression, then a normal slice count (12–80).
#
# **b) Load and sort** all slices by their physical position (`ImagePositionPatient`), never by filename.
#
# **c) Canonical orientation.** Sites store the same plane in different ways (rotated, mirrored). Every series is flipped or
# transposed so its axes point the same way in patient space:
#
# | Plane | image columns → | image rows ↓ | slices go |
# |---|---|---|---|
# | Sagittal | posterior (A left, P right) | inferior (S top) | right → left |
# | Coronal | patient's left (radiological view) | inferior | anterior → posterior |
# | Axial | patient's left | posterior (A top) | inferior → superior |
#
# **d) Left vs right knee.** For a right knee the medial side is towards the patient's left; for a left knee it is the other
# way. The cache keeps **patient** orientation and stores the `Laterality` tag (if present) plus the knee's x-position in the
# index, so training can mirror left knees later. We don't guess inside the cache.
#
# **e) Intensity:** per-series percentile scaling (0.5 %–99.5 %) to 0–255. MRI has no absolute units.
#
# **f) Crop:** the knee is found on the average of all slices (tissue = brighter than 30 % of the 99th percentile; the
# tissue box is where at least 5 % of a row or column is tissue). A `CROP_MM × CROP_MM` window is centred on that box and
# moved back inside the image if needed. A field of view smaller than the window is padded with black. The share of tissue
# left outside the window is stored as `tissue_outside` so we can check that no knee is cut.
#
# **g) Shape:** the window is resampled to `DEPTH × IMG × IMG` (linear between slices, anti-aliased in-plane) and stored as
# `uint8` (1 byte per voxel).

# %% [code] {"jupyter":{"outputs_hidden":false}}
PREPROC_SRC = r'''
"""knee_preproc.py: shared preprocessing for the RSNA knee competition (train cache AND submission)."""
from pathlib import Path
from collections import Counter
import time
import numpy as np
import pydicom
import torch
import torch.nn.functional as F

# canonical axes in patient space (LPS: +x = patient left, +y = posterior, +z = superior)
GEOMETRY = {
    "Sagittal": dict(slice_axis=(1, 0, 0), col_axis=(0, 1, 0), row_axis=(0, 0, -1)),
    "Coronal":  dict(slice_axis=(0, 1, 0), col_axis=(1, 0, 0), row_axis=(0, 0, -1)),
    "Axial":    dict(slice_axis=(0, 0, 1), col_axis=(1, 0, 0), row_axis=(0, 1, 0)),
}

def _t(ds, name, default=None):
    v = getattr(ds, name, default)
    return default if v is None or v == "" else v

def select_series(study_rows, plane, fluid, series_root):
    """Pick the best series for one role. study_rows: rows of *_series.csv for one study."""
    cand = study_rows[study_rows.Anatomical_Plane == plane].copy()
    if cand.empty:
        return None, None
    cand["n_files"] = [len(list((series_root / s).glob("*.dcm"))) for s in cand.SeriesInstanceUID]
    cand = cand[cand.n_files > 0]
    if cand.empty:
        return None, None
    cand["score"] = ((cand.Fluid_Sensitive == fluid) * 4
                     + ((cand.Fat_Suppression == 1) & (fluid == 1)) * 2
                     + cand.n_files.between(12, 80) * 1)
    best = cand.sort_values(["score", "n_files"], ascending=False).iloc[0]
    return best, bool(best.Fluid_Sensitive != fluid)

def load_series(series_dir, slice_axis):
    """All slices of one series as float32 (D, H, W), sorted along slice_axis, plus the header of the first slice."""
    sl = [pydicom.dcmread(p) for p in Path(series_dir).glob("*.dcm")]
    main = Counter((s.Rows, s.Columns) for s in sl).most_common(1)[0][0]
    sl = [s for s in sl if (s.Rows, s.Columns) == main]
    ax = np.asarray(slice_axis, float)
    if all(_t(s, "ImagePositionPatient") is not None for s in sl):
        sl.sort(key=lambda s: float(np.dot(ax, np.asarray(s.ImagePositionPatient, float))))
        pos = np.array([np.dot(ax, np.asarray(s.ImagePositionPatient, float)) for s in sl])
    else:
        sl.sort(key=lambda s: int(_t(s, "InstanceNumber", 0)))
        pos = None
    vol = []
    for s in sl:
        a = s.pixel_array.astype(np.float32)
        a = a * float(_t(s, "RescaleSlope", 1)) + float(_t(s, "RescaleIntercept", 0))
        if _t(s, "PhotometricInterpretation") == "MONOCHROME1":
            a = a.max() - a
        vol.append(a)
    return np.stack(vol), sl[0], pos

def canonicalize(vol, hdr, plane):
    """Flip/transpose so image columns and rows follow GEOMETRY[plane]. Returns vol and (row_mm, col_mm) spacing."""
    g = GEOMETRY[plane]
    iop = np.asarray(_t(hdr, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), float)
    r, c = iop[:3], iop[3:]                       # r: direction of increasing column index, c: of increasing row index
    ps = [float(x) for x in _t(hdr, "PixelSpacing", [1.0, 1.0])]   # [between rows, between columns]
    u, v = np.asarray(g["col_axis"], float), np.asarray(g["row_axis"], float)
    if abs(np.dot(r, u)) < abs(np.dot(c, u)):     # image is rotated 90 degrees: swap its axes
        vol = vol.transpose(0, 2, 1)
        r, c = c, r
        ps = ps[::-1]
    if np.dot(r, u) < 0:
        vol = vol[:, :, ::-1]
    if np.dot(c, v) < 0:
        vol = vol[:, ::-1, :]
    return np.ascontiguousarray(vol), ps

def normalize_u8(vol, lo_pct=0.5, hi_pct=99.5):
    sample = vol[:, ::4, ::4]
    lo, hi = np.percentile(sample, [lo_pct, hi_pct])
    return np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1)

CROP_MM = __CROP_MM__                             # in-plane window around the knee (mm); 0 = whole field of view

def crop_knee(vol01, ps, crop_mm):
    """Cut a crop_mm x crop_mm window centred on the knee (zero-padded where the field of view is smaller).
    Returns the window (D, h, w) in source pixels, and info about it."""
    D, H, W = vol01.shape
    proj = vol01.mean(0)
    tissue = proj > 0.3 * np.percentile(proj, 99)
    rows, cols = np.flatnonzero(tissue.mean(1) > 0.05), np.flatnonzero(tissue.mean(0) > 0.05)
    cy = (rows[0] + rows[-1] + 1) / 2 if len(rows) else H / 2
    cx = (cols[0] + cols[-1] + 1) / 2 if len(cols) else W / 2
    hh, ww = crop_mm / ps[0], crop_mm / ps[1]     # window size in source pixels
    def start(c, win, n):
        return (n - win) / 2 if win >= n else min(max(c - win / 2, 0), n - win)
    y0, x0 = int(round(start(cy, hh, H))), int(round(start(cx, ww, W)))
    h, w = max(1, int(round(hh))), max(1, int(round(ww)))
    out = np.zeros((D, h, w), np.float32)
    sy0, sx0, sy1, sx1 = max(y0, 0), max(x0, 0), min(y0 + h, H), min(x0 + w, W)
    out[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = vol01[:, sy0:sy1, sx0:sx1]
    inside = tissue[sy0:sy1, sx0:sx1].sum()
    info = dict(fov_mm=round(max(H * ps[0], W * ps[1]), 1),
                tissue_outside=round(float(1 - inside / max(1, tissue.sum())), 4))
    return out, info

def resample(vol01, ps, depth, img, crop_mm=None):
    """Crop to the knee (or pad the whole field of view to a square), resize to (depth, img, img), return uint8 and info."""
    crop_mm = CROP_MM if crop_mm is None else crop_mm
    D, H, W = vol01.shape
    if crop_mm and crop_mm > 0:
        win, info = crop_knee(vol01, ps, crop_mm)
    else:                                         # whole field of view, padded to a square (the first cache's layout)
        side = max(H * ps[0], W * ps[1])
        win, info = crop_knee(vol01, ps, side)
        info["tissue_outside"] = 0.0
    t = torch.from_numpy(win)[None, None]
    t = F.interpolate(t, size=(depth, win.shape[1], win.shape[2]), mode="trilinear", align_corners=False)[0, 0]
    t = F.interpolate(t[:, None], size=(img, img), mode="bilinear", antialias=True, align_corners=False)[:, 0]
    return (t.clamp(0, 1) * 255).round().to(torch.uint8).numpy(), info

def process_study(study_uid, study_rows, series_root, roles, depth, img):
    """Returns ({role: uint8 array or None}, [meta dict per role])."""
    torch.set_num_threads(1)
    arrays, metas = {}, []
    for role, (plane, fluid) in roles.items():
        m = {"StudyInstanceUID": study_uid, "role": role, "ok": False, "error": None}
        t0 = time.time()
        try:
            best, fallback = select_series(study_rows, plane, fluid, Path(series_root) / study_uid)
            if best is None:
                raise LookupError(f"no {plane} series")
            m.update(SeriesInstanceUID=best.SeriesInstanceUID, fallback=fallback, n_files=int(best.n_files),
                     fluid=int(best.Fluid_Sensitive), fatsat=int(best.Fat_Suppression))
            vol, hdr, pos = load_series(Path(series_root) / study_uid / best.SeriesInstanceUID,
                                        GEOMETRY[plane]["slice_axis"])
            m.update(n_slices=vol.shape[0], rows=vol.shape[1], cols=vol.shape[2],
                     slice_gap_mm=float(np.median(np.diff(pos))) if pos is not None and len(pos) > 1 else None,
                     transfer_syntax=hdr.file_meta.TransferSyntaxUID.name,
                     manufacturer=_t(hdr, "Manufacturer"), field_T=_t(hdr, "MagneticFieldStrength"),
                     laterality=_t(hdr, "Laterality") or _t(hdr, "ImageLaterality"),
                     x_center=float(np.asarray(_t(hdr, "ImagePositionPatient", [np.nan] * 3), float)[0]))
            vol, ps = canonicalize(vol, hdr, plane)
            m.update(pixel_mm=round(ps[0], 4))
            arrays[role], info = resample(normalize_u8(vol), ps, depth, img)
            m.update(info)
            m["ok"] = True
        except Exception as e:
            arrays[role] = None
            m["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        m["seconds"] = round(time.time() - t0, 2)
        metas.append(m)
    return arrays, metas

class CacheReader:
    """Read the cache: CacheReader(dir).load(study_uid) -> {role: uint8 (DEPTH, IMG, IMG) or None}."""
    def __init__(self, cache_dir):
        import pandas as pd
        self.dir = Path(cache_dir)
        self.index = pd.read_csv(self.dir / "index.csv")
        self.shard_of = dict(zip(self.index.StudyInstanceUID, self.index.shard))
        self.roles = sorted(self.index.role.unique())
        self._open = {}
    def load(self, study_uid):
        shard = self.shard_of[study_uid]
        if shard not in self._open:
            self._open[shard] = np.load(self.dir / shard)
        z = self._open[shard]
        return {r: (z[f"{study_uid}__{r}"] if f"{study_uid}__{r}" in z.files else None) for r in self.roles}
'''
PREPROC_SRC = PREPROC_SRC.replace("__CROP_MM__", repr(CROP_MM))
(OUT_DIR / "knee_preproc.py").write_text(PREPROC_SRC)
sys.dont_write_bytecode = True                 # no __pycache__ folder in the output
sys.path.insert(0, str(OUT_DIR))
os.environ["PYTHONPATH"] = str(OUT_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")   # parallel workers must find the module too
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import importlib, knee_preproc as kp
importlib.reload(kp)
print("wrote and imported", OUT_DIR / "knee_preproc.py")

# %% [markdown]
# ## 4. Which series are available? (CSV only, fast)
#
# Before touching any image: how many studies have a series of each role, and how often we must fall back to a
# non-fluid-sensitive series. A role that is missing for many studies means the model must handle missing inputs
# (the cache simply stores nothing for that role, and training uses a mask).

# %% [code] {"jupyter":{"outputs_hidden":false}}
train = pd.read_csv(ROOT / "train.csv", usecols=["StudyInstanceUID"])
series = pd.read_csv(ROOT / "train_series.csv")
studies = train.StudyInstanceUID.tolist()[:LIMIT] if LIMIT else train.StudyInstanceUID.tolist()
PART_K, PART_N = (int(x) for x in PART.split("/"))
studies = studies[PART_K - 1::PART_N]
print(f"part {PART_K} of {PART_N}: {len(studies)} studies")
print(f"studies to process: {len(studies)} | series in CSV: {len(series)}")

avail = []
for sid, grp in series.groupby("StudyInstanceUID"):
    row = {"study": sid}
    for role, (plane, fluid) in ROLES.items():
        p = grp[grp.Anatomical_Plane == plane]
        row[role] = "exact" if (p.Fluid_Sensitive == fluid).any() else ("fallback" if len(p) else "missing")
    avail.append(row)
avail = pd.DataFrame(avail)
summary = pd.DataFrame({r: avail[r].value_counts(normalize=True) for r in ROLES}).T.reindex(columns=["exact", "fallback", "missing"]).fillna(0)
ax = summary.plot.barh(stacked=True, color=["#1f5fa8", "#e59866", "#c0392b"], figsize=(10, 2.8))
ax.set_xlabel("share of studies"); ax.set_title("Series available per role"); ax.legend(loc="center left", bbox_to_anchor=(1, .5))
plt.tight_layout(); plt.show()
print(summary.round(3).to_string())

# %% [markdown]
# ## 5. Try it on a sample and look at the result
#
# Before the full run: process a few studies, **look at them**, and measure speed and compressed size.
# Check in the picture that every sagittal looks like a sagittal with the front of the knee on the left and the top up,
# that coronals and axials are consistently oriented, and that nothing is black or inverted.

# %% [code] {"jupyter":{"outputs_hidden":false}}
from joblib import Parallel, delayed

SERIES_ROOT = ROOT / "train_series"
rows_of = {sid: g for sid, g in series.groupby("StudyInstanceUID")}

def run(sid):
    return kp.process_study(sid, rows_of.get(sid, series.iloc[:0]), SERIES_ROOT, ROLES, DEPTH, IMG)

rng = np.random.default_rng(0)
has_files = [s for s in studies if (SERIES_ROOT / s).exists()]
sample_ids = list(rng.choice(has_files, size=min(24, len(has_files)), replace=False))
t0 = time.time()
sample = Parallel(n_jobs=N_JOBS)(delayed(run)(s) for s in sample_ids)
wall = time.time() - t0
print(f"{len(sample_ids)} studies in {wall:.1f} s wall → {wall / len(sample_ids) * N_JOBS:.2f} s per study per core")

buf = io.BytesIO()
np.savez_compressed(buf, **{f"{s}__{r}": a for s, (arrs, _) in zip(sample_ids, sample) for r, a in arrs.items() if a is not None})
per_study_mb = buf.getbuffer().nbytes / 1e6 / len(sample_ids)
est_gb = per_study_mb * len(studies) / 1e3
est_min = wall / len(sample_ids) * len(studies) / 60
print(f"compressed size: {per_study_mb:.2f} MB per study → estimated total {est_gb:.1f} GB (limit {SIZE_LIMIT_GB} GB)")
print(f"estimated runtime for {len(studies)} studies: {est_min:.0f} min")

sample_meta = pd.DataFrame([m for _, metas in sample for m in metas])
print("\nsample: roles ok", f"{sample_meta.ok.mean():.0%}", "| fallback", f"{sample_meta.fallback.fillna(False).mean():.0%}")
if (~sample_meta.ok).any():
    print(sample_meta.loc[~sample_meta.ok, ["StudyInstanceUID", "role", "error"]].head(10).to_string())
print(sample_meta.groupby("role")[["fov_mm", "pixel_mm", "tissue_outside"]].describe().round(3).T.to_string())

# %% [code] {"jupyter":{"outputs_hidden":false}}
SHOW = min(6, len(sample_ids))
roles = list(ROLES)
fig, axes = plt.subplots(SHOW, len(roles) * 3, figsize=(2.1 * len(roles) * 3, 2.2 * SHOW))
axes = np.atleast_2d(axes)
for i in range(SHOW):
    arrs = sample[i][0]
    for j, role in enumerate(roles):
        for k, frac in enumerate([0.3, 0.5, 0.7]):
            ax = axes[i, j * 3 + k]; ax.axis("off")
            a = arrs.get(role)
            if a is not None:
                ax.imshow(a[int(frac * (DEPTH - 1))], cmap="gray", vmin=0, vmax=255)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", color="red", transform=ax.transAxes)
            if i == 0:
                ax.set_title(f"{role} {int(frac * 100)}%", fontsize=9)
plt.suptitle("Sample studies after preprocessing (rows = studies; 3 depths per role)", y=1.0)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. Full run, written in shards
#
# Studies are processed in parallel on all CPUs and written `SHARD_SIZE` at a time into compressed `.npz` files
# (`shard_000.npz`, `shard_001.npz`, ...). Inside a shard, each array is stored under the key `<StudyInstanceUID>__<role>`.
#
# The run stops early if the size estimate exceeds `SIZE_LIMIT_GB`. In that case lower `IMG` (e.g. 192) or `DEPTH` and rerun.
# Shards that already exist are skipped, so an interrupted interactive run can continue.

# %% [code] {"jupyter":{"outputs_hidden":false}}
assert est_gb < SIZE_LIMIT_GB, f"estimated {est_gb:.1f} GB > {SIZE_LIMIT_GB} GB: split into more parts (PART = \"1/3\", \"2/3\", \"3/3\") and rerun"

all_meta, t0 = [], time.time()
shards = [studies[i:i + SHARD_SIZE] for i in range(0, len(studies), SHARD_SIZE)]
for n, chunk in enumerate(shards):
    name = f"shard_p{PART_K}_{n:03d}.npz"
    meta_path = CACHE_DIR / f"meta_p{PART_K}_{n:03d}.csv"
    if (CACHE_DIR / name).exists() and meta_path.exists():
        all_meta.append(pd.read_csv(meta_path)); continue
    res = Parallel(n_jobs=N_JOBS)(delayed(run)(s) for s in chunk)
    np.savez_compressed(CACHE_DIR / name,
                        **{f"{s}__{r}": a for s, (arrs, _) in zip(chunk, res) for r, a in arrs.items() if a is not None})
    meta = pd.DataFrame([m for _, metas in res for m in metas]).assign(shard=name)
    meta.to_csv(meta_path, index=False)
    all_meta.append(meta)
    done = (n + 1) * SHARD_SIZE; el = time.time() - t0
    print(f"{name}: {min(done, len(studies))}/{len(studies)} studies | {el / 60:.1f} min | "
          f"ETA {el / min(done, len(studies)) * (len(studies) - min(done, len(studies))) / 60:.0f} min", flush=True)

index = pd.concat(all_meta, ignore_index=True)
index.to_csv(CACHE_DIR / "index.csv", index=False)
for f in CACHE_DIR.glob("meta_*.csv"):
    f.unlink()                                   # merged into index.csv; keeps the output file count low
size_gb = sum(f.stat().st_size for f in CACHE_DIR.glob("*.npz")) / 1e9
print(f"\ndone in {(time.time() - t0) / 60:.1f} min | {len(list(CACHE_DIR.glob('*.npz')))} shards | {size_gb:.2f} GB")

# %% [markdown]
# ## 7. Quality report
#
# - **ok rate per role**: share of studies where the role was produced. Missing = no series of that plane, or decoding failed.
# - **errors by type**: if you see decoding errors for one transfer syntax, the decoders are the problem, not the data.
# - **metadata**: scanner vendor, field strength and laterality are stored in `index.csv` for later
#   (site-grouped cross-validation, mirroring left knees).

# %% [code] {"jupyter":{"outputs_hidden":false}}
q = index.groupby("role").agg(ok=("ok", "mean"), fallback=("fallback", lambda s: s.fillna(False).astype(bool).mean()),
                              median_slices=("n_slices", "median"), median_sec=("seconds", "median"),
                              median_fov_mm=("fov_mm", "median"), fov_over_crop=("fov_mm", lambda s: (s > CROP_MM).mean()),
                              tissue_out_median=("tissue_outside", "median"), tissue_out_p95=("tissue_outside", lambda s: s.quantile(0.95)),
                              tissue_out_over_10pct=("tissue_outside", lambda s: (s > 0.10).mean()))
print(q.round(3).to_string())          # commit logs do not show display() tables
print("\n'tissue_out' = share of the tissue outside the crop window. Sagittal and coronal windows cut femur and tibia shafts")
print("on purpose, so 10-30 % there is normal; what matters is that the joint itself is inside (see the pictures).")
errs = index[~index.ok.astype(bool)]
if len(errs):
    print("errors by type:")
    print(errs.error.str.split(":").str[0].value_counts().head(10).to_string())
    print("\nerrors by transfer syntax (of series that were found):")
    print(errs.transfer_syntax.fillna("—").value_counts().head(10).to_string())

fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
index.drop_duplicates("StudyInstanceUID").manufacturer.fillna("?").value_counts().head(8).plot.bar(ax=axes[0], color="#4a7ab5")
axes[0].set_title("scanner vendor (per study)"); axes[0].tick_params(axis="x", rotation=30)
index.drop_duplicates("StudyInstanceUID").laterality.fillna("not stored").value_counts().plot.bar(ax=axes[1], color="#4a7ab5")
axes[1].set_title("Laterality tag"); axes[1].tick_params(axis="x", rotation=0)
index.n_slices.clip(upper=100).plot.hist(bins=50, ax=axes[2], color="#4a7ab5")
axes[2].set_title("original slices per selected series (clipped at 100)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 8. Read it back
#
# The final check uses `CacheReader` from `knee_preproc.py`, exactly as the training notebook will: pick random studies,
# load them from the shards, and show the middle slice of each role.

# %% [code] {"jupyter":{"outputs_hidden":false}}
reader = kp.CacheReader(CACHE_DIR)
ok_ids = reader.index.loc[reader.index.ok.astype(bool), "StudyInstanceUID"].unique()
check = list(np.random.default_rng(1).choice(ok_ids, size=min(4, len(ok_ids)), replace=False))
fig, axes = plt.subplots(len(check), len(reader.roles), figsize=(3 * len(reader.roles), 3 * len(check)))
axes = np.atleast_2d(axes)
for i, sid in enumerate(check):
    t0 = time.time(); arrs = reader.load(sid); ms = (time.time() - t0) * 1000
    for j, role in enumerate(reader.roles):
        ax = axes[i, j]; ax.axis("off")
        if arrs[role] is not None:
            ax.imshow(arrs[role][DEPTH // 2], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{role}" + (f"  ({ms:.0f} ms)" if j == 0 else ""), fontsize=9)
plt.suptitle("Loaded back from the cache (middle slice)", y=1.0); plt.tight_layout(); plt.show()
print("output files:", len(list(OUT_DIR.rglob("*"))), "(Kaggle limit: 500)")

# %% [markdown]
# ## 9. Turn the output into a dataset
#
# 1. Run this notebook with **Save Version → Save & Run All** (CPU is enough).
# 2. Open the finished version → **Output** → **New Dataset**, named after the setting (section 1): **`knee-mri-cache-joint`**
#    for the 90 mm joint crop. Never add it as a version of an existing cache: a notebook can attach only one version.
# 3. The dataset contains:
#    - `cache/shard_*.npz`: the images,
#    - `cache/index.csv`: one row per study and role (which series, fallback, errors, scanner, laterality, ...),
#    - `knee_preproc.py`: the code. The training notebook uses `CacheReader`; the submission notebook calls
#      `process_study(...)` on the hidden test series with the **same** `ROLES`, `DEPTH` and `IMG`.
#
# Settings used for this cache (write them down; training and submission must match):

# %% [code] {"jupyter":{"outputs_hidden":false}}
config = {"ROLES": ROLES, "DEPTH": DEPTH, "IMG": IMG, "CROP_MM": CROP_MM, "PART": PART, "SHARD_SIZE": SHARD_SIZE, "studies": len(studies)}
(CACHE_DIR / "config.json").write_text(json.dumps(config, indent=1))
print(json.dumps(config, indent=1))
