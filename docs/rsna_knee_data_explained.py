# %% [markdown]
# # RSNA Knee: understanding the data, one study at a time
#
# This notebook takes **one training study** and **one of its series** and explains, step by step, what is inside:
#
# 1. **The folder hierarchy**: why there are two levels of folders before the `.dcm` files
# 2. **One DICOM file**: header (tags) and pixel data
# 3. **One slice**: the pixels and their physical size
# 4. **One series**: stacking slices into a 3D volume, and why sort order matters
# 5. **Geometry**: where the slices sit in the patient's body, in millimetres
# 6. **Intensities**: why MRI needs per-series normalization
# 7. **The whole study**: all sequences side by side, and how they intersect in 3D
# 8. **Labels and report** for this study
# 9. **Preparing model input**: resizing to a fixed shape
# 10. A short look at the whole dataset (CSV only, fast)
#
# Every section ends with a short **Takeaway**.

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os, textwrap
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom

pd.set_option("display.max_colwidth", 120)
plt.rcParams.update({"figure.dpi": 100, "axes.titlesize": 10})

# ---- the input ---------------------------------------------------------------
ROOT = Path(os.environ.get("KNEE_ROOT",
            "/kaggle/input/competitions/rsna-knee-abnormality-detection"))
STUDY_UID  = "1.2.826.0.1.3680043.8.498.10004873229099053869093324292195817260"
SERIES_UID = "1.2.826.0.1.3680043.8.498.12343110195036213483454091715412333772"

STUDY_DIR  = ROOT / "train_series" / STUDY_UID
SERIES_DIR = STUDY_DIR / SERIES_UID
print("root exists:  ", ROOT.exists())
print("study exists: ", STUDY_DIR.exists())
print("series exists:", SERIES_DIR.exists())
print("pydicom", pydicom.__version__)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Compressed series (JPEG 2000 / JPEG Lossless) need extra decoders.
# This series is uncompressed, so the notebook works without them, but check what is available:
for mod in ["pylibjpeg", "libjpeg", "openjpeg", "gdcm"]:
    try:
        __import__(mod); print(f"{mod:10s} available")
    except ImportError:
        print(f"{mod:10s} missing  (only needed for compressed series)")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Small helper: DICOM files in this competition keep only 86 allowlisted tags,
# so any tag may be missing. Always read tags defensively.
def tag(ds, name, default=None):
    v = getattr(ds, name, default)
    return default if v is None or v == "" else v

def short(uid, n=12):
    return "…" + str(uid)[-n:]

# %% [markdown]
# ## 1. The folder hierarchy
#
# DICOM organizes imaging as **Patient → Study → Series → Instance**:
#
# ```
# train_series/
# └── <StudyInstanceUID>/          one MRI exam (one visit to the scanner)   ← labels live here (train.csv)
#     ├── <SeriesInstanceUID>/     one acquisition, e.g. sagittal PD fat-sat ← sequence info (train_series.csv)
#     │   ├── <SOPInstanceUID>.dcm one 2D slice
#     │   ├── ...
#     ├── <SeriesInstanceUID>/
#     ...
# ```
#
# MRI stores **each slice as its own file**. You get a 3D volume by stacking all files of one series.
# Let's check the study folder:

# %% [code] {"jupyter":{"outputs_hidden":false}}
rows = []
for sdir in sorted(STUDY_DIR.iterdir()):
    files = list(sdir.glob("*.dcm"))
    rows.append({
        "SeriesInstanceUID": sdir.name,
        "n_slices": len(files),
        "total_MB": sum(f.stat().st_size for f in files) / 1e6,
        "avg_file_kB": np.mean([f.stat().st_size for f in files]) / 1e3,
    })
folder_df = pd.DataFrame(rows)
print(f"Study {short(STUDY_UID)} has {len(folder_df)} series, "
      f"{folder_df.n_slices.sum()} slice files, {folder_df.total_MB.sum():.1f} MB")
folder_df.assign(SeriesInstanceUID=folder_df.SeriesInstanceUID.map(short)).round(2)

# %% [markdown]
# The folders on their own tell us nothing about what each series *is*. That information is in **`train_series.csv`**,
# and the study-level labels and report are in **`train.csv`**. The UIDs are the join keys between CSVs and folders.

# %% [code] {"jupyter":{"outputs_hidden":false}}
train_df  = pd.read_csv(ROOT / "train.csv")
series_df = pd.read_csv(ROOT / "train_series.csv")

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
LABELS = [c for c in LABELS if c in train_df.columns]

study_series = series_df[series_df.StudyInstanceUID == STUDY_UID].copy()
study_series = study_series.merge(folder_df, on="SeriesInstanceUID", how="outer")
study_series["short_uid"] = study_series.SeriesInstanceUID.map(short)
study_series["selected"] = np.where(study_series.SeriesInstanceUID == SERIES_UID, "◀ this one", "")
study_series[["short_uid", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression",
              "n_slices", "avg_file_kB", "selected"]].round(1)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# A picture of the same hierarchy for this study
fig, ax = plt.subplots(figsize=(11, 0.9 + 0.7 * len(study_series)))
ax.axis("off")
ax.text(0.01, 0.5, f"Study\n{short(STUDY_UID)}", va="center", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", fc="#dbe8f5", ec="#4a7ab5"))
n = len(study_series)
for i, r in enumerate(study_series.itertuples()):
    y = 1 - (i + 0.5) / n
    ax.annotate("", xy=(0.30, y), xytext=(0.17, 0.5),
                arrowprops=dict(arrowstyle="->", color="#888"))
    fs = "fluid" if r.Fluid_Sensitive == 1 else "non-fluid"
    fat = "fat-sat" if r.Fat_Suppression == 1 else "no fat-sat"
    hl = r.SeriesInstanceUID == SERIES_UID
    ax.text(0.30, y, f"Series {r.short_uid}   {r.Anatomical_Plane}, {fs}, {fat}",
            va="center", fontsize=9, fontweight="bold" if hl else "normal",
            bbox=dict(boxstyle="round", fc="#fbe3c8" if hl else "#eeeeee", ec="#c08040" if hl else "#999"))
    ax.text(0.83, y, f"→ {int(r.n_slices)} × .dcm", va="center", fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("Study → Series → Slices (this study)", loc="left")
plt.show()

# %% [markdown]
# **Takeaway:** a study is one exam. It contains several series (different planes and contrasts), and each series
# is a stack of single-slice `.dcm` files. Labels belong to the **study**, sequence descriptors to the **series**,
# and pixels to the **slice**.

# %% [markdown]
# ## 2. Anatomy of one DICOM file
#
# A `.dcm` file has two parts:
#
# | Part | What it is |
# |---|---|
# | **Header** | a dictionary of *tags* (metadata): sizes, spacing, 3D position, orientation, compression... |
# | **Pixel data** | the image, usually 16-bit integers, possibly compressed |
#
# We start with the first file (by filename) of the selected series.

# %% [code] {"jupyter":{"outputs_hidden":false}}
series_files = sorted(SERIES_DIR.glob("*.dcm"))
f0 = series_files[0]
ds = pydicom.dcmread(f0)
print("file:", f0.name)
print("size on disk:", f0.stat().st_size, "bytes")
print("transfer syntax:", ds.file_meta.TransferSyntaxUID, "=", ds.file_meta.TransferSyntaxUID.name)
print("number of tags in the dataset:", len(ds))

# %% [code] {"jupyter":{"outputs_hidden":false}}
# All tags in a readable table (pixel data itself excluded)
tag_rows = []
for el in ds:
    if el.keyword == "PixelData":
        tag_rows.append((str(el.tag), el.keyword, el.VR, f"<{len(el.value):,} bytes of pixels>"))
        continue
    val = str(el.value)
    tag_rows.append((str(el.tag), el.keyword or el.name, el.VR, val[:80] + ("…" if len(val) > 80 else "")))
tags_df = pd.DataFrame(tag_rows, columns=["tag", "keyword", "VR", "value"])
tags_df

# %% [markdown]
# ### The tags that matter for modelling
#
# `(gggg,eeee)` is the numeric tag ID and the **VR** (value representation) is its type: `DS` decimal string, `US` unsigned short, `UI` UID...
# These are the ones you will actually use:

# %% [code] {"jupyter":{"outputs_hidden":false}}
important = {
    "Rows": "image height (px)",
    "Columns": "image width (px)",
    "PixelSpacing": "mm per pixel [row, col]",
    "SliceThickness": "thickness of one slice (mm)",
    "SpacingBetweenSlices": "centre-to-centre slice distance (mm)",
    "ImagePositionPatient": "3D position (mm) of the top-left pixel",
    "ImageOrientationPatient": "direction cosines of rows and columns",
    "InstanceNumber": "slice number assigned by the scanner",
    "BitsAllocated": "bits per pixel in storage",
    "BitsStored": "bits actually used",
    "PixelRepresentation": "0 = unsigned, 1 = signed",
    "PhotometricInterpretation": "MONOCHROME2 = normal, MONOCHROME1 = inverted",
    "RescaleSlope": "intensity = raw * slope + intercept",
    "RescaleIntercept": "",
    "Manufacturer": "scanner vendor (site proxy)",
    "MagneticFieldStrength": "1.5 T / 3 T (site proxy)",
    "Laterality": "left/right knee, if kept",
    "BodyPartExamined": "",
}
pd.DataFrame([(k, tag(ds, k, "— missing —"), v) for k, v in important.items()],
             columns=["keyword", "value", "meaning"])

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Where do the bytes go? Header vs pixels
px_bytes = len(ds.PixelData)
total = f0.stat().st_size
print(f"pixel data : {px_bytes:>9,} B  = {ds.Rows} × {ds.Columns} × {ds.BitsAllocated // 8} bytes"
      f"  → {'uncompressed' if px_bytes == ds.Rows * ds.Columns * ds.BitsAllocated // 8 else 'compressed'}")
print(f"header etc.: {total - px_bytes:>9,} B")
print(f"total      : {total:>9,} B")

fig, ax = plt.subplots(figsize=(8, 1.1))
ax.barh([0], [px_bytes], color="#4a7ab5", label="pixel data")
ax.barh([0], [total - px_bytes], left=[px_bytes], color="#e08a3c", label="header")
ax.set_yticks([]); ax.set_xlabel("bytes"); ax.legend(loc="lower right", fontsize=8)
ax.set_title("Almost the whole file is pixels", loc="left"); plt.show()

# %% [markdown]
# **Takeaway:** each file is one 2D image plus a header. The header matters as much as the pixels:
# it tells you where the slice is in space and how to interpret its values.

# %% [markdown]
# ## 3. The pixels of a single slice
#
# `ds.pixel_array` decodes the pixel data into a NumPy array.

# %% [code] {"jupyter":{"outputs_hidden":false}}
img = ds.pixel_array
print("shape:", img.shape, "| dtype:", img.dtype, "| min:", img.min(), "| max:", img.max(),
      "| mean:", round(float(img.mean()), 1))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("raw pixels, auto contrast (min→black, max→white)")
lo, hi = np.percentile(img, [1, 99.5])
axes[1].imshow(img, cmap="gray", vmin=lo, vmax=hi)
axes[1].set_title(f"windowed to 1st–99.5th percentile [{lo:.0f}, {hi:.0f}]")
axes[2].hist(img.ravel(), bins=200, color="#4a7ab5")
axes[2].axvline(lo, color="r", ls="--"); axes[2].axvline(hi, color="r", ls="--")
axes[2].set_yscale("log"); axes[2].set_title("intensity histogram (log y)"); axes[2].set_xlabel("raw value")
for a in axes[:2]: a.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# The histogram typically shows a huge spike near zero (**air/background**), with tissue spread over a wide range
# and a few very bright outliers. Auto-contrast gets dominated by those outliers, which is why percentile windowing
# (middle panel) looks better. We come back to this in section 5.

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Physical size: pixels are not "just pixels", every one covers a real area
ps = [float(x) for x in tag(ds, "PixelSpacing", [1, 1])]
h_mm, w_mm = img.shape[0] * ps[0], img.shape[1] * ps[1]
print(f"pixel spacing {ps[0]:.3f} × {ps[1]:.3f} mm → field of view {h_mm:.0f} × {w_mm:.0f} mm")

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, extent=[0, w_mm, h_mm, 0])
ax.set_xlabel("mm"); ax.set_ylabel("mm"); ax.set_title("same slice in millimetres")
plt.show()

# %% [markdown]
# ## 4. From slices to a 3D volume
#
# Stacking all files of the series gives a volume of shape `(depth, height, width)`.
# The key question is **in which order**. Candidates:
#
# - **filename**: the names are random UIDs, so the order is meaningless
# - **`InstanceNumber`**: usually right, but not guaranteed
# - **physical position**: project `ImagePositionPatient` onto the slice normal. This is always correct.

# %% [code] {"jupyter":{"outputs_hidden":false}}
slices = [pydicom.dcmread(p) for p in series_files]
print(len(slices), "slices loaded")

iop = np.array(tag(slices[0], "ImageOrientationPatient"), dtype=float)
row_dir, col_dir = iop[:3], iop[3:]
normal = np.cross(row_dir, col_dir)

def position(s):
    return float(np.dot(normal, np.array(s.ImagePositionPatient, dtype=float)))

order_df = pd.DataFrame({
    "file_idx": range(len(slices)),
    "filename": [short(p.stem, 8) for p in series_files],
    "InstanceNumber": [tag(s, "InstanceNumber") for s in slices],
    "position_mm": [position(s) for s in slices],
})
order_df.head(8)

# %% [code] {"jupyter":{"outputs_hidden":false}}
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(order_df.file_idx, order_df.position_mm, "o-", color="#c0504d")
axes[0].set_xlabel("index when sorted by filename"); axes[0].set_ylabel("position along normal (mm)")
axes[0].set_title("Filename order: positions jump around")

if order_df.InstanceNumber.notna().all():
    o = order_df.sort_values("InstanceNumber")
    axes[1].plot(o.InstanceNumber, o.position_mm, "o-", color="#4a7ab5")
    axes[1].set_xlabel("InstanceNumber")
    axes[1].set_title("InstanceNumber order vs. position")
else:
    axes[1].text(0.5, 0.5, "InstanceNumber missing", ha="center"); axes[1].axis("off")
axes[1].set_ylabel("position along normal (mm)")
plt.tight_layout(); plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false}}
def load_series(series_dir):
    # Load a series as a (D, H, W) float32 volume, sorted by physical position.
    sl = [pydicom.dcmread(p) for p in Path(series_dir).glob("*.dcm")]
    main_shape = Counter((s.Rows, s.Columns) for s in sl).most_common(1)[0][0]
    sl = [s for s in sl if (s.Rows, s.Columns) == main_shape]      # drop odd-sized slices (e.g. localizers)
    iop = np.array(sl[0].ImageOrientationPatient, dtype=float)
    nrm = np.cross(iop[:3], iop[3:])
    sl.sort(key=lambda s: np.dot(nrm, np.array(s.ImagePositionPatient, dtype=float)))
    vol = []
    for s in sl:
        a = s.pixel_array.astype(np.float32)
        a = a * float(tag(s, "RescaleSlope", 1)) + float(tag(s, "RescaleIntercept", 0))
        if tag(s, "PhotometricInterpretation") == "MONOCHROME1":
            a = a.max() - a
        vol.append(a)
    return np.stack(vol), sl

vol, sorted_slices = load_series(SERIES_DIR)
pos = np.array([np.dot(normal, np.array(s.ImagePositionPatient, dtype=float)) for s in sorted_slices])
gaps = np.diff(pos)
print("volume shape (D, H, W):", vol.shape)
print(f"slice gaps: min {gaps.min():.3f}, median {np.median(gaps):.3f}, max {gaps.max():.3f} mm")
print("SliceThickness:", tag(ds, "SliceThickness"), "| SpacingBetweenSlices:", tag(ds, "SpacingBetweenSlices"))

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Every slice of the series, in the correct physical order
lo, hi = np.percentile(vol, [1, 99.5])
n = len(vol); ncol = 6; nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.6 * nrow))
for i, ax in enumerate(axes.ravel()):
    ax.axis("off")
    if i < n:
        ax.imshow(vol[i], cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(f"#{i}  {pos[i]:.1f} mm", fontsize=8)
plt.suptitle(f"All {n} slices of series {short(SERIES_UID)} (sorted by position)", y=1.0)
plt.tight_layout(); plt.show()

# %% [markdown]
# **Takeaway:** always sort slices by position along the slice normal. The first and last slices usually only
# graze the edge of the knee, and the diagnostic content is in the middle of the stack.

# %% [markdown]
# ## 5. Geometry: where the slices are in the body
#
# DICOM uses the patient coordinate system **LPS**, in millimetres:
#
# - **x** → patient's **L**eft
# - **y** → patient's **P**osterior (back)
# - **z** → **S**uperior (towards the head)
#
# `ImageOrientationPatient` = 6 numbers: the 3D direction of an image row and of an image column.
# Their cross product is the **slice normal**, the direction in which the stack advances. From the normal you can read the plane:
#
# | normal mostly along | plane |
# |---|---|
# | x (left–right) | **Sagittal** |
# | y (front–back) | **Coronal** |
# | z (up–down) | **Axial** |

# %% [code] {"jupyter":{"outputs_hidden":false}}
def plane_from_iop(iop):
    n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
    return ["Sagittal", "Coronal", "Axial"][int(np.argmax(np.abs(n)))], n

p, n_vec = plane_from_iop(iop)
print("row direction   :", np.round(row_dir, 3))
print("column direction:", np.round(col_dir, 3))
print("slice normal    :", np.round(n_vec, 3))
print("plane from IOP  :", p)
print("plane from CSV  :", study_series.loc[study_series.SeriesInstanceUID == SERIES_UID, "Anatomical_Plane"].tolist())

# %% [markdown]
# ### Voxels are not cubes
#
# In-plane pixels are small (fraction of a mm), while the distance between slices is several mm.
# Cutting the volume *across* the slices shows how coarse the third dimension is:

# %% [code] {"jupyter":{"outputs_hidden":false}}
dz = float(np.median(np.abs(gaps)))
ps = [float(x) for x in tag(sorted_slices[0], "PixelSpacing", [1, 1])]
print(f"voxel size: {dz:.2f} (between slices) × {ps[0]:.2f} × {ps[1]:.2f} mm → anisotropy {dz / ps[0]:.1f}×")

mid = len(vol) // 2
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), gridspec_kw={"width_ratios": [1, 1, 1]})
axes[0].imshow(vol[mid], cmap="gray", vmin=lo, vmax=hi, extent=[0, vol.shape[2]*ps[1], vol.shape[1]*ps[0], 0])
axes[0].axhline(vol.shape[1]//2 * ps[0], color="y", lw=1); axes[0].axvline(vol.shape[2]//2 * ps[1], color="c", lw=1)
axes[0].set_title(f"acquired slice #{mid} (native plane)")
axes[1].imshow(vol[:, vol.shape[1]//2, :], cmap="gray", vmin=lo, vmax=hi, aspect="auto",
               extent=[0, vol.shape[2]*ps[1], len(vol)*dz, 0])
axes[1].set_title("reformat along yellow line (true mm scale)"); axes[1].set_ylabel("mm across slices")
axes[2].imshow(vol[:, :, vol.shape[2]//2].T, cmap="gray", vmin=lo, vmax=hi, aspect="auto",
               extent=[0, len(vol)*dz, vol.shape[1]*ps[0], 0])
axes[2].set_title("reformat along cyan line (true mm scale)"); axes[2].set_ylabel("mm")
for a in axes: a.set_xlabel("mm")
plt.tight_layout(); plt.show()

# %% [markdown]
# **Takeaway:** the reformats are blocky because there are only a few slices several mm apart.
# That is why most solutions treat knee MRI as **2.5D** (a 2D CNN on each slice, then aggregation over slices)
# rather than as a true 3D volume, and why the study contains **several planes**: each gives full resolution in a different direction.

# %% [markdown]
# ## 6. Intensities: MRI has no absolute units
#
# In CT, values are Hounsfield units, so water = 0 on every scanner. In MRI the raw values depend on the scanner,
# coil, sequence and patient, and **only relative contrast inside a series means anything**.
# Here are the intensity distributions of all series in this study:

# %% [code] {"jupyter":{"outputs_hidden":false}}
study_vols = {}
for r in study_series.itertuples():
    v, sl = load_series(STUDY_DIR / r.SeriesInstanceUID)
    study_vols[r.SeriesInstanceUID] = (v, sl)

def label_of(r):
    return (f"{r.Anatomical_Plane} {'fluid' if r.Fluid_Sensitive == 1 else 'non-fluid'}"
            f"{' FS' if r.Fat_Suppression == 1 else ''} ({r.short_uid[-5:]})")

def normalize(v):
    lo, hi = np.percentile(v, [1, 99.5])
    return np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
for r in study_series.itertuples():
    v = study_vols[r.SeriesInstanceUID][0]
    fg = v[v > np.percentile(v, 5)]                     # skip most of the background
    axes[0].hist(fg, bins=150, histtype="step", density=True, label=label_of(r))
    nv = normalize(v)
    axes[1].hist(nv[nv > 0.02].ravel(), bins=150, histtype="step", density=True, label=label_of(r))
axes[0].set_title("raw values (foreground)"); axes[0].set_xlabel("raw intensity")
axes[1].set_title("after per-series percentile normalization"); axes[1].set_xlabel("normalized [0, 1]")
axes[0].legend(fontsize=7)
plt.tight_layout(); plt.show()

# %% [markdown]
# **Takeaway:** raw ranges differ even between series of the *same* exam, and they differ far more between sites.
# Normalize **each series separately** (percentile clipping to [0, 1] is a simple, robust default).

# %% [markdown]
# ## 7. The whole study: all sequences side by side
#
# Each series is a different "view" of the same knee. Fluid-sensitive, fat-suppressed sequences make fluid and edema
# bright (effusion, bone bruise, tears). T1 or non-fat-suppressed sequences show anatomy, fat and bone structure.

# %% [code] {"jupyter":{"outputs_hidden":false}}
meta_rows = []
for r in study_series.itertuples():
    v, sl = study_vols[r.SeriesInstanceUID]
    s0 = sl[0]
    ps_ = [float(x) for x in tag(s0, "PixelSpacing", [np.nan, np.nan])]
    iop_ = tag(s0, "ImageOrientationPatient")
    meta_rows.append({
        "series": r.short_uid, "CSV plane": r.Anatomical_Plane,
        "IOP plane": plane_from_iop(iop_)[0] if iop_ is not None else None,
        "fluid": r.Fluid_Sensitive, "fatsat": r.Fat_Suppression,
        "slices": v.shape[0], "rows×cols": f"{v.shape[1]}×{v.shape[2]}",
        "pixel mm": round(ps_[0], 3), "thickness mm": tag(s0, "SliceThickness"),
        "FOV mm": f"{v.shape[1]*ps_[0]:.0f}×{v.shape[2]*ps_[1]:.0f}",
        "transfer syntax": s0.file_meta.TransferSyntaxUID.name,
    })
pd.DataFrame(meta_rows)

# %% [code] {"jupyter":{"outputs_hidden":false}}
k = len(study_series)
fig, axes = plt.subplots(3, k, figsize=(3.6 * k, 11))
axes = np.array(axes).reshape(3, k)
for j, r in enumerate(study_series.itertuples()):
    v = normalize(study_vols[r.SeriesInstanceUID][0])
    for i, frac in enumerate([0.3, 0.5, 0.7]):
        idx = int(frac * (len(v) - 1))
        axes[i, j].imshow(v[idx], cmap="gray"); axes[i, j].axis("off")
        if i == 0:
            axes[i, j].set_title(label_of(r), fontsize=9,
                                 fontweight="bold" if r.SeriesInstanceUID == SERIES_UID else "normal")
        axes[i, j].text(3, 12, f"slice {idx}/{len(v)-1}", color="y", fontsize=7)
plt.suptitle("Every series of the study at 30 % / 50 % / 70 % depth", y=1.0)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### How the series intersect in 3D
#
# Each slice has a physical position, so we can draw all slice centres of all series in one patient-space plot.
# Sagittal stacks advance left→right, coronal front→back, axial bottom→top, and all cover the same knee.

# %% [code] {"jupyter":{"outputs_hidden":false}}
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection="3d")
colors = plt.cm.tab10.colors
for j, r in enumerate(study_series.itertuples()):
    v, sl = study_vols[r.SeriesInstanceUID]
    c = colors[j % 10]
    for q, s in enumerate(sl):
        o = np.array(s.ImagePositionPatient, float)
        io = np.array(s.ImageOrientationPatient, float)
        rd, cd = io[:3], io[3:]
        psp = [float(x) for x in s.PixelSpacing]
        W, H = s.Columns * psp[1], s.Rows * psp[0]
        corners = [o, o + rd * W, o + rd * W + cd * H, o + cd * H]
        # draw outline of every slice; fill only the middle slice of each series
        xs, ys, zs = zip(*(corners + [corners[0]]))
        ax.plot(xs, ys, zs, color=c, lw=0.4, alpha=0.5)
        if q == len(sl) // 2:
            ax.add_collection3d(Poly3DCollection([corners], alpha=0.25, facecolor=c))
    ax.plot([], [], color=c, lw=3, label=label_of(r))
ax.set_xlabel("x → patient left (mm)"); ax.set_ylabel("y → posterior (mm)"); ax.set_zlabel("z → superior (mm)")
ax.legend(fontsize=8, loc="upper left")
ax.set_title("Slice outlines of all series in patient space (middle slice filled)")
plt.tight_layout(); plt.show()

# %% [markdown]
# **Takeaway:** the series are geometrically registered: they share one coordinate system, so a point in the knee
# can be located in every series. Different findings are best seen on different sequences, which is why routing
# series by `(plane, fluid, fat-sat)` helps the model.
#
# | finding | usually best seen on |
# |---|---|
# | ACL, menisci, effusion, contusion | sagittal fluid-sensitive fat-sat |
# | MCL, meniscal body, medial/lateral OA | coronal |
# | PF OA, synovitis, Baker's cyst | axial |
# | fracture lines, osteophytes | non-fat-sat T1 / PD |

# %% [markdown]
# ### Interactive 3D viewer
#
# The same study, now interactive. **Left**: every series in patient space (LPS, mm). Faint lines are the outlines of all slices, and
# the textured plane is each series' current slice. **Right**: the selected series in 2D.
#
# - **scroll** over the 2D image (or ↑/↓) to move through its slices and watch the plane move in 3D
# - the **dashed lines** in 2D show where the other series' current slices cut this image (the "reference lines" a radiologist's viewer shows)
# - **click** a point in 2D: every other series jumps to the slice passing through that point (white dot in 3D)
# - use the sliders to move any series, the **2D** radio to choose which one is shown on the right, and **3D** to hide a series
# - drag in 3D to rotate, right-drag to pan, wheel to zoom. Letters L/R, A/P, S/I mark patient directions.
#
# The viewer loads three.js from a CDN, so **Internet must be on** in the notebook settings. The page is also saved as
# `knee_viewer.html` in the output folder, and you can download it and open it in any browser.

# %% [code] {"jupyter":{"outputs_hidden":false}}
VIEWER_HTML = r'''<!doctype html>
<html><head><meta charset="utf-8">
<title>Knee MRI Viewer</title>
<style>
  :root{--bg:#111418;--panel:#1a1f26;--line:#2c343f;--text:#e6e9ee;--muted:#8d96a3;--accent:#f0b43c}
  *{box-sizing:border-box}
  html,body{margin:0;background:var(--bg);color:var(--text);font:13px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
  #app{display:flex;flex-direction:column;gap:8px;padding:8px;height:100vh}
  #views{display:flex;gap:8px;flex:1;min-height:0}
  .view{position:relative;background:#000;border:1px solid var(--line);border-radius:6px;overflow:hidden}
  #v3d{flex:1.35}#v2d{flex:1}
  canvas{display:block;width:100%;height:100%}
  .hud{position:absolute;left:8px;top:6px;pointer-events:none;font-size:12px;text-shadow:0 0 3px #000}
  .hint{position:absolute;left:8px;bottom:6px;color:var(--muted);font-size:11px;pointer-events:none;text-shadow:0 0 3px #000}
  .ori{position:absolute;color:var(--accent);font-weight:600;font-size:14px;pointer-events:none;text-shadow:0 0 3px #000}
  #controls{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:8px 10px}
  table{width:100%;border-collapse:collapse}
  td,th{padding:3px 6px;text-align:left;white-space:nowrap}
  th{color:var(--muted);font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
  td.sl{width:100%}
  input[type=range]{width:100%;accent-color:var(--accent)}
  .sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:6px}
  tr.active td{background:#252c36}
  .global{display:flex;gap:18px;align-items:center;margin-top:6px;color:var(--muted);flex-wrap:wrap}
  .global label{display:flex;gap:6px;align-items:center}
  button{background:#2a323d;color:var(--text);border:1px solid var(--line);border-radius:4px;padding:3px 10px;cursor:pointer}
  button:hover{background:#34404d}
  #err{display:none;padding:20px;color:#ff8a80}
</style></head>
<body>
<div id="err"></div>
<div id="app">
  <div id="views">
    <div class="view" id="v3d"><canvas id="c3d"></canvas>
      <div class="hud" id="hud3d">3D: patient space (LPS, mm)</div>
      <div class="hint">drag = rotate · right-drag = pan · wheel = zoom</div></div>
    <div class="view" id="v2d"><canvas id="c2d"></canvas>
      <div class="hud" id="hud2d"></div>
      <span class="ori" id="oT"></span><span class="ori" id="oB"></span><span class="ori" id="oL"></span><span class="ori" id="oR"></span>
      <div class="hint">wheel / ↑↓ = scroll slices · click = jump other series to that point</div></div>
  </div>
  <div id="controls">
    <table><thead><tr><th>2D</th><th>3D</th><th>series</th><th>slice</th><th></th></tr></thead><tbody id="rows"></tbody></table>
    <div class="global">
      <label><input type="checkbox" id="outl" checked> slice outlines</label>
      <label>slice opacity <input type="range" id="opac" min="0.2" max="1" step="0.05" value="0.9" style="width:120px"></label>
      <label><input type="checkbox" id="refl" checked> reference lines in 2D</label>
      <button id="reset">reset 3D view</button>
      <button id="mid">all to middle slice</button>
    </div>
  </div>
</div>
<script type="application/json" id="payload">__PAYLOAD__</script>
__THREE_TAG__
<script>
(function(){
if (typeof THREE === "undefined") {
  const e = document.getElementById("err"); e.style.display = "block";
  e.textContent = "three.js could not be loaded. In Kaggle, turn on Internet in the notebook settings, or download knee_viewer.html from the output and open it in a browser.";
  document.getElementById("app").style.display = "none"; return;
}
const P = JSON.parse(document.getElementById("payload").textContent);
const V = (a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]], S=(a,s)=>[a[0]*s,a[1]*s,a[2]*s], D3=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const SUB=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const CROSS=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];

// ---------- decode data ----------
const series = P.series.map((s,i)=>{
  const bin = atob(s.data), u8 = new Uint8Array(bin.length);
  for (let k=0;k<bin.length;k++) u8[k]=bin.charCodeAt(k);
  s.vox = u8; s.idx = Math.floor((s.D-1)/2); s.show3d = true; s.i = i;
  s.slices.forEach(q=>{ q.n = CROSS(q.r,q.c); q.pos = D3(q.n,q.o); });
  return s;
});
let active = Math.max(0, series.findIndex(s=>s.selected));
let clickPoint = null;

function corners(s,k){ const q=s.slices[k];
  return [q.o, V(q.o,S(q.r,s.W)), V(V(q.o,S(q.r,s.W)),S(q.c,s.H)), V(q.o,S(q.c,s.H))]; }

// ---------- 3D ----------
const c3 = document.getElementById("c3d");
const renderer = new THREE.WebGLRenderer({canvas:c3, antialias:true});
renderer.setPixelRatio(window.devicePixelRatio||1);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x06080b);
const camera = new THREE.PerspectiveCamera(35, 1, 1, 5000); camera.up.set(0,0,1);

let lo=[1e9,1e9,1e9], hi=[-1e9,-1e9,-1e9];
series.forEach(s=>s.slices.forEach((q,k)=>corners(s,k).forEach(p=>{for(let a=0;a<3;a++){lo[a]=Math.min(lo[a],p[a]);hi[a]=Math.max(hi[a],p[a]);}})));
const center = S(V(lo,hi),0.5), extent = Math.max(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2]);
const orbit = {theta:-1.05, phi:0.35, dist:extent*2.3, target:new THREE.Vector3(...center)};
function placeCamera(){
  const t=orbit.target, d=orbit.dist;
  camera.position.set(t.x + d*Math.cos(orbit.phi)*Math.cos(orbit.theta),
                      t.y + d*Math.cos(orbit.phi)*Math.sin(orbit.theta),
                      t.z + d*Math.sin(orbit.phi));
  camera.lookAt(t);
}

series.forEach(s=>{
  const col = new THREE.Color(s.color);
  const pts=[];
  s.slices.forEach((q,k)=>{ const c=corners(s,k); for(let e=0;e<4;e++){ pts.push(...c[e],...c[(e+1)%4]); } });
  const g=new THREE.BufferGeometry(); g.setAttribute("position",new THREE.Float32BufferAttribute(pts,3));
  s.outline = new THREE.LineSegments(g,new THREE.LineBasicMaterial({color:col,transparent:true,opacity:0.28}));
  scene.add(s.outline);
  const qg=new THREE.BufferGeometry();
  qg.setAttribute("position",new THREE.Float32BufferAttribute(new Float32Array(12),3));
  qg.setAttribute("uv",new THREE.Float32BufferAttribute([0,0,1,0,1,1,0,1],2));
  qg.setIndex([0,1,2,0,2,3]);
  s.mat = new THREE.MeshBasicMaterial({side:THREE.DoubleSide,transparent:true,opacity:0.9});
  s.quad = new THREE.Mesh(qg,s.mat); scene.add(s.quad);
  const fg=new THREE.BufferGeometry(); fg.setAttribute("position",new THREE.Float32BufferAttribute(new Float32Array(12),3));
  s.frame = new THREE.LineLoop(fg,new THREE.LineBasicMaterial({color:col})); scene.add(s.frame);
  s.rgba = new Uint8Array(s.w*s.h*4);
  s.tex = new THREE.DataTexture(s.rgba, s.w, s.h, THREE.RGBAFormat);
  s.tex.magFilter = THREE.LinearFilter; s.tex.minFilter = THREE.LinearFilter;
  s.mat.map = s.tex;
});

// orientation labels
function label(text,pos){
  const cv=document.createElement("canvas"); cv.width=cv.height=64; const x=cv.getContext("2d");
  x.fillStyle="#f0b43c"; x.font="bold 44px sans-serif"; x.textAlign="center"; x.textBaseline="middle"; x.fillText(text,32,34);
  const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(cv),depthTest:false}));
  sp.position.set(...pos); sp.scale.set(extent*0.07,extent*0.07,1); scene.add(sp);
}
const m = extent*0.62;
label("L",V(center,[m,0,0])); label("R",V(center,[-m,0,0]));
label("P",V(center,[0,m,0])); label("A",V(center,[0,-m,0]));
label("S",V(center,[0,0,m])); label("I",V(center,[0,0,-m]));
const marker = new THREE.Mesh(new THREE.SphereGeometry(extent*0.012,16,12), new THREE.MeshBasicMaterial({color:0xffffff,depthTest:false}));
marker.visible=false; scene.add(marker);

function updateSeries3D(s){
  const c=corners(s,s.idx), arr=[].concat(...c);
  s.quad.geometry.attributes.position.array.set(arr); s.quad.geometry.attributes.position.needsUpdate=true;
  s.quad.geometry.computeBoundingSphere();
  s.frame.geometry.attributes.position.array.set(arr); s.frame.geometry.attributes.position.needsUpdate=true;
  s.frame.geometry.computeBoundingSphere();
  const off=s.idx*s.w*s.h, v=s.vox, o=s.rgba;
  for(let i=0,n=s.w*s.h;i<n;i++){ const g=v[off+i]; o[4*i]=g;o[4*i+1]=g;o[4*i+2]=g;o[4*i+3]=255; }
  s.tex.needsUpdate=true;
  s.quad.visible = s.frame.visible = s.show3d;
  s.outline.visible = s.show3d && document.getElementById("outl").checked;
}

// orbit controls
let drag=null;
c3.addEventListener("contextmenu",e=>e.preventDefault());
c3.addEventListener("pointerdown",e=>{drag={x:e.clientX,y:e.clientY,b:e.button}; c3.setPointerCapture(e.pointerId);});
c3.addEventListener("pointerup",()=>drag=null);
c3.addEventListener("pointermove",e=>{
  if(!drag) return; const dx=e.clientX-drag.x, dy=e.clientY-drag.y; drag.x=e.clientX; drag.y=e.clientY;
  if(drag.b===2||e.shiftKey){
    const f=orbit.dist*0.0015, right=new THREE.Vector3(), up=new THREE.Vector3();
    camera.matrix.extractBasis(right,up,new THREE.Vector3());
    orbit.target.addScaledVector(right,-dx*f).addScaledVector(up,dy*f);
  } else { orbit.theta-=dx*0.008; orbit.phi=Math.max(-1.5,Math.min(1.5,orbit.phi+dy*0.008)); }
  placeCamera(); render3d();
});
c3.addEventListener("wheel",e=>{e.preventDefault(); orbit.dist*=Math.exp(e.deltaY*0.001); placeCamera(); render3d();},{passive:false});

function render3d(){ renderer.render(scene,camera); }

// ---------- 2D ----------
const c2=document.getElementById("c2d"), ctx=c2.getContext("2d");
const off=document.createElement("canvas"), octx=off.getContext("2d");
let view2 = null;
const LET = v=>{ const a=v.map(Math.abs), i=a.indexOf(Math.max(...a)); return [["R","L"],["A","P"],["I","S"]][i][v[i]>0?1:0]; };

function render2d(){
  const s=series[active], q=s.slices[s.idx], dpr=window.devicePixelRatio||1;
  const cw=c2.clientWidth, ch=c2.clientHeight; c2.width=cw*dpr; c2.height=ch*dpr;
  ctx.setTransform(dpr,0,0,dpr,0,0); ctx.fillStyle="#000"; ctx.fillRect(0,0,cw,ch);
  off.width=s.w; off.height=s.h; const img=octx.createImageData(s.w,s.h);
  const o0=s.idx*s.w*s.h; for(let i=0;i<s.w*s.h;i++){const g=s.vox[o0+i]; img.data[4*i]=img.data[4*i+1]=img.data[4*i+2]=g; img.data[4*i+3]=255;}
  octx.putImageData(img,0,0);
  const top=80, bot=34, sc=Math.min((cw-50)/s.W,(ch-top-bot)/s.H), dw=s.W*sc, dh=s.H*sc, x0=(cw-dw)/2, y0=top+(ch-top-bot-dh)/2;
  view2={sc,x0,y0};
  ctx.imageSmoothingEnabled=true; ctx.drawImage(off,x0,y0,dw,dh);
  ctx.strokeStyle=s.color; ctx.lineWidth=2; ctx.strokeRect(x0-1,y0-1,dw+2,dh+2);
  // reference lines: where the other series' current slices cut this slice
  if(document.getElementById("refl").checked) series.forEach(t=>{
    if(t===s||!t.show3d) return; const qb=t.slices[t.idx];
    const a=D3(qb.n,q.r), b=D3(qb.n,q.c), c=D3(qb.n,SUB(q.o,qb.o));
    if(Math.abs(a)<1e-3&&Math.abs(b)<1e-3) return;
    const pts=[];
    if(Math.abs(b)>1e-9){ for(const u of [0,s.W]){const v=-(a*u+c)/b; if(v>=0&&v<=s.H) pts.push([u,v]);} }
    if(Math.abs(a)>1e-9){ for(const v of [0,s.H]){const u=-(b*v+c)/a; if(u>=0&&u<=s.W) pts.push([u,v]);} }
    if(pts.length<2) return;
    ctx.strokeStyle=t.color; ctx.lineWidth=1.5; ctx.setLineDash([6,4]); ctx.beginPath();
    ctx.moveTo(x0+pts[0][0]*sc,y0+pts[0][1]*sc); ctx.lineTo(x0+pts[1][0]*sc,y0+pts[1][1]*sc); ctx.stroke(); ctx.setLineDash([]);
  });
  if(clickPoint){ const d=SUB(clickPoint,q.o), u=D3(d,q.r), v=D3(d,q.c), dist=D3(d,q.n);
    if(Math.abs(dist)<(s.gap||4)){ ctx.strokeStyle="#fff"; ctx.lineWidth=1.5; const X=x0+u*sc,Y=y0+v*sc;
      ctx.beginPath(); ctx.arc(X,Y,6,0,7); ctx.moveTo(X-11,Y);ctx.lineTo(X-4,Y);ctx.moveTo(X+4,Y);ctx.lineTo(X+11,Y);
      ctx.moveTo(X,Y-11);ctx.lineTo(X,Y-4);ctx.moveTo(X,Y+4);ctx.lineTo(X,Y+11); ctx.stroke(); } }
  document.getElementById("hud2d").innerHTML =
    `<b style="color:${s.color}">${s.label}</b><br>slice ${s.idx} / ${s.D-1} · position ${q.pos.toFixed(1)} mm<br>`+
    `${s.W.toFixed(0)} × ${s.H.toFixed(0)} mm · ${s.origRows}×${s.origCols} px (shown ${s.w}×${s.h})`;
  const place=(id,txt,css)=>{const e=document.getElementById(id); e.textContent=txt; Object.assign(e.style,css);};
  place("oR",LET(q.r),{left:(x0+dw+6)+"px",top:(y0+dh/2-9)+"px"});
  place("oL",LET(S(q.r,-1)),{left:(x0-18)+"px",top:(y0+dh/2-9)+"px"});
  place("oB",LET(q.c),{left:(x0+dw/2-5)+"px",top:(y0+dh+2)+"px"});
  place("oT",LET(S(q.c,-1)),{left:(x0+dw/2-5)+"px",top:(y0-20)+"px"});
}

c2.addEventListener("wheel",e=>{e.preventDefault(); step(active, e.deltaY>0?1:-1);},{passive:false});
c2.addEventListener("click",e=>{
  if(!view2) return; const s=series[active], q=s.slices[s.idx], r=c2.getBoundingClientRect();
  const u=(e.clientX-r.left-view2.x0)/view2.sc, v=(e.clientY-r.top-view2.y0)/view2.sc;
  if(u<0||v<0||u>s.W||v>s.H) return;
  clickPoint = V(V(q.o,S(q.r,u)),S(q.c,v));
  series.forEach(t=>{ if(t===s) return; let best=0,bd=1e9;
    t.slices.forEach((qq,k)=>{const d=Math.abs(D3(qq.n,SUB(clickPoint,qq.o))); if(d<bd){bd=d;best=k;}}); t.idx=best; });
  marker.position.set(...clickPoint); marker.visible=true; refreshAll();
});
window.addEventListener("keydown",e=>{ if(e.key==="ArrowUp"){step(active,-1);e.preventDefault();} if(e.key==="ArrowDown"){step(active,1);e.preventDefault();} });

// ---------- controls ----------
const tbody=document.getElementById("rows");
series.forEach((s,i)=>{
  const tr=document.createElement("tr");
  tr.innerHTML=`<td><input type="radio" name="act" ${i===active?"checked":""}></td>
    <td><input type="checkbox" checked></td>
    <td><span class="sw" style="background:${s.color}"></span>${s.label}</td>
    <td class="sl"><input type="range" min="0" max="${s.D-1}" value="${s.idx}"></td>
    <td class="ix" style="min-width:60px;color:var(--muted)"></td>`;
  const [rad,chk,,]=tr.querySelectorAll("input"); const rng=tr.querySelector("input[type=range]");
  rad.onchange=()=>{active=i; refreshAll();};
  chk.onchange=()=>{s.show3d=chk.checked; refreshAll();};
  rng.oninput=()=>{s.idx=+rng.value; refreshAll();};
  s.tr=tr; s.rng=rng; s.ix=tr.querySelector(".ix"); tbody.appendChild(tr);
});
document.getElementById("outl").onchange=refreshAll;
document.getElementById("refl").onchange=refreshAll;
document.getElementById("opac").oninput=e=>{series.forEach(s=>s.mat.opacity=+e.target.value); render3d();};
document.getElementById("reset").onclick=()=>{orbit.theta=-1.05;orbit.phi=0.35;orbit.dist=extent*2.3;orbit.target.set(...center);placeCamera();render3d();};
document.getElementById("mid").onclick=()=>{series.forEach(s=>s.idx=Math.floor((s.D-1)/2)); clickPoint=null; marker.visible=false; refreshAll();};

function step(i,d){ const s=series[i]; s.idx=Math.max(0,Math.min(s.D-1,s.idx+d)); refreshAll(); }
function refreshAll(){
  series.forEach((s,i)=>{ updateSeries3D(s); s.rng.value=s.idx; s.ix.textContent=`${s.idx} / ${s.D-1}`;
    s.tr.className = i===active?"active":""; s.tr.querySelector("input[type=radio]").checked = i===active; });
  render2d(); render3d();
}
function resize(){ const r=c3.parentElement.getBoundingClientRect(); renderer.setSize(r.width,r.height,false);
  camera.aspect=r.width/Math.max(1,r.height); camera.updateProjectionMatrix(); refreshAll(); }
window.addEventListener("resize",resize);
placeCamera(); resize();
})();
</script>
</body></html>
'''

# %% [code] {"jupyter":{"outputs_hidden":false}}
import json, base64, html as html_lib
import torch
import torch.nn.functional as F
from IPython.display import HTML, display

MAX_VOXELS_PER_SERIES = 1_200_000   # keeps the page light; raise for sharper images
MAX_INPLANE = 224                   # longest image side sent to the viewer (px)
THREE_URL = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
PALETTE = ["#4e9cf5", "#f0843c", "#4cc38a", "#e35d6a", "#b07ce8", "#e8c547", "#3ec5d6", "#f27fbf", "#9aa5b1", "#8fd14f"]

def series_payload(r, color):
    v, sl = study_vols[r.SeriesInstanceUID]
    D, H0, W0 = v.shape
    scale = min(1.0, MAX_INPLANE / max(H0, W0), np.sqrt(MAX_VOXELS_PER_SERIES / (D * H0 * W0)))
    h, w = max(8, round(H0 * scale)), max(8, round(W0 * scale))
    vn = torch.from_numpy(normalize(v).astype(np.float32))[None]           # (1, D, H, W)
    small = F.interpolate(vn, size=(h, w), mode="area")[0].numpy()
    u8 = (small * 255).round().astype(np.uint8)
    ps = [float(x) for x in sl[0].PixelSpacing]                            # [row spacing, col spacing]
    slices = [{"o": [float(x) for x in s.ImagePositionPatient],
               "r": [float(x) for x in s.ImageOrientationPatient[:3]],
               "c": [float(x) for x in s.ImageOrientationPatient[3:]]} for s in sl]
    n = np.cross(slices[0]["r"], slices[0]["c"])
    pos = [float(np.dot(n, q["o"])) for q in slices]
    gap = float(np.median(np.abs(np.diff(pos)))) if len(pos) > 1 else 4.0
    return {"label": label_of(r), "color": color, "selected": r.SeriesInstanceUID == SERIES_UID,
            "D": int(D), "h": int(h), "w": int(w), "origRows": int(H0), "origCols": int(W0),
            "W": W0 * ps[1], "H": H0 * ps[0], "gap": gap, "slices": slices,
            "data": base64.b64encode(u8.tobytes()).decode()}

payload = {"series": [series_payload(r, PALETTE[i % len(PALETTE)])
                      for i, r in enumerate(study_series.itertuples())]}
page = (VIEWER_HTML
        .replace("__PAYLOAD__", json.dumps(payload).replace("</", "<\\/"))
        .replace("__THREE_TAG__", f'<script src="{THREE_URL}"></script>'))

out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
(out_dir / "knee_viewer.html").write_text(page, encoding="utf-8")
print(f"viewer payload {len(page) / 1e6:.1f} MB, also saved to {out_dir / 'knee_viewer.html'}")
display(HTML(f'<iframe srcdoc="{html_lib.escape(page)}" style="width:100%;height:860px;border:0;border-radius:6px"></iframe>'))

# %% [markdown]
# ## 8. Labels and report for this study
#
# Only a small subset of training studies has gold labels. The rest have only the radiology report.

# %% [code] {"jupyter":{"outputs_hidden":false}}
row = train_df[train_df.StudyInstanceUID == STUDY_UID]
if row.empty:
    print("Study not found in train.csv")
else:
    row = row.iloc[0]
    lab = row[LABELS]
    if lab.isna().all():
        print("This study has NO gold labels → labels must be derived from the report.\n")
    else:
        print("Gold labels:")
        display(lab.to_frame("value").T)
    report = str(row.get("Report", ""))
    print(f"Report ({len(report)} characters):\n")
    print(textwrap.fill(report, 110) if "\n" not in report else report)

# %% [markdown]
# ## 9. Preparing model input
#
# Series differ in resolution and slice count, but a model needs a fixed shape. The simplest approach is to
# normalize and resample each series to e.g. `32 × 256 × 256`.

# %% [code] {"jupyter":{"outputs_hidden":false}}
import torch
import torch.nn.functional as F

def resize_volume(v, depth=32, size=256):
    t = torch.from_numpy(v.astype(np.float32))[None, None]
    return F.interpolate(t, size=(depth, size, size), mode="trilinear", align_corners=False)[0, 0].numpy()

v_norm = normalize(vol)
v_fixed = resize_volume(v_norm)
print("original :", vol.shape, vol.dtype, f"{vol.nbytes/1e6:.1f} MB")
print("model in :", v_fixed.shape, "float16 →", f"{v_fixed.astype(np.float16).nbytes/1e6:.1f} MB per series")

fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
for j, frac in enumerate([0.2, 0.4, 0.6, 0.8]):
    a = int(frac * (len(v_norm) - 1)); b = int(frac * (len(v_fixed) - 1))
    axes[0, j].imshow(v_norm[a], cmap="gray"); axes[0, j].set_title(f"original slice {a}/{len(v_norm)-1}")
    axes[1, j].imshow(v_fixed[b], cmap="gray"); axes[1, j].set_title(f"resampled slice {b}/31")
for a in axes.ravel(): a.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# **Note:** resampling depth from ~20 to 32 slices interpolates between slices and adds no information.
# Alternatives are padding or cropping to a fixed count, or a model that pools over a variable number of slices.
# Whatever you pick, **use the exact same function at training and inference**, and cache the result
# (`.npy` / zarr) so DICOM decoding does not happen every epoch.

# %% [markdown]
# ## 10. Zooming out: the whole dataset (CSV only)
#
# A quick look at how typical this study is. These cells read only the CSVs, so they run in seconds.

# %% [code] {"jupyter":{"outputs_hidden":false}}
per_study = series_df.groupby("StudyInstanceUID").size()
combo = (series_df.Anatomical_Plane + " | " +
         series_df.Fluid_Sensitive.map({1: "fluid", 0: "non-fluid"}) + " | " +
         series_df.Fat_Suppression.map({1: "FS", 0: "no FS"}))

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
vc = per_study.value_counts().sort_index()
axes[0].bar(vc.index, vc.values, color="#4a7ab5")
axes[0].axvline(per_study.get(STUDY_UID, np.nan), color="r", ls="--")
axes[0].set_title(f"series per study (red = this study: {per_study.get(STUDY_UID, 0)})")
axes[0].set_xlabel("number of series"); axes[0].set_ylabel("studies")

series_df.Anatomical_Plane.value_counts().plot.bar(ax=axes[1], color="#9bbb59")
axes[1].set_title("series by plane"); axes[1].tick_params(axis="x", rotation=0)

combo.value_counts().plot.barh(ax=axes[2], color="#e08a3c")
axes[2].invert_yaxis(); axes[2].set_title("plane | fluid | fat-sat combinations")
plt.tight_layout(); plt.show()

# how often each study has at least one series of each plane
has = series_df.groupby("StudyInstanceUID").Anatomical_Plane.agg(set)
for p in ["Sagittal", "Coronal", "Axial"]:
    print(f"studies with ≥1 {p:8s}: {has.map(lambda s: p in s).mean():6.1%}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
labeled = train_df[LABELS].notna().all(axis=1)
print(f"training studies: {len(train_df)} | with gold labels: {labeled.sum()} | report only: {(~labeled).sum()}")

if labeled.any():
    prev = train_df.loc[labeled, LABELS].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    prev.plot.barh(ax=ax, color="#4a7ab5")
    for i, v in enumerate(prev.values):
        ax.text(v + 0.005, i, f"{v:.0%}", va="center", fontsize=8)
    ax.set_xlabel("share positive"); ax.set_xlim(0, 1)
    ax.set_title(f"Label prevalence among the {labeled.sum()} gold-labelled studies")
    plt.tight_layout(); plt.show()

rep_len = train_df["Report"].fillna("").str.len()
print(f"report length: median {rep_len.median():.0f} chars, 5–95 %: {rep_len.quantile(.05):.0f}–{rep_len.quantile(.95):.0f}")

# %% [markdown]
# ## Summary
#
# | Concept | In this competition |
# |---|---|
# | **Study** | one knee MRI exam = one row of `train.csv` = one prediction row (12 probabilities) |
# | **Series** | one sequence (plane + contrast) = one row of `train_series.csv` = one folder of slices |
# | **Instance** | one `.dcm` file = one 2D slice + header |
# | Slice order | sort by `ImagePositionPatient` projected onto the slice normal |
# | Geometry | LPS millimetres; `ImageOrientationPatient` gives the plane; voxels are strongly anisotropic |
# | Intensities | no absolute units → normalize per series |
# | Labels | gold for a small subset; the rest must come from the (multilingual) report |
#
# **Next steps:** scan all headers with `pydicom.dcmread(..., stop_before_pixels=True)` to see the distribution of
# resolutions, transfer syntaxes and slice counts across the whole dataset, then build and cache the preprocessing.
