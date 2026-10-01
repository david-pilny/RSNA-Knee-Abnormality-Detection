# %% [markdown]
# # RSNA Knee: submission (trained models → `submission.csv`)
#
# This is the notebook Kaggle re-runs on the hidden test set (~1,300 studies), **offline**, within 9 hours.
# It joins everything we built:
#
# ```
# hidden test DICOMs ──► knee_preproc.process_study()  (the SAME code that built the training cache)
#                              │  3 series per study → 24 × 224 × 224 uint8
#                              ▼
#                     model_fold0.pt … model_fold4.pt   (whatever folds are in the dataset)
#                              │  each model sees the study twice: as is, and mirrored (left↔right knee)
#                              ▼
#                     average of all predictions  ──►  submission.csv
# ```
#
# **Inputs to attach:** the competition data, `knee-models` (weights + `train_config.json`), `knee-mri-cache`
# (only for `knee_preproc.py` and its `config.json`; the 10 GB of shards are not read), and `knee-dicom-wheels`
# (offline DICOM decoders).
# **Settings:** Accelerator **GPU T4 ×2** (one GPU is used), Internet **off** (required for submission).
#
# Safety rules carried over from the dummy submission: a valid `submission.csv` is written **first**; every study is
# processed inside `try/except`, and a study whose images fail gets the training prior instead of crashing the run.

# %% [markdown]
# ## 1. Settings and paths

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os, sys, time, glob, json, subprocess, traceback
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

IS_RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
T_START = time.time()
CHUNK = 32                    # studies preprocessed in parallel, then predicted together
USE_TTA = True                # also predict the mirrored knee and average
RAISE_IF_DECODE_FAIL = None   # e.g. 0.05: in the scoring run, fail on purpose if >5 % of series can't be read
                              # (a one-off diagnostic: "Submission error" then means decoding is broken)

OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT = Path(os.environ.get("INPUT_BASE", "/kaggle/input"))
SLUG = "rsna-knee-abnormality-detection"

def find_dirs_with(filename, base=INPUT, max_depth=6, skip=()):
    # shallow search for folders that contain `filename` (never crawls the DICOM folders)
    hits, frontier = [], [Path(base)]
    for _ in range(max_depth):
        nxt = []
        for d in frontier:
            if d in skip or d.name in ("train_series", "test_series"):
                continue
            try:
                kids = list(d.iterdir())
            except OSError:
                continue
            if any(k.name == filename for k in kids):
                hits.append(d)
            nxt += [k for k in kids if k.is_dir()]
        frontier = nxt
    return hits

ROOT = next((Path(p) for p in [os.environ.get("KNEE_ROOT", ""), f"{INPUT}/competitions/{SLUG}", f"{INPUT}/{SLUG}"]
             if p and (Path(p) / "test.csv").exists()), None) or find_dirs_with("test.csv")[0]
MODEL_DIR = find_dirs_with("train_config.json", skip={ROOT})[0]
PREPROC_DIR = find_dirs_with("knee_preproc.py", skip={ROOT})[0]
CACHE_CFG = json.loads(next(PREPROC_DIR.rglob("config.json")).read_text())
TRAIN_CFG = json.loads((MODEL_DIR / "train_config.json").read_text())
MODEL_FILES = sorted(MODEL_DIR.glob("model_fold*.pt"))

print("scoring re-run:", IS_RERUN)
print("competition :", ROOT)
print("models      :", MODEL_DIR, [f.name for f in MODEL_FILES])
print("preprocess  :", PREPROC_DIR)
print("backbone", TRAIN_CFG["backbone"], "| roles", TRAIN_CFG["roles"], "| DEPTH", CACHE_CFG["DEPTH"], "| IMG", CACHE_CFG["IMG"])
assert MODEL_FILES, "no model_fold*.pt found: is the knee-models dataset attached?"

# %% [markdown]
# ## 2. CSVs, prior, and the safety-net submission
#
# Same as in the dummy notebook: `sample_submission.csv` defines rows and columns; the prior (label prevalence among gold
# studies, here taken from the training config's label list with 0.5 as fallback) is what a study gets if its images
# cannot be processed. The safety net is written before any image is touched.

# %% [code] {"jupyter":{"outputs_hidden":false}}
test_series = pd.read_csv(ROOT / "test_series.csv")
sample = pd.read_csv(ROOT / "sample_submission.csv")
ID_COL, LABELS = sample.columns[0], list(sample.columns[1:])
assert LABELS == TRAIN_CFG["labels"], "label columns differ from training"

prior = {l: 0.5 for l in LABELS}
try:
    tr = pd.read_csv(ROOT / "train.csv", usecols=[ID_COL] + LABELS).dropna()
    if len(tr):
        prior.update(tr[LABELS].mean().clip(0.01, 0.99).to_dict())
except Exception as e:
    print("prior fallback 0.5:", e)

def make_submission(preds):
    sub = sample[[ID_COL]].copy()
    for lab in LABELS:
        sub[lab] = sub[ID_COL].map(lambda u: preds.get(u, {}).get(lab, np.nan)).fillna(prior[lab]).clip(0, 1)
    return sub

def validate_and_write(sub, path=OUT_DIR / "submission.csv"):
    assert list(sub.columns) == list(sample.columns)
    assert sub[ID_COL].tolist() == sample[ID_COL].tolist()
    assert not sub[LABELS].isna().any().any()
    sub.to_csv(path, index=False)
    print(f"✓ wrote {path} ({len(sub)} rows)")

validate_and_write(make_submission({}))
print(f"test studies: {len(sample)} | test series: {len(test_series)}")

# %% [markdown]
# ## 3. Offline DICOM decoders
#
# Compressed series (JPEG 2000, JPEG Lossless) need `pylibjpeg` + plugins. Without internet they are installed from the
# attached `knee-dicom-wheels` dataset. If it is missing, uncompressed series still work and section 6 reports how many
# series failed.

# %% [code] {"jupyter":{"outputs_hidden":false}}
def find_wheel_dirs(base=INPUT, max_depth=6):
    hits, frontier = [], [Path(base)]
    for _ in range(max_depth):
        nxt = []
        for d in frontier:
            if d == ROOT or d.name in ("train_series", "test_series", "cache"):
                continue
            try:
                kids = list(d.iterdir())
            except OSError:
                continue
            if any(k.suffix == ".whl" for k in kids):
                hits.append(str(d))
            nxt += [k for k in kids if k.is_dir()]
        frontier = nxt
    return hits

wheel_dirs = find_wheel_dirs()
if wheel_dirs:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", wheel_dirs[0], "-q",
                        "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], capture_output=True, text=True)
    print("offline install from", wheel_dirs[0], "→", "ok" if r.returncode == 0 else r.stderr[-400:])
else:
    print("no wheels found: compressed DICOMs may fail to decode")
for mod in ["pylibjpeg", "libjpeg", "openjpeg"]:
    try:
        __import__(mod); print(f"  {mod:10s} available")
    except ImportError:
        print(f"  {mod:10s} missing")

# %% [markdown]
# ## 4. Load the preprocessing code and the models
#
# - `knee_preproc.py` is imported from the cache dataset: the **identical** code that produced the training images.
# - Each `model_fold*.pt` is loaded into the same architecture as in training (copied below unchanged), with
#   `pretrained=False`: the ImageNet weights are not needed, our trained weights replace them.

# %% [code] {"jupyter":{"outputs_hidden":false}}
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(PREPROC_DIR))
os.environ["PYTHONPATH"] = f"{PREPROC_DIR}:{os.environ.get('PYTHONPATH', '')}"   # for the parallel workers
import knee_preproc as kp

import torch, torch.nn as nn, timm
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHANNELS_LAST = DEVICE == "cuda"
torch.backends.cudnn.benchmark = True
ROLES_ALL = {k: tuple(v) for k, v in CACHE_CFG["ROLES"].items()}
ROLES = TRAIN_CFG["roles"]
DEPTH, IMG, NT = CACHE_CFG["DEPTH"], CACHE_CFG["IMG"], TRAIN_CFG["n_triplets"]
MEAN, STD = TRAIN_CFG["mean"], TRAIN_CFG["std"]

# ---- architecture: identical to the training notebook ----
class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))
    def forward(self, h):
        w = torch.softmax(self.a(h), dim=1)
        return (w * h).sum(1)

class KneeNet(nn.Module):
    def __init__(self, backbone, pretrained, n_roles, n_out=12):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=3)
        d = self.enc.num_features
        self.pool = AttnPool(d)
        self.role_emb = nn.Parameter(torch.zeros(n_roles, d))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(n_roles * d + n_roles, 512), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(512, n_out))
    def forward(self, x, rmask):
        B, R, T = x.shape[:3]
        imgs = x.flatten(0, 2)
        if CHANNELS_LAST:
            imgs = imgs.contiguous(memory_format=torch.channels_last)
        f = self.enc(imgs)
        f = self.pool(f.view(B * R, T, -1)).view(B, R, -1)
        f = (f + self.role_emb) * rmask.unsqueeze(-1)
        return self.head(torch.cat([f.flatten(1), rmask], dim=1))

def make_triplets(x, n):
    D = x.shape[2]
    centers = torch.linspace(1, D - 2, n, device=x.device).round().long()
    return x[:, :, torch.stack([centers - 1, centers, centers + 1], dim=1)]

def prepare(x_u8):
    x = x_u8.to(DEVICE).float() / 255.0
    return (make_triplets(x, NT) - MEAN) / STD

def mirror(x_u8):
    # the same "other knee" mirror used as training augmentation
    x = x_u8.clone()
    for j, role in enumerate(ROLES):
        x[:, j] = x[:, j].flip(1) if role.startswith("sag") else x[:, j].flip(3)
    return x

models = []
for f in MODEL_FILES:
    m = KneeNet(TRAIN_CFG["backbone"], False, len(ROLES)).to(DEVICE)
    m.load_state_dict({k: v.float() for k, v in torch.load(f, map_location=DEVICE).items()})
    if CHANNELS_LAST:
        m = m.to(memory_format=torch.channels_last)
    models.append(m.eval())
print(f"loaded {len(models)} model(s) on {DEVICE}")

@torch.no_grad()
def predict_batch(x_u8, rmask):
    rm = rmask.to(DEVICE)
    views = [x_u8, mirror(x_u8)] if USE_TTA else [x_u8]
    probs = []
    for m in models:
        for v in views:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                probs.append(torch.sigmoid(m(prepare(v), rm).float()))
    return torch.stack(probs).mean(0).cpu().numpy()

# %% [markdown]
# ## 5. Preprocess and predict, chunk by chunk
#
# Studies go through in chunks of `CHUNK`: the 4 CPUs preprocess a chunk in parallel (DICOM decoding is the slow part),
# then the GPU predicts it. Per study and role we keep a small log (ok, error, transfer syntax) for section 6.
# Every few chunks the submission file is rewritten, so even a crash late in the run leaves real predictions behind.

# %% [code] {"jupyter":{"outputs_hidden":false}}
from joblib import Parallel, delayed

SERIES_ROOT = ROOT / "test_series"
rows_of = {s: g for s, g in test_series.groupby("StudyInstanceUID")}
roles_used = {r: ROLES_ALL[r] for r in ROLES}

def run(study):
    try:
        return kp.process_study(study, rows_of.get(study, test_series.iloc[:0]), SERIES_ROOT, roles_used, DEPTH, IMG)
    except Exception as e:
        return {r: None for r in ROLES}, [{"StudyInstanceUID": study, "role": "all", "ok": False, "error": repr(e)[:200]}]

studies = sample[ID_COL].tolist()
preds, metas, t0 = {}, [], time.time()
n_jobs = os.cpu_count() or 2
for c in range(0, len(studies), CHUNK):
    chunk = studies[c:c + CHUNK]
    try:
        res = Parallel(n_jobs=n_jobs)(delayed(run)(s) for s in chunk)
    except Exception:
        traceback.print_exc(); continue
    X = np.zeros((len(chunk), len(ROLES), DEPTH, IMG, IMG), np.uint8)
    RM = np.zeros((len(chunk), len(ROLES)), np.float32)
    for i, (arrs, meta) in enumerate(res):
        metas += meta
        for j, role in enumerate(ROLES):
            if arrs.get(role) is not None:
                X[i, j] = arrs[role]; RM[i, j] = 1
    keep = RM.sum(1) > 0                                    # studies with at least one usable series
    try:
        for b in range(0, int(keep.sum()), 8):
            idx = np.where(keep)[0][b:b + 8]
            p = predict_batch(torch.from_numpy(X[idx]), torch.from_numpy(RM[idx]))
            for k, i in enumerate(idx):
                preds[chunk[i]] = dict(zip(LABELS, p[k].tolist()))
    except Exception:
        traceback.print_exc()
    done = min(c + CHUNK, len(studies)); el = time.time() - t0
    if (c // CHUNK) % 5 == 0 or done == len(studies):
        validate_and_write(make_submission(preds))
        print(f"{done}/{len(studies)} studies | {el / 60:.1f} min | ETA {el / done * (len(studies) - done) / 60:.0f} min", flush=True)

meta_df = pd.DataFrame(metas)
print(f"predicted {len(preds)}/{len(studies)} studies in {(time.time() - t0) / 60:.1f} min")

# %% [markdown]
# ## 6. What went wrong, if anything
#
# - **Role success rate:** share of test studies where each series type could be read.
# - **Errors by type and transfer syntax:** decoding errors concentrated on one compression mean the decoder wheels are
#   missing or broken; "no … series" means the study simply has no such series (normal, the model handles missing ones).
# - Studies without any usable series got the prior.

# %% [code] {"jupyter":{"outputs_hidden":false}}
if len(meta_df):
    print("role success rate:", meta_df.groupby("role").ok.mean().round(3).to_dict())
    err = meta_df[~meta_df.ok.astype(bool)]
    if len(err):
        print("\nerrors by type:\n", err.error.fillna("?").str.split(":").str[0].value_counts().head(8).to_string())
        if "transfer_syntax" in err:
            print("\nerrors by transfer syntax:\n", err.transfer_syntax.fillna("—").value_counts().head(8).to_string())
    decode_fail = (~meta_df.ok.astype(bool) & ~meta_df.error.fillna("").str.startswith("LookupError")).mean()
    print(f"\nseries that exist but failed to decode: {decode_fail:.1%}")
    if IS_RERUN and RAISE_IF_DECODE_FAIL is not None and decode_fail > RAISE_IF_DECODE_FAIL:
        raise RuntimeError(f"decode failure rate {decode_fail:.1%} > {RAISE_IF_DECODE_FAIL:.0%} (diagnostic stop)")
print("studies falling back to the prior:", len(studies) - len(preds))

# %% [markdown]
# ## 7. Final submission

# %% [code] {"jupyter":{"outputs_hidden":false}}
sub = make_submission(preds)
validate_and_write(sub)
print(f"total runtime {(time.time() - T_START) / 60:.1f} min | models {len(models)} | TTA {USE_TTA}")
sub.head()

# %% [markdown]
# ## 8. Submitting
#
# 1. Settings: **Internet off**, accelerator **GPU T4 ×2**, all four inputs attached.
# 2. **Save Version → Save & Run All**. The interactive/commit run only sees the 3 example test studies, so it finishes in
#    a few minutes; check that sections 5–6 show predictions and no errors.
# 3. Competition page → **Submit Prediction** → this notebook/version.
# 4. When more folds are trained, add their `model_fold*.pt` to the `knee-models` dataset (new version), update the input
#    here to the new version, and submit again: the notebook automatically uses every fold it finds.
