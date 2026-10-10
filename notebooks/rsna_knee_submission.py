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
#                     model group 1 (e.g. trained on labels_v3):  model_fold0.pt … model_fold4.pt
#                     model group 2 (e.g. trained on labels_v4):  model_fold0.pt … model_fold4.pt
#                              │  each model sees the study twice: as is, and mirrored (left↔right knee)
#                              ▼
#                     average inside each group  →  rank-blend the groups per finding  ──►  submission.csv
# ```
#
# **Model groups.** Every attached dataset that contains a `train_config.json` is one group (its `model_fold*.pt` files
# are averaged). With one group the notebook behaves as before. With several, the groups are blended **by rank**
# (section 2 explains why), with an optional weight per group and finding.
#
# **Two crops (v8).** A group's `train_config.json` records the crop its cache was built with (`crop_mm`: 140 for the
# whole-knee models, 90 for the joint-region models). The DICOMs are decoded once per study and resampled once per
# distinct crop, so attaching both kinds costs decoding nothing extra; each group predicts on its own crop.
#
# **Inputs to attach:** the competition data, one or more model datasets (weights + `train_config.json`), **one** cache
# dataset (only for `knee_preproc.py` and its `config.json`; the shards are not read; any cropped cache will do, the crop
# itself comes from each group), and `knee-dicom-wheels` (offline DICOM decoders).
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
# Weight of a model group for a finding (default 1 for everything not listed). A group is named after the label file it
# was trained on. The v3 labels used the wrong definitions for ACL, MCL and PF OA, so the v3 models are left out there.
# Ignored when only one group is attached. {} = equal weights everywhere.
BLEND_WEIGHTS = {}           # e.g. {"labels_v3": {"ACL": 0}}: per-group, per-finding weights (default 1)

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
PREPROC_DIR = find_dirs_with("knee_preproc.py", skip={ROOT})[0]
CACHE_CFG = json.loads(next(PREPROC_DIR.rglob("config.json")).read_text())

GROUPS = []                   # one per attached model dataset
for d in sorted(find_dirs_with("train_config.json", skip={ROOT})):
    cfg = json.loads((d / "train_config.json").read_text())
    files = sorted(d.glob("model_fold*.pt"))
    if not files:
        continue
    name = cfg.get("group_name") or str(cfg.get("labels_file", d.name)).rsplit(".", 1)[0]
    if any(g["name"] == name for g in GROUPS):                # two groups trained on the same labels: add the folder name
        name = f"{name}@{d.name}"
    GROUPS.append({"name": name, "dir": d, "cfg": cfg, "files": files})

print("scoring re-run:", IS_RERUN)
print("competition :", ROOT)
print("preprocess  :", PREPROC_DIR, "| DEPTH", CACHE_CFG["DEPTH"], "| IMG", CACHE_CFG["IMG"])
assert GROUPS, "no train_config.json with model_fold*.pt found: is a models dataset attached?"
for g in GROUPS:
    g["crop"] = float(g["cfg"].get("crop_mm") or CACHE_CFG.get("CROP_MM") or 0)   # 0 = whole field of view (first cache)
    print(f"model group '{g['name']}': {g['dir']} | {g['cfg']['backbone']} | roles {g['cfg']['roles']} | crop {g['crop']:g} mm | "
          f"{[f.name for f in g['files']]} | gold AUC at training {g['cfg'].get('gold_macro_auc', float('nan')):.3f}")
    assert g["cfg"].get("depth", CACHE_CFG["DEPTH"]) == CACHE_CFG["DEPTH"] and g["cfg"].get("img", CACHE_CFG["IMG"]) == CACHE_CFG["IMG"], \
        "this group was trained on another cache shape than knee_preproc produces"
G = len(GROUPS)
CROPS = sorted({g["crop"] for g in GROUPS})                   # each test study is resampled once per crop
print("crops to preprocess (mm):", CROPS)

# %% [markdown]
# ## 2. CSVs, prior, and the safety-net submission
#
# Same as in the dummy notebook: `sample_submission.csv` defines rows and columns; the prior (label prevalence among gold
# studies, here taken from the training config's label list with 0.5 as fallback) is what a study gets if its images
# cannot be processed. The safety net is written before any image is touched.
#
# **Blending several groups by rank.** The metric (AUC) only looks at the *order* of the studies within a finding. Two
# model groups trained on different labels can order the studies equally well and still output numbers on different
# scales (the v4 labels have fewer positives, so those models give lower probabilities). Averaging the raw numbers would
# let the group with the wider scale dominate. So each group's predictions are first replaced by their rank among all
# test studies (0 = lowest, 1 = highest), and the ranks are averaged with the weights from section 1.

# %% [code] {"jupyter":{"outputs_hidden":false}}
test_series = pd.read_csv(ROOT / "test_series.csv")
sample = pd.read_csv(ROOT / "sample_submission.csv")
ID_COL, LABELS = sample.columns[0], list(sample.columns[1:])
from scipy.stats import rankdata
for g in GROUPS:
    assert LABELS == g["cfg"]["labels"], f"label columns differ from training (group {g['name']})"

W = np.ones((G, len(LABELS)))                                 # weight per group and finding
if G > 1:
    for gname, per in BLEND_WEIGHTS.items():
        assert gname in [g["name"] for g in GROUPS], f"BLEND_WEIGHTS names '{gname}', attached groups: {[g['name'] for g in GROUPS]}"
        for lab, w in per.items():
            W[[g["name"] for g in GROUPS].index(gname), LABELS.index(lab)] = w
    assert (W.sum(0) > 0).all(), "every finding needs at least one group with a positive weight"
    print("blend weights (rows = groups, columns = findings):")
    print(pd.DataFrame(W, index=[g["name"] for g in GROUPS], columns=LABELS).to_string())

prior = {l: 0.5 for l in LABELS}
try:
    tr = pd.read_csv(ROOT / "train.csv", usecols=[ID_COL] + LABELS).dropna()
    if len(tr):
        prior.update(tr[LABELS].mean().clip(0.01, 0.99).to_dict())
except Exception as e:
    print("prior fallback 0.5:", e)

def make_submission(preds):
    # preds: {study: array (groups, findings)} → one number per study and finding
    sub = sample[[ID_COL]].copy()
    out = pd.DataFrame(index=pd.Index([], name=ID_COL), columns=LABELS, dtype=float)
    if preds:
        uids = list(preds)
        A = np.stack([preds[u] for u in uids])                # (studies, groups, findings)
        if G > 1:
            A = rankdata(A, axis=0) / len(uids)               # rank of each study within its group and finding
        out = pd.DataFrame((A * W[None]).sum(1) / W.sum(0)[None], index=uids, columns=LABELS)
    for lab in LABELS:
        fill = prior[lab] if G == 1 else 0.5                  # a failed study: prior (one group) or the middle rank
        sub[lab] = sub[ID_COL].map(out[lab]).fillna(fill).clip(0, 1)
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
#   `pretrained=False`: the ImageNet weights are not needed, our trained weights replace them. Every group is built
#   from its own `train_config.json` (backbone, series, images per series, normalisation).

# %% [code] {"jupyter":{"outputs_hidden":false}}
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(PREPROC_DIR))
os.environ["PYTHONPATH"] = str(PREPROC_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")   # for the parallel workers
import knee_preproc as kp
import inspect
assert "crop_mm" in inspect.signature(kp.resample).parameters, "attach a cropped cache (v5 or later): its knee_preproc.py takes crop_mm"

def process_study_crops(study_uid, study_rows, series_root, roles, depth, img, crops):
    # kp.process_study, but the decoded series is resampled once per crop: ({crop: {role: uint8 array or None}}, metas)
    torch.set_num_threads(1)
    arrays, metas = {c: {} for c in crops}, []
    for role, (plane, fluid) in roles.items():
        m = {"StudyInstanceUID": study_uid, "role": role, "ok": False, "error": None}
        t0 = time.time()
        try:
            best, fallback = kp.select_series(study_rows, plane, fluid, Path(series_root) / study_uid)
            if best is None:
                raise LookupError(f"no {plane} series")
            m.update(SeriesInstanceUID=best.SeriesInstanceUID, fallback=fallback, n_files=int(best.n_files))
            vol, hdr, pos = kp.load_series(Path(series_root) / study_uid / best.SeriesInstanceUID, kp.GEOMETRY[plane]["slice_axis"])
            m.update(n_slices=vol.shape[0], transfer_syntax=hdr.file_meta.TransferSyntaxUID.name)
            vol, ps = kp.canonicalize(vol, hdr, plane)
            vol01 = kp.normalize_u8(vol)
            for c in crops:
                arrays[c][role], info = kp.resample(vol01, ps, depth, img, crop_mm=c)
                m[f"tissue_outside_{c:g}"] = info.get("tissue_outside")
            m["ok"] = True
        except Exception as e:
            for c in crops:
                arrays[c][role] = None
            m["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        m["seconds"] = round(time.time() - t0, 2)
        metas.append(m)
    return arrays, metas

import torch, torch.nn as nn, timm
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHANNELS_LAST = DEVICE == "cuda"
torch.backends.cudnn.benchmark = True
ROLES_ALL = {k: tuple(v) for k, v in CACHE_CFG["ROLES"].items()}
ROLES = []                                                    # every series type any group needs, preprocessed once
for g in GROUPS:
    ROLES += [r for r in g["cfg"]["roles"] if r not in ROLES]
DEPTH, IMG = CACHE_CFG["DEPTH"], CACHE_CFG["IMG"]

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

def prepare(x_u8, cfg):
    x = x_u8.to(DEVICE).float() / 255.0
    return (make_triplets(x, cfg["n_triplets"]) - cfg["mean"]) / cfg["std"]

def mirror(x_u8, sel=None):
    # the same "other knee" mirror used as training augmentation; sel = bool tensor of the studies to mirror (default all)
    x = x_u8.clone()
    sel = slice(None) if sel is None else sel
    for j, role in enumerate(ROLES):
        x[sel, j] = x[sel, j].flip(1) if role.startswith("sag") else x[sel, j].flip(3)
    return x

def load_model(path, backbone, n_roles, n_out=12):
    m = KneeNet(backbone, False, n_roles, n_out).to(DEVICE)
    m.load_state_dict({k: v.float() for k, v in torch.load(path, map_location=DEVICE).items()})
    if CHANNELS_LAST:
        m = m.to(memory_format=torch.channels_last)
    return m.eval()

# Canonical-side groups (v7): trained with every knee shown as a right knee, without mirror augmentation or TTA. Such a
# group ships the left/right classifier it was built with (`side_model.pt`); at test time the classifier decides the
# side of each study, left knees are mirrored, and the group predicts the single canonical view.
side_models = {}                                              # one classifier per distinct file (groups share it)
for g in GROUPS:
    g["role_idx"] = [ROLES.index(r) for r in g["cfg"]["roles"]]         # this group's series among the preprocessed ones
    g["models"] = [load_model(f, g["cfg"]["backbone"], len(g["cfg"]["roles"])) for f in g["files"]]
    g["canon"] = g["cfg"].get("canonical_side")
    g["tta"] = USE_TTA and g["cfg"].get("tta", True)
    if g["canon"]:
        sp = g["dir"] / g["cfg"].get("side_model", "side_model.pt")
        scfg = json.loads((g["dir"] / "side_config.json").read_text())
        key = (sp.stat().st_size, scfg["backbone"], tuple(scfg["roles"]))
        if key not in side_models:
            side_models[key] = dict(model=load_model(sp, scfg["backbone"], len(scfg["roles"]), 1), cfg=scfg,
                                    role_idx=[ROLES.index(r) for r in scfg["roles"]])
        g["side"] = side_models[key]
    print(f"group '{g['name']}': loaded {len(g['models'])} model(s) on {DEVICE}" +
          (f" | canonical side {g['canon']} (side classifier {g['side']['cfg']['backbone']}, "
           f"hold-out accuracy at training {g['side']['cfg'].get('holdout_accuracy')})" if g["canon"] else "") +
          f" | mirror TTA {g['tta']}")
side_log = []                                                 # p_left per study, for section 6

@torch.no_grad()
def side_p_left(x_u8, rmask, s):
    # antisymmetric average: p = (f(x) + 1 - f(mirror x)) / 2, exactly as in the side job
    rm = rmask[:, s["role_idx"]].to(DEVICE)
    out = 0
    for flip in (False, True):
        v = mirror(x_u8) if flip else x_u8
        x = prepare(v[:, s["role_idx"]], s["cfg"])
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            p = torch.sigmoid(s["model"](x, rm).float())[:, 0]
        out = out + (1 - p if flip else p)
    return 0.5 * out

@torch.no_grad()
def predict_batch(xs, rmask):
    # xs: {crop: uint8 tensor (studies, roles, DEPTH, IMG, IMG)} → array (groups, studies, findings): inside a group,
    # the mean over its fold models and its view(s), on the crop it was trained with
    out, canon_views = [], {}
    for g in GROUPS:
        x_u8 = xs[g["crop"]]
        if g["canon"]:
            key = (id(g["side"]), g["crop"])
            if key not in canon_views:
                p_left = side_p_left(x_u8, rmask, g["side"])
                side_log.extend(p_left.cpu().numpy().tolist())
                need_flip = (p_left > 0.5) if g["canon"] == "R" else (p_left <= 0.5)
                canon_views[key] = mirror(x_u8, need_flip.cpu()) if bool(need_flip.any()) else x_u8
            base = canon_views[key]
        else:
            base = x_u8
        views = [base, mirror(base)] if g["tta"] else [base]
        rm, probs = rmask[:, g["role_idx"]].to(DEVICE), []
        for v in views:
            x = prepare(v[:, g["role_idx"]], g["cfg"])
            for m in g["models"]:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                    probs.append(torch.sigmoid(m(x, rm).float()))
        out.append(torch.stack(probs).mean(0))
    return torch.stack(out).cpu().numpy()

# %% [markdown]
# ## 5. Preprocess and predict, chunk by chunk
#
# Studies go through in chunks of `CHUNK`: the 4 CPUs preprocess a chunk in parallel (DICOM decoding is the slow part;
# a second crop only adds a cheap resample), then the GPU predicts it, each group on its crop. Per study and role we keep a small log (ok, error, transfer syntax) for section 6.
# Every few chunks the submission file is rewritten, so even a crash late in the run leaves real predictions behind.

# %% [code] {"jupyter":{"outputs_hidden":false}}
from joblib import Parallel, delayed

SERIES_ROOT = ROOT / "test_series"
rows_of = {s: g for s, g in test_series.groupby("StudyInstanceUID")}
roles_used = {r: ROLES_ALL[r] for r in ROLES}

def run(study):
    try:
        return process_study_crops(study, rows_of.get(study, test_series.iloc[:0]), SERIES_ROOT, roles_used, DEPTH, IMG, CROPS)
    except Exception as e:
        return {c: {r: None for r in ROLES} for c in CROPS}, [{"StudyInstanceUID": study, "role": "all", "ok": False, "error": repr(e)[:200]}]

studies = sample[ID_COL].tolist()
preds, metas, t0 = {}, [], time.time()
n_jobs = os.cpu_count() or 2
for c in range(0, len(studies), CHUNK):
    chunk = studies[c:c + CHUNK]
    try:
        res = Parallel(n_jobs=n_jobs)(delayed(run)(s) for s in chunk)
    except Exception:
        traceback.print_exc(); continue
    X = {cr: np.zeros((len(chunk), len(ROLES), DEPTH, IMG, IMG), np.uint8) for cr in CROPS}
    RM = np.zeros((len(chunk), len(ROLES)), np.float32)   # a role is present for all crops or for none
    for i, (arrs, meta) in enumerate(res):
        metas += meta
        for j, role in enumerate(ROLES):
            if arrs[CROPS[0]].get(role) is not None:
                for cr in CROPS:
                    X[cr][i, j] = arrs[cr][role]
                RM[i, j] = 1
    keep = RM.sum(1) > 0                                    # studies with at least one usable series
    try:
        for b in range(0, int(keep.sum()), 8):
            idx = np.where(keep)[0][b:b + 8]
            p = predict_batch({cr: torch.from_numpy(X[cr][idx]) for cr in CROPS}, torch.from_numpy(RM[idx]))
            for k, i in enumerate(idx):
                preds[chunk[i]] = p[:, k]                         # (groups, findings)
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
# - Studies without any usable series got the prior (with several model groups: the middle rank, 0.5).

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
    for cr in CROPS:
        col = f"tissue_outside_{cr:g}"
        if col in meta_df:
            print(f"crop {cr:g} mm: median share of tissue outside the window per role:",
                  meta_df.groupby("role")[col].median().round(3).to_dict())
print("studies falling back to the prior:", len(studies) - len(preds))
if side_log:
    sl = np.asarray(side_log)
    print(f"knee side (canonical groups): {len(sl)} studies | left {np.mean(sl > 0.5):.1%} | "
          f"uncertain (|p - 0.5| < 0.1): {np.mean(np.abs(sl - 0.5) < 0.1):.1%} | "
          f"p_left quantiles 5/50/95 %: {np.percentile(sl, [5, 50, 95]).round(3).tolist()}")

# %% [markdown]
# ## 7. Final submission

# %% [code] {"jupyter":{"outputs_hidden":false}}
sub = make_submission(preds)
validate_and_write(sub)
print(f"total runtime {(time.time() - T_START) / 60:.1f} min | models " +
      ", ".join(f"{g['name']}: {len(g['models'])} @ {g['crop']:g} mm{' canonical ' + g['canon'] if g['canon'] else ''}" for g in GROUPS) +
      f" | TTA {[g['tta'] for g in GROUPS]}")
sub.head()

# %% [markdown]
# ## 8. Submitting
#
# 1. Settings: **Internet off**, accelerator **GPU T4 ×2**, all four inputs attached.
# 2. **Save Version → Save & Run All**. The interactive/commit run only sees the 3 example test studies, so it finishes in
#    a few minutes; check that sections 5–6 show predictions and no errors.
# 3. Competition page → **Submit Prediction** → this notebook/version.
# 4. To blend model groups, attach several model datasets (for example `knee-models` and `knee-models-v4`): each one
#    with a `train_config.json` becomes a group, and section 2 prints the weights that are used. Attach only the groups
#    you want in the submission.
