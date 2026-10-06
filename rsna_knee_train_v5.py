# %% [markdown]
# # RSNA Knee: training v5 (cropped 256 px cache, both GPUs)
#
# Same model idea as the baseline (2.5D multi-view: slice triplets → shared CNN → attention pooling per series → MLP),
# with three changes:
#
# | Change | Why |
# |---|---|
# | Reads **`knee-mri-cache-v5`** (knee cropped to 140 mm, 256 px) | menisci and lateral OA are limited by image detail, not labels |
# | **Both T4s used**: every training job (one backbone × one fold) runs as its own process on its own GPU | twice the work per GPU-quota hour. `DataParallel` stays banned (it was 20× slower) |
# | The cache is unpacked **once** into an uncompressed file on the local disk; the jobs read it through the operating system's file cache | two processes share one copy in RAM instead of needing two 21 GB copies |
#
# **Jobs.** A job is `{name, fold, + any settings to change}`. Jobs with the same `name` form one **model group** (a
# backbone + settings), saved in its own output folder `<name>/` with `model_fold*.pt` and `train_config.json`. The
# submission treats every such folder as one group.
#
# **Benchmark job.** `{"name": "bench", "bench": [...]}` times a few backbones (forward + backward on one real-sized batch)
# and prints seconds per step and GPU memory, so we can plan longer runs on facts.
#
# **Settings:** Accelerator **GPU T4 ×2**, Internet **on** (pretrained backbones download once). Inputs: `knee-report-labels-v4`
# (`labels_v4.csv`) and `knee-mri-cache-v5`.

# %% [markdown]
# ## 1. Settings
#
# `BASE` holds the defaults for every job; a job overrides any of them. Run lists for the plan:
#
# - **Session 1 (comparison on fold 0):** benchmark + EfficientNet-B0 + ConvNeXt-Nano, each on fold 0.
#   Compare their fold-0 validation AUC with the old 224 px B0 run's fold 0 (same folds, same labels).
# - **Session 2 (full run):** the winner on all five folds: `[dict(name=..., fold=k) for k in range(5)]`.

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os, sys, json, time, math, shutil, subprocess, threading, queue
from pathlib import Path

import numpy as np
import pandas as pd

BASE = dict(
    backbone="efficientnet_b0",
    pretrained=os.environ.get("PRETRAINED", "1") == "1",
    roles=["sag_fs", "cor_fs", "ax_fs"],
    n_triplets=8,              # images per series: 8 groups of 3 neighbouring slices
    epochs=int(os.environ.get("EPOCHS", 10)),
    batch_size=int(os.environ.get("BATCH", 8)),   # studies per step
    lr=3e-4, weight_decay=1e-2, warmup_epochs=1,
    drop_path=0.0,             # stochastic depth inside the backbone (0 = off, as in the baseline)
    grad_ckpt=False,           # trade speed for GPU memory (only needed for large backbones)
    log_every=50,
    seed=42,
)
CNXN = {"backbone": "convnext_nano.in12k_ft_in1k", "lr": 1.5e-4, "weight_decay": 0.05, "drop_path": 0.1}
# Session 1 (done 5 Oct): bench + {"name": "v5_b0", "fold": 0} + {"name": "v5_cnxn", "fold": 0, **CNXN}.
# Session 2: folds 1-4 of both groups; fold 0 is reused from the attached session-1 output (section 6).
JOBS = []
for k in [1, 2, 3, 4]:
    JOBS += [{"name": "v5_cnxn", "fold": k, **CNXN}, {"name": "v5_b0", "fold": k}]
if os.environ.get("JOBS"):                      # local tests override the job list
    JOBS = json.loads(os.environ["JOBS"])
N_FOLDS = 5
TIME_LIMIT_H = 11.0            # don't start a job that would likely end after this (sessions die at 12 h and save nothing)
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None   # debug: use only N pool studies

OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
T0 = time.time()

import torch
N_GPU = torch.cuda.device_count()
print("GPUs:", N_GPU, [torch.cuda.get_device_name(i) for i in range(N_GPU)])
print(json.dumps(BASE, indent=1))
for j in JOBS:
    print("job:", j)

# %% [markdown]
# ## 2. Find the two datasets

# %% [code] {"jupyter":{"outputs_hidden":false}}
SLUG = "rsna-knee-abnormality-detection"

def find(name, base="/kaggle/input", max_depth=6):
    base = Path(os.environ.get("INPUT_BASE", base))
    frontier = [base]
    for _ in range(max_depth):
        nxt = []
        for d in frontier:
            if SLUG in d.name:
                continue
            try:
                for c in d.iterdir():
                    if c.name == name:
                        return c
                    if c.is_dir():
                        nxt.append(c)
            except OSError:
                pass
        frontier = nxt
    raise FileNotFoundError(f"{name} not found under {base}: is the dataset attached?")

LABELS_CSV = find("labels_v4.csv")
CACHE_DIR = find("index.csv").parent
CACHE_CFG = json.loads((CACHE_DIR / "config.json").read_text())
CACHE_ROLES = list(CACHE_CFG["ROLES"])
DEPTH, IMG = CACHE_CFG["DEPTH"], CACHE_CFG["IMG"]
print("labels:", LABELS_CSV)
print("cache :", CACHE_DIR, "|", {k: CACHE_CFG.get(k) for k in ["DEPTH", "IMG", "CROP_MM"]})
assert CACHE_CFG.get("CROP_MM"), "this is the old (uncropped) cache: attach knee-mri-cache-v5"

# %% [markdown]
# ## 3. Labels, gold hold-out and folds
#
# Exactly as in the baseline: report-labelled studies form the training pool, the 58 gold studies are never trained on,
# and the folds are grouped by site proxy (`language|vendor`), so the fold-0 numbers are comparable with the old runs.

# %% [code] {"jupyter":{"outputs_hidden":false}}
from sklearn.model_selection import GroupKFold

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
lab = pd.read_csv(LABELS_CSV)
idx = pd.read_csv(CACHE_DIR / "index.csv")

ok = idx[idx.ok.astype(bool)].pivot_table(index="StudyInstanceUID", columns="role", values="ok", aggfunc="size").reindex(columns=BASE["roles"]).notna()
vendor = idx.sort_values("role").groupby("StudyInstanceUID").manufacturer.first().fillna("?")
df = lab.merge(ok.reset_index(), on="StudyInstanceUID", how="inner")
df["n_roles"] = df[BASE["roles"]].sum(axis=1)
df = df[df.n_roles > 0].copy()
df["vendor"] = df.StudyInstanceUID.map(vendor).fillna("?")
df["site"] = df.language.fillna("?").astype(str) + "|" + df.vendor.astype(str).str.upper().str.split().str[0]
df["n_known"] = df[LABELS].notna().sum(axis=1)

gold_df = df[df.is_gold.astype(bool)].reset_index(drop=True)
pool = df[~df.is_gold.astype(bool) & (df.n_known > 0)].reset_index(drop=True)
if LIMIT:
    pool = pool.sample(min(LIMIT, len(pool)), random_state=0).reset_index(drop=True)
pool["fold"] = -1
for f, (_, va) in enumerate(GroupKFold(n_splits=N_FOLDS).split(pool, groups=pool.site)):
    pool.loc[va, "fold"] = f
print(f"training pool: {len(pool)} | gold hold-out: {len(gold_df)} | site groups: {pool.site.nunique()}")
print(pool.groupby("fold").agg(studies=("StudyInstanceUID", "size"), sites=("site", "nunique")).to_string())

# %% [markdown]
# ## 4. Unpack the cache once, to a plain file on the local disk
#
# The shards are compressed; decompressing during training is too slow. The baseline decompressed everything into RAM, but
# two training processes would each need their own copy (2 × 21 GB > 32 GB). Instead, everything is written once into one
# uncompressed array file (`X.npy`) on the VM's local disk (not `/kaggle/working`, which is the 20 GB output folder). Both
# jobs open it as a memory map: Linux keeps the file in its page cache once and both processes read from that.

# %% [code] {"jupyter":{"outputs_hidden":false}}
import psutil
from concurrent.futures import ThreadPoolExecutor

all_ids = pd.concat([pool.StudyInstanceUID, gold_df.StudyInstanceUID]).drop_duplicates().tolist()
need_gb = len(all_ids) * len(CACHE_ROLES) * DEPTH * IMG * IMG / 1e9
cands = [os.environ.get("WORK_DIR", ""), "/kaggle/temp", "/tmp", str(OUT_DIR.parent / "knee_work")]
WORK = None
for c in [c for c in cands if c]:
    try:
        Path(c).mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(c).free / 1e9
        print(f"  {c}: {free:.0f} GB free")
        if free > need_gb + 3 and WORK is None:
            WORK = Path(c) / "knee_v5"
    except OSError:
        pass
assert WORK is not None, f"no local disk with {need_gb:.0f} GB free"
WORK.mkdir(parents=True, exist_ok=True)
print(f"array: {need_gb:.1f} GB → {WORK} | RAM total {psutil.virtual_memory().total / 1e9:.0f} GB, "
      f"available {psutil.virtual_memory().available / 1e9:.0f} GB")

X_PATH, RMASK_PATH, IDS_PATH = WORK / "X.npy", WORK / "rmask.npy", WORK / "ids.csv"
X = np.lib.format.open_memmap(X_PATH, mode="w+", dtype=np.uint8, shape=(len(all_ids), len(CACHE_ROLES), DEPTH, IMG, IMG))
RMASK = np.zeros((len(all_ids), len(CACHE_ROLES)), np.float32)
ROW = {s: i for i, s in enumerate(all_ids)}
shard_of = idx.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").shard
by_shard = pd.Series(all_ids).groupby(pd.Series(all_ids).map(shard_of)).apply(list)

def load_shard(item):
    shard, sids = item
    with np.load(CACHE_DIR / shard) as z:
        names = set(z.files)
        for s in sids:
            for j, role in enumerate(CACHE_ROLES):
                key = f"{s}__{role}"
                if key in names:
                    X[ROW[s], j] = z[key]; RMASK[ROW[s], j] = 1
    return len(sids)

t0 = time.time(); done_n = 0
with ThreadPoolExecutor(max_workers=4) as ex:
    for n in ex.map(load_shard, by_shard.items()):
        done_n += n
X.flush(); del X
np.save(RMASK_PATH, RMASK)
pd.DataFrame({"StudyInstanceUID": all_ids}).to_csv(IDS_PATH, index=False)
pool.to_csv(WORK / "pool.csv", index=False)
gold_df.to_csv(WORK / "gold.csv", index=False)
print(f"unpacked {done_n} studies in {(time.time() - t0) / 60:.1f} min | series present: "
      f"{dict(zip(CACHE_ROLES, RMASK.mean(0).round(3)))} | RAM available {psutil.virtual_memory().available / 1e9:.0f} GB")

COMMON = WORK / "common.json"
COMMON.write_text(json.dumps({
    "base": BASE, "labels": LABELS, "cache_roles": CACHE_ROLES, "out_dir": str(OUT_DIR),
    "x_path": str(X_PATH), "rmask_path": str(RMASK_PATH), "ids_path": str(IDS_PATH),
    "pool_path": str(WORK / "pool.csv"), "gold_path": str(WORK / "gold.csv"),
}))

# %% [markdown]
# ## 5. The training job, as a script
#
# Everything a job needs (data reading, GPU augmentation, model, training loop, evaluation) is written to
# `knee_train_worker.py` and started once per job. Model, augmentation and slice triplets are the baseline's, unchanged
# (the submission's copy of `KneeNet` stays valid):
#
# - **data:** batches are read from the memory-mapped array by a background thread while the GPU works on the previous one;
# - **augmentation on the GPU:** joint knee mirror (p = 0.5), small affine per series, brightness/contrast/gamma per series;
# - **model:** timm backbone → attention pooling over the 8 triplets of a series → role embedding × role mask → MLP → 12;
# - **training:** AdamW, 1 warm-up epoch + cosine, fp16, `channels_last`, masked soft-label BCE; after every epoch the
#   validation fold is scored, and the best epoch (validation macro AUC) is kept;
# - **at the end:** the best weights predict the validation fold and the gold studies **with mirror TTA** (as the
#   submission does), written as `oof_fold{k}.csv` and `gold_fold{k}.csv`.

# %% [code] {"jupyter":{"outputs_hidden":false}}
WORKER_SRC = r'''
import os, sys, json, time, math, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import timm
from sklearn.metrics import roc_auc_score

job = json.loads(sys.argv[1]); C = json.loads(Path(sys.argv[2]).read_text())
CFG = {**C["base"], **job}
LABELS, ROLES = C["labels"], CFG["roles"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEVICE == "cuda"
torch.backends.cudnn.benchmark = True
MEAN, STD = 0.45, 0.225

def log(*a):
    print(*a, flush=True)

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ---------------------------------------------------------------- model (identical to the baseline / submission)
class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))
    def forward(self, h):
        w = torch.softmax(self.a(h), dim=1)
        return (w * h).sum(1)

class KneeNet(nn.Module):
    def __init__(self, backbone, pretrained, n_roles, n_out=12, drop_path=0.0):
        super().__init__()
        kw = {"drop_path_rate": drop_path} if drop_path else {}
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=3, **kw)
        d = self.enc.num_features
        self.pool = AttnPool(d)
        self.role_emb = nn.Parameter(torch.zeros(n_roles, d))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(n_roles * d + n_roles, 512), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(512, n_out))
    def forward(self, x, rmask):
        B, R, T = x.shape[:3]
        imgs = x.flatten(0, 2)
        if AMP:
            imgs = imgs.contiguous(memory_format=torch.channels_last)
        f = self.enc(imgs)
        f = self.pool(f.view(B * R, T, -1)).view(B, R, -1)
        f = (f + self.role_emb) * rmask.unsqueeze(-1)
        return self.head(torch.cat([f.flatten(1), rmask], dim=1))

def build(backbone, pretrained):
    m = KneeNet(backbone, pretrained, len(ROLES), drop_path=CFG.get("drop_path", 0.0))
    if CFG.get("grad_ckpt"):
        m.enc.set_grad_checkpointing(True)
    m = m.to(DEVICE)
    return m.to(memory_format=torch.channels_last) if AMP else m

# ---------------------------------------------------------------- GPU-side preprocessing
def make_triplets(x, n, train):
    D = x.shape[2]
    centers = torch.linspace(1, D - 2, n, device=x.device).round().long()
    if train:
        centers = (centers + torch.randint(-1, 2, (n,), device=x.device)).clamp(1, D - 2)
    idx = torch.stack([centers - 1, centers, centers + 1], dim=1)
    return x[:, :, idx]

def mirror(x, sel=None):
    """Mirror the whole knee: reverse the sagittal slice order, flip coronal and axial left-right."""
    x = x.clone()
    sel = slice(None) if sel is None else sel
    for j, role in enumerate(ROLES):
        x[sel, j] = x[sel, j].flip(1) if role.startswith("sag") else x[sel, j].flip(3)
    return x

def augment(x):
    B, R, D, H, W = x.shape
    flip = torch.rand(B, device=x.device) < 0.5
    if flip.any():
        x = mirror(x, flip)
    ang = (torch.rand(B * R, device=x.device) - 0.5) * math.radians(20)
    scale = 1 + (torch.rand(B * R, device=x.device) - 0.5) * 0.2
    tx, ty = [(torch.rand(B * R, device=x.device) - 0.5) * 0.1 for _ in range(2)]
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, (B * R, D, H, W), align_corners=False)
    x = F.grid_sample(x.reshape(B * R, D, H, W), grid, align_corners=False, padding_mode="zeros").reshape(B, R, D, H, W)
    c = 1 + (torch.rand(B, R, 1, 1, 1, device=x.device) - 0.5) * 0.3
    b = (torch.rand(B, R, 1, 1, 1, device=x.device) - 0.5) * 0.1
    g = torch.exp((torch.rand(B, R, 1, 1, 1, device=x.device) - 0.5) * 0.4)
    return ((x.clamp(0, 1) ** g) * c + b).clamp(0, 1)

def prepare(x_u8, train, flip=False):
    x = x_u8.to(DEVICE, non_blocking=True).float() / 255.0
    if flip:
        x = mirror(x)
    if train:
        x = augment(x)
    return (make_triplets(x, CFG["n_triplets"], train) - MEAN) / STD

# ---------------------------------------------------------------- benchmark mode
if "bench" in job:
    B, R, T = CFG["batch_size"], len(ROLES), CFG["n_triplets"]
    D, H = 24, 256
    try:
        xs = np.load(C["x_path"], mmap_mode="r")
        D, H = xs.shape[2], xs.shape[3]
    except Exception:
        pass
    log(f"benchmark: batch {B} studies x {R} series x {T} triplets at {H} px | device {DEVICE}")
    rows = []
    for name in job["bench"]:
        for ckpt in [False, True]:
            try:
                torch.cuda.empty_cache() if AMP else None
                torch.cuda.reset_peak_memory_stats() if AMP else None
                m = KneeNet(name, False, R).to(DEVICE)
                if ckpt:
                    m.enc.set_grad_checkpointing(True)
                if AMP:
                    m = m.to(memory_format=torch.channels_last)
                opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
                scaler = torch.amp.GradScaler(enabled=AMP)
                x = torch.randint(0, 255, (B, R, D, H, H), dtype=torch.uint8)
                rm = torch.ones(B, R, device=DEVICE); y = torch.rand(B, 12, device=DEVICE)
                n_steps = 12 if AMP else 2
                for s in range(n_steps + 3):
                    if s == 3:
                        torch.cuda.synchronize() if AMP else None
                        t0 = time.time()
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                        out = m(prepare(x, True), rm)
                    loss = F.binary_cross_entropy_with_logits(out.float(), y)
                    opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                torch.cuda.synchronize() if AMP else None
                sps = (time.time() - t0) / n_steps
                mem = torch.cuda.max_memory_allocated() / 1e9 if AMP else float("nan")
                rows.append(dict(backbone=name, grad_ckpt=ckpt, s_per_step=round(sps, 3), gpu_mem_gb=round(mem, 1),
                                 params_M=round(sum(p.numel() for p in m.parameters()) / 1e6, 1)))
            except torch.cuda.OutOfMemoryError:
                rows.append(dict(backbone=name, grad_ckpt=ckpt, s_per_step=None, gpu_mem_gb="OOM"))
            except Exception as e:
                rows.append(dict(backbone=name, grad_ckpt=ckpt, s_per_step=None, gpu_mem_gb=f"{type(e).__name__}: {str(e)[:80]}"))
            m = opt = None
            log(rows[-1])
    t = pd.DataFrame(rows)
    t["min_per_epoch"] = t.s_per_step.astype(float) * 3480 / CFG["batch_size"] / 60
    log("\n" + t.round(2).to_string())
    sys.exit(0)

# ---------------------------------------------------------------- data
X = np.load(C["x_path"], mmap_mode="r")
RMASK_ALL = np.load(C["rmask_path"])
ROW = {s: i for i, s in enumerate(pd.read_csv(C["ids_path"]).StudyInstanceUID)}
ROLE_IDX = [C["cache_roles"].index(r) for r in ROLES]
pool = pd.read_csv(C["pool_path"]); gold = pd.read_csv(C["gold_path"])

def batches(frame, bs, shuffle, drop_last=False, seed=0):
    rows = frame.StudyInstanceUID.map(ROW).to_numpy()
    y = frame[LABELS].to_numpy(np.float32)
    order = np.random.default_rng(seed).permutation(len(frame)) if shuffle else np.arange(len(frame))
    chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
    if drop_last and chunks and len(chunks[-1]) < bs:
        chunks = chunks[:-1]
    def load(ch):
        r = rows[ch]
        x = np.ascontiguousarray(X[np.sort(r)][:, ROLE_IDX][np.argsort(np.argsort(r))])   # read in file order, restore batch order
        yy = y[ch]
        return (torch.from_numpy(x), torch.from_numpy(RMASK_ALL[r][:, ROLE_IDX]),
                torch.from_numpy(np.nan_to_num(yy)), torch.from_numpy((~np.isnan(yy)).astype(np.float32)))
    if not chunks:
        return
    with ThreadPoolExecutor(1) as ex:
        nxt = ex.submit(load, chunks[0])
        for i in range(len(chunks)):
            cur = nxt.result()
            nxt = ex.submit(load, chunks[i + 1]) if i + 1 < len(chunks) else None
            yield cur

# ---------------------------------------------------------------- metrics
def masked_bce(logits, y, ymask):
    l = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (l * ymask).sum() / ymask.sum().clamp(min=1)

def auc_table(pred, target):
    out = {}
    for k, name in enumerate(LABELS):
        t = target[:, k]; p = pred[:, k]
        keep = ~np.isnan(t) & (t != 0.5)
        yt = (t[keep] > 0.5).astype(int)
        out[name] = roc_auc_score(yt, p[keep]) if len(np.unique(yt)) == 2 else np.nan
    return pd.Series(out)

@torch.no_grad()
def predict(model, frame, tta=False):
    model.eval()
    preds = []
    for x, rmask, _, _ in batches(frame, CFG["batch_size"], shuffle=False):
        rmask = rmask.to(DEVICE)
        views = [False, True] if tta else [False]
        p = 0
        for flip in views:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                p = p + torch.sigmoid(model(prepare(x, False, flip), rmask).float())
        preds.append((p / len(views)).cpu().numpy())
    return np.concatenate(preds)

# ---------------------------------------------------------------- training
fold = int(job["fold"])
out = Path(C["out_dir"]) / CFG["name"]; out.mkdir(parents=True, exist_ok=True)
seed_all(CFG["seed"] + fold)
tr, va = pool[pool.fold != fold], pool[pool.fold == fold]
model = build(CFG["backbone"], CFG["pretrained"])
log(f"{CFG['backbone']}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters | train {len(tr)} | valid {len(va)}")
steps_ep = len(tr) // CFG["batch_size"]
opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
steps = CFG["epochs"] * steps_ep; warm = CFG["warmup_epochs"] * steps_ep
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, warm)) *
                                          0.5 * (1 + math.cos(math.pi * min(1, max(0, s - warm) / max(1, steps - warm)))))
scaler = torch.amp.GradScaler(enabled=AMP)
best, hist, t_job = -1, [], time.time()
for ep in range(CFG["epochs"]):
    model.train(); t0 = time.time(); losses = []; t_wait = 0.0; t_last = time.time()
    for step, (x, rmask, y, ymask) in enumerate(batches(tr, CFG["batch_size"], True, True, seed=CFG["seed"] * 100 + fold * 10 + ep)):
        t_wait += time.time() - t_last
        x = prepare(x, train=True); rmask, y, ymask = rmask.to(DEVICE), y.to(DEVICE), ymask.to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
            logits = model(x, rmask)
        loss = masked_bce(logits.float(), y, ymask)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(opt); scaler.update(); sched.step()
        losses.append(loss.item())
        if (step + 1) % CFG["log_every"] == 0 or step == 0:
            el = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 1e9 if AMP else 0
            log(f"ep {ep + 1} step {step + 1}/{steps_ep} | loss {np.mean(losses[-CFG['log_every']:]):.4f} | "
                f"{el / (step + 1):.2f} s/step | waiting for data {t_wait / el:.0%} | GPU mem {mem:.1f} GB | "
                f"epoch ETA {el / (step + 1) * (steps_ep - step - 1) / 60:.1f} min")
        t_last = time.time()
    aucs = auc_table(predict(model, va), va[LABELS].to_numpy(dtype=float))
    row = {"epoch": ep + 1, "train_loss": float(np.mean(losses)), "val_macro_auc": float(aucs.mean()),
           "minutes": (time.time() - t0) / 60}
    hist.append(row)
    flag = ""
    if aucs.mean() > best:
        best = aucs.mean(); flag = "  <- best, saved"
        torch.save({k: v.half() for k, v in model.state_dict().items()}, out / f"model_fold{fold}.pt")
    log(f"epoch {ep + 1:2d} | loss {row['train_loss']:.4f} | val macro AUC {row['val_macro_auc']:.4f} | {row['minutes']:.1f} min{flag}")
pd.DataFrame(hist).to_csv(out / f"hist_fold{fold}.csv", index=False)

# ---------------------------------------------------------------- final predictions with mirror TTA
sd = torch.load(out / f"model_fold{fold}.pt", map_location=DEVICE)
model.load_state_dict({k: v.float() for k, v in sd.items()})
pv = predict(model, va, tta=True)
pd.DataFrame(pv, columns=LABELS).assign(StudyInstanceUID=va.StudyInstanceUID.values, fold=fold).to_csv(out / f"oof_fold{fold}.csv", index=False)
pg = predict(model, gold, tta=True)
pd.DataFrame(pg, columns=LABELS).assign(StudyInstanceUID=gold.StudyInstanceUID.values).to_csv(out / f"gold_fold{fold}.csv", index=False)
va_auc = auc_table(pv, va[LABELS].to_numpy(dtype=float)).mean()
gold_auc = auc_table(pg, gold[LABELS].to_numpy(dtype=float)).mean()
log(f"DONE {CFG['name']} fold {fold} | val macro AUC (TTA) {va_auc:.4f} | gold macro AUC, this fold alone {gold_auc:.4f} | "
    f"{(time.time() - t_job) / 60:.0f} min")
'''
WORKER = WORK / "knee_train_worker.py"
WORKER.write_text(WORKER_SRC)
print("wrote", WORKER)

# %% [markdown]
# ## 6. Run the jobs, one per GPU at a time
#
# Each GPU takes the next job from the list as soon as it is free. Every log line is prefixed with the GPU and the job.
# A failed job (e.g. out of GPU memory) is reported and the others continue. A job is not started if it would probably
# run past `TIME_LIMIT_H` (estimated from the longest finished job).

# %% [code] {"jupyter":{"outputs_hidden":false}}
def run_jobs(jobs):
    slots = list(range(max(1, N_GPU)))
    pending, running, durations, status = list(jobs), {}, [], []
    lines = queue.Queue()

    def pump(proc, tag):
        for line in proc.stdout:
            lines.put(f"{tag} {line.rstrip()}")

    while pending or running:
        while pending and slots:
            el_h = (time.time() - T0) / 3600
            if durations and el_h + max(durations) / 3600 > TIME_LIMIT_H:
                print(f"time limit: not starting {[j['name'] + '/f' + str(j.get('fold')) for j in pending]}")
                pending.clear(); break
            job, g = pending.pop(0), slots.pop(0)
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            if N_GPU:
                env["CUDA_VISIBLE_DEVICES"] = str(g)
            p = subprocess.Popen([sys.executable, str(WORKER), json.dumps(job), str(COMMON)], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            tag = f"[gpu{g} {job['name']}" + (f" f{job['fold']}]" if "fold" in job else "]")
            threading.Thread(target=pump, args=(p, tag), daemon=True).start()
            running[p] = (job, g, time.time(), tag)
            print(f"{tag} started")
        time.sleep(2)
        while not lines.empty():
            print(lines.get(), flush=True)
        for p in list(running):
            if p.poll() is not None:
                job, g, t, tag = running.pop(p)
                time.sleep(1)
                while not lines.empty():
                    print(lines.get(), flush=True)
                mins = (time.time() - t) / 60
                if "bench" not in job:
                    durations.append(mins * 60)
                status.append({"job": tag, "exit": p.returncode, "minutes": round(mins, 1)})
                print(f"{tag} finished with exit code {p.returncode} after {mins:.1f} min "
                      f"| session {(time.time() - T0) / 3600:.2f} h", flush=True)
                slots.append(g)
    return pd.DataFrame(status)

# Reuse folds finished in an earlier run: if an attached dataset has a folder with the same group name and the same
# training settings, its fold files are copied into this run's output and those folds are not trained again.
SAME_KEYS = ["backbone", "roles", "n_triplets", "epochs", "batch_size", "lr", "weight_decay", "warmup_epochs",
             "drop_path", "seed", "labels_file", "depth", "img", "crop_mm"]

def find_all(name, base=os.environ.get("INPUT_BASE", "/kaggle/input"), max_depth=6):
    hits, frontier = [], [Path(base)]
    for _ in range(max_depth):
        nxt = []
        for d in frontier:
            if SLUG in d.name or d.name == "cache":
                continue
            try:
                for c in d.iterdir():
                    if c.name == name:
                        hits.append(c)
                    elif c.is_dir():
                        nxt.append(c)
            except OSError:
                pass
        frontier = nxt
    return hits

def job_cfg(job):
    return {**BASE, **job, "labels_file": LABELS_CSV.name, "depth": DEPTH, "img": IMG, "crop_mm": CACHE_CFG.get("CROP_MM")}

reused = []
for cfg_path in find_all("train_config.json"):
    old = json.loads(cfg_path.read_text())
    name = old.get("group_name")
    mine = [j for j in JOBS if j.get("name") == name and "fold" in j]
    if not mine:
        continue
    diff = [k for k in SAME_KEYS if old.get(k) != job_cfg(mine[0]).get(k)]
    if diff:
        print(f"not reusing {cfg_path.parent}: settings differ in {diff}"); continue
    planned = {int(j["fold"]) for j in mine}
    for mp in sorted(cfg_path.parent.glob("model_fold*.pt")):
        k = int(mp.stem.replace("model_fold", ""))
        dst = OUT_DIR / name
        if k in planned or (dst / mp.name).exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for f in [mp] + [cfg_path.parent / f"{p}_fold{k}.csv" for p in ["oof", "gold", "hist"]]:
            if f.exists():
                shutil.copy(f, dst / f.name)
        reused.append(f"{name} fold {k}")
        JOBS.append({"name": name, "fold": k, "reused": True})          # so section 7 knows the group
print("reused from attached datasets:", reused or "nothing")

status = run_jobs([j for j in JOBS if not j.get("reused")])
print(status.to_string())

# %% [markdown]
# ## 7. Results per model group
#
# For every group: validation AUC per trained fold (report labels, mirror TTA), and the gold AUC of the group's fold
# ensemble (only the folds trained in this run). With a single fold the gold number is one model, not an ensemble, and
# noisy (58 studies: ± 0.02 macro). **Compare groups on validation fold AUC first**, gold second.
#
# Every group folder gets `train_config.json`, `oof_predictions.csv` and `gold_predictions.csv`, ready to be used as a
# model dataset by the submission.

# %% [code] {"jupyter":{"outputs_hidden":false}}
from sklearn.metrics import roc_auc_score

def auc_table(pred, target):
    out = {}
    for k, name in enumerate(LABELS):
        t = target[:, k]; p = pred[:, k]
        keep = ~np.isnan(t) & (t != 0.5)
        yt = (t[keep] > 0.5).astype(int)
        out[name] = roc_auc_score(yt, p[keep]) if len(np.unique(yt)) == 2 else np.nan
    return pd.Series(out)

pool_y = pool.set_index("StudyInstanceUID")[LABELS]
gold_y = gold_df[LABELS].to_numpy(dtype=float)
compare, gold_by_group = {}, {}
for name in dict.fromkeys(j["name"] for j in JOBS if "bench" not in j):
    d = OUT_DIR / name
    oof_files = sorted(d.glob("oof_fold*.csv"))
    if not oof_files:
        print(f"{name}: no finished folds"); continue
    oof = pd.concat([pd.read_csv(f) for f in oof_files], ignore_index=True)
    cols = {}
    for f in sorted(oof.fold.unique()):
        o = oof[oof.fold == f]
        cols[f"val fold {f}"] = auc_table(o[LABELS].to_numpy(), pool_y.loc[o.StudyInstanceUID].to_numpy(dtype=float))
    gp = [pd.read_csv(f).set_index("StudyInstanceUID").loc[gold_df.StudyInstanceUID, LABELS].to_numpy()
          for f in sorted(d.glob("gold_fold*.csv"))]
    gens = np.mean(gp, axis=0)
    gold_by_group[name] = gens
    cols["GOLD (folds above)"] = auc_table(gens, gold_y)
    table = pd.DataFrame(cols); table.loc["macro"] = table.mean()
    print(f"\n=== {name} ===\n" + table.round(3).to_string())
    compare[name] = table.loc["macro"]

    folds = sorted(int(f) for f in oof.fold.unique())
    job0 = next(j for j in JOBS if j["name"] == name)
    cfg = {**BASE, **{k: v for k, v in job0.items() if k != "fold"}}
    cfg.update(group_name=name, labels=LABELS, labels_file=LABELS_CSV.name, depth=DEPTH, img=IMG,
               crop_mm=CACHE_CFG.get("CROP_MM"), mean=0.45, std=0.225, channels_last=N_GPU > 0,
               trained_folds=folds, gold_macro_auc=float(table.loc["macro", "GOLD (folds above)"]))
    (d / "train_config.json").write_text(json.dumps(cfg, indent=1))
    oof.to_csv(d / "oof_predictions.csv", index=False)
    pd.DataFrame(gens, columns=LABELS).assign(StudyInstanceUID=gold_df.StudyInstanceUID.values).to_csv(d / "gold_predictions.csv", index=False)

if compare:
    print("\n=== macro AUC by group ===\n" + pd.DataFrame(compare).round(4).to_string())
if len(gold_by_group) > 1:          # equal rank blend of all groups on gold
    ranks = [pd.DataFrame(g).rank(pct=True).to_numpy() for g in gold_by_group.values()]
    print(f"\ngold macro AUC, equal rank blend of {list(gold_by_group)}: {auc_table(np.mean(ranks, 0), gold_y).mean():.4f}")

# %% [markdown]
# ## 8. Output
#
# **Save Version → Save & Run All**, then on the finished version: **Output → New Dataset** (e.g. `knee-models-v5`).
# Every group folder is one model group for the submission. The unpacked cache lives outside `/kaggle/working` and is not
# saved.

# %% [code] {"jupyter":{"outputs_hidden":false}}
for p in sorted(OUT_DIR.rglob("*")):
    if p.is_file():
        print(f"{p.relative_to(OUT_DIR)}  {p.stat().st_size / 1e6:.1f} MB")
print(f"session time {(time.time() - T0) / 3600:.2f} h")
