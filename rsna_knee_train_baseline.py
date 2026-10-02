# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # RSNA Knee: baseline training (images → 12 findings)
# #
# This notebook trains the model that will eventually be submitted. It brings together the two datasets we built:
# #
# | Input dataset | Gives | Role |
# |---|---|---|
# | `knee-report-labels` → `labels_v4.csv` | 12 soft labels per study (0 / 0.5 / 1, NaN where unknown) | **y** |
# | `knee-mri-cache` → shards + `knee_preproc.py` | 3 preprocessed series per study, 24 × 224 × 224, uint8 | **X** |
# #
# **The model in one picture (2.5D, multi-view):**
# ```
#  sag_fs (24 slices) ─► 8 "RGB" images of 3 neighbouring slices ─┐
#  cor_fs (24 slices) ─► 8 images ────────────────────────────────┼─► shared 2D CNN (ImageNet-pretrained)
#  ax_fs  (24 slices) ─► 8 images ────────────────────────────────┘        │ one feature vector per image
#                                                                          ▼
#                                     attention pooling over the 8 images of each series
#                                                                          ▼
#                          3 series vectors (missing series → masked) → concatenate → MLP → 12 logits
# ```
# #
# **Validation:** the 58 **gold** studies are never trained on. They are our cleanest estimate of the leaderboard score,
# because the test labels are presumably made the same way as gold. Besides that, a 5-fold split of the report-labelled
# studies, grouped by a site proxy, is used for model selection.
# #
# **Settings:** Accelerator **GPU T4 ×2**, Internet **on** (downloads the pretrained backbone once). One fold takes roughly
# 40–60 min; the notebook prints the measured time per epoch after the first one.

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 1. Settings
# #
# Start with `FOLDS_TO_RUN = [0]`: one fold is enough to see whether everything works and how good the baseline is. When
# it is, run the remaining folds (possibly in separate sessions: every fold saves its own weights).

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os, sys, json, time, math, glob, random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CFG = dict(
    backbone=os.environ.get("BACKBONE", "efficientnet_b0"),   # any timm model name
    pretrained=os.environ.get("PRETRAINED", "1") == "1",
    roles=["sag_fs", "cor_fs", "ax_fs"],
    n_triplets=8,              # images per series: 8 groups of 3 neighbouring slices
    img=224,
    epochs=int(os.environ.get("EPOCHS", 10)),
    batch_size=8,              # studies per step (each = 3 series × 8 images)
    lr=3e-4, weight_decay=1e-2, warmup_epochs=1,
    n_folds=5,
    folds_to_run=[0, 1, 2, 3, 4],
    num_workers=4,
    preload=True,              # load the whole cache into RAM once (fast training); falls back to disk if RAM is short
    log_every=50,              # print progress every N training steps
    seed=42,
    limit=int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None,   # debug: use only N studies
)
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

import torch, torch.nn as nn, torch.nn.functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True        # fixed input sizes → let cuDNN pick the fastest kernels
N_GPU = torch.cuda.device_count()

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(CFG["seed"])
print(json.dumps(CFG, indent=1), "\ndevice:", DEVICE, "| GPUs:", N_GPU)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 2. Find the two datasets
# #
# Kaggle mounts attached datasets somewhere under `/kaggle/input`, and the exact path depends on your username and the
# dataset name. Instead of hard-coding it, we search (shallowly, skipping the huge competition folder) for the two files that
# identify them: `labels_v3.csv` and `cache/index.csv`.

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
sys.path.insert(0, str(CACHE_DIR.parent))          # knee_preproc.py sits next to the cache folder
sys.dont_write_bytecode = True
import knee_preproc as kp
print("labels:", LABELS_CSV)
print("cache :", CACHE_DIR, "| module:", kp.__file__)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 3. Put labels and images together
# #
# - **Training pool:** report-labelled studies that have at least one cached series and at least one known label.
#   Unknown labels (NaN, from parse errors) are **masked** in the loss, so a study with 11 of 12 labels still counts.
# - **Gold hold-out:** the 58 gold studies, used only for evaluation.
# - **Site proxy for grouping:** report language + scanner vendor. Studies from one site then fall into the same fold,
#   so validation measures how well the model generalises to *another* hospital, which is what the test set demands.

# %% [code] {"jupyter":{"outputs_hidden":false}}
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
lab = pd.read_csv(LABELS_CSV)
idx = pd.read_csv(CACHE_DIR / "index.csv")

ok = idx[idx.ok.astype(bool)].pivot_table(index="StudyInstanceUID", columns="role", values="ok", aggfunc="size").reindex(columns=CFG["roles"]).notna()
vendor = idx.sort_values("role").groupby("StudyInstanceUID").manufacturer.first().fillna("?")

df = lab.merge(ok.reset_index(), on="StudyInstanceUID", how="inner")
df["n_roles"] = df[CFG["roles"]].sum(axis=1)
df = df[df.n_roles > 0].copy()
df["vendor"] = df.StudyInstanceUID.map(vendor).fillna("?")
df["site"] = df.language.fillna("?").astype(str) + "|" + df.vendor.astype(str).str.upper().str.split().str[0]
df["n_known"] = df[LABELS].notna().sum(axis=1)

gold_df = df[df.is_gold.astype(bool)].reset_index(drop=True)
pool = df[~df.is_gold.astype(bool) & (df.n_known > 0)].reset_index(drop=True)
if CFG["limit"]:
    pool = pool.sample(min(CFG["limit"], len(pool)), random_state=0).reset_index(drop=True)

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
pool["fold"] = -1
if pool.site.nunique() >= CFG["n_folds"]:
    splits = GroupKFold(n_splits=CFG["n_folds"]).split(pool, groups=pool.site)
else:   # too few site groups for grouping: plain random folds
    print("warning: fewer site groups than folds, using random folds")
    splits = KFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"]).split(pool)
for f, (_, va) in enumerate(splits):
    pool.loc[va, "fold"] = f

print(f"training pool: {len(pool)} | gold hold-out: {len(gold_df)} | site groups: {pool.site.nunique()}")
print("series available:", {r: f"{pool[r].mean():.1%}" for r in CFG["roles"]})
fold_table = pool.groupby("fold").agg(studies=("StudyInstanceUID", "size"), sites=("site", "nunique"),
                                      effusion_pos=("Effusion", lambda s: (s > 0.5).mean())).round(3)
display(fold_table)
print(fold_table.to_string())          # commit logs do not show display() tables

# %% [code] {"jupyter":{"outputs_hidden":false}}
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
pos = pd.DataFrame({"train pool (soft label > 0.5)": (pool[LABELS] > 0.5).mean(),
                    "gold": (gold_df[LABELS] > 0.5).mean()})
pos.plot.barh(ax=axes[0], color=["#e59866", "#1f4e79"]); axes[0].invert_yaxis(); axes[0].set_title("share of positives")
pool.site.value_counts().head(20).plot.bar(ax=axes[1], color="#4a7ab5"); axes[1].set_title("largest site groups (language|vendor)")
axes[1].tick_params(axis="x", rotation=60, labelsize=8)
plt.tight_layout(); plt.show()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Dataset: from cache to tensors
# #
# Each item is one study: a `uint8` tensor `(3 roles, 24 slices, 224, 224)`, a role mask (which series exist), the 12
# targets and the 12-label mask. Everything else (choosing slice triplets, augmentation, normalisation) happens on the GPU
# in section 5, which is much faster than doing it per item on the CPU.

# %% [code] {"jupyter":{"outputs_hidden":false}}
from torch.utils.data import Dataset, DataLoader

class KneeDS(Dataset):
    def __init__(self, frame):
        self.f = frame.reset_index(drop=True)
        self.reader, self.pid = None, None                   # opened lazily, once per process
    def __len__(self):
        return len(self.f)
    def __getitem__(self, i):
        if RAM is None and (self.reader is None or self.pid != os.getpid()):   # a file handle shared across processes corrupts reads
            self.reader, self.pid = kp.CacheReader(CACHE_DIR), os.getpid()
        r = self.f.iloc[i]
        if RAM is not None:                                  # preloaded: just index the big array
            x, rmask = RAM[ROW[r.StudyInstanceUID]], RMASK[ROW[r.StudyInstanceUID]]
        else:                                                # from disk: decompress from the shard
            arrs = self.reader.load(r.StudyInstanceUID)
            x = np.zeros((len(CFG["roles"]), kp_depth, CFG["img"], CFG["img"]), np.uint8)
            rmask = np.zeros(len(CFG["roles"]), np.float32)
            for j, role in enumerate(CFG["roles"]):
                a = arrs.get(role)
                if a is not None:
                    x[j] = a; rmask[j] = 1
        y = r[LABELS].to_numpy(dtype=np.float32)
        ymask = (~np.isnan(y)).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(rmask), torch.from_numpy(np.nan_to_num(y)), torch.from_numpy(ymask)

kp_depth = json.loads((CACHE_DIR / "config.json").read_text())["DEPTH"]
RAM, ROW, RMASK = None, None, None
t0 = time.time(); item = KneeDS(pool.head(4))[0]
print("one item from disk:", [tuple(t.shape) for t in item], f"| {1000 * (time.time() - t0):.0f} ms to load")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### Load the whole cache into RAM
# #
# The first version read every study from the compressed shards on disk during training. That costs ~140 ms per study,
# and with ~3,400 studies per epoch the GPUs spent most of their time waiting for data. Instead, all studies are now
# decompressed **once** into one big array in memory (≈ 16 GB for ~4,400 studies; Kaggle GPU sessions have ~29 GB),
# using 4 threads. After that, fetching a study is a simple array lookup. If there is not enough free RAM, the notebook
# says so and falls back to reading from disk.

# %% [code] {"jupyter":{"outputs_hidden":false}}
import psutil
from concurrent.futures import ThreadPoolExecutor

all_ids = pd.concat([pool.StudyInstanceUID, gold_df.StudyInstanceUID]).drop_duplicates().tolist()
R = len(CFG["roles"])
need_gb = len(all_ids) * R * kp_depth * CFG["img"] ** 2 / 1e9
avail_gb = psutil.virtual_memory().available / 1e9
print(f"RAM needed {need_gb:.1f} GB | available {avail_gb:.1f} GB")

if CFG["preload"] and need_gb < avail_gb - 5:
    ROW = {s: i for i, s in enumerate(all_ids)}
    RAM = np.zeros((len(all_ids), R, kp_depth, CFG["img"], CFG["img"]), np.uint8)
    RMASK = np.zeros((len(all_ids), R), np.float32)
    shard_of = idx.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").shard
    by_shard = pd.Series(all_ids).groupby(pd.Series(all_ids).map(shard_of)).apply(list)

    def load_shard(item):
        shard, sids = item
        with np.load(CACHE_DIR / shard) as z:                # each thread opens its own handle
            names = set(z.files)
            for s in sids:
                for j, role in enumerate(CFG["roles"]):
                    key = f"{s}__{role}"
                    if key in names:
                        RAM[ROW[s], j] = z[key]; RMASK[ROW[s], j] = 1
        return len(sids)

    t0 = time.time(); done_n = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for n in ex.map(load_shard, by_shard.items()):
            done_n += n
    print(f"preloaded {done_n} studies in {(time.time() - t0) / 60:.1f} min | RAM now {psutil.virtual_memory().percent:.0f}% used")
    CFG["num_workers"] = 0                                   # lookups take ~2 ms: no worker processes needed (and no
                                                             # risk of extra copies of the 16 GB array)
else:
    print("not preloading: training reads from disk (slower)")

t0 = time.time(); item = KneeDS(pool.head(4))[0]
print("one item now:", f"{1000 * (time.time() - t0):.1f} ms to load")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 5. GPU-side preprocessing and augmentation
# #
# **Slice triplets.** From the 24 slices, 8 centre positions are spread evenly; each image is the centre slice with its two
# neighbours as the 3 colour channels. The CNN was pretrained on RGB photos, and three neighbouring slices give it a bit of
# 3D context for free. During training the centres jitter by ±1 slice.
# #
# **Augmentations (training only):**
# - small random rotation / zoom / shift, the same for all slices of a series;
# - brightness, contrast and gamma jitter per series (scanners differ);
# - **mirror the whole knee** with probability 0.5: flip coronal and axial left–right and reverse the sagittal slice order.
#   That turns a right knee into a (valid) left knee with the same labels. Flipping only one series would create an
#   impossible knee, so all three are flipped together.

# %% [code] {"jupyter":{"outputs_hidden":false}}
MEAN, STD = 0.45, 0.225

def make_triplets(x, n, train):
    # x: (B, R, D, H, W) float → (B, R, n, 3, H, W)
    D = x.shape[2]
    centers = torch.linspace(1, D - 2, n, device=x.device).round().long()
    if train:
        centers = (centers + torch.randint(-1, 2, (n,), device=x.device)).clamp(1, D - 2)
    idx = torch.stack([centers - 1, centers, centers + 1], dim=1)          # (n, 3)
    return x[:, :, idx]                                                    # (B, R, n, 3, H, W)

def augment(x):
    # x: (B, R, D, H, W) in [0, 1]
    B, R, D, H, W = x.shape
    # joint mirror of the knee
    flip = torch.rand(B, device=x.device) < 0.5
    if flip.any():
        roles = CFG["roles"]
        for j, role in enumerate(roles):
            if role.startswith("sag"):
                x[flip, j] = x[flip, j].flip(1)          # reverse slice order (right↔left)
            else:
                x[flip, j] = x[flip, j].flip(3)          # left-right flip in-plane
    # affine per series
    ang = (torch.rand(B * R, device=x.device) - 0.5) * math.radians(20)
    scale = 1 + (torch.rand(B * R, device=x.device) - 0.5) * 0.2
    tx, ty = [(torch.rand(B * R, device=x.device) - 0.5) * 0.1 for _ in range(2)]
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, (B * R, D, H, W), align_corners=False)
    x = F.grid_sample(x.reshape(B * R, D, H, W), grid, align_corners=False, padding_mode="zeros").reshape(B, R, D, H, W)
    # intensity per series
    c = 1 + (torch.rand(B, R, 1, 1, 1, device=x.device) - 0.5) * 0.3
    b = (torch.rand(B, R, 1, 1, 1, device=x.device) - 0.5) * 0.1
    g = torch.exp((torch.rand(B, R, 1, 1, 1, device=x.device) - 0.5) * 0.4)
    return ((x.clamp(0, 1) ** g) * c + b).clamp(0, 1)

CHANNELS_LAST = False   # switched on in section 8 (GPU only)

def prepare(x_u8, train):
    x = x_u8.to(DEVICE, non_blocking=True).float() / 255.0
    if train:
        x = augment(x)
    x = make_triplets(x, CFG["n_triplets"], train)
    return (x - MEAN) / STD

# quick visual check of the augmentation on one study
xb = torch.stack([KneeDS(pool.head(1))[0][0]])
fig, axes = plt.subplots(2, len(CFG["roles"]), figsize=(4 * len(CFG["roles"]), 7.5))
for row, train in enumerate([False, True]):
    x = prepare(xb, train)[0]
    for j, role in enumerate(CFG["roles"]):
        axes[row, j].imshow((x[j, CFG["n_triplets"] // 2, 1] * STD + MEAN).cpu(), cmap="gray", vmin=0, vmax=1)
        axes[row, j].set_title(f"{role} {'augmented' if train else 'plain'}"); axes[row, j].axis("off")
plt.tight_layout(); plt.show()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 6. The model
# #
# - **Backbone:** a timm CNN (`efficientnet_b0` by default: small and fast on T4), shared by all series and slices.
#   It turns each 3-slice image into one feature vector.
# - **Attention pooling:** learns which of the 8 images of a series matter (e.g. the ones through the ACL) and averages
#   them with those weights.
# - **Role embedding + mask:** each series gets a learned "which view am I" vector; missing series are zeroed out and
#   the mask count is fed to the head, so the model knows what it didn't see.
# - **Head:** concatenated series features → MLP → 12 logits (one per finding; sigmoid gives the probability).

# %% [code] {"jupyter":{"outputs_hidden":false}}
import timm

class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))
    def forward(self, h):                                  # h: (N, T, d)
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
    def forward(self, x, rmask):                           # x: (B, R, T, 3, H, W), rmask: (B, R)
        B, R, T = x.shape[:3]
        imgs = x.flatten(0, 2)                             # (B*R*T, 3, H, W)
        if CHANNELS_LAST:
            imgs = imgs.contiguous(memory_format=torch.channels_last)
        f = self.enc(imgs)                                 # (B*R*T, d)
        f = self.pool(f.view(B * R, T, -1)).view(B, R, -1)
        f = (f + self.role_emb) * rmask.unsqueeze(-1)
        return self.head(torch.cat([f.flatten(1), rmask], dim=1))

m = KneeNet(CFG["backbone"], CFG["pretrained"], len(CFG["roles"]))
print(f"{CFG['backbone']}: {sum(p.numel() for p in m.parameters()) / 1e6:.1f} M parameters")
del m

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 7. Loss and metrics
# #
# - **Loss:** binary cross-entropy against the **soft** labels, averaged only over known labels (`ymask`). A soft label of
#   0.5 ("uncertain") teaches the model to be unsure there, instead of forcing a guess.
# - **Validation AUC (report labels):** labels > 0.5 count as positive, < 0.5 as negative; 0.5 and NaN are skipped.
# - **Gold AUC:** the same metric against the 58 gold studies, the most honest number we have. With so few studies,
#   individual findings can swing a lot; the macro average is more stable.

# %% [code] {"jupyter":{"outputs_hidden":false}}
from sklearn.metrics import roc_auc_score

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
def predict(model, frame):
    model.eval()
    dl = DataLoader(KneeDS(frame), batch_size=CFG["batch_size"], shuffle=False, num_workers=CFG["num_workers"])
    preds = []
    for x, rmask, _, _ in dl:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            logits = model(prepare(x, train=False), rmask.to(DEVICE))
        preds.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(preds)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 8. One GPU, channels_last (what the speed diagnostic showed)
# #
# The diagnostic notebook measured, on a T4:
# #
# | Setup | Result |
# |---|---|
# | 1 GPU, 8 studies per step, `channels_last`, fp16 | **0.39 s/step** (healthy) |
# | 2 GPUs with `DataParallel` | **7.5 s/step**, and with `channels_last` it **crashes** (`CUDA error: misaligned address`) |
# #
# So `DataParallel` was the problem all along: it made every step ~20× slower in this PyTorch version, and it is the common
# factor of all the runs whose kernel died. Training now uses **one GPU** with `channels_last`. Expected: ~3 min per epoch,
# ~30–35 min per fold. (The second T4 stays idle; using it properly would mean training two folds in parallel processes,
# a possible later speed-up.)

# %% [code] {"jupyter":{"outputs_hidden":false}}
CHANNELS_LAST = DEVICE == "cuda"
USE_DP = False

def build_net(use_dp=False):
    model = KneeNet(CFG["backbone"], CFG["pretrained"], len(CFG["roles"])).to(DEVICE)
    if CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    return model, model

steps_per_epoch = (len(pool) * (CFG["n_folds"] - 1) // CFG["n_folds"]) // CFG["batch_size"]
print(f"{steps_per_epoch} steps per epoch → ≈ {steps_per_epoch * 0.4 / 60:.0f} min per epoch at the measured 0.4 s/step")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 9. Training loop
# #
# Per fold: AdamW, one warm-up epoch, then cosine decay; mixed precision (fp16) for speed on T4; one GPU (see section 8).
# After every epoch: validation loss and AUC on the held-out fold. The weights with the best validation
# macro AUC are saved as `model_fold{k}.pt`.

# %% [code] {"jupyter":{"outputs_hidden":false}}
def train_fold(fold):
    tr, va = pool[pool.fold != fold], pool[pool.fold == fold]
    dl = DataLoader(KneeDS(tr), batch_size=CFG["batch_size"], shuffle=True, drop_last=True,
                    num_workers=CFG["num_workers"], pin_memory=CFG["num_workers"] > 0, persistent_workers=CFG["num_workers"] > 0)
    model, net = build_net(USE_DP)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    steps = CFG["epochs"] * len(dl); warm = CFG["warmup_epochs"] * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, warm)) *
                                              0.5 * (1 + math.cos(math.pi * min(1, max(0, s - warm) / max(1, steps - warm)))))
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    best, hist = -1, []
    print(f"\n===== fold {fold}: train {len(tr)} | valid {len(va)} =====")
    for ep in range(CFG["epochs"]):
        net.train(); t0 = time.time(); losses = []; t_wait = 0.0; t_last = time.time()
        for step, (x, rmask, y, ymask) in enumerate(dl):
            t_wait += time.time() - t_last                     # time spent waiting for the data loader
            x = prepare(x, train=True); rmask, y, ymask = rmask.to(DEVICE), y.to(DEVICE), ymask.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                logits = net(x, rmask)
            loss = masked_bce(logits.float(), y, ymask)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item())
            if (step + 1) % CFG["log_every"] == 0 or step == 0:
                el = time.time() - t0
                print(f"  ep {ep + 1} step {step + 1}/{len(dl)} | loss {np.mean(losses[-CFG['log_every']:]):.4f} | "
                      f"{el / (step + 1):.2f} s/step | waiting for data {t_wait / el:.0%} | "
                      f"epoch ETA {el / (step + 1) * (len(dl) - step - 1) / 60:.1f} min | RAM {psutil.virtual_memory().percent:.0f}%", flush=True)
            t_last = time.time()
        p = predict(net, va)
        aucs = auc_table(p, va[LABELS].to_numpy(dtype=float))
        row = {"epoch": ep + 1, "train_loss": np.mean(losses), "val_macro_auc": aucs.mean(), "minutes": (time.time() - t0) / 60}
        hist.append(row)
        flag = ""
        if aucs.mean() > best:
            best = aucs.mean(); flag = "  ← best, saved"
            torch.save({k: v.half() for k, v in model.state_dict().items()}, OUT_DIR / f"model_fold{fold}.pt")
        print(f"epoch {ep + 1:2d} | loss {row['train_loss']:.4f} | val macro AUC {row['val_macro_auc']:.4f} | "
              f"{row['minutes']:.1f} min{flag}", flush=True)
    return pd.DataFrame(hist)

histories = {f: train_fold(f) for f in CFG["folds_to_run"]}

# %% [code] {"jupyter":{"outputs_hidden":false}}
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for f, h in histories.items():
    axes[0].plot(h.epoch, h.train_loss, marker="o", label=f"fold {f}")
    axes[1].plot(h.epoch, h.val_macro_auc, marker="o", label=f"fold {f}")
axes[0].set_title("training loss"); axes[1].set_title("validation macro AUC (report labels)")
for a in axes: a.set_xlabel("epoch"); a.legend()
plt.tight_layout(); plt.show()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 10. Evaluation: validation fold and gold hold-out
# #
# For every trained fold we reload the best weights and predict (a) its validation fold and (b) the 58 gold studies.
# The gold predictions of all trained folds are averaged (an ensemble), which is exactly what the submission will do.
# #
# **How to read it:** the gold macro AUC is the number to watch between experiments. Validation AUC on report labels is
# useful too (much more data), but it partly measures agreement with the labeller's mistakes.

# %% [code] {"jupyter":{"outputs_hidden":false}}
def load_model(fold):
    model = KneeNet(CFG["backbone"], False, len(CFG["roles"])).to(DEVICE)
    sd = torch.load(OUT_DIR / f"model_fold{fold}.pt", map_location=DEVICE)
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    return model.to(memory_format=torch.channels_last) if CHANNELS_LAST else model

val_rows, gold_preds, oof = [], [], []
for f in CFG["folds_to_run"]:
    model = load_model(f)
    va = pool[pool.fold == f]
    pv = predict(model, va)
    oof.append(pd.DataFrame(pv, columns=LABELS).assign(StudyInstanceUID=va.StudyInstanceUID.values, fold=f))
    val_rows.append(auc_table(pv, va[LABELS].to_numpy(dtype=float)).rename(f"val fold {f}"))
    gold_preds.append(predict(model, gold_df))

gold_ens = np.mean(gold_preds, axis=0)
table = pd.concat(val_rows + [auc_table(gold_ens, gold_df[LABELS].to_numpy(dtype=float)).rename("GOLD (ensemble)")], axis=1)
table.loc["macro"] = table.mean()
display(table.round(3))
print(table.round(3).to_string())          # commit logs do not show display() tables

fig, ax = plt.subplots(figsize=(10, 4.5))
table.drop("macro").plot.bar(ax=ax, width=0.8)
ax.axhline(0.5, color="grey", ls=":"); ax.set_ylim(0.4, 1); ax.set_title("AUC per finding")
ax.tick_params(axis="x", rotation=45); plt.tight_layout(); plt.show()

pd.concat(oof).to_csv(OUT_DIR / "oof_predictions.csv", index=False)
pd.DataFrame(gold_ens, columns=LABELS).assign(StudyInstanceUID=gold_df.StudyInstanceUID.values).to_csv(OUT_DIR / "gold_predictions.csv", index=False)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 11. Save everything the submission needs
# #
# The submission notebook runs offline, so it gets the model from this notebook's output (saved as a dataset, e.g.
# `knee-models`):
# - `model_fold*.pt`: weights (fp16, a few MB each for efficientnet_b0),
# - `train_config.json`: backbone, roles, triplets, image size. The submission rebuilds the exact same model from it,
#   with `pretrained=False` (no download needed, the weights come from the `.pt` file).
# #
# **Next:** save this version's output as a dataset → build the real submission notebook, which runs `knee_preproc` on the
# hidden test DICOMs, feeds the model, averages the folds and writes `submission.csv`.

# %% [code] {"jupyter":{"outputs_hidden":false}}
cfg_out = {**CFG, "channels_last": CHANNELS_LAST, "labels": LABELS, "depth": kp_depth, "mean": MEAN, "std": STD,
           "labels_file": LABELS_CSV.name, "trained_folds": CFG["folds_to_run"],
           "gold_macro_auc": float(table.loc["macro", "GOLD (ensemble)"])}
(OUT_DIR / "train_config.json").write_text(json.dumps(cfg_out, indent=1))
print(json.dumps({k: cfg_out[k] for k in ["backbone", "roles", "trained_folds", "gold_macro_auc"]}, indent=1))
print("files:", sorted(p.name for p in OUT_DIR.iterdir()))