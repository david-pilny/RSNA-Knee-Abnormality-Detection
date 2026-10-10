# %% [markdown]
# # RSNA Knee: speed and memory diagnostic (run interactively, ~5 min)
#
# Training is ~20× slower than expected and the kernel dies. This notebook finds out **where** the time and memory go,
# step by step, on a small amount of data. Run it as an **interactive session** (not Save & Run All), so you see each
# result immediately, and if the kernel dies, Kaggle shows the reason (for example "out of memory").
#
# Settings: GPU T4 ×2, Internet on, inputs `knee-mri-cache` (the labels are not needed here).

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os, sys, time, json, glob
from pathlib import Path
import numpy as np, pandas as pd, psutil
import torch, torch.nn as nn, torch.nn.functional as F
import timm

proc = psutil.Process()
def mem(tag=""):
    gpu = " | ".join(f"GPU{i} {torch.cuda.memory_allocated(i) / 1e9:.1f}/{torch.cuda.max_memory_allocated(i) / 1e9:.1f} GB"
                     for i in range(torch.cuda.device_count()))
    print(f"[{tag}] host RSS {proc.memory_info().rss / 1e9:.1f} GB, system RAM {psutil.virtual_memory().percent:.0f}% | {gpu}")

print("torch", torch.__version__, "| timm", timm.__version__, "| CUDA", torch.version.cuda, "| cuDNN", torch.backends.cudnn.version())
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("CPU cores:", os.cpu_count(), "| RAM total:", round(psutil.virtual_memory().total / 1e9, 1), "GB")
mem("start")

# %% [markdown]
# ## 1. Pure backbone speed (no data, no pipeline)
#
# Random images straight into EfficientNet-B0, forward + backward, fp16, on **one** GPU. This is the speed limit of the
# hardware. 96 images = one GPU's share of our batch (8 studies × 3 series × 8 images ÷ 2 GPUs).

# %% [code] {"jupyter":{"outputs_hidden":false}}
def bench_backbone(n_img, channels_last, amp=True, steps=5):
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0).cuda()
    if channels_last:
        m = m.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(m.parameters(), 1e-4); scaler = torch.amp.GradScaler()
    x = torch.randn(n_img, 3, 224, 224, device="cuda")
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 1):
        if i == 1:
            torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = m(x).float().mean()
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize(); dt = (time.time() - t0) / steps
    peak = torch.cuda.max_memory_allocated() / 1e9
    del m, opt, x; torch.cuda.empty_cache()
    return dt, n_img / dt, peak

rows = []
for n, cl, amp in [(32, False, True), (96, False, True), (96, True, True), (96, False, False)]:
    dt, ips, peak = bench_backbone(n, cl, amp)
    rows.append({"images": n, "channels_last": cl, "fp16": amp, "s/step": round(dt, 3), "images/s": round(ips), "GPU peak GB": round(peak, 1)})
    print(rows[-1], flush=True)
display(pd.DataFrame(rows))
mem("after backbone bench")

# %% [markdown]
# **How to read it:** a healthy T4 does a few hundred images/s here. If this table is already slow, the problem is the
# environment (GPU, drivers, settings), not our code. If it is fast, the problem is in our pipeline; the next cells find it.

# %% [markdown]
# ## 2. Our pipeline on real cached studies, piece by piece

# %% [code] {"jupyter":{"outputs_hidden":false}}
def find(name, base="/kaggle/input", max_depth=6):
    frontier = [Path(base)]
    for _ in range(max_depth):
        nxt = []
        for d in frontier:
            if "rsna-knee-abnormality-detection" in d.name:
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
CACHE_DIR = find("index.csv").parent
sys.path.insert(0, str(CACHE_DIR.parent)); sys.dont_write_bytecode = True
import knee_preproc as kp
reader = kp.CacheReader(CACHE_DIR)
ROLES = ["sag_fs", "cor_fs", "ax_fs"]
okc = reader.index[reader.index.ok.astype(bool)].groupby("StudyInstanceUID").role.nunique()
ids = okc[okc == len(ROLES)].index[:16]          # studies that have all three series
t0 = time.time()
X = np.stack([np.stack([reader.load(s)[r] for r in ROLES]) for s in ids])      # (16, 3, 24, 224, 224) uint8
print(f"loaded {len(ids)} studies from disk in {time.time() - t0:.1f} s → {X.shape}, {X.nbytes / 1e6:.0f} MB")
mem("after loading 16 studies")

# %% [code] {"jupyter":{"outputs_hidden":false}}
MEAN, STD, NT = 0.45, 0.225, 8

def make_triplets(x, n):
    D = x.shape[2]
    c = torch.linspace(1, D - 2, n, device=x.device).round().long()
    return x[:, :, torch.stack([c - 1, c, c + 1], 1)]

def augment(x):
    B, R, D, H, W = x.shape
    ang = (torch.rand(B * R, device=x.device) - 0.5) * 0.35
    sc = 1 + (torch.rand(B * R, device=x.device) - 0.5) * 0.2
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    z = torch.zeros_like(cos)
    theta = torch.stack([torch.stack([cos, -sin, z], 1), torch.stack([sin, cos, z], 1)], 1)
    grid = F.affine_grid(theta, (B * R, D, H, W), align_corners=False)
    return F.grid_sample(x.reshape(B * R, D, H, W), grid, align_corners=False).reshape(B, R, D, H, W)

class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__(); self.a = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))
    def forward(self, h):
        return (torch.softmax(self.a(h), 1) * h).sum(1)

class KneeNet(nn.Module):
    def __init__(self, cl):
        super().__init__()
        self.cl = cl
        self.enc = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        d = self.enc.num_features
        self.pool = AttnPool(d); self.head = nn.Linear(3 * d, 12)
    def forward(self, x):
        B, R, T = x.shape[:3]
        imgs = x.flatten(0, 2)
        if self.cl:
            imgs = imgs.contiguous(memory_format=torch.channels_last)
        f = self.pool(self.enc(imgs).view(B * R, T, -1)).view(B, -1)
        return self.head(f)

def bench_pipeline(bs, cl, dp, steps=4):
    model = KneeNet(cl).cuda()
    if cl:
        model = model.to(memory_format=torch.channels_last)
    net = nn.DataParallel(model) if dp else model
    opt = torch.optim.AdamW(model.parameters(), 1e-4); scaler = torch.amp.GradScaler()
    t = {"to_gpu": 0, "augment+triplets": 0, "forward": 0, "backward+step": 0}
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 1):
        xb = torch.from_numpy(X[np.random.choice(len(X), bs)])
        torch.cuda.synchronize(); a = time.time()
        x = xb.cuda().float() / 255
        torch.cuda.synchronize(); b = time.time()
        x = (make_triplets(augment(x), NT) - MEAN) / STD
        torch.cuda.synchronize(); c = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = net(x).float().mean()
        torch.cuda.synchronize(); d = time.time()
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize(); e = time.time()
        if i > 0:
            for k, v in zip(t, [b - a, c - b, d - c, e - d]):
                t[k] += v / steps
    peak = max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9
    del model, net, opt; torch.cuda.empty_cache()
    total = sum(t.values())
    return {"studies/batch": bs, "channels_last": cl, "2 GPUs": dp, "s/step": round(total, 2),
            **{k: round(v, 3) for k, v in t.items()}, "GPU peak GB": round(peak, 1)}

rows = []
for bs, cl, dp in [(2, False, False), (4, False, False), (4, True, False), (8, True, False), (8, True, True)]:
    try:
        rows.append(bench_pipeline(bs, cl, dp))
    except torch.cuda.OutOfMemoryError:
        rows.append({"studies/batch": bs, "channels_last": cl, "2 GPUs": dp, "s/step": "GPU out of memory"})
        torch.cuda.empty_cache()
    print(rows[-1], flush=True)
    mem(f"after bs={bs} cl={cl} dp={dp}")
display(pd.DataFrame(rows))

# %% [markdown]
# ## 3. Send back
#
# Copy the two tables and the `[...] host RSS` lines. They tell us:
# - whether the GPU itself is fast (section 1),
# - which part of our step is slow (section 2: `to_gpu`, `augment+triplets`, `forward`, `backward+step`),
# - whether one GPU or two is faster, and whether `channels_last` helps,
# - how much host RAM and GPU memory each setup needs, which explains the dying kernel.
