# %% [markdown]
# ## Our member: EfficientNet-B0 models on v3 / v4 labels, blended into the public predictions
#
# **Where this cell goes:** in the forked public notebook ("Bend the Knee to Speedy Raptors"), as a new cell directly
# **before the last cell** (the one starting with `# Publish submission.csv only after …`). Attach our datasets:
# `knee-models` (v4 models), `knee-models-v3`, `knee-mri-cache` (for `knee_preproc.py`) and `knee-dicom-wheels`.
#
# What it does:
# 1. copies the public notebook's staged predictions (`_pipeline_stage.csv`, rank percentiles) to a backup;
# 2. runs our own pipeline on the test DICOMs, exactly as in our 0.906 submission (same preprocessing, the ten models in two
#    groups, mirror TTA, rank blend with the v3 group switched off for ACL / MCL / PF OA);
# 3. blends our ranks into the staged predictions with weight `OUR_WEIGHT` and re-ranks;
# 4. if anything fails or our time allowance runs out, the backup is restored, so the public result is never at risk.
#
# Everything is found by searching `/kaggle/input` (no hard-coded dataset paths). The only globals used from the public
# notebook are `T0` and `TIME_BUDGET` (its clock), and they are optional.

# %% [code] {"jupyter":{"outputs_hidden":false}}
OUR_WEIGHT = 0.25            # share of our member in the final blend (0 = switch our member off)
OUR_TIME_LIMIT_H = 2.0       # our part stops after this many hours; studies not reached keep the public prediction
OUR_BLEND_WEIGHTS = {}       # inside our member, per group and finding (default 1), e.g. {"v5_b0": {"ACL": 0}}
OUR_CHUNK = 32

import os, sys, time, json, shutil, subprocess, traceback
from pathlib import Path
import numpy as np
import pandas as pd

_our_t0 = time.time()
_our_work = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
_our_stage = _our_work / "_pipeline_stage.csv"
_our_backup = _our_work / "_pipeline_stage_before_our_member.csv"
_our_log = lambda msg: print(f"[our member +{(time.time() - _our_t0) / 60:5.1f} min] {msg}", flush=True)

def our_member():
    from scipy.stats import rankdata
    INPUT = Path(os.environ.get("INPUT_BASE", "/kaggle/input"))
    SLUG = "rsna-knee-abnormality-detection"
    LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
              "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

    def find_dirs_with(filename, max_depth=6, skip=()):
        hits, frontier = [], [INPUT]
        for _ in range(max_depth):
            nxt = []
            for d in frontier:
                if d in skip or d.name in ("train_series", "test_series", "cache"):
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

    root = globals().get("ROOT")                       # the public notebook's competition root, if it exists
    root = Path(root) if root and (Path(root) / "test.csv").exists() else next(
        (Path(p) for p in [os.environ.get("KNEE_ROOT", ""), f"{INPUT}/competitions/{SLUG}", f"{INPUT}/{SLUG}"]
         if p and (Path(p) / "test.csv").exists()), None) or find_dirs_with("test.csv")[0]
    stage = pd.read_csv(_our_stage, dtype={"StudyInstanceUID": str})
    assert stage.columns.tolist() == ["StudyInstanceUID"] + LABELS, "staged file has another layout than expected"
    studies = stage.StudyInstanceUID.tolist()
    _our_log(f"competition root {root} | staged predictions for {len(studies)} studies")

    # ---- our inputs
    preproc_dir = find_dirs_with("knee_preproc.py", skip={root})[0]
    cache_cfg = json.loads(next(preproc_dir.rglob("config.json")).read_text())
    groups = []
    for d in sorted(find_dirs_with("train_config.json", skip={root})):
        cfg, files = json.loads((d / "train_config.json").read_text()), sorted(d.glob("model_fold*.pt"))
        if files:
            groups.append({"name": cfg.get("group_name") or str(cfg.get("labels_file", d.name)).rsplit(".", 1)[0], "cfg": cfg, "files": files})
    assert groups, "no model datasets (train_config.json + model_fold*.pt) attached"
    for g in groups:
        assert g["cfg"]["labels"] == LABELS and g["cfg"].get("depth", cache_cfg["DEPTH"]) == cache_cfg["DEPTH"]
        _our_log(f"group '{g['name']}': {len(g['files'])} models, {g['cfg']['backbone']}, gold AUC at training {g['cfg'].get('gold_macro_auc', float('nan')):.3f}")
    W = np.ones((len(groups), len(LABELS)))
    if len(groups) > 1:
        for gname, per in OUR_BLEND_WEIGHTS.items():
            assert gname in [g["name"] for g in groups], f"OUR_BLEND_WEIGHTS names '{gname}', attached: {[g['name'] for g in groups]}"
            for lab, w in per.items():
                W[[g["name"] for g in groups].index(gname), LABELS.index(lab)] = w
    wheel_dirs = []
    frontier = [INPUT]
    for _ in range(6):
        nxt = []
        for d in frontier:
            if d == root or d.name in ("train_series", "test_series", "cache"):
                continue
            try:
                kids = list(d.iterdir())
            except OSError:
                continue
            if any(k.suffix == ".whl" and k.name.lower().startswith("pylibjpeg") for k in kids):
                wheel_dirs.append(str(d))                  # only folders with OUR decoder wheels (the fork has others)
            nxt += [k for k in kids if k.is_dir()]
        frontier = nxt
    import importlib, importlib.util
    mods = ["pylibjpeg", "libjpeg", "openjpeg"]
    if all(importlib.util.find_spec(m) for m in mods):
        _our_log("decoder modules already installed")
    elif wheel_dirs:
        links = [a for d in wheel_dirs for a in ("--find-links", d)]
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", *links, "-q",
                            "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], capture_output=True, text=True)
        importlib.invalidate_caches()
        _our_log(f"decoder wheels from {wheel_dirs}: " + ("installed" if r.returncode == 0 else "install failed: " + r.stderr[-200:]))
    else:
        _our_log("no pylibjpeg wheels found: compressed DICOMs may fail (those studies keep the public prediction)")
    _our_log("decoders: " + ", ".join(f"{m} {'ok' if importlib.util.find_spec(m) else 'MISSING'}" for m in mods))

    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if str(preproc_dir) not in sys.path:
        sys.path.insert(0, str(preproc_dir))
    os.environ["PYTHONPATH"] = str(preproc_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")
    import knee_preproc as kp
    import torch, torch.nn as nn, timm
    from joblib import Parallel, delayed

    device = "cuda" if torch.cuda.is_available() else "cpu"
    channels_last = device == "cuda"
    if device == "cuda":
        torch.cuda.empty_cache()
    roles = []
    for g in groups:
        roles += [r for r in g["cfg"]["roles"] if r not in roles]
    roles_used = {r: tuple(cache_cfg["ROLES"][r]) for r in roles}
    DEPTH, IMG = cache_cfg["DEPTH"], cache_cfg["IMG"]

    # ---- architecture: identical to our training notebook
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
            if channels_last:
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
        x = x_u8.to(device).float() / 255.0
        return (make_triplets(x, cfg["n_triplets"]) - cfg["mean"]) / cfg["std"]

    def mirror(x_u8):
        x = x_u8.clone()
        for j, role in enumerate(roles):
            x[:, j] = x[:, j].flip(1) if role.startswith("sag") else x[:, j].flip(3)
        return x

    for g in groups:
        g["role_idx"] = [roles.index(r) for r in g["cfg"]["roles"]]
        g["models"] = []
        for f in g["files"]:
            m = KneeNet(g["cfg"]["backbone"], False, len(g["cfg"]["roles"])).to(device)
            m.load_state_dict({k: v.float() for k, v in torch.load(f, map_location=device).items()})
            if channels_last:
                m = m.to(memory_format=torch.channels_last)
            g["models"].append(m.eval())
    _our_log(f"loaded {sum(len(g['models']) for g in groups)} models on {device}")

    @torch.no_grad()
    def predict_batch(x_u8, rmask):
        views = [x_u8, mirror(x_u8)]
        out = []
        for g in groups:
            rm, probs = rmask[:, g["role_idx"]].to(device), []
            for v in views:
                x = prepare(v[:, g["role_idx"]], g["cfg"])
                for m in g["models"]:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                        probs.append(torch.sigmoid(m(x, rm).float()))
            out.append(torch.stack(probs).mean(0))
        return torch.stack(out).cpu().numpy()                   # (groups, studies, findings)

    # ---- preprocess and predict, chunk by chunk, within our time allowance
    series_root = root / "test_series"
    test_series = pd.read_csv(root / "test_series.csv")
    rows_of = {s: g for s, g in test_series.groupby("StudyInstanceUID")}

    def run(study):
        try:
            return kp.process_study(study, rows_of.get(study, test_series.iloc[:0]), series_root, roles_used, DEPTH, IMG)
        except Exception as e:
            return {r: None for r in roles}, [{"StudyInstanceUID": study, "role": "all", "ok": False, "error": repr(e)[:200]}]

    deadline = _our_t0 + OUR_TIME_LIMIT_H * 3600
    budget_end = globals().get("T0", _our_t0) + globals().get("TIME_BUDGET", 8 * 3600) - 20 * 60   # the public clock, 20 min spare
    preds, n_fail, stopped = {}, 0, False
    for c in range(0, len(studies), OUR_CHUNK):
        if time.time() > min(deadline, budget_end):
            stopped = True
            break
        chunk = studies[c:c + OUR_CHUNK]
        res = Parallel(n_jobs=os.cpu_count() or 2)(delayed(run)(s) for s in chunk)
        X = np.zeros((len(chunk), len(roles), DEPTH, IMG, IMG), np.uint8)
        RM = np.zeros((len(chunk), len(roles)), np.float32)
        for i, (arrs, meta) in enumerate(res):
            for j, role in enumerate(roles):
                if arrs.get(role) is not None:
                    X[i, j] = arrs[role]; RM[i, j] = 1
        keep = np.where(RM.sum(1) > 0)[0]
        n_fail += len(chunk) - len(keep)
        for b in range(0, len(keep), 8):
            idx = keep[b:b + 8]
            p = predict_batch(torch.from_numpy(X[idx]), torch.from_numpy(RM[idx]))
            for k, i in enumerate(idx):
                preds[chunk[i]] = p[:, k]
        if (c // OUR_CHUNK) % 5 == 0:
            done = min(c + OUR_CHUNK, len(studies)); el = time.time() - _our_t0
            _our_log(f"{done}/{len(studies)} studies | ETA {el / done * (len(studies) - done) / 60:.0f} min")
    _our_log(f"predicted {len(preds)}/{len(studies)} studies ({n_fail} without usable images)" + (" | STOPPED at the time limit" if stopped else ""))
    assert len(preds) >= max(1, len(studies) // 2), "fewer than half of the studies predicted: keeping the public result"

    # ---- our member's own rank blend (as in our submission), then blend into the staged predictions
    uids = list(preds)
    A = np.stack([preds[u] for u in uids])                                       # (studies, groups, findings)
    if len(groups) > 1:
        A = rankdata(A, axis=0) / len(uids)
    ours = pd.DataFrame((A * W[None]).sum(1) / W.sum(0)[None], index=uids, columns=LABELS)
    ours = ours.rank(method="average", pct=True)
    pub = stage.set_index("StudyInstanceUID")[LABELS].astype(float)
    have = pub.index.isin(ours.index)
    blended = pub.copy()
    blended.loc[have] = (1 - OUR_WEIGHT) * pub.loc[have] + OUR_WEIGHT * ours.loc[pub.index[have]].to_numpy()
    blended = blended.rank(method="average", pct=True)
    corr = {l: round(float(pub.loc[have, l].corr(ours.loc[pub.index[have], l], method="spearman")), 2) for l in LABELS} if have.sum() > 2 else {}
    _our_log(f"rank correlation public vs ours: {corr}")
    assert np.isfinite(blended.to_numpy()).all() and len(blended) == len(stage)
    out = blended.reset_index()[["StudyInstanceUID"] + LABELS]
    out.to_csv(_our_stage, index=False)
    _our_log(f"blended with weight {OUR_WEIGHT} for {int(have.sum())} studies → {_our_stage.name} rewritten")

if OUR_WEIGHT <= 0:
    _our_log("switched off (OUR_WEIGHT = 0): public predictions unchanged")
elif not _our_stage.exists():
    _our_log(f"{_our_stage} not found: nothing to blend into, public pipeline unchanged")
else:
    shutil.copy(_our_stage, _our_backup)
    try:
        our_member()
    except Exception:
        traceback.print_exc()
        shutil.copy(_our_backup, _our_stage)
        _our_log("FAILED: staged predictions restored, the public result is submitted unchanged")
    _our_log(f"done in {(time.time() - _our_t0) / 60:.1f} min")
