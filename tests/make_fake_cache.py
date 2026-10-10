"""Fake training inputs (random images, random labels) for testing the training notebooks locally. Never real data.

Usage: python tests/make_fake_cache.py <root> [n_studies] [img] [crop_mm]
Creates <root>/labels/labels_v4.csv and <root>/cache/{index.csv, config.json, shard_*.npz} (+ knee_preproc.py stub).
Set INPUT_BASE=<root> for the training notebook. Some studies miss a series; 6 studies are gold.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ROLES = {"sag_fs": ["Sagittal", 1], "cor_fs": ["Coronal", 1], "ax_fs": ["Axial", 1]}


def make(root, n=60, img=64, crop_mm=140.0, depth=24, shard_size=25, seed=0):
    root, rng = Path(root), np.random.default_rng(seed)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    ids = [f"1.2.7.{i}" for i in range(n)]
    lab = pd.DataFrame({"StudyInstanceUID": ids})
    for l in LABELS:
        lab[l] = rng.choice([0.0, 0.2, 0.5, 1.0, np.nan], n, p=[.5, .1, .1, .25, .05])
    lab["is_gold"] = [i < 6 for i in range(n)]
    lab["language"] = rng.choice(["en", "es", "de", "tr", "fr", "nl", "ru"], n)
    lab.to_csv(root / "labels" / "labels_v4.csv", index=False)
    # Every fake knee has a side: a bright block near the patient-left edge of coronal/axial images and in the last
    # (left-most) sagittal slices for a left knee, mirrored for a right knee. Laterality is tagged for ~70 % of studies.
    side = rng.choice(["L", "R"], n)
    tag = np.where(rng.random(n) < 0.7, side, None)
    tag[::23] = "B"                                       # a few odd tag values the notebook must ignore
    rows = []
    for s0 in range(0, n, shard_size):
        name = f"shard_{s0 // shard_size:03d}.npz"
        arrs = {}
        for i in range(s0, min(n, s0 + shard_size)):
            for role in ROLES:
                ok = not (role == "ax_fs" and i % 7 == 3)
                if ok:
                    a = rng.integers(0, 120, (depth, img, img), dtype=np.uint8)
                    q = max(2, img // 4)
                    if role == "sag_fs":
                        sl = slice(depth - depth // 4, depth) if side[i] == "L" else slice(0, depth // 4)
                        a[sl, q:-q, q:-q] = 230
                    else:
                        cs = slice(img - q, img) if side[i] == "L" else slice(0, q)
                        a[:, q:-q, cs] = 230
                    arrs[f"{ids[i]}__{role}"] = a
                rows.append({"StudyInstanceUID": ids[i], "role": role, "ok": ok, "shard": name,
                             "manufacturer": rng.choice(["SIEMENS", "GE MEDICAL", "Philips"]),
                             "laterality": tag[i], "x_center": (30 if side[i] == "L" else -30) + rng.normal(0, 25)})
        np.savez_compressed(cache / name, **arrs)
    pd.DataFrame({"StudyInstanceUID": ids, "true_side": side}).to_csv(root / "true_side.csv", index=False)
    pd.DataFrame(rows).to_csv(cache / "index.csv", index=False)
    (cache / "config.json").write_text(json.dumps({"ROLES": ROLES, "DEPTH": depth, "IMG": img, "CROP_MM": crop_mm}))
    (root / "knee_preproc.py").write_text("# stub\n")
    return root


if __name__ == "__main__":
    a = sys.argv[1:]
    print(make(a[0], *(int(x) for x in a[1:3]), *(float(x) for x in a[3:4])))
