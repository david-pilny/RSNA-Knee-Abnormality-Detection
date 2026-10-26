"""Turn the numbers-only project files in data/ into small tables for the script's pgfplots figures.

Run from the project root:  .venv/bin/python docs/script/make_figure_data.py
Reads:  data/labels_v4.csv, data/llm_probs_v4.csv, data/v4/gold_predictions.csv (no report text anywhere)
Writes: docs/script/figdata/*.dat (space-separated, with a header row) and numbers.tex (macros for the text)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figdata"
OUT.mkdir(exist_ok=True)

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SHORT = {"Medial Meniscus": "Med.~meniscus", "Lateral Meniscus": "Lat.~meniscus", "Medial OA": "Med.~OA",
         "Lateral OA": "Lat.~OA", "Baker's": "Baker's"}
CODES = ["P", "M", "U", "A", "N"]


def short(name):
    return SHORT.get(name, name)


lab = pd.read_csv(ROOT / "data/labels_v4.csv")
gold, pool = lab[lab.is_gold], lab[~lab.is_gold]

# 1) languages of the 4,349 report-labelled studies
lang = pool.language.value_counts()
names = {"en": "English", "es": "Spanish", "tr": "Turkish", "hr": "Croatian", "el": "Greek", "de": "German",
         "ru": "Russian", "nl": "Dutch", "fr": "French", "bg": "Bulgarian", "sr": "Serbian", "th": "Thai"}
with open(OUT / "languages.dat", "w") as f:
    f.write("lang n\n")
    for code, n in lang.items():
        f.write(f"{{{names.get(code, code)}}} {n}\n")

# 2) how common each finding is: gold (58, image-derived) vs the report labels (share with soft label > 0.5)
with open(OUT / "prevalence.dat", "w") as f:
    f.write("finding gold pool\n")
    for name in LABELS:
        f.write(f"{{{short(name)}}} {100 * gold[name].mean():.1f} {100 * (pool[name] > 0.5).mean():.1f}\n")

# 3) the labeller's most likely answer letter per finding (pool), in percent
probs = pd.read_csv(ROOT / "data/llm_probs_v4.csv")
probs = probs[probs.StudyInstanceUID.isin(pool.StudyInstanceUID)]
with open(OUT / "letters.dat", "w") as f:
    f.write("x y share\n")
    for i, name in enumerate(LABELS):
        p = probs[[f"{name}__{c}" for c in CODES]].to_numpy()
        share = np.bincount(p.argmax(1), minlength=5) / len(p) * 100
        for j, s in enumerate(share):
            f.write(f"{j} {i} {s:.1f}\n")
        f.write("\n")

# 4) soft labels are continuous: histogram of the Effusion soft label in the pool
counts, edges = np.histogram(pool["Effusion"], bins=20, range=(0, 1))
with open(OUT / "effusion_hist.dat", "w") as f:
    f.write("left n\n")
    for e, n in zip(edges[:-1], counts):
        f.write(f"{e:.2f} {n}\n")
    f.write(f"{edges[-1]:.2f} 0\n")          # closing edge for pgfplots' ybar interval

# 5) real ROC curves: our v4 five-fold model on the 58 gold studies
pred = pd.read_csv(ROOT / "data/v4/gold_predictions.csv").set_index("StudyInstanceUID")
truth = gold.set_index("StudyInstanceUID").loc[pred.index]
aucs = {}
for name, tag in [("ACL", "acl"), ("Synovitis", "syn"), ("Effusion", "eff")]:
    y, s = truth[name].to_numpy().astype(int), pred[name].to_numpy()
    fpr, tpr, _ = roc_curve(y, s)
    aucs[tag] = roc_auc_score(y, s)
    with open(OUT / f"roc_{tag}.dat", "w") as f:
        f.write("fpr tpr\n")
        for a, b in zip(fpr, tpr):
            f.write(f"{a:.4f} {b:.4f}\n")

# 5b) how noisy is a 58-study gold score? bootstrap the macro AUC of the same predictions
rng = np.random.default_rng(0)
Y, P = truth[LABELS].to_numpy().astype(int), pred[LABELS].to_numpy()
boot = []
for _ in range(2000):
    idx = rng.integers(0, len(Y), len(Y))
    vals = [roc_auc_score(Y[idx, k], P[idx, k]) for k in range(len(LABELS)) if 0 < Y[idx, k].sum() < len(idx)]
    boot.append(np.mean(vals))
boot = np.array(boot)
counts, edges = np.histogram(boot, bins=30, range=(0.80, 0.95))
with open(OUT / "gold_bootstrap.dat", "w") as f:
    f.write("left n\n")
    for e, n in zip(edges[:-1], counts):
        f.write(f"{e:.4f} {n}\n")
    f.write(f"{edges[-1]:.4f} 0\n")
macro = np.mean([roc_auc_score(Y[:, k], P[:, k]) for k in range(len(LABELS))])
lo, hi = np.percentile(boot, [5, 95])

# 6) macros for numbers quoted in the text
with open(OUT / "numbers.tex", "w") as f:
    f.write(f"\\newcommand{{\\nPool}}{{{len(pool):,}}}\n")
    f.write(f"\\newcommand{{\\nGold}}{{{len(gold)}}}\n")
    f.write(f"\\newcommand{{\\nLang}}{{{lang.size}}}\n")
    for tag, v in aucs.items():
        f.write(f"\\newcommand{{\\auc{tag}}}{{{v:.3f}}}\n")
    f.write(f"\\newcommand{{\\goldPosACL}}{{{int(gold['ACL'].sum())}}}\n")
    f.write(f"\\newcommand{{\\goldPosSyn}}{{{int(gold['Synovitis'].sum())}}}\n")
    syn_n = (probs[[f"Synovitis__{c}" for c in CODES]].to_numpy().argmax(1) == 4).mean() * 100
    f.write(f"\\newcommand{{\\synSilent}}{{{syn_n:.0f}}}\n")
    f.write(f"\\newcommand{{\\goldMacro}}{{{macro:.3f}}}\n")
    f.write(f"\\newcommand{{\\bootLo}}{{{lo:.3f}}}\n\\newcommand{{\\bootHi}}{{{hi:.3f}}}\n")
    f.write(f"\\newcommand{{\\bootSD}}{{{boot.std():.3f}}}\n")

print("written:", sorted(p.name for p in OUT.iterdir()))
print("gold AUC (v4 model):", {k: round(v, 3) for k, v in aucs.items()})
