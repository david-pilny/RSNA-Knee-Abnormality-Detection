"""Offline analysis of the label sets and model predictions against the 58 gold studies (numbers only, no report text).

Reads data/: labels_v3.csv, labels_v4.csv, llm_probs_v4.csv, v3/ and v4/ gold_predictions.csv + oof_predictions.csv.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

pd.set_option("display.width", 220)
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
CODES = ["P", "M", "U", "A", "N"]
D = "data/"

l3, l4 = (pd.read_csv(D + f).set_index("StudyInstanceUID") for f in ("labels_v3.csv", "labels_v4.csv"))
pr = pd.read_csv(D + "llm_probs_v4.csv").set_index("StudyInstanceUID")
g3, g4 = (pd.read_csv(D + f"{v}/gold_predictions.csv").set_index("StudyInstanceUID") for v in ("v3", "v4"))
o3, o4 = (pd.read_csv(D + f"{v}/oof_predictions.csv").set_index("StudyInstanceUID") for v in ("v3", "v4"))
gold = l4[l4.is_gold][LABELS]                      # gold labels (identical in both label files)
assert (l3.loc[gold.index, LABELS] == gold).all().all()
Y = gold.to_numpy()

def aucs(pred):
    p = pred.loc[gold.index, LABELS].to_numpy()
    return pd.Series([roc_auc_score(Y[:, k], p[:, k]) for k in range(12)], index=LABELS)

def rank(df):
    return pd.DataFrame(np.column_stack([rankdata(df[l]) / len(df) for l in LABELS]), index=df.index, columns=LABELS)

if __name__ == "__main__":
    print("gold studies:", len(gold), "| positives:", dict(gold.sum().astype(int)))
    t = pd.DataFrame({"model v3": aucs(g3), "model v4": aucs(g4)})
    blend = (rank(g3.loc[gold.index]) + rank(g4.loc[gold.index])) / 2
    t["rank mean v3+v4"] = aucs(blend)
    t["best of two"] = t[["model v3", "model v4"]].max(axis=1)
    t.loc["macro"] = t.mean()
    print("\n--- image models on gold ---\n", t.round(3).to_string())
