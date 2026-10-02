"""v4 labeller against gold, rebuilt from llm_probs_v4.csv (what the labeller's section 7 printed)."""
from gold_check import *

DEFAULT = {"P": 1.0, "M": 0.2, "U": 0.5, "A": 0.0, "N": 0.0}
V3_LABEL_AUC = {"ACL": 0.832, "MCL": 0.836, "Medial Meniscus": 0.886, "Lateral Meniscus": 0.774, "Medial OA": 0.920,
                "Lateral OA": 0.845, "PF OA": 0.758, "Effusion": 0.752, "Synovitis": 0.676, "Baker's": 0.904,
                "Contusion": 0.713, "Fracture": 0.806}

def P(frame):                                    # (n, 12, 5)
    return frame[[f"{l}__{c}" for l in LABELS for c in CODES]].to_numpy(float).reshape(len(frame), 12, 5)

def fit_maps(Pg, Y, strength=4):
    top, maps = Pg.argmax(-1), {}
    for k, l in enumerate(LABELS):
        m = dict(DEFAULT)
        for j, c in enumerate(CODES):
            if c in "MUN":
                s = top[:, k] == j
                m[c] = (Y[s, k].sum() + strength * DEFAULT[c]) / (s.sum() + strength)
        maps[l] = m
    return maps

def soft(Pm, maps):
    V = np.array([[maps[l][c] for c in CODES] for l in LABELS])
    return (Pm * V[None]).sum(-1)

if __name__ == "__main__":
    print("run key:", pr.run_key.unique(), "| rows:", len(pr), "| NaN rows:", int(pr.filter(like="__P").isna().any(axis=1).sum()))
    Pg = P(pr.loc[gold.index]); top = Pg.argmax(-1)
    ct = pd.DataFrame([{**{"gold_pos": int(Y[:, k].sum())},
                        **{c: f"{int((top[:, k] == j).sum())} ({int(Y[top[:, k] == j, k].sum())})" for j, c in enumerate(CODES)}}
                       for k in range(12)], index=LABELS)
    print("\n--- v4 answers on gold: n (gold-positive) ---\n", ct.to_string())
    maps = fit_maps(Pg, Y)
    print("\n--- measured label values ---\n", pd.DataFrame(maps).T[CODES].round(2).to_string())
    a = pd.DataFrame({"v3 labels": pd.Series(V3_LABEL_AUC),
                      "v4 default": [roc_auc_score(Y[:, k], soft(Pg, {l: DEFAULT for l in LABELS})[:, k]) for k in range(12)],
                      "v4 measured": [roc_auc_score(Y[:, k], soft(Pg, maps)[:, k]) for k in range(12)],
                      "model v3": aucs(g3), "model v4": aucs(g4)}, index=LABELS)
    a.loc["macro"] = a.mean()
    print("\n--- AUC vs gold: labels and models ---\n", a.round(3).to_string())
    m = pr.filter(like="__mass").loc[gold.index].to_numpy()
    print(f"\nletter mass on gold: mean {m.mean():.3f}, min {m.min():.3f} | all studies: mean {pr.filter(like='__mass').to_numpy().mean():.3f}")

    # are the folds of the two training runs the same?
    j = o3[["fold"]].join(o4[["fold"]], lsuffix="_v3", rsuffix="_v4", how="inner")
    print(f"\nfolds: {len(j)} studies in both runs, same fold for {(j.fold_v3 == j.fold_v4).mean():.1%}")
    print("language column v3 == v4 file:", (l3.language.fillna('?') == l4.language.fillna('?')).mean().round(3),
          "| languages:", l4.language.value_counts().head(14).to_dict())
