"""Why did Fracture get worse, how do v3 and v4 labels differ, and what do blends give on gold?"""
from gold_check import *
from labels_vs_gold import P, DEFAULT

Pm = P(pr); top = pd.DataFrame(np.array(CODES)[Pm.argmax(-1)], index=pr.index, columns=LABELS)
ng = l4.index[~l4.is_gold]

print("--- hard positives (soft label > 0.5) on the 4,349 report-labelled studies, % ---")
s = pd.DataFrame({"v3": (l3.loc[ng, LABELS] > 0.5).mean() * 100, "v4": (l4.loc[ng, LABELS] > 0.5).mean() * 100,
                  "v3 pos → v4 neg": [int(((l3.loc[ng, l] > 0.5) & (l4.loc[ng, l] <= 0.5)).sum()) for l in LABELS],
                  "v3 neg → v4 pos": [int(((l3.loc[ng, l] <= 0.5) & (l4.loc[ng, l] > 0.5)).sum()) for l in LABELS],
                  "gold %": gold.mean() * 100})
print(s.round(1).to_string())
print("\n--- v4 answers on all studies, % ---")
print((pd.DataFrame({l: top.loc[ng, l].value_counts(normalize=True) for l in LABELS}).T.reindex(columns=CODES).fillna(0) * 100).round(1).to_string())

print("\n--- Fracture: gold-positive studies (18) by v4 Fracture answer × v4 Contusion answer ---")
gp = gold.index[gold.Fracture == 1]
print(pd.crosstab(top.loc[gp, "Fracture"], top.loc[gp, "Contusion"], margins=True).to_string())
print("gold Fracture × gold Contusion:\n", pd.crosstab(gold.Fracture, gold.Contusion).to_string())
print("v3 grade for those 18:", l3.loc[gp, "grade_Fracture"].value_counts().to_dict(), "| v3 Contusion grade:", l3.loc[gp, "grade_Contusion"].value_counts().to_dict())
r3, r4 = rank(g3.loc[gold.index]), rank(g4.loc[gold.index])
d = pd.DataFrame({"v4 Fracture": top.loc[gp, "Fracture"], "v4 Contusion": top.loc[gp, "Contusion"], "v3 grade": l3.loc[gp, "grade_Fracture"],
                  "rank model v3": r3.loc[gp, "Fracture"].round(2), "rank model v4": r4.loc[gp, "Fracture"].round(2)}).reset_index(drop=True)
print(d.sort_values("rank model v4").to_string())
gn = gold.index[gold.Fracture == 0]
print("gold-NEGATIVE studies ranked high by model v4 (rank > 0.6):")
dn = pd.DataFrame({"v4 Fracture": top.loc[gn, "Fracture"], "v4 Contusion": top.loc[gn, "Contusion"], "gold Contusion": gold.loc[gn, "Contusion"],
                   "rank model v3": r3.loc[gn, "Fracture"].round(2), "rank model v4": r4.loc[gn, "Fracture"].round(2)}).reset_index(drop=True)
print(dn[dn["rank model v4"] > 0.6].sort_values("rank model v4", ascending=False).to_string())

print("\n--- cross-evaluation on out-of-fold predictions (4,348 studies): AUC of each model against each label set (hard labels) ---")
rows = {}
for name, o in (("model v3", o3), ("model v4", o4)):
    for lname, L in (("v3 labels", l3), ("v4 labels", l4)):
        idx = o.index.intersection(ng)
        rows[f"{name} vs {lname}"] = [roc_auc_score((L.loc[idx, l] > 0.5)[L.loc[idx, l].notna()], o.loc[idx, l][L.loc[idx, l].notna()]) for l in LABELS]
x = pd.DataFrame(rows, index=LABELS); x.loc["macro"] = x.mean(); print(x.round(3).to_string())
j = o3[LABELS].join(o4[LABELS], lsuffix="_3", rsuffix="_4", how="inner")
print("Spearman between the two models' OOF predictions:", {l: round(j[f"{l}_3"].corr(j[f"{l}_4"], method="spearman"), 2) for l in LABELS})

print("\n--- blends on gold ---")
def macro(p): return aucs(p).mean()
for w in (0, 0.25, 0.5, 0.75, 1):
    print(f"rank blend, weight on v4 = {w}: macro {macro((1 - w) * r3 + w * r4):.4f} | prob blend: {macro((1 - w) * g3.loc[gold.index] + w * g4.loc[gold.index]):.4f}")
hyb = r4.copy(); hyb["Fracture"] = r3["Fracture"]
print(f"v4 everywhere, Fracture from v3: {macro(hyb):.4f}")
hyb2 = (r3 + r4) / 2; hyb2["Fracture"] = r3["Fracture"]
for l in ("ACL", "MCL", "PF OA"): hyb2[l] = r4[l]
print(f"rank mean, but Fracture from v3 and ACL/MCL/PF OA from v4: {macro(hyb2):.4f}")

rng = np.random.default_rng(0); B = 4000; n = len(gold)
def boot(pa, pb):
    out = []
    a, b = pa.loc[gold.index, LABELS].to_numpy(), pb.loc[gold.index, LABELS].to_numpy()
    for _ in range(B):
        i = rng.integers(0, n, n)
        if any(len(np.unique(Y[i, k])) < 2 for k in range(12)): continue
        out.append(np.mean([roc_auc_score(Y[i, k], b[i, k]) - roc_auc_score(Y[i, k], a[i, k]) for k in range(12)]))
    return np.percentile(out, [5, 50, 95]).round(3), round(float(np.mean(np.array(out) > 0)), 2)
print("paired bootstrap of the macro difference [5%, median, 95%], P(>0):")
print("  v4 − v3:", *boot(g3, g4)); print("  rank mean − v4:", *boot(g4, (r3 + r4) / 2)); print("  rank mean − v3:", *boot(g3, (r3 + r4) / 2))
