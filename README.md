# RSNA Knee Abnormality Detection (Kaggle 2026)

Our entry for the RSNA 2026 AI Challenge: per-study probabilities for 12 knee MRI findings, scored by macro ROC-AUC.
Only 58 training studies have gold labels; the rest have a free-text radiology report, so the pipeline is

1. **Report labeller** – an open-weights LLM (Qwen2.5-32B, 4-bit) turns each report into 12 soft labels, inside Kaggle.
2. **Preprocessing cache** – DICOM series → canonical orientation, knee crop, `24 × 256 × 256` uint8 per series.
3. **Training** – 2.5D multi-view CNN (EfficientNet-B0 / ConvNeXt-Nano, attention pooling over slice triplets), 5-fold.
4. **Submission** – offline inference with rank-blended model groups; also used as an extra member of a public ensemble.

Own pipeline: **0.918** public LB. Forked public notebook with or without our member: **0.943**.
`CLAUDE.md` is the full working log (results, rules, lessons, next steps).

## Layout

| Path | Contents |
|---|---|
| `notebooks/` | the current Kaggle notebooks as percent-format `.py` scripts (labeller v4, cache v5, training v5, submission, member cell) |
| `public/` | the member cell exactly as pasted into the forked public notebook |
| `archive/` | superseded notebooks (v3 labeller, 224 px cache, baseline training, dummy submission, speed diagnostic) |
| `analysis/` | offline analysis of labels and predictions against the gold studies (needs the git-ignored `data/`) |
| `tests/` | fake-data generators and local tests; `make_kaggle_ipynb.py` builds `.ipynb` files for import into Kaggle |
| `docs/` | data explainer, PDF explainers, decision records |

## Running locally

```
uv venv .venv && uv pip install torch timm transformers pydicom jupytext nbconvert pandas scikit-learn joblib
.venv/bin/jupytext --to notebook notebooks/<name>.py          # or execute with nbconvert, see CLAUDE.md §3
.venv/bin/python tests/make_fake_dicoms.py /tmp/fake_comp 8   # fake inputs; no real data is ever stored here
```

Report text never leaves Kaggle and is not kept in this repository; `data/` holds numbers only and is git-ignored.
