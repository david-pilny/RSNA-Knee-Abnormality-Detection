# CLAUDE.md: RSNA Knee Abnormality Detection (Kaggle 2026)

Hand-off notes for continuing this project in a new session. Last updated **1 Oct 2026**.

---

## 1. Competition in one screen

- **Kaggle:** `rsna-knee-abnormality-detection` (RSNA 2026 AI Challenge, $77k). Code competition: the submission notebook is
  re-run on the hidden test set **offline (Internet off), ≤ 9 h**.
- **Task:** per study (one knee MRI exam), predict probabilities for 12 findings:
  `ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`.
- **Metric:** macro ROC-AUC over the 12 findings (ranking only; rare findings weigh as much as common ones).
- **Data:** 4,407 training studies / 24,371 series (DICOM, one file per slice, `train_series/<study>/<series>/<sop>.dcm`).
  `train_series.csv` gives per series: `Anatomical_Plane` (Sagittal/Coronal/Axial), `Fluid_Sensitive`, `Fat_Suppression`.
  Only **58 studies have gold labels**; the other 4,349 have only a free-text radiology **report** (~12 languages).
  Test: ~1,300 studies, **images only, no reports**.
- **Deadlines:** entry / team merge **15 Oct 2026**, final submission **22 Oct 2026**. Kaggle lets you pick 2 final submissions.
- **Leaderboard (1 Oct):** our best **0.882** public (4 folds, see §6) = rank **3,239 / 4,750**. Top of LB **0.961**.
  The LB is crowded around public notebooks (~0.94), so small gains move the rank a lot in that band. Public write-ups: one approach with
  0.903 (EffNet-B3, 288 px, 12 slices/series, 35B labelling LLM, "label quality dominates"); one with 0.939 (labeller AUC 0.869
  vs gold, CoAtNet 2.5D MIL ensemble; mentions public notebooks at ~0.94).

## 2. Hard rules / constraints

- **Never send report text to hosted LLM APIs** (ChatGPT, Claude, Gemini…): the data-security rule (4.b) plausibly forbids it.
  The labeller runs an open-weights model inside Kaggle. This also applies to the assistant: **don't ask the user to paste
  report text**; share numbers/charts only. The labeller has `SHOW_REPORT_TEXT = False` by default for this reason.
- **Do not use GPU P100** on Kaggle (current PyTorch has no Pascal kernels). Use **GPU T4 ×2**.
- **Do not use `nn.DataParallel`** on Kaggle's torch 2.10 + T4: 7.5 s/step instead of 0.4, kernel deaths, and
  `CUDA error: misaligned address` with channels_last. Use **one GPU** (to use the second one, run two folds in separate processes).
- Kaggle notebook output limits: **≤ 500 files, ≤ 20 GB** → the image cache is written as shards.
- GPU quota ~30 h/week. One training fold ≈ 40 min.

## 3. How the user works (conventions)

- User: David, frontend dev (Vue/TS) with a biomedical-informatics background (ultrasound segmentation thesis). New to Kaggle
  competitions and DICOM; likes understanding *why*, likes visual/interactive explanations and PDF explainers.
- **Deliver Kaggle notebooks as "percent-format" `.py` scripts** that he pastes into Kaggle and converts to a notebook:
  ```
  # %% [markdown]
  # text…
  # %% [code] {"jupyter":{"outputs_hidden":false}}
  code…
  ```
  Each step gets a markdown explanation. Test locally before delivering (convert with `jupytext`, execute with
  `nbconvert`, using fake data + env overrides like `KNEE_ROOT`, `OUT_DIR`, `INPUT_BASE`).
- He runs things with **Save & Run All** (background commits) and pastes back the **log** text. Commit logs don't show
  `display()` tables; ask for specific printed lines or the notebook output when needed.
- Prefers direct, honest feedback; flag mistakes plainly.
- Kaggle username in paths: `davidpiln` (datasets mount at `/kaggle/input/datasets/davidpiln/<name>/…`; competition at
  `/kaggle/input/competitions/rsna-knee-abnormality-detection`). Notebooks locate inputs by shallow search, not hard-coded paths.

## 4. Pipeline and Kaggle datasets

```
competition data ──► ① report labeller (Qwen2.5-7B, 2×T4) ──► dataset knee-report-labels  (labels_v3.csv  = y)
                 └─► ② preprocess cache (CPU)             ──► dataset knee-mri-cache      (shards + knee_preproc.py = X)
                                                              │
                     ③ training (EffNet-B0, 1×T4) ◄───────────┘ ──► dataset knee-models (model_fold*.pt, train_config.json)
                     ④ submission (offline) ◄── knee-models + knee-mri-cache (for knee_preproc.py) + knee-dicom-wheels
```

| Kaggle dataset | Contents | Made by |
|---|---|---|
| `knee-report-labels` | `labels_v3.csv`, `llm_raw_v3.csv`, `label_maps_v3.json` | labeller run (full, v3) |
| `knee-mri-cache` | `cache/shard_000..044.npz`, `cache/index.csv`, `cache/config.json`, `knee_preproc.py` | preprocess run |
| `knee-models` | `model_fold{k}.pt` (fp16 state dicts), `train_config.json`, oof/gold preds | training runs |
| `knee-dicom-wheels` | `wheels/*.whl` for pylibjpeg, pylibjpeg-libjpeg, pylibjpeg-openjpeg | 1-cell `pip download` notebook |

**Open issue:** the current `knee-models` version contains only folds **1–4** (the second training run's output replaced
fold 0). Fix: new dataset version with all five `model_fold*.pt` (fold 0 from the first successful training run's output).

## 5. Notebook scripts (files delivered so far)

| File | Purpose | Settings | Status |
|---|---|---|---|
| `rsna_knee_data_explained.py` | data explainer: hierarchy, DICOM tags, sorting, geometry, intensities, **interactive three.js 3D viewer** | GPU optional, Internet on (CDN) | done |
| `rsna_knee_dummy_submission.py` | constant-prior submission; tests offline pipeline + decoding | CPU, Internet off | submitted → **0.500** |
| `rsna_knee_report_labeler.py` | LLM labeller (see §6.1) | GPU T4×2, Internet on | full run done (3.8 h) |
| `rsna_knee_preprocess_cache.py` | DICOM → cache shards + `knee_preproc.py` | CPU, Internet on | done (98 min, 10.4 GB) |
| `rsna_knee_train_baseline.py` | training (see §6.3) | GPU T4×2 (uses 1), Internet on | fold 0 + folds 1–4 trained |
| `rsna_knee_speed_diagnostic.py` | GPU speed / memory diagnostic (found the DataParallel bug) | GPU T4×2, interactive | done, not needed again |
| `rsna_knee_submission.py` | real submission (see §6.4) | GPU T4×2, **Internet off** | submitted → **0.882** (4 folds) |
| `rsna_knee_explainer.pdf`, `rsna_knee_project_report.pdf` | explainer PDFs for the user | – | – |

## 6. Component details and results

### 6.1 Report labeller (`rsna_knee_report_labeler.py`)
- Model: `Qwen/Qwen2.5-7B-Instruct` via transformers, fp16, `device_map="auto"` over 2×T4 (downloaded from HF or Kaggle Models).
- Prompt: definitions of 12 findings + grades `present / minimal / uncertain / absent / not_mentioned` + report language.
  Answer is **prefilled** with `{"language": "` to force a JSON object (v1 without prefill → model returned a list, 100 % parse errors).
- Batching by **token budget** (`TOKEN_BUDGET=12000`, `BATCH_SIZE≤16`) + OOM split-and-retry (a fixed batch of 16 long reports OOM'd).
- Tolerant parser (synonyms incl. copied report words like "ruptured", "tear", "fraktür"); unparseable → NaN (masked in training).
- Calibration on gold: per finding, a borderline grade's value (0/0.5/1) changes only if ≥ 4 gold studies have that grade.
  Result `label_maps_v3.json`: defaults (present 1, minimal 1, uncertain 0.5, absent 0, not_mentioned 0) except
  **Effusion minimal → 0.5**, **MCL minimal → 0**.
- Agreement with gold (58): v2 macro AUC 0.818, **v3 0.809** (within noise; v3 better on medial meniscus 0.88, worse on
  synovitis 0.68, contusion 0.71). Full run: 4,407 reports, 0.32 reports/s, 3.6 % studies with some parse error,
  4,247 fully labelled. ACL and synovitis disagreements resisted prompt fixes (gold likely from image review).
- `labels_v3.csv` columns: `StudyInstanceUID, is_gold, <12 soft labels>, language, grade_<finding>…, has_label`;
  gold studies carry their gold labels.

### 6.2 Preprocessing cache (`rsna_knee_preprocess_cache.py` → `knee_preproc.py`)
- Roles: `sag_fs` (Sagittal, fluid), `cor_fs` (Coronal, fluid), `ax_fs` (Axial, fluid); prefer fat-sat; else fallback to any
  series of that plane (flag `fallback`). All 3 roles available for 100 % of training-pool studies.
- Per series: load all slices, sort by `ImagePositionPatient` along a canonical axis, **canonical orientation**
  (flip/transpose via `ImageOrientationPatient`; verified identical output for 5 stored variants), percentile 0.5–99.5
  normalisation, square-mm pixels + pad, resample to **24 × 224 × 224 uint8**.
- Left/right knee NOT canonicalised; `Laterality` tag and x-position stored in `index.csv`.
- Shards: 100 studies each, keys `<StudyInstanceUID>__<role>`; `CacheReader(cache_dir).load(uid)`.
- `knee_preproc.process_study(uid, study_rows, series_root, roles, depth, img)` is used **identically** in the submission.

### 6.3 Training (`rsna_knee_train_baseline.py`)
- 2.5D multi-view: each series → 8 "RGB" images of 3 neighbouring slices (centres jitter ±1 in training) → shared
  timm `efficientnet_b0` (ImageNet-pretrained) → attention pooling over the 8 → + role embedding, × role mask → concat
  (+ mask) → MLP → 12 logits. Masked soft-label BCE (NaN labels masked).
- Augment on GPU: per-series small affine + brightness/contrast/gamma; **joint knee mirror** p=0.5 (reverse sagittal slice
  order + flip coronal/axial left-right together).
- Data: whole cache **preloaded into RAM** (15.9 GB, ~0.8 min; Kaggle T4×2 VM has ~32 GB), `num_workers=0`.
  (Per-process `CacheReader` if reading from disk: a shared zip handle across processes corrupts reads.)
- 1 GPU, fp16 autocast, `channels_last`, `cudnn.benchmark`; AdamW lr 3e-4, wd 1e-2, 1 warm-up epoch + cosine, 10 epochs,
  batch 8 studies → **0.48 s/step, 3.7 min/epoch, ~40 min/fold**.
- Validation: 5-fold `GroupKFold` by site proxy (`language|vendor`, 31 groups); the **58 gold studies are never trained on**
  and are evaluated by the fold ensemble.
- **Fold 0 result:** val macro AUC (report labels) peaked **0.786** at epoch 7 (overfits after); **gold macro AUC 0.836**.
- Outputs: `model_fold{k}.pt`, `train_config.json` (backbone, roles, n_triplets, mean/std, channels_last, labels…),
  `oof_predictions.csv`, `gold_predictions.csv`. **Per-finding table (section 10) not yet reviewed** — ask the user for it.

### 6.4 Submission (`rsna_knee_submission.py`)
- Safety-net `submission.csv` first; offline install of wheels; loads `knee_preproc.py` from the cache dataset and **every
  `model_fold*.pt`** found; preprocess in chunks of 32 (joblib, 4 CPUs) → predict on GPU; **mirror TTA**; average all;
  failed studies → prior; rewrites submission every few chunks; reports role success / decode failures
  (optional `RAISE_IF_DECODE_FAIL` diagnostic).
- Current architecture code is copied from training; **if training's `KneeNet` changes, update the submission copy too.**
  For multi-architecture ensembles it must be extended to several model groups, each with its own config.

## 7. Next steps (agreed plan)

1. **Restore fold 0** in `knee-models` → resubmit 5-fold ensemble.
2. Get the **per-finding AUC table** (training section 10) to see weak findings.
3. Review the **top public notebooks** (Code tab, sorted by score, ~0.94) — user can share links/code (public, not competition data).
4. Experiments, **one change at a time, fold 0 only**, judged on val AUC + gold AUC (gold SE ≈ 0.02; don't chase < 0.005):
   bigger backbone (EffNet-B3 / ConvNeXt-Nano), more images per series (12–16), higher resolution (288 → needs a new cache),
   then label quality (larger LLM, e.g. 4-bit Qwen2.5-32B on 2×T4, or targeted prompt fixes).
5. Add an experiment name + log to the training notebook; consider 2 folds in parallel (one per GPU, separate processes).
6. Final week: best config on 5 folds + a second architecture for the ensemble; runtime check; choose 2 final submissions.
   Don't tune to the public LB (subset of test → overfitting risk).

## 8. Lessons learned (avoid repeating)

- Measure before fixing: a 5-minute diagnostic found DataParallel after two blind fixes failed.
- Always print step-level progress (s/step, % waiting for data, RAM) in long runs; committed logs are all we see.
- "Safety net first" in every submission; every study in try/except.
- Committed runs start with an empty `/kaggle/working`; creating a new dataset version from a run's output **replaces**
  files (that's how fold 0 got lost).
- With 58 gold studies, stop prompt/label polishing at the noise floor.
