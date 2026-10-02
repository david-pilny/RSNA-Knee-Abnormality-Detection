# CLAUDE.md: RSNA Knee Abnormality Detection (Kaggle 2026)

Hand-off notes for continuing this project in a new session. Last updated **2 Oct 2026**.

**Open decision:** see `DECISION.md` (start from the public 0.94 notebook, or keep improving our own pipeline). David decides;
don't start either path before he has answered.

---

## 1. Competition in one screen

- **Kaggle:** `rsna-knee-abnormality-detection` (RSNA 2026 AI Challenge, $77k). Code competition: the submission notebook is
  re-run on the hidden test set **offline (Internet off), ≤ 9 h**.
- **Task:** per study (one knee MRI exam), predict probabilities for 12 findings:
  `ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`.
- **Metric:** macro ROC-AUC over the 12 findings (ranking only; rare findings weigh as much as common ones).
- **Data:** 4,407 training studies / 24,371 series (DICOM, one file per slice, `train_series/<study>/<series>/<sop>.dcm`).
  `train_series.csv` gives per series: `Anatomical_Plane` (Sagittal/Coronal/Axial), `Fluid_Sensitive`, `Fat_Suppression`.
  Only **58 studies have gold labels**; the other 4,349 have only a free-text radiology **report** (12 languages; en, es, tr,
  hr, el, de, ru, nl, fr are the big ones). Reports: median 977 characters, max 4,743. Test: ~1,300 studies, **images only**.
- **Gold labels are image-derived and thresholded** (two MSK radiologists + adjudicator; borderline = negative). Thresholds as
  quoted by another participant from the competition overview (**not yet checked by David against the Kaggle page**):
  ACL high-grade partial/complete tear (>50 % of fibres); MCL acute high-grade/complete only; menisci: signal reaching the
  surface on ≥ 2 images or a morphologic abnormality; Medial/Lateral/PF OA: ≥ ~1 cm area of > 50 % cartilage-thickness loss in
  that compartment; Effusion and Baker's: moderate or large only; Contusion: impact marrow edema without a fracture line;
  Fracture: acute cortical break / fracture line. Gold is ~3× enriched in positives compared with the corpus.
- **Deadlines:** entry / team merge **15 Oct 2026**, final submission **22 Oct 2026**. Kaggle lets you pick 2 final submissions.
- **Leaderboard (2 Oct):** our best **0.906** public (v3 + v4 blend; see §6.5). Top of LB 0.961, 0.954+ is private work.
  The LB is crowded at ~0.94 because the **public notebooks are forks of one inference notebook with published weights**
  (§6.6). Rank at 0.882 was 3,239 / 4,750; not re-checked since.

## 2. Hard rules / constraints

- **Never send report text to hosted LLM APIs** (ChatGPT, Claude, Gemini…): the data-security rule (4.b) plausibly forbids it.
  The labeller runs an open-weights model inside Kaggle. This also applies to the assistant: **don't ask the user to paste
  report text**, and don't read files that may contain it. Numbers-only files (labels, probabilities, predictions) are fine
  and live in `data/` (git-ignored).
- **Do not use GPU P100** on Kaggle (current PyTorch has no Pascal kernels). Use **GPU T4 ×2**.
- **Do not use `nn.DataParallel`** on Kaggle's torch 2.10 + T4: 7.5 s/step instead of 0.4, kernel deaths, and
  `CUDA error: misaligned address` with channels_last. Use **one GPU** (to use the second one, run two folds in separate processes).
- **On T4 use fp16, never bf16** (bf16 is ~3× slower; ready-made 4-bit model files default to bf16 and must be switched).
- Kaggle notebook output limits: **≤ 500 files, ≤ 20 GB** → the image cache is written as shards.
- A notebook can attach **only one version of a dataset**: things that must be used together need separate datasets.
- GPU quota ~30 h/week; sessions end at 12 h (a killed run saves nothing). One training fold ≈ 33 min, five folds ≈ 2.7 h;
  the v4 labeller's full run took an unknown number of hours (its log was not shared; estimate 3–6 h).

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
  Each step gets a markdown explanation. Files he exports back from Kaggle have cosmetic differences (`# #` for blank
  markdown lines, metadata on markdown cells); leave them.
- **Test locally before delivering.** The environment is `.venv` in the project root (uv, Python 3.12; git-ignored):
  torch, transformers, timm, pydicom, jupytext, nbconvert. Convert with `.venv/bin/jupytext --to notebook` and execute with
  `.venv/bin/jupyter nbconvert --execute`, using fake data + env overrides (`KNEE_ROOT`, `OUT_DIR`, `INPUT_BASE`, …):
  - `tests/make_fake_reports.py <root> [n] [n_gold]`: fake `train.csv` with made-up reports (labeller tests);
  - `tests/make_fake_dicoms.py <root> [n]`: fake test set with tiny DICOM series (submission tests);
  - `tests/test_labeler_v4_engine.py`: the labeller's fast path against a from-scratch recomputation + OOM handling;
  - a fake image cache for training tests is built ad hoc (random arrays + `knee_preproc.py` extracted from the preprocess script).
  This Mac has ~3 GB free RAM (training's preload check then falls back to worker processes, which crash on macOS: lower
  the threshold in a test copy) and its MPS backend crashes on a plain forward pass, so **two-device code paths cannot be
  reproduced locally**. To stop a test, kill by its specific command line, never all `ipykernel` processes.
- He runs things with **Save & Run All** (background commits) and pastes back the **log** text. Commit logs don't show
  `display()` tables: **print every table as text** (`print(df.to_string())`).
- Prefers direct, honest feedback; flag mistakes plainly.
- Kaggle username in paths: `davidpiln` (datasets mount at `/kaggle/input/datasets/davidpiln/<name>/…`; competition at
  `/kaggle/input/competitions/rsna-knee-abnormality-detection`). Notebooks locate inputs by shallow search, not hard-coded paths.
- He commits directly to `main` himself; GitHub remote `david-pilny/RSNA-Knee-Abnormality-Detection`.

## 4. Pipeline and Kaggle datasets

```
competition data ──► ① report labeller v4 (Qwen2.5-32B 4-bit, 2×T4) ──► knee-report-labels-v4 (labels_v4.csv = y)
                 └─► ② preprocess cache (CPU)                       ──► knee-mri-cache (shards + knee_preproc.py = X)
                                                                        │
                     ③ training (EffNet-B0, 1×T4) ◄─────────────────────┘ ──► knee-models (latest version = v4 models)
                     ④ submission (offline) ◄── one or more model datasets + knee-mri-cache (knee_preproc.py) + knee-dicom-wheels
```

| Kaggle dataset | Contents | Made by |
|---|---|---|
| `knee-report-labels` | `labels_v3.csv`, `llm_raw_v3.csv`, `label_maps_v3.json` | v3 labeller (Qwen2.5-7B) |
| `knee-report-labels-v4` | `labels_v4.csv`, `llm_probs_v4.csv`, `label_maps_v4.json`, `gold_agreement_v4.csv`, `run_summary_v4.json` | v4 labeller |
| `knee-mri-cache` | `cache/shard_000..044.npz`, `cache/index.csv`, `cache/config.json`, `knee_preproc.py` | preprocess run |
| `knee-models` | **latest version: five v4 fold models** (`model_fold{k}.pt`, fp16) + `train_config.json`, oof/gold preds; older versions hold the v3 models | training runs |
| `knee-models-v3` | the five v3 fold models + `train_config.json` (made from the v3 training notebook version's output) | training run |
| `knee-dicom-wheels` | `wheels/*.whl` for pylibjpeg, pylibjpeg-libjpeg, pylibjpeg-openjpeg | 1-cell `pip download` notebook |

Local copies (numbers only, git-ignored `data/`): `labels_v3.csv`, `labels_v4.csv`, `llm_probs_v4.csv`, `v3/` and `v4/` with
`gold_predictions.csv` + `oof_predictions.csv`.

## 5. Files in the repo

| File | Purpose | Status |
|---|---|---|
| `rsna_knee_data_explained.py` | data explainer with interactive three.js 3D viewer | done |
| `rsna_knee_dummy_submission.py` | constant-prior submission; tests the offline pipeline | submitted → 0.500 |
| `rsna_knee_report_labeler.py` | v3 labeller (Qwen2.5-7B, one word per finding, parsed) | superseded by v4 |
| `rsna_knee_report_labeler_v4.py` | **v4 labeller** (§6.1) | full run done |
| `rsna_knee_preprocess_cache.py` | DICOM → cache shards + `knee_preproc.py` | done (98 min, 10.4 GB) |
| `rsna_knee_train_baseline.py` | training (§6.3); currently reads `labels_v4.csv`, runs all five folds | v3 and v4 five-fold runs done |
| `rsna_knee_speed_diagnostic.py` | GPU speed / memory diagnostic (found the DataParallel bug) | done |
| `rsna_knee_submission.py` | submission with rank-blended model groups (§6.4) | submitted → 0.906 |
| `analysis/*.py` | offline analysis of labels and predictions against gold (run from the project root with `PYTHONPATH=analysis .venv/bin/python analysis/<script>.py`; needs `data/`) | – |
| `tests/` | fake-data generators and the labeller engine test (§3) | – |
| `DECISION.md` | the open decision about the next step | waiting for David |
| `rsna_knee_explainer.pdf`, `rsna_knee_project_report.pdf` | explainer PDFs for the user | older state |

## 6. Component details and results

### 6.1 Report labeller v4 (`rsna_knee_report_labeler_v4.py`)
- **Why v4:** v3's prompt said "any grade / any size counts", which contradicts the host thresholds (§1), and it turned
  silence into `absent` (`not_mentioned` in 0–2 of 58 gold reports per finding).
- **Answers:** one letter per finding: `P` present and meets the threshold, `M` present but below it, `U` unclear (severity
  not stated, or hedged), `A` explicitly absent, `N` not mentioned.
- **Probabilities, not text:** the prompt ends with the start of the JSON answer; the notebook reads the model's
  probabilities for the 5 letters, appends the most likely one plus the next key, and repeats (12 findings + language =
  13 short model calls per batch instead of ~110 generated tokens). Soft label = Σ p(letter) · value. No parsing, no parse
  errors; the "letter mass" (probability on the 5 letters within the whole vocabulary) was 1.000 for every study.
- **Values:** `P` = 1, `A` = 0; `M`, `U`, `N` = share of gold-positives among gold studies with that answer, shrunk towards
  defaults (0.2 / 0.5 / 0.0) by 4 virtual studies. Measuring them did not beat the defaults on gold (0.875 vs 0.879).
- **Model:** `unsloth/Qwen2.5-32B-Instruct-bnb-4bit` (19 GB) across 2×T4 with `device_map="sequential"`; 4-bit layers
  switched to fp16; loads in ~150 s including download. The shared instruction prefix (1,409 of a median 1,747 prompt
  tokens) is computed once and reused (KV cache), with explicit position ids and left padding.
- **Safeguards:** self-tests at start-up (shortcut = full model, alone = in a padded batch, cached = recomputed, two made-up
  reports); gold studies first, then the full run only if gold macro AUC (default values) ≥ 0.84 (`RUN_MODE="auto"`);
  `TIME_LIMIT_H=10.5` stop and resume from `llm_probs_v4.csv` found among the inputs; `labels_v4.csv` only when complete.
- **Result vs gold (58):** macro AUC **0.879** (v3: 0.809). Per finding v3 → v4: ACL 0.832 → 0.990, MCL 0.836 → 0.989,
  Medial Meniscus 0.886 → 0.929, Lateral Meniscus 0.774 → 0.872, Medial OA 0.920 → 0.933, Lateral OA 0.845 → 0.839,
  PF OA 0.758 → 0.837, Effusion 0.752 → 0.869, Synovitis 0.676 → 0.774, Baker's 0.904 → 0.903, Contusion 0.713 → 0.729,
  Fracture 0.806 → 0.886. All 4,407 reports labelled, none failed.
- **Known weakness:** MCL is graded too strictly (only 1.4 % `P`; 6 of 9 gold-positive MCL studies answered `M`). Synovitis is
  `N` in 84 % of reports (gold-positive in 15 of 41 silent gold reports): reports cannot label it well.
- `labels_v4.csv` has the v3 layout plus `language_v4`; `language` is copied from `labels_v3.csv` so the folds stay the same
  (99.3 % of studies are in the same fold in the v3 and v4 training runs).
- v3 labeller, for reference: Qwen2.5-7B fp16, grades `present / minimal / uncertain / absent / not_mentioned`, tolerant
  parser, 3.8 h, 3.6 % studies with a parse error.

### 6.2 Preprocessing cache (`rsna_knee_preprocess_cache.py` → `knee_preproc.py`)
- Roles: `sag_fs` (Sagittal, fluid), `cor_fs` (Coronal, fluid), `ax_fs` (Axial, fluid); prefer fat-sat; else fallback to any
  series of that plane (flag `fallback`). All 3 roles available for 100 % of training-pool studies.
- Per series: load all slices, sort by `ImagePositionPatient` along a canonical axis, **canonical orientation**
  (flip/transpose via `ImageOrientationPatient`; verified identical output for 5 stored variants), percentile 0.5–99.5
  normalisation, square-mm pixels + pad, resample to **24 × 224 × 224 uint8**. The whole field of view is kept (no crop).
- Left/right knee NOT canonicalised; `Laterality` tag and x-position stored in `index.csv`. (Evidence from other teams is
  mixed: one flips every knee to one side, another measured a canonical-orientation change as much worse.)
- Shards: 100 studies each, keys `<StudyInstanceUID>__<role>`; `CacheReader(cache_dir).load(uid)`.
- `knee_preproc.process_study(uid, study_rows, series_root, roles, depth, img)` is used **identically** in the submission.

### 6.3 Training (`rsna_knee_train_baseline.py`)
- 2.5D multi-view: each series → 8 "RGB" images of 3 neighbouring slices (centres jitter ±1 in training) → shared
  timm `efficientnet_b0` (ImageNet-pretrained) → attention pooling over the 8 → + role embedding, × role mask → concat
  (+ mask) → MLP → 12 logits. Masked soft-label BCE (NaN labels masked).
- Augment on GPU: per-series small affine + brightness/contrast/gamma; **joint knee mirror** p=0.5 (reverse sagittal slice
  order + flip coronal/axial left-right together).
- Data: whole cache **preloaded into RAM** (15.9 GB, ~0.8 min; Kaggle T4×2 VM has ~32 GB), `num_workers=0`.
- 1 GPU, fp16 autocast, `channels_last`, `cudnn.benchmark`; AdamW lr 3e-4, wd 1e-2, 1 warm-up epoch + cosine, 10 epochs,
  batch 8 studies → **0.41 s/step, 3.2 min/epoch, ~33 min/fold**.
- Validation: 5-fold `GroupKFold` by site proxy (`language|vendor`, 31 groups; pool 4,349); the **58 gold studies are never
  trained on** and are evaluated by the fold ensemble. Validation AUC thresholds the (now continuous) labels at 0.5 and is
  **not comparable between label versions**; gold AUC is.
- Outputs: `model_fold{k}.pt`, `train_config.json` (incl. `labels_file`, which names the model group in the submission),
  `oof_predictions.csv`, `gold_predictions.csv`. The fold table and the section 10 table are also printed as text.
  A run overwrites these files, and they cover only the folds of that run.

### 6.4 Submission (`rsna_knee_submission.py`)
- Safety-net `submission.csv`; offline install of wheels; loads `knee_preproc.py` from the cache dataset; preprocess in
  chunks of 32 (joblib, 4 CPUs) → predict on GPU with **mirror TTA**; rewrites the submission every few chunks; reports role
  success / decode failures.
- **Model groups:** every attached dataset with a `train_config.json` is one group, named after its `labels_file`
  (`labels_v3`, `labels_v4`); its folds and the two views are averaged. Several groups are **blended by rank** per finding
  with `BLEND_WEIGHTS` (default: the v3 group gets weight 0 for ACL, MCL, PF OA). A wrong group name stops the commit run.
  With one group the output equals the old single-group script. Failed studies get the prior (one group) or 0.5 (blend).
- Architecture code is copied from training; **if training's `KneeNet` changes, update the submission copy too.** Groups may
  differ in backbone, roles and triplet count, but must share the cache shape (DEPTH, IMG).

### 6.5 Results so far

| Models | Gold macro AUC (58) | Public LB |
|---|---|---|
| v3 labels, 4 folds | – | 0.882 |
| v3 labels, 5 folds | 0.876 | 0.889 |
| v4 labels, 5 folds | 0.884 | **0.905** |
| v3 + v4 rank blend (v3 off for ACL / MCL / PF OA) | 0.896 | **0.906** |

Gold AUC per finding, five-fold ensembles, v3 → v4 labels: ACL 0.887 → 0.934, MCL 0.810 → 0.887, Medial Meniscus
0.909 → 0.900, Lateral Meniscus 0.778 → 0.784, Medial OA 0.966 → 0.977, Lateral OA 0.818 → 0.820, PF OA 0.799 → 0.900,
Effusion 0.952 → 0.955, Synovitis 0.740 → 0.711, Baker's 0.976 → 0.980, Contusion 0.901 → 0.901, Fracture 0.974 → 0.865.

- The label upgrade fixed ACL, MCL and PF OA; **the menisci did not move** → an image limit (resolution), not labels.
- Fracture fell on gold (18 positives), but the v4 labeller answered those studies like v3 did, and on 4,348 out-of-fold
  studies the v4 model beats the v3 model against both label sets (macro 0.842 vs 0.814 on v4 labels, 0.783 vs 0.771 on v3
  labels), Fracture included → probably mostly noise.
- Weakest findings now: Synovitis 0.71, Lateral Meniscus 0.78, Lateral OA 0.82.
- Gold noise: macro SE ≈ 0.02, one finding ≈ 0.05–0.09. Paired bootstrap, equal v3 + v4 blend minus v4: +0.004, 90 % CI
  [−0.005, +0.014]; the leaderboard gave +0.001.

### 6.6 What the public notebooks are (researched 2 Oct via public GitHub write-ups)
- Kaggle's own pages are JS-rendered and not readable by WebFetch; no Kaggle API token on this Mac.
- The ~0.94 band = forks of one inference notebook with ~13 public datasets of trained weights. Reference:
  `mattiaangeli/bend-the-knee-to-speedy-raptors-the-original` (0.940); public ceiling 0.941–0.943.
- Core model: **CoAtNet-RMLP-2, 2.5D attention-MIL, 336–384 px, 44–64 slices over 5 plane×contrast slots** ("Raptor"
  checkpoints by `dreaddevelopment`): ~0.914 LB and 0.91–0.92 on gold per checkpoint; a CoAtNet-only pipeline scored 0.939
  in ~3 h. DINOv2 (0.84) and RadImageNet ResNet50 (0.85) arms add diversity only.
- Measured by others: label/pipeline diversity pays, backbone swaps on the same labels and views do not (+0.001); more
  checkpoints of one pipeline add nothing; blend-weight tuning does nothing; mixing out-of-fold predictions into silent
  labels hurt; CoAtNet is slow on T4 (2.6 s/step reported; another team: one model in 5–7 h on 2×T4); five public label
  tables copy the gold labels verbatim; the top public forks use per-finding weights probed on the public LB.
- Sources: GitHub `NTejas-1/RSNA-Knee-Abnormality-Detection`, `TranBaDat2607/RSNA-Knee-Abnormality-Detection`,
  `ShivenKhurana1/rsna-knee-abnormality`, `Daniel766hi/RSNA`; Hugging Face blog `bishnoiyash/rsna-competetion`.
- The copy of the competition rules in one of these repos allows external data and models that are "reasonably accessible
  to all"; some public label tables were made with hosted LLMs by their authors.

## 7. Next steps

1. **David decides between the two options in `DECISION.md`:** (A) Copy & Edit the public 0.94 notebook and append our v4
   models as an extra member, or (B) stay with our own pipeline and raise the effective image resolution. My recommendation
   is A as the base and B as the experiment that makes our member stronger and more different.
2. For B, first experiment: a centre-cropped cache (same 24 × 224 × 224 shape, joint fills the frame) → five folds or fold 0
   on v4 labels → compare gold per finding (menisci, lateral OA). Then higher resolution / more series / larger backbone;
   the 15.9 GB RAM preload limits what fits (288 px × 3 roles would need ~26 GB).
3. Cheap label follow-ups, only if a labeller rerun is planned anyway: loosen the MCL wording; verify the thresholds against
   the Kaggle overview. Label values can be recomputed offline from `llm_probs_v4.csv` without the LLM.
4. Final week: runtime check, choose 2 final submissions. Don't tune to the public LB.

## 8. Lessons learned (avoid repeating)

- Measure before fixing: a 5-minute diagnostic found DataParallel after two blind fixes failed.
- **Read the organisers' label definitions before writing a labelling prompt.** v3 lost ~0.07 label AUC to wrong definitions.
- **Look at the public notebooks early.** The leaderboard band and the strongest known model class were public all along.
- Always print step-level progress (s/step, % waiting for data, RAM) and every table as text; committed logs are all we see.
- "Safety net first" in every submission; every study in try/except.
- Committed runs start with an empty `/kaggle/working`; a new dataset version made from a run's output **replaces** the
  files. Old versions stay available, but a notebook can attach only one version: use a new dataset for things used together.
- With a model split over two GPUs, `model(...)` returns its output on the first GPU while the output layer sits on the
  last: keep index tensors on the CPU. After an out-of-memory error, free and retry **outside** the `except` block.
- Blending two models that share architecture and images gives almost nothing (+0.001 LB), however different the labels.
- With 58 gold studies, per-finding differences below ~0.05 and macro differences below ~0.02 are noise; use out-of-fold
  predictions on the 4,349 report-labelled studies for a second opinion.
