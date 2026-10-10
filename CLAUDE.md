# CLAUDE.md: RSNA Knee Abnormality Detection (Kaggle 2026)

Hand-off notes for continuing this project in a new session. Last updated **10 Oct 2026**.

**Where we are:** David chose to build on the public 0.94 notebook (`DECISION.md` records the decision and its outcome). The fork
alone scores **0.943**; our own models (v5: 0.918 alone) add nothing to it (§6.5). Resolution is exhausted (320 px: +0.003 val).
**10 Oct: canonical knee side (v7, §7) tested on fold 0 — no gain (B0 +0.001, ConvNeXt −0.007 val): stopped.** Every
model-side lever (labels, resolution, side, backbone) has now been measured at ≤ +0.003. What remains: pick the two finals
(fork 0.943 + our fast v5 pipeline for the Efficiency track), and the signed-in forum reads (§6.7, §7 open items).
The public landscape moved: a public **0.949 single CoAtNet** thread exists (§6.7).

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
- **Leaderboard (10 Oct, read signed-out in Chrome):** our best **0.943** public = the forked public notebook, with or without
  our member (§6.5); **rank 1,758 / 5,658**, 14 entries. Our own pipeline alone: 0.918 (v5). Top of LB 0.964, top 49 all
  ≥ 0.958; bronze (top 10 % ≈ rank 566) probably needs ≈ 0.95+. Public LB = ~30 % of the test set (~390 studies): ±0.005 is noise.
  The LB is crowded at ~0.94 because the **public notebooks are forks of one inference notebook with published weights**
  (§6.6); since early Oct a public **0.949 single-CoAtNet** thread exists (§6.7).
- **Efficiency track** (separate prizes $7k/$6k/$5k): `Efficiency = AUC/(Benchmark − maxAUC) + Runtime/32400`, minimised on
  the private set → ≈ `−2.16·AUC + runtime_h/9`: **one hour of runtime costs 0.05 AUC**. The submission must be one of the two
  selected finals and beat `sample_submission`. Our v5 pipeline (0.918, ~1 h?) scores ≈ −1.87 vs ≈ −1.38 for the 6 h public fork.
  The host publishes a daily "Efficiency Leaderboard" notebook (ranks only). Not yet checked for our rank.

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
| forked public notebook, unchanged (our cell with `OUR_WEIGHT = 0`) | – | **0.943** |
| fork + our v3/v4 blend as extra member, `OUR_WEIGHT = 0.25` | – | 0.943 |
| fork + our member, `OUR_WEIGHT = 0.4` | – | 0.936 |
| **v5: cropped 256 px, B0 + ConvNeXt-Nano, 5 folds each, rank blend** | B0 0.904, CNXN 0.903 (training log) | **0.918** |
| fork + v5 member, `OUR_WEIGHT = 0.25` | – | 0.943 |
| fork + v5 member, `OUR_WEIGHT = 0.4` | – | 0.939 |

**Our member adds nothing to the public ensemble**: neutral at weight 0.25, harmful at 0.4. A 0.906 model that is only partly
decorrelated from a 0.943 ensemble is not strong enough to help; it would need to be around 0.92+ on its own (a single
public CoAtNet is 0.914) and different in input to pay off.

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

### 6.6 Our member cell for the public notebook (`rsna_knee_public_member_cell.py`)
- Goes into the fork of `mattiaangeli/bend-the-knee-to-speedy-raptors-the-original` as a code cell directly before its last
  (publish) cell. The plain code is also in `public/our_member_cell_code.py`; the downloaded fork is `public/bend-the-knee.ipynb`.
- Mechanics: the public notebook keeps rank-percentile predictions in `/kaggle/working/_pipeline_stage.csv` and publishes
  `submission.csv` in its last cell after integrity gates. Our cell backs the staged file up, runs our pipeline (same code as
  §6.4, both model groups, mirror TTA, rank blend), blends our ranks in with `OUR_WEIGHT`, re-ranks, writes the staged file
  back; any exception or the time limit (`OUR_TIME_LIMIT_H = 1.5`, or 20 min before the public `TIME_BUDGET`) restores the
  backup. It reads only `ROOT`, `T0`, `TIME_BUDGET` from the public notebook, all optional. Tested with a local harness
  (fake staged file + fake DICOMs): normal, weight 0, time-out, failure.
- Attach to the fork: `knee-models` (v4), `knee-models-v3`, `knee-mri-cache`, `knee-dicom-wheels`. The commit run takes ~8 min
  (3 studies, mostly model loading); **scoring a submission takes ~6 h** before a result or error appears.
- Two runs failed because a stray one-line cell containing `[code]` sat before our cell (a paste accident); check the cell
  list before committing.

### 6.7 What the public notebooks are (researched 2 Oct via public GitHub write-ups)
- Kaggle's own pages are JS-rendered and not readable by WebFetch; no Kaggle API token on this Mac.
- The ~0.94 band = forks of one inference notebook with ~13 public datasets of trained weights. Reference:
  `mattiaangeli/bend-the-knee-to-speedy-raptors-the-original` (its current version scores 0.943 for us); public ceiling 0.941–0.943.
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
- **Forum, 10 Oct (titles only; threads could not be opened signed-out):** "Reaching 0.949 with one CoAtNet: labels, native
  crops, and anatomical mirroring" (39 comments) and "Does using the 0.949 weights count as a violation?" → public 0.949
  weights exist, the 0.943 fork is probably no longer the best public base. "8 public LLM label sets vs gold-58: 11 of 12
  findings show no clear difference" (labels saturated). "Please clarify which external knee MRI datasets are allowed"
  (88 votes, active) and "are public weights trained on OAI knee MRI allowed?" → unresolved rules question about public
  weights. Pinned host threads: "Knee Abnormality Detection AI Challenge Overview" (**this is where the label thresholds
  of §1 live**; not on the Overview/Data pages) and "Use of Commercially Hosted LLMs" (the host's position on rule 4.b).
  David should read these three signed in.

## 7. Next steps

**10 Oct: v7 = canonical knee side** (`rsna_knee_train_v5.py`, default `JOBS`; submission + member cell updated, tested on fake data).
Idea: mirror augmentation (p = 0.5) + mirror TTA made the model side-blind, yet 4 of 12 targets are side-specific
(Medial/Lateral Meniscus, Medial/Lateral OA) and our weakest findings are Lateral Meniscus (0.78) and Lateral OA (0.82).
The public 0.949 thread lists "anatomical mirroring" as one of its three ingredients.
- **Side job** `{"name": "side", "side": True}` (runs first, alone, ~15 min): trains a left/right `KneeNet` (B0, 1 output,
  3 epochs) on the studies with a DICOM `Laterality` tag (cache `index.csv`; coverage unknown until the run prints it —
  the job needs ≥ 30 tagged studies per side), half of every batch mirrored with the target swapped; predicts all studies
  with the antisymmetric average `(f(x) + 1 − f(mirror x)) / 2`; prints hold-out accuracy and the sign-of-x-position
  agreement; writes `side/side.csv`, `side_model.pt`, `side_config.json`. Reused from an attached dataset in later sessions.
- **Groups `v7_b0`, `v7_cnxn`** (`canonical_side="R"`, `mirror_aug=False`, `tta=False`): left knees are mirrored in the
  loader (`need_flip`), single-view predictions. `train_config.json` carries these keys plus `side_model`; the side model
  is copied into each group folder. `SAME_KEYS` includes them, so old folds are not reused by mistake.
- **Submission / member cell:** a group with `canonical_side` loads its `side_model.pt`, decides each test study's side
  (same antisymmetric average), mirrors the left knees, predicts one view; other groups unchanged. Section 6 prints the
  side distribution and the share of uncertain studies (|p − 0.5| < 0.1).
- **Decision rule:** session 1 = side job + fold 0 of both groups (~1.3 h wall, ~2.1 GPU h). Compare with the 256 px fold 0
  (B0 0.8607, CNXN 0.8568 val, mirror TTA), especially Lateral Meniscus / Lateral OA. If the side classifier is weak
  (hold-out accuracy < ~0.97) the experiment is inconclusive, not negative. Gain ≥ +0.005 macro or ≥ +0.02 on the lateral
  findings → folds 1–4 (~4 GPU h on both T4s) → own submission (also the Efficiency-track candidate) → fork + member at 0.25.
- Local test settings (this Mac: 35 s/step at 64 px on CPU!): fake cache 48 studies at 32 px, `n_triplets=2`, 1–2 epochs.
  `tests/make_fake_cache.py` now gives every fake knee a side (bright block on the side-specific edge, `true_side.csv`),
  tags ~70 % of them (`laterality`, with a few odd `B` values) and stores `x_center`. Tested 10 Oct: training notebook
  (side job + `v7_b0` fold 0; the fake side model outputs a constant, so its p_left is 0.5 — random-init B0 after 20 CPU
  steps, the mirror logic itself was checked directly), submission notebook with the canonical group, member-cell harness
  with a canonical + a plain group (all 3 cases pass). `kaggle_upload/*.ipynb` rebuilt.
- **Kaggle run (session 1):** notebook `rsna_knee_train_v5` with inputs `knee-report-labels-v4` + `knee-mri-cache` (latest =
  256 px version), GPU T4 ×2, Internet on. Read first: the Laterality coverage line (section 3), then the side job's
  hold-out accuracy; if coverage is tiny the job stops with "too few tagged studies" and a side heuristic is needed instead.
- **Result (10 Oct, 1.32 h wall / 2.1 GPU h): stop rule fired.** Side classifier is reliable: Laterality tagged on 49.8 %
  of studies (1,088 L / 1,108 R), hold-out accuracy **0.988** (AUC 0.989), 0.1 % of untagged studies uncertain, 11 min;
  the sign of the x-position agrees with the tag in only 94 % (not a usable shortcut). Fold 0 vs the 256 px run (same folds):
  v7_b0 val **0.8616** (no TTA) vs 0.8607 (TTA), gold 0.889 vs 0.886 — a wash: canonical single-view = old model with TTA.
  v7_cnxn val **0.8497** vs 0.8568, gold 0.885 vs 0.898 — clearly worse (mirror augmentation regularised the larger
  backbone). Gold Lateral Meniscus B0 0.876 → 0.850; blend gold 0.903 → 0.8965. Side-blindness does not limit the lateral
  findings. Folds 1–4 not run. Output of the run: `side/`, `v7_b0/`, `v7_cnxn/` (fold 0 only) — keep as a dataset only if
  the side classifier is ever needed for something else. Lesson for §8: the v7 code paths stay in the notebook (defaults
  `mirror_aug=True, tta=True, canonical_side=None` reproduce the old behaviour); set `JOBS` back before any other run.

**Decided 5 Oct: option (a), go all in on our own member** (30 GPU h left this week), then blend it into the fork the same
way as before (§6.6). Files for it (tested locally on fake data, Windows `.venv` now set up with `uv`):
- `rsna_knee_preprocess_cache_v5.py` → dataset **`knee-mri-cache-v5`**: knee cropped to a fixed 140 mm window (centre of the
  tissue box on the slice-average), 256 px (0.55 mm/px), anti-aliased resize; `tissue_outside` per series in `index.csv`.
  `IMG` / `CROP_MM` env-overridable; if the size estimate exceeds 18 GB, rerun with `IMG=240`. Crop test: `tests/test_preproc_v5_crop.py`.
- `rsna_knee_train_v5.py` → dataset `knee-models-v5`: unpacks the cache once into `X.npy` on local disk (memmap), runs
  **jobs** (`{name, fold, overrides}`) as separate processes, **one per GPU, both T4s busy**; a `bench` job times backbones.
  Every `name` is one group folder with `train_config.json` (incl. `group_name`, `crop_mm`), oof/gold predictions (mirror TTA).
  Fake inputs: `tests/make_fake_cache.py`.
- Submission and member cell now name groups by `group_name`; `BLEND_WEIGHTS` / `OUR_BLEND_WEIGHTS` default to `{}`.
  A v5 submission attaches `knee-mri-cache-v5` (not the old cache) + the v5 model dataset; old 224 px groups can't be mixed in.

**Session 1 result (5 Oct, 1.2 GPU h; the v5 cache was saved as a new *version* of `knee-mri-cache`, not a new dataset;
old 224 px notebooks must pin the older version):** fold 0, mirror TTA. B0 (44 min/fold, 0.53 s/step): val 0.861,
gold 0.886 (single model = old five-fold v4 ensemble; Lateral Meniscus 0.784 → 0.876). ConvNeXt-Nano (62 min/fold):
val 0.857, gold 0.898, still improving at epoch 10. **Rank blend of the two single models: gold 0.903** (old best 0.896).
Bench at 256 px (s/step): B0 0.52; B2 1.01, ConvNeXt-Tiny 1.29 and EffNetV2-S 1.23 only with grad checkpointing.
**Own v5 submission: 0.918 public** (commit run 1.3 min for 3 studies; `kaggle_upload/*.ipynb` built by `tests/make_kaggle_ipynb.py`, imported with File → Import Notebook; detach `knee-models-v3`, the shape check stops on 224 px groups). Member cell: `OUR_TIME_LIMIT_H = 2.0`; harness `tests/test_member_cell.py`. Next: fork + v5 member at 0.25 and 0.4.
Session 2: folds 1–4 of both groups; fold 0 reused from the attached session-1 output (reuse step in section 6 copies
fold files when group name + settings match).

**6 Oct, plan A (one more shot, stop rule):** 320 px / 130 mm cache in two parts (`PART="1/2"`, `"2/2"` → new datasets `knee-mri-cache-320a`, `-320b`; training joins all attached parts and checks they are complete); groups `v6_b0` (grad ckpt, ~80 min/fold) and `v6_cnxn` (grad ckpt, 15 epochs, ~155 min/fold). Session 1 = fold 0 of both (~2.6 h): go on only if clearly above the 256 px fold 0 (B0 0.8607, CNXN 0.8568 val). Then folds 1-4 (~7.8 h) → own submission; blend into the fork only if own LB ≥ ~0.925, otherwise stop and pick finals (plain fork + fork with v5 at 0.25). GPU quota on 6 Oct: 23 h.
**320 px session 1 result (3.67 GPU h, more than the 2.6 h estimate: ConvNeXt 1.83 s/step, 216 min/fold; B0 1.12 s/step, 90 min/fold; data waiting 0 %):** fold-0 val (TTA) B0 0.8607 → 0.8648 (+0.004), ConvNeXt 0.8568 → 0.8586 (+0.002); gold B0 0.886 → 0.888, ConvNeXt 0.898 → 0.899, blend 0.903 → 0.903. Below the +0.005 stop rule; folds 1-4 would cost ~10 GPU h for maybe +0.003 own LB. 320 px datasets `knee-mri-cache-320a/-320b` exist.

Plan: (1) CPU: build `knee-mri-cache-v5` (~1.5–2 h). (2) GPU session 1: bench + B0 fold 0 + ConvNeXt-Nano fold 0 in parallel;
compare fold-0 val AUC with the old 224 px v4 B0 fold 0 (same folds). (3) Winner × 5 folds (~3 rounds on 2 GPUs).
(4) Own submission alone; it needs ~0.92 before the fork blend can help. (5) Fork + member at 0.25.
Further levers if time allows: larger backbone, more epochs, a 4th series (sagittal non-fat-sat PD for menisci).

**Old decision text (5 Oct, before choosing):** with 17 days left, either
- **(a) make our member strong enough to matter** — the only remaining route to beat 0.943 with our own work. Judge it on our
  own-pipeline leaderboard score first: it needs roughly 0.92+ alone before it can help the public ensemble. Steps, in order:
  centre-cropped cache (same 24 × 224 × 224 shape, joint fills the frame; CPU only, ~1.5 h) → five folds on v4 labels
  (2.7 GPU h) → own submission; then higher resolution / more series / larger backbone if the crop helps. RAM limit: the
  training preload holds 15.9 GB at 224 px; 288 px × 3 roles would need ~26 GB of the ~32 GB.
- **(b) stop here** and choose the final two submissions: the unchanged fork (0.943) and the 0.25 variant (equal score, a
  hedge that differs slightly). Don't tune `OUR_WEIGHT` further; the public LB cannot resolve it and that is how forks overfit.

Other open items:
- Verify the host label thresholds (§1): they are in the pinned forum thread "Knee Abnormality Detection AI Challenge
  Overview", not on the Overview/Data pages — still unchecked (needs a signed-in session).
- Efficiency track (§1): find our rank in the host's efficiency-leaderboard notebook and the full-test runtime of the v5
  submission (shown on the submissions page). **Recommendation (10 Oct): finals = unchanged fork (0.943) + own v5
  submission (0.918) for the Efficiency track** instead of the fork+0.25 hedge (same private score, no upside). If the
  v5 runtime is dominated by the models rather than DICOM decoding, a B0-only, no-TTA variant (v7 showed single-view B0
  loses nothing) is the cheapest way to cut it; check the runtime split in the submission log first.
- Check the Code tab sorted by score for a public notebook > 0.943 (the 0.949 thread) and the external-data rules thread
  before building finals on new public weights.
- Housekeeping: branch `blend-submission-and-notes` carries everything since 2 Oct and is not merged into `main`.
- `rsna_knee_report_labeler_v4.py` MCL wording is too strict; only relevant if a labeller rerun happens.
- GPU quota: the labeller and two five-fold runs used most of one week; check what is left before planning (a).
- Final week: runtime check of the chosen submissions; pick 2 final submissions on Kaggle before 22 Oct.

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
- A 0.906 model adds nothing to a 0.943 ensemble (neutral at weight 0.25, −0.007 at 0.4): an extra member must be close to
  the ensemble's strength and different in input to help.
- Submitting a version whose commit run failed wastes a submission: the scoring run fails too, but only after ~6 h.
- With 58 gold studies, per-finding differences below ~0.05 and macro differences below ~0.02 are noise; use out-of-fold
  predictions on the 4,349 report-labelled studies for a second opinion.
- Mirror augmentation is regularisation, not just side-blindness: removing it (canonical side, v7) cost ConvNeXt-Nano
  0.007 val AUC and gained B0 nothing. A one-fold, ~2 GPU h test with a written stop rule settled this cheaply.
