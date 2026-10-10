# %% [markdown]
# # RSNA Knee: dummy submission (end-to-end pipeline test)
#
# **Goal of this notebook:** not a good score, but a *working pipeline*. It scores about **0.5 AUC**, the same as random guessing.
#
# The competition is a **code competition**: Kaggle re-runs this notebook on hidden test data (~1,300 studies),
# **offline** and within **9 hours**. A lot can go wrong in that re-run that you never see in the interactive session:
#
# | Risk | How this notebook checks it |
# |---|---|
# | wrong paths / file layout of the hidden test set | reads everything through `test.csv` and `test_series.csv`, finds the root automatically |
# | compressed DICOMs that cannot be decoded (JPEG 2000, JPEG Lossless) | decodes **every** test slice and logs failures by transfer syntax |
# | packages missing without internet | optional offline install of decoders from an attached dataset |
# | running out of the 9-hour limit | measures decoding time per study and extrapolates |
# | a crash means **no submission at all** | writes a valid `submission.csv` **first**, before anything risky |
# | wrong submission format | validates rows, columns, NaNs and value range against `sample_submission.csv` |
#
# Once this notebook submits successfully, every later model only needs to replace the prediction step.

# %% [markdown]
# ## 1. Imports and settings
#
# `IS_RERUN` tells us whether Kaggle is currently scoring the notebook on the hidden test set.
# Kaggle sets the environment variable `KAGGLE_IS_COMPETITION_RERUN` only during that scoring run.
# In the interactive session the test set is just 3 example studies.

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os, sys, time, glob, subprocess, traceback
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

IS_RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
N_JOBS = os.cpu_count() or 2
T_START = time.time()

OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_PATH = OUT_DIR / "submission.csv"

print(f"scoring re-run: {IS_RERUN} | CPUs: {N_JOBS} | python {sys.version.split()[0]}")

# %% [markdown]
# ## 2. Find the competition data
#
# Depending on how the data is attached, Kaggle mounts it at `/kaggle/input/competitions/<slug>` (newer layout)
# or `/kaggle/input/<slug>` (older layout). Instead of hard-coding one, we look for the folder that contains `test.csv`.

# %% [code] {"jupyter":{"outputs_hidden":false}}
SLUG = "rsna-knee-abnormality-detection"
def find_dirs(base, name, max_depth=4, skip=()):
    # shallow search for folders called `name` (a full recursive glob would crawl ~100k DICOM files)
    hits, frontier = [], [Path(base)]
    for _ in range(max_depth):
        nxt = []
        for d in frontier:
            try:
                for c in d.iterdir():
                    if c.is_dir() and c not in skip:
                        (hits if c.name == name else nxt).append(c)
            except (PermissionError, FileNotFoundError):
                pass
        frontier = nxt
    return hits

candidates = [os.environ.get("KNEE_ROOT", ""),
              f"/kaggle/input/competitions/{SLUG}",
              f"/kaggle/input/{SLUG}"]
ROOT = next((Path(c) for c in candidates if c and (Path(c) / "test.csv").exists()), None)
if ROOT is None:   # fallback: any attached folder holding test.csv + test_series/
    ROOT = next((d.parent for d in find_dirs("/kaggle/input", "test_series") if (d.parent / "test.csv").exists()), None)
assert ROOT is not None, f"competition data not found, tried: {candidates}"
print("data root:", ROOT)
print("contents :", sorted(p.name for p in ROOT.iterdir()))

# %% [markdown]
# ## 3. Load the CSV files
#
# - **`test.csv`**: one row per test study, the rows we must predict. During scoring, Kaggle swaps this file for the real ~1,300 studies.
# - **`test_series.csv`**: one row per series (plane, fluid-sensitive, fat suppression). Also swapped during scoring.
# - **`sample_submission.csv`**: the exact format Kaggle expects: an ID column plus 12 probability columns.
#   We take the column names and order from it rather than typing them ourselves.
# - **`train.csv`**: used only to compute label prevalence for the constant prediction.

# %% [code] {"jupyter":{"outputs_hidden":false}}
test_df   = pd.read_csv(ROOT / "test.csv")
series_df = pd.read_csv(ROOT / "test_series.csv")
sample    = pd.read_csv(ROOT / "sample_submission.csv")

ID_COL = sample.columns[0]
LABELS = list(sample.columns[1:])

print(f"test studies: {len(test_df)} | test series: {len(series_df)} | sample_submission rows: {len(sample)}")
print("ID column:", ID_COL)
print("label columns:", LABELS)
sample.head()

# %% [markdown]
# ## 4. The "model": a constant prediction per label
#
# We predict the same probability for every study: the share of positives for each label among the gold-labelled
# training studies (0.5 if no gold labels are found).
#
# **Why this scores exactly 0.5.** ROC-AUC measures *ranking*: does a positive study get a higher score than a negative one?
# If every study gets the same score, nothing is ranked, and AUC is 0.5 for every label. The actual constant does not matter.
# We use prevalence only so the numbers look sensible.

# %% [code] {"jupyter":{"outputs_hidden":false}}
prior = {lab: 0.5 for lab in LABELS}
try:
    train_df = pd.read_csv(ROOT / "train.csv", usecols=lambda c: c in LABELS or c == ID_COL)
    gold = train_df.dropna(subset=[l for l in LABELS if l in train_df.columns])
    if len(gold):
        prior.update(gold[LABELS].mean().clip(0.01, 0.99).to_dict())
    print(f"gold-labelled training studies: {len(gold)}")
except Exception as e:
    print("could not compute prevalence, using 0.5:", e)

pd.Series(prior, name="predicted probability").round(3).to_frame()

# %% [markdown]
# ## 5. Safety net: write a valid submission **now**
#
# Everything after this cell touches the DICOM files, which is where the hidden test set can surprise us.
# If a later cell crashes, the re-run still finds a valid `submission.csv` and the submission gets scored instead of failing.
#
# `make_submission` is reused at the end. It starts from `sample_submission.csv`, so the row order and columns are
# always exactly what Kaggle expects.

# %% [code] {"jupyter":{"outputs_hidden":false}}
def make_submission(preds: dict | None = None) -> pd.DataFrame:
    # preds: {StudyInstanceUID: {label: prob}}; studies without a prediction get the prior
    sub = sample[[ID_COL]].copy()
    for lab in LABELS:
        sub[lab] = prior[lab]
    if preds:
        for lab in LABELS:
            mapped = sub[ID_COL].map(lambda u: preds.get(u, {}).get(lab, np.nan))
            sub[lab] = mapped.fillna(prior[lab])
    return sub

def validate_and_write(sub: pd.DataFrame, path: Path = SUBMISSION_PATH):
    assert list(sub.columns) == list(sample.columns), "columns differ from sample_submission"
    assert len(sub) == len(sample), f"{len(sub)} rows, expected {len(sample)}"
    assert sub[ID_COL].tolist() == sample[ID_COL].tolist(), "study IDs or their order differ"
    assert not sub[LABELS].isna().any().any(), "NaN in predictions"
    assert ((sub[LABELS] >= 0) & (sub[LABELS] <= 1)).all().all(), "predictions outside [0, 1]"
    sub.to_csv(path, index=False)
    print(f"✓ wrote {path} ({len(sub)} rows × {len(sub.columns)} columns)")

validate_and_write(make_submission())

# %% [markdown]
# **Check:** do all studies in `test.csv` appear in `sample_submission.csv`? They should, since Kaggle generates both from
# the same hidden list. If not, `sample_submission.csv` is the source of truth for what gets scored.

# %% [code] {"jupyter":{"outputs_hidden":false}}
missing_in_sample = set(test_df[ID_COL]) - set(sample[ID_COL])
missing_in_test   = set(sample[ID_COL]) - set(test_df[ID_COL])
print("in test.csv but not in sample_submission:", len(missing_in_sample))
print("in sample_submission but not in test.csv:", len(missing_in_test))

# %% [markdown]
# ## 6. DICOM decoders (offline)
#
# DICOM pixel data can be stored uncompressed or compressed. This competition mixes four *transfer syntaxes*:
#
# | Transfer syntax | Can pydicom decode it alone? | Extra package |
# |---|---|---|
# | Explicit VR Little Endian (uncompressed) | yes | – |
# | Implicit VR Little Endian (uncompressed) | yes | – |
# | JPEG Lossless | **no** | `pylibjpeg` + `pylibjpeg-libjpeg`, or `gdcm` |
# | JPEG 2000 | **no** (sometimes via Pillow) | `pylibjpeg` + `pylibjpeg-openjpeg`, or `gdcm` |
#
# A submission runs **without internet**, so `pip install` from PyPI is impossible. The standard trick:
#
# 1. In a separate notebook **with internet on**, download the wheels:
#    ```
#    !pip download pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg -d /kaggle/working/wheels
#    ```
# 2. Save that notebook's output as a dataset (e.g. `knee-dicom-wheels`) and attach it to this notebook.
# 3. The cell below installs from that folder with `--no-index` (no internet needed).
#
# If the dataset is not attached, the cell does nothing and section 7 tells you whether it was needed.

# %% [code] {"jupyter":{"outputs_hidden":false}}
wheel_dirs = [str(d) for d in find_dirs(os.environ.get("WHEELS_BASE", "/kaggle/input"), "wheels", skip={ROOT})
              if glob.glob(f"{d}/*.whl")]
if wheel_dirs:
    wd = wheel_dirs[0]
    print("installing decoders offline from", wd)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", wd, "-q",
                        "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], capture_output=True, text=True)
    print(r.stdout[-500:] or "done", r.stderr[-500:])
else:
    print("no wheels dataset attached, skipping offline install")

import pydicom
print("pydicom", pydicom.__version__)
for mod in ["pylibjpeg", "libjpeg", "openjpeg", "gdcm", "PIL"]:
    try:
        __import__(mod); print(f"  {mod:10s} available")
    except ImportError:
        print(f"  {mod:10s} missing")

# %% [markdown]
# ## 7. Decode every test slice (the real test)
#
# For each test study we walk its series exactly as a real model would: list the series from `test_series.csv`, read every
# `.dcm` file, and decode the pixels. Per series we log:
#
# - number of slices, image size, transfer syntax
# - whether decoding worked, and the error if not
# - time taken
#
# Errors are caught **per series**, so one broken file never stops the run. Studies are processed in parallel on all CPUs
# with `joblib`, as a real inference pipeline would.

# %% [code] {"jupyter":{"outputs_hidden":false}}
from joblib import Parallel, delayed

SERIES_ROOT = ROOT / "test_series"
series_by_study = series_df.groupby("StudyInstanceUID")

def probe_series(study_uid: str, series_uid: str) -> dict:
    row = {"study": study_uid, "series": series_uid, "n_files": 0, "ok": False,
           "transfer_syntax": None, "shape": None, "error": None, "seconds": 0.0}
    t0 = time.time()
    try:
        files = sorted((SERIES_ROOT / study_uid / series_uid).glob("*.dcm"))
        row["n_files"] = len(files)
        if not files:
            raise FileNotFoundError("series folder missing or empty")
        shapes = Counter()
        for f in files:
            ds = pydicom.dcmread(f)
            row["transfer_syntax"] = ds.file_meta.TransferSyntaxUID.name
            shapes[ds.pixel_array.shape] += 1          # this line does the actual decoding
        row["shape"] = str(shapes.most_common(1)[0][0])
        row["ok"] = True
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    row["seconds"] = time.time() - t0
    return row

def probe_study(study_uid: str) -> list[dict]:
    if study_uid not in series_by_study.groups:
        return [{"study": study_uid, "series": None, "ok": False, "error": "no rows in test_series.csv", "seconds": 0.0}]
    return [probe_series(study_uid, s) for s in series_by_study.get_group(study_uid)["SeriesInstanceUID"]]

studies = sample[ID_COL].tolist()
t0 = time.time()
try:
    results = Parallel(n_jobs=N_JOBS, verbose=0)(delayed(probe_study)(s) for s in studies)
    probe_df = pd.DataFrame([r for rows in results for r in rows])
except Exception:
    traceback.print_exc()
    probe_df = pd.DataFrame(columns=["study", "series", "ok", "transfer_syntax", "error", "seconds", "n_files"])
decode_wall = time.time() - t0
print(f"probed {len(studies)} studies / {len(probe_df)} series in {decode_wall:.1f} s wall time")

# %% [markdown]
# ### Decoding report
#
# The table below is the most important output of this notebook. Each transfer syntax should have `failed = 0`.
# A failure here means the model would get **no images** for those series, so fix it (usually by attaching the wheels)
# before building anything else.

# %% [code] {"jupyter":{"outputs_hidden":false}}
if len(probe_df):
    report = (probe_df.assign(failed=~probe_df.ok.astype(bool))
              .groupby(probe_df.transfer_syntax.fillna("unknown (read failed)"))
              .agg(series=("series", "size"), failed=("failed", "sum"), slices=("n_files", "sum"),
                   seconds=("seconds", "sum")))
    display(report)
    errs = probe_df[~probe_df.ok.astype(bool)]
    if len(errs):
        print("\nfirst errors:")
        display(errs[["study", "series", "transfer_syntax", "error"]].head(10))
    else:
        print("all series decoded successfully ✓")

# %% [markdown]
# ### Time budget
#
# The limit is **9 hours** for the whole notebook. Here we measure decoding time per study and extrapolate to the ~1,300
# hidden test studies. In the interactive session this is based on only 3 studies, so treat it as a rough estimate.
# Model inference comes on top of this later, so decoding should stay well below the limit.

# %% [code] {"jupyter":{"outputs_hidden":false}}
N_HIDDEN = 1300
if len(probe_df) and probe_df.study.nunique():
    per_study_cpu = probe_df.groupby("study").seconds.sum()
    est_hours = per_study_cpu.mean() * N_HIDDEN / N_JOBS / 3600
    print(f"decode time per study (single core): mean {per_study_cpu.mean():.2f} s, max {per_study_cpu.max():.2f} s")
    print(f"estimated decoding for {N_HIDDEN} studies on {N_JOBS} cores: {est_hours * 60:.1f} min "
          f"({est_hours / 9:.1%} of the 9 h budget)")
    if IS_RERUN:
        print(f"actual decoding in this scoring run: {decode_wall / 60:.1f} min")

# %% [markdown]
# ## 8. Final submission
#
# A real model would fill `preds` with `{StudyInstanceUID: {label: probability}}` here. For the dummy, `preds` stays empty
# and every study gets the constant prior. `make_submission` + `validate_and_write` are exactly what the real pipeline will
# call later, so this cell never needs to change.

# %% [code] {"jupyter":{"outputs_hidden":false}}
preds = {}          # ← later: preds = run_model(studies)

sub = make_submission(preds)
validate_and_write(sub)
print(f"total runtime: {(time.time() - T_START) / 60:.1f} min")
sub.head()

# %% [markdown]
# ## 9. How to submit
#
# 1. **Settings → Internet: Off.** Code competitions reject notebooks with internet on.
# 2. **Accelerator: None** is fine for this dummy (it only needs CPU). Keep your GPU hours for training.
# 3. **Save Version → Save & Run All (Commit)**, wait for it to finish, and check that `submission.csv` is in the output.
# 4. In the competition, go to **Submit Prediction**, choose this notebook and version, and submit.
# 5. Expect about **0.5** on the public leaderboard. If the submission errors instead, look at the version's log:
#    section 7's decoding report shows what failed.
#
# **What 0.5 proves:** paths, file reading, decoders, runtime and submission format all work on the hidden data.
# From here, only section 4 (the "model") and the `preds` line in section 8 need replacing.
