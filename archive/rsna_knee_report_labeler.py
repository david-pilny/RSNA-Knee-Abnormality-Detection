# %% [markdown]
# # RSNA Knee: turning radiology reports into training labels
#
# **The problem.** Only a few dozen training studies have gold labels for the 12 findings. The other ~4,350 have only a
# free-text radiology report, in one of about a dozen languages. A model can't learn from free text, so we turn every
# report into 12 labels.
#
# **The approach.**
# 1. An open-weights multilingual LLM runs **inside this notebook** and reads each report.
#    No report text leaves Kaggle: the competition's data-security rule plausibly forbids sending it to hosted APIs
#    (ChatGPT, Claude, Gemini...).
# 2. The LLM does **not** answer yes/no. It grades each finding: `present`, `minimal`, `uncertain`, `absent`, `not_mentioned`.
# 3. The gold-labelled studies decide how each grade maps to a label. For example, does a "small effusion" (`minimal`)
#    count as Effusion = 1 for the organizers? We don't guess; we measure it on the gold set.
# 4. The output is a CSV of **soft labels** (0, 0.5, 1) for every training study, with gold labels kept wherever they exist.
#
# **Settings for this notebook:** Accelerator **GPU T4 ×2** (not P100: current PyTorch has no kernels for it),
# Internet **on** (to download the model, unless you attach it from Kaggle Models). Expect roughly 1–3 hours for all reports;
# the notebook checkpoints its progress, so a crashed session can resume.

# %% [markdown]
# ## 1. Settings
#
# | Setting | Meaning |
# |---|---|
# | `MODEL_ID` | Hugging Face model to download if none is attached. Qwen2.5-7B-Instruct is strong multilingually and fits on 2× T4 in fp16. |
# | `BACKEND` | `"transformers"` runs the LLM. `"dry_run"` skips the LLM and returns random grades, to test the rest of the notebook without a GPU. |
# | `RUN_ALL` | `False`: label only the gold studies (fast, for improving the prompt). `True`: label all training reports. |
# | `BATCH_SIZE` | Reports per GPU batch. Lower it if you run out of GPU memory. |

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os, re, json, time, glob, textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BACKEND = os.environ.get("LABELER_BACKEND", "transformers")     # "transformers" or "dry_run"
RUN_ALL = True
BATCH_SIZE = 16              # upper limit of reports per batch
TOKEN_BUDGET = 12000         # max (reports in batch × longest prompt in tokens): keeps long reports from running out of GPU memory
SHOW_REPORT_TEXT = False     # True prints example reports and disagreements. Keep False for committed runs: the log then
                             # contains no report text, so you can share it safely
MAX_NEW_TOKENS = 200
MAX_REPORT_CHARS = 6000       # very long reports are truncated (the findings are almost always in the first part)
PROMPT_VERSION = "v3"         # change when you edit the prompt; old outputs are then not reused

OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_PATH = OUT_DIR / f"llm_raw_{PROMPT_VERSION}.csv"

SLUG = "rsna-knee-abnormality-detection"
ROOT = next(Path(p) for p in [os.environ.get("KNEE_ROOT", ""), f"/kaggle/input/competitions/{SLUG}", f"/kaggle/input/{SLUG}"]
            if p and (Path(p) / "train.csv").exists())
print("data root:", ROOT, "| backend:", BACKEND, "| run all:", RUN_ALL)

# %% [markdown]
# ## 2. The reports
#
# First a look at the data: how many studies have gold labels, how long the reports are, and a few examples.

# %% [code] {"jupyter":{"outputs_hidden":false}}
train = pd.read_csv(ROOT / "train.csv")
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
LABELS = [l for l in LABELS if l in train.columns]

train["Report"] = train["Report"].fillna("").astype(str)
train["is_gold"] = train[LABELS].notna().all(axis=1)
gold = train[train.is_gold]
print(f"training studies: {len(train)} | gold-labelled: {len(gold)} | report only: {(~train.is_gold).sum()}")

lens = train.Report.str.len()
print(f"report length (characters): median {lens.median():.0f}, 95th percentile {lens.quantile(.95):.0f}, max {lens.max()}")
print(f"reports longer than {MAX_REPORT_CHARS} chars (will be truncated): {(lens > MAX_REPORT_CHARS).sum()}")

fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))
axes[0].hist(lens.clip(upper=lens.quantile(.99)), bins=60, color="#4a7ab5")
axes[0].set_title("report length (characters)"); axes[0].set_xlabel("characters")
if len(gold):
    gold[LABELS].mean().sort_values().plot.barh(ax=axes[1], color="#4a7ab5")
    axes[1].set_title(f"label prevalence in the {len(gold)} gold studies"); axes[1].set_xlim(0, 1)
plt.tight_layout(); plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Three example reports, with their gold labels if available (only when SHOW_REPORT_TEXT = True)
for _, r in ((gold if len(gold) else train).sample(3, random_state=0).iterrows() if SHOW_REPORT_TEXT else []):
    print("=" * 110)
    if r.is_gold:
        print("gold positives:", [l for l in LABELS if r[l] == 1] or "none")
    print(textwrap.shorten(r.Report, 900, placeholder=" …"))

# %% [markdown]
# ## 3. Label definitions and the prompt
#
# This is the part to iterate on. The definitions say what counts for each finding, and the grades let the gold set decide
# the borderline cases later:
#
# | Grade | Meaning | Examples |
# |---|---|---|
# | `present` | clearly described | "complete ACL tear", "moderate effusion" |
# | `minimal` | present but trivial | "small / trace / mild / grade 1", "physiological amount of fluid" |
# | `uncertain` | hedged | "possible", "suspected", "cannot be excluded", "equivocal" |
# | `absent` | explicitly normal | "ACL intact", "no effusion", "menisci normal" |
# | `not_mentioned` | the report says nothing about it | |
#
# The LLM also returns the report's **language**. That is free extra information: the language is a good proxy for the
# imaging site, which we will need later for site-grouped cross-validation.

# %% [code] {"jupyter":{"outputs_hidden":false}}
DEFINITIONS = {
    "ACL": "anterior cruciate ligament injury: sprain, partial or complete tear, rupture, or tear/insufficiency of an ACL graft. ONLY an explicitly described tear/sprain/rupture counts. Mucoid degeneration, thickening, increased or heterogeneous signal WITHOUT a stated tear, and an intact graft are absent.",
    "MCL": "medial collateral ligament injury: sprain (any grade), partial or complete tear, edema in or around the MCL from injury. Old healed thickening without acute injury is minimal.",
    "Medial Meniscus": "tear of the medial meniscus (any type: horizontal, radial, complex, bucket-handle, root, flap, oblique). ONLY an explicitly described tear counts. Degeneration, intrasubstance or grade 1-2 signal, mucoid change or extrusion WITHOUT a stated tear are absent. Post-meniscectomy change without a new tear is uncertain.",
    "Lateral Meniscus": "tear of the lateral meniscus, same rules as the medial meniscus.",
    "Medial OA": "osteoarthritis of the medial tibiofemoral compartment: cartilage thinning/loss/chondral defects, osteophytes, joint space narrowing, subchondral sclerosis/cysts/degenerative edema in the medial femur or tibia.",
    "Lateral OA": "osteoarthritis of the lateral tibiofemoral compartment, same signs as Medial OA but in the lateral femur or tibia.",
    "PF OA": "patellofemoral osteoarthritis or cartilage damage: chondromalacia patellae of any grade, cartilage softening, fissuring, thinning, defects or loss of the patella or trochlea, patellofemoral osteophytes.",
    "Effusion": "joint effusion (excess fluid in the knee joint) of any size. A 'physiological' or 'trace' amount of fluid is minimal.",
    "Synovitis": "synovitis in the broad sense: synovitis, synovial thickening, proliferation or hypertrophy, inflamed/enhancing synovium, debris or 'effusion-synovitis', Hoffa fat pad edema/synovitis. Effusion alone without any synovial description is absent.",
    "Baker's": "Baker's cyst / popliteal cyst of any size (a small one is minimal).",
    "Contusion": "bone contusion / bone bruise / post-traumatic bone marrow edema. Marrow edema attributed to osteoarthritis, degeneration or a stress reaction is absent for this finding.",
    "Fracture": "any fracture: including osteochondral, impaction, avulsion (e.g. Segond), stress or insufficiency fractures.",
}
GRADES = ["present", "minimal", "uncertain", "absent", "not_mentioned"]

SYSTEM_PROMPT = (
    "You are an expert musculoskeletal radiologist. You read one knee MRI report, which may be written in any language, "
    "and grade 12 findings.\n\nFindings:\n"
    + "\n".join(f"- {k}: {v}" for k, v in DEFINITIONS.items())
    + "\n\nGrades (use exactly these words):\n"
    "- present: the finding is described as present\n"
    "- minimal: present but described as small, trace, minimal, mild, low-grade, grade 1, or physiological\n"
    "- uncertain: possible, suspected, questionable, cannot be excluded, equivocal\n"
    "- absent: explicitly described as normal, intact, or not present\n"
    "- not_mentioned: the report does not address this finding\n\n"
    "Rules: judge only this knee and only what the report states; do not infer findings from the clinical question. "
    "Negations in any language (e.g. 'no', 'kein', 'sin', 'bez', 'pas de') mean absent.\n\n"
    "Answer with ONE JSON object and nothing else, in exactly this format (language = ISO 639-1 code of the report, "
    "each <grade> replaced by one of the five grades):\n"
    + json.dumps({"language": "xx", **{l: "<grade>" for l in LABELS}}, ensure_ascii=False)
)

# The answer is "prefilled": the prompt already ends with the start of the JSON object, so the model can only continue it.
# This forces the object format (v1 without it: Qwen answered with a bare list of values, which cannot be parsed reliably).
PREFILL = '{"language": "'

def build_messages(report: str):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Report:\n\"\"\"\n" + report[:MAX_REPORT_CHARS] + "\n\"\"\""}]

print(SYSTEM_PROMPT)
print(f"\nprompt length ≈ {len(SYSTEM_PROMPT)} characters")

# %% [markdown]
# ## 4. Parsing the answer
#
# LLMs don't always answer perfectly: extra text around the JSON, synonyms ("yes", "mild", "not mentioned"), or broken JSON.
# The parser extracts the JSON object, falls back to a regex per key, and maps synonyms onto the five grades.
# Anything it cannot read becomes `parse_error` and is excluded from training later (not silently turned into 0).

# %% [code] {"jupyter":{"outputs_hidden":false}}
SYNONYMS = {"yes": "present", "positive": "present", "true": "present", "moderate": "present", "severe": "present",
            "large": "present", "mild": "minimal", "small": "minimal", "trace": "minimal", "physiological": "minimal",
            "possible": "uncertain", "suspected": "uncertain", "equivocal": "uncertain", "unclear": "uncertain",
            "no": "absent", "negative": "absent", "false": "absent", "normal": "absent", "intact": "absent",
            "not mentioned": "not_mentioned", "not_reported": "not_mentioned", "none": "not_mentioned",
            "unknown": "not_mentioned", "n/a": "not_mentioned",
            # v3 run: the model sometimes copied the report's own word instead of a grade
            "tear": "present", "torn": "present", "rupture": "present", "ruptured": "present", "ruptur": "present",
            "ruptüre": "present", "riss": "present", "rotura": "present", "rottura": "present", "sprain": "present",
            "fracture": "present", "fraktur": "present", "fraktür": "present", "fractura": "present", "frattura": "present",
            "degeneration": "absent", "degenerated": "absent"}

_GRADE_WORDS = {**{g_: g_ for g_ in GRADES}, **{g_.replace("_", " "): g_ for g_ in GRADES}, **SYNONYMS}
_GRADE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _GRADE_WORDS), key=len, reverse=True)) + r")\b")

def norm_grade(v) -> str:
    v = str(v).strip().lower().replace("-", "_")
    v = SYNONYMS.get(v, SYNONYMS.get(v.replace("_", " "), v))
    if v in GRADES:
        return v
    m = _GRADE_RE.search(v.replace("_", " "))   # e.g. "present (complex tear)" -> present
    return _GRADE_WORDS[m.group(1)] if m else "parse_error"

def parse_output(text: str) -> dict:
    out = {"language": None, **{l: "parse_error" for l in LABELS}}
    obj = None
    lst = re.search(r"\[.*\]", text, flags=re.S)
    if lst and "{" not in text:          # bare list of values: usable only if complete and in order
        try:
            vals = json.loads(lst.group(0))
            if len(vals) == len(LABELS) + 1:
                text = json.dumps(dict(zip(["language"] + LABELS, vals)))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    if obj is None:   # fallback: pick "key": "value" pairs one by one
        obj = dict(re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', text))
    for k, v in obj.items():
        k2 = str(k).strip()
        if k2.lower() == "language":
            out["language"] = str(v).strip().lower()[:5]
        else:
            match = next((l for l in LABELS if l.lower() == k2.lower().replace("’", "'")), None)
            if match:
                out[match] = norm_grade(v)
    return out

# quick self-test
print(parse_output('Sure! {"language": "de", "ACL": "Present", "Effusion": "mild", "Baker\'s": "no"} Hope this helps.'))
print(parse_output(json.dumps(["es"] + ["absent"] * len(LABELS))))
print(parse_output('{"language": "en", "ACL": "present (complete tear)", "MCL": "not mentioned", "Effusion": "minimal/small"}'))

# %% [markdown]
# ## 5. Load the LLM
#
# The model is loaded in **fp16** and split across both T4 GPUs (`device_map="auto"`); 7B parameters × 2 bytes ≈ 15 GB.
#
# If you attached the model through **Add Input → Models** (for example Qwen2.5 7B Instruct from Kaggle Models), it is found
# under `/kaggle/input` and nothing is downloaded. Otherwise it downloads `MODEL_ID` from Hugging Face (Internet must be on).
#
# **Padding side:** for batched generation, shorter prompts are padded on the **left**, so every prompt ends right where the
# answer begins.

# %% [code] {"jupyter":{"outputs_hidden":false}}
def find_local_model(hint="qwen", max_depth=6):
    # look for a folder with config.json + tokenizer in /kaggle/input, without crawling the DICOM folders
    frontier = [Path("/kaggle/input")]
    for _ in range(max_depth):
        nxt = []
        for d in frontier:
            if d == ROOT or not d.is_dir():
                continue
            try:
                kids = list(d.iterdir())
            except OSError:
                continue
            names = {k.name for k in kids}
            if "config.json" in names and ({"tokenizer.json", "tokenizer_config.json"} & names) and hint in str(d).lower():
                return str(d)
            nxt += [k for k in kids if k.is_dir()]
        frontier = nxt
    return None

if BACKEND == "transformers":
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")   # less memory fragmentation
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    MODEL_PATH = find_local_model() or MODEL_ID
    print("loading", MODEL_PATH, "| GPUs:", torch.cuda.device_count())
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16, device_map="auto")
    except TypeError:   # older transformers
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    print(f"loaded in {time.time() - t0:.0f} s")

    def to_prompt(report):
        kw = {"tokenize": False, "add_generation_prompt": True}
        try:
            p = tok.apply_chat_template(build_messages(report), enable_thinking=False, **kw)   # Qwen3-style models
        except TypeError:
            p = tok.apply_chat_template(build_messages(report), **kw)
        return p + PREFILL

    def generate_batch(reports):
        enc = tok([to_prompt(r) for r in reports], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        return [PREFILL + t for t in tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]

else:   # dry run: random grades, only for testing the pipeline without a GPU
    rng = np.random.default_rng(0)
    def generate_batch(reports):
        return [json.dumps({"language": "en", **{l: str(rng.choice(GRADES, p=[.2, .1, .05, .25, .4])) for l in LABELS}})
                for _ in reports]
    print("DRY RUN: no LLM, grades are random")

test_out = generate_batch([train.Report.iloc[0]])[0]
print(test_out)
print("parsed:", parse_output(test_out))
assert sum(v != "parse_error" for k, v in parse_output(test_out).items() if k != "language") >= len(LABELS) - 1, \
    "the model's answer cannot be parsed: fix the prompt/parser before labelling everything"

# %% [markdown]
# ## 6. Run the labeller (with checkpoints)
#
# Order: **gold studies first**, so section 7 can evaluate as soon as possible, then the rest.
# Reports are sorted by length inside each group, so each batch has similar lengths and little padding (faster).
# Batches are capped by a token budget (long reports → smaller batches), and a batch that still runs out of GPU memory
# is split in half and retried instead of crashing the run.
#
# Every few batches the raw outputs are saved to `llm_raw_<version>.csv`. If the session dies, rerunning the notebook skips
# everything already done. To reuse outputs across sessions, save this notebook's output as a dataset and copy the file
# back into `/kaggle/working` before running.

# %% [code] {"jupyter":{"outputs_hidden":false}}
done = pd.read_csv(RAW_PATH) if RAW_PATH.exists() else pd.DataFrame(columns=["StudyInstanceUID", "raw"])
done_ids = set(done.StudyInstanceUID)

todo = train[~train.StudyInstanceUID.isin(done_ids)].copy()
if not RUN_ALL:
    todo = todo[todo.is_gold]
todo["len"] = todo.Report.str.len()
todo = todo.sort_values(["is_gold", "len"], ascending=[False, True])
print(f"already labelled: {len(done_ids)} | to do now: {len(todo)}")

# Token-aware batching: a batch of 16 short reports is fine, 16 long ones (v3 first run) ran out of GPU memory.
# Each batch is limited to TOKEN_BUDGET = (number of reports) × (longest prompt in tokens).
if BACKEND == "transformers":
    todo["ntok"] = [len(tok(to_prompt(r), add_special_tokens=False).input_ids) for r in todo.Report]
else:
    todo["ntok"] = todo["len"] // 3 + 900
print(f"prompt tokens: median {todo.ntok.median():.0f}, max {todo.ntok.max()}")

batches, cur = [], []
for idx, n in zip(todo.index, todo.ntok):
    if cur and (len(cur) + 1 > BATCH_SIZE or (len(cur) + 1) * max(n, todo.ntok.loc[cur].max()) > TOKEN_BUDGET):
        batches.append(todo.loc[cur]); cur = []
    cur.append(idx)
if cur:
    batches.append(todo.loc[cur])
print(f"{len(batches)} batches, {len(todo) / max(1, len(batches)):.1f} reports per batch on average")

OOM = torch.cuda.OutOfMemoryError if BACKEND == "transformers" else MemoryError

def generate_safe(reports):
    # if a batch still runs out of memory, split it in half and try again
    try:
        return generate_batch(reports)
    except OOM:
        if BACKEND == "transformers":
            torch.cuda.empty_cache()
        if len(reports) == 1:
            print("  one report is too long even alone, marked as parse error", flush=True)
            return [""]
        h = len(reports) // 2
        return generate_safe(reports[:h]) + generate_safe(reports[h:])

rows, t0, SAVE_EVERY, n_done = [], time.time(), 20, 0
for b, batch in enumerate(batches):
    outs = generate_safe(batch.Report.tolist())
    rows += [{"StudyInstanceUID": u, "raw": o} for u, o in zip(batch.StudyInstanceUID, outs)]
    n_done += len(batch)
    if (b + 1) % SAVE_EVERY == 0 or b == len(batches) - 1:
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True); rows = []
        done.to_csv(RAW_PATH, index=False)
        rate = n_done / (time.time() - t0)
        print(f"{n_done}/{len(todo)} reports | {rate:.2f} reports/s | ETA {(len(todo) - n_done) / rate / 60:.0f} min", flush=True)

print(f"total labelled: {len(done)} | saved to {RAW_PATH}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Parse all raw outputs into one table of grades
parsed = pd.DataFrame([{"StudyInstanceUID": u, **parse_output(r)} for u, r in zip(done.StudyInstanceUID, done.raw.astype(str))])
parsed = parsed.drop_duplicates("StudyInstanceUID", keep="last")
err_rate = (parsed[LABELS] == "parse_error").mean()
print(f"studies parsed: {len(parsed)} | outputs with at least one parse error: {(parsed[LABELS] == 'parse_error').any(axis=1).mean():.1%}")
print("parse errors per label:", err_rate[err_rate > 0].round(3).to_dict() or "none")

# What did the unparseable answers look like? Only the short grade values are shown, never report text.
bad_vals = []
for raw in done.raw.astype(str):
    pairs = dict(re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', raw))
    found = {k for k in pairs}
    for lab in LABELS:
        if lab not in found:
            bad_vals.append("<key missing>")
        elif norm_grade(pairs[lab]) == "parse_error":
            bad_vals.append(pairs[lab][:40])
if bad_vals:
    print("unparseable values (most common):")
    for v, n in pd.Series(bad_vals).value_counts().head(12).items():
        print(f"  {n:4d} × {v!r}")
    print("answers that were cut off (no closing brace):", int((~done.raw.astype(str).str.contains("}")).sum()))

grade_counts = parsed[LABELS].apply(lambda c: c.value_counts(normalize=True)).T.reindex(columns=GRADES + ["parse_error"]).fillna(0)
ax = grade_counts.plot.barh(stacked=True, figsize=(12, 5), color=["#c0392b", "#e59866", "#f4d03f", "#5dade2", "#d5d8dc", "#000000"])
ax.invert_yaxis(); ax.set_xlabel("share of studies"); ax.set_title("How the LLM graded each finding")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5)); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 7. Compare with the gold labels
#
# The crosstabs below are the core of this notebook. For each finding: rows are the LLM's grades, columns the gold label.
#
# - A good finding has `present` mostly in column 1 and `absent`/`not_mentioned` mostly in column 0.
# - The interesting rows are `minimal` and `uncertain`: they show where the organizers draw the line.
#   If most `minimal` effusions are gold 1, then "small effusion" counts as Effusion.

# %% [code] {"jupyter":{"outputs_hidden":false}}
g = gold[["StudyInstanceUID"] + LABELS].merge(parsed, on="StudyInstanceUID", suffixes=("_gold", ""))
print(f"gold studies with LLM output: {len(g)}")

ncol = 4; nrow = int(np.ceil(len(LABELS) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow))
for ax, lab in zip(axes.ravel(), LABELS):
    ct = pd.crosstab(g[lab], g[f"{lab}_gold"]).reindex(index=GRADES + ["parse_error"], columns=[0, 1]).fillna(0)
    ax.imshow(ct.values, cmap="Blues", aspect="auto")
    for (i, j), v in np.ndenumerate(ct.values):
        ax.text(j, i, int(v), ha="center", va="center", fontsize=9, color="white" if v > ct.values.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["gold 0", "gold 1"])
    ax.set_yticks(range(len(ct))); ax.set_yticklabels(ct.index, fontsize=8)
    ax.set_title(lab)
for ax in axes.ravel()[len(LABELS):]:
    ax.axis("off")
plt.suptitle("LLM grade (rows) vs gold label (columns)", y=1.0); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 8. Calibrate: which grade means which label?
#
# Every grade is mapped to a soft label. The defaults:
#
# | present | minimal | uncertain | absent | not_mentioned |
# |---|---|---|---|---|
# | 1 | 1 | 0.5 | 0 | 0 |
#
# (`not_mentioned` → 0 because radiologists reliably report positive findings, so silence usually means normal.)
#
# Then, for each finding and each borderline grade (`minimal`, `uncertain`, `not_mentioned`) separately, we look at the gold
# studies that got that grade and pick 0, 0.5 or 1, whichever is closest to their gold labels. This happens only when at
# least `MIN_SUPPORT` gold studies have that grade; otherwise the default stays. (v2 tried all combinations at once and
# changed mappings based on one or two studies, which is overfitting.)

# %% [code] {"jupyter":{"outputs_hidden":false}}
DEFAULT_MAP = {"present": 1.0, "minimal": 1.0, "uncertain": 0.5, "absent": 0.0, "not_mentioned": 0.0}
MIN_SUPPORT = 4   # gold studies needed with a grade before its mapping may change

label_maps, calib_rows = {}, []
for lab in LABELS:
    grades, y = g[lab], g[f"{lab}_gold"].astype(float)
    m = dict(DEFAULT_MAP)
    row = {"label": lab, "gold_pos": int(y.sum())}
    for grade in ["minimal", "uncertain", "not_mentioned"]:
        sel = grades == grade
        n = int(sel.sum())
        if n >= MIN_SUPPORT:
            m[grade] = min([0.0, 0.5, 1.0], key=lambda v: ((v - y[sel]) ** 2).sum())
        row[f"{grade} (n)"] = n
        row[f"{grade}→"] = m[grade]
    row["changed"] = m != DEFAULT_MAP
    label_maps[lab] = m
    calib_rows.append(row)
calib = pd.DataFrame(calib_rows)
calib

# %% [markdown]
# ### Agreement per finding
#
# - **accuracy**: soft label > 0.5 → predicted 1, compared with gold (0.5 counts as wrong, deliberately strict)
# - **AUC**: how well the soft labels *rank* gold positives above negatives, the same metric as the competition
# - **gold positives**: with only a handful of positives, a single study moves the numbers a lot. Read these as rough signals.
#
# A finding with low agreement here caps what the image model can learn for it. Fix it first: look at the disagreements in
# section 9, refine its definition in section 3, bump `PROMPT_VERSION`, and rerun with `RUN_ALL = False`.

# %% [code] {"jupyter":{"outputs_hidden":false}}
from sklearn.metrics import roc_auc_score

def to_soft(df):
    return pd.DataFrame({lab: df[lab].map(label_maps[lab]).astype(float) for lab in LABELS}, index=df.index)

g_soft = to_soft(g)
agree = []
for lab in LABELS:
    y, s = g[f"{lab}_gold"].astype(float), g_soft[lab]
    ok = s.notna()
    auc = roc_auc_score(y[ok], s[ok]) if y[ok].nunique() == 2 else np.nan
    agree.append({"label": lab, "gold_pos": int(y.sum()), "gold_n": int(ok.sum()),
                  "accuracy": ((s[ok] > 0.5).astype(int) == y[ok]).mean(), "AUC": auc})
agree = pd.DataFrame(agree).set_index("label")
print(f"mean accuracy {agree.accuracy.mean():.3f} | macro AUC {agree.AUC.mean():.3f}")

fig, ax = plt.subplots(figsize=(10, 4.5))
agree[["accuracy", "AUC"]].plot.bar(ax=ax, color=["#5dade2", "#1f4e79"])
ax.axhline(0.9, color="grey", ls=":", lw=1); ax.set_ylim(0, 1.05)
ax.set_title("LLM labels vs gold, per finding"); ax.tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.show()
agree.round(3)

# %% [markdown]
# ## 9. Read the disagreements
#
# Numbers tell you *which* finding is weak; the reports tell you *why*. For each disagreement you see the finding, the LLM's
# grade, the gold label and the report. Typical causes: a definition that differs from the organizers' (e.g. what counts as
# OA), a language-specific negation the model missed, or a finding mentioned only in the impression.

# %% [code] {"jupyter":{"outputs_hidden":false}}
SHOW = 12
dis = []
for lab in LABELS:
    wrong = (g_soft[lab] - g[f"{lab}_gold"]).abs() >= 0.5
    for i in g.index[wrong.fillna(False)]:
        dis.append((lab, g.at[i, lab], int(g.at[i, f"{lab}_gold"]), g.at[i, "StudyInstanceUID"]))
print(f"{len(dis)} disagreements in total" + (f", showing {min(SHOW, len(dis))}\n" if SHOW_REPORT_TEXT else
      " (set SHOW_REPORT_TEXT = True in an interactive session to read them)"))
if not SHOW_REPORT_TEXT:
    dis = []
reports = train.set_index("StudyInstanceUID").Report
for lab, grade, y, uid in dis[:SHOW]:
    print(f"── {lab}: LLM said '{grade}', gold = {y}")
    print(textwrap.indent(textwrap.fill(textwrap.shorten(reports[uid], 1200, placeholder=" …"), 110), "   "), "\n")

# %% [markdown]
# ## 10. Final labels for training
#
# For every training study:
# - **gold studies** keep their gold labels (they are the ground truth);
# - all other studies get the calibrated soft labels; `parse_error` stays **NaN**, so the training loss can mask it out.
#
# The file also keeps the raw grades and the report language, which is useful later for site-grouped cross-validation.
# **Save this notebook's output as a dataset** (e.g. `knee-report-labels`) to use `labels_<version>.csv` in the training notebook.

# %% [code] {"jupyter":{"outputs_hidden":false}}
soft_all = to_soft(parsed.set_index("StudyInstanceUID"))
final = train[["StudyInstanceUID", "is_gold"]].set_index("StudyInstanceUID").join(soft_all)
gold_idx = gold.set_index("StudyInstanceUID").index
final.loc[gold_idx, LABELS] = gold.set_index("StudyInstanceUID")[LABELS].astype(float)
final["language"] = parsed.set_index("StudyInstanceUID")["language"].reindex(final.index)
final = final.join(parsed.set_index("StudyInstanceUID")[LABELS].add_prefix("grade_"))
final["has_label"] = final[LABELS].notna().all(axis=1)

out_path = OUT_DIR / f"labels_{PROMPT_VERSION}.csv"
final.reset_index().to_csv(out_path, index=False)
json.dump(label_maps, open(OUT_DIR / f"label_maps_{PROMPT_VERSION}.json", "w"), indent=1)
print(f"saved {out_path}: {len(final)} studies | fully labelled {final.has_label.sum()} | gold {final.is_gold.sum()}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
cmp = pd.DataFrame({"gold studies": gold[LABELS].mean(),
                    "LLM-labelled studies": final.loc[~final.is_gold, LABELS].mean()})
cmp.plot.barh(ax=axes[0], color=["#1f4e79", "#e59866"])
axes[0].set_title("prevalence: gold vs LLM labels (soft labels averaged)"); axes[0].invert_yaxis()
final.language.fillna("?").value_counts().head(15).plot.bar(ax=axes[1], color="#4a7ab5")
axes[1].set_title("report language (from the LLM)"); axes[1].tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.show()

# %% [markdown]
# **How to read the prevalence chart:** the gold studies were picked by the organizers, so their prevalence need not match
# the rest. A large gap for one finding is still worth a look: it can mean the LLM over- or under-calls it.
#
# ## Next steps
# 1. Look at the findings with the weakest agreement in section 8, read their disagreements, refine `DEFINITIONS`,
#    set `PROMPT_VERSION = "v3"` (and so on) and `RUN_ALL = False`, and rerun on gold only (a few minutes).
# 2. When gold agreement stops improving, set `RUN_ALL = True` and label everything.
# 3. Save the output as a dataset. The training notebook reads `labels_<version>.csv` and trains on the soft labels
#    with BCE, masking NaNs.
