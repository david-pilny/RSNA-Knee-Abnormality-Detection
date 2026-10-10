# %% [markdown]
# # RSNA Knee: report labeller v4 (stricter definitions, probabilities, bigger model)
#
# The v3 labels agree with the 58 gold studies at macro AUC **0.809**. Other teams report 0.86–0.89, and for ACL and MCL
# almost 1.0. The image model cannot be better than its labels on the findings where the report is the main source of truth,
# so this notebook rebuilds the labels. Four things change:
#
# | | v3 | v4 |
# |---|---|---|
# | **Definitions** | "any grade counts" (any effusion, any sprain, any chondromalacia) | the organisers' **severity thresholds**: only moderate/large or high-grade findings are positive, borderline ones are negative |
# | **Silence** | the model almost never answered `not_mentioned` (0–2 of 58 gold reports per finding) | "explicitly normal" and "not mentioned" are separate answers, with separate label values |
# | **Answer** | one word per finding, parsed from generated text → 3 label values (0, 0.5, 1) | the model's **probability** for each of 5 answers → a continuous soft label, no parsing, no parse errors |
# | **Model** | Qwen2.5-7B (fp16) | Qwen2.5-32B (4-bit), same family and prompt format |
#
# It also protects the GPU quota: the gold studies are labelled first (a few minutes), compared with v3, and the long run
# over all reports only starts **if the new labels are clearly better** (the "gate"). A time limit stops the run cleanly
# before Kaggle's 12-hour cut-off, and a second run continues where the first one stopped.
#
# **Settings for this notebook:** Accelerator **GPU T4 ×2**, Internet **on** (downloads the model, ~19 GB).
# No report text is printed anywhere, so the log is safe to share.

# %% [markdown]
# ## 1. Settings
#
# | Setting | Meaning |
# |---|---|
# | `MODEL_ID` | Hugging Face model. The default is a ready-made 4-bit version of Qwen2.5-32B-Instruct (19 GB, fits on 2× T4). To compare, `Qwen/Qwen2.5-7B-Instruct` (the v3 model) also works here. |
# | `RUN_MODE` | `"gold"`: only the 58 gold studies (a quick check). `"auto"`: gold first, then everything **if the gate passes**. `"all"`: everything, no gate. |
# | `GATE_MIN_AUC` | the gate: macro AUC against gold that the new labels must reach before the long run starts (v3 has 0.809) |
# | `TIME_LIMIT_H` | stop labelling after this many hours and save what is done (Kaggle kills GPU sessions at 12 h, and a killed run saves nothing) |
# | `TOKEN_BUDGET`, `BATCH_SIZE` | size of one GPU batch (reports × longest prompt). Lower `TOKEN_BUDGET` if the log shows many out-of-memory splits. |

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os, re, gc, json, time, copy, shutil, hashlib, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

T_START = time.time()
MODEL_ID = os.environ.get("LABELER_MODEL", "unsloth/Qwen2.5-32B-Instruct-bnb-4bit")
BACKEND = os.environ.get("LABELER_BACKEND", "transformers")     # "transformers" or "dry_run" (random answers, no GPU)
RUN_MODE = os.environ.get("LABELER_RUN", "auto")                # "gold" | "auto" | "all"
GATE_MIN_AUC = float(os.environ.get("LABELER_GATE", 0.84))
TIME_LIMIT_H = float(os.environ.get("LABELER_TIME_LIMIT_H", 10.5))
BATCH_SIZE = 16
TOKEN_BUDGET = int(os.environ.get("LABELER_TOKEN_BUDGET", 16000))   # upper limit; lowered automatically to what the GPUs can hold
USE_PREFIX_CACHE = os.environ.get("LABELER_PREFIX_CACHE", "1") == "1"   # reuse the instructions' computation (section 5)
COMPUTE_DTYPE = "float16"            # T4 is ~3x slower in bfloat16; change only if the self-test reports NaN
MAX_MEMORY = {0: "10.5GiB", 1: "14GiB"}   # GPU 0 is filled up to its limit, GPU 1 takes the rest: about half of the model each
MAX_REPORT_CHARS = 6000
PROMPT_VERSION = "v4"
SHOW_REPORT_TEXT = False             # keep False for committed runs: the log then contains no report text
SAVE_EVERY = 20                      # batches between checkpoints

OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT = Path(os.environ.get("INPUT_BASE", "/kaggle/input"))
SLUG = "rsna-knee-abnormality-detection"
ROOT = next(Path(p) for p in [os.environ.get("KNEE_ROOT", ""), f"{INPUT}/competitions/{SLUG}", f"{INPUT}/{SLUG}"]
            if p and (Path(p) / "train.csv").exists())

def find_input(name, max_depth=6):
    # shallow search of the attached datasets for one file (never walks the DICOM folders)
    frontier = [INPUT]
    for _ in range(max_depth):
        nxt = []
        for d in frontier:
            if SLUG in d.name or d.name in ("train_series", "test_series"):
                continue
            try:
                kids = list(d.iterdir())
            except OSError:
                continue
            for k in kids:
                if k.name == name and k.is_file():
                    return k
            nxt += [k for k in kids if k.is_dir()]
        frontier = nxt
    return None

def show(df, title=None):
    # commit logs do not show display() tables, so every table is printed as text
    if title:
        print(f"\n--- {title} ---")
    print(df.to_string())

print(f"data root: {ROOT} | backend: {BACKEND} | model: {MODEL_ID} | run mode: {RUN_MODE} | gate: {GATE_MIN_AUC}")

# %% [markdown]
# ## 2. The reports

# %% [code] {"jupyter":{"outputs_hidden":false}}
train = pd.read_csv(ROOT / "train.csv")
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
assert all(l in train.columns for l in LABELS), "label columns missing in train.csv"

train["Report"] = train["Report"].fillna("").astype(str)
train["is_gold"] = train[LABELS].notna().all(axis=1)
gold = train[train.is_gold]
lens = train.Report.str.len()
print(f"training studies: {len(train)} | gold-labelled: {len(gold)} | report only: {(~train.is_gold).sum()}")
print(f"report length (characters): median {lens.median():.0f}, 95th percentile {lens.quantile(.95):.0f}, max {lens.max()}"
      f" | longer than {MAX_REPORT_CHARS} (truncated): {(lens > MAX_REPORT_CHARS).sum()}")

# %% [markdown]
# ## 3. What counts as positive: the prompt
#
# **The gold labels were made by radiologists looking at the images, with strict thresholds**, and "on the fence" findings
# were graded negative. Other participants quote these thresholds from the competition's overview; **please compare the
# definitions below with the Overview page / pinned discussion** and correct them if the official wording differs.
# Our own v3 calibration already pointed the same way: gold treated most "minimal" MCL findings as negative and about half
# of the "minimal" effusions as negative.
#
# The model answers with one letter per finding:
#
# | Code | Meaning | Example | Label value |
# |---|---|---|---|
# | `P` | present **and meets the threshold** | "complete ACL rupture", "large effusion", "grade IV chondropathy" | 1 |
# | `M` | present but **below the threshold** | "grade 1 MCL sprain", "small effusion", "tiny Baker's cyst" | low (measured on gold) |
# | `U` | **unclear** whether the threshold is met: severity not stated, or hedged | "joint effusion", "possible tear" | middle (measured on gold) |
# | `A` | explicitly **absent** or normal | "ACL intact", "no effusion" | 0 |
# | `N` | **not mentioned** at all | (nothing about the synovium) | low (measured on gold) |
#
# The values for `M`, `U` and `N` are not guessed: section 7 measures, per finding, how often gold says "positive" for
# each answer. That makes the labels robust even where a definition is slightly off.

# %% [code] {"jupyter":{"outputs_hidden":false}}
CODES = ["P", "M", "U", "A", "N"]
CODE_WORD = {"P": "present", "M": "below_threshold", "U": "unclear", "A": "absent", "N": "not_mentioned"}

TFJ = ("high-grade cartilage loss in the {side} tibiofemoral compartment: grade 3-4 chondropathy, loss of more than half of "
       "the cartilage thickness or full-thickness loss over a larger area (about 1 cm or more), or moderate / severe / "
       "advanced osteoarthritis of that compartment")
OA_BELOW = "grade 1-2 or mild chondropathy, superficial fibrillation, a small focal defect, osteophytes alone, early or incipient degeneration"
MENISCUS = ("tear of the {side} meniscus of any type (horizontal, radial, complex, bucket-handle, root, flap: signal "
            "reaching an articular surface) or a displaced or missing fragment")
MENISCUS_BELOW = "intrasubstance or mucoid degeneration, grade 1-2 signal that does not reach the surface, fraying, extrusion without a tear"

# finding: (what meets the threshold = P, what is below the threshold = M)
DEFINITIONS = {
    "ACL": ("high-grade partial or complete tear of the anterior cruciate ligament: fibre discontinuity, more than half of "
            "the fibres torn, rupture (acute or chronic), or a torn ACL graft",
            "low-grade partial tear, sprain, mucoid degeneration, thickening or signal change without discontinuity"),
    "MCL": ("ACUTE high-grade partial or complete tear of the medial collateral ligament (grade 2-3, disrupted fibres)",
            "grade 1 or low-grade sprain, edema around an intact ligament, chronic or old thickening or scarring"),
    "Medial Meniscus": (MENISCUS.format(side="medial"), MENISCUS_BELOW),
    "Lateral Meniscus": (MENISCUS.format(side="lateral"), MENISCUS_BELOW),
    "Medial OA": (TFJ.format(side="MEDIAL"), OA_BELOW),
    "Lateral OA": (TFJ.format(side="LATERAL"), OA_BELOW),
    "PF OA": ("high-grade cartilage loss in the patellofemoral compartment (patella or trochlea): grade 3-4 chondromalacia, "
              "loss of more than half of the cartilage thickness or full-thickness loss over a larger area, or moderate / "
              "severe patellofemoral osteoarthritis", OA_BELOW),
    "Effusion": ("moderate or large joint effusion", "small, trace, minimal, mild or physiological amount of joint fluid"),
    "Synovitis": ("synovitis: inflammation or thickening of the synovium (synovial thickening, proliferation, hypertrophy "
                  "or enhancement)", "minimal or mild synovitis"),
    "Baker's": ("moderate or large Baker's (popliteal) cyst, including a ruptured one", "small or tiny Baker's cyst"),
    "Contusion": ("bone contusion / bone bruise: traumatic bone marrow edema without a fracture line",
                  "marrow edema attributed to degeneration, osteoarthritis, overload or a stress reaction"),
    "Fracture": ("acute fracture: a fracture line or cortical break, including impaction, avulsion, osteochondral, stress "
                 "and insufficiency fractures", "old healed fracture or old post-traumatic deformity"),
}

SYSTEM_PROMPT = (
    "You are an expert musculoskeletal radiologist. You read one knee MRI report, which may be written in any language, "
    "and grade 12 findings for a research dataset.\n\n"
    "The reference standard uses strict severity thresholds, and borderline findings count as negative. "
    "For each finding, P is listed first and M (present but below the threshold) second:\n"
    + "\n".join(f"- {k}: P = {p}. M = {m}." for k, (p, m) in DEFINITIONS.items())
    + "\n\nCodes (answer with exactly one capital letter per finding):\n"
    "- P: present and meets the threshold\n"
    "- M: present but clearly below the threshold\n"
    "- U: unclear whether the threshold is met: the finding is described but its severity, size or grade is not stated "
    "(for example 'joint effusion' or 'Baker's cyst' without a size, 'chondropathy' without a grade, 'partial ACL tear' "
    "without a grade), or the wording is hedged (possible, suspected, cannot be excluded), or a post-operative state is "
    "described without a statement about a new lesion (for example after partial meniscectomy)\n"
    "- A: explicitly described as absent, normal or intact\n"
    "- N: not mentioned: the report says nothing about this finding or its structure\n\n"
    "Rules:\n"
    "1. Judge only this knee and only what the report states. Do not infer findings from the clinical question or history.\n"
    "2. A statement about several structures applies to each of them: 'cruciate and collateral ligaments intact' gives "
    "ACL = A and MCL = A; 'menisci normal' gives both menisci = A; 'cartilage preserved' gives all three OA findings = A; "
    "'normal bone marrow signal' or 'no bone lesion' gives Contusion = A and Fracture = A; 'normal knee MRI' gives A everywhere.\n"
    "3. Negations in any language (for example 'no', 'kein', 'sin', 'bez', 'pas de', 'geen', 'yok') mean A.\n"
    "4. Use N only for silence, and never use A for silence. Effusion alone says nothing about the synovium: Synovitis is then N.\n"
    "5. If the findings section and the conclusion disagree, follow the conclusion.\n\n"
    "Answer with ONE JSON object and nothing else, in exactly this format (each <code> is one of P, M, U, A, N; "
    "language is the ISO 639-1 code of the report's language):\n"
    + json.dumps({**{l: "<code>" for l in LABELS}, "language": "xx"}, ensure_ascii=False)
)

# The answer is not generated freely. The prompt already ends with the start of the JSON object, and after each letter the
# notebook itself appends the next key. The model only ever chooses the letters (and the language code).
PREFILL = '{"' + LABELS[0] + '": "'
PIECES = ['", "' + l + '": "' for l in LABELS[1:]] + ['", "language": "']    # text between two answers

def build_messages(report: str):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Report:\n\"\"\"\n" + report[:MAX_REPORT_CHARS] + "\n\"\"\""}]

PROMPT_SHA = hashlib.sha1((SYSTEM_PROMPT + PREFILL + str(MAX_REPORT_CHARS)).encode()).hexdigest()[:10]
RUN_KEY = f"{MODEL_ID.split('/')[-1]}|{PROMPT_SHA}"       # outputs of another model or prompt are never mixed in
print(SYSTEM_PROMPT)
print(f"\nprompt length ≈ {len(SYSTEM_PROMPT)} characters | run key: {RUN_KEY}")

# Two made-up reports with known answers: used for the self-tests in section 5 (they are ours, so they may be printed)
SELFTEST = [
    ("MRI of the right knee. Complete rupture of the anterior cruciate ligament. Medial collateral ligament intact. "
     "Horizontal tear of the posterior horn of the medial meniscus reaching the inferior surface. Lateral meniscus normal. "
     "Cartilage preserved in all compartments. Large joint effusion. Small Baker's cyst. Bone bruise in the lateral femoral "
     "condyle. No fracture.", "PAPAAAAPNMPA"),
    ("MRT des linken Kniegelenks. Kreuzbänder und Kollateralbänder intakt. Menisken unauffällig. Retropatellar Chondropathie "
     "Grad IV mit großflächigem Knorpelverlust, Knorpel im medialen und lateralen Kompartiment regelrecht. Geringer "
     "Gelenkerguss. Keine Baker-Zyste. Kein Knochenmarködem, keine Fraktur.", "AAAAAAPMNAAA"),
]

# %% [markdown]
# ## 4. Load the LLM
#
# - **Why 4-bit:** Qwen2.5-32B in fp16 needs 65 GB of GPU memory; 2× T4 have 30 GB. Stored with 4 bits per weight it needs
#   about 19 GB and loses very little accuracy. The model is split across both GPUs, about half on each.
# - **Why force fp16:** the ready-made 4-bit files ask for `bfloat16` arithmetic, which T4 GPUs do not accelerate
#   (about 3× slower). The cell switches every quantised layer to `float16`.
# - **Disk:** the download goes to the folder with the most free space, not to `/kaggle/working` (20 GB limit).

# %% [code] {"jupyter":{"outputs_hidden":false}}
if BACKEND == "transformers":
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")   # less memory fragmentation
    if Path("/kaggle").exists() and "HF_HOME" not in os.environ:
        cands = [d for d in ["/kaggle/temp", "/tmp", str(Path.home() / ".cache")] if Path(d).is_dir()]
        best = max(cands, key=lambda d: shutil.disk_usage(d).free)
        os.environ["HF_HOME"] = f"{best}/hf"
        print(f"model cache: {os.environ['HF_HOME']} ({shutil.disk_usage(best).free / 1e9:.0f} GB free)")

    import torch
    LOAD_4BIT = "4bit" in MODEL_ID.lower()
    if LOAD_4BIT:
        try:
            import bitsandbytes
        except ImportError:
            r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes"], capture_output=True, text=True)
            print("pip install bitsandbytes:", "ok" if r.returncode == 0 else r.stderr[-400:])
    from transformers import AutoTokenizer, AutoModelForCausalLM

    CUDA, N_GPU = torch.cuda.is_available(), torch.cuda.device_count()
    DTYPE = getattr(torch, COMPUTE_DTYPE) if CUDA else torch.float32
    kw = {}
    if os.environ.get("LABELER_DEVICE_MAP"):          # tests: split a small model over two devices like on Kaggle
        kw["device_map"] = json.loads(os.environ["LABELER_DEVICE_MAP"])
    elif CUDA and N_GPU > 1 and MAX_MEMORY:
        # "auto" put 6.6 GiB on GPU 0 and 11.3 GiB on GPU 1 (first run), leaving GPU 1 only 3 GiB to work with.
        # "sequential" fills GPU 0 up to MAX_MEMORY[0] first, which gives an even split.
        kw["device_map"] = "sequential"
        kw["max_memory"] = {i: MAX_MEMORY[i] for i in range(N_GPU) if i in MAX_MEMORY}
    elif CUDA:
        kw["device_map"] = "auto"
    print(f"loading {MODEL_ID} | GPUs: {N_GPU} | dtype: {DTYPE} | 4-bit: {LOAD_4BIT}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, **kw)
    except TypeError:   # older transformers
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, **kw)
    model.eval()

    if LOAD_4BIT and CUDA:
        import bitsandbytes as bnb
        n4 = 0
        for m in model.modules():
            if isinstance(m, bnb.nn.Linear4bit):
                m.compute_dtype, m.compute_type_is_set = DTYPE, True
                qs = getattr(m.weight, "quant_state", None)
                if qs is not None:
                    qs.dtype = DTYPE
                n4 += 1
        print(f"4-bit layers switched to {DTYPE}: {n4}")
    print(f"loaded in {time.time() - t0:.0f} s" + ("" if not CUDA else " | memory: " + ", ".join(
        f"GPU{i} {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB" for i in range(N_GPU))))

    # How many prompt tokens fit into one batch? The model keeps a "KV cache" for every token and layer; it may use
    # at most half of the free memory of the GPU that holds those layers (the rest is for the computation itself).
    try:
        c = model.config
        kv_bytes = 2 * c.num_key_value_heads * (c.hidden_size // c.num_attention_heads) * torch.finfo(DTYPE).bits // 8
        layers_on = {}
        for name, dev in getattr(model, "hf_device_map", {}).items():
            if re.search(r"layers\.\d+$", name) and isinstance(dev, int):
                layers_on[dev] = layers_on.get(dev, 0) + 1
        if layers_on:
            fit = int(min(0.5 * torch.cuda.mem_get_info(dev)[0] / (n * kv_bytes) for dev, n in layers_on.items()))
            print(f"layers per GPU: {layers_on} | token budget per batch: {min(TOKEN_BUDGET, fit)} "
                  f"(setting {TOKEN_BUDGET}, memory allows {fit})")
            TOKEN_BUDGET = min(TOKEN_BUDGET, fit)
    except Exception as e:
        print(f"token budget stays at {TOKEN_BUDGET} ({type(e).__name__}: {str(e)[:100]})")
else:
    print("DRY RUN: no LLM, answers are random")

# %% [markdown]
# ## 5. Reading probabilities instead of text
#
# v3 let the model write the whole JSON answer (about 110 tokens, one after another) and parsed the text. v4 uses a
# property of language models: before writing a token, the model computes a **probability for every possible next token**.
# We read that directly.
#
# ```
# prompt … {"ACL": "      → model: P 0.03  M 0.01  U 0.02  A 0.93  N 0.01   → keep these 5 numbers, append the best letter
# … A", "MCL": "          → model: P 0.01  M 0.62  U 0.30  A 0.06  N 0.01   → and so on, 12 times, then the language
# ```
#
# What this buys:
# - **Soft labels with real information.** "62 % below threshold, 30 % unclear" becomes a label between the two, instead of
#   a hard choice. For a ranking metric like AUC, such tie-breaking is worth a lot.
# - **No parse errors**, because nothing is parsed.
# - **Speed.** 13 short model calls per batch instead of ~110, which is what makes a 32B model affordable.
#
# Two more speed measures:
# - **The instructions are computed once.** Every prompt starts with the same ~1,400 tokens of instructions. The model's
#   internal state after reading them (the "KV cache") is stored and reused for every report, so only the report itself is
#   processed each time.
# - **Batches by token budget**, shortest reports first, as in v3.
#
# **Self-tests** (the cell stops or falls back if one fails):
# 1. the shortcut "model body + output layer on the last token only" gives the same numbers as the full model;
# 2. a report gets the same probabilities alone and inside a padded batch;
# 3. with and without the stored instructions the probabilities are the same (otherwise the cache is switched off);
# 4. the two made-up reports from section 3 are graded and compared with their expected answers (printed).

# %% [code] {"jupyter":{"outputs_hidden":false}}
NL, NC = len(LABELS), len(CODES)

if BACKEND == "transformers":
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    LETTER_IDS = [enc(c) for c in CODES]
    assert all(len(x) == 1 for x in LETTER_IDS) and len({x[0] for x in LETTER_IDS}) == NC, "letter codes must be single, distinct tokens"
    LETTER_IDS = [x[0] for x in LETTER_IDS]
    PIECE_IDS = [enc(p) for p in PIECES]
    canon = enc(PREFILL + "A" + PIECES[0] + "N" + PIECES[1]) == enc(PREFILL) + [LETTER_IDS[3]] + PIECE_IDS[0] + [LETTER_IDS[4]] + PIECE_IDS[1]
    print("answer pieces tokenise exactly like free text:", canon)

    def prompt_ids(report):
        kw = {"tokenize": False, "add_generation_prompt": True}
        try:
            p = tok.apply_chat_template(build_messages(report), enable_thinking=False, **kw)   # Qwen3-style models
        except TypeError:
            p = tok.apply_chat_template(build_messages(report), **kw)
        return enc(p + PREFILL)

    BASE, HEAD = model.base_model, model.get_output_embeddings()
    DEV, HEAD_DEV = model.get_input_embeddings().weight.device, HEAD.weight.device
    PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    LETTER_T = torch.tensor(LETTER_IDS)               # stays on the CPU: index tensors on the CPU work with any device
    PIECE_T = [torch.tensor(p, device=DEV) for p in PIECE_IDS]

    def letter_logp(h):
        # last hidden states (B, d), on any device → log-probabilities of the 5 letters within the whole vocabulary.
        # With the model split over two GPUs, hidden states, output layer and results can each sit on a different GPU.
        return HEAD(h.to(HEAD_DEV)).float().log_softmax(-1).cpu()[:, LETTER_T]
    PREFIX_IDS, PREFIX_CACHE = [], None

    def expand_cache(cache, B):
        c = copy.deepcopy(cache)                 # the model appends to the cache in place: never hand out the original
        if hasattr(c, "batch_repeat_interleave"):
            c.batch_repeat_interleave(B)
            return c
        return tuple(tuple(t.repeat_interleave(B, 0) for t in layer) for layer in c)

    @torch.inference_mode()
    def grade_batch(id_lists, use_prefix=True):
        # id_lists: token ids of full prompts → probabilities (B, 12, 5), probability mass on the 5 letters (B, 12), languages
        B = len(id_lists)
        P = len(PREFIX_IDS) if (use_prefix and PREFIX_CACHE is not None) else 0
        var = [ids[P:] for ids in id_lists]
        L = max(len(v) for v in var)
        inp = torch.full((B, L), PAD_ID, dtype=torch.long)
        msk = torch.zeros((B, L), dtype=torch.long)
        for i, v in enumerate(var):                                   # pad on the left: every prompt ends at the same place
            inp[i, L - len(v):] = torch.tensor(v)
            msk[i, L - len(v):] = 1
        inp, msk = inp.to(DEV), msk.to(DEV)
        pos = (P + msk.cumsum(1) - 1).clamp(min=0)                    # real tokens continue right after the instructions
        attn = torch.cat([torch.ones((B, P), dtype=torch.long, device=DEV), msk], 1)
        past = expand_cache(PREFIX_CACHE, B) if P else None
        out = BASE(input_ids=inp, attention_mask=attn, position_ids=pos, past_key_values=past, use_cache=True)
        past, h, last = out.past_key_values, out.last_hidden_state[:, -1], pos[:, -1]
        probs = torch.zeros((B, NL, NC)); mass = torch.zeros((B, NL))
        for k in range(NL):
            logp = letter_logp(h)                                                 # (B, 5), within the whole vocabulary
            mass[:, k] = logp.exp().sum(-1).cpu()
            p = logp.softmax(-1)                                                  # renormalised over the 5 letters
            probs[:, k] = p.cpu()
            nxt = torch.cat([LETTER_T[p.argmax(-1).cpu()].to(DEV)[:, None], PIECE_T[k].expand(B, -1)], 1)
            m = nxt.shape[1]
            attn = torch.cat([attn, torch.ones((B, m), dtype=torch.long, device=DEV)], 1)
            out = BASE(input_ids=nxt, attention_mask=attn, position_ids=last[:, None] + 1 + torch.arange(m, device=DEV),
                       past_key_values=past, use_cache=True)
            past, h, last = out.past_key_values, out.last_hidden_state[:, -1], last + m
        lang_ids = HEAD(h.to(HEAD_DEV)).argmax(-1).cpu().tolist()
        langs = [re.sub(r"[^a-z]", "", tok.decode([i]).lower())[:3] or None for i in lang_ids]
        return probs.numpy(), mass.numpy(), langs

    OOM = torch.cuda.OutOfMemoryError
else:
    rng_of = lambda uid: np.random.default_rng(int(hashlib.sha1(str(uid).encode()).hexdigest()[:8], 16))
    OOM = MemoryError

# ---- tokenise every prompt once; find the shared instruction prefix
if BACKEND == "transformers":
    t0 = time.time()
    IDS = dict(zip(train.StudyInstanceUID, (prompt_ids(r) for r in train.Report)))
    TEST_IDS = [prompt_ids(r) for r, _ in SELFTEST]
    def common(a, b):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n
    pre = TEST_IDS[0][:common(TEST_IDS[0], TEST_IDS[1])]
    n_pre = min([len(pre)] + [common(pre, ids) for ids in IDS.values()])
    NTOK = {u: len(ids) for u, ids in IDS.items()}
    print(f"tokenised {len(IDS)} prompts in {time.time() - t0:.0f} s | prompt tokens: median {np.median(list(NTOK.values())):.0f}, "
          f"max {max(NTOK.values())} | shared instruction prefix: {n_pre} tokens "
          f"({n_pre / np.mean(list(NTOK.values())):.0%} of an average prompt)")
else:
    NTOK = dict(zip(train.StudyInstanceUID, train.Report.str.len() // 3 + 1000))

# ---- self-tests
if BACKEND == "transformers":
    def maxdiff(a, b):
        return float(np.abs(a - b).max())
    TOL = 0.1                    # fp16 arithmetic differs slightly between batch shapes; a real bug gives differences near 1
    with torch.inference_mode():
        ids = torch.tensor([TEST_IDS[0]], device=DEV)
        full = model(input_ids=ids).logits[0, -1].float().log_softmax(-1).cpu()[LETTER_T].exp().numpy()
        mine = letter_logp(BASE(input_ids=ids).last_hidden_state[:, -1])[0].exp().numpy()
    d1 = maxdiff(full, mine)
    print(f"self-test 1 (shortcut = full model): max difference {d1:.4f}")
    assert np.isfinite(mine).all(), f"NaN in the model output: try COMPUTE_DTYPE = 'bfloat16'"
    assert d1 < TOL, "the model-body shortcut does not reproduce the full model for this architecture"

    single = [grade_batch([x], use_prefix=False)[0][0] for x in TEST_IDS]
    padded = grade_batch(TEST_IDS, use_prefix=False)[0]
    d2 = max(maxdiff(single[i], padded[i]) for i in range(2))
    print(f"self-test 2 (alone = in a padded batch): max difference {d2:.4f}")
    if not d2 < TOL:
        BATCH_SIZE = 1
        print("  ! padding changes the result: falling back to one report per batch (slower, but exact)")

    if USE_PREFIX_CACHE and n_pre > 50:
        try:
            PREFIX_IDS = pre[:n_pre]
            with torch.inference_mode():
                PREFIX_CACHE = BASE(input_ids=torch.tensor([PREFIX_IDS], device=DEV), use_cache=True).past_key_values
            cached = grade_batch(TEST_IDS[:BATCH_SIZE], use_prefix=True)[0]
            d3 = max(maxdiff(single[i], cached[i]) for i in range(len(cached)))
            print(f"self-test 3 (stored instructions = recomputed): max difference {d3:.4f}")
            assert d3 < TOL
        except Exception as e:
            PREFIX_IDS, PREFIX_CACHE = [], None
            print(f"  ! instruction cache switched off ({type(e).__name__}: {str(e)[:120]}): slower, results unaffected")
    print("instruction cache:", "on" if PREFIX_CACHE is not None else "off")

    t0 = time.time()
    pr, ms, lg = grade_batch(TEST_IDS[:BATCH_SIZE])
    for i in range(len(pr)):
        got = "".join(CODES[j] for j in pr[i].argmax(-1))
        exp = SELFTEST[i][1]
        print(f"self-test 4, made-up report {i + 1} ({lg[i]}): expected {exp} | got {got} | "
              f"{sum(a == b for a, b in zip(exp, got))}/12 match | letter mass {ms[i].min():.2f}–{ms[i].max():.2f}")
    print(f"  ({time.time() - t0:.1f} s for this batch; 'letter mass' near 1.0 means the model wanted to answer with one of the 5 letters)")

# %% [markdown]
# ## 6. Stage 1: the gold studies
#
# The runner below is used for both stages. It keeps every result in one table (`llm_probs_v4.csv`: 5 probabilities per
# finding and study, numbers only) and saves it every few batches.
#
# **Resuming.** If a previous version of this notebook was stopped by the time limit, save its output as a dataset and
# attach it here: the notebook finds `llm_probs_v4.csv` among the inputs and only labels what is missing. Results made
# with another model or another prompt are ignored (the `run key` must match).

# %% [code] {"jupyter":{"outputs_hidden":false}}
PROBS_PATH = OUT_DIR / f"llm_probs_{PROMPT_VERSION}.csv"
PCOLS = [f"{l}__{c}" for l in LABELS for c in CODES]
MCOLS = [f"{l}__mass" for l in LABELS]

src = PROBS_PATH if PROBS_PATH.exists() else find_input(PROBS_PATH.name)
done = pd.read_csv(src) if src else pd.DataFrame(columns=["StudyInstanceUID", "run_key", "language"] + PCOLS + MCOLS)
n_all = len(done)
done = done[done.run_key == RUN_KEY].drop_duplicates("StudyInstanceUID", keep="last").reset_index(drop=True)
print(f"previous results: {src or 'none'} | usable rows (same model and prompt): {len(done)} of {n_all}")

def grade(uids):
    if BACKEND != "transformers":
        probs = np.stack([rng_of(u).dirichlet(np.ones(NC) * 0.3, size=NL) for u in uids])
        return probs, np.ones((len(uids), NL)), ["en"] * len(uids)
    try:
        return grade_batch([IDS[u] for u in uids])
    except OOM:
        pass
    # retry outside the except block: inside it, Python still holds the failed batch's tensors (via the traceback)
    gc.collect(); torch.cuda.empty_cache()
    STATS["oom"] += 1
    if len(uids) == 1:
        print("  one report is too long even alone: left unlabelled", flush=True)
        return np.full((1, NL, NC), np.nan), np.full((1, NL), np.nan), [None]
    h = len(uids) // 2
    a, b = grade(uids[:h]), grade(uids[h:])
    return np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]]), a[2] + b[2]

STATS = {"oom": 0}

def run_stage(uids, name, deadline=None):
    # label the given studies (shortest first), with checkpoints; returns (labelled, seconds, prompt tokens)
    global done
    have = set(done.StudyInstanceUID)
    uids = sorted([u for u in uids if u not in have], key=lambda u: NTOK[u])
    batches, cur, cur_max = [], [], 0
    for u in uids:
        if cur and (len(cur) + 1 > BATCH_SIZE or (len(cur) + 1) * max(cur_max, NTOK[u]) > TOKEN_BUDGET):
            batches.append(cur); cur, cur_max = [], 0
        cur.append(u); cur_max = max(cur_max, NTOK[u])
    if cur:
        batches.append(cur)
    print(f"{name}: {len(uids)} reports to label in {len(batches)} batches", flush=True)
    rows, t0, n, ntok, stopped = [], time.time(), 0, 0, False
    for b, batch in enumerate(batches):
        if deadline and time.time() > deadline:
            stopped = True
            break
        probs, mass, langs = grade(batch)
        for i, u in enumerate(batch):
            rows.append({"StudyInstanceUID": u, "run_key": RUN_KEY, "language": langs[i],
                         **dict(zip(PCOLS, np.round(probs[i].ravel(), 5))), **dict(zip(MCOLS, np.round(mass[i], 4)))})
        n += len(batch); ntok += sum(NTOK[u] for u in batch)
        if (b + 1) % SAVE_EVERY == 0 or b == len(batches) - 1:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if len(done) else pd.DataFrame(rows)
            rows = []
            done.to_csv(PROBS_PATH, index=False)
            el = time.time() - t0
            mem = "" if BACKEND != "transformers" or not CUDA else " | peak " + ", ".join(
                f"GPU{i} {torch.cuda.max_memory_allocated(i) / 2**30:.1f}G" for i in range(N_GPU))
            print(f"  {n}/{len(uids)} | {el / n:.2f} s/report | ETA {(len(uids) - n) * el / n / 60:.0f} min | "
                  f"OOM splits {STATS['oom']} | total runtime {(time.time() - T_START) / 3600:.2f} h{mem}", flush=True)
    if rows:
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if len(done) else pd.DataFrame(rows)
        done.to_csv(PROBS_PATH, index=False)
    if stopped:
        print(f"  TIME LIMIT reached: {len(uids) - n} reports of this stage are still unlabelled", flush=True)
    return n, time.time() - t0, ntok

n1, sec1, tok1 = run_stage(gold.StudyInstanceUID.tolist(), "stage 1 (gold)")

# %% [markdown]
# ## 7. Gold check: is v4 better than v3?
#
# Three tables, all printed as text so they appear in the log:
#
# 1. **Answers against gold.** For every finding and code: how many gold studies got that answer, and how many of those
#    are gold-positive (`n (positive)`). A good finding has almost only positives under `P` and almost none under `A`.
# 2. **Label values.** `P` = 1 and `A` = 0 are fixed. For `M`, `U` and `N` the value is the share of gold-positives among
#    the studies with that answer, pulled towards a default by 4 "virtual" studies so that two or three studies cannot
#    swing it. The soft label of a study is the probability-weighted average of these values.
# 3. **AUC per finding**, next to v3. Two versions for v4: with default values (nothing fitted to gold, used for the
#    gate) and with the measured values (comparable to v3's 0.809, which was also calibrated on gold).
#
# With 58 studies, one finding's AUC is uncertain by roughly ±0.05–0.09; the macro average by about ±0.02.

# %% [code] {"jupyter":{"outputs_hidden":false}}
from sklearn.metrics import roc_auc_score

DEFAULT_VALUE = {"P": 1.0, "M": 0.2, "U": 0.5, "A": 0.0, "N": 0.0}
PRIOR_STRENGTH = 4            # "virtual" studies at the default value
V3_AUC = {"ACL": 0.832, "MCL": 0.836, "Medial Meniscus": 0.886, "Lateral Meniscus": 0.774, "Medial OA": 0.920,
          "Lateral OA": 0.845, "PF OA": 0.758, "Effusion": 0.752, "Synovitis": 0.676, "Baker's": 0.904,
          "Contusion": 0.713, "Fracture": 0.806}            # v3 labels vs gold (macro 0.809)

def probs_of(frame):
    return frame[PCOLS].to_numpy(dtype=float).reshape(len(frame), NL, NC)

def soft_labels(frame, maps):
    V = np.array([[maps[l][c] for c in CODES] for l in LABELS])            # (12, 5)
    return pd.DataFrame((probs_of(frame) * V[None]).sum(-1), columns=LABELS, index=frame.index)

def auc(y, s):
    ok = ~np.isnan(s)
    return roc_auc_score(y[ok], s[ok]) if ok.sum() and len(np.unique(y[ok])) == 2 else np.nan

g = gold[["StudyInstanceUID"] + LABELS].merge(done, on="StudyInstanceUID", suffixes=("_gold", ""))
n_tried = len(g)
g = g[g[PCOLS].notna().all(axis=1)].reset_index(drop=True)
print(f"gold studies with LLM output: {len(g)} of {len(gold)} ({n_tried - len(g)} failed)")

default_maps = {l: dict(DEFAULT_VALUE) for l in LABELS}
label_maps, gate_ok, agree = default_maps, False, None
if len(g):
    Pg = probs_of(g)
    top = Pg.argmax(-1)                                                    # (n, 12) index of the most likely code
    Y = g[LABELS].to_numpy(dtype=float)
    ct, label_maps = [], {}
    for k, lab in enumerate(LABELS):
        row, m = {"finding": lab, "gold_pos": int(Y[:, k].sum())}, dict(DEFAULT_VALUE)
        for j, c in enumerate(CODES):
            sel = top[:, k] == j
            n, pos = int(sel.sum()), int(Y[sel, k].sum())
            row[c] = f"{n} ({pos})"
            if c in ("M", "U", "N"):
                m[c] = round((pos + PRIOR_STRENGTH * DEFAULT_VALUE[c]) / (n + PRIOR_STRENGTH), 3)
        ct.append(row); label_maps[lab] = m
    show(pd.DataFrame(ct).set_index("finding"), "answers against gold: n (gold-positive) per code")
    show(pd.DataFrame(label_maps).T[CODES], "label value per answer (M, U, N measured on gold)")

    s_def, s_cal = soft_labels(g, default_maps), soft_labels(g, label_maps)
    s_hard = pd.DataFrame(np.array([[default_maps[l][CODES[j]] for j in top[:, k]] for k, l in enumerate(LABELS)]).T, columns=LABELS)
    agree = pd.DataFrame({"gold_pos": Y.sum(0).astype(int),
                          "v3": pd.Series(V3_AUC),
                          "v4 best letter only": [auc(Y[:, k], s_hard[l].to_numpy()) for k, l in enumerate(LABELS)],
                          "v4 default values": [auc(Y[:, k], s_def[l].to_numpy()) for k, l in enumerate(LABELS)],
                          "v4 measured values": [auc(Y[:, k], s_cal[l].to_numpy()) for k, l in enumerate(LABELS)]},
                         index=LABELS)
    agree["change vs v3"] = agree["v4 measured values"] - agree["v3"]
    agree.loc["macro"] = agree.mean()
    agree["gold_pos"] = agree["gold_pos"].astype(int).astype(str)
    agree.loc["macro", "gold_pos"] = ""
    show(agree.round(3), "AUC against gold per finding")
    agree.to_csv(OUT_DIR / f"gold_agreement_{PROMPT_VERSION}.csv")

    macro_def, macro_cal = agree.loc["macro", "v4 default values"], agree.loc["macro", "v4 measured values"]
    gate_ok = bool(len(g) >= 20 and macro_def >= GATE_MIN_AUC)
    print(f"\nGOLD CHECK: v3 macro AUC 0.809 | v4 {macro_def:.3f} (default values) / {macro_cal:.3f} (measured values) | "
          f"gate {GATE_MIN_AUC} → {'PASSED' if gate_ok else 'NOT PASSED'}")
    print(f"letter mass on gold: mean {g[MCOLS].to_numpy().mean():.3f}, lowest {g[MCOLS].to_numpy().min():.3f} "
          f"(low values mean the model wanted to write something other than a letter)")

    fig, ax = plt.subplots(figsize=(11, 4.2))
    agree.drop("macro")[["v3", "v4 measured values"]].plot.bar(ax=ax, color=["#b0b7c3", "#1f4e79"], width=0.8)
    ax.axhline(0.9, color="grey", ls=":", lw=1); ax.set_ylim(0.4, 1.02); ax.set_title("labels vs gold: AUC per finding")
    ax.tick_params(axis="x", rotation=45); plt.tight_layout(); plt.show()

rest = train.loc[~train.is_gold & ~train.StudyInstanceUID.isin(done.StudyInstanceUID), "StudyInstanceUID"].tolist()
if n1 and rest:
    proj = sec1 / max(tok1, 1) * sum(NTOK[u] for u in rest) / 3600
    left = TIME_LIMIT_H - (time.time() - T_START) / 3600
    print(f"speed on gold: {sec1 / n1:.2f} s/report → projected {proj:.1f} h for the remaining {len(rest)} reports "
          f"({left:.1f} h left before the time limit" + ("" if proj < left else ": a second run will be needed to finish") + ")")

# %% [markdown]
# ## 8. Stage 2: all other reports
#
# Runs only if `RUN_MODE` is `"all"`, or `"auto"` and the gate passed. If the gate did not pass, nothing more is spent:
# send the three tables above and we look at which findings are still weak before trying again.

# %% [code] {"jupyter":{"outputs_hidden":false}}
go = RUN_MODE == "all" or (RUN_MODE == "auto" and gate_ok)
if go and rest:
    run_stage(rest, "stage 2 (all other reports)", deadline=T_START + TIME_LIMIT_H * 3600)
elif not rest:
    print("stage 2: nothing left to label")
else:
    print(f"stage 2 skipped (run mode '{RUN_MODE}', gate {'passed' if gate_ok else 'not passed'})")

# %% [markdown]
# ## 9. Final labels for training
#
# Same layout as `labels_v3.csv`, so the training notebook only needs the new file name:
# - **gold studies** keep their gold labels;
# - all other studies get the soft label from section 7 (a number between 0 and 1);
# - `grade_<finding>` holds the most likely answer in words; the full probabilities stay in `llm_probs_v4.csv`.
#
# **Language.** If the `knee-report-labels` dataset (with `labels_v3.csv`) is attached, the language column is copied from
# it. Training groups its folds by language and scanner vendor; identical languages keep the folds identical, so a v3 model
# and a v4 model are compared on the same split.
#
# The file is called `labels_v4.csv` only when every report was processed. Otherwise it is `labels_v4_PARTIAL.csv`, and the
# last line of the log says how to continue. A report the model could not handle (too long for the GPU) keeps empty
# labels, which training masks out, as it did for v3's parse errors.

# %% [code] {"jupyter":{"outputs_hidden":false}}
d = done.drop_duplicates("StudyInstanceUID", keep="last").set_index("StudyInstanceUID")
d = d[d[PCOLS].notna().all(axis=1)]
soft_all = soft_labels(d, label_maps)
final = train[["StudyInstanceUID", "is_gold"]].set_index("StudyInstanceUID").join(soft_all)
final.loc[gold.StudyInstanceUID.values, LABELS] = gold[LABELS].to_numpy(dtype=float)

v3_path = find_input("labels_v3.csv")
if v3_path:
    final["language"] = pd.read_csv(v3_path, usecols=["StudyInstanceUID", "language"]).set_index("StudyInstanceUID").language.reindex(final.index)
    print("language column: copied from", v3_path)
else:
    final["language"] = d.language.reindex(final.index)
    print("language column: from this run (labels_v3.csv not attached: folds may differ from earlier experiments)")
final["language_v4"] = d.language.reindex(final.index)

top_all = pd.DataFrame(probs_of(d).argmax(-1), index=d.index, columns=LABELS).apply(lambda col: col.map(lambda j: CODE_WORD[CODES[j]]))
final = final.join(top_all.add_prefix("grade_"))
final["has_label"] = final[LABELS].notna().all(axis=1)

tried = final.index.isin(done.StudyInstanceUID)
missing = int((~final.is_gold & ~tried).sum())                      # not attempted yet (gate, time limit)
failed = int((~final.is_gold & tried & ~final.has_label).sum())     # attempted, no usable answer: stays NaN, masked in training
complete = missing == 0
out_path = OUT_DIR / (f"labels_{PROMPT_VERSION}.csv" if complete else f"labels_{PROMPT_VERSION}_PARTIAL.csv")
final.reset_index().to_csv(out_path, index=False)
json.dump(label_maps, open(OUT_DIR / f"label_maps_{PROMPT_VERSION}.json", "w"), indent=1)
summary = {"model": MODEL_ID, "run_key": RUN_KEY, "run_mode": RUN_MODE, "gate_min_auc": GATE_MIN_AUC, "gate_passed": gate_ok,
           "gold_macro_auc_default": None if agree is None else round(float(agree.loc["macro", "v4 default values"]), 4),
           "gold_macro_auc_measured": None if agree is None else round(float(agree.loc["macro", "v4 measured values"]), 4),
           "studies": len(final), "labelled_by_llm": int(len(d)), "missing": missing, "failed": failed, "complete": complete,
           "runtime_h": round((time.time() - T_START) / 3600, 2)}
json.dump(summary, open(OUT_DIR / f"run_summary_{PROMPT_VERSION}.json", "w"), indent=1)

# %% [markdown]
# ### What the labels look like
#
# - **Answers per finding:** how often each code is the most likely one. `N` should now be common for findings that
#   reports often skip (synovitis, Baker's cyst) and rare for the ligaments and menisci.
# - **Positive rate per language:** a language where one finding is almost never positive, while it is common elsewhere,
#   points at a wording the model does not understand (fold 3's ACL validation AUC of 0.53 suggested such a case in v3).
# - **Against v3** (if attached): how many hard labels changed.

# %% [code] {"jupyter":{"outputs_hidden":false}}
llm = final[~final.is_gold & final.has_label]
if len(llm):
    dist = pd.DataFrame({l: llm[f"grade_{l}"].value_counts(normalize=True) for l in LABELS}).T.reindex(columns=list(CODE_WORD.values())).fillna(0)
    show((dist * 100).round(1), f"most likely answer per finding, % of {len(llm)} LLM-labelled studies")
    ax = dist.plot.barh(stacked=True, figsize=(12, 5), color=["#c0392b", "#e59866", "#f4d03f", "#5dade2", "#d5d8dc"])
    ax.invert_yaxis(); ax.set_xlabel("share of studies"); ax.set_title("How the LLM graded each finding")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5)); plt.tight_layout(); plt.show()

    cmp = pd.DataFrame({"gold studies": gold[LABELS].mean(), "v4, LLM-labelled (mean soft label)": llm[LABELS].mean(),
                        "v4, soft label > 0.5": (llm[LABELS] > 0.5).mean()})
    show((cmp * 100).round(1), "prevalence in % (gold is enriched in positives, so it should be higher)")

    by_lang = llm.assign(language=llm.language.fillna("?")).groupby("language")
    big = by_lang.size()[by_lang.size() >= 30].index
    if len(big):
        rate = (llm[LABELS] > 0.5).groupby(llm.language.fillna("?")).mean().loc[big]
        rate.insert(0, "n", by_lang.size().loc[big])
        show(pd.concat([rate[["n"]], (rate[LABELS] * 100).round(0)], axis=1).sort_values("n", ascending=False),
             "positive rate in % per language (soft label > 0.5), languages with ≥ 30 studies")

    if v3_path:
        v3 = pd.read_csv(v3_path).set_index("StudyInstanceUID").reindex(llm.index)
        rows = []
        for l in LABELS:
            ok = v3[l].notna()
            a, b = v3.loc[ok, l] > 0.5, llm.loc[ok, l] > 0.5
            rows.append({"finding": l, "v3 positive %": 100 * a.mean(), "v4 positive %": 100 * b.mean(),
                         "same hard label %": 100 * (a == b).mean(), "v3 pos → v4 neg": int((a & ~b).sum()), "v3 neg → v4 pos": int((~a & b).sum())})
        show(pd.DataFrame(rows).set_index("finding").round(1), "v4 against v3 on the LLM-labelled studies")

print("\n" + "=" * 100)
print("RESULT:", json.dumps(summary))
if complete:
    print(f"DONE: {out_path.name} is complete ({len(final)} studies, {failed} without labels). Save this version's output as a NEW dataset "
          f"(e.g. knee-report-labels-{PROMPT_VERSION}; a new version of knee-report-labels would replace the v3 files) "
          f"and point the training notebook at labels_{PROMPT_VERSION}.csv.")
elif not go:
    print("NOT LABELLED: the gate did not pass (or RUN_MODE = 'gold'). Share the tables of section 7; nothing else was spent.")
else:
    print(f"INCOMPLETE: {missing} reports are still unlabelled. Save this version's output as a dataset, attach it to this "
          f"notebook and run again: it continues from {PROBS_PATH.name}.")
print("files:", sorted(p.name for p in OUT_DIR.iterdir() if p.is_file()))
