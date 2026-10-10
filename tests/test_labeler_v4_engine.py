"""Check the v4 labeller's fast path against a slow, obviously-correct recomputation.

The notebook reads 12 answers with a stored instruction cache, left padding and explicit positions. Here every one of the
12 steps is recomputed from scratch (full prompt + answers so far, one sequence, no cache) with the full model, and the
5 letter probabilities must match.

Usage (fake data, small model on CPU):
    python tests/make_fake_reports.py /tmp/fake 12 6
    KNEE_ROOT=/tmp/fake OUT_DIR=/tmp/out INPUT_BASE=/tmp/none LABELER_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
        LABELER_RUN=gold MPLBACKEND=Agg python tests/test_labeler_v4_engine.py
"""
import runpy
from pathlib import Path

import numpy as np
import torch

nb = runpy.run_path(str(Path(__file__).resolve().parents[1] / "notebooks" / "rsna_knee_report_labeler_v4.py"))
model, grade_batch, IDS, TEST_IDS = nb["model"], nb["grade_batch"], nb["IDS"], nb["TEST_IDS"]
LETTER_IDS, PIECE_IDS, CODES = nb["LETTER_IDS"], nb["PIECE_IDS"], nb["CODES"]
assert nb["PREFIX_CACHE"] is not None, "instruction cache is off: nothing to test"

prompts = TEST_IDS + list(IDS.values())[:3]                      # different lengths → padding is exercised
fast, mass, _ = grade_batch(prompts, use_prefix=True)

worst = 0.0
with torch.inference_mode():
    for i, ids in enumerate(prompts):
        seq = list(ids)
        for k in range(len(PIECE_IDS)):
            logp = model(input_ids=torch.tensor([seq])).logits[0, -1].float().log_softmax(-1)[LETTER_IDS]
            slow = logp.softmax(-1).numpy()
            worst = max(worst, float(np.abs(slow - fast[i, k]).max()))
            assert abs(float(logp.exp().sum()) - mass[i, k]) < 1e-3
            seq += [LETTER_IDS[int(fast[i, k].argmax())]] + PIECE_IDS[k]
        print(f"prompt {i} ({len(ids)} tokens): answers {''.join(CODES[j] for j in fast[i].argmax(-1))}")
print(f"max difference fast path vs recomputation over {len(prompts)} prompts × 12 steps: {worst:.6f}")
assert worst < 1e-3

# out-of-memory handling: a batch that fails is split and retried; a single report that fails stays unlabelled
G = nb["grade"].__globals__
real, uids = G["grade_batch"], list(IDS)[:5]
want = real([IDS[u] for u in uids])[0]
def flaky(id_lists, use_prefix=True):
    if len(id_lists) > 2:
        raise torch.cuda.OutOfMemoryError("simulated")
    return real(id_lists, use_prefix)
G["grade_batch"], G["STATS"]["oom"] = flaky, 0
got = nb["grade"](uids)
assert np.abs(got[0] - want).max() < 1e-4 and len(got[2]) == 5 and G["STATS"]["oom"] >= 1
def always(id_lists, use_prefix=True):
    raise torch.cuda.OutOfMemoryError("simulated")
G["grade_batch"] = always
got = nb["grade"](uids[:2])
assert np.isnan(got[0]).all() and got[0].shape[0] == 2 and got[2] == [None, None]
G["grade_batch"] = real
print("out-of-memory split and retry: OK")
print("OK")
