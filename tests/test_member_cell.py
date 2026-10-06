"""Runs public/our_member_cell_code.py the way the forked public notebook does, on fake inputs (no real data).

Usage (from the project root): .venv/Scripts/python tests/test_member_cell.py <input_base> <competition_root> <work_dir>
  <input_base>: folder with the preprocessing module (knee_preproc.py + cache/config.json) and model group folders
  <competition_root>: fake competition data from tests/make_fake_dicoms.py
Cases: normal blend, weight 0 (unchanged), missing models (fails → staged file restored).
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
CELL = Path(__file__).resolve().parents[1] / "public" / "our_member_cell_code.py"
input_base, comp, work = (Path(a) for a in sys.argv[1:4])


def run_case(name, weight, input_dir):
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    ids = pd.read_csv(comp / "test.csv").StudyInstanceUID
    rng = np.random.default_rng(0)
    stage = pd.DataFrame({"StudyInstanceUID": ids, **{l: rng.random(len(ids)) for l in LABELS}})
    stage.to_csv(work / "_pipeline_stage.csv", index=False)
    os.environ.update(OUT_DIR=str(work), INPUT_BASE=str(input_dir), KNEE_ROOT=str(comp))
    src = CELL.read_text(encoding="utf-8").replace("OUR_WEIGHT = 0.25", f"OUR_WEIGHT = {weight}", 1)
    print(f"\n===== {name} =====", flush=True)
    exec(compile(src, str(CELL), "exec"), {"__name__": "__main__"})
    after = pd.read_csv(work / "_pipeline_stage.csv")
    changed = not np.allclose(after[LABELS].to_numpy(), stage[LABELS].to_numpy())
    ok = len(after) == len(stage) and list(after.StudyInstanceUID) == list(stage.StudyInstanceUID) \
        and np.isfinite(after[LABELS].to_numpy()).all()
    return ok, changed


results = {}
ok, changed = run_case("normal blend, weight 0.25", 0.25, input_base)
results["normal: valid and changed"] = ok and changed
ok, changed = run_case("weight 0", 0, input_base)
results["weight 0: unchanged"] = ok and not changed
empty = work.parent / "member_empty_input"
shutil.rmtree(empty, ignore_errors=True)
shutil.copytree(input_base, empty, ignore=shutil.ignore_patterns("model_fold*.pt", "train_config.json"))
ok, changed = run_case("no models attached (must fail safely)", 0.25, empty)
results["failure: restored unchanged"] = ok and not changed
print()
for k, v in results.items():
    print(("PASS " if v else "FAIL ") + k)
sys.exit(0 if all(results.values()) else 1)
