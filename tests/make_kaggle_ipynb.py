"""Build an importable Kaggle notebook (.ipynb) from a percent-format script, with the kernel metadata Kaggle's
papermill runner requires ("No kernel name found in notebook" otherwise).

Usage (from the project root): .venv/Scripts/python tests/make_kaggle_ipynb.py rsna_knee_submission.py [...]
Writes kaggle_upload/<name>.ipynb (git-ignored like every .ipynb).
"""
import sys
from collections import Counter
from pathlib import Path

import jupytext

for src in sys.argv[1:]:
    nb = jupytext.read(src)
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}}
    out = Path("kaggle_upload") / (Path(src).stem + ".ipynb")
    out.parent.mkdir(exist_ok=True)
    jupytext.write(nb, out, fmt="ipynb")
    kinds = Counter(c.cell_type for c in nb.cells)
    assert nb.cells[0].cell_type == "markdown", "first cell should be the markdown title"
    print(f"{out}: {len(nb.cells)} cells {dict(kinds)}")
