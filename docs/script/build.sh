#!/bin/sh
# Rebuild the script: chart data from the numbers-only files in data/, then the PDF with Tectonic.
# Output: docs/rsna_knee_script.pdf
set -e
cd "$(dirname "$0")/../.."
.venv/bin/python docs/script/make_figure_data.py
cd docs/script
tectonic -X compile main.tex --outdir ..
mv ../main.pdf ../rsna_knee_script.pdf
echo "written: docs/rsna_knee_script.pdf"
