# Guidance for future coding models

Read [MODEL_HANDOFF.md](MODEL_HANDOFF.md) before changing code or running data.
It is the authoritative operational handoff for this repository. Follow its
links to the dataset inventory, September calibration workflow, and pending
decisions.

Important constraints:

- Raw ND2 files and completed output runs belong to the user. Never delete,
  overwrite, rename, or modify them unless Shelly explicitly requests it.
- Every analysis invocation must create a new timestamped output directory.
- Shelly chose the September method on 2026-10-04 (hybrid correction, 0–24 µg/mL
  linear calibration, Z11 main result with Z3–Z12 robustness check); see
  `docs/NEXT_STEPS.md`. Use `scripts/run_gradient_review.py` and reuse existing
  measurement caches (`--reuse`) instead of re-reading the ND2 file when possible.
- Treat ND2 timepoint indices as zero-based. Experimental time is index × 2 h,
  so `t012` is 24 h.
- Treat displayed Z numbers as one-based unless a field explicitly says
  `z_index_zero_based`.
- The September gradient contains a DIC channel; fluorescence analysis must use
  GFP and Cy5 only. The separate large DIC image is used only for context.
- Quantification must use TIFF/ND2 values, never display-scaled PNG data.
- Keep stains visible in saved images. Robust masking/statistics may exclude
  them from quantitative profiles, but this must be recorded in QC metadata.
- Before committing, run `.\.venv\Scripts\python.exe -m unittest discover -s tests -q`.
- The GitHub repository is `https://github.com/shellyefraim41/Gradientanalysis`.

Current development branch:

`exp106-2026-09-27-calibration-pilot`

September workflow implementation baseline:

`0c36700 Add calibrated September t12 pilot workflow`

Use `git log -1 --oneline` for the later documentation commit at the current
branch tip.
