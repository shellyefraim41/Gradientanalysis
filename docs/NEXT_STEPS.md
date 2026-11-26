# Pending decisions and next steps

## Decisions made on 2026-10-04 (Shelly)

After the five-timepoint comparison (`outputs/2026_09_27_exp106_try4_gradient_review_5tp/`):

- **Correction:** hybrid = calibration FFC + residual overlap quadratic, one correction
  per channel and Z pooled over all timepoints. Held-out seam mismatch ≈1–3% (FFC only
  ≈5–10%, none ≈12–18%).
- **Concentration conversion:** straight line through the 0–24 µg/mL standards with an
  intercept. The 32 µg/mL standard is excluded automatically when every profile value
  is below the 24 µg/mL standard (true for this dataset: brightest data ≈50% of it).
  Piecewise interpolation is no longer primary: with one acquisition per standard it
  inflated slopes by ≈25%. Line-through-zero is reported alongside (≈5% lower slopes).
- **Depth:** main result = single plane Z11 (one-based; nominal +90 µm above the glass,
  homeIndex=1, bottom-to-top). Robustness check = mean of Z3–Z12. The full-stack
  integration is no longer used: Z14–Z15 is a boundary (membrane underside, curved,
  ~40 µm lower near the bridge), and planes above it are not liquid layer data.
  Standards share the Z reference, so each depth is calibrated with the same planes.
- **Stain:** fixed objective rule (`gradient_review.STAIN_RULE`); stain columns are
  excluded from fits only, always plotted, and unmasked fits are saved for comparison.
- **Bridge:** detected from the large DIC image (≈2.06–4.03 mm, ≈2 mm as designed) and
  marked on every profile plot; fit tables state midpoint position relative to it.
- **Flagged tanh fits** are kept in the main results and additionally analysed with
  model-free slopes and a relaxed-plateau refit in a separate subfolder. Slopes are
  robust to the plateau limits; midpoints of flagged fits are not determined.
- The experiment tests whether the device is diffusion-only (no cells). Profiles suggest
  net flow; candidate causes (Shelly): stage movement, membrane holes. July data are
  analysed the same way for comparison.

Workflow: `scripts/run_gradient_review.py` (module `gradient_analysis/gradient_review.py`,
measurement layer `gradient_analysis/correction_comparison.py`).

## 2026-10-08 dataset (Exp106 try5), analysed 2026-10-11

- `grad.nd2`: 31 timepoints at non-uniform intervals (real times from frame timestamps:
  0, 0.11, 0.28, 0.45 h, every 0.5 h to 1.95 h, then every 2 h to 49.95 h), 14 positions,
  16 Z planes (glass at Z2), **10x objective (NA 0.45, 0.6455 µm pixels)**.
- Calibration standards (file names in µg/mL; `calibration_filename_unit: "ug_per_ml"`)
  and `32withdevice.nd2` are **20x**. Shelly chose: overlap-quadratic correction only,
  intensity units (no µg/mL, no FFC) for this dataset; 20x files used only to explain
  methods. To add µg/mL later, image standards at 10x, ideally in the device.
- 32 µg/mL with the device is ~40% (GFP) / ~37% (Cy5) of 32 µg/mL in a regular well at Z11;
  most likely less out-of-focus light under the 150 µm chamber + black membrane. This
  implies regular-well standards over-estimate signal per µg/mL for device measurements.
- Background: blank median = 100 in every plane; dye-free gradient regions at 0 h = 100.
  Constant 100 used.
- Bridge DIC: `Exp106_try5_largeimag10X_area_0001.nd2` (does not cover the first 0.68 mm).
- Config: `exp106_2026_10_08_try5_config.json`; outputs in
  `outputs/2026_10_08_exp106_try5_full_experiment/`.

## Superseded: decision that was required before the full September run

Shelly should choose the final correction policy after reviewing the t012 pilot.
The current evidence supports testing a hybrid, but that choice has not yet been
approved for all 25 timepoints.

Options:

1. **Calibration FFC only**

   Best separation between independent calibration and experimental signal;
   corrects both X and Y. Residual overlap mismatch remains about 9.6% for GFP
   and 6.4% for Cy5 in log-ratio terms at Z11.

2. **Calibration FFC plus residual overlap correction**

   Apply the 2-D FFC first, then estimate a positive median-one residual X
   correction from corresponding overlap pixels. This is the leading candidate,
   but its calibration consistency and behavior across Z/time must be validated.

3. **Overlap correction for shape; calibration as a separate mapping**

   Retains the established seam performance but does not use the independent
   reference as the primary illumination correction. A clear policy is needed
   for applying the same correction basis to standards and experiment.

## Recommended hybrid pilot

Before a complete run:

1. Select a small temporal subset such as `t000`, `t006`, `t012`, `t018`, and
   `t024`.
2. Apply the fixed 2-D calibration FFC to every tile.
3. Estimate residual overlap coefficients after FFC for every channel and Z.
4. Quality-weight and smooth those coefficients across Z; assess whether one
   fixed residual across time is sufficient.
5. Normalize every residual correction to median 1.
6. Apply the residual policy consistently when measuring both standards and
   experiment, then rebuild the concentration curve.
7. Compare raw, FFC-only, overlap-only, and hybrid seam residuals.
8. Confirm that the broad GFP/Cy5 gradients, concentration endpoints, midpoint,
   and tanh slope do not change implausibly.
9. Confirm that the central stain stays visible in images and does not generate
   a false feature in robust profiles.
10. Only then expand to all 25 timepoints.

## Performance work before scaling

The full-resolution t012 calibrated pilot took roughly 35–40 minutes. A naive
25-timepoint expansion would be unnecessarily slow.

Recommended optimizations:

- cache measured backgrounds, hot-pixel masks, FFC maps, and calibration models;
- avoid recalculating the eight calibration stacks for every experimental run;
- validate a regular Y subsample or block-median implementation against the
  current full-resolution robust column profile;
- stream or memory-map integrated tiles where possible;
- write progress logs at calibration, channel, timepoint, and DIC stages;
- retain plane-at-a-time ND2 reads and never load the 193 GB source as one array.

Do not trade away the stain resistance without numerical comparison to the
validated t012 profile.

## Outputs still likely to be requested

After the final method is approved:

- complete 0–48 h concentration profiles and tanh fits;
- signed and absolute slope-over-time plots and tables;
- selected-timepoint all-Z summaries if still scientifically useful;
- stitched GFP, Cy5, and merge images with DIC context;
- PowerPoint presentations analogous to the July presentations;
- a revised concise manuscript Methods paragraph describing calibration and
  concentration conversion;
- merge of the feature branch into `main` only after Shelly approves the method.

## Git workflow

Continue on:

`exp106-2026-09-27-calibration-pilot`

Do not merge directly to `main` merely because tests pass. The unresolved item
is scientific method selection, not software correctness.

Before each push:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
git status --short --branch
git diff --check
```
