# Pending decisions and next steps

## Decision required before the full September run

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
