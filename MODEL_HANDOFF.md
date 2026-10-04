# Model handoff: GradientAnalysis

Last updated: 2026-10-04

This is the entry point for continuing Shelly's gradient-analysis work after a
model change. Read this document and the three linked references before making
scientific or implementation decisions.

- [Project and data inventory](docs/PROJECT_INVENTORY.md)
- [September calibration workflow](docs/SEPTEMBER_CALIBRATION_WORKFLOW.md)
- [Pending decisions and next steps](docs/NEXT_STEPS.md)
- [General pipeline README](README.md)
- [Completed t012 pilot report](PILOT_2026_09_27_T012_REPORT.md)

## User and project intent

The user is Shelly. She is analyzing opposing GFP- and Cy5-labeled dextran
gradients acquired as overlapping microscope fields, Z stacks, and time series.
The pipeline must preserve the physical dextran gradient while correcting
camera-coordinate illumination, combining overlaps smoothly, and fitting tanh
gradient models.

Shelly prefers concrete execution with new output folders and visual QC. Do not
silently overwrite completed runs. Explain scientific limitations plainly,
especially when a calibration or correction is preliminary.

## Repository state

- Local repository: `D:\Users\Shelly\GradientAnalysis`
- GitHub: `https://github.com/shellyefraim41/Gradientanalysis`
- Main branch commit: `2dc3ed0`
- Active branch: `exp106-2026-09-27-calibration-pilot`
- September workflow implementation commit: `0c36700`
- The model-handoff documentation was committed afterward; use
  `git log -1 --oneline` for the current branch tip.
- The active branch is pushed and tracks
  `origin/exp106-2026-09-27-calibration-pilot`.
- The worktree was clean when this handoff was prepared.
- Analysis outputs and ND2 files are intentionally ignored by Git.

## What is already implemented

### July 2026 workflow

The established pipeline supports:

- subtracting a constant background of 100 and flooring negatives at zero;
- overlap-derived quadratic X-illumination correction;
- physical-coordinate cosine feathering of overlapping fields;
- per-channel profile normalization;
- robust tanh fitting and midpoint-slope extraction;
- selected-timepoint all-Z analysis with per-Z overlap corrections smoothed
  across Z;
- stage-aligned two-dimensional mosaics;
- slope-over-time plots and signed/absolute slope tables;
- PowerPoint generation for image and profile summaries.

The primary validated July single-Z analysis used one-based Z11. Historical
completed outputs must remain untouched.

### September 2026 pilot

Commit `0c36700` added a separate calibrated workflow selected with:

`"correction_method": "calibration_flatfield"`

`gradient_analysis.pipeline_v2.run_pipeline` dispatches that configuration to
`gradient_analysis.calibrated_pipeline.run_calibrated_pipeline`. The existing
overlap workflow remains available and unchanged.

The calibrated pilot:

1. Reads numeric calibration filenames as ng/mL and reports them in µg/mL.
2. Uses the 0 µg/mL acquisition to estimate a scalar background and persistent
   hot-pixel mask for each channel.
3. Uses all Z planes of the 32 µg/mL standard to estimate an artifact-resistant,
   median-one 2-D flat field for GFP and Cy5.
4. Normalizes each reference Z plane before combining it, so absolute Z offsets
   between the calibration and experiment plates do not enter the FFC.
5. Builds a concentration calibration from robust block medians integrated
   through the complete Z stack.
6. Uses monotonic empirical interpolation for conversion and retains an ordinary
   linear fit only as a diagnostic.
7. Processes only `t012` in the current pilot, corresponding to 24 h.
8. Ignores the DIC channel embedded in `gradient.nd2`.
9. Keeps the plate stain visible in images while using robust Y statistics for
   quantitative profiles.
10. Crops the separate 10× DIC overview using stage coordinates and places the
    100–250 display-window crop above the fluorescence mosaic.

## Validated pilot outcome

Successful output:

`D:\Users\Shelly\GradientAnalysis\outputs\2026_09_27_exp106_try4_calibration_flatfield_t12_24h\run_gradient_calibrated_20261001_135136`

Ignore the incomplete earlier attempt:

`...\run_gradient_calibrated_20261001_134845`

Do not delete that incomplete folder without Shelly's permission.

Key results from the successful run:

| Metric | GFP | Cy5 |
|---|---:|---:|
| Linear calibration R² | 0.9712 | 0.9762 |
| Pearson correlation | 0.9855 | 0.9880 |
| t012 concentration range (µg/mL) | 1.32–12.38 | 0.49–11.44 |
| Tanh fit R² | 0.9896 | 0.9922 |
| Signed slope (µg/mL/mm) | −6.05 | +4.53 |
| Raw overlap mismatch | 0.1137 | 0.1567 |
| FFC overlap mismatch | 0.0964 | 0.0640 |
| Overlap-quadratic residual | 0.0096 | 0.0182 |

No detector-clipped pixels were found. All profile values remained inside the
measured calibration range. The 32 µg/mL standard is above the otherwise
near-linear trend in both channels, so absolute concentrations are preliminary
and should not be presented as higher precision than the standards support.

## Scientific conclusion that must be preserved

The different calibration plate does not prevent a broad optical FFC. The same
fluorescence optics, pixel geometry, channels, and acquisition settings were
used, and plane-wise normalization plus full-stack integration removes the need
to match absolute Z positions.

However, the real-data seam comparison shows that FFC alone does not remove all
experiment-specific X variation. The calibration FFC is physically preferable
for two-dimensional illumination and concentration calibration, while the
overlap-derived quadratic is substantially better at eliminating residual
seams. The leading candidate for the full experiment is therefore a validated
hybrid, not an unexamined replacement of one method by the other.

Do not start the complete 25-timepoint run until Shelly chooses whether to:

- keep calibration FFC alone;
- use a hybrid FFC plus residual overlap correction; or
- retain overlap correction for gradient shape and use calibration only as a
  separate intensity-to-concentration mapping.

See [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md) for the proposed validation.

## Verification status

- 50 unit tests pass.
- Python compilation checks pass.
- The successful run contains 15 TIFFs and 21 PNGs plus CSV/JSON metadata.
- All PNG files verify with Pillow.
- Integrated float TIFFs intentionally contain NaNs only outside the valid
  slanted union mosaic; finite pixels are valid measurement values.
- `PAPER_METHODS.docx` is a valid ZIP/DOCX and its repository hyperlink was
  updated to the renamed GitHub repository.

## Standard commands

Run tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Run the completed September t012 calibrated pilot into a new output root:

```powershell
.\.venv\Scripts\python.exe GradientAnalysis.py `
  "D:\Users\Shelly\2026_09_27_Exp106_try4\gradient.nd2" `
  --config exp106_2026_09_27_t12_calibrated_pilot_config.json `
  --output "outputs\2026_09_27_exp106_try4_calibration_flatfield_t12_24h"
```

This full-resolution pilot took roughly 35–40 minutes. Do not rerun it merely to
inspect results; use the completed output directory above.
