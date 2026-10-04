# Project and data inventory

Last updated: 2026-10-04

## Repository map

- `GradientAnalysis.py`: command-line entry point.
- `gradient_analysis/config.py`: configuration dataclasses and validation.
- `gradient_analysis/nd2_source.py`: lazy plane-at-a-time ND2 access and stage
  metadata parsing.
- `gradient_analysis/processing.py`: pure numerical operations, including
  overlap correction, feathering, flat-field construction, robust profiles, and
  tanh fits.
- `gradient_analysis/pipeline_v2.py`: established July overlap pipeline and
  dispatch to the calibrated workflow.
- `gradient_analysis/calibrated_pipeline.py`: September calibration/FFC pilot.
- `gradient_analysis/outputs.py`: TIFF, PNG, CSV, JSON, and plot helpers.
- `scripts/create_gradient_presentations.py`: July presentation builder.
- `tests/`: 50 unit tests at handoff.

## July dataset

Source folder:

`D:\Users\Shelly\2026_07_30_exp106_grad_2C`

Primary ND2:

`D:\Users\Shelly\2026_07_30_exp106_grad_2C\2026_07_30_gradient.nd2`

Metadata:

- 37 timepoints, 2 h apart;
- 10 positions;
- 26 Z planes, 10 µm apart;
- GFP and Cy5 channels;
- 2304 × 2304 pixels per tile;
- XY pixel size approximately 0.32298 µm;
- adjacent overlap approximately 21.6%, or about 498 columns;
- adjacent stage-Y displacement approximately 17 pixels.

Important configs:

- `exp106_2026_07_30_z11_config.json`: validated 37-timepoint Z11 run.
- `exp106_2026_07_30_overlap_config.json`: Z15 overlap workflow.
- `exp106_2026_07_30_all_z_pilot_config.json`: reduced all-Z pilot.

Primary completed Z11 run:

`D:\Users\Shelly\GradientAnalysis\outputs\2026_07_30_exp106_grad_2C_z11_overlap_quadratic_subtract100\run_2026_07_30_gradient_20260907_113630`

Presentations inside that run:

- `2026_07_30_z11_gradient_images_timecourse_5-per-slide_t2.pptx`
- `2026_07_30_z11_profiles_before_vs_after_tanh_fit.pptx`

## September dataset

Source folder:

`D:\Users\Shelly\2026_09_27_Exp106_try4`

### Gradient time series

`gradient.nd2`

- approximately 192.7 GB;
- 25 timepoints: `t000–t024`;
- each index represents 2 h, so the acquisition spans 0–48 h;
- 11 overlapping positions;
- 22 Z planes at 10 µm spacing;
- channels: Cy5, GFP, and X20-DIC;
- 2304 × 2304 pixels per field;
- XY pixel size approximately 0.32333 µm;
- adjacent X spacing approximately 524.77 µm;
- overlap approximately 681 pixels, or 29.6%;
- adjacent stage-Y displacement approximately 17 pixels;
- fluorescence analysis must ignore X20-DIC.

### Calibration standards

Numeric filenames encode ng/mL and are divided by 1000 for µg/mL:

| File | Concentration (µg/mL) |
|---|---:|
| `00000.nd2` | 0 |
| `2531.nd2` | 2.531 |
| `5062.nd2` | 5.062 |
| `10125.nd2` | 10.125 |
| `13500.nd2` | 13.5 |
| `18000.nd2` | 18 |
| `24000.nd2` | 24 |
| `32000.nd2` | 32 |

Each standard contains 22 Z planes, two fluorescence channels, 2304 × 2304
pixels, approximately 0.32333 µm XY pixels, and 10 µm Z spacing. The acquisition
note states 10% lasers and 300 ms exposure.

The 32 µg/mL file is the flat-field reference. The 0 µg/mL file is the blank.

### Other September files

- `Exp106_try4_largeimageX4_before.nd2`: one 27187 × 27187 X10-DIC overview,
  approximately 0.64549 µm/pixel. Used for the stage-coordinate context crop.
- `GFPafter48hours.nd2` and `Cy5after48hours.nd2`: endpoint fluorescence files;
  not used by the current pilot.
- `2026_09_27_Exp106_try4.txt`: acquisition notes.

## September configs and outputs

- `exp106_2026_09_27_t12_calibrated_pilot_config.json`: main calibrated pilot.
- `exp106_2026_09_27_t12_overlap_pilot_config.json`: old-method comparison at
  t012/Z11.

Successful calibrated output:

`D:\Users\Shelly\GradientAnalysis\outputs\2026_09_27_exp106_try4_calibration_flatfield_t12_24h\run_gradient_calibrated_20261001_135136`

Incomplete output to ignore but not delete without permission:

`D:\Users\Shelly\GradientAnalysis\outputs\2026_09_27_exp106_try4_calibration_flatfield_t12_24h\run_gradient_calibrated_20261001_134845`

Comparison overlap-pilot output:

`D:\Users\Shelly\GradientAnalysis\outputs\2026_09_27_exp106_try4_t12_overlap_pilot\run_gradient_20261001_133255`

## Git state at handoff

- Remote: `https://github.com/shellyefraim41/Gradientanalysis.git`
- `main`: `2dc3ed0`
- active branch: `exp106-2026-09-27-calibration-pilot`
- September workflow implementation commit: `0c36700`
- use `git log -1 --oneline` for the current documentation-inclusive branch tip;
- active branch is pushed and synchronized with its remote branch.
