# Exp106 try 4 calibration pilot — t012 (24 h)

The successful pilot output is:

`outputs/2026_09_27_exp106_try4_calibration_flatfield_t12_24h/run_gradient_calibrated_20261001_135136`

## Dataset and method

- `gradient.nd2` contains 25 timepoints at 2-hour intervals, 11 positions,
  22 Z planes at 10 µm spacing, and Cy5, GFP, and DIC channels.
- Only `t012` (24 h) was analyzed. The gradient-file DIC channel was ignored.
- Numeric calibration filenames were converted from ng/mL to µg/mL. The
  standards are 0, 2.531, 5.062, 10.125, 13.5, 18, 24, and 32 µg/mL.
- One smooth 2-D flat-field per fluorescence channel was estimated from all Z
  planes of the homogeneous 32 µg/mL standard. Each plane was normalized before
  combining it, so unmatched absolute Z positions on the two plates do not
  enter the illumination map.
- Concentration was derived from background-subtracted, flat-field-corrected
  signal integrated through the complete Z stack. This avoids matching one
  calibration Z plane to one experimental Z plane.
- Persistent and isolated hot pixels were identified and replaced by local
  medians. Saved images still show the plate stain; quantitative profiles use a
  robust median over Y so the localized stain does not dominate the gradient.

## Validation results

- No detector-clipped pixels were found in the standards or experimental pilot.
- The linear calibration diagnostic gave R² = 0.971 for GFP and R² = 0.976 for
  Cy5. The 32 µg/mL standard lies above the otherwise near-linear trend, so the
  primary conversion uses monotonic empirical interpolation and retains the
  linear fit only as a diagnostic.
- The t012 profiles remained within the calibration range: approximately
  1.32–12.38 µg/mL for GFP and 0.49–11.44 µg/mL for Cy5.
- Tanh fits passed QC with R² = 0.990 for GFP and R² = 0.992 for Cy5. The fitted
  midpoint slopes were −6.05 µg/mL/mm for GFP and +4.53 µg/mL/mm for Cy5.
- At Z11, median absolute overlap log-mismatch changed from 0.114 to 0.096 for
  GFP and from 0.157 to 0.064 for Cy5 after calibration FFC. The overlap-derived
  quadratic reached 0.0096 and 0.018, respectively.

## Interpretation

The separate-plate 32 µg/mL reference is suitable for a broad optical FFC
because the acquisition geometry and fluorescence channels match, and combining
normalized Z planes avoids requiring absolute Z correspondence. It also corrects
both camera X and Y, which the overlap-only quadratic cannot do.

However, the measured overlap residuals show that calibration FFC alone does
not remove all experiment-specific X variation. For the complete timecourse,
the strongest candidate is a validated hybrid: apply the calibration-derived
2-D FFC first, then estimate a median-one residual overlap correction and apply
that same residual policy consistently to calibration and experimental data.
This should be tested on a small multi-timepoint subset before the full run.

The large 10× DIC image was cropped from stage metadata to the physical rectangle
covered by all 11 gradient positions. Its display is fixed to 100–250 and the
aligned composite places it directly above the Z-integrated fluorescence mosaic.
