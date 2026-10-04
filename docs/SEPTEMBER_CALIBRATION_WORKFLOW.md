# September calibration and correction workflow

## Why Z is integrated

The calibration standards and gradient were imaged on different plates, so an
absolute calibration Z coordinate cannot be paired confidently with an
experimental Z coordinate. Both acquisitions contain 22 planes at the same
10 µm spacing. The pilot therefore integrates fluorescence across the complete
stack after correction. This is insensitive to an overall axial offset as long
as the complete fluorescent depth is captured in both stacks.

The integration is trapezoidal and is reported in `a.u.·µm`.

## Background and hot pixels

The 0 µg/mL stack is sampled across all Z planes. Its channel medians were
exactly 100 intensity units in the pilot, consistent with the older fixed
background. The code nevertheless records the measured scalar separately for
GFP and Cy5.

Persistent hot pixels are detected from the minimum intensity observed at each
camera pixel across the blank stack. An isolated pixel that remains brighter
than its local 3 × 3 neighborhood is flagged. Each acquired plane also receives
a conservative isolated-spike check. Flagged pixels are replaced with the local
median before background subtraction. Counts are saved in diagnostics.

No clipped 16-bit pixels were detected in the calibration series or t012 pilot.

## Two-dimensional flat-field correction

The 32 µg/mL standard is used because it has high signal without detector
clipping. For each channel:

1. Hot pixels are repaired.
2. The measured blank scalar is subtracted and negatives are floored at zero.
3. The image is reduced to robust spatial blocks.
4. Each Z plane is divided by its own robust median. This removes global
   axial-intensity changes while retaining the camera-coordinate illumination
   shape.
5. Normalized planes are combined with a median across Z.
6. Local spatial defects are rejected relative to a broad preliminary field.
7. Normalized convolution produces a smooth 2-D illumination map.
8. The map is normalized to median 1 and floored to prevent unstable edge
   amplification.

Correction is `signal / flatfield(y, x)`. The same channel map is reused for
every experimental position and Z plane.

Observed map ranges were approximately 0.518–1.272 for GFP and 0.456–1.301 for
Cy5. The shapes are smooth, broad, and parabolic, as expected for illumination
falloff.

## Calibration curve

Every standard is background-subtracted and flat-field corrected. Images are
divided into 128-pixel blocks; block medians are integrated through Z. Spatial
block outliers are excluded, and the retained block median is the standard's
signal. The block IQR is plotted as spatial uncertainty.

The primary signal-to-concentration conversion is monotonic piecewise-linear
interpolation through the measured standards. Signals below 0 or above the
32 µg/mL standard are clipped to the calibrated range and explicitly counted.
No t012 profile values required clipping.

A straight-line fit is saved only as a diagnostic. Its R² values are 0.971 for
GFP and 0.976 for Cy5. The 32 µg/mL point is visibly higher than the near-linear
0–24 µg/mL sequence in both channels. Because there is only one acquisition per
concentration, the empirical conversion should be described as preliminary and
not as a high-precision absolute assay.

## Stain handling

The September fluorescence mosaic contains a broad dark stain near the center.
The saved TIFFs and PNGs retain it. The quantitative gradient is calculated with
a robust median over Y for each X column before Z integration. A localized stain
that occupies less than half of a column therefore does not control the profile.
Additional extreme Y outliers are excluded and their fractions are written to
`artifact_exclusion_by_tile_z.csv`.

Do not inpaint, erase, or flat-field the stain as though it were an optical
illumination pattern. It is plate/sample-local, not camera-local.

## Position combination and fitting

Per-position profiles are mapped to one physical X grid using stage metadata.
Duplicate overlap values are combined with complementary cosine weights. The
calibrated concentration profile is divided by 32 µg/mL for the existing bounded
tanh fitter:

`y(x) = left + (right-left)/2 × [1 + tanh((x-midpoint)/width)]`

The normalized midpoint slope is `(right-left)/(2×width)`. Multiplying by
32 µg/mL gives the reported concentration slope in µg/mL/mm.

## DIC context image

The large 10× DIC image and 20× gradient share the same camera-to-stage
orientation metadata. Gradient position centers and tile dimensions define a
stage-coordinate rectangle. That rectangle is transformed into the large-image
pixel grid and cropped.

The TIFF crop preserves raw DIC intensities. The PNG uses a fixed 100–250 window.
For the composite only, the crop is resized to the fluorescence mosaic dimensions
and placed directly above it.

## FFC versus overlap correction

At selected one-based Z11, median absolute overlap log-mismatch was:

| Method | GFP | Cy5 |
|---|---:|---:|
| Before correction | 0.1137 | 0.1567 |
| Calibration FFC | 0.0964 | 0.0640 |
| Overlap quadratic residual | 0.0096 | 0.0182 |

The FFC improves the raw result and supplies genuine two-dimensional correction,
but the overlap model is much better at experiment-specific seam removal. This
is the evidence behind the pending hybrid-correction proposal.
