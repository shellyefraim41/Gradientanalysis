# Image analysis and gradient quantification

Time-lapse fluorescence images of fluorophore-conjugated dextran were analyzed
with a custom Python pipeline. The Nikon ND2 dataset contained 37 timepoints,
10 overlapping stage positions, 26 Z planes, and GFP and Cy5 channels. Images
were 2304 × 2304 pixels with a calibrated pixel size of 0.32298 µm. The primary
time-course analysis was performed at one-based Z=11 (ND2 index 10), selected
after comparison of the gradients across Z. Experimental time was calculated
as twice the zero-based acquisition index (0–72; units as defined in the
acquisition protocol). Images were read plane-by-plane, and channel identities,
pixel size, and stage coordinates were obtained from the ND2 metadata.

A constant background of 100 intensity units was subtracted from every image,
and negative values were set to zero. Adjacent fields overlapped by 496–497
pixels (21.56% of the field width). Corresponding pixels were identified from
the stage X and Y coordinates, including the approximately 17-pixel Y offset
between neighboring fields. Background-level and detector-clipped pixels were
excluded. For each valid overlap column, the median log intensity ratio between
matching pixels was calculated across Y. These ratios were used to estimate a
channel-specific camera-X illumination profile,

\[
I(u)=\exp(\beta_1u+\beta_2u^2),
\]

where \(u\) is the centered and scaled camera X coordinate. Coefficients were
estimated by robust iteratively reweighted least squares, and each profile was
normalized to a median of 1. For the Z11 analysis, all nine adjacent position
pairs and all timepoints were pooled to obtain one fixed profile for each
channel. Background-subtracted images were corrected as
\(S_{\mathrm{corr}}=S/I\). No independent flat-field image was used.

Fluorescence was averaged along Y to obtain a one-dimensional profile for each
field. Profiles were positioned on a common physical-X axis using the stage
metadata and pixel calibration. Duplicate measurements in the overlaps were
combined with complementary cosine weights that summed to 1, producing a
continuous profile without double-counting. Corrected profiles were normalized
separately for GFP and Cy5 by the maximum value observed across all 37 profiles
of that channel. The same channel-specific normalization constant was applied
to every timepoint; corrected TIFF images remained in background-subtracted
intensity units.

Each normalized profile was fitted by robust bounded nonlinear least squares
to

\[
y(x)=L+\frac{R-L}{2}\left[1+\tanh\left(\frac{x-x_0}{w}\right)\right],
\]

where \(L\) and \(R\) are the left and right plateaus, \(x_0\) is the transition
midpoint, and \(w\) is the transition width. Profiles were reduced, when
necessary, to at most 1,024 contiguous median bins before fitting, and a
soft-L1 loss was used to reduce sensitivity to outliers. Gradient slope was
defined as the derivative at the fitted midpoint,
\(m=(R-L)/(2w)\), in normalized intensity per millimeter. Signed slopes
described gradient direction, whereas absolute slopes described steepness.
Fits were retained but flagged if optimization failed, a parameter reached its
bound, the result was non-finite, or the fitted amplitude was below 0.05.

For the axial analysis, timepoints t002, t010, t014, t024, and t036 were
analyzed across all 26 Z planes. A separate overlap-derived illumination model
was estimated for each channel and Z plane. The linear and quadratic
coefficients were quality-weighted according to sample count and residual error
and smoothed across Z using a second-difference penalty of 10. Corrected fields
were aligned in two dimensions from their stage coordinates, cropped to their
shared physical-Y region, and cosine-feathered before profile extraction and
tanh fitting. This analysis used separate channel-specific normalization
constants calculated across the complete selected timepoint-by-Z dataset.

Quantitative results and quality-control metrics were exported as CSV and JSON
files. PNG files were used only for visualization; quantitative measurements
were calculated from TIFF data. The analysis was implemented in Python 3.12
using nd2, Dask, NumPy, SciPy, tifffile, Matplotlib, and Pillow. Source code,
configuration files, and numerical tests are available at
https://github.com/shellyefraim41/GradientanalysisExp106.
