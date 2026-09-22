# Methods: computational analysis of fluorescence gradients

## Image acquisition structure and data handling

Time-lapse fluorescence images were stored as a Nikon ND2 file containing 37
timepoints, 10 partially overlapping stage positions, 26 axial (Z) planes, and
two fluorescence channels (GFP and Cy5). Each field of view comprised
2304 × 2304 pixels, with a calibrated pixel size of 0.32298 µm. Experimental
time was calculated as twice the zero-based ND2 time index, giving values from
0 to 72 in the time units defined by the acquisition protocol. The primary
time-course analysis was performed at one-based Z=11 (ND2 index 10), which was
selected from the axial analysis as the plane with the best-defined gradient.

Images were analyzed with a custom Python 3.12 pipeline. The ND2 file was read
lazily with `nd2` and Dask so that only the required two-dimensional plane was
loaded at a given time. Channel identities, pixel size, and stage coordinates
were obtained from the ND2 metadata. Stage positions were sorted from left to
right according to their X coordinates. Numerical processing used NumPy and
SciPy; TIFF and graphical outputs were produced with tifffile, Matplotlib, and
Pillow.

## Background subtraction and overlap-derived illumination correction

A constant microscope background of 100 intensity units was subtracted from
every pixel before any correction or profile calculation. Negative values were
set to zero. The acquisition positions overlapped by 496–497 pixels
(approximately 21.56% of the image width). Stage coordinates were used to map
pixels in each pair of adjacent fields to the same physical location, including
the approximately 17-pixel displacement along Y.

The overlaps were used to estimate the illumination profile independently for
GFP and Cy5, without using the dextran gradient itself as a flat-field
reference. Pixels with background-subtracted intensity ≤5 or at the detector
limit were excluded. For each valid overlap column, the median across Y of the
log intensity ratio between matching pixels was calculated. Taking the median
across Y reduced the influence of isolated bright pixels and image noise.

The illumination profile across the camera X coordinate was modeled as a
positive log-quadratic function,

\[
I(u)=\exp(\beta_1u+\beta_2u^2),
\]

where \(u\) is the camera X coordinate centered and scaled to approximately
−1 to 1. The coefficients were obtained from the overlap-ratio constraints by
eight iterations of robust reweighted least squares using median-absolute-
deviation scaling and Huber-type weights. The fitted profile was normalized to
a median of 1. For the primary Z11 time course, all nine adjacent position
pairs and all 37 timepoints were pooled to estimate one fixed profile per
channel. Each background-subtracted tile was corrected before spatial
averaging by

\[
S_{\mathrm{corrected}}(x,y)=\frac{S_{\mathrm{background-subtracted}}(x,y)}{I(x)}.
\]

This operation corrected the camera-position-dependent illumination pattern
while retaining relative intensity differences between timepoints and between
physical locations in the device.

## Construction and normalization of one-dimensional gradients

For each corrected tile, fluorescence was averaged along Y to obtain a
one-dimensional profile along the gradient axis. The profile of each field was
placed on a common physical-X grid using the recorded stage coordinate and
pixel calibration. Duplicate measurements in adjacent overlaps were combined
using complementary cosine weights. Across an overlap, the weight of the left
tile decreased from 1 to 0 and the weight of the right tile increased from 0
to 1; the weights were equal at the overlap midpoint and summed to 1 at every
coordinate. This produced one continuous profile without duplicated overlap
pixels or abrupt field boundaries.

Normalization was performed after illumination correction and feathering. For
each channel, a single normalization constant was defined as the maximum value
among all 37 corrected, Y-averaged Z11 profiles. Every profile from that
channel was divided by the same constant. GFP and Cy5 were therefore each
scaled to a maximum of 1 while preserving relative changes across time. The
normalization was applied only to analytical profiles and model fits;
background-subtracted corrected TIFF images were retained in intensity units.

## Hyperbolic-tangent fitting and gradient-slope calculation

Each normalized channel profile at each timepoint was fitted to a
four-parameter hyperbolic-tangent model,

\[
y(x)=L+\frac{R-L}{2}\left[1+\tanh\left(\frac{x-x_0}{w}\right)\right],
\]

where \(L\) and \(R\) are the left and right plateaus, \(x_0\) is the
transition midpoint, and \(w>0\) describes the transition width. Before
fitting, the physical profile was reduced, when necessary, to at most 1,024
contiguous median bins. Fits were obtained by bounded nonlinear least squares
with a soft-L1 loss. Plateau values were bounded between −0.1 and 1.1, the
midpoint was restricted to the measured X range, and the width was bounded
between one physical pixel and twice the total measured span. Initial plateau
estimates were the medians of the first and last 5% of the profile, and the
initial midpoint was the location nearest their half-transition value.

The gradient slope was defined as the derivative of the fitted curve at its
midpoint, where the tanh transition is steepest:

\[
m=\left.\frac{dy}{dx}\right|_{x=x_0}=\frac{R-L}{2w}.
\]

Signed slopes were retained to describe gradient direction, and absolute
slopes were used to compare gradient steepness independently of direction.
Slope units were normalized fluorescence intensity per millimeter. Signed GFP
and Cy5 slopes and their absolute values were plotted against experimental time
and exported as CSV, JSON, and image tables. The fit output additionally
included the plateaus, amplitude, midpoint, width, coefficient of determination
(R²), root-mean-square error, optimizer status, and quality-control flags.
Fits were retained but flagged when the optimizer did not converge, a fitted
parameter reached a bound, a non-finite result occurred, or the normalized
amplitude was below 0.05.

## Axial analysis

To evaluate the dependence of the measurement on focal plane, timepoints t002,
t010, t014, t024, and t036 were analyzed across all 26 Z planes. A separate
overlap-derived illumination model was estimated for each channel and Z plane
from the selected timepoints. Because dim or noisy planes provided less stable
overlap estimates, the fitted linear and quadratic coefficients were smoothed
across Z by quality-weighted penalized least squares. Quality weights increased
with the number of valid overlap samples and decreased with the squared fit
residual; a second-difference penalty of 10 constrained abrupt coefficient
changes between neighboring planes.

For visualization and axial profile extraction, corrected fields were placed
using both stage X and stage Y coordinates, cropped to the physical Y region
shared by all fields, and cosine-feathered into a two-dimensional mosaic. The
quantitative X profile was calculated directly from this mosaic. The axial
analysis used its own channel-specific normalization constants, calculated
across the complete set of selected timepoints and Z planes, and applied the
same tanh fitting and quality-control procedure described above.

## Quality control and reproducibility

Correction quality was evaluated from the median absolute overlap log-ratio
before and after correction for every adjacent position pair and timepoint.
The pipeline recorded valid-pixel counts, fitted illumination coefficients,
residual errors, normalization constants, fit diagnostics, stage geometry, and
all configuration settings in machine-readable CSV/JSON files and in
`run_metadata.json`. Corrected values were converted to unsigned 16-bit TIFF
only for image output, and the number of clipped pixels was recorded. PNG files
were display copies and were not used for quantitative analysis; their display
limits were set from the 1st and 99.8th percentiles. Core numerical operations
were verified with synthetic unit tests covering overlap recovery in the
presence of a Y shift and physical gradient, exclusion of dark and clipped
pixels, cosine feathering, per-channel normalization, axial coefficient
smoothing, and recovery of increasing and decreasing tanh slopes.
