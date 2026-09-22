# GradientAnalysis

Memory-conscious analysis of a Nikon ND2 time course containing multiple stage
positions, Z planes, and GFP/Cy5 channels. The program reads one 2-D plane at a
time rather than loading the complete acquisition into RAM.

## Analysis

The full experiment configuration treats `Z=15` as one-based (ND2 index 14),
while `exp106_2026_07_30_z11_config.json` runs the primary time-course analysis
at one-based `Z=11` (ND2 index 10). Both sort stage positions left-to-right from
their metadata and process all 37 timepoints. Every tile is processed in this
order:

`raw - 100, floored at 0 -> overlap-derived X correction -> average over Y`

Adjacent fields overlap by about 21.6% (approximately 498 columns). Matching
pixels from all adjacent overlaps, including the measured Y displacement, are
used to robustly estimate one positive quadratic camera-X illumination profile
for GFP and one for Cy5. Dark and detector-clipped pixels are excluded. Each
single-Z illumination profile is normalized to median 1 and reused unchanged
for every timepoint in that run.

After Y averaging, duplicate physical locations are combined with complementary
cosine weights. A tile receives more weight farther from its camera edge; the
weights are equal at the overlap midpoint and always sum to 1. The result is one
continuous physical-X profile with no double-counting.

The output steps are:

1. Background-subtracted left-to-right contact sheets and GFP/Cy5 previews.
2. Continuous cosine-feathered profiles before illumination correction.
3. Corrected uint16 TIFF tiles/contact sheets and overlap-fit diagnostics.
4. Corrected, normalized timecourse profiles, one plot per channel.
5. Corrected, normalized GFP/Cy5 profile for each timepoint.
6. Robust hyperbolic-tangent fits and gradient summaries.
7. The existing background-subtracted P02/P05 maximum-difference analysis.
8. Corrected, feathered, normalized analysis across every Z plane for `t002`,
   `t010`, `t014`, `t024`, and `t036`. Step 8 independently estimates one
   overlap-derived quadratic profile per channel and Z. The raw linear and
   quadratic coefficients are quality-weighted and smoothed across Z before a
   median-one correction curve is reconstructed for each plane. Dim or noisy
   planes therefore borrow strength from neighboring Z planes without forcing
   their overall intensity to match.
9. Signed GFP and Cy5 tanh-slope timecourses, a combined absolute-slope
   timecourse, machine-readable slope tables, and separate signed/absolute
   table images. Experimental time is the zero-based ND2 index multiplied by 2.

Step 8 also creates true stage-coordinate mosaics. Corrected tiles are aligned
with both stage X and stage Y, cropped to their shared physical-Y band, and
cosine-feathered in their 2-D overlaps. The quantitative X profile is averaged
from this same mosaic. Each timepoint folder contains uint16 corrected mosaic
TIFFs, GFP/Cy5 previews, and a merged preview for every Z. PNGs use one shared
channel-specific display range across all selected timepoints and Z planes;
TIFFs retain measurement intensities.

For Steps 4-6 and 9, one maximum is calculated from all corrected, feathered
single-Z profiles for each channel. Every GFP profile is divided by the GFP
maximum and every Cy5 profile by the Cy5 maximum. Corrected TIFFs remain in
intensity units; normalization applies only to analytical profiles and fits.
Step 8 calculates its own two normalization constants across its selected
T-by-Z collection.

Steps 6 and 8 fit the full profile, reduced to at most 1,024 equal-width median
bins, to:

`y(x) = left + (right-left)/2 * [1 + tanh((x-midpoint)/width)]`

The signed slope is `(right-left)/(2*width)` in normalized intensity per
millimeter. CSV and JSON tables also contain the absolute slope, plateaus,
amplitude, midpoint, width, R², RMSE, optimizer status, and QC flags. The old
P01-P06 linear-slope calculation is no longer generated. Its legacy config keys
are accepted and ignored so older configuration files still load.

TIFF files contain measurement values. Files ending in `_preview.png` are
8-bit display copies. Local previews make each timepoint easy to inspect;
`_global_preview.png` files use one shared channel-specific range for fair visual
comparison. Step 8 mosaic previews always use a shared range across its complete
selected T-by-Z collection. Display limits, raw and smoothed per-Z coefficients,
overlap residuals, mosaic geometry, and all analysis settings are recorded in
the Step 8 diagnostics and `run_metadata.json`.

## Installation and use

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python GradientAnalysis.py "D:\path\experiment.nd2" --config example_config.json --output outputs
```

Each invocation creates a new `run_<original-name>_<date_time>` directory, so
existing completed runs are not overwritten. Metadata records the channel and
position mapping, overlap correction, feathering, normalization constants, tanh
model and slope definition, all-Z settings, and Z indices.

Use `exp106_2026_07_30_overlap_pilot_config.json` for a two-timepoint Z15 trial,
`exp106_2026_07_30_all_z_pilot_config.json` for a one-timepoint all-Z trial, and
`exp106_2026_07_30_overlap_config.json` for the full Z15 experiment. Use
`exp106_2026_07_30_z11_config.json` for the 37-timepoint Z11 analysis without
rerunning the all-Z section. A concise paper-style description of the
implemented workflow is provided as `THESIS_METHODS.md`, `PAPER_METHODS.txt`,
and `PAPER_METHODS.docx`.

## Tests

The numerical tests use small synthetic images and require no ND2 file:

```powershell
python -m unittest discover -s tests -v
```
