# GradientAnalysis

Memory-conscious analysis of a Nikon ND2 time course containing multiple stage
positions, Z planes, and GFP/Cy5 channels. The program reads one 2-D plane at a
time rather than loading the complete acquisition into RAM.

## Analysis

The experiment configuration treats `Z=15` as one-based (ND2 index 14), sorts
stage positions left-to-right from their metadata, and processes all 37
timepoints. Every tile is processed in this order:

`raw - 100, floored at 0 -> overlap-derived X correction -> average over Y`

Adjacent fields overlap by about 21.6% (approximately 498 columns). Matching
pixels from all adjacent overlaps, including the measured Y displacement, are
used to robustly estimate one positive quadratic camera-X illumination profile
for GFP and one for Cy5. Dark and detector-clipped pixels are excluded. Each
illumination profile is normalized to median 1 and reused unchanged for every
timepoint.

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
8. The same corrected, feathered, normalized analysis across every Z plane for
   `t002`, `t010`, `t014`, `t024`, and `t036`. Step 8 also saves corrected
   display-only stitched PNGs under `stitched_images/tXXX/`: one GFP image, one
   Cy5 image, and one GFP/Cy5 merge for each Z plane. These PNGs use local
   percentile scaling and are not measurement data; duplicate all-Z TIFF stacks
   are not written.

For Steps 4-6, one maximum is calculated from all corrected, feathered Z15
profiles for each channel. Every GFP profile is divided by the GFP maximum and
every Cy5 profile by the Cy5 maximum. Corrected TIFFs remain in intensity units;
normalization applies only to analytical profiles and fits. Step 8 calculates
its own two normalization constants across its selected T-by-Z collection.

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
comparison. Display limits and all analysis settings are recorded in
`run_metadata.json`.

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
`exp106_2026_07_30_overlap_config.json` for the full experiment.

## Tests

The numerical tests use small synthetic images and require no ND2 file:

```powershell
python -m unittest discover -s tests -v
```
