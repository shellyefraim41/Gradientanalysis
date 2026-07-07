# GradientAnalysis

Memory-conscious analysis of a Nikon ND2 time course containing multiple stage
positions, Z planes, and GFP/Cy5 channels. The program reads one 2-D plane at a
time, so it does not attempt to load the approximately 250 GB acquisition into
RAM. It uses the maintained `nd2` package's delayed Dask reader.

## Analysis

The default configuration treats `Z=15` as the fifteenth plane (zero-based
index 14), sorts stage positions by descending metadata X, and processes every
timepoint. Descending X matches this experiment's recorded left-to-right order;
`position_x_order` can also be set to `ascending` or `acquisition`.

1. Stitch positions left-to-right without overlap, gaps, or blending. Save a
   raw uint16 TIFF and display-scaled PNG for each channel, plus a GFP/Cy5 merge
   PNG. Single-channel `_preview.png` files are colorized for viewing: GFP is
   green and Cy5 is magenta.
2. Average each stitched image over Y and save the two channel x profiles on one
   graph. Dashed lines show smoothed visual trendlines; they do not change the
   saved data.
3. Work from the separate position images. For each timepoint and channel,
   automatically choose the bright position with the flattest x profile. Fit a
   quadratic trendline to that reference profile and calculate:

   `corrected(y, x) = raw(y, x) * median(fitted_laser_profile) / fitted_laser_profile(x)`

   The fitted trendline is used as the estimated microscope laser profile. The
   correction is applied to every position in that channel/timepoint without
   forcing the stitched chemical gradient to become flat. Save viewer-compatible
   corrected uint16 TIFF tiles, a corrected stitched TIFF/PNG, a before/after
   graph showing the dashed fitted laser trendline, and a JSON record of the
   selected reference. Graphs are calculated from float corrected values before
   TIFF rounding. To make the scientific choice manually, set a zero-based ND2
   position index under `reference_positions` in the config.
4. Save one graph per channel containing the corrected stitched x profile for
   every timepoint. Dashed lines show smoothed visual trendlines. The P04 bridge
   region is shaded gray, each timepoint has a dotted bridge fit, and a side
   table lists bridge slope by timepoint with units in the table header.
5. Save one GFP/Cy5 graph per timepoint, matching Step 2 but using corrected
   values. Mark the P04 bridge region and overlay the fitted bridge slope for
   each channel.
6. Calculate the corrected x-profile slope under the bridge, defined as P04
   by default. Save one CSV/JSON table with slope, intercept, R², and slope per
   micrometer when pixel size metadata are available. Save one slope-over-time
   graph per channel.

TIFF files contain measurement values. Files ending in `_preview.png` are 8-bit
colorized display copies using local per-image scaling, so dim timepoints are
easy to inspect. Files ending in `_global_preview.png` use one shared range per
channel across the selected timepoints, so they are better for fair visual
comparison. Additional `_grayscale_preview.png` files are saved for grayscale
viewing. Display limits are derived from the 1st and 99.8th percentiles and are
recorded in `run_metadata.json`.

P07 is not omitted. In the pilot at t=0 its GFP signal is near background and
its Cy5 signal is weak, so it is expected to remain darker than the other
positions even after consistent display scaling.

## Installation and use

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python GradientAnalysis.py "D:\path\experiment.nd2" --config example_config.json --output outputs
```

Each invocation creates a new folder named
`run_<original-name>_<date_time>`. It contains one folder for every requested
step and `run_metadata.json`, which records dimensions, channel mapping,
position order, and the exact Z index used. Existing runs are not overwritten.

For a quick trial, set `"timepoints": [0, 18]` in a copied config. Verify the
reported channel mapping, left-to-right position order, and Step 3 reference
selection before starting all 49 timepoints.

The included `pilot_config.json` is already restricted to `t=0` and `t=18` for
this check.

## Tests

The numerical tests use small synthetic images and require no ND2 file:

```powershell
python -m unittest discover -s tests -v
```
