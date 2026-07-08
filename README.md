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

1. Save a left-to-right contact sheet for image inspection. These Step 1 images
   still place the positions directly beside each other, because they are meant
   for quick visual checking, not for distance measurement. Save a raw uint16
   TIFF and display-scaled PNG for each channel, plus a GFP/Cy5 merge PNG.
   Single-channel `_preview.png` files are colorized for viewing: GFP is green
   and Cy5 is magenta.
2. Average each position image over Y and save the two channel x profiles on one
   graph using the real physical x-axis in millimeters. Stage metadata X values
   are treated as image centers, image columns are converted using the pixel
   size metadata, and the left edge of P01 is defined as `x = 0 mm`. Solid lines
   show measured image data only. Dashed lines across gaps are visual
   interpolation over unmeasured space; they are not added to the saved data or
   used for slope fitting.
3. Work from the separate position images. For each channel, scan all selected
   timepoints and positions and choose one fixed correction reference from
   bright, unsaturated candidates whose normalized local-x profile shape changes
   least over time. Fit a quadratic trendline to that single reference profile
   and calculate:

   `corrected(y, x) = raw(y, x) * median(fitted_laser_profile) / fitted_laser_profile(x)`

   The fitted trendline is used as the estimated microscope laser profile for
   that channel. The same correction curve is then reused for every timepoint
   and every position in that channel. This avoids changing the correction from
   timepoint to timepoint and reduces the chance of flipping the chemical
   gradient shape. Save viewer-compatible corrected uint16 TIFF tiles, a
   corrected contact-sheet TIFF/PNG, and a normalization graph with three
   panels: raw local-x profiles, corrected local-x profiles, and the same
   corrected values placed on the physical-mm device axis. The local-x panel can
   look much straighter because it shows each field of view separately; the
   physical-mm panel preserves the between-position gradient. Graphs are
   calculated from float corrected values before TIFF rounding. Candidate
   reference scores and selected-reference stability plots are saved under
   `step_03_illumination_corrected/reference_selection/`. A diagnostic
   `smoothed_profile_comparison` folder also compares the conservative
   quadratic laser fit with a more edge-following correction that uses the
   smoothed reference profile directly. On the 2-D flat-field experiment branch,
   `flatfield_2d_comparison` additionally tests correction from a smoothed 2-D
   reference tile illumination map.
4. Save one graph per channel containing the corrected physical-mm x profile for
   every timepoint. Solid line segments are measured image data. Dashed lines
   span the unmeasured gaps between positions as visual interpolation only. The
   P01-P06 slope range is shaded gray, each timepoint has a dotted P01-P06
   linear fit, and a side table lists gradient slope by timepoint with units in
   the table header.
5. Save one GFP/Cy5 graph per timepoint, matching Step 2 but using corrected
   values. The P01-P06 fit is overlaid for each channel. Raw versions of the
   Step 4, Step 5, and Step 6 analyses are saved in `raw_comparison` folders so
   pilot runs can be checked before running all timepoints.
6. Calculate the corrected gradient slope from measured pixels in P01 through
   P06, excluding P07. Save one CSV/JSON table with slope in `a.u./mm`,
   intercept, and R². Save one slope-over-time graph per channel.

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
position order, physical-mm position ranges, fixed correction references, and
the exact Z index used. Existing runs are not overwritten.

For a quick trial, set `"timepoints": [0, 18]` in a copied config. Verify the
reported channel mapping, left-to-right position order, physical-mm spacing,
Step 3 fixed correction reference, and raw-versus-corrected comparison outputs
before starting all 49 timepoints.

The included `pilot_config.json` is already restricted to `t=0` and `t=18` for
this check.

## Tests

The numerical tests use small synthetic images and require no ND2 file:

```powershell
python -m unittest discover -s tests -v
```
