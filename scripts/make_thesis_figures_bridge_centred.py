"""Thesis figures for 2026_10_08_exp106_try5 (Z11, overlap-corrected), re-plotted with the bridge centre as x = 0.

Nothing is re-analysed. The script only reads tables and float32 TIFFs that the analysis already saved:

* per-position normalized profiles (CSV, every 4th pixel = 2.6 um spacing) and the stain-masked tanh fit table;
* the bridge location (from the large DIC image) and the stain-excluded positions;
* the aligned large-DIC crop (raw uint16 TIFF) for the strip above the profile;
* the chemograph (kymograph) float32 TIFFs (median over Y of each 8x-binned mosaic column).

Changes relative to the original figures (display only):

* position axis shifted so that 0 = bridge centre (left of the bridge < 0, right > 0);
* the bridge is marked in one colour (#d9c27a, the band colour of the tanh-fit plots) on every figure;
* GFP -> Alexa Fluor 488, Cy5 -> Alexa Fluor 647 in all labels;
* kymograph transposed: time on the x axis, position on the y axis.

Outputs go to a new timestamped folder; existing outputs are never touched.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gradient_analysis.config import AnalysisConfig  # noqa: E402
from gradient_analysis.processing import merge_rgb, tanh_profile  # noqa: E402

DATASET = Path(r"D:\Users\Shelly\GradientAnalysis\outputs\2026_10_08_exp106_try5_0-22h_from_Z4")
RUN = DATASET / "run_grad_gradient_review_20261020_110631"
STEP06 = RUN / "step_06_MAIN_RESULT_overlap-corrected_Z11_profiles_with_tanh_fits_per_timepoint"
PROFILES = STEP06 / "2_normalized_intensity" / "2026_10_08_exp106_try5_profile_data_overlap-corrected_Z11_normalized_all_timepoints.csv"
FITS = STEP06 / "tanh_fit_results_overlap_Z11_stain-masked.csv"
BRIDGE = RUN / "step_02_bridge_location_from_large_DIC_image" / "bridge_location_mm.json"
DIC_TIF = RUN / "step_02_bridge_location_from_large_DIC_image" / "2026_10_08_exp106_try5_large_DIC_crop_matching_gradient_fields_raw_values.tif"
STAIN = RUN / "step_03_stain_mask_regions_excluded_from_fits" / "stain_mask_rule_and_excluded_positions.json"
CHEMO = DATASET / "chemograph_overlap-corrected_Z11_0-21.95h_20261101_113004"
PREFIX = "2026_10_08_exp106_try5"

TIMEPOINT = 8  # t008 = 5.95 h, the "~6 h" profile
NAMES = {"GFP": "Alexa Fluor 488", "Cy5": "Alexa Fluor 647"}
LINE_COLORS = {"GFP": "#22aa44", "Cy5": "#d6268c"}  # same as the original profile plots
BRIDGE_COLOR = "#d9c27a"  # band colour of the original tanh-fit plots
BRIDGE_ALPHA = 0.25
X_LABEL = "Position along the device [mm]  (0 = bridge centre)"
POSITION_MARGIN_MM = 0.15  # padding around midpoints / bridge for the shared midpoint-plot range

WIDE_NAME = f"3a-wide_{PREFIX}_Z11_tanh_midpoint_vs_time_linear_stretched"
WIDE_SIZE = (14.0, 6.2)  # same height as 3a, wider time axis

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 12, "legend.fontsize": 10,
    # Editable text for Adobe Illustrator: TrueType (Type 42) fonts in PDF instead of Type 3,
    # and real <text> elements in SVG instead of glyph outlines.
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    # A font Illustrator has installed, so the text opens in the same font (DejaVu Sans is the fallback).
    # Listed directly in font.family so matplotlib can take single missing glyphs (₀) from DejaVu Sans.
    "font.family": ["Arial", "DejaVu Sans"],
})

# Figure keys = the file-name prefix before the first "_" (1, 2a-2d, 3a-3c, 3a-wide). None = all.
WANTED: set[str] | None = None


def save(fig, out: Path, stem: str) -> None:
    if WANTED is None or stem.split("_", 1)[0] in WANTED:
        fig.savefig(out / f"{stem}.png", dpi=300)
        fig.savefig(out / f"{stem}.pdf")
        fig.savefig(out / f"{stem}.svg")
    plt.close(fig)


def read_fits() -> list[dict]:
    with open(FITS, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for r in rows:
        for key in ("time_h", "r_squared", "midpoint_mm", "width_mm", "normalized_left_plateau",
                    "normalized_right_plateau", "normalized_signed_slope_per_mm"):
            r[key] = float(r[key])
        r["timepoint"] = int(r["timepoint"])
        r["fit_valid"] = r["fit_valid"] == "True"
    return rows


def read_profiles() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with open(PROFILES, newline="", encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    reader = csv.DictReader(lines)
    data = list(reader)
    x = np.array([float(r["position_mm"]) for r in data])
    columns = {name: np.array([float(r[name]) for r in data]) for name in reader.fieldnames if "_t" in name}
    return x, columns


def main(wide_only_into: str | None = None) -> None:
    bridge = json.loads(BRIDGE.read_text(encoding="utf-8"))
    centre = float(bridge["center_mm"])
    b_left, b_right = bridge["left_foot_mm"] - centre, bridge["right_foot_mm"] - centre
    stain = [(a - centre, b - centre) for a, b in json.loads(STAIN.read_text(encoding="utf-8"))["excluded_intervals_mm"]]
    fits = read_fits()

    # ------------------------------------------------------------------ shared range for midpoint plot + cropped kymograph
    valid = [r for r in fits if r["fit_valid"]]
    mids = [r["midpoint_mm"] - centre for r in valid]
    pos_range = (min(min(mids), b_left) - POSITION_MARGIN_MM, max(max(mids), b_right) + POSITION_MARGIN_MM)

    # ------------------------------------------------------------------ 3) tanh midpoint over time (plot function)
    def midpoint_plot(name: str, scale: str, size: tuple[float, float] = (9.5, 6.2)) -> None:
        fig, ax = plt.subplots(figsize=size, constrained_layout=True)
        ax.axhspan(b_left, b_right, color=BRIDGE_COLOR, alpha=BRIDGE_ALPHA, lw=0,
                   label=f"Bridge from DIC ({b_left:.2f} to +{b_right:.2f} mm)")
        ax.axhline(0.0, color="0.55", lw=1, ls=":", label="Bridge centre (0 mm)")
        for channel in ("GFP", "Cy5"):
            d = sorted((r for r in valid if r["channel"] == channel), key=lambda r: r["time_h"])
            ax.plot([r["time_h"] for r in d], [r["midpoint_mm"] - centre for r in d], "-o",
                    color=LINE_COLORS[channel], ms=7, lw=1.3, label=NAMES[channel])
        ax.set_ylim(*pos_range)
        if scale == "linear":
            ax.set_xlabel("Time [h]")
        elif scale == "symlog":
            ax.set_xscale("symlog", linthresh=0.1, linscale=0.4)
            ticks = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_xlim(-0.01, 25)
            ax.set_xlabel("Time [h]  (logarithmic above 0.1 h; linear between 0 and 0.1 h)")
        elif scale == "sqrt":
            ax.set_xscale("function", functions=(lambda t: np.sqrt(np.clip(t, 0, None)), lambda s: s ** 2))
            ticks = [0, 0.1, 0.5, 1, 2, 4, 6, 10, 15, 20]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_xlim(0, 23)
            ax.set_xlabel("Time [h]  (square-root scale)")
        # plain text (not $x_0$ math text, which is written letter by letter and splits apart in Illustrator);
        # Arial has no "₀" glyph, so matplotlib takes that one character from the fallback font (DejaVu Sans)
        ax.set_ylabel("tanh midpoint x₀ [mm]  (0 = bridge centre)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10, loc="upper right")
        save(fig, out, name)

    if wide_only_into is not None:
        # add only the wide linear midpoint plot to an existing figure folder, never overwriting a file
        out = Path(wide_only_into)
        if not out.is_dir() or list(out.glob(f"{WIDE_NAME}.*")):
            raise SystemExit(f"{out} is not a folder or already contains {WIDE_NAME}")
        midpoint_plot(WIDE_NAME, "linear", size=WIDE_SIZE)
        print(out)
        return

    subset = "" if WANTED is None else "_figures-" + "-".join(sorted(WANTED))
    out = DATASET / f"THESIS_FIGURES_bridge-centred_editable-text{subset}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=False, exist_ok=False)

    def bridge_band_x(axis, label=True):
        axis.axvspan(b_left, b_right, color=BRIDGE_COLOR, alpha=BRIDGE_ALPHA, lw=0,
                     label=f"Bridge from DIC ({bridge['width_between_feet_mm']:.2f} mm)" if label else None)

    # ------------------------------------------------------------------ 1) t008 profile with tanh fits
    x_abs, columns = read_profiles()
    x_rel = x_abs - centre
    x_fine = np.linspace(x_abs[0], x_abs[-1], 4000)
    hours = next(r["time_h"] for r in fits if r["timepoint"] == TIMEPOINT)
    tag = f"t{TIMEPOINT:03d}_{hours:.2f}h"

    dic_raw = tifffile.imread(DIC_TIF).astype(np.float32)
    dic_pixel_mm = 0.645492759807901 / 1000.0
    dic_offset = float(bridge["dic_covers_profile_mm"][0])
    low, high = (float(v) for v in np.percentile(dic_raw, [0.5, 99.5]))  # same display rule as the original strip
    stride = max(1, dic_raw.shape[1] // 1600)
    dic_img = np.clip((dic_raw[::stride, ::stride] - low) / (high - low), 0, 1)
    dic_extent = [dic_offset - centre, dic_offset + dic_raw.shape[1] * dic_pixel_mm - centre, 0.0, dic_raw.shape[0] * dic_pixel_mm]

    fig = plt.figure(figsize=(13, 6.8), constrained_layout=True)
    grid = GridSpec(2, 1, figure=fig, height_ratios=[0.9, 5.0])
    top = fig.add_subplot(grid[0])
    axis = fig.add_subplot(grid[1], sharex=top)
    top.imshow(dic_img, cmap="gray", extent=dic_extent, aspect="auto")
    for edge in (b_left, b_right):
        top.axvline(edge, color=BRIDGE_COLOR, lw=1.6)
    top.set_yticks([])
    top.set_ylabel("DIC", rotation=0, labelpad=18, va="center")
    bridge_band_x(axis)
    for index, (a, b) in enumerate(stain):
        axis.axvspan(a, b, facecolor="none", edgecolor="#b22222", hatch="///", lw=0,
                     label="Stain: excluded from fit" if index == 0 else None)
    for channel in ("GFP", "Cy5"):
        row = next(r for r in fits if r["channel"] == channel and r["timepoint"] == TIMEPOINT)
        data = columns[next(k for k in columns if k.startswith(f"{channel}_t{TIMEPOINT:03d}_"))]
        fitted = tanh_profile(x_fine, row["normalized_left_plateau"], row["normalized_right_plateau"], row["midpoint_mm"], row["width_mm"])
        axis.plot(x_rel, data, color=LINE_COLORS[channel], lw=1.1, label=f"{NAMES[channel]} data")
        axis.plot(x_fine - centre, fitted, color=LINE_COLORS[channel], lw=2.0, ls="--",
                  label=(f"{NAMES[channel]} tanh fit: slope {row['normalized_signed_slope_per_mm']:.3g} 1/mm, "
                         f"midpoint {row['midpoint_mm'] - centre:.2f} mm, R²={row['r_squared']:.3f}"))
    axis.set_xlim(x_rel[0], x_rel[-1])
    axis.set_ylim(bottom=min(0.0, axis.get_ylim()[0]))
    axis.set(xlabel=X_LABEL, ylabel="Normalized intensity")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=9, loc="center right")
    save(fig, out, f"1_{PREFIX}_{tag}_Z11_normalized_tanh-fit_bridge-centred")

    # ------------------------------------------------------------------ 2) kymographs: time on x, position on y
    meta = json.loads((CHEMO / "chemograph_metadata.json").read_text(encoding="utf-8"))
    times = np.asarray(meta["time_h"], dtype=float)
    pixel_mm = float(meta["pixel_size_mm"])
    ranges = {c: tuple(meta["display_ranges"][c]) for c in ("GFP", "Cy5")}
    stem = f"{PREFIX}_chemograph_overlap_Z11_0-21.95h"
    kymo = {c: tifffile.imread(CHEMO / f"{stem}_{c}_1px-per-timepoint_float32.tif") for c in ("GFP", "Cy5")}
    rgb_colors = {c.label: c.rgb for c in AnalysisConfig().channels}
    panels = [(f"{NAMES[c]} (display 0–{ranges[c][1]:.0f} a.u.)", merge_rgb([np.nan_to_num(kymo[c])], [rgb_colors[c]], [ranges[c]]))
              for c in ("GFP", "Cy5")]
    panels.append(("Merge", merge_rgb([np.nan_to_num(kymo[c]) for c in ("GFP", "Cy5")],
                                      [rgb_colors[c] for c in ("GFP", "Cy5")], [ranges[c] for c in ("GFP", "Cy5")])))
    width = kymo["GFP"].shape[1]
    y_edges = np.arange(width + 1) * pixel_mm - centre  # column c spans [c, c+1] * pixel (centre at (c+0.5)*pixel)
    mid = (times[1:] + times[:-1]) / 2.0
    t_edges = np.concatenate([[max(0.0, times[0] - (mid[0] - times[0]))], mid, [times[-1] + (times[-1] - mid[-1])]])

    def kymograph(name: str, y_limits: tuple[float, float] | None, only_merge: bool = False) -> None:
        use = panels[-1:] if only_merge else panels
        fig, axes = plt.subplots(1, len(use), figsize=(5.2 * len(use) + 0.6, 6.0), sharey=True, constrained_layout=True, squeeze=False)
        for ax, (title, rgb) in zip(axes[0], use):
            ax.pcolormesh(t_edges, y_edges, np.transpose(rgb, (1, 0, 2)) / 255.0, shading="flat", rasterized=True)
            for edge in (b_left, b_right):
                ax.axhline(edge, color=BRIDGE_COLOR, lw=1.6, ls="--")
            ax.set_xlim(t_edges[0], t_edges[-1])
            ax.set_ylim(*(y_limits or (y_edges[0], y_edges[-1])))
            ax.set_title(title, loc="left")
            ax.set_xlabel("Time [h]")
        axes[0][0].set_ylabel(X_LABEL)
        axes[0][-1].plot([], [], color=BRIDGE_COLOR, lw=1.6, ls="--", label="Bridge edges (DIC)")
        axes[0][-1].legend(loc="lower right", framealpha=0.85)
        save(fig, out, name)

    kymograph(f"2a_{PREFIX}_kymograph_Z11_time-x_position-y_full-device", None)
    kymograph(f"2b_{PREFIX}_kymograph_Z11_time-x_position-y_cropped-to-midpoint-plot-range", pos_range)
    kymograph(f"2c_{PREFIX}_kymograph_Z11_merge-only_full-device", None, only_merge=True)
    kymograph(f"2d_{PREFIX}_kymograph_Z11_merge-only_cropped-to-midpoint-plot-range", pos_range, only_merge=True)

    # ------------------------------------------------------------------ 3) tanh midpoint over time
    midpoint_plot(f"3a_{PREFIX}_Z11_tanh_midpoint_vs_time_linear", "linear")
    midpoint_plot(f"3b_{PREFIX}_Z11_tanh_midpoint_vs_time_log-time", "symlog")
    midpoint_plot(f"3c_{PREFIX}_Z11_tanh_midpoint_vs_time_sqrt-time", "sqrt")
    midpoint_plot(WIDE_NAME, "linear", size=WIDE_SIZE)

    # ------------------------------------------------------------------ table + README
    with open(out / "tanh_midpoints_bridge-centred.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dye", "original_channel", "timepoint_zero_based", "time_h", "midpoint_mm_bridge_centred",
                         "midpoint_mm_original_axis", "width_mm", "r_squared", "fit_valid"])
        for r in sorted(fits, key=lambda r: (r["channel"] != "GFP", r["timepoint"])):
            writer.writerow([NAMES[r["channel"]], r["channel"], r["timepoint"], f"{r['time_h']:.4f}",
                             f"{r['midpoint_mm'] - centre:.4f}", f"{r['midpoint_mm']:.4f}", f"{r['width_mm']:.4f}",
                             f"{r['r_squared']:.4f}", r["fit_valid"]])

    (out / "README.txt").write_text(
        "Thesis figures, 2026_10_08_exp106_try5, Z11, overlap-corrected, stain-masked fits.\n"
        "Re-plotted from saved results only; nothing was re-analysed (same data, same fits, same display ranges).\n\n"
        f"Position axis: x_new = x_original - {centre:.4f} mm (bridge centre from the large DIC image).\n"
        f"  Bridge (between its feet): {b_left:.3f} to +{b_right:.3f} mm; negative = left of the bridge.\n"
        f"  Bridge colour on every figure: {BRIDGE_COLOR} (the yellow band of the original tanh-fit plots).\n"
        "Dye names: GFP channel = Alexa Fluor 488, Cy5 channel = Alexa Fluor 647.\n\n"
        f"1_*  t{TIMEPOINT:03d} = {hours:.2f} h normalized profile with tanh fits; DIC strip on top; red hatching = stain\n"
        "      positions excluded from the fit (data still shown). Profile data: every 4th pixel (2.6 um) of the saved table.\n"
        "2a/2b kymograph, time on x, position on y (0 = bridge centre, positive up); median over Y of each column of the\n"
        "      overlap-corrected Z11 mosaic; each column of pixels fills the time to the neighbouring timepoints.\n"
        "      Same display ranges as the original chemograph. 2b is cropped to the y range of the midpoint plots\n"
        f"      ({pos_range[0]:.2f} to +{pos_range[1]:.2f} mm). 2c/2d: merge panel only.\n"
        "3a   tanh midpoint over time, linear time axis.\n"
        "3b   same, time axis linear from 0 to 0.1 h and logarithmic above (so t = 0 can be shown) - stretches the start.\n"
        "3c   same, square-root time axis (a diffusing front moves ~ sqrt(t), so this is a physically natural scale).\n"
        "PNG = 300 dpi; PDF and SVG = same figure, vector, with text kept as real editable text\n"
        "(PDF: embedded TrueType/Type 42 Arial; SVG: <text> elements in Arial) for editing in Adobe Illustrator.\n"
        "The kymograph and DIC images inside the figures are embedded bitmaps.\n"
        f"Figures made in this folder: {'all' if WANTED is None else ', '.join(sorted(WANTED))}.\n\n"
        f"Sources:\n  {PROFILES}\n  {FITS}\n  {BRIDGE}\n  {STAIN}\n  {DIC_TIF}\n  {CHEMO}\n",
        encoding="utf-8")
    shutil.copy2(__file__, out / Path(__file__).name)
    print(out)


if __name__ == "__main__":
    # optional: --add-wide-midpoint <existing THESIS_FIGURES folder>
    #           --only 1,2d,3a,3a-wide   (make just these figures, in a new folder)
    if len(sys.argv) > 2 and sys.argv[1] == "--only":
        WANTED = {key.strip() for key in sys.argv[2].split(",") if key.strip()}
        main()
    else:
        main(sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--add-wide-midpoint" else None)
