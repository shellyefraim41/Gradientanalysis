"""Chemograph (kymograph): every timepoint's mosaic collapsed to a 1-pixel-tall row, stacked over time.

For each timepoint and channel the saved overlap-corrected single-plane mosaic TIFF of a
gradient-review run (8x-binned float32, background-subtracted intensity) is reduced to one row
by the median over Y of every column - the same robust Y statistic the pipeline uses for its
profiles, so the plate stain cannot pull a column down. Rows are stacked from the first
timepoint (top) to the last (bottom). Values are never taken from display-scaled PNGs.

Outputs (new timestamped folder):

* float32 TIFF per channel, 1 px per timepoint (rows = timepoints, columns = position);
* plain PNGs of the same pixels (GFP, Cy5, merge; same display range for all timepoints);
* annotated figures: one band per timepoint, and on a real time axis (non-uniform intervals);
* chemograph_metadata.json: source files, times, x axis, display ranges, reduction rule.

Example:
    .\\.venv\\Scripts\\python.exe scripts\\make_chemograph.py ^
        --run outputs\\2026_10_08_exp106_try5_0-22h_from_Z4\\run_grad_gradient_review_20261020_110631 ^
        --output outputs\\2026_10_08_exp106_try5_0-22h_from_Z4
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gradient_analysis.config import AnalysisConfig  # noqa: E402
from gradient_analysis.processing import merge_rgb  # noqa: E402

DISPLAY_PERCENTILE = 99.5


def _hl(hours: float) -> str:
    return f"{hours:.2f}".rstrip("0").rstrip(".") if hours else "0"


def collapse_to_row(image: np.ndarray) -> np.ndarray:
    """Median over Y of every column, ignoring NaNs outside the valid mosaic."""
    return np.nanmedian(image, axis=0).astype(np.float32)


def time_edges(times_h: np.ndarray) -> np.ndarray:
    """Row edges halfway between neighbouring timepoints; the end rows are symmetric."""
    mid = (times_h[1:] + times_h[:-1]) / 2.0
    first = max(0.0, times_h[0] - (mid[0] - times_h[0]))
    last = times_h[-1] + (times_h[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def _shade(axis, bridge: dict | None, stain: list[list[float]]) -> None:
    if bridge:
        for key in ("left_foot_mm", "right_foot_mm"):
            axis.axvline(bridge[key], color="gold", lw=1.2, ls="--")
    for low, high in stain:
        axis.axvspan(low, high, facecolor="none", edgecolor="red", hatch="////", lw=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path, help="gradient-review run folder")
    parser.add_argument("--output", required=True, type=Path, help="root folder for the new timestamped output")
    parser.add_argument("--method", default="overlap", help="correction tag in the mosaic file names")
    parser.add_argument("--band-px", type=int, default=24, help="display height per timepoint in the band figure")
    parser.add_argument("--mark-stain", action="store_true", help="hatch the stain-excluded positions on the figures")
    args = parser.parse_args()

    meta = json.loads((args.run / "run_metadata.json").read_text(encoding="utf-8"))
    label = meta["experiment_label"]
    timepoints = meta["timepoints"]
    times_h = np.asarray(meta["time_h"], dtype=float)
    channels = meta["channels"]
    depth = meta["main_depth"]
    pixel_mm = meta["pixel_size_um"] * meta["bin_factor_for_mosaics"] / 1000.0
    colors = {c.label: c.rgb for c in AnalysisConfig().channels}

    tiff_dir = next(args.run.glob("step_01_*/*_mosaics_8x-binned_float32_TIFF"))
    bridge_file = next(args.run.glob("step_02_*/bridge_location_mm.json"), None)
    stain_file = next(args.run.glob("step_03_*/stain_mask_rule_and_excluded_positions.json"), None)
    bridge = json.loads(bridge_file.read_text(encoding="utf-8")) if bridge_file else None
    stain = json.loads(stain_file.read_text(encoding="utf-8"))["excluded_intervals_mm"] if stain_file else []

    rows: dict[str, list[np.ndarray]] = {c: [] for c in channels}
    sources: dict[str, list[str]] = {c: [] for c in channels}
    for t, hours in zip(timepoints, times_h):
        for c in channels:
            matches = sorted(tiff_dir.glob(f"{label}_t{t:03d}_*_{c}_{args.method}_*_8x-binned_float32.tif"))
            if len(matches) != 1:
                raise FileNotFoundError(f"expected one {c} {args.method} mosaic for t{t:03d}, found {len(matches)}")
            rows[c].append(collapse_to_row(tifffile.imread(matches[0])))
            sources[c].append(str(matches[0]))
    kymo = {c: np.vstack(rows[c]) for c in channels}
    width = kymo[channels[0]].shape[1]
    x_mm = (np.arange(width) + 0.5) * pixel_mm
    ranges = {c: (0.0, float(np.nanpercentile(kymo[c], DISPLAY_PERCENTILE))) for c in channels}

    span = f"{_hl(times_h[0])}-{_hl(times_h[-1])}h"
    out = args.output / f"chemograph_{args.method}-corrected_{depth}_{span}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    stem = f"{label}_chemograph_{args.method}_{depth}_{span}"

    # ---- quantitative data and plain pixel images ----------------------------------
    for c in channels:
        tifffile.imwrite(out / f"{stem}_{c}_1px-per-timepoint_float32.tif", kymo[c])
    plain = {c: merge_rgb([np.nan_to_num(kymo[c])], [colors[c]], [ranges[c]]) for c in channels}
    plain["merge"] = merge_rgb([np.nan_to_num(kymo[c]) for c in channels], [colors[c] for c in channels],
                               [ranges[c] for c in channels])
    for name, rgb in plain.items():
        Image.fromarray(rgb, mode="RGB").save(out / f"{stem}_{name}_1px-per-timepoint.png")

    # ---- annotated figures ---------------------------------------------------------
    panels = [(c, plain[c]) for c in channels] + [("merge", plain["merge"])]
    time_labels = [f"{_hl(h)}" for h in times_h]
    extent_x = (0.0, width * pixel_mm)

    def figure(kind: str) -> None:
        n = len(times_h)
        height = (n * args.band_px / 100.0 + 1.0) if kind == "bands" else 4.2
        fig, axes = plt.subplots(len(panels), 1, figsize=(14, height * len(panels) / 1.6), constrained_layout=True)
        for axis, (name, rgb) in zip(axes, panels):
            if kind == "bands":
                axis.imshow(rgb, aspect="auto", interpolation="nearest", extent=(*extent_x, n - 0.5, -0.5))
                axis.set_yticks(range(n), time_labels, fontsize=7)
                axis.set_ylabel("Time [h]\n(one row per timepoint)")
            else:
                edges = time_edges(times_h)
                axis.pcolormesh(np.concatenate([[0.0], x_mm + pixel_mm / 2.0]), edges, rgb / 255.0, shading="flat")
                axis.set_ylim(edges[-1], edges[0])
                axis.set_ylabel("Time [h]")
            _shade(axis, bridge, stain if args.mark_stain else [])
            unit = "" if name == "merge" else f" (display 0–{ranges[name][1]:.0f} a.u.)"
            axis.set_title(f"{name}{unit}", loc="left", fontsize=10)
        axes[-1].set_xlabel("Position [mm] (0 = left edge of the leftmost gradient field)")
        title = "one band per timepoint" if kind == "bands" else "real time axis (rows fill the time to the neighbouring timepoints)"
        legend = "gold dashed = bridge edges (DIC)" + (", red hatching = stain (rows still show the data)" if args.mark_stain else "")
        fig.suptitle(f"{label} chemograph · {args.method}-corrected {depth}, median over Y · {title}\n{legend}", fontsize=10)
        fig.savefig(out / f"{stem}_annotated_{'one-band-per-timepoint' if kind == 'bands' else 'real-time-axis'}.png", dpi=150)
        plt.close(fig)

    figure("bands")
    figure("time")

    (out / "chemograph_metadata.json").write_text(json.dumps({
        "experiment_label": label,
        "source_run": str(args.run),
        "method": args.method,
        "depth": depth,
        "reduction": "median over Y of each column of the 8x-binned float32 mosaic (NaNs outside the valid mosaic ignored)",
        "stain_handling": "stain kept in the data; the Y median is robust to it"
                          + ("; excluded fit intervals marked with hatching" if args.mark_stain else "; not marked on the figures"),
        "rows": "one row per timepoint, top = first timepoint",
        "timepoints_zero_based": timepoints,
        "time_h": times_h.tolist(),
        "pixel_size_mm": pixel_mm,
        "x_mm_of_column_center": "(column + 0.5) * pixel_size_mm",
        "shape_rows_columns": list(kymo[channels[0]].shape),
        "intensity_unit": "a.u. (background-subtracted, illumination-corrected)",
        "display_ranges": {c: list(ranges[c]) for c in channels},
        "display_rule": f"0 to the {DISPLAY_PERCENTILE}th percentile of the channel's chemograph, same for all timepoints",
        "bridge_mm": [bridge["left_foot_mm"], bridge["right_foot_mm"]] if bridge else None,
        "stain_excluded_intervals_mm": stain,
        "source_tiffs": sources,
    }, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
