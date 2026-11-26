"""Plateau / linearity check: raw versus corrected profiles, each normalized to its own maximum.

For every timepoint and channel the single-plane profile is built twice from the cached
measurements of a gradient-review run: without illumination correction ("raw") and with the
run's primary correction. Each profile is divided by its own (lightly smoothed) maximum, so
only the shape is compared. Shape measures:

* source-edge slope: straight-line slope over the 0.5 mm at the dye's source edge
  (GFP: left edge, Cy5: right edge), where a plateau would appear;
* steepest slope anywhere in the profile (0.1 mm smoothing, stain positions excluded);
* plateau index = |source-edge slope| / |steepest slope|: 0 = flat plateau at the edge,
  1 = the profile is as steep at the edge as anywhere (no plateau);
* the same for the far edge (the low plateau opposite the source), plus its level;
* R² of one straight line over the whole field versus R² of the tanh fit.

Example:
    .\\.venv\\Scripts\\python.exe scripts\\analyze_profile_shape.py ^
        --run outputs\\2026_09_27_exp106_try4_full_experiment\\run_... --label 2026_09_27_exp106_try4 ^
        --config exp106_2026_09_27_full_experiment_config.json ^
        --nd2 "D:\\Users\\Shelly\\2026_09_27_Exp106_try4\\gradient.nd2" --output outputs\\profile_shape_check
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
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gradient_analysis.config import AnalysisConfig  # noqa: E402
from gradient_analysis.correction_comparison import (  # noqa: E402
    RunLog, collect_measurements, fit_z_series, log_illumination, pair_samples,
)
from gradient_analysis.outputs import write_rows_csv  # noqa: E402
from gradient_analysis.processing import feather_profiles, fit_tanh_profile, tanh_profile  # noqa: E402

EDGE_MM = 0.5
SOURCE_SIDE = {"GFP": "left", "Cy5": "right"}


def shape_measures(x: np.ndarray, y: np.ndarray, excluded: np.ndarray, side: str) -> dict[str, float]:
    step = float(np.median(np.diff(x)))
    keep = np.isfinite(y) & ~excluded
    filled = np.interp(x, x[keep], y[keep])
    smooth = ndimage.gaussian_filter1d(filled, 0.1 / step)
    derivative = np.gradient(smooth, x)
    derivative[~keep] = 0.0
    steepest = float(np.max(np.abs(derivative)))
    edge = (x <= x[0] + EDGE_MM) if side == "left" else (x >= x[-1] - EDGE_MM)
    far = (x >= x[-1] - EDGE_MM) if side == "left" else (x <= x[0] + EDGE_MM)
    edge &= keep
    far &= keep
    edge_slope = float(np.polyfit(x[edge], y[edge], 1)[0])
    far_slope = float(np.polyfit(x[far], y[far], 1)[0])
    line = np.polyfit(x[keep], y[keep], 1)
    residual = y[keep] - np.polyval(line, x[keep])
    ss_total = float(np.sum((y[keep] - y[keep].mean()) ** 2))
    fit_y = y.copy()
    fit_y[excluded] = np.nan
    tanh = fit_tanh_profile(x, fit_y, max_points=1024, pixel_size_mm=step)
    return {
        "source_edge_slope_per_mm": edge_slope,
        "steepest_slope_per_mm": steepest,
        "plateau_index": abs(edge_slope) / steepest if steepest > 0 else float("nan"),
        "far_edge_slope_per_mm": far_slope,
        "far_edge_level": float(np.median(y[far])),
        "far_plateau_index": abs(far_slope) / steepest if steepest > 0 else float("nan"),
        "straight_line_r_squared": 1.0 - float(np.sum(residual ** 2)) / ss_total,
        "tanh_r_squared": float(tanh["r_squared"]),
        "tanh_fit_valid": bool(tanh["fit_valid"]),
        "tanh_flags": ";".join(tanh["qc_flags"]),
        "_tanh": tanh,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Completed gradient-review run directory (read only).")
    parser.add_argument("--label", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--nd2", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--z-first", type=int, default=1, help="First one-based Z plane used for the correction fit.")
    args = parser.parse_args()

    run = Path(args.run)
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    config = AnalysisConfig.from_json(args.config)
    cache = run / "cache_measurements_do_not_edit"
    sources = json.loads((cache / "cache_sources.json").read_text(encoding="utf-8"))["reused_from"]
    out = Path(args.output) / f"{args.label}_profile_shape_raw_vs_corrected_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    log = RunLog(out / "progress_log.txt")
    B = collect_measurements(args.nd2, config, out / "scratch_cache_copy", log, reuse_dirs=[cache, *map(Path, sources)])
    primary = metadata["primary_method"]
    base = "ffc" if primary == "hybrid" else "raw"
    z = config.zero_based_z
    labels, timepoints = B["labels"], B["timepoints"]
    T, K, Z = len(timepoints), len(B["pair_geometry"]), B["z_count"]
    hour_of = dict(zip(timepoints, B["time_h"]))

    def hl(value: float) -> str:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    stain = metadata.get("stain_mask") or {}
    bridge = metadata.get("bridge_from_DIC")

    rows: list[dict[str, object]] = []
    shapes: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray, dict]] = {}
    for c, label in enumerate(labels):
        samples = []
        first = args.z_first - 1
        for zz in range(first, Z):
            parts = [pair_samples(B["ratios"][base][t, c, zz], B["x_offsets"], list(range(K))) for t in range(T)]
            samples.append(tuple(np.concatenate([p[i] for p in parts]) for i in range(3)))
        fit = fit_z_series(samples, B["width"], float(config.all_z_coefficient_smoothing_penalty))
        correction = np.exp(log_illumination(B["width"], fit["linear"][z - first], fit["quadratic"][z - first]))
        for ti, t in enumerate(timepoints):
            for version, tiles in (("raw", B["profiles"]["raw"][ti, c, z]),
                                   ("corrected", B["profiles"][base][ti, c, z] / correction[np.newaxis, :])):
                x, values, _ = feather_profiles(B["x_axes"], list(tiles))
                x = x - x[0]
                excluded = np.zeros(x.shape, dtype=bool)
                for start, stop in stain.get("excluded_intervals_mm", []):
                    excluded |= (x >= start) & (x <= stop)
                smooth_max = float(np.max(ndimage.gaussian_filter1d(values, 0.05 / float(x[1] - x[0]))))
                y = values / smooth_max
                measures = shape_measures(x, y, excluded, SOURCE_SIDE.get(label, "left"))
                shapes[(label, version, t)] = (x, y, measures)
                rows.append({"channel": label, "version": version, "timepoint": t, "time_h": hour_of[t],
                             **{k: v for k, v in measures.items() if not k.startswith("_")}})
    write_rows_csv(out / f"{args.label}_profile_shape_measures_raw_vs_corrected.csv", rows)

    corrected_name = {"hybrid": "corrected (FFC + residual overlap)", "overlap": "corrected (overlap quadratic)"}[primary]
    per_t = out / "1_raw_vs_corrected_profiles_each_normalized_to_its_own_maximum"
    for t in timepoints:
        fig, axes = plt.subplots(len(labels), 1, figsize=(13, 4.4 * len(labels)), constrained_layout=True, sharex=True)
        for axis, label in zip(np.atleast_1d(axes), labels):
            if bridge:
                axis.axvspan(bridge["left_foot_mm"], bridge["right_foot_mm"], color="#d9c27a", alpha=0.25, lw=0, label="Bridge (from DIC)")
            for start, stop in stain.get("excluded_intervals_mm", []):
                axis.axvspan(start, stop, facecolor="none", edgecolor="#b22222", hatch="///", lw=0)
            for version, color in (("raw", "#7f7f7f"), ("corrected", "#2ca02c")):
                x, y, m = shapes[(label, version, t)]
                tanh = m["_tanh"]
                axis.plot(x, y, color=color, lw=1.0,
                          label=f"{'raw (no illumination correction)' if version == 'raw' else corrected_name}: plateau index "
                                f"{m['plateau_index']:.2f}, straight-line R² {m['straight_line_r_squared']:.3f}, tanh R² {m['tanh_r_squared']:.3f}"
                                + ("" if m["tanh_fit_valid"] else " (tanh flagged)"))
                axis.plot(x, tanh_profile(x, tanh["left_plateau"], tanh["right_plateau"], tanh["midpoint_mm"], tanh["width_mm"]),
                          color=color, lw=1.0, ls="--")
            side = SOURCE_SIDE.get(label, "left")
            x = shapes[(label, "raw", t)][0]
            edge = (x[0], x[0] + EDGE_MM) if side == "left" else (x[-1] - EDGE_MM, x[-1])
            axis.axvspan(*edge, color="#9ecae1", alpha=0.35, lw=0, label=f"Source edge ({EDGE_MM:g} mm) where a plateau is expected")
            axis.set(xlabel="Position along device [mm]", ylabel=f"{label} intensity [fraction of this profile's maximum]")
            axis.set_ylim(-0.05, 1.08)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7.5, loc="best")
        np.atleast_1d(axes)[0].set_title(
            f"{args.label} · t{t:03d} ({hl(hour_of[t])} h) · Z{z + 1} · raw vs corrected, each normalized to its own maximum "
            f"(solid = data, dashed = tanh fit)")
        per_t.mkdir(parents=True, exist_ok=True)
        fig.savefig(per_t / f"{args.label}_t{t:03d}_{hl(hour_of[t])}h_Z{z + 1:02d}_raw_vs_corrected_own-max-normalized.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, len(labels), figsize=(7.5 * len(labels), 8), constrained_layout=True, squeeze=False)
    for c, label in enumerate(labels):
        for version, color in (("raw", "#7f7f7f"), ("corrected", "#2ca02c")):
            selected = [r for r in rows if r["channel"] == label and r["version"] == version]
            times = [r["time_h"] for r in selected]
            axes[0, c].plot(times, [r["plateau_index"] for r in selected], "o-", color=color, ms=4,
                            label=f"source edge, {'raw' if version == 'raw' else corrected_name}")
            axes[0, c].plot(times, [r["far_plateau_index"] for r in selected], "s--", color=color, ms=4, mfc="none",
                            label=f"far edge, {'raw' if version == 'raw' else corrected_name}")
            axes[1, c].plot(times, [r["straight_line_r_squared"] for r in selected], "o-", color=color, ms=4,
                            label=f"straight line, {'raw' if version == 'raw' else 'corrected'}")
            axes[1, c].plot(times, [r["tanh_r_squared"] for r in selected], "s--", color=color, ms=4, mfc="none",
                            label=f"tanh, {'raw' if version == 'raw' else 'corrected'}")
        axes[0, c].axhline(0, color="black", lw=0.6)
        axes[0, c].set(xlabel="Time [h]", ylabel="Plateau index [–] (0 = flat edge, 1 = no plateau)",
                       title=f"{args.label} · {label}: how flat are the two ends of the profile?")
        axes[1, c].set(xlabel="Time [h]", ylabel="Fit quality R² [–]",
                       title=f"{args.label} · {label}: straight line vs tanh over the whole field")
        for axis in axes[:, c]:
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
    fig.savefig(out / f"{args.label}_plateau_index_and_linearity_over_time.png", dpi=160)
    plt.close(fig)
    (out / "README.txt").write_text(__doc__ + "\nRead-only inputs: " + str(run) + "\n", encoding="utf-8")
    log("Done")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
