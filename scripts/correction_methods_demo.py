"""Explain the illumination corrections with images: raw, FFC only, overlap quadratic only, hybrid.

Uses the cached measurements of a completed calibration-backed review run (read only):
correction maps for one channel and Z plane, one example tile with every correction,
and median-over-Y profiles of that tile.

    .\\.venv\\Scripts\\python.exe scripts\\correction_methods_demo.py ^
        --run outputs\\2026_09_27_exp106_try4_full_experiment\\run_... ^
        --config exp106_2026_09_27_full_experiment_config.json ^
        --nd2 "D:\\Users\\Shelly\\2026_09_27_Exp106_try4\\gradient.nd2" --label 2026_09_27_exp106_try4 ^
        --output outputs\\2026_10_08_exp106_try5_full_experiment
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gradient_analysis.config import AnalysisConfig  # noqa: E402
from gradient_analysis.correction_comparison import (  # noqa: E402
    RunLog, collect_measurements, fit_z_series, log_illumination, pair_samples,
)

NAMES = {"raw": "No correction", "ffc": "FFC only (calibration flat field)",
         "overlap": "Overlap quadratic only", "hybrid": "Hybrid: FFC + residual overlap quadratic"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--run", "--config", "--nd2", "--label", "--output"):
        parser.add_argument(name, required=True)
    parser.add_argument("--channel", default="GFP")
    parser.add_argument("--timepoint-index", type=int, default=None, help="Index into the run's timepoints (default: middle).")
    args = parser.parse_args()
    run = Path(args.run)
    config = AnalysisConfig.from_json(args.config)
    cache = run / "cache_measurements_do_not_edit"
    sources = json.loads((cache / "cache_sources.json").read_text(encoding="utf-8"))["reused_from"]
    out = Path(args.output) / f"correction_methods_explained_with_{args.label}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    B = collect_measurements(args.nd2, config, out / "scratch_cache_copy", RunLog(out / "progress_log.txt"),
                             reuse_dirs=[cache, *map(Path, sources)])
    c = B["labels"].index(args.channel)
    z = config.zero_based_z
    T, K, Z, W = len(B["timepoints"]), len(B["pair_geometry"]), B["z_count"], B["width"]
    ti = T // 2 if args.timepoint_index is None else args.timepoint_index
    penalty = float(config.all_z_coefficient_smoothing_penalty)

    def correction(base: str) -> np.ndarray:
        samples = []
        for zz in range(Z):
            parts = [pair_samples(B["ratios"][base][t, c, zz], B["x_offsets"], list(range(K))) for t in range(T)]
            samples.append(tuple(np.concatenate([p[i] for p in parts]) for i in range(3)))
        fit = fit_z_series(samples, W, penalty)
        return np.exp(log_illumination(W, fit["linear"][z], fit["quadratic"][z]))

    flatfield = B["flatfields"][args.channel].astype(np.float64)
    overlap_curve = correction("raw")
    residual_curve = correction("ffc")
    maps = {
        "ffc": flatfield,
        "overlap": np.broadcast_to(overlap_curve[np.newaxis, :], flatfield.shape),
        "hybrid": flatfield * residual_curve[np.newaxis, :],
    }
    factor = B["bin_factor"]
    centers = (np.arange(B["binned_shape"][1]) * factor + (factor - 1) / 2.0).astype(int)
    tile_index = len(B["positions"]) // 2
    raw_tile = B["bins"]["raw"][ti, c, z, tile_index].astype(np.float64)
    ffc_tile = B["bins"]["ffc"][ti, c, z, tile_index].astype(np.float64)
    tiles = {"raw": raw_tile, "ffc": ffc_tile, "overlap": raw_tile / overlap_curve[centers][np.newaxis, :],
             "hybrid": ffc_tile / residual_curve[centers][np.newaxis, :]}
    hours = B["time_h"][ti]

    # correction maps
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), constrained_layout=True, gridspec_kw={"height_ratios": [3, 1.6]})
    low = min(float(m.min()) for m in maps.values())
    high = max(float(m.max()) for m in maps.values())
    for col, key in enumerate(("ffc", "overlap", "hybrid")):
        image = axes[0, col].imshow(maps[key], cmap="viridis", vmin=low, vmax=high)
        axes[0, col].set(title=f"{NAMES[key]}\nrelative illumination the correction divides out", xlabel="Camera X [pixel]", ylabel="Camera Y [pixel]")
        fig.colorbar(image, ax=axes[0, col], shrink=0.8, label="Relative illumination [–] (1 = average)")
        axes[1, col].plot(np.median(maps[key], axis=0), color="tab:green", label="along camera X (median over Y)")
        axes[1, col].plot(np.median(maps[key], axis=1), color="tab:blue", ls="--", label="along camera Y (median over X)")
        axes[1, col].axhline(1, color="black", lw=0.6)
        axes[1, col].set(xlabel="Camera position [pixel]", ylabel="Relative illumination [–]", ylim=(low * 0.95, high * 1.05))
        axes[1, col].grid(alpha=0.2)
        axes[1, col].legend(fontsize=8)
    fig.suptitle(f"{args.label} · {args.channel} · Z{z + 1}: what each correction removes (20× data; FFC needs a 20× uniform standard)")
    fig.savefig(out / f"{args.label}_{args.channel}_Z{z + 1:02d}_correction_maps_FFC_quadratic_hybrid.png", dpi=140)
    plt.close(fig)

    # example tile before/after
    fig, axes = plt.subplots(2, 4, figsize=(21, 9.5), constrained_layout=True, gridspec_kw={"height_ratios": [3, 1.7]})
    low, high = np.percentile(np.concatenate([t.ravel() for t in tiles.values()]), [1, 99.5])
    x = np.arange(raw_tile.shape[1]) * factor
    for col, key in enumerate(("raw", "ffc", "overlap", "hybrid")):
        axes[0, col].imshow(tiles[key], cmap="gray", vmin=low, vmax=high)
        axes[0, col].set(title=NAMES[key], xticks=[], yticks=[])
        profile = np.median(tiles[key], axis=0)
        axes[1, col].plot(x, profile, color="black")
        axes[1, col].set(xlabel="Camera X [pixel]", ylabel="Intensity [a.u.] (median over Y)", ylim=(low, high))
        axes[1, col].grid(alpha=0.2)
    fig.suptitle(f"{args.label} · one tile ({B['positions'][tile_index].name}) · {args.channel} · Z{z + 1} · {hours:.2f}".rstrip("0").rstrip(".") + " h: "
                 "the same image with each correction (same display range). The real gradient runs along X; "
                 "dark edges are illumination fall-off.")
    fig.savefig(out / f"{args.label}_{args.channel}_Z{z + 1:02d}_one_tile_with_each_correction.png", dpi=140)
    plt.close(fig)
    (out / "README.txt").write_text(
        "FFC (flat-field correction): divide every image by the 2-D illumination pattern measured on a uniform\n"
        "fluorescent standard. Overlap quadratic: neighbouring tiles image the same spot twice; the brightness ratio\n"
        "between the two views depends only on where the spot falls in each camera image, so a smooth curve along camera X\n"
        "(exp of a quadratic) is fitted that makes every overlap agree, then divided out. Hybrid: FFC first, then a\n"
        "residual quadratic fitted the same way on the FFC-corrected overlaps.\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
