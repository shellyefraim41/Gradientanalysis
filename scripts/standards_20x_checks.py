"""Checks on the calibration standards: background choice, 32 µg/mL with vs without device, FFC example.

All files in the calibration folder share one objective (20x for 2026-10-08), so these
comparisons are internally valid even when the gradient itself was imaged differently.

    .\\.venv\\Scripts\\python.exe scripts\\standards_20x_checks.py ^
        --calibration "D:\\Users\\Shelly\\2026_10_08_Exp106_try5" --unit ug_per_ml ^
        --device-file 32withdevice.nd2 --label 2026_10_08_exp106_try5 ^
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
import nd2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gradient_analysis.calibrated_pipeline import _discover_standards  # noqa: E402
from gradient_analysis.outputs import write_json, write_rows_csv  # noqa: E402
from gradient_analysis.processing import block_median, build_artifact_safe_flatfield, despike_image  # noqa: E402

CHANNELS = {"GFP": "gfp", "Cy5": "cy5"}
COLORS = {"GFP": "#20a83e", "Cy5": "#d62a8b"}
BLOCK = 128


def _stack(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """All planes per channel as float32 arrays (Z, Y, X), plus Z-loop metadata."""
    with nd2.ND2File(path) as handle:
        names = [str(c.channel.name).casefold() for c in handle.metadata.channels]
        data = np.asarray(handle.asarray(), dtype=np.float32)  # Z, C, Y, X
        loop = next((l for l in handle.experiment if l.type == "ZStackLoop"), None)
        info = {"home_index_zero_based": int(loop.parameters.homeIndex) if loop else 0,
                "step_um": float(loop.parameters.stepUm) if loop else 10.0,
                "objective": handle.metadata.channels[0].microscope.objectiveName,
                "pixel_um": float(handle.voxel_size().x)}
    out = {}
    for label, key in CHANNELS.items():
        index = next(i for i, name in enumerate(names) if key in name)
        out[label] = data[:, index]
    return out, info


def _heights(info: dict[str, object], z_count: int) -> np.ndarray:
    return (np.arange(z_count) - int(info["home_index_zero_based"])) * float(info["step_um"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--unit", default="ng_per_ml")
    parser.add_argument("--device-file", default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--background", type=float, default=100.0)
    args = parser.parse_args()
    directory = Path(args.calibration)
    standards = _discover_standards(directory, args.unit)
    out = Path(args.output) / f"{args.label}_calibration_standards_checks_{datetime.now():%Y%m%d_%H%M%S}"
    folders = {
        "background": out / "1_background_blank_image_vs_constant_100",
        "device": out / "2_32ugmL_regular_vs_32ugmL_with_device",
        "ffc": out / "3_flat_field_correction_example",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=False)
    summary: dict[str, object] = {"calibration_folder": str(directory), "standards_ug_ml": [c for c, _ in standards]}

    # ---- 1. background: blank image vs constant 100 ------------------------------------
    blank, info = _stack(next(p for c, p in standards if c == 0.0))
    summary["objective"] = info["objective"]
    heights = _heights(info, next(iter(blank.values())).shape[0])
    lowest_c, lowest_path = next((c, p) for c, p in standards if c > 0)
    lowest, _ = _stack(lowest_path)
    background_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)
    for row, label in enumerate(CHANNELS):
        stack = blank[label]
        per_z = np.median(stack.reshape(stack.shape[0], -1), axis=1)
        blank_map = np.median(stack, axis=0)  # per camera pixel, median over all Z planes
        axes[row, 0].plot(heights, per_z, "o-", color=COLORS[label])
        axes[row, 0].axhline(args.background, color="black", ls="--", label=f"constant {args.background:g}")
        axes[row, 0].set(xlabel="Nominal height above glass [µm]", ylabel="Median blank intensity [a.u.]",
                         title=f"{label}: blank (0 µg/mL) median in every Z plane")
        axes[row, 0].legend()
        image = axes[row, 1].imshow(blank_map - args.background, cmap="coolwarm", vmin=-4, vmax=4)
        fig.colorbar(image, ax=axes[row, 1], label="Blank map − 100 [a.u.]")
        axes[row, 1].set(title=f"{label}: blank image (median over Z) minus 100", xlabel="Camera X [pixel]", ylabel="Camera Y [pixel]")
        axes[row, 2].hist((blank_map - args.background).ravel(), bins=200, range=(-10, 10), color=COLORS[label])
        axes[row, 2].set(xlabel="Blank map − 100 [a.u.]", ylabel="Pixels [count]", title=f"{label}: how far the blank map is from 100")
        plane = lowest[label][int(info["home_index_zero_based"]) + 9]  # nominal +90 µm (Z11)
        signal_constant = plane - args.background
        signal_blank = plane - blank_map
        smoothed_blank = block_median(blank_map, 16)
        background_rows.append({
            "channel": label,
            "blank_median_per_Z_min": float(per_z.min()), "blank_median_per_Z_max": float(per_z.max()),
            "blank_map_mean_minus_100": float(blank_map.mean() - args.background),
            "blank_map_1st_percentile_minus_100": float(np.percentile(blank_map, 1) - args.background),
            "blank_map_99th_percentile_minus_100": float(np.percentile(blank_map, 99) - args.background),
            "blank_map_spatial_sd": float(blank_map.std()),
            "blank_map_large_scale_range_16px_blocks": float(smoothed_blank.max() - smoothed_blank.min()),
            "lowest_standard_ug_ml": lowest_c,
            "lowest_standard_Z11_median_signal_constant100": float(np.median(signal_constant)),
            "lowest_standard_Z11_median_signal_blankmap": float(np.median(signal_blank)),
            "lowest_standard_Z11_negative_fraction_constant100": float(np.mean(signal_constant < 0)),
            "lowest_standard_Z11_negative_fraction_blankmap": float(np.mean(signal_blank < 0)),
        })
    fig.suptitle(f"{args.label}: background from the 0 µg/mL standard versus the constant camera background of {args.background:g} "
                 f"({info['objective']})")
    fig.savefig(folders["background"] / f"{args.label}_blank_vs_constant_100.png", dpi=150)
    plt.close(fig)
    write_rows_csv(folders["background"] / "background_comparison.csv", background_rows)
    summary["background"] = background_rows

    # ---- 2. 32 regular vs 32 with device --------------------------------------------------
    device_rows = []
    if args.device_file:
        regular, _ = _stack(next(p for c, p in standards if np.isclose(c, 32.0)))
        device, device_info = _stack(directory / args.device_file)
        z_count = regular["GFP"].shape[0]
        heights = _heights(info, z_count)
        fig, axes = plt.subplots(2, 4, figsize=(21, 9.5), constrained_layout=True)
        for row, label in enumerate(CHANNELS):
            a = np.maximum(regular[label] - args.background, 0)
            b = np.maximum(device[label] - args.background, 0)
            per_z_a = np.median(a.reshape(z_count, -1), axis=1)
            per_z_b = np.median(b.reshape(z_count, -1), axis=1)
            axes[row, 0].plot(heights, per_z_a, "o-", color=COLORS[label], label="32 µg/mL (regular)")
            axes[row, 0].plot(heights, per_z_b, "s--", color="black", label="32 µg/mL with device")
            axes[row, 0].set(xlabel="Nominal height above glass [µm]", ylabel="Median intensity − 100 [a.u.]",
                             title=f"{label}: brightness in every Z plane")
            axes[row, 0].legend(fontsize=8)
            z11 = int(info["home_index_zero_based"]) + 9
            low, high = np.percentile(np.concatenate([a[z11].ravel(), b[z11].ravel()]), [1, 99.5])
            axes[row, 1].imshow(a[z11], cmap="gray", vmin=low, vmax=high)
            axes[row, 1].set(title=f"{label} regular, Z11 (same display)", xticks=[], yticks=[])
            axes[row, 2].imshow(b[z11], cmap="gray", vmin=low, vmax=high)
            axes[row, 2].set(title=f"{label} with device, Z11 (same display)", xticks=[], yticks=[])
            ratio = block_median(b[z11], BLOCK) / np.maximum(block_median(a[z11], BLOCK), 1e-6)
            image = axes[row, 3].imshow(ratio, cmap="coolwarm", vmin=np.nanpercentile(ratio, 2), vmax=np.nanpercentile(ratio, 98))
            fig.colorbar(image, ax=axes[row, 3], label="With device / regular [–]")
            axes[row, 3].set(title=f"{label}: ratio map at Z11 (128-px blocks)", xticks=[], yticks=[])
            layer = slice(int(info["home_index_zero_based"]) + 1, int(info["home_index_zero_based"]) + 11)  # Z3-Z12
            norm_a = block_median(a[z11], BLOCK) / np.median(block_median(a[z11], BLOCK))
            norm_b = block_median(b[z11], BLOCK) / np.median(block_median(b[z11], BLOCK))
            device_rows.append({
                "channel": label,
                "Z11_median_regular": float(per_z_a[z11]), "Z11_median_with_device": float(per_z_b[z11]),
                "Z11_ratio_device_over_regular": float(per_z_b[z11] / per_z_a[z11]),
                "Z3_Z12_mean_ratio_device_over_regular": float(per_z_b[layer].mean() / per_z_a[layer].mean()),
                "top_plane_ratio_device_over_regular": float(per_z_b[-1] / per_z_a[-1]),
                "per_Z_ratio": [float(v) for v in per_z_b / np.maximum(per_z_a, 1e-6)],
                "Z11_spatial_CV_regular": float(np.std(norm_a) / np.mean(norm_a)),
                "Z11_spatial_CV_with_device": float(np.std(norm_b) / np.mean(norm_b)),
                "Z11_illumination_shape_difference_max_abs": float(np.max(np.abs(norm_b - norm_a))),
            })
        fig.suptitle(f"{args.label}: 32 µg/mL in a regular well vs 32 µg/mL with the device ({device_info['objective']})")
        fig.savefig(folders["device"] / f"{args.label}_32ugmL_regular_vs_with_device.png", dpi=140)
        plt.close(fig)
        write_rows_csv(folders["device"] / "32_regular_vs_with_device.csv",
                       [{k: v for k, v in r.items() if k != "per_Z_ratio"} for r in device_rows])
        summary["32_regular_vs_with_device"] = device_rows

    # ---- 3. flat-field correction example ------------------------------------------------------
    reference, _ = _stack(next(p for c, p in standards if np.isclose(c, 32.0)))
    example_c, example_path = min(((c, p) for c, p in standards if 0 < c < 32), key=lambda item: abs(item[0] - 16))
    example, _ = _stack(example_path)
    z11 = int(info["home_index_zero_based"]) + 9
    fig, axes = plt.subplots(2, 4, figsize=(21, 9.5), constrained_layout=True)
    for row, label in enumerate(CHANNELS):
        planes = []
        for z in range(reference[label].shape[0]):
            repaired, _ = despike_image(reference[label][z])
            coarse = block_median(np.maximum(repaired - args.background, 0), 8)
            planes.append(coarse / np.median(coarse))
        flatfield, _ = build_artifact_safe_flatfield(np.stack(planes), reference[label].shape[1:])
        before = np.maximum(example[label][z11] - args.background, 0)
        after = before / flatfield
        low, high = np.percentile(np.concatenate([before.ravel(), after.ravel()]), [1, 99.5])
        image = axes[row, 0].imshow(flatfield, cmap="viridis")
        fig.colorbar(image, ax=axes[row, 0], label="Relative illumination [–]")
        axes[row, 0].set(title=f"{label}: flat field from 32 µg/mL (all Z planes)", xticks=[], yticks=[])
        axes[row, 1].imshow(before, cmap="gray", vmin=low, vmax=high)
        axes[row, 1].set(title=f"{label}: {example_c:g} µg/mL at Z11 before FFC", xticks=[], yticks=[])
        axes[row, 2].imshow(after, cmap="gray", vmin=low, vmax=high)
        axes[row, 2].set(title=f"{label}: same image after FFC", xticks=[], yticks=[])
        x = np.arange(before.shape[1])
        for data, style, name in ((before, "-", "before FFC"), (after, "-", "after FFC")):
            axes[row, 3].plot(x, np.median(data, axis=0) / np.median(data), style, label=f"{name}: along X")
            axes[row, 3].plot(np.arange(before.shape[0]), np.median(data, axis=1) / np.median(data), "--", label=f"{name}: along Y")
        axes[row, 3].axhline(1, color="black", lw=0.6)
        axes[row, 3].set(xlabel="Camera position [pixel]", ylabel="Relative intensity [–]",
                         title=f"{label}: uniform solution should be flat (= 1)")
        axes[row, 3].legend(fontsize=7)
        summary.setdefault("ffc_example", []).append({
            "channel": label, "example_ug_ml": example_c,
            "center_to_corner_ratio_before": float(np.median(before[1000:1300, 1000:1300]) / np.median(before[:200, :200])),
            "center_to_corner_ratio_after": float(np.median(after[1000:1300, 1000:1300]) / np.median(after[:200, :200])),
            "flatfield_min": float(flatfield.min()), "flatfield_max": float(flatfield.max()),
        })
    fig.suptitle(f"{args.label}: flat-field correction (FFC) example on a uniform {example_c:g} µg/mL standard ({info['objective']})")
    fig.savefig(folders["ffc"] / f"{args.label}_flat_field_correction_before_after.png", dpi=140)
    plt.close(fig)
    write_json(out / "summary.json", summary)
    print(out)
    print(json.dumps(summary, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
