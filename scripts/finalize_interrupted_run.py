"""Finish preview PNGs and metadata after an analysis timeout during previews."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gradient_analysis.config import AnalysisConfig
from gradient_analysis.outputs import save_color_preview, save_preview, save_rgb, write_json
from gradient_analysis.processing import merge_rgb


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def finalize(
    run_dir: Path,
    reference_metadata_path: Path,
    completed_pilot_metadata_path: Path,
    config: AnalysisConfig,
) -> Path:
    reference = _load(reference_metadata_path)
    pilot = _load(completed_pilot_metadata_path)
    display = reference["display_scaling"]
    raw_ranges = display["raw_ranges_by_channel"]
    corrected_ranges = display["corrected_ranges_by_channel"]
    local = display["local_ranges_by_image"]
    base = Path(str(reference["source"])).stem

    for timepoint in range(int(reference["dimensions"]["T"])):
        t = f"t{timepoint:03d}"
        step1 = run_dir / "step_01_stitched_images" / t
        raw_images = []
        for channel in config.channels:
            path = next(step1.glob(f"*_{channel.label}_positions_left-to-right.tif"))
            image = tifffile.imread(path)
            raw_images.append(image)
            low, high = local["raw"][f"{t}_{channel.label}"]
            outputs = (
                (path.with_name(f"{path.stem}_grayscale_preview.png"), "gray"),
                (path.with_name(f"{path.stem}_preview.png"), "color"),
                (path.with_name(f"{path.stem}_global_preview.png"), "global"),
            )
            for output, kind in outputs:
                if output.exists():
                    continue
                if kind == "gray":
                    save_preview(output, image, low, high)
                elif kind == "color":
                    save_color_preview(output, image, low, high, channel.rgb)
                else:
                    global_low, global_high = raw_ranges[channel.label]
                    save_color_preview(output, image, global_low, global_high, channel.rgb)
        merge_path = step1 / f"{base}_t{timepoint:03d}_z{config.z_index:02d}_GFP-Cy5_merge.png"
        if not merge_path.exists():
            save_rgb(
                merge_path,
                merge_rgb(
                    raw_images,
                    [channel.rgb for channel in config.channels],
                    [tuple(raw_ranges[channel.label]) for channel in config.channels],
                ),
            )
        for channel in config.channels:
            path = next(
                (run_dir / "step_03_illumination_corrected" / t / channel.label).glob(
                    "*_corrected_contact-sheet.tif"
                )
            )
            image = tifffile.imread(path)
            low, high = local["corrected"][f"{t}_{channel.label}"]
            outputs = (
                (path.with_name(f"{path.stem}_grayscale_preview.png"), "gray"),
                (path.with_name(f"{path.stem}_preview.png"), "color"),
                (path.with_name(f"{path.stem}_global_preview.png"), "global"),
            )
            for output, kind in outputs:
                if output.exists():
                    continue
                if kind == "gray":
                    save_preview(output, image, low, high)
                elif kind == "color":
                    save_color_preview(output, image, low, high, channel.rgb)
                else:
                    global_low, global_high = corrected_ranges[channel.label]
                    save_color_preview(output, image, global_low, global_high, channel.rgb)

    step8 = run_dir / "step_08_selected_timepoints_all_z"
    all_z = deepcopy(pilot["all_z_analysis"])
    selected = list(config.all_z_timepoints)
    all_z["selected_timepoints"] = selected
    all_z["record_count"] = len(selected) * int(all_z["z_count"]) * len(config.channels)
    all_z["normalization_constants"] = _load(step8 / "all_z_normalization_constants.json")
    all_z["correction_timepoints"] = selected
    all_z["coefficient_smoothing_penalty"] = config.all_z_coefficient_smoothing_penalty
    all_z["per_z_models"] = _load(
        step8 / "per_z_correction_diagnostics" / "per_z_quadratic_models.json"
    )
    display_rows = _load(step8 / "stitched_images" / "stitched_image_display_ranges.json")
    ranges = {}
    for channel in config.channels:
        row = next(item for item in display_rows if item["channel"] == channel.label)
        ranges[channel.label] = [int(row["display_low"]), int(row["display_high"])]
    all_z["stitched_images"].update(
        count=len(selected) * int(all_z["z_count"]) * 3,
        display_ranges=ranges,
        display_sample_stride=config.all_z_display_sample_stride,
        mosaic_tiffs_saved=config.all_z_save_mosaic_tiffs,
        clipped_pixel_count=sum(int(row["clipped_pixel_count"]) for row in display_rows),
    )

    metadata = deepcopy(reference)
    metadata["correction_policy"] = (
        "Z15 uses one fixed profile per channel across time; Step 8 estimates one "
        "profile per Z and quality-smooths its coefficients across Z"
    )
    metadata["analysis_controls"].update(
        all_z_coefficient_smoothing_penalty=config.all_z_coefficient_smoothing_penalty,
        all_z_display_sample_stride=config.all_z_display_sample_stride,
        all_z_save_mosaic_tiffs=config.all_z_save_mosaic_tiffs,
    )
    metadata["all_z_analysis"] = all_z
    write_json(run_dir / "run_metadata.json", metadata)
    return run_dir / "run_metadata.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("reference_metadata", type=Path)
    parser.add_argument("completed_pilot_metadata", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    path = finalize(
        args.run_dir,
        args.reference_metadata,
        args.completed_pilot_metadata,
        AnalysisConfig.from_json(args.config),
    )
    print(f"Finalized interrupted run: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
