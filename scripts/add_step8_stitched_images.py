"""Add corrected all-Z stitched PNG previews to an existing completed run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gradient_analysis.config import AnalysisConfig
from gradient_analysis.nd2_source import ND2Source
from gradient_analysis.outputs import (
    save_color_preview,
    save_rgb,
    write_json,
    write_rows_csv,
)
from gradient_analysis.processing import (
    correct_tile,
    image_percentile_range,
    merge_rgb,
    stitch,
    subtract_background_floor,
    to_uint16,
)


def _correction_from_model(model: dict[str, object], tile_width: int) -> np.ndarray:
    center = float(model["normalized_x_center_px"])
    scale = float(model["normalized_x_scale_px"])
    normalized_x = (np.arange(tile_width, dtype=np.float64) - center) / scale
    illumination = np.exp(
        float(model["linear_coefficient"]) * normalized_x
        + float(model["quadratic_coefficient"]) * normalized_x**2
    )
    return 1.0 / illumination


def add_stitched_images(
    nd2_path: Path,
    run_dir: Path,
    config: AnalysisConfig,
) -> Path:
    step8 = run_dir / "step_08_selected_timepoints_all_z"
    models_path = (
        run_dir
        / "step_03_illumination_corrected"
        / "overlap_correction_diagnostics"
        / "overlap_quadratic_models.json"
    )
    models = json.loads(models_path.read_text(encoding="utf-8"))
    stitched_root = step8 / "stitched_images"
    stitched_root.mkdir(parents=True, exist_ok=True)
    base = nd2_path.stem
    rows: list[dict[str, object]] = []

    with ND2Source(nd2_path, config) as source:
        z_count = int(source.sizes.get("Z", 1))
        tile_width = int(source.sizes["X"])
        corrections = {
            channel.label: _correction_from_model(models[channel.label], tile_width)
            for channel in config.channels
        }
        for timepoint in config.all_z_timepoints:
            timepoint_dir = stitched_root / f"t{timepoint:03d}"
            timepoint_dir.mkdir(parents=True, exist_ok=True)
            for z_index in range(z_count):
                stitched_images: list[np.ndarray] = []
                display_ranges: list[tuple[int, int]] = []
                for channel in config.channels:
                    tiles: list[np.ndarray] = []
                    clipped_pixels = 0
                    for position in source.positions:
                        signal = subtract_background_floor(
                            source.plane(timepoint, position.index, channel, z_index),
                            config.microscope_background,
                        )
                        converted, clipped = to_uint16(
                            correct_tile(signal, corrections[channel.label])
                        )
                        tiles.append(converted)
                        clipped_pixels += clipped
                    contact_sheet = stitch(tiles)
                    limits = image_percentile_range(
                        contact_sheet,
                        config.preview_low_percentile,
                        config.preview_high_percentile,
                    )
                    stitched_images.append(contact_sheet)
                    display_ranges.append(limits)
                    save_color_preview(
                        timepoint_dir
                        / f"{base}_t{timepoint:03d}_z{z_index + 1:02d}_{channel.label}_positions_left-to-right_preview.png",
                        contact_sheet,
                        *limits,
                        channel.rgb,
                    )
                    rows.append(
                        {
                            "timepoint": timepoint,
                            "z_index_zero_based": z_index,
                            "z_index_one_based": z_index + 1,
                            "channel": channel.label,
                            "display_low": limits[0],
                            "display_high": limits[1],
                            "clipped_pixel_count": clipped_pixels,
                            "scaling": "local percentile display only",
                        }
                    )
                save_rgb(
                    timepoint_dir
                    / f"{base}_t{timepoint:03d}_z{z_index + 1:02d}_GFP-Cy5_merge.png",
                    merge_rgb(
                        stitched_images,
                        [channel.rgb for channel in config.channels],
                        display_ranges,
                    ),
                )
                print(f"Completed t{timepoint:03d}, Z={z_index + 1}/{z_count}", flush=True)

    write_rows_csv(stitched_root / "stitched_image_display_ranges.csv", rows)
    write_json(stitched_root / "stitched_image_display_ranges.json", rows)
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["all_z_analysis"]["stitched_images"] = {
        "folder": "step_08_selected_timepoints_all_z/stitched_images",
        "count": len(config.all_z_timepoints) * int(metadata["all_z_analysis"]["z_count"]) * 3,
        "channels": [channel.label for channel in config.channels],
        "merge": "GFP-Cy5",
        "scaling": "local percentile display only; PNGs are not measurement data",
        "clipped_pixel_count": sum(int(row["clipped_pixel_count"]) for row in rows),
    }
    write_json(metadata_path, metadata)
    return stitched_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("nd2_file", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    output = add_stitched_images(
        args.nd2_file,
        args.run_dir,
        AnalysisConfig.from_json(args.config),
    )
    print(f"Step 8 stitched images complete: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
