"""Six-step orchestration for the gradient experiment."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

from .config import AnalysisConfig
from .nd2_source import ND2Source
from .outputs import (
    save_color_preview,
    save_normalization_plot,
    save_preview,
    save_profile_plot,
    save_rgb,
    save_slope_timecourse_plot,
    save_tiff,
    save_timecourse_plot,
    write_csv,
    write_json,
)
from .processing import (
    choose_reference,
    correct_tile,
    empty_uint16_histogram,
    fitted_illumination_profile,
    histogram_percentile_range,
    image_percentile_range,
    linear_fit,
    merge_rgb,
    position_span,
    stitch,
    to_uint16,
    update_uint16_histogram,
    x_profile,
)


def _safe_stem(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text).strip("_")


def _timepoint_folder(parent: Path, timepoint: int) -> Path:
    """Return the zero-padded timepoint subfolder used by Step 1."""
    return parent / f"t{timepoint:03d}"


def run_pipeline(nd2_path: str | Path, output_root: str | Path, config: AnalysisConfig) -> Path:
    """Run all requested steps and return the newly created run directory."""
    source_path = Path(nd2_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"run_{_safe_stem(source_path.stem)}_{stamp}"
    step1 = run_dir / "step_01_stitched_images"
    step2 = run_dir / "step_02_x_profiles"
    step3 = run_dir / "step_03_illumination_corrected"
    step4 = run_dir / "step_04_timecourse_profiles"
    step5 = run_dir / "step_05_corrected_x_profiles"
    step6 = run_dir / "step_06_bridge_slopes"
    for folder in (step1, step2, step3, step4, step5, step6):
        folder.mkdir(parents=True, exist_ok=False)

    timecourse: dict[str, dict[int, np.ndarray]] = {c.label: {} for c in config.channels}
    raw_histograms = {c.label: empty_uint16_histogram() for c in config.channels}
    corrected_histograms = {c.label: empty_uint16_histogram() for c in config.channels}
    raw_tiff_paths: dict[str, dict[int, Path]] = {c.label: {} for c in config.channels}
    corrected_tiff_paths: dict[str, dict[int, Path]] = {c.label: {} for c in config.channels}
    bridge_records: list[dict[str, float | int | str | None]] = []
    local_preview_ranges: dict[str, dict[str, list[int]]] = {"raw": {}, "corrected": {}}
    metadata: dict[str, object] = {}
    base = _safe_stem(source_path.stem)

    with ND2Source(source_path, config) as source:
        bridge_list_index = config.bridge_position - 1
        if not 0 <= bridge_list_index < len(source.positions):
            raise ValueError(
                f"Bridge position {config.bridge_position} is outside the available "
                f"positions 1-{len(source.positions)}."
            )

        metadata = {
            "source": str(source_path.resolve()),
            "dimensions": source.sizes,
            "nd2_channel_names": source.channel_names,
            "resolved_channels": source.channel_indices,
            "z_index_requested": config.z_index,
            "z_index_zero_based": config.zero_based_z,
            "positions_left_to_right": [
                {"label": p.name, "nd2_index": p.index, "x_um": p.x_um, "y_um": p.y_um}
                for p in source.positions
            ],
            "pixel_size_um": source.pixel_size_um,
            "correction_method": "quadratic_reference_trendline",
            "correction_formula": (
                "corrected(y,x) = raw(y,x) * median(fitted_laser_profile) "
                "/ fitted_laser_profile(x)"
            ),
            "correction_fit_degree": config.correction_fit_degree,
            "bridge_position_one_based": config.bridge_position,
            "bridge_position_label": source.positions[bridge_list_index].name,
            "bridge_position_nd2_index": source.positions[bridge_list_index].index,
            "step_04_values": "corrected stitched x profiles with P04 bridge slope annotations",
            "step_05_values": "corrected stitched x profiles with P04 bridge slope overlays",
            "step_06_values": "linear slopes fitted to corrected P04 bridge profiles",
        }

        for t in source.timepoints:
            raw_tiles: dict[str, list[np.ndarray]] = {}
            raw_stitched_profiles: dict[str, np.ndarray] = {}

            # Load at most two channels x seven positions for one timepoint.
            for channel in config.channels:
                tiles = [source.plane(t, p.index, channel) for p in source.positions]
                raw_tiles[channel.label] = tiles
                for tile in tiles:
                    update_uint16_histogram(raw_histograms[channel.label], tile)
                large = stitch(tiles)
                raw_stitched_profiles[channel.label] = x_profile(large)
                prefix = f"{base}_t{t:03d}_z{config.z_index:02d}_{channel.label}_positions_left-to-right"
                tiff_path = _timepoint_folder(step1, t) / f"{prefix}.tif"
                save_tiff(tiff_path, large)
                raw_tiff_paths[channel.label][t] = tiff_path

            save_profile_plot(
                step2 / f"{base}_t{t:03d}_z{config.z_index:02d}_GFP-Cy5_x-profile.png",
                raw_stitched_profiles,
                config.channels,
                f"{source_path.name} — t={t}, Z={config.z_index} (raw)",
                config.trendline_window_px,
            )

            bridge_fits_for_t: dict[str, dict[str, float | int | str | None]] = {}
            for channel in config.channels:
                tiles = raw_tiles[channel.label]
                local_profiles = [x_profile(tile) for tile in tiles]
                manual_position = config.reference_positions.get(channel.label)
                if manual_position is None:
                    ref_list_index = choose_reference(local_profiles, config.smoothing_window_px)
                else:
                    indices = [p.index for p in source.positions]
                    if manual_position not in indices:
                        raise ValueError(
                            f"Reference position {manual_position} for {channel.label} is not in {indices}"
                        )
                    ref_list_index = indices.index(manual_position)

                fitted_laser_profile, plateau = fitted_illumination_profile(
                    local_profiles[ref_list_index],
                    config.smoothing_window_px,
                    config.correction_fit_degree,
                )
                curve = plateau / fitted_laser_profile
                corrected = [correct_tile(tile, curve) for tile in tiles]
                corrected_profiles = [x_profile(tile) for tile in corrected]
                # Step 4, Step 5, and Step 6 all use these float-precision values.
                timecourse[channel.label][t] = np.concatenate(corrected_profiles)

                start_px, end_px = position_span(bridge_list_index, tiles[bridge_list_index].shape[1])
                fit = linear_fit(corrected_profiles[bridge_list_index], source.pixel_size_um)
                bridge_record = {
                    "timepoint": t,
                    "channel": channel.label,
                    "position_label": source.positions[bridge_list_index].name,
                    "position_one_based": config.bridge_position,
                    "position_nd2_index": source.positions[bridge_list_index].index,
                    "start_px": start_px,
                    "end_px": end_px,
                    **fit,
                }
                bridge_records.append(bridge_record)
                bridge_fits_for_t[channel.label] = bridge_record

                converted_tiles: list[np.ndarray] = []
                clipped_pixels = 0
                for tile in corrected:
                    converted, clipped = to_uint16(tile)
                    converted_tiles.append(converted)
                    clipped_pixels += clipped
                    update_uint16_histogram(corrected_histograms[channel.label], converted)

                channel_dir = step3 / f"t{t:03d}" / channel.label
                channel_dir.mkdir(parents=True, exist_ok=True)
                if config.save_corrected_tiles:
                    for position, tile in zip(source.positions, converted_tiles):
                        save_tiff(
                            channel_dir
                            / f"{base}_t{t:03d}_z{config.z_index:02d}_{channel.label}_{position.name}_corrected.tif",
                            tile,
                        )
                corrected_tiff = (
                    channel_dir
                    / f"{base}_t{t:03d}_z{config.z_index:02d}_{channel.label}_corrected_stitched.tif"
                )
                save_tiff(corrected_tiff, stitch(converted_tiles))
                corrected_tiff_paths[channel.label][t] = corrected_tiff

                save_normalization_plot(
                    channel_dir
                    / f"{base}_t{t:03d}_z{config.z_index:02d}_{channel.label}_normalization_profiles.png",
                    local_profiles,
                    corrected_profiles,
                    fitted_laser_profile,
                    ref_list_index,
                    [p.name for p in source.positions],
                    channel,
                    f"{source_path.name} — t={t}, {channel.label}; plateau={plateau:.3g}",
                )
                write_json(
                    channel_dir / "normalization_details.json",
                    {
                        "reference_position_label": source.positions[ref_list_index].name,
                        "reference_nd2_index": source.positions[ref_list_index].index,
                        "selection": "automatic" if manual_position is None else "manual",
                        "plateau": plateau,
                        "smoothing_window_px": config.smoothing_window_px,
                        "trendline_window_px": config.trendline_window_px,
                        "correction_fit_degree": config.correction_fit_degree,
                        "correction_method": "quadratic_reference_trendline",
                        "corrected_tiff_dtype": "uint16",
                        "rounded_or_clipped_range": [0, 65535],
                        "clipped_pixel_count": clipped_pixels,
                    },
                )

            corrected_at_t = {c.label: timecourse[c.label][t] for c in config.channels}
            save_profile_plot(
                step5
                / f"{base}_t{t:03d}_z{config.z_index:02d}_GFP-Cy5_corrected_x-profile.png",
                corrected_at_t,
                config.channels,
                f"{source_path.name} — t={t}, Z={config.z_index} (corrected)",
                config.trendline_window_px,
                bridge_fits_for_t,
            )

        raw_ranges = {
            c.label: histogram_percentile_range(
                raw_histograms[c.label],
                config.preview_low_percentile,
                config.preview_high_percentile,
            )
            for c in config.channels
        }
        corrected_ranges = {
            c.label: histogram_percentile_range(
                corrected_histograms[c.label],
                config.preview_low_percentile,
                config.preview_high_percentile,
            )
            for c in config.channels
        }
        metadata["display_scaling"] = {
            "global_method": "global uint16 histogram percentiles across selected timepoints",
            "local_method": "per-image uint16 histogram percentiles",
            "percentiles": [
                config.preview_low_percentile,
                config.preview_high_percentile,
            ],
            "raw_ranges_by_channel": {key: list(value) for key, value in raw_ranges.items()},
            "corrected_ranges_by_channel": {
                key: list(value) for key, value in corrected_ranges.items()
            },
            "png_note": (
                "Single-channel *_preview.png files use local per-image scaling for "
                "visibility. *_global_preview.png files use one shared channel range "
                "across selected timepoints for fair comparison. TIFF files retain "
                "uint16 intensity values."
            ),
        }

        # The global limits are known only after the streaming pass. Create PNGs
        # from saved TIFFs, avoiding a second read of the 250 GB ND2.
        for t in source.timepoints:
            raw_images: list[np.ndarray] = []
            for channel in config.channels:
                path = raw_tiff_paths[channel.label][t]
                image = tifffile.imread(path)
                raw_images.append(image)
                local_range = image_percentile_range(
                    image,
                    config.preview_low_percentile,
                    config.preview_high_percentile,
                )
                local_preview_ranges["raw"][f"t{t:03d}_{channel.label}"] = list(local_range)
                save_preview(
                    path.with_name(f"{path.stem}_grayscale_preview.png"),
                    image,
                    *local_range,
                )
                save_color_preview(
                    path.with_name(f"{path.stem}_preview.png"),
                    image,
                    *local_range,
                    channel.rgb,
                )
                save_color_preview(
                    path.with_name(f"{path.stem}_global_preview.png"),
                    image,
                    *raw_ranges[channel.label],
                    channel.rgb,
                )

            merged = merge_rgb(
                raw_images,
                [c.rgb for c in config.channels],
                [raw_ranges[c.label] for c in config.channels],
            )
            save_rgb(
                _timepoint_folder(step1, t)
                / f"{base}_t{t:03d}_z{config.z_index:02d}_GFP-Cy5_merge.png",
                merged,
            )

            for channel in config.channels:
                path = corrected_tiff_paths[channel.label][t]
                image = tifffile.imread(path)
                local_range = image_percentile_range(
                    image,
                    config.preview_low_percentile,
                    config.preview_high_percentile,
                )
                local_preview_ranges["corrected"][f"t{t:03d}_{channel.label}"] = list(local_range)
                save_preview(
                    path.with_name(f"{path.stem}_grayscale_preview.png"),
                    image,
                    *local_range,
                )
                save_color_preview(
                    path.with_name(f"{path.stem}_preview.png"),
                    image,
                    *local_range,
                    channel.rgb,
                )
                save_color_preview(
                    path.with_name(f"{path.stem}_global_preview.png"),
                    image,
                    *corrected_ranges[channel.label],
                    channel.rgb,
                )
        metadata["display_scaling"]["local_ranges_by_image"] = local_preview_ranges

        for channel in config.channels:
            save_timecourse_plot(
                step4
                / f"{base}_z{config.z_index:02d}_{channel.label}_all-timepoints_x-profiles.png",
                timecourse[channel.label],
                channel,
                f"{source_path.name} — {channel.label}, all corrected timepoints, Z={config.z_index}",
                config.trendline_window_px,
                bridge_records,
            )
            save_slope_timecourse_plot(
                step6 / f"{base}_z{config.z_index:02d}_{channel.label}_P04_bridge_slope_over_time.png",
                bridge_records,
                channel,
                f"{source_path.name} — {channel.label}, corrected P04 bridge slope",
            )

        write_csv(step6 / f"{base}_z{config.z_index:02d}_P04_bridge_slopes.csv", bridge_records)
        write_json(step6 / f"{base}_z{config.z_index:02d}_P04_bridge_slopes.json", bridge_records)

    write_json(run_dir / "run_metadata.json", metadata)
    return run_dir
