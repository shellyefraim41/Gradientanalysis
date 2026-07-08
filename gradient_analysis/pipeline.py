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
    correct_tile,
    empty_uint16_histogram,
    fitted_illumination_profile,
    flatten_segments,
    histogram_percentile_range,
    image_percentile_range,
    linear_fit_xy,
    merge_rgb,
    physical_x_axes_mm,
    split_equal_width,
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


def _empty_reference() -> dict[str, object]:
    return {
        "mean_intensity": -np.inf,
        "timepoint": None,
        "position_list_index": None,
        "position_label": None,
        "position_nd2_index": None,
        "saturated_fraction": None,
        "profile": None,
        "used_saturated_fallback": False,
    }


def _maybe_update_reference(
    candidate: dict[str, object],
    *,
    mean_intensity: float,
    saturated_fraction: float,
    timepoint: int,
    position_list_index: int,
    position_label: str,
    position_nd2_index: int,
    profile: np.ndarray,
) -> None:
    if mean_intensity <= float(candidate["mean_intensity"]):
        return
    candidate.update(
        {
            "mean_intensity": mean_intensity,
            "timepoint": timepoint,
            "position_list_index": position_list_index,
            "position_label": position_label,
            "position_nd2_index": position_nd2_index,
            "saturated_fraction": saturated_fraction,
            "profile": profile.copy(),
        }
    )


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
    step6 = run_dir / "step_06_gradient_slopes"
    for folder in (step1, step2, step3, step4, step5, step6):
        folder.mkdir(parents=True, exist_ok=False)

    timecourse: dict[str, dict[int, list[tuple[np.ndarray, np.ndarray]]]] = {
        c.label: {} for c in config.channels
    }
    raw_histograms = {c.label: empty_uint16_histogram() for c in config.channels}
    corrected_histograms = {c.label: empty_uint16_histogram() for c in config.channels}
    raw_tiff_paths: dict[str, dict[int, Path]] = {c.label: {} for c in config.channels}
    corrected_tiff_paths: dict[str, dict[int, Path]] = {c.label: {} for c in config.channels}
    slope_records: list[dict[str, float | int | str | None]] = []
    local_preview_ranges: dict[str, dict[str, list[int]]] = {"raw": {}, "corrected": {}}
    metadata: dict[str, object] = {}
    base = _safe_stem(source_path.stem)

    with ND2Source(source_path, config) as source:
        if source.pixel_size_um is None:
            raise ValueError("Cannot build a physical mm axis because ND2 pixel size metadata were not found.")
        slope_start_index = config.slope_start_position - 1
        slope_end_index = config.slope_end_position - 1
        if not 0 <= slope_start_index <= slope_end_index < len(source.positions):
            raise ValueError(
                f"Slope positions must fall within 1-{len(source.positions)} and start <= end."
            )

        tile_width = int(source.sizes["X"])
        x_axes_mm = physical_x_axes_mm(
            [p.x_um for p in source.positions],
            source.pixel_size_um,
            tile_width,
        )
        position_ranges_mm = [
            {
                "label": position.name,
                "nd2_index": position.index,
                "x_start_mm": float(axis[0]),
                "x_end_mm": float(axis[-1]),
                "stage_x_um": position.x_um,
                "stage_y_um": position.y_um,
            }
            for position, axis in zip(source.positions, x_axes_mm)
        ]

        best_unsaturated = {c.label: _empty_reference() for c in config.channels}
        best_any = {c.label: _empty_reference() for c in config.channels}

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
            "position_ranges_mm_p01_left_edge_zero": position_ranges_mm,
            "pixel_size_um": source.pixel_size_um,
            "correction_method": "fixed_channel_reference_max_mean_quadratic_trendline",
            "correction_formula": (
                "corrected(y,x) = raw(y,x) * median(fitted_laser_profile) "
                "/ fitted_laser_profile(x); one fitted_laser_profile per channel"
            ),
            "correction_fit_degree": config.correction_fit_degree,
            "saturation_fraction_threshold": config.saturation_fraction_threshold,
            "slope_start_position": source.positions[slope_start_index].name,
            "slope_end_position": source.positions[slope_end_index].name,
            "step_02_values": "raw measured x profiles on physical mm axis; dashed lines span unmeasured gaps",
            "step_04_values": "corrected measured x profiles on physical mm axis with P01-P06 slope table",
            "step_05_values": "corrected measured x profiles per timepoint on physical mm axis",
            "step_06_values": "linear slopes fitted to corrected measured pixels from P01 through P06",
        }

        # Pass 1: read the ND2 once, save raw contact-sheet images, make raw
        # physical-axis plots, and select one fixed correction reference per channel.
        for t in source.timepoints:
            raw_segmented_profiles: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
            for channel in config.channels:
                tiles = [source.plane(t, p.index, channel) for p in source.positions]
                for i, (position, tile) in enumerate(zip(source.positions, tiles)):
                    update_uint16_histogram(raw_histograms[channel.label], tile)
                    profile = x_profile(tile)
                    mean_intensity = float(np.mean(tile))
                    saturated_fraction = float(np.count_nonzero(tile == np.iinfo(np.uint16).max) / tile.size)
                    _maybe_update_reference(
                        best_any[channel.label],
                        mean_intensity=mean_intensity,
                        saturated_fraction=saturated_fraction,
                        timepoint=t,
                        position_list_index=i,
                        position_label=position.name,
                        position_nd2_index=position.index,
                        profile=profile,
                    )
                    if saturated_fraction <= config.saturation_fraction_threshold:
                        _maybe_update_reference(
                            best_unsaturated[channel.label],
                            mean_intensity=mean_intensity,
                            saturated_fraction=saturated_fraction,
                            timepoint=t,
                            position_list_index=i,
                            position_label=position.name,
                            position_nd2_index=position.index,
                            profile=profile,
                        )

                raw_segmented_profiles[channel.label] = [
                    (axis, x_profile(tile)) for axis, tile in zip(x_axes_mm, tiles)
                ]
                large = stitch(tiles)
                prefix = f"{base}_t{t:03d}_z{config.z_index:02d}_{channel.label}_positions_left-to-right"
                tiff_path = _timepoint_folder(step1, t) / f"{prefix}.tif"
                save_tiff(tiff_path, large)
                raw_tiff_paths[channel.label][t] = tiff_path

            save_profile_plot(
                step2 / f"{base}_t{t:03d}_z{config.z_index:02d}_GFP-Cy5_x-profile-mm.png",
                raw_segmented_profiles,
                config.channels,
                f"{source_path.name} — t={t}, Z={config.z_index} (raw, physical x)",
                config.trendline_window_px,
            )

        fixed_corrections: dict[str, dict[str, object]] = {}
        for channel in config.channels:
            selected = best_unsaturated[channel.label]
            if selected["profile"] is None:
                selected = best_any[channel.label]
                selected["used_saturated_fallback"] = True
            fitted_laser_profile, plateau = fitted_illumination_profile(
                selected["profile"],  # type: ignore[arg-type]
                config.smoothing_window_px,
                config.correction_fit_degree,
            )
            fixed_corrections[channel.label] = {
                **{k: v for k, v in selected.items() if k != "profile"},
                "plateau": plateau,
                "fitted_laser_profile": fitted_laser_profile,
                "curve": plateau / fitted_laser_profile,
            }
        metadata["fixed_correction_references"] = {
            label: {k: v for k, v in details.items() if k not in {"fitted_laser_profile", "curve"}}
            for label, details in fixed_corrections.items()
        }

        # Pass 2: apply fixed per-channel correction to raw TIFF contact sheets.
        for t in source.timepoints:
            gradient_fits_for_t: dict[str, dict[str, float | int | str | None]] = {}
            corrected_at_t: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
            for channel in config.channels:
                raw_large = tifffile.imread(raw_tiff_paths[channel.label][t])
                tiles = split_equal_width(raw_large, len(source.positions))
                local_profiles = [x_profile(tile) for tile in tiles]
                curve = fixed_corrections[channel.label]["curve"]
                fitted_laser_profile = fixed_corrections[channel.label]["fitted_laser_profile"]
                plateau = float(fixed_corrections[channel.label]["plateau"])
                corrected = [correct_tile(tile, curve) for tile in tiles]  # type: ignore[arg-type]
                corrected_profiles = [x_profile(tile) for tile in corrected]
                corrected_segments = [
                    (axis, profile) for axis, profile in zip(x_axes_mm, corrected_profiles)
                ]
                corrected_at_t[channel.label] = corrected_segments
                timecourse[channel.label][t] = corrected_segments

                fit_x, fit_y = flatten_segments(
                    x_axes_mm,
                    corrected_profiles,
                    slope_start_index,
                    slope_end_index,
                )
                fit = linear_fit_xy(fit_x, fit_y)
                slope_record = {
                    "timepoint": t,
                    "channel": channel.label,
                    "start_position_label": source.positions[slope_start_index].name,
                    "end_position_label": source.positions[slope_end_index].name,
                    "start_position_one_based": config.slope_start_position,
                    "end_position_one_based": config.slope_end_position,
                    "x_start_mm": float(fit_x[0]),
                    "x_end_mm": float(fit_x[-1]),
                    "slope_au_per_mm": fit["slope"],
                    "intercept": fit["intercept"],
                    "r_squared": fit["r_squared"],
                }
                slope_records.append(slope_record)
                gradient_fits_for_t[channel.label] = slope_record

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
                    / f"{base}_t{t:03d}_z{config.z_index:02d}_{channel.label}_corrected_contact-sheet.tif"
                )
                save_tiff(corrected_tiff, stitch(converted_tiles))
                corrected_tiff_paths[channel.label][t] = corrected_tiff

                fixed_ref = fixed_corrections[channel.label]
                ref_index = (
                    int(fixed_ref["position_list_index"])
                    if fixed_ref["timepoint"] == t and fixed_ref["position_list_index"] is not None
                    else -1
                )
                save_normalization_plot(
                    channel_dir
                    / f"{base}_t{t:03d}_z{config.z_index:02d}_{channel.label}_normalization_profiles.png",
                    local_profiles,
                    corrected_profiles,
                    fitted_laser_profile,  # type: ignore[arg-type]
                    ref_index,
                    [p.name for p in source.positions],
                    channel,
                    (
                        f"{source_path.name} — t={t}, {channel.label}; fixed ref="
                        f"t{int(fixed_ref['timepoint']):03d} {fixed_ref['position_label']}; plateau={plateau:.3g}"
                    ),
                    corrected_segments,
                )
                write_json(
                    channel_dir / "normalization_details.json",
                    {
                        "fixed_reference_timepoint": fixed_ref["timepoint"],
                        "fixed_reference_position_label": fixed_ref["position_label"],
                        "fixed_reference_nd2_index": fixed_ref["position_nd2_index"],
                        "fixed_reference_mean_intensity": fixed_ref["mean_intensity"],
                        "fixed_reference_saturated_fraction": fixed_ref["saturated_fraction"],
                        "used_saturated_fallback": fixed_ref["used_saturated_fallback"],
                        "plateau": plateau,
                        "smoothing_window_px": config.smoothing_window_px,
                        "correction_fit_degree": config.correction_fit_degree,
                        "correction_method": "fixed_channel_reference_max_mean_quadratic_trendline",
                        "corrected_tiff_dtype": "uint16",
                        "rounded_or_clipped_range": [0, 65535],
                        "clipped_pixel_count": clipped_pixels,
                    },
                )

            save_profile_plot(
                step5
                / f"{base}_t{t:03d}_z{config.z_index:02d}_GFP-Cy5_corrected_x-profile-mm.png",
                corrected_at_t,
                config.channels,
                f"{source_path.name} — t={t}, Z={config.z_index} (corrected, physical x)",
                config.trendline_window_px,
                gradient_fits_for_t,
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
                save_preview(path.with_name(f"{path.stem}_grayscale_preview.png"), image, *local_range)
                save_color_preview(path.with_name(f"{path.stem}_preview.png"), image, *local_range, channel.rgb)
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
                save_preview(path.with_name(f"{path.stem}_grayscale_preview.png"), image, *local_range)
                save_color_preview(path.with_name(f"{path.stem}_preview.png"), image, *local_range, channel.rgb)
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
                / f"{base}_z{config.z_index:02d}_{channel.label}_all-timepoints_x-profiles-mm.png",
                timecourse[channel.label],
                channel,
                f"{source_path.name} — {channel.label}, all corrected timepoints, Z={config.z_index}",
                config.trendline_window_px,
                slope_records,
            )
            save_slope_timecourse_plot(
                step6 / f"{base}_z{config.z_index:02d}_{channel.label}_P01-P06_slope_over_time.png",
                slope_records,
                channel,
                f"{source_path.name} — {channel.label}, corrected P01-P06 slope",
            )

        write_csv(step6 / f"{base}_z{config.z_index:02d}_P01-P06_slopes.csv", slope_records)
        write_json(step6 / f"{base}_z{config.z_index:02d}_P01-P06_slopes.json", slope_records)

    write_json(run_dir / "run_metadata.json", metadata)
    return run_dir
