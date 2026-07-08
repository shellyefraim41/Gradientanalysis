"""Six-step orchestration for the gradient experiment."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

from .config import AnalysisConfig
from .nd2_source import ND2Source
from .outputs import (
    save_correction_method_comparison_plot,
    save_color_preview,
    save_flatfield_2d_comparison_plot,
    save_normalization_plot,
    save_preview,
    save_profile_plot,
    save_reference_selection_plot,
    save_rgb,
    save_slope_timecourse_plot,
    save_tiff,
    save_timecourse_plot,
    write_csv,
    write_json,
    write_rows_csv,
)
from .processing import (
    correct_tile,
    correct_tile_2d,
    empty_uint16_histogram,
    fitted_illumination_profile,
    flatten_segments,
    histogram_percentile_range,
    image_percentile_range,
    linear_fit_xy,
    merge_rgb,
    physical_x_axes_mm,
    profile_curvature,
    reference_candidate_score,
    smoothed_illumination_image,
    smoothed_illumination_profile,
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


def _slope_record(
    *,
    timepoint: int,
    channel_label: str,
    x_axes_mm: list[np.ndarray],
    profiles: list[np.ndarray],
    slope_start_index: int,
    slope_end_index: int,
    start_position_label: str,
    end_position_label: str,
    slope_start_position: int,
    slope_end_position: int,
) -> dict[str, float | int | str | None]:
    fit_x, fit_y = flatten_segments(
        x_axes_mm,
        profiles,
        slope_start_index,
        slope_end_index,
    )
    fit = linear_fit_xy(fit_x, fit_y)
    return {
        "timepoint": timepoint,
        "channel": channel_label,
        "start_position_label": start_position_label,
        "end_position_label": end_position_label,
        "start_position_one_based": slope_start_position,
        "end_position_one_based": slope_end_position,
        "x_start_mm": float(fit_x[0]),
        "x_end_mm": float(fit_x[-1]),
        "slope_au_per_mm": fit["slope"],
        "intercept": fit["intercept"],
        "r_squared": fit["r_squared"],
    }


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
    raw_step4 = step4 / "raw_comparison"
    raw_step5 = step5 / "raw_comparison"
    raw_step6 = step6 / "raw_comparison"
    diagnostics_dir = step3 / "reference_selection"
    smoothed_comparison_dir = step3 / "smoothed_profile_comparison"
    flatfield_2d_comparison_dir = step3 / "flatfield_2d_comparison"
    for folder in (
        step1,
        step2,
        step3,
        step4,
        step5,
        step6,
        raw_step4,
        raw_step5,
        raw_step6,
        diagnostics_dir,
        smoothed_comparison_dir,
        flatfield_2d_comparison_dir,
    ):
        folder.mkdir(parents=True, exist_ok=False)

    timecourse: dict[str, dict[int, list[tuple[np.ndarray, np.ndarray]]]] = {
        c.label: {} for c in config.channels
    }
    raw_timecourse: dict[str, dict[int, list[tuple[np.ndarray, np.ndarray]]]] = {
        c.label: {} for c in config.channels
    }
    raw_histograms = {c.label: empty_uint16_histogram() for c in config.channels}
    corrected_histograms = {c.label: empty_uint16_histogram() for c in config.channels}
    raw_tiff_paths: dict[str, dict[int, Path]] = {c.label: {} for c in config.channels}
    corrected_tiff_paths: dict[str, dict[int, Path]] = {c.label: {} for c in config.channels}
    slope_records: list[dict[str, float | int | str | None]] = []
    raw_slope_records: list[dict[str, float | int | str | None]] = []
    diagnostic_records: list[dict[str, object]] = []
    smoothed_diagnostic_records: list[dict[str, object]] = []
    flatfield_2d_diagnostic_records: list[dict[str, object]] = []
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

        reference_candidates = {
            c.label: [
                {
                    "position_list_index": i,
                    "position_label": p.name,
                    "position_nd2_index": p.index,
                    "profiles": [],
                    "tiles": [],
                    "mean_intensities": [],
                    "saturated_fractions": [],
                    "timepoints": [],
                }
                for i, p in enumerate(source.positions)
            ]
            for c in config.channels
        }

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
            "correction_method": "fixed_channel_reference_stable_bright_quadratic_trendline",
            "comparison_correction_method": "fixed_channel_reference_stable_bright_smoothed_profile",
            "flatfield_2d_comparison_method": "fixed_channel_reference_stable_bright_smoothed_2d_flatfield",
            "correction_formula": (
                "corrected(y,x) = raw(y,x) * median(fitted_laser_profile) "
                "/ fitted_laser_profile(x); one fitted_laser_profile per channel"
            ),
            "comparison_note": (
                "Primary corrected TIFFs use the quadratic fitted laser profile. "
                "step_03_illumination_corrected/smoothed_profile_comparison stores "
                "diagnostic plots and tables using the smoothed reference profile directly. "
                "step_03_illumination_corrected/flatfield_2d_comparison stores a "
                "smoothed 2-D reference-tile flat-field comparison."
            ),
            "correction_fit_degree": config.correction_fit_degree,
            "saturation_fraction_threshold": config.saturation_fraction_threshold,
            "slope_start_position": source.positions[slope_start_index].name,
            "slope_end_position": source.positions[slope_end_index].name,
            "step_02_values": "raw measured x profiles on physical mm axis; dashed lines span unmeasured gaps",
            "step_04_values": "corrected measured x profiles on physical mm axis with P01-P06 slope table; raw_comparison contains the same raw analysis",
            "step_05_values": "corrected measured x profiles per timepoint on physical mm axis; raw_comparison contains the same raw analysis",
            "step_06_values": "linear slopes fitted to corrected measured pixels from P01 through P06; raw_comparison contains raw slopes",
        }

        # Pass 1: read the ND2 once, save raw contact-sheet images, make raw
        # physical-axis plots, and select one fixed correction reference per channel.
        for t in source.timepoints:
            raw_segmented_profiles: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
            raw_gradient_fits_for_t: dict[str, dict[str, float | int | str | None]] = {}
            for channel in config.channels:
                tiles = [source.plane(t, p.index, channel) for p in source.positions]
                local_profiles: list[np.ndarray] = []
                for i, (position, tile) in enumerate(zip(source.positions, tiles)):
                    update_uint16_histogram(raw_histograms[channel.label], tile)
                    profile = x_profile(tile)
                    local_profiles.append(profile)
                    mean_intensity = float(np.mean(tile))
                    saturated_fraction = float(np.count_nonzero(tile == np.iinfo(np.uint16).max) / tile.size)
                    candidate = reference_candidates[channel.label][i]
                    candidate["profiles"].append(profile.copy())
                    candidate["tiles"].append(tile.copy())
                    candidate["mean_intensities"].append(mean_intensity)
                    candidate["saturated_fractions"].append(saturated_fraction)
                    candidate["timepoints"].append(t)

                raw_segmented_profiles[channel.label] = [
                    (axis, profile) for axis, profile in zip(x_axes_mm, local_profiles)
                ]
                raw_timecourse[channel.label][t] = raw_segmented_profiles[channel.label]
                raw_slope_record = _slope_record(
                    timepoint=t,
                    channel_label=channel.label,
                    x_axes_mm=x_axes_mm,
                    profiles=local_profiles,
                    slope_start_index=slope_start_index,
                    slope_end_index=slope_end_index,
                    start_position_label=source.positions[slope_start_index].name,
                    end_position_label=source.positions[slope_end_index].name,
                    slope_start_position=config.slope_start_position,
                    slope_end_position=config.slope_end_position,
                )
                raw_slope_records.append(raw_slope_record)
                raw_gradient_fits_for_t[channel.label] = raw_slope_record
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

            save_profile_plot(
                raw_step5 / f"{base}_t{t:03d}_z{config.z_index:02d}_GFP-Cy5_raw_x-profile-mm.png",
                raw_segmented_profiles,
                config.channels,
                f"{source_path.name} - t={t}, Z={config.z_index} (raw comparison, physical x)",
                config.trendline_window_px,
                raw_gradient_fits_for_t,
            )

        fixed_corrections: dict[str, dict[str, object]] = {}
        reference_candidate_rows: list[dict[str, object]] = []
        for channel in config.channels:
            candidates = reference_candidates[channel.label]
            brightness_scale = max(
                float(np.median(candidate["mean_intensities"]))  # type: ignore[arg-type]
                for candidate in candidates
            )
            scored_candidates: list[dict[str, object]] = []
            for candidate in candidates:
                score = reference_candidate_score(
                    candidate["profiles"],  # type: ignore[arg-type]
                    candidate["mean_intensities"],  # type: ignore[arg-type]
                    candidate["saturated_fractions"],  # type: ignore[arg-type]
                    config.smoothing_window_px,
                    config.saturation_fraction_threshold,
                    brightness_scale,
                )
                scored = {**candidate, **score}
                scored_candidates.append(scored)
                reference_candidate_rows.append(
                    {
                        "channel": channel.label,
                        "position_label": candidate["position_label"],
                        "position_list_index": candidate["position_list_index"],
                        "position_nd2_index": candidate["position_nd2_index"],
                        "score": score["score"],
                        "median_brightness": score["median_brightness"],
                        "brightness_fraction": score["brightness_fraction"],
                        "shape_variability": score["shape_variability"],
                        "max_saturated_fraction": score["max_saturated_fraction"],
                        "unsaturated": score["unsaturated"],
                        "best_profile_index": score["best_profile_index"],
                    }
                )
            selected = max(scored_candidates, key=lambda item: float(item["score"]))
            best_profile_index = int(selected["best_profile_index"])
            selected_profile = selected["profiles"][best_profile_index]  # type: ignore[index]
            selected_tile = selected["tiles"][best_profile_index]  # type: ignore[index]
            selected_timepoint = selected["timepoints"][best_profile_index]  # type: ignore[index]
            selected_mean = selected["mean_intensities"][best_profile_index]  # type: ignore[index]
            selected_saturated = selected["saturated_fractions"][best_profile_index]  # type: ignore[index]
            fitted_laser_profile, plateau = fitted_illumination_profile(
                selected_profile,  # type: ignore[arg-type]
                config.smoothing_window_px,
                config.correction_fit_degree,
            )
            smoothed_laser_profile, smoothed_plateau = smoothed_illumination_profile(
                selected_profile,  # type: ignore[arg-type]
                config.smoothing_window_px,
            )
            flatfield_2d_map, flatfield_2d_plateau = smoothed_illumination_image(
                selected_tile,  # type: ignore[arg-type]
                config.smoothing_window_px,
                config.smoothing_window_px,
            )
            fixed_corrections[channel.label] = {
                "mean_intensity": float(selected_mean),
                "timepoint": int(selected_timepoint),
                "position_list_index": int(selected["position_list_index"]),
                "position_label": selected["position_label"],
                "position_nd2_index": selected["position_nd2_index"],
                "saturated_fraction": float(selected_saturated),
                "used_saturated_fallback": not bool(selected["unsaturated"]),
                "reference_selection_score": float(selected["score"]),
                "reference_shape_variability": float(selected["shape_variability"]),
                "reference_brightness_fraction": float(selected["brightness_fraction"]),
                "reference_max_saturated_fraction": float(selected["max_saturated_fraction"]),
                "plateau": plateau,
                "fitted_laser_profile": fitted_laser_profile,
                "curve": plateau / fitted_laser_profile,
                "smoothed_profile_plateau": smoothed_plateau,
                "smoothed_fitted_laser_profile": smoothed_laser_profile,
                "smoothed_profile_curve": smoothed_plateau / smoothed_laser_profile,
                "flatfield_2d_plateau": flatfield_2d_plateau,
                "flatfield_2d_map": flatfield_2d_map,
            }
            save_reference_selection_plot(
                diagnostics_dir / f"{base}_z{config.z_index:02d}_{channel.label}_selected_reference_shapes.png",
                list(zip(selected["timepoints"], selected["profiles"])),  # type: ignore[arg-type]
                channel,
                str(selected["position_label"]),
                (
                    f"{source_path.name} - {channel.label} selected reference "
                    f"{selected['position_label']} normalized profile stability"
                ),
                config.smoothing_window_px,
            )
        write_rows_csv(diagnostics_dir / "reference_candidate_scores.csv", reference_candidate_rows)
        write_json(diagnostics_dir / "reference_candidate_scores.json", reference_candidate_rows)
        metadata["fixed_correction_references"] = {
            label: {
                k: v
                for k, v in details.items()
                if k not in {
                    "fitted_laser_profile",
                    "curve",
                    "smoothed_fitted_laser_profile",
                    "smoothed_profile_curve",
                    "flatfield_2d_map",
                }
            }
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
                smoothed_curve = fixed_corrections[channel.label]["smoothed_profile_curve"]
                smoothed_laser_profile = fixed_corrections[channel.label]["smoothed_fitted_laser_profile"]
                smoothed_plateau = float(fixed_corrections[channel.label]["smoothed_profile_plateau"])
                flatfield_2d_map = fixed_corrections[channel.label]["flatfield_2d_map"]
                flatfield_2d_plateau = float(fixed_corrections[channel.label]["flatfield_2d_plateau"])
                corrected = [correct_tile(tile, curve) for tile in tiles]  # type: ignore[arg-type]
                corrected_profiles = [x_profile(tile) for tile in corrected]
                smoothed_corrected = [correct_tile(tile, smoothed_curve) for tile in tiles]  # type: ignore[arg-type]
                smoothed_corrected_profiles = [x_profile(tile) for tile in smoothed_corrected]
                flatfield_2d_corrected = [
                    correct_tile_2d(tile, flatfield_2d_map, flatfield_2d_plateau)  # type: ignore[arg-type]
                    for tile in tiles
                ]
                flatfield_2d_corrected_profiles = [x_profile(tile) for tile in flatfield_2d_corrected]
                corrected_segments = [
                    (axis, profile) for axis, profile in zip(x_axes_mm, corrected_profiles)
                ]
                corrected_at_t[channel.label] = corrected_segments
                timecourse[channel.label][t] = corrected_segments

                slope_record = _slope_record(
                    timepoint=t,
                    channel_label=channel.label,
                    x_axes_mm=x_axes_mm,
                    profiles=corrected_profiles,
                    slope_start_index=slope_start_index,
                    slope_end_index=slope_end_index,
                    start_position_label=source.positions[slope_start_index].name,
                    end_position_label=source.positions[slope_end_index].name,
                    slope_start_position=config.slope_start_position,
                    slope_end_position=config.slope_end_position,
                )
                slope_records.append(slope_record)
                gradient_fits_for_t[channel.label] = slope_record
                raw_record = next(
                    record
                    for record in raw_slope_records
                    if record["timepoint"] == t and record["channel"] == channel.label
                )
                _, raw_fit_y = flatten_segments(
                    x_axes_mm,
                    local_profiles,
                    slope_start_index,
                    slope_end_index,
                )
                _, corrected_fit_y = flatten_segments(
                    x_axes_mm,
                    corrected_profiles,
                    slope_start_index,
                    slope_end_index,
                )
                _, smoothed_fit_y = flatten_segments(
                    x_axes_mm,
                    smoothed_corrected_profiles,
                    slope_start_index,
                    slope_end_index,
                )
                _, flatfield_2d_fit_y = flatten_segments(
                    x_axes_mm,
                    flatfield_2d_corrected_profiles,
                    slope_start_index,
                    slope_end_index,
                )
                smoothed_slope_record = _slope_record(
                    timepoint=t,
                    channel_label=channel.label,
                    x_axes_mm=x_axes_mm,
                    profiles=smoothed_corrected_profiles,
                    slope_start_index=slope_start_index,
                    slope_end_index=slope_end_index,
                    start_position_label=source.positions[slope_start_index].name,
                    end_position_label=source.positions[slope_end_index].name,
                    slope_start_position=config.slope_start_position,
                    slope_end_position=config.slope_end_position,
                )
                flatfield_2d_slope_record = _slope_record(
                    timepoint=t,
                    channel_label=channel.label,
                    x_axes_mm=x_axes_mm,
                    profiles=flatfield_2d_corrected_profiles,
                    slope_start_index=slope_start_index,
                    slope_end_index=slope_end_index,
                    start_position_label=source.positions[slope_start_index].name,
                    end_position_label=source.positions[slope_end_index].name,
                    slope_start_position=config.slope_start_position,
                    slope_end_position=config.slope_end_position,
                )
                fixed_ref = fixed_corrections[channel.label]
                diagnostic_records.append(
                    {
                        "timepoint": t,
                        "channel": channel.label,
                        "raw_slope_au_per_mm": raw_record["slope_au_per_mm"],
                        "corrected_slope_au_per_mm": slope_record["slope_au_per_mm"],
                        "raw_curvature": profile_curvature(raw_fit_y, config.trendline_window_px),
                        "corrected_curvature": profile_curvature(corrected_fit_y, config.trendline_window_px),
                        "reference_position_label": fixed_ref["position_label"],
                        "reference_timepoint": fixed_ref["timepoint"],
                        "reference_selection_score": fixed_ref["reference_selection_score"],
                        "reference_shape_variability": fixed_ref["reference_shape_variability"],
                    }
                )
                smoothed_diagnostic_records.append(
                    {
                        "timepoint": t,
                        "channel": channel.label,
                        "raw_slope_au_per_mm": raw_record["slope_au_per_mm"],
                        "quadratic_slope_au_per_mm": slope_record["slope_au_per_mm"],
                        "smoothed_profile_slope_au_per_mm": smoothed_slope_record["slope_au_per_mm"],
                        "raw_curvature": profile_curvature(raw_fit_y, config.trendline_window_px),
                        "quadratic_curvature": profile_curvature(corrected_fit_y, config.trendline_window_px),
                        "smoothed_profile_curvature": profile_curvature(smoothed_fit_y, config.trendline_window_px),
                        "reference_position_label": fixed_ref["position_label"],
                        "reference_timepoint": fixed_ref["timepoint"],
                        "quadratic_plateau": plateau,
                        "smoothed_profile_plateau": smoothed_plateau,
                        "reference_selection_score": fixed_ref["reference_selection_score"],
                    }
                )
                flatfield_2d_diagnostic_records.append(
                    {
                        "timepoint": t,
                        "channel": channel.label,
                        "raw_slope_au_per_mm": raw_record["slope_au_per_mm"],
                        "quadratic_slope_au_per_mm": slope_record["slope_au_per_mm"],
                        "smoothed_profile_slope_au_per_mm": smoothed_slope_record["slope_au_per_mm"],
                        "flatfield_2d_slope_au_per_mm": flatfield_2d_slope_record["slope_au_per_mm"],
                        "raw_curvature": profile_curvature(raw_fit_y, config.trendline_window_px),
                        "quadratic_curvature": profile_curvature(corrected_fit_y, config.trendline_window_px),
                        "smoothed_profile_curvature": profile_curvature(smoothed_fit_y, config.trendline_window_px),
                        "flatfield_2d_curvature": profile_curvature(flatfield_2d_fit_y, config.trendline_window_px),
                        "reference_position_label": fixed_ref["position_label"],
                        "reference_timepoint": fixed_ref["timepoint"],
                        "flatfield_2d_plateau": flatfield_2d_plateau,
                        "reference_selection_score": fixed_ref["reference_selection_score"],
                    }
                )

                converted_tiles: list[np.ndarray] = []
                clipped_pixels = 0
                for tile in corrected:
                    converted, clipped = to_uint16(tile)
                    converted_tiles.append(converted)
                    clipped_pixels += clipped
                    update_uint16_histogram(corrected_histograms[channel.label], converted)

                channel_dir = step3 / f"t{t:03d}" / channel.label
                channel_dir.mkdir(parents=True, exist_ok=True)
                comparison_channel_dir = smoothed_comparison_dir / f"t{t:03d}" / channel.label
                comparison_channel_dir.mkdir(parents=True, exist_ok=True)
                flatfield_2d_channel_dir = flatfield_2d_comparison_dir / f"t{t:03d}" / channel.label
                flatfield_2d_channel_dir.mkdir(parents=True, exist_ok=True)
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
                save_correction_method_comparison_plot(
                    comparison_channel_dir
                    / f"{base}_t{t:03d}_z{config.z_index:02d}_{channel.label}_quadratic-vs-smoothed-profile.png",
                    local_profiles,
                    corrected_profiles,
                    smoothed_corrected_profiles,
                    fitted_laser_profile,  # type: ignore[arg-type]
                    smoothed_laser_profile,  # type: ignore[arg-type]
                    ref_index,
                    [p.name for p in source.positions],
                    channel,
                    (
                        f"{source_path.name} - t={t}, {channel.label}; fixed ref="
                        f"t{int(fixed_ref['timepoint']):03d} {fixed_ref['position_label']}"
                    ),
                )
                save_flatfield_2d_comparison_plot(
                    flatfield_2d_channel_dir
                    / f"{base}_t{t:03d}_z{config.z_index:02d}_{channel.label}_1d-vs-2d-flatfield.png",
                    local_profiles,
                    corrected_profiles,
                    smoothed_corrected_profiles,
                    flatfield_2d_corrected_profiles,
                    fitted_laser_profile,  # type: ignore[arg-type]
                    smoothed_laser_profile,  # type: ignore[arg-type]
                    flatfield_2d_map,  # type: ignore[arg-type]
                    ref_index,
                    [p.name for p in source.positions],
                    channel,
                    (
                        f"{source_path.name} - t={t}, {channel.label}; fixed ref="
                        f"t{int(fixed_ref['timepoint']):03d} {fixed_ref['position_label']}"
                    ),
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
                        "correction_method": "fixed_channel_reference_stable_bright_quadratic_trendline",
                        "reference_selection_score": fixed_ref["reference_selection_score"],
                        "reference_shape_variability": fixed_ref["reference_shape_variability"],
                        "reference_brightness_fraction": fixed_ref["reference_brightness_fraction"],
                        "reference_max_saturated_fraction": fixed_ref["reference_max_saturated_fraction"],
                        "comparison_correction_method": "fixed_channel_reference_stable_bright_smoothed_profile",
                        "smoothed_profile_plateau": smoothed_plateau,
                        "smoothed_profile_note": (
                            "Diagnostic only: uses smooth_profile(reference_profile) directly "
                            "as the fitted laser profile, with the same median-preserving correction formula."
                        ),
                        "flatfield_2d_comparison_method": "fixed_channel_reference_stable_bright_smoothed_2d_flatfield",
                        "flatfield_2d_plateau": flatfield_2d_plateau,
                        "flatfield_2d_note": (
                            "Diagnostic only: uses a smoothed 2-D selected reference tile "
                            "as the fitted illumination map, with the same median-preserving correction formula."
                        ),
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
                raw_step4
                / f"{base}_z{config.z_index:02d}_{channel.label}_all-timepoints_raw_x-profiles-mm.png",
                raw_timecourse[channel.label],
                channel,
                f"{source_path.name} - {channel.label}, all raw timepoints, Z={config.z_index}",
                config.trendline_window_px,
                raw_slope_records,
            )
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

        for channel in config.channels:
            save_slope_timecourse_plot(
                raw_step6 / f"{base}_z{config.z_index:02d}_{channel.label}_P01-P06_raw_slope_over_time.png",
                raw_slope_records,
                channel,
                f"{source_path.name} - {channel.label}, raw P01-P06 slope",
            )
        write_csv(raw_step6 / f"{base}_z{config.z_index:02d}_P01-P06_raw_slopes.csv", raw_slope_records)
        write_json(raw_step6 / f"{base}_z{config.z_index:02d}_P01-P06_raw_slopes.json", raw_slope_records)
        write_rows_csv(step6 / "raw_vs_corrected_diagnostics.csv", diagnostic_records)
        write_json(step6 / "raw_vs_corrected_diagnostics.json", diagnostic_records)
        write_rows_csv(
            smoothed_comparison_dir / "quadratic_vs_smoothed_profile_diagnostics.csv",
            smoothed_diagnostic_records,
        )
        write_json(
            smoothed_comparison_dir / "quadratic_vs_smoothed_profile_diagnostics.json",
            smoothed_diagnostic_records,
        )
        write_rows_csv(
            flatfield_2d_comparison_dir / "quadratic_vs_smoothed_profile_vs_2d_flatfield_diagnostics.csv",
            flatfield_2d_diagnostic_records,
        )
        write_json(
            flatfield_2d_comparison_dir / "quadratic_vs_smoothed_profile_vs_2d_flatfield_diagnostics.json",
            flatfield_2d_diagnostic_records,
        )
        write_csv(step6 / f"{base}_z{config.z_index:02d}_P01-P06_slopes.csv", slope_records)
        write_json(step6 / f"{base}_z{config.z_index:02d}_P01-P06_slopes.json", slope_records)

    write_json(run_dir / "run_metadata.json", metadata)
    return run_dir
