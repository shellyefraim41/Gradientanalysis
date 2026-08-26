"""Feathered, normalized gradient analysis with tanh and all-Z fitting."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import numpy as np
import tifffile

from .config import AnalysisConfig
from .nd2_source import ND2Source
from .outputs import (
    save_color_preview, save_continuous_profile_plot, save_continuous_timecourse_plot,
    save_fit_heatmap, save_fit_metric_by_z_plot, save_fit_metric_timecourse_plot,
    save_max_difference_timecourse_plot, save_overlap_correction_plot,
    save_overlap_residual_plot, save_preview, save_rgb, save_tanh_fit_plot,
    save_z_correction_coefficients_plot,
    save_tiff, write_json, write_rows_csv,
)
from .processing import (
    correct_tile, empty_uint16_histogram, feather_profiles, feather_tiles_2d,
    fit_overlap_log_quadratic, fit_tanh_profile, histogram_percentile_range,
    image_percentile_range, merge_rgb, overlap_log_ratio_samples,
    physical_x_axes_mm, profile_normalization_constants, quadratic_illumination_profile,
    smooth_z_coefficients, split_equal_width,
    stitch, subtract_background_floor, to_uint16, update_uint16_histogram, x_profile,
)


def _safe_stem(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text).strip("_")


def _timepoint_folder(parent: Path, timepoint: int) -> Path:
    return parent / f"t{timepoint:03d}"


def _position_max_difference_record(*, timepoint: int, channel_label: str, p02_tile: np.ndarray, p05_tile: np.ndarray) -> dict[str, object]:
    p02_max, p05_max = float(np.max(p02_tile)), float(np.max(p05_tile))
    formula, difference = (
        ("max(P05)-max(P02)", p05_max - p02_max)
        if channel_label == "Cy5"
        else ("max(P02)-max(P05)", p02_max - p05_max)
    )
    return {"timepoint": timepoint, "channel": channel_label, "formula": formula,
            "p02_max": p02_max, "p05_max": p05_max, "max_intensity_difference": difference}


def _fit(x: np.ndarray, y: np.ndarray, channel: str, timepoint: int, config: AnalysisConfig, z: int | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "timepoint": timepoint, "channel": channel,
        **fit_tanh_profile(x, y, max_points=config.tanh_max_points, pixel_size_mm=float(x[1] - x[0])),
    }
    if z is not None:
        record.update(z_index_zero_based=z, z_index_one_based=z + 1)
    return record


def _save_previews(timepoints, channels, raw_paths, corrected_paths, raw_hist, corrected_hist, config, step1, base):
    raw_ranges = {c.label: histogram_percentile_range(raw_hist[c.label], config.preview_low_percentile, config.preview_high_percentile) for c in channels}
    corrected_ranges = {c.label: histogram_percentile_range(corrected_hist[c.label], config.preview_low_percentile, config.preview_high_percentile) for c in channels}
    local: dict[str, dict[str, list[int]]] = {"raw": {}, "corrected": {}}
    for t in timepoints:
        raw_images = []
        for c in channels:
            path = raw_paths[c.label][t]; image = tifffile.imread(path); raw_images.append(image)
            limits = image_percentile_range(image, config.preview_low_percentile, config.preview_high_percentile)
            local["raw"][f"t{t:03d}_{c.label}"] = list(limits)
            save_preview(path.with_name(f"{path.stem}_grayscale_preview.png"), image, *limits)
            save_color_preview(path.with_name(f"{path.stem}_preview.png"), image, *limits, c.rgb)
            save_color_preview(path.with_name(f"{path.stem}_global_preview.png"), image, *raw_ranges[c.label], c.rgb)
        save_rgb(_timepoint_folder(step1, t) / f"{base}_t{t:03d}_z{config.z_index:02d}_GFP-Cy5_merge.png",
                 merge_rgb(raw_images, [c.rgb for c in channels], [raw_ranges[c.label] for c in channels]))
        for c in channels:
            path = corrected_paths[c.label][t]; image = tifffile.imread(path)
            limits = image_percentile_range(image, config.preview_low_percentile, config.preview_high_percentile)
            local["corrected"][f"t{t:03d}_{c.label}"] = list(limits)
            save_preview(path.with_name(f"{path.stem}_grayscale_preview.png"), image, *limits)
            save_color_preview(path.with_name(f"{path.stem}_preview.png"), image, *limits, c.rgb)
            save_color_preview(path.with_name(f"{path.stem}_global_preview.png"), image, *corrected_ranges[c.label], c.rgb)
    return {"percentiles": [config.preview_low_percentile, config.preview_high_percentile],
            "raw_ranges_by_channel": {k: list(v) for k, v in raw_ranges.items()},
            "corrected_ranges_by_channel": {k: list(v) for k, v in corrected_ranges.items()},
            "local_ranges_by_image": local}


def _run_all_z(source, config, x_axes, step8, base):
    selected = tuple(config.all_z_timepoints)
    invalid = [t for t in selected if t not in source.timepoints]
    if invalid:
        raise ValueError(f"All-Z timepoints are not in the selected analysis: {invalid}")
    z_count = int(source.sizes.get("Z", 1))
    profiles = {c.label: {} for c in config.channels}
    stitched_root = step8 / "stitched_images"
    stitched_root.mkdir(parents=True, exist_ok=True)
    correction_root = step8 / "per_z_correction_diagnostics"
    correction_root.mkdir(parents=True, exist_ok=True)
    display_rows: list[dict[str, object]] = []
    correction_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    samples = {
        c.label: {z: [] for z in range(z_count)} for c in config.channels
    }
    sampled_tiles = {c.label: [] for c in config.channels}
    tile_width = int(source.sizes["X"])
    pixel_size_um = float(source.pixel_size_um)
    saturation_signal = float(np.iinfo(np.uint16).max) - config.microscope_background
    stride = int(config.all_z_display_sample_stride)
    pair_geometry = []
    for left, right in zip(source.positions, source.positions[1:]):
        pair_geometry.append((
            left,
            right,
            int(round(abs(right.x_um - left.x_um) / pixel_size_um)),
            int(round((right.y_um - left.y_um) / pixel_size_um)),
        ))

    # First pass: estimate one raw overlap model per Z and retain sparse pixels
    # for a shared display range. No complete corrected stack is held in RAM.
    for t in selected:
        for z in range(z_count):
            for c in config.channels:
                tiles = [
                    subtract_background_floor(
                        source.plane(t, p.index, c, z), config.microscope_background
                    )
                    for p in source.positions
                ]
                for tile in tiles:
                    sampled_tiles[c.label].append((z, to_uint16(tile[::stride, ::stride])[0]))
                for index, (left, right, x_offset, y_shift) in enumerate(pair_geometry):
                    xl, xr, ratios, details = overlap_log_ratio_samples(
                        tiles[index], tiles[index + 1], x_offset, y_shift,
                        config.overlap_min_signal, saturation_signal,
                    )
                    row = {
                        "timepoint": t, "z_index_zero_based": z,
                        "z_index_one_based": z + 1, "channel": c.label,
                        "position_pair": f"{left.name}-{right.name}",
                        "raw_per_z_corrected_median_abs_log_residual": float("nan"),
                        "smoothed_corrected_median_abs_log_residual": float("nan"),
                        **details,
                    }
                    overlap_rows.append(row)
                    if ratios.size:
                        samples[c.label][z].append((xl, xr, ratios, row))

    corrections = {c.label: {} for c in config.channels}
    models = {c.label: {} for c in config.channels}
    for c in config.channels:
        raw_linear = np.full(z_count, np.nan, dtype=np.float64)
        raw_quadratic = np.full(z_count, np.nan, dtype=np.float64)
        counts = np.zeros(z_count, dtype=np.float64)
        residuals = np.full(z_count, np.nan, dtype=np.float64)
        for z in range(z_count):
            z_samples = samples[c.label][z]
            if not z_samples:
                continue
            xl = np.concatenate([item[0] for item in z_samples])
            xr = np.concatenate([item[1] for item in z_samples])
            ratios = np.concatenate([item[2] for item in z_samples])
            try:
                _, model = fit_overlap_log_quadratic(tile_width, xl, xr, ratios)
            except (ValueError, np.linalg.LinAlgError):
                continue
            raw_linear[z] = float(model["linear_coefficient"])
            raw_quadratic[z] = float(model["quadratic_coefficient"])
            counts[z] = float(model["sample_count"])
            residuals[z] = float(model["median_abs_log_residual"])
            raw_profile = quadratic_illumination_profile(
                tile_width, raw_linear[z], raw_quadratic[z]
            )
            for sx, sy, sr, row in z_samples:
                predicted = np.log(
                    raw_profile[sx.astype(int)] / raw_profile[sy.astype(int)]
                )
                row["raw_per_z_corrected_median_abs_log_residual"] = float(
                    np.median(np.abs(sr - predicted))
                )
        valid = np.isfinite(raw_linear) & np.isfinite(raw_quadratic) & (counts > 0)
        if np.count_nonzero(valid) < 2:
            raise ValueError(f"Fewer than two valid all-Z overlap models for {c.label}.")
        count_scale = max(float(np.median(counts[valid])), 1.0)
        residual_scale = max(float(np.median(residuals[valid])), 1e-6)
        quality = np.zeros(z_count, dtype=np.float64)
        quality[valid] = (counts[valid] / count_scale) / np.maximum(
            residuals[valid] / residual_scale, 0.25
        ) ** 2
        smooth_linear, normalized_weights = smooth_z_coefficients(
            raw_linear, quality, config.all_z_coefficient_smoothing_penalty
        )
        smooth_quadratic, _ = smooth_z_coefficients(
            raw_quadratic, quality, config.all_z_coefficient_smoothing_penalty
        )
        for z in range(z_count):
            profile = quadratic_illumination_profile(
                tile_width, float(smooth_linear[z]), float(smooth_quadratic[z])
            )
            corrections[c.label][z] = 1.0 / profile
            model = {
                "model": "exp(linear*x + quadratic*x^2)",
                "z_index_zero_based": z,
                "z_index_one_based": z + 1,
                "raw_linear_coefficient": float(raw_linear[z]),
                "raw_quadratic_coefficient": float(raw_quadratic[z]),
                "smoothed_linear_coefficient": float(smooth_linear[z]),
                "smoothed_quadratic_coefficient": float(smooth_quadratic[z]),
                "quality_weight": float(normalized_weights[z]),
                "sample_count": int(counts[z]),
                "raw_median_abs_log_residual": float(residuals[z]),
                "profile_median": float(np.median(profile)),
                "profile_min": float(np.min(profile)),
                "profile_max": float(np.max(profile)),
            }
            models[c.label][z] = model
            correction_rows.append({"channel": c.label, **model})
            channel_dir = correction_root / c.label
            save_overlap_correction_plot(
                channel_dir / f"{base}_z{z + 1:02d}_{c.label}_smoothed_overlap_profile.png",
                profile, c,
                f"{source.path.name} - {c.label}, Z={z + 1} smoothed overlap illumination",
            )
            for sx, sy, sr, row in samples[c.label][z]:
                predicted = np.log(profile[sx.astype(int)] / profile[sy.astype(int)])
                row["smoothed_corrected_median_abs_log_residual"] = float(
                    np.median(np.abs(sr - predicted))
                )
        save_z_correction_coefficients_plot(
            correction_root / f"{base}_{c.label}_raw-vs-smoothed_coefficients.png",
            correction_rows, c,
            f"{source.path.name} - {c.label} overlap coefficients across Z",
        )
    write_rows_csv(correction_root / "per_z_quadratic_coefficients.csv", correction_rows)
    write_json(correction_root / "per_z_quadratic_models.json", models)
    write_rows_csv(correction_root / "per_z_overlap_residuals.csv", overlap_rows)
    write_json(correction_root / "per_z_overlap_residuals.json", overlap_rows)

    # A stratified sparse sample establishes one comparable display range for
    # every Z and timepoint without storing a second complete image collection.
    display_histograms = {c.label: empty_uint16_histogram() for c in config.channels}
    sampled_x = np.arange(0, tile_width, stride)
    for c in config.channels:
        for z, tile in sampled_tiles[c.label]:
            corrected_sample = tile.astype(np.float32) * corrections[c.label][z][sampled_x][np.newaxis, :]
            update_uint16_histogram(display_histograms[c.label], to_uint16(corrected_sample)[0])
    display_ranges = {
        c.label: histogram_percentile_range(
            display_histograms[c.label], config.preview_low_percentile,
            config.preview_high_percentile,
        )
        for c in config.channels
    }
    del sampled_tiles

    pixel_mm = pixel_size_um / 1000.0
    x_origin = min(float(axis[0]) for axis in x_axes)
    x_starts_px = [(float(axis[0]) - x_origin) / pixel_mm for axis in x_axes]
    first_y = float(source.positions[0].y_um)
    y_starts_px = [(float(p.y_um) - first_y) / pixel_size_um for p in source.positions]
    mosaic_details = None
    clipped_total = 0
    for t in selected:
        timepoint_dir = _timepoint_folder(stitched_root, t)
        timepoint_dir.mkdir(parents=True, exist_ok=True)
        for z in range(z_count):
            display_images: list[np.ndarray] = []
            for c in config.channels:
                corrected_tiles = []
                for p in source.positions:
                    signal = subtract_background_floor(
                        source.plane(t, p.index, c, z), config.microscope_background
                    )
                    corrected_tiles.append(correct_tile(signal, corrections[c.label][z]))
                mosaic, mosaic_details = feather_tiles_2d(
                    corrected_tiles, x_starts_px, y_starts_px
                )
                x = np.arange(mosaic.shape[1], dtype=np.float64) * pixel_mm
                profiles[c.label][(t, z)] = (x, x_profile(mosaic))
                converted, clipped = to_uint16(mosaic)
                clipped_total += clipped
                stem = f"{base}_t{t:03d}_z{z + 1:02d}_{c.label}_corrected_feathered_mosaic"
                tiff_path = timepoint_dir / f"{stem}.tif"
                if config.all_z_save_mosaic_tiffs:
                    save_tiff(tiff_path, converted)
                low, high = display_ranges[c.label]
                save_color_preview(
                    timepoint_dir / f"{stem}_preview.png", converted, low, high, c.rgb
                )
                display_images.append(converted)
                display_rows.append({
                    "timepoint": t, "z_index_zero_based": z,
                    "z_index_one_based": z + 1, "channel": c.label,
                    "display_low": low, "display_high": high,
                    "clipped_pixel_count": clipped,
                    "scaling": "shared channel range across selected T and Z",
                    "mosaic_tiff": str(tiff_path) if config.all_z_save_mosaic_tiffs else "not saved",
                })
            save_rgb(
                timepoint_dir / f"{base}_t{t:03d}_z{z + 1:02d}_GFP-Cy5_corrected_feathered_mosaic.png",
                merge_rgb(
                    display_images, [c.rgb for c in config.channels],
                    [display_ranges[c.label] for c in config.channels],
                ),
            )
    write_rows_csv(stitched_root / "stitched_image_display_ranges.csv", display_rows)
    write_json(stitched_root / "stitched_image_display_ranges.json", display_rows)
    constants = profile_normalization_constants({c.label: [v[1] for v in profiles[c.label].values()] for c in config.channels})
    normalized = {c.label: {k: (v[0], v[1] / constants[c.label]) for k, v in profiles[c.label].items()} for c in config.channels}
    records: list[dict[str, object]] = []
    fit_dir = step8 / "profile_fits"; fit_dir.mkdir(parents=True, exist_ok=True)
    for t in selected:
        for z in range(z_count):
            plot_profiles, fits = {}, {}
            for c in config.channels:
                x, y = normalized[c.label][(t, z)]
                record = _fit(x, y, c.label, t, config, z)
                records.append(record); fits[c.label] = record; plot_profiles[c.label] = (x, y)
            save_tanh_fit_plot(fit_dir / f"{base}_t{t:03d}_z{z + 1:02d}_GFP-Cy5_tanh-fit.png",
                               plot_profiles, fits, config.channels,
                               f"{source.path.name} - t={t}, Z={z + 1} (ND2 index {z})")
    write_rows_csv(step8 / "all_z_tanh_fits.csv", records)
    write_json(step8 / "all_z_tanh_fits.json", records)
    write_json(step8 / "all_z_normalization_constants.json", constants)
    metrics = (("signed_slope_per_mm", "Signed slope (normalized/mm)", "signed_slope"),
               ("absolute_slope_per_mm", "Absolute slope (normalized/mm)", "absolute_slope"),
               ("midpoint_mm", "Midpoint (mm)", "midpoint"),
               ("width_mm", "Width (mm)", "width"))
    for t in selected:
        for metric, ylabel, suffix in metrics:
            save_fit_metric_by_z_plot(step8 / f"{base}_t{t:03d}_{suffix}_by_z.png", records,
                                      config.channels, t, metric, ylabel,
                                      f"{source.path.name} - t={t}, {ylabel} by Z")
    for c in config.channels:
        for metric, suffix in (("signed_slope_per_mm", "signed_slope"), ("absolute_slope_per_mm", "absolute_slope")):
            save_fit_heatmap(step8 / f"{base}_{c.label}_{suffix}_T-by-Z_heatmap.png", records,
                             c, selected, z_count, metric,
                             f"{source.path.name} - {c.label} {suffix.replace('_', ' ')}")
    return {"selected_timepoints": list(selected), "z_count": z_count, "record_count": len(records),
            "normalization_constants": constants,
            "correction_source": "one overlap model per Z, coefficients quality-weighted and smoothed across Z",
            "correction_timepoints": list(selected),
            "coefficient_smoothing_penalty": config.all_z_coefficient_smoothing_penalty,
            "per_z_models": models,
            "mosaic_geometry": mosaic_details,
            "stitched_images": {
                "count": len(selected) * z_count * 3,
                "channels": [c.label for c in config.channels],
                "merge": "GFP-Cy5",
                "method": "stage-aligned 2-D cosine-feathered mosaic",
                "scaling": "one sampled global percentile range per channel across selected T and Z",
                "display_ranges": {key: list(value) for key, value in display_ranges.items()},
                "display_sample_stride": stride,
                "mosaic_tiffs_saved": config.all_z_save_mosaic_tiffs,
                "clipped_pixel_count": clipped_total,
            }}


def run_pipeline(nd2_path: str | Path, output_root: str | Path, config: AnalysisConfig) -> Path:
    if config.apply_illumination_correction and config.correction_method != "overlap_quadratic":
        raise ValueError("The revised pipeline requires correction_method='overlap_quadratic'.")
    source_path = Path(nd2_path); base = _safe_stem(source_path.stem)
    run_dir = Path(output_root) / f"run_{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    steps = {1: "step_01_stitched_images", 2: "step_02_feathered_profiles_before_correction",
             3: "step_03_illumination_corrected", 4: "step_04_normalized_timecourse_profiles",
             5: "step_05_normalized_profiles", 6: "step_06_tanh_gradient_fits",
             7: "step_07_max_intensity_differences", 8: "step_08_selected_timepoints_all_z"}
    dirs = {n: run_dir / name for n, name in steps.items()}
    for n in range(1, 8): dirs[n].mkdir(parents=True, exist_ok=False)
    if config.all_z_enabled: dirs[8].mkdir(parents=True, exist_ok=False)
    overlap_dir = dirs[3] / "overlap_correction_diagnostics"; overlap_dir.mkdir()
    raw_paths = {c.label: {} for c in config.channels}; corrected_paths = {c.label: {} for c in config.channels}
    raw_hist = {c.label: empty_uint16_histogram() for c in config.channels}; corrected_hist = {c.label: empty_uint16_histogram() for c in config.channels}
    raw_profiles = {c.label: {} for c in config.channels}; corrected_profiles = {c.label: {} for c in config.channels}
    samples = {c.label: [] for c in config.channels}; overlap_rows = []; max_records = []
    with ND2Source(source_path, config) as source:
        if source.pixel_size_um is None: raise ValueError("ND2 pixel size metadata are required.")
        tile_width = int(source.sizes["X"]); timepoints = source.timepoints
        x_axes = physical_x_axes_mm([p.x_um for p in source.positions], source.pixel_size_um, tile_width)
        feather_details = None
        for t in timepoints:
            plot_profiles = {}
            for c in config.channels:
                acquired = [source.plane(t, p.index, c) for p in source.positions]
                tiles = [to_uint16(subtract_background_floor(image, config.microscope_background))[0] for image in acquired]
                for tile in tiles: update_uint16_histogram(raw_hist[c.label], tile)
                x, y, feather_details = feather_profiles(x_axes, [x_profile(tile) for tile in tiles])
                raw_profiles[c.label][t] = (x, y); plot_profiles[c.label] = (x, y)
                if len(tiles) >= 5:
                    max_records.append(_position_max_difference_record(timepoint=t, channel_label=c.label, p02_tile=tiles[1], p05_tile=tiles[4]))
                for i in range(len(source.positions) - 1):
                    left, right = source.positions[i], source.positions[i + 1]
                    x_offset = int(round(abs(right.x_um - left.x_um) / source.pixel_size_um))
                    y_shift = int(round((right.y_um - left.y_um) / source.pixel_size_um))
                    xl, xr, ratios, details = overlap_log_ratio_samples(tiles[i], tiles[i + 1], x_offset, y_shift,
                                                                        config.overlap_min_signal,
                                                                        float(np.iinfo(np.uint16).max) - config.microscope_background)
                    row = {"timepoint": t, "channel": c.label, "position_pair": f"{left.name}-{right.name}",
                           "corrected_median_abs_log_residual": float("nan"), **details}
                    overlap_rows.append(row)
                    if ratios.size: samples[c.label].append((xl, xr, ratios, row))
                path = _timepoint_folder(dirs[1], t) / f"{base}_t{t:03d}_z{config.z_index:02d}_{c.label}_positions_left-to-right.tif"
                save_tiff(path, stitch(tiles)); raw_paths[c.label][t] = path
            save_continuous_profile_plot(dirs[2] / f"{base}_t{t:03d}_z{config.z_index:02d}_GFP-Cy5_feathered_before-correction.png",
                                         plot_profiles, config.channels,
                                         f"{source_path.name} - t={t}, Z={config.z_index} before correction", normalized=False)
        corrections, models = {}, {}
        for c in config.channels:
            if config.apply_illumination_correction:
                if not samples[c.label]: raise ValueError(f"No valid overlap samples for {c.label}.")
                xl = np.concatenate([s[0] for s in samples[c.label]]); xr = np.concatenate([s[1] for s in samples[c.label]])
                ratios = np.concatenate([s[2] for s in samples[c.label]])
                profile, model = fit_overlap_log_quadratic(tile_width, xl, xr, ratios)
                corrections[c.label] = 1.0 / profile; models[c.label] = model
                save_overlap_correction_plot(overlap_dir / f"{base}_z{config.z_index:02d}_{c.label}_overlap_quadratic_profile.png",
                                             profile, c, f"{source_path.name} - {c.label} overlap illumination")
                for sx, sy, sr, row in samples[c.label]:
                    predicted = np.log(profile[sx.astype(int)] / profile[sy.astype(int)])
                    row["corrected_median_abs_log_residual"] = float(np.median(np.abs(sr - predicted)))
            else:
                corrections[c.label] = np.ones(tile_width); models[c.label] = {"model": "none"}
        write_rows_csv(overlap_dir / "overlap_pair_diagnostics.csv", overlap_rows); write_json(overlap_dir / "overlap_pair_diagnostics.json", overlap_rows)
        write_json(overlap_dir / "overlap_quadratic_models.json", models)
        for c in config.channels:
            save_overlap_residual_plot(overlap_dir / f"{base}_z{config.z_index:02d}_{c.label}_overlap_before-after.png",
                                       overlap_rows, c, f"{source_path.name} - {c.label} overlap mismatch")
        for t in timepoints:
            for c in config.channels:
                raw_tiles = split_equal_width(tifffile.imread(raw_paths[c.label][t]), len(source.positions))
                corrected_float = [correct_tile(tile, corrections[c.label]) for tile in raw_tiles]
                x, y, feather_details = feather_profiles(x_axes, [x_profile(tile) for tile in corrected_float])
                corrected_profiles[c.label][t] = (x, y)
                converted, clipped = [], 0
                for tile in corrected_float:
                    image, count = to_uint16(tile); converted.append(image); clipped += count; update_uint16_histogram(corrected_hist[c.label], image)
                channel_dir = dirs[3] / f"t{t:03d}" / c.label; channel_dir.mkdir(parents=True)
                if config.save_corrected_tiles:
                    for p, image in zip(source.positions, converted):
                        save_tiff(channel_dir / f"{base}_t{t:03d}_z{config.z_index:02d}_{c.label}_{p.name}_corrected.tif", image)
                path = channel_dir / f"{base}_t{t:03d}_z{config.z_index:02d}_{c.label}_corrected_contact-sheet.tif"
                save_tiff(path, stitch(converted)); corrected_paths[c.label][t] = path
                write_json(channel_dir / "normalization_details.json", {"correction_method": "overlap_quadratic", "clipped_pixel_count": clipped, "profile_feathering": feather_details})
        constants = profile_normalization_constants({c.label: [v[1] for v in corrected_profiles[c.label].values()] for c in config.channels})
        normalized = {c.label: {t: (v[0], v[1] / constants[c.label]) for t, v in corrected_profiles[c.label].items()} for c in config.channels}
        write_json(dirs[5] / "profile_normalization_constants.json", constants)
        fit_records = []
        for t in timepoints:
            plot_profiles = {c.label: normalized[c.label][t] for c in config.channels}
            save_continuous_profile_plot(dirs[5] / f"{base}_t{t:03d}_z{config.z_index:02d}_GFP-Cy5_normalized.png", plot_profiles,
                                         config.channels, f"{source_path.name} - t={t}, Z={config.z_index} normalized", normalized=True)
            fits = {}
            for c in config.channels:
                x, y = plot_profiles[c.label]; record = _fit(x, y, c.label, t, config)
                fits[c.label] = record; fit_records.append(record)
            save_tanh_fit_plot(dirs[6] / f"{base}_t{t:03d}_z{config.z_index:02d}_GFP-Cy5_tanh-fit.png",
                               plot_profiles, fits, config.channels, f"{source_path.name} - t={t}, Z={config.z_index} tanh fits")
        write_rows_csv(dirs[6] / "tanh_fit_results.csv", fit_records); write_json(dirs[6] / "tanh_fit_results.json", fit_records)
        for c in config.channels:
            save_continuous_timecourse_plot(dirs[4] / f"{base}_z{config.z_index:02d}_{c.label}_normalized_timecourse.png",
                                            normalized[c.label], c, f"{source_path.name} - {c.label} normalized profiles", normalized=True)
        for metric, ylabel, suffix in (("signed_slope_per_mm", "Signed slope (normalized/mm)", "signed_slope"),
                                       ("absolute_slope_per_mm", "Absolute slope (normalized/mm)", "absolute_slope"),
                                       ("midpoint_mm", "Midpoint (mm)", "midpoint"), ("width_mm", "Width (mm)", "width")):
            save_fit_metric_timecourse_plot(dirs[6] / f"{base}_z{config.z_index:02d}_{suffix}_over_time.png", fit_records,
                                            config.channels, metric, ylabel, f"{source_path.name} - tanh {ylabel.lower()} over time")
        save_max_difference_timecourse_plot(dirs[7] / f"{base}_z{config.z_index:02d}_P02-P05_max_difference_over_time.png",
                                            max_records, config.channels, f"{source_path.name} - P02/P05 max differences")
        write_rows_csv(dirs[7] / f"{base}_z{config.z_index:02d}_P02-P05_max_differences.csv", max_records)
        write_json(dirs[7] / f"{base}_z{config.z_index:02d}_P02-P05_max_differences.json", max_records)
        all_z = _run_all_z(source, config, x_axes, dirs[8], base) if config.all_z_enabled else None
        display = _save_previews(timepoints, config.channels, raw_paths, corrected_paths, raw_hist, corrected_hist, config, dirs[1], base)
        overlap_widths = list(feather_details["overlap_widths_px"]) if feather_details else []
        measured_overlap = (
            float(np.mean(overlap_widths) / tile_width * 100.0) if overlap_widths else 0.0
        )
        metadata = {"source": str(source_path.resolve()), "dimensions": source.sizes, "z_index_requested": config.z_index,
                    "z_index_zero_based": config.zero_based_z, "pixel_size_um": source.pixel_size_um,
                    "positions_left_to_right": [{"label": p.name, "nd2_index": p.index, "x_um": p.x_um, "y_um": p.y_um} for p in source.positions],
                    "microscope_background": config.microscope_background, "correction_method": "overlap_quadratic",
                    "measured_overlap_percent": measured_overlap,
                    "measured_overlap_widths_px": overlap_widths,
                    "correction_policy": "Z15 uses one fixed profile per channel across time; Step 8 estimates one profile per Z and quality-smooths its coefficients across Z",
                    "overlap_models": models, "profile_feathering": feather_details,
                    "profile_normalization_constants": constants,
                    "profile_normalization_scope": "one maximum per channel across selected Z15 feathered profiles",
                    "tanh_model": "left+(right-left)/2*(1+tanh((x-midpoint)/width))",
                    "tanh_slope_definition": "(right-left)/(2*width), normalized/mm",
                    "analysis_controls": {"feather_overlaps": config.feather_overlaps,
                                          "normalize_profiles": config.normalize_profiles,
                                          "tanh_fit_enabled": config.tanh_fit_enabled,
                                          "tanh_max_points": config.tanh_max_points,
                                           "all_z_enabled": config.all_z_enabled,
                                           "all_z_timepoints": list(config.all_z_timepoints),
                                           "all_z_coefficient_smoothing_penalty": config.all_z_coefficient_smoothing_penalty,
                                           "all_z_display_sample_stride": config.all_z_display_sample_stride,
                                           "all_z_save_mosaic_tiffs": config.all_z_save_mosaic_tiffs},
                    "legacy_linear_slopes_removed": True, "all_z_analysis": all_z, "display_scaling": display}
    write_json(run_dir / "run_metadata.json", metadata)
    return run_dir
