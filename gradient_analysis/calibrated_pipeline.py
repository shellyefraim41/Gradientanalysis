"""Calibration-flat-field pilot for the September 2026 Exp106 acquisition."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nd2
import numpy as np
from PIL import Image
from scipy.integrate import trapezoid
import tifffile

from .config import AnalysisConfig, ChannelConfig
from .nd2_source import ND2Source
from .outputs import save_color_preview, save_rgb, write_json, write_rows_csv
from .processing import (
    block_median,
    build_artifact_safe_flatfield,
    colorize_scaled,
    despike_image,
    empirical_concentration,
    feather_profiles,
    feather_tiles_2d_union,
    fit_overlap_log_quadratic,
    fit_tanh_profile,
    image_percentile_range,
    merge_rgb,
    overlap_log_ratio_samples,
    persistent_hot_pixel_mask,
    physical_x_axes_mm,
    robust_y_profile,
    subtract_background_floor,
    tanh_profile,
    to_uint16,
)


def _safe_stem(text: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in text).strip("_")


def _channel_indices(file: nd2.ND2File, channels: tuple[ChannelConfig, ...]) -> dict[str, int]:
    names = [str(item.channel.name or "") for item in file.metadata.channels]
    lowered = [name.casefold() for name in names]
    result: dict[str, int] = {}
    for fallback, channel in enumerate(channels):
        if channel.index is not None:
            index = channel.index
        else:
            matches = [
                index
                for index, name in enumerate(lowered)
                if any(term.casefold() in name for term in channel.match)
            ]
            index = matches[0] if matches else fallback
        if not 0 <= index < len(names):
            raise ValueError(f"Cannot resolve {channel.label} in calibration channels {names}.")
        result[channel.label] = index
    return result


def _discover_standards(directory: Path) -> list[tuple[float, Path]]:
    standards = sorted(
        (int(path.stem) / 1000.0, path)
        for path in directory.glob("*.nd2")
        if path.stem.isdigit()
    )
    if len(standards) < 3:
        raise ValueError(f"At least three numeric calibration ND2 files are required in {directory}.")
    concentrations = [item[0] for item in standards]
    if 0.0 not in concentrations:
        raise ValueError("A 0 ng/mL calibration file is required for background estimation.")
    return standards


def _reference_path(
    standards: list[tuple[float, Path]], requested_concentration: float
) -> Path:
    concentration, path = min(standards, key=lambda item: abs(item[0] - requested_concentration))
    if not np.isclose(concentration, requested_concentration):
        raise ValueError(
            f"No {requested_concentration:g} µg/mL calibration file was found; closest is {concentration:g}."
        )
    return path


def _save_flatfield_plot(
    path: Path, flatfield: np.ndarray, channel: ChannelConfig, title: str
) -> None:
    correction = 1.0 / flatfield
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    first = axes[0].imshow(flatfield, cmap="viridis")
    axes[0].set(title="Relative illumination", xlabel="Camera X", ylabel="Camera Y")
    fig.colorbar(first, ax=axes[0], shrink=0.8)
    second = axes[1].imshow(correction, cmap="magma")
    axes[1].set(title="Correction factor", xlabel="Camera X", ylabel="Camera Y")
    fig.colorbar(second, ax=axes[1], shrink=0.8)
    axes[2].plot(np.median(flatfield, axis=0), color=channel.plot_color, label="median over Y")
    axes[2].plot(np.median(flatfield, axis=1), color="tab:blue", label="median over X")
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[2].set(title="Flat-field profiles", xlabel="Camera coordinate", ylabel="Relative illumination")
    axes[2].grid(alpha=0.2)
    axes[2].legend()
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _calibration_background_and_flatfields(
    standards: list[tuple[float, Path]],
    config: AnalysisConfig,
    output: Path,
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    blank_path = next(path for concentration, path in standards if concentration == 0.0)
    reference_path = _reference_path(standards, config.calibration_reference_concentration_ug_ml)
    backgrounds: dict[str, float] = {}
    hot_masks: dict[str, np.ndarray] = {}
    flatfields: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    with nd2.ND2File(blank_path) as blank, nd2.ND2File(reference_path) as reference:
        if dict(blank.sizes) != dict(reference.sizes):
            raise ValueError("Blank and high-standard calibration files have different dimensions.")
        if not {"Z", "C", "Y", "X"}.issubset(blank.sizes):
            raise ValueError("Calibration files must contain Z, C, Y, and X axes.")
        blank_indices = _channel_indices(blank, config.channels)
        reference_indices = _channel_indices(reference, config.channels)
        blank_data = blank.to_dask()
        reference_data = reference.to_dask()
        z_count = int(blank.sizes["Z"])
        image_shape = (int(blank.sizes["Y"]), int(blank.sizes["X"]))
        for channel in config.channels:
            minimum = np.full(image_shape, np.iinfo(np.uint16).max, dtype=np.uint16)
            samples: list[np.ndarray] = []
            for z in range(z_count):
                image = np.asarray(blank_data[z, blank_indices[channel.label]].compute())
                np.minimum(minimum, image, out=minimum)
                samples.append(image[::16, ::16].reshape(-1))
            background = float(np.median(np.concatenate(samples)))
            hot_mask, hot_details = persistent_hot_pixel_mask(minimum)
            backgrounds[channel.label] = background
            hot_masks[channel.label] = hot_mask

            coarse_planes: list[np.ndarray] = []
            despiked_counts: list[int] = []
            plane_normalizations: list[float] = []
            clipped_count = 0
            for z in range(z_count):
                acquired = np.asarray(reference_data[z, reference_indices[channel.label]].compute())
                clipped_count += int(np.count_nonzero(acquired == np.iinfo(acquired.dtype).max))
                repaired, spike_mask = despike_image(acquired, hot_mask)
                despiked_counts.append(int(np.count_nonzero(spike_mask)))
                signal = subtract_background_floor(repaired, background)
                coarse = block_median(signal, config.flatfield_block_size_px)
                normalization = float(np.nanmedian(coarse[coarse > config.overlap_min_signal]))
                if not np.isfinite(normalization) or normalization <= 0:
                    raise ValueError(f"Invalid {channel.label} flat-field normalization at Z={z + 1}.")
                plane_normalizations.append(normalization)
                coarse_planes.append(coarse / normalization)
            flatfield, flatfield_details = build_artifact_safe_flatfield(
                np.stack(coarse_planes),
                image_shape,
                smoothing_sigma=config.flatfield_smoothing_sigma_coarse_px,
                minimum_relative_illumination=config.flatfield_min_relative_illumination,
            )
            flatfields[channel.label] = flatfield
            channel_dir = output / channel.label
            channel_dir.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(channel_dir / f"{channel.label}_relative_illumination_float32.tif", flatfield)
            tifffile.imwrite(channel_dir / f"{channel.label}_correction_factor_float32.tif", 1.0 / flatfield)
            tifffile.imwrite(channel_dir / f"{channel.label}_persistent_hot_pixel_mask.tif", hot_mask.astype(np.uint8))
            _save_flatfield_plot(
                channel_dir / f"{channel.label}_calibration_flatfield.png",
                flatfield,
                channel,
                f"{channel.label}: artifact-safe FFC from {config.calibration_reference_concentration_ug_ml:g} µg/mL",
            )
            diagnostics[channel.label] = {
                "blank_file": str(blank_path.resolve()),
                "reference_file": str(reference_path.resolve()),
                "reference_concentration_ug_ml": config.calibration_reference_concentration_ug_ml,
                "background_scalar": background,
                "z_plane_count": z_count,
                "plane_normalizations": plane_normalizations,
                "reference_clipped_pixel_count": clipped_count,
                "despiked_pixel_counts_by_z": despiked_counts,
                "hot_pixel_detection": hot_details,
                "flatfield": flatfield_details,
                "policy": "normalize each reference Z plane, median across Z, reject spatial defects, then broad 2-D smoothing",
            }
    write_json(output / "flatfield_diagnostics.json", diagnostics)
    return backgrounds, hot_masks, flatfields, diagnostics


def _measure_calibration_series(
    standards: list[tuple[float, Path]],
    config: AnalysisConfig,
    backgrounds: dict[str, float],
    hot_masks: dict[str, np.ndarray],
    flatfields: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for concentration, path in standards:
        with nd2.ND2File(path) as file:
            indices = _channel_indices(file, config.channels)
            data = file.to_dask()
            z_count = int(file.sizes.get("Z", 1))
            z_step_um = float(file.voxel_size().z)
            for channel in config.channels:
                blocks: list[np.ndarray] = []
                clipped_count = 0
                despiked_count = 0
                for z in range(z_count):
                    acquired = np.asarray(data[z, indices[channel.label]].compute())
                    clipped = acquired == np.iinfo(acquired.dtype).max
                    clipped_count += int(np.count_nonzero(clipped))
                    repaired, spike_mask = despike_image(acquired, hot_masks[channel.label])
                    despiked_count += int(np.count_nonzero(spike_mask))
                    signal = subtract_background_floor(repaired, backgrounds[channel.label])
                    corrected = signal / flatfields[channel.label]
                    corrected[clipped] = np.nan
                    blocks.append(block_median(corrected, config.calibration_block_size_px).reshape(-1))
                block_stack = np.stack(blocks)
                integrated = trapezoid(block_stack, dx=z_step_um, axis=0)
                center = float(np.nanmedian(integrated))
                mad = float(np.nanmedian(np.abs(integrated - center)))
                tolerance = max(6.0 * 1.4826 * mad, np.finfo(float).eps)
                valid = np.isfinite(integrated) & (np.abs(integrated - center) <= tolerance)
                retained = integrated[valid]
                if retained.size < max(4, integrated.size // 2):
                    raise ValueError(f"Too few valid calibration blocks for {path.name} {channel.label}.")
                rows.append(
                    {
                        "file": path.name,
                        "concentration_ug_ml": concentration,
                        "channel": channel.label,
                        "z_plane_count": z_count,
                        "z_step_um": z_step_um,
                        "integrated_signal_au_um": float(np.median(retained)),
                        "block_q25_au_um": float(np.percentile(retained, 25)),
                        "block_q75_au_um": float(np.percentile(retained, 75)),
                        "block_mad_au_um": float(np.median(np.abs(retained - np.median(retained)))),
                        "retained_block_count": int(retained.size),
                        "excluded_block_count": int(integrated.size - retained.size),
                        "clipped_pixel_count": clipped_count,
                        "despiked_pixel_count": despiked_count,
                    }
                )
    models: dict[str, dict[str, Any]] = {}
    for channel in config.channels:
        selected = sorted(
            (row for row in rows if row["channel"] == channel.label),
            key=lambda row: float(row["concentration_ug_ml"]),
        )
        concentrations = np.asarray([float(row["concentration_ug_ml"]) for row in selected])
        signals = np.asarray([float(row["integrated_signal_au_um"]) for row in selected])
        if np.any(np.diff(signals) <= 0):
            raise ValueError(f"{channel.label} calibration signal is not strictly monotonic.")
        coefficients = np.polyfit(concentrations, signals, 1)
        predicted = np.polyval(coefficients, concentrations)
        residual = signals - predicted
        ss_total = float(np.sum((signals - np.mean(signals)) ** 2))
        r_squared = 1.0 - float(np.sum(residual**2)) / ss_total
        pearson = float(np.corrcoef(concentrations, signals)[0, 1])
        models[channel.label] = {
            "primary_conversion": "monotonic piecewise-linear interpolation of all measured standards; values outside 0-32 µg/mL are clipped and flagged",
            "concentrations_ug_ml": concentrations.tolist(),
            "integrated_signals_au_um": signals.tolist(),
            "linear_diagnostic": {
                "signal_intercept_au_um": float(coefficients[1]),
                "signal_per_ug_ml": float(coefficients[0]),
                "r_squared": r_squared,
                "pearson_r": pearson,
                "residuals_au_um": residual.tolist(),
            },
        }
    return rows, models


def _save_calibration_plot(
    path: Path,
    rows: list[dict[str, Any]],
    models: dict[str, dict[str, Any]],
    channels: tuple[ChannelConfig, ...],
) -> None:
    fig, axes = plt.subplots(1, len(channels), figsize=(7 * len(channels), 5.5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, channel in zip(axes, channels):
        selected = sorted(
            (row for row in rows if row["channel"] == channel.label),
            key=lambda row: float(row["concentration_ug_ml"]),
        )
        concentration = np.asarray([float(row["concentration_ug_ml"]) for row in selected])
        signal = np.asarray([float(row["integrated_signal_au_um"]) for row in selected])
        low = signal - np.asarray([float(row["block_q25_au_um"]) for row in selected])
        high = np.asarray([float(row["block_q75_au_um"]) for row in selected]) - signal
        axis.errorbar(
            concentration,
            signal,
            yerr=np.vstack([low, high]),
            fmt="o",
            capsize=3,
            color=channel.plot_color,
            label="Measured block median ± IQR",
        )
        axis.plot(concentration, signal, color=channel.plot_color, linewidth=1.3, label="Empirical conversion")
        linear = models[channel.label]["linear_diagnostic"]
        fit = float(linear["signal_intercept_au_um"]) + float(linear["signal_per_ug_ml"]) * concentration
        axis.plot(concentration, fit, color="black", linestyle="--", label=f"Linear diagnostic, R²={float(linear['r_squared']):.3f}")
        axis.set(
            title=f"{channel.label} calibration",
            xlabel="Dextran concentration (µg/mL)",
            ylabel="Z-integrated corrected intensity (a.u.·µm)",
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_profile_plot(
    path: Path,
    profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    channels: tuple[ChannelConfig, ...],
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, axis = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    for channel in channels:
        x, y = profiles[channel.label]
        axis.plot(x, y, color=channel.plot_color, linewidth=1.2, label=channel.label)
    axis.set(title=title, xlabel="Position along device (mm)", ylabel=ylabel)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.grid(alpha=0.2)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_tanh_plot(
    path: Path,
    profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    fits: dict[str, dict[str, Any]],
    channels: tuple[ChannelConfig, ...],
    title: str,
) -> None:
    fig, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    for channel in channels:
        x, y = profiles[channel.label]
        fit = fits[channel.label]
        axis.plot(x, y, color=channel.plot_color, linewidth=0.8, alpha=0.55, label=f"{channel.label} data")
        predicted = tanh_profile(
            x,
            float(fit["left_plateau"]),
            float(fit["right_plateau"]),
            float(fit["midpoint_mm"]),
            float(fit["width_mm"]),
        )
        axis.plot(
            x,
            predicted,
            color=channel.plot_color,
            linestyle="--",
            linewidth=2.2,
            label=f"{channel.label} fit; slope={float(fit['signed_slope_ug_ml_per_mm']):.3g} µg/mL/mm",
        )
    axis.set(
        title=title,
        xlabel="Position along device (mm)",
        ylabel="Concentration / 32 µg/mL",
        ylim=(-0.02, 1.02),
    )
    axis.grid(alpha=0.2)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_overlap_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [str(row["channel"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.26
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.bar(x - width, [row["before_median_abs_log_ratio"] for row in rows], width, label="Before correction", color="gray")
    axis.bar(x, [row["ffc_median_abs_log_ratio"] for row in rows], width, label="Calibration FFC", color="tab:blue")
    axis.bar(x + width, [row["overlap_quadratic_residual"] for row in rows], width, label="Overlap quadratic", color="tab:orange")
    axis.set_xticks(x, labels)
    axis.set(ylabel="Median absolute overlap log-ratio", title="Correction comparison at the selected Z plane")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _camera_matrix(file: nd2.ND2File) -> np.ndarray:
    frame = file.frame_metadata(0).channels[0]
    m11, m12, m21, m22 = frame.volume.cameraTransformationMatrix
    # Nikon stores the two camera-axis vectors consecutively; columns map
    # camera X/Y offsets to stage X/Y offsets.
    return np.asarray([[m11, m21], [m12, m22]], dtype=np.float64)


def _large_dic_crop(
    large_path: Path,
    gradient_positions: list[tuple[float, float]],
    gradient_shape: tuple[int, int],
    gradient_pixel_um: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    with nd2.ND2File(large_path) as file:
        if set(file.sizes) != {"Y", "X"}:
            raise ValueError("The large DIC file must contain one 2-D image.")
        height, width = int(file.sizes["Y"]), int(file.sizes["X"])
        pixel_um = float(file.voxel_size().x)
        frame = file.frame_metadata(0).channels[0]
        stage = frame.position.stagePositionUm
        center_stage = np.asarray([float(stage.x), float(stage.y)])
        transform = _camera_matrix(file)
        inverse = np.linalg.inv(transform)
        camera_centers = np.stack(
            [inverse @ (np.asarray(position) - center_stage) / pixel_um for position in gradient_positions]
        )
        camera_centers[:, 0] += width / 2.0
        camera_centers[:, 1] += height / 2.0
        tile_height, tile_width = gradient_shape
        half_x = tile_width * gradient_pixel_um / (2.0 * pixel_um)
        half_y = tile_height * gradient_pixel_um / (2.0 * pixel_um)
        x0 = max(0, int(np.floor(np.min(camera_centers[:, 0] - half_x))))
        x1 = min(width, int(np.ceil(np.max(camera_centers[:, 0] + half_x))))
        y0 = max(0, int(np.floor(np.min(camera_centers[:, 1] - half_y))))
        y1 = min(height, int(np.ceil(np.max(camera_centers[:, 1] + half_y))))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Gradient stage rectangle lies outside the large DIC image.")
        complete = file.read_frame(0)
        crop = np.asarray(complete[y0:y1, x0:x1]).copy()
        del complete
        return crop, {
            "large_image": str(large_path.resolve()),
            "large_image_shape_yx": [height, width],
            "large_pixel_size_um": pixel_um,
            "large_stage_center_um": center_stage.tolist(),
            "crop_bounds_xyxy_px": [x0, y0, x1, y1],
            "crop_shape_yx": list(crop.shape),
            "gradient_position_centers_in_large_px": camera_centers.tolist(),
            "camera_to_stage_matrix": transform.tolist(),
        }


def _save_dic_and_composite(
    output: Path,
    crop: np.ndarray,
    merge_path: Path,
    low: float,
    high: float,
) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    crop_tiff = output / "large_DIC_gradient_stage_rectangle.tif"
    tifffile.imwrite(crop_tiff, crop)
    scaled = np.round(np.clip((crop.astype(np.float32) - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    crop_png = output / f"large_DIC_gradient_stage_rectangle_display_{low:g}-{high:g}.png"
    Image.fromarray(scaled, mode="L").save(crop_png)
    fluorescence = Image.open(merge_path).convert("RGB")
    dic = Image.fromarray(scaled, mode="L").convert("RGB").resize(fluorescence.size, Image.Resampling.BILINEAR)
    composite = Image.new("RGB", (fluorescence.width, fluorescence.height * 2), "black")
    composite.paste(dic, (0, 0))
    composite.paste(fluorescence, (0, fluorescence.height))
    composite_path = output / "large_DIC_above_t012_24h_integrated_GFP-Cy5.png"
    composite.save(composite_path)
    return crop_png, composite_path


def run_calibrated_pipeline(
    nd2_path: str | Path, output_root: str | Path, config: AnalysisConfig
) -> Path:
    """Run a Z-integrated, calibration-derived FFC pilot."""
    source_path = Path(nd2_path)
    calibration_directory = Path(str(config.calibration_directory))
    if not calibration_directory.is_dir():
        raise ValueError(f"Calibration directory not found: {calibration_directory}")
    if not config.calibration_z_integrated:
        raise ValueError("The calibration-flatfield workflow currently requires calibration_z_integrated=true.")
    standards = _discover_standards(calibration_directory)
    base = _safe_stem(source_path.stem)
    run_dir = Path(output_root) / f"run_{base}_calibrated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    step_names = {
        1: "step_01_stitched_images",
        2: "step_02_profiles_before_correction",
        3: "step_03_calibration_flatfield_corrected",
        4: "step_04_calibration_curve",
        5: "step_05_concentration_profiles",
        6: "step_06_tanh_gradient_fits",
        7: "step_07_dic_context",
        8: "step_08_quality_control",
    }
    steps = {number: run_dir / name for number, name in step_names.items()}
    for directory in steps.values():
        directory.mkdir(parents=True, exist_ok=False)

    backgrounds, hot_masks, flatfields, flatfield_diagnostics = _calibration_background_and_flatfields(
        standards, config, steps[3] / "flatfield_diagnostics"
    )
    calibration_rows, calibration_models = _measure_calibration_series(
        standards, config, backgrounds, hot_masks, flatfields
    )
    write_rows_csv(steps[4] / "calibration_measurements.csv", calibration_rows)
    write_json(steps[4] / "calibration_measurements.json", calibration_rows)
    write_json(steps[4] / "calibration_models.json", calibration_models)
    _save_calibration_plot(
        steps[4] / "GFP-Cy5_Z-integrated_intensity_vs_concentration.png",
        calibration_rows,
        calibration_models,
        config.channels,
    )

    timepoints = tuple(config.timepoints or ())
    if timepoints != (12,):
        raise ValueError("This pilot is intentionally limited to timepoint t012.")
    selected_z = config.zero_based_z
    artifact_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    before_profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    corrected_signal_profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    concentration_profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    normalized_profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fit_records: list[dict[str, Any]] = []
    selected_raw_tiles: dict[str, list[np.ndarray]] = {}
    selected_corrected_tiles: dict[str, list[np.ndarray]] = {}
    integrated_tiles_by_channel: dict[str, list[np.ndarray]] = {}
    union_geometry: dict[str, Any] | None = None
    source_metadata: dict[str, Any]

    with ND2Source(source_path, config) as source:
        if source.pixel_size_um is None:
            raise ValueError("Gradient ND2 pixel size metadata are required.")
        z_count = int(source.sizes.get("Z", 1))
        if z_count < 2:
            raise ValueError("Z-integrated calibration requires at least two gradient Z planes.")
        z_step_um = 10.0
        for loop in source._file.experiment:
            if getattr(loop, "type", "") == "ZStackLoop":
                z_step_um = float(loop.parameters.stepUm)
        height, width = int(source.sizes["Y"]), int(source.sizes["X"])
        for channel in config.channels:
            if flatfields[channel.label].shape != (height, width):
                raise ValueError(f"Calibration FFC shape does not match gradient tiles for {channel.label}.")
        x_axes = physical_x_axes_mm(
            [position.x_um for position in source.positions], float(source.pixel_size_um), width
        )
        pixel_mm = float(source.pixel_size_um) / 1000.0
        x_origin = min(float(axis[0]) for axis in x_axes)
        x_starts = [(float(axis[0]) - x_origin) / pixel_mm for axis in x_axes]
        first_y = float(source.positions[0].y_um)
        y_starts = [(float(position.y_um) - first_y) / float(source.pixel_size_um) for position in source.positions]

        for channel in config.channels:
            raw_z_profiles = np.empty((z_count, len(source.positions), width), dtype=np.float64)
            corrected_z_profiles = np.empty_like(raw_z_profiles)
            integrated_tiles = [np.zeros((height, width), dtype=np.float32) for _ in source.positions]
            raw_at_selected: list[np.ndarray] = []
            corrected_at_selected: list[np.ndarray] = []
            masks_at_selected: list[np.ndarray] = []
            clipped_total = 0
            despiked_total = 0
            for z in range(z_count):
                trapezoid_weight = 0.5 if z in {0, z_count - 1} else 1.0
                for position_index, position in enumerate(source.positions):
                    acquired = source.plane(12, position.index, channel, z)
                    clipped = acquired == np.iinfo(acquired.dtype).max
                    clipped_total += int(np.count_nonzero(clipped))
                    repaired, spike_mask = despike_image(acquired, hot_masks[channel.label])
                    despiked_total += int(np.count_nonzero(spike_mask))
                    signal = subtract_background_floor(repaired, backgrounds[channel.label])
                    raw_profile, _, raw_details = robust_y_profile(
                        signal,
                        outlier_sigma=config.profile_outlier_sigma,
                        minimum_deviation=config.profile_minimum_deviation,
                    )
                    corrected = signal / flatfields[channel.label]
                    corrected_profile, valid, corrected_details = robust_y_profile(
                        corrected,
                        outlier_sigma=config.profile_outlier_sigma,
                        minimum_deviation=config.profile_minimum_deviation,
                    )
                    raw_z_profiles[z, position_index] = raw_profile
                    corrected_z_profiles[z, position_index] = corrected_profile
                    integrated_tiles[position_index] += corrected * (trapezoid_weight * z_step_um)
                    artifact_rows.append(
                        {
                            "timepoint": 12,
                            "experimental_time_hours": 12 * config.experimental_hours_per_timepoint,
                            "z_index_zero_based": z,
                            "z_index_one_based": z + 1,
                            "position": position.name,
                            "channel": channel.label,
                            "clipped_pixel_count": int(np.count_nonzero(clipped)),
                            "despiked_pixel_count": int(np.count_nonzero(spike_mask)),
                            "before_excluded_fraction": raw_details["excluded_pixel_fraction"],
                            "after_ffc_excluded_fraction": corrected_details["excluded_pixel_fraction"],
                        }
                    )
                    if z == selected_z:
                        raw_at_selected.append(signal)
                        corrected_at_selected.append(corrected)
                        masks_at_selected.append(~valid)
            integrated_tiles_by_channel[channel.label] = integrated_tiles
            selected_raw_tiles[channel.label] = raw_at_selected
            selected_corrected_tiles[channel.label] = corrected_at_selected
            raw_integrated = trapezoid(raw_z_profiles, dx=z_step_um, axis=0)
            corrected_integrated = trapezoid(corrected_z_profiles, dx=z_step_um, axis=0)
            before_profiles[channel.label] = feather_profiles(x_axes, list(raw_integrated))[0:2]
            corrected_signal_profiles[channel.label] = feather_profiles(x_axes, list(corrected_integrated))[0:2]

            mask_mosaic, _, _ = feather_tiles_2d_union(
                [mask.astype(np.float32) for mask in masks_at_selected], x_starts, y_starts
            )
            tifffile.imwrite(
                steps[8] / f"t012_z{selected_z + 1:02d}_{channel.label}_quantitative_exclusion_mask.tif",
                np.nan_to_num(mask_mosaic, nan=0.0).astype(np.uint8),
            )
            Image.fromarray(
                (np.nan_to_num(mask_mosaic, nan=0.0) > 0.5).astype(np.uint8) * 255,
                mode="L",
            ).save(steps[8] / f"t012_z{selected_z + 1:02d}_{channel.label}_quantitative_exclusion_mask.png")

            for pair_index, (left_position, right_position) in enumerate(zip(source.positions, source.positions[1:])):
                x_offset = int(round(abs(right_position.x_um - left_position.x_um) / float(source.pixel_size_um)))
                y_shift = int(round((right_position.y_um - left_position.y_um) / float(source.pixel_size_um)))
                _, _, raw_ratio, _ = overlap_log_ratio_samples(
                    raw_at_selected[pair_index], raw_at_selected[pair_index + 1],
                    x_offset, y_shift, config.overlap_min_signal,
                    float(np.iinfo(np.uint16).max) - backgrounds[channel.label],
                )
                _, _, ffc_ratio, _ = overlap_log_ratio_samples(
                    corrected_at_selected[pair_index], corrected_at_selected[pair_index + 1],
                    x_offset, y_shift, config.overlap_min_signal,
                    float(np.iinfo(np.uint16).max) - backgrounds[channel.label],
                )
                overlap_rows.append(
                    {
                        "channel": channel.label,
                        "position_pair": f"{left_position.name}-{right_position.name}",
                        "before_median_abs_log_ratio": float(np.median(np.abs(raw_ratio))) if raw_ratio.size else float("nan"),
                        "ffc_median_abs_log_ratio": float(np.median(np.abs(ffc_ratio))) if ffc_ratio.size else float("nan"),
                    }
                )
            raw_samples = []
            for pair_index, (left_position, right_position) in enumerate(zip(source.positions, source.positions[1:])):
                x_offset = int(round(abs(right_position.x_um - left_position.x_um) / float(source.pixel_size_um)))
                y_shift = int(round((right_position.y_um - left_position.y_um) / float(source.pixel_size_um)))
                xl, xr, ratio, _ = overlap_log_ratio_samples(
                    raw_at_selected[pair_index], raw_at_selected[pair_index + 1],
                    x_offset, y_shift, config.overlap_min_signal,
                    float(np.iinfo(np.uint16).max) - backgrounds[channel.label],
                )
                if ratio.size:
                    raw_samples.append((xl, xr, ratio))
            if raw_samples:
                _, model = fit_overlap_log_quadratic(
                    width,
                    np.concatenate([item[0] for item in raw_samples]),
                    np.concatenate([item[1] for item in raw_samples]),
                    np.concatenate([item[2] for item in raw_samples]),
                )
                for row in overlap_rows:
                    if row["channel"] == channel.label:
                        row["overlap_quadratic_residual"] = float(model["median_abs_log_residual"])
            else:
                for row in overlap_rows:
                    if row["channel"] == channel.label:
                        row["overlap_quadratic_residual"] = float("nan")

            integrated_mosaic, _, union_geometry = feather_tiles_2d_union(integrated_tiles, x_starts, y_starts)
            tifffile.imwrite(
                steps[3] / f"gradient_t012_24h_{channel.label}_Z-integrated_FFC_float32.tif",
                integrated_mosaic.astype(np.float32),
            )
            finite = integrated_mosaic[np.isfinite(integrated_mosaic)]
            low, high = np.percentile(finite, [config.preview_low_percentile, config.preview_high_percentile])
            save_color_preview(
                steps[3] / f"gradient_t012_24h_{channel.label}_Z-integrated_FFC_preview.png",
                np.nan_to_num(integrated_mosaic, nan=0.0),
                float(low), float(high), channel.rgb,
            )
            selected_mosaic, _, _ = feather_tiles_2d_union(corrected_at_selected, x_starts, y_starts)
            selected_uint16, selected_clipped = to_uint16(selected_mosaic)
            tifffile.imwrite(
                steps[3] / f"gradient_t012_24h_z{selected_z + 1:02d}_{channel.label}_FFC_mosaic.tif",
                selected_uint16,
            )
            local_low, local_high = image_percentile_range(selected_uint16, config.preview_low_percentile, config.preview_high_percentile)
            save_color_preview(
                steps[3] / f"gradient_t012_24h_z{selected_z + 1:02d}_{channel.label}_FFC_mosaic_preview.png",
                selected_uint16, local_low, local_high, channel.rgb,
            )
            source_metadata = {
                "clipped_pixel_count": clipped_total,
                "despiked_pixel_count": despiked_total,
                "selected_z_uint16_conversion_clipped_count": selected_clipped,
            }

        raw_display: list[np.ndarray] = []
        integrated_display: list[np.ndarray] = []
        for channel in config.channels:
            raw_mosaic, _, _ = feather_tiles_2d_union(selected_raw_tiles[channel.label], x_starts, y_starts)
            raw_uint16, _ = to_uint16(raw_mosaic)
            tifffile.imwrite(
                steps[1] / f"gradient_t012_24h_z{selected_z + 1:02d}_{channel.label}_background-subtracted_mosaic.tif",
                raw_uint16,
            )
            low, high = image_percentile_range(raw_uint16, config.preview_low_percentile, config.preview_high_percentile)
            save_color_preview(
                steps[1] / f"gradient_t012_24h_z{selected_z + 1:02d}_{channel.label}_background-subtracted_mosaic_preview.png",
                raw_uint16, low, high, channel.rgb,
            )
            raw_display.append(raw_uint16)
            integrated_mosaic, _, _ = feather_tiles_2d_union(integrated_tiles_by_channel[channel.label], x_starts, y_starts)
            finite = integrated_mosaic[np.isfinite(integrated_mosaic)]
            low, high = np.percentile(finite, [config.preview_low_percentile, config.preview_high_percentile])
            integrated_display.append(
                np.nan_to_num(integrated_mosaic, nan=0.0).astype(np.float32)
            )
        raw_ranges = [image_percentile_range(image, config.preview_low_percentile, config.preview_high_percentile) for image in raw_display]
        save_rgb(
            steps[1] / f"gradient_t012_24h_z{selected_z + 1:02d}_GFP-Cy5_background-subtracted_merge.png",
            merge_rgb(raw_display, [channel.rgb for channel in config.channels], raw_ranges),
        )
        integrated_ranges = [
            tuple(np.percentile(image[image > 0], [config.preview_low_percentile, config.preview_high_percentile]))
            for image in integrated_display
        ]
        integrated_merge_path = steps[3] / "gradient_t012_24h_GFP-Cy5_Z-integrated_FFC_merge.png"
        save_rgb(
            integrated_merge_path,
            merge_rgb(integrated_display, [channel.rgb for channel in config.channels], integrated_ranges),
        )

        source_metadata = {
            "source": str(source_path.resolve()),
            "dimensions": source.sizes,
            "channels_analyzed": [channel.label for channel in config.channels],
            "ignored_channel": "X20-DIC",
            "timepoint": 12,
            "experimental_time_hours": 12 * config.experimental_hours_per_timepoint,
            "z_plane_count": z_count,
            "z_step_um": z_step_um,
            "selected_display_z_one_based": selected_z + 1,
            "pixel_size_um": source.pixel_size_um,
            "positions_left_to_right": [
                {
                    "name": position.name,
                    "nd2_index": position.index,
                    "x_um": position.x_um,
                    "y_um": position.y_um,
                }
                for position in source.positions
            ],
            "union_mosaic_geometry": union_geometry,
        }
        gradient_positions = [(position.x_um, position.y_um) for position in source.positions]

    _save_profile_plot(
        steps[2] / "gradient_t012_24h_GFP-Cy5_Z-integrated_before-correction.png",
        before_profiles,
        config.channels,
        "t012 (24 h): Z-integrated profiles before illumination correction",
        "Background-subtracted integrated intensity (a.u.·µm)",
    )
    _save_profile_plot(
        steps[3] / "gradient_t012_24h_GFP-Cy5_Z-integrated_after-FFC.png",
        corrected_signal_profiles,
        config.channels,
        "t012 (24 h): Z-integrated profiles after calibration FFC",
        "FFC-corrected integrated intensity (a.u.·µm)",
    )

    conversion_rows: list[dict[str, Any]] = []
    for channel in config.channels:
        model = calibration_models[channel.label]
        x, signal = corrected_signal_profiles[channel.label]
        concentration, outside = empirical_concentration(
            signal,
            np.asarray(model["integrated_signals_au_um"]),
            np.asarray(model["concentrations_ug_ml"]),
        )
        concentration_profiles[channel.label] = (x, concentration)
        normalized_profiles[channel.label] = (
            x,
            concentration / config.calibration_reference_concentration_ug_ml,
        )
        conversion_rows.append(
            {
                "channel": channel.label,
                "profile_point_count": int(signal.size),
                "below_calibration_count": int(np.count_nonzero(signal < model["integrated_signals_au_um"][0])),
                "above_calibration_count": int(np.count_nonzero(signal > model["integrated_signals_au_um"][-1])),
                "out_of_range_fraction": float(np.count_nonzero(outside) / outside.size),
                "minimum_concentration_ug_ml": float(np.min(concentration)),
                "maximum_concentration_ug_ml": float(np.max(concentration)),
            }
        )
        fit = fit_tanh_profile(
            x,
            normalized_profiles[channel.label][1],
            max_points=config.tanh_max_points,
            pixel_size_mm=float(x[1] - x[0]),
        )
        fit.update(
            {
                "timepoint": 12,
                "experimental_time_hours": 12 * config.experimental_hours_per_timepoint,
                "channel": channel.label,
                "normalization_concentration_ug_ml": config.calibration_reference_concentration_ug_ml,
                "signed_slope_ug_ml_per_mm": float(fit["signed_slope_per_mm"]) * config.calibration_reference_concentration_ug_ml,
                "absolute_slope_ug_ml_per_mm": float(fit["absolute_slope_per_mm"]) * config.calibration_reference_concentration_ug_ml,
            }
        )
        fit_records.append(fit)
    _save_profile_plot(
        steps[5] / "gradient_t012_24h_GFP-Cy5_concentration_profiles.png",
        concentration_profiles,
        config.channels,
        "t012 (24 h): calibrated dextran concentration profiles",
        "Dextran concentration (µg/mL)",
        (0.0, config.calibration_reference_concentration_ug_ml * 1.02),
    )
    _save_profile_plot(
        steps[5] / "gradient_t012_24h_GFP-Cy5_normalized_concentration_profiles.png",
        normalized_profiles,
        config.channels,
        "t012 (24 h): calibrated profiles normalized to 32 µg/mL",
        "Concentration / 32 µg/mL",
        (0.0, 1.02),
    )
    write_rows_csv(steps[5] / "concentration_conversion_qc.csv", conversion_rows)
    write_json(steps[5] / "concentration_conversion_qc.json", conversion_rows)
    fit_by_channel = {str(record["channel"]): record for record in fit_records}
    _save_tanh_plot(
        steps[6] / "gradient_t012_24h_GFP-Cy5_calibrated_tanh_fit.png",
        normalized_profiles,
        fit_by_channel,
        config.channels,
        "t012 (24 h): tanh fits to calibrated Z-integrated gradients",
    )
    write_rows_csv(steps[6] / "tanh_fit_results.csv", fit_records)
    write_json(steps[6] / "tanh_fit_results.json", fit_records)

    pair_rows = overlap_rows
    summary_rows: list[dict[str, Any]] = []
    for channel in config.channels:
        selected = [row for row in pair_rows if row["channel"] == channel.label]
        summary_rows.append(
            {
                "channel": channel.label,
                "pair_count": len(selected),
                "before_median_abs_log_ratio": float(np.nanmedian([row["before_median_abs_log_ratio"] for row in selected])),
                "ffc_median_abs_log_ratio": float(np.nanmedian([row["ffc_median_abs_log_ratio"] for row in selected])),
                "overlap_quadratic_residual": float(np.nanmedian([row["overlap_quadratic_residual"] for row in selected])),
            }
        )
    write_rows_csv(steps[8] / "overlap_pair_comparison.csv", pair_rows)
    write_rows_csv(steps[8] / "overlap_correction_summary.csv", summary_rows)
    write_json(steps[8] / "overlap_correction_summary.json", summary_rows)
    write_rows_csv(steps[8] / "artifact_exclusion_by_tile_z.csv", artifact_rows)
    write_json(steps[8] / "artifact_exclusion_by_tile_z.json", artifact_rows)
    _save_overlap_comparison(steps[8] / "FFC_vs_overlap_quadratic.png", summary_rows)

    dic_metadata: dict[str, Any] | None = None
    if config.large_dic_path:
        crop, dic_metadata = _large_dic_crop(
            Path(config.large_dic_path),
            gradient_positions,
            (int(source_metadata["dimensions"]["Y"]), int(source_metadata["dimensions"]["X"])),
            float(source_metadata["pixel_size_um"]),
        )
        _, composite_path = _save_dic_and_composite(
            steps[7], crop, integrated_merge_path,
            config.large_dic_display_low, config.large_dic_display_high,
        )
        dic_metadata.update(
            {
                "display_range": [config.large_dic_display_low, config.large_dic_display_high],
                "aligned_composite": str(composite_path),
                "alignment_policy": "stage-coordinate rectangle from the union of all gradient-position fields; same camera-to-stage orientation",
            }
        )
        write_json(steps[7] / "DIC_crop_metadata.json", dic_metadata)

    metadata = {
        **source_metadata,
        "correction_method": "calibration_flatfield",
        "correction_reference_concentration_ug_ml": config.calibration_reference_concentration_ug_ml,
        "calibration_directory": str(calibration_directory.resolve()),
        "calibration_files": [
            {"file": path.name, "concentration_ug_ml": concentration}
            for concentration, path in standards
        ],
        "background_by_channel": backgrounds,
        "flatfield_policy": "one artifact-safe 2-D map per channel from all Z planes of the 32 µg/mL homogeneous standard; reused for every experimental Z",
        "z_matching_policy": "no calibration-to-experiment Z matching; concentration uses full-stack trapezoidal integration at 10 µm spacing",
        "concentration_conversion": calibration_models,
        "stain_policy": "stains remain visible in saved images; localized Y outliers are excluded only from quantitative column profiles and recorded",
        "hot_pixel_policy": "persistent blank-derived mask plus per-image isolated-spike replacement with the local median",
        "flatfield_diagnostics": flatfield_diagnostics,
        "concentration_conversion_qc": conversion_rows,
        "overlap_validation": summary_rows,
        "tanh_model": "left+(right-left)/2*(1+tanh((x-midpoint)/width))",
        "tanh_slope": "normalized slope=(right-left)/(2*width); concentration slope=normalized slope*32 µg/mL",
        "dic_context": dic_metadata,
    }
    write_json(run_dir / "run_metadata.json", metadata)
    return run_dir
