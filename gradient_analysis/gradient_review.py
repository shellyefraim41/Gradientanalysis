"""Chronological gradient review built on cached full-resolution measurements.

Output folders are numbered in the order the analysis is performed and named
for what they contain, so the run can be read from top to bottom:

calibration-backed data (September):
  01 background, hot pixels, and flat field from the calibration standards
  02 calibration curves (intensity vs concentration) and conversion choices
  03 illumination-correction methods compared at tile overlaps (seams)
  04 bridge location from the large DIC image
  05 stain mask: regions excluded from fits
  06 Z selection: tanh fits for every Z plane
  07 Z-integrated profiles: correction methods compared
  08 primary-method Z-integrated profiles with tanh fits, one plot per timepoint

Data without calibration, DIC, or stain (July) keep the same order and simply
omit the steps that do not apply.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import tifffile

from .calibrated_pipeline import _large_dic_crop, _safe_stem, _save_flatfield_plot
from .config import AnalysisConfig
from .correction_comparison import (
    FITTED_METHODS,
    METHOD_BASE,
    METHOD_COLORS,
    METHOD_LABELS,
    METHODS,
    RunLog,
    calibration_model_from_block_values,
    collect_measurements,
    correct_ratios,
    fit_z_series,
    log_illumination,
    median_abs,
    pair_samples,
    profile_pair_mismatch,
)
from .nd2_source import ND2Source
from .outputs import write_json, write_rows_csv
from .processing import (
    empirical_concentration,
    feather_profiles,
    feather_tiles_2d_union,
    fit_tanh_profile,
    merge_rgb,
    quadratic_illumination_profile,
    tanh_profile,
)

X_LABEL = "Position along device [mm]"
REPRESENTATIONS = {
    "concentration": {
        "axis": "Dextran concentration [µg/mL]",
        "slope_unit": "µg/mL/mm",
        "folder": "1_concentration_ug_per_mL",
        "word": "calibrated concentration",
    },
    "normalized": {
        "axis": "Normalized intensity [fraction of channel maximum]",
        "slope_unit": "1/mm",
        "folder": "2_normalized_intensity",
        "word": "normalized intensity",
    },
    "intensity": {
        "axis": "Z-integrated intensity [a.u.·µm]",
        "slope_unit": "a.u.·µm/mm",
        "folder": "3_intensity_au_um",
        "word": "intensity",
    },
}
LINEAR_CALIBRATION_MAX_UG_ML = 24.0
RELAXED_PLATEAU_BOUNDS = (-1.0, 3.0)
HELD_OUT_TIME_GROUPS = 5
STAIN_RULE = {
    "pixel_darker_than_fraction_of_column_reference": 0.7,
    "column_reference_percentile": 90,
    "minimum_column_reference_fraction_of_image_max": 0.15,
    "minimum_fraction_of_images_voting_stain": 0.5,
    "minimum_images_evaluated": 3,
    "minimum_object_area_um2": 2000.0,
    "column_excluded_if_stain_fraction_above": 0.10,
    "excluded_region_dilation_um": 25.0,
}


# ---------------------------------------------------------------------------
# Pure helpers (unit tested)
# ---------------------------------------------------------------------------


def _hl(hours: float) -> str:
    """Compact hour label: 24, 0.45, 23.95 (no trailing zeros)."""
    return f"{hours:.2f}".rstrip("0").rstrip(".")


def detect_bridge_edges(
    column_profile: np.ndarray,
    pixel_mm: float,
    *,
    smoothing_px: float = 15.0,
    plateau_halfwidth_mm: float = 0.4,
    outer_window_mm: tuple[float, float] = (0.15, 0.45),
    foot_fraction: float = 0.1,
) -> dict[str, float]:
    """Locate the bright central DIC plateau (the bridge) and its edges.

    Half-maximum edges sit midway between the bridge level and the level just
    outside it; outer "foot" edges are where the signal has fallen 90% of the
    way to the outside level.
    """
    s = ndimage.gaussian_filter1d(np.asarray(column_profile, dtype=np.float64), smoothing_px)
    n = s.size
    center = n // 2
    halfwidth = max(1, int(plateau_halfwidth_mm / pixel_mm))
    bright = float(np.median(s[max(0, center - halfwidth): center + halfwidth]))
    first_threshold = 0.5 * (bright + float(np.median(s)))
    left_below = np.flatnonzero(s[:center] < first_threshold)
    right_below = np.flatnonzero(s[center:] < first_threshold)
    if not left_below.size or not right_below.size:
        raise ValueError("No bright central plateau found in the DIC column profile.")
    left_guess = int(left_below.max()) + 1
    right_guess = center + int(right_below.min()) - 1
    near, far = (max(1, int(value / pixel_mm)) for value in outer_window_mm)
    outer_left = float(np.median(s[max(0, left_guess - far): max(1, left_guess - near)]))
    outer_right = float(np.median(s[min(n - 1, right_guess + near): min(n, right_guess + far)]))

    def last_below(values: np.ndarray, threshold: float, stop: int) -> int:
        hits = np.flatnonzero(values[:stop] < threshold)
        return int(hits.max()) + 1 if hits.size else 0

    def first_below(values: np.ndarray, threshold: float, start: int) -> int:
        hits = np.flatnonzero(values[start:] < threshold)
        return start + int(hits.min()) - 1 if hits.size else n - 1

    left_half = last_below(s, 0.5 * (bright + outer_left), center)
    right_half = first_below(s, 0.5 * (bright + outer_right), center)
    left_foot = last_below(s, outer_left + foot_fraction * (bright - outer_left), left_half)
    right_foot = first_below(s, outer_right + foot_fraction * (bright - outer_right), right_half)
    return {
        "left_foot_mm": left_foot * pixel_mm,
        "left_half_max_mm": left_half * pixel_mm,
        "right_half_max_mm": right_half * pixel_mm,
        "right_foot_mm": right_foot * pixel_mm,
        "width_between_feet_mm": (right_foot - left_foot) * pixel_mm,
        "width_at_half_max_mm": (right_half - left_half) * pixel_mm,
        "center_mm": 0.5 * (left_foot + right_foot) * pixel_mm,
        "bridge_level": bright,
        "outside_level_left": outer_left,
        "outside_level_right": outer_right,
    }


def detect_stain_mask(
    mosaics: list[np.ndarray], pixel_um: float, rule: dict[str, float] = STAIN_RULE
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Objective stain mask: pixels much darker than the bright part of their column.

    The same physical stain appears in every channel and timepoint, so a pixel
    is called stain only if it is flagged in most images that have enough
    signal in that column. Returns (mask, vote_fraction, details).
    """
    stack = [np.asarray(item, dtype=np.float64) for item in mosaics]
    votes = np.zeros(stack[0].shape)
    counts = np.zeros(stack[0].shape)
    for image in stack:
        with np.errstate(invalid="ignore"):
            reference = np.nanpercentile(image, rule["column_reference_percentile"], axis=0)
        finite_reference = reference[np.isfinite(reference)]
        if not finite_reference.size:
            continue
        enough = np.isfinite(reference) & (reference > rule["minimum_column_reference_fraction_of_image_max"] * finite_reference.max())
        evaluated = np.isfinite(image) & enough[np.newaxis, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            flagged = evaluated & (image < rule["pixel_darker_than_fraction_of_column_reference"] * reference[np.newaxis, :])
        votes += flagged
        counts += evaluated
    fraction = np.where(counts > 0, votes / np.maximum(counts, 1), 0.0)
    mask = (fraction >= rule["minimum_fraction_of_images_voting_stain"]) & (counts >= rule["minimum_images_evaluated"])
    mask = ndimage.binary_opening(mask, np.ones((3, 3), dtype=bool))
    labels, count = ndimage.label(mask)
    pixel_area = pixel_um * pixel_um
    objects = []
    keep = np.zeros(count + 1, dtype=bool)
    for index in range(1, count + 1):
        area = float(np.count_nonzero(labels == index)) * pixel_area
        if area >= rule["minimum_object_area_um2"]:
            keep[index] = True
            ys, xs = np.nonzero(labels == index)
            objects.append({"area_mm2": area / 1e6, "x_range_px": [int(xs.min()), int(xs.max())]})
    mask = keep[labels]
    return mask, fraction, {"objects": objects, "images_used": len(stack)}


def excluded_profile_positions(
    mask: np.ndarray,
    valid: np.ndarray,
    column_mm: np.ndarray,
    profile_mm: np.ndarray,
    rule: dict[str, float] = STAIN_RULE,
) -> tuple[np.ndarray, np.ndarray]:
    """Profile positions excluded from fits, plus the stain fraction per column."""
    fraction = mask.sum(axis=0) / np.maximum(valid.sum(axis=0), 1)
    on_profile = np.interp(profile_mm, column_mm, fraction, left=0.0, right=0.0)
    excluded = on_profile > rule["column_excluded_if_stain_fraction_above"]
    step = float(np.median(np.diff(profile_mm)))
    dilation = int(round(rule["excluded_region_dilation_um"] / 1000.0 / step))
    if dilation > 0 and excluded.any():
        excluded = ndimage.binary_dilation(excluded, iterations=dilation)
    return excluded, fraction


def _align_dic_crop_to_profile(crop: np.ndarray, dic_meta: dict[str, Any], tile_width_mm: float,
                               dic_pixel_mm: float) -> tuple[np.ndarray, float]:
    """Orient the DIC crop like the profile and return its start position on the profile axis.

    The profile runs from the first to the last gradient position. If the large image is
    mirrored relative to that order the crop is flipped; if it does not reach the
    profile's first edge, the uncovered length is returned as an offset in mm.
    """
    centers_x = np.asarray(dic_meta["gradient_position_centers_in_large_px"], dtype=float)[:, 0]
    half = tile_width_mm / (2.0 * dic_pixel_mm)
    x0, _, x1, _ = dic_meta["crop_bounds_xyxy_px"]
    if centers_x[-1] >= centers_x[0]:
        offset_px = x0 - (centers_x.min() - half)
        return crop, max(0.0, float(offset_px)) * dic_pixel_mm
    offset_px = (centers_x.max() + half) - x1
    return crop[:, ::-1].copy(), max(0.0, float(offset_px)) * dic_pixel_mm


def intervals_from_mask(x: np.ndarray, excluded: np.ndarray) -> list[tuple[float, float]]:
    edges = np.flatnonzero(np.diff(np.r_[0, excluded.astype(int), 0]))
    return [(float(x[start]), float(x[stop - 1])) for start, stop in zip(edges[::2], edges[1::2])]


def linear_conversion(
    concentrations: np.ndarray, signals: np.ndarray, maximum_concentration: float, *, through_origin: bool
) -> dict[str, Any]:
    """Least-squares line through the standards up to ``maximum_concentration``."""
    concentrations = np.asarray(concentrations, dtype=np.float64)
    signals = np.asarray(signals, dtype=np.float64)
    used = concentrations <= maximum_concentration + 1e-9
    c, s = concentrations[used], signals[used]
    if through_origin:
        slope = float(np.sum(c * s) / np.sum(c * c))
        intercept = 0.0
    else:
        slope, intercept = (float(v) for v in np.polyfit(c, s, 1))
    predicted = intercept + slope * c
    ss_total = float(np.sum((s - s.mean()) ** 2))
    return {
        "standards_used_ug_ml": c.tolist(),
        "signal_per_ug_ml": slope,
        "signal_intercept_au_um": intercept,
        "intercept_as_concentration_ug_ml": intercept / slope,
        "through_origin": through_origin,
        "r_squared": 1.0 - float(np.sum((s - predicted) ** 2)) / ss_total,
        "residuals_au_um": (s - predicted).tolist(),
    }


def alternative_gradient_metrics(
    x: np.ndarray, y: np.ndarray, excluded: np.ndarray, *, smoothing_mm: float = 0.1, band: tuple[float, float] = (0.2, 0.8)
) -> dict[str, float]:
    """Model-free gradient measures for profiles that are not sigmoidal.

    * linear slope over the transition band, i.e. where the smoothed profile is
      between 20% and 80% of its range;
    * the steepest local slope of the smoothed profile and where it occurs.
    Excluded (stain) positions are bridged by interpolation for smoothing only
    and never used for either measure.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(y) & ~excluded
    filled = np.interp(x, x[keep], y[keep])
    step = float(np.median(np.diff(x)))
    smooth = ndimage.gaussian_filter1d(filled, smoothing_mm / step)
    low, high = float(np.min(smooth[keep])), float(np.max(smooth[keep]))
    in_band = keep & (smooth >= low + band[0] * (high - low)) & (smooth <= low + band[1] * (high - low))
    slope, intercept = (float(v) for v in np.polyfit(x[in_band], y[in_band], 1))
    derivative = np.gradient(smooth, x)
    derivative[~keep] = 0.0
    steepest = int(np.argmax(np.abs(derivative)))
    return {
        "band_linear_slope_per_mm": slope,
        "band_linear_intercept": intercept,
        "band_start_mm": float(x[in_band].min()),
        "band_end_mm": float(x[in_band].max()),
        "steepest_local_slope_per_mm": float(derivative[steepest]),
        "steepest_local_slope_position_mm": float(x[steepest]),
        "smoothing_mm": smoothing_mm,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _shade(axis, bridge: dict[str, float] | None, stain: list[tuple[float, float]]) -> None:
    if bridge:
        axis.axvspan(bridge["left_foot_mm"], bridge["right_foot_mm"], color="#d9c27a", alpha=0.25, lw=0,
                     label=f"Bridge from DIC ({bridge['width_between_feet_mm']:.2f} mm)")
    for index, (start, stop) in enumerate(stain):
        axis.axvspan(start, stop, facecolor="none", edgecolor="#b22222", hatch="///", lw=0,
                     label="Stain: excluded from fit" if index == 0 else None)


def _dic_strip(axis, dic: dict[str, Any], bridge: dict[str, float] | None) -> None:
    axis.imshow(dic["image"], cmap="gray", extent=dic["extent"], aspect="auto")
    if "profile_range_mm" in dic:
        axis.set_xlim(*dic["profile_range_mm"])  # keep the full profile visible where the DIC image does not reach
    if bridge:
        for key in ("left_foot_mm", "right_foot_mm"):
            axis.axvline(bridge[key], color="#d9a400", lw=1.2)
    axis.set_yticks([])
    axis.set_ylabel("DIC", rotation=0, labelpad=18, va="center")


def _figure(dic: dict[str, Any] | None, rows: int = 1, height: float = 5.6):
    if dic is None:
        fig, axes = plt.subplots(rows, 1, figsize=(13, height * rows), constrained_layout=True, squeeze=False)
        return fig, None, [row[0] for row in axes]
    fig = plt.figure(figsize=(13, height * rows + 1.2), constrained_layout=True)
    grid = GridSpec(rows + 1, 1, figure=fig, height_ratios=[0.9] + [5.0] * rows)
    top = fig.add_subplot(grid[0])
    axes = [fig.add_subplot(grid[i + 1], sharex=top) for i in range(rows)]
    return fig, top, axes


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_labeled_stack(path: Path, panels: list[tuple[str, np.ndarray]], title: str) -> None:
    band = 26
    width = max(panel.shape[1] for _, panel in panels)
    height = band + sum(panel.shape[0] + band for _, panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), title, fill="black")
    y = band
    for label, panel in panels:
        draw.text((6, y + 6), label, fill="black")
        y += band
        canvas.paste(Image.fromarray(panel, mode="RGB"), (0, y))
        y += panel.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def depth_modes(config: AnalysisConfig, layer: tuple[int, int] | None) -> list[dict[str, Any]]:
    """Main single-plane mode (config Z) and an optional mean-of-planes robustness mode."""
    z = config.zero_based_z
    modes = [{
        "key": f"Z{z + 1:02d}", "kind": "plane", "planes": [z], "title": f"single plane Z{z + 1}",
        "file_tag": f"Z{z + 1:02d}", "intensity_axis": f"Intensity at Z{z + 1} [a.u.]", "intensity_unit": "a.u.",
    }]
    if layer:
        first, last = layer
        modes.append({
            "key": f"Z{first:02d}-Z{last:02d}_mean", "kind": "mean", "planes": list(range(first - 1, last)),
            "title": f"mean of planes Z{first}–Z{last}", "file_tag": f"mean-Z{first:02d}-Z{last:02d}",
            "intensity_axis": f"Mean intensity over Z{first}–Z{last} [a.u.]", "intensity_unit": "a.u.",
        })
    return modes


def representation_info(rep: str, mode: dict[str, Any]) -> dict[str, str]:
    info = dict(REPRESENTATIONS[rep])
    if rep == "intensity":
        info["axis"] = mode["intensity_axis"]
        info["slope_unit"] = f"{mode['intensity_unit']}/mm"
        info["folder"] = "3_intensity_au"
    info["word"] = f"{info['word']} ({mode['title']})"
    return info


def run_gradient_review(
    nd2_path: str | Path,
    output_root: str | Path,
    config: AnalysisConfig,
    *,
    experiment_label: str,
    workers: int = 48,
    bin_factor: int = 8,
    reuse_dirs: list[Path] | None = None,
    z_analysis_timepoints: tuple[int, ...] | None = None,
    stain_mask_enabled: bool = True,
    layer: tuple[int, int] | None = None,
    run_tag: str = "gradient_review",
    z_first_one_based: int = 1,
    glass_plane_one_based: int | None = None,
    flagged_fit_analysis: bool = True,
    diffusion_analysis: bool = False,
    expected_d: tuple[float, float] | None = None,
) -> Path:
    source_path = Path(nd2_path)
    has_cal = bool(config.calibration_directory)
    methods = METHODS if has_cal else ("raw", "overlap")
    fitted = tuple(m for m in FITTED_METHODS if m in methods)
    primary = "hybrid" if has_cal else "overlap"
    reps = ["concentration", "normalized", "intensity"] if has_cal else ["normalized", "intensity"]
    modes = depth_modes(config, layer)
    main_mode = modes[0]

    specs: list[tuple[str, str]] = []
    if has_cal:
        specs += [
            ("flatfield", "background_hot-pixels_and_flat-field_from_calibration_standards"),
            ("calibration", "calibration_curves_intensity_vs_concentration"),
        ]
    specs.append(("seams", "illumination_correction_methods_compared_at_tile_overlaps"))
    if config.large_dic_path:
        specs.append(("bridge", "bridge_location_from_large_DIC_image"))
    if stain_mask_enabled:
        specs.append(("stain", "stain_mask_regions_excluded_from_fits"))
    specs += [
        ("zsel", "Z_selection_tanh_fits_for_every_Z_plane"),
        ("methods", f"{main_mode['file_tag']}_profiles_correction_methods_compared"),
        ("mode0", f"MAIN_RESULT_{primary}-corrected_{main_mode['file_tag']}_profiles_with_tanh_fits_per_timepoint"),
    ]
    for index, mode in enumerate(modes[1:], start=1):
        specs.append((f"mode{index}", f"robustness_check_{primary}-corrected_{mode['file_tag']}_profiles_with_tanh_fits"))
    if diffusion_analysis:
        specs.append(("diffusion", "front_displacement_speed_and_diffusion_coefficient_over_time"))
    run_dir = Path(output_root) / f"run_{_safe_stem(source_path.stem)}_{run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    steps = {key: run_dir / f"step_{index + 1:02d}_{name}" for index, (key, name) in enumerate(specs)}
    for directory in steps.values():
        directory.mkdir(parents=True, exist_ok=False)
    log = RunLog(run_dir / "progress_log.txt")
    log(f"Run directory: {run_dir}")

    B = collect_measurements(
        source_path, config, run_dir / "cache_measurements_do_not_edit", log,
        workers=workers, bin_factor=bin_factor, reuse_dirs=reuse_dirs,
        flatfield_output=run_dir / "cache_measurements_do_not_edit" / "flatfield_build",
    )
    z0 = int(z_first_one_based) - 1  # planes below this (e.g. inside the glass) are excluded from the analysis
    if z0 > 0:
        for key in ("profiles", "ratios", "bins"):
            B[key] = {base: values[:, :, z0:] for base, values in B[key].items()}
        if B.get("standard_blocks"):
            B["standard_blocks"] = {base: values[:, :, z0:] for base, values in B["standard_blocks"].items()}
        B["z_count"] -= z0
        log(f"Excluding Z1-Z{z0}: analysis uses Z{z0 + 1}-Z{z0 + B['z_count']}")
        for mode in modes:
            mode["planes"] = [plane - z0 for plane in mode["planes"]]
            if min(mode["planes"]) < 0:
                raise ValueError(f"{mode['key']} uses planes below the first analysed plane Z{z0 + 1}.")
    labels: list[str] = B["labels"]
    timepoints: tuple[int, ...] = B["timepoints"]
    hour_of: dict[int, float] = dict(zip(timepoints, B["time_h"]))
    T, C, Z = len(timepoints), len(labels), B["z_count"]
    for mode in modes:
        if max(mode["planes"]) >= Z or min(mode["planes"]) < 0:
            raise ValueError(f"{mode['key']} is outside the {Z} acquired Z planes.")
    K = len(B["pair_geometry"])
    width = B["width"]
    x_offsets, positions = B["x_offsets"], B["positions"]
    profiles, ratios, bins = B["profiles"], B["ratios"], B["bins"]
    channel_cfg = {channel.label: channel for channel in config.channels}
    all_t, all_k = list(range(T)), list(range(K))

    def hours(timepoint: int) -> str:
        return f"{_hl(hour_of[timepoint])}h"

    def name(timepoint: int, *parts: str) -> str:
        return "_".join([experiment_label, f"t{timepoint:03d}", hours(timepoint), *parts])

    # ---- overlap-fitted corrections ------------------------------------------
    log("Fitting overlap corrections (pooled over all timepoints and pairs)")

    def samples_for(base: str, c: int, t_list: list[int], k_list: list[int]):
        out = []
        for z in range(Z):
            parts = [pair_samples(ratios[base][t, c, z], x_offsets, k_list) for t in t_list]
            out.append(tuple(np.concatenate([part[i] for part in parts]) for i in range(3)))
        return out

    def log_profiles_from(fit: dict[str, np.ndarray]) -> np.ndarray:
        return np.stack([log_illumination(width, fit["linear"][z], fit["quadratic"][z]) for z in range(Z)])

    penalty = float(config.all_z_coefficient_smoothing_penalty)
    fits: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    log_p: dict[tuple[str, int], np.ndarray] = {}
    per_t_fits: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
    coefficient_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for method in fitted:
        base = METHOD_BASE[method]
        for c, label in enumerate(labels):
            fit = fit_z_series(samples_for(base, c, all_t, all_k), width, penalty)
            fits[(method, c)] = fit
            log_p[(method, c)] = log_profiles_from(fit)
            for z in range(Z):
                coefficient_rows.append({
                    "method": method, "channel": label, "z_index_one_based": z + 1 + z0,
                    "raw_linear": fit["raw_linear"][z], "raw_quadratic": fit["raw_quadratic"][z],
                    "smoothed_linear": fit["linear"][z], "smoothed_quadratic": fit["quadratic"][z],
                    "quality_weight": fit["quality_weight"][z], "overlap_sample_count": fit["sample_count"][z],
                    "correction_factor_min": float(np.exp(-log_p[(method, c)][z]).min()),
                    "correction_factor_max": float(np.exp(-log_p[(method, c)][z]).max()),
                })
            for ti, timepoint in enumerate(timepoints):
                try:
                    fit_t = fit_z_series(samples_for(base, c, [ti], all_k), width, penalty)
                except ValueError:
                    continue
                per_t_fits[(method, c, ti)] = fit_t
                deviation = np.abs(log_profiles_from(fit_t) - log_p[(method, c)]).max(axis=1)
                stability_rows.append({
                    "method": method, "channel": label, "timepoint": timepoint, "time_h": hour_of[timepoint],
                    "median_over_Z_of_max_difference_percent": 100 * (np.exp(float(np.median(deviation))) - 1),
                    "max_over_Z_of_max_difference_percent": 100 * (np.exp(float(np.max(deviation))) - 1),
                })

    def tile_profiles(method: str, c: int, ti: int, planes: list[int], lp: np.ndarray | None = None) -> np.ndarray:
        """Corrected tile profiles (P, W) averaged over the given Z planes."""
        lp = log_p.get((method, c)) if lp is None and method in fitted else lp
        stack = profiles[METHOD_BASE[method]][ti, c][planes]
        if lp is not None:
            stack = stack / np.exp(lp[planes])[:, np.newaxis, :]
        return stack.mean(axis=0)

    # ---- seam validation -----------------------------------------------------
    log("Seam validation (in-sample and held-out)")
    # Timepoints are split into at most HELD_OUT_TIME_GROUPS interleaved groups; each
    # (group, tile pair) is refitted without that group and pair and tested on it.
    held_out: dict[tuple[str, int, int, int], np.ndarray] = {}
    groups = [all_t[start::min(HELD_OUT_TIME_GROUPS, T)] for start in range(min(HELD_OUT_TIME_GROUPS, T))]
    for method in fitted:
        base = METHOD_BASE[method]
        for c in range(C):
            for group in groups:
                training_t = [t for t in all_t if t not in group] or all_t
                for k in all_k:
                    fit = fit_z_series(samples_for(base, c, training_t, [p for p in all_k if p != k]), width, penalty)
                    log_fit = log_profiles_from(fit)
                    for ti in group:
                        held_out[(method, c, ti, k)] = log_fit
    minimum_profile = float(config.overlap_min_signal)
    main_planes = main_mode["planes"]
    seam_rows: list[dict[str, Any]] = []
    profile_seam_rows: list[dict[str, Any]] = []
    for method in methods:
        base = METHOD_BASE[method]
        for c, label in enumerate(labels):
            for ti, timepoint in enumerate(timepoints):
                inside_profiles = tile_profiles(method, c, ti, main_planes)
                for k in all_k:
                    pair = f"{positions[k].name}-{positions[k + 1].name}"
                    for z in range(Z):
                        ratio = ratios[base][ti, c, z, k]
                        if method in fitted:
                            inside = correct_ratios(ratio, x_offsets[k], log_p[(method, c)][z])
                            outside = correct_ratios(ratio, x_offsets[k], held_out[(method, c, ti, k)][z])
                        else:
                            inside = outside = ratio
                        seam_rows.append({
                            "method": method, "channel": label, "timepoint": timepoint, "pair": pair,
                            "z_index_one_based": z + 1 + z0,
                            "in_sample_median_abs_log_ratio": median_abs(inside),
                            "held_out_median_abs_log_ratio": median_abs(outside),
                        })
                    held_profiles = (tile_profiles(method, c, ti, main_planes, held_out[(method, c, ti, k)])
                                     if method in fitted else inside_profiles)
                    profile_seam_rows.append({
                        "method": method, "channel": label, "timepoint": timepoint, "pair": pair,
                        "in_sample_mismatch": profile_pair_mismatch(inside_profiles[k], inside_profiles[k + 1], x_offsets[k], minimum_profile),
                        "held_out_mismatch": profile_pair_mismatch(held_profiles[k], held_profiles[k + 1], x_offsets[k], minimum_profile),
                    })
    seam_summary: list[dict[str, Any]] = []
    for method in methods:
        for label in labels:
            plane = [r for r in seam_rows if r["method"] == method and r["channel"] == label]
            plane_main = [r for r in plane if r["z_index_one_based"] - 1 - z0 in main_planes]
            prof = [r for r in profile_seam_rows if r["method"] == method and r["channel"] == label]

            def pct(values: list[float]) -> float:
                return 100 * (np.exp(float(np.nanmedian(values))) - 1)

            seam_summary.append({
                "method": method, "method_name": METHOD_LABELS[method], "channel": label,
                "every_Z_fitted_on_same_data_percent": pct([r["in_sample_median_abs_log_ratio"] for r in plane]),
                "every_Z_held_out_percent": pct([r["held_out_median_abs_log_ratio"] for r in plane]),
                f"{main_mode['file_tag']}_held_out_percent": pct([r["held_out_median_abs_log_ratio"] for r in plane_main]),
                f"{main_mode['file_tag']}_profile_held_out_percent": pct([r["held_out_mismatch"] for r in prof]),
                f"{main_mode['file_tag']}_profile_held_out_90th_percentile_percent":
                    100 * (np.exp(float(np.nanpercentile([r["held_out_mismatch"] for r in prof], 90))) - 1),
            })
    seams = steps["seams"]
    write_rows_csv(seams / "seam_mismatch_summary_percent.csv", seam_summary)
    write_rows_csv(seams / "seam_mismatch_every_plane_and_tile_pair.csv", seam_rows)
    write_rows_csv(seams / f"seam_mismatch_{main_mode['file_tag']}_profiles_every_tile_pair.csv", profile_seam_rows)
    write_rows_csv(seams / "overlap_correction_coefficients_by_Z.csv", coefficient_rows)
    write_rows_csv(seams / "overlap_correction_single-timepoint_vs_pooled_difference.csv", stability_rows)
    _plot_seam_summary(seams / f"{experiment_label}_seam_mismatch_summary_by_method.png", seam_summary, labels, methods,
                       experiment_label, main_mode["file_tag"])
    _plot_seam_by_z(seams / f"{experiment_label}_held-out_seam_mismatch_by_Z.png", seam_rows, labels, methods, Z, experiment_label, z0)
    _plot_correction_profiles(seams / f"{experiment_label}_overlap_illumination_profiles_per_timepoint_vs_pooled.png",
                              fits, per_t_fits, labels, timepoints, width, Z, fitted, hour_of)

    # ---- profiles and binned mosaics for every depth mode ------------------------
    log("Profiles and binned mosaics for every depth mode")
    signal: dict[tuple[str, str, int, int], np.ndarray] = {}
    x_mm: np.ndarray | None = None
    for mode in modes:
        for method in methods:
            for c in range(C):
                for ti in all_t:
                    x, values, _ = feather_profiles(B["x_axes"], list(tile_profiles(method, c, ti, mode["planes"])))
                    signal[(mode["key"], method, c, ti)] = values
                    x_mm = x - x[0]
    assert x_mm is not None
    channel_max = {(mode["key"], m, c): max(float(np.nanmax(signal[(mode["key"], m, c, ti)])) for ti in all_t)
                   for mode in modes for m in methods for c in range(C)}
    factor = B["bin_factor"]
    binned_x = [value / factor for value in B["x_starts"]]
    binned_y = [value / factor for value in B["y_starts"]]
    bin_centers = (np.arange(B["binned_shape"][1]) * factor + (factor - 1) / 2.0).astype(int)

    def mosaic(mode: dict[str, Any], method: str, c: int, ti: int) -> np.ndarray:
        planes = mode["planes"]
        stack = bins[METHOD_BASE[method]][ti, c][planes].astype(np.float64)
        if method in fitted:
            stack = stack / np.exp(log_p[(method, c)][planes][:, bin_centers])[:, np.newaxis, np.newaxis, :]
        image, _, _ = feather_tiles_2d_union(list(stack.mean(axis=0).astype(np.float32)), binned_x, binned_y)
        return image

    main_mosaics = {(m, c, ti): mosaic(main_mode, m, c, ti) for m in methods for c in range(C) for ti in all_t}
    mosaic_column_mm = (np.arange(main_mosaics[(primary, 0, 0)].shape[1]) * factor + (factor - 1) / 2.0) * B["pixel_um"] / 1000.0
    _save_mosaic_previews(seams / f"{main_mode['file_tag']}_mosaic_previews_each_method_same_display_range",
                          seams / f"{main_mode['file_tag']}_mosaics_8x-binned_float32_TIFF",
                          main_mosaics, methods, labels, timepoints, channel_cfg, config, experiment_label, hour_of, main_mode["file_tag"])

    # ---- bridge from DIC ---------------------------------------------------------
    bridge: dict[str, float] | None = None
    dic_strip: dict[str, Any] | None = None
    if "bridge" in steps:
        log("Locating the bridge in the large DIC image")
        crop, dic_meta = _large_dic_crop(Path(config.large_dic_path), [(p.x_um, p.y_um) for p in positions],
                                         (B["height"], width), B["pixel_um"])
        dic_pixel_mm = float(dic_meta["large_pixel_size_um"]) / 1000.0
        crop, dic_offset_mm = _align_dic_crop_to_profile(crop, dic_meta, width * B["pixel_um"] / 1000.0, dic_pixel_mm)
        bridge = detect_bridge_edges(np.median(crop, axis=0), dic_pixel_mm)
        for key in ("left_foot_mm", "left_half_max_mm", "right_half_max_mm", "right_foot_mm", "center_mm"):
            bridge[key] += dic_offset_mm
        bridge["dic_covers_profile_mm"] = [dic_offset_mm, dic_offset_mm + crop.shape[1] * dic_pixel_mm]
        values = crop.astype(np.float32)
        low, high = (float(v) for v in np.percentile(values, [0.5, 99.5]))
        stride = max(1, crop.shape[1] // 1600)
        dic_strip = {
            "image": np.clip((values[::stride, ::stride] - low) / (high - low), 0, 1),
            "extent": [dic_offset_mm, dic_offset_mm + crop.shape[1] * dic_pixel_mm, 0.0, crop.shape[0] * dic_pixel_mm],
            "profile_range_mm": (float(x_mm[0]), float(x_mm[-1])),
        }
        mosaic_width_mm = main_mosaics[(primary, 0, 0)].shape[1] * factor * B["pixel_um"] / 1000.0
        _bridge_outputs(steps["bridge"], crop, values, bridge, dic_meta, (low, high), config, main_mosaics, primary,
                        channel_cfg, labels, timepoints, experiment_label, hour_of, main_mode["file_tag"],
                        dic_offset_mm, mosaic_width_mm)

    # ---- stain mask --------------------------------------------------------------
    excluded = np.zeros(x_mm.shape, dtype=bool)
    stain_intervals: list[tuple[float, float]] = []
    stain_details: dict[str, Any] = {"enabled": False}
    if "stain" in steps:
        log("Detecting the stain with a fixed, objective rule")
        detection_mode = modes[-1]
        images = [mosaic(detection_mode, primary, c, ti) for ti in all_t for c in range(C)]
        stain_mask, vote, stain_details = detect_stain_mask(images, B["pixel_um"] * factor)
        valid = np.isfinite(images[0])
        excluded, column_fraction = excluded_profile_positions(stain_mask, valid, mosaic_column_mm, x_mm)
        stain_intervals = intervals_from_mask(x_mm, excluded)
        stain_details.update({
            "enabled": True, "rule": STAIN_RULE,
            "excluded_intervals_mm": stain_intervals,
            "excluded_fraction_of_profile": float(excluded.mean()),
            "images": f"{primary}-corrected mosaics ({detection_mode['title']}), both channels, all {T} timepoints",
        })
        _stain_outputs(steps["stain"], stain_mask, vote, column_fraction, mosaic_column_mm, stain_details,
                       images[(T - 1) * C], images[(T // 2) * C + C - 1], B["pixel_um"] * factor,
                       experiment_label, labels, timepoints[T - 1], timepoints[T // 2], hour_of)

    # ---- calibration and conversions -----------------------------------------------
    conversion: dict[tuple[str, str, int], dict[str, Any]] = {}
    calibration_models: dict[str, dict[str, Any]] = {}
    if has_cal:
        log("Calibration curves for every depth mode and method")
        conversion, calibration_models = _calibration_step(
            steps["calibration"], B, modes, methods, fitted, log_p, signal, labels, all_t, experiment_label,
            int(config.calibration_block_size_px))

    # ---- tanh fits -----------------------------------------------------------------
    log("Tanh fits")
    reference = float(config.calibration_reference_concentration_ug_ml)
    pixel_mm = float(x_mm[1] - x_mm[0])
    fit_rows: dict[str, list[dict[str, Any]]] = {mode["key"]: [] for mode in modes}
    unmasked_rows: list[dict[str, Any]] = []

    def fit_record(mode: dict[str, Any], method: str, c: int, ti: int, mask: np.ndarray | None) -> dict[str, Any]:
        key = mode["key"]
        maximum = channel_max[(key, method, c)]
        y = signal[(key, method, c, ti)] / maximum
        if mask is not None:
            y = y.copy()
            y[mask] = np.nan
        fit = fit_tanh_profile(x_mm, y, max_points=config.tanh_max_points, pixel_size_mm=pixel_mm)
        row: dict[str, Any] = {
            "depth": key, "method": method, "method_name": METHOD_LABELS[method], "channel": labels[c],
            "timepoint": timepoints[ti], "time_h": hour_of[timepoints[ti]],
            "stain_mask_applied": mask is not None and bool(mask.any()),
            "fit_valid": fit["fit_valid"], "qc_flags": ";".join(fit["qc_flags"]),
            "r_squared": fit["r_squared"], "midpoint_mm": fit["midpoint_mm"], "width_mm": fit["width_mm"],
            "normalized_left_plateau": fit["left_plateau"], "normalized_right_plateau": fit["right_plateau"],
            "normalized_signed_slope_per_mm": fit["signed_slope_per_mm"],
            "normalized_absolute_slope_per_mm": fit["absolute_slope_per_mm"],
            "intensity_unit": mode["intensity_unit"], "channel_maximum_intensity": maximum,
            "intensity_left_plateau": fit["left_plateau"] * maximum,
            "intensity_right_plateau": fit["right_plateau"] * maximum,
            "intensity_signed_slope_per_mm": fit["signed_slope_per_mm"] * maximum,
        }
        if bridge:
            mid = fit["midpoint_mm"]
            row["midpoint_relative_to_bridge"] = (
                "left of bridge" if mid < bridge["left_foot_mm"] else "right of bridge" if mid > bridge["right_foot_mm"] else "on bridge")
            row["midpoint_minus_bridge_center_mm"] = mid - bridge["center_mm"]
        if has_cal:
            lines = conversion[(key, method, c)]
            m, b = lines["primary"]["signal_per_ug_ml"], lines["primary"]["signal_intercept"]
            row.update({
                "concentration_left_plateau_ug_ml": (fit["left_plateau"] * maximum - b) / m,
                "concentration_right_plateau_ug_ml": (fit["right_plateau"] * maximum - b) / m,
                "concentration_signed_slope_ug_ml_per_mm": fit["signed_slope_per_mm"] * maximum / m,
                "concentration_absolute_slope_ug_ml_per_mm": fit["absolute_slope_per_mm"] * maximum / m,
                "slope_if_line_forced_through_zero_ug_ml_per_mm": fit["signed_slope_per_mm"] * maximum / lines["origin"]["signal_per_ug_ml"],
            })
            model = calibration_models[key][method][labels[c]]
            if model["strictly_monotonic"]:
                piecewise, _ = empirical_concentration(signal[(key, method, c, ti)], np.asarray(model["signals"]),
                                                       np.asarray(model["concentrations_ug_ml"]))
                yp = piecewise / reference
                if mask is not None:
                    yp = yp.copy()
                    yp[mask] = np.nan
                try:
                    fit_p = fit_tanh_profile(x_mm, yp, max_points=config.tanh_max_points, pixel_size_mm=pixel_mm)
                    row["slope_if_piecewise_conversion_ug_ml_per_mm"] = fit_p["signed_slope_per_mm"] * reference
                except ValueError:
                    row["slope_if_piecewise_conversion_ug_ml_per_mm"] = float("nan")
        return row

    mask_or_none = excluded if excluded.any() else None
    for mode in modes:
        for method in (methods if mode is main_mode else (primary,)):
            for c in range(C):
                for ti in all_t:
                    fit_rows[mode["key"]].append(fit_record(mode, method, c, ti, mask_or_none))
                    if mode is main_mode and method == primary and mask_or_none is not None:
                        unmasked_rows.append(fit_record(mode, method, c, ti, None))

    def series(mode: dict[str, Any], method: str, c: int, ti: int, rep: str) -> np.ndarray:
        values = signal[(mode["key"], method, c, ti)]
        if rep == "normalized":
            return values / channel_max[(mode["key"], method, c)]
        if rep == "concentration":
            line = conversion[(mode["key"], method, c)]["primary"]
            return (values - line["signal_intercept"]) / line["signal_per_ug_ml"]
        return values

    def curve(row: dict[str, Any], rep: str) -> tuple[np.ndarray, str]:
        if rep == "normalized":
            left, right, slope = row["normalized_left_plateau"], row["normalized_right_plateau"], row["normalized_signed_slope_per_mm"]
        elif rep == "intensity":
            left, right, slope = row["intensity_left_plateau"], row["intensity_right_plateau"], row["intensity_signed_slope_per_mm"]
        else:
            left, right, slope = row["concentration_left_plateau_ug_ml"], row["concentration_right_plateau_ug_ml"], row["concentration_signed_slope_ug_ml_per_mm"]
        text = f"{slope:.3g}" if abs(slope) < 1000 else f"{slope:.4g}"
        return tanh_profile(x_mm, left, right, row["midpoint_mm"], row["width_mm"]), text

    def find(rows: list[dict[str, Any]], method: str, c: int, ti: int) -> dict[str, Any]:
        return next(r for r in rows if r["method"] == method and r["channel"] == labels[c] and r["timepoint"] == timepoints[ti])

    # ---- step: methods compared (main depth) -------------------------------------------
    log("Plots: correction methods compared")
    methods_dir = steps["methods"]
    main_rows = fit_rows[main_mode["key"]]
    for rep in reps:
        info = representation_info(rep, main_mode)
        for ti, timepoint in enumerate(timepoints):
            fig, top, axes = _figure(dic_strip, rows=C, height=4.4)
            title = (f"{experiment_label} · t{timepoint:03d} ({hours(timepoint)}) · {info['word']} for every correction method\n"
                     f"(solid = data, dashed = tanh fit)")
            (top if top is not None else axes[0]).set_title(title)
            if top is not None:
                _dic_strip(top, dic_strip, bridge)
            for c, axis in enumerate(axes):
                _shade(axis, bridge, stain_intervals)
                for method in methods:
                    row = find(main_rows, method, c, ti)
                    fitted_curve, text = curve(row, rep)
                    axis.plot(x_mm, series(main_mode, method, c, ti, rep), color=METHOD_COLORS[method], lw=1.0,
                              label=f"{METHOD_LABELS[method]}: slope {text} {info['slope_unit']}, R²={row['r_squared']:.3f}")
                    axis.plot(x_mm, fitted_curve, color=METHOD_COLORS[method], lw=1.0, ls="--")
                axis.set(xlabel=X_LABEL, ylabel=f"{labels[c]} {info['axis']}")
                axis.grid(alpha=0.2)
                axis.legend(fontsize=7.5, loc="best")
            _save(fig, methods_dir / info["folder"] / f"{name(timepoint, main_mode['file_tag'], rep, 'all-correction-methods', 'tanh-fits')}.png")
    write_rows_csv(methods_dir / "tanh_fit_results_every_method_and_timepoint.csv", main_rows)
    _plot_parameters_over_time(methods_dir / f"{experiment_label}_{main_mode['file_tag']}_tanh_parameters_over_time_every_method.png",
                               main_rows, labels, methods, reps, bridge, experiment_label, main_mode["title"])

    # ---- steps: primary method, one folder per depth mode ---------------------------------
    for index, mode in enumerate(modes):
        folder = steps[f"mode{index}"]
        rows = [r for r in fit_rows[mode["key"]] if r["method"] == primary]
        log(f"Plots: {primary}-corrected {mode['title']} per timepoint")
        for rep in reps:
            info = representation_info(rep, mode)
            for ti, timepoint in enumerate(timepoints):
                fig, top, axes = _figure(dic_strip, rows=1, height=5.6)
                axis = axes[0]
                (top if top is not None else axis).set_title(
                    f"{experiment_label} · t{timepoint:03d} ({hours(timepoint)}) · {METHOD_LABELS[primary]}\n{info['word']} with tanh fits")
                if top is not None:
                    _dic_strip(top, dic_strip, bridge)
                _shade(axis, bridge, stain_intervals)
                for c, label in enumerate(labels):
                    row = find(rows, primary, c, ti)
                    fitted_curve, text = curve(row, rep)
                    flag = "" if row["fit_valid"] else f"  ⚠ {row['qc_flags']}"
                    axis.plot(x_mm, series(mode, primary, c, ti, rep), color=channel_cfg[label].plot_color, lw=1.1, label=f"{label} data")
                    axis.plot(x_mm, fitted_curve, color=channel_cfg[label].plot_color, lw=2.0, ls="--",
                              label=f"{label} tanh fit: slope {text} {info['slope_unit']}, midpoint {row['midpoint_mm']:.2f} mm, R²={row['r_squared']:.3f}{flag}")
                axis.set(xlabel=X_LABEL, ylabel=info["axis"])
                axis.set_ylim(bottom=min(0.0, axis.get_ylim()[0]))
                axis.grid(alpha=0.2)
                axis.legend(fontsize=8, loc="best")
                _save(fig, folder / info["folder"] / f"{name(timepoint, primary + '-corrected', mode['file_tag'], rep, 'GFP-Cy5_tanh-fit')}.png")
            _plot_overlay_all_timepoints(
                folder / info["folder"] / f"{experiment_label}_all_timepoints_{primary}-corrected_{mode['file_tag']}_{rep}.png",
                x_mm, {(c, ti): series(mode, primary, c, ti, rep) for c in range(C) for ti in all_t},
                labels, timepoints, hour_of, info, bridge, stain_intervals, dic_strip,
                f"{experiment_label} · all timepoints · {METHOD_LABELS[primary]} · {info['word']}")
            _write_profile_table(
                folder / info["folder"] / f"{experiment_label}_profile_data_{primary}-corrected_{mode['file_tag']}_{rep}_all_timepoints.csv",
                x_mm, excluded, bridge, {f"{labels[c]}_t{t:03d}_{_hl(hour_of[t])}h": series(mode, primary, c, ti, rep)
                                         for c in range(C) for ti, t in enumerate(timepoints)}, info["axis"])
        write_rows_csv(folder / f"tanh_fit_results_{primary}_{mode['file_tag']}{'_stain-masked' if mask_or_none is not None else ''}.csv", rows)
        _plot_parameters_over_time(folder / f"{experiment_label}_{primary}_{mode['file_tag']}_tanh_slopes_and_midpoints_over_time.png",
                                   rows, labels, (primary,), reps, bridge, experiment_label, mode["title"])
        if index > 0:
            _plot_mode_comparison(folder / f"{experiment_label}_{primary}_{main_mode['file_tag']}_vs_{mode['file_tag']}_slopes_over_time.png",
                                  [r for r in main_rows if r["method"] == primary], rows, main_mode, mode, labels, has_cal, experiment_label)

    main_primary_rows = [r for r in main_rows if r["method"] == primary]
    main_folder = steps["mode0"]
    if unmasked_rows:
        folder = main_folder / "4_fits_without_stain_mask_for_comparison"
        write_rows_csv(folder / f"tanh_fit_results_{primary}_{main_mode['file_tag']}_without_stain_mask.csv", unmasked_rows)
        _plot_mask_effect(folder / f"{experiment_label}_effect_of_stain_mask_on_{primary}_{main_mode['file_tag']}_fits.png",
                          main_primary_rows, unmasked_rows, labels, has_cal, experiment_label)

    # alternative analysis for non-sigmoidal profiles (main depth)
    flagged = [r for r in main_primary_rows if not r["fit_valid"]] if flagged_fit_analysis else []
    alt_dir = main_folder / "5_alternative_analysis_for_profiles_without_sigmoid_shape"
    if flagged_fit_analysis:
        alt_dir.mkdir(parents=True, exist_ok=True)
    alt_rows: list[dict[str, Any]] = []
    first_rep = reps[0]
    for row in flagged:
        c, ti = labels.index(row["channel"]), timepoints.index(row["timepoint"])
        maximum = channel_max[(main_mode["key"], primary, c)]
        metrics = alternative_gradient_metrics(x_mm, signal[(main_mode["key"], primary, c, ti)] / maximum, excluded)
        record = {"channel": row["channel"], "timepoint": row["timepoint"], "time_h": row["time_h"],
                  "tanh_qc_flags": row["qc_flags"], **metrics,
                  "band_linear_slope_intensity_per_mm": metrics["band_linear_slope_per_mm"] * maximum,
                  "steepest_local_slope_intensity_per_mm": metrics["steepest_local_slope_per_mm"] * maximum}
        y_relaxed = signal[(main_mode["key"], primary, c, ti)] / maximum
        if mask_or_none is not None:
            y_relaxed = y_relaxed.copy()
            y_relaxed[mask_or_none] = np.nan
        relaxed = fit_tanh_profile(x_mm, y_relaxed, max_points=config.tanh_max_points, pixel_size_mm=pixel_mm,
                                   plateau_bounds=RELAXED_PLATEAU_BOUNDS)
        record.update({
            "tanh_slope_normalized_per_mm": row["normalized_signed_slope_per_mm"],
            "tanh_slope_with_relaxed_plateau_limits_normalized_per_mm": relaxed["signed_slope_per_mm"],
            "slope_change_with_relaxed_limits_percent": 100 * (relaxed["signed_slope_per_mm"] / row["normalized_signed_slope_per_mm"] - 1),
            "tanh_midpoint_mm": row["midpoint_mm"],
            "tanh_midpoint_with_relaxed_plateau_limits_mm": relaxed["midpoint_mm"],
            "relaxed_fit_flags": ";".join(relaxed["qc_flags"]),
        })
        line = conversion[(main_mode["key"], primary, c)]["primary"] if has_cal else None
        if line is not None:
            record["band_linear_slope_ug_ml_per_mm"] = metrics["band_linear_slope_per_mm"] * maximum / line["signal_per_ug_ml"]
            record["steepest_local_slope_ug_ml_per_mm"] = metrics["steepest_local_slope_per_mm"] * maximum / line["signal_per_ug_ml"]
        alt_rows.append(record)
        _plot_alternative(alt_dir / f"{name(row['timepoint'], row['channel'], primary + '-corrected', main_mode['file_tag'], 'alternative_gradient_measures')}.png",
                          x_mm, series(main_mode, primary, c, ti, first_rep), record, representation_info(first_rep, main_mode), first_rep,
                          row, bridge, stain_intervals, dic_strip,
                          f"{experiment_label} · t{row['timepoint']:03d} ({hours(row['timepoint'])}) · {row['channel']} · {main_mode['title']}\n"
                          f"tanh fit flagged ({row['qc_flags']}): model-free gradient measures",
                          channel_cfg[row["channel"]].plot_color, line, maximum)
    if alt_rows:
        write_rows_csv(alt_dir / "alternative_gradient_measures_for_flagged_tanh_fits.csv", alt_rows)
    if flagged_fit_analysis:
        (alt_dir / "README.txt").write_text(
        "These profiles received a tanh fit like every other timepoint (see the main folder), but the fit was flagged\n"
        "(for example a plateau or the midpoint at the edge of the field). Here they are additionally described without\n"
        "assuming a sigmoid: (1) a straight-line slope over the transition band where the smoothed profile is between\n"
        "20% and 80% of its range, and (2) the steepest local slope of the profile smoothed over 0.1 mm, with its position.\n"
        "Stain-excluded positions are not used. These results are kept separate and are not mixed into the main tables.\n"
        "\n"
        "Sensitivity check: the main tanh fit limits both plateaus to -0.1...1.1 of the channel maximum. Most flags mean a\n"
        "plateau reached that limit because the true plateau lies outside the imaged field. Each flagged profile is refitted\n"
        f"with relaxed limits ({RELAXED_PLATEAU_BOUNDS[0]:g}...{RELAXED_PLATEAU_BOUNDS[1]:g}); the CSV lists how much the slope and\n"
        "midpoint change. If the slope barely changes, the slope is trustworthy; a midpoint that moves a lot is not determined\n"
        "by the data and should not be interpreted.\n"
        f"Flagged fits: {len(flagged)}\n", encoding="utf-8")

    if diffusion_analysis:
        log("Front displacement, speed, and D(t) from the tanh midpoints")
        origin = min(float(axis[0]) for axis in B["x_axes"])
        centers = np.array([(float(axis[0]) + float(axis[-1])) / 2.0 - origin for axis in B["x_axes"]])
        main_z = int(config.zero_based_z)
        with ND2Source(source_path, config) as time_source:
            cache_times: dict[tuple[str, int, int], float] = {}

            def tile_hours(label: str, timepoint: int, midpoint_mm: float) -> float:
                position = int(np.argmin(np.abs(centers - midpoint_mm)))
                key = (label, timepoint, position)
                if key not in cache_times:
                    cache_times[key] = time_source.frame_hours(timepoint, positions[position].index, main_z, label)
                return cache_times[key]

            _diffusion_step(steps["diffusion"], main_primary_rows, bridge, labels, experiment_label, main_mode["title"],
                            tile_hours, expected_d)

    # ---- Z selection ----------------------------------------------------------------------
    log("Z selection: tanh fit for every Z plane")
    z_times = [t for t in (z_analysis_timepoints or timepoints) if t in timepoints]
    _z_selection(steps["zsel"], z_times, timepoints, labels, Z, primary,
                 lambda method, c, ti, z: tile_profiles(method, c, ti, [z]), B["x_axes"], x_mm,
                 excluded, stain_intervals, bridge, dic_strip, config, channel_cfg, experiment_label, hour_of, log, z0)

    # ---- flat field step ---------------------------------------------------------------------
    if "flatfield" in steps:
        for label in labels:
            _save_flatfield_plot(steps["flatfield"] / f"{experiment_label}_{label}_flat-field_from_32ugmL_standard.png",
                                 B["flatfields"][label], channel_cfg[label],
                                 f"{label}: relative illumination measured from the 32 µg/mL standard (all Z planes)")
            tifffile.imwrite(steps["flatfield"] / f"{experiment_label}_{label}_flat-field_relative_illumination_float32.tif",
                             B["flatfields"][label])
        write_json(steps["flatfield"] / "background_hot-pixels_flat-field_diagnostics.json",
                   {"backgrounds": B["backgrounds"], "diagnostics": B["flatfield_diagnostics"], "standards_qc": B["standard_qc"]})

    # ---- metadata, summary, and folder guide ---------------------------------------------------
    metadata = {
        "experiment_label": experiment_label, "source": str(source_path.resolve()),
        "timepoints": list(timepoints), "time_h": [hour_of[t] for t in timepoints],
        "channels": labels, "nd2_channels": B["channel_names"], "z_plane_count": Z, "z_step_um": B["z_step_um"],
        "z_stack": B["z_info"], "z_first_analysed_one_based": z0 + 1,
        "glass_plane_one_based": glass_plane_one_based or int(B["z_info"]["home_index_zero_based"]) + 1,
        "flagged_fit_analysis": flagged_fit_analysis, "time_source": "frame timestamps (first frame of each timepoint), hours from the first timepoint",
        "pixel_size_um": B["pixel_um"], "positions_left_to_right": [p.name for p in positions],
        "depth_modes": modes, "main_depth": main_mode["key"],
        "methods": {m: METHOD_LABELS[m] for m in methods}, "primary_method": primary,
        "overlap_fit_policy": "one quadratic per channel and Z, pooled over all timepoints and tile pairs; July quality weights; smoothed across Z",
        "held_out_policy": f"timepoints split into {HELD_OUT_TIME_GROUPS} interleaved groups (each timepoint alone if fewer); each (group, tile pair) refit without that group and pair and tested on it",
        "tanh_fit_policy": "fit to normalized intensity; intensity and concentration fits are the same curve rescaled (concentration via the primary linear calibration)",
        "calibration_policy": "standards measured at the same Z planes as the experiment (same glass reference, 10 µm steps); per-plane block medians, or their mean over the robustness planes",
        "calibration_conversions": {f"{k}|{m}|{labels[c]}": v for (k, m, c), v in conversion.items()},
        "bridge_from_DIC": bridge, "stain_mask": stain_details,
        "bin_factor_for_mosaics": factor, "reused_measurement_caches": [str(p) for p in (reuse_dirs or [])],
    }
    write_json(run_dir / "run_metadata.json", metadata)
    _write_summary(run_dir / "SUMMARY_key_results.md", experiment_label, seam_summary, fit_rows, conversion, labels,
                   timepoints, hour_of, bridge, stain_details, primary, has_cal, modes)
    _write_guide(run_dir / "00_README_how_to_read_this_folder.md", specs, steps, experiment_label, primary, reps, has_cal, modes)
    log("Done")
    return run_dir


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------


def _calibration_step(folder, B, modes, methods, fitted, log_p, signal, labels, all_t, experiment_label, block_size):
    blocks_per_row = B["width"] // block_size
    block_center_x = ((np.arange(blocks_per_row) + 0.5) * block_size).astype(int)
    concentrations = [c for c, _ in B["standards"]]
    conversion: dict[tuple[str, str, int], dict[str, Any]] = {}
    models: dict[str, dict[str, dict[str, Any]]] = {}
    point_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    primary = "hybrid" if "hybrid" in methods else methods[-1]
    for mode in modes:
        key = mode["key"]
        models[key] = {}
        for method in methods:
            models[key][method] = {}
            for c, label in enumerate(labels):
                values = []
                for s, path in enumerate(B["standard_paths"]):
                    stack = B["standard_blocks"][METHOD_BASE[method]][s, c][mode["planes"]]  # (n, blocks)
                    if method in fitted:
                        per_plane = np.exp(log_p[(method, c)][mode["planes"]][:, block_center_x])  # (n, columns)
                        stack = stack / np.tile(per_plane, (1, stack.shape[1] // blocks_per_row))
                    values.append(stack.mean(axis=0))
                rows, model = calibration_model_from_block_values(concentrations, values)
                models[key][method][label] = model
                conc = np.asarray(model["concentrations_ug_ml"])
                sig = np.asarray(model["signals"])
                cutoff_signal = float(sig[np.isclose(conc, LINEAR_CALIBRATION_MAX_UG_ML)][0])
                data_max = max(float(np.nanmax(signal[(key, method, c, ti)])) for ti in all_t)
                applies = data_max < cutoff_signal
                limit = LINEAR_CALIBRATION_MAX_UG_ML if applies else float(conc.max())
                primary_line = linear_conversion(conc, sig, limit, through_origin=False)
                origin_line = linear_conversion(conc, sig, limit, through_origin=True)
                all_points = linear_conversion(conc, sig, float(conc.max()), through_origin=False)
                for line in (primary_line, origin_line, all_points):
                    line["signal_intercept"] = line.pop("signal_intercept_au_um")
                    line["residuals"] = line.pop("residuals_au_um")
                primary_line.update({
                    "rule": f"standards above {LINEAR_CALIBRATION_MAX_UG_ML:g} µg/mL excluded when all data are below the {LINEAR_CALIBRATION_MAX_UG_ML:g} µg/mL standard",
                    "rule_applied": bool(applies), "brightest_data_fraction_of_24_standard": data_max / cutoff_signal})
                conversion[(key, method, c)] = {"primary": primary_line, "origin": origin_line, "all_points": all_points,
                                                "data_max": data_max, "intensity_unit": mode["intensity_unit"]}
                for row in rows:
                    point_rows.append({"depth": key, "method": method, "channel": label, **row})
                for kind, line in (("PRIMARY line 0-24 with intercept", primary_line), ("line 0-24 forced through zero", origin_line),
                                   ("line through all 8 standards", all_points)):
                    table_rows.append({"depth": key, "method": method, "channel": label, "conversion": kind,
                                       "signal_per_ug_ml": line["signal_per_ug_ml"], "intercept": line["signal_intercept"],
                                       "intercept_as_ug_ml": line["intercept_as_concentration_ug_ml"], "r_squared": line["r_squared"],
                                       "rule_applied": line.get("rule_applied", ""),
                                       "standards_used_ug_ml": " ".join(f"{v:g}" for v in line["standards_used_ug_ml"])})
        fig, axes = plt.subplots(2, len(labels), figsize=(7.5 * len(labels), 9), constrained_layout=True,
                                 gridspec_kw={"height_ratios": [3, 1.4]}, squeeze=False)
        for c, label in enumerate(labels):
            model = models[key][primary][label]
            conc = np.asarray(model["concentrations_ug_ml"])
            sig = np.asarray(model["signals"])
            points = sorted((r for r in point_rows if r["depth"] == key and r["method"] == primary and r["channel"] == label),
                            key=lambda r: r["concentration_ug_ml"])
            err = np.vstack([[r["standard_signal"] - r["block_q25"] for r in points], [r["block_q75"] - r["standard_signal"] for r in points]])
            axis = axes[0, c]
            axis.errorbar(conc, sig, yerr=err, fmt="o", color="black", capsize=3, label="Standards (block median ± IQR)")
            grid = np.linspace(0, conc.max(), 200)
            lines = conversion[(key, primary, c)]
            axis.plot(grid, lines["primary"]["signal_intercept"] + lines["primary"]["signal_per_ug_ml"] * grid, color="tab:green", lw=2,
                      label=f"PRIMARY: line 0–24 µg/mL with intercept, R²={lines['primary']['r_squared']:.4f}")
            axis.plot(grid, lines["origin"]["signal_per_ug_ml"] * grid, color="tab:purple", ls="--",
                      label=f"Line 0–24 µg/mL forced through zero, R²={lines['origin']['r_squared']:.4f}")
            axis.plot(conc, sig, color="gray", lw=0.8, ls=":", label="Piecewise between standards")
            axis.axhline(lines["data_max"], color="tab:red", lw=0.8, ls="-.", label="Brightest experimental profile value")
            axis.set(title=f"{experiment_label} · {label} calibration · {mode['title']} · {METHOD_LABELS[primary]}",
                     xlabel="Dextran concentration [µg/mL]", ylabel=mode["intensity_axis"])
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
            residual_axis = axes[1, c]
            for kind, color, marker in (("primary", "tab:green", "o"), ("origin", "tab:purple", "s")):
                line = lines[kind]
                used = np.asarray(line["standards_used_ug_ml"])
                residual_axis.plot(used, 100 * np.asarray(line["residuals"]) / (line["signal_per_ug_ml"] * max(used.max(), 1)),
                                   marker, color=color, label="PRIMARY line" if kind == "primary" else "line through zero")
            residual_axis.axhline(0, color="black", lw=0.6)
            residual_axis.set(xlabel="Dextran concentration [µg/mL]", ylabel="Residual [% of 24 µg/mL signal]",
                              title=f"{label}: distance of each standard from the line")
            residual_axis.grid(alpha=0.2)
            residual_axis.legend(fontsize=8)
        _save(fig, folder / f"{experiment_label}_{mode['file_tag']}_calibration_curve_{primary}_with_conversion_choices.png")
    write_rows_csv(folder / "calibration_standard_measurements_every_depth_and_method.csv", point_rows)
    write_rows_csv(folder / "calibration_conversion_lines_every_depth_and_method.csv", table_rows)
    write_json(folder / "calibration_models_every_depth_and_method.json",
               {"models": models, "conversions": {f"{k}|{m}|{labels[c]}": v for (k, m, c), v in conversion.items()}})
    return conversion, models


def _bridge_outputs(folder, crop, values, bridge, dic_meta, auto_range, config, mosaics, primary, channel_cfg, labels,
                    timepoints, experiment_label, hour_of, depth_tag, offset_mm=0.0, profile_width_mm=None):
    tifffile.imwrite(folder / f"{experiment_label}_large_DIC_crop_matching_gradient_fields_raw_values.tif", crop)
    windows = {"auto": auto_range, "requested": (float(config.large_dic_display_low), float(config.large_dic_display_high))}
    for kind, (low, high) in windows.items():
        scaled = np.round(np.clip((values - low) / (high - low), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(scaled, mode="L").save(folder / f"{experiment_label}_large_DIC_crop_display_{low:.0f}-{high:.0f}_{kind}.png")
    pixel_mm = float(dic_meta["large_pixel_size_um"]) / 1000.0
    column = np.median(crop, axis=0)
    x = offset_mm + np.arange(column.size) * pixel_mm
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [1.2, 2]})
    low, high = auto_range
    stride = max(1, crop.shape[1] // 1600)
    axes[0].imshow(np.clip((values[::stride, ::stride] - low) / (high - low), 0, 1), cmap="gray",
                   extent=[x[0], x[-1], 0, crop.shape[0] * pixel_mm], aspect="auto")
    axes[0].set(ylabel="Y [mm]", title=f"{experiment_label} · large 10× DIC image cropped to the gradient fields "
                                       f"(display {low:.0f}–{high:.0f}); detected bridge edges")
    axes[1].plot(x, column, lw=0.5, color="gray", label="Median over Y of each DIC column")
    axes[1].plot(x, ndimage.gaussian_filter1d(column.astype(float), 15), color="black", lw=1.2, label="Smoothed")
    for axis in axes:
        axis.axvspan(bridge["left_foot_mm"], bridge["right_foot_mm"], color="#d9c27a", alpha=0.25, lw=0)
        for key, style in (("left_foot_mm", "-"), ("right_foot_mm", "-"), ("left_half_max_mm", ":"), ("right_half_max_mm", ":")):
            axis.axvline(bridge[key], color="#b8860b", ls=style, lw=1.2)
    axes[1].set(xlabel=X_LABEL, ylabel="DIC intensity [a.u.]",
                title=f"Bridge: {bridge['left_foot_mm']:.2f}–{bridge['right_foot_mm']:.2f} mm (outer edges, width "
                      f"{bridge['width_between_feet_mm']:.2f} mm); half-height edges {bridge['left_half_max_mm']:.2f}–"
                      f"{bridge['right_half_max_mm']:.2f} mm (width {bridge['width_at_half_max_mm']:.2f} mm)")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    _save(fig, folder / f"{experiment_label}_bridge_edges_detected_from_DIC_column_profile.png")
    middle = len(timepoints) // 2
    images = [np.nan_to_num(mosaics[(primary, c, middle)], nan=0.0) for c in range(len(labels))]
    ranges = [tuple(np.percentile(image[image > 0], [config.preview_low_percentile, config.preview_high_percentile])) for image in images]
    merge = Image.fromarray(merge_rgb(images, [channel_cfg[l].rgb for l in labels], ranges), mode="RGB")
    for kind, (low, high) in windows.items():
        scaled = np.round(np.clip((values - low) / (high - low), 0, 1) * 255).astype(np.uint8)
        total_mm = profile_width_mm or (offset_mm + crop.shape[1] * pixel_mm)
        scale = merge.width / total_mm
        dic_width = max(1, int(round(crop.shape[1] * pixel_mm * scale)))
        dic = Image.fromarray(scaled, mode="L").convert("RGB").resize((dic_width, merge.height), Image.Resampling.LANCZOS)
        composite = Image.new("RGB", (merge.width, merge.height * 2), "black")
        composite.paste(dic, (int(round(offset_mm * scale)), 0))
        composite.paste(merge, (0, merge.height))
        draw = ImageDraw.Draw(composite)
        for key in ("left_foot_mm", "right_foot_mm"):
            xpix = int(round(bridge[key] * scale))
            draw.line([(xpix, 0), (xpix, composite.height)], fill=(255, 200, 0), width=2)
        t = timepoints[middle]
        composite.save(folder / f"{experiment_label}_DIC_display_{low:.0f}-{high:.0f}_{kind}_above_t{t:03d}_{_hl(hour_of[t])}h_{primary}_{depth_tag}_merge_bridge-marked.png")
    write_json(folder / "bridge_location_mm.json", {**bridge, "position_axis": "same as the profile x axis: 0 mm = left edge of the leftmost gradient field",
                                                    "DIC_crop": dic_meta})


def _stain_outputs(folder, mask, vote, column_fraction, column_mm, details, image_a, image_b, pixel_um, experiment_label,
                   labels, t_a, t_b, hour_of):
    tifffile.imwrite(folder / f"{experiment_label}_stain_mask_8x-binned.tif", mask.astype(np.uint8))
    height_mm = mask.shape[0] * pixel_um / 1000.0
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [2, 2, 1.4]})
    for axis, image, label, t in ((axes[0], image_a, labels[0], t_a), (axes[1], image_b, labels[-1], t_b)):
        finite = image[np.isfinite(image)]
        low, high = np.percentile(finite, [1, 99.8])
        axis.imshow(np.nan_to_num(image), cmap="gray", vmin=low, vmax=high, extent=[0, column_mm[-1], 0, height_mm], aspect="auto")
        axis.contour(np.linspace(0, column_mm[-1], mask.shape[1]), np.linspace(height_mm, 0, mask.shape[0]), mask, [0.5], colors="red", linewidths=1)
        axis.set(ylabel="Y [mm]", title=f"{label}, t{t:03d} ({_hl(hour_of[t])} h): stain outline (red) found by the rule")
    axes[2].plot(column_mm, 100 * column_fraction, color="black")
    axes[2].axhline(100 * STAIN_RULE["column_excluded_if_stain_fraction_above"], color="red", ls="--",
                    label="Exclusion threshold: >10% of the column height is stain")
    for index, (start, stop) in enumerate(details["excluded_intervals_mm"]):
        axes[2].axvspan(start, stop, facecolor="none", edgecolor="#b22222", hatch="///", lw=0, label="Excluded from fits" if index == 0 else None)
    axes[2].set(xlabel=X_LABEL, ylabel="Stain fraction of column [%]",
                title=f"Profile positions excluded from tanh fits: {', '.join(f'{a:.2f}–{b:.2f} mm' for a, b in details['excluded_intervals_mm'])} "
                      f"({100 * details['excluded_fraction_of_profile']:.1f}% of the profile)")
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=8)
    _save(fig, folder / f"{experiment_label}_stain_mask_and_excluded_profile_positions.png")
    write_json(folder / "stain_mask_rule_and_excluded_positions.json", details)
    (folder / "README.txt").write_text(
        "How the stain is handled (fixed rule, decided before looking at fit results, identical for every timepoint and method):\n"
        "1. In each Z-integrated mosaic, a pixel is 'dark' if it is below 70% of the 90th-percentile brightness of its own column\n"
        "   (columns with too little signal are skipped).\n"
        "2. A pixel is 'stain' if it is dark in at least half of all evaluated images (both channels, all timepoints) - the stain\n"
        "   is fixed on the plate, while real gradient features move.\n"
        "3. Tiny specks (< 2000 µm²) are ignored.\n"
        "4. A profile position is excluded from the tanh fit if more than 10% of that column is stain (+25 µm margin).\n"
        "Nothing is removed from images or data: the stain stays visible, all data points are still plotted (hatched region),\n"
        "and fits WITHOUT the mask are saved next to the main results for comparison.\n", encoding="utf-8")


def _z_selection(folder, z_times, timepoints, labels, Z, primary, corrected_tile_profiles, x_axes, x_mm, excluded,
                 stain_intervals, bridge, dic_strip, config, channel_cfg, experiment_label, hour_of, log, z0=0):
    C = len(labels)
    signal_z: dict[tuple[int, int, int], np.ndarray] = {}
    for t in z_times:
        ti = timepoints.index(t)
        for c in range(C):
            for z in range(Z):
                _, values, _ = feather_profiles(x_axes, list(corrected_tile_profiles(primary, c, ti, z)))
                signal_z[(ti, c, z)] = values
    maximum = {c: max(float(np.nanmax(v)) for (ti, cc, z), v in signal_z.items() if cc == c) for c in range(C)}
    pixel_mm = float(x_mm[1] - x_mm[0])
    records: list[dict[str, Any]] = []
    for (ti, c, z), values in signal_z.items():
        y = values / maximum[c]
        y_fit = y.copy()
        y_fit[excluded] = np.nan
        fit = fit_tanh_profile(x_mm, y_fit, max_points=config.tanh_max_points, pixel_size_mm=pixel_mm)
        records.append({
            "timepoint": timepoints[ti], "time_h": hour_of[timepoints[ti]], "channel": labels[c],
            "z_index_one_based": z + 1 + z0, "median_intensity_au": float(np.nanmedian(values)),
            "normalization_maximum_au": maximum[c], "signed_slope_per_mm": fit["signed_slope_per_mm"],
            "absolute_slope_per_mm": fit["absolute_slope_per_mm"], "midpoint_mm": fit["midpoint_mm"],
            "width_mm": fit["width_mm"], "left_plateau": fit["left_plateau"], "right_plateau": fit["right_plateau"],
            "r_squared": fit["r_squared"], "fit_valid": fit["fit_valid"], "qc_flags": ";".join(fit["qc_flags"]),
        })
    write_rows_csv(folder / "tanh_fits_every_Z_plane.csv", records)
    metrics = [
        ("signed_slope_per_mm", "Signed midpoint slope [1/mm]", "coolwarm", "Signed midpoint slope of the tanh fit"),
        ("absolute_slope_per_mm", "Absolute midpoint slope [1/mm]", "viridis", "Absolute midpoint slope of the tanh fit"),
        ("midpoint_mm", "Tanh midpoint position [mm]", "viridis", "Midpoint position of the tanh fit"),
        ("width_mm", "Tanh width [mm]", "viridis", "Width of the tanh fit"),
        ("r_squared", "Fit quality R² [–]", "magma", "Fit quality (R²) of the tanh fit"),
        ("median_intensity_au", "Median profile intensity [a.u.]", "cividis", "Median intensity of the single-plane profile (brightness of each Z plane)"),
    ]
    heat_dir = folder / "1_heatmaps_time_by_Z"
    for c, label in enumerate(labels):
        for key, title, cmap, description in metrics:
            grid = np.full((len(z_times), Z), np.nan)
            invalid = np.zeros_like(grid, dtype=bool)
            for r in records:
                if r["channel"] == label:
                    row, col = z_times.index(r["timepoint"]), r["z_index_one_based"] - 1 - z0
                    grid[row, col] = r[key]
                    invalid[row, col] = not r["fit_valid"]
            fig, axis = plt.subplots(figsize=(max(11, 0.55 * Z), 0.8 * len(z_times) + 2.2), constrained_layout=True)
            if cmap == "coolwarm":
                bound = float(np.nanmax(np.abs(grid)))
                image = axis.imshow(grid, aspect="auto", origin="lower", cmap=cmap, vmin=-bound, vmax=bound)
            else:
                image = axis.imshow(grid, aspect="auto", origin="lower", cmap=cmap)
            for row in range(grid.shape[0]):
                for col in range(grid.shape[1]):
                    if np.isfinite(grid[row, col]):
                        text = f"{grid[row, col]:.2f}" if abs(grid[row, col]) < 100 else f"{grid[row, col]:.0f}"
                        axis.text(col, row, text + ("*" if invalid[row, col] else ""), ha="center", va="center", fontsize=6.5,
                                  color="white" if cmap in ("viridis", "magma", "cividis") else "black")
            axis.set_xticks(np.arange(Z), np.arange(1, Z + 1) + z0, fontsize=8)
            axis.set_yticks(np.arange(len(z_times)), [f"{_hl(hour_of[t])}" for t in z_times])
            axis.set(xlabel="Z plane [one-based index; 10 µm steps]", ylabel="Time [h]",
                     title=f"{experiment_label} · {label} · {METHOD_LABELS[primary]}\n{description}, for every Z plane and time (* = fit flagged)")
            fig.colorbar(image, ax=axis, label=title)
            _save(fig, heat_dir / f"{experiment_label}_{label}_{key}_time-by-Z_heatmap.png")
    by_z_dir = folder / "2_fit_parameters_versus_Z_per_timepoint"
    overlay_dir = folder / "3_profiles_of_every_Z_plane_overlaid"
    for t in z_times:
        ti = timepoints.index(t)
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for axis, (key, title, _, _) in zip(axes.flat, [metrics[0], metrics[2], metrics[4], metrics[5]]):
            for label in labels:
                rows = sorted((r for r in records if r["channel"] == label and r["timepoint"] == t), key=lambda r: r["z_index_one_based"])
                axis.plot([r["z_index_one_based"] for r in rows], [r[key] for r in rows], "o-", ms=3,
                          color=channel_cfg[label].plot_color, label=label)
            if key == "midpoint_mm" and bridge:
                axis.axhspan(bridge["left_foot_mm"], bridge["right_foot_mm"], color="#d9c27a", alpha=0.3, label="Bridge")
            axis.set(xlabel="Z plane [one-based index]", ylabel=title)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        fig.suptitle(f"{experiment_label} · t{t:03d} ({_hl(hour_of[t])} h) · {METHOD_LABELS[primary]} · tanh-fit parameters for every Z plane")
        _save(fig, by_z_dir / f"{experiment_label}_t{t:03d}_{_hl(hour_of[t])}h_tanh_parameters_versus_Z.png")
        fig, top, axes = _figure(dic_strip, rows=C, height=4.2)
        cmap = plt.get_cmap("viridis", Z)
        for c, axis in enumerate(axes):
            _shade(axis, bridge, stain_intervals)
            for z in range(Z):
                axis.plot(x_mm, signal_z[(ti, c, z)] / maximum[c], color=cmap(z), lw=0.8)
            axis.set(xlabel=X_LABEL, ylabel=f"{labels[c]} normalized intensity [–]")
            axis.grid(alpha=0.2)
            scalar = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1 + z0, Z + z0))
            fig.colorbar(scalar, ax=axis, label="Z plane [one-based index]")
        title = (f"{experiment_label} · t{t:03d} ({_hl(hour_of[t])} h) · {METHOD_LABELS[primary]} · profile of every Z plane\n"
                 f"(intensity normalized to the channel maximum over all Z planes and selected times)")
        (top if top is not None else axes[0]).set_title(title)
        if top is not None:
            _dic_strip(top, dic_strip, bridge)
        _save(fig, overlay_dir / f"{experiment_label}_t{t:03d}_{_hl(hour_of[t])}h_profiles_every_Z_overlaid.png")
        fit_dir = folder / "4_tanh_fit_plot_for_every_Z_plane" / f"t{t:03d}_{_hl(hour_of[t])}h"
        for z in range(Z):
            fig, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
            _shade(axis, bridge, stain_intervals)
            for c, label in enumerate(labels):
                r = next(r for r in records if r["channel"] == label and r["timepoint"] == t and r["z_index_one_based"] == z + 1 + z0)
                axis.plot(x_mm, signal_z[(ti, c, z)] / maximum[c], color=channel_cfg[label].plot_color, lw=1.0, label=f"{label} data")
                axis.plot(x_mm, tanh_profile(x_mm, r["left_plateau"], r["right_plateau"], r["midpoint_mm"], r["width_mm"]),
                          color=channel_cfg[label].plot_color, lw=1.8, ls="--",
                          label=f"{label} fit: slope {r['signed_slope_per_mm']:.3g} 1/mm, R²={r['r_squared']:.3f}" + ("" if r["fit_valid"] else " ⚠"))
            axis.set(xlabel=X_LABEL, ylabel="Normalized intensity [–]",
                     title=f"{experiment_label} · t{t:03d} ({_hl(hour_of[t])} h) · Z{z + 1 + z0:02d} · {METHOD_LABELS[primary]} · single-plane profiles with tanh fits\n"
                           f"(intensity normalized to the channel maximum over all Z planes and selected times)")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
            _save(fig, fit_dir / f"{experiment_label}_t{t:03d}_{_hl(hour_of[t])}h_Z{z + 1 + z0:02d}_GFP-Cy5_tanh-fit.png")
    (folder / "README.txt").write_text(
        "Z selection: each Z plane is analysed on its own (no Z integration) with the same correction, stain exclusion,\n"
        "and tanh model as the main analysis. Values are normalized intensity (fraction of the channel maximum over all\n"
        "selected timepoints and Z planes); slopes are in 1/mm. Concentration is NOT given per Z: the calibration standards\n"
        "were on a different plate whose Z positions do not match, so the calibration is only valid for the Z-integrated signal.\n"
        "1_heatmaps_time_by_Z: one heatmap per channel and fit parameter (rows = time, columns = Z; * = flagged fit).\n"
        "2_fit_parameters_versus_Z_per_timepoint: the same numbers as line plots, one figure per timepoint.\n"
        "3_profiles_of_every_Z_plane_overlaid: every Z plane's profile on one plot (colour = Z), with the bridge and stain marked.\n"
        "4_tanh_fit_plot_for_every_Z_plane: one plot per timepoint and Z with both channels and their fits.\n", encoding="utf-8")


def _save_mosaic_previews(preview_dir, tiff_dir, mosaics, methods, labels, timepoints, channel_cfg, config, experiment_label, hour_of, depth_tag):
    for ti, t in enumerate(timepoints):
        tag = f"{experiment_label}_t{t:03d}_{_hl(hour_of[t])}h"
        ranges = []
        reference_method = "ffc" if "ffc" in methods else methods[-1]
        for c, label in enumerate(labels):
            finite = mosaics[(reference_method, c, ti)][np.isfinite(mosaics[(reference_method, c, ti)])]
            ranges.append(tuple(float(v) for v in np.percentile(finite, [config.preview_low_percentile, config.preview_high_percentile])))
            panels = [(METHOD_LABELS[m], merge_rgb([np.nan_to_num(mosaics[(m, c, ti)], nan=0.0)], [channel_cfg[label].rgb], [ranges[c]])) for m in methods]
            _save_labeled_stack(preview_dir / f"{tag}_{label}_{depth_tag}_every_method_stacked.png", panels,
                                f"{tag} {label}: {depth_tag} mosaic, one row per correction method, same display range {ranges[c][0]:.0f}-{ranges[c][1]:.0f}")
            tiff_dir.mkdir(parents=True, exist_ok=True)
            for m in methods:
                tifffile.imwrite(tiff_dir / f"{tag}_{label}_{m}_{depth_tag}_8x-binned_float32.tif", mosaics[(m, c, ti)].astype(np.float32))
        panels = [(METHOD_LABELS[m], merge_rgb([np.nan_to_num(mosaics[(m, c, ti)], nan=0.0) for c in range(len(labels))],
                                               [channel_cfg[l].rgb for l in labels], ranges)) for m in methods]
        _save_labeled_stack(preview_dir / f"{tag}_GFP-Cy5_merge_{depth_tag}_every_method_stacked.png", panels,
                            f"{tag} GFP/Cy5 merge: {depth_tag} mosaic, one row per correction method")


def _plot_seam_summary(path, summary, labels, methods, experiment_label, main_tag):
    metrics = [
        ("every_Z_fitted_on_same_data_percent", "Every Z plane,\ncorrection fitted on the same data"),
        ("every_Z_held_out_percent", "Every Z plane,\nheld out (fair test)"),
        (f"{main_tag}_profile_held_out_percent", f"{main_tag} profiles,\nheld out (fair test)"),
    ]
    fig, axes = plt.subplots(1, len(labels), figsize=(7.5 * len(labels), 5.4), constrained_layout=True, sharey=True, squeeze=False)
    width = 0.8 / len(methods)
    for axis, label in zip(axes[0], labels):
        x = np.arange(len(metrics))
        for i, method in enumerate(methods):
            row = next(r for r in summary if r["method"] == method and r["channel"] == label)
            bars = axis.bar(x + (i - (len(methods) - 1) / 2) * width, [row[k] for k, _ in metrics], width,
                            color=METHOD_COLORS[method], label=METHOD_LABELS[method])
            axis.bar_label(bars, fmt="%.1f", fontsize=7)
        axis.set_xticks(x, [n for _, n in metrics])
        axis.set(ylabel="Brightness mismatch of the same spot in neighbouring tiles [%]",
                 title=f"{experiment_label} · {label}: seam mismatch after each correction (lower is better)")
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    _save(fig, path)


def _plot_seam_by_z(path, rows, labels, methods, z_count, experiment_label, z0=0):
    fig, axes = plt.subplots(1, len(labels), figsize=(7.5 * len(labels), 4.8), constrained_layout=True, sharey=True, squeeze=False)
    for axis, label in zip(axes[0], labels):
        for method in methods:
            values = [np.nanmedian([r["held_out_median_abs_log_ratio"] for r in rows
                                    if r["method"] == method and r["channel"] == label and r["z_index_one_based"] == z + 1 + z0])
                      for z in range(z_count)]
            axis.plot(np.arange(1, z_count + 1) + z0, 100 * (np.exp(values) - 1), "o-", ms=3, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axis.set(xlabel="Z plane [one-based index]", ylabel="Held-out seam mismatch [%]",
                 title=f"{experiment_label} · {label}: seam mismatch for each Z plane (fair test)")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    _save(fig, path)


def _plot_correction_profiles(path, fits, per_t_fits, labels, timepoints, width, z_count, fitted, hour_of):
    fig, axes = plt.subplots(len(fitted), len(labels), figsize=(7 * len(labels), 4.6 * len(fitted)), constrained_layout=True, squeeze=False)
    x = np.arange(width)
    cmap = plt.get_cmap("viridis", max(len(timepoints), 2))
    for r, method in enumerate(fitted):
        for c, label in enumerate(labels):
            axis = axes[r, c]
            for ti, t in enumerate(timepoints):
                fit_t = per_t_fits.get((method, c, ti))
                if fit_t is None:
                    continue
                for z in range(0, z_count, 3):
                    axis.plot(x, quadratic_illumination_profile(width, fit_t["linear"][z], fit_t["quadratic"][z]),
                              color=cmap(ti), alpha=0.35, lw=0.8, label=f"fitted on {_hl(hour_of[t])} h only" if z == 0 else None)
            for z in range(0, z_count, 3):
                fit = fits[(method, c)]
                axis.plot(x, quadratic_illumination_profile(width, fit["linear"][z], fit["quadratic"][z]),
                          color="black", lw=1.4, label="pooled over all timepoints (used)" if z == 0 else None)
            axis.axhline(1, color="gray", lw=0.6, ls="--")
            axis.set(title=f"{METHOD_LABELS[method]} · {label} (every 3rd Z plane)", xlabel="Camera X [pixel]",
                     ylabel="Relative illumination along camera X [–]")
            axis.grid(alpha=0.2)
            if len(timepoints) <= 8:
                axis.legend(fontsize=7, ncol=2)
    _save(fig, path)


def _plot_parameters_over_time(path, rows, labels, methods, reps, bridge, experiment_label, depth_title):
    slope_keys = []
    if "concentration" in reps:
        slope_keys.append(("concentration_signed_slope_ug_ml_per_mm", "Midpoint slope [µg/mL/mm]"))
    slope_keys += [("normalized_signed_slope_per_mm", "Midpoint slope [1/mm, normalized intensity]"),
                   ("midpoint_mm", "Tanh midpoint position [mm]"), ("r_squared", "Fit quality R² [–]")]
    fig, axes = plt.subplots(len(slope_keys), len(labels), figsize=(7.5 * len(labels), 3.3 * len(slope_keys)),
                             constrained_layout=True, squeeze=False)
    for r, (key, ylabel) in enumerate(slope_keys):
        for c, label in enumerate(labels):
            axis = axes[r, c]
            if key == "midpoint_mm" and bridge:
                axis.axhspan(bridge["left_foot_mm"], bridge["right_foot_mm"], color="#d9c27a", alpha=0.3, label="Bridge (from DIC)")
            for method in methods:
                selected = sorted((x for x in rows if x["method"] == method and x["channel"] == label), key=lambda x: x["timepoint"])
                times = [x["time_h"] for x in selected]
                values = [x[key] for x in selected]
                axis.plot(times, values, "-", color=METHOD_COLORS[method], label=METHOD_LABELS[method])
                good = [x["fit_valid"] for x in selected]
                axis.scatter([t for t, g in zip(times, good) if g], [v for v, g in zip(values, good) if g], color=METHOD_COLORS[method], s=18)
                axis.scatter([t for t, g in zip(times, good) if not g], [v for v, g in zip(values, good) if not g],
                             facecolors="none", edgecolors=METHOD_COLORS[method], s=40, label="flagged fit (open circle)" if method == methods[0] else None)
            axis.set(xlabel="Time [h]", ylabel=ylabel, title=f"{experiment_label} · {label} · {depth_title}")
            axis.grid(alpha=0.2)
            if r == 0 or key == "midpoint_mm":
                axis.legend(fontsize=7)
    _save(fig, path)


def _plot_overlay_all_timepoints(path, x, values, labels, timepoints, hour_of, info, bridge, stain, dic, title):
    fig, top, axes = _figure(dic, rows=len(labels), height=4.2)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(hour_of[timepoints[0]], max(hour_of[timepoints[-1]], hour_of[timepoints[0]] + 1e-9))
    for c, axis in enumerate(axes):
        _shade(axis, bridge, stain)
        for ti, t in enumerate(timepoints):
            axis.plot(x, values[(c, ti)], color=cmap(norm(hour_of[t])), lw=1.0,
                      label=f"{_hl(hour_of[t])} h" if len(timepoints) <= 12 else None)
        axis.set(xlabel=X_LABEL, ylabel=f"{labels[c]} {info['axis']}")
        axis.grid(alpha=0.2)
        if len(timepoints) <= 12:
            axis.legend(fontsize=7, title="Time", ncol=2)
        else:
            fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axis, label="Time [h] (colour = acquisition time)")
    (top if top is not None else axes[0]).set_title(title)
    if top is not None:
        _dic_strip(top, dic, bridge)
    _save(fig, path)


def _plot_mask_effect(path, masked, unmasked, labels, has_cal, experiment_label):
    key = "concentration_signed_slope_ug_ml_per_mm" if has_cal else "normalized_signed_slope_per_mm"
    unit = "µg/mL/mm" if has_cal else "1/mm"
    fig, axes = plt.subplots(2, len(labels), figsize=(7.5 * len(labels), 7), constrained_layout=True, squeeze=False)
    for c, label in enumerate(labels):
        a = sorted((r for r in masked if r["channel"] == label), key=lambda r: r["timepoint"])
        b = sorted((r for r in unmasked if r["channel"] == label), key=lambda r: r["timepoint"])
        for r, (k, ylabel) in enumerate(((key, f"Midpoint slope [{unit}]"), ("midpoint_mm", "Tanh midpoint position [mm]"))):
            axis = axes[r, c]
            axis.plot([x["time_h"] for x in a], [x[k] for x in a], "o-", label="Stain excluded (main result)")
            axis.plot([x["time_h"] for x in b], [x[k] for x in b], "s--", label="All positions used")
            axis.set(xlabel="Time [h]", ylabel=ylabel, title=f"{experiment_label} · {label}: effect of the stain mask")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
    _save(fig, path)


def _plot_alternative(path, x, y, record, info, rep, row, bridge, stain, dic, title, color, line, maximum):
    fig, top, axes = _figure(dic, rows=1, height=5.4)
    axis = axes[0]
    (top if top is not None else axis).set_title(title)
    if top is not None:
        _dic_strip(top, dic, bridge)
    _shade(axis, bridge, stain)
    axis.plot(x, y, color=color, lw=1.0, label=f"{row['channel']} data")
    band = (x >= record["band_start_mm"]) & (x <= record["band_end_mm"])
    norm_line = record["band_linear_intercept"] + record["band_linear_slope_per_mm"] * x[band]
    if rep == "concentration" and line is not None:
        shown = (norm_line * maximum - line["signal_intercept"]) / line["signal_per_ug_ml"]
        slope_text = f"{record['band_linear_slope_ug_ml_per_mm']:.3g} µg/mL/mm"
        steep_text = f"{record['steepest_local_slope_ug_ml_per_mm']:.3g} µg/mL/mm"
    elif rep == "intensity":
        shown = norm_line * maximum
        slope_text = f"{record['band_linear_slope_intensity_per_mm']:.4g} {info['slope_unit']}"
        steep_text = f"{record['steepest_local_slope_intensity_per_mm']:.4g} {info['slope_unit']}"
    else:
        shown = norm_line
        slope_text = f"{record['band_linear_slope_per_mm']:.3g} 1/mm"
        steep_text = f"{record['steepest_local_slope_per_mm']:.3g} 1/mm"
    axis.plot(x[band], shown, color="black", lw=2, label=f"Straight line over the 20–80% transition band: slope {slope_text}")
    axis.axvline(record["steepest_local_slope_position_mm"], color="black", ls=":",
                 label=f"Steepest local slope {steep_text} at {record['steepest_local_slope_position_mm']:.2f} mm")
    axis.set(xlabel=X_LABEL, ylabel=info["axis"])
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    _save(fig, path)


def diffusion_from_widths(times_h: np.ndarray, widths_mm: np.ndarray) -> dict[str, float]:
    """D from the growth of the tanh width.

    A diffusing front from a constant source is erfc-shaped; erf(u) and tanh(k*u) agree best near
    the midpoint for k = 2/sqrt(pi), which makes the tanh width w = sqrt(pi*D*(t - t0)).
    A straight line through w^2 versus time therefore has slope pi*D.
    """
    t = np.asarray(times_h, dtype=float) * 3600.0
    w2 = (np.asarray(widths_mm, dtype=float) * 1000.0) ** 2
    slope, intercept = np.polyfit(t, w2, 1)
    predicted = slope * t + intercept
    ss_total = float(np.sum((w2 - w2.mean()) ** 2))
    return {
        "D_um2_per_s": float(slope / np.pi),
        "t0_h": float(-intercept / slope / 3600.0) if slope != 0 else float("nan"),
        "r_squared": 1.0 - float(np.sum((w2 - predicted) ** 2)) / ss_total if ss_total > 0 else float("nan"),
        "points": int(t.size),
    }


def front_motion_table(times_h: np.ndarray, midpoints_mm: np.ndarray) -> list[dict[str, float]]:
    """Front displacement from the first timepoint, its average speed, and (dX)^2/dt.

    X0 is the midpoint at the first timepoint and dt the time since that timepoint, as
    defined by Shelly and her advisor. (X - X0)/dt is a speed; (X - X0)^2/dt has the units
    of a diffusion coefficient.
    """
    t = np.asarray(times_h, dtype=float)
    x = np.asarray(midpoints_mm, dtype=float)
    rows = []
    for time, position in zip(t[1:], x[1:]):
        dt_s = (time - t[0]) * 3600.0
        dx_um = (position - x[0]) * 1000.0
        rows.append({
            "dt_h": float(time - t[0]),
            "displacement_um": float(dx_um),
            "front_speed_um_per_s": float(dx_um / dt_s),
            "front_speed_mm_per_h": float(dx_um / dt_s * 3.6),
            "D_dx2_over_dt_um2_per_s": float(dx_um * dx_um / dt_s),
        })
    return rows


def _diffusion_step(folder, rows, bridge, labels, experiment_label, depth_title, tile_hours=None, expected_d=None):
    """D(t) = (X - X0)^2 / dt and front speed (X - X0) / dt from the tanh midpoints.

    ``tile_hours(label, timepoint, midpoint_mm)`` returns the acquisition time of the tile
    that contains the front; when given, a second version uses those times for X and X0.
    """
    colors = {"GFP": "#20a83e", "Cy5": "#d62a8b"}
    table: list[dict[str, Any]] = []
    fig, axes = plt.subplots(3, 1, figsize=(13, 13), constrained_layout=True, sharex=True)
    for label in labels:
        selected = sorted((r for r in rows if r["channel"] == label), key=lambda r: r["time_h"])
        t_start = np.array([r["time_h"] for r in selected])
        mid = np.array([r["midpoint_mm"] for r in selected])
        versions = [("timepoint start time", t_start, "o-")]
        if tile_hours is not None:
            t_tile = np.array([tile_hours(label, r["timepoint"], r["midpoint_mm"]) for r in selected])
            versions.append(("time the front's tile was imaged", t_tile, "s--"))
        color = colors.get(label, "black")
        axes[0].plot(t_start, mid, "o-", color=color, label=f"{label} front (tanh midpoint)")
        axes[0].axhline(mid[0], color=color, lw=0.8, ls=":", label=f"{label} X0 = {mid[0]:.2f} mm (first timepoint)")
        for name, times, style in versions:
            motion = front_motion_table(times, mid)
            for row, record in zip(motion, selected[1:]):
                table.append({"channel": label, "timepoint": record["timepoint"], "time_h": record["time_h"],
                              "time_basis": name, "X_mm": record["midpoint_mm"], "X0_mm": float(mid[0]), **row})
            axes[1].plot(t_start[1:], [m["front_speed_um_per_s"] for m in motion], style, color=color, ms=4,
                         mfc="none" if style.startswith("s") else color, label=f"{label}, Δt from {name}")
            axes[2].plot(t_start[1:], [m["D_dx2_over_dt_um2_per_s"] for m in motion], style, color=color, ms=4,
                         mfc="none" if style.startswith("s") else color, label=f"{label}, Δt from {name}")
    if bridge:
        axes[0].axhspan(bridge["left_foot_mm"], bridge["right_foot_mm"], color="#d9c27a", alpha=0.3, label="Bridge (from DIC)")
    if expected_d:
        axes[2].axhspan(expected_d[0], expected_d[1], color="#9ecae1", alpha=0.4,
                        label=f"Expected D for the dextran ({expected_d[0]:g}-{expected_d[1]:g} µm²/s)")
    axes[1].axhline(0, color="black", lw=0.6)
    axes[0].set(ylabel="Front position X [mm]", title="Front position: tanh midpoint at each timepoint")
    axes[1].set(ylabel="(X − X0) / Δt [µm/s]", title="Average front speed since the first timepoint (positive = moving right)")
    axes[2].set(xlabel="Time [h]", ylabel="(X − X0)² / Δt [µm²/s]", yscale="log",
                title="D(t) = (X − X0)² / Δt since the first timepoint (units of a diffusion coefficient)")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
    fig.suptitle(f"{experiment_label} · {depth_title}: front displacement from the first timepoint")
    _save(fig, folder / f"{experiment_label}_front_displacement_speed_and_D_over_time.png")
    write_rows_csv(folder / "front_displacement_speed_and_D_over_time.csv", table)
    (folder / "README.txt").write_text(
        "Front X = tanh midpoint at each timepoint; X0 = midpoint at the first timepoint; Δt = time since the first timepoint.\n"
        "(X - X0)/Δt is the average speed of the front since the start [µm/s; positive = moving right].\n"
        "(X - X0)²/Δt has the units of a diffusion coefficient [µm²/s] (D = Δx²/τ). For a front spreading by diffusion\n"
        "alone it stays roughly constant in time; a value that falls as 1/Δt means the front has stopped moving.\n"
        "Two time bases are given: the start time of each timepoint, and the time the tile containing the front was\n"
        "imaged (a timepoint takes about 6 minutes to image tile by tile, which matters for the first intervals).\n",
        encoding="utf-8")


def _write_profile_table(path, x, excluded, bridge, columns: dict[str, np.ndarray], unit_text: str, stride: int = 4) -> None:
    """Profile values for re-plotting (every ``stride``-th pixel, about 1.3 µm spacing for stride 4)."""
    import csv

    keep = np.arange(0, x.size, stride)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"# values: {unit_text}; position: {X_LABEL}"])
        header = ["position_mm", "excluded_from_fit_stain"] + (["on_bridge"] if bridge else []) + list(columns)
        writer.writerow(header)
        for index in keep:
            row = [f"{x[index]:.5f}", int(excluded[index])]
            if bridge:
                row.append(int(bridge["left_foot_mm"] <= x[index] <= bridge["right_foot_mm"]))
            row += [f"{values[index]:.6g}" for values in columns.values()]
            writer.writerow(row)


def _plot_mode_comparison(path, main_rows, other_rows, main_mode, other_mode, labels, has_cal, experiment_label):
    key = "concentration_signed_slope_ug_ml_per_mm" if has_cal else "normalized_signed_slope_per_mm"
    unit = "µg/mL/mm" if has_cal else "1/mm"
    fig, axes = plt.subplots(2, len(labels), figsize=(7.5 * len(labels), 7), constrained_layout=True, squeeze=False)
    for c, label in enumerate(labels):
        a = sorted((r for r in main_rows if r["channel"] == label), key=lambda r: r["timepoint"])
        b = sorted((r for r in other_rows if r["channel"] == label), key=lambda r: r["timepoint"])
        for r, (k, ylabel) in enumerate(((key, f"Midpoint slope [{unit}]"), ("midpoint_mm", "Tanh midpoint position [mm]"))):
            axis = axes[r, c]
            axis.plot([x["time_h"] for x in a], [x[k] for x in a], "o-", label=f"{main_mode['title']} (main result)")
            axis.plot([x["time_h"] for x in b], [x[k] for x in b], "s--", label=f"{other_mode['title']} (robustness check)")
            axis.set(xlabel="Time [h]", ylabel=ylabel, title=f"{experiment_label} · {label}: main result vs robustness check")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
    _save(fig, path)


def _write_summary(path, experiment_label, seam_summary, fit_rows, conversion, labels, timepoints, hour_of, bridge,
                   stain, primary, has_cal, modes):
    main = modes[0]
    lines = [f"# {experiment_label}: key results", "",
             f"Main result: {METHOD_LABELS[primary]}, {main['title']}. "
             + (f"Robustness check: {modes[1]['title']}." if len(modes) > 1 else ""), ""]
    lines += ["## Seam mismatch (fair, held-out test)", "",
              f"| Method | Channel | Every Z plane | {main['file_tag']} profile |", "|---|---|---:|---:|"]
    for r in seam_summary:
        lines.append(f"| {r['method_name']} | {r['channel']} | {r['every_Z_held_out_percent']:.1f}% | "
                     f"{r[main['file_tag'] + '_profile_held_out_percent']:.1f}% |")
    if bridge:
        lines += ["", "## Bridge (from DIC)", "",
                  f"Outer edges {bridge['left_foot_mm']:.2f}–{bridge['right_foot_mm']:.2f} mm (width {bridge['width_between_feet_mm']:.2f} mm); "
                  f"half-height edges {bridge['left_half_max_mm']:.2f}–{bridge['right_half_max_mm']:.2f} mm (width {bridge['width_at_half_max_mm']:.2f} mm)."]
    if stain.get("enabled"):
        lines += ["", "## Stain mask", "", "Excluded from fits: " + ", ".join(f"{a:.2f}–{b:.2f} mm" for a, b in stain["excluded_intervals_mm"])
                  + f" ({100 * stain['excluded_fraction_of_profile']:.1f}% of the profile)."]
    if has_cal:
        lines += ["", f"## Calibration lines ({METHOD_LABELS[primary]})", "",
                  "| Depth | Channel | Line | Standards used [µg/mL] | µg/mL at zero signal | R² |", "|---|---|---|---|---:|---:|"]
        for mode in modes:
            for c, label in enumerate(labels):
                for kind, text in (("primary", "PRIMARY, with intercept"), ("origin", "forced through zero")):
                    line = conversion[(mode["key"], primary, c)][kind]
                    used = ", ".join(f"{v:g}" for v in line["standards_used_ug_ml"])
                    lines.append(f"| {mode['title']} | {label} | {text} | {used} | {line['intercept_as_concentration_ug_ml']:.2f} | {line['r_squared']:.4f} |")
    unit = "µg/mL/mm" if has_cal else "1/mm"
    key = "concentration_signed_slope_ug_ml_per_mm" if has_cal else "normalized_signed_slope_per_mm"
    for mode in modes:
        rows = [r for r in fit_rows[mode["key"]] if r["method"] == primary]
        lines += ["", f"## Tanh fits: {METHOD_LABELS[primary]}, {mode['title']}", "",
                  f"| Channel | Time [h] | Slope [{unit}] | Midpoint [mm] | Midpoint vs bridge | R² | Flags |", "|---|---:|---:|---:|---|---:|---|"]
        for r in sorted(rows, key=lambda r: (r["channel"], r["timepoint"])):
            lines.append(f"| {r['channel']} | {_hl(r['time_h'])} | {r[key]:.3g} | {r['midpoint_mm']:.2f} | {r.get('midpoint_relative_to_bridge', '')} | "
                         f"{r['r_squared']:.3f} | {r['qc_flags'] or 'ok'} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_guide(path, specs, steps, experiment_label, primary, reps, has_cal, modes):
    main = modes[0]
    descriptions = {
        "flatfield": "How bright the microscope illumination is at every camera pixel, measured from the uniform 32 µg/mL standard. "
                     "Also the camera background (from the 0 µg/mL standard) and hot pixels.",
        "calibration": "Intensity of every calibration standard and the line used to convert intensity to µg/mL, for each depth "
                       f"({', '.join(m['title'] for m in modes)}). The standards were imaged with the same Z reference as the experiment, "
                       "so the same Z planes are used. Compares the primary line (0–24 µg/mL with intercept), the same line forced "
                       "through zero, and piecewise interpolation.",
        "seams": "How well each illumination correction makes neighbouring tiles agree where they overlap. 'Held out' = the "
                 "correction was fitted without the tile pair and timepoint being tested (the fair test). Also stacked mosaic images "
                 f"at {main['title']} (one row per method, same brightness scale) so seams can be judged by eye.",
        "bridge": "The bridge position, found automatically from the large DIC image, in the same mm coordinates as the profiles.",
        "stain": "Which positions are excluded from the fits because of the plate stain, and the fixed rule used to find them.",
        "zsel": "Every Z plane analysed separately at selected timepoints, to choose a Z plane. Heatmaps are rows = time, columns = Z.",
        "methods": f"Profiles at {main['title']} for every timepoint with all correction methods on the same plot, with their tanh fits.",
        "mode0": f"THE MAIN RESULT: {METHOD_LABELS[primary]}, {main['title']}, one plot per timepoint with both channels and their "
                 "tanh fits. Subfolders 1–3 are the same fits shown with different Y axes; 4 (if present) = the same fits without "
                 "the stain mask (for comparison); 5 (if present) = an additional, model-free analysis of flagged tanh fits.",
        "diffusion": "Front position X (tanh midpoint) relative to the first timepoint X0: average front speed (X − X0)/Δt and "
                     "D(t) = (X − X0)²/Δt over time, with Δt from the timepoint start or from the time the front's tile was imaged.",
    }
    for index, mode in enumerate(modes[1:], start=1):
        descriptions[f"mode{index}"] = (f"Robustness check: the same analysis on the {mode['title']} (all below the membrane). "
                                        "Includes a plot comparing its slopes and midpoints with the main result.")
    lines = [f"# How to read this folder ({experiment_label})", "",
             "Folders are numbered in the order the analysis happens. Read them from top to bottom.", ""]
    for key, _ in specs:
        lines += [f"## {steps[key].name}", "", descriptions[key], ""]
    lines += ["## File names", "",
              f"`{experiment_label}_t012_24h_...` = experiment, zero-based timepoint index (t012), experimental time "
              "(24 h = 12 × 2 h), then what the file shows.", "", "Y-axis representations (each in its own subfolder):", ""]
    for rep in reps:
        info = representation_info(rep, main)
        lines.append(f"- `{info['folder']}`: {info['axis']}")
    lines += ["", "Marks on the profile plots:", "",
              "- yellow band: the bridge (from DIC);", "- red hatching: positions excluded from the fit because of the stain (data still shown);",
              "- dashed lines: tanh fits; ⚠ in a legend = the fit was flagged (e.g. a plateau or midpoint at the edge of the field).", "",
              "`cache_measurements_do_not_edit` holds the numerical measurements so later analyses do not have to re-read the ND2 file "
              "(`cache_sources.json` lists earlier cache folders that were reused). `SUMMARY_key_results.md` lists the main numbers. "
              "`run_metadata.json` records every setting."]
    if not has_cal:
        lines += ["", "This dataset has no calibration standards, so results are in intensity units only (no µg/mL), and only "
                      "'no correction' and 'overlap quadratic' are compared."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
