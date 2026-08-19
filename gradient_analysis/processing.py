"""Pure numerical operations used by the pipeline."""

from __future__ import annotations

import numpy as np

UINT16_LEVELS = 65_536


def stitch(tiles: list[np.ndarray]) -> np.ndarray:
    """Place equal-height positions directly end-to-end from left to right."""
    if not tiles:
        raise ValueError("At least one tile is required.")
    heights = {tile.shape[0] for tile in tiles}
    if len(heights) != 1:
        raise ValueError(f"All tiles must have the same height; received {sorted(heights)}")
    return np.concatenate(tiles, axis=1)


def x_profile(image: np.ndarray) -> np.ndarray:
    """Mean intensity for every image column (average over Y)."""
    return np.mean(image, axis=0, dtype=np.float64)


def physical_x_axes_mm(
    position_x_um: list[float],
    pixel_size_um: float,
    tile_width: int,
) -> list[np.ndarray]:
    """Return one physical x-axis per position, with P01 left edge at 0 mm.

    Stage X metadata are treated as image-center coordinates. Positions are
    expected in the already-resolved left-to-right order. Since all tiles have
    equal width, the left-edge offset between positions equals the stage-center
    offset between those positions.
    """
    if not position_x_um:
        raise ValueError("At least one position is required.")
    if pixel_size_um <= 0:
        raise ValueError("Pixel size must be positive.")
    if tile_width <= 0:
        raise ValueError("Tile width must be positive.")
    direction = 1.0
    if len(position_x_um) > 1 and position_x_um[-1] > position_x_um[0]:
        direction = -1.0
    pixel_mm = pixel_size_um / 1000.0
    first_center = float(position_x_um[0])
    local = np.arange(tile_width, dtype=np.float64) * pixel_mm
    return [
        direction * (first_center - float(center_um)) / 1000.0 + local
        for center_um in position_x_um
    ]


def feather_profiles(
    x_axes: list[np.ndarray], profiles: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Interpolate onto one physical grid and blend overlaps with cosine weights."""
    if not x_axes or len(x_axes) != len(profiles):
        raise ValueError("x_axes and profiles must be non-empty and equal length.")
    if any(axis.ndim != 1 or profile.ndim != 1 or axis.size != profile.size for axis, profile in zip(x_axes, profiles)):
        raise ValueError("Every physical axis must match its one-dimensional profile.")
    pixel_steps = np.concatenate([np.diff(axis) for axis in x_axes if axis.size > 1])
    pixel_mm = float(np.median(pixel_steps))
    if not np.isfinite(pixel_mm) or pixel_mm <= 0:
        raise ValueError("Physical axes must be strictly increasing.")
    origin = min(float(axis[0]) for axis in x_axes)
    fractional_starts = [(float(axis[0]) - origin) / pixel_mm for axis in x_axes]
    starts = [int(round(value)) for value in fractional_starts]
    alignment_errors = [
        abs(value - start) for value, start in zip(fractional_starts, starts)
    ]
    end = max(float(axis[-1]) for axis in x_axes)
    total = int(np.floor((end - origin) / pixel_mm + 1e-9)) + 1
    axis = origin + np.arange(total, dtype=np.float64) * pixel_mm
    values = [
        np.interp(axis, local_x, profile.astype(np.float64), left=np.nan, right=np.nan)
        for local_x, profile in zip(x_axes, profiles)
    ]
    tile_weights = [np.isfinite(value).astype(np.float64) for value in values]
    overlaps: list[int] = []
    for index in range(len(profiles) - 1):
        overlap_mask = np.isfinite(values[index]) & np.isfinite(values[index + 1])
        overlap_indices = np.flatnonzero(overlap_mask)
        overlap = int(overlap_indices.size)
        if overlap == 0 and float(x_axes[index][-1]) < float(x_axes[index + 1][0]):
            raise ValueError("Profiles contain a physical gap and cannot be feathered.")
        overlaps.append(overlap)
        if overlap:
            phase = np.linspace(0.0, np.pi, overlap, dtype=np.float64)
            tile_weights[index][overlap_indices] *= 0.5 * (1.0 + np.cos(phase))
            tile_weights[index + 1][overlap_indices] *= 0.5 * (1.0 - np.cos(phase))
    weighted = np.zeros(total, dtype=np.float64)
    weights = np.zeros(total, dtype=np.float64)
    for value, weight in zip(values, tile_weights):
        valid = np.isfinite(value)
        weighted[valid] += value[valid] * weight[valid]
        weights[valid] += weight[valid]
    if np.any(weights <= 0):
        raise ValueError("Feathering produced uncovered physical coordinates.")
    return axis, weighted / weights, {
        "method": "cosine_feather",
        "placement": "linear interpolation from subpixel stage coordinates onto the camera-pixel grid",
        "pixel_size_mm": pixel_mm,
        "position_start_indices": starts,
        "position_start_coordinates_px": fractional_starts,
        "overlap_widths_px": overlaps,
        "output_width_px": total,
        "maximum_alignment_error_px": max(alignment_errors, default=0.0),
    }


def profile_normalization_constants(
    profiles_by_channel: dict[str, list[np.ndarray]],
) -> dict[str, float]:
    """Return one exact positive maximum per channel across 1-D profiles."""
    constants: dict[str, float] = {}
    for channel, profiles in profiles_by_channel.items():
        if not profiles:
            raise ValueError(f"No profiles supplied for channel {channel}.")
        maximum = max(float(np.nanmax(profile)) for profile in profiles)
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError(f"Channel {channel} has no positive finite profile maximum.")
        constants[channel] = maximum
    return constants


def tanh_profile(
    x: np.ndarray,
    left: float,
    right: float,
    midpoint: float,
    width: float,
) -> np.ndarray:
    """Evaluate the four-parameter hyperbolic-tangent gradient model."""
    return left + 0.5 * (right - left) * (1.0 + np.tanh((x - midpoint) / width))


def _median_bin_profile(
    x: np.ndarray, y: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep].astype(np.float64)
    y = y[keep].astype(np.float64)
    if x.size < 4:
        raise ValueError("At least four finite profile points are required.")
    order = np.argsort(x)
    x, y = x[order], y[order]
    if x.size <= max_points:
        return x, y
    edges = np.linspace(0, x.size, max_points + 1, dtype=int)
    return (
        np.array([np.median(x[edges[i] : edges[i + 1]]) for i in range(max_points)]),
        np.array([np.median(y[edges[i] : edges[i + 1]]) for i in range(max_points)]),
    )


def fit_tanh_profile(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_points: int = 1024,
    pixel_size_mm: float | None = None,
) -> dict[str, object]:
    """Robustly fit a normalized profile and report its midpoint slope."""
    from scipy.optimize import least_squares

    fit_x, fit_y = _median_bin_profile(x, y, max_points)
    span = float(fit_x[-1] - fit_x[0])
    if span <= 0:
        raise ValueError("Physical x coordinates must span a positive distance.")
    edge_count = max(1, int(np.ceil(0.05 * fit_y.size)))
    initial_left = float(np.clip(np.median(fit_y[:edge_count]), -0.1, 1.1))
    initial_right = float(np.clip(np.median(fit_y[-edge_count:]), -0.1, 1.1))
    halfway = 0.5 * (initial_left + initial_right)
    initial_midpoint = float(fit_x[np.argmin(np.abs(fit_y - halfway))])
    minimum_width = max(float(pixel_size_mm or np.median(np.diff(fit_x))), np.finfo(float).eps)
    lower = np.array([-0.1, -0.1, fit_x[0], minimum_width], dtype=np.float64)
    upper = np.array([1.1, 1.1, fit_x[-1], 2.0 * span], dtype=np.float64)
    initial = np.array(
        [initial_left, initial_right, initial_midpoint, max(span / 10.0, minimum_width * 2.0)]
    )
    initial = np.minimum(np.maximum(initial, lower + 1e-9), upper - 1e-9)
    try:
        result = least_squares(
            lambda parameters: tanh_profile(fit_x, *parameters) - fit_y,
            initial,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.02,
            max_nfev=5000,
        )
        parameters = result.x
        optimizer_success = bool(result.success)
        optimizer_message = str(result.message)
    except Exception as error:  # retain a flagged record instead of aborting the run
        parameters = initial
        optimizer_success = False
        optimizer_message = f"{type(error).__name__}: {error}"
    left, right, midpoint, width = (float(value) for value in parameters)
    predicted = tanh_profile(fit_x, left, right, midpoint, width)
    residual = fit_y - predicted
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((fit_y - np.mean(fit_y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    amplitude = right - left
    signed_slope = amplitude / (2.0 * width)
    tolerance = 1e-4
    at_bound = bool(
        np.any(np.isclose(parameters, lower, rtol=0, atol=tolerance))
        or np.any(np.isclose(parameters, upper, rtol=0, atol=tolerance))
    )
    flags: list[str] = []
    if not optimizer_success:
        flags.append("non_convergence")
    if at_bound:
        flags.append("parameter_at_bound")
    if abs(amplitude) < 0.05:
        flags.append("low_amplitude")
    values = np.array([left, right, midpoint, width, signed_slope, r_squared])
    if not np.all(np.isfinite(values)):
        flags.append("non_finite_result")
    return {
        "left_plateau": left,
        "right_plateau": right,
        "amplitude": amplitude,
        "midpoint_mm": midpoint,
        "width_mm": width,
        "signed_slope_per_mm": signed_slope,
        "absolute_slope_per_mm": abs(signed_slope),
        "r_squared": r_squared,
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "optimizer_success": optimizer_success,
        "optimizer_message": optimizer_message,
        "fit_valid": not flags,
        "qc_flags": flags,
        "fit_point_count": int(fit_x.size),
    }


def smooth_profile(profile: np.ndarray, window: int) -> np.ndarray:
    """Robustly smooth a 1-D profile using an odd reflected moving average."""
    if profile.size < 3:
        return profile.astype(np.float64, copy=True)
    window = max(3, min(int(window), int(profile.size)))
    if window % 2 == 0:
        window -= 1
    pad = window // 2
    padded = np.pad(profile.astype(np.float64), pad, mode="reflect")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def smooth_image(image: np.ndarray, window_y: int, window_x: int) -> np.ndarray:
    """Smooth a 2-D image with reflected separable moving averages."""
    if image.ndim != 2:
        raise ValueError("Only 2-D images can be smoothed.")
    smoothed = image.astype(np.float64, copy=False)
    for axis, requested_window in ((0, window_y), (1, window_x)):
        size = smoothed.shape[axis]
        if size < 3:
            continue
        window = max(3, min(int(requested_window), int(size)))
        if window % 2 == 0:
            window -= 1
        pad = window // 2
        padded = np.pad(
            smoothed,
            ((pad, pad), (0, 0)) if axis == 0 else ((0, 0), (pad, pad)),
            mode="reflect",
        )
        kernel = np.ones(window, dtype=np.float64) / window
        smoothed = np.apply_along_axis(lambda values: np.convolve(values, kernel, mode="valid"), axis, padded)
    return smoothed


def fitted_illumination_profile(
    reference_profile: np.ndarray,
    smoothing_window: int,
    degree: int = 2,
) -> tuple[np.ndarray, float]:
    """Fit a broad polynomial laser/illumination profile to one reference tile."""
    if degree < 0:
        raise ValueError("Polynomial degree must be non-negative.")
    smooth = smooth_profile(reference_profile, smoothing_window)
    x = np.linspace(-1.0, 1.0, smooth.size)
    coefficients = np.polyfit(x, smooth, min(degree, smooth.size - 1))
    fitted = np.polyval(coefficients, x).astype(np.float64)
    plateau = float(np.median(fitted))
    floor = max(plateau * 0.05, np.finfo(float).eps)
    fitted = np.maximum(fitted, floor)
    return fitted, plateau


def smoothed_illumination_profile(
    reference_profile: np.ndarray,
    smoothing_window: int,
) -> tuple[np.ndarray, float]:
    """Use the smoothed reference x-profile directly as the illumination profile."""
    fitted = smooth_profile(reference_profile, smoothing_window)
    plateau = float(np.median(fitted))
    floor = max(plateau * 0.05, np.finfo(float).eps)
    fitted = np.maximum(fitted, floor)
    return fitted.astype(np.float64), plateau


def smoothed_illumination_image(
    reference_tile: np.ndarray,
    smoothing_window_y: int,
    smoothing_window_x: int,
) -> tuple[np.ndarray, float]:
    """Use a smoothed 2-D reference tile directly as the illumination map."""
    fitted = smooth_image(reference_tile, smoothing_window_y, smoothing_window_x)
    plateau = float(np.median(fitted))
    floor = max(plateau * 0.05, np.finfo(float).eps)
    fitted = np.maximum(fitted, floor)
    return fitted.astype(np.float64), plateau


def normalized_profile_shape(profile: np.ndarray, smoothing_window: int) -> np.ndarray:
    """Return a smoothed x-profile normalized by its median brightness."""
    smooth = smooth_profile(profile, smoothing_window)
    median = max(float(np.median(smooth)), np.finfo(float).eps)
    return smooth / median


def profile_curvature(profile: np.ndarray, smoothing_window: int) -> float:
    """Estimate broad quadratic curvature after median normalization."""
    normalized = normalized_profile_shape(profile, smoothing_window)
    x = np.linspace(-1.0, 1.0, normalized.size)
    coefficients = np.polyfit(x, normalized, min(2, normalized.size - 1))
    return float(coefficients[0]) if coefficients.size == 3 else 0.0


def reference_candidate_score(
    profiles: list[np.ndarray],
    mean_intensities: list[float],
    saturated_fractions: list[float],
    smoothing_window: int,
    saturation_threshold: float,
    brightness_scale: float | None = None,
) -> dict[str, float | bool | int]:
    """Score one position as a stable, bright illumination reference candidate."""
    if not (profiles and len(profiles) == len(mean_intensities) == len(saturated_fractions)):
        raise ValueError("Reference candidate inputs must be non-empty and equal length.")

    normalized = np.stack([normalized_profile_shape(profile, smoothing_window) for profile in profiles])
    shape_variability = float(np.mean(np.std(normalized, axis=0))) if len(profiles) > 1 else 0.0
    median_brightness = float(np.median(mean_intensities))
    max_saturated_fraction = float(np.max(saturated_fractions))
    unsaturated = max_saturated_fraction <= saturation_threshold
    scale = brightness_scale if brightness_scale and brightness_scale > 0 else median_brightness
    brightness_fraction = median_brightness / max(float(scale), np.finfo(float).eps)
    saturation_penalty = 0.0 if unsaturated else 50.0 * (max_saturated_fraction - saturation_threshold)
    score = brightness_fraction / (1.0 + 20.0 * shape_variability + saturation_penalty)
    if not unsaturated:
        score *= 0.01

    reference_shape = np.median(normalized, axis=0)
    distances = np.mean(np.abs(normalized - reference_shape[np.newaxis, :]), axis=1)
    best_profile_index = int(np.argmin(distances))
    return {
        "score": float(score),
        "median_brightness": median_brightness,
        "brightness_fraction": float(brightness_fraction),
        "shape_variability": shape_variability,
        "max_saturated_fraction": max_saturated_fraction,
        "unsaturated": bool(unsaturated),
        "best_profile_index": best_profile_index,
    }


def linearity_score(profile: np.ndarray, smoothing_window: int) -> float:
    """Score bright, flat profiles highly; used only for automatic selection."""
    smooth = smooth_profile(profile, smoothing_window)
    brightness = max(float(np.median(smooth)), np.finfo(float).eps)
    x = np.linspace(-1.0, 1.0, smooth.size)
    slope = abs(float(np.polyfit(x, smooth / brightness, 1)[0]))
    roughness = float(np.std(smooth / brightness))
    return brightness / (1.0 + 8.0 * slope + 8.0 * roughness)


def choose_reference(profiles: list[np.ndarray], smoothing_window: int) -> int:
    """Return the list index of the brightest, flattest candidate profile."""
    if not profiles:
        raise ValueError("No profiles were supplied.")
    return int(np.argmax([linearity_score(p, smoothing_window) for p in profiles]))


def correction_curve(reference_profile: np.ndarray, smoothing_window: int) -> tuple[np.ndarray, float]:
    """Return plateau/fitted_reference(x), with a floor to prevent noise blow-up."""
    fitted, plateau = fitted_illumination_profile(reference_profile, smoothing_window, degree=2)
    return plateau / fitted, plateau


def correct_tile(tile: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """Apply a local-x illumination correction and preserve floating precision."""
    if tile.shape[1] != curve.size:
        raise ValueError("Correction curve width does not match tile width.")
    return tile.astype(np.float32) * curve.astype(np.float32)[np.newaxis, :]


def overlap_log_ratio_samples(
    left_tile: np.ndarray,
    right_tile: np.ndarray,
    x_offset_px: int,
    y_shift_px: int,
    min_signal: float,
    saturation_signal: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    """Return robust column-wise log ratios from the physical tile overlap.

    ``x_offset_px`` maps local x=0 in the right tile to that local x in the
    left tile. ``y_shift_px`` maps a right-tile row to the corresponding
    left-tile row. Medians across Y make the fit resistant to isolated pixels.
    """
    if left_tile.shape != right_tile.shape or left_tile.ndim != 2:
        raise ValueError("Overlap tiles must be equally sized 2-D arrays.")
    height, width = left_tile.shape
    if not 0 < x_offset_px < width:
        raise ValueError("x_offset_px must produce a non-empty overlap.")
    overlap = width - x_offset_px
    if abs(y_shift_px) >= height:
        raise ValueError("y_shift_px leaves no overlapping rows.")
    if y_shift_px >= 0:
        left_rows = slice(y_shift_px, height)
        right_rows = slice(0, height - y_shift_px)
    else:
        left_rows = slice(0, height + y_shift_px)
        right_rows = slice(-y_shift_px, height)
    left = left_tile[left_rows, x_offset_px:].astype(np.float64)
    right = right_tile[right_rows, :overlap].astype(np.float64)
    valid = (
        (left > min_signal)
        & (right > min_signal)
        & (left < saturation_signal)
        & (right < saturation_signal)
    )
    log_ratio = np.full(overlap, np.nan, dtype=np.float64)
    valid_counts = np.count_nonzero(valid, axis=0)
    for column in np.flatnonzero(valid_counts):
        mask = valid[:, column]
        log_ratio[column] = float(np.median(np.log(left[mask, column] / right[mask, column])))
    keep = np.isfinite(log_ratio)
    x_left = np.flatnonzero(keep).astype(np.float64) + x_offset_px
    x_right = np.flatnonzero(keep).astype(np.float64)
    diagnostics: dict[str, float | int] = {
        "x_offset_px": x_offset_px,
        "overlap_width_px": overlap,
        "y_shift_px": y_shift_px,
        "valid_pixel_count": int(valid.sum()),
        "valid_column_count": int(keep.sum()),
        "raw_median_abs_log_ratio": float(np.median(np.abs(log_ratio[keep]))) if keep.any() else float("nan"),
    }
    return x_left, x_right, log_ratio[keep], diagnostics


def fit_overlap_log_quadratic(
    tile_width: int,
    x_left: np.ndarray,
    x_right: np.ndarray,
    log_ratios: np.ndarray,
    iterations: int = 8,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit a positive, median-normalized illumination profile from overlaps."""
    if not (x_left.size == x_right.size == log_ratios.size) or x_left.size < 3:
        raise ValueError("At least three equal-length overlap samples are required.")
    center = (tile_width - 1) / 2.0
    scale = max(center, 1.0)
    a = (x_left - center) / scale
    b = (x_right - center) / scale
    design = np.column_stack((a - b, a * a - b * b))
    weights = np.ones(log_ratios.size, dtype=np.float64)
    coefficients = np.zeros(2, dtype=np.float64)
    for _ in range(iterations):
        root_w = np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(
            design * root_w[:, None], log_ratios * root_w, rcond=None
        )
        residual = log_ratios - design @ coefficients
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        robust_scale = max(1.4826 * mad, np.finfo(float).eps)
        normalized = np.abs(residual - median) / (1.345 * robust_scale)
        weights = np.ones_like(normalized)
        high = normalized > 1.0
        weights[high] = 1.0 / normalized[high]
    local_x = (np.arange(tile_width, dtype=np.float64) - center) / scale
    log_profile = coefficients[0] * local_x + coefficients[1] * local_x * local_x
    log_profile -= float(np.median(log_profile))
    profile = np.exp(log_profile)
    residual = log_ratios - design @ coefficients
    details: dict[str, object] = {
        "model": "exp(linear*x + quadratic*x^2)",
        "normalized_x_center_px": center,
        "normalized_x_scale_px": scale,
        "linear_coefficient": float(coefficients[0]),
        "quadratic_coefficient": float(coefficients[1]),
        "profile_median": float(np.median(profile)),
        "profile_min": float(np.min(profile)),
        "profile_max": float(np.max(profile)),
        "sample_count": int(log_ratios.size),
        "median_abs_log_residual": float(np.median(np.abs(residual))),
    }
    return profile, details


def subtract_background_floor(image: np.ndarray, background: float) -> np.ndarray:
    """Subtract a constant microscope background and floor at zero."""
    return np.maximum(image.astype(np.float32) - float(background), 0.0)


def correct_tile_2d(tile: np.ndarray, illumination_map: np.ndarray, plateau: float) -> np.ndarray:
    """Apply a 2-D illumination correction and preserve floating precision."""
    if tile.shape != illumination_map.shape:
        raise ValueError("2-D illumination map shape does not match tile shape.")
    curve = float(plateau) / illumination_map.astype(np.float32)
    return tile.astype(np.float32) * curve


def to_uint16(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Round corrected values to viewer-compatible uint16 and report clipping."""
    nonfinite = int(np.count_nonzero(~np.isfinite(image)))
    below = int(np.count_nonzero(image < 0))
    above = int(np.count_nonzero(image > np.iinfo(np.uint16).max))
    safe = np.nan_to_num(
        image,
        nan=0.0,
        posinf=float(np.iinfo(np.uint16).max),
        neginf=0.0,
    )
    converted = np.clip(np.rint(safe), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return converted, nonfinite + below + above


def empty_uint16_histogram() -> np.ndarray:
    """Create an exact intensity histogram suitable for streaming accumulation."""
    return np.zeros(UINT16_LEVELS, dtype=np.int64)


def update_uint16_histogram(histogram: np.ndarray, image: np.ndarray) -> None:
    """Add one uint16 image to an existing histogram in-place."""
    if histogram.shape != (UINT16_LEVELS,) or histogram.dtype != np.int64:
        raise ValueError("Histogram must be an int64 array with 65,536 bins.")
    if image.dtype != np.uint16:
        raise ValueError(f"Histogram input must be uint16, received {image.dtype}.")
    histogram += np.bincount(image.reshape(-1), minlength=UINT16_LEVELS)


def histogram_percentile_range(
    histogram: np.ndarray, low_percentile: float, high_percentile: float
) -> tuple[int, int]:
    """Return robust display limits from an accumulated uint16 histogram."""
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100.")
    total = int(histogram.sum())
    if total == 0:
        raise ValueError("Cannot calculate a display range from an empty histogram.")
    cumulative = np.cumsum(histogram)

    def value_at(percentile: float) -> int:
        rank = percentile / 100.0 * (total - 1)
        return int(np.searchsorted(cumulative, rank, side="right"))

    low = value_at(low_percentile)
    high = value_at(high_percentile)
    if high <= low:
        occupied = np.flatnonzero(histogram)
        low, high = int(occupied[0]), int(occupied[-1])
    return low, high


def image_percentile_range(
    image: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> tuple[int, int]:
    """Return robust display limits for a single image."""
    if image.dtype == np.uint16:
        histogram = empty_uint16_histogram()
        update_uint16_histogram(histogram, image)
        return histogram_percentile_range(histogram, low_percentile, high_percentile)
    low, high = np.percentile(image, [low_percentile, high_percentile])
    if high <= low:
        high = low + 1
    return int(round(float(low))), int(round(float(high)))


def split_equal_width(image: np.ndarray, count: int) -> list[np.ndarray]:
    """Split a stitched/contact-sheet image back into equal-width tiles."""
    if count <= 0 or image.shape[1] % count:
        raise ValueError("Image width must divide evenly by count.")
    width = image.shape[1] // count
    return [image[:, i * width : (i + 1) * width] for i in range(count)]


def display_scale(image: np.ndarray, low_value: float, high_value: float) -> np.ndarray:
    """Linearly map fixed intensity limits to uint8 for PNG visualization only."""
    if high_value <= low_value:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = np.clip(
        (image.astype(np.float32) - low_value) / (high_value - low_value),
        0.0,
        1.0,
    )
    return np.round(scaled * 255).astype(np.uint8)


def colorize_scaled(
    image: np.ndarray,
    low_value: float,
    high_value: float,
    color: tuple[float, float, float],
) -> np.ndarray:
    """Map one grayscale image into an RGB channel-colored preview."""
    scaled = display_scale(image, low_value, high_value).astype(np.float32) / 255.0
    rgb = scaled[..., None] * np.asarray(color, dtype=np.float32)
    return np.round(np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def merge_rgb(
    images: list[np.ndarray],
    colors: list[tuple[float, float, float]],
    display_ranges: list[tuple[float, float]],
) -> np.ndarray:
    """Create an additive RGB merge using one fixed range per channel."""
    if not (len(images) == len(colors) == len(display_ranges)):
        raise ValueError("Images, colors, and display ranges must have equal lengths.")
    rgb = np.zeros((*images[0].shape, 3), dtype=np.float32)
    for image, color, limits in zip(images, colors, display_ranges):
        scaled = display_scale(image, *limits).astype(np.float32) / 255.0
        rgb += scaled[..., None] * np.asarray(color, dtype=np.float32)
    return np.round(np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
