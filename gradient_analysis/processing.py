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


def flatten_segments(
    x_segments: list[np.ndarray],
    y_segments: list[np.ndarray],
    start_index: int = 0,
    end_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate selected measured profile segments for fitting."""
    if end_index is None:
        end_index = len(x_segments) - 1
    if not (0 <= start_index <= end_index < len(x_segments) == len(y_segments)):
        raise ValueError("Invalid segment range for flattening.")
    return (
        np.concatenate(x_segments[start_index : end_index + 1]),
        np.concatenate(y_segments[start_index : end_index + 1]),
    )


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


def position_span(position_list_index: int, tile_width: int) -> tuple[int, int]:
    """Return the stitched x start/end pixel coordinates for one position."""
    if position_list_index < 0:
        raise ValueError("Position list index must be non-negative.")
    if tile_width <= 0:
        raise ValueError("Tile width must be positive.")
    start = int(position_list_index) * int(tile_width)
    return start, start + int(tile_width)


def linear_fit_xy(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Fit y = slope*x + intercept and report slope plus R²."""
    if x.size != y.size or x.size < 2:
        raise ValueError("x and y must have equal length >= 2.")
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r_squared = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
    }


def linear_fit(profile: np.ndarray, pixel_size_um: float | None = None) -> dict[str, float | None]:
    """Fit y = slope*x + intercept using pixel coordinate x and report R²."""
    y = profile.astype(np.float64)
    x = np.arange(y.size, dtype=np.float64)
    fit = linear_fit_xy(x, y)
    slope = fit["slope"]
    slope_per_um = None if not pixel_size_um else float(slope / pixel_size_um)
    return {
        "slope_per_pixel": float(slope),
        "slope_per_um": slope_per_um,
        "intercept": fit["intercept"],
        "r_squared": fit["r_squared"],
    }


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
