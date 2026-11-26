"""Compare illumination-correction policies on a subset of September timepoints.

Four policies are evaluated on identical measurements:

``raw``      background-subtracted, no illumination correction;
``ffc``      calibration-derived 2-D flat field only;
``overlap``  overlap-derived camera-X quadratic only (July method);
``hybrid``   calibration FFC followed by a residual overlap quadratic.

Every correction compared here is either the fixed 2-D flat field or a
camera-X curve. An X-only curve is constant within a column, so dividing a tile
by it commutes with the robust column median over Y and with column-wise overlap
log-ratios. The expensive full-resolution pass therefore stores, for every
timepoint, channel, Z, and position, the robust Y profile and the overlap
log-ratios of the raw and FFC planes. All four policies, the held-out seam
tests, and the calibration rebuild are then exact operations on those arrays.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Any

import nd2
import numpy as np
from scipy.integrate import trapezoid

from .calibrated_pipeline import (
    _calibration_background_and_flatfields,
    _channel_indices,
    _discover_standards,
)
from .config import AnalysisConfig
from .nd2_source import ND2Source
from .outputs import write_json
from .processing import (
    block_median,
    despike_image,
    fit_overlap_log_quadratic,
    overlap_log_ratio_samples,
    physical_x_axes_mm,
    quadratic_illumination_profile,
    robust_y_profile,
    smooth_z_coefficients,
    subtract_background_floor,
)

METHODS = ("raw", "ffc", "overlap", "hybrid")
METHOD_BASE = {"raw": "raw", "ffc": "ffc", "overlap": "raw", "hybrid": "ffc"}
METHOD_LABELS = {
    "raw": "No correction",
    "ffc": "Calibration FFC only",
    "overlap": "Overlap quadratic only",
    "hybrid": "Hybrid: FFC + residual overlap",
}
METHOD_COLORS = {"raw": "#7f7f7f", "ffc": "#1f77b4", "overlap": "#ff7f0e", "hybrid": "#2ca02c"}
FITTED_METHODS = ("overlap", "hybrid")
LINEAR_CALIBRATION_MAX_UG_ML = 24.0
PROFILE_UNITS = {
    "concentration": ("Dextran concentration (µg/mL)", "µg/mL/mm"),
    "normalized": ("Normalized intensity (channel maximum = 1)", "per mm"),
    "intensity": ("Z-integrated intensity (a.u.·µm)", "a.u.·µm/mm"),
}


# ---------------------------------------------------------------------------
# Pure numerical helpers (unit tested with synthetic data)
# ---------------------------------------------------------------------------


def log_illumination(width: int, linear: float, quadratic: float) -> np.ndarray:
    """Log of the positive median-one camera-X illumination profile."""
    return np.log(quadratic_illumination_profile(width, float(linear), float(quadratic)))


def correct_ratios(ratio: np.ndarray, x_offset: int, log_profile: np.ndarray) -> np.ndarray:
    """Remove an X-illumination model from NaN-padded column log-ratios.

    ``ratio[j]`` is log(left[j + x_offset] / right[j]); dividing both tiles by
    the same illumination profile subtracts ``log p(j + x_offset) - log p(j)``.
    """
    columns = np.arange(ratio.size)
    valid = columns + x_offset < log_profile.size
    corrected = np.full(ratio.shape, np.nan, dtype=np.float64)
    corrected[valid] = ratio[valid] - (log_profile[columns[valid] + x_offset] - log_profile[columns[valid]])
    return corrected


def pair_samples(
    ratios: np.ndarray, x_offsets: list[int], pairs: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate finite overlap samples for the selected pairs.

    ``ratios`` has shape (pairs, padded overlap width).
    """
    xl: list[np.ndarray] = []
    xr: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for pair in pairs:
        row = ratios[pair]
        keep = np.flatnonzero(np.isfinite(row))
        xr.append(keep.astype(np.float64))
        xl.append(keep.astype(np.float64) + x_offsets[pair])
        values.append(row[keep])
    if not values:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty
    return np.concatenate(xl), np.concatenate(xr), np.concatenate(values)


def fit_z_series(
    samples_by_z: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    width: int,
    penalty: float,
) -> dict[str, np.ndarray]:
    """Fit one overlap quadratic per Z, quality-weight, and smooth across Z.

    This reproduces the July all-Z policy: weights increase with the number of
    overlap samples and decrease with the squared robust residual.
    """
    z_count = len(samples_by_z)
    raw_linear = np.full(z_count, np.nan)
    raw_quadratic = np.full(z_count, np.nan)
    counts = np.zeros(z_count)
    residuals = np.full(z_count, np.nan)
    for z, (xl, xr, values) in enumerate(samples_by_z):
        if values.size < 3:
            continue
        try:
            _, model = fit_overlap_log_quadratic(width, xl, xr, values)
        except (ValueError, np.linalg.LinAlgError):
            continue
        raw_linear[z] = float(model["linear_coefficient"])
        raw_quadratic[z] = float(model["quadratic_coefficient"])
        counts[z] = float(model["sample_count"])
        residuals[z] = float(model["median_abs_log_residual"])
    valid = np.isfinite(raw_linear) & np.isfinite(raw_quadratic) & (counts > 0)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Fewer than two valid per-Z overlap models.")
    count_scale = max(float(np.median(counts[valid])), 1.0)
    residual_scale = max(float(np.median(residuals[valid])), 1e-6)
    quality = np.zeros(z_count)
    quality[valid] = (counts[valid] / count_scale) / np.maximum(residuals[valid] / residual_scale, 0.25) ** 2
    linear, weights = smooth_z_coefficients(raw_linear, quality, penalty)
    quadratic, _ = smooth_z_coefficients(raw_quadratic, quality, penalty)
    return {
        "raw_linear": raw_linear,
        "raw_quadratic": raw_quadratic,
        "linear": linear,
        "quadratic": quadratic,
        "quality_weight": weights,
        "sample_count": counts,
        "fit_median_abs_log_residual": residuals,
    }


def median_abs(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(np.abs(finite))) if finite.size else float("nan")


def profile_pair_mismatch(
    left_profile: np.ndarray, right_profile: np.ndarray, x_offset: int, minimum: float
) -> float:
    """Median absolute log mismatch of two 1-D tile profiles in their overlap."""
    width = left_profile.size
    left = left_profile[x_offset:width]
    right = right_profile[: width - x_offset]
    valid = np.isfinite(left) & np.isfinite(right) & (left > minimum) & (right > minimum)
    if not np.any(valid):
        return float("nan")
    return float(np.median(np.abs(np.log(left[valid] / right[valid]))))


def calibration_model_from_blocks(
    concentrations: list[float],
    block_stacks: list[np.ndarray],
    z_steps_um: list[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Integrate block medians through Z and build the empirical conversion.

    ``block_stacks[i]`` has shape (Z, blocks); the robust retained-block median
    matches the calibrated pilot.
    """
    integrated_values = [
        trapezoid(np.asarray(stack, dtype=np.float64), dx=z_step, axis=0)
        for stack, z_step in zip(block_stacks, z_steps_um)
    ]
    return calibration_model_from_block_values(concentrations, integrated_values)


def calibration_model_from_block_values(
    concentrations: list[float], block_values: list[np.ndarray]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Robust retained-block median per standard and the empirical conversion.

    ``block_values[i]`` holds one value per spatial block (already reduced over
    Z: integrated, a single plane, or a mean of planes).
    """
    rows: list[dict[str, Any]] = []
    for concentration, integrated in zip(concentrations, block_values):
        integrated = np.asarray(integrated, dtype=np.float64)
        center = float(np.nanmedian(integrated))
        mad = float(np.nanmedian(np.abs(integrated - center)))
        tolerance = max(6.0 * 1.4826 * mad, np.finfo(float).eps)
        valid = np.isfinite(integrated) & (np.abs(integrated - center) <= tolerance)
        retained = integrated[valid]
        if retained.size < max(4, integrated.size // 2):
            raise ValueError(f"Too few valid calibration blocks at {concentration:g} µg/mL.")
        rows.append(
            {
                "concentration_ug_ml": float(concentration),
                "standard_signal": float(np.median(retained)),
                "block_q25": float(np.percentile(retained, 25)),
                "block_q75": float(np.percentile(retained, 75)),
                "retained_block_count": int(retained.size),
                "excluded_block_count": int(integrated.size - retained.size),
            }
        )
    rows.sort(key=lambda row: row["concentration_ug_ml"])
    concentration_values = np.asarray([row["concentration_ug_ml"] for row in rows])
    signals = np.asarray([row["standard_signal"] for row in rows])
    coefficients = np.polyfit(concentration_values, signals, 1)
    predicted = np.polyval(coefficients, concentration_values)
    ss_total = float(np.sum((signals - signals.mean()) ** 2))
    model = {
        "concentrations_ug_ml": concentration_values.tolist(),
        "signals": signals.tolist(),
        "strictly_monotonic": bool(np.all(np.diff(signals) > 0)),
        "linear_diagnostic": {
            "signal_intercept_au_um": float(coefficients[1]),
            "signal_per_ug_ml": float(coefficients[0]),
            "r_squared": 1.0 - float(np.sum((signals - predicted) ** 2)) / ss_total,
            "residuals_au_um": (signals - predicted).tolist(),
        },
    }
    return rows, model


def bin_mean(image: np.ndarray, factor: int) -> np.ndarray:
    height = image.shape[0] // factor * factor
    width = image.shape[1] // factor * factor
    return (
        image[:height, :width]
        .reshape(height // factor, factor, width // factor, factor)
        .mean(axis=(1, 3), dtype=np.float64)
        .astype(np.float32)
    )


# ---------------------------------------------------------------------------
# Worker processes (one full-resolution pass)
# ---------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _init_worker(state: dict[str, Any]) -> None:
    import dask

    dask.config.set(scheduler="synchronous")
    _WORKER.clear()
    _WORKER.update(state)
    _WORKER["files"] = {}


def _worker_gradient_source() -> ND2Source:
    source = _WORKER.get("gradient_source")
    if source is None:
        source = ND2Source(_WORKER["gradient_path"], _WORKER["config"]).__enter__()
        _WORKER["gradient_source"] = source
    return source


def _worker_standard(path: str) -> tuple[Any, dict[str, int]]:
    files = _WORKER["files"]
    if path not in files:
        handle = nd2.ND2File(path)
        files[path] = (handle, handle.to_dask(), _channel_indices(handle, _WORKER["config"].channels))
    handle, data, indices = files[path]
    return data, indices


def _measure_standard_plane(job: tuple[str, str, int]) -> dict[str, Any]:
    path, label, z = job
    data, indices = _worker_standard(path)
    acquired = np.asarray(data[z, indices[label]].compute())
    clipped = acquired == np.iinfo(acquired.dtype).max
    repaired, spike_mask = despike_image(acquired, _WORKER["hot_masks"][label])
    signal = subtract_background_floor(repaired, _WORKER["backgrounds"][label])
    corrected = signal / _WORKER["flatfields"][label]
    signal[clipped] = np.nan
    corrected[clipped] = np.nan
    block = int(_WORKER["config"].calibration_block_size_px)
    return {
        "job": job,
        "raw_blocks": block_median(signal, block).astype(np.float64),
        "ffc_blocks": block_median(corrected, block).astype(np.float64),
        "clipped": int(np.count_nonzero(clipped)),
        "despiked": int(np.count_nonzero(spike_mask)),
    }


def _measure_gradient_z(job: tuple[int, str, int]) -> dict[str, Any]:
    timepoint, label, z = job
    config: AnalysisConfig = _WORKER["config"]
    source = _worker_gradient_source()
    channel = next(item for item in config.channels if item.label == label)
    flatfield = _WORKER["flatfields"][label]
    background = _WORKER["backgrounds"][label]
    factor = int(_WORKER["bin_factor"])
    position_indices: list[int] = _WORKER["position_indices"]
    geometry: list[tuple[int, int]] = _WORKER["pair_geometry"]
    width_padded = int(_WORKER["overlap_width_padded"])
    bases: tuple[str, ...] = _WORKER["bases"]
    profiles: dict[str, list[np.ndarray]] = {base: [] for base in bases}
    bins: dict[str, list[np.ndarray]] = {base: [] for base in bases}
    tiles: dict[str, list[np.ndarray]] = {base: [] for base in bases}
    stats: list[dict[str, Any]] = []
    for position in position_indices:
        acquired = source.plane(timepoint, position, channel, z)
        clipped = acquired == np.iinfo(acquired.dtype).max
        repaired, spike_mask = despike_image(acquired, _WORKER["hot_masks"][label])
        signal = subtract_background_floor(repaired, background)
        images = {"raw": signal}
        if "ffc" in bases:
            images["ffc"] = signal / flatfield
        record = {
            "position_nd2_index": position,
            "clipped_pixel_count": int(np.count_nonzero(clipped)),
            "despiked_pixel_count": int(np.count_nonzero(spike_mask)),
        }
        for base, image in images.items():
            profile, _, details = robust_y_profile(
                image,
                outlier_sigma=config.profile_outlier_sigma,
                minimum_deviation=config.profile_minimum_deviation,
            )
            profiles[base].append(profile)
            bins[base].append(bin_mean(image, factor))
            tiles[base].append(image)
            record[f"{base}_excluded_fraction"] = details["excluded_pixel_fraction"]
        stats.append(record)
    saturation = float(np.iinfo(np.uint16).max) - background
    ratios = {}
    for base in bases:
        padded = np.full((len(geometry), width_padded), np.nan)
        for pair, (x_offset, y_shift) in enumerate(geometry):
            _, xr, values, _ = overlap_log_ratio_samples(
                tiles[base][pair], tiles[base][pair + 1], x_offset, y_shift,
                config.overlap_min_signal, saturation,
            )
            padded[pair, xr.astype(int)] = values
        ratios[base] = padded
    return {
        "job": job,
        "profiles": {base: np.stack(values) for base, values in profiles.items()},
        "bins": {base: np.stack(values) for base, values in bins.items()},
        "ratios": ratios,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Orchestration: one full-resolution pass, cached per timepoint
# ---------------------------------------------------------------------------


class RunLog:
    def __init__(self, path: Path):
        self.path = path
        self.start = time.time()

    def __call__(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')} +{(time.time() - self.start) / 60:5.1f} min] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _run_pool(jobs, function, state, workers, log, label):
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(state,)) as pool:
        futures = [pool.submit(function, job) for job in jobs]
        step = max(1, len(futures) // 10)
        for done, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if done % step == 0 or done == len(futures):
                log(f"{label}: {done}/{len(futures)}")
    return results


def _find_cached(reuse_dirs: list[Path], name: str) -> Path | None:
    for directory in reuse_dirs:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def collect_measurements(
    nd2_path: str | Path,
    config: AnalysisConfig,
    cache: Path,
    log: RunLog,
    *,
    workers: int = 48,
    bin_factor: int = 8,
    reuse_dirs: list[Path] | None = None,
    flatfield_output: Path | None = None,
) -> dict[str, Any]:
    """Measure (or load cached) robust profiles, overlap ratios, and binned tiles.

    Without a calibration directory (July data) only the background-subtracted
    ``raw`` base is measured, using the constant microscope background.
    """
    reuse_dirs = [Path(item) for item in (reuse_dirs or [])]
    source_path = Path(nd2_path)
    timepoints = tuple(config.timepoints or ())
    labels = [channel.label for channel in config.channels]
    has_calibration = bool(config.calibration_directory)
    bases = ("raw", "ffc") if has_calibration else ("raw",)
    cache.mkdir(parents=True, exist_ok=True)

    with ND2Source(source_path, config) as source:
        sizes = dict(source.sizes)
        if not timepoints:
            timepoints = tuple(range(int(sizes.get("T", 1))))
        invalid = [t for t in timepoints if not 0 <= t < int(sizes.get("T", 1))]
        if invalid:
            raise ValueError(f"Invalid timepoints {invalid}.")
        z_count = int(sizes["Z"])
        height, width = int(sizes["Y"]), int(sizes["X"])
        pixel_um = float(source.pixel_size_um)
        z_step_um = 10.0
        for loop in source._file.experiment:
            if getattr(loop, "type", "") == "ZStackLoop":
                z_step_um = float(loop.parameters.stepUm)
        positions = list(source.positions)
        channel_names = list(source.channel_names)
        all_hours = source.timepoint_hours()
        z_info = source.z_stack_info()
    position_indices = [position.index for position in positions]
    pair_geometry = [
        (
            int(round(abs(right.x_um - left.x_um) / pixel_um)),
            int(round((right.y_um - left.y_um) / pixel_um)),
        )
        for left, right in zip(positions, positions[1:])
    ]
    x_offsets = [item[0] for item in pair_geometry]
    overlap_width_padded = width - min(x_offsets)
    x_axes = physical_x_axes_mm([position.x_um for position in positions], pixel_um, width)
    pixel_mm = pixel_um / 1000.0
    x_origin = min(float(axis[0]) for axis in x_axes)
    x_starts = [(float(axis[0]) - x_origin) / pixel_mm for axis in x_axes]
    y_starts = [(position.y_um - positions[0].y_um) / pixel_um for position in positions]

    standards: list[tuple[float, Path]] = []
    flatfield_diagnostics: dict[str, Any] = {}
    if has_calibration:
        standards = _discover_standards(Path(str(config.calibration_directory)), config.calibration_filename_unit)
        cached = _find_cached(reuse_dirs, "calibration_state.npz")
        if cached is not None:
            log(f"Loading cached background/hot-pixel/flat-field state from {cached.parent}")
            stored = np.load(cached)
            backgrounds = {label: float(stored[f"background_{label}"]) for label in labels}
            hot_masks = {label: stored[f"hot_{label}"] for label in labels}
            flatfields = {label: stored[f"flatfield_{label}"] for label in labels}
            flatfield_diagnostics = json.loads((cached.parent / "flatfield_diagnostics.json").read_text(encoding="utf-8"))
        else:
            log("Estimating blank background, hot pixels, and 32 µg/mL flat field")
            backgrounds, hot_masks, flatfields, flatfield_diagnostics = _calibration_background_and_flatfields(
                standards, config, flatfield_output or cache
            )
        np.savez(
            cache / "calibration_state.npz",
            **{f"background_{label}": backgrounds[label] for label in labels},
            **{f"hot_{label}": hot_masks[label] for label in labels},
            **{f"flatfield_{label}": flatfields[label] for label in labels},
        )
        write_json(cache / "flatfield_diagnostics.json", flatfield_diagnostics)
    else:
        backgrounds = {label: float(config.microscope_background) for label in labels}
        hot_masks = {label: np.zeros((height, width), dtype=bool) for label in labels}
        flatfields = {label: np.ones((height, width), dtype=np.float32) for label in labels}
    log(f"Backgrounds: {backgrounds}")

    state = {
        "config": config, "gradient_path": str(source_path), "backgrounds": backgrounds,
        "hot_masks": hot_masks, "flatfields": flatfields, "bin_factor": bin_factor,
        "position_indices": position_indices, "pair_geometry": pair_geometry,
        "overlap_width_padded": overlap_width_padded, "bases": bases,
    }
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"

    standard_paths = [str(path) for _, path in standards]
    standard_z: dict[str, tuple[int, float]] = {}
    standard_blocks: dict[str, np.ndarray] = {}
    standard_qc: dict[str, Any] = {}
    if has_calibration:
        for path in standard_paths:
            with nd2.ND2File(path) as handle:
                standard_z[path] = (int(handle.sizes.get("Z", 1)), float(handle.voxel_size().z))
        cached = _find_cached(reuse_dirs, "standards.npz")
        if cached is not None:
            stored = np.load(cached)
            standard_blocks = {base: stored[base] for base in ("raw", "ffc")}
            standard_qc = json.loads((cached.parent / "standards_qc.json").read_text(encoding="utf-8"))
            log("Loaded cached calibration-standard measurements")
        else:
            jobs = [(path, label, z) for path in standard_paths for label in labels for z in range(standard_z[path][0])]
            log(f"Measuring calibration standards: {len(jobs)} planes")
            results = _run_pool(jobs, _measure_standard_plane, state, workers, log, "standards")
            max_z = max(item[0] for item in standard_z.values())
            sample = results[0]["raw_blocks"].size
            standard_blocks = {base: np.full((len(standard_paths), len(labels), max_z, sample), np.nan) for base in ("raw", "ffc")}
            standard_qc = {f"{Path(path).name}|{label}": {"clipped": 0, "despiked": 0} for path in standard_paths for label in labels}
            for result in results:
                path, label, z = result["job"]
                s, c = standard_paths.index(path), labels.index(label)
                standard_blocks["raw"][s, c, z] = result["raw_blocks"].reshape(-1)
                standard_blocks["ffc"][s, c, z] = result["ffc_blocks"].reshape(-1)
                standard_qc[f"{Path(path).name}|{label}"]["clipped"] += result["clipped"]
                standard_qc[f"{Path(path).name}|{label}"]["despiked"] += result["despiked"]
        np.savez(cache / "standards.npz", **standard_blocks)
        write_json(cache / "standards_qc.json", standard_qc)

    T, C, Z, P, K = len(timepoints), len(labels), z_count, len(positions), len(pair_geometry)
    binned_shape = (height // bin_factor, width // bin_factor)
    profiles = {base: np.empty((T, C, Z, P, width)) for base in bases}
    ratios = {base: np.full((T, C, Z, K, overlap_width_padded), np.nan) for base in bases}
    bins = {base: np.empty((T, C, Z, P, *binned_shape), dtype=np.float32) for base in bases}
    plane_rows: list[dict[str, Any]] = []
    for ti, timepoint in enumerate(timepoints):
        name = f"gradient_t{timepoint:03d}.npz"
        cached = _find_cached(reuse_dirs, name)
        if cached is not None:
            stored = np.load(cached)
            for base in bases:
                profiles[base][ti] = stored[f"profiles_{base}"]
                ratios[base][ti] = stored[f"ratios_{base}"]
                bins[base][ti] = stored[f"bins_{base}"]
            rows_t = json.loads((cached.parent / f"gradient_t{timepoint:03d}_qc.json").read_text(encoding="utf-8"))
            log(f"t{timepoint:03d}: loaded cached measurements from {cached.parent}")
        else:
            jobs = [(timepoint, label, z) for label in labels for z in range(z_count)]
            log(f"t{timepoint:03d} ({all_hours[timepoint]:.2f} h): measuring {len(jobs) * P} planes")
            results = _run_pool(jobs, _measure_gradient_z, state, workers, log, f"t{timepoint:03d}")
            rows_t = []
            for result in results:
                _, label, z = result["job"]
                c = labels.index(label)
                for base in bases:
                    profiles[base][ti, c, z] = result["profiles"][base]
                    ratios[base][ti, c, z] = result["ratios"][base]
                    bins[base][ti, c, z] = result["bins"][base]
                for position, record in zip(positions, result["stats"]):
                    rows_t.append({
                        "timepoint": timepoint, "channel": label, "z_index_zero_based": z,
                        "z_index_one_based": z + 1, "position": position.name, **record,
                    })
            np.savez(
                cache / name,
                **{f"profiles_{base}": profiles[base][ti] for base in bases},
                **{f"ratios_{base}": ratios[base][ti] for base in bases},
                **{f"bins_{base}": bins[base][ti] for base in bases},
            )
            write_json(cache / f"gradient_t{timepoint:03d}_qc.json", rows_t)
        plane_rows.extend(rows_t)
    write_json(cache / "cache_sources.json", {"reused_from": [str(path) for path in reuse_dirs]})
    log("Full-resolution measurements available")
    return {
        "source_path": source_path, "timepoints": timepoints, "labels": labels, "channel_names": channel_names,
        "time_h": [all_hours[t] if t < len(all_hours) else t * config.experimental_hours_per_timepoint for t in timepoints],
        "z_info": z_info,
        "has_calibration": has_calibration, "bases": bases, "sizes": sizes,
        "z_count": z_count, "height": height, "width": width, "pixel_um": pixel_um, "z_step_um": z_step_um,
        "positions": positions, "pair_geometry": pair_geometry, "x_offsets": x_offsets,
        "x_axes": x_axes, "x_starts": x_starts, "y_starts": y_starts,
        "backgrounds": backgrounds, "flatfields": flatfields, "flatfield_diagnostics": flatfield_diagnostics,
        "standards": standards, "standard_paths": standard_paths, "standard_z": standard_z,
        "standard_blocks": standard_blocks, "standard_qc": standard_qc,
        "profiles": profiles, "ratios": ratios, "bins": bins, "binned_shape": binned_shape,
        "bin_factor": bin_factor, "plane_rows": plane_rows,
    }
