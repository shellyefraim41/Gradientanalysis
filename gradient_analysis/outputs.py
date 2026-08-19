"""Image and graph output helpers."""

from __future__ import annotations

import json
import os
import csv
from pathlib import Path

# Keep Matplotlib's font cache writable and local to this project on managed
# Windows systems. This must be set before importing matplotlib.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib_cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image

from .config import ChannelConfig
from .processing import colorize_scaled, display_scale, normalized_profile_shape, smooth_profile


def _segments(profile: object) -> list[tuple[np.ndarray, np.ndarray]]:
    """Normalize legacy arrays and physical profile segments for plotting."""
    if isinstance(profile, np.ndarray):
        return [(np.arange(profile.size, dtype=np.float64), profile)]
    return [(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)) for x, y in profile]  # type: ignore[arg-type]


def _plot_measured_segments(
    ax: object,
    segments: list[tuple[np.ndarray, np.ndarray]],
    color: object,
    label: str,
    linewidth: float,
    alpha: float = 0.9,
) -> None:
    for i, (x, y) in enumerate(segments):
        ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, label=label if i == 0 else None)


def _plot_gap_trendlines(
    ax: object,
    segments: list[tuple[np.ndarray, np.ndarray]],
    color: object,
    linewidth: float = 1.4,
) -> None:
    """Draw dashed visual interpolation only between measured positions."""
    for left, right in zip(segments, segments[1:]):
        lx, ly = left
        rx, ry = right
        ax.plot(
            [lx[-1], rx[0]],
            [ly[-1], ry[0]],
            color=color,
            linestyle="--",
            linewidth=linewidth,
            alpha=0.85,
        )


def save_tiff(path: Path, image: np.ndarray) -> None:
    """Write uint16 analysis data without display rescaling."""
    if image.dtype != np.uint16:
        raise ValueError(f"TIFF output must be uint16, received {image.dtype}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, image, photometric="minisblack")


def save_preview(path: Path, image: np.ndarray, low_value: float, high_value: float) -> None:
    """Save an 8-bit grayscale display copy; not for measurement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(display_scale(image, low_value, high_value)).save(path)


def save_color_preview(
    path: Path,
    image: np.ndarray,
    low_value: float,
    high_value: float,
    color: tuple[float, float, float],
) -> None:
    """Save an RGB display copy using the channel color; not for measurement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(colorize_scaled(image, low_value, high_value, color), mode="RGB").save(path)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def save_profile_plot(
    path: Path,
    profiles: dict[str, np.ndarray],
    channels: tuple[ChannelConfig, ...],
    title: str,
    trendline_window: int | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    for channel in channels:
        segments = _segments(profiles[channel.label])
        _plot_measured_segments(ax, segments, channel.plot_color, channel.label, linewidth=1.2)
        _plot_gap_trendlines(ax, segments, channel.plot_color)
        if trendline_window and len(segments) == 1:
            profile = segments[0][1]
            ax.plot(
                segments[0][0],
                smooth_profile(profile, trendline_window),
                color=channel.plot_color,
                linestyle="--",
                linewidth=1.8,
                alpha=0.9,
                label=f"{channel.label} smooth trend",
            )
    ax.set(title=title, xlabel="Position along device (mm)", ylabel="Mean intensity (a.u.)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_normalization_plot(
    path: Path,
    raw_profiles: list[np.ndarray],
    corrected_profiles: list[np.ndarray],
    fitted_laser_profile: np.ndarray,
    reference_list_index: int,
    position_labels: list[str],
    channel: ChannelConfig,
    title: str,
    corrected_physical_segments: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    plot_physical = corrected_physical_segments is not None
    rows = 3 if plot_physical else 2
    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(12, 11 if plot_physical else 8),
        constrained_layout=True,
    )
    for i, (raw, corrected) in enumerate(zip(raw_profiles, corrected_profiles)):
        suffix = " (reference)" if i == reference_list_index else ""
        axes[0].plot(raw, label=position_labels[i] + suffix, alpha=0.85)
        axes[1].plot(corrected, label=position_labels[i] + suffix, alpha=0.85)
    axes[0].plot(
        fitted_laser_profile,
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="quadratic laser trendline",
    )
    axes[0].set(title=f"{title} — raw local-x profiles", ylabel="Mean intensity (a.u.)")
    axes[1].set(
        title="After quadratic trendline correction",
        xlabel="Local x coordinate (pixels)",
        ylabel="Corrected mean intensity",
    )
    if plot_physical:
        segments = _segments(corrected_physical_segments)
        for i, (x, corrected) in enumerate(segments):
            axes[2].plot(x, corrected, label=position_labels[i], alpha=0.85)
        _plot_gap_trendlines(axes[2], segments, channel.plot_color)
        axes[2].set(
            title="Same corrected values placed at true physical stage positions",
            xlabel="Position along device (mm)",
            ylabel="Corrected mean intensity",
        )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(ncol=4, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_timecourse_plot(
    path: Path,
    profiles: dict[int, np.ndarray],
    channel: ChannelConfig,
    title: str,
    trendline_window: int | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(18, 8), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.76, top=0.9, bottom=0.18)
    colors = plt.colormaps["viridis"](np.linspace(0.0, 1.0, len(profiles)))
    for color, (timepoint, profile) in zip(colors, sorted(profiles.items())):
        segments = _segments(profile)
        _plot_measured_segments(ax, segments, color, f"t={timepoint}", linewidth=0.9, alpha=0.85)
        _plot_gap_trendlines(ax, segments, color, linewidth=1.2)
        if trendline_window and len(segments) == 1:
            ax.plot(
                segments[0][0],
                smooth_profile(segments[0][1], trendline_window),
                color=color,
                linestyle="--",
                linewidth=1.6,
                alpha=0.95,
            )
    ax.set(title=title, xlabel="Position along device (mm)", ylabel="Mean intensity (a.u.)")
    ax.grid(alpha=0.2)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=8,
        fontsize=7,
        frameon=True,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_max_difference_timecourse_plot(
    path: Path,
    records: list[dict[str, object]],
    channels: tuple[ChannelConfig, ...],
    title: str,
) -> None:
    """Plot both channel-specific P02/P05 max differences over time."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for channel in channels:
        selected = sorted(
            [record for record in records if record["channel"] == channel.label],
            key=lambda record: int(record["timepoint"]),
        )
        ax.plot(
            [int(record["timepoint"]) for record in selected],
            [float(record["max_intensity_difference"]) for record in selected],
            marker="o",
            color=channel.plot_color,
            label=f"{channel.label} {selected[0]['formula']}" if selected else channel.label,
        )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set(title=title, xlabel="Timepoint", ylabel="Background-subtracted max difference (a.u.)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_reference_selection_plot(
    path: Path,
    profiles_by_timepoint: list[tuple[int, np.ndarray]],
    channel: ChannelConfig,
    position_label: str,
    title: str,
    smoothing_window: int,
) -> None:
    """Plot normalized profile shapes for the selected correction reference position."""
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    colors = plt.colormaps["viridis"](np.linspace(0.0, 1.0, len(profiles_by_timepoint)))
    for color, (timepoint, profile) in zip(colors, profiles_by_timepoint):
        ax.plot(
            normalized_profile_shape(profile, smoothing_window),
            color=color,
            linewidth=1.2,
            alpha=0.9,
            label=f"t={timepoint}",
        )
    ax.set(
        title=title,
        xlabel=f"{position_label} local x coordinate (pixels)",
        ylabel="Smoothed profile / median",
    )
    ax.grid(alpha=0.2)
    ax.legend(ncol=6, fontsize=7)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_correction_method_comparison_plot(
    path: Path,
    raw_profiles: list[np.ndarray],
    quadratic_profiles: list[np.ndarray],
    smoothed_profiles: list[np.ndarray],
    quadratic_laser_profile: np.ndarray,
    smoothed_laser_profile: np.ndarray,
    reference_list_index: int,
    position_labels: list[str],
    channel: ChannelConfig,
    title: str,
) -> None:
    """Compare quadratic and smoothed-profile illumination correction on local x."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    for i, (raw, quadratic, smoothed) in enumerate(zip(raw_profiles, quadratic_profiles, smoothed_profiles)):
        suffix = " (reference)" if i == reference_list_index else ""
        label = position_labels[i] + suffix
        axes[0].plot(raw, label=label, alpha=0.85)
        axes[1].plot(quadratic, label=label, alpha=0.85)
        axes[2].plot(smoothed, label=label, alpha=0.85)
    axes[0].plot(
        quadratic_laser_profile,
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="quadratic fitted laser profile",
    )
    axes[0].plot(
        smoothed_laser_profile,
        color=channel.plot_color,
        linestyle=":",
        linewidth=2.0,
        label="smoothed fitted laser profile",
    )
    axes[0].set(title=f"{title} - raw local-x profiles", ylabel="Mean intensity (a.u.)")
    axes[1].set(title="After quadratic correction", ylabel="Corrected mean intensity")
    axes[2].set(
        title="After smoothed-profile correction",
        xlabel="Local x coordinate (pixels)",
        ylabel="Corrected mean intensity",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(ncol=4, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_flatfield_2d_comparison_plot(
    path: Path,
    raw_profiles: list[np.ndarray],
    quadratic_profiles: list[np.ndarray],
    smoothed_1d_profiles: list[np.ndarray],
    flatfield_2d_profiles: list[np.ndarray],
    quadratic_laser_profile: np.ndarray,
    smoothed_1d_laser_profile: np.ndarray,
    flatfield_2d_map: np.ndarray,
    reference_list_index: int,
    position_labels: list[str],
    channel: ChannelConfig,
    title: str,
) -> None:
    """Compare 1-D and 2-D illumination correction methods on local x profiles."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 15), constrained_layout=True)
    for i, (raw, quadratic, smoothed_1d, flatfield_2d) in enumerate(
        zip(raw_profiles, quadratic_profiles, smoothed_1d_profiles, flatfield_2d_profiles)
    ):
        suffix = " (reference)" if i == reference_list_index else ""
        label = position_labels[i] + suffix
        axes[0].plot(raw, label=label, alpha=0.85)
        axes[1].plot(quadratic, label=label, alpha=0.85)
        axes[2].plot(smoothed_1d, label=label, alpha=0.85)
        axes[3].plot(flatfield_2d, label=label, alpha=0.85)
    axes[0].plot(
        quadratic_laser_profile,
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="quadratic fitted laser profile",
    )
    axes[0].plot(
        smoothed_1d_laser_profile,
        color=channel.plot_color,
        linestyle=":",
        linewidth=2.0,
        label="smoothed 1-D fitted laser profile",
    )
    axes[0].plot(
        np.mean(flatfield_2d_map, axis=0),
        color="tab:orange",
        linestyle="-.",
        linewidth=2.0,
        label="2-D flat-field x mean",
    )
    axes[0].set(title=f"{title} - raw local-x profiles", ylabel="Mean intensity (a.u.)")
    axes[1].set(title="After quadratic 1-D correction", ylabel="Corrected mean intensity")
    axes[2].set(title="After smoothed-profile 1-D correction", ylabel="Corrected mean intensity")
    axes[3].set(
        title="After smoothed 2-D flat-field correction",
        xlabel="Local x coordinate (pixels)",
        ylabel="Corrected mean intensity",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(ncol=4, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a generic table using the keys from the first row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_overlap_correction_plot(
    path: Path,
    illumination_profile: np.ndarray,
    channel: ChannelConfig,
    title: str,
) -> None:
    """Plot the fitted illumination and reciprocal correction curves."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(illumination_profile.size)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].plot(x, illumination_profile, color=channel.plot_color, linewidth=2)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set(title=title, ylabel="Relative illumination")
    axes[1].plot(x, 1.0 / illumination_profile, color=channel.plot_color, linewidth=2)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set(xlabel="Local camera x (pixels)", ylabel="Correction factor")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_overlap_residual_plot(
    path: Path,
    rows: list[dict[str, object]],
    channel: ChannelConfig,
    title: str,
) -> None:
    """Plot median absolute overlap mismatch before and after correction."""
    selected = [row for row in rows if row.get("channel") == channel.label]
    labels = [f"t{int(row['timepoint']):03d} {row['position_pair']}" for row in selected]
    before = [float(row["raw_median_abs_log_ratio"]) for row in selected]
    after = [float(row["corrected_median_abs_log_residual"]) for row in selected]
    x = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(max(10, len(selected) * 0.08), 5), constrained_layout=True)
    ax.plot(x, before, color="gray", alpha=0.65, linewidth=0.8, label="Before")
    ax.plot(x, after, color=channel.plot_color, alpha=0.85, linewidth=0.8, label="After")
    tick_step = max(1, len(labels) // 18)
    ticks = x[::tick_step]
    ax.set_xticks(ticks, [labels[i] for i in ticks], rotation=70, ha="right", fontsize=7)
    ax.set(title=title, ylabel="Median absolute log-ratio", xlabel="Overlap sample")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_continuous_profile_plot(
    path: Path,
    profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    channels: tuple[ChannelConfig, ...],
    title: str,
    *,
    normalized: bool,
) -> None:
    """Plot one feathered physical profile per channel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    for channel in channels:
        x, y = profiles[channel.label]
        ax.plot(x, y, color=channel.plot_color, linewidth=1.2, label=channel.label)
    ax.set(
        title=title,
        xlabel="Position along device (mm)",
        ylabel="Normalized intensity" if normalized else "Mean intensity (a.u.)",
    )
    if normalized:
        ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_continuous_timecourse_plot(
    path: Path,
    profiles: dict[int, tuple[np.ndarray, np.ndarray]],
    channel: ChannelConfig,
    title: str,
    *,
    normalized: bool,
) -> None:
    """Plot feathered profiles for all selected timepoints without slope tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
    colors = plt.colormaps["viridis"](np.linspace(0.0, 1.0, len(profiles)))
    for color, (timepoint, (x, y)) in zip(colors, sorted(profiles.items())):
        ax.plot(x, y, color=color, linewidth=0.9, alpha=0.85, label=f"t={timepoint}")
    ax.set(
        title=title,
        xlabel="Position along device (mm)",
        ylabel="Normalized intensity" if normalized else "Mean intensity (a.u.)",
    )
    if normalized:
        ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.2)
    ax.legend(ncol=8, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_tanh_fit_plot(
    path: Path,
    profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    fit_records: dict[str, dict[str, object]],
    channels: tuple[ChannelConfig, ...],
    title: str,
) -> None:
    """Plot normalized feathered profiles with their fitted tanh curves."""
    from .processing import tanh_profile

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    for channel in channels:
        x, y = profiles[channel.label]
        record = fit_records[channel.label]
        ax.plot(x, y, color=channel.plot_color, linewidth=0.8, alpha=0.55, label=f"{channel.label} data")
        fitted = tanh_profile(
            x,
            float(record["left_plateau"]),
            float(record["right_plateau"]),
            float(record["midpoint_mm"]),
            float(record["width_mm"]),
        )
        slope = float(record["signed_slope_per_mm"])
        ax.plot(x, fitted, color=channel.plot_color, linestyle="--", linewidth=2.2, label=f"{channel.label} tanh; slope={slope:.4g}/mm")
        ax.axvline(float(record["midpoint_mm"]), color=channel.plot_color, linestyle=":", alpha=0.5)
    ax.set(title=title, xlabel="Position along device (mm)", ylabel="Normalized intensity", ylim=(-0.02, 1.02))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_fit_metric_timecourse_plot(
    path: Path,
    records: list[dict[str, object]],
    channels: tuple[ChannelConfig, ...],
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for channel in channels:
        selected = sorted(
            (record for record in records if record["channel"] == channel.label),
            key=lambda record: int(record["timepoint"]),
        )
        ax.plot(
            [int(record["timepoint"]) for record in selected],
            [float(record[metric]) for record in selected],
            marker="o",
            color=channel.plot_color,
            label=channel.label,
        )
    ax.set(title=title, xlabel="Timepoint", ylabel=ylabel)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_fit_metric_by_z_plot(
    path: Path,
    records: list[dict[str, object]],
    channels: tuple[ChannelConfig, ...],
    timepoint: int,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for channel in channels:
        selected = sorted(
            (record for record in records if record["channel"] == channel.label and int(record["timepoint"]) == timepoint),
            key=lambda record: int(record["z_index_zero_based"]),
        )
        ax.plot(
            [int(record["z_index_one_based"]) for record in selected],
            [float(record[metric]) for record in selected],
            marker="o",
            color=channel.plot_color,
            label=channel.label,
        )
    ax.set(title=title, xlabel="Z plane (one-based)", ylabel=ylabel)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_fit_heatmap(
    path: Path,
    records: list[dict[str, object]],
    channel: ChannelConfig,
    timepoints: tuple[int, ...],
    z_count: int,
    metric: str,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.full((len(timepoints), z_count), np.nan, dtype=float)
    timepoint_index = {timepoint: index for index, timepoint in enumerate(timepoints)}
    for record in records:
        if record["channel"] == channel.label:
            values[timepoint_index[int(record["timepoint"])], int(record["z_index_zero_based"])] = float(record[metric])
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", origin="lower", cmap="coolwarm" if metric == "signed_slope_per_mm" else "viridis")
    ax.set(title=title, xlabel="Z plane (one-based)", ylabel="Timepoint")
    ax.set_xticks(np.arange(z_count), np.arange(1, z_count + 1), fontsize=7)
    ax.set_yticks(np.arange(len(timepoints)), timepoints)
    fig.colorbar(image, ax=ax, label=metric)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
