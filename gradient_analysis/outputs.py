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
from .processing import colorize_scaled, display_scale, smooth_profile


def slope_table_value(slope: float) -> str:
    """Format one Step 4 slope table value without repeating units."""
    return f"{slope:.4g}"


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
    bridge_fits: dict[str, dict[str, float | int | str | None]] | None = None,
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
        if bridge_fits and channel.label in bridge_fits:
            fit = bridge_fits[channel.label]
            start = float(fit["x_start_mm"])
            end = float(fit["x_end_mm"])
            x = np.linspace(start, end, 200)
            y = float(fit["slope_au_per_mm"]) * x + float(fit["intercept"])
            ax.plot(
                x,
                y,
                color=channel.plot_color,
                linestyle=":",
                linewidth=2.4,
                label=f"{channel.label} P01-P06 slope={float(fit['slope_au_per_mm']):.4g}/mm",
            )
            ax.axvspan(start, end, color="gray", alpha=0.08, label="P01-P06 fit range")
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
    bridge_records: list[dict[str, float | int | str | None]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(18, 8), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.76, top=0.9, bottom=0.18)
    colors = plt.colormaps["viridis"](np.linspace(0.0, 1.0, len(profiles)))
    records_by_timepoint = {
        int(record["timepoint"]): record
        for record in bridge_records or []
        if record["channel"] == channel.label
    }
    slope_span: tuple[float, float] | None = None
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
        record = records_by_timepoint.get(int(timepoint))
        if record:
            start = float(record["x_start_mm"])
            end = float(record["x_end_mm"])
            x = np.linspace(start, end, 200)
            y = float(record["slope_au_per_mm"]) * x + float(record["intercept"])
            ax.plot(
                x,
                y,
                color=color,
                linestyle=":",
                linewidth=2.3,
            )
            slope_span = (start, end)
    if slope_span is not None:
        ax.axvspan(*slope_span, color="gray", alpha=0.10, label="P01-P06 fit range")
    ax.set(title=title, xlabel="Position along device (mm)", ylabel="Mean intensity (a.u.)")
    ax.grid(alpha=0.2)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=8,
        fontsize=7,
        frameon=True,
    )
    if records_by_timepoint:
        table_rows = [
            [str(timepoint), slope_table_value(float(record["slope_au_per_mm"]))]
            for timepoint, record in sorted(records_by_timepoint.items())
        ]
        table_ax = fig.add_axes([0.79, 0.18, 0.19, 0.72])
        table_ax.axis("off")
        table_ax.set_title(f"{channel.label} P01-P06 slope", fontsize=10, pad=8)
        table = table_ax.table(
            cellText=table_rows,
            colLabels=["Timepoint", "Slope P01-P06\n(a.u./mm)"],
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7 if len(table_rows) <= 25 else 5.5)
        table.scale(1.0, 1.15 if len(table_rows) <= 25 else 0.75)
        for (row, _column), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#eeeeee")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_slope_timecourse_plot(
    path: Path,
    records: list[dict[str, float | int | str | None]],
    channel: ChannelConfig,
    title: str,
) -> None:
    """Plot P01-P06 gradient slope values across time for one channel."""
    selected = sorted(
        [record for record in records if record["channel"] == channel.label],
        key=lambda record: int(record["timepoint"]),
    )
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x = [int(record["timepoint"]) for record in selected]
    y = [float(record["slope_au_per_mm"]) for record in selected]
    ax.plot(x, y, marker="o", color=channel.plot_color, label=f"{channel.label} P01-P06 slope")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set(title=title, xlabel="Timepoint", ylabel="Slope P01-P06 (a.u./mm)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a simple table with stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timepoint",
        "channel",
        "start_position_label",
        "end_position_label",
        "start_position_one_based",
        "end_position_one_based",
        "x_start_mm",
        "x_end_mm",
        "slope_au_per_mm",
        "intercept",
        "r_squared",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
