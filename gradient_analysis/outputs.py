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


def bridge_slope_value(slope_per_pixel: float) -> str:
    """Format one Step 4 bridge slope table value without repeating units."""
    return f"{slope_per_pixel:.4g}"


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
    bridge_span: tuple[float, float] | None = None
    for channel in channels:
        profile = profiles[channel.label]
        ax.plot(profile, color=channel.plot_color, label=channel.label, linewidth=1.2)
        if trendline_window:
            ax.plot(
                smooth_profile(profile, trendline_window),
                color=channel.plot_color,
                linestyle="--",
                linewidth=1.8,
                alpha=0.9,
                label=f"{channel.label} smooth trend",
            )
        if bridge_fits and channel.label in bridge_fits:
            fit = bridge_fits[channel.label]
            start = int(fit["start_px"])
            end = int(fit["end_px"])
            x = np.arange(start, end, dtype=np.float64)
            y = float(fit["slope_per_pixel"]) * np.arange(end - start) + float(fit["intercept"])
            ax.plot(
                x,
                y,
                color=channel.plot_color,
                linestyle=":",
                linewidth=2.4,
                label=f"{channel.label} P04 slope={float(fit['slope_per_pixel']):.4g}/px",
            )
            bridge_span = (start, end)
    if bridge_span is not None:
        ax.axvspan(*bridge_span, color="gray", alpha=0.12, label="P04 bridge")
    ax.set(title=title, xlabel="Stitched x coordinate (pixels)", ylabel="Mean intensity (a.u.)")
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
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
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
    bridge_span: tuple[float, float] | None = None
    for color, (timepoint, profile) in zip(colors, sorted(profiles.items())):
        ax.plot(profile, color=color, linewidth=0.9, alpha=0.85, label=f"t={timepoint}")
        if trendline_window:
            ax.plot(
                smooth_profile(profile, trendline_window),
                color=color,
                linestyle="--",
                linewidth=1.6,
                alpha=0.95,
            )
        record = records_by_timepoint.get(int(timepoint))
        if record:
            start = int(record["start_px"])
            end = int(record["end_px"])
            local_x = np.arange(end - start, dtype=np.float64)
            y = float(record["slope_per_pixel"]) * local_x + float(record["intercept"])
            ax.plot(
                np.arange(start, end, dtype=np.float64),
                y,
                color=color,
                linestyle=":",
                linewidth=2.3,
            )
            bridge_span = (start, end)
    if bridge_span is not None:
        ax.axvspan(*bridge_span, color="gray", alpha=0.12, label="P04 bridge")
    ax.set(title=title, xlabel="Stitched x coordinate (pixels)", ylabel="Mean intensity (a.u.)")
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
            [str(timepoint), bridge_slope_value(float(record["slope_per_pixel"]))]
            for timepoint, record in sorted(records_by_timepoint.items())
        ]
        table_ax = fig.add_axes([0.79, 0.18, 0.19, 0.72])
        table_ax.axis("off")
        table_ax.set_title(f"{channel.label} P04 slope", fontsize=10, pad=8)
        table = table_ax.table(
            cellText=table_rows,
            colLabels=["Timepoint", "Δ under bridge\n(a.u./px)"],
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
    """Plot bridge slope values across time for one channel."""
    selected = sorted(
        [record for record in records if record["channel"] == channel.label],
        key=lambda record: int(record["timepoint"]),
    )
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x = [int(record["timepoint"]) for record in selected]
    y = [float(record["slope_per_pixel"]) for record in selected]
    ax.plot(x, y, marker="o", color=channel.plot_color, label=f"{channel.label} slope")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set(title=title, xlabel="Timepoint", ylabel="Slope in P04 (intensity / pixel)")
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
        "position_label",
        "position_one_based",
        "position_nd2_index",
        "start_px",
        "end_px",
        "slope_per_pixel",
        "slope_per_um",
        "intercept",
        "r_squared",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
