"""Configuration loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ChannelConfig:
    """How one ND2 channel should be identified and displayed."""

    label: str
    match: tuple[str, ...]
    plot_color: str
    rgb: tuple[float, float, float]
    index: int | None = None


@dataclass(frozen=True)
class AnalysisConfig:
    """All user-adjustable analysis settings."""

    z_index: int = 15
    z_is_one_based: bool = True
    timepoints: tuple[int, ...] | None = None
    position_x_order: str = "descending"
    channels: tuple[ChannelConfig, ...] = field(
        default_factory=lambda: (
            ChannelConfig("GFP", ("gfp", "488"), "#20a83e", (0.0, 1.0, 0.0)),
            ChannelConfig("Cy5", ("cy5", "647"), "#d62a8b", (1.0, 0.0, 1.0)),
        )
    )
    microscope_background: float = 100.0
    apply_illumination_correction: bool = True
    correction_method: str = "overlap_quadratic"
    overlap_min_signal: float = 5.0
    preview_low_percentile: float = 1.0
    preview_high_percentile: float = 99.8
    save_corrected_tiles: bool = True
    feather_overlaps: bool = True
    normalize_profiles: bool = True
    tanh_fit_enabled: bool = True
    tanh_max_points: int = 1024
    all_z_enabled: bool = True
    all_z_timepoints: tuple[int, ...] = (2, 10, 14, 24, 36)
    all_z_coefficient_smoothing_penalty: float = 10.0
    all_z_display_sample_stride: int = 32
    all_z_save_mosaic_tiffs: bool = True

    @classmethod
    def from_json(cls, path: str | Path | None) -> "AnalysisConfig":
        if path is None:
            return cls()
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        channels = tuple(
            ChannelConfig(
                label=item["label"],
                match=tuple(item.get("match", ())),
                plot_color=item["plot_color"],
                rgb=tuple(item["rgb"]),
                index=item.get("index"),
            )
            for item in raw.pop("channels", [])
        )
        if channels:
            raw["channels"] = channels
        if raw.get("timepoints") is not None:
            raw["timepoints"] = tuple(raw["timepoints"])
        if raw.get("all_z_timepoints") is not None:
            raw["all_z_timepoints"] = tuple(raw["all_z_timepoints"])
        # Removed settings are accepted but intentionally ignored so older
        # configuration files remain loadable.
        legacy_keys = (
            "slope_start_position",
            "slope_end_position",
            "reference_positions",
            "smoothing_window_px",
            "trendline_window_px",
            "correction_fit_degree",
            "bridge_position",
            "saturation_fraction_threshold",
        )
        for key in legacy_keys:
            raw.pop(key, None)
        return cls(**raw)

    @property
    def zero_based_z(self) -> int:
        return self.z_index - 1 if self.z_is_one_based else self.z_index

    def __post_init__(self) -> None:
        if self.correction_method != "overlap_quadratic":
            raise ValueError("correction_method must be 'overlap_quadratic'")
        if self.overlap_min_signal < 0:
            raise ValueError("overlap_min_signal must be non-negative")
        if self.tanh_max_points < 4:
            raise ValueError("tanh_max_points must be at least 4")
        if any(timepoint < 0 for timepoint in self.all_z_timepoints):
            raise ValueError("all_z_timepoints must be non-negative")
        if self.all_z_coefficient_smoothing_penalty < 0:
            raise ValueError("all_z_coefficient_smoothing_penalty must be non-negative")
        if self.all_z_display_sample_stride < 1:
            raise ValueError("all_z_display_sample_stride must be at least 1")
