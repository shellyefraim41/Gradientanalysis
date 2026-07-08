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
    reference_positions: dict[str, int | None] = field(
        default_factory=lambda: {"GFP": None, "Cy5": None}
    )
    smoothing_window_px: int = 101
    trendline_window_px: int = 201
    correction_fit_degree: int = 2
    bridge_position: int = 4
    slope_start_position: int = 1
    slope_end_position: int = 6
    saturation_fraction_threshold: float = 0.001
    preview_low_percentile: float = 1.0
    preview_high_percentile: float = 99.8
    save_corrected_tiles: bool = True
    microscope_background: float = 100.0
    position_gaussian_references: dict[str, int] = field(
        default_factory=lambda: {"GFP": 2, "Cy5": 6}
    )
    position_gaussian_sigma_px: float | None = None
    position_gaussian_clip_min: float = 0.25
    position_gaussian_clip_max: float = 4.0

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
        return cls(**raw)

    @property
    def zero_based_z(self) -> int:
        return self.z_index - 1 if self.z_is_one_based else self.z_index
