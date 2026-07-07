"""Lazy, plane-at-a-time access to ND2 data and acquisition metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import AnalysisConfig, ChannelConfig


@dataclass(frozen=True)
class Position:
    """One acquisition position, retaining its original ND2 index."""

    index: int
    x_um: float
    y_um: float
    name: str


class ND2Source:
    """Context-managed ND2 reader that computes only requested 2-D planes."""

    def __init__(self, path: str | Path, config: AnalysisConfig):
        self.path = Path(path)
        self.config = config
        self._file: Any = None
        self._array: Any = None
        self.axis_order: tuple[str, ...] = ()
        self.sizes: dict[str, int] = {}
        self.channel_indices: dict[str, int] = {}
        self.channel_names: list[str] = []
        self.positions: list[Position] = []
        self.pixel_size_um: float | None = None

    def __enter__(self) -> "ND2Source":
        try:
            import nd2
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc
        self._file = nd2.ND2File(self.path)
        self.sizes = dict(self._file.sizes)
        self.axis_order = tuple(self.sizes)
        self._validate_dimensions()
        self.channel_names = self._read_channel_names()
        self.channel_indices = self._resolve_channels()
        self.positions = self._read_positions()
        self.pixel_size_um = self._read_pixel_size_um()
        # Dask keeps the 250 GB acquisition on disk. Each call to plane() computes
        # just one T/P/Z/C selection.
        self._array = self._file.to_dask()
        return self

    def __exit__(self, *_: object) -> None:
        if self._file is not None:
            self._file.close()

    def _validate_dimensions(self) -> None:
        missing = {"Y", "X"} - self.sizes.keys()
        if missing:
            raise ValueError(f"ND2 is missing required dimensions: {sorted(missing)}")
        z = self.config.zero_based_z
        if z < 0 or z >= self.sizes.get("Z", 1):
            raise ValueError(
                f"Requested Z={self.config.z_index} resolves to index {z}, "
                f"but the file contains {self.sizes.get('Z', 1)} Z planes."
            )
        if self.sizes.get("C", 1) < len(self.config.channels):
            raise ValueError("The ND2 contains fewer channels than requested.")
        if self.config.position_x_order not in {"ascending", "descending", "acquisition"}:
            raise ValueError(
                "position_x_order must be 'ascending', 'descending', or 'acquisition'."
            )

    def _read_channel_names(self) -> list[str]:
        try:
            return [str(c.channel.name or f"Channel {i}") for i, c in enumerate(self._file.metadata.channels)]
        except (AttributeError, TypeError):
            return [f"Channel {i}" for i in range(self.sizes.get("C", 1))]

    def _resolve_channels(self) -> dict[str, int]:
        result: dict[str, int] = {}
        lowered = [name.casefold() for name in self.channel_names]
        for fallback, channel in enumerate(self.config.channels):
            if channel.index is not None:
                index = channel.index
            else:
                matches = [i for i, name in enumerate(lowered) if any(key.casefold() in name for key in channel.match)]
                index = matches[0] if matches else fallback
            if not 0 <= index < self.sizes.get("C", 1):
                raise ValueError(f"Cannot resolve channel {channel.label!r}; ND2 channels: {self.channel_names}")
            result[channel.label] = index
        if len(set(result.values())) != len(result):
            raise ValueError(f"Channels resolved to duplicate indices: {result}. Set channel indices in the config.")
        return result

    def _read_positions(self) -> list[Position]:
        count = self.sizes.get("P", 1)
        points: list[Any] = []
        for loop in self._file.experiment:
            if getattr(loop, "type", "") == "XYPosLoop":
                points = list(getattr(loop.parameters, "points", []))
                break
        positions = []
        for index in range(count):
            point = points[index] if index < len(points) else None
            xyz = getattr(point, "stagePositionUm", None)
            if xyz is None:
                x_um, y_um = float(index), 0.0
            elif hasattr(xyz, "x"):
                # nd2 0.11 exposes a StagePosition named tuple/object.
                x_um, y_um = float(xyz.x), float(xyz.y)
            else:
                # Retain compatibility with older nd2 versions that returned a list.
                x_um, y_um = float(xyz[0]), float(xyz[1])
            name = getattr(point, "name", None) or f"P{index + 1:02d}"
            positions.append(Position(index, x_um, y_um, str(name)))
        if self.config.position_x_order == "acquisition":
            return positions
        return sorted(
            positions,
            key=lambda p: p.x_um,
            reverse=self.config.position_x_order == "descending",
        )

    def _read_pixel_size_um(self) -> float | None:
        """Best-effort read of the XY pixel size in micrometers."""
        for name in ("voxel_size", "voxelSize"):
            candidate = getattr(self._file, name, None)
            if callable(candidate):
                try:
                    value = candidate()
                except TypeError:
                    value = candidate
            else:
                value = candidate
            x_value = getattr(value, "x", None)
            if x_value is not None:
                return float(x_value)
        return None

    @property
    def timepoints(self) -> tuple[int, ...]:
        count = self.sizes.get("T", 1)
        selected = self.config.timepoints or tuple(range(count))
        invalid = [t for t in selected if not 0 <= t < count]
        if invalid:
            raise ValueError(f"Invalid timepoint indices {invalid}; file has {count} timepoints.")
        return selected

    def plane(self, timepoint: int, position: int, channel: ChannelConfig) -> np.ndarray:
        """Read one YX plane, indexing absent singleton dimensions safely."""
        selection: list[int | slice] = []
        values = {
            "T": timepoint,
            "P": position,
            "Z": self.config.zero_based_z,
            "C": self.channel_indices[channel.label],
        }
        for axis in self.axis_order:
            selection.append(values.get(axis, slice(None)))
        plane = np.asarray(self._array[tuple(selection)].compute()).squeeze()
        if plane.ndim != 2:
            raise ValueError(f"Expected a 2-D YX plane, got shape {plane.shape} after selection.")
        return plane
