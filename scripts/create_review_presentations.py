"""PowerPoint summaries for gradient-review runs, and a July-vs-September comparison deck.

Per-run deck (reads a completed run read-only and writes into a new folder):

    .\\.venv\\Scripts\\python.exe scripts\\create_review_presentations.py run ^
        --run outputs\\2026_09_27_exp106_try4_full_experiment\\run_... --label 2026_09_27_exp106_try4 ^
        --shape outputs\\2026_09_27_exp106_try4_full_experiment\\2026_09_27_exp106_try4_profile_shape_raw_vs_corrected_... ^
        --output outputs\\2026_09_27_exp106_try4_full_experiment

Comparison deck:

    .\\.venv\\Scripts\\python.exe scripts\\create_review_presentations.py compare ^
        --sept-run ... --july-run ... --sept-shape ... --july-shape ... --output outputs
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from pptx.util import Inches, Pt
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from create_gradient_presentations import (  # noqa: E402
    DARK, GRAY, _add_picture, _blank_slide, _new_presentation, _section_slide, _text, _title_slide,
)

KEY_HOURS = (0, 12, 24, 36, 48)
CHANNEL_COLORS = {"GFP": "#20a83e", "Cy5": "#d62a8b"}
CHANNEL_RGB = {"GFP": (0.0, 1.0, 0.0), "Cy5": (1.0, 0.0, 1.0)}


# ---------------------------------------------------------------------------
# Slide helpers
# ---------------------------------------------------------------------------


def _bullets(presentation, title: str, lines: list[str], size: int = 17) -> None:
    slide = _blank_slide(presentation)
    _text(slide, title, 0.6, 0.35, 12.1, 0.7, size=26, bold=True)
    box = slide.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(11.8), Inches(5.8))
    frame = box.text_frame
    frame.word_wrap = True
    for index, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = ("• " if not line.startswith("  ") else "    – ") + line.strip()
        paragraph.font.name = "Aptos"
        paragraph.font.size = Pt(size if not line.startswith("  ") else size - 2)
        paragraph.font.color.rgb = DARK if not line.startswith("  ") else GRAY
        paragraph.space_after = Pt(6)


def _picture_slide(presentation, title: str, path: Path, note: str = "") -> None:
    slide = _blank_slide(presentation)
    _text(slide, title, 0.4, 0.15, 12.5, 0.55, size=20, bold=True)
    top, height = 0.75, 6.55 if not note else 6.15
    _add_picture(slide, path, 0.3, top, 12.73, height, compress=True)
    if note:
        _text(slide, note, 0.5, 6.95, 12.3, 0.45, size=12, color=GRAY)


def _two_picture_slide(presentation, title: str, first: Path, second: Path, *, vertical: bool = True, note: str = "") -> None:
    slide = _blank_slide(presentation)
    _text(slide, title, 0.4, 0.15, 12.5, 0.55, size=20, bold=True)
    if vertical:
        _add_picture(slide, first, 0.3, 0.75, 12.73, 3.2, compress=True)
        _add_picture(slide, second, 0.3, 3.98, 12.73, 3.2, compress=True)
    else:
        _add_picture(slide, first, 0.2, 0.8, 6.4, 6.2, compress=True)
        _add_picture(slide, second, 6.73, 0.8, 6.4, 6.2, compress=True)
    if note:
        _text(slide, note, 0.5, 7.1, 12.3, 0.35, size=11, color=GRAY)


def _one(folder: Path, pattern: str) -> Path | None:
    matches = sorted(folder.glob(pattern))
    return matches[0] if matches else None


def _by_timepoint(folder: Path, pattern: str) -> list[tuple[int, Path]]:
    found = []
    for path in sorted(folder.glob(pattern)):
        match = re.search(r"_t(\d{3})_", path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def _step(run: Path, pattern: str) -> Path | None:
    matches = sorted(run.glob(f"step_*{pattern}*"))
    return matches[0] if matches else None


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Corrected gradient images (several timepoints per slide)
# ---------------------------------------------------------------------------

IMAGES_PER_SLIDE = 5


def _corrected_image_strips(run: Path, label: str, metadata: dict, folder: Path) -> dict[str, list[tuple[str, Path]]]:
    """Render the primary-method mosaics with one display range per channel over all timepoints."""
    seams = _step(run, "illumination_correction")
    method, tag = metadata["primary_method"], metadata["depth_modes"][0]["file_tag"]
    pixel_mm = float(metadata["pixel_size_um"]) * float(metadata["bin_factor_for_mosaics"]) / 1000.0
    bridge = metadata.get("bridge_from_DIC")
    stacks: dict[str, list[tuple[int, np.ndarray]]] = {}
    for channel in ("GFP", "Cy5"):
        for timepoint in metadata["timepoints"]:
            matches = sorted(seams.glob(f"*TIFF/{label}_t{timepoint:03d}_*_{channel}_{method}_{tag}_8x-binned_float32.tif"))
            if matches:
                stacks.setdefault(channel, []).append((timepoint, tifffile.imread(matches[0]).astype(np.float32)))
    if not stacks:
        return {}
    ranges = {}
    for channel, items in stacks.items():
        pooled = np.concatenate([image[np.isfinite(image) & (image > 0)][::7] for _, image in items])
        ranges[channel] = tuple(float(v) for v in np.percentile(pooled, [1.0, 99.8]))
    folder.mkdir(parents=True, exist_ok=True)
    hours = dict(zip(metadata["timepoints"], metadata["time_h"]))
    out: dict[str, list[tuple[str, Path]]] = {"merge": [], "GFP": [], "Cy5": []}
    for index, (timepoint, _) in enumerate(stacks["GFP"]):
        scaled = {}
        for channel in ("GFP", "Cy5"):
            image = np.nan_to_num(stacks[channel][index][1], nan=0.0)
            low, high = ranges[channel]
            scaled[channel] = np.clip((image - low) / (high - low), 0, 1)
        variants = {
            "merge": scaled["GFP"][..., None] * np.asarray(CHANNEL_RGB["GFP"]) + scaled["Cy5"][..., None] * np.asarray(CHANNEL_RGB["Cy5"]),
            "GFP": scaled["GFP"][..., None] * np.asarray(CHANNEL_RGB["GFP"]),
            "Cy5": scaled["Cy5"][..., None] * np.asarray(CHANNEL_RGB["Cy5"]),
        }
        for kind, rgb in variants.items():
            picture = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
            draw = ImageDraw.Draw(picture)
            height, width = picture.height, picture.width
            if bridge:
                for key in ("left_foot_mm", "right_foot_mm"):
                    x = int(round(bridge[key] / pixel_mm))
                    draw.line([(x, 0), (x, height // 6)], fill=(255, 210, 0), width=4)
                    draw.line([(x, height - height // 6), (x, height)], fill=(255, 210, 0), width=4)
            bar = int(round(1.0 / pixel_mm))
            draw.rectangle([width - bar - 20, height - 22, width - 20, height - 14], fill=(255, 255, 255))
            path = folder / f"{label}_t{timepoint:03d}_{hours[timepoint]:.2f}h_{kind}_{method}_{tag}_corrected_same-scale.png"
            picture.save(path)
            out[kind].append((f"{hours[timepoint]:.2f}".rstrip("0").rstrip(".") + f" h (t{timepoint:03d})", path))
    (folder / "README.txt").write_text(
        f"{method} correction at {tag}, 8x-binned mosaics. One display range per channel for all timepoints: "
        + ", ".join(f"{c} {lo:.0f}-{hi:.0f}" for c, (lo, hi) in ranges.items())
        + ". Yellow ticks: bridge edges (from DIC). White bar: 1 mm. Display images only; use the TIFFs in the run for measurement.\n",
        encoding="utf-8")
    return out


def _image_strip_slide(presentation, title: str, items: list[tuple[str, Path]], note: str) -> None:
    slide = _blank_slide(presentation)
    _text(slide, title, 0.4, 0.12, 12.5, 0.5, size=18, bold=True)
    top, row = 0.68, 1.30
    for index, (label, path) in enumerate(items):
        y = top + index * row
        _text(slide, label, 0.2, y + 0.3, 1.3, 0.7, size=13, bold=True)
        _add_picture(slide, path, 1.45, y, 11.7, row - 0.06, compress=True)
    _text(slide, note, 0.5, 7.12, 12.3, 0.35, size=11, color=GRAY)


# ---------------------------------------------------------------------------
# Value tables
# ---------------------------------------------------------------------------

TABLE_ROWS_PER_SLIDE = 18


def _table_slides(presentation, title: str, header: list[str], rows: list[list[str]], note: str = "",
                  first_column_width: float = 1.0) -> None:
    """Add one or more slides with a plain table (split when there are many rows)."""
    for start in range(0, max(len(rows), 1), TABLE_ROWS_PER_SLIDE):
        chunk = rows[start:start + TABLE_ROWS_PER_SLIDE]
        slide = _blank_slide(presentation)
        suffix = f" ({start // TABLE_ROWS_PER_SLIDE + 1}/{-(-len(rows) // TABLE_ROWS_PER_SLIDE)})" if len(rows) > TABLE_ROWS_PER_SLIDE else ""
        _text(slide, title + suffix, 0.4, 0.15, 12.5, 0.55, size=20, bold=True)
        height = 0.32 * (len(chunk) + 1)
        shape = slide.shapes.add_table(len(chunk) + 1, len(header), Inches(0.35), Inches(0.8), Inches(12.6), Inches(height))
        table = shape.table
        other = (12.6 - first_column_width) / max(len(header) - 1, 1)
        for column in range(len(header)):
            table.columns[column].width = Inches(first_column_width if column == 0 else other)
        for row_index, values in enumerate([header] + chunk):
            for column, value in enumerate(values):
                cell = table.cell(row_index, column)
                cell.text = str(value)
                paragraph = cell.text_frame.paragraphs[0]
                paragraph.font.size = Pt(10.5 if row_index else 10)
                paragraph.font.name = "Aptos"
                paragraph.font.bold = row_index == 0
        if note:
            _text(slide, note, 0.5, min(0.9 + height + 0.1, 7.0), 12.3, 0.4, size=11, color=GRAY)


def _write_table_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _fit_value_table(presentation, rows: list[dict[str, str]], labels: list[str], folder: Path, title_suffix: str) -> None:
    has_concentration = any(r.get("concentration_signed_slope_ug_ml_per_mm") for r in rows)
    slope_key, slope_name = (("concentration_signed_slope_ug_ml_per_mm", "slope [µg/mL/mm]") if has_concentration
                             else ("normalized_signed_slope_per_mm", "slope [1/mm]"))
    times = sorted({float(r["time_h"]) for r in rows})
    header = ["Time [h]"]
    for label in labels:
        header += [f"{label} {slope_name}", f"{label} slope [a.u./mm]", f"{label} midpoint [mm]", f"{label} width [mm]", f"{label} R²"]
    table, flagged = [], False
    for time in times:
        line = [f"{time:.2f}"]
        for label in labels:
            match = next((r for r in rows if r["channel"] == label and abs(float(r["time_h"]) - time) < 1e-6), None)
            if match is None:
                line += ["", "", "", "", ""]
                continue
            mark = "" if match["fit_valid"] == "True" else " ⚠"
            flagged |= bool(mark)
            line += [f"{float(match[slope_key]):.3g}{mark}", f"{float(match['intensity_signed_slope_per_mm']):.4g}",
                     f"{float(match['midpoint_mm']):.2f}",
                     f"{float(match['width_mm']):.2f}", f"{float(match['r_squared']):.3f}"]
        table.append(line)
    note = ("Slope = (right − left)/(2·width) at the midpoint; negative = decreasing to the right. [1/mm] is relative to the "
            "channel maximum over the analysed period; [a.u./mm] is in measured intensity units. "
            + ("⚠ = flagged fit (plateau outside the field): slope usable, midpoint/width not determined." if flagged else "All fits passed the checks."))
    _table_slides(presentation, f"Tanh fit values per timepoint{title_suffix}", header, table, note)
    _write_table_csv(folder / "tables" / "tanh_fit_values_per_timepoint.csv", header, table)


# ---------------------------------------------------------------------------
# Extra explanation slides
# ---------------------------------------------------------------------------


def _z_first(metadata: dict) -> int:
    return int(metadata.get("z_first_analysed_one_based", 1))


def _z_heights(metadata: dict) -> np.ndarray:
    """Nominal heights above the glass of the analysed planes."""
    z = metadata.get("z_stack") or {"home_index_zero_based": 0, "step_um": metadata.get("z_step_um", 10.0)}
    glass = int(metadata.get("glass_plane_one_based") or int(z["home_index_zero_based"]) + 1)
    first = _z_first(metadata)
    count = metadata["z_plane_count"]  # already the number of analysed planes
    return (np.arange(first, first + count) - glass) * float(z["step_um"])


def _timepoints_slide(presentation, metadata: dict) -> None:
    slide = _blank_slide(presentation)
    hours = metadata["time_h"]
    intervals = np.diff(hours) if len(hours) > 1 else np.array([0.0])
    _text(slide, "Timepoints and their real acquisition times", 0.6, 0.3, 12.1, 0.6, size=24, bold=True)
    _text(slide, "Read from the frame timestamps in the ND2 file (start of each timepoint, hours after the first). "
                 f"Intervals range from {intervals.min() * 60:.0f} min to {intervals.max():.2g} h.",
          0.6, 0.95, 12.1, 0.5, size=13, color=GRAY)
    entries = [f"t{t:03d}   {h:6.2f} h" for t, h in zip(metadata["timepoints"], hours)]
    per_column = int(np.ceil(len(entries) / 4))
    for column in range(4):
        chunk = entries[column * per_column:(column + 1) * per_column]
        if not chunk:
            continue
        box = slide.shapes.add_textbox(Inches(0.8 + column * 3.1), Inches(1.6), Inches(3.0), Inches(5.6))
        frame = box.text_frame
        for index, entry in enumerate(chunk):
            paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
            paragraph.text = entry
            paragraph.font.name = "Consolas"
            paragraph.font.size = Pt(14)
            paragraph.font.color.rgb = DARK


def _z_explanation_figure(path: Path, metadata: dict, chamber_height_um: float | None) -> Path:
    heights = _z_heights(metadata)
    modes = metadata["depth_modes"]
    main_planes = modes[0]["planes"]
    layer = modes[1]["planes"] if len(modes) > 1 else []
    fig, (side, heat) = plt.subplots(1, 2, figsize=(15, 7.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.1, 1]})
    top = max(heights.max() + 20, (chamber_height_um or 0) + 40)
    side.axhspan(-40, 0, color="#cfe8f3", label="Glass (bottom of the dish)")
    ceiling = chamber_height_um or top
    side.axhspan(0, ceiling, color="#f3f9ff", label="Liquid with the dextran gradient")
    if chamber_height_um:
        side.axhspan(chamber_height_um, chamber_height_um + 8, color="#333333", label=f"Membrane (designed at {chamber_height_um:g} µm)")
        side.axhspan(chamber_height_um + 8, top, color="#bbbbbb", alpha=0.5, label="Device above the membrane (not imaged)")
    if layer:
        side.axhspan(heights[layer[0]] - 4, heights[layer[-1]] + 4, color="#ffe9a8", alpha=0.7,
                     label=f"Robustness check: {modes[1]['title']}")
    for index, height in enumerate(heights):
        is_main = index in main_planes
        side.axhline(height, color="#d62728" if is_main else "#1f77b4", lw=3 if is_main else 1)
        side.text(0.02, height + 1.5, f"Z{index + _z_first(metadata)}  ({height:g} µm)" + ("  <- main result" if is_main else ""),
                  transform=side.get_yaxis_transform(), fontsize=8.5, color="#d62728" if is_main else "#1f4e79",
                  fontweight="bold" if is_main else "normal")
    side.set(xlim=(0, 1), ylim=(-40, top), xticks=[], ylabel="Nominal focus height above the glass [µm]",
             title="Side view: where each Z plane is (lines = optical slices)")
    side.legend(loc="upper right", fontsize=8, framealpha=0.95)
    rows, columns = 5, len(heights)
    rng = np.random.default_rng(1)
    demo = np.tile(np.linspace(0.9, 0.3, rows)[:, None], (1, columns)) + rng.normal(0, 0.015, (rows, columns))
    heat.imshow(demo, cmap="viridis", aspect="auto", origin="lower")
    heat.set_xticks(range(columns), [str(i + _z_first(metadata)) for i in range(columns)], fontsize=8)
    heat.set_yticks(range(rows), ["time 1", "time 2", "time 3", "time 4", "time 5"])
    heat.set(xlabel="Z plane [one-based index]: one column per height", ylabel="Time: one row per timepoint",
             title="Heatmap: one tanh fit per cell (example layout)")
    heat.add_patch(plt.Rectangle((-0.5, 1.5), columns, 1, fill=False, ec="white", lw=2.5))
    heat.annotate("one row = one timepoint,\nall heights", xy=(columns - 1, 2), xytext=(columns * 0.55, 3.6), color="white",
                  fontsize=10, arrowprops={"arrowstyle": "->", "color": "white"})
    heat.add_patch(plt.Rectangle((main_planes[0] - 0.5, -0.5), 1, rows, fill=False, ec="#ff6b6b", lw=2.5))
    heat.annotate("one column = one Z plane,\nall timepoints", xy=(main_planes[0], 0.3), xytext=(1, 0.6), color="#ffdddd",
                  fontsize=10, arrowprops={"arrowstyle": "->", "color": "#ffdddd"})
    heat.annotate("one cell = the gradient's slope\n(or brightness) seen in that\nsingle plane at that time", xy=(2, 4),
                  xytext=(0.2, 3.0), color="white", fontsize=10, arrowprops={"arrowstyle": "->", "color": "white"})
    fig.suptitle("Z analysis: every Z plane is analysed on its own, then the results are compared across heights "
                 "(heights are nominal; with the dry objective the real height in water is about 1.2-1.3x larger)", fontsize=11)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _standards_slides(presentation, folder: Path) -> None:
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    objective = summary.get("objective", "")
    _section_slide(presentation, "Calibration standards", f"Imaged with {objective}: flat field, background, and 32 µg/mL with the device")
    picture = _one(folder / "3_flat_field_correction_example", "*.png")
    if picture:
        ffc = {r["channel"]: r for r in summary.get("ffc_example", [])}
        note = " · ".join(f"{c}: centre/corner {r['center_to_corner_ratio_before']:.2f} -> {r['center_to_corner_ratio_after']:.2f}"
                          for c, r in ffc.items())
        _picture_slide(presentation, "Flat-field correction (FFC): a uniform standard before and after", picture,
                       f"FFC divides each image by the illumination pattern measured on a uniform standard. {note} (1.00 = perfectly flat).")
    picture = _one(folder / "1_background_blank_image_vs_constant_100", "*.png")
    if picture:
        _picture_slide(presentation, "Background: the 0 µg/mL standard versus the constant camera background (100)", picture)
        lines = []
        for r in summary["background"]:
            lines.append(f"{r['channel']}: blank median = {r['blank_median_per_Z_min']:g} in every Z plane; blank image - 100 = "
                         f"{r['blank_map_mean_minus_100']:+.2f} on average, pixel-to-pixel spread ±{r['blank_map_spatial_sd']:.1f}, "
                         f"slow shading {r['blank_map_large_scale_range_16px_blocks']:.1f} units.")
            lines.append(f"  Lowest standard ({r['lowest_standard_ug_ml']:g} µg/mL) at Z11: signal "
                         f"{r['lowest_standard_Z11_median_signal_constant100']:.1f} (minus 100) vs "
                         f"{r['lowest_standard_Z11_median_signal_blankmap']:.1f} (minus blank image); negative pixels "
                         f"{100 * r['lowest_standard_Z11_negative_fraction_constant100']:.2f}% vs "
                         f"{100 * r['lowest_standard_Z11_negative_fraction_blankmap']:.2f}%.")
        lines.append("Conclusion: the two backgrounds are the same number; subtracting the blank image pixel by pixel only adds camera "
                     "noise. All analysis uses the constant 100.")
        _bullets(presentation, "Background: which subtraction to use", lines, size=14)
    picture = _one(folder / "2_32ugmL_regular_vs_32ugmL_with_device", "*.png")
    if picture:
        _picture_slide(presentation, "32 µg/mL in a regular well vs 32 µg/mL with the device", picture)
        lines = []
        for r in summary.get("32_regular_vs_with_device", []):
            lines.append(f"{r['channel']}: with the device the signal is {100 * r['Z11_ratio_device_over_regular']:.0f}% of the regular "
                         f"well at Z11 ({100 * r['Z3_Z12_mean_ratio_device_over_regular']:.0f}% averaged over Z3-Z12).")
            lines.append(f"  Spatial variation at Z11: {100 * r['Z11_spatial_CV_regular']:.0f}% (regular) vs "
                         f"{100 * r['Z11_spatial_CV_with_device']:.0f}% (with device); the illumination shape differs by up to "
                         f"{100 * r['Z11_illumination_shape_difference_max_abs']:.0f}%.")
        lines += ["With the device the liquid layer starts higher (signal jumps between +10 and +30 µm) and gets dimmer towards the top.",
                  "A regular well has a deep column of fluorescent liquid above the focal plane; some of its out-of-focus light still "
                  "reaches the camera. Under the device the liquid is only about 150 µm deep with a black membrane on top, so much "
                  "less extra light is collected: the most likely reason for the ~2.5x difference (a lower real concentration in the "
                  "device is the alternative).",
                  "Implication: to convert intensity to µg/mL, standards should be imaged in the same geometry as the experiment "
                  "(in the device); standards in a regular well over-estimate the signal per µg/mL."]
        _bullets(presentation, "32 µg/mL with the device: what it means", lines, size=14)


def _diffusion_slides(presentation, folder: Path, output_folder: Path) -> None:
    _section_slide(presentation, "Front movement and diffusion rate", "From the tanh midpoint (the front) at every timepoint")
    rows = _read_rows(folder / "front_displacement_speed_and_D_over_time.csv")
    lines = [
        "Front X = tanh midpoint at each timepoint; X0 = midpoint at the first timepoint; Δt = time since the first timepoint.",
        "Average front speed = (X − X0)/Δt [µm/s]: how fast the front has moved since the start (positive = to the right).",
        "D(t) = (X − X0)²/Δt [µm²/s]: the displacement squared over the time, which has the units of a diffusion coefficient. "
        "For a front spreading by diffusion it stays roughly constant; if the front stops moving it falls as 1/Δt.",
        "Two time bases: the start of each timepoint, and the moment the tile containing the front was imaged "
        "(one timepoint takes about 6 minutes to image, which matters for the first intervals).",
    ]
    for channel in sorted({r["channel"] for r in rows}):
        selected = [r for r in rows if r["channel"] == channel and r["time_basis"] == "timepoint start time"]
        if selected:
            first, last = selected[0], selected[-1]
            lines.append(f"  {channel}: X0 = {float(first['X0_mm']):.2f} mm; first interval ({float(first['dt_h']) * 60:.0f} min): "
                         f"speed {float(first['front_speed_um_per_s']):.2g} µm/s, D {float(first['D_dx2_over_dt_um2_per_s']):.3g} µm²/s; "
                         f"at {float(last['time_h']):.3g} h: speed {float(last['front_speed_um_per_s']):.2g} µm/s, "
                         f"D {float(last['D_dx2_over_dt_um2_per_s']):.3g} µm²/s.")
    _bullets(presentation, "How the front movement and D(t) are calculated", lines, size=14)
    picture = _one(folder, "*.png")
    if picture:
        _picture_slide(presentation, "Front position, average front speed, and D(t) = (X − X0)²/Δt over time", picture)
    basis = "timepoint start time"
    channels = sorted({r["channel"] for r in rows}, key=lambda c: (c != "GFP", c))
    times = sorted({float(r["time_h"]) for r in rows})
    header = ["Time [h]", "Δt [h]"]
    for channel in channels:
        header += [f"{channel} X [mm]", f"{channel} X − X0 [µm]", f"{channel} speed [µm/s]", f"{channel} D(t) [µm²/s]"]
    table = []
    for time in times:
        line = [f"{time:.2f}", ""]
        for channel in channels:
            match = next((r for r in rows if r["channel"] == channel and r["time_basis"] == basis
                          and abs(float(r["time_h"]) - time) < 1e-6), None)
            if match is None:
                line += ["", "", "", ""]
                continue
            line[1] = f"{float(match['dt_h']):.2f}"
            line += [f"{float(match['X_mm']):.2f}", f"{float(match['displacement_um']):.0f}",
                     f"{float(match['front_speed_um_per_s']):.3f}", f"{float(match['D_dx2_over_dt_um2_per_s']):.3g}"]
        table.append(line)
    x0 = {r["channel"]: float(r["X0_mm"]) for r in rows}
    _table_slides(presentation, "Front position, speed, and D(t) per timepoint", header, table,
                  "X0 (first timepoint): " + ", ".join(f"{c} {x0[c]:.2f} mm" for c in channels)
                  + ". Speed = (X − X0)/Δt; D(t) = (X − X0)²/Δt; Δt from the start of each timepoint.")
    _write_table_csv(output_folder / "tables" / "front_position_speed_and_D_per_timepoint.csv", header, table)


def _corrections_slides(presentation, folder: Path, method_name: str) -> None:
    _bullets(presentation, "How the illumination corrections work", [
        "Microscope illumination is brightest in the centre of each image and dimmer at the edges. Without correction this "
        "draws a dome on every tile and steps (seams) where tiles meet.",
        "FFC (flat-field correction): divide every image by the 2-D illumination pattern measured on a uniform standard "
        "(standard method). It needs a uniform standard imaged with the same objective.",
        "Overlap quadratic: neighbouring tiles image the same spot twice, once near the right edge of one tile and once near "
        "the left edge of the next.",
        "  The real dextran there is identical, so the brightness ratio of the two views depends only on the illumination at "
        "those two camera positions: the real gradient cancels out and cannot be removed by mistake.",
        "  Thousands of such ratios (all overlaps, all timepoints) are fitted with one smooth curve along camera X: "
        "illumination(x) = exp(a·x + b·x²), scaled to average 1; every tile is divided by it.",
        "  It corrects along X (the gradient direction) only; one curve per Z plane, smoothed between neighbouring planes.",
        "Hybrid: FFC first, then the overlap quadratic fitted on what is left.",
        f"This dataset uses: {method_name}.",
    ], size=15)
    picture = _one(folder, "*correction_maps_FFC_quadratic_hybrid.png")
    if picture:
        _picture_slide(presentation, "What each correction divides out (example: 2026-09-27, 20x data, GFP, Z11)", picture,
                       "FFC removes the full 2-D dome; the overlap quadratic removes the left-right curve only; the hybrid combines both.")
    picture = _one(folder, "*one_tile_with_each_correction.png")
    if picture:
        _picture_slide(presentation, "The same tile with each correction (example: 2026-09-27, 20x data)", picture,
                       "Bottom: median over Y. Uncorrected, both edges fall off; each correction flattens the illumination differently.")


# ---------------------------------------------------------------------------
# Per-run deck
# ---------------------------------------------------------------------------


def run_deck(run: Path, label: str, shape: Path | None, output_root: Path, *, standards_checks: Path | None = None,
             corrections_demo: Path | None = None, chamber_height_um: float | None = None,
             notes: list[str] | None = None) -> Path:
    folder = output_root / f"{label}_presentations_{datetime.now():%Y%m%d_%H%M%S}"
    folder.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    hours = metadata["time_h"]
    primary = metadata["primary_method"]
    method_name = metadata["methods"][primary]
    modes = metadata["depth_modes"]
    has_cal = bool(metadata.get("calibration_conversions"))
    presentation = _new_presentation()
    _title_slide(presentation, f"{label}: dextran gradient analysis",
                 f"{len(hours)} timepoints, {hours[0]:.3g}–{hours[-1]:.3g} h · main result: {method_name}, {modes[0]['title']}"
                 f" · {datetime.now():%Y-%m-%d}")

    lines = [
        f"Illumination correction: {method_name} (chosen from a held-out seam test of {len(metadata['methods'])} methods).",
        f"Depth: main result at {modes[0]['title']}" + (f"; robustness check: {modes[1]['title']}." if len(modes) > 1 else "."),
    ]
    if has_cal:
        lines += ["Concentration: straight line through the 0–24 µg/mL standards with an intercept (32 µg/mL standard excluded "
                  "because all data are below the 24 µg/mL standard); line-through-zero and piecewise results are in the tables.",
                  "  Calibration standards were imaged at the same Z planes (same glass reference) as the experiment."]
    if metadata.get("bridge_from_DIC"):
        b = metadata["bridge_from_DIC"]
        lines.append(f"Bridge located automatically in the large DIC image: {b['left_foot_mm']:.2f}–{b['right_foot_mm']:.2f} mm "
                     f"(width {b['width_between_feet_mm']:.2f} mm); shown as a yellow band on every profile.")
    stain = metadata.get("stain_mask") or {}
    if stain.get("enabled"):
        lines.append("Stain: found with a fixed rule; excluded from fits only (red hatching), always shown in images and plots: "
                     + ", ".join(f"{a:.2f}–{c:.2f} mm" for a, c in stain["excluded_intervals_mm"]) + ".")
    lines.append("Tanh model: y = left + (right − left)/2 · [1 + tanh((x − midpoint)/width)]; midpoint slope = (right − left)/(2·width).")
    if metadata.get("flagged_fit_analysis", True):
        lines.append("Fits flagged ⚠ (a plateau outside the imaged field) are kept; they get extra model-free measures in their own folder.")
    if metadata.get("z_first_analysed_one_based", 1) > 1:
        lines.append(f"Z planes below Z{metadata['z_first_analysed_one_based']} are excluded (inside the glass); "
                     f"the glass surface is taken as Z{metadata.get('glass_plane_one_based')}.")
    _bullets(presentation, "What was done", lines)
    _timepoints_slide(presentation, metadata)
    if notes:
        _bullets(presentation, "Important notes about this dataset", notes, size=16)

    flat, calib = _step(run, "flat-field"), _step(run, "calibration_curves")
    if calib:
        _section_slide(presentation, "Calibration", "Background, flat field, and intensity → concentration")
        if flat:
            first, second = sorted(flat.glob("*flat-field_from_32ugmL_standard.png"))[:2]
            _two_picture_slide(presentation, "Flat field from the uniform 32 µg/mL standard (GFP left, Cy5 right)", first, second, vertical=False)
        picture = _one(calib, f"*{modes[0]['file_tag']}_calibration_curve_*.png")
        if picture:
            _picture_slide(presentation, f"Calibration curve at {modes[0]['title']}", picture,
                           "Green = primary line (0–24 µg/mL with intercept); purple = forced through zero; red = brightest experimental value.")

    if not calib and standards_checks:
        _standards_slides(presentation, standards_checks)

    seams = _step(run, "illumination_correction")
    if seams:
        _section_slide(presentation, "Illumination correction", "How well neighbouring tiles agree where they overlap")
        if corrections_demo:
            _corrections_slides(presentation, corrections_demo, method_name)
        picture = _one(seams, "*seam_mismatch_summary_by_method.png")
        if picture:
            _picture_slide(presentation, "Seam mismatch after each correction (lower is better; held out = fair test)", picture)
        previews = sorted((seams).glob("*mosaic_previews*/*GFP-Cy5_merge_*_every_method_stacked.png"))
        if previews:
            middle = previews[len(previews) // 2]
            when = re.search(r"_t(\d{3})_([\d.]+h)_", middle.name)
            _picture_slide(presentation, f"Mosaic with each correction method ({when.group(2)}, t{when.group(1)})" if when else
                           "Mosaic with each correction method", middle, "Same display range for every row; seams appear as vertical bands.")

    bridge = _step(run, "bridge")
    if bridge:
        _section_slide(presentation, "Bridge location", "From the large 10× DIC image, in the profile coordinates")
        picture = _one(bridge, "*bridge_edges_detected*.png")
        if picture:
            _picture_slide(presentation, "Bridge edges detected from the DIC image", picture)
        picture = _one(bridge, "*_auto_above_*bridge-marked.png")
        if picture:
            _picture_slide(presentation, "DIC (auto contrast) above the fluorescence mosaic, bridge edges marked", picture)

    stain_dir = _step(run, "stain_mask")
    if stain_dir:
        picture = _one(stain_dir, "*stain_mask_and_excluded_profile_positions.png")
        if picture:
            _picture_slide(presentation, "Stain: found by a fixed rule and excluded from the fits only", picture)

    zsel = _step(run, "Z_selection")
    if zsel:
        _section_slide(presentation, "Z selection", "Every Z plane analysed on its own at selected timepoints")
        explanation = _z_explanation_figure(folder / "Z_analysis_explained.png", metadata, chamber_height_um)
        _picture_slide(presentation, "How the Z analysis works", explanation)
        z_heights = _z_heights(metadata)
        _bullets(presentation, "How to read the Z analysis", [
            "The microscope records the same field at several heights (Z planes, 10 µm apart). Each plane is a thin optical "
            "slice of the liquid at that height.",
            "For each selected timepoint, the gradient profile along the device is measured in ONE plane at a time and fitted "
            "with the same tanh curve as the main result.",
            "The heatmaps collect all of these fits: each row is a timepoint, each column a Z plane, and the colour/number is "
            "the result for that plane (steepness of the gradient, or brightness of the plane).",
            "  Similar colours along a row = the gradient looks the same at those heights; a sudden change = a boundary "
            "(glass at the bottom, membrane at the top) or a real difference with height.",
            "A good plane sits in the middle of a uniform block of columns: away from the glass and from the membrane.",
            f"This dataset: {len(z_heights)} analysed planes (Z{_z_first(metadata)}-Z{_z_first(metadata) + len(z_heights) - 1}), "
            f"nominal heights {z_heights[0]:g} to {z_heights[-1]:g} µm above the glass"
            + (f"; the membrane is designed at {chamber_height_um:g} µm, so the stack stays below it." if chamber_height_um else "."),
            f"Main result: {modes[0]['title']}" + (f"; robustness check: {modes[1]['title']}." if len(modes) > 1 else "."),
        ], size=15)
        heat = zsel / "1_heatmaps_time_by_Z"
        for key, title in (("absolute_slope_per_mm", "Absolute midpoint slope for every Z plane and time"),
                           ("median_intensity_au", "Brightness of every Z plane (a sudden drop marks a boundary: glass or membrane)")):
            files = sorted(heat.glob(f"*_{key}_time-by-Z_heatmap.png"))
            if len(files) >= 2:
                _two_picture_slide(presentation, title, files[0], files[1])
        overlays = sorted((zsel / "3_profiles_of_every_Z_plane_overlaid").glob("*.png"))
        if overlays:
            _picture_slide(presentation, "Profiles of every Z plane at one timepoint (colour = Z)", overlays[len(overlays) // 2])

    main = _step(run, "MAIN_RESULT")
    reps = [folder for folder in sorted(main.iterdir()) if folder.is_dir() and folder.name[0] in "123"]
    first_rep = reps[0]
    _section_slide(presentation, "Main result", f"{method_name} · {modes[0]['title']} · {first_rep.name[2:].replace('_', ' ')}")
    strips = _corrected_image_strips(run, label, metadata, folder / "corrected_gradient_images")
    for kind, title in (("merge", "GFP (green) + Cy5 (magenta)"), ("GFP", "GFP"), ("Cy5", "Cy5")):
        items = strips.get(kind, [])
        for start in range(0, len(items), IMAGES_PER_SLIDE):
            _image_strip_slide(presentation, f"Corrected gradient images · {title} · {method_name}, {modes[0]['title']}",
                               items[start:start + IMAGES_PER_SLIDE],
                               "Same brightness scale for every timepoint; yellow ticks = bridge edges; white bar = 1 mm; stain left visible.")
    picture = _one(first_rep, "*_all_timepoints_*.png")
    if picture:
        _picture_slide(presentation, "All timepoints overlaid (colour = time)", picture)
    picture = _one(main, "*tanh_slopes_and_midpoints_over_time.png")
    if picture:
        _picture_slide(presentation, "Tanh slope, midpoint, and fit quality over time (open circles = flagged fits)", picture)
    fit_table = _one(main, "tanh_fit_results_*.csv")
    if fit_table:
        _fit_value_table(presentation, _read_rows(fit_table), metadata["channels"], folder,
                         f" · {method_name}, {modes[0]['title']}")
    for timepoint, path in _by_timepoint(first_rep, "*GFP-Cy5_tanh-fit.png"):
        _picture_slide(presentation, f"{hours[metadata['timepoints'].index(timepoint)]:.2f} h (t{timepoint:03d}) · {method_name} · "
                                     f"{modes[0]['title']}", path)

    diffusion = _step(run, "diffusion_coefficient")
    if diffusion:
        _diffusion_slides(presentation, diffusion, folder)

    robust = _step(run, "robustness_check")
    if robust:
        picture = _one(robust, "*_vs_*_slopes_over_time.png")
        if picture:
            _picture_slide(presentation, f"Robustness check: {modes[0]['title']} vs {modes[1]['title']}", picture)

    alt = _one(main, "5_*")
    if alt and (alt / "alternative_gradient_measures_for_flagged_tanh_fits.csv").exists():
        rows = _read_rows(alt / "alternative_gradient_measures_for_flagged_tanh_fits.csv")
        changes = [abs(float(r["slope_change_with_relaxed_limits_percent"])) for r in rows]
        _section_slide(presentation, "Flagged tanh fits", f"{len(rows)} of {2 * len(hours)} fits")
        _bullets(presentation, "What the flags mean", [
            "Flag 'parameter_at_bound': a fitted plateau lies outside the imaged field, so the fit is extrapolating.",
            f"Refitting with relaxed plateau limits changes the slope by a median of {np.median(changes):.1f}% (maximum {max(changes):.1f}%).",
            "  Slopes are therefore mostly reliable; midpoints of flagged fits are not determined and should not be interpreted.",
            "Each flagged profile also has model-free measures: a straight-line slope over the 20–80% band and the steepest local slope.",
        ])
        plots = sorted(alt.glob("*.png"))
        for first, second in zip(plots[::2], plots[1::2] + [None] * (len(plots) % 2)):
            if second is None:
                _picture_slide(presentation, "Model-free gradient measures for a flagged fit", first)
            else:
                _two_picture_slide(presentation, "Model-free gradient measures for flagged fits", first, second)

    key_timepoints = {metadata["timepoints"][int(np.argmin(np.abs(np.asarray(hours) - h)))] for h in KEY_HOURS if h <= hours[-1] + 1}
    if shape:
        _section_slide(presentation, "Do the profiles have plateaus?", "Raw vs corrected, each normalized to its own maximum")
        rows = _read_rows(next(shape.glob("*_profile_shape_measures_raw_vs_corrected.csv")))
        lines = []
        for channel in sorted({r["channel"] for r in rows}):
            for version in ("raw", "corrected"):
                selected = [r for r in rows if r["channel"] == channel and r["version"] == version]
                index = [float(r["plateau_index"]) for r in selected]
                lines.append(f"{channel} {version}: source-edge plateau index median {np.median(index):.2f} "
                             f"(range {min(index):.2f}–{max(index):.2f}); 0 = flat edge, 1 = no plateau.")
        lines += ["Raw profiles look less flat at the source edge because the outermost tile edges are dimmed by uneven illumination "
                  "(vignetting); after correction the source edge is close to flat.",
                  "Where tanh fits are flagged, the missing plateau is usually the low one at the far end: the dye has spread across "
                  "the whole field, so the profile is still sloping at the far edge and becomes closer to a straight line."]
        _bullets(presentation, "Plateau check: summary", lines, size=15)
        picture = _one(shape, "*plateau_index_and_linearity_over_time.png")
        if picture:
            _picture_slide(presentation, "How flat are the profile ends, and is a straight line as good as tanh?", picture)
        for timepoint, path in _by_timepoint(shape / "1_raw_vs_corrected_profiles_each_normalized_to_its_own_maximum", "*.png"):
            if timepoint in key_timepoints:
                _picture_slide(presentation, f"{hours[metadata['timepoints'].index(timepoint)]:.4g} h: raw vs corrected, "
                                             "each normalized to its own maximum", path)

    for rep in reps[1:]:
        _section_slide(presentation, f"Appendix: {rep.name[2:].replace('_', ' ')}", "Same fits, different Y axis")
        picture = _one(rep, "*_all_timepoints_*.png")
        if picture:
            _picture_slide(presentation, "All timepoints overlaid", picture)
        items = _by_timepoint(rep, "*GFP-Cy5_tanh-fit.png")
        for first, second in zip(items[::2], items[1::2] + [None] * (len(items) % 2)):
            if second is None:
                _picture_slide(presentation, f"t{first[0]:03d}", first[1])
            else:
                _two_picture_slide(presentation, f"t{first[0]:03d} and t{second[0]:03d}", first[1], second[1])

    path = folder / f"{label}_gradient_analysis_summary.pptx"
    presentation.save(path)
    (folder / "SOURCE_RUN.txt").write_text(f"Built from (read only): {run}\nPlateau check: {shape}\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# July vs September comparison deck
# ---------------------------------------------------------------------------


def _profiles(run: Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict | None]:
    main = _step(run, "MAIN_RESULT")
    path = next((main / "2_normalized_intensity").glob("*profile_data*.csv"))
    with path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        header = next(reader)
        data = np.array([[float(v) for v in row] for row in reader])
    columns = {name: data[:, index] for index, name in enumerate(header)}
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    return columns["position_mm"], columns, metadata


def _column(columns: dict[str, np.ndarray], channel: str, hour: int) -> np.ndarray | None:
    for name, values in columns.items():
        if name.startswith(f"{channel}_t{hour // 2:03d}_"):
            return values
    return None


def _mosaic_merge(run: Path, label: str, hour: int, method: str, tag: str) -> np.ndarray | None:
    seams = _step(run, "illumination_correction")
    images, colors = [], []
    for channel in ("GFP", "Cy5"):
        matches = sorted(seams.glob(f"*TIFF/{label}_t{hour // 2:03d}_*_{channel}_{method}_{tag}_8x-binned_float32.tif"))
        if not matches:
            return None
        image = np.nan_to_num(tifffile.imread(matches[0]).astype(np.float32), nan=0.0)
        low, high = np.percentile(image[image > 0], [1, 99.8])
        images.append(np.clip((image - low) / (high - low), 0, 1))
        colors.append(CHANNEL_RGB[channel])
    rgb = sum(image[..., None] * np.asarray(color) for image, color in zip(images, colors))
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def comparison_deck(sept: Path, july: Path, sept_label: str, july_label: str, sept_shape: Path | None,
                    july_shape: Path | None, output_root: Path) -> Path:
    folder = output_root / f"comparison_{july_label}_vs_{sept_label}_{datetime.now():%Y%m%d_%H%M%S}"
    figures = folder / "figures"
    figures.mkdir(parents=True, exist_ok=False)
    sx, s_cols, s_meta = _profiles(sept)
    jx, j_cols, j_meta = _profiles(july)
    bridge = s_meta.get("bridge_from_DIC")
    s_tag, j_tag = s_meta["depth_modes"][0]["file_tag"], j_meta["depth_modes"][0]["file_tag"]
    s_method, j_method = s_meta["primary_method"], j_meta["primary_method"]

    profile_figures = []
    for hour in KEY_HOURS:
        fig, axes = plt.subplots(2, 1, figsize=(13, 8.4), constrained_layout=True, sharex=True)
        for axis, channel in zip(axes, ("GFP", "Cy5")):
            if bridge:
                axis.axvspan(bridge["left_foot_mm"], bridge["right_foot_mm"], color="#d9c27a", alpha=0.25, lw=0,
                             label="Bridge (September DIC)")
            s = _column(s_cols, channel, hour)
            j = _column(j_cols, channel, hour)
            if s is not None:
                axis.plot(sx, s, color=CHANNEL_COLORS[channel], lw=1.4, label=f"September ({s_meta['methods'][s_method]}, {s_tag})")
            if j is not None:
                axis.plot(jx, j, color="black", lw=1.2, ls="--", label=f"July ({j_meta['methods'][j_method]}, {j_tag})")
            axis.set(xlabel="Position along device [mm]", ylabel=f"{channel} normalized intensity [fraction of channel maximum]")
            axis.set_ylim(-0.03, 1.05)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=9, loc="best")
        axes[0].set_title(f"July vs September at {hour} h · single plane Z11 · each experiment normalized to its own channel maximum over time")
        path = figures / f"July_vs_September_{hour:02d}h_Z11_normalized_profiles.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        profile_figures.append((hour, path))

    s_rows = _read_rows(next(_step(sept, "MAIN_RESULT").glob("tanh_fit_results_*.csv")))
    j_rows = _read_rows(next(_step(july, "MAIN_RESULT").glob("tanh_fit_results_*.csv")))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True)
    for c, channel in enumerate(("GFP", "Cy5")):
        for r, (key, ylabel) in enumerate((("normalized_signed_slope_per_mm", "Midpoint slope [1/mm, normalized intensity]"),
                                           ("midpoint_mm", "Tanh midpoint position [mm]"))):
            axis = axes[r, c]
            if key == "midpoint_mm" and bridge:
                axis.axhspan(bridge["left_foot_mm"], bridge["right_foot_mm"], color="#d9c27a", alpha=0.3, label="Bridge (September DIC)")
            for rows, name, style in ((s_rows, "September", "-"), (j_rows, "July", "--")):
                selected = sorted((x for x in rows if x["channel"] == channel), key=lambda x: float(x["time_h"]))
                times = [float(x["time_h"]) for x in selected]
                values = [float(x[key]) for x in selected]
                valid = [x["fit_valid"] == "True" for x in selected]
                color = CHANNEL_COLORS[channel] if name == "September" else "black"
                axis.plot(times, values, style, color=color, label=name)
                axis.scatter([t for t, v in zip(times, valid) if v], [y for y, v in zip(values, valid) if v], color=color, s=14)
                axis.scatter([t for t, v in zip(times, valid) if not v], [y for y, v in zip(values, valid) if not v],
                             facecolors="none", edgecolors=color, s=30)
            axis.set(xlabel="Time [h]", ylabel=ylabel, title=f"{channel}: July vs September (open circles = flagged fits)")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
    parameters_path = figures / "July_vs_September_tanh_slope_and_midpoint_over_time.png"
    fig.savefig(parameters_path, dpi=160)
    plt.close(fig)

    shape_path = None
    if sept_shape and july_shape:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
        for axis, channel in zip(axes, ("GFP", "Cy5")):
            for shape, name, style in ((sept_shape, "September", "-"), (july_shape, "July", "--")):
                rows = _read_rows(next(shape.glob("*_profile_shape_measures_raw_vs_corrected.csv")))
                selected = [r for r in rows if r["channel"] == channel and r["version"] == "corrected"]
                color = CHANNEL_COLORS[channel] if name == "September" else "black"
                axis.plot([float(r["time_h"]) for r in selected], [float(r["plateau_index"]) for r in selected], style,
                          color=color, label=f"{name}: source edge")
                axis.plot([float(r["time_h"]) for r in selected], [float(r["far_plateau_index"]) for r in selected], ":",
                          color=color, label=f"{name}: far edge")
            axis.set(xlabel="Time [h]", ylabel="Plateau index [–] (0 = flat, 1 = no plateau)",
                     title=f"{channel}: how flat are the profile ends? (corrected profiles)")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        shape_path = figures / "July_vs_September_plateau_index_over_time.png"
        fig.savefig(shape_path, dpi=160)
        plt.close(fig)

    mosaic_figures = []
    for hour in KEY_HOURS:
        panels = []
        for run, label, method, tag, name in ((sept, sept_label, s_method, s_tag, "September"), (july, july_label, j_method, j_tag, "July")):
            merge = _mosaic_merge(run, label, hour, method, tag)
            if merge is not None:
                panels.append((f"{name} · {hour} h · {tag} · GFP green / Cy5 magenta (display range per experiment)", merge))
        if not panels:
            continue
        width = max(p.shape[1] for _, p in panels)
        height = sum(p.shape[0] + 30 for _, p in panels)
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        y = 0
        for text, panel in panels:
            draw.text((6, y + 8), text, fill="black")
            canvas.paste(Image.fromarray(panel), (0, y + 30))
            y += panel.shape[0] + 30
        path = figures / f"July_vs_September_{hour:02d}h_Z11_mosaics.png"
        canvas.save(path)
        mosaic_figures.append((hour, path))

    presentation = _new_presentation()
    _title_slide(presentation, "July vs September: dextran gradients in the device",
                 f"{july_label} ({j_meta['time_h'][-1]:g} h) and {sept_label} ({s_meta['time_h'][-1]:g} h) · key timepoints "
                 f"{', '.join(f'{h} h' for h in KEY_HOURS)} · {datetime.now():%Y-%m-%d}")
    _bullets(presentation, "How to compare the two experiments", [
        "Both: single plane Z11, spinning-disk confocal 20× NA 0.8, 10 µm Z steps, same tanh model.",
        f"September: {s_meta['methods'][s_method]}; July: {j_meta['methods'][j_method]} (no calibration standards in July).",
        "  So profiles are compared as normalized intensity: each experiment divided by its own channel maximum over time.",
        "Z11 is not exactly the same height: in July the glass interface is at about Z3, in September below Z1; "
        "both Z11 planes are inside the liquid layer below the membrane.",
        "The membrane boundary is flat in July (Z16–17) and curved in September (Z14 near the bridge, Z18–19 at the edges).",
        "Bridge position is known only for September (from its DIC image); it is shown on all plots for orientation.",
        "July covers 0–72 h, September 0–48 h; slopes and midpoints are compared over the common times and beyond.",
    ], size=15)
    for (hour, profile), mosaic in zip(profile_figures, [dict(mosaic_figures).get(h) for h, _ in profile_figures]):
        _section_slide(presentation, f"{hour} h", "Profiles and mosaics")
        _picture_slide(presentation, f"{hour} h: normalized Z11 profiles, July (dashed) vs September (solid)", profile)
        if mosaic is not None:
            _picture_slide(presentation, f"{hour} h: Z11 mosaics (top September, bottom July)", mosaic)
    _section_slide(presentation, "Over time", "Midpoints, slopes, and plateaus")
    _picture_slide(presentation, "Tanh slope and midpoint over time: July vs September", parameters_path,
                   "Midpoints of flagged fits (open circles) are not well determined.")
    if shape_path:
        _picture_slide(presentation, "Do the corrected profiles reach plateaus? July vs September", shape_path)
    _bullets(presentation, "Observations to discuss", [
        "Neither experiment looks like symmetric diffusion from two opposite sources.",
        "July: Cy5 fills most of the field and advances to the left; GFP is pushed back towards the left edge.",
        "September: Cy5 advances left across the bridge until about 18 h, then retreats right; GFP rises everywhere after about 20 h "
        "and its front moves right across the bridge.",
        "  The apparent net direction differs between the experiments, which fits an external pressure/flow difference better "
        "than a property of the device itself (to be checked: stage movement, membrane holes, reservoir levels).",
    ], size=16)
    path = folder / "July_vs_September_key_timepoint_comparison.pptx"
    presentation.save(path)
    (folder / "SOURCE_RUNS.txt").write_text(f"September (read only): {sept}\nJuly (read only): {july}\n"
                                            f"Plateau checks: {sept_shape} ; {july_shape}\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run")
    one.add_argument("--run", required=True)
    one.add_argument("--label", required=True)
    one.add_argument("--shape", default=None)
    one.add_argument("--output", required=True)
    one.add_argument("--standards-checks", default=None)
    one.add_argument("--corrections-demo", default=None)
    one.add_argument("--chamber-height-um", type=float, default=None)
    one.add_argument("--note", action="append", default=None, help="Bullet for an 'important notes' slide (repeatable).")
    both = sub.add_parser("compare")
    for name in ("--sept-run", "--july-run", "--output"):
        both.add_argument(name, required=True)
    both.add_argument("--sept-label", default="2026_09_27_exp106_try4")
    both.add_argument("--july-label", default="2026_07_30_exp106")
    both.add_argument("--sept-shape", default=None)
    both.add_argument("--july-shape", default=None)
    args = parser.parse_args()
    if args.command == "run":
        print(run_deck(Path(args.run), args.label, Path(args.shape) if args.shape else None, Path(args.output),
                       standards_checks=Path(args.standards_checks) if args.standards_checks else None,
                       corrections_demo=Path(args.corrections_demo) if args.corrections_demo else None,
                       chamber_height_um=args.chamber_height_um, notes=args.note))
    else:
        print(comparison_deck(Path(args.sept_run), Path(args.july_run), args.sept_label, args.july_label,
                              Path(args.sept_shape) if args.sept_shape else None,
                              Path(args.july_shape) if args.july_shape else None, Path(args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
