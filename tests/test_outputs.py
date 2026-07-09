"""Compatibility tests for files written by the pipeline."""

import unittest
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from gradient_analysis.outputs import (
    save_color_preview,
    save_max_difference_timecourse_plot,
    save_normalization_plot,
    save_slope_timecourse_plot,
    save_tiff,
    save_timecourse_plot,
    slope_table_value,
    write_csv,
    write_rows_csv,
)
from gradient_analysis.config import ChannelConfig


class OutputTests(unittest.TestCase):
    def test_uint16_tiff_opens_with_tifffile_and_pillow(self):
        expected = np.arange(48, dtype=np.uint16).reshape(6, 8)
        folder = Path.cwd() / ".test_outputs"
        folder.mkdir(exist_ok=True)
        path = folder / "corrected.tif"
        save_tiff(path, expected)
        np.testing.assert_array_equal(tifffile.imread(path), expected)
        with Image.open(path) as image:
            self.assertEqual(image.size, (8, 6))
            self.assertEqual(np.asarray(image).dtype, np.uint16)

    def test_color_preview_opens_as_rgb(self):
        folder = Path.cwd() / ".test_outputs"
        folder.mkdir(exist_ok=True)
        path = folder / "preview.png"
        save_color_preview(path, np.arange(12, dtype=np.uint16).reshape(3, 4), 0, 11, (1, 0, 1))
        with Image.open(path) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (4, 3))

    def test_slope_outputs_are_written(self):
        folder = Path.cwd() / ".test_outputs"
        folder.mkdir(exist_ok=True)
        rows = [
            {
                "timepoint": 0,
                "channel": "GFP",
                "start_position_label": "P01",
                "end_position_label": "P06",
                "start_position_one_based": 1,
                "end_position_one_based": 6,
                "x_start_mm": 0.0,
                "x_end_mm": 8.0,
                "slope_au_per_mm": 1.5,
                "intercept": 2.0,
                "r_squared": 0.99,
            }
        ]
        csv_path = folder / "slopes.csv"
        png_path = folder / "slopes.png"
        write_csv(csv_path, rows)
        save_slope_timecourse_plot(
            png_path,
            rows,
            ChannelConfig("GFP", ("gfp",), "#20a83e", (0, 1, 0)),
            "Slope test",
        )
        self.assertTrue(csv_path.read_text(encoding="utf-8").startswith("timepoint,channel"))
        with Image.open(png_path) as image:
            self.assertGreater(image.size[0], 0)

    def test_step4_slope_table_value_format_has_no_units(self):
        self.assertEqual(slope_table_value(-0.1705084), "-0.1705")

    def test_combined_max_difference_plot_outputs_image(self):
        folder = Path.cwd() / ".test_outputs"
        folder.mkdir(exist_ok=True)
        path = folder / "max_differences.png"
        records = [
            {
                "timepoint": 0,
                "channel": "GFP",
                "formula": "max(P02)-max(P05)",
                "max_intensity_difference": 120.0,
            },
            {
                "timepoint": 0,
                "channel": "Cy5",
                "formula": "max(P05)-max(P02)",
                "max_intensity_difference": 80.0,
            },
        ]
        channels = (
            ChannelConfig("GFP", ("gfp",), "#20a83e", (0, 1, 0)),
            ChannelConfig("Cy5", ("cy5",), "#d62a8b", (1, 0, 1)),
        )
        save_max_difference_timecourse_plot(path, records, channels, "Difference test")
        with Image.open(path) as image:
            self.assertGreater(image.size[0], 0)

    def test_generic_rows_csv_uses_first_row_fields(self):
        folder = Path.cwd() / ".test_outputs"
        folder.mkdir(exist_ok=True)
        path = folder / "diagnostics.csv"
        write_rows_csv(path, [{"channel": "GFP", "raw": 1.0, "corrected": 2.0}])
        self.assertTrue(path.read_text(encoding="utf-8").startswith("channel,raw,corrected"))

    def test_timecourse_plot_accepts_gradient_slope_records(self):
        folder = Path.cwd() / ".test_outputs"
        folder.mkdir(exist_ok=True)
        path = folder / "step4_with_gradient_slope.png"
        profiles = {
            0: [(np.linspace(0, 1, 5), np.linspace(0, 4, 5)), (np.linspace(2, 3, 5), np.linspace(8, 12, 5))],
            18: [(np.linspace(0, 1, 5), np.linspace(10, 6, 5)), (np.linspace(2, 3, 5), np.linspace(2, -2, 5))],
        }
        records = [
            {
                "timepoint": 0,
                "channel": "GFP",
                "start_position_label": "P01",
                "end_position_label": "P06",
                "start_position_one_based": 1,
                "end_position_one_based": 6,
                "x_start_mm": 0.0,
                "x_end_mm": 3.0,
                "slope_au_per_mm": 1.0,
                "intercept": 0.0,
                "r_squared": 1.0,
            },
            {
                "timepoint": 18,
                "channel": "GFP",
                "start_position_label": "P01",
                "end_position_label": "P06",
                "start_position_one_based": 1,
                "end_position_one_based": 6,
                "x_start_mm": 0.0,
                "x_end_mm": 3.0,
                "slope_au_per_mm": -1.0,
                "intercept": 10.0,
                "r_squared": 1.0,
            },
        ]
        save_timecourse_plot(
            path,
            profiles,
            ChannelConfig("GFP", ("gfp",), "#20a83e", (0, 1, 0)),
            "Step 4 bridge test",
            trendline_window=3,
            bridge_records=records,
        )
        with Image.open(path) as image:
            self.assertGreater(image.size[0], 0)

    def test_normalization_plot_accepts_physical_corrected_segments(self):
        folder = Path.cwd() / ".test_outputs"
        folder.mkdir(exist_ok=True)
        path = folder / "normalization_with_physical_axis.png"
        raw_profiles = [np.linspace(5, 10, 6), np.linspace(10, 20, 6)]
        corrected_profiles = [np.linspace(8, 9, 6), np.linspace(12, 13, 6)]
        physical_segments = [
            (np.linspace(0.0, 0.5, 6), corrected_profiles[0]),
            (np.linspace(1.5, 2.0, 6), corrected_profiles[1]),
        ]
        save_normalization_plot(
            path,
            raw_profiles,
            corrected_profiles,
            np.linspace(1, 2, 6),
            reference_list_index=0,
            position_labels=["P01", "P02"],
            channel=ChannelConfig("GFP", ("gfp",), "#20a83e", (0, 1, 0)),
            title="Normalization test",
            corrected_physical_segments=physical_segments,
        )
        with Image.open(path) as image:
            self.assertGreater(image.size[0], 0)


if __name__ == "__main__":
    unittest.main()
