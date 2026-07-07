"""Compatibility tests for files written by the pipeline."""

import unittest
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from gradient_analysis.outputs import (
    bridge_slope_value,
    save_color_preview,
    save_slope_timecourse_plot,
    save_tiff,
    save_timecourse_plot,
    write_csv,
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
                "position_label": "P04",
                "position_one_based": 4,
                "position_nd2_index": 3,
                "start_px": 6,
                "end_px": 8,
                "slope_per_pixel": 1.5,
                "slope_per_um": 4.6,
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
        self.assertEqual(bridge_slope_value(-0.1705084), "-0.1705")

    def test_timecourse_plot_accepts_bridge_records(self):
        folder = Path.cwd() / ".test_outputs"
        folder.mkdir(exist_ok=True)
        path = folder / "step4_with_bridge.png"
        profiles = {
            0: np.linspace(0, 9, 10),
            18: np.linspace(10, 1, 10),
        }
        records = [
            {
                "timepoint": 0,
                "channel": "GFP",
                "position_label": "P04",
                "position_one_based": 4,
                "position_nd2_index": 3,
                "start_px": 2,
                "end_px": 8,
                "slope_per_pixel": 1.0,
                "slope_per_um": 3.0,
                "intercept": 2.0,
                "r_squared": 1.0,
            },
            {
                "timepoint": 18,
                "channel": "GFP",
                "position_label": "P04",
                "position_one_based": 4,
                "position_nd2_index": 3,
                "start_px": 2,
                "end_px": 8,
                "slope_per_pixel": -1.0,
                "slope_per_um": -3.0,
                "intercept": 8.0,
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


if __name__ == "__main__":
    unittest.main()
