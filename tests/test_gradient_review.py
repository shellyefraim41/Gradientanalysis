"""Synthetic tests for bridge detection, stain masking, calibration lines, and model-free slopes."""

import unittest
from pathlib import Path

import numpy as np

from gradient_analysis.calibrated_pipeline import _discover_standards
from gradient_analysis.gradient_review import (
    _align_dic_crop_to_profile,
    _hl,
    diffusion_from_widths,
    front_motion_table,
    alternative_gradient_metrics,
    detect_bridge_edges,
    detect_stain_mask,
    excluded_profile_positions,
    intervals_from_mask,
    linear_conversion,
)


class BridgeTests(unittest.TestCase):
    def test_three_level_dic_profile_returns_central_plateau(self):
        pixel_mm = 0.001
        x = np.arange(6000) * pixel_mm
        profile = np.full(x.size, 103.0)
        profile[(x >= 1.2) & (x < 4.9)] = 108.0
        profile[(x >= 2.0) & (x < 4.0)] = 119.0
        profile += np.random.default_rng(1).normal(0, 0.5, x.size)
        edges = detect_bridge_edges(profile, pixel_mm)
        self.assertAlmostEqual(edges["left_half_max_mm"], 2.0, delta=0.02)
        self.assertAlmostEqual(edges["right_half_max_mm"], 4.0, delta=0.02)
        self.assertAlmostEqual(edges["width_between_feet_mm"], 2.0, delta=0.08)
        self.assertLessEqual(edges["left_foot_mm"], edges["left_half_max_mm"])
        self.assertGreaterEqual(edges["right_foot_mm"], edges["right_half_max_mm"])


class StainTests(unittest.TestCase):
    def test_static_dark_blob_is_masked_but_moving_gradient_is_not(self):
        images = []
        yy, xx = np.mgrid[0:100, 0:400]
        blob = (yy - 50) ** 2 + (xx - 200) ** 2 < 25 ** 2
        for shift in range(5):
            image = 100.0 + 0.5 * xx + 40 * np.exp(-((xx - 80 - 40 * shift) / 15.0) ** 2)
            image[blob] *= 0.4
            images.append(image)
        mask, _, details = detect_stain_mask(images, pixel_um=2.6)
        self.assertGreater(np.count_nonzero(mask & blob) / np.count_nonzero(blob), 0.9)
        self.assertEqual(np.count_nonzero(mask & ~blob), 0)
        self.assertEqual(len(details["objects"]), 1)

    def test_columns_with_large_stain_fraction_are_excluded_with_margin(self):
        mask = np.zeros((100, 400), dtype=bool)
        mask[20:70, 190:210] = True
        mask[0:5, 50:60] = True  # 5% of the column: below threshold
        valid = np.ones_like(mask)
        column_mm = np.arange(400) * 0.01
        profile_mm = np.arange(4000) * 0.001
        excluded, _ = excluded_profile_positions(mask, valid, column_mm, profile_mm)
        intervals = intervals_from_mask(profile_mm, excluded)
        self.assertEqual(len(intervals), 1)
        start, stop = intervals[0]
        self.assertLess(start, 1.90)
        self.assertGreater(stop, 2.09)


class CalibrationLineTests(unittest.TestCase):
    def test_line_excludes_standards_above_cutoff_and_origin_variant_has_zero_intercept(self):
        concentration = np.array([0, 2.5, 5, 10, 13.5, 18, 24, 32])
        signal = 1000 * concentration + 500
        signal[-1] = 40000  # off-trend top standard
        line = linear_conversion(concentration, signal, 24.0, through_origin=False)
        self.assertAlmostEqual(line["signal_per_ug_ml"], 1000.0, places=6)
        self.assertAlmostEqual(line["signal_intercept_au_um"], 500.0, places=6)
        self.assertNotIn(32.0, line["standards_used_ug_ml"])
        origin = linear_conversion(concentration, signal, 24.0, through_origin=True)
        self.assertEqual(origin["signal_intercept_au_um"], 0.0)
        self.assertGreater(origin["signal_per_ug_ml"], 1000.0)


class AlternativeMetricTests(unittest.TestCase):
    def test_linear_profile_slope_and_steepest_position(self):
        x = np.linspace(0, 6, 3001)
        y = 1.0 - 0.1 * x
        excluded = (x > 2.4) & (x < 2.8)
        y_with_dip = y.copy()
        y_with_dip[excluded] -= 0.3
        metrics = alternative_gradient_metrics(x, y_with_dip, excluded)
        self.assertAlmostEqual(metrics["band_linear_slope_per_mm"], -0.1, places=4)
        self.assertAlmostEqual(metrics["steepest_local_slope_per_mm"], -0.1, delta=0.01)


class DatasetHandlingTests(unittest.TestCase):
    def test_calibration_filenames_in_ng_and_ug_per_ml(self):
        folder = Path.cwd() / ".test_outputs" / "standards_names"
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("00000.nd2", "2531.nd2", "32.nd2", "2.5.nd2", "32withdevice.nd2", "grad.nd2"):
            (folder / name).write_bytes(b"")
        ug = [c for c, _ in _discover_standards(folder, "ug_per_ml")]
        ng = [c for c, _ in _discover_standards(folder, "ng_per_ml")]
        self.assertEqual(ug, [0.0, 2.5, 32.0, 2531.0])
        self.assertEqual(ng, [0.0, 0.0025, 0.032, 2.531])

    def test_partial_dic_crop_offset_and_orientation(self):
        crop = np.arange(10, dtype=float)[np.newaxis, :].repeat(3, axis=0)
        meta = {"gradient_position_centers_in_large_px": [[100.0, 0.0], [300.0, 0.0]], "crop_bounds_xyxy_px": [80, 0, 90, 3]}
        aligned, offset = _align_dic_crop_to_profile(crop, meta, tile_width_mm=0.1, dic_pixel_mm=0.001)
        self.assertAlmostEqual(offset, 0.03)  # tile half-width 50 px: profile starts at 50, crop at 80
        self.assertTrue(np.array_equal(aligned, crop))
        mirrored = {"gradient_position_centers_in_large_px": [[300.0, 0.0], [100.0, 0.0]], "crop_bounds_xyxy_px": [50, 0, 340, 3]}
        flipped, offset = _align_dic_crop_to_profile(crop, mirrored, tile_width_mm=0.1, dic_pixel_mm=0.001)
        self.assertAlmostEqual(offset, 0.01)
        self.assertTrue(np.array_equal(flipped, crop[:, ::-1]))

    def test_diffusion_from_width_growth(self):
        d_um2_s, t0_h = 50.0, -0.2
        t = np.array([0.0, 0.25, 0.5, 1.0, 2.0])
        w_mm = np.sqrt(np.pi * d_um2_s * (t - t0_h) * 3600.0) / 1000.0
        result = diffusion_from_widths(t, w_mm)
        self.assertAlmostEqual(result["D_um2_per_s"], d_um2_s, places=6)
        self.assertAlmostEqual(result["t0_h"], t0_h, places=6)

    def test_front_motion_speed_and_d(self):
        rows = front_motion_table(np.array([0.0, 1.0, 4.0]), np.array([5.0, 4.4, 3.8]))
        self.assertAlmostEqual(rows[0]["displacement_um"], -600.0)
        self.assertAlmostEqual(rows[0]["front_speed_um_per_s"], -600.0 / 3600.0)
        self.assertAlmostEqual(rows[1]["D_dx2_over_dt_um2_per_s"], 1200.0 ** 2 / (4 * 3600.0))

    def test_hour_labels(self):
        self.assertEqual(_hl(24.0), "24")
        self.assertEqual(_hl(0.45), "0.45")
        self.assertEqual(_hl(23.951), "23.95")


if __name__ == "__main__":
    unittest.main()
