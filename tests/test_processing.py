"""Tests for the data-independent analysis functions."""

import unittest
from unittest.mock import patch

import numpy as np

from gradient_analysis.processing import (
    choose_reference,
    colorize_scaled,
    correct_tile,
    correct_tile_2d,
    correction_curve,
    empty_uint16_histogram,
    fit_overlap_log_quadratic,
    fit_tanh_profile,
    feather_profiles,
    feather_tiles_2d,
    fitted_illumination_profile,
    histogram_percentile_range,
    image_percentile_range,
    merge_rgb,
    overlap_log_ratio_samples,
    physical_x_axes_mm,
    profile_curvature,
    profile_normalization_constants,
    quadratic_illumination_profile,
    reference_candidate_score,
    smoothed_illumination_image,
    smoothed_illumination_profile,
    subtract_background_floor,
    smooth_image,
    smooth_z_coefficients,
    split_equal_width,
    stitch,
    to_uint16,
    update_uint16_histogram,
    x_profile,
)


class OverlapCorrectionTests(unittest.TestCase):
    def test_recovers_profile_with_y_shift_and_gradient(self):
        height, width, offset, y_shift = 80, 120, 94, 7
        overlap = width - offset
        local_x = (np.arange(width) - (width - 1) / 2) / ((width - 1) / 2)
        illumination = np.exp(0.08 * local_x - 0.28 * local_x**2)
        global_x = np.arange(width + offset, dtype=float)
        scene = 800.0 + 2.5 * global_x
        left = np.broadcast_to(scene[:width] * illumination, (height, width)).copy()
        right = np.broadcast_to(scene[offset : offset + width] * illumination, (height, width)).copy()
        xl, xr, ratios, details = overlap_log_ratio_samples(
            left, right, offset, y_shift, 5.0, 65535.0
        )
        fitted, model = fit_overlap_log_quadratic(width, xl, xr, ratios)
        expected = illumination / np.median(illumination)
        np.testing.assert_allclose(fitted, expected, rtol=0.02, atol=0.01)
        corrected_left = correct_tile(left, 1.0 / fitted)
        corrected_right = correct_tile(right, 1.0 / fitted)
        np.testing.assert_allclose(
            corrected_left[y_shift:, offset:], corrected_right[:-y_shift, :overlap], rtol=0.02
        )
        self.assertEqual(details["overlap_width_px"], overlap)
        self.assertEqual(model["sample_count"], overlap)

    def test_excludes_dark_and_clipped_pixels(self):
        left = np.full((5, 10), 100.0)
        right = np.full((5, 10), 50.0)
        left[:, 7] = 0.0
        right[:, 1] = 65535.0
        _, _, ratios, details = overlap_log_ratio_samples(left, right, 7, 0, 5.0, 65535.0)
        self.assertEqual(ratios.size, 1)
        self.assertEqual(details["valid_column_count"], 1)


class FeatherAndTanhTests(unittest.TestCase):
    def test_cosine_feather_is_continuous_and_uses_one_physical_grid(self):
        axes = [np.arange(6, dtype=float), np.arange(4, 10, dtype=float)]
        profiles = [np.full(6, 10.0), np.full(6, 20.0)]
        x, combined, details = feather_profiles(axes, profiles)
        np.testing.assert_array_equal(x, np.arange(10, dtype=float))
        np.testing.assert_allclose(combined, [10, 10, 10, 10, 10, 20, 20, 20, 20, 20])
        self.assertEqual(details["overlap_widths_px"], [2])

    def test_cosine_feather_preserves_identical_gradient(self):
        axes = [np.arange(6, dtype=float), np.arange(4, 10, dtype=float)]
        profiles = [2 * axes[0] + 3, 2 * axes[1] + 3]
        x, combined, _ = feather_profiles(axes, profiles)
        np.testing.assert_allclose(combined, 2 * x + 3)

    def test_unequal_overlaps_remain_gap_free(self):
        axes = [np.arange(7, dtype=float), np.arange(4, 11, dtype=float), np.arange(9, 16, dtype=float)]
        profiles = [axis.copy() for axis in axes]
        x, combined, details = feather_profiles(axes, profiles)
        np.testing.assert_allclose(combined, x)
        self.assertEqual(details["overlap_widths_px"], [3, 2])
        self.assertEqual(x.size, 16)

    def test_2d_feather_uses_stage_x_y_and_preserves_physical_image(self):
        height, width = 5, 6
        x_starts = [0.0, 4.0]
        y_starts = [0.0, -1.0]
        tiles = []
        for x_start, y_start in zip(x_starts, y_starts):
            local_y, local_x = np.indices((height, width), dtype=float)
            tiles.append(100.0 + 3.0 * (local_x + x_start) + 7.0 * (local_y + y_start))
        mosaic, details = feather_tiles_2d(tiles, x_starts, y_starts)
        global_y, global_x = np.indices((4, 10), dtype=float)
        expected = 100.0 + 3.0 * global_x + 7.0 * global_y
        np.testing.assert_allclose(mosaic, expected, atol=1e-5)
        self.assertEqual(details["overlap_widths_px"], [2])
        self.assertEqual(details["common_y_height_px"], 4)
        self.assertEqual(details["output_width_px"], 10)

    def test_z_coefficient_smoothing_downweights_bad_plane(self):
        expected = np.linspace(-0.2, 0.2, 9)
        measured = expected.copy()
        measured[4] = 2.0
        weights = np.ones(9)
        weights[4] = 0.01
        smoothed, normalized_weights = smooth_z_coefficients(measured, weights, penalty=10.0)
        self.assertLess(abs(smoothed[4] - expected[4]), abs(measured[4] - expected[4]))
        self.assertLess(normalized_weights[4], normalized_weights[3])

    def test_quadratic_profile_is_positive_and_median_normalized(self):
        profile = quadratic_illumination_profile(101, -0.12, -0.25)
        self.assertTrue(np.all(profile > 0))
        self.assertAlmostEqual(float(np.median(profile)), 1.0, places=12)

    def test_channel_normalization_uses_one_global_profile_maximum(self):
        profiles = {"GFP": [np.array([1.0, 5.0]), np.array([2.0, 10.0])],
                    "Cy5": [np.array([3.0, 6.0]), np.array([1.0, 2.0])]}
        constants = profile_normalization_constants(profiles)
        self.assertEqual(constants, {"GFP": 10.0, "Cy5": 6.0})
        for channel in profiles:
            normalized = [profile / constants[channel] for profile in profiles[channel]]
            self.assertEqual(max(float(np.max(profile)) for profile in normalized), 1.0)

    def test_tanh_fit_recovers_increasing_and_decreasing_slopes(self):
        x = np.linspace(0.0, 10.0, 3000)
        for left, right in ((0.1, 0.9), (0.9, 0.1)):
            y = left + 0.5 * (right - left) * (1 + np.tanh((x - 4.0) / 1.2))
            fit = fit_tanh_profile(x, y, max_points=512, pixel_size_mm=x[1] - x[0])
            self.assertAlmostEqual(float(fit["midpoint_mm"]), 4.0, places=2)
            self.assertAlmostEqual(float(fit["width_mm"]), 1.2, places=2)
            self.assertAlmostEqual(
                float(fit["signed_slope_per_mm"]), (right - left) / 2.4, places=2
            )
            self.assertAlmostEqual(
                float(fit["absolute_slope_per_mm"]), abs((right - left) / 2.4), places=2
            )

    def test_tanh_fit_flags_flat_profile(self):
        x = np.linspace(0.0, 5.0, 100)
        fit = fit_tanh_profile(x, np.full_like(x, 0.4), max_points=100)
        self.assertIn("low_amplitude", fit["qc_flags"])
        self.assertFalse(fit["fit_valid"])

    def test_tanh_optimizer_failure_returns_flagged_record(self):
        x = np.linspace(0.0, 5.0, 100)
        y = 0.2 + 0.6 * (1 + np.tanh((x - 2.5) / 0.8)) / 2
        with patch("scipy.optimize.least_squares", side_effect=RuntimeError("synthetic failure")):
            fit = fit_tanh_profile(x, y, max_points=100)
        self.assertFalse(fit["optimizer_success"])
        self.assertFalse(fit["fit_valid"])
        self.assertIn("non_convergence", fit["qc_flags"])
        self.assertIn("synthetic failure", fit["optimizer_message"])


class ProcessingTests(unittest.TestCase):
    def test_background_subtraction_floors_negative_values(self):
        image = np.array([[75, 100, 125]], dtype=np.uint16)
        np.testing.assert_array_equal(subtract_background_floor(image, 100), [[0, 0, 25]])

    def test_quadratic_correction_uses_zero_based_signal_without_restoring_background(self):
        x = np.linspace(-1, 1, 101)
        signal = 300 - 60 * x**2
        acquired = np.repeat((signal + 100)[None, :], 8, axis=0)
        zero_based = subtract_background_floor(acquired, 100)
        fitted, plateau = fitted_illumination_profile(x_profile(zero_based), 9, degree=2)
        corrected = correct_tile(zero_based, plateau / fitted)
        self.assertLess(np.std(x_profile(corrected)), 1.0)
        self.assertAlmostEqual(float(np.mean(corrected)), plateau, delta=1.0)
        self.assertLess(float(np.mean(corrected)), float(np.mean(acquired)) - 90)

    def test_stitch_and_x_profile(self):
        left = np.full((3, 2), 2, dtype=np.uint16)
        right = np.full((3, 2), 6, dtype=np.uint16)
        result = stitch([left, right])
        np.testing.assert_array_equal(x_profile(result), [2, 2, 6, 6])

    def test_correction_flattens_parabolic_illumination(self):
        x = np.linspace(-1, 1, 101)
        illumination = 100 - 40 * x**2
        tile = np.repeat(illumination[None, :], 8, axis=0)
        curve, _ = correction_curve(x_profile(tile), smoothing_window=9)
        corrected_tile = correct_tile(tile, curve)
        self.assertEqual(corrected_tile.dtype, np.float32)
        corrected = x_profile(corrected_tile)
        self.assertLess(np.std(corrected) / np.mean(corrected), 0.015)

    def test_quadratic_fit_tracks_broad_laser_profile(self):
        x = np.linspace(-1, 1, 101)
        illumination = 100 - 35 * x**2
        noisy = illumination + np.sin(np.arange(101)) * 3
        fitted, plateau = fitted_illumination_profile(noisy, smoothing_window=9, degree=2)
        self.assertEqual(fitted.shape, noisy.shape)
        self.assertGreater(plateau, 80)
        self.assertLess(np.std((illumination / fitted) / np.mean(illumination / fitted)), 0.04)

    def test_smoothed_profile_correction_handles_sharp_edge_falloff_better_than_quadratic(self):
        x = np.linspace(-1, 1, 301)
        illumination = 100 - 20 * x**2
        illumination[:24] *= np.linspace(0.35, 1.0, 24)
        illumination[-24:] *= np.linspace(1.0, 0.35, 24)
        tile = np.repeat(illumination[None, :], 8, axis=0)
        reference = x_profile(tile)
        quadratic, quadratic_plateau = fitted_illumination_profile(reference, smoothing_window=15, degree=2)
        smoothed, smoothed_plateau = smoothed_illumination_profile(reference, smoothing_window=15)
        quadratic_corrected = x_profile(correct_tile(tile, quadratic_plateau / quadratic))
        smoothed_corrected = x_profile(correct_tile(tile, smoothed_plateau / smoothed))
        raw_variation = np.std(reference) / np.mean(reference)
        quadratic_variation = np.std(quadratic_corrected) / np.mean(quadratic_corrected)
        smoothed_variation = np.std(smoothed_corrected) / np.mean(smoothed_corrected)
        self.assertLess(quadratic_variation, raw_variation)
        self.assertLess(smoothed_variation, quadratic_variation)

    def test_2d_flatfield_correction_flattens_y_and_x_illumination(self):
        y = np.linspace(-1, 1, 61)[:, None]
        x = np.linspace(-1, 1, 81)[None, :]
        illumination = 100 - 18 * x**2 - 12 * y**2
        illumination[:, :8] *= np.linspace(0.5, 1.0, 8)
        illumination[:8, :] *= np.linspace(0.7, 1.0, 8)[:, None]
        one_d_profile = x_profile(illumination)
        one_d_fit, one_d_plateau = smoothed_illumination_profile(one_d_profile, 9)
        one_d_corrected = correct_tile(illumination, one_d_plateau / one_d_fit)
        fitted, plateau = smoothed_illumination_image(illumination, 9, 9)
        corrected = correct_tile_2d(illumination, fitted, plateau)
        raw_variation = np.std(illumination) / np.mean(illumination)
        one_d_variation = np.std(one_d_corrected) / np.mean(one_d_corrected)
        two_d_variation = np.std(corrected) / np.mean(corrected)
        self.assertLess(one_d_variation, raw_variation)
        self.assertLess(two_d_variation, one_d_variation)

    def test_smooth_image_rejects_non_2d_inputs(self):
        with self.assertRaises(ValueError):
            smooth_image(np.arange(5), 3, 3)

    def test_reference_selection_prefers_bright_flat_profile(self):
        curved = 100 - 30 * np.linspace(-1, 1, 101) ** 2
        flat = np.full(101, 95.0)
        dim_flat = np.full(101, 20.0)
        self.assertEqual(choose_reference([curved, flat, dim_flat], 9), 1)

    def test_stable_reference_score_prefers_normalized_shape_stability(self):
        x = np.linspace(-1, 1, 101)
        laser = 100 - 30 * x**2
        stable_profiles = [laser * 1.0, laser * 1.4, laser * 0.8]
        changing_profiles = [laser, laser * (1.0 + 0.3 * x), laser * (1.0 - 0.3 * x)]
        stable = reference_candidate_score(stable_profiles, [100, 140, 80], [0, 0, 0], 9, 0.001, 140)
        changing = reference_candidate_score(changing_profiles, [120, 120, 120], [0, 0, 0], 9, 0.001, 140)
        self.assertLess(stable["shape_variability"], changing["shape_variability"])
        self.assertGreater(stable["score"], changing["score"])

    def test_reference_score_penalizes_saturation(self):
        profile = np.full(51, 100.0)
        clean = reference_candidate_score([profile], [100], [0.0], 5, 0.001, 100)
        saturated = reference_candidate_score([profile], [100], [0.01], 5, 0.001, 100)
        self.assertTrue(clean["unsaturated"])
        self.assertFalse(saturated["unsaturated"])
        self.assertLess(saturated["score"], clean["score"])

    def test_profile_curvature_reports_quadratic_component(self):
        x = np.linspace(-1, 1, 101)
        curved = 100 - 25 * x**2
        self.assertLess(profile_curvature(curved, 9), 0.0)

    def test_merge_has_green_and_magenta_components(self):
        green = np.arange(16, dtype=float).reshape(4, 4)
        magenta = np.flip(green, axis=1)
        result = merge_rgb(
            [green, magenta],
            [(0, 1, 0), (1, 0, 1)],
            [(0, 100), (0, 100)],
        )
        self.assertEqual(result.shape, (4, 4, 3))
        np.testing.assert_array_equal(result[..., 0], result[..., 2])

    def test_colorized_preview_uses_channel_color(self):
        image = np.array([[0, 50], [100, 150]], dtype=np.uint16)
        result = colorize_scaled(image, 0, 100, (0.0, 1.0, 0.0))
        self.assertEqual(result.shape, (2, 2, 3))
        self.assertTrue(np.all(result[..., 0] == 0))
        self.assertTrue(np.all(result[..., 2] == 0))
        self.assertGreater(result[..., 1].max(), 0)

    def test_local_and_global_preview_ranges_can_differ(self):
        dim = np.array([[0, 5], [10, 15]], dtype=np.uint16)
        bright = np.array([[0, 500], [1000, 1500]], dtype=np.uint16)
        local = image_percentile_range(dim, 0, 100)
        histogram = empty_uint16_histogram()
        update_uint16_histogram(histogram, dim)
        update_uint16_histogram(histogram, bright)
        global_range = histogram_percentile_range(histogram, 0, 100)
        self.assertEqual(local, (0, 15))
        self.assertEqual(global_range, (0, 1500))

    def test_physical_x_axes_start_p01_left_edge_at_zero(self):
        axes = physical_x_axes_mm([1000.0, 0.0], pixel_size_um=500.0, tile_width=2)
        np.testing.assert_allclose(axes[0], [0.0, 0.5])
        np.testing.assert_allclose(axes[1], [1.0, 1.5])
        self.assertGreater(axes[1][0] - axes[0][-1], 0.0)

    def test_split_equal_width_recovers_tiles(self):
        image = np.arange(12, dtype=np.uint16).reshape(2, 6)
        tiles = split_equal_width(image, 3)
        self.assertEqual(len(tiles), 3)
        np.testing.assert_array_equal(tiles[1], image[:, 2:4])

    def test_streaming_histogram_produces_one_global_range(self):
        histogram = empty_uint16_histogram()
        update_uint16_histogram(histogram, np.array([[0, 10]], dtype=np.uint16))
        update_uint16_histogram(histogram, np.array([[20, 30]], dtype=np.uint16))
        self.assertEqual(histogram_percentile_range(histogram, 0, 100), (0, 30))

    def test_uint16_conversion_rounds_and_reports_clipping(self):
        values = np.array([[-2.0, 1.4, 2.6, 70_000.0, np.nan]], dtype=np.float32)
        converted, clipped = to_uint16(values)
        np.testing.assert_array_equal(converted, [[0, 1, 3, 65_535, 0]])
        self.assertEqual(converted.dtype, np.uint16)
        self.assertEqual(clipped, 3)

    def test_concatenated_corrected_profiles_equal_stitched_profile(self):
        left = np.arange(12, dtype=np.float32).reshape(3, 4)
        right = left + 20
        expected = x_profile(stitch([left, right]))
        actual = np.concatenate([x_profile(left), x_profile(right)])
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
