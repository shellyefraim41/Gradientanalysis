"""Tests for the data-independent analysis functions."""

import unittest

import numpy as np

from gradient_analysis.processing import (
    choose_reference,
    colorize_scaled,
    correct_tile,
    correct_tile_2d,
    correction_curve,
    empty_uint16_histogram,
    fitted_illumination_profile,
    histogram_percentile_range,
    image_percentile_range,
    linear_fit,
    linear_fit_xy,
    flatten_segments,
    merge_rgb,
    physical_x_axes_mm,
    profile_curvature,
    reference_candidate_score,
    position_span,
    smoothed_illumination_image,
    smoothed_illumination_profile,
    subtract_background_floor,
    smooth_image,
    split_equal_width,
    stitch,
    to_uint16,
    update_uint16_histogram,
    x_profile,
)


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

    def test_bridge_position_span_for_p04(self):
        self.assertEqual(position_span(3, 2304), (6912, 9216))

    def test_physical_x_axes_start_p01_left_edge_at_zero(self):
        axes = physical_x_axes_mm([1000.0, 0.0], pixel_size_um=500.0, tile_width=2)
        np.testing.assert_allclose(axes[0], [0.0, 0.5])
        np.testing.assert_allclose(axes[1], [1.0, 1.5])
        self.assertGreater(axes[1][0] - axes[0][-1], 0.0)

    def test_flatten_segments_and_mm_linear_fit(self):
        axes = [np.array([0.0, 0.5]), np.array([2.0, 2.5])]
        profiles = [10.0 + 3.0 * axes[0], 10.0 + 3.0 * axes[1]]
        x, y = flatten_segments(axes, profiles, 0, 1)
        fit = linear_fit_xy(x, y)
        self.assertAlmostEqual(fit["slope"], 3.0)
        self.assertAlmostEqual(fit["intercept"], 10.0)
        self.assertAlmostEqual(fit["r_squared"], 1.0)

    def test_split_equal_width_recovers_tiles(self):
        image = np.arange(12, dtype=np.uint16).reshape(2, 6)
        tiles = split_equal_width(image, 3)
        self.assertEqual(len(tiles), 3)
        np.testing.assert_array_equal(tiles[1], image[:, 2:4])

    def test_linear_fit_reports_known_slope(self):
        profile = 4.0 * np.arange(10) + 7.0
        fit = linear_fit(profile, pixel_size_um=0.5)
        self.assertAlmostEqual(fit["slope_per_pixel"], 4.0)
        self.assertAlmostEqual(fit["slope_per_um"], 8.0)
        self.assertAlmostEqual(fit["intercept"], 7.0)
        self.assertAlmostEqual(fit["r_squared"], 1.0)

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
