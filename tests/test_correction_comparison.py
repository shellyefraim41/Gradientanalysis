"""Synthetic tests for the correction-comparison helpers."""

import unittest

import numpy as np

from gradient_analysis.correction_comparison import (
    calibration_model_from_blocks,
    correct_ratios,
    fit_z_series,
    log_illumination,
    pair_samples,
    profile_pair_mismatch,
)


def _synthetic_ratios(width, offsets, log_profile, gradient, noise=0.0, seed=0):
    """Column log-ratios for tiles that see a true gradient times illumination."""
    rng = np.random.default_rng(seed)
    padded = np.full((len(offsets), width - min(offsets)), np.nan)
    start = 0
    for pair, offset in enumerate(offsets):
        columns = np.arange(width - offset)
        physical = start + offset + columns
        left = gradient(physical) * np.exp(log_profile[columns + offset])
        right = gradient(physical) * np.exp(log_profile[columns])
        padded[pair, columns] = np.log(left / right) + rng.normal(0, noise, columns.size)
        start += offset
    return padded


class CorrectionComparisonTests(unittest.TestCase):
    width = 400
    offsets = [280, 281, 279, 280]

    def test_correct_ratios_removes_known_illumination_exactly(self):
        log_profile = log_illumination(self.width, 0.1, -0.4)
        ratios = _synthetic_ratios(self.width, self.offsets, log_profile, lambda x: 50 + 0.2 * x)
        for pair, offset in enumerate(self.offsets):
            corrected = correct_ratios(ratios[pair], offset, log_profile)
            np.testing.assert_allclose(corrected[np.isfinite(corrected)], 0.0, atol=1e-12)

    def test_true_gradient_cancels_in_overlap_ratios(self):
        flat = np.zeros(self.width)
        ratios = _synthetic_ratios(self.width, self.offsets, flat, lambda x: 10 + np.exp(x / 300))
        self.assertTrue(np.allclose(ratios[np.isfinite(ratios)], 0.0))

    def test_fit_z_series_recovers_quadratic_and_generalizes_to_held_out_pair(self):
        log_profile = log_illumination(self.width, 0.05, -0.5)
        samples = []
        held_out = []
        for z in range(6):
            ratios = _synthetic_ratios(self.width, self.offsets, log_profile, lambda x: 20 + x, noise=0.002, seed=z)
            samples.append(pair_samples(ratios, self.offsets, [0, 1, 2]))
            held_out.append(ratios[3])
        fit = fit_z_series(samples, self.width, penalty=10.0)
        np.testing.assert_allclose(fit["quadratic"], -0.5, atol=0.02)
        for z in range(6):
            recovered = log_illumination(self.width, fit["linear"][z], fit["quadratic"][z])
            residual = correct_ratios(held_out[z], self.offsets[3], recovered)
            self.assertLess(np.nanmedian(np.abs(residual)), 0.005)

    def test_profile_pair_mismatch_measures_overlap_step(self):
        left = np.full(self.width, 100.0)
        right = np.full(self.width, 110.0)
        self.assertAlmostEqual(profile_pair_mismatch(left, right, 300, 1.0), np.log(1.1), places=12)

    def test_calibration_model_from_blocks_integrates_through_z(self):
        concentrations = [0.0, 10.0, 20.0]
        stacks = [np.full((3, 16), 5.0 * c + 1.0) for c in concentrations]
        rows, model = calibration_model_from_blocks(concentrations, stacks, [10.0, 10.0, 10.0])
        self.assertEqual([row["concentration_ug_ml"] for row in rows], concentrations)
        np.testing.assert_allclose(model["signals"], [20.0, 1020.0, 2020.0])
        self.assertAlmostEqual(model["linear_diagnostic"]["r_squared"], 1.0)
        self.assertTrue(model["strictly_monotonic"])


if __name__ == "__main__":
    unittest.main()
