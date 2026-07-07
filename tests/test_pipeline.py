"""Tests for path/layout helpers used by the pipeline."""

import unittest
from pathlib import Path

from gradient_analysis.pipeline import _timepoint_folder


class PipelineTests(unittest.TestCase):
    def test_step1_timepoint_folder_is_zero_padded(self):
        self.assertEqual(_timepoint_folder(Path("step_01_stitched_images"), 18), Path("step_01_stitched_images") / "t018")


if __name__ == "__main__":
    unittest.main()
