"""Tests for path/layout helpers used by the pipeline."""

import unittest
from pathlib import Path

import numpy as np

from gradient_analysis.pipeline import _position_max_difference_record, _timepoint_folder


class PipelineTests(unittest.TestCase):
    def test_step1_timepoint_folder_is_zero_padded(self):
        self.assertEqual(_timepoint_folder(Path("step_01_stitched_images"), 18), Path("step_01_stitched_images") / "t018")

    def test_max_difference_direction_depends_on_channel(self):
        p02 = np.array([[30]], dtype=np.uint16)
        p05 = np.array([[10]], dtype=np.uint16)
        gfp = _position_max_difference_record(
            timepoint=0, channel_label="GFP", p02_tile=p02, p05_tile=p05
        )
        cy5 = _position_max_difference_record(
            timepoint=0, channel_label="Cy5", p02_tile=p02, p05_tile=p05
        )
        self.assertEqual(gfp["max_intensity_difference"], 20.0)
        self.assertEqual(cy5["max_intensity_difference"], -20.0)


if __name__ == "__main__":
    unittest.main()
