import os
import shutil
import tempfile
import unittest
import numpy as np
import tifffile
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from inverter import process_positives
from tiff_writer import write_16bit_tiff

class TestHighContrastEdgeInversion(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_sharp_edge_no_black_blotch_collapse(self):
        """
        Verify that sharp high-contrast negative transitions (e.g., bright sky next to dark silhouette)
        do not produce zero-collapsed black blotches or non-monotonic artifacts.
        """
        h, w = 100, 100
        # Create a step edge negative: left side is clear film base (high transmission = 55000),
        # right side is dense negative highlight (low transmission = 2000)
        neg_data = np.zeros((h, w, 3), dtype=np.uint16)
        neg_data[:, :50, :] = 55000  # Clear film base / shadow in positive
        neg_data[:, 50:, :] = 2000   # Dense negative / bright sky in positive

        # Add slight boundary transition (1 pixel wide transition)
        neg_data[:, 49, :] = 35000

        input_path = os.path.join(self.test_dir, "test_edge_neg.tiff")
        write_16bit_tiff(input_path, neg_data, is_monochrome=False)

        pos_dir = os.path.join(self.test_dir, "Positives")
        process_positives(input_path, output_dir=pos_dir, clip=0.1, gamma=1.0)

        out_path = os.path.join(pos_dir, "Positive_test_edge_neg.tiff")
        self.assertTrue(os.path.exists(out_path))

        out_img = tifffile.imread(out_path)
        self.assertEqual(out_img.shape, (h, w, 3))

        # Check positive brightness: right side should be bright, left side should be dark
        left_val = np.mean(out_img[:, :40, :])
        right_val = np.mean(out_img[:, 60:, :])
        self.assertGreater(right_val, left_val)

        # Ensure the transition region values are monotonic across columns (no sudden zero-dip/black blotch)
        row_profile = out_img[50, :, 0]
        # From column 40 to 60, values must be non-decreasing
        diffs = np.diff(row_profile[40:60])
        self.assertTrue(np.all(diffs >= 0), f"Edge profile dipped non-monotonically: {row_profile[40:60]}")

if __name__ == '__main__':
    unittest.main()
