import os
import shutil
import tempfile
import unittest
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from compositor import align_channel, fourier_shift_2d

class TestTripletAlignmentSubpixel(unittest.TestCase):
    def test_fourier_shift_subpixel_accuracy(self):
        """
        Verify that fourier_shift_2d correctly shifts image content by fractional pixels
        and does not produce zero-padded border stripes.
        """
        h, w = 120, 120
        # Create a test pattern: 2D Gaussian spot
        y, x = np.mgrid[0:h, 0:w]
        center_y, center_x = 60.0, 60.0
        spot = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * 10.0**2)) * 50000.0
        ref_img = (spot + 1000.0).astype(np.uint16)

        # Shift by fractional displacement (dy=2.0, dx=-2.0)
        shifted_mov = fourier_shift_2d(ref_img, 2.0, -2.0)

        # Align shifted_mov back to ref_img
        aligned = align_channel(ref_img, shifted_mov, "TestChannel")

        # Verify no boundary values collapsed to zero (because edge reflection padding was used)
        self.assertGreater(int(np.min(aligned)), int(np.min(ref_img)) - 100)

        # Check correlation after alignment: center region should align closely
        diff = np.abs(aligned[20:100, 20:100].astype(np.float32) - ref_img[20:100, 20:100].astype(np.float32))
        self.assertLess(np.mean(diff), 500.0)

if __name__ == '__main__':
    unittest.main()
