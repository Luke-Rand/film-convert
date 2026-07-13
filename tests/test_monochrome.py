import os
import shutil
import tempfile
import unittest
import numpy as np
import tifffile
import sys

# Add src/ folder to python path so we can import modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from inverter import process_positives
from dng_writer import write_linear_dng

class TestMonochromeInversion(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
        # Create a dummy 16-bit RGB TIFF image (10x10x3) with different channel behavior:
        # - Red channel: horizontal gradient (1000 to 1900)
        # - Green channel: flat (15000)
        # - Blue channel: vertical gradient (1000 to 1900)
        self.rgb_data = np.zeros((10, 10, 3), dtype=np.uint16)
        for y in range(10):
            for x in range(10):
                self.rgb_data[y, x, 0] = 1000 + x * 100
                self.rgb_data[y, x, 1] = 15000
                self.rgb_data[y, x, 2] = 1000 + y * 100
        
        self.rgb_filepath = os.path.join(self.test_dir, "test_RGB.dng")
        write_linear_dng(self.rgb_filepath, self.rgb_data, is_monochrome=False)
        
        # Create a dummy 16-bit single-channel (grayscale) TIFF image (10x10)
        self.gray_data = np.zeros((10, 10), dtype=np.uint16)
        for y in range(10):
            for x in range(10):
                self.gray_data[y, x] = 1000 + x * 100
        self.gray_filepath = os.path.join(self.test_dir, "test_gray.dng")
        write_linear_dng(self.gray_filepath, self.gray_data, is_monochrome=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_native_grayscale_handling(self):
        # Native 2D images should be processed and saved as grayscale without throwing errors
        process_positives(self.gray_filepath, clip=0.0, gamma=1.0)
        
        out_filepath = os.path.join(self.test_dir, "Positives", "Positive_test_gray.dng")
        self.assertTrue(os.path.exists(out_filepath))
        
        out_img = tifffile.imread(out_filepath)
        self.assertEqual(out_img.ndim, 2)
        self.assertEqual(out_img.dtype, np.uint16)

    def test_monochrome_red_channel(self):
        # Extracting red channel (horizontal gradient) should produce a horizontal gradient
        process_positives(self.rgb_filepath, clip=0.0, gamma=1.0, monochrome=True, monochrome_channel="red")
        
        out_filepath = os.path.join(self.test_dir, "Positives", "Positive_test_RGB.dng")
        self.assertTrue(os.path.exists(out_filepath))
        
        out_img = tifffile.imread(out_filepath)
        self.assertEqual(out_img.ndim, 2)
        
        # Verify it has a horizontal gradient (differences across columns, but same down rows)
        self.assertTrue(np.any(np.diff(out_img, axis=1) != 0)) # cols differ
        self.assertTrue(np.all(np.diff(out_img, axis=0) == 0)) # rows are identical

    def test_monochrome_green_channel(self):
        # Extracting green channel (flat) should produce a flat/constant output
        process_positives(self.rgb_filepath, clip=0.0, gamma=1.0, monochrome=True, monochrome_channel="green")
        
        out_filepath = os.path.join(self.test_dir, "Positives", "Positive_test_RGB.dng")
        self.assertTrue(os.path.exists(out_filepath))
        
        out_img = tifffile.imread(out_filepath)
        self.assertEqual(out_img.ndim, 2)
        
        # Should be a flat image of zeroes (since clip/stretch normalizes the flat 1/15000 to constant)
        self.assertTrue(np.all(out_img == 0))

    def test_monochrome_blue_channel(self):
        # Extracting blue channel (vertical gradient) should produce a vertical gradient
        process_positives(self.rgb_filepath, clip=0.0, gamma=1.0, monochrome=True, monochrome_channel="blue")
        
        out_filepath = os.path.join(self.test_dir, "Positives", "Positive_test_RGB.dng")
        self.assertTrue(os.path.exists(out_filepath))
        
        out_img = tifffile.imread(out_filepath)
        self.assertEqual(out_img.ndim, 2)
        
        # Verify it has a vertical gradient (differences down rows, but same across columns)
        self.assertTrue(np.all(np.diff(out_img, axis=1) == 0)) # cols are identical
        self.assertTrue(np.any(np.diff(out_img, axis=0) != 0)) # rows differ

    def test_monochrome_average(self):
        # Average channel should combine all 3 channels
        process_positives(self.rgb_filepath, clip=0.0, gamma=1.0, monochrome=True, monochrome_channel="average")
        
        out_filepath = os.path.join(self.test_dir, "Positives", "Positive_test_RGB.dng")
        self.assertTrue(os.path.exists(out_filepath))
        
        out_img = tifffile.imread(out_filepath)
        self.assertEqual(out_img.ndim, 2)
        
        # Since it averages vertical + horizontal + flat, both rows and columns should vary
        self.assertTrue(np.any(np.diff(out_img, axis=0) != 0))
        self.assertTrue(np.any(np.diff(out_img, axis=1) != 0))

if __name__ == '__main__':
    unittest.main()
