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

class TestReversalProcessing(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
        # Create synthetic 16-bit RGB image where pixel (0,0) is dark (1000) and (9,9) is bright (60000)
        self.rgb_data = np.zeros((10, 10, 3), dtype=np.uint16)
        for y in range(10):
            for x in range(10):
                val = 1000 + (x + y) * 3000
                self.rgb_data[y, x, :] = val

        self.rgb_filepath = os.path.join(self.test_dir, "test_reversal.tiff")
        write_16bit_tiff(self.rgb_filepath, self.rgb_data, is_monochrome=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_reversal_processing_no_inversion(self):
        # When reversal=True, dark pixels stay darker than bright pixels (no density inversion)
        process_positives(self.rgb_filepath, clip=0.0, gamma=1.0, reversal=True)
        
        out_filepath = os.path.join(self.test_dir, "Positives", "Positive_test_reversal.tiff")
        self.assertTrue(os.path.exists(out_filepath))
        
        out_img = tifffile.imread(out_filepath)
        self.assertEqual(out_img.shape, (10, 10, 3))
        self.assertEqual(out_img.dtype, np.uint16)

        # Confirm dark pixel remains darker than bright pixel (positive polarity preserved)
        dark_pixel_val = out_img[0, 0, 0]
        bright_pixel_val = out_img[9, 9, 0]
        self.assertLess(dark_pixel_val, bright_pixel_val)

    def test_negative_processing_inversion(self):
        # When reversal=False (default negative behavior), dark input pixels invert to bright output values
        process_positives(self.rgb_filepath, clip=0.0, gamma=1.0, reversal=False)
        
        out_filepath = os.path.join(self.test_dir, "Positives", "Positive_test_reversal.tiff")
        self.assertTrue(os.path.exists(out_filepath))
        
        out_img = tifffile.imread(out_filepath)
        dark_input_out_val = out_img[0, 0, 0]
        bright_input_out_val = out_img[9, 9, 0]
        # Inverted: dark input produces high output value, bright input produces lower output value
        self.assertGreater(dark_input_out_val, bright_input_out_val)

    def test_direct_raw_positives_without_tiff_conversion(self):
        # When convert_to_tiff=False, the input file is copied directly into positives folder without rendering TIFF
        process_positives(self.rgb_filepath, reversal=True, convert_to_tiff=False)
        
        out_filepath = os.path.join(self.test_dir, "Positives", "Positive_test_reversal.tiff")
        self.assertTrue(os.path.exists(out_filepath))
        
        # Verify file size matches original file size
        self.assertEqual(os.path.getsize(out_filepath), os.path.getsize(self.rgb_filepath))

if __name__ == '__main__':
    unittest.main()
