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
from tiff_writer import write_16bit_tiff

class TestTIFFArtifacts(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_linear_tiff_preserves_grain_without_double_demosaicing(self):
        """
        Verify that processing a 16-bit TIFF image does not pass through rawpy demosaicing
        which causes maze grid artifacts on high frequency grain noise.
        """
        np.random.seed(42)
        h, w = 60, 60
        # Create image with high-frequency random noise (film grain)
        grain_data = np.random.randint(10000, 40000, size=(h, w, 3), dtype=np.uint16)
        
        input_tiff = os.path.join(self.test_dir, "test_grain_input.tiff")
        write_16bit_tiff(input_tiff, grain_data, is_monochrome=False)
        
        # Process positive
        process_positives(input_tiff, output_dir=os.path.join(self.test_dir, "Positives"), clip=0.0, gamma=1.0)
        
        output_tiff = os.path.join(self.test_dir, "Positives", "Positive_test_grain_input.tiff")
        self.assertTrue(os.path.exists(output_tiff))
        
        out_img = tifffile.imread(output_tiff)
        self.assertEqual(out_img.shape, (h, w, 3))
        
        # Calculate standard deviation of high-frequency noise difference
        # Verify that output remains sharp high-frequency structure
        self.assertGreater(np.std(out_img), 1000)

if __name__ == '__main__':
    unittest.main()
