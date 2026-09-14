import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Add src/ folder to python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import compositor
import inverter

class TestGridArtifactFix(unittest.TestCase):
    def test_compositor_passes_four_color_rgb(self):
        """
        Verify that compositor.py passes four_color_rgb=True to rawpy postprocess
        to prevent G1/G2 channel imbalance maze grid artifacts.
        """
        mock_raw = MagicMock()
        mock_raw.postprocess.return_value = np.zeros((100, 100, 3), dtype=np.uint16)
        
        with patch('rawpy.imread') as mock_imread:
            mock_imread.return_value.__enter__.return_value = mock_raw
            with patch('os.path.exists', return_value=True):
                with patch('os.path.getsize', return_value=50000000): # Large, non-mock file
                    with patch('compositor.write_16bit_tiff'):
                        try:
                            compositor.process_triplet(
                                ["test_r.cr3", "test_g.cr3", "test_b.cr3"],
                                "out.tiff"
                            )
                        except Exception:
                            pass
                            
        self.assertTrue(mock_raw.postprocess.called)
        _, kwargs = mock_raw.postprocess.call_args
        self.assertTrue(kwargs.get('four_color_rgb', False), "four_color_rgb must be True in compositor.py")

    def test_inverter_passes_four_color_rgb(self):
        """
        Verify that inverter.py passes four_color_rgb=True to rawpy postprocess
        when demosaicing camera RAW files.
        """
        mock_raw = MagicMock()
        mock_raw.postprocess.return_value = np.zeros((100, 100, 3), dtype=np.uint16)
        
        with patch('rawpy.imread') as mock_imread:
            mock_imread.return_value.__enter__.return_value = mock_raw
            with patch('os.path.isfile', return_value=True):
                with patch('os.path.exists', return_value=True):
                    with patch('inverter.write_16bit_tiff'):
                        try:
                            inverter.process_positives(
                                "test_capture.cr3",
                                output_dir="Positives",
                                convert_to_tiff=True
                            )
                        except Exception:
                            pass
                            
        self.assertTrue(mock_raw.postprocess.called)
        _, kwargs = mock_raw.postprocess.call_args
        self.assertTrue(kwargs.get('four_color_rgb', False), "four_color_rgb must be True in inverter.py")

if __name__ == '__main__':
    unittest.main()
