import os
import shutil
import tempfile
import unittest
import numpy as np
import tifffile
import sys
import json
import subprocess

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from compositor import process_triplet
from inverter import process_positives
from tiff_writer import write_16bit_tiff
from web_ui import app, session


class TestFilmBaseNeutralizationPicker(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.orig_root = session.root_folder
        session.root_folder = self.test_dir
        self.app = app.test_client()
        self.app.testing = True

    def tearDown(self):
        session.root_folder = self.orig_root
        shutil.rmtree(self.test_dir)

    def test_compositor_with_custom_base_ratios(self):
        """
        Test that compositor.process_triplet scales channels according to custom base_ratios.
        """
        h, w = 50, 50
        # Create synthetic R, G, B channels with known orange mask levels
        # e.g., R=50000, G=30000, B=15000
        r_img = np.full((h, w), 50000, dtype=np.uint16)
        g_img = np.full((h, w), 30000, dtype=np.uint16)
        b_img = np.full((h, w), 15000, dtype=np.uint16)

        r_path = os.path.join(self.test_dir, "frame_001_R.tiff")
        g_path = os.path.join(self.test_dir, "frame_001_G.tiff")
        b_path = os.path.join(self.test_dir, "frame_001_B.tiff")

        tifffile.imwrite(r_path, r_img)
        tifffile.imwrite(g_path, g_img)
        tifffile.imwrite(b_path, b_img)

        output_path = os.path.join(self.test_dir, "frame_001_Composite.tiff")
        
        # Test with custom base ratios matching the orange mask: (50000, 30000, 15000) -> normalized (1.0, 0.6, 0.3)
        res = process_triplet(
            [r_path, g_path, b_path],
            output_filepath=output_path,
            neutralize_base=True,
            base_ratios=(1.0, 0.6, 0.3)
        )

        self.assertTrue(res)
        self.assertTrue(os.path.exists(output_path))
        comp = tifffile.imread(output_path)

        # After scaling with base_ratios, all 3 channels at the rebate should be neutralized to equal values
        mean_r = np.mean(comp[:, :, 0])
        mean_g = np.mean(comp[:, :, 1])
        mean_b = np.mean(comp[:, :, 2])

        self.assertAlmostEqual(mean_r, mean_g, delta=500)
        self.assertAlmostEqual(mean_g, mean_b, delta=500)

    def test_inverter_with_custom_base_ratios(self):
        """
        Test that inverter.process_positives uses custom base_ratios when computing density.
        """
        h, w = 60, 60
        # Synthetic negative with orange mask rebate
        # Film base: R=50000, G=30000, B=15000
        neg_data = np.full((h, w, 3), 0, dtype=np.uint16)
        neg_data[:, :, 0] = 50000
        neg_data[:, :, 1] = 30000
        neg_data[:, :, 2] = 15000

        # Add exposed content in center
        neg_data[15:45, 15:45, 0] = 10000
        neg_data[15:45, 15:45, 1] = 8000
        neg_data[15:45, 15:45, 2] = 5000

        input_path = os.path.join(self.test_dir, "test_neg.tiff")
        write_16bit_tiff(input_path, neg_data, is_monochrome=False)

        pos_dir = os.path.join(self.test_dir, "Positives")
        process_positives(
            input_path,
            output_dir=pos_dir,
            clip=0.0,
            gamma=1.0,
            global_levels=True,
            base_ratios=(1.0, 0.6, 0.3)
        )

        out_path = os.path.join(pos_dir, "Positive_test_neg.tiff")
        self.assertTrue(os.path.exists(out_path))
        pos_img = tifffile.imread(out_path)

        # At rebate region (e.g. [0:10, 0:10]), density should be 0, so inverted positive should be ~0 (pure black / Dmin)
        rebate_r = np.mean(pos_img[0:10, 0:10, 0])
        rebate_g = np.mean(pos_img[0:10, 0:10, 1])
        rebate_b = np.mean(pos_img[0:10, 0:10, 2])

        self.assertLess(rebate_r, 2000)
        self.assertLess(rebate_g, 2000)
        self.assertLess(rebate_b, 2000)

    def test_api_config_get_and_post(self):
        """
        Test /api/config endpoint to update and read back base_ratios.
        """
        # Set base_ratios via POST
        post_data = {
            "neutralize": True,
            "base_ratios": [1.0, 0.55, 0.25]
        }
        res = self.app.post('/api/config', data=json.dumps(post_data), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data["config"]["base_ratios"], [1.0, 0.55, 0.25])

        # Get config
        res = self.app.get('/api/config')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data["config"]["base_ratios"], [1.0, 0.55, 0.25])

        # Reset base_ratios to None
        res = self.app.post('/api/config', data=json.dumps({"base_ratios": None}), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsNone(data["config"]["base_ratios"])

    def test_api_sample_rebate_direct_rgb(self):
        """
        Test /api/sample_rebate with direct sRGB preview values.
        """
        payload = {
            "rgb": [240, 150, 75]
        }
        res = self.app.post('/api/sample_rebate', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        ratios = data.get("base_ratios")
        self.assertIsNotNone(ratios)
        self.assertEqual(len(ratios), 3)
        self.assertEqual(ratios[0], 1.0)
        self.assertAlmostEqual(ratios[1], (150.0 / 240.0) ** 2.2, delta=0.05)
        self.assertAlmostEqual(ratios[2], (75.0 / 240.0) ** 2.2, delta=0.05)

    def test_api_sample_rebate_file_coordinate(self):
        """
        Test /api/sample_rebate with file path and relative coordinates (x_ratio, y_ratio).
        """
        h, w = 100, 100
        test_img = np.zeros((h, w, 3), dtype=np.uint16)
        # Set rebate region around center (50, 50) to R=60000, G=30000, B=15000
        test_img[40:60, 40:60, 0] = 60000
        test_img[40:60, 40:60, 1] = 30000
        test_img[40:60, 40:60, 2] = 15000

        img_path = os.path.join(self.test_dir, "sample_test.tiff")
        write_16bit_tiff(img_path, test_img, is_monochrome=False)

        payload = {
            "path": img_path,
            "x_ratio": 0.5,
            "y_ratio": 0.5
        }
        res = self.app.post('/api/sample_rebate', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        ratios = data.get("base_ratios")
        self.assertIsNotNone(ratios)
        self.assertAlmostEqual(ratios[0], 1.0, delta=0.01)
        self.assertAlmostEqual(ratios[1], 0.5, delta=0.01)
        self.assertAlmostEqual(ratios[2], 0.25, delta=0.01)

    def test_cli_base_ratios_parsing(self):
        """
        Verify that CLI scripts accept --base-ratios with 3 floats without errors.
        """
        import subprocess
        venv_python = sys.executable

        # Test inverter --help
        res = subprocess.run([venv_python, "src/inverter.py", "--help"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("--base-ratios", res.stdout)

        # Test compositor --help
        res = subprocess.run([venv_python, "src/compositor.py", "--help"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("--base-ratios", res.stdout)

        # Test reprocess_rolls --help
        res = subprocess.run([venv_python, "reprocess_rolls.py", "--help"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("--base-ratios", res.stdout)


if __name__ == '__main__':
    unittest.main()
