import os
import sys
import time
import tempfile
import shutil
import unittest
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

# Ensure src is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from compositor import fourier_shift_2d, align_channel, _HAS_SCIPY_FFT
from camera_manager import CameraManager
from batch_worker import (
    worker_process_triplet,
    worker_process_positives,
    worker_process_triplet_pipeline
)

class TestPerfConcurrency(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_scipy_fft_acceleration_available(self):
        """Verify that scipy.fft acceleration is detected and functional."""
        self.assertTrue(_HAS_SCIPY_FFT, "scipy.fft should be installed and enabled for multi-threaded SIMD acceleration")

    def test_fourier_shift_and_align_accuracy(self):
        """Test sub-pixel accuracy of accelerated Fourier phase shift and correlation."""
        h, w = 256, 256
        y, x = np.mgrid[0:h, 0:w]
        spot = np.exp(-((x - 128.0)**2 + (y - 128.0)**2) / (2 * 12.0**2)) * 40000.0
        ref_img = (spot + 2000.0).astype(np.uint16)

        # Shift by sub-pixel offset
        dy, dx = 1.5, -2.25
        shifted = fourier_shift_2d(ref_img, dy, dx)

        # Re-align back to ref_img
        aligned = align_channel(ref_img, shifted, "PerfTest")

        # Aligned image should closely match ref_img
        diff = np.abs(aligned[30:220, 30:220].astype(np.float32) - ref_img[30:220, 30:220].astype(np.float32))
        self.assertLess(float(np.mean(diff)), 400.0)

    def test_liveview_fast_focus_thumbnail_cache(self):
        """Verify skip-frame and downscaling thumbnail cache during fast manual focus."""
        cm = CameraManager()
        cm.simulated = True
        cm.set_liveview(True)
        cm.start()
        try:
            # Initially no focus activity
            self.assertFalse(cm.is_fast_focus_active(window_sec=0.5))

            # Simulate manual focus adjustment
            cm.notify_focus_adjustment()
            self.assertTrue(cm.is_fast_focus_active(window_sec=1.0))

            # Wait for at least one live view frame to generate
            for _ in range(50):
                if cm.latest_frame:
                    break
                time.sleep(0.02)
            self.assertIsNotNone(cm.latest_frame)

            # Request downscaled thumbnail
            thumb = cm.get_fast_focus_frame(max_width=320, quality=65)
            self.assertIsNotNone(thumb)
            self.assertLess(len(thumb), len(cm.latest_frame))

            # Verify cache hit
            thumb2 = cm.get_fast_focus_frame(max_width=320, quality=65)
            self.assertEqual(thumb, thumb2)
        finally:
            cm.stop()

    def test_multiprocessing_batch_workers(self):
        """Verify ProcessPoolExecutor successfully executes batch workers out-of-process."""
        # Create mock triplet files
        f1 = os.path.join(self.test_dir, "Frame_01_Capture_red.cr3")
        f2 = os.path.join(self.test_dir, "Frame_01_Capture_green.cr3")
        f3 = os.path.join(self.test_dir, "Frame_01_Capture_blue.cr3")
        for f in [f1, f2, f3]:
            with open(f, "w") as fp:
                fp.write("MOCK RAW DATA")

        out_composite = os.path.join(self.test_dir, "Frame_01_Composite.dng")
        out_positives = os.path.join(self.test_dir, "Positives")
        os.makedirs(out_positives, exist_ok=True)

        config = {
            "neutralize": False,
            "compress_tiff": False,
            "align_channels": False,
            "color_profile": "adobe_rgb",
            "embed_metadata": False,
            "clip": 0.1,
            "gamma": 2.2,
            "margin": 0.1,
            "scurve": 0.0,
            "autocrop": False,
            "monochrome": False
        }

        # Run out-of-process via ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                worker_process_triplet_pipeline,
                group=[f1, f2, f3],
                composite_filepath=out_composite,
                positives_dir=out_positives,
                config=config
            )
            success, result, logs = future.result(timeout=15.0)

        self.assertTrue(success, f"Pipeline worker failed: {result}")
        self.assertTrue(os.path.exists(out_composite))
        self.assertIn("Frame_01_Composite", logs)
        self.assertEqual(len(result), 3) # (r_mean, g_mean, b_mean)

if __name__ == '__main__':
    unittest.main()
