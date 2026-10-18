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

    def test_in_memory_triplet_to_positive_handoff(self):
        """Verify process_triplet return_image=True and process_positives preloaded_image avoid disk re-reads."""
        from compositor import process_triplet
        from inverter import process_positives
        import tifffile
        from unittest.mock import patch

        f1 = os.path.join(self.test_dir, "Frame_01_Capture_red.cr3")
        f2 = os.path.join(self.test_dir, "Frame_01_Capture_green.cr3")
        f3 = os.path.join(self.test_dir, "Frame_01_Capture_blue.cr3")
        for f in [f1, f2, f3]:
            with open(f, "w") as fp:
                fp.write("MOCK RAW DATA")

        out_composite = os.path.join(self.test_dir, "Frame_01_Composite.dng")
        out_positives = os.path.join(self.test_dir, "Positives")
        os.makedirs(out_positives, exist_ok=True)

        means, comp_rgb = process_triplet(
            group=[f1, f2, f3],
            output_filepath=out_composite,
            preserve_metadata=False,
            return_image=True
        )
        self.assertEqual(len(means), 3)
        self.assertEqual(comp_rgb.dtype, np.uint16)

        # Ensure tifffile.imread is NOT called when preloaded_image is supplied
        with patch("inverter.tifffile.imread", side_effect=AssertionError("Should not read composite from disk!")):
            process_positives(
                input_path=out_composite,
                output_dir=out_positives,
                preserve_metadata=False,
                preloaded_image=comp_rgb
            )

        expected_pos = os.path.join(out_positives, "Frame_01_Positive.tiff")
        self.assertTrue(os.path.exists(expected_pos))

    def test_frontend_es_modules_and_no_duplicate_functions(self):
        """Verify ES module split (app.js, liveview.js, histogram.js, gallery.js, shortcuts.js) and single updateSliderVal."""
        from web_ui import app

        client = app.test_client()
        index_res = client.get("/")
        self.assertEqual(index_res.status_code, 200)
        self.assertIn(b'type="module" src="/static/js/app.js"', index_res.data)

        for mod in ["app.js", "liveview.js", "histogram.js", "gallery.js", "shortcuts.js"]:
            res = client.get(f"/static/js/{mod}")
            self.assertEqual(res.status_code, 200, f"Module {mod} failed to load")
            text = res.data.decode("utf-8")
            self.assertGreater(len(text), 100)
            if mod == "app.js":
                self.assertEqual(text.count("function updateSliderVal("), 1)
            res.close()

    def test_waitress_production_server_configuration(self):
        """Verify run_production_server delegates to waitress.serve with thread pool sizing."""
        from unittest.mock import patch
        from web_ui import app, run_production_server

        with patch("waitress.serve") as mock_serve:
            run_production_server(host="127.0.0.1", port=5099, threads=24)
            mock_serve.assert_called_once()
            args, kwargs = mock_serve.call_args
            self.assertIs(args[0], app)
            self.assertEqual(kwargs["host"], "127.0.0.1")
            self.assertEqual(kwargs["port"], 5099)
            self.assertEqual(kwargs["threads"], 24)
            self.assertEqual(kwargs["channel_timeout"], 300)
            self.assertEqual(kwargs["send_bytes"], 1)

    def test_local_preview_cache_avoids_network_tiff_reread(self):
        """Verify write_16bit_image pre-populates local SSD preview cache and /api/preview serves it without tifffile.imread."""
        from unittest.mock import patch
        from tiff_writer import write_16bit_image
        from metadata_preservation import get_preview_cache_key
        from web_ui import app, session

        session_dir = os.path.join(self.test_dir, "TestRoll-135-01")
        neg_dir = os.path.join(session_dir, "negatives")
        proc_dir = os.path.join(session_dir, "processed_raws")
        pos_dir = os.path.join(session_dir, "positives")
        os.makedirs(neg_dir, exist_ok=True)
        os.makedirs(proc_dir, exist_ok=True)
        os.makedirs(pos_dir, exist_ok=True)

        comp_neg_path = os.path.join(neg_dir, "Frame_01_Composite.dng")
        comp_proc_path = os.path.join(proc_dir, "Frame_01_Composite.dng")
        pos_path = os.path.join(pos_dir, "Frame_01_Positive.tiff")

        # Preview cache key must remain identical when Composite.dng moves from negatives/ to processed_raws/
        self.assertEqual(get_preview_cache_key(comp_neg_path), get_preview_cache_key(comp_proc_path))

        img_16 = np.full((128, 192, 3), 32000, dtype=np.uint16)
        write_16bit_image(comp_neg_path, img_16, compress=False)
        shutil.move(comp_neg_path, comp_proc_path)
        write_16bit_image(pos_path, img_16, compress=False)

        self.assertTrue(os.path.isfile(get_preview_cache_key(comp_proc_path)))
        self.assertTrue(os.path.isfile(get_preview_cache_key(pos_path)))

        with session.lock:
            saved_root = session.root_folder
            session.root_folder = self.test_dir
        try:
            client = app.test_client()
            with patch("web_ui.tifffile.imread", side_effect=AssertionError("Should serve from local preview cache without reading 16-bit TIFF/DNG!")):
                res_pos = client.get(f"/api/preview?path={pos_path}&w=400")
                self.assertEqual(res_pos.status_code, 200)
                self.assertEqual(res_pos.mimetype, "image/jpeg")
                res_pos.close()

                res_neg = client.get(f"/api/preview?path={comp_proc_path}&w=400")
                self.assertEqual(res_neg.status_code, 200)
                self.assertEqual(res_neg.mimetype, "image/jpeg")
                res_neg.close()
        finally:
            with session.lock:
                session.root_folder = saved_root

    def test_gphoto2_drain_preserves_file_added_and_stages_raw_locally(self):
        """Verify GPhoto2CameraBackend never drops GP_EVENT_FILE_ADDED during event drains and stages RAW files locally."""
        from unittest.mock import MagicMock, patch
        import camera_backends.gphoto2_backend as gp_mod
        from metadata_preservation import resolve_fast_read_path, cleanup_staged_raw

        backend = gp_mod.GPhoto2CameraBackend()
        mock_cam = MagicMock()
        backend.camera = mock_cam

        fake_gp = MagicMock()
        fake_gp.GP_EVENT_FILE_ADDED = 101
        fake_gp.GP_EVENT_TIMEOUT = 102
        fake_gp.GP_FILE_TYPE_NORMAL = 1

        evt_obj = MagicMock()
        evt_obj.folder = "//"
        evt_obj.name = "capt0000.cr3"
        mock_cam.wait_for_event.side_effect = [
            (fake_gp.GP_EVENT_FILE_ADDED, evt_obj),
            (fake_gp.GP_EVENT_TIMEOUT, None)
        ]

        def save_mock(dest_path):
            with open(dest_path, "wb") as fp:
                fp.write(b"RAW_CAMERA_BYTES")

        mock_cam_file = MagicMock()
        mock_cam_file.save.side_effect = save_mock
        mock_cam.file_get.return_value = mock_cam_file

        mock_session = MagicMock()
        mock_session.dirs = {"negatives": self.test_dir}
        mock_session.mode = "triplet"
        mock_session.get_next_frame_number.return_value = 1
        backend.session_manager = mock_session

        with patch.object(gp_mod, "GPHOTO2_AVAILABLE", True), patch.object(gp_mod, "gp", fake_gp):
            with backend.lock:
                backend._drain_events_locked(max_events=3, timeout_ms=10)
            self.assertEqual(backend._pending_file_events, [("//", "capt0000.cr3")])

            local_path = backend._download_camera_file("//", "capt0000.cr3")
            try:
                self.assertTrue(os.path.isfile(local_path))
                # Staged copy on local SSD must also exist for fast downstream compositor reading
                staged_path = resolve_fast_read_path(local_path)
                self.assertNotEqual(staged_path, local_path)
                self.assertTrue(os.path.isfile(staged_path))
            finally:
                cleanup_staged_raw(local_path)

if __name__ == '__main__':
    unittest.main()

