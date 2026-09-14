import os
import sys
import json
import unittest

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from camera_manager import CameraManager
from web_ui import app

class TestCameraConnectionManager(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_initial_state_machine(self):
        cm = CameraManager()
        from camera_manager import GPHOTO2_AVAILABLE
        if GPHOTO2_AVAILABLE:
            self.assertFalse(cm.simulated)
            self.assertEqual(cm.connection_state, "searching")
        else:
            self.assertTrue(cm.simulated)
            self.assertEqual(cm.connection_state, "simulated")

    def test_status_schema(self):
        cm = CameraManager()
        cm.simulated = True
        status = cm.get_status()
        self.assertIn("state", status)
        self.assertIn("model", status)
        self.assertIn("connected", status)
        self.assertIn("simulated", status)
        self.assertIn("settings", status)
        self.assertIn("choices", status)
        self.assertTrue(status["connected"])
        self.assertTrue(status["simulated"])

    def test_status_endpoint(self):
        response = self.app.get('/api/camera/status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("state", data)
        self.assertIn("model", data)
        self.assertIn("connected", data)

    def test_frame_endpoint_no_content_on_empty(self):
        from web_ui import camera_manager
        with camera_manager.frame_lock:
            saved_frame = camera_manager.latest_frame
            camera_manager.latest_frame = None

        try:
            response = self.app.get('/api/camera/frame')
            self.assertEqual(response.status_code, 204)
        finally:
            with camera_manager.frame_lock:
                camera_manager.latest_frame = saved_frame

    def test_frame_event_increment(self):
        cm = CameraManager()
        cm.simulated = True
        self.assertEqual(cm.frame_id, 0)
        
        frame = cm._grab_preview_frame()
        self.assertIsNotNone(frame)
        with cm.frame_lock:
            cm.latest_frame = frame
            cm.frame_id += 1
            cm.frame_event.set()
        
        self.assertEqual(cm.frame_id, 1)
        self.assertTrue(cm.frame_event.is_set())

if __name__ == '__main__':
    unittest.main()
