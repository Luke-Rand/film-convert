import os
import unittest
import sys
import json

# Add src/ folder to python path so we can import modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from web_ui import app, camera_manager

class TestCameraAutofocus(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        
        # Ensure camera is in simulated mode for testing
        camera_manager.simulated = True
        if not camera_manager.worker_thread:
            camera_manager.start()

    def tearDown(self):
        camera_manager.stop()

    def test_capture_endpoint_default(self):
        # Test standard capture (autofocus defaults to True)
        response = self.app.post('/api/camera/capture')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('path', data)

    def test_capture_endpoint_no_autofocus(self):
        # Test capture with autofocus=False
        response = self.app.post(
            '/api/camera/capture',
            data=json.dumps({'autofocus': False}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('path', data)

if __name__ == '__main__':
    unittest.main()
