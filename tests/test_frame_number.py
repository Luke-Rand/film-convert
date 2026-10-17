import os
import shutil
import tempfile
import unittest
import sys

# Add src/ folder to python path so we can import modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from web_ui import SessionManager
from scanning_session import get_next_frame_number as get_next_frame_number_session
from compositor import get_next_frame_number as get_next_frame_number_compositor

class TestFrameNumber(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_web_ui_frame_number(self):
        # Setup directories in SessionManager
        session = SessionManager()
        negatives = os.path.join(self.test_dir, "negatives")
        positives = os.path.join(self.test_dir, "positives")
        processed = os.path.join(self.test_dir, "processed_raws")
        
        os.makedirs(negatives)
        os.makedirs(positives)
        os.makedirs(processed)
        
        session.dirs = {
            "negatives": negatives,
            "positives": positives,
            "processed": processed
        }
        
        # Initially, frame number should be 1
        self.assertEqual(session.get_next_frame_number(negatives), 1)
        
        # If composite is in negatives
        open(os.path.join(negatives, "Frame_01_Composite.tiff"), 'w').close()
        self.assertEqual(session.get_next_frame_number(negatives), 2)
        
        # Move composite to processed, positive in positives
        os.remove(os.path.join(negatives, "Frame_01_Composite.tiff"))
        open(os.path.join(processed, "Frame_01_Composite.tiff"), 'w').close()
        open(os.path.join(positives, "Frame_01_Positive.tiff"), 'w').close()
        # Should still be 2
        self.assertEqual(session.get_next_frame_number(negatives), 2)
        
        # Add another frame in positives
        open(os.path.join(positives, "Frame_05_Positive.tiff"), 'w').close()
        self.assertEqual(session.get_next_frame_number(negatives), 6)

        # Verify that raw _Capture files in negatives/processed do NOT inflate next frame number to 4!
        open(os.path.join(negatives, "Frame_06_Capture_01_Red_1001.cr3"), 'w').close()
        open(os.path.join(negatives, "Frame_06_Capture_02_Green_1002.cr3"), 'w').close()
        open(os.path.join(negatives, "Frame_06_Capture_03_Blue_1003.cr3"), 'w').close()
        # Even if legacy capture files were named Frame_07_Capture, Frame_08_Capture:
        open(os.path.join(negatives, "Frame_07_Capture_Green_1002.cr3"), 'w').close()
        open(os.path.join(negatives, "Frame_08_Capture_Blue_1003.cr3"), 'w').close()
        self.assertEqual(session.get_next_frame_number(negatives), 6)

    def test_first_triplet_is_frame_01_not_frame_04(self):
        session = SessionManager()
        negatives = os.path.join(self.test_dir, "negatives")
        positives = os.path.join(self.test_dir, "positives")
        processed = os.path.join(self.test_dir, "processed_raws")
        os.makedirs(negatives)
        os.makedirs(positives)
        os.makedirs(processed)
        session.dirs = {
            "negatives": negatives,
            "positives": positives,
            "processed": processed
        }

        # Simulate 3 raw captures for the first triplet in negatives/
        open(os.path.join(negatives, "Frame_01_Capture_01_Red_100.cr3"), 'w').close()
        open(os.path.join(negatives, "Frame_01_Capture_02_Green_101.cr3"), 'w').close()
        open(os.path.join(negatives, "Frame_01_Capture_03_Blue_102.cr3"), 'w').close()

        # First composite/positive must be Frame_01, not Frame_04
        self.assertEqual(session.get_next_frame_number(negatives), 1)
        self.assertEqual(get_next_frame_number_session(session.dirs), 1)
        self.assertEqual(get_next_frame_number_compositor(negatives), 1)

    def test_camera_manager_capture_naming(self):
        from camera_manager import CameraManager
        session = SessionManager()
        negatives = os.path.join(self.test_dir, "negatives")
        positives = os.path.join(self.test_dir, "positives")
        processed = os.path.join(self.test_dir, "processed_raws")
        os.makedirs(negatives)
        os.makedirs(positives)
        os.makedirs(processed)
        session.dirs = {
            "negatives": negatives,
            "positives": positives,
            "processed": processed
        }
        session.mode = "triplet"
        cm = CameraManager(session_manager=session)

        # Shot 1 of triplet 1 -> (1, 1, True)
        self.assertEqual(cm._get_next_capture_naming(negatives), (1, 1, True))
        open(os.path.join(negatives, "Frame_01_Capture_01_Red_100.cr3"), 'w').close()

        # Shot 2 of triplet 1 -> (1, 2, True)
        self.assertEqual(cm._get_next_capture_naming(negatives), (1, 2, True))
        open(os.path.join(negatives, "Frame_01_Capture_02_Green_101.cr3"), 'w').close()

        # Shot 3 of triplet 1 -> (1, 3, True)
        self.assertEqual(cm._get_next_capture_naming(negatives), (1, 3, True))
        open(os.path.join(negatives, "Frame_01_Capture_03_Blue_102.cr3"), 'w').close()

        # If a second triplet is captured before the first finishes processing -> (2, 1, True)
        self.assertEqual(cm._get_next_capture_naming(negatives), (2, 1, True))

    def test_inverter_format_positive_filename(self):
        from inverter import _format_positive_filename
        self.assertEqual(_format_positive_filename("Frame_01_Composite", ".tiff"), "Frame_01_Positive.tiff")
        self.assertEqual(_format_positive_filename("Frame_01_Capture_1729182001", ".tiff"), "Frame_01_Positive.tiff")
        self.assertEqual(_format_positive_filename("Frame_01_Capture_01_Red_1729182001", ".dng"), "Frame_01_Positive.dng")
        self.assertEqual(_format_positive_filename("test_RGB", ".tiff"), "Positive_test_RGB.tiff")

    def test_scanning_session_frame_number(self):
        negatives = os.path.join(self.test_dir, "negatives")
        positives = os.path.join(self.test_dir, "positives")
        processed = os.path.join(self.test_dir, "processed_raws")
        
        os.makedirs(negatives)
        os.makedirs(positives)
        os.makedirs(processed)
        
        dirs = {
            "negatives": negatives,
            "positives": positives,
            "processed": processed
        }
        
        self.assertEqual(get_next_frame_number_session(dirs), 1)
        
        open(os.path.join(positives, "Frame_02_Positive.tiff"), 'w').close()
        self.assertEqual(get_next_frame_number_session(dirs), 3)

    def test_compositor_frame_number(self):
        self.assertEqual(get_next_frame_number_compositor(self.test_dir), 1)
        
        open(os.path.join(self.test_dir, "Frame_03_Composite.tiff"), 'w').close()
        self.assertEqual(get_next_frame_number_compositor(self.test_dir), 4)

if __name__ == '__main__':
    unittest.main()
