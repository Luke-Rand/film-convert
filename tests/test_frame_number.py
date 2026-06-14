import os
import shutil
import tempfile
import unittest
import sys

# Add root folder to python path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
