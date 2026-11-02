import os
import shutil
import tempfile
import unittest
from pathlib import Path
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, 'src'))

import reprocess_rolls


class TestReprocessRolls(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_is_roll_dir_ignores_digital_and_session_folders(self):
        for name in ['digital', 'Digital', 'c1', 'capture1', 'trash']:
            p = os.path.join(self.test_dir, name)
            os.makedirs(p, exist_ok=True)
            # Create a RAW file inside
            Path(os.path.join(p, "IMG_0001.CR3")).touch()
            self.assertFalse(reprocess_rolls.is_roll_dir(p))

    def test_find_raw_source_dir_requires_frame_or_manifest_for_root(self):
        roll_dir = os.path.join(self.test_dir, "RandomCameraFolder")
        os.makedirs(roll_dir, exist_ok=True)
        raw_path = os.path.join(roll_dir, "9H0A0001.CR3")
        Path(raw_path).touch()
        # Random RAWs in root without 'frame' or session.json should NOT be treated as a roll
        self.assertIsNone(reprocess_rolls.find_raw_source_dir(roll_dir))

        # Adding 'frame' in name makes it detected
        frame_raw = os.path.join(roll_dir, "Frame_01_Capture.CR3")
        Path(frame_raw).touch()
        self.assertEqual(reprocess_rolls.find_raw_source_dir(roll_dir), roll_dir)

    def test_find_raw_source_dir_detects_processed_raws(self):
        roll_dir = os.path.join(self.test_dir, "KodakGold200-135-01")
        processed_dir = os.path.join(roll_dir, "processed_raws")
        os.makedirs(processed_dir, exist_ok=True)
        Path(os.path.join(processed_dir, "Frame_01_Capture.CR3")).touch()
        self.assertEqual(reprocess_rolls.find_raw_source_dir(roll_dir), processed_dir)

    def test_discover_rolls_multiple_directories(self):
        r1 = os.path.join(self.test_dir, "Roll1", "processed_raws")
        r2 = os.path.join(self.test_dir, "Roll2", "processed_raws")
        os.makedirs(r1, exist_ok=True)
        os.makedirs(r2, exist_ok=True)
        Path(os.path.join(r1, "Frame_01.cr3")).touch()
        Path(os.path.join(r2, "Frame_01.cr3")).touch()

        discovered = reprocess_rolls.discover_rolls(self.test_dir)
        self.assertEqual(len(discovered), 2)

    def test_web_ui_discovers_last_frame_in_negatives(self):
        import web_ui
        mgr = web_ui.SessionManager()
        negatives = os.path.join(self.test_dir, "negatives")
        positives = os.path.join(self.test_dir, "positives")
        processed = os.path.join(self.test_dir, "processed_raws")
        os.makedirs(negatives, exist_ok=True)
        os.makedirs(positives, exist_ok=True)
        os.makedirs(processed, exist_ok=True)

        mgr.dirs = {
            "negatives": negatives,
            "positives": positives,
            "processed": processed
        }

        # Place a positive in positives and a composite in negatives
        Path(os.path.join(positives, "Frame_01_Positive.tiff")).touch()
        Path(os.path.join(negatives, "Frame_01_Composite.dng")).touch()

        last = mgr._discover_last_frame()
        self.assertIsNotNone(last)
        self.assertEqual(last["frame_number"], 1)
        self.assertEqual(last["negative_name"], "Frame_01_Composite.dng")

