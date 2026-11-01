"""
Composite-only mode: the session stops after compositing and hands its folders to an external
converter (NegPy). These tests cover the switch itself, the manifest accounting, and the worker
that must skip inversion while leaving the composite in place.
"""

import os
import shutil
import sys
import tempfile
import time
import unittest
from argparse import Namespace
from unittest import mock

import numpy as np
import tifffile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, 'src'))

import batch_worker
import reprocess_rolls
import web_ui
from schemas import SessionConfigSchema, SessionConfigUpdateSchema
from session_manifest import (
    count_session_frames,
    read_session_manifest,
    write_session_manifest,
)


class TestCompositeOnlySchema(unittest.TestCase):
    def test_defaults_to_off(self):
        self.assertFalse(SessionConfigSchema().composite_only)

    def test_accepts_explicit_true(self):
        self.assertTrue(SessionConfigSchema(composite_only=True).composite_only)

    def test_partial_update_accepts_the_flag(self):
        self.assertTrue(SessionConfigUpdateSchema(composite_only=True).composite_only)

    def test_unknown_keys_do_not_set_it(self):
        self.assertFalse(SessionConfigSchema.model_validate({"unrelated_key": True}).composite_only)


class TestCompositeOnlyManifest(unittest.TestCase):
    def setUp(self):
        self.session_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.session_dir, ignore_errors=True)

    def test_flag_round_trips_through_session_json(self):
        write_session_manifest(
            session_dir=self.session_dir,
            session_name="Portra400-135-01",
            config={"composite_only": True},
        )
        manifest = read_session_manifest(self.session_dir)
        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertTrue(manifest["config"]["composite_only"])

    def test_flag_defaults_off_in_a_manifest_that_never_set_it(self):
        write_session_manifest(session_dir=self.session_dir, session_name="Portra400-135-02")
        manifest = read_session_manifest(self.session_dir)
        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertFalse(manifest["config"]["composite_only"])

    def test_frame_count_reads_the_composites_folder(self):
        """A composite-only session has no positives; its frame count comes from composites/."""
        comp_dir = os.path.join(self.session_dir, "composites")
        os.makedirs(comp_dir)
        os.makedirs(os.path.join(self.session_dir, "positives"))
        for n in (1, 2, 3):
            tifffile.imwrite(
                os.path.join(comp_dir, f"Frame_{n:02d}_Composite.dng"),
                np.zeros((8, 8, 3), dtype=np.uint16),
            )
        self.assertEqual(count_session_frames(self.session_dir), 3)

    def test_frame_count_ignores_contact_sheets(self):
        comp_dir = os.path.join(self.session_dir, "composites")
        os.makedirs(comp_dir)
        tifffile.imwrite(os.path.join(comp_dir, "Frame_01_Composite.dng"), np.zeros((8, 8, 3), dtype=np.uint16))
        tifffile.imwrite(os.path.join(comp_dir, "roll_contact_sheet.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertEqual(count_session_frames(self.session_dir), 1)


class TestCompositeOnlyWorker(unittest.TestCase):
    """The worker composites either way; only the inversion step is conditional."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.group = []
        for color in ("red", "green", "blue"):
            path = os.path.join(self.tmp, f"Frame_01_Capture_{color}.cr3")
            with open(path, "wb") as f:
                f.write(b"stub")
            self.group.append(path)
        self.composite_path = os.path.join(self.tmp, "Frame_01_Composite.dng")
        self.positives_dir = os.path.join(self.tmp, "positives")
        os.makedirs(self.positives_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fake_composite(self, group, output_filepath, **kwargs):
        tifffile.imwrite(output_filepath, np.full((8, 8, 3), 1000, dtype=np.uint16))
        return (1.0, 2.0, 3.0), np.full((8, 8, 3), 1000, dtype=np.uint16)

    def test_composite_only_skips_inversion_and_keeps_the_negative(self):
        with mock.patch.object(batch_worker, "process_triplet", side_effect=self._fake_composite), \
             mock.patch.object(batch_worker, "process_positives") as invert:
            success, means, logs = batch_worker.worker_process_triplet_pipeline(
                group=self.group,
                composite_filepath=self.composite_path,
                positives_dir=self.positives_dir,
                config={"composite_only": True},
            )

        self.assertTrue(success, logs)
        self.assertEqual(means, (1.0, 2.0, 3.0))
        invert.assert_not_called()
        self.assertTrue(os.path.isfile(self.composite_path), "the handed-off negative must survive")
        self.assertEqual(os.listdir(self.positives_dir), [])

    def test_default_mode_still_inverts(self):
        with mock.patch.object(batch_worker, "process_triplet", side_effect=self._fake_composite), \
             mock.patch.object(batch_worker, "process_positives") as invert:
            success, _, logs = batch_worker.worker_process_triplet_pipeline(
                group=self.group,
                composite_filepath=self.composite_path,
                positives_dir=self.positives_dir,
                config={},
            )

        self.assertTrue(success, logs)
        invert.assert_called_once()


class TestCompositeOnlyMonitor(unittest.TestCase):
    """Drives the real SessionManager loop: the deliverable must land in composites/ and stay there."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.session_name = "Portra400-135-09"
        self.session = web_ui.SessionManager()

    def tearDown(self):
        try:
            self.session.stop_monitoring()
        except Exception:
            pass
        shutil.rmtree(self.root, ignore_errors=True)

    def test_single_shot_negative_is_handed_off_not_inverted(self):
        ok, msg = self.session.start_monitoring(
            self.root, self.session_name, "single", config={"composite_only": True}
        )
        self.assertTrue(ok, msg)

        session_dir = os.path.join(self.root, self.session_name)
        neg_dir = os.path.join(session_dir, "negatives")
        source = os.path.join(neg_dir, "Scan_0001.tif")
        tifffile.imwrite(source, np.full((32, 32, 3), 40000, dtype=np.uint16))

        handed_off = os.path.join(session_dir, "composites", "Scan_0001.tif")
        deadline = time.time() + 20
        while time.time() < deadline and not os.path.isfile(handed_off):
            time.sleep(0.2)

        self.assertTrue(os.path.isfile(handed_off), "the negative must land in composites/")
        self.assertFalse(os.path.isfile(source), "the negative must leave negatives/")
        self.assertEqual(os.listdir(os.path.join(session_dir, "positives")), [],
                         "composite-only must not produce positives")
        self.assertEqual(os.listdir(os.path.join(session_dir, "processed_raws")), [],
                         "a handed-off negative must not be filed as a processed RAW")

    def test_default_mode_still_inverts_through_the_monitor(self):
        ok, msg = self.session.start_monitoring(self.root, self.session_name, "single")
        self.assertTrue(ok, msg)

        session_dir = os.path.join(self.root, self.session_name)
        source = os.path.join(session_dir, "negatives", "Scan_0001.tif")
        tifffile.imwrite(source, np.full((32, 32, 3), 40000, dtype=np.uint16))

        positives_dir = os.path.join(session_dir, "positives")
        deadline = time.time() + 25
        produced = []
        while time.time() < deadline:
            produced = [f for f in os.listdir(positives_dir) if not f.startswith(".")]
            if produced:
                break
            time.sleep(0.2)

        self.assertEqual(len(produced), 1, "the default path must still invert into positives/")
        self.assertEqual(len(os.listdir(os.path.join(session_dir, "composites"))), 0)


class TestFilesEndpoint(unittest.TestCase):
    """The gallery reads /api/files, so it has to list composites when that is the deliverable."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.saved_dirs = dict(web_ui.session.dirs)
        self.saved_config = dict(web_ui.session.config)

    def tearDown(self):
        web_ui.session.dirs = self.saved_dirs
        web_ui.session.config = self.saved_config
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _session_dirs(self):
        dirs = {}
        for key in ("negatives", "positives", "processed", "errors", "composites"):
            path = os.path.join(self.tmp, key)
            os.makedirs(path, exist_ok=True)
            dirs[key] = path
        return dirs

    def test_lists_composites_when_composite_only(self):
        web_ui.session.dirs = self._session_dirs()
        web_ui.session.config["composite_only"] = True
        tifffile.imwrite(
            os.path.join(web_ui.session.dirs["composites"], "Frame_01_Composite.dng"),
            np.zeros((8, 8, 3), dtype=np.uint16),
        )

        data = web_ui.app.test_client().get('/api/files').get_json()

        self.assertTrue(data["composite_only"])
        self.assertEqual([f["name"] for f in data["positives"]], ["Frame_01_Composite.dng"])

    def test_lists_positives_when_not_composite_only(self):
        web_ui.session.dirs = self._session_dirs()
        web_ui.session.config["composite_only"] = False
        tifffile.imwrite(
            os.path.join(web_ui.session.dirs["positives"], "Frame_01_Positive.tiff"),
            np.zeros((8, 8, 3), dtype=np.uint16),
        )
        tifffile.imwrite(
            os.path.join(web_ui.session.dirs["composites"], "Frame_01_Composite.dng"),
            np.zeros((8, 8, 3), dtype=np.uint16),
        )

        data = web_ui.app.test_client().get('/api/files').get_json()

        self.assertFalse(data["composite_only"])
        self.assertEqual([f["name"] for f in data["positives"]], ["Frame_01_Positive.tiff"])


class TestReprocessFlagOverride(unittest.TestCase):
    """--no-composite-only must be able to undo a manifest that recorded composite_only."""

    def setUp(self):
        self.roll_dir = tempfile.mkdtemp()
        write_session_manifest(
            session_dir=self.roll_dir,
            session_name="Portra400-135-05",
            config={"composite_only": True},
        )

    def tearDown(self):
        shutil.rmtree(self.roll_dir, ignore_errors=True)

    def _args(self, explicit):
        # pick() evaluates some defaults eagerly, so the flags accessors must exist.
        return Namespace(
            base_ratios=None,
            neutralize=False,
            no_align=False,
            no_metadata=False,
            no_composite_only=False,
            composite_only=False,
            contact_sheet=True,
            _explicit=set(explicit),
        )

    def test_manifest_value_applies_by_default(self):
        _, eff = reprocess_rolls._resolve_effective_config(self.roll_dir, self._args([]))
        self.assertTrue(eff["composite_only"])

    def test_no_composite_only_overrides_the_manifest(self):
        _, eff = reprocess_rolls._resolve_effective_config(
            self.roll_dir, self._args(["no_composite_only"])
        )
        self.assertFalse(eff["composite_only"])

    def test_explicit_flag_still_wins(self):
        """With the manifest at False, only the explicit flag can turn it on."""
        plain_roll = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, plain_roll, True)
        write_session_manifest(
            session_dir=plain_roll,
            session_name="Portra400-135-06",
            config={"composite_only": False},
        )

        _, eff = reprocess_rolls._resolve_effective_config(
            plain_roll, self._args(["composite_only"])
        )
        self.assertTrue(eff["composite_only"], "explicit --composite-only must override a False manifest")


if __name__ == '__main__':
    unittest.main()
