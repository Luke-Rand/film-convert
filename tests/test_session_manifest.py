import os
import sys
import shutil
import tempfile
import unittest
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from session_manifest import (
    read_session_manifest,
    write_session_manifest,
    parse_roll_identifiers,
    count_session_frames,
    get_manifest_path
)
from web_ui import app, session as global_session
from reprocess_rolls import _resolve_effective_config


class TestSessionManifest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        if global_session.status == "monitoring":
            global_session.stop_monitoring()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_write_and_read_manifest(self):
        roll_dir = os.path.join(self.test_dir, "KodakPortra400-135-02")
        os.makedirs(os.path.join(roll_dir, "positives"), exist_ok=True)
        open(os.path.join(roll_dir, "positives", "Frame_01_Positive.tiff"), "w").close()
        open(os.path.join(roll_dir, "positives", "Frame_02_Positive.tiff"), "w").close()

        written = write_session_manifest(
            session_dir=roll_dir,
            session_name="KodakPortra400-135-02",
            mode="triplet",
            config={"gamma": 2.4, "base_ratios": [1.0, 0.58, 0.23], "neutralize": True},
            sampled_rgb=[240, 145, 70],
            last_triplet_means={"r_mean": 0.45, "g_mean": 0.42, "b_mean": 0.40}
        )

        self.assertIsNotNone(written)
        self.assertTrue(os.path.isfile(get_manifest_path(roll_dir)))
        self.assertEqual(written["stock"], "KodakPortra400")
        self.assertEqual(written["format"], "135")
        self.assertEqual(written["roll"], "02")
        self.assertEqual(written["frame_count"], 2)
        self.assertEqual(written["config"]["gamma"], 2.4)
        self.assertEqual(written["config"]["base_ratios"], [1.0, 0.58, 0.23])
        self.assertEqual(written["sampled_rgb"], [240, 145, 70])

        loaded = read_session_manifest(roll_dir)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["session_name"], "KodakPortra400-135-02")
        self.assertEqual(loaded["created_at"], written["created_at"])

        # Updating preserves created_at and updates config
        updated = write_session_manifest(
            session_dir=roll_dir,
            config={"gamma": 2.6}
        )
        self.assertEqual(updated["created_at"], written["created_at"])
        self.assertEqual(updated["config"]["gamma"], 2.6)
        self.assertEqual(updated["config"]["base_ratios"], [1.0, 0.58, 0.23])

    def test_web_ui_session_load_and_manifest_endpoints(self):
        client = app.test_client()
        roll_dir = os.path.join(self.test_dir, "IlfordHP5-120-05")
        os.makedirs(roll_dir, exist_ok=True)

        write_session_manifest(
            session_dir=roll_dir,
            session_name="IlfordHP5-120-05",
            stock="IlfordHP5",
            film_format="120",
            roll="05",
            mode="single",
            config={"monochrome": True, "gamma": 2.1, "autocrop": True}
        )

        # GET /api/session/manifest
        res = client.get(f"/api/session/manifest?path={roll_dir}")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["manifest"]["stock"], "IlfordHP5")

        # POST /api/session/load
        res_load = client.post("/api/session/load", json={"session_dir": roll_dir})
        self.assertEqual(res_load.status_code, 200)
        load_data = res_load.get_json()
        self.assertTrue(load_data["success"])
        self.assertEqual(load_data["manifest"]["root_dir"], os.path.abspath(self.test_dir))
        self.assertEqual(load_data["status_data"]["status"], "idle")
        self.assertEqual(global_session.session_name, "IlfordHP5-120-05")
        self.assertEqual(global_session.stock, "IlfordHP5")
        self.assertEqual(global_session.format, "120")
        self.assertEqual(global_session.roll, "05")
        self.assertEqual(global_session.mode, "single")
        self.assertTrue(global_session.config["monochrome"])
        self.assertEqual(global_session.config["gamma"], 2.1)

        # POST /api/start and /api/stop return status_data
        res_start = client.post("/api/start", json={
            "root_dir": self.test_dir,
            "stock": "IlfordHP5",
            "format": "120",
            "roll": "05",
            "mode": "single"
        })
        self.assertEqual(res_start.status_code, 200)
        start_data = res_start.get_json()
        self.assertTrue(start_data["success"])
        self.assertEqual(start_data["status_data"]["status"], "monitoring")

        res_stop = client.post("/api/stop")
        self.assertEqual(res_stop.status_code, 200)
        stop_data = res_stop.get_json()
        self.assertTrue(stop_data["success"])
        self.assertEqual(stop_data["status_data"]["status"], "idle")

        # GET /api/browse expands ~
        res_browse = client.get("/api/browse?path=~")
        self.assertEqual(res_browse.status_code, 200)
        browse_data = res_browse.get_json()
        self.assertEqual(browse_data["current"], str(os.path.realpath(os.path.expanduser("~"))))

    def test_reprocess_rolls_uses_manifest_unless_overridden(self):
        roll_dir = os.path.join(self.test_dir, "Cinestill800T-135-01")
        os.makedirs(roll_dir, exist_ok=True)
        write_session_manifest(
            session_dir=roll_dir,
            session_name="Cinestill800T-135-01",
            config={
                "gamma": 2.5,
                "clip": 0.25,
                "base_ratios": [1.0, 0.61, 0.29],
                "neutralize": True,
                "autocrop": True
            }
        )

        # Default CLI args (nothing in _explicit)
        args = argparse.Namespace(
            gamma=2.2,
            clip=0.1,
            margin=0.03,
            scurve=0.0,
            autocrop=False,
            global_levels=False,
            neutralize=False,
            base_ratios=None,
            compress=False,
            no_align=False,
            monochrome=False,
            monochrome_channel="luminance",
            reversal=False,
            icc_profile="adobe_rgb",
            no_metadata=False,
            contact_sheet=True,
            contact_sheet_columns=6,
            _explicit=set()
        )
        _, eff = _resolve_effective_config(roll_dir, args)
        self.assertEqual(eff["gamma"], 2.5)
        self.assertEqual(eff["clip"], 0.25)
        self.assertEqual(eff["base_ratios"], [1.0, 0.61, 0.29])
        self.assertTrue(eff["neutralize"])
        self.assertTrue(eff["autocrop"])

        # Explicit CLI override for gamma
        args.gamma = 1.8
        args._explicit = {"gamma"}
        _, eff_override = _resolve_effective_config(roll_dir, args)
        self.assertEqual(eff_override["gamma"], 1.8)
        self.assertEqual(eff_override["base_ratios"], [1.0, 0.61, 0.29])


if __name__ == "__main__":
    unittest.main()
