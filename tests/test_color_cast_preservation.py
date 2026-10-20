import os
import shutil
import tempfile
import unittest
import numpy as np
import tifffile
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from inverter import process_positives
from tiff_writer import write_16bit_tiff

class TestColorCastPreservation(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_global_levels_preserves_chromaticity(self):
        """
        Verify that a non-neutral scene (e.g. green foliage, where Green exposure is high
        and Red is low in positive space) preserves its channel ratios when global_levels=True,
        rather than auto-stretching Red to match Green.
        """
        h, w = 100, 100
        # In a negative of a vibrant green scene:
        # Green dye was heavily exposed (dense negative = low transmission I_G = 3000 to 15000)
        # Red and Blue dyes were lightly exposed (clear negative = high transmission I_R = 45000, I_B = 42000)
        # Clear film base is 55000 across all channels.
        neg_data = np.full((h, w, 3), 55000, dtype=np.uint16)
        
        # Inner scene content
        y, x = np.mgrid[20:80, 20:80]
        # Red channel remains close to base (shadow in positive): 48000
        neg_data[20:80, 20:80, 0] = 48000
        # Green channel has high exposure density (bright green highlight in positive): 4000 to 12000
        neg_data[20:80, 20:80, 1] = (4000 + (x - 20) * 100).astype(np.uint16)
        # Blue channel remains close to base: 45000
        neg_data[20:80, 20:80, 2] = 45000

        input_path = os.path.join(self.test_dir, "test_green_scene.tiff")
        write_16bit_tiff(input_path, neg_data, is_monochrome=False)

        pos_dir = os.path.join(self.test_dir, "Positives")
        process_positives(input_path, output_dir=pos_dir, clip=0.0, gamma=1.0, global_levels=True, ignore_margin=0.0)

        out_path = os.path.join(pos_dir, "Positive_test_green_scene.tiff")
        self.assertTrue(os.path.exists(out_path))

        out_img = tifffile.imread(out_path)

        # In positive space, Green should be significantly brighter than Red and Blue
        mean_r = np.mean(out_img[25:75, 25:75, 0])
        mean_g = np.mean(out_img[25:75, 25:75, 1])
        mean_b = np.mean(out_img[25:75, 25:75, 2])

        self.assertGreater(mean_g, mean_r * 2.0, f"Green ({mean_g}) should dominate Red ({mean_r})")
        self.assertGreater(mean_g, mean_b * 2.0, f"Green ({mean_g}) should dominate Blue ({mean_b})")

    def test_default_is_global_levels_not_per_channel(self):
        """
        Regression guard: the default (global_levels omitted) must behave as global_levels=True
        (joint exposure scaling), NOT per-channel. Per-channel levels over-amplify the narrow
        red/blue density channels of a color negative and produce a magenta cast.
        """
        h, w = 100, 100
        # A negative with a modest red/blue channel imbalance (narrower density range in R/B).
        neg_data = np.full((h, w, 3), 55000, dtype=np.uint16)
        y, x = np.mgrid[20:80, 20:80]
        neg_data[20:80, 20:80, 0] = 48000
        neg_data[20:80, 20:80, 1] = (4000 + (x - 20) * 100).astype(np.uint16)
        neg_data[20:80, 20:80, 2] = 45000

        input_path = os.path.join(self.test_dir, "test_imbalance.tiff")
        write_16bit_tiff(input_path, neg_data, is_monochrome=False)

        def run(label, gl_kwarg):
            d = os.path.join(self.test_dir, label)
            os.makedirs(d, exist_ok=True)
            process_positives(input_path, output_dir=d, clip=0.0, gamma=1.0,
                              ignore_margin=0.0, **gl_kwarg)
            out = next(f for f in os.listdir(d) if f.lower().endswith(".tiff"))
            img = tifffile.imread(os.path.join(d, out))
            return tuple(np.mean(img[25:75, 25:75, c]) for c in range(3))

        default_means = run("default", {})
        explicit_global = run("explicit_global", {"global_levels": True})
        legacy_per_channel = run("legacy_per_channel", {"global_levels": False})

        # Default must equal explicit global levels.
        for c, (a, b) in enumerate(zip(default_means, explicit_global)):
            self.assertAlmostEqual(a, b, places=4,
                msg=f"channel {c}: default {a} != global_levels=True {b}")

        # And must NOT equal per-channel (legacy) behavior.
        self.assertNotEqual(default_means, legacy_per_channel,
            "default unexpectedly matches per-channel (legacy magenta) behavior")

if __name__ == '__main__':
    unittest.main()
