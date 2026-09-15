import os
import shutil
import tempfile
import unittest
import subprocess
import numpy as np
import tifffile
import sys

# Add src/ folder to python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from icc_manager import get_icc_profile, normalize_profile_name
from metadata_preservation import find_exiftool, extract_camera_info, get_dng_extratags, transfer_raw_metadata
from tiff_writer import write_16bit_image, write_16bit_tiff
from compositor import process_triplet
from inverter import process_positives


class TestMetadataAndICC(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sample_raw = "/Users/lukerand/Pictures/Scans/KodakGold200-135-01/processed_raws/Frame_02_Capture_1789482682.cr3"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_icc_manager_loads_bundled_profiles(self):
        """Verify that ICC profiles can be loaded for all supported color spaces."""
        adobe = get_icc_profile("adobe_rgb")
        self.assertIsNotNone(adobe)
        self.assertGreater(len(adobe), 500)

        prophoto = get_icc_profile("prophoto_rgb")
        self.assertIsNotNone(prophoto)
        self.assertGreater(len(prophoto), 500)

        srgb = get_icc_profile("srgb")
        self.assertIsNotNone(srgb)
        self.assertGreater(len(srgb), 1000)

        gray = get_icc_profile("gray", is_monochrome=True)
        self.assertIsNotNone(gray)
        self.assertGreater(len(gray), 1000)

        none_prof = get_icc_profile("none")
        self.assertIsNone(none_prof)

    def test_write_16bit_tiff_with_adobe_rgb(self):
        """Verify that writing a 16-bit TIFF embeds the Adobe RGB (1998) ICC profile tag."""
        data = np.zeros((80, 80, 3), dtype=np.uint16)
        out_tiff = os.path.join(self.test_dir, "test_adobe.tiff")
        write_16bit_image(out_tiff, data, icc_profile="adobe_rgb")

        self.assertTrue(os.path.exists(out_tiff))

        # Check with tifffile
        with tifffile.TiffFile(out_tiff) as tif:
            page = tif.pages[0]
            self.assertEqual(page.photometric, 2)
            # Tag 34675 is InterColorProfile
            icc_tag = page.tags.get(34675)
            self.assertIsNotNone(icc_tag)
            self.assertEqual(len(icc_tag.value), len(get_icc_profile("adobe_rgb")))

        # Check with exiftool if available
        exiftool = find_exiftool()
        if exiftool:
            res = subprocess.run([exiftool, "-s3", "-ProfileDescription", out_tiff], capture_output=True, text=True)
            self.assertIn("Adobe RGB (1998)", res.stdout)

    def test_write_16bit_tiff_with_prophoto_rgb(self):
        """Verify that writing a 16-bit TIFF embeds the ProPhoto RGB (ROMM) profile."""
        data = np.zeros((80, 80, 3), dtype=np.uint16)
        out_tiff = os.path.join(self.test_dir, "test_prophoto.tiff")
        write_16bit_image(out_tiff, data, icc_profile="prophoto_rgb")

        self.assertTrue(os.path.exists(out_tiff))
        with tifffile.TiffFile(out_tiff) as tif:
            page = tif.pages[0]
            icc_tag = page.tags.get(34675)
            self.assertIsNotNone(icc_tag)

        exiftool = find_exiftool()
        if exiftool:
            res = subprocess.run([exiftool, "-s3", "-ProfileDescription", out_tiff], capture_output=True, text=True)
            self.assertTrue("ROMM" in res.stdout or "ProPhoto" in res.stdout)

    def test_write_16bit_monochrome_gray_profile(self):
        """Verify that monochrome images receive the Gray Gamma 2.2 profile."""
        data = np.zeros((80, 80), dtype=np.uint16)
        out_tiff = os.path.join(self.test_dir, "test_mono.tiff")
        write_16bit_image(out_tiff, data, is_monochrome=True)

        self.assertTrue(os.path.exists(out_tiff))
        with tifffile.TiffFile(out_tiff) as tif:
            page = tif.pages[0]
            self.assertEqual(page.photometric, 1)
            icc_tag = page.tags.get(34675)
            self.assertIsNotNone(icc_tag)

    def test_true_dng_matrix_and_tags(self):
        """Verify that writing a .dng file injects full DNG specification tags and matrix."""
        data = np.random.randint(1000, 50000, (60, 60, 3), dtype=np.uint16)
        out_dng = os.path.join(self.test_dir, "test_true.dng")
        write_16bit_image(out_dng, data, icc_profile="adobe_rgb")

        self.assertTrue(os.path.exists(out_dng))

        with tifffile.TiffFile(out_dng) as tif:
            page = tif.pages[0]
            # DNG LinearRaw Photometric
            self.assertEqual(page.photometric, 34892)
            # DNGVersion tag 50706
            self.assertIn(page.tags[50706].value, [(1, 4, 0, 0), b'\x01\x04\x00\x00', [1, 4, 0, 0]])
            # DNGBackwardVersion tag 50707
            self.assertIn(page.tags[50707].value, [(1, 3, 0, 0), b'\x01\x03\x00\x00', [1, 3, 0, 0]])
            # ColorMatrix1 tag 50721
            self.assertIn(50721, page.tags)
            # CalibrationIlluminant1 tag 50778 (D65 = 21)
            self.assertEqual(page.tags[50778].value, 21)
            # UniqueCameraModel tag 50708
            self.assertIn(50708, page.tags)

        exiftool = find_exiftool()
        if exiftool:
            res = subprocess.run([exiftool, "-s3", "-DNGVersion", out_dng], capture_output=True, text=True)
            self.assertEqual(res.stdout.strip(), "1.4.0.0")

    def test_raw_metadata_preservation(self):
        """Verify that EXIF/IPTC camera metadata is preserved from camera RAW into final DNG and TIFF."""
        if not os.path.exists(self.sample_raw):
            self.skipTest(f"Sample RAW file not present: {self.sample_raw}")

        data = np.zeros((50, 50, 3), dtype=np.uint16)
        out_dng = os.path.join(self.test_dir, "composite_frame.dng")

        write_16bit_image(
            out_dng,
            data,
            icc_profile="adobe_rgb",
            source_metadata_path=self.sample_raw
        )

        self.assertTrue(os.path.exists(out_dng))

        exiftool = find_exiftool()
        if exiftool:
            res = subprocess.run(
                [exiftool, "-s", "-Make", "-Model", "-LensModel", "-SerialNumber", "-ExposureTime", "-ISO", out_dng],
                capture_output=True,
                text=True
            )
            output = res.stdout
            self.assertIn("Canon", output)
            self.assertIn("EOS R6 Mark III", output)
            self.assertIn("EF100mm", output)
            self.assertIn("159202003410", output)
            self.assertIn("1/40", output)

    def test_inverter_preserves_icc_and_metadata(self):
        """Verify that process_positives propagates the ICC profile and metadata."""
        # Create input 16-bit negative TIFF
        h, w = 60, 60
        neg_data = np.random.randint(10000, 40000, size=(h, w, 3), dtype=np.uint16)
        input_tiff = os.path.join(self.test_dir, "Frame_01_Composite.tiff")

        source_meta = self.sample_raw if os.path.exists(self.sample_raw) else None
        write_16bit_image(input_tiff, neg_data, icc_profile="prophoto_rgb", source_metadata_path=source_meta)

        pos_dir = os.path.join(self.test_dir, "Positives")
        process_positives(
            input_path=input_tiff,
            output_dir=pos_dir,
            clip=0.0,
            gamma=2.2,
            icc_profile="prophoto_rgb",
            preserve_metadata=True
        )

        expected_pos = os.path.join(pos_dir, "Frame_01_Positive.tiff")
        self.assertTrue(os.path.exists(expected_pos))

        with tifffile.TiffFile(expected_pos) as tif:
            page = tif.pages[0]
            self.assertIn(34675, page.tags)

        if source_meta and find_exiftool():
            res = subprocess.run(
                [find_exiftool(), "-s3", "-Model", expected_pos],
                capture_output=True,
                text=True
            )
            self.assertIn("EOS R6 Mark III", res.stdout)


if __name__ == '__main__':
    unittest.main()
