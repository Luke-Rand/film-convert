import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Any

# Standard D50 to sRGB Color Matrix (SRATIONAL, 9 signed integer pairs)
DEFAULT_D50_SRGB_MATRIX = [
    (31339, 10000), (-16169, 10000), (-4906, 10000),
    (-9788, 10000), (19161, 10000), (335, 10000),
    (719, 10000), (-2290, 10000), (14052, 10000)
]


def find_exiftool() -> Optional[str]:
    """Locates exiftool binary on PATH or standard install locations."""
    tool = shutil.which("exiftool")
    if tool and os.path.exists(tool):
        return tool
        
    common_paths = [
        "/opt/homebrew/bin/exiftool",
        "/usr/local/bin/exiftool",
        "/usr/bin/exiftool",
    ]
    for p in common_paths:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
            
    return None


def extract_camera_info(source_path: str) -> Tuple[Optional[List[Tuple[int, int]]], Optional[str]]:
    """
    Extracts camera matrix and model from source RAW using rawpy or exiftool.
    Returns (color_matrix_srational_pairs, camera_model_name).
    """
    if not source_path or not os.path.exists(source_path):
        return None, None

    camera_model = None
    color_matrix = None

    # Try rawpy first
    try:
        import rawpy
        import numpy as np
        with rawpy.imread(source_path) as raw:
            # rgb_xyz_matrix is 3x3 (cam to xyz)
            cam_xyz = getattr(raw, "rgb_xyz_matrix", None)
            if cam_xyz is not None and len(cam_xyz) >= 3:
                cam_3x3 = np.array(cam_xyz[:3, :3], dtype=np.float64)
                if np.linalg.matrix_rank(cam_3x3) == 3:
                    xyz_cam = np.linalg.inv(cam_3x3)
                    pairs = []
                    for row in xyz_cam:
                        for val in row:
                            pairs.append((int(round(val * 10000)), 10000))
                    color_matrix = pairs
    except Exception:
        pass

    # Try exiftool for camera model
    exiftool = find_exiftool()
    if exiftool:
        try:
            res = subprocess.run(
                [exiftool, "-s3", "-Model", source_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            model_str = res.stdout.strip()
            if model_str:
                camera_model = model_str
        except Exception:
            pass

    return color_matrix, camera_model


def get_dng_extratags(
    source_path: Optional[str] = None,
    color_matrix_pairs: Optional[List[Tuple[int, int]]] = None,
    camera_model: Optional[str] = None,
    icc_profile_bytes: Optional[bytes] = None,
    is_monochrome: bool = False
) -> List[Tuple[Any, ...]]:
    """
    Builds the standard DNG specification extratags for tifffile.imwrite.
    """
    if source_path and (not color_matrix_pairs or not camera_model):
        extracted_matrix, extracted_model = extract_camera_info(source_path)
        if not color_matrix_pairs and extracted_matrix:
            color_matrix_pairs = extracted_matrix
        if not camera_model and extracted_model:
            camera_model = extracted_model

    if not color_matrix_pairs:
        color_matrix_pairs = DEFAULT_D50_SRGB_MATRIX

    matrix_flat = []
    for num, den in color_matrix_pairs:
        matrix_flat.extend([num, den])

    model_name = camera_model or "FilmConvert Linear DNG"

    dng_version = b'\x01\x04\x00\x00'
    dng_backward = b'\x01\x03\x00\x00'

    extratags = [
        (254, 'I', 1, 0, True),                                  # NewSubfileType = 0 (Full-resolution)
        (50706, 'B', 4, dng_version, True),                      # DNGVersion = 1.4.0.0
        (50707, 'B', 4, dng_backward, True),                     # DNGBackwardVersion = 1.3.0.0
        (50708, 's', len(model_name) + 1, model_name, True),     # UniqueCameraModel
        (50714, '2I', 1, (0, 1), True),                          # BlackLevel = 0
        (50717, 'I', 1, 65535, True),                            # WhiteLevel = 65535
        (50730, '2i', 1, (0, 100), True),                        # BaselineExposure = 0.0 EV
        (50734, '2I', 1, (1, 1), True),                          # LinearResponseLimit = 1.0
        (50736, 'I', 1, 1, True),                                # BaselineInterpretation = 1
    ]

    if not is_monochrome:
        extratags.extend([
            (50721, '2i', 9, matrix_flat, True),                 # ColorMatrix1
            (50728, '2I', 3, [1, 1, 1, 1, 1, 1], True),          # AsShotNeutral (1, 1, 1)
            (50778, 'H', 1, 21, True),                           # CalibrationIlluminant1 (D65)
        ])

    if icc_profile_bytes:
        extratags.append((50831, 'B', len(icc_profile_bytes), icc_profile_bytes, True)) # AsShotICCProfile

    return extratags


def transfer_raw_metadata(source_path: str, target_path: str) -> bool:
    """
    Preserves EXIF/IPTC/XMP camera metadata (exposure, lens model, camera serial,
    capture timestamp, etc.) from the source RAW file into the final target TIFF/DNG.
    Safely excludes image raster geometry tags and the embedded ICC profile.
    """
    if not source_path or not os.path.exists(source_path):
        return False
    if not target_path or not os.path.exists(target_path):
        return False

    exiftool = find_exiftool()
    if exiftool:
        cmd = [
            exiftool,
            "-overwrite_original",
            "-tagsFromFile", source_path,
            "-all:all",
            # Exclude raster dimensions & structure tags
            "--ImageWidth", "--ImageHeight", "--ImageLength",
            "--BitsPerSample", "--Compression", "--PhotometricInterpretation",
            "--StripOffsets", "--RowsPerStrip", "--StripByteCounts",
            "--TileOffsets", "--TileByteCounts", "--SamplesPerPixel",
            "--PlanarConfiguration",
            "--NewSubfileType", "--SubfileType",
            # Exclude ICC profile tag so our tagged color profile is preserved
            "--ICC_Profile",
            target_path
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if res.returncode == 0:
                return True
            else:
                print(f"  -> Exiftool warning copying metadata: {res.stderr.strip() or res.stdout.strip()}")
        except Exception as e:
            print(f"  -> Exiftool error: {e}")

    # Fallback to piexif if available
    try:
        import piexif
        exif_dict = piexif.load(source_path)
        if exif_dict:
            # Strip 0th IFD geometry
            for tag in [piexif.ImageIFD.ImageWidth, piexif.ImageIFD.ImageLength, piexif.ImageIFD.BitsPerSample,
                        piexif.ImageIFD.Compression, piexif.ImageIFD.PhotometricInterpretation,
                        piexif.ImageIFD.StripOffsets, piexif.ImageIFD.RowsPerStrip, piexif.ImageIFD.StripByteCounts]:
                exif_dict.get("0th", {}).pop(tag, None)
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, target_path)
            return True
    except Exception:
        pass

    return False
