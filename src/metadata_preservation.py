import os
import shutil
import subprocess
import tempfile
import hashlib
import time
from pathlib import Path
from typing import Optional, Tuple, List, Any, Dict

# Standard D50 to sRGB Color Matrix (SRATIONAL, 9 signed integer pairs)
DEFAULT_D50_SRGB_MATRIX = [
    (31339, 10000), (-16169, 10000), (-4906, 10000),
    (-9788, 10000), (19161, 10000), (335, 10000),
    (719, 10000), (-2290, 10000), (14052, 10000)
]

STAGE_DIR = os.path.join(tempfile.gettempdir(), "film_convert_stage")
PREVIEW_CACHE_DIR = os.path.join(tempfile.gettempdir(), "film_convert_previews")

# In-memory caches to avoid re-reading 40MB RAW files over SMB or spawning duplicate subprocesses
_EXIFTOOL_PATH_CACHE: Optional[str] = None
_EXIFTOOL_PROBED: bool = False
_CAMERA_INFO_CACHE: Dict[str, Tuple[Optional[List[Tuple[int, int]]], Optional[str]]] = {}


def get_stage_dir() -> str:
    os.makedirs(STAGE_DIR, exist_ok=True)
    return STAGE_DIR


def get_preview_cache_dir() -> str:
    os.makedirs(PREVIEW_CACHE_DIR, exist_ok=True)
    return PREVIEW_CACHE_DIR


def get_staged_raw_path(filepath: str) -> str:
    """Returns the local SSD staging path for a captured RAW file."""
    return os.path.join(get_stage_dir(), os.path.basename(filepath))


def resolve_fast_read_path(filepath: str) -> str:
    """
    If a local staged copy of `filepath` exists in STAGE_DIR, returns the local SSD path
    to avoid reading across network storage (SMB/NFS). Otherwise returns `filepath`.
    """
    if not filepath:
        return filepath
    try:
        staged = os.path.join(STAGE_DIR, os.path.basename(filepath))
        if os.path.isfile(staged) and os.path.getsize(staged) > 0:
            return staged
    except OSError:
        pass
    return filepath


def cleanup_staged_raw(filepath: str, max_age_sec: float = 1800.0) -> None:
    """Removes the local staged copy of `filepath` and prunes stale staging files."""
    if not filepath:
        return
    try:
        staged = os.path.join(STAGE_DIR, os.path.basename(filepath))
        if os.path.isfile(staged):
            os.remove(staged)
    except OSError:
        pass

    try:
        if os.path.isdir(STAGE_DIR):
            now = time.time()
            for entry in os.listdir(STAGE_DIR):
                full_p = os.path.join(STAGE_DIR, entry)
                try:
                    if os.path.isfile(full_p) and (now - os.path.getmtime(full_p)) > max_age_sec:
                        os.remove(full_p)
                except OSError:
                    pass
    except OSError:
        pass


def get_preview_cache_key(filepath: str) -> str:
    """
    Computes a stable preview cache key for a frame across session subdirectories
    (so moving a Composite.dng from negatives/ to processed_raws/ retains its cached preview).
    """
    abs_p = os.path.abspath(filepath)
    parent = os.path.dirname(abs_p)
    parent_name = os.path.basename(parent).lower()
    if parent_name in ("negatives", "positives", "processed_raws", "error_raws", "composites"):
        session_root = os.path.dirname(parent)
    else:
        session_root = parent
    key_str = f"{session_root.lower()}::{os.path.basename(abs_p).lower()}"
    digest = hashlib.md5(key_str.encode("utf-8")).hexdigest()
    return os.path.join(get_preview_cache_dir(), f"{digest}.jpg")


def find_exiftool() -> Optional[str]:
    """Locates exiftool binary on PATH or standard install locations (cached)."""
    global _EXIFTOOL_PATH_CACHE, _EXIFTOOL_PROBED
    if _EXIFTOOL_PROBED and _EXIFTOOL_PATH_CACHE and os.path.exists(_EXIFTOOL_PATH_CACHE):
        return _EXIFTOOL_PATH_CACHE

    tool = shutil.which("exiftool")
    if tool and os.path.exists(tool):
        _EXIFTOOL_PATH_CACHE = tool
        _EXIFTOOL_PROBED = True
        return tool
        
    common_paths = [
        "/opt/homebrew/bin/exiftool",
        "/usr/local/bin/exiftool",
        "/usr/bin/exiftool",
    ]
    for p in common_paths:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            _EXIFTOOL_PATH_CACHE = p
            _EXIFTOOL_PROBED = True
            return p
            
    _EXIFTOOL_PROBED = True
    return None


def _matrix_to_srational_pairs(cam_xyz: Any) -> Optional[List[Tuple[int, int]]]:
    try:
        import numpy as np
        if cam_xyz is not None and len(cam_xyz) >= 3:
            cam_3x3 = np.array(cam_xyz[:3, :3], dtype=np.float64)
            if np.linalg.matrix_rank(cam_3x3) == 3:
                xyz_cam = np.linalg.inv(cam_3x3)
                pairs = []
                for row in xyz_cam:
                    for val in row:
                        pairs.append((int(round(val * 10000)), 10000))
                return pairs
    except Exception:
        pass
    return None


def cache_camera_matrix_from_raw(source_path: str, cam_xyz: Any) -> Optional[List[Tuple[int, int]]]:
    """
    Caches the camera color matrix extracted from an already-open rawpy handle
    so get_dng_extratags does not need to re-open the 40MB RAW file over the network.
    """
    if not source_path:
        return None
    pairs = _matrix_to_srational_pairs(cam_xyz)
    if pairs:
        abs_p = os.path.abspath(source_path)
        dir_key = f"dir::{os.path.dirname(abs_p)}::{os.path.splitext(abs_p)[1].lower()}"
        existing_model = _CAMERA_INFO_CACHE.get(abs_p, (None, None))[1] or _CAMERA_INFO_CACHE.get(dir_key, (None, None))[1]
        _CAMERA_INFO_CACHE[abs_p] = (pairs, existing_model)
        _CAMERA_INFO_CACHE[dir_key] = (pairs, existing_model)
    return pairs


def extract_camera_info(source_path: str) -> Tuple[Optional[List[Tuple[int, int]]], Optional[str]]:
    """
    Extracts camera matrix and model from source RAW using rawpy or exiftool.
    Returns (color_matrix_srational_pairs, camera_model_name).
    Uses in-memory roll/file cache and local staging path resolution to avoid redundant network reads.
    """
    if not source_path or not os.path.exists(source_path):
        return None, None

    abs_p = os.path.abspath(source_path)
    dir_key = f"dir::{os.path.dirname(abs_p)}::{os.path.splitext(abs_p)[1].lower()}"

    cached_matrix, cached_model = _CAMERA_INFO_CACHE.get(abs_p, (None, None))
    if not cached_matrix or not cached_model:
        dir_matrix, dir_model = _CAMERA_INFO_CACHE.get(dir_key, (None, None))
        cached_matrix = cached_matrix or dir_matrix
        cached_model = cached_model or dir_model

    if cached_matrix and cached_model:
        return cached_matrix, cached_model

    fast_source = resolve_fast_read_path(source_path)
    camera_model = cached_model
    color_matrix = cached_matrix

    # Try rawpy first if matrix not already cached
    if not color_matrix:
        try:
            import rawpy
            with rawpy.imread(fast_source) as raw:
                color_matrix = _matrix_to_srational_pairs(getattr(raw, "rgb_xyz_matrix", None))
        except Exception:
            pass

    # Try exiftool for camera model if not already cached
    if not camera_model:
        exiftool = find_exiftool()
        if exiftool:
            try:
                res = subprocess.run(
                    [exiftool, "-s3", "-Model", fast_source],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                model_str = res.stdout.strip()
                if model_str:
                    camera_model = model_str
            except Exception:
                pass

    if color_matrix or camera_model:
        _CAMERA_INFO_CACHE[abs_p] = (color_matrix, camera_model)
        _CAMERA_INFO_CACHE[dir_key] = (color_matrix, camera_model)

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
    if not source_path:
        return False
    fast_source = resolve_fast_read_path(source_path)
    if not os.path.exists(fast_source):
        return False
    if not target_path or not os.path.exists(target_path):
        return False

    exiftool = find_exiftool()
    if exiftool:
        cmd = [
            exiftool,
            "-overwrite_original",
            "-tagsFromFile", fast_source,
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
        exif_dict = piexif.load(fast_source)
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
