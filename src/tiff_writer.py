import os
import shutil
import tempfile
import tifffile
import numpy as np
from PIL import Image
from typing import Optional

from icc_manager import get_icc_profile
from metadata_preservation import (
    get_dng_extratags,
    transfer_raw_metadata,
    get_stage_dir,
    get_staged_raw_path,
    get_preview_cache_key,
    resolve_fast_read_path,
)


def _write_preview_cache_from_array(filepath: str, img_data: np.ndarray, is_dng: bool) -> None:
    """
    Generates a lightweight 8-bit JPEG preview in local SSD cache directly from the in-memory
    16-bit image array, eliminating 190MB network reads when the UI requests /api/preview.
    """
    try:
        h, w = img_data.shape[:2]
        # Strided downsample so 6960x4640 reduces to ~1740x1160 in <1ms without full-array allocation
        step = max(1, max(w, h) // 1800)
        sub = img_data[::step, ::step]
        if sub.ndim == 3 and sub.shape[2] > 3:
            sub = sub[:, :, :3]

        if sub.dtype == np.uint16:
            if is_dng:
                sub_float = sub.astype(np.float32) / 65535.0
                sub_8bit = np.clip((sub_float ** (1.0 / 2.2)) * 255.0, 0, 255).astype(np.uint8)
            else:
                sub_8bit = (sub >> 8).astype(np.uint8)
        else:
            sub_8bit = sub.astype(np.uint8)

        cache_path = get_preview_cache_key(filepath)
        tmp_cache = f"{cache_path}.{os.getpid()}.tmp"
        pil_img = Image.fromarray(sub_8bit)
        pil_img.save(tmp_cache, format="JPEG", quality=88)
        os.replace(tmp_cache, cache_path)
    except Exception:
        pass


def _atomic_place_file(src_local_path: str, dst_filepath: str, keep_staged_for_read: bool = False) -> None:
    """
    Moves `src_local_path` to `dst_filepath` atomically on the same filesystem,
    or copies via a 4MB-buffered `.part` file + atomic rename across volumes (e.g. SMB/NAS).
    When `keep_staged_for_read=True` and copying across volumes, retains `src_local_path` at
    `get_staged_raw_path(dst_filepath)` so downstream pipeline stages read from local SSD.
    """
    dst_dir = os.path.dirname(os.path.abspath(dst_filepath)) or "."
    os.makedirs(dst_dir, exist_ok=True)

    try:
        # Fast O(1) rename when temp dir and destination are on the same volume
        os.replace(src_local_path, dst_filepath)
        return
    except OSError:
        pass

    part_path = os.path.join(dst_dir, f".{os.path.basename(dst_filepath)}.{os.getpid()}.part")
    try:
        with open(src_local_path, "rb") as fsrc, open(part_path, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst, length=4 * 1024 * 1024)
        os.replace(part_path, dst_filepath)
        if keep_staged_for_read and os.path.exists(src_local_path):
            try:
                os.replace(src_local_path, get_staged_raw_path(dst_filepath))
            except OSError:
                pass
    finally:
        if os.path.exists(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass
        if os.path.exists(src_local_path):
            try:
                os.remove(src_local_path)
            except OSError:
                pass


def write_16bit_image(
    filepath: str,
    img_data: np.ndarray,
    is_monochrome: bool = False,
    compress: bool = False,
    icc_profile: Optional[str] = "adobe_rgb",
    source_metadata_path: Optional[str] = None
) -> None:
    """
    Writes a 16-bit NumPy image array as a high-precision 16-bit TIFF or True Linear DNG.
    - Stages file creation and ExifTool metadata tagging on local SSD before placing onto
      destination storage, eliminating ~400MB of redundant SMB read/rewrite traffic per image.
    - Pre-generates a local 8-bit preview JPEG directly from RAM for instant UI display.
    - Embeds standard ICC profile tags (e.g. Adobe RGB 1998, ProPhoto RGB, sRGB, Gray Gamma 2.2).
    - Injects full DNG specification tags & camera matrix when writing .dng files.
    - Preserves archival EXIF/IPTC/XMP camera metadata when source_metadata_path is provided.
    - Uses rowsperstrip=64 for streaming buffer compatibility with video NLEs like DaVinci Resolve.
    """
    # Ensure uint16 datatype
    if img_data.dtype != np.uint16:
        img_data = img_data.astype(np.uint16)

    # Convert to contiguous array
    img_data = np.ascontiguousarray(img_data)

    # Auto-detect monochrome / grayscale
    is_mono = is_monochrome or (img_data.ndim == 2) or (img_data.ndim == 3 and img_data.shape[2] == 1)

    # Fetch binary ICC profile
    icc_bytes = get_icc_profile(icc_profile, is_monochrome=is_mono)

    # Configure compression (zlib if requested, otherwise None for uncompressed)
    compression = 'zlib' if compress else None

    ext = os.path.splitext(filepath)[1].lower()
    is_dng = (ext == '.dng')

    fast_source_meta = resolve_fast_read_path(source_metadata_path) if source_metadata_path else None
    has_source_meta = bool(fast_source_meta and os.path.exists(fast_source_meta))

    stage_dir = get_stage_dir()
    fd, local_tmp_path = tempfile.mkstemp(prefix="fc_img_", suffix=ext or ".tiff", dir=stage_dir)
    os.close(fd)

    try:
        if is_dng:
            # DNG LinearRaw (34892) for color, BlackIsZero (1) for monochrome
            photometric = 1 if is_mono else 34892
            extratags = get_dng_extratags(
                source_path=source_metadata_path,
                icc_profile_bytes=icc_bytes,
                is_monochrome=is_mono
            )
            tifffile.imwrite(
                local_tmp_path,
                img_data,
                photometric=photometric,
                compression=compression,
                rowsperstrip=64,
                iccprofile=icc_bytes,
                extratags=extratags
            )
        else:
            # Standard TIFF: Photometric 1 for monochrome (BlackIsZero), 2 for RGB
            photometric = 1 if is_mono else 2
            tifffile.imwrite(
                local_tmp_path,
                img_data,
                photometric=photometric,
                compression=compression,
                rowsperstrip=64,
                iccprofile=icc_bytes
            )

        # Transfer archival camera EXIF/IPTC provenance metadata on local SSD before network copy
        if has_source_meta:
            transfer_raw_metadata(source_metadata_path, local_tmp_path)

        # Place completed file into destination atomically (retaining staged copy for DNG composites)
        _atomic_place_file(local_tmp_path, filepath, keep_staged_for_read=is_dng)

        # Cache lightweight 8-bit preview JPEG directly from in-memory array
        _write_preview_cache_from_array(filepath, img_data, is_dng=is_dng)
    finally:
        if os.path.exists(local_tmp_path):
            try:
                os.remove(local_tmp_path)
            except OSError:
                pass


# Alias for backward compatibility with existing tests and codebase
write_16bit_tiff = write_16bit_image
