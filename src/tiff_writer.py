import os
import tifffile
import numpy as np
from typing import Optional

from icc_manager import get_icc_profile
from metadata_preservation import get_dng_extratags, transfer_raw_metadata


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

    if is_dng:
        # DNG LinearRaw (34892) for color, BlackIsZero (1) for monochrome
        photometric = 1 if is_mono else 34892
        extratags = get_dng_extratags(
            source_path=source_metadata_path,
            icc_profile_bytes=icc_bytes,
            is_monochrome=is_mono
        )
        tifffile.imwrite(
            filepath,
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
            filepath,
            img_data,
            photometric=photometric,
            compression=compression,
            rowsperstrip=64,
            iccprofile=icc_bytes
        )

    # Transfer archival camera EXIF/IPTC provenance metadata if source is available
    if source_metadata_path and os.path.exists(source_metadata_path):
        transfer_raw_metadata(source_metadata_path, filepath)


# Alias for backward compatibility with existing tests and codebase
write_16bit_tiff = write_16bit_image
