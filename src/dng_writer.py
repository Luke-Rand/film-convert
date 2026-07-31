import tifffile
import numpy as np

def write_linear_dng(filepath, img_data, is_monochrome=False, compress=False):
    """
    Writes a 16-bit NumPy image array (RGB or Grayscale) as a valid Linear DNG file
    using the tifffile library, injecting standard DNG tags.
    """
    # Ensure uint16 datatype
    if img_data.dtype != np.uint16:
        img_data = img_data.astype(np.uint16)
    
    # Convert to contiguous array
    img_data = np.ascontiguousarray(img_data)
    
    # Auto-detect monochrome / grayscale
    is_mono = is_monochrome or (img_data.ndim == 2) or (img_data.ndim == 3 and img_data.shape[2] == 1)
    
    # DNG Version 1.4.0.0
    dng_version = b'\x01\x04\x00\x00'
    camera_model = "FilmConvert Linear DNG"
    
    # D50 to sRGB Color Matrix: converts D50 CIE XYZ to linear sRGB camera space.
    # Store as SRATIONAL (pairs of signed integers: numerator, denominator).
    color_matrix = [
        (31339, 10000), (-16169, 10000), (-4906, 10000),
        (-9788, 10000), (19161, 10000), (335, 10000),
        (719, 10000), (-2290, 10000), (14052, 10000)
    ]
    
    color_matrix_flat = []
    for num, den in color_matrix:
        color_matrix_flat.extend([num, den])
        
    # AsShotNeutral (RATIONAL, 3 values) - set to neutral (1.0, 1.0, 1.0)
    as_shot_neutral = [1, 1, 1, 1, 1, 1]
    
    # Common extratags:
    # Tag 254 (NewSubfileType) -> 0 (Main image)
    # Tag 262 (PhotometricInterpretation) -> 34892 (LinearRaw) for RGB, 1 (minisblack) for Monochrome
    # Tag 50706 (DNGVersion) -> [1, 4, 0, 0]
    # Tag 50708 (UniqueCameraModel) -> "FilmConvert Linear DNG"
    
    extratags = [
        (254, 'I', 1, 0, True),
        (50706, 'B', 4, dng_version, True),
        (50708, 's', len(camera_model) + 1, camera_model, True),
    ]
    
    if not is_mono and img_data.ndim == 3 and img_data.shape[2] == 3:
        # Add color-specific tags for RGB
        extratags.extend([
            (50721, '2i', 9, color_matrix_flat, True),  # ColorMatrix1
            (50728, '2I', 3, as_shot_neutral, True),   # AsShotNeutral
            (50778, 'H', 1, 21, True),                 # CalibrationIlluminant1 (D65)
        ])
    
    # Configure compression
    compression = 'zlib' if compress else None
    
    # Save using tifffile. TiffWriter writes it out.
    # Set photometric to 1 for monochrome, 34892 for color RGB
    photometric = 1 if is_mono else 34892
    tifffile.imwrite(
        filepath,
        img_data,
        photometric=photometric,
        compression=compression,
        extratags=extratags
    )
