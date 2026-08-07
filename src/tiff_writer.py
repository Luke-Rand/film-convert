import tifffile
import numpy as np

def write_16bit_tiff(filepath, img_data, is_monochrome=False, compress=False):
    """
    Writes a 16-bit NumPy image array (RGB or Grayscale) as a high-precision 16-bit TIFF file.
    Uses rowsperstrip=64 for streaming buffer compatibility with video NLEs like DaVinci Resolve.
    """
    # Ensure uint16 datatype
    if img_data.dtype != np.uint16:
        img_data = img_data.astype(np.uint16)
    
    # Convert to contiguous array
    img_data = np.ascontiguousarray(img_data)
    
    # Auto-detect monochrome / grayscale
    is_mono = is_monochrome or (img_data.ndim == 2) or (img_data.ndim == 3 and img_data.shape[2] == 1)
    
    # Photometric: 1 for monochrome (BlackIsZero), 2 for RGB
    photometric = 1 if is_mono else 2
    
    # Configure compression (zlib if requested, otherwise None for uncompressed)
    compression = 'zlib' if compress else None
    
    # Write 16-bit TIFF with 64 rows per strip for NLE streaming compatibility
    tifffile.imwrite(
        filepath,
        img_data,
        photometric=photometric,
        compression=compression,
        rowsperstrip=64
    )
