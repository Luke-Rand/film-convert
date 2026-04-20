import os
from pathlib import Path
import numpy as np
import tifffile
import argparse
import rawpy

def process_positives(input_path, output_dir=None, clip=0.1, gamma=2.2, compress_tiff=False, global_levels=False, ignore_margin=0.15, scurve=0.0, autocrop=False):
    """
    Scans a directory or processes a single 16-bit TIFF/DNG file, inverts it (negative to positive),
    normalizes the black and white points, applies a gamma curve, optionally crops, and saves the results.
    """
    # 1. Gather all supported image files (case-insensitive)
    supported_exts = {'.tiff', '.tif', '.dng'}
    image_files = []
    
    if os.path.isfile(input_path):
        if os.path.splitext(input_path)[1].lower() in supported_exts:
            image_files.append(input_path)
        base_dir = os.path.dirname(input_path) or "."
    elif os.path.isdir(input_path):
        image_files = [
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, f)) and os.path.splitext(f)[1].lower() in supported_exts
        ]
        base_dir = input_path
    else:
        print(f"Error: '{input_path}' is not a valid file or directory.")
        return
    
    if not image_files:
        print(f"No valid .tiff, .tif, or .dng files found for input: {input_path}")
        return

    # 2. Sort files alphabetically 
    image_files.sort()
    total_files = len(image_files)
    print(f"Found {total_files} files to process.")

    # 3. Setup output directory
    if output_dir is None:
        output_dir = os.path.join(base_dir, "Positives")
    os.makedirs(output_dir, exist_ok=True)

    # 4. Process each file
    for filepath in image_files:
        filename = Path(filepath).name
        print(f"Processing {filename}...")
        
        try:
            # Read the file based on its extension
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.dng':
                with rawpy.imread(filepath) as raw:
                    # Extract as linear 16-bit to match compositor TIFFs
                    img = raw.postprocess(
                        gamma=(1, 1),
                        no_auto_bright=True,
                        use_camera_wb=False,
                        user_wb=[1.0, 1.0, 1.0, 1.0],
                        output_color=rawpy.ColorSpace.raw,
                        output_bps=16
                    )
            else:
                img = tifffile.imread(filepath)
            
            # Ensure we are working with 16-bit data
            if img.dtype != np.uint16:
                print(f"  -> WARNING: {filename} is not 16-bit (uint16). Skipping.")
                continue
                
            # THE FIX: Strip alpha channels from stitched panoramas
            # Stitched TIFFs often have a 4th Alpha channel. If left in, it becomes fully 
            # transparent during math, making the final image appear entirely white.
            if img.ndim == 3 and img.shape[2] > 3:
                print("  -> Stripping Alpha/Extra channels...")
                img = img[:, :, :3]
                
            # --- STEP 1: INVERSION ---
            # Convert to float32 for precise math
            img_float = img.astype(np.float32)
            
            # THE FIX: True Linear Inversion
            # Inverting linear RAW data with subtraction (max - pixel) crushes highlight 
            # contrast because it's the wrong math for density. Film density is logarithmic.
            # To get linear scene exposure back from linear transmission, we must divide.
            # Avoid division by zero by using a minimum value of 1.0.
            img_float = 1.0 / np.maximum(img_float, 1.0)
            
            # --- STEP 2: CROPPING & LEVELS NORMALIZATION ---
            h, w = img_float.shape[:2]
            h_margin = int(h * ignore_margin)
            w_margin = int(w * ignore_margin)
            
            if autocrop:
                print(f"  -> Auto-cropping {ignore_margin*100:.0f}% margins (maintaining aspect ratio)...")
                # Physically crop the image array
                img_float = img_float[h_margin:h-h_margin, w_margin:w-w_margin]
                # The region for level analysis is now the entire remaining image
                analysis_region = img_float
            else:
                # Keep the full image, but only use the center to calculate percentiles
                analysis_region = img_float[h_margin:h-h_margin, w_margin:w-w_margin]
                
            print(f"  -> Normalizing levels (clip={clip}%)...")
            
            if global_levels:
                # Global normalization
                p_low = np.percentile(analysis_region, clip)
                p_high = np.percentile(analysis_region, 100 - clip)
                
                if p_high > p_low:
                    img_float = (img_float - p_low) / (p_high - p_low) * 65535.0
            else:
                # Per-channel normalization (Auto-Color): Calculates black/white points
                # independently for R, G, and B. This acts as an automated color correction 
                # to remove residual film base color casts.
                for c in range(3):
                    analysis_channel = analysis_region[:, :, c]
                    p_low = np.percentile(analysis_channel, clip)
                    p_high = np.percentile(analysis_channel, 100 - clip)
                    
                    if p_high > p_low:
                        img_float[:, :, c] = (img_float[:, :, c] - p_low) / (p_high - p_low) * 65535.0
            
            # Clip any mathematical overshoots to the absolute 0-65535 bounds
            img_float = np.clip(img_float, 0, 65535)
            
            # --- STEP 3: GAMMA AND CONTRAST ---
            # Linear scans look very dark/muddy when inverted. We apply a standard
            # viewing gamma (e.g., 2.2) to lift the midtones properly.
            if gamma != 1.0 or scurve > 0.0:
                print(f"  -> Applying tone curve (gamma={gamma}, scurve={scurve})...")
                # Normalize to 0.0-1.0
                img_norm = img_float / 65535.0
                
                # Apply power law for gamma
                if gamma != 1.0:
                    img_norm = img_norm ** (1.0 / gamma)
                
                # Apply S-Curve for contrast
                if scurve > 0.0:
                    c = 1.0 + scurve
                    mask = img_norm < 0.5
                    
                    # Piecewise contrast curve that pins black/white but stretches midtones
                    img_norm[mask] = 0.5 * (2.0 * img_norm[mask]) ** c
                    img_norm[~mask] = 1.0 - 0.5 * (2.0 * (1.0 - img_norm[~mask])) ** c
                
                # Scale back to 16-bit
                img_float = img_norm * 65535.0
                
            # --- STEP 4: SAVE OUT ---
            # Convert back to contiguous 16-bit integer array
            final_img = img_float.astype(np.uint16)
            final_img = np.ascontiguousarray(final_img)
            
            # Generate new filename ensuring it ends in .tiff
            base_name = os.path.splitext(filename)[0]
            if "_Composite" in base_name:
                out_filename = base_name.replace("_Composite", "_Positive") + ".tiff"
            else:
                out_filename = f"Positive_{base_name}.tiff"
                
            output_filepath = os.path.join(output_dir, out_filename)
            
            # Setup compression
            tiff_compression = 'zlib' if compress_tiff else None
            tifffile.imwrite(output_filepath, final_img, photometric='rgb', compression=tiff_compression)
            
            print(f"  -> Saved positive to: {out_filename}\n")
            
        except Exception as e:
            print(f"  -> ERROR processing {filename}: {e}\n")
            
    print("Inversion and normalization complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invert, Normalize, and Gamma Correct 16-bit linear TIFF and RAW DNG film scans.")
    
    # Define command line arguments
    parser.add_argument("-i", "--input", type=str, required=True, 
                        help="Path to a single 16-bit composite TIFF/RAW DNG file, or a directory containing them")
    parser.add_argument("-c", "--compress", action="store_true", 
                        help="Enable lossless compression (zlib/deflate) for output TIFFs")
    parser.add_argument("-p", "--clip", type=float, default=0.1,
                        help="Percentile to clip for black/white points (default: 0.1%% to ignore dust/scratches)")
    parser.add_argument("-g", "--gamma", type=float, default=2.2,
                        help="Gamma correction curve to apply (default: 2.2). Set to 1.0 for strictly linear output.")
    parser.add_argument("-s", "--scurve", type=float, default=0.0,
                        help="Strength of the contrast S-Curve to apply (default: 0.0 = none). Try 0.2 to 0.5 for a film-like punch.")
    parser.add_argument("-m", "--margin", type=float, default=0.03,
                        help="Fraction of outer edge to ignore when calculating levels (default: 0.03 = 3%%). Prevents film holders from skewing brightness.")
    parser.add_argument("-a", "--autocrop", action="store_true",
                        help="Physically crop off the outer margins defined by --margin from the final saved image.")
    parser.add_argument("--global-levels", action="store_true",
                        help="Stretch levels globally instead of per-channel. Use this if you relied on the compositor's neutralization and want to perfectly maintain that color balance.")
    
    args = parser.parse_args()
    
    process_positives(
        input_path=args.input, 
        clip=args.clip, 
        gamma=args.gamma, 
        compress_tiff=args.compress,
        global_levels=args.global_levels,
        ignore_margin=args.margin,
        scurve=args.scurve,
        autocrop=args.autocrop
    )