import os
from pathlib import Path
import numpy as np
import tifffile
import argparse
import rawpy

def process_positives(directory_path, output_dir=None, clip=0.1, gamma=2.2, compress_tiff=False, global_levels=False, ignore_margin=0.15):
    """
    Scans a directory for 16-bit TIFF and DNG files, inverts them (negative to positive),
    normalizes the black and white points (ignoring outer margins), applies a gamma curve, and saves the results.
    """
    # 1. Gather all supported image files (case-insensitive)
    supported_exts = {'.tiff', '.tif', '.dng'}
    image_files = [
        os.path.join(directory_path, f) for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f)) and os.path.splitext(f)[1].lower() in supported_exts
    ]
    
    if not image_files:
        print(f"No .tiff, .tif, or .dng files found in {directory_path}")
        return

    # 2. Sort files alphabetically 
    image_files.sort()
    total_files = len(image_files)
    print(f"Found {total_files} files to process.")

    # 3. Setup output directory
    if output_dir is None:
        output_dir = os.path.join(directory_path, "Positives")
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
                
            # --- STEP 1: INVERSION ---
            # Convert to float32 for precise math
            img_float = img.astype(np.float32)
            
            # THE FIX: True Linear Inversion
            # Inverting linear RAW data with subtraction (max - pixel) crushes highlight 
            # contrast because it's the wrong math for density. Film density is logarithmic.
            # To get linear scene exposure back from linear transmission, we must divide.
            # Avoid division by zero by using a minimum value of 1.0.
            img_float = 1.0 / np.maximum(img_float, 1.0)
            
            # --- STEP 2: LEVELS NORMALIZATION ---
            print(f"  -> Normalizing levels (clip={clip}%, ignoring {ignore_margin*100:.0f}% margins)...")
            
            h, w = img_float.shape[:2]
            h_margin = int(h * ignore_margin)
            w_margin = int(w * ignore_margin)
            
            # Use only the center of the image to calculate percentiles
            # This prevents film holders (pure black) and unexposed film base (pure white) 
            # from skewing the auto-levels.
            analysis_region = img_float[h_margin:h-h_margin, w_margin:w-w_margin]
            
            if global_levels:
                # Global normalization: preserves the exact color ratio/balance 
                # produced by the compositor's --neutralize flag.
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
            
            # --- STEP 3: GAMMA CORRECTION ---
            # Linear scans look very dark/muddy when inverted. We apply a standard
            # viewing gamma (e.g., 2.2) to lift the midtones properly.
            if gamma != 1.0:
                print(f"  -> Applying gamma correction (gamma={gamma})...")
                # Normalize to 0.0-1.0, apply power law, scale back to 16-bit
                img_float = ((img_float / 65535.0) ** (1.0 / gamma)) * 65535.0
                
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
                        help="Path to the directory containing 16-bit composite TIFF or RAW DNG files")
    parser.add_argument("-c", "--compress", action="store_true", 
                        help="Enable lossless compression (zlib/deflate) for output TIFFs")
    parser.add_argument("-p", "--clip", type=float, default=0.1,
                        help="Percentile to clip for black/white points (default: 0.1%% to ignore dust/scratches)")
    parser.add_argument("-g", "--gamma", type=float, default=2.2,
                        help="Gamma correction curve to apply (default: 2.2). Set to 1.0 for strictly linear output.")
    parser.add_argument("-m", "--margin", type=float, default=0.15,
                        help="Fraction of outer edge to ignore when calculating levels (default: 0.15 = 15%%). Prevents film holders from skewing brightness.")
    parser.add_argument("--global-levels", action="store_true",
                        help="Stretch levels globally instead of per-channel. Use this if you relied on the compositor's neutralization and want to perfectly maintain that color balance.")
    
    args = parser.parse_args()
    
    process_positives(
        directory_path=args.input, 
        clip=args.clip, 
        gamma=args.gamma, 
        compress_tiff=args.compress,
        global_levels=args.global_levels,
        ignore_margin=args.margin
    )