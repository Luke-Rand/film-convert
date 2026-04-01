import os
from pathlib import Path
import numpy as np
import tifffile
import argparse

def process_positives(directory_path, output_dir=None, clip=0.1, gamma=2.2, compress_tiff=False, global_levels=False):
    """
    Scans a directory for 16-bit TIFF files, inverts them (negative to positive),
    normalizes the black and white points, applies a gamma curve, and saves the results.
    """
    # 1. Gather all supported TIFF files (case-insensitive)
    supported_exts = {'.tiff', '.tif'}
    tiff_files = [
        os.path.join(directory_path, f) for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f)) and os.path.splitext(f)[1].lower() in supported_exts
    ]
    
    if not tiff_files:
        print(f"No .tiff or .tif files found in {directory_path}")
        return

    # 2. Sort files alphabetically 
    tiff_files.sort()
    total_files = len(tiff_files)
    print(f"Found {total_files} TIFF files to process.")

    # 3. Setup output directory
    if output_dir is None:
        output_dir = os.path.join(directory_path, "Positives")
    os.makedirs(output_dir, exist_ok=True)

    # 4. Process each file
    for filepath in tiff_files:
        filename = Path(filepath).name
        print(f"Processing {filename}...")
        
        try:
            # Read the TIFF file
            img = tifffile.imread(filepath)
            
            # Ensure we are working with 16-bit data from the compositor
            if img.dtype != np.uint16:
                print(f"  -> WARNING: {filename} is not 16-bit (uint16). Skipping.")
                continue
                
            # --- STEP 1: INVERSION ---
            # Subtract pixel values from the maximum 16-bit value (65535)
            inverted = 65535 - img
            
            # Convert to float32 for precise math during normalization and gamma
            img_float = inverted.astype(np.float32)
            
            # --- STEP 2: LEVELS NORMALIZATION ---
            print(f"  -> Normalizing levels (clip={clip}%)...")
            
            if global_levels:
                # Global normalization: preserves the exact color ratio/balance 
                # produced by the compositor's --neutralize flag.
                p_low = np.percentile(img_float, clip)
                p_high = np.percentile(img_float, 100 - clip)
                
                if p_high > p_low:
                    img_float = (img_float - p_low) / (p_high - p_low) * 65535.0
            else:
                # Per-channel normalization (Auto-Color): Calculates black/white points
                # independently for R, G, and B. This acts as an automated color correction 
                # to remove residual film base color casts.
                for c in range(3):
                    channel = img_float[:, :, c]
                    p_low = np.percentile(channel, clip)
                    p_high = np.percentile(channel, 100 - clip)
                    
                    if p_high > p_low:
                        img_float[:, :, c] = (channel - p_low) / (p_high - p_low) * 65535.0
            
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
            
            # Generate new filename
            out_filename = filename.replace("_Composite", "_Positive")
            if out_filename == filename:
                out_filename = f"Positive_{filename}"
                
            output_filepath = os.path.join(output_dir, out_filename)
            
            # Setup compression
            tiff_compression = 'zlib' if compress_tiff else None
            tifffile.imwrite(output_filepath, final_img, photometric='rgb', compression=tiff_compression)
            
            print(f"  -> Saved positive to: {out_filename}\n")
            
        except Exception as e:
            print(f"  -> ERROR processing {filename}: {e}\n")
            
    print("Inversion and normalization complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invert, Normalize, and Gamma Correct 16-bit linear TIFF film scans.")
    
    # Define command line arguments
    parser.add_argument("-i", "--input", type=str, required=True, 
                        help="Path to the directory containing 16-bit composite TIFF files")
    parser.add_argument("-c", "--compress", action="store_true", 
                        help="Enable lossless compression (zlib/deflate) for output TIFFs")
    parser.add_argument("-p", "--clip", type=float, default=0.1,
                        help="Percentile to clip for black/white points (default: 0.1%% to ignore dust/scratches)")
    parser.add_argument("-g", "--gamma", type=float, default=2.2,
                        help="Gamma correction curve to apply (default: 2.2). Set to 1.0 for strictly linear output.")
    parser.add_argument("--global-levels", action="store_true",
                        help="Stretch levels globally instead of per-channel. Use this if you relied on the compositor's neutralization and want to perfectly maintain that color balance.")
    
    args = parser.parse_args()
    
    process_positives(
        directory_path=args.input, 
        clip=args.clip, 
        gamma=args.gamma, 
        compress_tiff=args.compress,
        global_levels=args.global_levels
    )