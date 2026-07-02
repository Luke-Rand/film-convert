import os
from pathlib import Path
import numpy as np
import tifffile
import argparse
import rawpy

def process_positives(input_path, output_dir=None, clip=0.1, gamma=2.2, compress_tiff=False, global_levels=False, ignore_margin=0.15, scurve=0.0, autocrop=False, monochrome=False, monochrome_channel="luminance"):
    """
    Processes 16-bit TIFF/DNG files: inverts, normalizes, applies gamma, crops, and saves.
    """
    # Find supported image files
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

    # Create output directory
    if output_dir is None:
        # Check if the base directory name is "negatives"
        if os.path.basename(base_dir).lower() == "negatives":
            # If so, place the 'Positives' folder one level up
            output_dir = os.path.join(os.path.dirname(base_dir), "Positives")
        else:
            # Otherwise, place it as a subdirectory of the base
            output_dir = os.path.join(base_dir, "Positives")
    os.makedirs(output_dir, exist_ok=True)

    # Process files
    for filepath in image_files:
        filename = Path(filepath).name
        print(f"Processing {filename}...")
        
        try:
            # Read file based on extension
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.dng':
                with rawpy.imread(filepath) as raw:
                    # Extract as linear 16-bit
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
            
            # Check for 16-bit data
            if img.dtype != np.uint16:
                print(f"  -> WARNING: {filename} is not 16-bit (uint16). Skipping.")
                continue
                
            # Remove alpha channels from stitched panos to prevent transparency issues
            if img.ndim == 3 and img.shape[2] > 3:
                print("  -> Stripping Alpha/Extra channels...")
                img = img[:, :, :3]
                
            # --- STEP 1: INVERSION ---
            # Convert to float32
            img_float = img.astype(np.float32)
            
            # --- STEP 0: MONOCHROME CONVERSION ---
            is_monochrome = monochrome or (img_float.ndim == 2) or (img_float.ndim == 3 and img_float.shape[2] == 1)
            
            if is_monochrome and img_float.ndim == 3 and img_float.shape[2] > 1:
                print(f"  -> Converting to monochrome using channel: {monochrome_channel}...")
                if monochrome_channel == "red":
                    img_float = img_float[:, :, 0]
                elif monochrome_channel == "green":
                    img_float = img_float[:, :, 1]
                elif monochrome_channel == "blue":
                    img_float = img_float[:, :, 2]
                elif monochrome_channel == "average":
                    img_float = np.mean(img_float, axis=2)
                else: # "luminance" or fallback
                    img_float = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]

            # --- STEP 1: INVERSION ---
            # True linear inversion: use division instead of subtraction for film density.
            # Set minimum value to 1.0 to avoid division by zero.
            img_float = 1.0 / np.maximum(img_float, 1.0)
            
            # --- STEP 2: CROPPING & LEVELS NORMALIZATION ---
            h, w = img_float.shape[:2]
            h_margin = int(h * ignore_margin)
            w_margin = int(w * ignore_margin)
            
            if autocrop:
                print(f"  -> Auto-cropping {ignore_margin*100:.0f}% margins (maintaining aspect ratio)...")
                # Crop the image array
                img_float = img_float[h_margin:h-h_margin, w_margin:w-w_margin]
                # Use the entire remaining image for level analysis
                analysis_region = img_float
            else:
                # Use the center of the image to calculate percentiles
                analysis_region = img_float[h_margin:h-h_margin, w_margin:w-w_margin]
                
            print(f"  -> Normalizing levels (clip={clip}%)...")
            
            if global_levels or is_monochrome:
                # Global normalization
                p_low = np.percentile(analysis_region, clip)
                p_high = np.percentile(analysis_region, 100 - clip)
                
                if p_high > p_low:
                    img_float = (img_float - p_low) / (p_high - p_low) * 65535.0
            else:
                # Per-channel normalization (Auto-Color) to remove color casts.
                for c in range(3):
                    analysis_channel = analysis_region[:, :, c]
                    p_low = np.percentile(analysis_channel, clip)
                    p_high = np.percentile(analysis_channel, 100 - clip)
                    
                    if p_high > p_low:
                        img_float[:, :, c] = (img_float[:, :, c] - p_low) / (p_high - p_low) * 65535.0
            
            # Clip values to 0-65535
            img_float = np.clip(img_float, 0, 65535)
            
            # --- STEP 3: GAMMA AND CONTRAST ---
            # Apply gamma curve to lift midtones.
            if gamma != 1.0 or scurve > 0.0:
                print(f"  -> Applying tone curve (gamma={gamma}, scurve={scurve})...")
                # Normalize to 0.0-1.0
                img_norm = img_float / 65535.0
                
                # Apply gamma curve
                if gamma != 1.0:
                    img_norm = img_norm ** (1.0 / gamma)
                
                # Apply S-Curve
                if scurve > 0.0:
                    c = 1.0 + scurve
                    mask = img_norm < 0.5
                    
                    # Piecewise curve to stretch midtones
                    img_norm[mask] = 0.5 * (2.0 * img_norm[mask]) ** c
                    img_norm[~mask] = 1.0 - 0.5 * (2.0 * (1.0 - img_norm[~mask])) ** c
                
                # Scale back to 16-bit
                img_float = img_norm * 65535.0
                
            # --- STEP 4: SAVE OUT ---
            # Convert back to contiguous 16-bit array
            final_img = img_float.astype(np.uint16)
            final_img = np.ascontiguousarray(final_img)
            
            # Generate output filename
            base_name = os.path.splitext(filename)[0]
            if "_Composite" in base_name:
                out_filename = base_name.replace("_Composite", "_Positive") + ".tiff"
            else:
                out_filename = f"Positive_{base_name}.tiff"
                
            output_filepath = os.path.join(output_dir, out_filename)
            
            # Set up compression
            tiff_compression = 'zlib' if compress_tiff else None
            photometric = 'minisblack' if is_monochrome else 'rgb'
            tifffile.imwrite(output_filepath, final_img, photometric=photometric, compression=tiff_compression)
            
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
    parser.add_argument("--monochrome", "--bw", action="store_true",
                        help="Convert output composite to monochrome / black and white positive")
    parser.add_argument("--monochrome-channel", "--bw-channel", type=str, default="luminance",
                        choices=["luminance", "average", "red", "green", "blue"],
                        help="Method to convert RGB to monochrome. Default: luminance (weighted). 'green' is recommended for high resolution on standard Bayer sensors.")
    
    args = parser.parse_args()
    
    process_positives(
        input_path=args.input, 
        clip=args.clip, 
        gamma=args.gamma, 
        compress_tiff=args.compress,
        global_levels=args.global_levels,
        ignore_margin=args.margin,
        scurve=args.scurve,
        autocrop=args.autocrop,
        monochrome=args.monochrome,
        monochrome_channel=args.monochrome_channel
    )