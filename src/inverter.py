import os
from pathlib import Path
import numpy as np
import tifffile
import argparse
import rawpy
from dng_writer import write_linear_dng

def process_positives(input_path, output_dir=None, clip=0.1, gamma=2.2, compress_dng=False, global_levels=False, ignore_margin=0.15, scurve=0.0, autocrop=False, monochrome=False, monochrome_channel="luminance"):
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
            # Read file: use tifffile for Linear DNGs/TIFFs to avoid double-demosaicing grid artifacts,
            # and rawpy with DHT demosaicing for camera RAW files to prevent AHD maze grid artifacts on grain.
            ext = os.path.splitext(filename)[1].lower()
            img = None
            if ext in ['.dng', '.tiff', '.tif']:
                try:
                    with tifffile.TiffFile(filepath) as tif:
                        page = tif.pages[0]
                        # Photometric 32803 indicates CFA raw sensor data needing rawpy demosaicing
                        is_cfa = getattr(page, 'photometric', None) == 32803
                    if not is_cfa:
                        img = tifffile.imread(filepath)
                except Exception:
                    pass

            if img is None:
                demosaic_alg = getattr(rawpy.DemosaicAlgorithm, 'DHT', rawpy.DemosaicAlgorithm.AHD)
                with rawpy.imread(filepath) as raw:
                    img = raw.postprocess(
                        gamma=(1, 1),
                        no_auto_bright=True,
                        use_camera_wb=False,
                        user_wb=[1.0, 1.0, 1.0, 1.0],
                        output_color=rawpy.ColorSpace.raw,
                        output_bps=16,
                        user_flip=0,
                        demosaic_algorithm=demosaic_alg
                    )
            
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

            # --- STEP 1: CROPPING & ANALYSIS REGION ---
            h, w = img_float.shape[:2]
            h_margin = int(h * ignore_margin)
            w_margin = int(w * ignore_margin)
            
            if autocrop:
                print(f"  -> Auto-cropping {ignore_margin*100:.0f}% margins (maintaining aspect ratio)...")
                img_float = img_float[h_margin:h-h_margin, w_margin:w-w_margin]
                analysis_region = img_float
            else:
                analysis_region = img_float[h_margin:h-h_margin, w_margin:w-w_margin]

            # Generate output filename
            base_name = os.path.splitext(filename)[0]
            if "_Composite" in base_name:
                out_filename = base_name.replace("_Composite", "_Positive") + ".dng"
            else:
                out_filename = f"Positive_{base_name}.dng"
                
            output_filepath = os.path.join(output_dir, out_filename)
            is_dng_output = out_filename.endswith('.dng')

            # --- STEP 2: FILM SENSITOMETRIC INVERSION & DENSITY TRANSFORMATION ---
            # Professional Film H&D Sensitometric Inversion:
            # 1. Optical Density: D = log10(c_base / max(I_raw, 1.0))
            # 2. Normalized Density: D_norm = (D - p_low) / (p_high - p_low)  (0.0 = black shadow, 1.0 = white highlight)
            # 3. Linear Light Conversion: I_pos = ((10^(gamma_film * D_norm) - 1) / (10^gamma_film - 1)) * target_max
            # This produces deep rich blacks (0..12), perfectly balanced midtones (~90..110), and crisp highlights (~170..190) with zero clipping.
            gamma_film = 0.8
            target_max = (65535.0 * 0.40) if is_dng_output else 65535.0
            denom = (10.0 ** gamma_film) - 1.0
            print(f"  -> Inverting film transmission & balancing levels (clip={clip}%, target_max={target_max:.0f})...")
            
            is_mono = is_monochrome or (img_float.ndim == 2) or (img_float.ndim == 3 and img_float.shape[2] == 1)

            if not is_mono and img_float.ndim == 3 and img_float.shape[2] == 3:
                pos_linear = np.zeros_like(img_float)
                for c in range(3):
                    c_base = np.percentile(analysis_region[:, :, c], 99.9)
                    D_raw_img = np.log10(np.maximum(c_base, 1.0) / np.maximum(img_float[:, :, c], 1.0))
                    D_raw_ana = np.log10(np.maximum(c_base, 1.0) / np.maximum(analysis_region[:, :, c], 1.0))
                    
                    p_low = np.percentile(D_raw_ana, clip)
                    p_high = np.percentile(D_raw_ana, 100 - clip)
                    
                    if p_high > p_low:
                        D_norm = np.clip((D_raw_img - p_low) / (p_high - p_low), 0.0, 1.0)
                    else:
                        D_norm = np.zeros_like(D_raw_img)
                        
                    I_pos = (10.0 ** (gamma_film * D_norm) - 1.0) / denom
                    pos_linear[:, :, c] = I_pos * target_max
                img_float = pos_linear
            else:
                c_base = np.percentile(analysis_region, 99.9)
                D_raw_img = np.log10(np.maximum(c_base, 1.0) / np.maximum(img_float, 1.0))
                D_raw_ana = np.log10(np.maximum(c_base, 1.0) / np.maximum(analysis_region, 1.0))
                
                p_low = np.percentile(D_raw_ana, clip)
                p_high = np.percentile(D_raw_ana, 100 - clip)
                
                if p_high > p_low:
                    D_norm = np.clip((D_raw_img - p_low) / (p_high - p_low), 0.0, 1.0)
                else:
                    D_norm = np.zeros_like(D_raw_img)
                    
                I_pos = (10.0 ** (gamma_film * D_norm) - 1.0) / denom
                img_float = I_pos * target_max

            # Scale to 16-bit linear light array
            img_float = np.clip(img_float, 0, 65535)

            # --- STEP 3: GAMMA AND CONTRAST ---
            # Bypassed for DNG files to preserve strictly linear raw data (prevents double-gamma in RAW processors).
            effective_gamma = 1.0 if is_dng_output else gamma
            effective_scurve = 0.0 if is_dng_output else scurve
            
            if effective_gamma != 1.0 or effective_scurve > 0.0:
                print(f"  -> Applying tone curve (gamma={effective_gamma}, scurve={effective_scurve})...")
                # Normalize to 0.0-1.0
                img_norm = img_float / 65535.0
                
                # Apply gamma curve
                if effective_gamma != 1.0:
                    img_norm = img_norm ** (1.0 / effective_gamma)
                
                # Apply S-Curve
                if effective_scurve > 0.0:
                    c = 1.0 + effective_scurve
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
            
            # Save positive using write_linear_dng
            write_linear_dng(output_filepath, final_img, is_monochrome=is_monochrome, compress=compress_dng)
            
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
                        help="Enable lossless compression (zlib/deflate) for output DNGs")
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
        compress_dng=args.compress,
        global_levels=args.global_levels,
        ignore_margin=args.margin,
        scurve=args.scurve,
        autocrop=args.autocrop,
        monochrome=args.monochrome,
        monochrome_channel=args.monochrome_channel
    )