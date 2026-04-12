import os
import glob
from pathlib import Path
import rawpy
import numpy as np
import tifffile
import argparse

def process_roll(directory_path, output_dir=None, neutralize_base=False, compress_tiff=False):
    """
    Scans a directory for CR3 and RAF files, groups them by 3 based on filename, 
    auto-detects the color of each shot, and creates linear composited TIFFs.
    """
    # 1. Gather all supported RAW files (case-insensitive)
    supported_exts = {'.cr3', '.raf'}
    raw_files = [
        os.path.join(directory_path, f) for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f)) and os.path.splitext(f)[1].lower() in supported_exts
    ]
    
    if not raw_files:
        print(f"No .cr3 or .raf files found in {directory_path}")
        return

    # 2. Sort files alphabetically by filename 
    raw_files.sort()
    
    total_files = len(raw_files)
    print(f"Found {total_files} RAW files.")
    
    if total_files % 3 != 0:
        print("WARNING: The number of files is not divisible by 3.")
        print("Please ensure there are exactly 3 shots (R, G, B) per frame.")
        print("The script will process as many complete groups of 3 as possible.\n")

    # 3. Setup output directory
    if output_dir is None:
        output_dir = os.path.join(directory_path, "Composites")
    os.makedirs(output_dir, exist_ok=True)

    # 4. Process in groups of 3
    frame_number = 1
    for i in range(0, total_files - 2, 3):
        group = raw_files[i:i+3]
        print(f"Frame {frame_number:02d}:")
        
        try:
            channels_data = {'red': None, 'green': None, 'blue': None}
            
            for filepath in group:
                print(f"    Analyzing {Path(filepath).name}...")
                with rawpy.imread(filepath) as raw:
                    # Process the RAW file into a linear 16-bit RGB image
                    # KEY FIXES: 
                    # 1. output_color=rawpy.ColorSpace.raw prevents the sRGB matrix from scrambling LEDs.
                    # 2. user_flip=0 ignores camera rotation metadata, preventing stacking errors 
                    #    if the copy stand gravity sensor gets confused.
                    linear_rgb = raw.postprocess(
                        gamma=(1, 1),
                        no_auto_bright=True,
                        use_camera_wb=False,
                        user_wb=[1.0, 1.0, 1.0, 1.0], 
                        output_color=rawpy.ColorSpace.raw,
                        output_bps=16,
                        user_flip=0
                    )
                    
                    # Auto-detect which light was used by finding the brightest channel
                    means = [
                        np.mean(linear_rgb[:, :, 0]), # Red channel average
                        np.mean(linear_rgb[:, :, 1]), # Green channel average
                        np.mean(linear_rgb[:, :, 2])  # Blue channel average
                    ]
                    
                    dominant_idx = int(np.argmax(means))
                    
                    if dominant_idx == 0:
                        channels_data['red'] = linear_rgb[:, :, 0]
                        print("      -> Detected as RED light shot")
                    elif dominant_idx == 1:
                        channels_data['green'] = linear_rgb[:, :, 1]
                        print("      -> Detected as GREEN light shot")
                    elif dominant_idx == 2:
                        channels_data['blue'] = linear_rgb[:, :, 2]
                        print("      -> Detected as BLUE light shot")
            
            # Ensure we found exactly one of each color in this group of 3
            if any(v is None for v in channels_data.values()):
                raise ValueError("Could not detect a distinct Red, Green, and Blue shot in this group. Verify your shots.")
            
            # Stack the 3 grayscale channels into a single (H, W, 3) RGB array
            composite_rgb = np.stack((
                channels_data['red'], 
                channels_data['green'], 
                channels_data['blue']
            ), axis=-1)
            
            # Print average brightness to help debug exposure settings
            r_mean = np.mean(channels_data['red'])
            g_mean = np.mean(channels_data['green'])
            b_mean = np.mean(channels_data['blue'])
            print(f"  -> Channel Data: R={r_mean:.0f}, G={g_mean:.0f}, B={b_mean:.0f}")
            
            if neutralize_base:
                print("  -> Neutralizing film base color cast...")
                # Convert to float for math
                composite_float = composite_rgb.astype(np.float32)
                
                # Find the 99.9th percentile to represent the unexposed film base (ignoring hot pixels)
                r_base = np.percentile(composite_float[:,:,0], 99.9)
                g_base = np.percentile(composite_float[:,:,1], 99.9)
                b_base = np.percentile(composite_float[:,:,2], 99.9)
                
                # Scale all channels so the film base is white/neutral (65535)
                composite_float[:,:,0] = np.clip((composite_float[:,:,0] / r_base) * 65535.0, 0, 65535)
                composite_float[:,:,1] = np.clip((composite_float[:,:,1] / g_base) * 65535.0, 0, 65535)
                composite_float[:,:,2] = np.clip((composite_float[:,:,2] / b_base) * 65535.0, 0, 65535)
                
                composite_rgb = composite_float.astype(np.uint16)
            
            # Save as 16-bit TIFF
            output_filename = f"Frame_{frame_number:02d}_Composite.tiff"
            output_filepath = os.path.join(output_dir, output_filename)
            
            # Enforce contiguous array to prevent TIFF reading errors in some viewers
            composite_rgb = np.ascontiguousarray(composite_rgb)
            
            # Setup compression
            tiff_compression = 'zlib' if compress_tiff else None
            tifffile.imwrite(output_filepath, composite_rgb, photometric='rgb', compression=tiff_compression)
            
            print(f"  -> Saved composite to: {output_filename}\n")
            
        except Exception as e:
            print(f"  -> ERROR processing Frame {frame_number:02d}: {e}\n")
            
        frame_number += 1
        
    print("Roll processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tri-Color Auto Compositor for RAW Film Scans")
    
    # Define command line arguments
    parser.add_argument("-i", "--input", type=str, required=True, 
                        help="Path to the directory containing RAW files (.CR3 or .RAF)")
    parser.add_argument("-c", "--compress", action="store_true", 
                        help="Enable lossless compression (zlib/deflate) for output TIFFs")
    parser.add_argument("-n", "--neutralize", action="store_true", 
                        help="Automatically balance the color channels to neutralize the film base")
    
    args = parser.parse_args()
    
    # The output directory will automatically be created as a subfolder inside the input path
    process_roll(args.input, neutralize_base=args.neutralize, compress_tiff=args.compress)