import os
import glob
from pathlib import Path
import rawpy
import numpy as np
import tifffile
import argparse
import time
import shutil
import sys

def align_channel(ref, mov, channel_name=""):
    """Aligns moving channel to reference channel using FFT phase correlation."""
    h, w = ref.shape
    size = 1024
    y_start = max(0, h // 2 - size // 2)
    y_end = min(h, h // 2 + size // 2)
    x_start = max(0, w // 2 - size // 2)
    x_end = min(w, w // 2 + size // 2)
    
    ref_crop = ref[y_start:y_end, x_start:x_end].astype(np.float32)
    mov_crop = mov[y_start:y_end, x_start:x_end].astype(np.float32)
    
    # Compute FFT phase correlation
    F = np.fft.fft2(ref_crop)
    G = np.fft.fft2(mov_crop)
    cross_power = F * np.conjugate(G)
    R = cross_power / (np.abs(cross_power) + 1e-8)
    r = np.fft.ifft2(R)
    
    peak = np.unravel_index(np.argmax(np.abs(r)), r.shape)
    dy, dx = peak
    if dy > r.shape[0] // 2:
        dy -= r.shape[0]
    if dx > r.shape[1] // 2:
        dx -= r.shape[1]
        
    print(f"    - [{channel_name}] Detected shift offset: dy={dy}, dx={dx}")
    
    # Safety threshold to avoid alignment distortion if files are mismatching
    if abs(dy) > 20 or abs(dx) > 20:
        print(f"      -> WARNING: Detected shift too large ({dy}, {dx}). Skipping alignment.")
        return mov
        
    if dy == 0 and dx == 0:
        return mov
        
    # Apply corrective shift (opposite of detected shift)
    shifted = np.zeros_like(mov)
    
    src_y_start = max(0, dy)
    src_y_end = min(h, h + dy)
    src_x_start = max(0, dx)
    src_x_end = min(w, w + dx)
    
    dst_y_start = max(0, -dy)
    dst_y_end = min(h, h - dy)
    dst_x_start = max(0, -dx)
    dst_x_end = min(w, w - dx)
    
    shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = mov[src_y_start:src_y_end, src_x_start:src_x_end]
    return shifted

def process_triplet(group, output_filepath, neutralize_base, compress_tiff, align_channels=False):
    """Processes exactly 3 RAW files into a single composite."""
    channels_data = {'red': None, 'green': None, 'blue': None}
    
    for filepath in group:
        print(f"    Analyzing {Path(filepath).name}...")
        with rawpy.imread(filepath) as raw:
            # Decode RAW to linear 16-bit RGB
            # Important settings:
            # - output_color=rawpy.ColorSpace.raw prevents sRGB color matrix interference.
            # - user_flip=0 ignores camera rotation metadata to avoid stacking errors.
            linear_rgb = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                use_camera_wb=False,
                user_wb=[1.0, 1.0, 1.0, 1.0], 
                output_color=rawpy.ColorSpace.raw,
                output_bps=16,
                user_flip=0
            )
            
            # Determine light source by finding the brightest channel
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
    
    # Check if we have one of each color (Red, Green, Blue)
    if any(v is None for v in channels_data.values()):
        raise ValueError("Could not detect a distinct Red, Green, and Blue shot in this group. Verify your shots.")
    
    if align_channels:
        print("  -> Aligning channels to Green reference channel...")
        channels_data['red'] = align_channel(channels_data['green'], channels_data['red'], "Red")
        channels_data['blue'] = align_channel(channels_data['green'], channels_data['blue'], "Blue")
        
    # Combine channels into an RGB image
    composite_rgb = np.stack((
        channels_data['red'], 
        channels_data['green'], 
        channels_data['blue']
    ), axis=-1)
    
    # Print channel averages for exposure debugging
    r_mean = np.mean(channels_data['red'])
    g_mean = np.mean(channels_data['green'])
    b_mean = np.mean(channels_data['blue'])
    print(f"  -> Channel Data: R={r_mean:.0f}, G={g_mean:.0f}, B={b_mean:.0f}")
    
    if neutralize_base:
        print("  -> Neutralizing film base color cast...")
        # Convert to float32
        composite_float = composite_rgb.astype(np.float32)
        
        # Use 99.9th percentile to estimate unexposed film base, ignoring hot pixels
        r_base = np.percentile(composite_float[:,:,0], 99.9)
        g_base = np.percentile(composite_float[:,:,1], 99.9)
        b_base = np.percentile(composite_float[:,:,2], 99.9)
        
        # Scale channels to neutralize film base (white = 65535)
        composite_float[:,:,0] = np.clip((composite_float[:,:,0] / r_base) * 65535.0, 0, 65535)
        composite_float[:,:,1] = np.clip((composite_float[:,:,1] / g_base) * 65535.0, 0, 65535)
        composite_float[:,:,2] = np.clip((composite_float[:,:,2] / b_base) * 65535.0, 0, 65535)
        
        composite_rgb = composite_float.astype(np.uint16)
    
    # Make array contiguous to avoid TIFF reader issues
    composite_rgb = np.ascontiguousarray(composite_rgb)
    
    # Set up compression
    tiff_compression = 'zlib' if compress_tiff else None
    tifffile.imwrite(output_filepath, composite_rgb, photometric='rgb', compression=tiff_compression)
    
    print(f"  -> Saved composite to: {os.path.basename(output_filepath)}\n")

def get_next_frame_number(directory):
    """Finds the next frame number based on existing files."""
    search_dirs = [directory]
    for sub in ["Processed_RAWs", "Composites", "Positives", "positives", "processed_raws"]:
        d = os.path.join(directory, sub)
        if os.path.exists(d):
            search_dirs.append(d)
            
    max_num = 0
    for d in search_dirs:
        for entry in os.listdir(d):
            if entry.startswith("Frame_"):
                try:
                    parts = entry.split('_')
                    if len(parts) > 1:
                        num = int(parts[1])
                        if num > max_num:
                            max_num = num
                except (ValueError, IndexError):
                    pass
    return max_num + 1

def hot_folder_mode(directory_path, neutralize_base=False, compress_tiff=False, timeout=60, align_channels=False):
    """Monitors a directory for RAW triplets and processes them."""
    print(f"\n{'='*60}")
    print(f"🔥 HOT FOLDER MODE ACTIVE 🔥")
    print(f"Monitoring: {directory_path}")
    print(f"Waiting for RAW triplets. Press Ctrl+C to exit.")
    print(f"{'='*60}\n")
    
    processed_dir = os.path.join(directory_path, "Processed_RAWs")
    error_dir = os.path.join(directory_path, "Error_RAWs")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)
    
    supported_exts = {'.cr3', '.raf'}
    frame_number = get_next_frame_number(directory_path)
    
    while True:
        try:
            raw_files = [
                os.path.join(directory_path, f) for f in os.listdir(directory_path)
                if os.path.isfile(os.path.join(directory_path, f)) and os.path.splitext(f)[1].lower() in supported_exts
            ]
            
            # Sort by modification time, oldest first
            raw_files.sort(key=lambda x: os.path.getmtime(x))
            
            if len(raw_files) >= 3:
                group = raw_files[:3]
                
                # Ensure the newest file is completely written (wait 2 seconds)
                if time.time() - os.path.getmtime(group[-1]) < 2:
                    time.sleep(1)
                    continue
                    
                print(f"\n{'-'*60}")
                print(f"📸 Triplet detected! Processing Frame {frame_number:02d}...")
                output_filename = f"Frame_{frame_number:02d}_Composite.tiff"
                # Save composites directly in the watched directory
                output_filepath = os.path.join(directory_path, output_filename)
                
                try:
                    process_triplet(group, output_filepath, neutralize_base, compress_tiff, align_channels)
                    
                    # Move original RAWs to processed folder
                    for f in group:
                        shutil.move(f, os.path.join(processed_dir, os.path.basename(f)))
                    print(f"\n{'*'*60}")
                    print(f"✅ SUCCESS: Frame {frame_number:02d} processed and saved.")
                    print(f"Moved original RAWs to {processed_dir}")
                    print(f"Waiting for next triplet...")
                    print(f"{'*'*60}\n")
                    frame_number += 1
                    
                except Exception as e:
                    print(f"\n{'!'*60}")
                    print(f"❌ ERROR PROCESSING TRIPLET: {e}")
                    print(f"Moving problematic files to Error_RAWs folder.")
                    print(f"{'!'*60}\n")
                    # Move failed files to avoid infinite loops
                    for f in group:
                        shutil.move(f, os.path.join(error_dir, os.path.basename(f)))
            
            elif 0 < len(raw_files) < 3:
                # Check timeout
                oldest_time = os.path.getmtime(raw_files[0])
                elapsed = time.time() - oldest_time
                if elapsed > timeout:
                    print(f"\n{'?'*60}")
                    print(f"⚠️ TIMEOUT ANOMALY: {int(elapsed)} seconds have passed!")
                    print(f"Found {len(raw_files)} file(s), but waiting for a full 3 to complete the triplet.")
                    print(f"Please check your camera or the hot folder!")
                    print(f"{'?'*60}\n")
                    time.sleep(10) # Snooze for 10s to avoid spamming
                    
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\nExiting Hot Folder Mode.")
            sys.exit(0)

def process_roll(directory_path, output_dir=None, neutralize_base=False, compress_tiff=False, align_channels=False):
    """
    Scans for RAW files, groups by 3, auto-detects colors, and creates linear TIFFs.
    """
    # Find all supported RAW files
    supported_exts = {'.cr3', '.raf'}
    raw_files = [
        os.path.join(directory_path, f) for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f)) and os.path.splitext(f)[1].lower() in supported_exts
    ]
    
    if not raw_files:
        print(f"No .cr3 or .raf files found in {directory_path}")
        return

    # Sort files alphabetically
    raw_files.sort()
    
    total_files = len(raw_files)
    print(f"Found {total_files} RAW files.")
    
    if total_files % 3 != 0:
        print("WARNING: The number of files is not divisible by 3.")
        print("Please ensure there are exactly 3 shots (R, G, B) per frame.")
        print("The script will process as many complete groups of 3 as possible.\n")

    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(directory_path, "Composites")
    os.makedirs(output_dir, exist_ok=True)

    # Process files in groups of 3
    frame_number = 1
    for i in range(0, total_files - 2, 3):
        group = raw_files[i:i+3]
        print(f"Frame {frame_number:02d}:")
        
        output_filename = f"Frame_{frame_number:02d}_Composite.tiff"
        output_filepath = os.path.join(output_dir, output_filename)
        
        try:
            process_triplet(group, output_filepath, neutralize_base, compress_tiff, align_channels)
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
    parser.add_argument("--hotfolder", action="store_true", 
                        help="Run in Hot Folder mode: monitor the directory, composite automatically, and move originals.")
    parser.add_argument("-t", "--timeout", type=int, default=60, 
                        help="Timeout in seconds to wait for a 3rd image in hot folder mode (default: 60)")
    
    parser.add_argument("-a", "--align", action="store_true", 
                        help="Auto-correct exposure alignment between channels (R, G, B) using FFT phase correlation")
    
    args = parser.parse_args()
    
    if args.hotfolder:
        hot_folder_mode(args.input, neutralize_base=args.neutralize, compress_tiff=args.compress, timeout=args.timeout, align_channels=args.align)
    else:
        # The output directory will automatically be created as a subfolder inside the input path
        process_roll(args.input, neutralize_base=args.neutralize, compress_tiff=args.compress, align_channels=args.align)