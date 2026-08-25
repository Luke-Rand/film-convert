#!/usr/bin/env python3
import os
import sys
import re
import time
import argparse
from pathlib import Path

# Ensure src is on python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from compositor import process_triplet
from inverter import process_positives

SUPPORTED_EXTS = {'.cr3', '.raf', '.nef', '.arw', '.rw2', '.nrw', '.dcr'}

def extract_frame_num(filename):
    """Extracts integer frame number from filenames like Frame_01_Capture..."""
    match = re.search(r'Frame_(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match_any = re.search(r'(\d+)', filename)
    return int(match_any.group(1)) if match_any else 0

def find_raw_source_dir(roll_dir):
    """
    Finds where RAW files are located within a roll directory.
    Checks processed_raws, negatives, or the directory itself.
    """
    for sub in ['processed_raws', 'negatives', 'Processed_RAWs', 'Negatives', 'RAWs', 'raws']:
        p = os.path.join(roll_dir, sub)
        if os.path.isdir(p):
            raws = [f for f in os.listdir(p) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]
            if raws:
                return p
                
    # Check root of roll_dir
    raws = [f for f in os.listdir(roll_dir) if os.path.isfile(os.path.join(roll_dir, f)) and os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]
    if raws:
        return roll_dir
        
    return None

def is_roll_dir(path):
    """Checks if a directory is a roll (contains RAW files or a processed_raws/negatives subfolder with RAWs)."""
    return find_raw_source_dir(path) is not None

def discover_rolls(target_path):
    """
    Discovers all roll directories to process from target_path.
    Can be a single roll folder or a parent directory containing multiple rolls.
    """
    target_path = os.path.abspath(os.path.expanduser(target_path))
    if not os.path.exists(target_path):
        print(f"Error: Path does not exist: {target_path}")
        return []

    # If the target path itself is a roll directory
    if is_roll_dir(target_path):
        return [target_path]

    # Search subdirectories (1 level deep)
    discovered = []
    for entry in sorted(os.listdir(target_path)):
        sub_path = os.path.join(target_path, entry)
        if os.path.isdir(sub_path) and not entry.startswith('.'):
            if is_roll_dir(sub_path):
                discovered.append(sub_path)
            else:
                # Check 2 levels deep for nested directory structures (e.g. Year/Month/Roll)
                for sub2 in sorted(os.listdir(sub_path)):
                    sub2_path = os.path.join(sub_path, sub2)
                    if os.path.isdir(sub2_path) and not sub2.startswith('.') and is_roll_dir(sub2_path):
                        discovered.append(sub2_path)

    return discovered

def reprocess_roll(roll_dir, args):
    roll_name = os.path.basename(roll_dir)
    print(f"\n{'='*70}")
    print(f"🎬 RE-PROCESSING ROLL: {roll_name}")
    print(f"   Path: {roll_dir}")
    print(f"{'='*70}\n")
    
    raw_source_dir = find_raw_source_dir(roll_dir)
    if not raw_source_dir:
        print(f"  -> No RAW files found in {roll_dir}. Skipping.\n")
        return 0

    positives_dir = os.path.join(roll_dir, 'positives')
    os.makedirs(positives_dir, exist_ok=True)
    
    # Collect all RAW files
    raw_files = [f for f in os.listdir(raw_source_dir) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]
    raw_files.sort(key=extract_frame_num)
    
    total_raws = len(raw_files)
    num_frames = total_raws // 3
    if num_frames == 0:
        print(f"  -> Found {total_raws} RAW files (need at least 3 for a triplet). Skipping.\n")
        return 0

    print(f"Found {total_raws} RAW files -> {num_frames} frames to process.\n")
    
    successful_frames = 0
    for i in range(0, total_raws - 2, 3):
        group_files = raw_files[i:i+3]
        frame_idx = (i // 3) + 1
        
        next_num = extract_frame_num(group_files[-1]) + 1
        comp_name = f"Frame_{next_num:02d}_Composite.dng"
        comp_path = os.path.join(raw_source_dir, comp_name)
        
        print(f"[{frame_idx:02d}/{num_frames:02d}] Processing Frame {next_num:02d} ({', '.join(group_files)})...")
        group_full_paths = [os.path.join(raw_source_dir, f) for f in group_files]
        
        start_t = time.time()
        try:
            # 1. Composite triplet with sub-pixel alignment & grain-safe demosaicing
            process_triplet(
                group=group_full_paths,
                output_filepath=comp_path,
                neutralize_base=args.neutralize,
                compress_tiff=args.compress,
                align_channels=not args.no_align
            )
            
            # 2. Sensitometric Inversion with smooth monotonic knee roll-off
            process_positives(
                input_path=comp_path,
                output_dir=positives_dir,
                clip=args.clip,
                gamma=args.gamma,
                compress_tiff=args.compress,
                global_levels=args.global_levels,
                ignore_margin=args.margin,
                scurve=args.scurve,
                autocrop=args.autocrop,
                monochrome=args.monochrome,
                monochrome_channel=args.monochrome_channel,
                reversal=args.reversal,
                convert_to_tiff=True
            )
            elapsed = time.time() - start_t
            print(f"  -> Successfully converted in {elapsed:.1f}s\n")
            successful_frames += 1
            
        except Exception as e:
            print(f"  -> ERROR processing Frame {next_num:02d}: {e}\n")

    return successful_frames

def main():
    parser = argparse.ArgumentParser(
        description="Batch re-process film scan roll directories with sub-pixel alignment and sensitometric inversion."
    )
    parser.add_argument(
        "directory", 
        type=str, 
        nargs="?",
        default="/Volumes/film-scans/2026/2026-08_DixonDR",
        help="Path to a single roll directory or parent folder containing rolls (default: /Volumes/film-scans/2026/2026-08_DixonDR)"
    )
    parser.add_argument("-g", "--gamma", type=float, default=2.2, help="Output gamma curve (default: 2.2, use 1.0 for linear)")
    parser.add_argument("-p", "--clip", type=float, default=0.1, help="Percentile clipping for black/white levels (default: 0.1%%)")
    parser.add_argument("-m", "--margin", type=float, default=0.03, help="Fraction of outer edge to ignore (default: 0.03 = 3%%)")
    parser.add_argument("-s", "--scurve", type=float, default=0.0, help="Contrast S-Curve strength (default: 0.0)")
    parser.add_argument("-a", "--autocrop", action="store_true", help="Auto-crop outer margins from final output")
    parser.add_argument("--global-levels", action="store_true", help="Preserve scene chromaticity with global exposure scaling")
    parser.add_argument("-n", "--neutralize", action="store_true", help="Neutralize film base during compositing")
    parser.add_argument("-c", "--compress", action="store_true", help="Enable zlib compression for output TIFFs")
    parser.add_argument("--no-align", action="store_true", help="Disable sub-pixel channel alignment")
    parser.add_argument("--monochrome", "--bw", action="store_true", help="Convert output to monochrome/B&W")
    parser.add_argument("--monochrome-channel", type=str, default="luminance", choices=["luminance", "average", "red", "green", "blue"])
    parser.add_argument("--reversal", action="store_true", help="Process positive slide / reversal film")
    
    args = parser.parse_args()
    
    rolls = discover_rolls(args.directory)
    if not rolls:
        print(f"No roll directories with RAW files found in: {args.directory}")
        sys.exit(1)
        
    print(f"\n{'='*70}")
    print(f"🔍 Discovered {len(rolls)} roll directory(s):")
    for r in rolls:
        print(f"   • {r}")
    print(f"{'='*70}\n")
    
    total_start = time.time()
    total_frames = 0
    for r in rolls:
        total_frames += reprocess_roll(r, args)
        
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"🎉 COMPLETED {total_frames} frames across {len(rolls)} roll(s) in {total_elapsed/60.0:.1f} minutes!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
