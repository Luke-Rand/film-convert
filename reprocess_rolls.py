import os
import sys
import re
import time

# Ensure src is on path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from compositor import process_triplet
from inverter import process_positives

def reprocess_roll(roll_dir):
    roll_name = os.path.basename(roll_dir)
    print(f"\n{'='*70}")
    print(f"🎬 RE-PROCESSING ROLL: {roll_name}")
    print(f"{'='*70}\n")
    
    processed_dir = os.path.join(roll_dir, 'processed_raws')
    positives_dir = os.path.join(roll_dir, 'positives')
    os.makedirs(positives_dir, exist_ok=True)
    
    # Collect all RAW files (.cr3, .raf, .nef)
    raw_files = [f for f in os.listdir(processed_dir) if f.lower().endswith(('.cr3', '.raf', '.nef'))]
    
    # Sort numerically by frame index in filename (e.g. Frame_01, Frame_02)
    def extract_frame_num(f):
        match = re.search(r'Frame_(\d+)', f, re.IGNORECASE)
        return int(match.group(1)) if match else 0

    raw_files.sort(key=extract_frame_num)
    
    total_raws = len(raw_files)
    num_frames = total_raws // 3
    print(f"Found {total_raws} RAW files -> {num_frames} frames to process.\n")
    
    for i in range(0, total_raws - 2, 3):
        group_files = raw_files[i:i+3]
        frame_idx = (i // 3) + 1
        
        # Determine composite file name (match original 4-based indexing e.g. Frame_04_Composite.dng or Frame_XX_Composite.tiff)
        # Check if existing composite filename in processed_raws corresponds to this frame
        next_num = extract_frame_num(group_files[-1]) + 1
        comp_name = f"Frame_{next_num:02d}_Composite.dng"
        comp_path = os.path.join(processed_dir, comp_name)
        
        print(f"[{frame_idx:02d}/{num_frames:02d}] Processing Frame {next_num:02d} ({', '.join(group_files)})...")
        group_full_paths = [os.path.join(processed_dir, f) for f in group_files]
        
        start_t = time.time()
        try:
            # 1. Composite triplet with sub-pixel alignment & grain-safe demosaicing
            process_triplet(
                group=group_full_paths,
                output_filepath=comp_path,
                neutralize_base=False,
                compress_tiff=False,
                align_channels=True
            )
            
            # 2. Sensitometric Inversion with monotonic knee roll-off
            process_positives(
                input_path=comp_path,
                output_dir=positives_dir,
                clip=0.1,
                gamma=2.2,
                compress_tiff=False,
                global_levels=False,
                ignore_margin=0.03,
                scurve=0.0,
                autocrop=False,
                monochrome=False,
                reversal=False,
                convert_to_tiff=True
            )
            elapsed = time.time() - start_t
            print(f"  -> Successfully converted in {elapsed:.1f}s\n")
            
        except Exception as e:
            print(f"  -> ERROR processing Frame {next_num:02d}: {e}\n")

if __name__ == "__main__":
    base_dir = "/Volumes/film-scans/2026/2026-08_DixonDR"
    rolls = ["KodakGold200-135-01", "KodakGold200-135-02", "KodakGold200-135-03"]
    
    total_start = time.time()
    for r in rolls:
        r_path = os.path.join(base_dir, r)
        if os.path.exists(r_path):
            reprocess_roll(r_path)
            
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"🎉 ALL 3 ROLLS COMPLETED in {total_elapsed/60.0:.1f} minutes!")
    print(f"{'='*70}\n")
