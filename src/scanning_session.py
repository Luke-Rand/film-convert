import os
import time
import shutil
import glob
from pathlib import Path
import sys

from compositor import process_triplet
from inverter import process_positives

def setup_session():
    """Interactively gather session details and create necessary folders."""
    print("\n" + "="*50)
    print("🎞️  NEW FILM SCANNING SESSION")
    print("="*50)
    
    # Get the base path for scans
    root_input = input("Enter the root directory for your scans (e.g., ~/Pictures/Scans): ").strip().strip("'\"")
    
    # Resolve ~ and make the path absolute
    root_folder = os.path.abspath(os.path.expanduser(root_input))
    
    # Create the root directory if it's missing
    if not os.path.exists(root_folder):
        create = input(f"Directory '{root_folder}' does not exist. Create it? (y/n): ").strip().lower()
        if create == 'y':
            os.makedirs(root_folder, exist_ok=True)
        else:
            print("Exiting. Please run the script again and provide a valid directory.")
            sys.exit(1)
            
    print(f"-> Working directory set to: {root_folder}\n")
    
    mode_choice = ""
    while mode_choice not in ['1', '2']:
        mode_choice = input("Select scanning mode:\n  1. Triplet (3x RAW for RGB)\n  2. Single-shot (DNG/TIFF negatives)\nChoice: ").strip()
    
    mode = 'triplet' if mode_choice == '1' else 'single'
    print() # Add a newline for spacing
    
    stock = input("Film Stock (e.g., KodakGold200): ").strip()
    fmt = input("Format (e.g., 135, 120): ").strip() 
    roll = input("Roll Number (e.g., 02): ").strip().zfill(2)
    
    is_monochrome = input("Is this a Black & White film session? (y/n) [default: n]: ").strip().lower() == 'y'
    monochrome_channel = "luminance"
    if is_monochrome:
        monochrome_channel = input("Choose monochrome conversion channel (luminance, average, red, green, blue) [default: luminance]: ").strip().lower()
        if monochrome_channel not in ["luminance", "average", "red", "green", "blue"]:
            monochrome_channel = "luminance"
            
    folder_name = f"{stock}-{fmt}-{roll}"
    session_dir = os.path.join(root_folder, folder_name)
    
    # Set up paths for the various stages of processing
    dirs = {
        "negatives": os.path.join(session_dir, "negatives"),
        "positives": os.path.join(session_dir, "positives"),
        "processed": os.path.join(session_dir, "processed_raws"),
        "errors": os.path.join(session_dir, "error_raws")
    }
    
    # Make sure all directories exist
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        
    print(f"\n✅ Session initialized at: {session_dir}")
    return dirs, mode, is_monochrome, monochrome_channel

def get_next_frame_number(dirs):
    """Figures out the next available frame number by looking at existing files in negatives, processed, and positives."""
    search_dirs = []
    for key in ['negatives', 'processed', 'positives']:
        d = dirs.get(key)
        if d and os.path.exists(d):
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

def run_triplet_pipeline(dirs, is_mono, mono_chan):
    """Watches for RAW triplets, composites them, and inverts them."""
    print(f"\n🔥 TRIPLET PIPELINE ACTIVE 🔥")
    print(f"Monitoring: {dirs['negatives']}")
    print(f"Waiting for RGB RAW triplets. Press Ctrl+C to exit.\n")
    
    supported_exts = {'.cr3', '.raf', '.nef'}
    frame_number = get_next_frame_number(dirs)
    
    while True:
        try:
            # Grab all supported RAW files in the negatives directory, sorted by age
            raw_files = [
                os.path.join(dirs['negatives'], f) for f in os.listdir(dirs['negatives'])
                if os.path.isfile(os.path.join(dirs['negatives'], f)) and os.path.splitext(f)[1].lower() in supported_exts
            ]
            raw_files.sort(key=lambda x: os.path.getmtime(x))
            
            if len(raw_files) >= 3:
                group = raw_files[:3]
                
                # Give the camera/OS a couple seconds to finish saving the newest file
                if time.time() - os.path.getmtime(group[-1]) < 2:
                    time.sleep(1)
                    continue
                    
                print(f"{'-'*50}\n📸 Triplet detected! Processing Frame {frame_number:02d}...")
                
                composite_filename = f"Frame_{frame_number:02d}_Composite.dng"
                composite_filepath = os.path.join(dirs['negatives'], composite_filename)
                
                try:
                    # 1. Combine the 3 exposures (don't neutralize, inverter handles orange mask)
                    process_triplet(
                        group=group, 
                        output_filepath=composite_filepath, 
                        neutralize_base=False, 
                        compress_dng=True
                    )
                    
                    # 2. Invert to positive (per-channel levels remove orange mask)
                    process_positives(
                        input_path=composite_filepath,
                        output_dir=dirs['positives'],
                        clip=0.1,
                        gamma=2.2,
                        compress_dng=True,
                        global_levels=False,
                        ignore_margin=0.03,
                        scurve=0.0,
                        autocrop=False,
                        monochrome=is_mono,
                        monochrome_channel=mono_chan
                    )
                    
                    # 3. Move original RAWs and intermediate composite out of the hot folder
                    for f in group:
                        shutil.move(f, os.path.join(dirs['processed'], os.path.basename(f)))
                    shutil.move(composite_filepath, os.path.join(dirs['processed'], composite_filename))
                    
                    print(f"✅ SUCCESS: Frame {frame_number:02d} completed and moved to Positives.")
                    frame_number += 1
                    
                except Exception as e:
                    print(f"❌ ERROR PROCESSING FRAME {frame_number:02d}: {e}")
                    # Move failed RAWs to the error folder
                    for f in group:
                        shutil.move(f, os.path.join(dirs['errors'], os.path.basename(f)))
                    # Also move the composite if it was created before the error
                    if os.path.exists(composite_filepath):
                        shutil.move(composite_filepath, os.path.join(dirs['errors'], composite_filename))
                        
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\nExiting scanning session.")
            break

def run_single_shot_pipeline(dirs, is_mono, mono_chan):
    """Watches for single DNG/TIFF negatives and inverts them."""
    print(f"\n🔥 SINGLE-SHOT PIPELINE ACTIVE 🔥")
    print(f"Monitoring: {dirs['negatives']}")
    print(f"Waiting for single DNG/TIFF negatives. Press Ctrl+C to exit.\n")
    
    supported_exts = {'.dng', '.tiff', '.tif'}
    
    while True:
        try:
            # Grab all supported files in the negatives directory, sorted by age
            neg_files = [
                os.path.join(dirs['negatives'], f) for f in os.listdir(dirs['negatives'])
                if os.path.isfile(os.path.join(dirs['negatives'], f)) and os.path.splitext(f)[1].lower() in supported_exts
            ]
            neg_files.sort(key=lambda x: os.path.getmtime(x))
            
            if neg_files:
                filepath = neg_files[0]
                filename = os.path.basename(filepath)
                
                # Give the OS a couple seconds to finish saving the file
                if time.time() - os.path.getmtime(filepath) < 2:
                    time.sleep(1)
                    continue
                    
                print(f"{'-'*50}\n🎞️  Negative detected! Processing {filename}...")
                
                try:
                    # Invert to positive and adjust colors
                    process_positives(
                        input_path=filepath,
                        output_dir=dirs['positives'],
                        clip=0.1,
                        gamma=2.2,
                        compress_dng=True,
                        global_levels=False, 
                        ignore_margin=0.03,
                        scurve=0.0,          
                        autocrop=False,
                        monochrome=is_mono,
                        monochrome_channel=mono_chan
                    )
                    
                    # Move the original negative out of the hot folder
                    shutil.move(filepath, os.path.join(dirs['processed'], filename))
                    
                    print(f"✅ SUCCESS: {filename} processed and positive saved.")
                    
                except Exception as e:
                    print(f"❌ ERROR PROCESSING {filename}: {e}")
                    shutil.move(filepath, os.path.join(dirs['errors'], filename))
                        
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\nExiting scanning session.")
            break

def run_pipeline(dirs, mode, is_mono, mono_chan):
    """Dispatches to the correct pipeline based on user's choice."""
    if mode == 'triplet':
        run_triplet_pipeline(dirs, is_mono, mono_chan)
    elif mode == 'single':
        run_single_shot_pipeline(dirs, is_mono, mono_chan)

if __name__ == "__main__":
    try:
        session_dirs, mode, is_mono, mono_chan = setup_session()
        run_pipeline(session_dirs, mode, is_mono, mono_chan)
    except KeyboardInterrupt:
        print("\nSession setup cancelled.")