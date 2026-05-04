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
    root_input = input("Enter the root directory for your scans (e.g., ~/Pictures/Scans): ").strip()
    
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
    
    stock = input("Film Stock (e.g., KodakGold200): ").strip()
    fmt = input("Format (e.g., 135, 120): ").strip() 
    roll = input("Roll Number (e.g., 02): ").strip().zfill(2)
    
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
    return dirs

def get_next_frame_number(composites_dir):
    """Figures out the next available frame number by looking at existing composite files."""
    existing = glob.glob(os.path.join(composites_dir, "Frame_*_Composite.tiff"))
    max_num = 0
    for f in existing:
        try:
            num = int(os.path.basename(f).split('_')[1])
            if num > max_num: max_num = num
        except ValueError:
            pass
    return max_num + 1

def run_pipeline(dirs):
    """Watches the negatives folder for new RAW files and processes them in groups of 3."""
    print(f"\n🔥 HOT FOLDER PIPELINE ACTIVE 🔥")
    print(f"Monitoring: {dirs['negatives']}")
    print(f"Waiting for RGB RAW triplets. Press Ctrl+C to exit.\n")
    
    supported_exts = {'.cr3', '.raf'}
    frame_number = get_next_frame_number(dirs['negatives'])
    
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
                
                composite_filename = f"Frame_{frame_number:02d}_Composite.tiff"
                composite_filepath = os.path.join(dirs['negatives'], composite_filename)
                
                try:
                    # 1. Combine the 3 exposures
                    # We skip neutralizing the base here; the inverter handles the orange mask better.
                    process_triplet(
                        group=group, 
                        output_filepath=composite_filepath, 
                        neutralize_base=False, 
                        compress_tiff=True
                    )
                    
                    # 2. Invert to positive and adjust colors
                    # Using per-channel levels (global_levels=False) naturally removes the orange mask.
                    process_positives(
                        input_path=composite_filepath,
                        output_dir=dirs['positives'],
                        clip=0.1,
                        gamma=2.2,
                        compress_tiff=True,
                        global_levels=False, 
                        ignore_margin=0.03,
                        scurve=0.2,          
                        autocrop=True
                    )
                    
                    # 3. Move the original RAWs out of the hot folder
                    for f in group:
                        shutil.move(f, os.path.join(dirs['processed'], os.path.basename(f)))
                    
                    print(f"✅ SUCCESS: Frame {frame_number:02d} completed and moved to Positives.")
                    frame_number += 1
                    
                except Exception as e:
                    print(f"❌ ERROR PROCESSING FRAME {frame_number:02d}: {e}")
                    for f in group:
                        shutil.move(f, os.path.join(dirs['errors'], os.path.basename(f)))
                        
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\nExiting scanning session.")
            break

if __name__ == "__main__":
    try:
        session_dirs = setup_session()
        run_pipeline(session_dirs)
    except KeyboardInterrupt:
        print("\nSession setup cancelled.")