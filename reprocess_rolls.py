#!/usr/bin/env python3
import os
import sys
import re
import time
import argparse
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Ensure src is on python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from compositor import process_triplet
from inverter import process_positives
from contact_sheet import generate_contact_sheet
from batch_worker import worker_process_triplet_pipeline
from session_manifest import read_session_manifest, write_session_manifest

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

def _resolve_effective_config(roll_dir, args):
    """Merges session.json manifest config (if present) with any explicit CLI arguments."""
    manifest = read_session_manifest(roll_dir)
    manifest_cfg = (manifest.get("config") or {}) if manifest else {}
    explicit = getattr(args, "_explicit", set())

    def pick(arg_name, cfg_key, default_val):
        if arg_name in explicit:
            return getattr(args, arg_name, default_val)
        if cfg_key in manifest_cfg and manifest_cfg[cfg_key] is not None:
            return manifest_cfg[cfg_key]
        return getattr(args, arg_name, default_val)

    base_ratios = args.base_ratios if ("base_ratios" in explicit or args.base_ratios is not None) else manifest_cfg.get("base_ratios")
    neutralize = (args.neutralize if "neutralize" in explicit else manifest_cfg.get("neutralize", args.neutralize)) or (base_ratios is not None)
    compress = pick("compress", "compress_tiff", False)
    align_channels = (not args.no_align) if "no_align" in explicit else manifest_cfg.get("align_channels", not args.no_align)
    icc_profile = pick("icc_profile", "color_profile", "adobe_rgb")
    preserve_metadata = (not args.no_metadata) if "no_metadata" in explicit else manifest_cfg.get("embed_metadata", not args.no_metadata)
    clip = pick("clip", "clip", 0.1)
    gamma = pick("gamma", "gamma", 2.2)
    global_levels = pick("global_levels", "global_levels", True)
    margin = pick("margin", "margin", 0.03)
    scurve = pick("scurve", "scurve", 0.0)
    autocrop = pick("autocrop", "autocrop", False)
    monochrome = pick("monochrome", "monochrome", False)
    monochrome_channel = pick("monochrome_channel", "monochrome_channel", "luminance")
    reversal = pick("reversal", "reversal", False)
    contact_sheet_columns = pick("contact_sheet_columns", "contact_sheet_columns", 6)
    contact_sheet_theme = manifest_cfg.get("contact_sheet_theme", "dark")

    return manifest, {
        "neutralize": bool(neutralize),
        "base_ratios": list(base_ratios) if base_ratios is not None else None,
        "compress_tiff": bool(compress),
        "align_channels": bool(align_channels),
        "color_profile": str(icc_profile),
        "embed_metadata": bool(preserve_metadata),
        "clip": float(clip),
        "gamma": float(gamma),
        "global_levels": bool(global_levels),
        "margin": float(margin),
        "scurve": float(scurve),
        "autocrop": bool(autocrop),
        "monochrome": bool(monochrome),
        "monochrome_channel": str(monochrome_channel),
        "reversal": bool(reversal),
        "convert_to_tiff": True,
        "auto_contact_sheet": bool(getattr(args, "contact_sheet", True)),
        "contact_sheet_columns": int(contact_sheet_columns),
        "contact_sheet_theme": str(contact_sheet_theme)
    }

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

    manifest, eff = _resolve_effective_config(roll_dir, args)
    if manifest:
        print(f"  📋 Loaded session manifest (session.json) for '{manifest.get('session_name', roll_name)}'")

    positives_dir = os.path.join(roll_dir, 'positives')
    os.makedirs(positives_dir, exist_ok=True)
    
    # Collect all RAW files and sort by (frame_number, filename)
    raw_files = [f for f in os.listdir(raw_source_dir) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]
    raw_files.sort(key=lambda f: (extract_frame_num(f), f.lower()))
    
    total_raws = len(raw_files)
    num_frames = total_raws // 3
    if num_frames == 0:
        print(f"  -> Found {total_raws} RAW files (need at least 3 for a triplet). Skipping.\n")
        return 0

    print(f"Found {total_raws} RAW files -> {num_frames} frames to process.\n")
    
    successful_frames = 0
    max_workers = getattr(args, "workers", None) or min(4, max(1, (os.cpu_count() or 2) - 1))

    if max_workers > 1 and num_frames > 1:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = {}
            for i in range(0, total_raws - 2, 3):
                group_files = raw_files[i:i+3]
                frame_idx = (i // 3) + 1
                comp_name = f"Frame_{frame_idx:02d}_Composite.dng"
                comp_path = os.path.join(raw_source_dir, comp_name)
                group_full_paths = [os.path.join(raw_source_dir, f) for f in group_files]
                print(f"[{frame_idx:02d}/{num_frames:02d}] Queueing Frame {frame_idx:02d} ({', '.join(group_files)})...")
                fut = executor.submit(
                    worker_process_triplet_pipeline,
                    group=group_full_paths,
                    composite_filepath=comp_path,
                    positives_dir=positives_dir,
                    config=eff,
                )
                futures[fut] = frame_idx

            for fut in as_completed(futures):
                frame_idx = futures[fut]
                try:
                    success, result, worker_logs = fut.result()
                    if worker_logs:
                        print(worker_logs.strip())
                    if success:
                        print(f"  -> [{frame_idx:02d}/{num_frames:02d}] Frame {frame_idx:02d} converted successfully\n")
                        successful_frames += 1
                    else:
                        print(f"  -> ERROR processing Frame {frame_idx:02d}: {result}\n")
                except Exception as e:
                    print(f"  -> ERROR processing Frame {frame_idx:02d}: {e}\n")
    else:
        for i in range(0, total_raws - 2, 3):
            group_files = raw_files[i:i+3]
            frame_idx = (i // 3) + 1
            
            comp_name = f"Frame_{frame_idx:02d}_Composite.dng"
            comp_path = os.path.join(raw_source_dir, comp_name)
            
            print(f"[{frame_idx:02d}/{num_frames:02d}] Processing Frame {frame_idx:02d} ({', '.join(group_files)})...")
            group_full_paths = [os.path.join(raw_source_dir, f) for f in group_files]
            
            start_t = time.time()
            try:
                preserve_meta = eff["embed_metadata"]
                neutralize_base = eff["neutralize"]
                source_meta = (group_full_paths[1] if len(group_full_paths) > 1 else group_full_paths[0]) if preserve_meta else None
                # 1. Composite triplet with sub-pixel alignment & grain-safe demosaicing (in-memory handoff)
                _, composite_rgb = process_triplet(
                    group=group_full_paths,
                    output_filepath=comp_path,
                    neutralize_base=neutralize_base,
                    compress_tiff=eff["compress_tiff"],
                    align_channels=eff["align_channels"],
                    icc_profile=eff["color_profile"],
                    preserve_metadata=preserve_meta,
                    base_ratios=eff["base_ratios"],
                    return_image=True
                )
                
                # 2. Sensitometric Inversion directly from in-memory composite_rgb
                process_positives(
                    input_path=comp_path,
                    output_dir=positives_dir,
                    clip=eff["clip"],
                    gamma=eff["gamma"],
                    compress_tiff=eff["compress_tiff"],
                    global_levels=eff["global_levels"],
                    ignore_margin=eff["margin"],
                    scurve=eff["scurve"],
                    autocrop=eff["autocrop"],
                    monochrome=eff["monochrome"],
                    monochrome_channel=eff["monochrome_channel"],
                    reversal=eff["reversal"],
                    convert_to_tiff=True,
                    icc_profile=eff["color_profile"],
                    preserve_metadata=preserve_meta,
                    base_ratios=None if neutralize_base else eff["base_ratios"],
                    preloaded_image=composite_rgb,
                    source_metadata_path=source_meta
                )
                elapsed = time.time() - start_t
                print(f"  -> Successfully converted in {elapsed:.1f}s\n")
                successful_frames += 1
                
            except Exception as e:
                print(f"  -> ERROR processing Frame {frame_idx:02d}: {e}\n")

    # Persist updated session manifest
    write_session_manifest(
        session_dir=roll_dir,
        session_name=(manifest.get("session_name") if manifest else None) or roll_name,
        stock=manifest.get("stock") if manifest else None,
        film_format=manifest.get("format") if manifest else None,
        roll=manifest.get("roll") if manifest else None,
        mode=(manifest.get("mode") if manifest else None) or "triplet",
        config=eff
    )

    if successful_frames > 0 and eff["auto_contact_sheet"]:
        print(f"  -> Generating archival roll contact sheet...")
        try:
            cs_res = generate_contact_sheet(
                session_dir=roll_dir,
                columns=eff["contact_sheet_columns"],
                theme=eff["contact_sheet_theme"],
                config=eff
            )
            if cs_res.get("success"):
                print(f"  ✅ Contact sheet created: {cs_res.get('message')}")
        except Exception as e:
            print(f"  ❌ Error generating contact sheet: {e}")

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
    parser.add_argument("--global-levels", action=argparse.BooleanOptionalAction, default=True,
                        help="Use joint global exposure scaling (default: on; preserves color balance). Per-channel levels can introduce a magenta cast on color negatives; pass --no-global-levels for legacy per-channel behavior.")
    parser.add_argument("-n", "--neutralize", action="store_true", help="Neutralize film base during compositing")
    parser.add_argument("--base-ratios", nargs=3, type=float, metavar=('R', 'G', 'B'), default=None,
                        help="Custom film base neutralizer RGB ratios (e.g. 1.0 0.58 0.23) sampled from film rebate")
    parser.add_argument("-c", "--compress", action="store_true", help="Enable zlib compression for output TIFFs")
    parser.add_argument("--no-align", action="store_true", help="Disable sub-pixel channel alignment")
    parser.add_argument("--monochrome", "--bw", action="store_true", help="Convert output to monochrome/B&W")
    parser.add_argument("--monochrome-channel", type=str, default="luminance", choices=["luminance", "average", "red", "green", "blue"])
    parser.add_argument("--reversal", action="store_true", help="Process positive slide / reversal film")
    parser.add_argument("--icc-profile", "--color-profile", type=str, default="adobe_rgb",
                        choices=["adobe_rgb", "prophoto_rgb", "srgb", "none"],
                        help="Embedded ICC color profile (default: adobe_rgb)")
    parser.add_argument("--no-metadata", action="store_true",
                        help="Disable embedding original camera RAW EXIF/IPTC metadata into composites and positives")
    parser.add_argument("--no-contact-sheet", dest="contact_sheet", action="store_false",
                        help="Disable automatic contact sheet generation after roll processing")
    parser.add_argument("--contact-sheet-columns", type=int, default=6,
                        help="Number of columns on contact sheet grid (default: 6)")
    parser.add_argument("-w", "--workers", type=int, default=None,
                        help="Number of parallel worker processes (default: auto based on CPU cores)")
    parser.set_defaults(contact_sheet=True)
    
    args = parser.parse_args()

    # Track which CLI flags were explicitly provided by the user
    explicit_dests = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if any(arg == opt or arg.startswith(opt + "=") for arg in sys.argv[1:]):
                explicit_dests.add(action.dest)
    args._explicit = explicit_dests
    
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
