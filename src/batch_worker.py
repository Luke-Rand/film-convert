"""
Batch Worker Module for Out-of-Process Task Execution.
Runs CPU-heavy RawPy demosaicing, Fourier channel alignment, and 16-bit TIFF
inversion in isolated multiprocessing workers to prevent GIL contention on the Flask server.
"""

import os
import sys
import io
import traceback
import contextlib

# Ensure src directory is in sys.path for worker subprocesses
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from compositor import process_triplet
from inverter import process_positives
from metadata_preservation import cleanup_staged_raw

def worker_process_triplet(group, output_filepath, neutralize_base, compress_tiff, align_channels, icc_profile, preserve_metadata, base_ratios):
    """
    Executes process_triplet in a background worker process.
    Returns (success: bool, result: tuple, logs: str)
    """
    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            r_mean, g_mean, b_mean = process_triplet(
                group=group,
                output_filepath=output_filepath,
                neutralize_base=neutralize_base,
                compress_tiff=compress_tiff,
                align_channels=align_channels,
                icc_profile=icc_profile,
                preserve_metadata=preserve_metadata,
                base_ratios=base_ratios
            )
        return (True, (float(r_mean), float(g_mean), float(b_mean)), capture.getvalue())
    except Exception as e:
        capture.write(f"\nWorker Exception: {str(e)}\n{traceback.format_exc()}\n")
        return (False, str(e), capture.getvalue())
    finally:
        for f in (group or []):
            cleanup_staged_raw(f)

def worker_process_positives(input_path, output_dir, clip, gamma, compress_tiff, global_levels, ignore_margin, scurve, autocrop, monochrome, monochrome_channel, reversal, convert_to_tiff, icc_profile, preserve_metadata, base_ratios):
    """
    Executes process_positives in a background worker process.
    Returns (success: bool, result: Optional[str], logs: str)
    """
    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            out_paths = process_positives(
                input_path=input_path,
                output_dir=output_dir,
                clip=clip,
                gamma=gamma,
                compress_tiff=compress_tiff,
                global_levels=global_levels,
                ignore_margin=ignore_margin,
                scurve=scurve,
                autocrop=autocrop,
                monochrome=monochrome,
                monochrome_channel=monochrome_channel,
                reversal=reversal,
                convert_to_tiff=convert_to_tiff,
                icc_profile=icc_profile,
                preserve_metadata=preserve_metadata,
                base_ratios=base_ratios
            )
        saved_path = out_paths[0] if out_paths else None
        return (True, saved_path, capture.getvalue())
    except Exception as e:
        capture.write(f"\nWorker Exception: {str(e)}\n{traceback.format_exc()}\n")
        return (False, str(e), capture.getvalue())
    finally:
        cleanup_staged_raw(input_path)

def worker_process_triplet_pipeline(group, composite_filepath, positives_dir, config):
    """
    Runs both composition and inversion sequentially within a single worker process invocation,
    minimizing process IPC overhead and passing the 16-bit composite array in-memory to
    eliminate redundant disk reads.
    Returns (success: bool, means: tuple, logs: str)
    """
    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            preserve_meta = config.get("embed_metadata", True)
            neutralize_base = config.get("neutralize", False)
            base_ratios = config.get("base_ratios")
            source_meta = (group[1] if len(group) > 1 else group[0]) if (preserve_meta and group) else None
            # 1. Composite (returns in-memory 16-bit RGB array alongside channel means)
            (r_mean, g_mean, b_mean), composite_rgb = process_triplet(
                group=group,
                output_filepath=composite_filepath,
                neutralize_base=neutralize_base,
                compress_tiff=config.get("compress_tiff", False),
                align_channels=config.get("align_channels", False),
                icc_profile=config.get("color_profile", "adobe_rgb"),
                preserve_metadata=preserve_meta,
                base_ratios=base_ratios,
                return_image=True
            )
            # 2. Invert composite directly from in-memory array without re-reading disk file.
            # If neutralize_base ran in process_triplet, the composite channels are already
            # normalized by base_ratios; do not apply base_ratios a second time in process_positives.
            process_positives(
                input_path=composite_filepath,
                output_dir=positives_dir,
                clip=config.get("clip", 0.1),
                gamma=config.get("gamma", 2.2),
                compress_tiff=config.get("compress_tiff", False),
                global_levels=config.get("global_levels", True),
                ignore_margin=config.get("margin", 0.15),
                scurve=config.get("scurve", 0.0),
                autocrop=config.get("autocrop", False),
                monochrome=config.get("monochrome", False),
                monochrome_channel=config.get("monochrome_channel", "luminance"),
                reversal=config.get("reversal", False),
                convert_to_tiff=config.get("convert_to_tiff", True),
                icc_profile=config.get("color_profile", "adobe_rgb"),
                preserve_metadata=preserve_meta,
                base_ratios=None if neutralize_base else base_ratios,
                preloaded_image=composite_rgb,
                source_metadata_path=source_meta
            )
        return (True, (float(r_mean), float(g_mean), float(b_mean)), capture.getvalue())
    except Exception as e:
        capture.write(f"\nWorker Exception: {str(e)}\n{traceback.format_exc()}\n")
        return (False, str(e), capture.getvalue())
    finally:
        for f in (group or []):
            cleanup_staged_raw(f)
        cleanup_staged_raw(composite_filepath)
