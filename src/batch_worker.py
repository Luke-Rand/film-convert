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

def worker_process_positives(input_path, output_dir, clip, gamma, compress_tiff, global_levels, ignore_margin, scurve, autocrop, monochrome, monochrome_channel, reversal, convert_to_tiff, icc_profile, preserve_metadata, base_ratios):
    """
    Executes process_positives in a background worker process.
    Returns (success: bool, result: None, logs: str)
    """
    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            process_positives(
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
        return (True, None, capture.getvalue())
    except Exception as e:
        capture.write(f"\nWorker Exception: {str(e)}\n{traceback.format_exc()}\n")
        return (False, str(e), capture.getvalue())

def worker_process_triplet_pipeline(group, composite_filepath, positives_dir, config):
    """
    Runs both composition and inversion sequentially within a single worker process invocation,
    minimizing process IPC overhead for live scanning session monitor.
    Returns (success: bool, means: tuple, logs: str)
    """
    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            # 1. Composite
            r_mean, g_mean, b_mean = process_triplet(
                group=group,
                output_filepath=composite_filepath,
                neutralize_base=config.get("neutralize", False),
                compress_tiff=config.get("compress_tiff", False),
                align_channels=config.get("align_channels", False),
                icc_profile=config.get("color_profile", "adobe_rgb"),
                preserve_metadata=config.get("embed_metadata", True),
                base_ratios=config.get("base_ratios")
            )
            # 2. Invert composite
            process_positives(
                input_path=composite_filepath,
                output_dir=positives_dir,
                clip=config.get("clip", 0.1),
                gamma=config.get("gamma", 2.2),
                compress_tiff=config.get("compress_tiff", False),
                global_levels=config.get("global_levels", False),
                ignore_margin=config.get("margin", 0.15),
                scurve=config.get("scurve", 0.0),
                autocrop=config.get("autocrop", False),
                monochrome=config.get("monochrome", False),
                monochrome_channel=config.get("monochrome_channel", "luminance"),
                reversal=config.get("reversal", False),
                convert_to_tiff=config.get("convert_to_tiff", True),
                icc_profile=config.get("color_profile", "adobe_rgb"),
                preserve_metadata=config.get("embed_metadata", True),
                base_ratios=config.get("base_ratios")
            )
        return (True, (float(r_mean), float(g_mean), float(b_mean)), capture.getvalue())
    except Exception as e:
        capture.write(f"\nWorker Exception: {str(e)}\n{traceback.format_exc()}\n")
        return (False, str(e), capture.getvalue())
