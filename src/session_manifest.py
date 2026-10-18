"""
Session Manifest (session.json) persistence and recovery for FilmConvert.

Stores per-roll metadata (stock, format, roll, mode, timestamps, frame count)
and complete processing configuration (SessionConfigSchema, sampled base_ratios,
triplet channel means) inside each session directory.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from schemas import SessionConfigSchema
except ImportError:
    from src.schemas import SessionConfigSchema

MANIFEST_FILENAME = "session.json"
MANIFEST_VERSION = "1.0"

VALID_POSITIVE_EXTS = {
    ".tif", ".tiff", ".dng", ".cr3", ".raf", ".nef",
    ".arw", ".rw2", ".nrw", ".dcr", ".jpg", ".jpeg", ".png"
}


def get_manifest_path(session_dir: str) -> str:
    """Returns the absolute path to session.json inside session_dir."""
    return os.path.join(os.path.abspath(os.path.expanduser(session_dir)), MANIFEST_FILENAME)


def parse_roll_identifiers(
    session_name: str,
    stock: str = "",
    film_format: str = "",
    roll: str = ""
) -> Tuple[str, str, str]:
    """
    Infers (stock, format, roll) from session_name (e.g. 'KodakGold200-135-01')
    when explicit values are not provided.
    """
    parts = [p.strip() for p in (session_name or "").split("-") if p.strip()]
    resolved_stock = stock.strip() if stock else (parts[0] if len(parts) >= 1 else "FilmStock")
    resolved_format = film_format.strip() if film_format else (parts[1] if len(parts) >= 2 else "135")
    resolved_roll = roll.strip().zfill(2) if roll else (parts[2].zfill(2) if len(parts) >= 3 else "01")
    return resolved_stock, resolved_format, resolved_roll


def count_session_frames(session_dir: str) -> int:
    """Counts completed positive frames in <session_dir>/positives (or Positives)."""
    if not session_dir or not os.path.isdir(session_dir):
        return 0

    for sub in ("positives", "Positives"):
        pos_dir = os.path.join(session_dir, sub)
        if os.path.isdir(pos_dir):
            try:
                return sum(
                    1 for f in os.listdir(pos_dir)
                    if os.path.isfile(os.path.join(pos_dir, f))
                    and os.path.splitext(f)[1].lower() in VALID_POSITIVE_EXTS
                    and "_contact_sheet" not in f.lower()
                )
            except OSError:
                return 0
    return 0


def read_session_manifest(session_dir: str) -> Optional[Dict[str, Any]]:
    """
    Reads and returns the parsed session.json dictionary from session_dir,
    or None if no valid manifest exists.
    """
    if not session_dir:
        return None
    manifest_path = get_manifest_path(session_dir)
    if not os.path.isfile(manifest_path):
        return None

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None

        # Normalize and validate config block through SessionConfigSchema
        raw_cfg = data.get("config")
        if isinstance(raw_cfg, dict):
            validated_cfg = SessionConfigSchema(**raw_cfg).model_dump()
            data["config"] = validated_cfg
        else:
            data["config"] = SessionConfigSchema().model_dump()

        return data
    except Exception:
        return None


def write_session_manifest(
    session_dir: str,
    *,
    session_name: str = "",
    stock: str = "",
    film_format: str = "",
    roll: str = "",
    mode: str = "",
    config: Optional[Dict[str, Any]] = None,
    sampled_rgb: Optional[List[int]] = None,
    last_triplet_means: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Creates or updates <session_dir>/session.json atomically.
    Preserves existing created_at and any unspecified metadata fields when updating.
    """
    abs_session_dir = os.path.abspath(os.path.expanduser(session_dir))
    os.makedirs(abs_session_dir, exist_ok=True)

    existing = read_session_manifest(abs_session_dir) or {}
    now_iso = datetime.now(timezone.utc).isoformat()

    resolved_session_name = session_name or existing.get("session_name") or os.path.basename(abs_session_dir)
    resolved_stock, resolved_format, resolved_roll = parse_roll_identifiers(
        resolved_session_name,
        stock=stock or existing.get("stock", ""),
        film_format=film_format or existing.get("format", ""),
        roll=roll or existing.get("roll", "")
    )
    resolved_mode = mode or existing.get("mode") or "triplet"

    merged_cfg = SessionConfigSchema().model_dump()
    if isinstance(existing.get("config"), dict):
        merged_cfg.update(existing["config"])
    if isinstance(config, dict):
        merged_cfg.update(config)
    validated_cfg = SessionConfigSchema(**merged_cfg).model_dump()

    manifest: Dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "session_name": resolved_session_name,
        "stock": resolved_stock,
        "format": resolved_format,
        "roll": resolved_roll,
        "mode": resolved_mode,
        "created_at": existing.get("created_at") or now_iso,
        "updated_at": now_iso,
        "frame_count": count_session_frames(abs_session_dir),
        "config": validated_cfg,
    }

    resolved_sampled_rgb = sampled_rgb if sampled_rgb is not None else existing.get("sampled_rgb")
    if resolved_sampled_rgb is not None:
        manifest["sampled_rgb"] = resolved_sampled_rgb

    resolved_means = last_triplet_means if last_triplet_means is not None else existing.get("last_triplet_means")
    if resolved_means is not None:
        manifest["last_triplet_means"] = resolved_means

    manifest_path = get_manifest_path(abs_session_dir)
    tmp_path = f"{manifest_path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_path, manifest_path)
    except Exception:
        # Fallback to direct write if atomic replace fails on network/external volume
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return manifest
