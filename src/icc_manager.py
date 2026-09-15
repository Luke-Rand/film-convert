import os
from pathlib import Path
from typing import Optional

_PROFILE_CACHE = {}

import sys
_MODULE_DIR = Path(__file__).resolve().parent
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    _BUNDLED_ICC_DIR = Path(sys._MEIPASS) / "icc_profiles"
else:
    _BUNDLED_ICC_DIR = _MODULE_DIR / "icc_profiles"

_SYSTEM_ICC_DIRS = [
    Path("/System/Library/ColorSync/Profiles"),
    Path("/Library/ColorSync/Profiles"),
    Path("/usr/share/color/icc"),
    Path("/usr/local/share/color/icc"),
]

# Map standardized names to candidate filenames (bundled and system)
_PROFILE_FILENAMES = {
    "adobe_rgb": [
        "AdobeRGB1998.icc",
        "AdobeRGB1998.ICC",
        "Adobe RGB (1998).icc",
    ],
    "prophoto_rgb": [
        "ROMM_RGB.icc",
        "ROMM RGB.icc",
        "ProPhotoRGB.icc",
        "ProPhoto.icc",
        "ProPhoto.ICC",
    ],
    "srgb": [
        "sRGB.icc",
        "sRGB Profile.icc",
        "sRGB.ICC",
        "sRGB Color Space Profile.icm",
    ],
    "gray": [
        "Generic_Gray_Gamma2.2.icc",
        "Generic Gray Gamma 2.2 Profile.icc",
        "Generic Gray Profile.icc",
        "Gray.icc",
    ],
}


def normalize_profile_name(name: Optional[str]) -> str:
    """Normalizes various user/cli strings into standard keys."""
    if not name:
        return "adobe_rgb"
    
    clean = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    
    if clean in ("none", "linear", "passthrough", "off", "false"):
        return "none"
    if clean in ("adobe", "adobergb", "adobe_rgb", "adobe_rgb_1998", "adobe_1998"):
        return "adobe_rgb"
    if clean in ("prophoto", "prophoto_rgb", "romm", "romm_rgb", "romm_rgb_iso_22028_2:2013"):
        return "prophoto_rgb"
    if clean in ("srgb", "s_rgb", "standard_rgb"):
        return "srgb"
    if clean in ("gray", "grey", "grayscale", "monochrome", "bw", "gray_gamma22"):
        return "gray"
    
    return clean


def _load_profile_bytes(profile_key: str) -> Optional[bytes]:
    """Attempts to load profile bytes from bundled dir, then system paths, then PIL."""
    candidates = _PROFILE_FILENAMES.get(profile_key, [f"{profile_key}.icc"])
    
    # 1. Bundled profiles
    for fname in candidates:
        bundled_path = _BUNDLED_ICC_DIR / fname
        if bundled_path.is_file():
            try:
                return bundled_path.read_bytes()
            except Exception:
                pass
                
    # 2. System profiles
    for sys_dir in _SYSTEM_ICC_DIRS:
        if sys_dir.is_dir():
            for fname in candidates:
                sys_path = sys_dir / fname
                if sys_path.is_file():
                    try:
                        return sys_path.read_bytes()
                    except Exception:
                        pass
                        
    # 3. PIL ImageCms fallback (e.g. for sRGB)
    if profile_key == "srgb":
        try:
            from PIL import ImageCms
            cms_prof = ImageCms.createProfile("sRGB")
            if hasattr(cms_prof, "tobytes"):
                return cms_prof.tobytes()
        except Exception:
            pass

    return None


def get_icc_profile(profile_name: Optional[str] = "adobe_rgb", is_monochrome: bool = False) -> Optional[bytes]:
    """
    Returns the binary ICC profile data for the requested color space.
    If is_monochrome is True, returns the standard Gray Gamma 2.2 profile unless profile_name is 'none'.
    """
    key = normalize_profile_name(profile_name)
    if key == "none":
        return None
        
    if is_monochrome:
        key = "gray"
        
    if key in _PROFILE_CACHE:
        return _PROFILE_CACHE[key]
        
    data = _load_profile_bytes(key)
    if data:
        _PROFILE_CACHE[key] = data
    return data


def get_available_profiles():
    """Returns the list of supported color profile options."""
    return [
        {"id": "adobe_rgb", "name": "Adobe RGB (1998)", "description": "Standard wide gamut for photography & print"},
        {"id": "prophoto_rgb", "name": "ProPhoto RGB (ROMM)", "description": "Ultra-wide gamut for high-dynamic-range negatives"},
        {"id": "srgb", "name": "sRGB", "description": "Standard color space for web and consumer screens"},
        {"id": "none", "name": "None / Linear", "description": "Untagged raw linear / sensor space"},
    ]
