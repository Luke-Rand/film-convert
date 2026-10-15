"""
Modular Camera Backends for FilmConvert.
Supports bundled libgphoto2 (macOS/Linux), native Windows bridge (Windows),
and simulated fallbacks.
"""

from .base import BaseCameraBackend
from .simulated import SimulatedCameraBackend
from .windows_bridge import WindowsBridgeBackend

__all__ = [
    "BaseCameraBackend",
    "SimulatedCameraBackend",
    "WindowsBridgeBackend"
]
