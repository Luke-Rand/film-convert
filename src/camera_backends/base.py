import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseCameraBackend(ABC):
    """Abstract base class defining the unified interface for all camera backends."""

    def __init__(self, log_callback=None):
        self.log_callback = log_callback or print
        self.camera_connected = False
        self.connection_state = "disconnected"
        self.model_name = ""
        self.camera_settings = {
            "iso": "Auto",
            "aperture": "5.6",
            "shutterspeed": "auto"
        }
        self.camera_choices = {
            "iso": ["Auto", "100", "200", "400", "800", "1600", "3200", "6400"],
            "aperture": ["2.8", "3.5", "4", "5.6", "8", "11", "16", "22"],
            "shutterspeed": ["1/500", "1/250", "1/125", "1/60", "1/30", "1/15", "1/8", "1/4", "1/2", "1", "2", "auto"]
        }

    def log(self, message: str):
        if self.log_callback:
            self.log_callback(f"[{self.__class__.__name__}] {message}")

    def _get_next_capture_naming(self, target_dir: Optional[str]):
        """
        Determines (frame_num, shot_num, is_triplet) for a new raw capture in target_dir.
        In triplet mode, all 3 shots of a triplet share the same film frame_num (e.g. Frame_01_Capture_01..03)
        so the resulting composite and positive for the first triplet are named Frame_01 instead of Frame_04.
        """
        base_frame_num = 1
        mode = "triplet"
        session_mgr = getattr(self, "session_manager", None)
        if session_mgr:
            mode = getattr(session_mgr, "mode", "triplet") or "triplet"
            if hasattr(session_mgr, "get_next_frame_number"):
                base_frame_num = session_mgr.get_next_frame_number(target_dir)

        pending_captures = 0
        raw_exts = {'.cr3', '.raf', '.nef', '.arw', '.rw2', '.nrw', '.dcr', '.dng', '.tif', '.tiff', '.jpg', '.jpeg'}
        if target_dir and os.path.exists(target_dir):
            try:
                for entry in os.listdir(target_dir):
                    full_p = os.path.join(target_dir, entry)
                    if not os.path.isfile(full_p):
                        continue
                    ext_l = os.path.splitext(entry)[1].lower()
                    name_l = entry.lower()
                    if ext_l in raw_exts and "_composite" not in name_l and "_positive" not in name_l:
                        pending_captures += 1
            except OSError:
                pass

        if mode == "triplet":
            frame_num = base_frame_num + (pending_captures // 3)
            shot_num = (pending_captures % 3) + 1
            return frame_num, shot_num, True
        else:
            frame_num = base_frame_num + pending_captures
            return frame_num, 1, False

    @abstractmethod
    def connect(self) -> bool:
        """Attempt to connect to a physical or simulated camera."""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect the camera and clean up resources."""
        pass

    @abstractmethod
    def grab_preview_frame(self) -> Optional[bytes]:
        """Capture and return the latest Live View JPEG frame as bytes."""
        pass

    @abstractmethod
    def capture_image(self, autofocus: bool = True) -> Optional[str]:
        """Trigger capture and download image. Returns local filepath to the captured image."""
        pass

    @abstractmethod
    def step_focus(self, direction: str, coarse: bool = False) -> bool:
        """Step motorized manual focus in given direction ('near' or 'far')."""
        pass

    @abstractmethod
    def trigger_autofocus(self) -> bool:
        """Trigger camera autofocus drive."""
        pass

    @abstractmethod
    def get_settings(self) -> Dict[str, str]:
        """Query and return active camera settings dictionary."""
        pass

    @abstractmethod
    def set_setting(self, name: str, value: str) -> bool:
        """Update a camera setting (iso, aperture, shutterspeed, etc.)."""
        pass

    @abstractmethod
    def get_choices(self) -> Dict[str, list]:
        """Return available choices for camera settings."""
        pass

    def get_status(self) -> Dict[str, Any]:
        """Return full status payload for UI serialization."""
        return {
            "connected": self.camera_connected,
            "simulated": False,
            "state": self.connection_state,
            "model": self.model_name,
            "settings": self.camera_settings,
            "choices": self.camera_choices
        }
