import os
import sys
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional

from .base import BaseCameraBackend

class WindowsBridgeBackend(BaseCameraBackend):
    """
    Windows Native Camera Bridge Backend.
    Uses bundled native Windows camera driver bridge (canon_bridge.exe / EDSDK)
    to communicate with DSLR / mirrorless cameras over standard Windows USB
    without requiring Zadig or driver changes.
    """

    def __init__(self, log_callback=None):
        super().__init__(log_callback=log_callback)
        self.bridge_proc = None
        self.reader_thread = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.bridge_path = self._locate_bridge_executable()
        self.bridge_available = self.bridge_path is not None
        self.camera_connected = False
        self.connection_state = "disconnected"
        self.model_name = "Canon EOS Camera"

    def _locate_bridge_executable(self) -> Optional[str]:
        search_paths = []
        if hasattr(sys, '_MEIPASS'):
            search_paths.append(os.path.join(sys._MEIPASS, 'camera_bridge', 'canon_bridge.exe'))
            search_paths.append(os.path.join(sys._MEIPASS, 'canon_bridge.exe'))
            search_paths.append(os.path.join(sys._MEIPASS, 'camera_bridge', 'camera_bridge.exe'))

        repo_root = Path(__file__).resolve().parent.parent.parent
        search_paths.extend([
            str(repo_root / 'src' / 'vendor' / 'camera_bridge_win' / 'canon_bridge.exe'),
            str(repo_root / 'vendor' / 'camera_bridge_win' / 'canon_bridge.exe'),
            str(repo_root / 'src' / 'vendor' / 'camera_bridge_win' / 'camera_bridge.exe'),
            r"C:\Users\lrand\OneDrive\Documents\repos\filmconvert-studio\bin\canon_bridge.exe"
        ])

        for path in search_paths:
            if os.path.isfile(path):
                return path
        return None

    def connect(self) -> bool:
        if sys.platform != 'win32':
            return False

        if not self.bridge_path:
            self.bridge_path = self._locate_bridge_executable()
            if not self.bridge_path:
                self.log("Native Windows camera bridge binary not located.")
                self.camera_connected = False
                self.connection_state = "disconnected"
                return False

        with self.lock:
            if self.bridge_proc and self.bridge_proc.poll() is None:
                self.camera_connected = True
                self.connection_state = "connected"
                return True

        self.log(f"Launching Windows camera bridge from: {self.bridge_path}")
        try:
            self.bridge_proc = subprocess.Popen(
                [self.bridge_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )

            time.sleep(0.4)
            ret = self.bridge_proc.poll()
            if ret is not None:
                err_out = ""
                try:
                    err_out = self.bridge_proc.stderr.read().decode('utf-8', errors='ignore')
                except Exception:
                    pass
                self.log(f"Bridge process exited immediately (code {ret}): {err_out.strip()}")
                self.camera_connected = False
                self.connection_state = "disconnected"
                self.bridge_proc = None
                return False

            self.camera_connected = True
            self.connection_state = "connected"
            self.model_name = "Canon EOS Camera"

            # Start background reader thread for EVF JPEG stream
            self.reader_thread = threading.Thread(
                target=self._read_evf_frames_worker,
                name="WindowsBridgeReader",
                daemon=True
            )
            self.reader_thread.start()
            self.log("Windows camera bridge connection established and reader active.")
            return True

        except Exception as e:
            self.log(f"Failed to start camera bridge: {e}")
            self.camera_connected = False
            self.connection_state = "error"
            self.bridge_proc = None
            return False

    def _read_exact(self, n: int) -> Optional[bytes]:
        data = bytearray()
        while len(data) < n:
            if not self.bridge_proc or not self.bridge_proc.stdout:
                return None
            chunk = self.bridge_proc.stdout.read(n - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)

    def _read_evf_frames_worker(self):
        MAGIC = 0x46564501
        self.log("Bridge reader loop started.")
        while self.camera_connected and self.bridge_proc and self.bridge_proc.poll() is None:
            try:
                header = self._read_exact(8)
                if not header or len(header) < 8:
                    break
                magic, length = struct.unpack("<II", header)
                if magic != MAGIC or length == 0 or length > 10000000:
                    continue
                jpeg_bytes = self._read_exact(length)
                if jpeg_bytes and len(jpeg_bytes) == length:
                    with self.lock:
                        self.latest_frame = jpeg_bytes
            except Exception as e:
                self.log(f"Bridge reader exception: {e}")
                break

        self.camera_connected = False
        self.connection_state = "disconnected"
        self.log("Bridge reader loop finished.")

    def disconnect(self):
        with self.lock:
            if self.bridge_proc:
                try:
                    if self.bridge_proc.stdin:
                        self.bridge_proc.stdin.write(b"QUIT\n")
                        self.bridge_proc.stdin.flush()
                except Exception:
                    pass
                try:
                    self.bridge_proc.terminate()
                except Exception:
                    pass
                self.bridge_proc = None
        self.camera_connected = False
        self.connection_state = "disconnected"
        self.log("Windows camera bridge disconnected.")

    def grab_preview_frame(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_frame

    def capture_image(self, autofocus: bool = True) -> Optional[str]:
        if not self.camera_connected or not self.bridge_proc:
            raise RuntimeError("Camera not connected.")
        try:
            self.bridge_proc.stdin.write(b"SHUTTER\n")
            self.bridge_proc.stdin.flush()
            self.log("Shutter command sent to camera bridge.")
            time.sleep(1.0)
            return None
        except Exception as e:
            self.log(f"Capture command error: {e}")
            raise

    def step_focus(self, direction: str, coarse: bool = False) -> bool:
        if not self.camera_connected or not self.bridge_proc:
            return False
        step = 3 if coarse else 1
        if direction.lower() == "near":
            step = -step
        try:
            cmd = f"FOCUS {step}\n".encode('ascii')
            self.bridge_proc.stdin.write(cmd)
            self.bridge_proc.stdin.flush()
            self.log(f"Focus command sent: {cmd.strip()}")
            return True
        except Exception as e:
            self.log(f"Focus command error: {e}")
            return False

    def trigger_autofocus(self) -> bool:
        return self.step_focus("near", coarse=False)

    def get_settings(self) -> Dict[str, str]:
        return self.camera_settings

    def set_setting(self, name: str, value: str) -> bool:
        self.camera_settings[name.lower()] = value
        return True

    def get_choices(self) -> Dict[str, list]:
        return self.camera_choices
