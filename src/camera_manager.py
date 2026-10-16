import io
import sys
import time
import queue
import threading
from PIL import Image

from camera_backends import (
    SimulatedCameraBackend,
    WindowsBridgeBackend,
    GPhoto2CameraBackend,
    GPHOTO2_AVAILABLE,
)


class CameraManager:
    """
    Coordinates the background camera worker queue, Live View frame caching,
    fast-focus thumbnail generation, and strategy delegation across
    SimulatedCameraBackend, WindowsBridgeBackend, and GPhoto2CameraBackend.
    """

    def __init__(self, session_manager=None):
        self.session_manager = session_manager
        self.log_callback = print

        # Strategy Backends
        self.simulated_backend = SimulatedCameraBackend(
            log_callback=self.log,
            session_manager=session_manager
        )
        self.simulated_backend.connect()

        self.gphoto2_backend = GPhoto2CameraBackend(
            log_callback=self.log,
            session_manager=session_manager
        )

        self.windows_bridge = None
        if sys.platform == 'win32':
            try:
                wb = WindowsBridgeBackend(log_callback=self.log)
                if wb.bridge_available:
                    self.windows_bridge = wb
            except Exception:
                pass

        # In simulated mode if neither gphoto2 nor windows bridge is available
        self.camera_connected = False
        self.simulated = not GPHOTO2_AVAILABLE and (self.windows_bridge is None)
        self.connection_state = "simulated" if self.simulated else "searching"
        self.model_name = "Simulated Camera" if self.simulated else ""

        # Thread safety control
        self.cmd_queue = queue.Queue()
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        # Live view configurations & states
        self.live_view_active = False
        self.pause_preview = False
        self.latest_frame = None
        self.frame_id = 0
        self.frame_event = threading.Event()
        self.frame_lock = threading.Lock()

        # Focus activity telemetry and downscaled thumbnail cache
        self.last_focus_time = 0.0
        self._fast_focus_thumbnail_cache = None
        self._cached_thumb_frame_id = -1

        # Share references with backends for backward compatibility
        self.sim_settings = self.simulated_backend.camera_settings
        self.sim_choices = self.simulated_backend.camera_choices
        self.mock_leds = self.simulated_backend.mock_leds

        self.camera_settings = self.gphoto2_backend.camera_settings
        self.camera_choices = self.gphoto2_backend.camera_choices

    def log(self, msg):
        if self.session_manager:
            self.session_manager.log(f"[Camera] {msg}")
        else:
            self.log_callback(f"[Camera] {msg}")

    def start(self):
        self.stop_event.clear()
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            name="CameraWorkerThread",
            daemon=True
        )
        self.worker_thread.start()
        self.log("Worker thread started.")

    def stop(self):
        self.log("Stopping worker thread...")
        self.stop_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=3.0)
            self.worker_thread = None
        self.disconnect()

    def disconnect(self):
        try:
            self.gphoto2_backend.disconnect()
        except Exception:
            pass

        if self.windows_bridge:
            try:
                self.windows_bridge.disconnect()
            except Exception:
                pass

        self.camera_connected = False
        self.connection_state = "simulated" if self.simulated else "disconnected"
        self.log("Camera disconnected.")

    def get_status(self):
        with self.frame_lock:
            if self.simulated:
                return {
                    "connected": True,
                    "simulated": True,
                    "state": "simulated",
                    "model": "Simulated Camera",
                    "settings": self.sim_settings,
                    "choices": self.sim_choices
                }
            elif self.camera_connected:
                return {
                    "connected": True,
                    "simulated": False,
                    "state": self.connection_state,
                    "model": self.model_name or "Physical Camera",
                    "settings": self.camera_settings,
                    "choices": self.camera_choices
                }
            else:
                return {
                    "connected": False,
                    "simulated": False,
                    "state": self.connection_state,
                    "model": self.model_name,
                    "settings": {},
                    "choices": {}
                }

    # Queue commands helpers
    def send_cmd(self, cmd, args=None, timeout=5.0):
        resp_q = queue.Queue()
        self.cmd_queue.put((cmd, args or {}, resp_q))
        try:
            success, val = resp_q.get(timeout=timeout)
            if not success:
                raise Exception(val)
            return val
        except queue.Empty:
            raise Exception("Command timed out waiting for camera response.")

    def update_config(self, name, value):
        return self.send_cmd("set_config", {"name": name, "value": value})

    def reconnect(self):
        return self.send_cmd("reconnect", {}, timeout=20.0)

    def capture_image(self, autofocus=True):
        return self.send_cmd("capture", {"autofocus": autofocus}, timeout=30.0)

    def set_liveview(self, active):
        self.live_view_active = active
        self.log(f"Live view streaming toggled: {active}")

    def update_mock_leds(self, r, g, b):
        self.simulated_backend.update_mock_leds(r, g, b)
        self.mock_leds = self.simulated_backend.mock_leds

    def get_latest_frame(self):
        with self.frame_lock:
            return self.latest_frame

    def notify_focus_adjustment(self):
        """Notifies the camera manager that a manual focus step or autofocus action is occurring."""
        self.last_focus_time = time.time()

    def is_fast_focus_active(self, window_sec=1.0):
        """Returns True if a manual focus command was issued within the last window_sec seconds."""
        return (time.time() - getattr(self, 'last_focus_time', 0.0)) < window_sec

    def get_fast_focus_frame(self, max_width=640, quality=70):
        """
        Returns an optimized, downscaled JPEG thumbnail of the latest live view frame
        to eliminate socket lag and reduce browser decompression memory during fast manual focus adjustments.
        Caches the encoded thumbnail per frame_id.
        """
        with self.frame_lock:
            raw_frame = self.latest_frame
            f_id = self.frame_id
            if not raw_frame:
                return None
            if getattr(self, '_cached_thumb_frame_id', -1) == f_id and getattr(self, '_fast_focus_thumbnail_cache', None) is not None:
                return self._fast_focus_thumbnail_cache

        try:
            with Image.open(io.BytesIO(raw_frame)) as img:
                w, h = img.size
                if w > max_width:
                    scale = max_width / float(w)
                    new_h = int(h * scale)
                    thumb_img = img.resize((max_width, new_h), Image.Resampling.BILINEAR)
                else:
                    thumb_img = img.copy()

                buf = io.BytesIO()
                thumb_img.save(buf, format='JPEG', quality=quality, optimize=False)
                thumb_bytes = buf.getvalue()

            with self.frame_lock:
                self._fast_focus_thumbnail_cache = thumb_bytes
                self._cached_thumb_frame_id = f_id
            return thumb_bytes
        except Exception:
            return raw_frame

    def _publish_frame(self, frame_bytes):
        if frame_bytes:
            with self.frame_lock:
                self.latest_frame = frame_bytes
                self.frame_id += 1
            self.frame_event.set()
            self.frame_event.clear()

    # Private loop running on worker thread
    def _worker_loop(self):
        self.log("Worker loop entering active state.")
        last_detect_attempt = 0.0
        while not self.stop_event.is_set():
            # 1. Execute commands queued by Flask threads FIRST
            try:
                while not self.cmd_queue.empty():
                    cmd, args, resp_q = self.cmd_queue.get_nowait()
                    try:
                        res = self._handle_worker_cmd(cmd, args)
                        resp_q.put((True, res))
                    except Exception as e:
                        resp_q.put((False, str(e)))
            except queue.Empty:
                pass

            now = time.time()
            # 2. If physical camera is not connected and not in forced simulated mode, attempt discovery
            if not self.simulated and not self.camera_connected:
                if now - last_detect_attempt >= 2.0:
                    last_detect_attempt = now
                    self._try_connect_physical_camera()
                if not self.camera_connected:
                    time.sleep(0.15)
                    continue

            # Viewfinder state transition for Nikon/Canon/Windows Bridge
            if self.camera_connected:
                if self.windows_bridge and self.windows_bridge.camera_connected:
                    self.connection_state = "streaming" if self.live_view_active else "connected"
                elif self.gphoto2_backend.camera_connected:
                    if self.live_view_active and not self.gphoto2_backend._physical_viewfinder_active:
                        success = self.gphoto2_backend.set_viewfinder(1, on_frame_callback=self._publish_frame)
                        if success:
                            self.connection_state = "streaming"
                    elif not self.live_view_active and self.gphoto2_backend._physical_viewfinder_active:
                        self.gphoto2_backend.set_viewfinder(0)
                        self.connection_state = "connected"

            # 3. Grab preview frame if live view is active and not paused for setting update
            if self.live_view_active and not self.pause_preview:
                try:
                    frame = self._grab_preview_frame()
                    if frame:
                        self._publish_frame(frame)
                except Exception as e:
                    err_str = str(e)
                    if "-52" in err_str or "Could not find the requested device" in err_str:
                        self.log("Camera device disconnected or reset during Live View ([-52]). Resetting connection state...")
                        self.disconnect()
                        self.connection_state = "searching"
                        time.sleep(1.0)
                    elif "-110" in err_str or "I/O in progress" in err_str:
                        time.sleep(0.1)
                    else:
                        self.log(f"Live view preview frame warning: {e}")
                        time.sleep(0.04)

            # 4. Poll hardware shutter events & periodic body dial settings when Live View is idle
            if self.camera_connected and self.gphoto2_backend.camera_connected and not self.live_view_active:
                self.gphoto2_backend.poll_events_and_settings(now)

            # Avoid tight loop when idle
            if not self.live_view_active:
                time.sleep(0.08)
            else:
                time.sleep(0.015)

        self.log("Worker loop exited.")

    def _try_connect_physical_camera(self):
        # 1. On Windows, attempt connection via native camera bridge (EDSDK)
        if sys.platform == 'win32' and self.windows_bridge and not self.simulated:
            try:
                if self.windows_bridge.connect():
                    self.camera_connected = True
                    self.simulated = False
                    self.model_name = self.windows_bridge.model_name
                    self.connection_state = "connected"
                    self.camera_settings = self.windows_bridge.get_settings()
                    self.camera_choices = self.windows_bridge.get_choices()
                    self.log(f"Successfully connected to camera via Windows Native Bridge: {self.model_name}")
                    return
            except Exception as e:
                self.log(f"Windows camera bridge connection attempt failed: {e}")

        if not GPHOTO2_AVAILABLE or self.simulated:
            self.simulated = True
            self.connection_state = "simulated"
            return

        # 2. Attempt connection via libgphoto2 backend
        if self.gphoto2_backend.connect():
            self.camera_connected = True
            self.simulated = False
            self.model_name = self.gphoto2_backend.model_name
            self.connection_state = "connected"
            self.camera_settings = self.gphoto2_backend.camera_settings
            self.camera_choices = self.gphoto2_backend.camera_choices
        else:
            self.camera_connected = False
            self.connection_state = "searching"

    def _handle_worker_cmd(self, cmd, args):
        if cmd == "get_status":
            if self.simulated:
                return {
                    "connected": True,
                    "simulated": True,
                    "settings": self.sim_settings,
                    "choices": self.sim_choices
                }
            if not self.camera_connected:
                return {
                    "connected": False,
                    "simulated": False,
                    "settings": {},
                    "choices": {}
                }
            if self.windows_bridge and self.windows_bridge.camera_connected:
                return {
                    "connected": True,
                    "simulated": False,
                    "settings": self.windows_bridge.get_settings(),
                    "choices": self.windows_bridge.get_choices()
                }
            return {
                "connected": True,
                "simulated": False,
                "settings": self.gphoto2_backend.get_settings(),
                "choices": self.gphoto2_backend.get_choices()
            }

        elif cmd == "reconnect":
            self.disconnect()
            self.simulated = False
            self.camera_connected = False
            self._try_connect_physical_camera()
            return self.camera_connected

        elif cmd == "set_config":
            name = args["name"].lower()
            val = args["value"]
            if name == "manualfocusdrive":
                self.notify_focus_adjustment()
            if self.simulated:
                self.simulated_backend.camera_settings = self.sim_settings
                return self.simulated_backend.set_setting(name, val)
            elif self.windows_bridge and self.windows_bridge.camera_connected:
                if name == "manualfocusdrive":
                    coarse = (str(val) in ["3", "-3", "near_coarse", "far_coarse"])
                    direction = "near" if ("-" in str(val) or "near" in str(val).lower()) else "far"
                    return self.windows_bridge.step_focus(direction, coarse=coarse)
                else:
                    return self.windows_bridge.set_setting(name, val)
            else:
                self.pause_preview = True
                time.sleep(0.15)
                try:
                    return self.gphoto2_backend.set_setting(name, val)
                except Exception as e:
                    self.log(f"Error setting camera property '{name}' to '{val}': {e}")
                    raise e
                finally:
                    self.pause_preview = False

        elif cmd == "test_widgets":
            if self.simulated or not self.gphoto2_backend.camera_connected:
                return {"error": "simulated mode"}
            return self.gphoto2_backend.test_widgets()

        elif cmd == "dump_config":
            if self.simulated or not self.gphoto2_backend.camera_connected:
                return ["Simulated Mode Active - No Physical Config"]
            return self.gphoto2_backend.dump_config()

        elif cmd == "dump_config_values":
            if self.simulated or not self.gphoto2_backend.camera_connected:
                return {"error": "simulated mode"}
            return self.gphoto2_backend.dump_config_values()

        elif cmd == "capture":
            autofocus = args.get("autofocus", True)
            if self.windows_bridge and self.windows_bridge.camera_connected:
                self.pause_preview = True
                self.connection_state = "capturing"
                try:
                    return self.windows_bridge.capture_image(autofocus=autofocus)
                finally:
                    self.pause_preview = False
                    self.connection_state = "streaming" if self.live_view_active else "connected"
            elif self.simulated or not self.camera_connected or not self.gphoto2_backend.camera_connected:
                self.simulated_backend.camera_settings = self.sim_settings
                self.simulated_backend.mock_leds = self.mock_leds
                return self.simulated_backend.capture_image(autofocus=autofocus)
            else:
                self.pause_preview = True
                self.connection_state = "capturing"
                try:
                    return self.gphoto2_backend.capture_image(autofocus=autofocus)
                finally:
                    self.pause_preview = False
                    self.connection_state = "streaming" if self.live_view_active else "connected"

        elif cmd == "autofocus":
            self.notify_focus_adjustment()
            if self.simulated:
                return self.simulated_backend.trigger_autofocus()
            elif self.windows_bridge and self.windows_bridge.camera_connected:
                return self.windows_bridge.trigger_autofocus()
            else:
                self.pause_preview = True
                try:
                    return self.gphoto2_backend.trigger_autofocus()
                finally:
                    self.pause_preview = False

        raise ValueError(f"Unknown worker command: {cmd}")

    def _grab_preview_frame(self):
        if self.simulated:
            self.simulated_backend.camera_settings = self.sim_settings
            self.simulated_backend.mock_leds = self.mock_leds
            return self.simulated_backend.grab_preview_frame()

        if self.windows_bridge and self.windows_bridge.camera_connected:
            frame = self.windows_bridge.grab_preview_frame()
            if frame:
                return frame
            time.sleep(0.02)
            return None

        return self.gphoto2_backend.grab_preview_frame()
