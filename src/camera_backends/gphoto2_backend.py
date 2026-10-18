import os
import re
import sys
import glob
import time
import shutil
import threading
import subprocess
from typing import Dict, Any, Optional
from .base import BaseCameraBackend
from metadata_preservation import get_staged_raw_path


def _configure_gphoto2_env():
    if 'CAMLIBS' not in os.environ or 'IOLIBS' not in os.environ:
        search_roots = []
        if hasattr(sys, '_MEIPASS'):
            search_roots.append(os.path.join(sys._MEIPASS, 'gphoto2', 'libgphoto2'))
            search_roots.append(os.path.join(sys._MEIPASS, 'gphoto2'))
            search_roots.append(sys._MEIPASS)

        # Repository root is two levels up from src/camera_backends/gphoto2_backend.py
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        search_roots.append(os.path.join(repo_root, 'src', 'vendor', 'gphoto2_mac'))
        search_roots.append(os.path.join(repo_root, 'vendor', 'gphoto2_mac'))
        search_roots.extend(['/opt/homebrew/lib', '/usr/local/lib', '/usr/lib64', '/usr/lib'])

        for root in search_roots:
            if not os.path.exists(root):
                continue
            if 'CAMLIBS' not in os.environ:
                direct_cam = os.path.join(root, 'camlibs')
                if os.path.isdir(direct_cam) and (os.path.exists(os.path.join(direct_cam, 'ptp2.so')) or os.path.exists(os.path.join(direct_cam, 'canon.so'))):
                    os.environ['CAMLIBS'] = direct_cam
                else:
                    matches = glob.glob(os.path.join(root, 'libgphoto2', '*')) + glob.glob(os.path.join(root, 'camlibs', '*'))
                    for m in sorted(matches, reverse=True):
                        if os.path.isdir(m) and (os.path.exists(os.path.join(m, 'ptp2.so')) or os.path.exists(os.path.join(m, 'canon.so'))):
                            os.environ['CAMLIBS'] = m
                            break
            if 'IOLIBS' not in os.environ:
                direct_io = os.path.join(root, 'iolibs')
                if os.path.isdir(direct_io) and (os.path.exists(os.path.join(direct_io, 'usb1.so')) or os.path.exists(os.path.join(direct_io, 'disk.so'))):
                    os.environ['IOLIBS'] = direct_io
                else:
                    matches = (
                        glob.glob(os.path.join(root, 'libgphoto2_port', '*'))
                        + glob.glob(os.path.join(root, 'libgphoto2', 'iolibs'))
                        + glob.glob(os.path.join(root, 'iolibs', '*'))
                    )
                    for m in sorted(matches, reverse=True):
                        if os.path.isdir(m) and (os.path.exists(os.path.join(m, 'usb1.so')) or os.path.exists(os.path.join(m, 'disk.so'))):
                            os.environ['IOLIBS'] = m
                            break

            if sys.platform == 'darwin':
                lib_dir = os.path.join(root, 'lib') if os.path.isdir(os.path.join(root, 'lib')) else root
                cur_dyld = os.environ.get('DYLD_FALLBACK_LIBRARY_PATH', '')
                if lib_dir not in cur_dyld:
                    os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = f"{lib_dir}:{cur_dyld}" if cur_dyld else lib_dir


_configure_gphoto2_env()

# Try to import gphoto2. If not installed, we fallback to simulated mode or Windows bridge.
try:
    import gphoto2 as gp
    GPHOTO2_AVAILABLE = True
except ImportError as e:
    import traceback
    print(f"[Camera] Warning: Failed to import python-gphoto2 bindings: {e}")
    traceback.print_exc()
    gp = None
    GPHOTO2_AVAILABLE = False


class GPhoto2CameraBackend(BaseCameraBackend):
    """
    libgphoto2 camera backend for macOS and Linux.
    Encapsulates USB PTP discovery, Canon/Nikon/Sony widget probing, Live View preview streaming,
    remote shutter release, and camera storage downloads.
    """

    def __init__(self, log_callback=None, session_manager=None):
        super().__init__(log_callback=log_callback)
        self.session_manager = session_manager
        self.camera = None
        self.lock = threading.Lock()
        self.is_canon = False
        self.is_nikon = False
        self.is_sony = False
        self._daemons_suspended = False
        self._physical_viewfinder_active = False
        self._consecutive_preview_errors = 0
        self._cached_dcim_folder = None
        self._captures_to_ram = False
        self._pending_file_events = []
        self._last_settings_poll = 0.0

        # Resolved widget names mapping (probed dynamically upon connection)
        self.resolved_names = {
            "iso": "iso",
            "aperture": "aperture",
            "shutterspeed": "shutterspeed",
            "manualfocusdrive": "manualfocusdrive",
            "eosremoterelease": "eosremoterelease",
            "autofocusdrive": "autofocusdrive",
            "focusmode": "focusmode"
        }

    def log(self, message: str):
        if self.log_callback:
            self.log_callback(message)

    def _drain_events_locked(self, max_events: int = 5, timeout_ms: int = 25):
        """
        Drain pending PTP events while holding self.lock.
        Crucially preserves any GP_EVENT_FILE_ADDED events in self._pending_file_events
        so shutter release or property updates never accidentally swallow a captured frame.
        """
        if not self.camera or not GPHOTO2_AVAILABLE:
            return
        try:
            for _ in range(max_events):
                evt_type, evt_data = self.camera.wait_for_event(timeout_ms)
                if evt_type == gp.GP_EVENT_TIMEOUT:
                    break
                if evt_type == gp.GP_EVENT_FILE_ADDED and evt_data:
                    self._pending_file_events.append((evt_data.folder, evt_data.name))
        except Exception:
            pass

    def connect(self) -> bool:
        if not GPHOTO2_AVAILABLE:
            self.camera_connected = False
            self.connection_state = "disconnected"
            return False

        try:
            # On macOS, claim exclusive access by suspending icdd and terminating ptpcamerad
            if sys.platform == 'darwin':
                try:
                    subprocess.run(["killall", "-STOP", "icdd"], capture_output=True)
                    subprocess.run(["killall", "-9", "ptpcamerad"], capture_output=True)
                    time.sleep(0.15)
                    subprocess.run(["killall", "-STOP", "ptpcamerad"], capture_output=True)
                    self._daemons_suspended = True
                except Exception:
                    pass
                time.sleep(0.3)

            cl = gp.Camera.autodetect()
            if len(cl) == 0:
                self.camera_connected = False
                self.connection_state = "searching"
                return False

            name, port_path = cl.get_name(0), cl.get_value(0)
            self.log(f"Autodetected device '{name}' on port '{port_path}'. Binding driver abilities...")

            name_lower = name.lower()
            self.is_canon = "canon" in name_lower
            self.is_nikon = "nikon" in name_lower
            self.is_sony = "sony" in name_lower
            self.model_name = name

            port_info_list = gp.PortInfoList()
            port_info_list.load()
            port_idx = port_info_list.lookup_path(port_path)
            port_info = port_info_list[port_idx]

            abilities_list = gp.CameraAbilitiesList()
            abilities_list.load()

            model_indices = []
            ab_idx = abilities_list.lookup_model(name)
            if ab_idx >= 0:
                model_indices.append((name, ab_idx))

            ptp_idx = abilities_list.lookup_model('USB PTP Class Camera')
            if ptp_idx >= 0 and ptp_idx != ab_idx:
                model_indices.append(('USB PTP Class Camera', ptp_idx))

            last_init_err = None
            camera = None
            for model_label, idx in model_indices:
                for attempt in range(3):
                    cam_try = None
                    try:
                        self.log(f"Attempting camera init with driver profile '{model_label}' (index {idx}, attempt {attempt + 1}/3)...")
                        cam_try = gp.Camera()
                        cam_try.set_abilities(abilities_list[idx])
                        cam_try.set_port_info(port_info)
                        with self.lock:
                            cam_try.init()
                        camera = cam_try
                        self.log(f"Successfully initialized camera using driver profile '{model_label}'")
                        break
                    except Exception as err:
                        self.log(f"Init with driver profile '{model_label}' (attempt {attempt + 1}) failed: {err}")
                        if cam_try:
                            try:
                                with self.lock:
                                    cam_try.exit()
                            except Exception:
                                pass
                        last_init_err = err
                        if sys.platform == 'darwin':
                            subprocess.run(["killall", "-9", "ptpcamerad"], capture_output=True)
                            time.sleep(0.15)
                            subprocess.run(["killall", "-STOP", "ptpcamerad"], capture_output=True)
                        elif sys.platform.startswith('linux'):
                            try:
                                subprocess.run(["gio", "mount", "-s", "gphoto2"], capture_output=True, timeout=2)
                                subprocess.run(["killall", "-9", "gvfsd-gphoto2", "gvfs-gphoto2-volume-monitor"], capture_output=True, timeout=2)
                            except Exception:
                                pass
                        time.sleep(0.3)
                if camera:
                    break

            if not camera:
                raise last_init_err or Exception("Failed to initialize camera with any driver profile.")

            self.camera = camera
            self.camera_connected = True
            self.connection_state = "connected"
            try:
                summary_line = camera.get_summary().text.splitlines()[0]
            except Exception:
                summary_line = name
            self.log(f"Successfully connected to camera: {summary_line}")

            # Probe and resolve setting widget names (Canon vs Nikon)
            probe_targets = [
                ("iso", ["iso", "eosiso"]),
                ("aperture", ["aperture", "f-number", "fnumber"]),
                ("shutterspeed", ["shutterspeed", "shutterspeed2"]),
                ("manualfocusdrive", ["manualfocusdrive"]),
                ("eosremoterelease", ["eosremoterelease"]),
                ("autofocusdrive", ["autofocusdrive"]),
                ("focusmode", ["focusmode", "focus_mode", "lensfocusmode", "canonfocusmode"]),
                ("eoszoom", ["eoszoom", "zoom", "canonzoom", "eoszoomposition"]),
                ("movieservoaf", ["movieservoaf"]),
                ("continuousaf", ["continuousaf"])
            ]
            with self.lock:
                for key, candidates in probe_targets:
                    for candidate in candidates:
                        try:
                            camera.get_single_config(candidate)
                            self.resolved_names[key] = candidate
                            self.log(f"Resolved camera setting '{key}' to widget '{candidate}'")
                            break
                        except Exception:
                            pass
            self._physical_viewfinder_active = False

            # Reset viewfinder on startup ONLY for Nikon / cameras that explicitly expose a single viewfinder widget
            if not self.is_canon:
                try:
                    with self.lock:
                        viewfinder = camera.get_single_config("viewfinder")
                        viewfinder.set_value(0)
                        camera.set_single_config("viewfinder", viewfinder)
                        self.log("Reset viewfinder to 0 on startup.")
                except Exception:
                    pass

            # Prevent camera from auto-powering off every 60 seconds while tethered
            for ap_widget in ["autopoweroff", "auto_power_off"]:
                try:
                    with self.lock:
                        ap = camera.get_single_config(ap_widget)
                        valid_c = [str(ap.get_choice(i)) for i in range(ap.count_choices())]
                        if '0' in valid_c:
                            self._set_widget_value_safely(ap, '0')
                        elif '1800' in valid_c:
                            self._set_widget_value_safely(ap, '1800')
                        camera.set_single_config(ap_widget, ap)
                        self.log(f"Configured camera {ap_widget} to prevent idle sleep while tethered.")
                        break
                except Exception:
                    pass

            # Query and cache settings and choices
            queried_settings = self.get_settings()
            queried_choices = self.get_choices()
            for k, v in queried_settings.items():
                if v and v != "Unknown":
                    self.camera_settings[k] = v
            for k, v in queried_choices.items():
                if v:
                    self.camera_choices[k] = v
            self.log(f"Initialized physical camera settings: {self.camera_settings}")
            return True

        except Exception as e:
            if self.camera:
                try:
                    with self.lock:
                        self.camera.exit()
                except Exception:
                    pass
            self.camera = None
            self.camera_connected = False
            self.connection_state = "searching"
            self.log(f"Camera connection attempt failed ({e}). Will retry in background...")
            return False

    def disconnect(self):
        if self.camera:
            try:
                self.set_viewfinder(0)
                with self.lock:
                    self.camera.exit()
            except Exception:
                pass
            self.camera = None

        self.camera_connected = False
        self._physical_viewfinder_active = False
        self._cached_dcim_folder = None
        self._captures_to_ram = False
        self._pending_file_events.clear()
        self.connection_state = "disconnected"

        # Safely resume macOS icdd daemon if suspended
        if sys.platform == 'darwin' and self._daemons_suspended:
            try:
                subprocess.run(["killall", "-CONT", "icdd"], capture_output=True)
                self._daemons_suspended = False
            except Exception:
                pass

    def grab_preview_frame(self) -> Optional[bytes]:
        if not self.camera:
            return None
        retries = 3
        last_err = None
        for _ in range(retries):
            try:
                # Capture actual preview (thread-safe lock to prevent USB bus collision)
                with self.lock:
                    camera_file = self.camera.capture_preview()
                    file_data = camera_file.get_data_and_size()
                    self._consecutive_preview_errors = 0
                    return memoryview(file_data).tobytes()
            except Exception as e:
                last_err = e
                # Drain pending events to unlock USB pipe
                with self.lock:
                    self._drain_events_locked(max_events=3, timeout_ms=25)
                time.sleep(0.1)

        self._consecutive_preview_errors += 1
        if self._consecutive_preview_errors == 10:
            self.log(f"Live view preview struggling with camera I/O: {last_err}")
        raise last_err

    def set_viewfinder(self, val: int, on_frame_callback=None) -> bool:
        if not self.camera:
            return False
        try:
            val_int = int(val)
            val_str = str(val)

            # 1. On Canon mirrorless cameras, live view stream is driven directly by capture_preview().
            # Trying to set non-existent 'viewfinder' widgets causes slow PTP timeouts.
            if self.is_canon:
                if val_int == 1:
                    try:
                        with self.lock:
                            camera_file = self.camera.capture_preview()
                            file_data = camera_file.get_data_and_size()
                            frame_bytes = memoryview(file_data).tobytes()
                        if on_frame_callback:
                            on_frame_callback(frame_bytes)
                        self._physical_viewfinder_active = True
                        self.log("Canon Live View stream initiated via capture_preview.")
                        return True
                    except Exception as e:
                        self.log(f"Notice: Canon capture_preview init: {e}")
                        return False
                else:
                    self._physical_viewfinder_active = False
                    self.log("Canon Live View stream stopped.")
                    return True

            # 2. Fast path for Nikon / other cameras using get_single_config
            for candidate in ["viewfinder", "evf_status"]:
                try:
                    with self.lock:
                        viewfinder = self.camera.get_single_config(candidate)
                        try:
                            viewfinder.set_value(val_int)
                        except Exception:
                            viewfinder.set_value(val_str)
                        self.camera.set_single_config(candidate, viewfinder)
                    self._physical_viewfinder_active = (val_int == 1)
                    self.log(f"Set {candidate} to {val} using get_single_config")
                    return True
                except Exception:
                    pass

            # 3. Fallback: If turning ON (val == 1) and widget single_config was not found, kickstart via capture_preview
            if val_int == 1:
                try:
                    with self.lock:
                        camera_file = self.camera.capture_preview()
                        file_data = camera_file.get_data_and_size()
                        frame_bytes = memoryview(file_data).tobytes()
                    if on_frame_callback:
                        on_frame_callback(frame_bytes)
                    self._physical_viewfinder_active = True
                    self.log("Live View stream kickstarted via capture_preview.")
                    return True
                except Exception as e:
                    self.log(f"Notice: capture_preview kickstart: {e}")

        except Exception as e:
            self.log(f"Notice setting viewfinder to {val}: {e}")
        return False

    def capture_image(self, autofocus: bool = True) -> Optional[str]:
        if not self.camera:
            raise Exception("Physical gphoto2 camera is not connected.")

        # Reset viewfinder state so EVF stream is re-engaged cleanly after capture
        self._physical_viewfinder_active = False

        # 1. Drain residual USB preview packets so bus is idle before setting configs or triggering shutter
        with self.lock:
            self._drain_events_locked(max_events=6, timeout_ms=20)

        # 2. Disengage hardware sensor zoom (eoszoom) if active to unlock camera shutter
        if self.resolved_names.get("eoszoom"):
            try:
                cur_zoom = str(self.camera_settings.get("eoszoom", "0")).lower()
                if cur_zoom not in ('0', 'off', 'none'):
                    self.log("Disengaging hardware zoom (eoszoom = 0) prior to capture to unlock shutter...")
                    self._set_camera_property("eoszoom", 0)
                    self.camera_settings["eoszoom"] = "0"
                    time.sleep(0.15)
            except Exception as zoom_err:
                self.log(f"Notice: Pre-capture eoszoom check: {zoom_err}")

        # 3. Snapshot DCIM directory on camera storage ONLY when camera does not capture to SDRAM
        dcim_dir = None
        initial_dcim_files = set()
        last_dcim_file = None
        predicted_candidates = []

        if not self._captures_to_ram:
            dcim_dir = self._get_dcim_folder()
            if dcim_dir:
                try:
                    with self.lock:
                        f_list = self.camera.folder_list_files(dcim_dir)
                        f_count = f_list.count()
                        initial_dcim_files = {f_list.get_name(i) for i in range(f_count)}
                        if f_count > 0:
                            last_dcim_file = f_list.get_name(f_count - 1)
                    self.log(f"Pre-capture DCIM snapshot ({dcim_dir}): {len(initial_dcim_files)} files. Last: {last_dcim_file}")
                except Exception as list_err:
                    self.log(f"Notice: Pre-capture DCIM listing: {list_err}")
            else:
                self.log("Notice: DCIM folder not located prior to shutter release.")

            if last_dcim_file:
                pred = self._predict_next_filename(last_dcim_file)
                if pred:
                    predicted_candidates.append(pred)
                    base_root = os.path.splitext(pred)[0]
                    for alt_ext in [".CR3", ".cr3", ".JPG", ".jpg"]:
                        cand = f"{base_root}{alt_ext}"
                        if cand not in predicted_candidates:
                            predicted_candidates.append(cand)
                self.log(f"Predicted next capture file candidate(s): {predicted_candidates}")

        file_path_info = None
        result_path = None

        if self.resolved_names.get("eosremoterelease"):
            # Ensure release state is clean
            for rel in ["Release", "Release Full", "Release Half"]:
                try:
                    self._set_camera_property("eosremoterelease", rel, drain_events=False)
                    break
                except Exception:
                    pass

            # For scanning film negatives, shutter release MUST NOT activate AF hunting
            # Canon EOS remote release: Press Half MF (metering only) -> Press Full MF (shutter trip)
            self.log("Triggering camera shutter via eosremoterelease (Press Half MF -> Press Full MF)...")
            self._set_camera_property("eosremoterelease", "Press Half MF", drain_events=False)
            time.sleep(0.08)  # Hold half-press briefly

            self._set_camera_property("eosremoterelease", "Press Full MF", drain_events=False)
            time.sleep(0.15)  # Hold full-press to trip shutter mechanism

            # Release shutter switches (without draining FILE_ADDED events!)
            for rel in ["Release", "Release Full", "Release Half"]:
                try:
                    self._set_camera_property("eosremoterelease", rel, drain_events=False)
                    break
                except Exception:
                    pass

            # Wait for capture event from camera or retrieve newly written file from storage
            t0 = time.time()
            last_pred_check = t0
            while time.time() - t0 < 10.0:
                # 0. Check if any FILE_ADDED event was captured during property retry drains
                if self._pending_file_events:
                    ev_folder, ev_name = self._pending_file_events.pop(0)
                    file_path_info = (ev_folder, ev_name)
                    if ev_folder in ("/", "//") or ev_name.lower().startswith("capt"):
                        self._captures_to_ram = True
                    self.log(f"Capture event detected: {file_path_info[0]}/{file_path_info[1]}")
                    result_path = self._download_camera_file(ev_folder, ev_name)
                    break

                # 1. Wait for PTP event (returns immediately on event, or after 120ms timeout)
                with self.lock:
                    try:
                        event_type, event_data = self.camera.wait_for_event(120)
                    except Exception:
                        event_type, event_data = gp.GP_EVENT_TIMEOUT, None

                if event_type == gp.GP_EVENT_FILE_ADDED and event_data:
                    file_path_info = (event_data.folder, event_data.name)
                    if event_data.folder in ("/", "//") or event_data.name.lower().startswith("capt"):
                        self._captures_to_ram = True
                    self.log(f"Capture event detected: {file_path_info[0]}/{file_path_info[1]}")
                    result_path = self._download_camera_file(event_data.folder, event_data.name)
                    break

                # 2. Fallback: direct storage file retrieval ONLY if camera is not capturing to RAM
                # and no event has arrived after 2.0s (polled at most once per 1.0s to avoid USB bus contention)
                now_ts = time.time()
                if (
                    not self._captures_to_ram
                    and dcim_dir
                    and predicted_candidates
                    and (now_ts - t0 >= 2.0)
                    and (now_ts - last_pred_check >= 1.0)
                ):
                    last_pred_check = now_ts
                    for cand_name in predicted_candidates:
                        try:
                            result_path = self._download_camera_file(dcim_dir, cand_name, quiet=True)
                            self.log(f"Successfully matched and downloaded predicted capture: {cand_name}")
                            file_path_info = (dcim_dir, cand_name)
                            break
                        except Exception:
                            pass
                    if file_path_info:
                        break

            if not result_path:
                raise Exception("Camera shutter release did not produce an image file on storage within timeout.")
        else:
            self.log("Triggering camera capture via camera.capture()...")
            with self.lock:
                file_path = self.camera.capture(gp.GP_CAPTURE_IMAGE)
            if file_path.folder in ("/", "//") or file_path.name.lower().startswith("capt"):
                self._captures_to_ram = True
            self.log(f"Capture successful. File created on camera: {file_path.folder}/{file_path.name}")
            result_path = self._download_camera_file(file_path.folder, file_path.name)

        # Post-capture event draining to clear remaining PTP notifications (like CAPTURE_COMPLETE)
        with self.lock:
            self._drain_events_locked(max_events=6, timeout_ms=20)
        time.sleep(0.05)
        return result_path

    def step_focus(self, direction: str, coarse: bool = False) -> bool:
        step_val = f"{direction.capitalize()} {'3' if coarse else '1'}"
        return self.set_setting("manualfocusdrive", step_val)

    def trigger_autofocus(self) -> bool:
        if not self.camera:
            raise Exception("Physical gphoto2 camera is not connected.")

        if self.resolved_names.get("eosremoterelease"):
            self.log("Triggering autofocus via eosremoterelease (Canon)...")
            for rel in ["Release", "Release Half"]:
                try:
                    self._set_camera_property("eosremoterelease", rel)
                    break
                except Exception:
                    pass
            self._set_camera_property("eosremoterelease", "Press Half AF")
            time.sleep(1.2)
            for rel in ["Release", "Release Half"]:
                try:
                    self._set_camera_property("eosremoterelease", rel)
                    break
                except Exception:
                    pass
        elif self.resolved_names.get("autofocusdrive"):
            self.log("Triggering autofocus via autofocusdrive (Nikon/Generic)...")
            self._set_camera_property("autofocusdrive", 1)
        else:
            raise Exception("Autofocus is not supported or resolved for this camera model.")

        with self.lock:
            self._drain_events_locked(max_events=5, timeout_ms=40)
        time.sleep(0.1)
        return True

    def get_settings(self) -> Dict[str, str]:
        if not self.camera:
            return {}
        settings = {}
        for key in ["iso", "aperture", "shutterspeed"]:
            widget_name = self.resolved_names.get(key)
            if not widget_name:
                settings[key] = "Unknown"
                continue
            try:
                widget = self.camera.get_single_config(widget_name)
                settings[key] = str(widget.get_value())
            except Exception as e:
                self.log(f"Error querying setting '{key}' (widget '{widget_name}'): {e}")
                settings[key] = "Unknown"
        return settings

    def get_choices(self) -> Dict[str, list]:
        if not self.camera:
            return {}
        choices = {}
        for key in ["iso", "aperture", "shutterspeed"]:
            widget_name = self.resolved_names.get(key)
            if not widget_name:
                choices[key] = []
                continue
            try:
                widget = self.camera.get_single_config(widget_name)
                opt_list = []
                for i in range(widget.count_choices()):
                    opt_list.append(str(widget.get_choice(i)))
                choices[key] = opt_list
            except Exception as e:
                self.log(f"Error querying choices for '{key}' (widget '{widget_name}'): {e}")
                choices[key] = []
        return choices

    def set_setting(self, name: str, value: str) -> bool:
        if not self.camera:
            raise Exception("Physical gphoto2 camera is not connected.")

        # Drain any pending PTP events to clear USB bus before updating property
        with self.lock:
            self._drain_events_locked(max_events=5, timeout_ms=25)
        self._set_camera_property(name, value)
        self.camera_settings[name] = value
        return True

    def poll_events_and_settings(self, now: float):
        """Check for hardware shutter events and periodically poll body dial settings when Live View is idle."""
        if not self.camera_connected or not self.camera:
            return

        # 1. Check for camera events (e.g., photo taken via hardware remote)
        try:
            with self.lock:
                event_type, event_data = self.camera.wait_for_event(20)
            if event_type == gp.GP_EVENT_FILE_ADDED:
                self.log(f"Hardware shutter event detected! File added: {event_data.name}")
                self._download_camera_file(event_data.folder, event_data.name)
        except Exception:
            pass

        # 2. Periodically update physical settings cache to capture body dial changes
        if now - self._last_settings_poll > 5.0:
            try:
                polled = self.get_settings()
                for k, v in polled.items():
                    if v and v != "Unknown":
                        self.camera_settings[k] = v
                self._last_settings_poll = now
            except Exception:
                pass

    def test_widgets(self) -> Dict[str, Any]:
        if not self.camera:
            return {"error": "simulated mode"}
        config = self.camera.get_config()
        res = {}
        keep_alive = [config]
        for name in ["iso", "aperture", "shutterspeed"]:
            widget = self._get_setting_widget(name, config, keep_alive)
            res[name] = {
                "found": widget is not None,
                "name": widget.get_name() if widget else None,
                "value": str(widget.get_value()) if widget else None,
                "choices_count": widget.count_choices() if widget else 0
            }
        return res

    def dump_config(self) -> list:
        if not self.camera:
            return ["Simulated Mode Active - No Physical Config"]
        config = self.camera.get_config()
        names = []

        def dump(w):
            names.append(w.get_name())
            for i in range(w.count_children()):
                dump(w.get_child(i))

        dump(config)
        return names

    def dump_config_values(self) -> Dict[str, Any]:
        if not self.camera:
            return {"error": "simulated mode"}
        try:
            config = self.camera.get_config()
            values = {}

            def traverse(w, path=None):
                if path is None:
                    path = []
                name = w.get_name()
                new_path = path + [name]
                try:
                    val = w.get_value()
                    if val is not None:
                        choices = []
                        try:
                            for i in range(w.count_choices()):
                                choices.append(str(w.get_choice(i)))
                        except Exception:
                            pass
                        values["/".join(new_path)] = {
                            "value": str(val),
                            "choices": choices
                        }
                except Exception:
                    pass
                for i in range(w.count_children()):
                    traverse(w.get_child(i), new_path)

            traverse(config)
            return values
        except Exception as e:
            return {"error": str(e)}

    # --- Internal gphoto2 helpers ---

    def _get_dcim_folder(self):
        if self._cached_dcim_folder:
            return self._cached_dcim_folder
        if not self.camera:
            return None
        try:
            with self.lock:
                root_folders = self.camera.folder_list_folders('/')
                for i in range(root_folders.count()):
                    top = f"/{root_folders.get_name(i)}"
                    try:
                        sub = self.camera.folder_list_folders(top)
                        for j in range(sub.count()):
                            name = sub.get_name(j)
                            if name.upper() == "DCIM":
                                dcim_path = f"{top}/{name}"
                                sub_dcim = self.camera.folder_list_folders(dcim_path)
                                if sub_dcim.count() > 0:
                                    target = f"{dcim_path}/{sub_dcim.get_name(sub_dcim.count() - 1)}"
                                    self._cached_dcim_folder = target
                                    self.log(f"Located camera DCIM storage directory: {target}")
                                    return target
                                self._cached_dcim_folder = dcim_path
                                self.log(f"Located camera DCIM storage directory: {dcim_path}")
                                return dcim_path
                    except Exception as sub_e:
                        self.log(f"Notice inspecting storage top {top}: {sub_e}")
        except Exception as e:
            self.log(f"Notice finding DCIM folder: {e}")
        return None

    def _predict_next_filename(self, filename):
        m = re.match(r'^([A-Za-z0-9_]+?)(\d+)(\.[A-Za-z0-9]+)$', filename)
        if m:
            prefix, num_str, ext = m.groups()
            num = int(num_str)
            num_len = len(num_str)
            next_num = (num + 1) % (10 ** num_len)
            return f"{prefix}{next_num:0{num_len}d}{ext}"
        return None

    def _download_camera_file(self, folder, name, quiet=False):
        try:
            with self.lock:
                camera_file = self.camera.file_get(
                    folder,
                    name,
                    gp.GP_FILE_TYPE_NORMAL
                )
        except Exception as e:
            if not quiet:
                self.log(f"Failed to download file {name}: {e}")
            raise e

        target_dir = None
        if self.session_manager and getattr(self.session_manager, 'dirs', None):
            target_dir = self.session_manager.dirs.get("negatives")

        if not target_dir or not os.path.exists(target_dir):
            target_dir = os.path.abspath("./negatives_download")
            os.makedirs(target_dir, exist_ok=True)

        ext = os.path.splitext(name)[1].lower()
        frame_num, shot_num, is_triplet = self._get_next_capture_naming(target_dir)

        if is_triplet:
            local_name = f"Frame_{frame_num:02d}_Capture_{shot_num:02d}_{int(time.time())}{ext}"
        else:
            local_name = f"Frame_{frame_num:02d}_Capture_{int(time.time())}{ext}"
        local_path = os.path.join(target_dir, local_name)

        if not quiet:
            self.log(f"Downloading {name} to {local_path}...")

        try:
            staged_path = get_staged_raw_path(local_path)
            with self.lock:
                camera_file.save(staged_path)

            part_path = os.path.join(target_dir, f".{local_name}.part")
            try:
                with open(staged_path, "rb") as fsrc, open(part_path, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst, length=4 * 1024 * 1024)
                os.replace(part_path, local_path)
            except Exception:
                if os.path.exists(part_path):
                    try:
                        os.remove(part_path)
                    except OSError:
                        pass
                raise

            self.log(f"Download complete: {local_name}")
            return local_path
        except Exception as e:
            if not quiet:
                self.log(f"Failed to download file {name}: {e}")
            raise e

    def _get_setting_widget(self, name, config, keep_alive):
        paths = {
            "iso": [
                ["main", "imgsettings", "iso"],
                ["main", "capturesettings", "iso"],
                ["main", "settings", "iso"],
                ["main", "imgsettings", "eosiso"],
                ["main", "capturesettings", "eosiso"],
            ],
            "aperture": [
                ["main", "capturesettings", "aperture"],
                ["main", "imgsettings", "aperture"],
                ["main", "settings", "f-number"],
                ["main", "capturesettings", "f-number"],
                ["main", "capturesettings", "fnumber"],
            ],
            "shutterspeed": [
                ["main", "capturesettings", "shutterspeed"],
                ["main", "imgsettings", "shutterspeed"],
                ["main", "settings", "shutterspeed"],
                ["main", "capturesettings", "shutterspeed2"],
            ]
        }

        search_paths = paths.get(name, [[name]])
        for path_components in search_paths:
            current = config
            temp_keep_alive = []
            success = True
            for comp in path_components:
                try:
                    current = current.get_child_by_name(comp)
                    temp_keep_alive.append(current)
                except Exception as e:
                    self.log(f"Path step failed for {name} on component '{comp}' in path {path_components}: {e}")
                    success = False
                    break
            if success:
                keep_alive.extend(temp_keep_alive)
                return current

        widget = self._find_widget_by_name(config, name, keep_alive)
        if widget:
            return widget
        return None

    def _find_widget_by_name(self, parent, name, keep_alive):
        if parent.get_name() == name:
            return parent
        for i in range(parent.count_children()):
            try:
                child = parent.get_child(i)
                keep_alive.append(child)
                res = self._find_widget_by_name(child, name, keep_alive)
                if res:
                    return res
            except Exception:
                pass
        return None

    def _set_widget_value_safely(self, widget, val):
        try:
            w_type = widget.get_type()
        except Exception:
            w_type = None

        if w_type in (gp.GP_WIDGET_RADIO, gp.GP_WIDGET_MENU, gp.GP_WIDGET_TEXT):
            types_to_try = [str, int, float]
        elif w_type in (gp.GP_WIDGET_INT, gp.GP_WIDGET_TOGGLE):
            types_to_try = [int, str, float]
        elif w_type == gp.GP_WIDGET_RANGE:
            types_to_try = [float, int, str]
        else:
            types_to_try = [str, int, float]

        for t in types_to_try:
            try:
                if t == str:
                    widget.set_value(str(val))
                elif t == int:
                    widget.set_value(int(str(val).replace('x', '').strip()))
                elif t == float:
                    widget.set_value(float(str(val).replace('x', '').strip()))
                return True
            except Exception:
                pass
        try:
            widget.set_value(val)
        except Exception:
            pass

    def _set_camera_property(self, name, value, drain_events: bool = True):
        if not self.camera:
            return False

        widget_name = self.resolved_names.get(name.lower())
        if not widget_name:
            widget_name = name.lower()

        widget = None
        for attempt in range(4):
            try:
                with self.lock:
                    widget = self.camera.get_single_config(widget_name)
                break
            except Exception as e:
                if attempt < 3 and ("-110" in str(e) or "I/O in progress" in str(e) or "busy" in str(e).lower()):
                    with self.lock:
                        self._drain_events_locked(max_events=5, timeout_ms=25)
                    time.sleep(0.1)
                else:
                    raise Exception(f"Setting '{name}' (widget '{widget_name}') not supported or found: {e}")

        try:
            widget_type = widget.get_type()
        except Exception:
            widget_type = None

        if widget_type == gp.GP_WIDGET_RANGE and name.lower() == "manualfocusdrive":
            val_str = str(value).lower()
            direction = 1 if "near" in val_str else -1

            speed_match = re.search(r'\d+', val_str)
            speed = int(speed_match.group()) if speed_match else 1

            if speed == 1:
                step = 150
            elif speed == 2:
                step = 1000
            else:
                step = 5000

            matched_choice = direction * step
            self._set_widget_value_safely(widget, matched_choice)
            self.log(f"Mapped manualfocusdrive '{value}' to range value: {matched_choice}")
        else:
            valid_choices = []
            try:
                for i in range(widget.count_choices()):
                    valid_choices.append(str(widget.get_choice(i)))
            except Exception:
                pass

            matched_choice = None
            if name.lower() == "eoszoom":
                v_lower = str(value).lower()
                if v_lower in ['0', '1', 'off', 'normal']:
                    targets = ['0', '1', 'off', 'normal']
                    default_zoom = 0
                elif '5' in v_lower:
                    targets = ['5', '5x']
                    default_zoom = 5
                elif '10' in v_lower:
                    targets = ['10', '10x']
                    default_zoom = 10
                else:
                    targets = [v_lower]
                    default_zoom = 0

                for choice in valid_choices:
                    if str(choice).lower() in targets:
                        matched_choice = choice
                        break
                if matched_choice is None:
                    matched_choice = default_zoom

            if name.lower() == "eosremoterelease":
                v_lower = str(value).strip().lower()
                for choice in valid_choices:
                    c_lower = str(choice).strip().lower()
                    if v_lower == c_lower:
                        matched_choice = choice
                        break
                if matched_choice is None:
                    for choice in valid_choices:
                        c_lower = str(choice).strip().lower()
                        if v_lower.replace(" ", "") == c_lower.replace(" ", ""):
                            matched_choice = choice
                            break

            if matched_choice is None:
                for choice in valid_choices:
                    if str(choice).lower() == str(value).lower():
                        matched_choice = choice
                        break

            if matched_choice is None:
                matched_choice = value
            self._set_widget_value_safely(widget, matched_choice)

        max_retries = 4
        last_err = None
        for attempt in range(max_retries):
            try:
                with self.lock:
                    if drain_events:
                        self._drain_events_locked(max_events=3, timeout_ms=20)
                    self.camera.set_single_config(widget_name, widget)
                self.log(f"Setting updated: {name} (widget '{widget_name}') = {matched_choice}")
                return True
            except Exception as first_err:
                last_err = first_err
                err_str = str(first_err)
                if name.lower() == "eoszoom":
                    self.log(f"Initial set_single_config for eoszoom='{value}' failed: {first_err}. Probing integer variants [0, 1]...")
                    success = False
                    for alt in [0, 1]:
                        try:
                            self._set_widget_value_safely(widget, alt)
                            with self.lock:
                                self.camera.set_single_config(widget_name, widget)
                            matched_choice = alt
                            success = True
                            self.log(f"Successfully applied eoszoom using alternative variant: {alt}")
                            break
                        except Exception:
                            pass
                    if success:
                        return True

                if ("-110" in err_str or "I/O in progress" in str(e if 'e' in locals() else first_err) or "busy" in err_str.lower()) and attempt < max_retries - 1:
                    self.log(f"Setting '{name}' attempt {attempt+1}/{max_retries} encountered I/O in progress. Retrying after event drain...")
                    with self.lock:
                        self._drain_events_locked(max_events=5, timeout_ms=25)
                    time.sleep(0.15)
                else:
                    break

        if last_err:
            self.log(f"Error applying setting '{name}' = '{value}': {last_err}")
            raise last_err
        return True
