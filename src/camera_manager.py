import os
import io
import time
import queue
import threading
from pathlib import Path
from PIL import Image, ImageDraw

def _configure_gphoto2_env():
    import sys
    import glob
    if 'CAMLIBS' not in os.environ or 'IOLIBS' not in os.environ:
        search_roots = []
        if hasattr(sys, '_MEIPASS'):
            search_roots.append(os.path.join(sys._MEIPASS, 'gphoto2'))
            search_roots.append(sys._MEIPASS)
        search_roots.extend(['/opt/homebrew/lib', '/usr/local/lib', '/usr/lib'])

        for root in search_roots:
            if 'CAMLIBS' not in os.environ:
                matches = glob.glob(os.path.join(root, 'libgphoto2', '*')) + glob.glob(os.path.join(root, 'camlibs', '*'))
                for m in sorted(matches, reverse=True):
                    if os.path.isdir(m) and (os.path.exists(os.path.join(m, 'ptp2.so')) or os.path.exists(os.path.join(m, 'canon.so'))):
                        os.environ['CAMLIBS'] = m
                        break
            if 'IOLIBS' not in os.environ:
                matches = glob.glob(os.path.join(root, 'libgphoto2_port', '*')) + glob.glob(os.path.join(root, 'iolibs', '*'))
                for m in sorted(matches, reverse=True):
                    if os.path.isdir(m) and (os.path.exists(os.path.join(m, 'usb1.so')) or os.path.exists(os.path.join(m, 'disk.so'))):
                        os.environ['IOLIBS'] = m
                        break

_configure_gphoto2_env()

# Try to import gphoto2. If not installed, we fallback to simulated mode.
try:
    import gphoto2 as gp
    GPHOTO2_AVAILABLE = True
except ImportError as e:
    import traceback
    print(f"[Camera] Warning: Failed to import python-gphoto2 bindings: {e}")
    traceback.print_exc()
    GPHOTO2_AVAILABLE = False

class CameraManager:
    def __init__(self, session_manager=None):
        self.session_manager = session_manager
        self.simulated = not GPHOTO2_AVAILABLE
        self.camera = None
        self.camera_connected = False
        self.connection_state = "simulated" if self.simulated else "searching"
        self.model_name = "Simulated Camera" if self.simulated else ""
        self.is_canon = False
        self.is_nikon = False
        self.is_sony = False
        self._daemons_suspended = False
        
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
        
        # Simulated mode states
        self.sim_settings = {
            "iso": "400",
            "aperture": "f/8.0",
            "shutterspeed": "1/60",
        }
        self.sim_choices = {
            "iso": ["100", "200", "400", "800", "1600", "3200", "6400"],
            "aperture": ["f/2.8", "f/4.0", "f/5.6", "f/8.0", "f/11", "f/16", "f/22"],
            "shutterspeed": ["1/250", "1/125", "1/60", "1/30", "1/15", "1/8", "1/4", "1/2", "1s", "2s"]
        }
        self.mock_leds = {"red": 255, "green": 255, "blue": 255}
        
        # Physical camera settings cache (initialized with sensible photography fallbacks)
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
        self._physical_viewfinder_active = False
        self._consecutive_preview_errors = 0
        self._cached_dcim_folder = None
        
        # Internal log helper
        self.log_callback = print

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
        if self.camera:
            try:
                self._set_camera_viewfinder(0)
                with self.lock:
                    self.camera.exit()
            except Exception:
                pass
            self.camera = None
        self.camera_connected = False
        self._physical_viewfinder_active = False
        self._cached_dcim_folder = None
        self.connection_state = "simulated" if self.simulated else "disconnected"
        
        # Safely resume macOS icdd daemon if suspended
        import sys
        import subprocess
        if sys.platform == 'darwin' and self._daemons_suspended:
            try:
                subprocess.run(["killall", "-CONT", "icdd"], capture_output=True)
                self._daemons_suspended = False
            except Exception:
                pass
                
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
        self.mock_leds = {"red": r, "green": g, "blue": b}

    def get_latest_frame(self):
        with self.frame_lock:
            return self.latest_frame

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
            if not self.simulated and not self.camera:
                if now - last_detect_attempt >= 2.0:
                    last_detect_attempt = now
                    self._try_connect_physical_camera()
                if not self.camera_connected:
                    time.sleep(0.15)
                    continue

            # Viewfinder state transition for Nikon/Canon
            if self.camera_connected and self.camera:
                if self.live_view_active and not self._physical_viewfinder_active:
                    success = self._set_camera_viewfinder(1)
                    if success:
                        self._physical_viewfinder_active = True
                        self.connection_state = "streaming"
                elif not self.live_view_active and self._physical_viewfinder_active:
                    self._set_camera_viewfinder(0)
                    self._physical_viewfinder_active = False
                    self.connection_state = "connected"

            # 3. Grab preview frame if live view is active and not paused for setting update
            if self.live_view_active and not self.pause_preview:
                try:
                    frame = self._grab_preview_frame()
                    if frame:
                        with self.frame_lock:
                            self.latest_frame = frame
                            self.frame_id += 1
                        self.frame_event.set()
                        self.frame_event.clear()
                except Exception as e:
                    err_str = str(e)
                    if "-52" in err_str or "Could not find the requested device" in err_str:
                        self.log(f"Camera device disconnected or reset during Live View ([-52]). Resetting connection state...")
                        self.disconnect()
                        self.connection_state = "searching"
                        time.sleep(1.0)
                    elif "-110" in err_str or "I/O in progress" in err_str:
                        time.sleep(0.1)
                    else:
                        self.log(f"Live view preview frame warning: {e}")
                        time.sleep(0.04)

            # 4. Check for camera events (e.g., photo taken via hardware remote) — only when Live View is idle
            if self.camera_connected and self.camera and not self.live_view_active:
                try:
                    with self.lock:
                        event_type, event_data = self.camera.wait_for_event(20)
                    if event_type == gp.GP_EVENT_FILE_ADDED:
                        self.log(f"Hardware shutter event detected! File added: {event_data.name}")
                        self._download_camera_file(event_data.folder, event_data.name)
                except Exception:
                    pass

            # 5. If live view is NOT active, periodically update physical settings cache to capture body dial changes
            if self.camera_connected and self.camera and not self.live_view_active:
                if not hasattr(self, '_last_settings_poll') or now - self._last_settings_poll > 5.0:
                    try:
                        polled = self._query_camera_settings()
                        for k, v in polled.items():
                            if v and v != "Unknown":
                                self.camera_settings[k] = v
                        self._last_settings_poll = now
                    except Exception:
                        pass

            # Avoid tight loop when idle
            if not self.live_view_active:
                time.sleep(0.08)
            else:
                time.sleep(0.015)

        self.log("Worker loop exited.")

    # Worker actions (Guaranteed to execute sequentially on the camera thread)
    def _try_connect_physical_camera(self):
        if not GPHOTO2_AVAILABLE or self.simulated:
            self.simulated = True
            self.connection_state = "simulated"
            return
            
        try:
            import sys
            import subprocess
            
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
                return

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
                        time.sleep(0.3)
                if camera:
                    break

            if not camera:
                raise last_init_err or Exception("Failed to initialize camera with any driver profile.")

            self.camera = camera
            self.camera_connected = True
            self.simulated = False
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
            queried_settings = self._query_camera_settings()
            queried_choices = self._query_camera_choices()
            for k, v in queried_settings.items():
                if v and v != "Unknown":
                    self.camera_settings[k] = v
            for k, v in queried_choices.items():
                if v:
                    self.camera_choices[k] = v
            self.log(f"Initialized physical camera settings: {self.camera_settings}")
                
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
            return {
                "connected": True,
                "simulated": False,
                "settings": self._query_camera_settings(),
                "choices": self._query_camera_choices()
            }

        elif cmd == "reconnect":
            self.disconnect()
            self.simulated = False
            self.camera = None
            self.camera_connected = False
            self._try_connect_physical_camera()
            return self.camera_connected

        elif cmd == "set_config":
            name = args["name"].lower()
            val = args["value"]
            if self.simulated:
                if name in self.sim_settings:
                    self.sim_settings[name] = val
                    self.log(f"Simulated setting updated: {name} = {val}")
                    return True
                elif name in ["manualfocusdrive", "eosremoterelease"]:
                    self.log(f"Simulated setting updated (action): {name} = {val}")
                    return True
                raise ValueError(f"Unknown setting: {name}")
            else:
                self.pause_preview = True
                time.sleep(0.15)
                try:
                    # Drain any pending PTP events to clear USB bus before updating property
                    if self.camera:
                        with self.lock:
                            try:
                                for _ in range(5):
                                    evt_type, _ = self.camera.wait_for_event(30)
                                    if evt_type == gp.GP_EVENT_TIMEOUT:
                                        break
                            except Exception:
                                pass
                    self._set_camera_property(name, val)
                    self.camera_settings[name] = val
                    return True
                except Exception as e:
                    self.log(f"Error setting camera property '{name}' to '{val}': {e}")
                    raise e
                finally:
                    self.pause_preview = False
                
        elif cmd == "test_widgets":
            if self.simulated or not self.camera:
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

        elif cmd == "dump_config":
            if self.simulated or not self.camera:
                return ["Simulated Mode Active - No Physical Config"]
            config = self.camera.get_config()
            names = []
            def dump(w):
                names.append(w.get_name())
                for i in range(w.count_children()):
                    dump(w.get_child(i))
            dump(config)
            return names

        elif cmd == "dump_config_values":
            if self.simulated or not self.camera:
                return {"error": "simulated mode"}
            try:
                config = self.camera.get_config()
                values = {}
                def traverse(w, path=[]):
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

        elif cmd == "capture":
            if self.simulated or not self.camera_connected or not self.camera:
                self.log("Simulating capture...")
                time.sleep(0.8) # simulate shutter release sound/lag
                return self._simulate_raw_capture()
            else:
                self.pause_preview = True
                self.connection_state = "capturing"
                # Reset viewfinder state so EVF stream is re-engaged cleanly after capture
                self._physical_viewfinder_active = False
                try:
                    # 1. Drain residual USB preview packets so bus is idle before setting configs or triggering shutter
                    with self.lock:
                        try:
                            for _ in range(6):
                                evt_type, _ = self.camera.wait_for_event(25)
                                if evt_type == gp.GP_EVENT_TIMEOUT:
                                    break
                        except Exception:
                            pass

                    # 2. Disengage hardware sensor zoom (eoszoom) if active to unlock camera shutter
                    if self.resolved_names.get("eoszoom"):
                        try:
                            cur_zoom = str(self.camera_settings.get("eoszoom", "0")).lower()
                            if cur_zoom not in ('0', 'off', 'none'):
                                self.log("Disengaging hardware zoom (eoszoom = 0) prior to capture to unlock shutter...")
                                self._set_camera_property("eoszoom", 0)
                                self.camera_settings["eoszoom"] = "0"
                                time.sleep(0.2)
                        except Exception as zoom_err:
                            self.log(f"Notice: Pre-capture eoszoom check: {zoom_err}")

                    # 3. Snapshot DCIM directory on camera storage so we can detect new captures
                    # even if capturetarget is Memory card or no FILE_ADDED PTP event is emitted
                    dcim_dir = self._get_dcim_folder()
                    initial_dcim_files = set()
                    last_dcim_file = None
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

                    predicted_candidates = []
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
                                self._set_camera_property("eosremoterelease", rel)
                                break
                            except Exception:
                                pass

                        # For scanning film negatives, shutter release MUST NOT activate AF hunting
                        # Canon EOS remote release: Press Half MF (metering only) -> Press Full MF (shutter trip)
                        self.log("Triggering camera shutter via eosremoterelease (Press Half MF -> Press Full MF)...")
                        self._set_camera_property("eosremoterelease", "Press Half MF")
                        time.sleep(0.15) # Hold half-press for exposure metering

                        self._set_camera_property("eosremoterelease", "Press Full MF")
                        time.sleep(0.25) # Hold full-press to trip shutter mechanism

                        # Release shutter switches
                        for rel in ["Release", "Release Full", "Release Half"]:
                            try:
                                self._set_camera_property("eosremoterelease", rel)
                                break
                            except Exception:
                                pass

                        # Wait for capture event from camera or retrieve newly written file from storage
                        t0 = time.time()
                        while time.time() - t0 < 10.0:
                            # 1. Check PTP events
                            with self.lock:
                                try:
                                    event_type, event_data = self.camera.wait_for_event(80)
                                except Exception:
                                    event_type, event_data = gp.GP_EVENT_TIMEOUT, None

                            if event_type == gp.GP_EVENT_FILE_ADDED and event_data:
                                file_path_info = (event_data.folder, event_data.name)
                                self.log(f"Capture event detected: {file_path_info[0]}/{file_path_info[1]}")
                                result_path = self._download_camera_file(event_data.folder, event_data.name)
                                break

                            # 2. Try direct storage file retrieval using predicted filename
                            if dcim_dir and predicted_candidates:
                                for cand_name in predicted_candidates:
                                    try:
                                        result_path = self._download_camera_file(dcim_dir, cand_name)
                                        self.log(f"Successfully matched and downloaded predicted capture: {cand_name}")
                                        file_path_info = (dcim_dir, cand_name)
                                        break
                                    except Exception:
                                        pass
                                if file_path_info:
                                    break

                            time.sleep(0.15)

                        if not result_path:
                            raise Exception("Camera shutter release did not produce an image file on storage within timeout.")
                    else:
                        self.log("Triggering camera capture via camera.capture()...")
                        with self.lock:
                            file_path = self.camera.capture(gp.GP_CAPTURE_IMAGE)
                        self.log(f"Capture successful. File created on camera: {file_path.folder}/{file_path.name}")
                        result_path = self._download_camera_file(file_path.folder, file_path.name)

                    # Post-capture event draining to clear remaining PTP notifications (like CAPTURE_COMPLETE)
                    try:
                        with self.lock:
                            for _ in range(12):
                                evt_type, _ = self.camera.wait_for_event(30)
                                if evt_type == gp.GP_EVENT_TIMEOUT:
                                    break
                    except Exception:
                        pass
                    time.sleep(0.2)
                    return result_path
                finally:
                    self.pause_preview = False
                    self.connection_state = "streaming" if self.live_view_active else "connected"

        elif cmd == "autofocus":
            if self.simulated:
                self.log("Simulating autofocus...")
                time.sleep(1.0)
                return True
            else:
                self.pause_preview = True
                try:
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

                    try:
                        with self.lock:
                            for _ in range(5):
                                evt_type, _ = self.camera.wait_for_event(50)
                                if evt_type == gp.GP_EVENT_TIMEOUT:
                                    break
                    except Exception:
                        pass
                    time.sleep(0.1)
                    return True
                finally:
                    self.pause_preview = False

        raise ValueError(f"Unknown worker command: {cmd}")

    def _grab_preview_frame(self):
        if self.simulated:
            # Generate simulated frame
            is_mono = False
            if self.session_manager:
                is_mono = self.session_manager.config.get("monochrome", False)
                
            return self._generate_simulated_frame(
                iso=self.sim_settings["iso"],
                aperture=self.sim_settings["aperture"],
                shutter=self.sim_settings["shutterspeed"],
                r_led=self.mock_leds["red"],
                g_led=self.mock_leds["green"],
                b_led=self.mock_leds["blue"],
                is_monochrome=is_mono
            )
            
        retries = 3
        last_err = None
        for attempt in range(retries):
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
                try:
                    with self.lock:
                        for _ in range(3):
                            evt_type, _ = self.camera.wait_for_event(30)
                            if evt_type == gp.GP_EVENT_TIMEOUT:
                                break
                except Exception:
                    pass
                time.sleep(0.1)

        self._consecutive_preview_errors += 1
        if self._consecutive_preview_errors == 10:
            self.log(f"Live view preview struggling with camera I/O: {last_err}")
        raise last_err

    def _set_camera_viewfinder(self, val):
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
                            with self.frame_lock:
                                self.latest_frame = memoryview(file_data).tobytes()
                                self.frame_id += 1
                            self.frame_event.set()
                            self.frame_event.clear()
                        self.log("Canon Live View stream initiated via capture_preview.")
                        return True
                    except Exception as e:
                        self.log(f"Notice: Canon capture_preview init: {e}")
                        return False
                else:
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
                        with self.frame_lock:
                            self.latest_frame = memoryview(file_data).tobytes()
                            self.frame_id += 1
                        self.frame_event.set()
                        self.frame_event.clear()
                    self.log("Live View stream kickstarted via capture_preview.")
                    return True
                except Exception as e:
                    self.log(f"Notice: capture_preview kickstart: {e}")

        except Exception as e:
            self.log(f"Notice setting viewfinder to {val}: {e}")
        return False

    def _get_dcim_folder(self):
        if hasattr(self, '_cached_dcim_folder') and self._cached_dcim_folder:
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
        import re
        m = re.match(r'^([A-Za-z0-9_]+?)(\d+)(\.[A-Za-z0-9]+)$', filename)
        if m:
            prefix, num_str, ext = m.groups()
            num = int(num_str)
            num_len = len(num_str)
            next_num = (num + 1) % (10 ** num_len)
            return f"{prefix}{next_num:0{num_len}d}{ext}"
        return None

    def _download_camera_file(self, folder, name):
        # Determine target directory
        target_dir = None
        if self.session_manager and getattr(self.session_manager, 'dirs', None):
            target_dir = self.session_manager.dirs.get("negatives")
            
        if not target_dir or not os.path.exists(target_dir):
            # Fallback to current workspace or Pictures if no active session
            target_dir = os.path.abspath("./negatives_download")
            os.makedirs(target_dir, exist_ok=True)

        # Build name based on next frame index
        ext = os.path.splitext(name)[1].lower()
        
        # If the file is not a supported RAW (.cr3, .raf) or JPEG/TIFF, just download as-is
        # Typically Canon outputs .CR3
        frame_num = 1
        if self.session_manager and hasattr(self.session_manager, 'get_next_frame_number'):
            frame_num = self.session_manager.get_next_frame_number(target_dir)
            
        # Format filename to keep stack aligned
        local_name = f"Frame_{frame_num:02d}_Capture_{int(time.time())}{ext}"
        local_path = os.path.join(target_dir, local_name)
        
        self.log(f"Downloading {name} to {local_path}...")
        
        try:
            with self.lock:
                camera_file = self.camera.file_get(
                    folder, 
                    name, 
                    gp.GP_FILE_TYPE_NORMAL
                )
                camera_file.save(local_path)
            self.log(f"Download complete: {local_name}")
            return local_path
        except Exception as e:
            self.log(f"Failed to download file {name}: {e}")
            raise e

    def _simulate_raw_capture(self):
        # Simulate RAW file creation in negatives folder
        target_dir = None
        if self.session_manager and getattr(self.session_manager, 'dirs', None):
            target_dir = self.session_manager.dirs.get("negatives")
            
        if not target_dir or not os.path.exists(target_dir):
            target_dir = os.path.abspath("./negatives_download")
            os.makedirs(target_dir, exist_ok=True)
            
        frame_num = 1
        if self.session_manager and hasattr(self.session_manager, 'get_next_frame_number'):
            frame_num = self.session_manager.get_next_frame_number(target_dir)

        # Write a mock CR3 file containing coordinates and setting strings
        # Size will be small so compositor detects it as mock
        # Format: Frame_XX_Capture_YYYY_red.cr3 (in sequential triplets, the compositor sorts alphabetically)
        # Let's figure out what color light is active based on the mock LEDs to add a hint to the name
        color_suffix = "white"
        r, g, b = self.mock_leds["red"], self.mock_leds["green"], self.mock_leds["blue"]
        if r > g and r > b:
            color_suffix = "red"
        elif g > r and g > b:
            color_suffix = "green"
        elif b > r and b > g:
            color_suffix = "blue"
            
        local_name = f"Frame_{frame_num:02d}_Capture_{color_suffix}.cr3"
        local_path = os.path.join(target_dir, local_name)
        
        with open(local_path, "w") as f:
            f.write(f"MOCK RAW CAPTURE DATA\n")
            f.write(f"Frame: {frame_num}\n")
            f.write(f"Color: {color_suffix}\n")
            f.write(f"ISO: {self.sim_settings['iso']}\n")
            f.write(f"Aperture: {self.sim_settings['aperture']}\n")
            f.write(f"Shutter: {self.sim_settings['shutterspeed']}\n")
            
        self.log(f"Simulated capture saved: {local_name}")
        return local_path

    def _get_setting_widget(self, name, config, keep_alive):
        # Component paths for setting traversal on Canon/Nikon cameras.
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
                # Add all successfully resolved wrappers to the main keep_alive list to keep them in scope
                keep_alive.extend(temp_keep_alive)
                return current
        
        # Fallback to recursive search if explicit paths did not resolve the widget
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

    def _query_camera_settings(self):
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

    def _query_camera_choices(self):
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

    def _query_camera_focus_mode(self):
        if not self.camera:
            return None
        widget_name = self.resolved_names.get("focusmode")
        if not widget_name:
            return None
        try:
            widget = self.camera.get_single_config(widget_name)
            return str(widget.get_value())
        except Exception as e:
            self.log(f"Error querying focus mode (widget '{widget_name}'): {e}")
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

    def _set_camera_property(self, name, value):
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
                        try:
                            for _ in range(5):
                                evt_type, _ = self.camera.wait_for_event(30)
                                if evt_type == gp.GP_EVENT_TIMEOUT:
                                    break
                        except Exception:
                            pass
                    time.sleep(0.1)
                else:
                    raise Exception(f"Setting '{name}' (widget '{widget_name}') not supported or found: {e}")

        # If it's a range widget (like manualfocusdrive on Nikon), handle mapping from speed/dir string to step integer
        try:
            widget_type = widget.get_type()
        except Exception:
            widget_type = None

        if widget_type == gp.GP_WIDGET_RANGE and name.lower() == "manualfocusdrive":
            val_str = str(value).lower()
            direction = 1 if "near" in val_str else -1
            
            import re
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
                    for _ in range(3):
                        evt_type, _ = self.camera.wait_for_event(20)
                        if evt_type == gp.GP_EVENT_TIMEOUT:
                            break
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

                if ("-110" in err_str or "I/O in progress" in err_str or "busy" in err_str.lower()) and attempt < max_retries - 1:
                    self.log(f"Setting '{name}' attempt {attempt+1}/{max_retries} encountered I/O in progress. Retrying after event drain...")
                    with self.lock:
                        try:
                            for _ in range(5):
                                evt_type, _ = self.camera.wait_for_event(30)
                                if evt_type == gp.GP_EVENT_TIMEOUT:
                                    break
                        except Exception:
                            pass
                    time.sleep(0.15)
                else:
                    break

        if last_err:
            self.log(f"Error applying setting '{name}' = '{value}': {last_err}")
            raise last_err
        return True

    def _generate_simulated_frame(self, iso, aperture, shutter, r_led, g_led, b_led, is_monochrome):
        # Create image container (640x480)
        width, height = 640, 480
        img = Image.new('RGB', (width, height), color=(20, 20, 25))
        draw = ImageDraw.Draw(img)
        
        # Base film color
        if is_monochrome:
            base_color = (95, 95, 95)
        else:
            base_color = (215, 115, 60) # classic Kodachrome orange negative base
            
        # Inner film border
        draw.rectangle([70, 50, width-70, height-50], fill=base_color)
        
        # Center Target Graphics
        draw.ellipse([width//2 - 90, height//2 - 90, width//2 + 90, height//2 + 90], outline=(255, 255, 255), width=2)
        draw.ellipse([width//2 - 30, height//2 - 30, width//2 + 30, height//2 + 30], outline=(240, 240, 240), width=1)
        draw.line([width//2 - 130, height//2, width//2 + 130, height//2], fill=(230, 230, 230), width=1)
        draw.line([width//2, height//2 - 130, width//2, height//2 + 130], fill=(230, 230, 230), width=1)
        
        # Scale pixel channels by LED intensities and exposure settings
        import numpy as np
        arr = np.array(img, dtype=np.float32)
        
        # ISO Exposure scalar
        try:
            iso_val = float(iso)
        except ValueError:
            iso_val = 400.0
            
        # Aperture exposure scalar
        try:
            ap_val = float(aperture.replace('f/', ''))
        except ValueError:
            ap_val = 8.0
            
        # Shutter speed exposure scalar
        try:
            if '/' in shutter:
                num, denom = shutter.split('/')
                shutter_val = float(num) / float(denom)
            else:
                shutter_val = float(shutter.replace('s', ''))
        except ValueError:
            shutter_val = 0.0166
            
        # Exposure math: base ISO 400, shutter 1/60 (0.0166s), aperture f/8.0
        exposure = iso_val * shutter_val * (1.0 / (ap_val ** 2))
        brightness = exposure / 0.103
        
        # Calculate dynamic Red, Green, Blue exposure multipliers
        r_mult = (r_led / 255.0) * brightness
        g_mult = (g_led / 255.0) * brightness
        b_mult = (b_led / 255.0) * brightness
        
        # Apply multipliers to pixel arrays
        arr[:, :, 0] *= r_mult
        arr[:, :, 1] *= g_mult
        arr[:, :, 2] *= b_mult
        
        # Inject ISO noise (film-grain simulation)
        if iso_val > 100:
            noise_sigma = (iso_val / 6400.0) * 40.0
            noise = np.random.normal(0, noise_sigma, arr.shape)
            arr += noise
            
        # Clip colors to valid 8-bit bounds
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        processed_img = Image.fromarray(arr)
        
        # Draw status info banner
        draw_ovr = ImageDraw.Draw(processed_img)
        draw_ovr.rectangle([0, height-25, width, height], fill=(15, 23, 42)) # Slate dark footer
        text_str = f"SIMULATED CAMERA  |  ISO {iso}  |  {aperture}  |  {shutter}s  |  LEDs [R:{r_led} G:{g_led} B:{b_led}]"
        draw_ovr.text((15, height-20), text_str, fill=(148, 163, 184)) # Cool slate gray text
        
        # Focus lines indicator (simulating focus assist overlay)
        draw_ovr.rectangle([10, 10, width-10, height-35], outline=(30, 41, 59), width=1) # Outer thin frame
        
        # Save image as JPEG stream
        out_buf = io.BytesIO()
        processed_img.save(out_buf, format='JPEG', quality=85)
        return out_buf.getvalue()
