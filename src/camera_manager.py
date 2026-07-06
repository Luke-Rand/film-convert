import os
import io
import time
import queue
import threading
from pathlib import Path
from PIL import Image, ImageDraw

# Try to import gphoto2. If not installed, we fallback to simulated mode.
try:
    import gphoto2 as gp
    GPHOTO2_AVAILABLE = True
except ImportError:
    GPHOTO2_AVAILABLE = False

class CameraManager:
    def __init__(self, session_manager=None):
        self.session_manager = session_manager
        self.simulated = not GPHOTO2_AVAILABLE
        self.camera = None
        self.camera_connected = False
        
        # Thread safety control
        self.cmd_queue = queue.Queue()
        self.worker_thread = None
        self.stop_event = threading.Event()
        
        # Live view configurations & states
        self.live_view_active = False
        self.latest_frame = None
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
                self.camera.exit()
            except Exception:
                pass
            self.camera = None
        self.camera_connected = False
        self._physical_viewfinder_active = False
        self.log("Camera disconnected.")

    def get_status(self):
        with self.frame_lock:
            if self.simulated:
                return {
                    "connected": True,
                    "simulated": True,
                    "settings": self.sim_settings,
                    "choices": self.sim_choices
                }
            elif self.camera_connected:
                return {
                    "connected": True,
                    "simulated": False,
                    "settings": self.camera_settings,
                    "choices": self.camera_choices
                }
            else:
                return {
                    "connected": False,
                    "simulated": False,
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
        return self.send_cmd("reconnect", {})

    def capture_image(self, autofocus=True):
        return self.send_cmd("capture", {"autofocus": autofocus})

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
        while not self.stop_event.is_set():
            if not self.simulated and not self.camera:
                self._try_connect_physical_camera()
                if not self.camera_connected:
                    time.sleep(2.0)
                    continue

            # 1. Execute commands queued by Flask threads
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

            # Viewfinder state transition for Nikon/Canon
            if self.camera_connected and self.camera:
                if self.live_view_active and not self._physical_viewfinder_active:
                    self._set_camera_viewfinder(1)
                    self._physical_viewfinder_active = True
                elif not self.live_view_active and self._physical_viewfinder_active:
                    self._set_camera_viewfinder(0)
                    self._physical_viewfinder_active = False

            # 2. Grab preview frame if live view is active
            if self.live_view_active:
                try:
                    frame = self._grab_preview_frame()
                    if frame:
                        with self.frame_lock:
                            self.latest_frame = frame
                except Exception as e:
                    self.log(f"Live view preview capture error: {e}. Disconnecting.")
                    self.disconnect()
                    time.sleep(1.0)

            # 3. Check for camera events (e.g., photo taken via hardware remote)
            if self.camera_connected and self.camera:
                try:
                    # Brief timeout so we don't hold the lock too long
                    event_type, event_data = self.camera.wait_for_event(20)
                    if event_type == gp.GP_EVENT_FILE_ADDED:
                        self.log(f"Hardware shutter event detected! File added: {event_data.name}")
                        self._download_camera_file(event_data.folder, event_data.name)
                except Exception as e:
                    # If it's a gphoto2 error indicating disconnection, handle it
                    if isinstance(e, gp.GPhoto2Error) and e.code in [-52, -53, -10, -110]:
                        self.log(f"Event wait connection error: {e}. Disconnecting.")
                        self.disconnect()

            # 4. If live view is NOT active, periodically update physical settings cache to capture body dial changes
            if self.camera_connected and self.camera and not self.live_view_active:
                now = time.time()
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
                # Target ~20-25 FPS live view (40-50ms intervals)
                time.sleep(0.04)

        self.log("Worker loop exited.")

    # Worker actions (Guaranteed to execute sequentially on the camera thread)
    def _try_connect_physical_camera(self):
        if not GPHOTO2_AVAILABLE:
            self.simulated = True
            return
            
        try:
            self.log("Attempting to connect to physical camera...")
            
            # On macOS, combat the ptpcamerad daemon that keeps locking USB camera ports
            import sys
            import subprocess
            
            t = None
            stop_kill = None
            if sys.platform == 'darwin':
                self.log("Detected macOS: Launching automated background release for ptpcamerad...")
                stop_kill = threading.Event()
                def kill_loop():
                    while not stop_kill.is_set():
                        try:
                            # Kill daemon silently
                            subprocess.run(["killall", "-9", "ptpcamerad"], capture_output=True)
                        except Exception:
                            pass
                        time.sleep(0.1)
                t = threading.Thread(target=kill_loop, name="PtpCameraKiller", daemon=True)
                t.start()
                time.sleep(0.3) # allow a moment for the process to be terminated
                
            try:
                camera = gp.Camera()
                camera.init()
                self.camera = camera
                self.camera_connected = True
                self.simulated = False
                self.log(f"Successfully connected to camera: {camera.get_summary().text.splitlines()[0]}")
                
                # Probe and resolve setting widget names (Canon vs Nikon)
                probe_targets = [
                    ("iso", ["iso", "eosiso"]),
                    ("aperture", ["aperture", "f-number", "fnumber"]),
                    ("shutterspeed", ["shutterspeed", "shutterspeed2"]),
                    ("manualfocusdrive", ["manualfocusdrive"]),
                    ("eosremoterelease", ["eosremoterelease"]),
                    ("autofocusdrive", ["autofocusdrive"]),
                    ("focusmode", ["focusmode", "focus_mode", "lensfocusmode", "canonfocusmode"])
                ]
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
                
                # Force viewfinder (live view) to OFF on startup to clear any mirror-up lock from previous crashed runs
                try:
                    config = camera.get_config()
                    keep_alive = [config]
                    viewfinder = self._find_widget_by_name(config, "viewfinder", keep_alive)
                    if viewfinder:
                        viewfinder.set_value(0)
                        camera.set_config(config)
                        self.log("Forced viewfinder (live view) to OFF on startup to reset camera state.")
                    else:
                        self.log("Viewfinder widget not found recursively in config tree (could not reset).")
                except Exception as e:
                    self.log(f"Failed to reset viewfinder on startup: {e}")
                
                # Query and cache settings and choices immediately while camera is in standby!
                queried_settings = self._query_camera_settings()
                queried_choices = self._query_camera_choices()
                for k, v in queried_settings.items():
                    if v and v != "Unknown":
                        self.camera_settings[k] = v
                for k, v in queried_choices.items():
                    if v:
                        self.camera_choices[k] = v
                self.log(f"Initialized physical camera settings (with fallbacks): {self.camera_settings}")
            finally:
                if t and stop_kill:
                    stop_kill.set()
                    t.join(timeout=1.0)
                    self.log("macOS background release loop stopped.")
                    
        except Exception as e:
            self.camera = None
            self.camera_connected = False
            # Fallback to simulated mode if no camera is available
            self.simulated = True
            self.log(f"No physical camera detected. Falling back to Simulated Mode. (Reason: {e})")

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
            self.simulated = False
            self.camera = None
            self.camera_connected = False
            return True

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
                try:
                    self._set_camera_property(name, val)
                except Exception as e:
                    self.log(f"Warning setting camera property '{name}' to '{val}': {e}. Cache updated.")
                self.camera_settings[name] = val
                return True
                
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
            autofocus = args.get("autofocus", True)
            if self.simulated:
                self.log("Simulating capture...")
                time.sleep(0.8) # simulate shutter release sound/lag
                return self._simulate_raw_capture()
            else:
                original_focus_mode = None
                widget_name = self.resolved_names.get("focusmode")
                if not autofocus and widget_name:
                    try:
                        original_focus_mode = self._query_camera_focus_mode()
                        if original_focus_mode and original_focus_mode.lower() != "manual":
                            self.log(f"Temporarily setting focus mode to Manual (was: {original_focus_mode})")
                            self._set_camera_property("focusmode", "Manual")
                    except Exception as e:
                        self.log(f"Failed to temporarily set focus mode to Manual: {e}")
                
                try:
                    self.log("Triggering camera capture...")
                    file_path = self.camera.capture(gp.GP_CAPTURE_IMAGE)
                    self.log(f"Capture successful. File created on camera: {file_path.folder}/{file_path.name}")
                    return self._download_camera_file(file_path.folder, file_path.name)
                finally:
                    if original_focus_mode and original_focus_mode.lower() != "manual" and widget_name:
                        try:
                            self.log(f"Restoring focus mode to {original_focus_mode}")
                            self._set_camera_property("focusmode", original_focus_mode)
                        except Exception as e:
                            self.log(f"Failed to restore focus mode to {original_focus_mode}: {e}")

        elif cmd == "autofocus":
            if self.simulated:
                self.log("Simulating autofocus...")
                time.sleep(1.0)
                return True
            else:
                if self.resolved_names.get("eosremoterelease"):
                    self.log("Triggering autofocus via eosremoterelease (Canon)...")
                    self._set_camera_property("eosremoterelease", "Press Half AF")
                    time.sleep(1.0)
                    self._set_camera_property("eosremoterelease", "Release Half")
                    return True
                elif self.resolved_names.get("autofocusdrive"):
                    self.log("Triggering autofocus via autofocusdrive (Nikon/Generic)...")
                    self._set_camera_property("autofocusdrive", 1)
                    return True
                else:
                    raise Exception("Autofocus is not supported or resolved for this camera model.")

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
            
        try:
            # Capture actual preview
            camera_file = self.camera.capture_preview()
            file_data = camera_file.get_data_and_size()
            return memoryview(file_data).tobytes()
        except Exception as e:
            raise e

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

    def _set_camera_viewfinder(self, val):
        if not self.camera:
            return
        try:
            try:
                # Fast path using get_single_config
                viewfinder = self.camera.get_single_config("viewfinder")
                viewfinder.set_value(val)
                self.camera.set_single_config("viewfinder", viewfinder)
                self.log(f"Set viewfinder to {val} using get_single_config")
                return
            except Exception:
                pass
                
            # Fallback using recursive config tree search
            config = self.camera.get_config()
            keep_alive = [config]
            viewfinder = self._find_widget_by_name(config, "viewfinder", keep_alive)
            if viewfinder:
                viewfinder.set_value(val)
                self.camera.set_config(config)
                self.log(f"Set viewfinder to {val} using config tree")
            else:
                self.log("Viewfinder widget not found recursively in config tree.")
        except Exception as e:
            self.log(f"Failed to set viewfinder to {val}: {e}")

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

    def _set_camera_property(self, name, value):
        if not self.camera:
            return False
        
        widget_name = self.resolved_names.get(name.lower())
        if not widget_name:
            widget_name = name.lower()
            
        try:
            widget = self.camera.get_single_config(widget_name)
        except Exception as e:
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
            widget.set_value(matched_choice)
            self.log(f"Mapped manualfocusdrive '{value}' to range value: {matched_choice}")
        else:
            valid_choices = []
            try:
                for i in range(widget.count_choices()):
                    valid_choices.append(str(widget.get_choice(i)))
            except Exception:
                pass

            matched_choice = None
            for choice in valid_choices:
                if choice.lower() == str(value).lower():
                    matched_choice = choice
                    break

            if not matched_choice:
                matched_choice = str(value)
                if valid_choices and name.lower() != "focusmode":
                    self.log(f"Warning: Set value '{value}' not in camera choices {valid_choices}. Attempting raw set.")

            widget.set_value(matched_choice)

        try:
            self.camera.set_single_config(widget_name, widget)
            self.log(f"Setting updated: {name} (widget '{widget_name}') = {matched_choice}")
        except Exception as e:
            self.log(f"Error applying setting '{name}' = '{value}': {e}")
            raise e
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
