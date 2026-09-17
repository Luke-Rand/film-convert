import os
import sys
import time
import shutil
import glob
import threading
import contextlib
import io
import queue
from pathlib import Path
from datetime import datetime
from collections import deque

from flask import Flask, request, jsonify, render_template, send_file, Response
import numpy as np
import tifffile
from PIL import Image
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from typing import Optional, Dict, Any, List, Tuple
from pydantic import ValidationError

try:
    from schemas import (
        SessionConfigSchema,
        SessionConfigUpdateSchema,
        StartSessionSchema,
        BatchJobSchema,
        ContactSheetGenerateSchema,
        SampleRebateSchema,
        CameraConfigSchema,
        CameraFocusStepSchema,
        CameraToggleLiveviewSchema,
        CameraMockLedsSchema
    )
except ImportError:
    from src.schemas import (
        SessionConfigSchema,
        SessionConfigUpdateSchema,
        StartSessionSchema,
        BatchJobSchema,
        ContactSheetGenerateSchema,
        SampleRebateSchema,
        CameraConfigSchema,
        CameraFocusStepSchema,
        CameraToggleLiveviewSchema,
        CameraMockLedsSchema
    )

# Import core logic from existing scripts
from compositor import process_triplet
from inverter import process_positives
from contact_sheet import generate_contact_sheet
from camera_manager import CameraManager
from batch_worker import (
    worker_process_triplet,
    worker_process_positives,
    worker_process_triplet_pipeline
)

if hasattr(sys, '_MEIPASS'):
    # Bundled path for PyInstaller
    template_dir = os.path.join(sys._MEIPASS, 'templates')
    static_dir = os.path.join(sys._MEIPASS, 'static')
else:
    # Resolve relative to the location of web_ui.py (in src/)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(base_dir, 'templates')
    static_dir = os.path.join(base_dir, 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

def format_validation_error(e: ValidationError) -> str:
    err_msgs = []
    for err in e.errors():
        loc = ".".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        err_msgs.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(err_msgs)

# Thread log redirector to capture stdout from composite/inverter scripts
class ThreadLogRedirector:
    def __init__(self, log_callback):
        self.log_callback = log_callback

    def write(self, s):
        text = s.strip()
        if text:
            # Also log to actual terminal
            sys.__stdout__.write(s)
            self.log_callback(text)

    def flush(self):
        sys.__stdout__.flush()

class SessionManager:
    def __init__(self) -> None:
        self.status: str = "idle"  # idle, monitoring, batch_processing
        self.mode: str = "triplet"  # triplet, single
        self.root_folder: str = os.path.abspath(os.path.expanduser("~/Pictures/Scans"))
        self.session_name: str = ""
        self.dirs: Dict[str, str] = {}
        self.config: Dict[str, Any] = SessionConfigSchema().model_dump()
        self.logs: deque = deque(maxlen=1000)
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event: threading.Event = threading.Event()
        self.lock: threading.Lock = threading.Lock()
        self.subscribers: List[queue.Queue] = []
        self.subscribers_lock: threading.Lock = threading.Lock()
        self.executor: Optional[ProcessPoolExecutor] = None
        self.log("System initialized. Ready.")

    def _get_executor(self) -> ProcessPoolExecutor:
        """Returns or lazily creates the background multiprocessing ProcessPoolExecutor."""
        with self.lock:
            if self.executor is None:
                ctx = multiprocessing.get_context('spawn')
                # Reserve 1 core for Flask / CameraManager
                max_w = min(4, max(1, (os.cpu_count() or 2) - 1))
                self.executor = ProcessPoolExecutor(max_workers=max_w, mp_context=ctx)
            return self.executor

    def shutdown_executor(self) -> None:
        """Cleanly shuts down the background ProcessPoolExecutor on exit."""
        with self.lock:
            if self.executor:
                try:
                    self.executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                self.executor = None

    def add_subscriber(self) -> queue.Queue:
        with self.subscribers_lock:
            q = queue.Queue()
            self.subscribers.append(q)
            return q

    def remove_subscriber(self, q: queue.Queue) -> None:
        with self.subscribers_lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def broadcast(self, event_type: str, data: Any) -> None:
        with self.subscribers_lock:
            for q in self.subscribers:
                q.put((event_type, data))

    def broadcast_status(self) -> None:
        with self.lock:
            status_data = {
                "status": self.status,
                "mode": self.mode,
                "root_folder": self.root_folder,
                "session_name": self.session_name,
                "dirs": self.dirs,
                "config": self.config
            }
        self.broadcast("status", status_data)

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        self.logs.append(formatted)
        try:
            sys.__stdout__.write(f"WEB_LOG: {formatted}\n")
            sys.__stdout__.flush()
        except UnicodeEncodeError:
            encoding = getattr(sys.__stdout__, 'encoding', 'utf-8') or 'utf-8'
            safe_msg = formatted.encode(encoding, errors='replace').decode(encoding)
            sys.__stdout__.write(f"WEB_LOG: {safe_msg}\n")
            sys.__stdout__.flush()
        except Exception:
            pass
        
        # Broadcast to web subscribers
        if hasattr(self, 'subscribers_lock'):
            self.broadcast("log", {"line": formatted})

    def clear_logs(self) -> None:
        with self.lock:
            self.logs.clear()
            self.log("Logs cleared.")

    def is_safe_path(self, path: str) -> bool:
        # Allow reading files that are within the current root folder or workspace
        try:
            real_path = os.path.realpath(path)
            # Allow workspace folder and the configured root folder
            workspace = os.path.realpath(".")
            root = os.path.realpath(self.root_folder)
            
            # On case-insensitive filesystems (like macOS and Windows), compare case-insensitively
            if os.name == 'nt' or sys.platform == 'darwin':
                return real_path.lower().startswith(workspace.lower()) or real_path.lower().startswith(root.lower())
            
            # Check if it starts with either
            return real_path.startswith(workspace) or real_path.startswith(root)
        except Exception:
            return False

    def start_monitoring(self, root_dir: str, session_name: str, mode: str, config: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        with self.lock:
            if self.status != "idle":
                return False, f"Cannot start monitoring, current status is '{self.status}'"

            try:
                self.root_folder = os.path.abspath(os.path.expanduser(root_dir))
                os.makedirs(self.root_folder, exist_ok=True)
                
                self.session_name = session_name
                self.mode = mode
                if config:
                    self.config.update(config)
                
                session_dir = os.path.join(self.root_folder, self.session_name)
                self.dirs = {
                    "negatives": os.path.join(session_dir, "negatives"),
                    "positives": os.path.join(session_dir, "positives"),
                    "processed": os.path.join(session_dir, "processed_raws"),
                    "errors": os.path.join(session_dir, "error_raws")
                }
                
                for d in self.dirs.values():
                    os.makedirs(d, exist_ok=True)
                    
                self.stop_event.clear()
                self.status = "monitoring"
                
                self.monitor_thread = threading.Thread(
                    target=self._monitor_loop, 
                    name="ScannerMonitorThread",
                    daemon=True
                )
                self.monitor_thread.start()
                
                self.log(f"Started monitoring session: '{self.session_name}' ({self.mode} mode)")
                self.log(f"Negatives folder: {self.dirs['negatives']}")
            except PermissionError as e:
                self.status = "idle"
                err_msg = f"Permission denied accessing '{root_dir}': {e}. If using an external drive or network share, please grant FilmConvert 'Full Disk Access' or 'Files and Folders' in macOS System Settings."
                self.log(f"Error starting monitor: {err_msg}")
                return False, err_msg
            except Exception as e:
                self.status = "idle"
                err_msg = f"Failed to initialize session directories: {str(e)}"
                self.log(f"Error starting monitor: {err_msg}")
                return False, err_msg
        
        self.broadcast_status()
        return True, "Monitoring started successfully"

    def stop_monitoring(self) -> Tuple[bool, str]:
        with self.lock:
            if self.status != "monitoring":
                return False, "Not currently monitoring"
            
            self.log("Stopping monitor...")
            self.stop_event.set()
            
        # Join outside lock to prevent deadlock if thread tries to lock during termination
        if self.monitor_thread:
            self.monitor_thread.join(timeout=3.0)
            
        with self.lock:
            self.status = "idle"
            self.monitor_thread = None
            self.log("Monitor stopped.")
            session_dirs_copy = dict(self.dirs) if self.dirs else {}
            session_name_copy = self.session_name
            root_folder_copy = self.root_folder
            cfg_copy = dict(self.config)

        self.broadcast_status()

        # Check if auto contact sheet generation is enabled and positive frames exist
        if cfg_copy.get("auto_contact_sheet", True) and session_dirs_copy.get("positives"):
            positives_dir = session_dirs_copy.get("positives")
            if os.path.isdir(positives_dir):
                valid_exts = {'.tiff', '.tif', '.dng', '.jpg', '.jpeg', '.png', '.cr3', '.raf', '.nef', '.arw'}
                has_frames = any(os.path.isfile(os.path.join(positives_dir, f)) and os.path.splitext(f)[1].lower() in valid_exts for f in os.listdir(positives_dir))
                if has_frames:
                    session_folder = os.path.join(root_folder_copy, session_name_copy)
                    self.log("Auto-generating archival roll summary & contact sheet...")
                    try:
                        cs_res = generate_contact_sheet(
                            session_dir=session_folder,
                            session_name=session_name_copy,
                            columns=cfg_copy.get("contact_sheet_columns", 6),
                            theme=cfg_copy.get("contact_sheet_theme", "dark"),
                            config=cfg_copy
                        )
                        if cs_res.get("success"):
                            self.log(f"Contact sheet complete: {cs_res.get('message')}")
                            self.broadcast("contact_sheet_generated", cs_res)
                        else:
                            self.log(f"Contact sheet note: {cs_res.get('message')}")
                    except Exception as cs_err:
                        self.log(f"Error auto-generating contact sheet: {cs_err}")

        return True, "Monitoring stopped successfully"

    def get_next_frame_number(self, negatives_dir: str) -> int:
        search_dirs = [negatives_dir]
        if self.dirs:
            for key in ['processed', 'positives']:
                d = self.dirs.get(key)
                if d and os.path.exists(d):
                    search_dirs.append(d)
        
        max_num = 0
        for d in search_dirs:
            for entry in os.listdir(d):
                if entry.startswith("Frame_"):
                    try:
                        parts = entry.split('_')
                        if len(parts) > 1:
                            num = int(parts[1])
                            if num > max_num:
                                max_num = num
                    except (ValueError, IndexError):
                        pass
        return max_num + 1

    def _monitor_loop(self) -> None:
        supported_triplet_exts = {'.cr3', '.raf', '.nef', '.arw', '.rw2', '.nrw', '.dcr'}
        supported_single_exts = {'.dng', '.tiff', '.tif', '.cr3', '.raf', '.nef', '.arw', '.rw2', '.nrw', '.dcr'}
        
        self.log("Background scanner monitor loop active.")
        
        # Setup redirector
        redirector = ThreadLogRedirector(self.log)
        
        while not self.stop_event.is_set():
            try:
                neg_dir = self.dirs.get("negatives")
                if not neg_dir or not os.path.exists(neg_dir):
                    self.stop_event.wait(1.0)
                    continue

                if self.mode == 'triplet':
                    # Find RAW triplet files
                    raw_files = [
                        os.path.join(neg_dir, f) for f in os.listdir(neg_dir)
                        if os.path.isfile(os.path.join(neg_dir, f)) and os.path.splitext(f)[1].lower() in supported_triplet_exts
                    ]
                    raw_files.sort(key=lambda x: os.path.getmtime(x))
                    
                    if len(raw_files) >= 3:
                        group = raw_files[:3]
                        
                        # Wait for OS to finish writing
                        if time.time() - os.path.getmtime(group[-1]) < 2:
                            self.stop_event.wait(0.5)
                            continue
                            
                        frame_number = self.get_next_frame_number(neg_dir)
                        self.log(f"Triplet detected! Processing Frame {frame_number:02d} in background worker...")
                        
                        composite_filename = f"Frame_{frame_number:02d}_Composite.dng"
                        composite_filepath = os.path.join(neg_dir, composite_filename)
                        
                        try:
                            executor = self._get_executor()
                            future = executor.submit(
                                worker_process_triplet_pipeline,
                                group=group,
                                composite_filepath=composite_filepath,
                                positives_dir=self.dirs['positives'],
                                config=dict(self.config)
                            )
                            success, result, worker_logs = future.result()
                            if worker_logs:
                                for line in worker_logs.strip().splitlines():
                                    if line.strip():
                                        self.log(line)
                            
                            if not success:
                                raise Exception(f"Worker failure: {result}")
                            
                            r_mean, g_mean, b_mean = result
                            self.broadcast("triplet_means", {
                                "r_mean": float(r_mean),
                                "g_mean": float(g_mean),
                                "b_mean": float(b_mean)
                            })
                            
                            # Move files
                            for f in group:
                                shutil.move(f, os.path.join(self.dirs['processed'], os.path.basename(f)))
                            shutil.move(composite_filepath, os.path.join(self.dirs['processed'], composite_filename))
                            
                            self.log(f"SUCCESS: Frame {frame_number:02d} processed and saved.")
                            
                        except Exception as e:
                            self.log(f"ERROR processing Frame {frame_number:02d}: {str(e)}")
                            for f in group:
                                try:
                                    shutil.move(f, os.path.join(self.dirs['errors'], os.path.basename(f)))
                                
                                except Exception: pass
                            if os.path.exists(composite_filepath):
                                try:
                                    shutil.move(composite_filepath, os.path.join(self.dirs['errors'], composite_filename))
                                except Exception: pass

                elif self.mode == 'single':
                    neg_files = [
                        os.path.join(neg_dir, f) for f in os.listdir(neg_dir)
                        if os.path.isfile(os.path.join(neg_dir, f)) and os.path.splitext(f)[1].lower() in supported_single_exts
                    ]
                    neg_files.sort(key=lambda x: os.path.getmtime(x))
                    
                    if neg_files:
                        filepath = neg_files[0]
                        filename = os.path.basename(filepath)
                        
                        if time.time() - os.path.getmtime(filepath) < 2:
                            self.stop_event.wait(0.5)
                            continue
                            
                        self.log(f"Negative detected! Processing {filename} in background worker...")
                        
                        try:
                            executor = self._get_executor()
                            future = executor.submit(
                                worker_process_positives,
                                input_path=filepath,
                                output_dir=self.dirs['positives'],
                                clip=self.config["clip"],
                                gamma=self.config["gamma"],
                                compress_tiff=self.config["compress_tiff"],
                                global_levels=self.config["global_levels"],
                                ignore_margin=self.config["margin"],
                                scurve=self.config["scurve"],
                                autocrop=self.config["autocrop"],
                                monochrome=self.config.get("monochrome", False),
                                monochrome_channel=self.config.get("monochrome_channel", "luminance"),
                                reversal=self.config.get("reversal", False),
                                convert_to_tiff=self.config.get("convert_to_tiff", True),
                                icc_profile=self.config.get("color_profile", "adobe_rgb"),
                                preserve_metadata=self.config.get("embed_metadata", True),
                                base_ratios=self.config.get("base_ratios")
                            )
                            success, result, worker_logs = future.result()
                            if worker_logs:
                                for line in worker_logs.strip().splitlines():
                                    if line.strip():
                                        self.log(line)

                            if not success:
                                raise Exception(f"Worker failure: {result}")
                            
                            shutil.move(filepath, os.path.join(self.dirs['processed'], filename))
                            self.log(f"SUCCESS: {filename} processed and saved.")
                            
                        except Exception as e:
                            self.log(f"ERROR processing {filename}: {str(e)}")
                            try:
                                shutil.move(filepath, os.path.join(self.dirs['errors'], filename))
                            except Exception: pass
                
            except PermissionError as pe:
                self.log(f"Permission denied accessing session directories: {pe}. Please grant FilmConvert 'Full Disk Access' or 'Files and Folders' (Network Volumes) in macOS System Settings.")
                self.stop_event.wait(5.0)
            except Exception as e:
                self.log(f"Monitor loop error: {str(e)}")
                self.stop_event.wait(1.0)
            
        self.log("Background scanner monitor loop stopped.")

    def run_batch_job(self, task_type: str, input_path: str, config: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        with self.lock:
            if self.status != "idle":
                return False, f"System is currently '{self.status}'"
            
            self.status = "batch_processing"
            if config:
                self.config.update(config)
            
        self.broadcast_status()

        def _batch_thread():
            self.log(f"Starting background batch task: {task_type} for '{input_path}'")
            
            try:
                # Resolve paths
                in_path = os.path.abspath(os.path.expanduser(input_path))
                if not os.path.exists(in_path):
                    raise FileNotFoundError(f"Input path '{in_path}' does not exist.")
                
                executor = self._get_executor()
                
                if task_type == 'composite':
                    # Composite RAW files in folder
                    out_dir = os.path.join(in_path, "Composites")
                    self.log(f"Processing RAW roll in: {in_path}")
                    self.log(f"Output folder: {out_dir}")
                    
                    # Search files
                    supported_exts = {'.cr3', '.raf', '.nef'}
                    raw_files = [
                        os.path.join(in_path, f) for f in os.listdir(in_path)
                        if os.path.isfile(os.path.join(in_path, f)) and os.path.splitext(f)[1].lower() in supported_exts
                    ]
                    
                    if not raw_files:
                        raise ValueError(f"No .cr3, .raf, or .nef files found in {in_path}")
                        
                    raw_files.sort()
                    total_files = len(raw_files)
                    self.log(f"Found {total_files} RAW files.")
                    
                    os.makedirs(out_dir, exist_ok=True)
                    
                    frame_number = 1
                    for i in range(0, total_files - 2, 3):
                        group = raw_files[i:i+3]
                        self.log(f"Processing Frame {frame_number:02d} ({[os.path.basename(f) for f in group]}) in background worker...")
                        output_filepath = os.path.join(out_dir, f"Frame_{frame_number:02d}_Composite.dng")
                        
                        future = executor.submit(
                            worker_process_triplet,
                            group=group,
                            output_filepath=output_filepath,
                            neutralize_base=self.config["neutralize"],
                            compress_tiff=self.config["compress_tiff"],
                            align_channels=self.config["align_channels"],
                            icc_profile=self.config.get("color_profile", "adobe_rgb"),
                            preserve_metadata=self.config.get("embed_metadata", True),
                            base_ratios=self.config.get("base_ratios")
                        )
                        success, result, worker_logs = future.result()
                        if worker_logs:
                            for line in worker_logs.strip().splitlines():
                                if line.strip():
                                    self.log(line)
                        if not success:
                            raise Exception(f"Batch worker error: {result}")
                        frame_number += 1
                    
                    self.log("Batch compositing complete!")
                    
                elif task_type == 'invert':
                    # Invert composite images in background process
                    self.log(f"Processing positive inversions for: {in_path} in background worker...")
                    future = executor.submit(
                        worker_process_positives,
                        input_path=in_path,
                        output_dir=None,  # let it auto-create subfolder Positives
                        clip=self.config["clip"],
                        gamma=self.config["gamma"],
                        compress_tiff=self.config["compress_tiff"],
                        global_levels=self.config["global_levels"],
                        ignore_margin=self.config["margin"],
                        scurve=self.config["scurve"],
                        autocrop=self.config["autocrop"],
                        monochrome=self.config.get("monochrome", False),
                        monochrome_channel=self.config.get("monochrome_channel", "luminance"),
                        reversal=self.config.get("reversal", False),
                        convert_to_tiff=self.config.get("convert_to_tiff", True),
                        icc_profile=self.config.get("color_profile", "adobe_rgb"),
                        preserve_metadata=self.config.get("embed_metadata", True),
                        base_ratios=self.config.get("base_ratios")
                    )
                    success, result, worker_logs = future.result()
                    if worker_logs:
                        for line in worker_logs.strip().splitlines():
                            if line.strip():
                                self.log(line)
                    if not success:
                        raise Exception(f"Batch worker error: {result}")
                    self.log("Batch inversion complete!")
                    
            except Exception as e:
                self.log(f"BATCH ERROR: {str(e)}")
            finally:
                with self.lock:
                    self.status = "idle"
                self.log("System idle.")
                self.broadcast_status()
 
        t = threading.Thread(target=_batch_thread, name="BatchProcessThread", daemon=True)
        t.start()
        return True, "Batch processing started"

# Instantiate global session manager
session = SessionManager()

# Initialize camera manager
camera_manager = CameraManager(session_manager=session)
camera_manager.start()

import atexit
import signal

def _cleanup():
    try:
        session.shutdown_executor()
    except Exception:
        pass
    try:
        camera_manager.stop()
    except Exception:
        pass

atexit.register(_cleanup)

def _sig_handler(signum, frame):
    _cleanup()
    sys.exit(0)

try:
    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)
except (ValueError, AttributeError):
    pass

# --- WEB CONTROLLER ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

# --- CAMERA CONTROLLER ROUTES ---

@app.route('/api/camera/status', methods=['GET'])
def get_camera_status():
    return jsonify(camera_manager.get_status())

@app.route('/api/camera/config', methods=['POST'])
def set_camera_config():
    try:
        payload = CameraConfigSchema.model_validate(request.json or {})
    except ValidationError as e:
        return jsonify({"success": False, "message": f"Invalid camera config: {format_validation_error(e)}"}), 400
    try:
        camera_manager.update_config(payload.name, payload.value)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/camera/focus_step', methods=['POST'])
def camera_focus_step():
    try:
        payload = CameraFocusStepSchema.model_validate(request.json or {})
    except ValidationError as e:
        return jsonify({"success": False, "message": f"Invalid focus step: {format_validation_error(e)}"}), 400
    direction = str(payload.direction).capitalize()
    value = f"{direction} {payload.speed}"
    try:
        camera_manager.notify_focus_adjustment()
        camera_manager.update_config("manualfocusdrive", value)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/camera/autofocus', methods=['POST'])
def camera_autofocus():
    try:
        camera_manager.notify_focus_adjustment()
        camera_manager.send_cmd("autofocus", {})
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/camera/capture', methods=['POST'])
def capture_camera_image():
    try:
        path = camera_manager.capture_image()
        return jsonify({"success": True, "path": path})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/camera/toggle_liveview', methods=['POST'])
def toggle_camera_liveview():
    try:
        payload = CameraToggleLiveviewSchema.model_validate(request.json or {})
    except ValidationError as e:
        return jsonify({"success": False, "message": f"Invalid liveview payload: {format_validation_error(e)}"}), 400
    camera_manager.set_liveview(payload.active)
    return jsonify({"success": True})

@app.route('/api/camera/reconnect', methods=['POST'])
def reconnect_camera():
    try:
        camera_manager.reconnect()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/camera/update_mock_leds', methods=['POST'])
def update_camera_mock_leds():
    try:
        payload = CameraMockLedsSchema.model_validate(request.json or {})
    except ValidationError as e:
        return jsonify({"success": False, "message": f"Invalid LED payload: {format_validation_error(e)}"}), 400
    camera_manager.update_mock_leds(payload.red, payload.green, payload.blue)
    return jsonify({"success": True})

@app.route('/api/camera/liveview')
def camera_liveview_feed():
    def generate():
        last_id = -1
        while True:
            camera_manager.frame_event.wait(timeout=0.05)
            with camera_manager.frame_lock:
                curr_id = camera_manager.frame_id
                frame = camera_manager.latest_frame
            
            if not frame or curr_id == last_id:
                time.sleep(0.02)
                continue

            # Skip-frame mechanism: Jump directly to latest frame ID
            last_id = curr_id

            # Downscaling thumbnail cache during fast manual focus adjustments
            if camera_manager.is_fast_focus_active(window_sec=1.0):
                frame_to_send = camera_manager.get_fast_focus_frame(max_width=640, quality=70) or frame
            else:
                frame_to_send = frame

            try:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')
            except GeneratorExit:
                break
            except Exception:
                break
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/frame')
def get_camera_single_frame():
    force_downscale = request.args.get('downscale', type=int)
    if force_downscale or camera_manager.is_fast_focus_active(window_sec=1.0):
        frame = camera_manager.get_fast_focus_frame(max_width=640, quality=70)
    else:
        frame = camera_manager.get_latest_frame()
    if not frame:
        return Response(status=204)
    return Response(frame, mimetype='image/jpeg')

@app.route('/api/debug/config_values')
def debug_config_values():
    try:
        res = camera_manager.send_cmd("dump_config_values", timeout=10.0)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/api/debug/widgets')
def debug_widgets():
    try:
        res = camera_manager.send_cmd("test_widgets", timeout=10.0)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/api/status', methods=['GET'])
def get_status():
    with session.lock:
        return jsonify({
            "status": session.status,
            "mode": session.mode,
            "root_folder": session.root_folder,
            "session_name": session.session_name,
            "dirs": session.dirs,
            "config": session.config
        })

@app.route('/api/stream', methods=['GET'])
def sse_stream():
    q = session.add_subscriber()
    
    def event_generator():
        # Yield initial status on connect
        with session.lock:
            initial_status = {
                "status": session.status,
                "mode": session.mode,
                "root_folder": session.root_folder,
                "session_name": session.session_name,
                "dirs": session.dirs,
                "config": session.config
            }
        import json
        yield f"event: status\ndata: {json.dumps(initial_status)}\n\n"
        
        try:
            while True:
                try:
                    # Wait for an event with a timeout (keep-alive check)
                    event_type, data = q.get(timeout=10.0)
                    yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                except queue.Empty:
                    # Send keep-alive ping to avoid connection drop
                    yield ": keep-alive\n\n"
        except GeneratorExit:
            # Browser closed the connection or navigation occurred
            pass
        finally:
            session.remove_subscriber(q)
            
    return Response(event_generator(), mimetype='text/event-stream')

@app.route('/api/start', methods=['POST'])
def start_session():
    try:
        try:
            payload = StartSessionSchema.model_validate(request.json or {})
        except ValidationError as e:
            return jsonify({"success": False, "message": f"Invalid start session payload: {format_validation_error(e)}"}), 400

        root_dir = payload.root_dir or "~/Pictures/Scans"
        session_name = payload.session_name or ""
        mode = payload.mode
        
        if isinstance(payload.config, (SessionConfigSchema, SessionConfigUpdateSchema)):
            config = payload.config.model_dump(exclude_unset=True)
        elif isinstance(payload.config, dict):
            config = payload.config
        else:
            config = {}
        
        if not session_name:
            # Generate default session name based on film stock etc.
            stock = str(payload.stock or "FilmStock").strip().replace(" ", "")
            fmt = str(payload.format or "135").strip().replace(" ", "")
            roll = str(payload.roll or "01").strip().zfill(2)
            session_name = f"{stock}-{fmt}-{roll}"

        success, msg = session.start_monitoring(root_dir, session_name, mode, config)
        return jsonify({"success": success, "message": msg})
    except Exception as e:
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

@app.route('/api/stop', methods=['POST'])
def stop_session():
    success, msg = session.stop_monitoring()
    return jsonify({"success": success, "message": msg})

@app.route('/api/logs', methods=['GET'])
def get_logs():
    return jsonify({"logs": list(session.logs)})

@app.route('/api/logs/clear', methods=['POST'])
def clear_logs():
    session.clear_logs()
    return jsonify({"success": True})

@app.route('/api/files', methods=['GET'])
def get_files():
    with session.lock:
        if not session.dirs:
            return jsonify({"success": False, "message": "No active session"})
            
        positives_dir = session.dirs.get("positives")
        processed_dir = session.dirs.get("processed")
        negatives_dir = session.dirs.get("negatives")
        
        positives = []
        if positives_dir and os.path.exists(positives_dir):
            valid_exts = {'.tif', '.tiff', '.dng', '.cr3', '.raf', '.nef', '.arw', '.rw2', '.nrw', '.dcr', '.jpg', '.jpeg', '.png'}
            files = [
                os.path.join(positives_dir, f) for f in os.listdir(positives_dir)
                if os.path.isfile(os.path.join(positives_dir, f)) and os.path.splitext(f)[1].lower() in valid_exts
            ]
            files.sort()
            for f in files:
                stat = os.stat(f)
                positives.append({
                    "name": os.path.basename(f),
                    "path": f,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime
                })
                
        processed = []
        if processed_dir and os.path.exists(processed_dir):
            files = os.listdir(processed_dir)
            files.sort()
            for f in files:
                p = os.path.join(processed_dir, f)
                if os.path.isfile(p):
                    stat = os.stat(p)
                    processed.append({
                        "name": f,
                        "path": p,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime
                    })
                    
        negatives = []
        if negatives_dir and os.path.exists(negatives_dir):
            files = os.listdir(negatives_dir)
            files.sort()
            for f in files:
                p = os.path.join(negatives_dir, f)
                if os.path.isfile(p):
                    stat = os.stat(p)
                    negatives.append({
                        "name": f,
                        "path": p,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime
                    })

        contact_sheets = []
        session_folder = os.path.join(session.root_folder, session.session_name) if session.session_name else None
        search_cs_dirs = [d for d in [session_folder, positives_dir] if d and os.path.isdir(d)]
        seen_cs = set()
        for csd in search_cs_dirs:
            for fname in os.listdir(csd):
                fl = fname.lower()
                if "_contact_sheet" in fl and (fl.endswith(".pdf") or fl.endswith(".jpg") or fl.endswith(".jpeg") or fl.endswith(".png")):
                    fp = os.path.join(csd, fname)
                    if fp not in seen_cs and os.path.isfile(fp):
                        seen_cs.add(fp)
                        stat = os.stat(fp)
                        contact_sheets.append({
                            "name": fname,
                            "path": fp,
                            "type": "pdf" if fl.endswith(".pdf") else "jpeg",
                            "size": stat.st_size,
                            "mtime": stat.st_mtime
                        })
        contact_sheets.sort(key=lambda x: (0 if x["type"] == "pdf" else 1, x["name"]))

        return jsonify({
            "success": True,
            "positives": positives,
            "processed": processed,
            "negatives": negatives,
            "contact_sheets": contact_sheets
        })


@app.route('/api/contact_sheet/generate', methods=['POST'])
def api_generate_contact_sheet():
    try:
        payload = ContactSheetGenerateSchema.model_validate(request.json or {})
    except ValidationError as e:
        return jsonify({"success": False, "message": f"Invalid contact sheet payload: {format_validation_error(e)}"}), 400

    target_dir = payload.session_dir
    with session.lock:
        if not target_dir:
            if session.dirs and session.session_name:
                target_dir = os.path.join(session.root_folder, session.session_name)
            else:
                return jsonify({"success": False, "message": "No active session. Please specify session_dir."}), 400
        cfg = dict(session.config)
        session_name = payload.session_name or session.session_name

    session.log(f"Generating contact sheet for: '{target_dir}'...")
    try:
        res = generate_contact_sheet(
            session_dir=target_dir,
            film_stock=payload.stock or "",
            film_format=payload.format or "",
            roll_number=payload.roll or "",
            session_name=session_name,
            columns=payload.columns,
            theme=payload.theme,
            config=cfg,
            export_pdf=payload.export_pdf,
            export_jpeg=payload.export_jpeg
        )
        if res.get("success"):
            session.log(f"Contact sheet complete: {res.get('message')}")
            session.broadcast("contact_sheet_generated", res)
            return jsonify(res)
        else:
            session.log(f"Contact sheet failed: {res.get('message')}")
            return jsonify(res), 400
    except Exception as e:
        session.log(f"Contact sheet error: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/contact_sheet/download', methods=['GET'])
def api_download_contact_sheet():
    file_path = request.args.get('path')
    if not file_path:
        return jsonify({"error": "Missing path parameter"}), 400
    if not session.is_safe_path(file_path):
        return jsonify({"error": "Unauthorized access path"}), 403
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    as_attachment = request.args.get('download', '0') == '1'
    mimetype = 'application/pdf' if file_path.lower().endswith('.pdf') else 'image/jpeg'
    return send_file(file_path, mimetype=mimetype, as_attachment=as_attachment, download_name=os.path.basename(file_path))

@app.route('/api/preview', methods=['GET'])
def get_preview():
    img_path = request.args.get('path')
    if not img_path:
        return jsonify({"error": "Missing path"}), 400
        
    if not session.is_safe_path(img_path):
        return jsonify({"error": "Unauthorized access path"}), 403
        
    if not os.path.exists(img_path):
        return jsonify({"error": "File not found"}), 404
        
    try:
        # Load TIFF or other format using tifffile
        ext = os.path.splitext(img_path)[1].lower()
        if ext in ['.tiff', '.tif', '.dng']:
            img = tifffile.imread(img_path)
            
            # Remove transparency or alpha channel if present
            if img.ndim == 3 and img.shape[2] > 3:
                img = img[:, :, :3]
                
            # If 16-bit, scale to 8-bit for web viewer
            if img.dtype == np.uint16:
                if ext == '.dng':
                    # Apply a standard 2.2 gamma curve to linear DNG data for web display
                    img_float = img.astype(np.float32) / 65535.0
                    img_gamma = np.clip(img_float ** (1.0 / 2.2) * 255.0, 0, 255)
                    img_8bit = img_gamma.astype(np.uint8)
                else:
                    img_8bit = (img >> 8).astype(np.uint8)
            else:
                img_8bit = img.astype(np.uint8)
                
            pil_img = Image.fromarray(img_8bit)
        elif ext in ['.cr3', '.raf', '.nef', '.arw', '.rw2', '.nrw', '.dcr']:
            try:
                import rawpy
                with rawpy.imread(img_path) as raw:
                    try:
                        thumb = raw.extract_thumb()
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            pil_img = Image.open(io.BytesIO(thumb.data))
                        elif thumb.format == rawpy.ThumbFormat.BITMAP:
                            pil_img = Image.fromarray(thumb.data)
                        else:
                            rgb = raw.postprocess(half_size=True)
                            pil_img = Image.fromarray(rgb)
                    except Exception:
                        rgb = raw.postprocess(half_size=True)
                        pil_img = Image.fromarray(rgb)
            except Exception:
                pil_img = Image.open(img_path)
        else:
            # Fallback for standard files like JPG/PNG
            pil_img = Image.open(img_path)

        # Handle thumbnail width request
        width = request.args.get('w', type=int)
        if width and width > 0:
            w_percent = (width / float(pil_img.size[0]))
            h_size = int((float(pil_img.size[1]) * float(w_percent)))
            pil_img = pil_img.resize((width, h_size), Image.Resampling.LANCZOS)
            
            
        img_io = io.BytesIO()
        # Serve as high-quality JPEG
        pil_img.save(img_io, 'JPEG', quality=85)
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
        
    except Exception as e:
        return jsonify({"error": f"Failed to generate preview: {str(e)}"}), 500

@app.route('/api/config', methods=['GET', 'POST'])
def manage_config():
    if request.method == 'POST':
        try:
            payload = SessionConfigUpdateSchema.model_validate(request.json or {})
        except ValidationError as e:
            return jsonify({"success": False, "message": f"Invalid configuration: {format_validation_error(e)}"}), 400
        
        updates = payload.model_dump(exclude_unset=True)
        with session.lock:
            session.config.update(updates)
            cfg = dict(session.config)
        session.broadcast("config_update", cfg)
        return jsonify({"success": True, "config": cfg})
    else:
        with session.lock:
            cfg = dict(session.config)
        return jsonify({"success": True, "config": cfg})

@app.route('/api/sample_rebate', methods=['POST'])
def sample_rebate():
    """
    Samples film rebate (orange mask) from either:
    1) Image path + normalized coordinates (x_ratio, y_ratio) from full-res TIFF/DNG/RAW
    2) Direct RGB values [r, g, b] sampled from UI preview / live view canvas
    Computes exact normalized neutralizer ratios [r_ratio, g_ratio, b_ratio],
    updates session config, and returns the result.
    """
    try:
        try:
            payload = SampleRebateSchema.model_validate(request.json or {})
        except ValidationError as e:
            return jsonify({"error": f"Invalid rebate payload: {format_validation_error(e)}"}), 400

        img_path = payload.path
        x_ratio = payload.x_ratio
        y_ratio = payload.y_ratio
        rgb_input = payload.rgb
        
        r_ratio, g_ratio, b_ratio = 1.0, 1.0, 1.0
        sampled_rgb = [255, 255, 255]
        
        if img_path and x_ratio is not None and y_ratio is not None:
            if not session.is_safe_path(img_path):
                return jsonify({"error": "Unauthorized access path"}), 403
            if not os.path.exists(img_path):
                return jsonify({"error": "File not found"}), 404
                
            ext = os.path.splitext(img_path)[1].lower()
            img = None
            if ext in ['.dng', '.tiff', '.tif']:
                try:
                    img = tifffile.imread(img_path)
                except Exception:
                    pass
            if img is None:
                try:
                    import rawpy
                    with rawpy.imread(img_path) as raw:
                        img = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16)
                except Exception:
                    from PIL import Image
                    pil = Image.open(img_path)
                    img = np.array(pil)
                    
            if img is not None:
                if img.ndim == 3 and img.shape[2] > 3:
                    img = img[:, :, :3]
                h, w = img.shape[:2]
                px = int(np.clip(float(x_ratio) * (w - 1), 0, w - 1))
                py = int(np.clip(float(y_ratio) * (h - 1), 0, h - 1))
                
                # Sample 5x5 patch around (py, px) to filter single-pixel sensor noise / grain
                patch_r = max(1, min(h // 100, 3))
                y0, y1 = max(0, py - patch_r), min(h, py + patch_r + 1)
                x0, x1 = max(0, px - patch_r), min(w, px + patch_r + 1)
                patch = img[y0:y1, x0:x1]
                
                if patch.ndim == 3 and patch.shape[2] >= 3:
                    r_val = float(np.median(patch[:, :, 0]))
                    g_val = float(np.median(patch[:, :, 1]))
                    b_val = float(np.median(patch[:, :, 2]))
                else:
                    val = float(np.median(patch))
                    r_val = g_val = b_val = val
                    
                max_val = max(r_val, g_val, b_val, 1e-6)
                r_ratio = round(r_val / max_val, 4)
                g_ratio = round(g_val / max_val, 4)
                b_ratio = round(b_val / max_val, 4)
                
                # Scale for display preview (0-255)
                if img.dtype == np.uint16:
                    sampled_rgb = [int(np.clip(r_val / 256.0, 0, 255)), int(np.clip(g_val / 256.0, 0, 255)), int(np.clip(b_val / 256.0, 0, 255))]
                else:
                    sampled_rgb = [int(np.clip(r_val, 0, 255)), int(np.clip(g_val, 0, 255)), int(np.clip(b_val, 0, 255))]
                    
        elif rgb_input is not None and len(rgb_input) >= 3:
            # Client sampled directly from canvas preview (8-bit sRGB)
            r_in, g_in, b_in = float(rgb_input[0]), float(rgb_input[1]), float(rgb_input[2])
            sampled_rgb = [int(np.clip(r_in, 0, 255)), int(np.clip(g_in, 0, 255)), int(np.clip(b_in, 0, 255))]
            
            # Linearize from 8-bit gamma 2.2 preview for exact transmission ratios
            r_lin = max(r_in / 255.0, 0.0) ** 2.2
            g_lin = max(g_in / 255.0, 0.0) ** 2.2
            b_lin = max(b_in / 255.0, 0.0) ** 2.2
            max_lin = max(r_lin, g_lin, b_lin, 1e-6)
            
            r_ratio = round(r_lin / max_lin, 4)
            g_ratio = round(g_lin / max_lin, 4)
            b_ratio = round(b_lin / max_lin, 4)
        else:
            return jsonify({"error": "Missing path/coordinates or rgb payload"}), 400

        with session.lock:
            session.config["base_ratios"] = [r_ratio, g_ratio, b_ratio]
            session.config["neutralize"] = True
            cfg = dict(session.config)
            
        session.log(f"Sampled film base rebate: R={r_ratio:.3f}, G={g_ratio:.3f}, B={b_ratio:.3f} (RGB: {sampled_rgb})")
        session.broadcast("config_update", cfg)
        
        return jsonify({
            "success": True,
            "base_ratios": [r_ratio, g_ratio, b_ratio],
            "sampled_rgb": sampled_rgb,
            "config": cfg
        })
    except Exception as e:
        return jsonify({"error": f"Failed to sample rebate: {str(e)}"}), 500

@app.route('/api/batch', methods=['POST'])
def run_batch():
    try:
        payload = BatchJobSchema.model_validate(request.json or {})
    except ValidationError as e:
        return jsonify({"success": False, "message": f"Invalid batch payload: {format_validation_error(e)}"}), 400

    if isinstance(payload.config, (SessionConfigSchema, SessionConfigUpdateSchema)):
        config = payload.config.model_dump(exclude_unset=True)
    elif isinstance(payload.config, dict):
        config = payload.config
    else:
        config = {}
        
    success, msg = session.run_batch_job(payload.task_type, payload.input_path, config)
    return jsonify({"success": success, "message": msg})

@app.route('/api/browse', methods=['GET'])
def browse_directory():
    path_str = request.args.get('path', '').strip()
    is_windows = (os.name == 'nt')
    
    # List Windows drives if requested
    if is_windows and (path_str == 'root'):
        import string
        import ctypes
        drives = []
        bitmask = ctypes.windll.kernel32.GetLogicalDrives()
        for letter in string.ascii_uppercase:
            if bitmask & 1:
                drives.append(f"{letter}:\\")
            bitmask >>= 1
        return jsonify({
            "current": "root",
            "parent": "",
            "drives": drives,
            "folders": []
        })

    # Default to user home if empty
    if not path_str:
        path = Path.home()
    else:
        path = Path(path_str)

    try:
        abs_path = path.resolve()
        
        # List subfolders
        folders = []
        for item in abs_path.iterdir():
            try:
                if item.is_dir() and not item.name.startswith('.'):
                    folders.append({
                        "name": item.name,
                        "path": str(item.absolute())
                    })
            except (PermissionError, FileNotFoundError):
                pass
                
        folders.sort(key=lambda x: x["name"].lower())
        
        parent = ""
        # If this is drive root (e.g. C:\) and on Windows, set parent to 'root' to go back to drives list
        if is_windows and abs_path.parent == abs_path:
            parent = "root"
        elif abs_path.parent != abs_path:
            parent = str(abs_path.parent)
            
        return jsonify({
            "current": str(abs_path),
            "parent": parent,
            "drives": [],
            "folders": folders
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "current": path_str,
            "parent": "root" if is_windows else "",
            "drives": [],
            "folders": []
        }), 400

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Start local Flask server
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5001))
    print("\n" + "="*60)
    print("STARTING FILM-CONVERT WEB UI")
    print(f"Open http://{host}:{port} in your browser.")
    print("="*60 + "\n")
    # Suppress per-request access log noise from Werkzeug (e.g. every /api/camera/frame line)
    import logging
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    app.run(host=host, port=port, debug=False)
