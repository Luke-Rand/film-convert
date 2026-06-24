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

# Import core logic from existing scripts
from compositor import process_triplet
from inverter import process_positives

app = Flask(__name__, template_folder='templates', static_folder='static')

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
    def __init__(self):
        self.status = "idle"  # idle, monitoring, batch_processing
        self.mode = "triplet"  # triplet, single
        self.root_folder = os.path.abspath(os.path.expanduser("~/Pictures/Scans"))
        self.session_name = ""
        self.dirs = {}
        self.config = {
            "clip": 0.1,
            "gamma": 2.2,
            "scurve": 0.0,
            "margin": 0.03,
            "autocrop": False,
            "global_levels": False,
            "compress_tiff": True,
            "neutralize": False,  # compositor neutralization
            "align_channels": False
        }
        self.logs = deque(maxlen=1000)
        self.monitor_thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.subscribers = []
        self.subscribers_lock = threading.Lock()
        self.log("System initialized. Ready.")

    def add_subscriber(self):
        with self.subscribers_lock:
            q = queue.Queue()
            self.subscribers.append(q)
            return q

    def remove_subscriber(self, q):
        with self.subscribers_lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def broadcast(self, event_type, data):
        with self.subscribers_lock:
            for q in self.subscribers:
                q.put((event_type, data))

    def broadcast_status(self):
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

    def log(self, message):
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

    def clear_logs(self):
        with self.lock:
            self.logs.clear()
            self.log("Logs cleared.")

    def is_safe_path(self, path):
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

    def start_monitoring(self, root_dir, session_name, mode, config):
        with self.lock:
            if self.status != "idle":
                return False, f"Cannot start monitoring, current status is '{self.status}'"

            self.root_folder = os.path.abspath(os.path.expanduser(root_dir))
            os.makedirs(self.root_folder, exist_ok=True)
            
            self.session_name = session_name
            self.mode = mode
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
        
        self.broadcast_status()
        return True, "Monitoring started successfully"

    def stop_monitoring(self):
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
            
        self.broadcast_status()
        return True, "Monitoring stopped successfully"

    def get_next_frame_number(self, negatives_dir):
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

    def _monitor_loop(self):
        supported_triplet_exts = {'.cr3', '.raf'}
        supported_single_exts = {'.dng', '.tiff', '.tif'}
        
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
                        self.log(f"Triplet detected! Processing Frame {frame_number:02d}...")
                        
                        composite_filename = f"Frame_{frame_number:02d}_Composite.tiff"
                        composite_filepath = os.path.join(neg_dir, composite_filename)
                        
                        try:
                            # 1. Composite (redirect stdout to web log)
                            with contextlib.redirect_stdout(redirector):
                                process_triplet(
                                    group=group,
                                    output_filepath=composite_filepath,
                                    neutralize_base=self.config["neutralize"],
                                    compress_tiff=self.config["compress_tiff"],
                                    align_channels=self.config["align_channels"]
                                )
                            
                            # 2. Invert (redirect stdout to web log)
                            with contextlib.redirect_stdout(redirector):
                                process_positives(
                                    input_path=composite_filepath,
                                    output_dir=self.dirs['positives'],
                                    clip=self.config["clip"],
                                    gamma=self.config["gamma"],
                                    compress_tiff=self.config["compress_tiff"],
                                    global_levels=self.config["global_levels"],
                                    ignore_margin=self.config["margin"],
                                    scurve=self.config["scurve"],
                                    autocrop=self.config["autocrop"]
                                )
                            
                            # 3. Move files
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
                            
                        self.log(f"Negative detected! Processing {filename}...")
                        
                        try:
                            with contextlib.redirect_stdout(redirector):
                                process_positives(
                                    input_path=filepath,
                                    output_dir=self.dirs['positives'],
                                    clip=self.config["clip"],
                                    gamma=self.config["gamma"],
                                    compress_tiff=self.config["compress_tiff"],
                                    global_levels=self.config["global_levels"],
                                    ignore_margin=self.config["margin"],
                                    scurve=self.config["scurve"],
                                    autocrop=self.config["autocrop"]
                                )
                            
                            shutil.move(filepath, os.path.join(self.dirs['processed'], filename))
                            self.log(f"SUCCESS: {filename} processed and saved.")
                            
                        except Exception as e:
                            self.log(f"ERROR processing {filename}: {str(e)}")
                            try:
                                shutil.move(filepath, os.path.join(self.dirs['errors'], filename))
                            except Exception: pass
                
            except Exception as e:
                self.log(f"Monitor loop error: {str(e)}")
                
            # Sleep using event wait to enable clean stop
            self.stop_event.wait(1.0)
            
        self.log("Background scanner monitor loop stopped.")

    def run_batch_job(self, task_type, input_path, config):
        with self.lock:
            if self.status != "idle":
                return False, f"System is currently '{self.status}'"
            
            self.status = "batch_processing"
            self.config.update(config)
            
        self.broadcast_status()

        def _batch_thread():
            redirector = ThreadLogRedirector(self.log)
            self.log(f"Starting batch task: {task_type} for '{input_path}'")
            
            try:
                # Resolve paths
                in_path = os.path.abspath(os.path.expanduser(input_path))
                if not os.path.exists(in_path):
                    raise FileNotFoundError(f"Input path '{in_path}' does not exist.")
                
                if task_type == 'composite':
                    # Composite RAW files in folder
                    out_dir = os.path.join(in_path, "Composites")
                    self.log(f"Processing RAW roll in: {in_path}")
                    self.log(f"Output folder: {out_dir}")
                    
                    # Search files
                    supported_exts = {'.cr3', '.raf'}
                    raw_files = [
                        os.path.join(in_path, f) for f in os.listdir(in_path)
                        if os.path.isfile(os.path.join(in_path, f)) and os.path.splitext(f)[1].lower() in supported_exts
                    ]
                    
                    if not raw_files:
                        raise ValueError(f"No .cr3 or .raf files found in {in_path}")
                        
                    raw_files.sort()
                    total_files = len(raw_files)
                    self.log(f"Found {total_files} RAW files.")
                    
                    os.makedirs(out_dir, exist_ok=True)
                    
                    frame_number = 1
                    for i in range(0, total_files - 2, 3):
                        group = raw_files[i:i+3]
                        self.log(f"Processing Frame {frame_number:02d} ({[os.path.basename(f) for f in group]})...")
                        output_filepath = os.path.join(out_dir, f"Frame_{frame_number:02d}_Composite.tiff")
                        
                        with contextlib.redirect_stdout(redirector):
                            process_triplet(
                                group=group,
                                output_filepath=output_filepath,
                                neutralize_base=self.config["neutralize"],
                                compress_tiff=self.config["compress_tiff"],
                                align_channels=self.config["align_channels"]
                            )
                        frame_number += 1
                    
                    self.log("Batch compositing complete!")
                    
                elif task_type == 'invert':
                    # Invert composite images
                    self.log(f"Processing positive inversions for: {in_path}")
                    
                    # We let the inverter script figure out outputs
                    with contextlib.redirect_stdout(redirector):
                        process_positives(
                            input_path=in_path,
                            output_dir=None,  # let it auto-create subfolder Positives
                            clip=self.config["clip"],
                            gamma=self.config["gamma"],
                            compress_tiff=self.config["compress_tiff"],
                            global_levels=self.config["global_levels"],
                            ignore_margin=self.config["margin"],
                            scurve=self.config["scurve"],
                            autocrop=self.config["autocrop"]
                        )
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

# --- WEB CONTROLLER ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

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
    data = request.json or {}
    root_dir = data.get("root_dir", "~/Pictures/Scans")
    session_name = data.get("session_name", "")
    mode = data.get("mode", "triplet")
    config = data.get("config", {})
    
    if not session_name:
        # Generate default session name based on film stock etc.
        stock = data.get("stock", "FilmStock").replace(" ", "")
        fmt = data.get("format", "135").replace(" ", "")
        roll = str(data.get("roll", "01")).strip().zfill(2)
        session_name = f"{stock}-{fmt}-{roll}"

    success, msg = session.start_monitoring(root_dir, session_name, mode, config)
    return jsonify({"success": success, "message": msg})

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
            files = glob.glob(os.path.join(positives_dir, "*.tif*"))
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
                    
        return jsonify({
            "success": True,
            "positives": positives,
            "processed": processed,
            "negatives": negatives
        })

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
        if ext in ['.tiff', '.tif']:
            img = tifffile.imread(img_path)
            
            # Remove transparency or alpha channel if present
            if img.ndim == 3 and img.shape[2] > 3:
                img = img[:, :, :3]
                
            # If 16-bit, scale to 8-bit for web viewer
            if img.dtype == np.uint16:
                img_8bit = (img >> 8).astype(np.uint8)
            else:
                img_8bit = img.astype(np.uint8)
                
            pil_img = Image.fromarray(img_8bit)
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

@app.route('/api/batch', methods=['POST'])
def run_batch():
    data = request.json or {}
    task_type = data.get("task_type")  # composite, invert
    input_path = data.get("input_path")
    config = data.get("config", {})
    
    if not task_type or not input_path:
        return jsonify({"success": False, "message": "Missing task_type or input_path"}), 400
        
    success, msg = session.run_batch_job(task_type, input_path, config)
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
    # Start local Flask server
    print("\n" + "="*60)
    print("STARTING FILM-CONVERT WEB UI")
    print("Open http://127.0.0.1:5001 in your browser.")
    print("="*60 + "\n")
    app.run(host="127.0.0.1", port=5001, debug=False)
