# FilmConvert Web UI API Reference

The FilmConvert Web UI exposes a JSON-based REST API and Server-Sent Events (SSE) stream to control tethered camera hardware, adjust parameters, manage active film roll sessions, and process digitized negative exposures.

---

## 1. Camera Control Endpoints

These endpoints manage communication with physical camera hardware (e.g., Canon EOS RP) via `libgphoto2` using a serialized worker-thread model. When no hardware is connected, it transparently falls back to simulated settings and visual preview generators.

### `GET /api/camera/status`
Reads the active connection state, cached parameters, and available choices from the camera.

* **Response (200 OK - Simulated Mode)**:
  ```json
  {
    "connected": true,
    "simulated": true,
    "settings": {
      "iso": "100",
      "aperture": "5.6",
      "shutterspeed": "0.4"
    },
    "choices": {
      "iso": ["Auto", "100", "200", "400", "800", "1600", "3200", "6400"],
      "aperture": ["2.8", "3.5", "4", "5.6", "8", "11", "16", "22"],
      "shutterspeed": ["1/500", "1/250", "1/125", "1/60", "1/30", "1/15", "1/8", "1/4", "1/2", "1", "2", "auto"]
    }
  }
  ```

### `POST /api/camera/config`
Updates camera settings like ISO, Aperture, or Shutter Speed.

* **Request Body**:
  ```json
  {
    "name": "iso",
    "value": "400"
  }
  ```
* **Response (200 OK)**:
  ```json
  {
    "success": true
  }
  ```

### `POST /api/camera/focus_step`
Steps the camera lens focus motor. Used for precision physical alignment.

* **Request Body**:
  ```json
  {
    "direction": "near",  // Options: "near" or "far"
    "speed": "3"          // Options: "1" (micro-step) or "3" (coarse step)
  }
  ```
* **Response (200 OK)**:
  ```json
  {
    "success": true
  }
  ```

### `POST /api/camera/autofocus`
Sequences an autofocus lock command on the camera by pressing half-way down, waiting for a latch delay, and releasing.

* **Response (200 OK)**:
  ```json
  {
    "success": true
  }
  ```

### `POST /api/camera/capture`
Triggers an immediate high-resolution capture and downloads the RAW image from the camera storage into the session's active `negatives/` directory.

* **Request Body (Optional)**:
  ```json
  {
    "autofocus": false  // Set to false to temporarily disable autofocus on capture (defaults to true)
  }
  ```
* **Response (200 OK)**:
  ```json
  {
    "success": true,
    "path": "/Users/lukerand/Pictures/Scans/SessionName/negatives/Frame_01_Capture_white.cr3"
  }
  ```

### `POST /api/camera/toggle_liveview`
Toggles active live view streaming state on the camera sensor.

* **Request Body**:
  ```json
  {
    "active": true  // Options: true (start stream) or false (shut down sensor preview loop)
  }
  ```
* **Response (200 OK)**:
  ```json
  {
    "success": true
  }
  ```

### `POST /api/camera/reconnect`
Disconnects active hardware instances and triggers a re-query on the USB bus. This forces the PTP connection routine to run, clearing state lockups without requiring a server shutdown.

* **Response (200 OK)**:
  ```json
  {
    "success": true
  }
  ```

### `POST /api/camera/update_mock_leds`
Tells the simulated mock engine what RGB light balances are active so it can adjust color casts inside mock preview frames.

* **Request Body**:
  ```json
  {
    "red": 255,
    "green": 200,
    "blue": 180
  }
  ```
* **Response (200 OK)**:
  ```json
  {
    "success": true
  }
  ```

### `GET /api/camera/liveview`
Serves a continuous real-time preview feed as an MJPEG stream.

* **Response**: `multipart/x-mixed-replace; boundary=frame`

### `GET /api/camera/frame`
Serves the latest viewfinder frame in the buffer as a single JPEG image. Used by the browser canvas polling loop for edge detection filters.

* **Response**: `image/jpeg` binary data.

---

## 2. Session & Hot Folder Monitor Endpoints

These endpoints manage roll monitoring settings and folder watcher threads.

### `GET /api/status`
Returns the active configuration and operational state of the hot folder watcher.

* **Response (200 OK)**:
  ```json
  {
    "status": "monitoring",  // Options: "idle" or "monitoring"
    "mode": "triplet",       // Options: "triplet" or "single"
    "root_folder": "/Users/lukerand/Pictures/Scans",
    "session_name": "Ektar100-135-01",
    "dirs": {
      "negatives": "/Users/lukerand/Pictures/Scans/Ektar100-135-01/negatives",
      "positives": "/Users/lukerand/Pictures/Scans/Ektar100-135-01/positives",
      "processed": "/Users/lukerand/Pictures/Scans/Ektar100-135-01/processed"
    },
    "config": {
      "gamma": 2.2,
      "clip": 0.005,
      "scurve": 1.2,
      "margin": 0.05,
      "autocrop": true,
      "compress_dng": true,
      "neutralize": true,
      "monochrome": false
    }
  }
  ```

### `POST /api/start`
Starts monitoring a folder directory for incoming camera file captures.

* **Request Body**:
  ```json
  {
    "root_dir": "~/Pictures/Scans",
    "stock": "Ektar100",
    "format": "135",
    "roll": "01",
    "mode": "triplet",
    "config": {
      "gamma": 2.2,
      "clip": 0.01
    }
  }
  ```
* **Response (200 OK)**:
  ```json
  {
    "success": true,
    "message": "Started monitoring /Users/lukerand/Pictures/Scans/Ektar100-135-01"
  }
  ```

### `POST /api/stop`
Stops the active folder monitor thread.

* **Response (200 OK)**:
  ```json
  {
    "success": true,
    "message": "Stopped folder monitoring."
  }
  ```

### `GET /api/stream`
Server-Sent Events (SSE) subscription endpoint. Streams status changes, processing progress events, and new log lines in real-time.

* **Mimetype**: `text/event-stream`
* **Pings**: Keep-alive ping sent every 10 seconds.
* **Events**:
  * `status`: Dispatches roll configuration and activity changes.
  * `log`: Dispatches new server-side activity logs.

---

## 3. Logs & File Gallery Endpoints

### `GET /api/logs`
Returns historical session log lines.

* **Response (200 OK)**:
  ```json
  {
    "logs": [
      "[10:39:35] System initialized. Ready.",
      "[10:39:35] [Camera] Worker thread started."
    ]
  }
  ```

### `POST /api/logs/clear`
Clears the active logs cache.

* **Response (200 OK)**:
  ```json
  {
    "success": true
  }
  ```

### `GET /api/files`
Returns a list of negatives, processed frames, and neutralized/inverted positives in the active session.

* **Response (200 OK)**:
  ```json
  {
    "success": true,
    "positives": [
      {
        "name": "Frame_01_Capture_white.tif",
        "path": "/Users/lukerand/Pictures/Scans/Ektar100-135-01/positives/Frame_01_Capture_white.tif",
        "size": 5038192,
        "mtime": 1782846831.0
      }
    ],
    "processed": [],
    "negatives": []
  }
  ```

### `GET /api/preview`
Loads, scales, and downsamples raw or processed images (including 16-bit TIFF files) to high-quality JPEGs for web rendering.

* **Query Parameters**:
  * `path` (required): Absolute path to the source image file.
  * `w` (optional): Requested target width in pixels for thumbnail sizing.
* **Response**: `image/jpeg` binary data.

### `GET /api/browse`
A folder-hierarchy browser API used to select host directories via local folder picker dialogs.

* **Query Parameters**:
  * `path` (optional): Path string to inspect. Defaults to user home directory. Pass `"root"` on Windows to list drive mount nodes.
* **Response (200 OK)**:
  ```json
  {
    "current": "/Users/lukerand/Pictures",
    "parent": "/Users/lukerand",
    "drives": [],
    "folders": [
      {
        "name": "Scans",
        "path": "/Users/lukerand/Pictures/Scans"
      }
    ]
  }
  ```

### `POST /api/batch`
Triggers an offline batch run to composites or inverts existing folders of frames.

* **Request Body**:
  ```json
  {
    "task_type": "invert",  // Options: "invert" or "composite"
    "input_path": "/Users/lukerand/Pictures/Scans/Ektar100-135-01/negatives",
    "config": {
      "gamma": 2.2,
      "clip": 0.005
    }
  }
  ```
* **Response (200 OK)**:
  ```json
  {
    "success": true,
    "message": "Batch processing started."
  }
  ```

---

## 4. Diagnostics & Debugging Endpoints

### `GET /api/debug/config_values`
Returns the raw camera configuration tree values from the tethered `gphoto2` context.

* **Response (200 OK)**: Key-value map of found settings parameters.

### `GET /api/debug/widgets`
Returns matching diagnostic tests for registered setting components.
