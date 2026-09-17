# FilmConvert Web UI API Reference

The FilmConvert Web UI exposes a JSON-based REST API and Server-Sent Events (SSE) stream to control tethered camera hardware, adjust image processing parameters, manage active film roll sessions, process digitized negative exposures, and generate archival contact sheets.

All input payloads are validated via [Pydantic schemas](file:///Users/lukerand/Documents/repos/film-convert/src/schemas.py). Invalid requests return HTTP `400 Bad Request` with descriptive field validation messages.

---

## 1. Camera Control Endpoints

These endpoints manage communication with physical camera hardware (e.g., Canon EOS R-series/DSLR, Nikon, or Sony) via `libgphoto2` using a serialized background worker thread. When no hardware is connected, FilmConvert transparently operates in simulated mode with synthetic preview feeds and parameter controls.

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

* **Request Body** (Validated by `CameraConfigSchema`):
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
Steps the camera lens motorized focus drive. Used for fine physical alignment and live framing.

* **Request Body** (Validated by `CameraFocusStepSchema`):
  ```json
  {
    "direction": "near",  // Options: "near" or "far"
    "speed": "1"          // Options: "1" (micro-step), "2" (medium), or "3" (coarse step)
  }
  ```
* **Response (200 OK)**:
  ```json
  {
    "success": true
  }
  ```

### `POST /api/camera/autofocus`
Triggers an autofocus sequence on the camera by pressing half-way down, waiting for focus lock, and releasing.

* **Response (200 OK)**:
  ```json
  {
    "success": true
  }
  ```

### `POST /api/camera/capture`
Triggers an immediate high-resolution capture and downloads the RAW image from camera storage into the session's active `negatives/` directory.

* **Request Body (Optional)**:
  ```json
  {
    "autofocus": false  // Set to false to temporarily bypass autofocus on capture (defaults to true)
  }
  ```
* **Response (200 OK)**:
  ```json
  {
    "success": true,
    "path": "/Users/lukerand/Pictures/Scans/Ektar100-135-01/negatives/Frame_01_Capture_white.cr3"
  }
  ```

### `POST /api/camera/toggle_liveview`
Toggles active live view viewfinder streaming on the camera sensor.

* **Request Body** (Validated by `CameraToggleLiveviewSchema`):
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
Disconnects active hardware instances, resets the macOS Image Capture / PTP daemon locks (`icdd` and `ptpcamerad`), and re-queries the USB bus to clear device lockups.

* **Response (200 OK)**:
  ```json
  {
    "success": true
  }
  ```

### `POST /api/camera/update_mock_leds`
Updates simulated LED brightness levels so the mock preview engine adjusts color casts in simulated mode.

* **Request Body** (Validated by `CameraMockLedsSchema`):
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
Serves a continuous real-time preview feed as an MJPEG stream with dynamic sleep backoff when paused.

* **Response**: `multipart/x-mixed-replace; boundary=frame`

### `GET /api/camera/frame`
Serves the latest viewfinder frame as a single JPEG image. Used by frontend canvas loops for high-contrast edge focus peaking and histogram rendering.

* **Response**: `image/jpeg` binary data.

---

## 2. Session & Hot Folder Monitor Endpoints

These endpoints manage scanning session configurations, folder watcher threads, and real-time processing updates.

### `GET /api/status`
Returns the active configuration, folder paths, and operational status of the session monitor.

* **Response (200 OK)**:
  ```json
  {
    "status": "monitoring",  // "idle" | "monitoring" | "batch_processing"
    "mode": "triplet",       // "triplet" | "single"
    "root_folder": "/Users/lukerand/Pictures/Scans",
    "session_name": "Ektar100-135-01",
    "dirs": {
      "negatives": "/Users/lukerand/Pictures/Scans/Ektar100-135-01/negatives",
      "positives": "/Users/lukerand/Pictures/Scans/Ektar100-135-01/positives",
      "processed": "/Users/lukerand/Pictures/Scans/Ektar100-135-01/processed_raws",
      "errors": "/Users/lukerand/Pictures/Scans/Ektar100-135-01/error_raws"
    },
    "config": {
      "clip": 0.1,
      "gamma": 2.2,
      "scurve": 0.0,
      "margin": 0.03,
      "autocrop": false,
      "global_levels": false,
      "compress_tiff": false,
      "neutralize": false,
      "base_ratios": [1.0, 0.58, 0.23],
      "align_channels": true,
      "monochrome": false,
      "monochrome_channel": "luminance",
      "reversal": false,
      "convert_to_tiff": true,
      "color_profile": "adobe_rgb",
      "embed_metadata": true,
      "auto_contact_sheet": true,
      "contact_sheet_columns": 6,
      "contact_sheet_theme": "dark"
    }
  }
  ```

### `POST /api/start`
Initializes session directories and starts the hot folder monitor thread.

* **Request Body** (Validated by `StartSessionSchema`):
  ```json
  {
    "root_dir": "~/Pictures/Scans",
    "stock": "Ektar100",
    "format": "135",
    "roll": "01",
    "mode": "triplet",
    "config": {
      "gamma": 2.2,
      "clip": 0.1,
      "align_channels": true,
      "neutralize": true,
      "color_profile": "adobe_rgb"
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
Stops the active hot folder monitor thread and triggers automatic contact sheet generation if enabled.

* **Response (200 OK)**:
  ```json
  {
    "success": true,
    "message": "Stopped folder monitoring."
  }
  ```

### `GET /api/config`
Retrieves the current session processing configuration.

* **Response (200 OK)**:
  ```json
  {
    "success": true,
    "config": {
      "clip": 0.1,
      "gamma": 2.2,
      "scurve": 0.0,
      "margin": 0.03,
      "autocrop": false,
      "global_levels": false,
      "compress_tiff": false,
      "neutralize": true,
      "base_ratios": [1.0, 0.58, 0.23],
      "align_channels": true,
      "monochrome": false,
      "monochrome_channel": "luminance",
      "reversal": false,
      "convert_to_tiff": true,
      "color_profile": "adobe_rgb",
      "embed_metadata": true,
      "auto_contact_sheet": true,
      "contact_sheet_columns": 6,
      "contact_sheet_theme": "dark"
    }
  }
  ```

### `POST /api/config`
Dynamically updates session processing configuration properties without requiring a session restart.

* **Request Body** (Validated by `SessionConfigUpdateSchema`):
  ```json
  {
    "gamma": 2.2,
    "scurve": 0.2,
    "clip": 0.05,
    "align_channels": true,
    "color_profile": "adobe_rgb"
  }
  ```
* **Response (200 OK)**:
  ```json
  {
    "success": true,
    "config": { ... }
  }
  ```

### `POST /api/sample_rebate`
Samples unexposed film rebate (orange mask) from either high-resolution image coordinates or direct RGB color values, computes normalized neutralizer ratios `[R, G, B]`, updates session configuration, and returns the calculated balance ratios.

* **Request Body** (Validated by `SampleRebateSchema`):
  * *Option A (High-Res Image Coordinates):*
    ```json
    {
      "path": "/Users/lukerand/Pictures/Scans/Ektar100-135-01/negatives/Frame_01_Capture_white.cr3",
      "x_ratio": 0.05,
      "y_ratio": 0.50
    }
    ```
  * *Option B (Direct RGB Values from Preview Canvas):*
    ```json
    {
      "rgb": [215, 128, 62]
    }
    ```
* **Response (200 OK)**:
  ```json
  {
    "success": true,
    "sampled_rgb": [215, 128, 62],
    "base_ratios": [1.0, 0.595, 0.288],
    "message": "Calculated film base ratios: R=1.000, G=0.595, B=0.288"
  }
  ```

### `GET /api/stream`
Server-Sent Events (SSE) subscription endpoint. Streams status changes, batch progress, log messages, and contact sheet notifications in real-time.

* **Mimetype**: `text/event-stream`
* **Keep-Alive**: Ping sent every 10 seconds (`: ping\n\n`).
* **Event Types**:
  * `status`: Dispatches session state, mode, folder locations, and active configuration.
  * `log`: Dispatches server log strings `{"line": "[10:39:35] ..."}`.
  * `config_update`: Dispatches updated session configuration dictionary.
  * `contact_sheet_generated`: Dispatches contact sheet creation metadata (`pdf_path`, `jpeg_paths`, `frame_count`, `pages_count`).
  * `batch_progress`: Dispatches batch processing completion counters.

---

## 3. Archival Contact Sheet Endpoints

### `POST /api/contact_sheet/generate`
Generates archival multi-page PDF documents and high-resolution JPEG contact sheets for a scanned roll session.

* **Request Body** (Validated by `ContactSheetGenerateSchema`):
  ```json
  {
    "session_dir": "/Users/lukerand/Pictures/Scans/Ektar100-135-01", // Optional (defaults to active session)
    "stock": "Kodak Ektar 100",
    "format": "135",
    "roll": "01",
    "session_name": "Ektar100-135-01",
    "columns": 6,                    // Grid columns (2 - 8, default: 6)
    "theme": "dark",                 // "dark" | "light" (default: "dark")
    "export_pdf": true,              // Generate PDF document
    "export_jpeg": true              // Generate JPEG index prints
  }
  ```
* **Response (200 OK)**:
  ```json
  {
    "success": true,
    "pdf_path": "/Users/lukerand/Pictures/Scans/Ektar100-135-01/Ektar100-135-01_contact_sheet.pdf",
    "jpeg_paths": [
      "/Users/lukerand/Pictures/Scans/Ektar100-135-01/Ektar100-135-01_contact_sheet_p1.jpg"
    ],
    "frame_count": 36,
    "pages_count": 1,
    "message": "Successfully generated contact sheet with 36 frames (1 page)"
  }
  ```

### `GET /api/contact_sheet/download`
Streams or downloads a generated contact sheet PDF or JPEG file.

* **Query Parameters**:
  * `path` (required): Absolute path to the contact sheet file.
  * `download` (optional): Set to `1` to send as an attachment download; set to `0` to view inline in the browser.
* **Response**: Binary stream with `application/pdf` or `image/jpeg` mimetype.

---

## 4. Logs & File Gallery Endpoints

### `GET /api/files`
Returns a structured listing of negatives, processed intermediate composites, converted positives, and contact sheets in the active session.

* **Response (200 OK)**:
  ```json
  {
    "success": true,
    "positives": [
      {
        "name": "Frame_01_Capture_white.tif",
        "path": "/Users/lukerand/Pictures/Scans/Ektar100-135-01/positives/Frame_01_Capture_white.tif",
        "size": 50381920,
        "mtime": 1782846831.0
      }
    ],
    "processed": [],
    "negatives": [
      {
        "name": "Frame_01_Capture_white.cr3",
        "path": "/Users/lukerand/Pictures/Scans/Ektar100-135-01/negatives/Frame_01_Capture_white.cr3",
        "size": 32184910,
        "mtime": 1782846800.0
      }
    ],
    "contact_sheets": [
      {
        "name": "Ektar100-135-01_contact_sheet.pdf",
        "path": "/Users/lukerand/Pictures/Scans/Ektar100-135-01/Ektar100-135-01_contact_sheet.pdf",
        "type": "pdf",
        "size": 4125890,
        "mtime": 1782847000.0
      }
    ]
  }
  ```

### `GET /api/preview`
Loads, scales, and downsamples images (16-bit TIFFs, DNGs, and camera RAWs) to high-quality JPEGs for web rendering. For linear DNGs, dynamically applies a `2.2` gamma display curve for accurate browser display.

* **Query Parameters**:
  * `path` (required): Absolute path to the source image file.
  * `w` (optional): Requested target width in pixels for thumbnail sizing.
* **Response**: `image/jpeg` binary data.

### `GET /api/browse`
A folder-hierarchy browser API used to navigate host file systems in folder picker dialogs.

* **Query Parameters**:
  * `path` (optional): Directory path string to inspect. Defaults to the user home directory.
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
Submits an offline batch job to composite or invert an existing directory of frame files using multiprocessing worker pools.

* **Request Body** (Validated by `BatchJobSchema`):
  ```json
  {
    "task_type": "invert",  // Options: "invert" | "composite"
    "input_path": "/Users/lukerand/Pictures/Scans/Ektar100-135-01/negatives",
    "config": {
      "gamma": 2.2,
      "clip": 0.005,
      "scurve": 0.2
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

---

## 5. Diagnostics & Hardware Debugging

### `GET /api/debug/config_values`
Returns the raw camera configuration tree values queried from the tethered `gphoto2` context.

* **Response (200 OK)**: Key-value map of camera settings parameters.

### `GET /api/debug/widgets`
Returns matching diagnostic tests and widget types for registered camera components.
