# Developer & Source Build Guide

This document is for developers, contributors, or advanced users who want to run FilmConvert from source, debug the application, or compile and package production installer binaries.

---

## 1. Prerequisites

To set up the development environment, ensure you have the following installed:
* **Python 3.7+**
* **Node.js (v18+) and npm**
* **Homebrew (macOS)** or **apt (Linux)** for system-level dependencies.

---

## 2. Setting Up the Project

### Step 1: Clone the Repository
Clone the codebase and enter the project folder:
```bash
git clone https://github.com/Luke-Rand/film-convert.git
cd film-convert
```

### Step 2: Set up a Python Virtual Environment
Initialize a virtual environment to isolate the Python dependencies:
* **Windows:**
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```
* **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Step 3: Install Python Dependencies
Install dependencies required by the stacking scripts, inversion processor, and web server:
```bash
pip install -r requirements.txt -r requirements-web.txt
```

### Step 4: Install gPhoto2 Binding (Optional)
If you require physical camera tethering/Live View streaming:
1. Install system library dependencies:
   * **macOS (via Homebrew):** `brew install gphoto2`
   * **Linux (Debian/Ubuntu):** `sudo apt install gphoto2 libgphoto2-dev`
2. Install the python binding wrapper:
   ```bash
   pip install gphoto2
   ```

### Step 5: Install Electron Frontend Dependencies
Install the Node.js packages required by the Electron shell wrapper:
```bash
npm install
```

---

## 3. Running in Development Mode

### Running the Python Backend Separately (Optional)
You can run the web server independently of Electron. This launches the backend on port `5001`:
```bash
python src/web_ui.py
```
Open `http://127.0.0.1:5001` in your browser.

### Running with Electron (Recommended for Frontend Dev)
To start the Electron shell in development mode (which launches both the Python Flask process and the native GUI wrapper automatically):
```bash
npm start
```

---

## 4. Packaging and Compiling Installer Binaries

To build a standalone distributable installer (e.g. a `.dmg` on macOS or an `.exe` on Windows):

### Step 1: Compile the Python Backend
Use PyInstaller to compile the Flask backend, scripts, and Python runtime into a standalone compiled binary:
*Make sure your Python virtual environment is active, and you have installed PyInstaller (`pip install pyinstaller`)*
```bash
npm run build:python
```
This writes the compiled backend helper executable to the `build/` and `dist/` subdirectories.

### Step 2: Package the Electron Wrapper
Package the Electron wrapper and include the compiled Python backend from Step 1:
```bash
npm run dist
```
* **macOS:** Produces a `.dmg` installer containing `FilmConvert.app`.
* **Windows:** Produces a setup `.exe` installer.

The final installer package will be available in the `dist-app/` directory.

### Step 3: Direct Local macOS Deployment
To build the Python backend, package the Electron application, copy it to `/Applications/FilmConvert.app`, clear quarantine bits, and sign it with required USB/camera entitlements:
```bash
npm run deploy:mac
```

---

## 5. Technical Note: Camera Tethering & macOS USB Setup

### macOS USB Daemons (`icdd` & `ptpcamerad`)
macOS includes system daemons that automatically claim any connected DSLR/mirrorless camera over USB as soon as it is plugged in or powered on:
- `icdd` (Image Capture Device Daemon): Detects hardware hotplug events.
- `ptpcamerad`: Manages PTP communications and locks the USB interface.

If active, third-party PTP libraries (like `libgphoto2`) are blocked from claiming the USB interface, resulting in `-53 (Could not claim the USB device)` or `-10 (Timeout)` errors.

Because macOS `launchd` immediately respawns `ptpcamerad` if it simply exits from `SIGKILL`, FilmConvert manages both daemons via `src/camera_manager.py`:
1. Suspends `icdd` with `SIGSTOP` (`killall -STOP icdd`) to suppress hotplug triggers.
2. Terminates any existing `ptpcamerad` process (`killall -9 ptpcamerad`).
3. Immediately places `ptpcamerad` into suspended state `T` with `SIGSTOP` (`killall -STOP ptpcamerad`). Because the process remains alive in state `T`, `launchd` does not respawn it and it cannot claim the USB interface, granting `libgphoto2` exclusive access.

### Canon EOS Remote Release & Predictive Capture Retrieval
For Canon EOS mirrorless and DSLR cameras (e.g., EOS R6 Mark III, EOS RP, EOS 5D):
1. **Manual Focus Remote Release:** To prevent the camera from hunting for focus between multi-spectral color frames, FilmConvert sequences `eosremoterelease`:
   - `Press Half MF` (engages metering without autofocus hunting)
   - `Press Full MF` (trips shutter mechanism)
   - `Release` (resets release switches)
2. **Predictive Storage & RAM Buffer Download:** Depending on camera configuration (`capturetarget` set to Memory Card or Internal RAM):
   - **RAM Capture:** The camera emits a PTP event `GP_EVENT_FILE_ADDED` with temporary buffer paths (e.g., `//capt0000.cr3`). FilmConvert catches this event and streams the RAW directly.
   - **Card Storage:** If saved directly to the SD card, `libgphoto2` internal folder listings may cache directory contents. FilmConvert snapshots the last written DCIM file before exposure (e.g., `9H0A8504.CR3`), predicts the next sequential filename candidate (`9H0A8505.CR3`), and polls direct file retrieval until write completion.

---

## 6. Technical Specifications: DNG Stacking & Inversion

FilmConvert outputs Digital Negative (DNG) files to offer standard RAW editing capabilities in editors like Lightroom.

### DNG Metadata Structure
DNG files are written using the `tifffile` library, packing 16-bit uint16 image arrays into a TIFF container containing standard DNG tags:
- **`DNGVersion` (Tag 50706):** Configured to `1.4.0.0` (bytes `b'\x01\x04\x00\x00'`).
- **`UniqueCameraModel` (Tag 50708):** Configured to `"FilmConvert Linear DNG"`.
- **`PhotometricInterpretation` (Tag 262):** Set to `34892` (LinearRaw) for RGB color files, and `1` (minisblack) for Monochrome files.
- **`ColorMatrix1` (Tag 50721):** Defines the transformation from CIE XYZ D50 to the native camera sRGB space (represented as `SRATIONAL` pairs).
- **`AsShotNeutral` (Tag 50728):** Sets default white balance multipliers to `[1.0, 1.0, 1.0]`.
- **`CalibrationIlluminant1` (Tag 50778):** Configured to `21` (D65).

### Double-Gamma Correction Prevention
To prevent double-gamma rendering in RAW editors (which automatically apply their own tone curves), `inverter.py` automatically bypasses gamma and contrast curve applications (forcing `effective_gamma = 1.0`) when exporting `.dng` files. 

### Web UI Display Gamma
Because output DNG files are saved strictly in their linear state, they would render too dark in standard web browsers. To solve this, the preview endpoint (`/api/preview` in `web_ui.py`) dynamically applies a standard `2.2` gamma display curve when generating preview thumbnails and lightbox images for the frontend.
