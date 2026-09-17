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

## 4. Automated Testing Suite

FilmConvert maintains a comprehensive testing suite covering backend processing algorithms, camera drivers, metadata preservation, ICC profiles, API validation, and frontend interactions.

### Running Python Unit & Integration Tests
Execute the full pytest suite from the project root:
```bash
pytest
```

To run specific subsystems:
```bash
# Test subpixel FFT alignment and channel registration
pytest tests/test_triplet_alignment_subpixel.py

# Test sensitometric inversion, curves, and base neutralization
pytest tests/test_inverter.py tests/test_film_base_neutralization_picker.py

# Test archival contact sheet generation and metadata extraction
pytest tests/test_contact_sheet.py

# Test true DNG tags, EXIF preservation, and ICC color profile tagging
pytest tests/test_metadata_and_icc.py tests/test_dng_artifacts.py

# Test Pydantic API validation schemas
pytest tests/test_schemas_and_validation.py

# Test multiprocessing concurrency and throughput
pytest tests/test_perf_concurrency.py
```

### Running Frontend End-to-End Tests (Playwright)
Run the automated browser test suite against the web frontend:
```bash
# Install test dependencies if running for the first time
npx playwright install --with-deps

# Run the Playwright test suite
npx playwright test
```

---

## 5. Packaging and Compiling Installer Binaries

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

## 6. Architecture & Subsystems Overview

```
FilmConvert Architecture
├── Frontend (Electron / Vanilla JS)
│   ├── static/js/app.js           <-- Live view canvas, shortcuts, REST/SSE client
│   └── static/js/scanlight.js     <-- WebSerial hardware LED controller
├── Web Backend (Flask / REST / SSE)
│   ├── src/web_ui.py              <-- API endpoints, SSE dispatcher, session monitor
│   └── src/schemas.py             <-- Pydantic request models & data validators
├── Hardware & Acquisition
│   └── src/camera_manager.py      <-- libgphoto2 USB tethering, daemon control, autofocus
└── Image Processing Core
    ├── src/batch_worker.py        <-- ProcessPoolExecutor multiprocessing tasks
    ├── src/compositor.py          <-- RAW parsing, FFT subpixel alignment, RGB stacking
    ├── src/inverter.py            <-- Density inversion, S-curves, channel auto-levels
    ├── src/tiff_writer.py         <-- Striped 16-bit TIFF & True DNG packing
    ├── src/icc_manager.py         <-- Embedded ICC profile tagging (Adobe RGB, ROMM, sRGB)
    ├── src/metadata_preservation.py <-- Lossless EXIF/IPTC copying via exiftool
    └── src/contact_sheet.py       <-- Archival multi-page PDF & JPEG roll summaries
```

### Multiprocessing & Concurrency (`src/batch_worker.py`)
To prevent heavy CPU computations (RAW demosaicing, FFT phase correlation, 16-bit float matrix inversions) from stalling the Flask HTTP server or interrupting the Live View camera thread, operations are executed across a spawned `ProcessPoolExecutor`. Functions in `src/batch_worker.py` are isolated top-level tasks designed for serialization across worker processes.

### Pydantic Validation Layer (`src/schemas.py`)
All REST API inputs are validated using Pydantic V2 models. Constraints (e.g. clipping percentiles `0.0 <= p <= 10.0`, gamma curves `0.1 <= g <= 5.0`, base transmission ratios length 3) are enforced before reaching core processing pipelines.

### Archival Contact Sheet Engine (`src/contact_sheet.py`)
Extracts frame-level EXIF parameters (exposure time, aperture, ISO, focal length, camera model, lens) and renders multi-page PDF documents and JPEG contact sheets. Dynamically formats frame badges, handles varying aspect ratios (135, 120 6x6/6x7), and optimizes font sizes across platforms.

### ICC Profile Management (`src/icc_manager.py`)
Bundles standardized ICC profiles in `src/icc_profiles/` (Adobe RGB 1998, ProPhoto RGB / ROMM, sRGB, and Generic Gray Gamma 2.2). Profiles are losslessly embedded into output TIFFs and DNGs to ensure exact color reproduction in color-managed NLEs and photo editing software.

### Metadata Preservation (`src/metadata_preservation.py`)
Uses `exiftool` to losslessly replicate full camera EXIF and IPTC structures from original RAW frames to composite and positive output files, including lens serial numbers, exposure timestamps, and shooting tags.

---

## 7. Technical Note: Camera Tethering & macOS USB Setup

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

## 8. Technical Specifications: DNG Stacking & Inversion

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

