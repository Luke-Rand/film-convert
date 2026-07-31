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

---

## 5. Technical Note: Camera Tethering & macOS USB Setup

### macOS USB Daemon (`ptpcamerad`)
macOS has a built-in background daemon (`ptpcamerad`) that claims any connected DSLR/mirrorless camera over USB as soon as it is powered on. This blocks third-party libraries (like `libgphoto2`) from claiming the USB device, resulting in `-53 (Could not claim the USB device)` or `-10 (Timeout)` connection errors.

FilmConvert handles this automatically in `src/camera_manager.py`:
1. Disables `com.apple.ptpcamerad` auto-respawn via `launchctl disable gui/$UID/com.apple.ptpcamerad`.
2. Releases any running instance via `killall -9 ptpcamerad` and allows a 0.3s settling window for `libgphoto2` to bind USB Interface 0 cleanly.

### Canon EOS R-Series Cameras (R6 Mark III, R5, R3, etc.)
- **CFexpress Type B Cards**: Inserting a CFexpress Type B card can cause Canon camera firmware to default to USB Mass Storage / Card Reader mode, blocking PTP tethering handshakes and causing `[-10] Timeout` errors. **Remove the CFexpress card** or ensure USB app mode is set to "Photo Import / Remote Control".
- **Wireless Connections**: Disable Wi-Fi / Bluetooth in the camera menu (`Wi-Fi/Bluetooth connection` -> `Disable`) so the camera unlocks the USB PTP interface.
