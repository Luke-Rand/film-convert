# Tri-Color Auto Compositor & Film Inverter for RAW Scans

A suite of Python scripts designed to automate the process of combining individual Red, Green, and Blue RAW film scans into high-quality, 16-bit linear composite TIFF images, and accurately inverting them into positive images. This toolset is perfect for archival film scanning setups that use distinct red, green, and blue monochromatic light sources to capture color film.

## Features

### Web User Interface (`web_ui.py`)
- **Visual Control Panel:** Set directories, select scanning modes, and fine-tune gamma, clipping, margin parameters, and tone curves interactively.
- **Tethered Camera Controls (New):** Connect, configure, and control physical digital cameras (e.g. Canon EOS RP via `libgphoto2`). Stream live view previews directly to the browser, adjust ISO, Aperture, and Shutter Speed, and trigger raw captures. Includes background daemon release for macOS system processes (like `ptpcamerad`).
- **Interactive Crop Guide Overlay (New):** Toggle a visual boundary overlay indicating the exact border crop area (configured via "Ignore Outer Margins") mapped precisely to the camera's contained aspect-ratio.
- **Real-Time Feed:** Stream process log updates dynamically to an event console and browse output positive files in a session gallery.
- **On-the-Fly Previews:** Dynamic resizing and rendering of 16-bit TIFF composites directly to standard JPEG in the browser without writing duplicate files to disk.
- **Manual Batch Jobs:** Execute bulk compositing or inversions on-demand on existing film folders.
- **Scanlight Controller (Optional):** Direct integration with the Jackw01 Big Scanlight and Scanlight v4 using the browser's WebSerial API. Adjust brightness, manage local presets, calibrate balance offsets, and run automated sequential captures.

### Session Manager (`scanning_session.py`)
- **Interactive Setup:** Prompts for film stock, format, and roll number to automatically generate organized directory structures.
- **End-to-End Automation:** Runs a hot folder monitor that instantly composites RAW triplets as they are captured and pipes them directly into the inverter.
- **Clean File Management:** Automatically moves processed and errored RAW files into designated subfolders to keep your working directory tidy.

### Compositor (`compositor.py`)
- **Auto-Color Detection:** Automatically determines which shot is the Red, Green, or Blue channel by analyzing the average brightness of the RAW data.
- **Linear 16-bit Processing:** Uses raw sensor data (`rawpy.ColorSpace.raw`) bypassing standard sRGB matrices to ensure pure channel data without cross-channel contamination.
- **Film Base Neutralization:** Optional automated color balancing that calculates the 99.9th percentile of brightness to neutralize the unexposed film base color cast.
- **Lossless Compression:** Optional `zlib`/`deflate` compression for the output TIFFs to save disk space.
- **Broad Camera Support:** Supports `.CR3` (Canon), `.RAF` (Fujifilm), and `.NEF` (Nikon) RAW formats.

### Inverter (`inverter.py`)
- **Accurate Density Inversion:** Uses true mathematical division for linear data, maintaining contrast across highlights and shadows.
- **Auto-Levels & Tone Curve Control:** Removes remaining color casts (per-channel clipping), applies a viewing gamma (default 2.2), and supports optional photographic S-curves for punchy contrast.
- **Auto-Cropping:** Option to physically crop or just ignore film holder borders during auto-level calculation.
- **Black & White (Monochrome) Inversion:** Convert RGB scans to high-quality single-channel grayscale positives using weighted luminance, average, or single-channel extraction. Handles native single-channel inputs automatically.

## Installation & Setup

### Path A: Desktop Application (Recommended - No Code Required)

If you just want to run FilmConvert, you do not need to install Python or compile the project from source. 

1. Download the latest packaged application for your operating system from the [GitHub Releases](https://github.com/Luke-Rand/film-convert/releases) page.
2. *(macOS Users)* Bypass Gatekeeper to allow the unsigned application to run:
   * Move the downloaded `FilmConvert.app` to your `/Applications` folder.
   * Open Terminal and execute:
     ```bash
     xattr -d com.apple.quarantine /Applications/FilmConvert.app
     ```
3. *(Optional)* If tethering a **physical camera** for Live View capture, install the `gphoto2` system driver:
   * **macOS (via Homebrew):** `brew install gphoto2`
   * **Linux (Debian/Ubuntu):** `sudo apt install gphoto2`

---

### Path B: Command Line & Source Development (Developers)

If you are developing, modifying the Electron wrapper, or prefer running via CLI scripts:

#### Prerequisites
* Python 3.7+
* Node.js (v18+) and npm (only required if packaging/modifying the Electron app)

#### Setup Steps
1. Clone this repository and enter the directory:
   ```bash
   git clone https://github.com/Luke-Rand/film-convert.git
   cd film-convert
   ```
2. Create and activate a Python virtual environment:
   * **Windows:**
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   * **macOS/Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt -r requirements-web.txt
   ```
4. *(Optional)* If using physical camera tethering:
   * Install system libraries:
     * **macOS:** `brew install gphoto2`
     * **Linux:** `sudo apt install gphoto2 libgphoto2-dev`
   * Install python gphoto2 bindings:
     ```bash
     pip install gphoto2
     ```
5. *(Optional)* Install Electron development dependencies:
   ```bash
   npm install
   ```

> [!NOTE]
> **macOS USB Connection Notice:** macOS has a built-in background daemon (`ptpcamerad`) that automatically claims any connected DSLR/mirrorless camera over USB. The Web UI runs a background release script to terminate it upon launch. If you still encounter `-53 (Could not claim USB device)` or `-10 (Timeout)` connection errors, power-cycle the camera.


## Usage

Depending on your setup method from [Installation & Setup](#installation--setup), choose the appropriate way to launch and use FilmConvert:

### 1. Launching the Desktop App (Path A)

Simply double-click the installed **FilmConvert** application. It will automatically initialize the local Python server in the background and launch the user interface inside a self-contained window.

### 2. Launching from Source (Path B)

#### Run the Web User Interface
To start the Flask-based local web server and run the entire suite in your web browser:
```bash
python web_ui.py
```
Open `http://127.0.0.1:5001` in your browser. You can configure scan settings, start/stop hot folder monitoring, view real-time log outputs, run manual batch tasks, and view completed scans in the gallery.

#### Scanlight Controller Integration (Optional)
If you own the **Jackw01 Big Scanlight** or **Scanlight v4**, you can access the **Scanlight** tab to control the device directly:
1. Connect the light to your computer via USB (using a Chromium-based browser like Google Chrome or Microsoft Edge).
2. Click **Connect Big Scanlight** to initialize communication. The device metrics (hardware model, firmware, VBUS input voltage, and temperature) will load in real-time.
3. Presets can be created and saved, and manual adjustment of LED levels is available.
4. Set shutter pulse length, post-shutter delays, and sequence preferences, then click any sequence button (like **Auto R,G,B**) to automatically capture sequential frames.
5. **Exposure Auto-Calibration:** You can automatically detect optimal RGB channel brightness values:
   * Start a folder monitoring session on the **Live Scanner** tab.
   * Go to the **Scanlight** tab and click **Auto-Calibrate RGB Exposures**.
   * The controller runs a test RGB sequence at a reference power level of 150.
   * The compositor measures the captured RAW frames' average channel intensity (0-65,535 in 16-bit linear RAW).
   * It calculates the proportional scaling factors to achieve a target exposure level of 55,000.
   * If any channel exceeds the maximum LED value of 255, the highest channel is capped and others are scaled proportionally to preserve color balance.
   * The new calibrated values are applied to your active Red, Green, and Blue sliders.

#### Run the Interactive CLI Session Pipeline
To use the fully automated CLI folder monitoring session manager:
```bash
python scanning_session.py
```
You will be prompted to enter a root directory, film stock, format, and roll number. The script will create an organized session folder and start monitoring the `negatives` subfolder. 

As you shoot your RGB triplets into the `negatives` folder, the script will automatically:
1. Detect the 3 RAW files.
2. Composite them into a 16-bit linear TIFF.
3. Invert, auto-color balance, and apply an S-curve to create a positive image.
4. Save the final positive to the `positives` folder.
5. Move the original RAWs and the intermediate composite to the `processed_raws` folder.

#### Manual Source-Build Compilation (Optional)
If you wish to build/compile a production binary of the application yourself:
```bash
# 1. Compile the Python backend
npm run build:python

# 2. Package the Electron app
npm run dist
```
*The packaged installer will be generated in the `dist-app/` directory.*

### 4. Manual Compositing RAWs

The compositor script runs via the command line and requires the path to a directory containing your RAW files.

### Basic Usage

```bash
python compositor.py -i /path/to/your/raw/files
````

*This will process the files, auto-detect the channels, and output uncompressed 16-bit TIFFs into a new `Composites` subfolder.*

### Advanced Usage (with Neutralization and Compression)

```bash
python compositor.py -i /path/to/your/raw/files --neutralize --compress
```

### Command Line Arguments

| **Argument** | **Short** | **Description** |
| :--- | :---: | :--- |
| `--input` | `-i` | **(Required)** Path to the directory containing RAW files (`.CR3`, `.RAF`, or `.NEF`). |
| `--compress` | `-c` | Enable lossless compression (`zlib`/`deflate`) for output TIFFs. |
| `--neutralize` | `-n` | Automatically balance the color channels to neutralize the film base. |
| `--align` | `-a` | Auto-correct exposure alignment between channels (R, G, B) using FFT phase correlation. |

## How It Works & Best Practices

1.  **File Organization:** The script sorts all `.CR3`, `.RAF`, and `.NEF` files in the provided directory alphabetically.
2.  **Groups of Three:** It processes files in **sequential groups of 3**. You *must* ensure that every 3 consecutive files represent the Red, Green, and Blue exposures for a single frame.
3.  **Color Independence:** Because the script auto-detects the color of the light source used for each shot based on channel luminosity, the 3 shots for a frame can be in any order (e.g., R-G-B, B-G-R, G-B-R), as long as they are grouped together.
4.  **Output:** Processed composites are saved as `Frame_01_Composite.tiff`, `Frame_02_Composite.tiff`, etc., in a `Composites` folder inside your input directory.

### Important Limitations

- The total number of RAW files in the directory **must be divisible by 3**. If you have misfires or test shots in the folder, remove them before running the script so the sequence isn't thrown off.
- The script expects distinct RGB monochromatic light sources. If a shot is heavily mixed or exposed incorrectly, auto-detection may fail.

### Black & White Film Scanning Best Practices

When converting black and white film negatives, selecting the correct channel extraction method depends on your light source and hardware:

* **Green Channel Method (Recommended for Bayer Sensors):** Digital camera sensors have double the density of green pixels compared to red or blue. Extracting only the green channel (`green`) yields the highest native spatial resolution, avoids demosaicing interpolation artifacts, and minimizes lens chromatic aberrations.
* **Using RGB Monochromatic Light (e.g. Scanlight):** If your light source allows individual color channel adjustment, illuminate **only the Green LED** and scan in **Single-Shot** mode. Discarding Red and Blue completely eliminates color fringing and channel crosstalk.
* **Using Standard White Light:** If scanning with a static white light pad, you can still use the **Green Channel** extraction method for maximum resolution. Alternatively, use the **Luminance** (`luminance`) method to combine channels into a traditional panchromatic tonal range.

### 5. Manual Inverting Composites

The inverter script takes your 16-bit composite TIFFs (or single RAW DNGs) and accurately inverts them, normalizes levels, and applies gamma and contrast curves.

### Basic Usage

```bash
python inverter.py -i /path/to/your/Composites
```

*This will process the files, invert them, apply default auto-levels (0.1% clipping) and gamma (2.2), and save them into a new `Positives` subfolder.*

### Advanced Usage

```bash
python inverter.py -i /path/to/your/Composites --compress --scurve 0.3 --autocrop
```

### Command Line Arguments

| **Argument** | **Short** | **Description** |
| :--- | :---: | :--- |
| `--input` | `-i` | **(Required)** Path to a single 16-bit composite TIFF/RAW DNG file, or a directory containing them. |
| `--compress` | `-c` | Enable lossless compression (`zlib`/`deflate`) for output TIFFs. |
| `--clip` | `-p` | Percentile to clip for black/white points (default: `0.1`% to ignore dust/scratches). |
| `--gamma` | `-g` | Gamma correction curve to apply (default: `2.2`). Set to 1.0 for strictly linear output. |
| `--scurve` | `-s` | Strength of the contrast S-Curve to apply (default: `0.0` = none). Try 0.2 to 0.5 for a film-like punch. |
| `--margin` | `-m` | Fraction of outer edge to ignore when calculating levels (default: `0.03` = 3%). Prevents film holders from skewing brightness. |
| `--autocrop` | `-a` | Physically crop off the outer margins defined by `--margin` from the final saved image. |
| `--global-levels` | | Stretch levels globally instead of per-channel. Use this if you relied on the compositor's neutralization and want to perfectly maintain that color balance. |
| `--monochrome` / `--bw` | | Convert output composite to monochrome / black and white positive. |
| `--monochrome-channel` / `--bw-channel` | | Method to convert RGB to monochrome: `luminance` (standard weighted), `average`, `red`, `green` (recommended), or `blue`. |

## Credits & Attributions

- The Scanlight control protocols, automatic sequence patterns, and device command structures are adapted from the official [Scanlight Project](https://github.com/jackw01/scanlight) created by [jackw01](https://github.com/jackw01). This feature is designed to interface with the Big Scanlight / Scanlight v4 hardware.
