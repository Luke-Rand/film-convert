# Command Line Interface (CLI) Usage Guide

FilmConvert provides a suite of CLI tools designed for scanning session automation, RAW stacking, and image density inversion. If you prefer working in the terminal or scripting workflows, use this guide.

---

## 1. Automated Session Manager (`src/scanning_session.py`)

The session manager watches a folder, automatically composites RAW triplets as they are captured by your camera, inverts them into positive positives, and organizes your files.

### Launching a Session
Run the script to launch the interactive setup:
```bash
python src/scanning_session.py
```

### Interactive Setup Steps:
1. **Root Directory:** Choose the base directory where scan folders will be created (e.g., `~/Pictures/Scans`).
2. **Scan Mode:** 
   * **Triplet:** Expects sequential triplets of RAW files (.CR3, .RAF, or .NEF) corresponding to Red, Green, and Blue exposures.
   * **Single-shot:** Expects single DNG captures (for standard white light sources).
3. **Details:** Enter Film Stock name, Format (135, 120), and Roll Number.
4. **Monochrome Check:** Specify if it's a Black & White scanning session and select a preferred extraction channel.

The script creates organized directories and runs a hot-folder loop:
```
SessionName/
├── negatives/       <-- Capture your images here
├── positives/       <-- Completed positive DNGs land here
├── processed_raws/  <-- Original RAWs are safely archived here
└── error_raws/      <-- Errored shots are moved here to prevent pipeline blocks
```

---

## 2. RAW Tri-Color Compositor (`src/compositor.py`)

The compositor sorts and groups RAW images (Canon `.CR3`, Fujifilm `.RAF`, or Nikon `.NEF`), auto-detects Red, Green, and Blue channels, and stacks them into a single 16-bit linear composite DNG.

### Basic Usage
Combine files in a folder into composites inside a new `Composites` subdirectory:
```bash
python src/compositor.py -i /path/to/raw/files
```

### Advanced Usage (With Channel Alignment and Compression)
Enable channel auto-alignment via FFT phase correlation, neutralize the film base orange cast, and compress the output DNG:
```bash
python src/compositor.py -i /path/to/raw/files --align --neutralize --compress
```

### CLI Arguments

| Argument | Short | Description |
| :--- | :---: | :--- |
| `--input` | `-i` | **(Required)** Path to the directory containing RAW files (`.CR3`, `.RAF`, or `.NEF`). |
| `--compress` | `-c` | Enable lossless compression (`zlib`/`deflate`) for output DNGs. |
| `--neutralize` | `-n` | Automatically balance the color channels to neutralize the film base. |
| `--align` | `-a` | Auto-correct exposure alignment between channels (R, G, B) using FFT phase correlation. |
| `--hotfolder` | | Run in Hot Folder mode to monitor a folder, composite files on the fly, and archive originals. |
| `--timeout` | `-t` | Timeout in seconds to wait for the 3rd exposure in hot folder mode (default: `60`). |

---

## 3. Density Inverter (`src/inverter.py`)

The inverter takes composite 16-bit linear DNGs or single RAW DNG files and inverts them, applies S-curves, stretches levels, and crops borders.

### Basic Usage
Invert linear images inside a folder into positives in a new `Positives` subdirectory:
```bash
python src/inverter.py -i /path/to/Composites
```

### Advanced Usage (Auto-Crop, Level Clipping, Contrast Curve)
Apply per-channel auto-levels with 0.1% clipping, a viewing gamma of 2.2, a 30% photographic S-curve, and physically crop out the outer margins:
```bash
python src/inverter.py -i /path/to/Composites --clip 0.1 --gamma 2.2 --scurve 0.3 --margin 0.03 --autocrop --compress
```

### CLI Arguments

| Argument | Short | Description |
| :--- | :---: | :--- |
| `--input` | `-i` | **(Required)** Path to a single 16-bit composite DNG or camera RAW DNG file, or a directory containing them. |
| `--compress` | `-c` | Enable lossless compression (`zlib`/`deflate`) for output DNGs. |
| `--clip` | `-p` | Percentile to clip for black/white points (default: `0.1`% to ignore dust/scratches). |
| `--gamma` | `-g` | Gamma correction curve to apply (default: `2.2`). Set to `1.0` for strictly linear output. |
| `--scurve` | `-s` | Strength of the contrast S-curve to apply (default: `0.0` = none). Try `0.2` to `0.5`. |
| `--margin` | `-m` | Fraction of outer edge to ignore when calculating levels (default: `0.03` = 3%). |
| `--autocrop` | `-a` | Physically crop off the outer margins defined by `--margin` from the final saved image. |
| `--global-levels` | | Stretch levels globally instead of per-channel (maintains compositor neutralization). |
| `--monochrome` / `--bw` | | Convert output composite to a single-channel grayscale positive. |
| `--monochrome-channel` | | Method to convert RGB to monochrome: `luminance`, `average`, `red`, `green` (recommended), or `blue`. |

---

## How It Works & Best Practices

### 1. File Organization & Triplets
* **Alphabetic Sorting:** The compositor sorts all supported RAW files (`.CR3`, `.RAF`, `.NEF`) in the input folder alphabetically.
* **Groups of Three:** Files are processed in sequential groups of three. You must ensure that every three consecutive files represent the Red, Green, and Blue exposures for a single film frame. Remove any test shots or misfires before running.
* **Color Auto-Detection:** The script automatically determines which shot corresponds to Red, Green, or Blue by analyzing average sensor brightness. As long as they are consecutive, you can capture them in any order (e.g., R-G-B, B-G-R).

### 2. Black & White Film Scanning Best Practices
When converting black and white film negatives, select the correct channel extraction method depending on your setup:
* **Green Channel Method (`green`):** (Recommended for Bayer Sensors) Digital camera sensors have double the density of green pixels compared to red or blue. Extracting only the green channel yields the highest native spatial resolution, avoids demosaicing interpolation artifacts, and minimizes lens chromatic aberrations.
* **Using RGB Monochromatic Light (e.g., Scanlight):** Illuminate only the Green LED on your light source and capture in Single-Shot mode. Discarding Red and Blue completely eliminates color fringing and channel crosstalk.
* **Using Standard White Light:** If scanning with a static white light pad, you can still use the **Green Channel** extraction method for maximum resolution. Alternatively, use the **Luminance** (`luminance`) method to combine channels into a traditional panchromatic tonal range.

### 3. Important Limitations
* The total number of RAW files in the directory must be divisible by 3 when running batch stack jobs.
* The compositor expects distinct monochromatic light sources. If a shot is heavily mixed or exposed incorrectly, auto-detection may fail.
