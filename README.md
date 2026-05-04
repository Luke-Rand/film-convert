# Tri-Color Auto Compositor & Film Inverter for RAW Scans

A suite of Python scripts designed to automate the process of combining individual Red, Green, and Blue RAW film scans into high-quality, 16-bit linear composite TIFF images, and accurately inverting them into positive images. This toolset is perfect for archival film scanning setups that use distinct red, green, and blue monochromatic light sources to capture color film.

## Features

### Session Manager (`scanning_session.py`)
- **Interactive Setup:** Prompts for film stock, format, and roll number to automatically generate organized directory structures.
- **End-to-End Automation:** Runs a hot folder monitor that instantly composites RAW triplets as they are captured and pipes them directly into the inverter.
- **Clean File Management:** Automatically moves processed and errored RAW files into designated subfolders to keep your working directory tidy.

### Compositor (`compositor.py`)
- **Auto-Color Detection:** Automatically determines which shot is the Red, Green, or Blue channel by analyzing the average brightness of the RAW data.
- **Linear 16-bit Processing:** Uses raw sensor data (`rawpy.ColorSpace.raw`) bypassing standard sRGB matrices to ensure pure channel data without cross-channel contamination.
- **Film Base Neutralization:** Optional automated color balancing that calculates the 99.9th percentile of brightness to neutralize the unexposed film base color cast.
- **Lossless Compression:** Optional `zlib`/`deflate` compression for the output TIFFs to save disk space.
- **Broad Camera Support:** Currently supports `.CR3` (Canon) and `.RAF` (Fujifilm) RAW formats.

### Inverter (`inverter.py`)
- **Accurate Density Inversion:** Uses true mathematical division for linear data, maintaining contrast across highlights and shadows.
- **Auto-Levels & Tone Curve Control:** Removes remaining color casts (per-channel clipping), applies a viewing gamma (default 2.2), and supports optional photographic S-curves for punchy contrast.
- **Auto-Cropping:** Option to physically crop or just ignore film holder borders during auto-level calculation.

## Prerequisites

- Python 3.7+
- Required Python packages: `rawpy`, `numpy`, `tifffile`

## Installation

1. Clone this repository or download the scripts.
2. (Recommended) Create and activate a Python virtual environment to keep dependencies isolated:

   **Windows:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS/Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required dependencies using pip:

```bash
pip install rawpy numpy tifffile
```

## Usage

### 1. End-to-End Workflow (Recommended)

The session manager provides an interactive, fully automated pipeline.

```bash
python scanning_session.py
```

You will be prompted to enter a root directory, film stock, format, and roll number. The script will create an organized session folder and start monitoring the `negatives` subfolder. 

As you shoot your RGB triplets into the `negatives` folder, the script will automatically:
1. Detect the 3 RAW files.
2. Composite them into a 16-bit linear TIFF.
3. Invert, auto-color balance, and apply an S-curve to create a positive image.
4. Save the final positive to the `positives` folder.
5. Move the original RAWs to the `processed_raws` folder.

### 2. Manual Compositing RAWs

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
| `--input` | `-i` | **(Required)** Path to the directory containing RAW files (`.CR3` or `.RAF`). |
| `--compress` | `-c` | Enable lossless compression (`zlib`/`deflate`) for output TIFFs. |
| `--neutralize` | `-n` | Automatically balance the color channels to neutralize the film base. |

## How It Works & Best Practices

1.  **File Organization:** The script sorts all `.CR3` and `.RAF` files in the provided directory alphabetically.
2.  **Groups of Three:** It processes files in **sequential groups of 3**. You *must* ensure that every 3 consecutive files represent the Red, Green, and Blue exposures for a single frame.
3.  **Color Independence:** Because the script auto-detects the color of the light source used for each shot based on channel luminosity, the 3 shots for a frame can be in any order (e.g., R-G-B, B-G-R, G-B-R), as long as they are grouped together.
4.  **Output:** Processed composites are saved as `Frame_01_Composite.tiff`, `Frame_02_Composite.tiff`, etc., in a `Composites` folder inside your input directory.

### Important Limitations

- The total number of RAW files in the directory **must be divisible by 3**. If you have misfires or test shots in the folder, remove them before running the script so the sequence isn't thrown off.
- The script expects distinct RGB monochromatic light sources. If a shot is heavily mixed or exposed incorrectly, auto-detection may fail.

### 3. Manual Inverting Composites

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
