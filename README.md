# Tri-Color Auto Compositor & Film Inverter for RAW Scans

A suite of Python scripts designed to automate the process of combining individual Red, Green, and Blue RAW film scans into high-quality, 16-bit linear composite TIFF images, and accurately inverting them into positive images. This toolset is perfect for archival film scanning setups that use distinct red, green, and blue monochromatic light sources to capture color film.

## Features

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

### 1. Compositing RAWs

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
