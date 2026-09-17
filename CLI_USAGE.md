# Command Line Interface (CLI) Usage Guide

FilmConvert provides a comprehensive suite of CLI tools designed for scanning session automation, RAW stacking, sensitometric density inversion, batch archive reprocessing, and archival contact sheet generation.

---

## 1. Automated Session Manager (`src/scanning_session.py`)

The session manager watches a folder, automatically composites RAW triplets as they are captured by your camera, inverts them into positives, preserves EXIF metadata, embeds ICC color profiles, and generates archival contact sheets upon session completion.

### Launching a Session
Run the script to launch the interactive setup:
```bash
python src/scanning_session.py
```

### Interactive Setup Steps:
1. **Root Directory:** Choose the base directory where scan folders will be created (e.g., `~/Pictures/Scans`).
2. **Scan Mode:** 
   * **Triplet:** Expects sequential triplets of RAW files (`.CR3`, `.RAF`, or `.NEF`) corresponding to Red, Green, and Blue exposures.
   * **Single-shot:** Expects single DNG/RAW/TIFF captures (for standard white light sources).
3. **Details:** Enter Film Stock name, Format (135, 120), and Roll Number.
4. **Monochrome Check:** Specify if it's a Black & White scanning session and select a preferred extraction channel.
5. **Slide / Reversal Film:** Specify if scanning positive transparency film (bypasses negative density inversion).

The script creates organized directories and runs a hot-folder loop:
```
SessionName/
├── negatives/       <-- Capture your images here
├── positives/       <-- Completed positive 16-bit TIFFs land here
├── processed_raws/  <-- Original RAWs are safely archived here
├── error_raws/      <-- Errored shots are moved here to prevent pipeline blocks
└── SessionName_contact_sheet.pdf  <-- Auto-generated roll summary document
```

---

## 2. RAW Tri-Color Compositor (`src/compositor.py`)

The compositor sorts and groups RAW images (Canon `.CR3`, Fujifilm `.RAF`, Nikon `.NEF`, or standard DNGs), auto-detects Red, Green, and Blue channels, aligns subpixel shifts via FFT phase correlation, and stacks them into a 16-bit linear composite TIFF or True DNG.

### Basic Usage
Combine files in a folder into composites inside a `Composites` subdirectory:
```bash
python src/compositor.py -i /path/to/raw/files
```

### Advanced Usage (Alignment, Neutralization, Custom Base Ratios, and Compression)
Enable subpixel FFT alignment, neutralize the orange mask using custom sampled transmission ratios, and compress the output:
```bash
python src/compositor.py -i /path/to/raw/files --align --base-ratios 1.0 0.58 0.23 --compress
```

### CLI Arguments

| Argument | Short | Description |
| :--- | :---: | :--- |
| `--input` | `-i` | **(Required)** Path to directory containing RAW files (`.CR3`, `.RAF`, `.NEF`, or `.DNG`). |
| `--compress` | `-c` | Enable optional zlib compression for output TIFFs (default: uncompressed for maximum NLE & DaVinci Resolve compatibility). |
| `--neutralize` | `-n` | Automatically balance color channels to neutralize the film base mask. |
| `--base-ratios` | | Custom film base neutralizer RGB transmission ratios (`R G B`, e.g. `1.0 0.58 0.23`) sampled from film rebate. Automatically enables neutralization. |
| `--align` | `-a` | Auto-correct subpixel shift between channels (R, G, B) using FFT phase correlation. |
| `--icc-profile` | | Embedded ICC color profile: `adobe_rgb` (default), `prophoto_rgb`, `srgb`, or `none`. |
| `--no-metadata` | | Disable copying original camera RAW EXIF/IPTC metadata into composites. |
| `--hotfolder` | | Run in Hot Folder mode to monitor directory, composite files in real-time, and archive originals. |
| `--timeout` | `-t` | Timeout in seconds to wait for subsequent exposures in hot folder mode (default: `60`). |

---

## 3. Density Inverter (`src/inverter.py`)

The inverter takes composite 16-bit linear TIFFs or camera RAW files and inverts them, applies tone curves, normalizes levels, embeds ICC profiles, and crops borders into 16-bit TIFF positives.

### Basic Usage
Invert linear images inside a folder into positives in a `Positives` subdirectory:
```bash
python src/inverter.py -i /path/to/Composites
```

### Advanced Usage (Auto-Crop, Level Clipping, Contrast Curve, Custom Film Base)
Apply per-channel auto-levels with 0.1% clipping, a viewing gamma of 2.2, a photographic S-curve, custom rebate ratios, and physical border auto-cropping:
```bash
python src/inverter.py -i /path/to/Composites --clip 0.1 --gamma 2.2 --scurve 0.3 --margin 0.03 --autocrop --base-ratios 1.0 0.58 0.23
```

### CLI Arguments

| Argument | Short | Description |
| :--- | :---: | :--- |
| `--input` | `-i` | **(Required)** Path to a single 16-bit composite TIFF/DNG file, or a directory containing them. |
| `--compress` | `-c` | Enable optional zlib compression for output TIFFs (default: uncompressed). |
| `--clip` | `-p` | Percentile to clip for black/white points (default: `0.1`% to reject dust and scratches). |
| `--gamma` | `-g` | Gamma correction curve to apply (default: `2.2`). Set to `1.0` for strictly linear output. |
| `--scurve` | `-s` | Strength of contrast S-curve to apply (default: `0.0` = none; `0.2` to `0.5` recommended for film punch). |
| `--margin` | `-m` | Fraction of outer edge to ignore when calculating histogram levels (default: `0.03` = 3%). |
| `--autocrop` | `-a` | Physically crop off the outer margins defined by `--margin` from the final saved image. |
| `--global-levels` | | Stretch levels globally across all RGB channels simultaneously instead of per-channel. |
| `--base-ratios` | | Custom film base neutralizer RGB transmission ratios (`R G B`, e.g. `1.0 0.58 0.23`). |
| `--monochrome` / `--bw` | | Convert output composite to a single-channel grayscale positive. |
| `--monochrome-channel` | | Channel extraction method: `luminance` (default), `average`, `red`, `green` (recommended for Bayer sensors), or `blue`. |
| `--reversal` | | Enable reversal / slide film mode (positive film). Bypasses negative density inversion. |
| `--no-tiff` | | Bypass conversion to TIFF and add original RAW files directly to the positives folder. |
| `--icc-profile` | | Embedded ICC color profile (`adobe_rgb`, `prophoto_rgb`, `srgb`, `none`; default: `adobe_rgb`). |
| `--no-metadata` | | Disable embedding original camera RAW EXIF/IPTC metadata into output files. |

---

## 4. Archival Contact Sheet & Roll Summary Generator (`src/contact_sheet.py`)

Generates archival multi-page PDF documents and high-resolution JPEG contact sheet index prints complete with frame numbers, exposure parameters (shutter speed, aperture, ISO), camera & lens tags, and roll summary headers.

### Basic Usage
Generate a contact sheet for a completed roll session:
```bash
python src/contact_sheet.py -i /path/to/SessionFolder
```

### Advanced Usage (Custom Roll Info, Layout, and Light Theme)
```bash
python src/contact_sheet.py -i /path/to/SessionFolder --stock "Kodak Portra 400" --format "135" --roll "02" --columns 6 --theme dark -o /path/to/output
```

### CLI Arguments

| Argument | Short | Description |
| :--- | :---: | :--- |
| `--input` | `-i` | **(Required)** Path to session directory (or `positives/` subdirectory) containing scans. |
| `--output` | `-o` | Output directory where contact sheets will be saved (defaults to session directory). |
| `--stock` | | Film stock name displayed in document header (e.g. `"Kodak Ektar 100"`). |
| `--format` | | Film format displayed in document header (e.g. `"135"`, `"120"`). |
| `--roll` | | Roll identification index string (e.g. `"01"`). |
| `--columns` | | Number of thumbnail grid columns per row (default: `6`, range: 2–8). |
| `--theme` | | Visual styling theme: `dark` (default darkroom style) or `light` (proof print style). |
| `--no-pdf` | | Disable generating the multi-page archival PDF document. |
| `--no-jpeg` | | Disable generating high-resolution JPEG index prints. |

---

## 5. Batch Roll Reprocessor (`reprocess_rolls.py`)

Automates batch discovery and re-processing of entire directories containing multiple scanned rolls. Automatically discovers nested roll folders, runs subpixel FFT channel alignment, applies sensitometric tone reproduction, and outputs contact sheets for every roll.

### Basic Usage
Reprocess all rolls in an archive directory:
```bash
python reprocess_rolls.py /path/to/scans_archive
```

### Advanced Usage (Custom Neutralization, Gamma, Contrast, and Contact Sheets)
```bash
python reprocess_rolls.py /path/to/scans_archive --gamma 2.2 --scurve 0.2 --neutralize --base-ratios 1.0 0.58 0.23 --autocrop --contact-sheet-columns 6
```

### CLI Arguments

| Argument | Short | Description |
| :--- | :---: | :--- |
| `directory` | | **(Positional)** Path to parent directory containing rolls or a single roll folder. |
| `--gamma` | `-g` | Output gamma curve (default: `2.2`; set to `1.0` for strictly linear output). |
| `--clip` | `-p` | Percentile clipping for black/white levels (default: `0.1`%). |
| `--margin` | `-m` | Fraction of outer edge to ignore for histogram calculations (default: `0.03` = 3%). |
| `--scurve` | `-s` | Contrast S-Curve strength (default: `0.0`). |
| `--autocrop` | `-a` | Automatically crop outer borders defined by `--margin`. |
| `--global-levels` | | Preserve scene chromaticity with joint RGB exposure scaling. |
| `--neutralize` | `-n` | Neutralize film base mask during compositing. |
| `--base-ratios` | | Custom film base neutralizer RGB ratios (`R G B`, e.g. `1.0 0.58 0.23`). |
| `--compress` | `-c` | Enable zlib compression for output TIFFs. |
| `--no-align` | | Disable sub-pixel FFT channel alignment. |
| `--monochrome` / `--bw` | | Convert output to monochrome/B&W. |
| `--monochrome-channel` | | Channel extraction method (`luminance`, `average`, `red`, `green`, `blue`). |
| `--reversal` | | Process positive slide / reversal film without density inversion. |
| `--icc-profile` | | Embedded ICC color profile (`adobe_rgb`, `prophoto_rgb`, `srgb`, `none`). |
| `--no-metadata` | | Disable embedding camera RAW EXIF/IPTC metadata into composites and positives. |
| `--no-contact-sheet` | | Disable automatic contact sheet generation after roll processing. |
| `--contact-sheet-columns` | | Number of columns on contact sheet grid (default: `6`). |

---

## How It Works & Best Practices

1. **16-Bit Precision & ColorSync / ICC Tagging:** Output files are stored in standard 16-bit TIFF or True DNG format with 64-row strip chunking for instant loading in video editors like DaVinci Resolve and image editors like Lightroom and Photoshop. Each output file has an embedded ICC profile tag (Adobe RGB 1998 or ProPhoto RGB / ROMM) to ensure consistent tone curve reproduction without clipping across all color-managed viewers.
2. **Archival Metadata Preservation & True DNG:** Original camera RAW EXIF/IPTC metadata (exposure, lens model, camera serial, capture timestamp) is losslessly copied from source files into the final composites and positives via `exiftool`. Output DNG files include full DNG specification tags (`DNGVersion 1.4.0.0`, `ColorMatrix1`, `CalibrationIlluminant1`, `UniqueCameraModel`).
3. **Channel Auto-Detection:** Triplet files are automatically analyzed by average brightness to identify Red, Green, and Blue exposures regardless of shot order.
4. **Subpixel FFT Alignment:** Phase correlation identifies subpixel mechanical shifts and optical divergence between monochromatic filter passes, eliminating color fringing.
