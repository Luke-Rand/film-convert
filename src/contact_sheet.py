"""
Automated Roll Summary & Contact Sheet Generator for FilmConvert.

Generates archival multi-page PDF documents and high-resolution JPEG index prints
containing thumbnail grids, frame numbers, exposure metadata, and roll identification
for lab archiving.
"""

import os
import sys
import re
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tifffile

try:
    from metadata_preservation import find_exiftool
except ImportError:
    try:
        from src.metadata_preservation import find_exiftool
    except ImportError:
        def find_exiftool():
            return None


def get_font(size: int = 14, bold: bool = False) -> ImageFont.ImageFont:
    """Loads a high-quality system font cross-platform with fallback to PIL default font."""
    font_candidates = []
    if sys.platform == "darwin":
        if bold:
            font_candidates = [
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/SFNSTextBold.ttf",
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/System/Library/Fonts/Menlo.ttc"
            ]
        else:
            font_candidates = [
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/SFNSText.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Menlo.ttc"
            ]
    elif sys.platform == "win32":
        font_candidates = [
            "C:\\Windows\\Fonts\\arialbd.ttf" if bold else "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\calibri.ttf",
            "arial.ttf"
        ]
    else:  # Linux / Unix
        font_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "DejaVuSans.ttf"
        ]

    for p in font_candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue

    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def extract_frame_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extracts exposure, camera, and geometry metadata for a single scan.
    """
    meta: Dict[str, Any] = {
        "filename": os.path.basename(image_path),
        "path": image_path,
        "width": 0,
        "height": 0,
        "megapixels": 0.0,
        "shutter": "",
        "aperture": "",
        "iso": "",
        "focal_length": "",
        "camera": "",
        "lens": "",
        "date_time": "",
        "frame_num": 0
    }

    # Extract frame number from filename (e.g. Frame_01_Positive.tiff -> 1)
    base_name = os.path.basename(image_path)
    match = re.search(r'Frame_(\d+)', base_name, re.IGNORECASE)
    if match:
        meta["frame_num"] = int(match.group(1))
    else:
        match_any = re.search(r'(\d+)', base_name)
        if match_any:
            meta["frame_num"] = int(match_any.group(1))

    # Fast exiftool extraction if available
    exiftool = find_exiftool()
    if exiftool and os.path.exists(image_path):
        try:
            cmd = [
                exiftool,
                "-s3",
                "-ImageWidth",
                "-ImageHeight",
                "-ExposureTime",
                "-FNumber",
                "-ISO",
                "-FocalLength",
                "-Model",
                "-LensModel",
                "-DateTimeOriginal",
                image_path
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if res.returncode == 0:
                lines = [l.strip() for l in res.stdout.splitlines()]
                if len(lines) >= 9:
                    w = int(lines[0]) if lines[0].isdigit() else 0
                    h = int(lines[1]) if lines[1].isdigit() else 0
                    if w and h:
                        meta["width"] = w
                        meta["height"] = h
                        meta["megapixels"] = round((w * h) / 1_000_000, 1)
                    if lines[2]:
                        meta["shutter"] = lines[2] if "s" in lines[2] or "/" in lines[2] else f"{lines[2]}s"
                    if lines[3]:
                        meta["aperture"] = f"f/{lines[3]}" if not lines[3].startswith("f/") else lines[3]
                    if lines[4]:
                        meta["iso"] = f"ISO {lines[4]}" if not lines[4].startswith("ISO") else lines[4]
                    if lines[5]:
                        meta["focal_length"] = lines[5]
                    if lines[6]:
                        meta["camera"] = lines[6]
                    if lines[7]:
                        meta["lens"] = lines[7]
                    if lines[8]:
                        meta["date_time"] = lines[8]
        except Exception:
            pass

    # If geometry not found, inspect with PIL or tifffile
    if meta["width"] == 0 or meta["height"] == 0:
        try:
            ext = os.path.splitext(image_path)[1].lower()
            if ext in ['.tiff', '.tif', '.dng']:
                with tifffile.TiffFile(image_path) as tif:
                    page = tif.pages[0]
                    meta["height"], meta["width"] = page.shape[:2]
                    meta["megapixels"] = round((meta["width"] * meta["height"]) / 1_000_000, 1)
            else:
                with Image.open(image_path) as img:
                    meta["width"], meta["height"] = img.size
                    meta["megapixels"] = round((meta["width"] * meta["height"]) / 1_000_000, 1)
        except Exception:
            pass

    return meta


def load_thumbnail(image_path: str, target_w: int, target_h: int) -> Optional[Image.Image]:
    """Loads an image file, normalizes it, and resizes to target bounding box."""
    try:
        ext = os.path.splitext(image_path)[1].lower()
        pil_img = None

        if ext in ['.tiff', '.tif', '.dng']:
            try:
                img_data = tifffile.imread(image_path)
                if img_data.ndim == 3 and img_data.shape[2] > 3:
                    img_data = img_data[:, :, :3]

                if img_data.dtype == np.uint16:
                    if ext == '.dng':
                        # 2.2 gamma curve for linear DNG
                        img_f = img_data.astype(np.float32) / 65535.0
                        img_gamma = np.clip(img_f ** (1.0 / 2.2) * 255.0, 0, 255)
                        img_8 = img_gamma.astype(np.uint8)
                    else:
                        img_8 = (img_data >> 8).astype(np.uint8)
                else:
                    img_8 = img_data.astype(np.uint8)

                pil_img = Image.fromarray(img_8)
            except Exception:
                pass

        if pil_img is None and ext in ['.cr3', '.raf', '.nef', '.arw', '.rw2', '.nrw', '.dcr']:
            try:
                import rawpy
                with rawpy.imread(image_path) as raw:
                    try:
                        thumb = raw.extract_thumb()
                        import io
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            pil_img = Image.open(io.BytesIO(thumb.data))
                        elif thumb.format == rawpy.ThumbFormat.BITMAP:
                            pil_img = Image.fromarray(thumb.data)
                        else:
                            rgb = raw.postprocess(half_size=True)
                            pil_img = Image.fromarray(rgb)
                    except Exception:
                        rgb = raw.postprocess(half_size=True)
                        pil_img = Image.fromarray(rgb)
            except Exception:
                pass

        if pil_img is None:
            pil_img = Image.open(image_path)

        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        # Compute aspect ratio fit
        orig_w, orig_h = pil_img.size
        aspect = orig_w / float(orig_h)

        if aspect > (target_w / float(target_h)):
            new_w = target_w
            new_h = max(1, int(target_w / aspect))
        else:
            new_h = target_h
            new_w = max(1, int(target_h * aspect))

        thumb = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return thumb

    except Exception as e:
        print(f"Warning: Failed loading thumbnail for {image_path}: {e}")
        return None


def generate_contact_sheet(
    session_dir: str,
    output_dir: Optional[str] = None,
    output_basename: Optional[str] = None,
    film_stock: str = "",
    film_format: str = "",
    roll_number: str = "",
    session_name: str = "",
    columns: int = 6,
    theme: str = "dark",  # "dark" or "light"
    dpi: int = 300,
    config: Optional[Dict[str, Any]] = None,
    export_pdf: bool = True,
    export_jpeg: bool = True
) -> Dict[str, Any]:
    """
    Generates archival PDF and high-resolution JPEG contact sheets for a session directory.

    Returns dict with paths to generated files:
    {
        "success": bool,
        "pdf_path": str or None,
        "jpeg_paths": [str],
        "frame_count": int,
        "pages_count": int,
        "message": str
    }
    """
    session_dir = os.path.abspath(os.path.expanduser(session_dir))
    if not os.path.exists(session_dir):
        return {"success": False, "message": f"Directory not found: {session_dir}"}

    # Discover positive scans
    valid_exts = {'.tiff', '.tif', '.dng', '.jpg', '.jpeg', '.png', '.cr3', '.raf', '.nef', '.arw'}
    positives_dir = os.path.join(session_dir, "positives")
    
    scan_files: List[str] = []
    if os.path.isdir(positives_dir):
        files = [
            os.path.join(positives_dir, f) for f in os.listdir(positives_dir)
            if os.path.isfile(os.path.join(positives_dir, f)) and os.path.splitext(f)[1].lower() in valid_exts
        ]
        scan_files.extend(files)

    if not scan_files:
        # Fallback to scanning session_dir directly if no 'positives' subfolder
        files = [
            os.path.join(session_dir, f) for f in os.listdir(session_dir)
            if os.path.isfile(os.path.join(session_dir, f)) and os.path.splitext(f)[1].lower() in valid_exts
            and not f.endswith("_contact_sheet.jpg") and not f.endswith("_contact_sheet.png")
        ]
        scan_files.extend(files)

    if not scan_files:
        return {"success": False, "message": f"No scan files found in {session_dir}"}

    # Sort files by frame number or filename
    def sort_key(p: str):
        meta = extract_frame_metadata(p)
        return (meta["frame_num"] if meta["frame_num"] > 0 else 999999, os.path.basename(p).lower())

    scan_files.sort(key=sort_key)

    total_frames = len(scan_files)

    # Determine roll identifiers
    if not session_name:
        session_name = os.path.basename(session_dir)

    if not film_stock or not film_format or not roll_number:
        parts = session_name.split("-")
        if len(parts) >= 3:
            film_stock = film_stock or parts[0]
            film_format = film_format or parts[1]
            roll_number = roll_number or parts[2]
        else:
            film_stock = film_stock or "FilmStock"
            film_format = film_format or "135"
            roll_number = roll_number or "01"

    # Contact sheet dimensions (300 DPI Letter/A4 canvas: 3508 x 2480 or 3600 x 2700 landscape)
    canvas_w = 3600
    canvas_h = 2700

    margin_x = 120
    header_h = 280
    footer_h = 90
    margin_bottom = 60

    grid_w = canvas_w - (margin_x * 2)
    grid_h = canvas_h - header_h - footer_h - margin_bottom

    # Grid columns and rows calculation
    cols = max(2, min(8, columns))
    # Standard 35mm 6x6 = 36 frames per sheet; 120 is often 4x4 or 3x4
    if film_format == "120" and columns == 6:
        cols = 4

    rows = 6 if cols >= 5 else 4
    frames_per_page = cols * rows
    total_pages = math.ceil(total_frames / frames_per_page)

    col_gap = 40
    row_gap = 50

    cell_w = int((grid_w - (col_gap * (cols - 1))) / cols)
    cell_h = int((grid_h - (row_gap * (rows - 1))) / rows)

    # Cell internal allocation: thumbnail + metadata label space
    label_h = 60
    thumb_box_w = cell_w
    thumb_box_h = cell_h - label_h

    # Color Palette
    if theme == "light":
        bg_color = (250, 252, 255)
        card_bg = (240, 243, 248)
        border_color = (210, 218, 230)
        text_primary = (15, 23, 42)
        text_secondary = (71, 85, 105)
        accent_color = (14, 116, 144)
        badge_bg = (224, 231, 255)
        badge_text = (30, 58, 138)
        divider_color = (226, 232, 240)
    else:  # darkroom dark theme
        bg_color = (15, 18, 24)
        card_bg = (22, 27, 36)
        border_color = (40, 48, 64)
        text_primary = (243, 244, 246)
        text_secondary = (156, 163, 175)
        accent_color = (56, 189, 248)
        badge_bg = (30, 41, 59)
        badge_text = (125, 211, 252)
        divider_color = (31, 41, 55)

    # Fonts
    font_title = get_font(size=46, bold=True)
    font_subtitle = get_font(size=22, bold=False)
    font_header_meta = get_font(size=20, bold=False)
    font_header_bold = get_font(size=20, bold=True)
    font_frame_num = get_font(size=18, bold=True)
    font_cell_meta = get_font(size=14, bold=False)
    font_cell_meta_bold = get_font(size=14, bold=True)
    font_footer = get_font(size=16, bold=False)

    pages: List[Image.Image] = []

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    processing_profile = (config.get("color_profile", "Adobe RGB (1998)") if config else "Adobe RGB (1998)").replace("_", " ").title()
    gamma_str = f"γ {config.get('gamma', 2.2):.1f}" if config else "γ 2.2"
    neutralize_str = "Film Base Neutralized" if (config and config.get("neutralize")) else "Standard Balance"

    for page_idx in range(total_pages):
        page_img = Image.new("RGB", (canvas_w, canvas_h), bg_color)
        draw = ImageDraw.Draw(page_img)

        # --- DRAW HEADER ---
        # Top banner background
        draw.rectangle([margin_x, 40, canvas_w - margin_x, header_h - 20], fill=card_bg, outline=border_color, width=2)
        
        # Header accent stripe
        draw.rectangle([margin_x, 40, margin_x + 8, header_h - 20], fill=accent_color)

        # Title: Roll & Film Identification
        title_x = margin_x + 35
        title_y = 65
        draw.text((title_x, title_y), f"FILM ARCHIVAL CONTACT SHEET", font=font_title, fill=text_primary)
        
        sub_text = f"SESSION: {session_name.upper()}  •  STOCK: {film_stock.upper()}  •  FORMAT: {film_format}  •  ROLL: #{roll_number}"
        draw.text((title_x, title_y + 60), sub_text, font=font_subtitle, fill=accent_color)

        # Header Right Metadata Box
        meta_right_x = canvas_w - margin_x - 700
        meta_y = 65
        
        draw.text((meta_right_x, meta_y), "DATE / TIME:", font=font_header_bold, fill=text_secondary)
        draw.text((meta_right_x + 160, meta_y), now_str, font=font_header_meta, fill=text_primary)
        
        draw.text((meta_right_x, meta_y + 32), "TOTAL FRAMES:", font=font_header_bold, fill=text_secondary)
        draw.text((meta_right_x + 160, meta_y + 32), f"{total_frames} Frames", font=font_header_meta, fill=text_primary)
        
        draw.text((meta_right_x, meta_y + 64), "COLOR PROFILE:", font=font_header_bold, fill=text_secondary)
        draw.text((meta_right_x + 160, meta_y + 64), processing_profile, font=font_header_meta, fill=text_primary)

        draw.text((meta_right_x, meta_y + 96), "SENSITOMETRY:", font=font_header_bold, fill=text_secondary)
        draw.text((meta_right_x + 160, meta_y + 96), f"{neutralize_str} ({gamma_str})", font=font_header_meta, fill=text_primary)

        # Header Divider line
        draw.line([(margin_x, header_h - 10), (canvas_w - margin_x, header_h - 10)], fill=divider_color, width=2)

        # --- DRAW THUMBNAIL GRID ---
        start_frame_idx = page_idx * frames_per_page
        end_frame_idx = min(start_frame_idx + frames_per_page, total_frames)

        grid_top = header_h + 10

        for idx in range(start_frame_idx, end_frame_idx):
            cell_rel_idx = idx - start_frame_idx
            c = cell_rel_idx % cols
            r = cell_rel_idx // cols

            cell_x = margin_x + c * (cell_w + col_gap)
            cell_y = grid_top + r * (cell_h + row_gap)

            file_path = scan_files[idx]
            meta = extract_frame_metadata(file_path)

            frame_num = meta["frame_num"] or (idx + 1)
            frame_label = f"FRAME {frame_num:02d}"

            # Cell Card Background & Border
            draw.rectangle(
                [cell_x, cell_y, cell_x + cell_w, cell_y + cell_h],
                fill=card_bg,
                outline=border_color,
                width=1
            )

            # Thumbnail Box
            thumb = load_thumbnail(file_path, thumb_box_w - 12, thumb_box_h - 12)
            if thumb:
                tw, th = thumb.size
                tx = cell_x + int((cell_w - tw) / 2)
                ty = cell_y + int((thumb_box_h - th) / 2) + 6
                # Thumbnail border
                draw.rectangle([tx - 1, ty - 1, tx + tw + 1, ty + th + 1], outline=border_color, width=1)
                page_img.paste(thumb, (tx, ty))
            else:
                # Placeholder if image cannot be rendered
                draw.rectangle(
                    [cell_x + 10, cell_y + 10, cell_x + cell_w - 10, cell_y + thumb_box_h - 10],
                    fill=bg_color,
                    outline=border_color,
                    width=1
                )
                draw.text((cell_x + 30, cell_y + thumb_box_h // 2 - 10), "Preview Unavailable", font=font_cell_meta, fill=text_secondary)

            # Metadata Bar at bottom of cell
            label_top = cell_y + thumb_box_h
            draw.line([(cell_x, label_top), (cell_x + cell_w, label_top)], fill=divider_color, width=1)

            # Frame Number Badge
            badge_w = 95
            badge_h = 24
            badge_x = cell_x + 8
            badge_y = label_top + 8
            draw.rectangle([badge_x, badge_y, badge_x + badge_w, badge_y + badge_h], fill=badge_bg, outline=border_color, width=1)
            draw.text((badge_x + 8, badge_y + 4), frame_label, font=font_frame_num, fill=badge_text)

            # Exposure metadata or file dimensions
            exp_text = ""
            if meta["shutter"] or meta["aperture"] or meta["iso"]:
                parts_exp = []
                if meta["shutter"]: parts_exp.append(meta["shutter"])
                if meta["aperture"]: parts_exp.append(meta["aperture"])
                if meta["iso"]: parts_exp.append(meta["iso"])
                exp_text = " • ".join(parts_exp)
            elif meta["megapixels"] > 0:
                exp_text = f"{meta['width']}×{meta['height']} ({meta['megapixels']} MP)"
            else:
                exp_text = "16-Bit Master"

            draw.text((badge_x + badge_w + 10, badge_y + 5), exp_text, font=font_cell_meta_bold, fill=text_primary)

            # File name line
            fname = meta["filename"]
            if len(fname) > 30:
                fname = fname[:27] + "..."
            draw.text((cell_x + 8, label_top + 36), fname, font=font_cell_meta, fill=text_secondary)

        # --- DRAW FOOTER ---
        footer_y = canvas_h - footer_h
        draw.line([(margin_x, footer_y), (canvas_w - margin_x, footer_y)], fill=divider_color, width=1)

        footer_left = f"FilmConvert RAW Suite • Archival Lab Contact Sheet System • Generated {now_str}"
        draw.text((margin_x, footer_y + 25), footer_left, font=font_footer, fill=text_secondary)

        footer_right = f"Page {page_idx + 1} of {total_pages}"
        draw.text((canvas_w - margin_x - 120, footer_y + 25), footer_right, font=font_footer, fill=accent_color)

        pages.append(page_img)

    # --- SAVE OUTPUT FILES ---
    if output_dir is None:
        output_dir = session_dir
    os.makedirs(output_dir, exist_ok=True)

    if output_basename is None:
        output_basename = f"{session_name}_contact_sheet"

    pdf_path: Optional[str] = None
    jpeg_paths: List[str] = []

    # Export PDF
    if export_pdf and pages:
        pdf_path = os.path.join(output_dir, f"{output_basename}.pdf")
        try:
            pages[0].save(
                pdf_path,
                "PDF",
                save_all=True,
                append_images=pages[1:] if len(pages) > 1 else [],
                resolution=float(dpi)
            )
        except Exception as e:
            print(f"Error saving contact sheet PDF: {e}")
            pdf_path = None

    # Export JPEG(s)
    if export_jpeg and pages:
        if len(pages) == 1:
            jpg_p = os.path.join(output_dir, f"{output_basename}.jpg")
            pages[0].save(jpg_p, "JPEG", quality=92, optimize=True)
            jpeg_paths.append(jpg_p)
        else:
            for p_num, p_img in enumerate(pages, start=1):
                jpg_p = os.path.join(output_dir, f"{output_basename}_p{p_num:02d}.jpg")
                p_img.save(jpg_p, "JPEG", quality=92, optimize=True)
                jpeg_paths.append(jpg_p)

    return {
        "success": True,
        "pdf_path": pdf_path,
        "jpeg_paths": jpeg_paths,
        "frame_count": total_frames,
        "pages_count": len(pages),
        "message": f"Successfully generated contact sheet with {total_frames} frames ({len(pages)} page{'s' if len(pages) > 1 else ''})"
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Archival Contact Sheet / Roll Summary for FilmConvert scans.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Session directory or positives directory")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output directory (defaults to session dir)")
    parser.add_argument("--stock", type=str, default="", help="Film stock name")
    parser.add_argument("--format", type=str, default="", help="Film format (135, 120)")
    parser.add_argument("--roll", type=str, default="", help="Roll number")
    parser.add_argument("--columns", type=int, default=6, help="Grid columns (default: 6)")
    parser.add_argument("--theme", type=str, default="dark", choices=["dark", "light"], help="Theme: dark or light")
    parser.add_argument("--no-pdf", action="store_true", help="Disable PDF generation")
    parser.add_argument("--no-jpeg", action="store_true", help="Disable JPEG generation")

    args = parser.parse_args()
    res = generate_contact_sheet(
        session_dir=args.input,
        output_dir=args.output,
        film_stock=args.stock,
        film_format=args.format,
        roll_number=args.roll,
        columns=args.columns,
        theme=args.theme,
        export_pdf=not args.no_pdf,
        export_jpeg=not args.no_jpeg
    )
    print(res["message"])
    if res.get("pdf_path"):
        print(f"  PDF:  {res['pdf_path']}")
    for jp in res.get("jpeg_paths", []):
        print(f"  JPEG: {jp}")
