import os
import sys
import shutil
import tempfile
import pytest
import numpy as np
from PIL import Image
import tifffile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from contact_sheet import generate_contact_sheet, extract_frame_metadata, get_font
from schemas import SessionConfigSchema, ContactSheetGenerateSchema
from web_ui import app, session


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_session_dir():
    temp_dir = tempfile.mkdtemp(prefix="filmconvert_test_session_")
    positives_dir = os.path.join(temp_dir, "positives")
    os.makedirs(positives_dir, exist_ok=True)

    # Create 6 synthetic positive 16-bit TIFF frames
    for i in range(1, 7):
        img = np.random.randint(5000, 60000, (400, 600, 3), dtype=np.uint16)
        fname = f"Frame_{i:02d}_Positive.tiff"
        fpath = os.path.join(positives_dir, fname)
        tifffile.imwrite(fpath, img)

    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_extract_frame_metadata(sample_session_dir):
    positives_dir = os.path.join(sample_session_dir, "positives")
    f1 = os.path.join(positives_dir, "Frame_01_Positive.tiff")
    meta = extract_frame_metadata(f1)
    
    assert meta["frame_num"] == 1
    assert meta["filename"] == "Frame_01_Positive.tiff"
    assert meta["width"] == 600
    assert meta["height"] == 400
    assert meta["megapixels"] == 0.2


def test_generate_contact_sheet_single_page(sample_session_dir):
    res = generate_contact_sheet(
        session_dir=sample_session_dir,
        session_name="KodakGold200-135-01",
        film_stock="KodakGold200",
        film_format="135",
        roll_number="01",
        columns=6,
        theme="dark",
        export_pdf=True,
        export_jpeg=True
    )

    assert res["success"] is True
    assert res["frame_count"] == 6
    assert res["pages_count"] == 1
    assert res["pdf_path"] is not None
    assert os.path.exists(res["pdf_path"])
    assert len(res["jpeg_paths"]) == 1
    assert os.path.exists(res["jpeg_paths"][0])

    # Verify JPEG file integrity and dimensions
    with Image.open(res["jpeg_paths"][0]) as jpg_img:
        assert jpg_img.size == (3600, 2700)
        assert jpg_img.mode == "RGB"


def test_generate_contact_sheet_multipage():
    temp_dir = tempfile.mkdtemp(prefix="filmconvert_multipage_")
    positives_dir = os.path.join(temp_dir, "positives")
    os.makedirs(positives_dir, exist_ok=True)

    try:
        # Create 40 frames (exceeds 36 frames per sheet for 6x6 grid)
        for i in range(1, 41):
            img = np.random.randint(1000, 50000, (100, 150, 3), dtype=np.uint16)
            tifffile.imwrite(os.path.join(positives_dir, f"Frame_{i:02d}_Positive.tiff"), img)

        res = generate_contact_sheet(
            session_dir=temp_dir,
            session_name="FujiSuperia400-135-02",
            columns=6,
            export_pdf=True,
            export_jpeg=True
        )

        assert res["success"] is True
        assert res["frame_count"] == 40
        assert res["pages_count"] == 2
        assert res["pdf_path"] is not None
        assert os.path.exists(res["pdf_path"])
        # Multipage generates _p01.jpg and _p02.jpg
        assert len(res["jpeg_paths"]) == 2
        for jp in res["jpeg_paths"]:
            assert os.path.exists(jp)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_generate_contact_sheet_light_theme_and_120_format(sample_session_dir):
    res = generate_contact_sheet(
        session_dir=sample_session_dir,
        session_name="Portra400-120-03",
        film_stock="Portra400",
        film_format="120",
        roll_number="03",
        columns=4,
        theme="light",
        export_pdf=True,
        export_jpeg=True
    )

    assert res["success"] is True
    assert res["pages_count"] == 1
    assert os.path.exists(res["pdf_path"])


def test_contact_sheet_flask_api(client, sample_session_dir):
    # Test POST /api/contact_sheet/generate
    payload = {
        "session_dir": sample_session_dir,
        "stock": "KodakGold200",
        "format": "135",
        "roll": "01",
        "columns": 6,
        "theme": "dark",
        "export_pdf": True,
        "export_jpeg": True
    }
    resp = client.post('/api/contact_sheet/generate', json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["frame_count"] == 6
    assert data["pdf_path"] is not None

    # Test GET /api/files discovers contact sheets
    with session.lock:
        session.root_folder = os.path.dirname(sample_session_dir)
        session.session_name = os.path.basename(sample_session_dir)
        session.dirs = {
            "positives": os.path.join(sample_session_dir, "positives"),
            "processed": os.path.join(sample_session_dir, "processed_raws"),
            "negatives": os.path.join(sample_session_dir, "negatives"),
            "errors": os.path.join(sample_session_dir, "error_raws")
        }

    files_resp = client.get('/api/files')
    assert files_resp.status_code == 200
    files_data = files_resp.get_json()
    assert files_data["success"] is True
    assert "contact_sheets" in files_data
    assert len(files_data["contact_sheets"]) >= 2  # PDF and JPEG

    # Test GET /api/contact_sheet/download
    pdf_item = next(cs for cs in files_data["contact_sheets"] if cs["type"] == "pdf")
    dl_resp = client.get(f'/api/contact_sheet/download?path={pdf_item["path"]}&download=1')
    assert dl_resp.status_code == 200
    assert dl_resp.headers["Content-Type"] == "application/pdf"


def test_batch_metadata_single_exiftool_invocation(sample_session_dir, monkeypatch):
    import subprocess
    import json
    import contact_sheet

    calls = []

    def fake_find_exiftool():
        return "/usr/bin/exiftool"

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        files = [arg for arg in cmd[1:] if not arg.startswith("-")]
        payload = [
            {
                "SourceFile": f,
                "ImageWidth": 600,
                "ImageHeight": 400,
                "ExposureTime": "1/125",
                "FNumber": 8.0,
                "ISO": 200,
                "FocalLength": "90.0 mm",
                "Model": "Canon EOS R5",
                "LensModel": "RF 100mm F2.8 L",
                "DateTimeOriginal": "2026:10:16 10:00:00"
            }
            for f in files
        ]

        class DummyResult:
            returncode = 0
            stdout = json.dumps(payload)
            stderr = ""

        return DummyResult()

    monkeypatch.setattr(contact_sheet, "find_exiftool", fake_find_exiftool)
    monkeypatch.setattr(subprocess, "run", fake_run)

    res = generate_contact_sheet(
        session_dir=sample_session_dir,
        session_name="BatchTest-135-01",
        columns=6,
        export_pdf=False,
        export_jpeg=True
    )

    assert res["success"] is True
    assert res["frame_count"] == 6
    # Should invoke exiftool exactly ONCE in batch mode (-j) across all 6 frames, not 2*6=12 times
    assert len(calls) == 1
    assert "-j" in calls[0]

