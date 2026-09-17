import pytest
from pydantic import ValidationError
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from schemas import (
    SessionConfigSchema,
    SessionConfigUpdateSchema,
    StartSessionSchema,
    BatchJobSchema,
    ContactSheetGenerateSchema,
    SampleRebateSchema,
    CameraConfigSchema,
    CameraFocusStepSchema,
    CameraToggleLiveviewSchema,
    CameraMockLedsSchema
)
from web_ui import app, session

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_session_config_schema_defaults():
    cfg = SessionConfigSchema()
    assert cfg.clip == 0.1
    assert cfg.gamma == 2.2
    assert cfg.scurve == 0.0
    assert cfg.margin == 0.03
    assert cfg.autocrop is False
    assert cfg.convert_to_tiff is True
    assert cfg.color_profile == "adobe_rgb"
    assert cfg.embed_metadata is True
    assert cfg.base_ratios is None
    assert cfg.auto_contact_sheet is True
    assert cfg.contact_sheet_columns == 6
    assert cfg.contact_sheet_theme == "dark"

def test_contact_sheet_generate_schema():
    # Defaults
    cs_cfg = ContactSheetGenerateSchema()
    assert cs_cfg.columns == 6
    assert cs_cfg.theme == "dark"
    assert cs_cfg.export_pdf is True
    assert cs_cfg.export_jpeg is True

    # Custom
    cs_custom = ContactSheetGenerateSchema(columns=4, theme="light", stock="Portra400")
    assert cs_custom.columns == 4
    assert cs_custom.theme == "light"
    assert cs_custom.stock == "Portra400"

    # Invalid columns (< 2 or > 8)
    with pytest.raises(ValidationError):
        ContactSheetGenerateSchema(columns=1)
    with pytest.raises(ValidationError):
        ContactSheetGenerateSchema(columns=10)

def test_session_config_schema_validation():
    # Valid base ratios
    cfg = SessionConfigSchema(base_ratios=[0.5, 0.8, 1.0])
    assert cfg.base_ratios == [0.5, 0.8, 1.0]

    # Invalid base ratios (wrong length)
    with pytest.raises(ValidationError):
        SessionConfigSchema(base_ratios=[0.5, 0.8])

    # Invalid base ratios (negative)
    with pytest.raises(ValidationError):
        SessionConfigSchema(base_ratios=[-0.1, 0.5, 1.0])

    # Invalid clip (> 10)
    with pytest.raises(ValidationError):
        SessionConfigSchema(clip=15.0)

    # Invalid gamma (< 0.1)
    with pytest.raises(ValidationError):
        SessionConfigSchema(gamma=0.0)

def test_start_session_schema():
    # Default values
    req = StartSessionSchema()
    assert req.mode == "triplet"
    assert req.stock == "FilmStock"
    assert req.format == "135"

    # Custom valid values
    req2 = StartSessionSchema(mode="single", stock="Portra400", format="120", roll="03")
    assert req2.mode == "single"
    assert req2.stock == "Portra400"

    # Invalid mode
    with pytest.raises(ValidationError):
        StartSessionSchema(mode="invalid_mode")

def test_batch_job_schema():
    # Valid composite
    job = BatchJobSchema(task_type="composite", input_path="/path/to/raws")
    assert job.task_type == "composite"
    assert job.input_path == "/path/to/raws"

    # Missing task_type or input_path
    with pytest.raises(ValidationError):
        BatchJobSchema(task_type="invalid", input_path="/path")
    with pytest.raises(ValidationError):
        BatchJobSchema(task_type="composite", input_path="")

def test_sample_rebate_schema():
    # Valid coords
    r1 = SampleRebateSchema(path="/path/to/img.dng", x_ratio=0.5, y_ratio=0.5)
    assert r1.x_ratio == 0.5

    # Valid RGB
    r2 = SampleRebateSchema(rgb=[180, 120, 80])
    assert r2.rgb == [180, 120, 80]

    # Missing both coords and RGB
    with pytest.raises(ValidationError):
        SampleRebateSchema()

    # Out of range coords
    with pytest.raises(ValidationError):
        SampleRebateSchema(path="/path", x_ratio=1.5, y_ratio=0.5)

def test_camera_schemas():
    # Camera config
    c = CameraConfigSchema(name="iso", value="400")
    assert c.name == "iso"
    assert c.value == "400"

    # Focus step
    f = CameraFocusStepSchema(direction="near", speed="2")
    assert f.direction == "near"
    assert f.speed == "2"

    with pytest.raises(ValidationError):
        CameraFocusStepSchema(direction="sideways")

    # Mock LEDs
    leds = CameraMockLedsSchema(red=255, green=128, blue=0)
    assert leds.red == 255
    assert leds.green == 128

    with pytest.raises(ValidationError):
        CameraMockLedsSchema(red=300)

def test_flask_endpoints_schema_validation(client):
    # Test invalid /api/config payload
    resp = client.post('/api/config', json={"clip": 50.0})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "clip" in data["message"]

    # Test valid /api/config payload
    resp = client.post('/api/config', json={"clip": 0.2, "gamma": 2.4})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["config"]["clip"] == 0.2
    assert data["config"]["gamma"] == 2.4

    # Test invalid /api/start payload
    resp = client.post('/api/start', json={"mode": "invalid_mode"})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False

    # Test invalid /api/batch payload
    resp = client.post('/api/batch', json={"task_type": "magic"})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False

    # Test invalid /api/camera/config payload
    resp = client.post('/api/camera/config', json={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False

    # Test invalid /api/camera/focus_step payload
    resp = client.post('/api/camera/focus_step', json={"direction": "up"})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False

    # Test invalid /api/camera/update_mock_leds payload
    resp = client.post('/api/camera/update_mock_leds', json={"red": 500})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
