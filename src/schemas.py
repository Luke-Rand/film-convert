"""
Pydantic schemas and models for FilmConvert API validation and session configuration.
"""

from typing import Optional, List, Literal, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class SessionConfigSchema(BaseModel):
    """Configuration options for image compositing and inversion pipelines."""
    clip: float = Field(default=0.1, ge=0.0, le=10.0, description="Black/white point clipping percentile")
    gamma: float = Field(default=2.2, ge=0.1, le=5.0, description="Target gamma curve")
    scurve: float = Field(default=0.0, ge=-1.0, le=1.0, description="Midtone S-curve contrast adjustment")
    margin: float = Field(default=0.03, ge=0.0, le=0.5, description="Border margin fraction to ignore during histogram analysis")
    autocrop: bool = Field(default=False, description="Whether to automatically crop borders")
    global_levels: bool = Field(default=False, description="Whether to apply joint RGB min/max levels instead of independent channels")
    compress_tiff: bool = Field(default=False, description="Whether to apply Deflate compression to output TIFFs")
    neutralize: bool = Field(default=False, description="Whether to apply film base neutralization during compositing")
    base_ratios: Optional[List[float]] = Field(default=None, description="Film base transmission ratios [R, G, B] normalized to max 1.0")
    align_channels: bool = Field(default=False, description="Whether to perform subpixel alignment between color channels")
    monochrome: bool = Field(default=False, description="Whether to output single-channel monochrome grayscale")
    monochrome_channel: str = Field(default="luminance", description="Channel extraction method: 'red', 'green', 'blue', 'luminance', 'average'")
    reversal: bool = Field(default=False, description="Color reversal (slide film) positive processing mode")
    convert_to_tiff: bool = Field(default=True, description="Save converted output as 16-bit TIFF alongside DNG")
    color_profile: str = Field(default="adobe_rgb", description="ICC output color profile ('adobe_rgb', 'srgb', 'prophoto_rgb', 'romm_rgb', 'generic_gray')")
    embed_metadata: bool = Field(default=True, description="Preserve EXIF and capture metadata in output files")
    auto_contact_sheet: bool = Field(default=True, description="Whether to automatically generate archival contact sheet on session completion")
    contact_sheet_columns: int = Field(default=6, ge=2, le=8, description="Number of columns in contact sheet grid")
    contact_sheet_theme: str = Field(default="dark", description="Contact sheet theme: 'dark' or 'light'")

    @field_validator("base_ratios")
    @classmethod
    def validate_base_ratios(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return None
        if len(v) != 3:
            raise ValueError("base_ratios must contain exactly 3 float values [R, G, B]")
        for ratio in v:
            if not isinstance(ratio, (int, float)) or ratio <= 0.0 or ratio > 2.0:
                raise ValueError("base_ratios values must be positive floats (typically <= 1.0)")
        return [float(r) for r in v]

    model_config = {
        "extra": "ignore"
    }


class SessionConfigUpdateSchema(BaseModel):
    """Partial configuration schema for updates."""
    clip: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    gamma: Optional[float] = Field(default=None, ge=0.1, le=5.0)
    scurve: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    margin: Optional[float] = Field(default=None, ge=0.0, le=0.5)
    autocrop: Optional[bool] = None
    global_levels: Optional[bool] = None
    compress_tiff: Optional[bool] = None
    neutralize: Optional[bool] = None
    base_ratios: Optional[List[float]] = None
    align_channels: Optional[bool] = None
    monochrome: Optional[bool] = None
    monochrome_channel: Optional[str] = None
    reversal: Optional[bool] = None
    convert_to_tiff: Optional[bool] = None
    color_profile: Optional[str] = None
    embed_metadata: Optional[bool] = None
    auto_contact_sheet: Optional[bool] = None
    contact_sheet_columns: Optional[int] = Field(default=None, ge=2, le=8)
    contact_sheet_theme: Optional[str] = None

    @field_validator("base_ratios")
    @classmethod
    def validate_base_ratios(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return None
        if len(v) != 3:
            raise ValueError("base_ratios must contain exactly 3 float values [R, G, B]")
        for ratio in v:
            if not isinstance(ratio, (int, float)) or ratio <= 0.0 or ratio > 2.0:
                raise ValueError("base_ratios values must be positive floats (typically <= 1.0)")
        return [float(r) for r in v]

    model_config = {
        "extra": "ignore"
    }


class StartSessionSchema(BaseModel):
    """Payload schema for starting a scanning monitoring session."""
    root_dir: Optional[str] = Field(default="~/Pictures/Scans", description="Root scan folder")
    session_name: Optional[str] = Field(default="", description="Session name / subdirectory")
    mode: Literal["triplet", "single"] = Field(default="triplet", description="Scanning mode")
    stock: Optional[str] = Field(default="FilmStock", description="Film stock name for auto-naming")
    format: Optional[str] = Field(default="135", description="Film format (135, 120, etc.)")
    roll: Optional[str] = Field(default="01", description="Roll index string")
    config: Optional[Union[SessionConfigUpdateSchema, SessionConfigSchema, dict]] = Field(default=None, description="Initial session configuration")

    model_config = {
        "extra": "ignore"
    }


class BatchJobSchema(BaseModel):
    """Payload schema for running a batch processing job."""
    task_type: Literal["composite", "invert"] = Field(..., description="Batch job type: composite or invert")
    input_path: str = Field(..., min_length=1, description="Path to folder or file to process")
    config: Optional[Union[SessionConfigUpdateSchema, SessionConfigSchema, dict]] = Field(default=None, description="Overrides for batch job config")

    model_config = {
        "extra": "ignore"
    }


class ContactSheetGenerateSchema(BaseModel):
    """Payload schema for generating contact sheets on demand."""
    session_dir: Optional[str] = Field(default=None, description="Target session folder path")
    stock: Optional[str] = Field(default="", description="Film stock name")
    format: Optional[str] = Field(default="", description="Film format")
    roll: Optional[str] = Field(default="", description="Roll number")
    session_name: Optional[str] = Field(default="", description="Session name")
    columns: int = Field(default=6, ge=2, le=8, description="Number of grid columns")
    theme: Literal["dark", "light"] = Field(default="dark", description="Visual theme: dark or light")
    export_pdf: bool = Field(default=True, description="Export archival PDF")
    export_jpeg: bool = Field(default=True, description="Export high-resolution JPEG")

    model_config = {
        "extra": "ignore"
    }


class SampleRebateSchema(BaseModel):
    """Payload schema for film base rebate sampling."""
    path: Optional[str] = Field(default=None, description="Path to source image")
    x_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Normalized X coordinate (0.0 - 1.0)")
    y_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Normalized Y coordinate (0.0 - 1.0)")
    rgb: Optional[List[Union[int, float]]] = Field(default=None, description="Direct RGB values [R, G, B] (0-255)")

    @model_validator(mode="after")
    def validate_source(self):
        has_coords = self.path is not None and self.x_ratio is not None and self.y_ratio is not None
        has_rgb = self.rgb is not None and len(self.rgb) >= 3
        if not has_coords and not has_rgb:
            raise ValueError("Must provide either (path, x_ratio, y_ratio) or rgb array [r, g, b]")
        return self

    model_config = {
        "extra": "ignore"
    }


class CameraConfigSchema(BaseModel):
    """Payload schema for setting a camera gphoto2 property."""
    name: str = Field(..., min_length=1, description="Configuration property name")
    value: Any = Field(..., description="Configuration property value")


class CameraFocusStepSchema(BaseModel):
    """Payload schema for camera manual focus stepping."""
    direction: Literal["near", "far", "Near", "Far"] = Field(default="near", description="Focus step direction")
    speed: Union[str, int] = Field(default="1", description="Focus step size (1, 2, 3)")


class CameraToggleLiveviewSchema(BaseModel):
    """Payload schema for toggling camera live view."""
    active: bool = Field(default=False, description="Live view active flag")


class CameraMockLedsSchema(BaseModel):
    """Payload schema for updating mock LED brightnesses."""
    red: int = Field(default=255, ge=0, le=255, description="Red LED brightness (0-255)")
    green: int = Field(default=255, ge=0, le=255, description="Green LED brightness (0-255)")
    blue: int = Field(default=255, ge=0, le=255, description="Blue LED brightness (0-255)")
