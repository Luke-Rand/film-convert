import os
import io
import time
from typing import Dict, Any, Optional
from PIL import Image, ImageDraw
import numpy as np

from .base import BaseCameraBackend

class SimulatedCameraBackend(BaseCameraBackend):
    """Simulated camera backend that generates synthetic frames and mock captures."""

    def __init__(self, log_callback=None, session_manager=None):
        super().__init__(log_callback=log_callback)
        self.session_manager = session_manager
        self.camera_connected = True
        self.connection_state = "simulated"
        self.model_name = "Simulated Camera"
        self.camera_settings = {
            "iso": "400",
            "aperture": "f/8.0",
            "shutterspeed": "1/60",
        }
        self.camera_choices = {
            "iso": ["100", "200", "400", "800", "1600", "3200", "6400"],
            "aperture": ["f/2.8", "f/4.0", "f/5.6", "f/8.0", "f/11", "f/16", "f/22"],
            "shutterspeed": ["1/250", "1/125", "1/60", "1/30", "1/15", "1/8", "f/1/4", "1/4", "1/2", "1s", "2s"]
        }
        # Normalize shutter choices
        self.camera_choices["shutterspeed"] = ["1/250", "1/125", "1/60", "1/30", "1/15", "1/8", "1/4", "1/2", "1s", "2s"]
        self.mock_leds = {"red": 255, "green": 255, "blue": 255}
        self.is_monochrome = False

    def connect(self) -> bool:
        self.camera_connected = True
        self.connection_state = "simulated"
        return True

    def disconnect(self):
        self.camera_connected = True
        self.connection_state = "simulated"

    def update_mock_leds(self, r: int, g: int, b: int):
        self.mock_leds = {"red": r, "green": g, "blue": b}

    def set_monochrome(self, mono: bool):
        self.is_monochrome = mono

    def grab_preview_frame(self) -> Optional[bytes]:
        is_mono = self.is_monochrome
        if self.session_manager and hasattr(self.session_manager, "config"):
            is_mono = self.session_manager.config.get("monochrome", is_mono)
        return self._generate_simulated_frame(
            iso=self.camera_settings.get("iso", "400"),
            aperture=self.camera_settings.get("aperture", "f/8.0"),
            shutter=self.camera_settings.get("shutterspeed", "1/60"),
            r_led=self.mock_leds.get("red", 255),
            g_led=self.mock_leds.get("green", 255),
            b_led=self.mock_leds.get("blue", 255),
            is_monochrome=is_mono
        )

    def capture_image(self, autofocus: bool = True) -> Optional[str]:
        self.log("Simulating capture...")
        time.sleep(0.8)  # simulate shutter release sound/lag
        return self._simulate_raw_capture()

    def _simulate_raw_capture(self) -> str:
        target_dir = None
        if self.session_manager and getattr(self.session_manager, 'dirs', None):
            target_dir = self.session_manager.dirs.get("negatives")

        if not target_dir or not os.path.exists(target_dir):
            target_dir = os.path.abspath("./negatives_download")
            os.makedirs(target_dir, exist_ok=True)

        frame_num, shot_num, is_triplet = self._get_next_capture_naming(target_dir)

        color_suffix = "white"
        r, g, b = self.mock_leds["red"], self.mock_leds["green"], self.mock_leds["blue"]
        if r > g and r > b:
            color_suffix = "red"
        elif g > r and g > b:
            color_suffix = "green"
        elif b > r and b > g:
            color_suffix = "blue"

        if is_triplet:
            local_name = f"Frame_{frame_num:02d}_Capture_{shot_num:02d}_{color_suffix}.cr3"
        else:
            local_name = f"Frame_{frame_num:02d}_Capture_{color_suffix}.cr3"
        local_path = os.path.join(target_dir, local_name)

        with open(local_path, "w") as f:
            f.write("MOCK RAW CAPTURE DATA\n")
            f.write(f"Frame: {frame_num}\n")
            f.write(f"Shot: {shot_num}\n")
            f.write(f"Color: {color_suffix}\n")
            f.write(f"ISO: {self.camera_settings['iso']}\n")
            f.write(f"Aperture: {self.camera_settings['aperture']}\n")
            f.write(f"Shutter: {self.camera_settings['shutterspeed']}\n")

        self.log(f"Simulated capture saved: {local_name}")
        return local_path

    def step_focus(self, direction: str, coarse: bool = False) -> bool:
        self.log(f"Simulated focus stepped: {direction} (coarse={coarse})")
        return True

    def trigger_autofocus(self) -> bool:
        self.log("Simulating autofocus...")
        time.sleep(0.8)
        return True

    def get_settings(self) -> Dict[str, str]:
        return self.camera_settings

    def set_setting(self, name: str, value: str) -> bool:
        norm_name = name.lower()
        if norm_name in self.camera_settings:
            self.camera_settings[norm_name] = value
            self.log(f"Simulated setting updated: {norm_name} = {value}")
            return True
        elif norm_name in ["manualfocusdrive", "eosremoterelease"]:
            self.log(f"Simulated action setting updated: {norm_name} = {value}")
            return True
        raise ValueError(f"Unknown simulated setting: {name}")

    def get_choices(self) -> Dict[str, list]:
        return self.camera_choices

    def get_status(self) -> Dict[str, Any]:
        return {
            "connected": True,
            "simulated": True,
            "state": "simulated",
            "model": "Simulated Camera",
            "settings": self.camera_settings,
            "choices": self.camera_choices
        }

    def _generate_simulated_frame(self, iso, aperture, shutter, r_led, g_led, b_led, is_monochrome):
        width, height = 640, 480
        img = Image.new('RGB', (width, height), color=(20, 20, 25))
        draw = ImageDraw.Draw(img)

        # Base film color
        if is_monochrome:
            base_color = (95, 95, 95)
        else:
            base_color = (215, 115, 60) # classic Kodachrome orange negative base

        # Inner film border
        draw.rectangle([70, 50, width-70, height-50], fill=base_color)

        # Center Target Graphics
        draw.ellipse([width//2 - 90, height//2 - 90, width//2 + 90, height//2 + 90], outline=(255, 255, 255), width=2)
        draw.ellipse([width//2 - 30, height//2 - 30, width//2 + 30, height//2 + 30], outline=(240, 240, 240), width=1)
        draw.line([width//2 - 130, height//2, width//2 + 130, height//2], fill=(230, 230, 230), width=1)
        draw.line([width//2, height//2 - 130, width//2, height//2 + 130], fill=(230, 230, 230), width=1)

        arr = np.array(img, dtype=np.float32)

        try:
            iso_val = float(iso)
        except ValueError:
            iso_val = 400.0

        try:
            ap_val = float(aperture.replace('f/', ''))
        except ValueError:
            ap_val = 8.0

        try:
            if '/' in shutter:
                num, denom = shutter.split('/')
                shutter_val = float(num) / float(denom)
            else:
                shutter_val = float(shutter.replace('s', ''))
        except ValueError:
            shutter_val = 0.0166

        exposure = iso_val * shutter_val * (1.0 / (ap_val ** 2))
        brightness = exposure / 0.103

        r_mult = (r_led / 255.0) * brightness
        g_mult = (g_led / 255.0) * brightness
        b_mult = (b_led / 255.0) * brightness

        arr[:, :, 0] *= r_mult
        arr[:, :, 1] *= g_mult
        arr[:, :, 2] *= b_mult

        if iso_val > 100:
            noise_sigma = (iso_val / 6400.0) * 40.0
            noise = np.random.normal(0, noise_sigma, arr.shape)
            arr += noise

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        processed_img = Image.fromarray(arr)

        draw_ovr = ImageDraw.Draw(processed_img)
        draw_ovr.rectangle([0, height-25, width, height], fill=(15, 23, 42))
        text_str = f"SIMULATED CAMERA  |  ISO {iso}  |  {aperture}  |  {shutter}s  |  LEDs [R:{r_led} G:{g_led} B:{b_led}]"
        draw_ovr.text((15, height-20), text_str, fill=(148, 163, 184))
        draw_ovr.rectangle([10, 10, width-10, height-35], outline=(30, 41, 59), width=1)

        out_buf = io.BytesIO()
        processed_img.save(out_buf, format='JPEG', quality=85)
        return out_buf.getvalue()
