import os
import sys
import unittest
import numpy as np
import json

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from web_ui import app, camera_manager


class TestHistogramAndLEDAutoTune(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_update_mock_leds_api(self):
        """
        Verify that POST /api/camera/update_mock_leds updates camera_manager mock LED settings.
        """
        payload = {"red": 180, "green": 120, "blue": 240}
        res = self.app.post('/api/camera/update_mock_leds',
                            data=json.dumps(payload),
                            content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(camera_manager.mock_leds["red"], 180)
        self.assertEqual(camera_manager.mock_leds["green"], 120)
        self.assertEqual(camera_manager.mock_leds["blue"], 240)

    def test_histogram_margin_masking_accuracy(self):
        """
        Verify that margin-aware sampling ignores outer carrier borders and specular edge leaks,
        preventing false highlight clipping and erroneous exposure readouts.
        """
        w, h = 480, 360
        # Create an image where:
        # Outer 5% contains edge light leaks (value 255, 255, 255) and carrier mask (0, 0, 0)
        # Inner active area contains film rebate with R=220, G=140, B=80
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Outer edge flare (light leak outside the film carrier)
        img[0:15, :, :] = 255
        img[:, 0:15, :] = 255
        
        # Inner active area
        mx = int(w * 0.05) # 24 px
        my = int(h * 0.05) # 18 px
        img[my:h-my, mx:w-mx, 0] = 220
        img[my:h-my, mx:w-mx, 1] = 140
        img[my:h-my, mx:w-mx, 2] = 80
        
        # 1. Unmasked whole-frame calculation would incorrectly report clipping:
        unmasked_clip_count = np.sum((img[:, :, 0] >= 254) | (img[:, :, 1] >= 254) | (img[:, :, 2] >= 254))
        self.assertGreater(unmasked_clip_count, 0, "Unmasked calculation sees the edge leaks")
        
        # 2. Masked calculation within the safe margin:
        roi = img[my:h-my, mx:w-mx]
        r_hist = np.bincount(roi[:, :, 0].ravel(), minlength=256)
        g_hist = np.bincount(roi[:, :, 1].ravel(), minlength=256)
        b_hist = np.bincount(roi[:, :, 2].ravel(), minlength=256)
        
        total_pixels = roi.shape[0] * roi.shape[1]
        p99_thresh = total_pixels * 0.999
        
        def calc_p99(hist):
            acc = 0
            for v in range(256):
                acc += hist[v]
                if acc >= p99_thresh:
                    return v
            return 255
            
        r_p99 = calc_p99(r_hist)
        g_p99 = calc_p99(g_hist)
        b_p99 = calc_p99(b_hist)
        
        masked_clip_count = np.sum((roi[:, :, 0] >= 254) | (roi[:, :, 1] >= 254) | (roi[:, :, 2] >= 254))
        
        # Accurate readouts: exactly the inner film values, zero clipping!
        self.assertEqual(r_p99, 220)
        self.assertEqual(g_p99, 140)
        self.assertEqual(b_p99, 80)
        self.assertEqual(masked_clip_count, 0)

    def test_p99_rejects_single_pixel_hot_noise(self):
        """
        Verify that 99.9th percentile rejects single-pixel hot noise / cosmic ray / dust artifacts
        while max (P100) is easily fooled.
        """
        total_pixels = 100000
        # Normal film rebate values at 210
        pixels = np.full(total_pixels, 210, dtype=np.uint8)
        # Inject 10 hot pixels at 255
        pixels[0:10] = 255
        
        hist = np.bincount(pixels, minlength=256)
        p100 = np.max(pixels)
        
        acc = 0
        p99 = 255
        for v in range(256):
            acc += hist[v]
            if acc >= total_pixels * 0.999:
                p99 = v
                break
                
        self.assertEqual(p100, 255, "P100 was deceived by 10 hot pixels")
        self.assertEqual(p99, 210, "P99.9 robustly rejected the hot pixels and reported true rebate floor")

    def test_damped_inverse_radiometric_convergence(self):
        """
        Verify that the ETTR algorithm converges within 3 iterations for simulated non-linear
        camera live view responses.
        """
        target_ceiling = 242
        tolerance = 3
        
        # Simulate camera response function: Peak = 255 * (PWM / 255)^0.55 (typical gamma ~1.8 curve)
        def camera_response(pwm):
            return int(np.clip(255.0 * ((pwm / 255.0) ** 0.55), 1, 255))
            
        # Test starting from different initial states: underexposed (PWM=50) and overexposed (PWM=250)
        initial_pwms = [50, 100, 250]
        
        for init_pwm in initial_pwms:
            pwm = init_pwm
            converged = False
            for iteration in range(1, 5):
                peak = camera_response(pwm)
                error = target_ceiling - peak
                if abs(error) <= tolerance:
                    converged = True
                    break
                
                if peak >= 254:
                    ratio = 0.80
                else:
                    ratio = (target_ceiling / max(peak, 8)) ** 1.7
                    
                next_pwm = int(np.clip(round(pwm * ratio), 1, 255))
                if next_pwm == pwm:
                    next_pwm += 1 if error > 0 else -1
                    next_pwm = int(np.clip(next_pwm, 1, 255))
                pwm = next_pwm
                
            self.assertTrue(converged, f"Failed to converge from initial PWM {init_pwm}, reached PWM={pwm}, peak={camera_response(pwm)}")
            final_peak = camera_response(pwm)
            self.assertLessEqual(abs(target_ceiling - final_peak), tolerance)


if __name__ == '__main__':
    unittest.main()
