// ============================================================
// CAMERA LIVE VIEW, FOCUS ASSIST, OVERLAYS & EYEDROPPER MODULE
// ============================================================

import {
    initHistogramCanvas,
    computeAndRenderHistogram,
    clearHistogramCanvas,
    syncMiniScanlightUI,
} from './histogram.js';

export let isLiveviewActive = false;
export let isLiveviewFullscreen = false;
export let marginOverlayVisible = false;
export let activeFilmBaseRatios = null;
export let activeFilmBaseRGB = null;
export let isEyedropperActive = false;
export let eyedropperSource = null;

window.isLiveviewActive = false;
window.isLiveviewFullscreen = false;
window.isEyedropperActive = false;
window.eyedropperSource = null;
window.activeFilmBaseRatios = null;
window.activeFilmBaseRGB = null;

let liveviewCanvas = null;
let offscreenCanvas = null;
let offscreenCtx = null;
let liveviewTimeout = null;
let cameraStatusInterval = null;
let isPollingFrame = false;

// Focus Assist States
let isZoomed = false;
let zoomFactor = 1; // 1, 3, 5
let zoomX = 0.5;
let zoomY = 0.5;
let focusPeakingActive = false;
let peakingThreshold = 30;
let liveviewRotated180 = localStorage.getItem('liveviewRotated180') === 'true';

export function setActiveFilmBaseState(ratios, rgb) {
    activeFilmBaseRatios = ratios;
    activeFilmBaseRGB = rgb;
    window.activeFilmBaseRatios = ratios;
    window.activeFilmBaseRGB = rgb;
}

export function stopCameraStatusPolling() {
    if (cameraStatusInterval) {
        clearInterval(cameraStatusInterval);
        cameraStatusInterval = null;
    }
}

// --- Margin Overlay ---
export function toggleMarginOverlay() {
    marginOverlayVisible = !marginOverlayVisible;
    const overlay = document.getElementById('margin-overlay');
    const btn = document.getElementById('btn-margin-overlay');
    if (!overlay) return;
    if (marginOverlayVisible) {
        updateMarginOverlay();
        overlay.style.display = 'block';
        if (btn) btn.classList.add('overlay-active');
    } else {
        overlay.style.display = 'none';
        if (btn) btn.classList.remove('overlay-active');
    }
}

export function getContainedImageBounds(img, tempImg = null) {
    if (!img) return null;
    const container = img.parentElement;
    if (!container) return null;

    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;

    const sourceWidth = tempImg ? (tempImg.naturalWidth || tempImg.width) : (img.tagName === 'CANVAS' ? img.width : img.naturalWidth);
    const sourceHeight = tempImg ? (tempImg.naturalHeight || tempImg.height) : (img.tagName === 'CANVAS' ? img.height : img.naturalHeight);

    if (img.style.display === 'none' || !sourceWidth || !sourceHeight) {
        return {
            left: 0,
            top: 0,
            width: containerWidth,
            height: containerHeight
        };
    }

    const imageRatio = sourceWidth / sourceHeight;
    const containerRatio = containerWidth / containerHeight;

    let w, h, l, t;
    if (imageRatio > containerRatio) {
        w = containerWidth;
        h = containerWidth / imageRatio;
        l = 0;
        t = (containerHeight - h) / 2;
    } else {
        h = containerHeight;
        w = containerHeight * imageRatio;
        l = (containerWidth - w) / 2;
        t = 0;
    }

    return { left: l, top: t, width: w, height: h };
}

export function updateMarginOverlay(tempImg = null) {
    const overlay = document.getElementById('margin-overlay');
    if (!overlay) return;

    if (!marginOverlayVisible || isZoomed) {
        overlay.style.display = 'none';
        return;
    }

    overlay.style.display = 'block';

    const img = document.getElementById('camera-liveview-canvas');
    const bounds = getContainedImageBounds(img, tempImg);

    if (bounds) {
        overlay.style.left = bounds.left + 'px';
        overlay.style.top = bounds.top + 'px';
        overlay.style.width = bounds.width + 'px';
        overlay.style.height = bounds.height + 'px';
    }

    const marginSlider = document.getElementById('config-margin');
    const marginFraction = marginSlider ? parseFloat(marginSlider.value) : 0.03;
    const pct = (marginFraction * 100).toFixed(1) + '%';
    overlay.style.setProperty('--margin-px', pct);
    overlay.style.boxShadow = `inset 0 0 0 ${pct} rgba(0,0,0,0.55)`;
}

window.addEventListener('resize', () => {
    if (marginOverlayVisible) {
        updateMarginOverlay();
    }
});

// --- Interactive Film Base Neutralization Picker ---
export function toggleFilmBaseEyedropper(source) {
    if (isEyedropperActive) {
        cancelFilmBaseEyedropper();
        return;
    }

    if (source === 'auto') {
        const lightboxEl = document.getElementById('image-lightbox');
        if (lightboxEl && lightboxEl.classList.contains('active')) {
            source = 'lightbox';
        } else {
            source = 'liveview';
        }
    }

    if (source === 'liveview') {
        if (!isLiveviewActive) {
            const toggle = document.getElementById('camera-liveview-toggle');
            if (toggle) {
                toggle.checked = true;
                toggleCameraLiveview(true);
            }
        }
    }

    isEyedropperActive = true;
    eyedropperSource = source;
    window.isEyedropperActive = true;
    window.eyedropperSource = source;

    ['btn-pick-film-base', 'btn-liveview-eyedropper', 'btn-lightbox-eyedropper'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.classList.add('btn-eyedropper-active');
    });

    if (source === 'liveview') {
        const banner = document.getElementById('liveview-eyedropper-banner');
        if (banner) banner.style.display = 'flex';
    } else if (source === 'lightbox') {
        const banner = document.getElementById('lightbox-eyedropper-banner');
        if (banner) banner.style.display = 'flex';
    }

    const lvCanvas = document.getElementById('camera-liveview-canvas');
    if (lvCanvas) lvCanvas.classList.add('canvas-eyedropper-mode');
    const lbImg = document.getElementById('lightbox-img');
    if (lbImg) lbImg.classList.add('img-eyedropper-mode');

    const loupe = document.getElementById('eyedropper-loupe');
    if (loupe) loupe.style.display = 'flex';

    window.appendLogLine?.("[Client] Film base eyedropper active. Click unexposed film rebate (orange mask) on preview to sample.");
}

export function cancelFilmBaseEyedropper() {
    isEyedropperActive = false;
    eyedropperSource = null;
    window.isEyedropperActive = false;
    window.eyedropperSource = null;

    ['btn-pick-film-base', 'btn-liveview-eyedropper', 'btn-lightbox-eyedropper'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.classList.remove('btn-eyedropper-active');
    });

    const lvBanner = document.getElementById('liveview-eyedropper-banner');
    if (lvBanner) lvBanner.style.display = 'none';
    const lbBanner = document.getElementById('lightbox-eyedropper-banner');
    if (lbBanner) lbBanner.style.display = 'none';

    const lvCanvas = document.getElementById('camera-liveview-canvas');
    if (lvCanvas) lvCanvas.classList.remove('canvas-eyedropper-mode');
    const lbImg = document.getElementById('lightbox-img');
    if (lbImg) lbImg.classList.remove('img-eyedropper-mode');

    const loupe = document.getElementById('eyedropper-loupe');
    if (loupe) loupe.style.display = 'none';
}

export function updateFilmBaseBadgeUI(ratios, rgb) {
    const badges = [
        { badgeId: 'film-base-badge', swatchId: 'film-base-swatch', textId: 'film-base-ratios-text' },
        { badgeId: 'batch-film-base-badge', swatchId: 'batch-film-base-swatch', textId: 'batch-film-base-ratios-text' }
    ];

    badges.forEach(({ badgeId, swatchId, textId }) => {
        const badge = document.getElementById(badgeId);
        const swatch = document.getElementById(swatchId);
        const text = document.getElementById(textId);

        if (!badge || !swatch || !text) return;

        if (ratios && Array.isArray(ratios) && ratios.length === 3) {
            badge.style.display = 'inline-flex';
            text.textContent = `R: ${ratios[0].toFixed(3)} | G: ${ratios[1].toFixed(3)} | B: ${ratios[2].toFixed(3)}`;
            if (rgb && Array.isArray(rgb) && rgb.length === 3) {
                swatch.style.backgroundColor = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
            } else {
                const rS = Math.round(245 * Math.pow(ratios[0], 1 / 2.2));
                const gS = Math.round(245 * Math.pow(ratios[1], 1 / 2.2));
                const bS = Math.round(245 * Math.pow(ratios[2], 1 / 2.2));
                swatch.style.backgroundColor = `rgb(${rS}, ${gS}, ${bS})`;
            }
        } else {
            badge.style.display = 'none';
        }
    });
}

export function onNeutralizeCheckboxChanged(checked) {
    const cfgNeutralize = document.getElementById('config-neutralize');
    if (cfgNeutralize) cfgNeutralize.checked = checked;
    const bchNeutralize = document.getElementById('batch-neutralize');
    if (bchNeutralize) bchNeutralize.checked = checked;

    fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            neutralize: checked,
            base_ratios: checked ? activeFilmBaseRatios : null
        })
    }).catch(err => console.error("Failed to sync neutralize state:", err));
}

export function resetFilmBaseToAuto() {
    setActiveFilmBaseState(null, null);
    updateFilmBaseBadgeUI(null, null);

    fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            base_ratios: null
        })
    })
    .then(res => res.json())
    .then(() => {
        window.appendLogLine?.("[Client] Film base neutralization reset to automatic 99.9th percentile mode.");
    })
    .catch(err => console.error("Failed to reset film base:", err));
}

export function sampleCanvasPatch(canvas, x, y, radius = 2) {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const w = canvas.width;
    const h = canvas.height;
    const x0 = Math.max(0, Math.min(w - 1, x - radius));
    const y0 = Math.max(0, Math.min(h - 1, y - radius));
    const sw = Math.min(radius * 2 + 1, w - x0);
    const sh = Math.min(radius * 2 + 1, h - y0);

    if (sw <= 0 || sh <= 0) return [255, 255, 255];

    try {
        const imgData = ctx.getImageData(x0, y0, sw, sh);
        const d = imgData.data;
        let rSum = 0, gSum = 0, bSum = 0, count = 0;
        for (let i = 0; i < d.length; i += 4) {
            rSum += d[i];
            gSum += d[i + 1];
            bSum += d[i + 2];
            count++;
        }
        if (count === 0) return [255, 255, 255];
        return [Math.round(rSum / count), Math.round(gSum / count), Math.round(bSum / count)];
    } catch (e) {
        return [255, 255, 255];
    }
}

export function spawnClickRipple(parent, clientX, clientY) {
    if (!parent) return;
    const rect = parent.getBoundingClientRect();
    const ripple = document.createElement('div');
    ripple.className = 'eyedropper-ripple';
    ripple.style.left = `${clientX - rect.left}px`;
    ripple.style.top = `${clientY - rect.top}px`;
    parent.appendChild(ripple);
    setTimeout(() => { if (ripple.parentNode) ripple.parentNode.removeChild(ripple); }, 700);
}

export function initLightboxEyedropperListeners() {
    const lbImg = document.getElementById('lightbox-img');
    if (!lbImg || lbImg.hasEyedropperListeners) return;
    lbImg.hasEyedropperListeners = true;

    const lbSampleCanvas = document.createElement('canvas');
    const lbSampleCtx = lbSampleCanvas.getContext('2d', { willReadFrequently: true });

    lbImg.addEventListener('mousemove', (e) => {
        if (!isEyedropperActive) return;
        const rect = lbImg.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return;

        if (lbSampleCanvas.width !== lbImg.naturalWidth || lbSampleCanvas.height !== lbImg.naturalHeight) {
            lbSampleCanvas.width = lbImg.naturalWidth || 800;
            lbSampleCanvas.height = lbImg.naturalHeight || 600;
            try {
                lbSampleCtx.drawImage(lbImg, 0, 0, lbSampleCanvas.width, lbSampleCanvas.height);
            } catch (err) {}
        }

        const normX = (e.clientX - rect.left) / rect.width;
        const normY = (e.clientY - rect.top) / rect.height;
        const px = Math.round(normX * (lbSampleCanvas.width - 1));
        const py = Math.round(normY * (lbSampleCanvas.height - 1));
        const rgb = sampleCanvasPatch(lbSampleCanvas, px, py, 2);

        const loupe = document.getElementById('eyedropper-loupe');
        const swatch = document.getElementById('loupe-swatch');
        const rgbTxt = document.getElementById('loupe-rgb');
        if (loupe) {
            loupe.style.left = `${e.clientX + 14}px`;
            loupe.style.top = `${e.clientY + 14}px`;
        }
        if (swatch) swatch.style.backgroundColor = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
        if (rgbTxt) rgbTxt.textContent = `R: ${rgb[0]} G: ${rgb[1]} B: ${rgb[2]}`;
    });

    lbImg.addEventListener('click', (e) => {
        if (!isEyedropperActive) return;
        e.stopPropagation();
        e.preventDefault();

        const rect = lbImg.getBoundingClientRect();
        const normX = Math.max(0, Math.min(1.0, (e.clientX - rect.left) / rect.width));
        const normY = Math.max(0, Math.min(1.0, (e.clientY - rect.top) / rect.height));

        spawnClickRipple(lbImg.parentElement, e.clientX, e.clientY);

        const lbPath = window.currentLightboxPath || '';
        const payload = lbPath ? {
            path: lbPath,
            x_ratio: normX,
            y_ratio: normY
        } : {
            rgb: [255, 255, 255]
        };

        fetch('/api/sample_rebate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                setActiveFilmBaseState(data.base_ratios, data.sampled_rgb);
                const cfgN = document.getElementById('config-neutralize');
                if (cfgN) cfgN.checked = true;
                const bchN = document.getElementById('batch-neutralize');
                if (bchN) bchN.checked = true;
                updateFilmBaseBadgeUI(activeFilmBaseRatios, activeFilmBaseRGB);
                window.appendLogLine?.(`[Client] Film base rebate sampled from image (${lbPath}): R=${activeFilmBaseRatios[0].toFixed(3)}, G=${activeFilmBaseRatios[1].toFixed(3)}, B=${activeFilmBaseRatios[2].toFixed(3)}`);
            }
        })
        .catch(err => console.error("Failed to sample rebate from image:", err))
        .finally(() => {
            cancelFilmBaseEyedropper();
        });
    });
}

// --- Live View Magnification & Fullscreen ---
export function setZoomMagnification(factor) {
    zoomFactor = factor;

    document.querySelectorAll('.btn-zoom-level').forEach(btn => btn.classList.remove('active'));
    const activeBtn = document.getElementById(`btn-zoom-${factor}x`);
    if (activeBtn) activeBtn.classList.add('active');

    const lvCanvas = document.getElementById('camera-liveview-canvas');

    if (factor === 5) {
        isZoomed = false;
        if (lvCanvas) lvCanvas.style.cursor = 'default';
        window.appendLogLine?.(`[Client] Enabling Canon 5x Native Hardware Sensor Zoom (1:1 Full Resolution)...`);
        setCameraConfig('eoszoom', '5');
    } else if (factor === 3) {
        isZoomed = true;
        if (lvCanvas) lvCanvas.style.cursor = 'crosshair';
        setCameraConfig('eoszoom', '1');
    } else {
        isZoomed = false;
        if (lvCanvas) lvCanvas.style.cursor = 'zoom-in';
        setCameraConfig('eoszoom', '1');
    }

    if (marginOverlayVisible) {
        updateMarginOverlay();
    }
}

export function toggleLiveviewFullscreen() {
    isLiveviewFullscreen = !isLiveviewFullscreen;
    window.isLiveviewFullscreen = isLiveviewFullscreen;
    const card = document.querySelector('.liveview-card');
    const btn = document.getElementById('btn-maximize-liveview');
    const hud = document.getElementById('fullscreen-focus-hud');

    if (card) {
        if (isLiveviewFullscreen) {
            card.classList.add('fullscreen-focus-mode');
            if (btn) {
                btn.innerHTML = '<span class="tool-icon">✖</span> <span class="tool-text">Exit (ESC)</span>';
                btn.classList.add('btn-fullscreen-active');
            }
            if (hud) hud.style.display = 'flex';
        } else {
            card.classList.remove('fullscreen-focus-mode');
            if (btn) {
                btn.innerHTML = '<span class="tool-icon">🔍</span> <span class="tool-text">Maximize</span>';
                btn.classList.remove('btn-fullscreen-active');
            }
            if (hud) hud.style.display = 'none';
        }
    }
}

export function initCameraUI() {
    initHistogramCanvas();

    liveviewCanvas = document.getElementById('camera-liveview-canvas');
    if (liveviewCanvas && !liveviewCanvas.hasZoomListener) {
        liveviewCanvas.hasZoomListener = true;

        liveviewCanvas.addEventListener('mousemove', (e) => {
            if (!isEyedropperActive) return;
            const rect = liveviewCanvas.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0) return;

            const scaleX = liveviewCanvas.width / rect.width;
            const scaleY = liveviewCanvas.height / rect.height;
            const cx = (e.clientX - rect.left) * scaleX;
            const cy = (e.clientY - rect.top) * scaleY;
            const rgb = sampleCanvasPatch(liveviewCanvas, Math.round(cx), Math.round(cy), 2);

            const loupe = document.getElementById('eyedropper-loupe');
            const swatch = document.getElementById('loupe-swatch');
            const rgbTxt = document.getElementById('loupe-rgb');
            if (loupe) {
                loupe.style.left = `${e.clientX + 14}px`;
                loupe.style.top = `${e.clientY + 14}px`;
            }
            if (swatch) swatch.style.backgroundColor = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
            if (rgbTxt) rgbTxt.textContent = `R: ${rgb[0]} G: ${rgb[1]} B: ${rgb[2]}`;
        });

        liveviewCanvas.addEventListener('click', (e) => {
            if (!isLiveviewActive) return;

            if (isEyedropperActive) {
                e.stopPropagation();
                e.preventDefault();

                const rect = liveviewCanvas.getBoundingClientRect();
                const scaleX = liveviewCanvas.width / rect.width;
                const scaleY = liveviewCanvas.height / rect.height;
                const cx = (e.clientX - rect.left) * scaleX;
                const cy = (e.clientY - rect.top) * scaleY;
                const rgb = sampleCanvasPatch(liveviewCanvas, Math.round(cx), Math.round(cy), 2);

                spawnClickRipple(liveviewCanvas.parentElement, e.clientX, e.clientY);

                fetch('/api/sample_rebate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ rgb: rgb })
                })
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        setActiveFilmBaseState(data.base_ratios, data.sampled_rgb);
                        const cfgN = document.getElementById('config-neutralize');
                        if (cfgN) cfgN.checked = true;
                        const bchN = document.getElementById('batch-neutralize');
                        if (bchN) bchN.checked = true;
                        updateFilmBaseBadgeUI(activeFilmBaseRatios, activeFilmBaseRGB);
                        window.appendLogLine?.(`[Client] Film base rebate sampled from Live View: R=${activeFilmBaseRatios[0].toFixed(3)}, G=${activeFilmBaseRatios[1].toFixed(3)}, B=${activeFilmBaseRatios[2].toFixed(3)} (RGB: ${rgb.join(',')})`);
                    }
                })
                .catch(err => console.error("Failed to sample film base from live view:", err))
                .finally(() => {
                    cancelFilmBaseEyedropper();
                });
                return;
            }

            const rect = liveviewCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            let normX = x / rect.width;
            let normY = y / rect.height;
            if (liveviewRotated180) {
                normX = 1.0 - normX;
                normY = 1.0 - normY;
            }

            if (isZoomed) {
                const dx = Math.abs(normX - zoomX);
                const dy = Math.abs(normY - zoomY);
                if (dx < 0.1 && dy < 0.1) {
                    setZoomMagnification(1);
                } else {
                    zoomX = normX;
                    zoomY = normY;
                    if (marginOverlayVisible) updateMarginOverlay();
                }
            } else {
                zoomX = normX;
                zoomY = normY;
                setZoomMagnification(zoomFactor > 1 ? zoomFactor : 3);
            }
        });
    }

    const btnRotate = document.getElementById('btn-rotate-liveview');
    if (btnRotate && liveviewRotated180) {
        btnRotate.classList.add('active');
        btnRotate.style.borderColor = 'var(--accent-red)';
        btnRotate.style.color = 'var(--accent-red)';
    }

    offscreenCanvas = document.createElement('canvas');
    offscreenCtx = offscreenCanvas.getContext('2d');

    const toggle = document.getElementById('camera-liveview-toggle');
    if (toggle && toggle.checked) {
        toggle.checked = false;
    }
    if (isLiveviewActive) {
        isLiveviewActive = false;
        window.isLiveviewActive = false;
        if (liveviewTimeout) { clearTimeout(liveviewTimeout); liveviewTimeout = null; }
        const canvas = document.getElementById('camera-liveview-canvas');
        const placeholder = document.getElementById('liveview-placeholder');
        if (canvas) canvas.style.display = 'none';
        if (placeholder) placeholder.style.display = 'flex';
    }

    syncMiniScanlightUI();
    fetchCameraStatus();

    if (cameraStatusInterval) clearInterval(cameraStatusInterval);
    cameraStatusInterval = setInterval(fetchCameraStatus, 3000);
}

export function fetchCameraStatus() {
    fetch('/api/camera/status')
        .then(res => res.json())
        .then(data => {
            const badge = document.getElementById('camera-status-badge');
            const placeholder = document.getElementById('liveview-placeholder');
            if (badge) {
                badge.className = 'camera-badge';
                if (data.simulated) {
                    badge.classList.add('badge-simulated');
                    badge.textContent = 'Simulated';
                    badge.title = 'Camera: Simulated Mock Device';
                } else if (data.state === 'searching') {
                    badge.classList.add('badge-disconnected');
                    badge.textContent = 'Searching...';
                    badge.title = 'Searching for USB camera...';
                } else if (data.state === 'capturing') {
                    badge.classList.add('badge-connected');
                    badge.textContent = 'Capturing...';
                    badge.title = 'Camera is actively capturing frames';
                } else if (data.connected) {
                    badge.classList.add('badge-connected');
                    badge.textContent = data.model ? data.model : 'Connected';
                    badge.title = `Camera Connected: ${data.model || 'Ready'}`;
                } else {
                    badge.classList.add('badge-disconnected');
                    badge.textContent = 'Disconnected';
                    badge.title = 'Camera Disconnected';
                }
            }

            if (placeholder && !isLiveviewActive) {
                const placeholderSpan = placeholder.querySelector('span:not(.placeholder-icon)');
                if (placeholderSpan) {
                    if (data.state === 'searching') {
                        placeholderSpan.textContent = 'Searching for camera... Turn on camera or connect USB.';
                    } else if (data.connected) {
                        placeholderSpan.textContent = `Live Feed Inactive (${data.model || 'Camera Ready'})`;
                    } else if (data.simulated) {
                        placeholderSpan.textContent = 'Live Feed Inactive (Simulated Mode)';
                    } else {
                        placeholderSpan.textContent = 'Live Feed Inactive';
                    }
                }
            }

            if (data.settings && data.choices) {
                populateCameraSelect('camera-iso-select', data.settings.iso, data.choices.iso);
                populateCameraSelect('camera-aperture-select', data.settings.aperture, data.choices.aperture);
                populateCameraSelect('camera-shutter-select', data.settings.shutterspeed, data.choices.shutterspeed);
            }
        })
        .catch(err => console.error("Error fetching camera status:", err));
}

export function populateCameraSelect(elemId, activeVal, choices) {
    const select = document.getElementById(elemId);
    if (!select) return;

    const userVal = select.value;
    select.innerHTML = '';

    if (!choices || choices.length === 0) {
        const opt = document.createElement('option');
        opt.value = activeVal || '';
        opt.textContent = activeVal || 'Not Supported';
        select.appendChild(opt);
        select.disabled = true;
        return;
    }

    select.disabled = false;
    choices.forEach(val => {
        const opt = document.createElement('option');
        opt.value = val;
        opt.textContent = val;
        const targetVal = (userVal && choices.includes(userVal)) ? userVal : activeVal;
        if (targetVal && val.toString().toLowerCase() === targetVal.toString().toLowerCase()) {
            opt.selected = true;
        }
        select.appendChild(opt);
    });
}

export function setCameraConfig(name, value) {
    if (!value) return;
    window.appendLogLine?.(`[Client] Updating camera ${name} to ${value}...`);

    fetch('/api/camera/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name, value: value })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            window.appendLogLine?.(`[Client] Camera ${name} updated successfully.`);
            fetchCameraStatus();
        } else {
            window.appendLogLine?.(`[Client] Error updating camera ${name}: ${data.message}`);
            alert(`Error: ${data.message}`);
        }
    })
    .catch(err => console.error("Error updating config:", err));
}

export function parseShutterDuration(str) {
    if (!str) return 0;
    const clean = str.toString().trim().toLowerCase().replace('s', '');
    if (clean === 'auto' || clean === 'bulb') return 999999;
    if (clean.includes('/')) {
        const parts = clean.split('/');
        const num = parseFloat(parts[0]);
        const den = parseFloat(parts[1]);
        if (!isNaN(num) && !isNaN(den) && den !== 0) {
            return num / den;
        }
    }
    const val = parseFloat(clean);
    return isNaN(val) ? 0 : val;
}

export function stepCameraShutter(direction) {
    const select = document.getElementById('camera-shutter-select');
    if (!select || select.disabled || select.options.length <= 1) {
        window.appendLogLine?.('[Camera] Shutter speed control not available or no choices loaded.');
        return;
    }

    const currentIdx = select.selectedIndex;
    if (currentIdx < 0) return;

    const currentVal = select.options[currentIdx].value;
    const currentDur = parseShutterDuration(currentVal);

    const opts = Array.from(select.options).map((opt, i) => ({
        index: i,
        value: opt.value,
        duration: parseShutterDuration(opt.value)
    }));

    let targetIdx = -1;

    if (direction === 'faster' || direction === 'up' || direction === 1) {
        const candidates = opts.filter(o => o.duration < currentDur && o.duration > 0);
        if (candidates.length > 0) {
            candidates.sort((a, b) => b.duration - a.duration);
            targetIdx = candidates[0].index;
        } else {
            return;
        }
    } else {
        const candidates = opts.filter(o => o.duration > currentDur);
        if (candidates.length > 0) {
            candidates.sort((a, b) => a.duration - b.duration);
            targetIdx = candidates[0].index;
        } else {
            return;
        }
    }

    if (targetIdx >= 0 && targetIdx !== currentIdx) {
        select.selectedIndex = targetIdx;
        const newVal = select.options[targetIdx].value;
        setCameraConfig('shutterspeed', newVal);
    }
}

export function toggleCameraLiveview(active) {
    isLiveviewActive = active;
    window.isLiveviewActive = active;

    if (!active && zoomFactor !== 1) {
        setZoomMagnification(1);
    } else {
        isZoomed = false;
    }

    const canvas = document.getElementById('camera-liveview-canvas');
    const placeholder = document.getElementById('liveview-placeholder');
    if (canvas) canvas.style.cursor = 'zoom-in';

    fetch('/api/camera/toggle_liveview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ active: active })
    })
    .then(res => res.json())
    .then(data => {
        if (!data.success) {
            console.error("Failed to toggle live view state on server.");
            return;
        }

        if (active) {
            if (placeholder) placeholder.style.display = 'none';
            if (canvas) canvas.style.display = 'block';

            if (liveviewTimeout) clearTimeout(liveviewTimeout);
            isPollingFrame = false;
            pollLiveviewFrame();
        } else {
            if (canvas) {
                canvas.style.display = 'none';
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
            if (placeholder) placeholder.style.display = 'flex';
            if (liveviewTimeout) {
                clearTimeout(liveviewTimeout);
                liveviewTimeout = null;
            }
            isPollingFrame = false;
            clearHistogramCanvas();
            if (marginOverlayVisible) {
                updateMarginOverlay();
            }
        }
    })
    .catch(err => console.error("Error toggling live view:", err));
}

export async function pollLiveviewFrame() {
    if (!isLiveviewActive) return;
    if (isPollingFrame) return;
    isPollingFrame = true;

    const canvas = document.getElementById('camera-liveview-canvas');
    if (!canvas) {
        isPollingFrame = false;
        return;
    }
    const ctx = canvas.getContext('2d');
    const startTime = Date.now();

    try {
        const response = await fetch('/api/camera/frame', { cache: 'no-store' });
        if (!isLiveviewActive) {
            isPollingFrame = false;
            return;
        }

        if (response.status === 204) {
            liveviewTimeout = setTimeout(pollLiveviewFrame, 35);
            isPollingFrame = false;
            return;
        }

        if (!response.ok) {
            liveviewTimeout = setTimeout(pollLiveviewFrame, 150);
            isPollingFrame = false;
            return;
        }

        const blob = await response.blob();
        if (!blob || blob.size === 0) {
            liveviewTimeout = setTimeout(pollLiveviewFrame, 35);
            isPollingFrame = false;
            return;
        }

        let imgSource = null;
        if (window.createImageBitmap) {
            try {
                imgSource = await createImageBitmap(blob);
            } catch (e) {
                imgSource = null;
            }
        }

        if (!imgSource) {
            await new Promise((resolve, reject) => {
                const img = new Image();
                const url = URL.createObjectURL(blob);
                img.onload = () => {
                    URL.revokeObjectURL(url);
                    imgSource = img;
                    resolve();
                };
                img.onerror = () => {
                    URL.revokeObjectURL(url);
                    reject();
                };
                img.src = url;
            });
        }

        if (!isLiveviewActive || !imgSource) {
            if (imgSource && imgSource.close) imgSource.close();
            isPollingFrame = false;
            return;
        }

        const nw = imgSource.naturalWidth || imgSource.width;
        const nh = imgSource.naturalHeight || imgSource.height;
        if (nw > 0 && nh > 0) {
            if (canvas.width !== nw || canvas.height !== nh) {
                canvas.width = nw;
                canvas.height = nh;
            }

            ctx.save();
            if (liveviewRotated180) {
                ctx.translate(nw / 2, nh / 2);
                ctx.rotate(Math.PI);
                ctx.translate(-nw / 2, -nh / 2);
            }

            if (isZoomed) {
                const factor = zoomFactor > 1 ? zoomFactor : 3;
                const sw = nw / factor;
                const sh = nh / factor;
                let sx = (zoomX * nw) - (sw / 2);
                let sy = (zoomY * nh) - (sh / 2);
                sx = Math.max(0, Math.min(nw - sw, sx));
                sy = Math.max(0, Math.min(nh - sh, sy));
                ctx.drawImage(imgSource, sx, sy, sw, sh, 0, 0, nw, nh);
            } else {
                ctx.drawImage(imgSource, 0, 0, nw, nh);
            }
            ctx.restore();

            if (focusPeakingActive) {
                applyFocusPeaking(canvas, ctx);
            }
        }

        if (marginOverlayVisible) {
            updateMarginOverlay(imgSource);
        }

        const aspect = nh > 0 ? (nw / nh) : (4 / 3);
        const sampleW = 480;
        const sampleH = Math.round(480 / aspect);
        if (offscreenCanvas.width !== sampleW || offscreenCanvas.height !== sampleH) {
            offscreenCanvas.width = sampleW;
            offscreenCanvas.height = sampleH;
        }
        offscreenCtx.drawImage(imgSource, 0, 0, sampleW, sampleH);
        if (imgSource.close) {
            imgSource.close();
        }

        try {
            computeAndRenderHistogram(offscreenCtx, sampleW, sampleH);
        } catch (e) {
            console.error("Histogram parsing failed:", e);
        }

        const elapsed = Date.now() - startTime;
        const delay = Math.max(5, 33 - elapsed);
        liveviewTimeout = setTimeout(pollLiveviewFrame, delay);

    } catch (err) {
        if (isLiveviewActive) {
            liveviewTimeout = setTimeout(pollLiveviewFrame, 150);
        }
    } finally {
        isPollingFrame = false;
    }
}

export function triggerCameraCapture() {
    const btn = document.getElementById('btn-camera-capture');
    if (btn) btn.disabled = true;

    if (zoomFactor !== 1) {
        setZoomMagnification(1);
    }

    window.appendLogLine?.("[Client] Triggering capture command...");

    fetch('/api/camera/capture', { method: 'POST' })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            window.appendLogLine?.(`[Client] Capture completed. Saved to: ${data.path}`);
        } else {
            window.appendLogLine?.(`[Client] Capture failed: ${data.message}`);
            alert(`Capture error: ${data.message}`);
        }
    })
    .catch(err => {
        console.error("Capture trigger error:", err);
        window.appendLogLine?.(`[Client] Network error during capture trigger.`);
    })
    .finally(() => {
        if (btn) btn.disabled = false;
    });
}

// --- Focus Assist Helper Functions ---
export function toggleFocusPeaking(active) {
    focusPeakingActive = active;
    const sliderGroup = document.getElementById('peaking-sensitivity-group');
    if (sliderGroup) {
        sliderGroup.style.display = active ? 'block' : 'none';
    }
}

export function updateFocusPeakingThreshold(val) {
    peakingThreshold = parseInt(val, 10);
}

export function applyFocusPeaking(canvas, ctx) {
    const width = canvas.width;
    const height = canvas.height;

    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;

    const output = ctx.createImageData(width, height);
    const outData = output.data;
    outData.set(data);

    const threshold = peakingThreshold;

    for (let y = 0; y < height - 1; y += 2) {
        for (let x = 0; x < width - 1; x += 2) {
            const idx = (y * width + x) * 4;

            const r1 = data[idx], g1 = data[idx + 1], b1 = data[idx + 2];
            const lum1 = (r1 * 77 + g1 * 150 + b1 * 29) >> 8;

            const idxRight = idx + 8;
            const r2 = data[idxRight], g2 = data[idxRight + 1], b2 = data[idxRight + 2];
            const lum2 = (r2 * 77 + g2 * 150 + b2 * 29) >> 8;

            const idxDown = idx + (width * 8);
            const r3 = data[idxDown], g3 = data[idxDown + 1], b3 = data[idxDown + 2];
            const lum3 = (r3 * 77 + g3 * 150 + b3 * 29) >> 8;

            const diff = Math.abs(lum1 - lum2) + Math.abs(lum1 - lum3);

            if (diff > threshold) {
                for (let dy = 0; dy < 2; dy++) {
                    for (let dx = 0; dx < 2; dx++) {
                        const targetIdx = ((y + dy) * width + (x + dx)) * 4;
                        if (targetIdx < outData.length) {
                            outData[targetIdx] = 0;
                            outData[targetIdx + 1] = 255;
                            outData[targetIdx + 2] = 0;
                        }
                    }
                }
            }
        }
    }
    ctx.putImageData(output, 0, 0);
}

export function driveFocus(direction, speed) {
    window.appendLogLine?.(`[Client] Stepping focus ${direction} (speed ${speed})...`);
    fetch('/api/camera/focus_step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ direction: direction, speed: speed })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            window.appendLogLine?.(`[Client] Focus adjusted successfully.`);
        } else {
            window.appendLogLine?.(`[Client] Focus step failed: ${data.message || 'unknown error'}`);
        }
    })
    .catch(err => console.error("Error stepping focus:", err));
}

export function triggerAutofocus() {
    const btn = document.getElementById('btn-trigger-af');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Focusing...';
    }
    window.appendLogLine?.(`[Client] Triggering camera autofocus sequence...`);
    fetch('/api/camera/autofocus', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            window.appendLogLine?.(`[Client] Autofocus lock complete.`);
        } else {
            window.appendLogLine?.(`[Client] Autofocus trigger failed: ${data.message || 'unknown error'}`);
            alert(`Autofocus failed: ${data.message}`);
        }
    })
    .catch(err => console.error("Error triggering autofocus:", err))
    .finally(() => {
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Autofocus';
        }
    });
}

export function reconnectCameraDevice() {
    const btn = document.getElementById('btn-camera-reconnect');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Connecting...';
    }

    const toggle = document.getElementById('camera-liveview-toggle');
    if (toggle && toggle.checked) {
        toggle.checked = false;
        toggleCameraLiveview(false);
    }

    window.appendLogLine?.(`[Client] Re-initializing connection to physical camera...`);
    fetch('/api/camera/reconnect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            window.appendLogLine?.(`[Client] Reconnect query dispatched successfully.`);
            setTimeout(fetchCameraStatus, 500);
        } else {
            window.appendLogLine?.(`[Client] Reconnect failed: ${data.message || 'unknown error'}`);
            alert(`Reconnect failed: ${data.message}`);
        }
    })
    .catch(err => {
        console.error("Error reconnecting camera:", err);
        window.appendLogLine?.(`[Client] Network error during camera reconnect command.`);
    })
    .finally(() => {
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Reconnect';
        }
    });
}

export function toggleLiveviewRotation() {
    liveviewRotated180 = !liveviewRotated180;
    localStorage.setItem('liveviewRotated180', liveviewRotated180);

    const btn = document.getElementById('btn-rotate-liveview');
    if (btn) {
        if (liveviewRotated180) {
            btn.classList.add('active');
            btn.style.borderColor = 'var(--accent-red)';
            btn.style.color = 'var(--accent-red)';
        } else {
            btn.classList.remove('active');
            btn.style.borderColor = '';
            btn.style.color = '';
        }
    }

    if (isLiveviewActive) {
        pollLiveviewFrame();
    }
}

// Expose to window for inline HTML handlers
Object.assign(window, {
    toggleMarginOverlay,
    getContainedImageBounds,
    updateMarginOverlay,
    toggleFilmBaseEyedropper,
    cancelFilmBaseEyedropper,
    updateFilmBaseBadgeUI,
    onNeutralizeCheckboxChanged,
    resetFilmBaseToAuto,
    sampleCanvasPatch,
    spawnClickRipple,
    initLightboxEyedropperListeners,
    setZoomMagnification,
    toggleLiveviewFullscreen,
    initCameraUI,
    fetchCameraStatus,
    populateCameraSelect,
    setCameraConfig,
    parseShutterDuration,
    stepCameraShutter,
    toggleCameraLiveview,
    pollLiveviewFrame,
    triggerCameraCapture,
    toggleFocusPeaking,
    updateFocusPeakingThreshold,
    applyFocusPeaking,
    driveFocus,
    triggerAutofocus,
    reconnectCameraDevice,
    toggleLiveviewRotation,
});
