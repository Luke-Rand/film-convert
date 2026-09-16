// Global State
let currentTab = 'scanner';
let systemStatus = 'idle';
window.systemStatus = 'idle';
let activeSessionDirs = {};
let eventSource = null;
let activeHistogramMode = 'all';

// ============================================================
// TOOLTIP ENGINE
// Uses data-tooltip="..." attribute for content.
// A single floating #fc-tooltip div is reused for all tips.
// ============================================================

let _tipEl = null;         // cached tooltip DOM node
let _tipShowTimer = null;
let _tipHideTimer = null;
let _tipCurrentEl = null;  // the element whose tip is pending/shown

const TOOLTIP_SHOW_DELAY = 600;  // ms before showing
const TOOLTIP_HIDE_DELAY = 80;   // ms after leave before hiding
const TOOLTIP_MARGIN     = 10;   // px from viewport edge
const TOOLTIP_OFFSET_Y   = 8;    // px gap below target element

function _getTipEl() {
    if (!_tipEl) {
        _tipEl = document.getElementById('fc-tooltip');
        if (!_tipEl) {
            _tipEl = document.createElement('div');
            _tipEl.id = 'fc-tooltip';
            document.body.appendChild(_tipEl);
        }
    }
    return _tipEl;
}

function _positionTip(anchorEl) {
    const t    = _getTipEl();
    const rect = anchorEl.getBoundingClientRect();
    const vw   = window.innerWidth;
    const vh   = window.innerHeight;
    const tipW = t.offsetWidth  || 240;
    const tipH = t.offsetHeight || 48;

    let top  = rect.bottom + TOOLTIP_OFFSET_Y;
    let left = rect.left;

    // Flip above if clipped at bottom
    if (top + tipH > vh - TOOLTIP_MARGIN) {
        top = rect.top - tipH - TOOLTIP_OFFSET_Y;
    }
    // Clamp horizontal
    left = Math.min(left, vw - tipW - TOOLTIP_MARGIN);
    left = Math.max(left, TOOLTIP_MARGIN);
    // Clamp vertical
    top  = Math.max(top, TOOLTIP_MARGIN);

    t.style.left = left + 'px';
    t.style.top  = top  + 'px';
}

function _showTip(el) {
    if (document.body.classList.contains('tooltips-disabled')) return;
    const text = el.dataset.tooltip;
    if (!text) return;

    const t = _getTipEl();
    t.textContent = text;

    // Reset to hidden state cleanly, then measure, then reveal
    t.style.removeProperty('left');
    t.style.removeProperty('top');
    t.classList.remove('fc-tooltip-visible');
    t.style.display = 'block';

    // One rAF to allow layout so offsetWidth/Height are accurate
    requestAnimationFrame(() => {
        _positionTip(el);
        // Second rAF so the browser paints the hidden (opacity:0) state first,
        // giving the CSS transition something to animate from
        requestAnimationFrame(() => {
            t.classList.add('fc-tooltip-visible');
        });
    });
}

function _hideTip() {
    const t = _getTipEl();
    t.classList.remove('fc-tooltip-visible');
    // After transition completes, set display:none so it can't intercept clicks
    t.addEventListener('transitionend', function onEnd() {
        t.removeEventListener('transitionend', onEnd);
        if (!t.classList.contains('fc-tooltip-visible')) {
            t.style.display = 'none';
        }
    }, { once: true });
}

function _initTooltipListeners() {
    // Ensure tip starts hidden
    const t = _getTipEl();
    t.style.display = 'none';

    document.addEventListener('mouseover', (e) => {
        const el = e.target.closest('[data-tooltip]');
        if (!el) return;

        // If we're still on the same element, keep the existing timer
        if (el === _tipCurrentEl) {
            clearTimeout(_tipHideTimer);
            return;
        }

        _tipCurrentEl = el;
        clearTimeout(_tipShowTimer);
        clearTimeout(_tipHideTimer);
        _tipShowTimer = setTimeout(() => _showTip(el), TOOLTIP_SHOW_DELAY);
    });

    document.addEventListener('mouseout', (e) => {
        const el = e.target.closest('[data-tooltip]');
        if (!el) return;

        // Only hide if the pointer is truly leaving the [data-tooltip] element,
        // not just moving to one of its child nodes.
        const related = e.relatedTarget;
        if (related && el.contains(related)) return;

        clearTimeout(_tipShowTimer);
        _tipCurrentEl = null;
        _tipHideTimer = setTimeout(_hideTip, TOOLTIP_HIDE_DELAY);
    });

    // Dismiss on scroll or mousedown
    document.addEventListener('scroll', () => {
        clearTimeout(_tipShowTimer);
        _hideTip();
    }, { capture: true, passive: true });

    document.addEventListener('mousedown', () => {
        clearTimeout(_tipShowTimer);
        _hideTip();
    }, { passive: true });
}

// Toggle tooltips on/off (persisted in localStorage)
function toggleTooltips() {
    const disabled = document.body.classList.toggle('tooltips-disabled');
    localStorage.setItem('tooltips-disabled', disabled ? 'true' : 'false');
    if (disabled) _hideTip();
    const btn   = document.getElementById('btn-tooltip-toggle');
    const label = btn ? btn.querySelector('.tt-label') : null;
    if (btn)   btn.classList.toggle('active', !disabled);
    if (label) label.textContent = disabled ? 'Tooltips Off' : 'Tooltips On';
}




// On Load Initialization
document.addEventListener('DOMContentLoaded', () => {
    // Initialize tooltip system (must run after DOM is ready)
    _initTooltipListeners();

    // Set default scanner path
    const rootInput = document.getElementById('scanner-root-dir');
    if (rootInput && !rootInput.value) {
        rootInput.value = '~/Pictures/Scans';
    }

    // Load initial logs
    fetchInitialLogs();
    
    // Connect to real-time event stream
    connectSSE();

    // Initialize Camera UI
    initCameraUI();


    // Restore camera/light/liveview visibility selections from localStorage
    const showLiveview = localStorage.getItem('show-liveview-controls') !== 'false';
    const showCamera = localStorage.getItem('show-camera-controls') !== 'false';
    const showLight = localStorage.getItem('show-light-controls') !== 'false';
    
    const liveviewCheckbox = document.getElementById('ui-show-liveview');
    const camCheckbox = document.getElementById('ui-show-camera');
    const lightCheckbox = document.getElementById('ui-show-light');
    
    // We toggle liveview first, then camera, then light so grid updates correctly
    if (liveviewCheckbox) {
        liveviewCheckbox.checked = showLiveview;
        toggleUiSection('liveview', showLiveview);
    }
    if (camCheckbox) {
        camCheckbox.checked = showCamera;
        toggleUiSection('camera', showCamera);
    }
    if (lightCheckbox) {
        lightCheckbox.checked = showLight;
        toggleUiSection('light', showLight);
    }

    // Restore sidebar collapsed preference
    const sidebarCollapsed = localStorage.getItem('sidebar-collapsed') === 'true';
    if (sidebarCollapsed) {
        const sidebar = document.querySelector('.sidebar');
        if (sidebar) {
            sidebar.classList.add('collapsed');
            const toggleIcon = document.querySelector('#btn-sidebar-toggle .toggle-icon');
            if (toggleIcon) {
                toggleIcon.textContent = '▶';
            }
        }
    }

    // Restore active histogram mode preference
    activeHistogramMode = localStorage.getItem('histogram-mode') || 'all';
    setHistogramMode(activeHistogramMode);

    // Restore tooltip preference
    const tooltipsDisabled = localStorage.getItem('tooltips-disabled') === 'true';
    const tooltipBtn = document.getElementById('btn-tooltip-toggle');
    const tooltipLabel = tooltipBtn ? tooltipBtn.querySelector('.tt-label') : null;
    if (tooltipsDisabled) {
        document.body.classList.add('tooltips-disabled');
        if (tooltipBtn) tooltipBtn.classList.remove('active');
        if (tooltipLabel) tooltipLabel.textContent = 'Tooltips Off';
    } else {
        document.body.classList.remove('tooltips-disabled');
        if (tooltipBtn) tooltipBtn.classList.add('active');
        if (tooltipLabel) tooltipLabel.textContent = 'Tooltips On';
    }
});

// Real-Time Server-Sent Events (SSE) Connection
function connectSSE() {
    if (eventSource) {
        eventSource.close();
    }

    eventSource = new EventSource('/api/stream');

    eventSource.addEventListener('status', (event) => {
        try {
            const data = JSON.parse(event.data);
            systemStatus = data.status;
            window.systemStatus = data.status;
            activeSessionDirs = data.dirs;
            updateStatusUI(data);
        } catch (err) {
            console.error("Error parsing status SSE data:", err);
        }
    });

    eventSource.addEventListener('log', (event) => {
        try {
            const data = JSON.parse(event.data);
            appendLogLine(data.line);
        } catch (err) {
            console.error("Error parsing log SSE data:", err);
        }
    });

    eventSource.addEventListener('triplet_means', (event) => {
        try {
            const data = JSON.parse(event.data);
            if (window.onTripletMeansReceived) {
                window.onTripletMeansReceived(data);
            }
        } catch (err) {
            console.error("Error parsing triplet_means SSE data:", err);
        }
    });

    eventSource.onerror = (err) => {
        console.error("SSE connection error:", err);
        eventSource.close();
        // Retry connection after 5 seconds
        setTimeout(connectSSE, 5000);
    };
}

// Fetch historical logs on page load
function fetchInitialLogs() {
    fetch('/api/logs')
        .then(res => res.json())
        .then(data => {
            const consoleOut = document.getElementById('console-output');
            if (consoleOut) consoleOut.innerHTML = '';
            const batchConsoleOut = document.getElementById('batch-console-output');
            if (batchConsoleOut) batchConsoleOut.innerHTML = '';
            const scanlightConsoleOut = document.getElementById('scanlight-console-output');
            if (scanlightConsoleOut) scanlightConsoleOut.innerHTML = '';
            
            const logs = data.logs || [];
            logs.forEach(line => {
                appendLogLine(line);
            });
        })
        .catch(err => console.error("Error fetching initial logs:", err));
}

// Tab Navigation
function switchTab(tabId) {
    currentTab = tabId;
    
    // Update nav button active states
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    const activeBtn = document.getElementById(`nav-${tabId}-btn`);
    if (activeBtn) activeBtn.classList.add('active');
    
    // Update panel active states
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.remove('active');
    });
    const activePanel = document.getElementById(`panel-${tabId}`);
    if (activePanel) activePanel.classList.add('active');

    // Perform tab specific actions
    if (tabId === 'gallery') {
        refreshFiles();
    }
    
    // Camera live view tab management
    if (tabId === 'scanner') {
        initCameraUI();
    } else {
        // Auto-pause live view when switching away to preserve camera battery
        const toggle = document.getElementById('camera-liveview-toggle');
        if (toggle && toggle.checked) {
            toggle.checked = false;
            toggleCameraLiveview(false);
        }
        if (cameraStatusInterval) {
            clearInterval(cameraStatusInterval);
            cameraStatusInterval = null;
        }
    }
}

// Slider value displays
function updateSliderVal(id) {
    const slider = document.getElementById(`config-${id}`);
    const valSpan = document.getElementById(`val-${id}`);
    if (slider && valSpan) {
        let val = slider.value;
        if (id === 'margin') {
            val = Math.round(val * 100) + '%';
        }
        valSpan.textContent = val;
    }
}

// Double handle update batch values
function updateSliderVal(id) {
    if (id === 'gamma') {
        document.getElementById('val-gamma').textContent = document.getElementById('config-gamma').value;
    } else if (id === 'clip') {
        document.getElementById('val-clip').textContent = document.getElementById('config-clip').value;
    } else if (id === 'scurve') {
        document.getElementById('val-scurve').textContent = document.getElementById('config-scurve').value;
    } else if (id === 'margin') {
        const val = parseFloat(document.getElementById('config-margin').value);
        document.getElementById('val-margin').textContent = Math.round(val * 100) + '%';
    } else if (id === 'batch-gamma') {
        document.getElementById('val-batch-gamma').textContent = document.getElementById('batch-gamma').value;
    } else if (id === 'batch-clip') {
        document.getElementById('val-batch-clip').textContent = document.getElementById('batch-clip').value;
    } else if (id === 'batch-scurve') {
        document.getElementById('val-batch-scurve').textContent = document.getElementById('batch-scurve').value;
    } else if (id === 'batch-margin') {
        const val = parseFloat(document.getElementById('batch-margin').value);
        document.getElementById('val-batch-margin').textContent = Math.round(val * 100) + '%';
    }
}

// --- Margin Overlay ---
let marginOverlayVisible = false;

function toggleMarginOverlay() {
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

function getContainedImageBounds(img, tempImg = null) {
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
        // Image is wider than container (letterbox top/bottom)
        w = containerWidth;
        h = containerWidth / imageRatio;
        l = 0;
        t = (containerHeight - h) / 2;
    } else {
        // Image is taller than container (letterbox sides)
        h = containerHeight;
        w = containerHeight * imageRatio;
        l = (containerWidth - w) / 2;
        t = 0;
    }
    
    return { left: l, top: t, width: w, height: h };
}

function updateMarginOverlay(tempImg = null) {
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

// Keep overlay updated on window resize
window.addEventListener('resize', () => {
    if (marginOverlayVisible) {
        updateMarginOverlay();
    }
});




// Toggle batch settings panel
function toggleBatchSettings() {
    const panel = document.getElementById('batch-settings-panel');
    const arrow = document.getElementById('batch-settings-arrow');
    if (panel.classList.contains('hidden')) {
        panel.classList.remove('hidden');
        arrow.textContent = '▲';
    } else {
        panel.classList.add('hidden');
        arrow.textContent = '▼';
    }
}

function toggleScannerSettings() {
    const panel = document.getElementById('scanner-settings-panel');
    const arrow = document.getElementById('scanner-settings-arrow');
    if (panel.classList.contains('hidden')) {
        panel.classList.remove('hidden');
        arrow.textContent = '▲';
    } else {
        panel.classList.add('hidden');
        arrow.textContent = '▼';
    }
}


// Update description of batch task on radio select
function updateBatchTaskDesc() {
    // Just visual helpers inside index.html templates
}

// Update all UI elements representing state
function updateStatusUI(data) {
    // Summary status dot in sidebar
    const dot = document.getElementById('summary-status-dot');
    const text = document.getElementById('summary-status-text');
    if (dot) dot.className = `status-dot ${data.status}`;
    
    // Human readable text
    let statusText = "System Idle";
    if (data.status === 'monitoring') statusText = "Scanner Active";
    if (data.status === 'batch_processing') statusText = "Running Batch Task";
    if (text) text.textContent = statusText;

    // Scanner Panel Active Status Dashboard
    const statusVal = document.getElementById('monitor-status-val');
    const pathVal = document.getElementById('monitor-path-val');
    const toggleBtn = document.getElementById('btn-toggle-monitor');
    
    if (data.status === 'monitoring') {
        if (statusVal) {
            statusVal.textContent = "Monitoring...";
            statusVal.style.color = "var(--accent-green)";
        }
        if (pathVal) pathVal.textContent = data.dirs.negatives || "None";
        if (toggleBtn) {
            toggleBtn.textContent = "🛑 Stop Monitoring Session";
            toggleBtn.className = "btn btn-primary btn-large monitoring-active";
            toggleBtn.disabled = false;
        }
        disableInputs(true);
    } else if (data.status === 'batch_processing') {
        if (statusVal) {
            statusVal.textContent = "Processing Batch...";
            statusVal.style.color = "var(--accent-orange)";
        }
        if (pathVal) pathVal.textContent = "Manual Task";
        if (toggleBtn) {
            toggleBtn.textContent = "System Busy (Batching)";
            toggleBtn.className = "btn btn-secondary btn-large";
            toggleBtn.disabled = true;
        }
        disableInputs(true);
    } else {
        if (statusVal) {
            statusVal.textContent = "Idle / Stopped";
            statusVal.style.color = "var(--text-muted)";
        }
        if (pathVal) pathVal.textContent = "None";
        if (toggleBtn) {
            toggleBtn.textContent = "⚡ Start Monitoring Hot Folder";
            toggleBtn.className = "btn btn-primary btn-large";
            toggleBtn.disabled = false;
        }
        disableInputs(false);
        
        // Sync inputs from server configuration if not active
        if (data.config) {
            syncConfigToUI(data.config);
        }
    }

    // Reactively update batch run button state
    const runBtn = document.getElementById('btn-run-batch');
    if (runBtn) {
        if (data.status === 'batch_processing') {
            runBtn.disabled = true;
            runBtn.textContent = "⏳ Running Task...";
        } else if (data.status === 'monitoring') {
            runBtn.disabled = true;
            runBtn.textContent = "System Busy (Monitoring)";
        } else {
            runBtn.disabled = false;
            runBtn.textContent = "🚀 Execute Batch Process";
        }
    }
    
    // Sync mini light controls status
    syncMiniScanlightUI();
}

// Freeze forms when running
function disableInputs(disabled) {
    document.getElementById('scanner-root-dir').disabled = disabled;
    document.getElementById('scanner-stock').disabled = disabled;
    document.getElementById('scanner-format').disabled = disabled;
    document.getElementById('scanner-roll').disabled = disabled;
    const scannerModeEl = document.getElementById('scanner-mode');
    if (scannerModeEl) scannerModeEl.disabled = disabled;
    
    // Sliders
    document.getElementById('config-gamma').disabled = disabled;
    document.getElementById('config-clip').disabled = disabled;
    document.getElementById('config-scurve').disabled = disabled;
    document.getElementById('config-margin').disabled = disabled;
    
    // Checkboxes
    document.getElementById('config-autocrop').disabled = disabled;
    document.getElementById('config-global-levels').disabled = disabled;
    document.getElementById('config-neutralize').disabled = disabled;
    document.getElementById('config-compress').disabled = disabled;
    document.getElementById('config-align-channels').disabled = disabled;
    const cfgEmbed = document.getElementById('config-embed-metadata');
    if (cfgEmbed) cfgEmbed.disabled = disabled;
    const cfgProf = document.getElementById('config-color-profile');
    if (cfgProf) cfgProf.disabled = disabled;
}

// Sync config from backend to HTML inputs
function syncConfigToUI(config) {
    // Only update if inputs are not currently active/dirty
    if (document.activeElement.tagName === 'INPUT') return;

    document.getElementById('config-gamma').value = config.gamma;
    document.getElementById('config-clip').value = config.clip;
    document.getElementById('config-scurve').value = config.scurve;
    document.getElementById('config-margin').value = config.margin;
    
    document.getElementById('config-autocrop').checked = config.autocrop;
    document.getElementById('config-global-levels').checked = config.global_levels;
    document.getElementById('config-neutralize').checked = config.neutralize;
    document.getElementById('config-compress').checked = config.compress_tiff;
    document.getElementById('config-align-channels').checked = config.align_channels;
    
    // Monochrome settings
    const isMono = config.monochrome || false;
    document.getElementById('config-monochrome').checked = isMono;
    document.getElementById('config-monochrome-channel').value = config.monochrome_channel || 'luminance';
    const monoGroup = document.getElementById('mini-mono-channel-group');
    if (monoGroup) monoGroup.style.display = isMono ? 'block' : 'none';

    // Reversal / slide film setting
    const isReversal = config.reversal || false;
    const cfgReversal = document.getElementById('config-reversal');
    if (cfgReversal) cfgReversal.checked = isReversal;
    const bchReversal = document.getElementById('batch-reversal');
    if (bchReversal) bchReversal.checked = isReversal;
    
    const convertTiff = config.convert_to_tiff !== undefined ? config.convert_to_tiff : true;
    const cfgConvertTiff = document.getElementById('config-convert-tiff');
    if (cfgConvertTiff) cfgConvertTiff.checked = convertTiff;
    const bchConvertTiff = document.getElementById('batch-convert-tiff');
    if (bchConvertTiff) bchConvertTiff.checked = convertTiff;

    // Metadata preservation and color profile
    const embedMeta = config.embed_metadata !== undefined ? config.embed_metadata : true;
    const cfgEmbed = document.getElementById('config-embed-metadata');
    if (cfgEmbed) cfgEmbed.checked = embedMeta;
    const bchEmbed = document.getElementById('batch-embed-metadata');
    if (bchEmbed) bchEmbed.checked = embedMeta;

    const colorProf = config.color_profile || 'adobe_rgb';
    const cfgProf = document.getElementById('config-color-profile');
    if (cfgProf) cfgProf.value = colorProf;
    const bchProf = document.getElementById('batch-color-profile');
    if (bchProf) bchProf.value = colorProf;
    
    // Film base neutralizer ratios & badge
    if (config.base_ratios && Array.isArray(config.base_ratios) && config.base_ratios.length === 3) {
        activeFilmBaseRatios = config.base_ratios;
        updateFilmBaseBadgeUI(activeFilmBaseRatios, activeFilmBaseRGB);
    } else if (config.base_ratios === null) {
        activeFilmBaseRatios = null;
        updateFilmBaseBadgeUI(null, null);
    }
    
    // Update labels
    document.getElementById('val-gamma').textContent = config.gamma;
    document.getElementById('val-clip').textContent = config.clip;
    document.getElementById('val-scurve').textContent = config.scurve;
    document.getElementById('val-margin').textContent = Math.round(config.margin * 100) + '%';
}

// Toggle Start/Stop monitoring
function toggleMonitor() {
    if (systemStatus === 'monitoring') {
        // Stop it
        fetch('/api/stop', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                appendLogLine(`[Client] Stop request sent: ${data.message}`);
            })
            .catch(err => appendLogLine(`[Client Error] Failed to stop: ${err}`));
    } else {
        // Start it
        const rootDir = document.getElementById('scanner-root-dir').value.trim();
        const stock = document.getElementById('scanner-stock').value.trim();
        const format = document.getElementById('scanner-format').value.trim();
        const roll = document.getElementById('scanner-roll').value.trim();
        
        const modeEl = document.getElementById('scanner-mode');
        const mode = modeEl ? modeEl.value : 'triplet';

        if (!rootDir) {
            alert("Please provide a root directory path.");
            return;
        }

        const config = {
            gamma: parseFloat(document.getElementById('config-gamma').value),
            clip: parseFloat(document.getElementById('config-clip').value),
            scurve: parseFloat(document.getElementById('config-scurve').value),
            margin: parseFloat(document.getElementById('config-margin').value),
            autocrop: document.getElementById('config-autocrop').checked,
            global_levels: document.getElementById('config-global-levels').checked,
            neutralize: document.getElementById('config-neutralize').checked,
            base_ratios: activeFilmBaseRatios,
            compress_tiff: document.getElementById('config-compress').checked,
            align_channels: document.getElementById('config-align-channels').checked,
            monochrome: document.getElementById('config-monochrome').checked,
            monochrome_channel: document.getElementById('config-monochrome-channel').value,
            reversal: document.getElementById('config-reversal') ? document.getElementById('config-reversal').checked : false,
            convert_to_tiff: document.getElementById('config-convert-tiff') ? document.getElementById('config-convert-tiff').checked : true,
            color_profile: document.getElementById('config-color-profile') ? document.getElementById('config-color-profile').value : 'adobe_rgb',
            embed_metadata: document.getElementById('config-embed-metadata') ? document.getElementById('config-embed-metadata').checked : true
        };

        const payload = {
            root_dir: rootDir,
            stock: stock,
            format: format,
            roll: roll,
            mode: mode,
            config: config
        };

        fetch('/api/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(async res => {
            const data = await res.json().catch(() => null);
            if (!res.ok) {
                const msg = (data && data.message) || `Server returned HTTP ${res.status}: ${res.statusText}`;
                appendLogLine(`[Client Error] Failed to start: ${msg}`);
                alert(`Error starting scanner: ${msg}`);
                return;
            }
            if (data && data.success) {
                appendLogLine(`[Client] Session monitor initialized successfully.`);
            } else {
                const msg = (data && data.message) || 'Unknown error starting session.';
                appendLogLine(`[Client Error] Could not start session: ${msg}`);
                alert(`Error starting scanner: ${msg}`);
            }
        })
        .catch(err => appendLogLine(`[Client Error] Failed to start: ${err}`));
    }
}

// Add a log entry to HTML output
function appendLogLine(line) {
    const consoleOut = document.getElementById('console-output');
    const batchConsoleOut = document.getElementById('batch-console-output');
    const scanlightConsoleOut = document.getElementById('scanlight-console-output');
    
    // Choose which target console
    if (systemStatus === 'batch_processing' && currentTab === 'batch') {
        if (batchConsoleOut) {
            appendLineToTarget(batchConsoleOut, line);
        }
    } else {
        if (consoleOut) {
            appendLineToTarget(consoleOut, line);
        }
        if (scanlightConsoleOut) {
            appendLineToTarget(scanlightConsoleOut, line);
        }
    }
}

function appendLineToTarget(target, line) {
    const div = document.createElement('div');
    div.className = 'console-line';
    
    // Simple color highlighting based on line content
    if (line.includes('SUCCESS') || line.includes('completed') || line.includes('complete')) {
        div.classList.add('text-success');
    } else if (line.includes('ERROR') || line.includes('Failed') || line.includes('!' * 10)) {
        div.classList.add('text-error');
    } else if (line.includes('WARNING') || line.includes('anomaly') || line.includes('?') * 10) {
        div.classList.add('text-warning');
    } else if (line.startsWith('[Client]') || line.startsWith('[Scanlight]')) {
        div.classList.add('text-muted');
    }

    div.textContent = line;
    target.appendChild(div);
    
    // Auto Scroll to bottom
    target.scrollTop = target.scrollHeight;
}

// Clear system logs
function clearLogs() {
    fetch('/api/logs/clear', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                const consoleOut = document.getElementById('console-output');
                if (consoleOut) consoleOut.innerHTML = '';
                const batchConsoleOut = document.getElementById('batch-console-output');
                if (batchConsoleOut) batchConsoleOut.innerHTML = '';
                const scanlightConsoleOut = document.getElementById('scanlight-console-output');
                if (scanlightConsoleOut) scanlightConsoleOut.innerHTML = '';
            }
        })
        .catch(err => console.error(err));
}

// Execute batch processing task
function runBatchJob() {
    const inputPath = document.getElementById('batch-input-path').value.trim();
    if (!inputPath) {
        alert("Please provide an input directory path.");
        return;
    }

    let taskType = 'composite';
    document.getElementsByName('batch-task').forEach(rad => {
        if (rad.checked) taskType = rad.value;
    });

    const config = {
        gamma: parseFloat(document.getElementById('batch-gamma').value),
        clip: parseFloat(document.getElementById('batch-clip').value),
        scurve: parseFloat(document.getElementById('batch-scurve').value),
        margin: parseFloat(document.getElementById('batch-margin').value),
        autocrop: document.getElementById('batch-autocrop').checked,
        global_levels: document.getElementById('batch-global-levels').checked,
        neutralize: document.getElementById('batch-neutralize').checked,
        base_ratios: activeFilmBaseRatios,
        compress_tiff: document.getElementById('batch-compress').checked,
        align_channels: document.getElementById('batch-align-channels').checked,
        monochrome: document.getElementById('batch-monochrome').checked,
        monochrome_channel: document.getElementById('batch-monochrome-channel').value,
        reversal: document.getElementById('batch-reversal') ? document.getElementById('batch-reversal').checked : false,
        convert_to_tiff: document.getElementById('batch-convert-tiff') ? document.getElementById('batch-convert-tiff').checked : true,
        color_profile: document.getElementById('batch-color-profile') ? document.getElementById('batch-color-profile').value : 'adobe_rgb',
        embed_metadata: document.getElementById('batch-embed-metadata') ? document.getElementById('batch-embed-metadata').checked : true
    };

    const payload = {
        task_type: taskType,
        input_path: inputPath,
        config: config
    };

    // Clear batch console output
    const batchConsoleOut = document.getElementById('batch-console-output');
    if (batchConsoleOut) batchConsoleOut.innerHTML = '<div class="console-line text-muted">Submitting batch task...</div>';
    
    const runBtn = document.getElementById('btn-run-batch');
    if (runBtn) {
        runBtn.disabled = true;
        runBtn.textContent = "⏳ Running Task...";
    }

    fetch('/api/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            appendLogLine(`[Client] Batch job started. Status: ${data.message}`);
        } else {
            alert(`Error starting batch job: ${data.message}`);
            if (runBtn) {
                runBtn.disabled = false;
                runBtn.textContent = "🚀 Execute Batch Process";
            }
        }
    })
    .catch(err => {
        alert(`Request failed: ${err}`);
        if (runBtn) {
            runBtn.disabled = false;
            runBtn.textContent = "🚀 Execute Batch Process";
        }
    });
}

// Refresh Positive Scans list and populate gallery
function refreshFiles() {
    const grid = document.getElementById('gallery-grid');
    const countInfo = document.getElementById('gallery-count-info');
    
    if (!grid) return;
    
    grid.innerHTML = '<div class="empty-gallery-msg">Loading session files...</div>';
    
    fetch('/api/files')
        .then(res => res.json())
        .then(data => {
            if (!data.success) {
                grid.innerHTML = `<div class="empty-gallery-msg">${data.message || 'No active session. Please start monitoring or run a batch task.'}</div>`;
                countInfo.textContent = '0 Positive Frames found in session';
                return;
            }
            
            const positives = data.positives;
            countInfo.textContent = `${positives.length} Positive Frame(s) found in session`;
            
            if (positives.length === 0) {
                grid.innerHTML = '<div class="empty-gallery-msg">No positive TIFF scans found yet. Triplets will appear here once processed.</div>';
                return;
            }
            
            grid.innerHTML = ''; // Clear loader
            
            positives.forEach(file => {
                const card = document.createElement('div');
                card.className = 'gallery-item';
                card.onclick = () => openLightbox(file.name, file.path, file.size);
                
                const thumbContainer = document.createElement('div');
                thumbContainer.className = 'thumbnail-container';
                
                const img = document.createElement('img');
                img.className = 'thumbnail-img';
                img.alt = file.name;
                img.loading = 'lazy';
                // Fetch dynamic thumbnail: limit size to 250px width for fast loading
                img.src = `/api/preview?path=${encodeURIComponent(file.path)}&w=250`;
                
                thumbContainer.appendChild(img);
                
                const meta = document.createElement('div');
                meta.className = 'item-meta';
                
                const title = document.createElement('div');
                title.className = 'item-title';
                title.textContent = file.name;
                
                const size = document.createElement('div');
                size.className = 'item-size';
                size.textContent = formatBytes(file.size);
                
                meta.appendChild(title);
                meta.appendChild(size);
                
                card.appendChild(thumbContainer);
                card.appendChild(meta);
                
                grid.appendChild(card);
            });
        })
        .catch(err => {
            grid.innerHTML = `<div class="empty-gallery-msg text-error">Failed to fetch files: ${err}</div>`;
        });
}

// Human readable file size formatter
function formatBytes(bytes, decimals = 1) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

// Lightbox modal view control
function openLightbox(name, path, size) {
    const box = document.getElementById('image-lightbox');
    const img = document.getElementById('lightbox-img');
    const title = document.getElementById('lightbox-title');
    const desc = document.getElementById('lightbox-desc');
    const dlLink = document.getElementById('lightbox-download-link');
    
    currentLightboxPath = path;

    // Clear preview image while loading new one to avoid layout jumps
    img.src = '';
    title.textContent = name;
    desc.textContent = `Size: ${formatBytes(size)} | Path: ${path}`;
    
    // Set dynamic source link
    dlLink.href = `/api/preview?path=${encodeURIComponent(path)}`;
    
    box.classList.add('active');
    
    // Load high-resolution preview (800px width limit for responsive viewport loading)
    img.src = `/api/preview?path=${encodeURIComponent(path)}&w=900`;

    initLightboxEyedropperListeners();
}

function closeLightbox() {
    if (isEyedropperActive && eyedropperSource === 'lightbox') {
        cancelFilmBaseEyedropper();
    }
    const box = document.getElementById('image-lightbox');
    if (box) box.classList.remove('active');
}

// ============================================================
// INTERACTIVE FILM BASE NEUTRALIZATION PICKER
// ============================================================

let activeFilmBaseRatios = null; // [r_ratio, g_ratio, b_ratio]
let activeFilmBaseRGB = null;    // [r, g, b] (0-255)
let isEyedropperActive = false;
let eyedropperSource = null;    // 'liveview', 'lightbox'
let currentLightboxPath = '';

function toggleFilmBaseEyedropper(source) {
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

    // Update active button state
    ['btn-pick-film-base', 'btn-liveview-eyedropper', 'btn-lightbox-eyedropper'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.classList.add('btn-eyedropper-active');
    });

    // Show banner overlay
    if (source === 'liveview') {
        const banner = document.getElementById('liveview-eyedropper-banner');
        if (banner) banner.style.display = 'flex';
    } else if (source === 'lightbox') {
        const banner = document.getElementById('lightbox-eyedropper-banner');
        if (banner) banner.style.display = 'flex';
    }

    // Update cursor
    const lvCanvas = document.getElementById('camera-liveview-canvas');
    if (lvCanvas) lvCanvas.classList.add('canvas-eyedropper-mode');
    const lbImg = document.getElementById('lightbox-img');
    if (lbImg) lbImg.classList.add('img-eyedropper-mode');

    // Show loupe
    const loupe = document.getElementById('eyedropper-loupe');
    if (loupe) loupe.style.display = 'flex';

    appendLogLine("[Client] Film base eyedropper active. Click unexposed film rebate (orange mask) on preview to sample.");
}

function cancelFilmBaseEyedropper() {
    isEyedropperActive = false;
    eyedropperSource = null;

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

function updateFilmBaseBadgeUI(ratios, rgb) {
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
                // Approximate display sRGB color from linear ratios (assuming orange base)
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

function onNeutralizeCheckboxChanged(checked) {
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

function resetFilmBaseToAuto() {
    activeFilmBaseRatios = null;
    activeFilmBaseRGB = null;
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
        appendLogLine("[Client] Film base neutralization reset to automatic 99.9th percentile mode.");
    })
    .catch(err => console.error("Failed to reset film base:", err));
}

// Sample a 5x5 patch from a canvas around (x, y) to eliminate single-pixel grain noise
function sampleCanvasPatch(canvas, x, y, radius = 2) {
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

// Visual click ripple
function spawnClickRipple(parent, clientX, clientY) {
    if (!parent) return;
    const rect = parent.getBoundingClientRect();
    const ripple = document.createElement('div');
    ripple.className = 'eyedropper-ripple';
    ripple.style.left = `${clientX - rect.left}px`;
    ripple.style.top = `${clientY - rect.top}px`;
    parent.appendChild(ripple);
    setTimeout(() => { if (ripple.parentNode) ripple.parentNode.removeChild(ripple); }, 700);
}

// Initialize Lightbox Eyedropper Event Listeners
function initLightboxEyedropperListeners() {
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

        const payload = currentLightboxPath ? {
            path: currentLightboxPath,
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
                activeFilmBaseRatios = data.base_ratios;
                activeFilmBaseRGB = data.sampled_rgb;
                const cfgN = document.getElementById('config-neutralize');
                if (cfgN) cfgN.checked = true;
                const bchN = document.getElementById('batch-neutralize');
                if (bchN) bchN.checked = true;
                updateFilmBaseBadgeUI(activeFilmBaseRatios, activeFilmBaseRGB);
                appendLogLine(`[Client] Film base rebate sampled from image (${currentLightboxPath}): R=${activeFilmBaseRatios[0].toFixed(3)}, G=${activeFilmBaseRatios[1].toFixed(3)}, B=${activeFilmBaseRatios[2].toFixed(3)}`);
            }
        })
        .catch(err => console.error("Failed to sample rebate from image:", err))
        .finally(() => {
            cancelFilmBaseEyedropper();
        });
    });
}

// Folder Browser State
let browserActiveInputId = '';
let browserCurrentPath = '';
let browserSelectedPath = '';
let browserAllFolders = [];

function openFolderBrowser(inputId) {
    browserActiveInputId = inputId;
    const currentVal = document.getElementById(inputId).value.trim();
    
    // Reset selected path
    browserSelectedPath = '';
    
    // Clear filter input
    const filterInput = document.getElementById('browser-filter-input');
    if (filterInput) filterInput.value = '';

    // Show modal
    const modal = document.getElementById('folder-browser-modal');
    if (modal) modal.classList.add('active');
    
    loadDirectory(currentVal);
}

function closeFolderBrowser() {
    const modal = document.getElementById('folder-browser-modal');
    if (modal) modal.classList.remove('active');
}

function loadDirectory(path) {
    const listContainer = document.getElementById('browser-folder-list');
    if (!listContainer) return;
    
    listContainer.innerHTML = '<div style="text-align: center; padding: 2rem; color: var(--text-muted);">Loading directories...</div>';
    
    let url = '/api/browse';
    if (path) {
        url += `?path=${encodeURIComponent(path)}`;
    }
    
    fetch(url)
        .then(res => {
            if (!res.ok) {
                // If failed, try loading without path (defaults to Home)
                if (path) {
                    appendLogLine(`[Folder Browser] Failed to read '${path}'. Falling back to Home folder.`);
                    return fetch('/api/browse');
                }
                throw new Error("Failed to load filesystem root");
            }
            return res;
        })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                listContainer.innerHTML = `<div class="browser-folder-empty text-error">Error: ${data.error}</div>`;
                return;
            }
            
            browserCurrentPath = data.current;
            browserSelectedPath = data.current; // Select current path by default
            browserAllFolders = data.folders || [];
            
            renderBrowser(data);
        })
        .catch(err => {
            listContainer.innerHTML = `<div class="browser-folder-empty text-error">Error: ${err.message}</div>`;
        });
}

function renderBrowser(data) {
    renderBreadcrumbs(data.current);
    
    const listContainer = document.getElementById('browser-folder-list');
    if (!listContainer) return;
    listContainer.innerHTML = '';
    
    // Check if we are viewing Windows drives list
    if (data.current === 'root' && data.drives && data.drives.length > 0) {
        data.drives.forEach(drive => {
            const item = document.createElement('div');
            item.className = 'browser-folder-item';
            item.onclick = (e) => {
                selectFolderItem(item, drive);
            };
            item.ondblclick = () => {
                traverseToFolder(drive);
            };
            
            item.innerHTML = `
                <span class="browser-folder-icon">💾</span>
                <span class="browser-folder-name">${drive}</span>
            `;
            listContainer.appendChild(item);
        });
        return;
    }
    
    // Add ".." (Parent directory) item if parent exists
    if (data.parent) {
        const item = document.createElement('div');
        item.className = 'browser-folder-item';
        item.onclick = () => {
            selectFolderItem(item, data.parent);
        };
        item.ondblclick = () => {
            traverseToFolder(data.parent);
        };
        
        const label = data.parent === 'root' ? '.. [System Drives]' : '.. (Parent Directory)';
        item.innerHTML = `
            <span class="browser-folder-icon">⬆️</span>
            <span class="browser-folder-name" style="font-weight: 500;">${label}</span>
        `;
        listContainer.appendChild(item);
    }
    
    renderFolderListItems(data.folders);
}

function renderFolderListItems(folders) {
    const listContainer = document.getElementById('browser-folder-list');
    if (!listContainer) return;
    
    if (folders.length === 0) {
        const childCount = listContainer.children.length;
        if (childCount === 0) {
            listContainer.innerHTML = '<div class="browser-folder-empty">This folder is empty</div>';
        }
        return;
    }
    
    folders.forEach(folder => {
        const item = document.createElement('div');
        item.className = 'browser-folder-item';
        item.onclick = () => {
            selectFolderItem(item, folder.path);
        };
        item.ondblclick = () => {
            traverseToFolder(folder.path);
        };
        
        item.innerHTML = `
            <span class="browser-folder-icon">📁</span>
            <span class="browser-folder-name">${folder.name}</span>
        `;
        listContainer.appendChild(item);
    });
}

function renderBreadcrumbs(pathStr) {
    const container = document.getElementById('browser-breadcrumbs');
    if (!container) return;
    
    container.innerHTML = '';
    
    if (pathStr === 'root') {
        const rootCrumb = document.createElement('span');
        rootCrumb.className = 'breadcrumb-item';
        rootCrumb.textContent = 'This PC';
        rootCrumb.onclick = () => traverseToFolder('root');
        container.appendChild(rootCrumb);
        return;
    }
    
    const isWindows = pathStr.includes('\\') || pathStr.includes(':');
    const separator = isWindows ? '\\' : '/';
    
    let parts = pathStr.split(/[\\/]/);
    
    if (!isWindows && pathStr.startsWith('/')) {
        parts[0] = '';
    } else {
        parts = parts.filter(p => p !== '');
    }

    if (isWindows) {
        const pcCrumb = document.createElement('span');
        pcCrumb.className = 'breadcrumb-item';
        pcCrumb.textContent = 'This PC';
        pcCrumb.onclick = () => traverseToFolder('root');
        container.appendChild(pcCrumb);
        
        const sep = document.createElement('span');
        sep.className = 'breadcrumb-separator';
        sep.textContent = ' > ';
        container.appendChild(sep);
    }
    
    let runningPath = '';
    parts.forEach((part, index) => {
        if (index > 0) {
            const sep = document.createElement('span');
            sep.className = 'breadcrumb-separator';
            sep.textContent = separator;
            container.appendChild(sep);
        }
        
        if (isWindows) {
            if (index === 0) {
                runningPath = part + '\\';
            } else {
                runningPath = runningPath + part + '\\';
            }
        } else {
            if (index === 0 && part === '') {
                runningPath = '/';
            } else {
                runningPath = runningPath + (runningPath.endsWith('/') ? '' : '/') + part;
            }
        }
        
        const crumb = document.createElement('span');
        crumb.className = 'breadcrumb-item';
        crumb.textContent = part === '' ? '/' : part;
        
        const targetPath = runningPath;
        crumb.onclick = () => traverseToFolder(targetPath);
        
        container.appendChild(crumb);
    });
}

function selectFolderItem(itemEl, path) {
    document.querySelectorAll('.browser-folder-item').forEach(el => el.classList.remove('selected'));
    itemEl.classList.add('selected');
    browserSelectedPath = path;
}

function traverseToFolder(path) {
    const filterInput = document.getElementById('browser-filter-input');
    if (filterInput) filterInput.value = '';
    loadDirectory(path);
}

function filterBrowserFolders() {
    const query = document.getElementById('browser-filter-input').value.toLowerCase().trim();
    
    const filtered = browserAllFolders.filter(f => f.name.toLowerCase().includes(query));
    
    const data = {
        current: browserCurrentPath,
        parent: browserCurrentPath === 'root' ? '' : getParentPath(browserCurrentPath),
        drives: [],
        folders: filtered
    };
    
    renderBrowser(data);
}

function getParentPath(pathStr) {
    if (pathStr === 'root') return '';
    const parts = pathStr.split(/[\\/]/).filter(p => p !== '');
    if (parts.length <= 1) {
        return 'root';
    }
    const sep = pathStr.includes('\\') ? '\\' : '/';
    return pathStr.substring(0, pathStr.lastIndexOf(sep));
}

function confirmFolderSelection() {
    if (!browserSelectedPath) {
        alert("Please select a folder first.");
        return;
    }
    
    const input = document.getElementById(browserActiveInputId);
    if (input) {
        input.value = browserSelectedPath;
        input.dispatchEvent(new Event('change', { bubbles: true }));
    }
    closeFolderBrowser();
}

// ==========================================
// CAMERA LIVE VIEW & HISTOGRAM MODULE
// ==========================================

let isLiveviewActive = false;
let histogramCanvas = null;
let histogramCtx = null;
let liveviewCanvas = null;
let offscreenCanvas = null;
let offscreenCtx = null;
let liveviewTimeout = null;
let cameraStatusInterval = null;

// Focus Assist States
let isZoomed = false;
let zoomFactor = 1; // 1, 3, 6, 10
let zoomX = 0.5;
let zoomY = 0.5;
let focusPeakingActive = false;
let peakingThreshold = 30;
let crispPixelsActive = true;
let focusScorePeak = 0;
let liveviewRotated180 = localStorage.getItem('liveviewRotated180') === 'true';

function setZoomMagnification(factor) {
    zoomFactor = factor;
    
    document.querySelectorAll('.btn-zoom-level').forEach(btn => btn.classList.remove('active'));
    const activeBtn = document.getElementById(`btn-zoom-${factor}x`);
    if (activeBtn) activeBtn.classList.add('active');
    
    const liveviewCanvas = document.getElementById('camera-liveview-canvas');
    
    if (factor === 5) {
        // Canon 5x Native Hardware Sensor Zoom (1:1 Sensor-level Readout)
        isZoomed = false;
        if (liveviewCanvas) liveviewCanvas.style.cursor = 'default';
        appendLogLine(`[Client] Enabling Canon 5x Native Hardware Sensor Zoom (1:1 Full Resolution)...`);
        setCameraConfig('eoszoom', '5');
    } else if (factor === 3) {
        // 3x Digital Grain Loupe
        isZoomed = true;
        if (liveviewCanvas) liveviewCanvas.style.cursor = 'crosshair';
        setCameraConfig('eoszoom', '1');
    } else {
        // 1x Full View
        isZoomed = false;
        if (liveviewCanvas) liveviewCanvas.style.cursor = 'zoom-in';
        setCameraConfig('eoszoom', '1');
    }
    
    if (marginOverlayVisible) {
        updateMarginOverlay();
    }
}

let isLiveviewFullscreen = false;

function toggleLiveviewFullscreen() {
    isLiveviewFullscreen = !isLiveviewFullscreen;
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

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        if (isEyedropperActive) {
            cancelFilmBaseEyedropper();
        } else if (isLiveviewFullscreen) {
            toggleLiveviewFullscreen();
        }
    }
});

// Initialize camera controls and fetch status
function initCameraUI() {
    histogramCanvas = document.getElementById('camera-histogram-canvas');
    if (histogramCanvas) {
        histogramCtx = histogramCanvas.getContext('2d');
    }
    
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
                        activeFilmBaseRatios = data.base_ratios;
                        activeFilmBaseRGB = data.sampled_rgb;
                        const cfgN = document.getElementById('config-neutralize');
                        if (cfgN) cfgN.checked = true;
                        const bchN = document.getElementById('batch-neutralize');
                        if (bchN) bchN.checked = true;
                        updateFilmBaseBadgeUI(activeFilmBaseRatios, activeFilmBaseRGB);
                        appendLogLine(`[Client] Film base rebate sampled from Live View: R=${activeFilmBaseRatios[0].toFixed(3)}, G=${activeFilmBaseRatios[1].toFixed(3)}, B=${activeFilmBaseRatios[2].toFixed(3)} (RGB: ${rgb.join(',')})`);
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

    // Set initial rotation state from localStorage
    const btnRotate = document.getElementById('btn-rotate-liveview');
    if (btnRotate) {
        if (liveviewRotated180) {
            btnRotate.classList.add('active');
            btnRotate.style.borderColor = 'var(--accent-red)';
            btnRotate.style.color = 'var(--accent-red)';
        }
    }
    
    // Create high-performance offscreen canvas for sampling
    offscreenCanvas = document.createElement('canvas');
    offscreenCtx = offscreenCanvas.getContext('2d');

    // Always reset live view toggle to OFF on init — the server starts with live view off
    // and we don't want a stale checked state from a previous page load.
    const toggle = document.getElementById('camera-liveview-toggle');
    if (toggle && toggle.checked) {
        toggle.checked = false;
    }
    if (isLiveviewActive) {
        isLiveviewActive = false;
        if (liveviewTimeout) { clearTimeout(liveviewTimeout); liveviewTimeout = null; }
        const canvas = document.getElementById('camera-liveview-canvas');
        const placeholder = document.getElementById('liveview-placeholder');
        if (canvas) canvas.style.display = 'none';
        if (placeholder) placeholder.style.display = 'flex';
    }
    
    // Sync Mini Scanlight sliders with Scanlight tab controllers if available
    syncMiniScanlightUI();

    // Fetch active camera property dropdowns
    fetchCameraStatus();
    
    // Poll camera status every 3 seconds to sync physical dials and connection changes
    if (cameraStatusInterval) clearInterval(cameraStatusInterval);
    cameraStatusInterval = setInterval(fetchCameraStatus, 3000);
}


// Fetch camera connection status and configure properties
function fetchCameraStatus() {
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

            // Populate property selects
            if (data.settings && data.choices) {
                populateCameraSelect('camera-iso-select', data.settings.iso, data.choices.iso);
                populateCameraSelect('camera-aperture-select', data.settings.aperture, data.choices.aperture);
                populateCameraSelect('camera-shutter-select', data.settings.shutterspeed, data.choices.shutterspeed);
            }
        })
        .catch(err => console.error("Error fetching camera status:", err));
}

function populateCameraSelect(elemId, activeVal, choices) {
    const select = document.getElementById(elemId);
    if (!select) return;
    
    // Save current selection if active to avoid cursor reset during polling
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
        // Prefer user selection if still valid, otherwise fall back to backend state
        const targetVal = (userVal && choices.includes(userVal)) ? userVal : activeVal;
        if (targetVal && val.toString().toLowerCase() === targetVal.toString().toLowerCase()) {
            opt.selected = true;
        }
        select.appendChild(opt);
    });
}

// Update camera properties on backend
function setCameraConfig(name, value) {
    if (!value) return;
    appendLogLine(`[Client] Updating camera ${name} to ${value}...`);
    
    fetch('/api/camera/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name, value: value })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            appendLogLine(`[Client] Camera ${name} updated successfully.`);
            fetchCameraStatus(); // refresh values
        } else {
            appendLogLine(`[Client] Error updating camera ${name}: ${data.message}`);
            alert(`Error: ${data.message}`);
        }
    })
    .catch(err => console.error("Error updating config:", err));
}

// Toggle Live View streaming state
function toggleCameraLiveview(active) {
    isLiveviewActive = active;
    
    // When stopping live view, reset zoom magnification so hardware zoom is disengaged
    if (!active && zoomFactor !== 1 && typeof setZoomMagnification === 'function') {
        setZoomMagnification(1);
    } else {
        isZoomed = false; // Reset zoom state when toggling
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
            
            // Start the adaptive live view frame loop
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

let isPollingFrame = false;

// Adaptive live view frame fetcher (eliminates request backlog and handles 204 standby)
async function pollLiveviewFrame() {
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
            // Frame not ready or standby, poll again with short delay
            const delay = 35;
            liveviewTimeout = setTimeout(pollLiveviewFrame, delay);
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
        
        // Update margin overlay position and sizing
        if (marginOverlayVisible) {
            updateMarginOverlay(imgSource);
        }
        
        // High-accuracy sampling canvas for real-time histogram & exposure calculation
        // Sample at 480px width (preserving aspect ratio) to avoid blurring/damping highlight peaks
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
            // Apply active scan margin to ignore film carrier borders / metal edges
            const marginSlider = document.getElementById('config-margin');
            const marginFraction = marginSlider ? parseFloat(marginSlider.value) : 0.04;
            const safeMargin = Math.max(0.02, Math.min(0.20, marginFraction));
            
            const mx = Math.floor(sampleW * safeMargin);
            const my = Math.floor(sampleH * safeMargin);
            const roiW = sampleW - (2 * mx);
            const roiH = sampleH - (2 * my);
            
            const imgData = offscreenCtx.getImageData(mx, my, roiW, roiH);
            const pixels = imgData.data;
            const totalPixels = roiW * roiH;
            
            const rHist = new Uint32Array(256);
            const gHist = new Uint32Array(256);
            const bHist = new Uint32Array(256);
            
            let clippedCount = 0;
            let rMax = 0, gMax = 0, bMax = 0;
            
            for (let i = 0; i < pixels.length; i += 4) {
                const r = pixels[i];
                const g = pixels[i + 1];
                const b = pixels[i + 2];
                
                rHist[r]++;
                gHist[g]++;
                bHist[b]++;
                
                if (r > rMax) rMax = r;
                if (g > gMax) gMax = g;
                if (b > bMax) bMax = b;
                
                if (r >= 254 || g >= 254 || b >= 254) {
                    clippedCount++;
                }
            }
            
            // Calculate 99.9th percentile values
            const p999Threshold = totalPixels * 0.999;
            const calcP99 = (hist) => {
                let acc = 0;
                for (let v = 0; v < 256; v++) {
                    acc += hist[v];
                    if (acc >= p999Threshold) return v;
                }
                return 255;
            };
            
            const rP99 = calcP99(rHist);
            const gP99 = calcP99(gHist);
            const bP99 = calcP99(bHist);
            const clipPct = ((clippedCount / totalPixels) * 100).toFixed(1);
            
            const stats = {
                rMax, gMax, bMax,
                rP99, gP99, bP99,
                clippedCount,
                clipPct,
                totalPixels,
                timestamp: Date.now()
            };
            window.lastHistogramStats = stats;
            
            renderRGBHistogram(rHist, gHist, bHist, stats);
        } catch (e) {
            console.error("Histogram parsing failed:", e);
        }
        
        const elapsed = Date.now() - startTime;
        const delay = Math.max(5, 33 - elapsed); // Target ~30 FPS
        liveviewTimeout = setTimeout(pollLiveviewFrame, delay);
        
    } catch (err) {
        if (isLiveviewActive) {
            liveviewTimeout = setTimeout(pollLiveviewFrame, 150);
        }
    } finally {
        isPollingFrame = false;
    }
}

// Manual raw image capture
function triggerCameraCapture() {
    const btn = document.getElementById('btn-camera-capture');
    if (btn) btn.disabled = true;
    
    // Disengage hardware zoom prior to capture if active
    if (zoomFactor !== 1 && typeof setZoomMagnification === 'function') {
        setZoomMagnification(1);
    }
    
    appendLogLine("[Client] Triggering capture command...");
    
    fetch('/api/camera/capture', { method: 'POST' })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            appendLogLine(`[Client] Capture completed. Saved to: ${data.path}`);
        } else {
            appendLogLine(`[Client] Capture failed: ${data.message}`);
            alert(`Capture error: ${data.message}`);
        }
    })
    .catch(err => {
        console.error("Capture trigger error:", err);
        appendLogLine(`[Client] Network error during capture trigger.`);
    })
    .finally(() => {
        if (btn) btn.disabled = false;
    });
}

function renderRGBHistogram(rHist, gHist, bHist, stats = null) {
    if (!histogramCtx || !histogramCanvas) return;
    
    const w = histogramCanvas.width;
    const h = histogramCanvas.height;
    
    // Clear canvas
    histogramCtx.clearRect(0, 0, w, h);
    
    // Draw vertical reference grid lines: 25%, 50% (middle gray), 75%, and 95% (ETTR ceiling)
    const gridPoints = [
        { pct: 0.25, stroke: 'rgba(255, 255, 255, 0.07)', dash: [2, 2] },
        { pct: 0.50, stroke: 'rgba(255, 255, 255, 0.12)', dash: [] }, // 18% middle gray
        { pct: 0.75, stroke: 'rgba(255, 255, 255, 0.07)', dash: [2, 2] },
        { pct: 242 / 255, stroke: 'rgba(245, 158, 11, 0.35)', dash: [3, 2] } // 95% ETTR target ceiling
    ];
    
    histogramCtx.save();
    gridPoints.forEach(pt => {
        const x = Math.round(pt.pct * w);
        histogramCtx.beginPath();
        histogramCtx.setLineDash(pt.dash);
        histogramCtx.moveTo(x, 0);
        histogramCtx.lineTo(x, h);
        histogramCtx.strokeStyle = pt.stroke;
        histogramCtx.lineWidth = 1;
        histogramCtx.stroke();
    });
    histogramCtx.restore();
    
    const mode = activeHistogramMode;
    
    if (mode === 'split') {
        const channelHeight = h / 3;
        
        const channels = [
            { data: rHist, fill: 'rgba(239, 68, 68, 0.2)', stroke: '#ef4444', offset: 0 },
            { data: gHist, fill: 'rgba(34, 197, 94, 0.2)', stroke: '#22c55e', offset: channelHeight },
            { data: bHist, fill: 'rgba(59, 130, 246, 0.2)', stroke: '#3b82f6', offset: channelHeight * 2 }
        ];
        
        channels.forEach(ch => {
            const chMax = Math.max(...ch.data) || 1;
            const startY = ch.offset + channelHeight;
            
            histogramCtx.beginPath();
            histogramCtx.moveTo(0, startY);
            
            for (let x = 0; x < 256; x++) {
                const val = ch.data[x];
                const px = (x / 255) * w;
                // Scale within the subchannel height space
                const py = startY - (val / chMax) * (channelHeight - 4);
                histogramCtx.lineTo(px, py);
            }
            
            histogramCtx.lineTo(w, startY);
            histogramCtx.closePath();
            
            histogramCtx.fillStyle = ch.fill;
            histogramCtx.fill();
            
            histogramCtx.lineWidth = 1;
            histogramCtx.strokeStyle = ch.stroke;
            histogramCtx.stroke();
            
            // Draw baseline for the channel
            histogramCtx.beginPath();
            histogramCtx.moveTo(0, startY);
            histogramCtx.lineTo(w, startY);
            histogramCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
            histogramCtx.lineWidth = 1;
            histogramCtx.stroke();
        });
        
    } else {
        // Find maximum count for scaling
        let maxVal = 1;
        
        const channels = [];
        if (mode === 'all' || mode === 'r') {
            channels.push({ data: rHist, fill: 'rgba(239, 68, 68, 0.2)', stroke: '#ef4444' });
        }
        if (mode === 'all' || mode === 'g') {
            channels.push({ data: gHist, fill: 'rgba(34, 197, 94, 0.2)', stroke: '#22c55e' });
        }
        if (mode === 'all' || mode === 'b') {
            channels.push({ data: bHist, fill: 'rgba(59, 130, 246, 0.2)', stroke: '#3b82f6' });
        }
        
        // Compute maxVal based on active channels
        const activeArrays = channels.map(c => c.data);
        if (activeArrays.length > 0) {
            const merged = [].concat(...activeArrays);
            maxVal = Math.max(...merged) || 1;
        }
        
        // Setup composite screen blending ONLY if mode is 'all'
        if (mode === 'all') {
            histogramCtx.globalCompositeOperation = 'screen';
        } else {
            histogramCtx.globalCompositeOperation = 'source-over';
        }
        
        channels.forEach(ch => {
            histogramCtx.beginPath();
            histogramCtx.moveTo(0, h);
            
            for (let x = 0; x < 256; x++) {
                const val = ch.data[x];
                const px = (x / 255) * w;
                const py = h - (val / maxVal) * (h - 10);
                histogramCtx.lineTo(px, py);
            }
            
            histogramCtx.lineTo(w, h);
            histogramCtx.closePath();
            
            // Fill area
            histogramCtx.fillStyle = ch.fill;
            histogramCtx.fill();
            
            // Draw path outline
            histogramCtx.lineWidth = 1.25;
            histogramCtx.strokeStyle = ch.stroke;
            histogramCtx.stroke();
        });
        
        // Reset blending mode
        histogramCtx.globalCompositeOperation = 'source-over';
    }
    
    // Update live numeric statistics badges
    if (stats) {
        const rEl = document.getElementById('hist-val-r');
        const gEl = document.getElementById('hist-val-g');
        const bEl = document.getElementById('hist-val-b');
        if (rEl) rEl.textContent = `${stats.rP99} (${Math.round(stats.rP99 / 2.55)}%)`;
        if (gEl) gEl.textContent = `${stats.gP99} (${Math.round(stats.gP99 / 2.55)}%)`;
        if (bEl) bEl.textContent = `${stats.bP99} (${Math.round(stats.bP99 / 2.55)}%)`;
        
        const clipText = document.getElementById('hist-clip-text');
        const clipBadge = document.getElementById('hist-clip-badge');
        if (clipText) clipText.textContent = `Clip: ${stats.clipPct}%`;
        if (clipBadge) {
            if (stats.clippedCount > 0 || parseFloat(stats.clipPct) > 0.0) {
                clipBadge.classList.add('clip-warning');
            } else {
                clipBadge.classList.remove('clip-warning');
            }
        }
    }
}

function setHistogramMode(mode) {
    activeHistogramMode = mode;
    
    // Toggle active classes on buttons
    const btnIds = {
        'all': 'btn-hist-all',
        'r': 'btn-hist-r',
        'g': 'btn-hist-g',
        'b': 'btn-hist-b',
        'split': 'btn-hist-split'
    };
    
    for (const [m, id] of Object.entries(btnIds)) {
        const btn = document.getElementById(id);
        if (btn) {
            if (m === mode) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        }
    }
    
    localStorage.setItem('histogram-mode', mode);
}

function clearHistogramCanvas() {
    if (histogramCtx && histogramCanvas) {
        histogramCtx.clearRect(0, 0, histogramCanvas.width, histogramCanvas.height);
    }
}

// Sync Mini-Scanlight UI with active global ScanlightController
function syncMiniScanlightUI() {
    const statusText = document.getElementById('mini-sl-status-text');
    if (!statusText) return;
    
    const isConnected = window.scanlightController && window.scanlightController.connected;
    
    if (isConnected) {
        statusText.innerHTML = 'Status: <span style="color: var(--accent-green); font-weight: 600;">Connected</span>';
        document.getElementById('btn-mini-sl-connect').textContent = 'Disconnect';
    } else {
        statusText.innerHTML = 'Status: <span style="color: var(--accent-red); font-weight: 600;">Disconnected</span>';
        document.getElementById('btn-mini-sl-connect').textContent = 'Connect Light';
    }
    
    if (window.scanlightController) {
        // Sync values
        const r = window.scanlightController.red;
        const g = window.scanlightController.green;
        const b = window.scanlightController.blue;
        
        const rEl = document.getElementById('mini-sl-red');
        if (rEl) {
            rEl.value = r;
            document.getElementById('val-mini-sl-red').textContent = r;
        }
        
        const gEl = document.getElementById('mini-sl-green');
        if (gEl) {
            gEl.value = g;
            document.getElementById('val-mini-sl-green').textContent = g;
        }
        
        const bEl = document.getElementById('mini-sl-blue');
        if (bEl) {
            bEl.value = b;
            document.getElementById('val-mini-sl-blue').textContent = b;
        }
    }
}

function toggleMiniScanlightConnection() {
    if (window.scanlightController) {
        window.scanlightController.toggleConnection()
            .then(() => {
                setTimeout(syncMiniScanlightUI, 100);
            })
            .catch(err => {
                alert("Scanlight connection failed: " + err);
                syncMiniScanlightUI();
            });
    } else {
        alert("Scanlight controller script is not loaded.");
    }
}

function updateMiniScanlightColor(channel, value) {
    const intVal = parseInt(value, 10);
    const valSpan = document.getElementById(`val-mini-sl-${channel}`);
    if (valSpan) valSpan.textContent = intVal;
    const slider = document.getElementById(`mini-sl-${channel}`);
    if (slider && slider.value !== String(intVal)) slider.value = intVal;
    
    let r = parseInt(document.getElementById('mini-sl-red')?.value || 255, 10);
    let g = parseInt(document.getElementById('mini-sl-green')?.value || 255, 10);
    let b = parseInt(document.getElementById('mini-sl-blue')?.value || 255, 10);
    if (channel === 'red') r = intVal;
    if (channel === 'green') g = intVal;
    if (channel === 'blue') b = intVal;
    
    if (window.scanlightController) {
        window.scanlightController[channel] = intVal;
        
        // Sync slider in main scanlight tab if present
        const mainSlider = document.getElementById(`scanlight-${channel}-slider`);
        const mainInput = document.getElementById(`scanlight-${channel}-val`);
        if (mainSlider) mainSlider.value = intVal;
        if (mainInput) mainInput.value = intVal;
        
        window.scanlightController.updateColor();
    }
    
    // Always notify mock backend for simulated camera shifts
    fetch('/api/camera/update_mock_leds', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ red: r, green: g, blue: b })
    }).catch(() => {});
}

let isAutoTuningLEDs = false;

async function autoTuneLEDLevels() {
    if (isAutoTuningLEDs) return;
    
    const btn = document.getElementById('btn-auto-tune-leds');
    const statusEl = document.getElementById('auto-tune-status');
    const headroomSelect = document.getElementById('auto-tune-headroom');
    const targetCeiling = headroomSelect ? parseInt(headroomSelect.value, 10) : 242;
    const tolerance = 3;
    
    // Check scanner mode: triplet vs single
    const scannerModeSelect = document.getElementById('scanner-mode');
    const isTripletMode = !scannerModeSelect || scannerModeSelect.value === 'triplet';
    
    // Check if live view is active
    if (!isLiveviewActive) {
        alert("Please enable Live View first so the algorithm can meter real-time film exposure.");
        return;
    }
    
    isAutoTuningLEDs = true;
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span>⏳</span> Tuning LEDs...';
    }
    if (statusEl) {
        statusEl.style.display = 'block';
        statusEl.className = 'auto-tune-status status-active';
        statusEl.textContent = isTripletMode
            ? `Calibrating Triplet Channels sequentially... Target: ${targetCeiling} / 255 (~${Math.round(targetCeiling / 2.55)}%)`
            : `Metering Single-Shot exposure... Target: ${targetCeiling} / 255 (~${Math.round(targetCeiling / 2.55)}%)`;
    }
    
    appendLogLine(`[Auto-Tune] Starting closed-loop LED ETTR calibration (Mode: ${isTripletMode ? 'Triplet (Sequential R->G->B)' : 'Single-Shot (Parallel)'}, Target: ${targetCeiling})...`);
    
    const channelList = [
        { name: 'red', statKey: 'rP99', mask: [1, 0, 0, 0, 0], symbol: 'R' },
        { name: 'green', statKey: 'gP99', mask: [0, 1, 0, 0, 0], symbol: 'G' },
        { name: 'blue', statKey: 'bP99', mask: [0, 0, 1, 0, 0], symbol: 'B' }
    ];
    
    const optimalPWM = {
        red: parseInt(document.getElementById('mini-sl-red')?.value || 255, 10),
        green: parseInt(document.getElementById('mini-sl-green')?.value || 255, 10),
        blue: parseInt(document.getElementById('mini-sl-blue')?.value || 255, 10)
    };
    
    try {
        if (isTripletMode) {
            // === SEQUENTIAL TRIPLET MODE CALIBRATION ===
            // Isolate each channel individually so spectral crosstalk does not skew exposure readings
            for (const chObj of channelList) {
                const ch = chObj.name;
                const statKey = chObj.statKey;
                const symbol = chObj.symbol;
                
                if (statusEl) {
                    statusEl.textContent = `Tuning isolated ${symbol} channel... (Target: ${targetCeiling})`;
                }
                appendLogLine(`[Auto-Tune] Isolating ${symbol} LED (G=0, B=0) for narrowband ETTR calibration...`);
                
                // Turn on ONLY this channel
                applyMiniScanlightPreset(chObj.mask);
                
                const maxPasses = 3;
                for (let pass = 1; pass <= maxPasses; pass++) {
                    await new Promise(r => setTimeout(r, 130));
                    
                    const stats = window.lastHistogramStats;
                    if (!stats) {
                        await new Promise(r => setTimeout(r, 100));
                        continue;
                    }
                    
                    const peak = stats[statKey];
                    const currentVal = optimalPWM[ch];
                    const error = targetCeiling - peak;
                    
                    appendLogLine(`[Auto-Tune] ${symbol} Pass ${pass}: Measured peak=${peak}, current PWM=${currentVal}, error=${error}`);
                    
                    if (Math.abs(error) <= tolerance) {
                        break;
                    }
                    if ((currentVal === 255 && error > 0) || (currentVal === 1 && error < 0)) {
                        break;
                    }
                    
                    let targetRatio;
                    if (peak >= 254) {
                        targetRatio = 0.80;
                    } else {
                        targetRatio = Math.pow(targetCeiling / Math.max(peak, 8), 1.7);
                    }
                    
                    let calculated = Math.round(currentVal * targetRatio);
                    calculated = Math.max(1, Math.min(255, calculated));
                    if (calculated === currentVal) {
                        calculated += (error > 0 ? 1 : -1);
                        calculated = Math.max(1, Math.min(255, calculated));
                    }
                    
                    optimalPWM[ch] = calculated;
                    updateMiniScanlightColor(ch, calculated);
                    // Re-apply single channel mask with new PWM
                    applyMiniScanlightPreset(chObj.mask);
                }
            }
            
            // Re-enable all channels for live view preview
            applyMiniScanlightPreset([1, 1, 1, 0, 0]);
            updateMiniScanlightColor('red', optimalPWM.red);
            updateMiniScanlightColor('green', optimalPWM.green);
            updateMiniScanlightColor('blue', optimalPWM.blue);
            
        } else {
            // === SIMULTANEOUS SINGLE-SHOT MODE CALIBRATION ===
            applyMiniScanlightPreset([1, 1, 1, 0, 0]);
            const maxIterations = 4;
            
            for (let iter = 1; iter <= maxIterations; iter++) {
                await new Promise(r => setTimeout(r, 130));
                const stats = window.lastHistogramStats;
                if (!stats) {
                    await new Promise(r => setTimeout(r, 100));
                    continue;
                }
                
                const currentPeaks = {
                    red: stats.rP99,
                    green: stats.gP99,
                    blue: stats.bP99
                };
                
                appendLogLine(`[Auto-Tune] Pass ${iter}/${maxIterations}: Measured P99 Peaks -> R:${currentPeaks.red}, G:${currentPeaks.green}, B:${currentPeaks.blue}`);
                if (statusEl) {
                    statusEl.textContent = `Pass ${iter}/${maxIterations}: R:${currentPeaks.red} G:${currentPeaks.green} B:${currentPeaks.blue} | Target: ${targetCeiling}`;
                }
                
                let allDone = true;
                for (const chObj of channelList) {
                    const ch = chObj.name;
                    const peak = currentPeaks[ch];
                    const currentVal = optimalPWM[ch];
                    const error = targetCeiling - peak;
                    
                    if (Math.abs(error) <= tolerance) continue;
                    if ((currentVal === 255 && error > 0) || (currentVal === 1 && error < 0)) continue;
                    
                    allDone = false;
                    let targetRatio;
                    if (peak >= 254) {
                        targetRatio = 0.80;
                    } else {
                        targetRatio = Math.pow(targetCeiling / Math.max(peak, 8), 1.7);
                    }
                    let calculated = Math.round(currentVal * targetRatio);
                    calculated = Math.max(1, Math.min(255, calculated));
                    if (calculated === currentVal) {
                        calculated += (error > 0 ? 1 : -1);
                        calculated = Math.max(1, Math.min(255, calculated));
                    }
                    optimalPWM[ch] = calculated;
                }
                
                if (allDone) break;
                
                updateMiniScanlightColor('red', optimalPWM.red);
                updateMiniScanlightColor('green', optimalPWM.green);
                updateMiniScanlightColor('blue', optimalPWM.blue);
            }
        }
        
        // Final UI feedback
        const rVal = optimalPWM.red;
        const gVal = optimalPWM.green;
        const bVal = optimalPWM.blue;
        
        if (statusEl) {
            statusEl.className = 'auto-tune-status status-success';
            const modeLabel = isTripletMode ? 'Triplet Narrowband' : 'Single-Shot';
            statusEl.innerHTML = `✓ <strong>Optimal ${modeLabel} ETTR Locked:</strong> R: ${rVal}, G: ${gVal}, B: ${bVal} (Target: ~${Math.round(targetCeiling / 2.55)}%).`;
        }
        appendLogLine(`[Auto-Tune] Successfully calibrated optimal LED levels for ${isTripletMode ? 'Triplet' : 'Single-Shot'} mode: R=${rVal}, G=${gVal}, B=${bVal}`);
        
    } catch (err) {
        console.error("Auto-tune error:", err);
        if (statusEl) {
            statusEl.className = 'auto-tune-status status-warning';
            statusEl.textContent = `Auto-tune error: ${err.message}`;
        }
    } finally {
        isAutoTuningLEDs = false;
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<span>⚡</span> Auto-Tune Optimal LEDs';
        }
    }
}

function applyMiniScanlightPreset(channels) {
    let r = parseInt(document.getElementById('mini-sl-red')?.value || 255, 10) * (channels[0] || 0);
    let g = parseInt(document.getElementById('mini-sl-green')?.value || 255, 10) * (channels[1] || 0);
    let b = parseInt(document.getElementById('mini-sl-blue')?.value || 255, 10) * (channels[2] || 0);

    if (window.scanlightController) {
        window.scanlightController.setEnabledChannels(channels);
        setTimeout(syncMiniScanlightUI, 50);
        r = window.scanlightController.red * channels[0];
        g = window.scanlightController.green * channels[1];
        b = window.scanlightController.blue * channels[2];
    }
    
    // Always notify mock backend for simulated camera shifts
    fetch('/api/camera/update_mock_leds', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ red: r, green: g, blue: b })
    }).catch(() => {});
}

function runMiniScanlightSequence(sequenceName) {
    if (!window.scanlightController) {
        alert('Scanlight controller script is not loaded.');
        return;
    }
    if (!window.scanlightController.connected) {
        alert('Scanlight is not connected. Please connect the light first.');
        return;
    }
    if (window.scanlightController.isSequenceRunning) {
        // Cancel if already running
        window.scanlightController.isSequenceRunning = false;
        return;
    }
    const btn = document.getElementById('btn-mini-seq-rgb');
    if (btn) {
        btn.textContent = '⏹ Stop Sequence';
        btn.classList.add('btn-danger');
        btn.classList.remove('btn-primary');
    }
    window.scanlightController.runSequence(sequenceName)
        .finally(() => {
            if (btn) {
                btn.textContent = '▶ Auto R → G → B';
                btn.classList.remove('btn-danger');
                btn.classList.add('btn-primary');
            }
            syncMiniScanlightUI();
        });
}

function updateScannerModeUI(value) {
    // Sync to backend config trigger
    appendLogLine(`[Client] Changed capture mode to: ${value}`);
}

function toggleUiSection(section, show) {
    if (section === 'liveview') {
        const card = document.getElementById('camera-liveview-canvas')?.parentElement;
        const liveviewCard = card ? card.closest('.card') : null;
        if (liveviewCard) {
            liveviewCard.style.display = show ? 'block' : 'none';
        }
        // Auto-disable live view streaming if panel is hidden to save resources
        if (!show) {
            const toggle = document.getElementById('camera-liveview-toggle');
            if (toggle && toggle.checked) {
                toggle.checked = false;
                toggleCameraLiveview(false);
            }
        }
    } else {
        let cardId = '';
        if (section === 'camera') cardId = 'card-camera-exposure';
        else if (section === 'focus') cardId = 'card-focus-assist';
        else if (section === 'light') cardId = 'card-light-source';
        
        const card = document.getElementById(cardId);
        if (card) {
            card.style.display = show ? 'block' : 'none';
        }
    }
    
    // Check all checkbox states
    const showLiveview = document.getElementById('ui-show-liveview') ? document.getElementById('ui-show-liveview').checked : true;
    const showCamera = document.getElementById('ui-show-camera') ? document.getElementById('ui-show-camera').checked : true;
    const showFocus = document.getElementById('ui-show-focus') ? document.getElementById('ui-show-focus').checked : true;
    const showLight = document.getElementById('ui-show-light') ? document.getElementById('ui-show-light').checked : true;
    
    const colLiveview = document.getElementById('scanner-col-liveview');
    const colControls = document.getElementById('scanner-col-controls');
    const grid = document.querySelector('.grid-layout-3');
    
    // Column 1 (Liveview + Focus) visibility
    const showColLiveview = showLiveview || showFocus;
    if (colLiveview) {
        colLiveview.style.display = showColLiveview ? 'flex' : 'none';
    }
    
    // Column 2 (Light + Camera) visibility
    const showColControls = showLight || showCamera;
    if (colControls) {
        colControls.style.display = showColControls ? 'flex' : 'none';
    }
    
    // Dynamically calculate grid columns based on active panels
    if (grid) {
        if (showColLiveview && showColControls) {
            grid.style.gridTemplateColumns = '1.3fr 0.95fr 0.95fr';
        } else if (showColLiveview && !showColControls) {
            grid.style.gridTemplateColumns = '1.3fr 0.95fr';
        } else if (!showColLiveview && showColControls) {
            grid.style.gridTemplateColumns = '1fr 1fr';
        } else {
            grid.style.gridTemplateColumns = '1fr';
        }
    }
    
    localStorage.setItem(`show-${section}-controls`, show);
    
    // Trigger window resize event so the margin overlay bounds recalculate based on new column sizes
    window.dispatchEvent(new Event('resize'));
}

// --- Focus Assist Helper Functions ---
function toggleFocusPeaking(active) {
    focusPeakingActive = active;
    const sliderGroup = document.getElementById('peaking-sensitivity-group');
    if (sliderGroup) {
        sliderGroup.style.display = active ? 'block' : 'none';
    }
}

function updateFocusPeakingThreshold(val) {
    peakingThreshold = parseInt(val);
}

function applyFocusPeaking(canvas, ctx) {
    const width = canvas.width;
    const height = canvas.height;
    
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;
    
    const output = ctx.createImageData(width, height);
    const outData = output.data;
    outData.set(data);
    
    const threshold = peakingThreshold;
    
    // Quick horizontal difference + vertical difference edge check to maximize performance:
    for (let y = 0; y < height - 1; y += 2) {
        for (let x = 0; x < width - 1; x += 2) {
            const idx = (y * width + x) * 4;
            
            const r1 = data[idx], g1 = data[idx+1], b1 = data[idx+2];
            const lum1 = (r1 * 77 + g1 * 150 + b1 * 29) >> 8;
            
            const idxRight = idx + 8;
            const r2 = data[idxRight], g2 = data[idxRight+1], b2 = data[idxRight+2];
            const lum2 = (r2 * 77 + g2 * 150 + b2 * 29) >> 8;
            
            const idxDown = idx + (width * 8);
            const r3 = data[idxDown], g3 = data[idxDown+1], b3 = data[idxDown+2];
            const lum3 = (r3 * 77 + g3 * 150 + b3 * 29) >> 8;
            
            const diff = Math.abs(lum1 - lum2) + Math.abs(lum1 - lum3);
            
            if (diff > threshold) {
                // Color the 2x2 block neon green
                for (let dy = 0; dy < 2; dy++) {
                    for (let dx = 0; dx < 2; dx++) {
                        const targetIdx = ((y + dy) * width + (x + dx)) * 4;
                        if (targetIdx < outData.length) {
                            outData[targetIdx] = 0;      // R
                            outData[targetIdx+1] = 255;  // G
                            outData[targetIdx+2] = 0;    // B
                        }
                    }
                }
            }
        }
    }
    ctx.putImageData(output, 0, 0);
}

function driveFocus(direction, speed) {
    appendLogLine(`[Client] Stepping focus ${direction} (speed ${speed})...`);
    fetch('/api/camera/focus_step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ direction: direction, speed: speed })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            appendLogLine(`[Client] Focus adjusted successfully.`);
        } else {
            appendLogLine(`[Client] Focus step failed: ${data.message || 'unknown error'}`);
        }
    })
    .catch(err => console.error("Error stepping focus:", err));
}

function triggerAutofocus() {
    const btn = document.getElementById('btn-trigger-af');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Focusing...';
    }
    appendLogLine(`[Client] Triggering camera autofocus sequence...`);
    fetch('/api/camera/autofocus', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            appendLogLine(`[Client] Autofocus lock complete.`);
        } else {
            appendLogLine(`[Client] Autofocus trigger failed: ${data.message || 'unknown error'}`);
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

function reconnectCameraDevice() {
    const btn = document.getElementById('btn-camera-reconnect');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Connecting...';
    }
    
    // Auto-disable live view before reconnecting to clear preview handles
    const toggle = document.getElementById('camera-liveview-toggle');
    if (toggle && toggle.checked) {
        toggle.checked = false;
        toggleCameraLiveview(false);
    }
    
    appendLogLine(`[Client] Re-initializing connection to physical camera...`);
    fetch('/api/camera/reconnect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            appendLogLine(`[Client] Reconnect query dispatched successfully.`);
            // Trigger an immediate camera status refresh
            setTimeout(fetchCameraStatus, 500);
        } else {
            appendLogLine(`[Client] Reconnect failed: ${data.message || 'unknown error'}`);
            alert(`Reconnect failed: ${data.message}`);
        }
    })
    .catch(err => {
        console.error("Error reconnecting camera:", err);
        appendLogLine(`[Client] Network error during camera reconnect command.`);
    })
    .finally(() => {
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Reconnect';
        }
    });
}

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    if (!sidebar) return;
    
    const isCollapsed = sidebar.classList.toggle('collapsed');
    
    // Update toggle icon
    const toggleIcon = document.querySelector('#btn-sidebar-toggle .toggle-icon');
    if (toggleIcon) {
        toggleIcon.textContent = isCollapsed ? '▶' : '◀';
    }
    
    // Save state to localStorage
    localStorage.setItem('sidebar-collapsed', isCollapsed);
    
    // Trigger window resize so canvas crop guide re-aligns
    setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
    }, 250); // wait for CSS width transition to complete
}

function toggleLiveviewRotation() {
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
    
    // Repoll immediate frame to show updated rotation if active
    if (isLiveviewActive) {
        pollLiveviewFrame();
    }
}





