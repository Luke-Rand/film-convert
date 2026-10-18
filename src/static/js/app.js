// ============================================================
// FILMCONVERT MAIN APPLICATION ENTRY MODULE
// ============================================================

import {
    setHistogramMode,
    syncMiniScanlightUI,
} from './histogram.js';

import {
    refreshFiles,
    openLightbox,
} from './gallery.js';

import {
    initCameraUI,
    toggleCameraLiveview,
    stopCameraStatusPolling,
    setActiveFilmBaseState,
    updateFilmBaseBadgeUI,
} from './liveview.js';

import {
    initKeyboardShortcuts,
} from './shortcuts.js';

// Global State
let currentTab = 'scanner';
let systemStatus = 'idle';
window.systemStatus = 'idle';
let activeSessionDirs = {};
let eventSource = null;
let lastFrameData = null;
let lastFrameViewMode = 'positive'; // 'positive' | 'negative'

// ============================================================
// TOOLTIP ENGINE
// Uses data-tooltip="..." attribute for content.
// A single floating #fc-tooltip div is reused for all tips.
// ============================================================

let _tipEl = null;
let _tipShowTimer = null;
let _tipHideTimer = null;
let _tipCurrentEl = null;

const TOOLTIP_SHOW_DELAY = 600;
const TOOLTIP_MARGIN = 10;
const TOOLTIP_OFFSET_Y = 8;
const TOOLTIP_HIDE_DELAY = 80;

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
    const t = _getTipEl();
    const rect = anchorEl.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const tipW = t.offsetWidth || 240;
    const tipH = t.offsetHeight || 48;

    let top = rect.bottom + TOOLTIP_OFFSET_Y;
    let left = rect.left;

    if (top + tipH > vh - TOOLTIP_MARGIN) {
        top = rect.top - tipH - TOOLTIP_OFFSET_Y;
    }
    left = Math.min(left, vw - tipW - TOOLTIP_MARGIN);
    left = Math.max(left, TOOLTIP_MARGIN);
    top = Math.max(top, TOOLTIP_MARGIN);

    t.style.left = left + 'px';
    t.style.top = top + 'px';
}

function _showTip(el) {
    if (document.body.classList.contains('tooltips-disabled')) return;
    const text = el.dataset.tooltip;
    if (!text) return;

    const t = _getTipEl();
    t.textContent = text;

    t.style.removeProperty('left');
    t.style.removeProperty('top');
    t.classList.remove('fc-tooltip-visible');
    t.style.display = 'block';

    requestAnimationFrame(() => {
        _positionTip(el);
        requestAnimationFrame(() => {
            t.classList.add('fc-tooltip-visible');
        });
    });
}

function _hideTip() {
    const t = _getTipEl();
    t.classList.remove('fc-tooltip-visible');
    t.addEventListener('transitionend', function onEnd() {
        t.removeEventListener('transitionend', onEnd);
        if (!t.classList.contains('fc-tooltip-visible')) {
            t.style.display = 'none';
        }
    }, { once: true });
}

function _initTooltipListeners() {
    const t = _getTipEl();
    t.style.display = 'none';

    document.addEventListener('mouseover', (e) => {
        const el = e.target.closest('[data-tooltip]');
        if (!el) return;

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

        const related = e.relatedTarget;
        if (related && el.contains(related)) return;

        clearTimeout(_tipShowTimer);
        _tipCurrentEl = null;
        _tipHideTimer = setTimeout(_hideTip, TOOLTIP_HIDE_DELAY);
    });

    document.addEventListener('scroll', () => {
        clearTimeout(_tipShowTimer);
        _hideTip();
    }, { capture: true, passive: true });

    document.addEventListener('mousedown', () => {
        clearTimeout(_tipShowTimer);
        _hideTip();
    }, { passive: true });
}

export function toggleTooltips() {
    const disabled = document.body.classList.toggle('tooltips-disabled');
    localStorage.setItem('tooltips-disabled', disabled ? 'true' : 'false');
    if (disabled) _hideTip();
    const btn = document.getElementById('btn-tooltip-toggle');
    const label = btn ? btn.querySelector('.tt-label') : null;
    if (btn) btn.classList.toggle('active', !disabled);
    if (label) label.textContent = disabled ? 'Tooltips Off' : 'Tooltips On';
}

// ============================================================
// INITIALIZATION & SSE
// ============================================================

function initApp() {
    _initTooltipListeners();
    initKeyboardShortcuts();

    const rootInput = document.getElementById('scanner-root-dir');
    if (rootInput && !rootInput.value) {
        rootInput.value = '~/Pictures/Scans';
    }

    fetchInitialLogs();
    fetchInitialStatus();
    connectSSE();
    initCameraUI();

    const showLiveview = localStorage.getItem('show-liveview-controls') !== 'false';
    const showCamera = localStorage.getItem('show-camera-controls') !== 'false';
    const showLight = localStorage.getItem('show-light-controls') !== 'false';

    const liveviewCheckbox = document.getElementById('ui-show-liveview');
    const camCheckbox = document.getElementById('ui-show-camera');
    const lightCheckbox = document.getElementById('ui-show-light');

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

    const savedHistMode = localStorage.getItem('histogram-mode') || 'all';
    setHistogramMode(savedHistMode);

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
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}

export function connectSSE() {
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

    eventSource.addEventListener('frame_processed', (event) => {
        try {
            const data = JSON.parse(event.data);
            updateLastFrameCard(data, true);
            if (currentTab === 'gallery') {
                refreshFiles();
            }
        } catch (err) {
            console.error("Error parsing frame_processed SSE data:", err);
        }
    });

    eventSource.addEventListener('contact_sheet_generated', (event) => {
        try {
            const data = JSON.parse(event.data);
            appendLogLine(`[Archival Contact Sheet] ${data.message}`);
            refreshFiles();
        } catch (err) {
            console.error("Error parsing contact_sheet_generated SSE data:", err);
        }
    });

    eventSource.onerror = (err) => {
        console.error("SSE connection error:", err);
        eventSource.close();
        setTimeout(connectSSE, 5000);
    };
}

export function fetchInitialStatus() {
    fetch('/api/status')
        .then(res => res.json())
        .then(data => {
            if (data && data.status) {
                systemStatus = data.status;
                window.systemStatus = data.status;
                activeSessionDirs = data.dirs || {};
                updateStatusUI(data);
            }
        })
        .catch(err => console.error("Error fetching initial status:", err));
}

export function fetchInitialLogs() {
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

// ============================================================
// NAVIGATION & UI CONTROLS
// ============================================================

export function switchTab(tabId) {
    currentTab = tabId;

    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    const activeBtn = document.getElementById(`nav-${tabId}-btn`);
    if (activeBtn) activeBtn.classList.add('active');

    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.remove('active');
    });
    const activePanel = document.getElementById(`panel-${tabId}`);
    if (activePanel) activePanel.classList.add('active');

    if (tabId === 'gallery') {
        refreshFiles();
    }

    if (tabId === 'scanner') {
        initCameraUI();
    } else {
        const toggle = document.getElementById('camera-liveview-toggle');
        if (toggle && toggle.checked) {
            toggle.checked = false;
            toggleCameraLiveview(false);
        }
        stopCameraStatusPolling();
    }
}

// Consolidated slider value display handler (supports both scanner and batch sliders)
export function updateSliderVal(id) {
    const sliderId = id.startsWith('batch-') ? id : `config-${id}`;
    const slider = document.getElementById(sliderId);
    const valSpan = document.getElementById(`val-${id}`);
    if (slider && valSpan) {
        let val = slider.value;
        if (id === 'margin' || id === 'batch-margin') {
            val = Math.round(parseFloat(val) * 100) + '%';
        }
        valSpan.textContent = val;
    }
}

export function toggleBatchSettings() {
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

export function toggleScannerSettings() {
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

export function updateBatchTaskDesc() {
    // Visual helper hook inside index.html templates
}

export function updateScannerModeUI(value) {
    appendLogLine(`[Client] Changed capture mode to: ${value}`);
}

export function toggleUiSection(section, show) {
    if (section === 'liveview') {
        const card = document.getElementById('camera-liveview-canvas')?.parentElement;
        const liveviewCard = card ? card.closest('.card') : null;
        if (liveviewCard) {
            liveviewCard.style.display = show ? 'block' : 'none';
        }
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

    const showLiveview = document.getElementById('ui-show-liveview') ? document.getElementById('ui-show-liveview').checked : true;
    const showCamera = document.getElementById('ui-show-camera') ? document.getElementById('ui-show-camera').checked : true;
    const showFocus = document.getElementById('ui-show-focus') ? document.getElementById('ui-show-focus').checked : true;
    const showLight = document.getElementById('ui-show-light') ? document.getElementById('ui-show-light').checked : true;

    const colLiveview = document.getElementById('scanner-col-liveview');
    const colControls = document.getElementById('scanner-col-controls');
    const grid = document.querySelector('.grid-layout-3');

    const showColLiveview = showLiveview || showFocus;
    if (colLiveview) {
        colLiveview.style.display = showColLiveview ? 'flex' : 'none';
    }

    const showColControls = showLight || showCamera;
    if (colControls) {
        colControls.style.display = showColControls ? 'flex' : 'none';
    }

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
    window.dispatchEvent(new Event('resize'));
}

export function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    if (!sidebar) return;

    const isCollapsed = sidebar.classList.toggle('collapsed');

    const toggleIcon = document.querySelector('#btn-sidebar-toggle .toggle-icon');
    if (toggleIcon) {
        toggleIcon.textContent = isCollapsed ? '▶' : '◀';
    }

    localStorage.setItem('sidebar-collapsed', isCollapsed);

    setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
    }, 250);
}

// ============================================================
// SESSION & BATCH CONTROL
// ============================================================

export function updateStatusUI(data) {
    const dot = document.getElementById('summary-status-dot');
    const text = document.getElementById('summary-status-text');
    if (dot) dot.className = `status-dot ${data.status}`;

    let statusText = "System Idle";
    if (data.status === 'monitoring') statusText = "Scanner Active";
    if (data.status === 'batch_processing') statusText = "Running Batch Task";
    if (text) text.textContent = statusText;

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
        if (document.activeElement.tagName !== 'INPUT') {
            if (data.root_folder) {
                const rootEl = document.getElementById('scanner-root-dir');
                if (rootEl) rootEl.value = data.root_folder;
            }
            if (data.stock) {
                const stockEl = document.getElementById('scanner-stock');
                if (stockEl) stockEl.value = data.stock;
            }
            if (data.format) {
                const fmtEl = document.getElementById('scanner-format');
                if (fmtEl) fmtEl.value = data.format;
            }
            if (data.roll) {
                const rollEl = document.getElementById('scanner-roll');
                if (rollEl) rollEl.value = data.roll;
            }
            if (data.mode) {
                const modeEl = document.getElementById('scanner-mode');
                if (modeEl) modeEl.value = data.mode;
            }
        }
        if (data.sampled_rgb !== undefined) {
            window.activeFilmBaseRGB = data.sampled_rgb;
        }
        if (data.config) {
            syncConfigToUI(data.config);
        }
    }

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

    if (data.last_frame !== undefined) {
        updateLastFrameCard(data.last_frame, false);
    }

    syncMiniScanlightUI();
}

export function updateLastFrameCard(frameInfo, flash = false) {
    lastFrameData = frameInfo || null;
    window.lastFrameData = lastFrameData;

    const card = document.getElementById('card-last-frame');
    const badge = document.getElementById('last-frame-badge');
    const toggleGroup = document.getElementById('last-frame-mode-toggle');
    const btnPos = document.getElementById('btn-lf-positive');
    const btnNeg = document.getElementById('btn-lf-negative');
    const imgEl = document.getElementById('last-frame-img');
    const emptyEl = document.getElementById('last-frame-empty');
    const footerEl = document.getElementById('last-frame-footer');
    const filenameEl = document.getElementById('last-frame-filename');

    if (!card || !imgEl || !emptyEl) return;

    if (!lastFrameData || !lastFrameData.positive_path) {
        if (badge) {
            badge.textContent = 'No frames yet';
            badge.classList.remove('active');
        }
        if (toggleGroup) toggleGroup.style.display = 'none';
        imgEl.style.display = 'none';
        imgEl.src = '';
        emptyEl.style.display = 'flex';
        if (footerEl) footerEl.style.display = 'none';
        return;
    }

    const frameNumStr = String(lastFrameData.frame_number || 1).padStart(2, '0');
    const totalCount = lastFrameData.frame_count || lastFrameData.frame_number || 1;
    if (badge) {
        badge.textContent = `Frame #${frameNumStr} (${totalCount} total)`;
        badge.classList.add('active');
    }

    const hasNegative = Boolean(lastFrameData.negative_path);
    if (toggleGroup) {
        toggleGroup.style.display = hasNegative ? 'flex' : 'none';
    }
    if (!hasNegative && lastFrameViewMode === 'negative') {
        lastFrameViewMode = 'positive';
    }

    if (btnPos) btnPos.classList.toggle('active', lastFrameViewMode === 'positive');
    if (btnNeg) btnNeg.classList.toggle('active', lastFrameViewMode === 'negative');

    const activePath = (lastFrameViewMode === 'negative' && hasNegative)
        ? lastFrameData.negative_path
        : lastFrameData.positive_path;
    const activeName = (lastFrameViewMode === 'negative' && hasNegative)
        ? (lastFrameData.negative_name || 'Negative')
        : (lastFrameData.positive_name || 'Positive');

    const cacheBuster = lastFrameData.mtime ? `&t=${Math.floor(lastFrameData.mtime * 1000)}` : '';
    imgEl.src = `/api/preview?path=${encodeURIComponent(activePath)}&w=600${cacheBuster}`;
    imgEl.style.display = 'block';
    emptyEl.style.display = 'none';

    if (footerEl) footerEl.style.display = 'flex';
    if (filenameEl) {
        filenameEl.textContent = activeName;
        filenameEl.title = activePath;
    }

    if (flash) {
        card.classList.remove('frame-flash');
        void card.offsetWidth;
        card.classList.add('frame-flash');
        setTimeout(() => card.classList.remove('frame-flash'), 1200);
    }
}

export function setLastFrameViewMode(mode) {
    if (mode !== 'positive' && mode !== 'negative') return;
    lastFrameViewMode = mode;
    updateLastFrameCard(lastFrameData, false);
}

export function openLastFrameInLightbox() {
    if (!lastFrameData || !lastFrameData.positive_path) return;
    const hasNegative = Boolean(lastFrameData.negative_path);
    const useNeg = (lastFrameViewMode === 'negative' && hasNegative);
    const path = useNeg ? lastFrameData.negative_path : lastFrameData.positive_path;
    const name = useNeg ? (lastFrameData.negative_name || 'Negative') : (lastFrameData.positive_name || 'Positive');
    const size = useNeg ? (lastFrameData.negative_size || 0) : (lastFrameData.positive_size || 0);
    openLightbox(name, path, size);
}

export function disableInputs(disabled) {
    document.getElementById('scanner-root-dir').disabled = disabled;
    document.getElementById('scanner-stock').disabled = disabled;
    document.getElementById('scanner-format').disabled = disabled;
    document.getElementById('scanner-roll').disabled = disabled;
    const scannerModeEl = document.getElementById('scanner-mode');
    if (scannerModeEl) scannerModeEl.disabled = disabled;
    const loadSessionBtn = document.getElementById('btn-load-session');
    if (loadSessionBtn) loadSessionBtn.disabled = disabled;

    document.getElementById('config-gamma').disabled = disabled;
    document.getElementById('config-clip').disabled = disabled;
    document.getElementById('config-scurve').disabled = disabled;
    document.getElementById('config-margin').disabled = disabled;

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

export function syncConfigToUI(config) {
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

    const isMono = config.monochrome || false;
    document.getElementById('config-monochrome').checked = isMono;
    document.getElementById('config-monochrome-channel').value = config.monochrome_channel || 'luminance';
    const monoGroup = document.getElementById('mini-mono-channel-group');
    if (monoGroup) monoGroup.style.display = isMono ? 'block' : 'none';

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

    if (config.base_ratios && Array.isArray(config.base_ratios) && config.base_ratios.length === 3) {
        setActiveFilmBaseState(config.base_ratios, window.activeFilmBaseRGB);
        updateFilmBaseBadgeUI(config.base_ratios, window.activeFilmBaseRGB);
    } else if (config.base_ratios === null) {
        setActiveFilmBaseState(null, null);
        updateFilmBaseBadgeUI(null, null);
    }

    document.getElementById('val-gamma').textContent = config.gamma;
    document.getElementById('val-clip').textContent = config.clip;
    document.getElementById('val-scurve').textContent = config.scurve;
    document.getElementById('val-margin').textContent = Math.round(config.margin * 100) + '%';
}

let isTogglingMonitor = false;

export function toggleMonitor() {
    if (isTogglingMonitor) return;

    const toggleBtn = document.getElementById('btn-toggle-monitor');

    if (systemStatus === 'monitoring') {
        isTogglingMonitor = true;
        if (toggleBtn) toggleBtn.disabled = true;
        fetch('/api/stop', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                if (data && data.status_data) {
                    systemStatus = data.status_data.status;
                    window.systemStatus = data.status_data.status;
                    activeSessionDirs = data.status_data.dirs || {};
                    updateStatusUI(data.status_data);
                } else {
                    fetchInitialStatus();
                }
                appendLogLine(`[Client] Stop request sent: ${data.message}`);
            })
            .catch(err => {
                appendLogLine(`[Client Error] Failed to stop: ${err}`);
                fetchInitialStatus();
            })
            .finally(() => {
                isTogglingMonitor = false;
                if (toggleBtn) toggleBtn.disabled = false;
            });
    } else {
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
            base_ratios: window.activeFilmBaseRatios,
            compress_tiff: document.getElementById('config-compress').checked,
            align_channels: document.getElementById('config-align-channels').checked,
            monochrome: document.getElementById('config-monochrome').checked,
            monochrome_channel: document.getElementById('config-monochrome-channel').value,
            reversal: document.getElementById('config-reversal') ? document.getElementById('config-reversal').checked : false,
            convert_to_tiff: document.getElementById('config-convert-tiff') ? document.getElementById('config-convert-tiff').checked : true,
            color_profile: document.getElementById('config-color-profile') ? document.getElementById('config-color-profile').value : 'adobe_rgb',
            embed_metadata: document.getElementById('config-embed-metadata') ? document.getElementById('config-embed-metadata').checked : true,
            auto_contact_sheet: document.getElementById('config-auto-contact-sheet') ? document.getElementById('config-auto-contact-sheet').checked : true
        };

        const payload = {
            root_dir: rootDir,
            stock: stock,
            format: format,
            roll: roll,
            mode: mode,
            config: config
        };

        isTogglingMonitor = true;
        if (toggleBtn) toggleBtn.disabled = true;

        fetch('/api/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(async res => {
            const data = await res.json().catch(() => null);
            if (data && data.status_data) {
                systemStatus = data.status_data.status;
                window.systemStatus = data.status_data.status;
                activeSessionDirs = data.status_data.dirs || {};
                updateStatusUI(data.status_data);
            } else {
                fetchInitialStatus();
            }
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
        .catch(err => {
            appendLogLine(`[Client Error] Failed to start: ${err}`);
            fetchInitialStatus();
        })
        .finally(() => {
            isTogglingMonitor = false;
            if (toggleBtn) toggleBtn.disabled = false;
        });
    }
}

export function appendLogLine(line) {
    const consoleOut = document.getElementById('console-output');
    const batchConsoleOut = document.getElementById('batch-console-output');
    const scanlightConsoleOut = document.getElementById('scanlight-console-output');

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

export function appendLineToTarget(target, line) {
    const div = document.createElement('div');
    div.className = 'console-line';

    if (line.includes('SUCCESS') || line.includes('completed') || line.includes('complete')) {
        div.classList.add('text-success');
    } else if (line.includes('ERROR') || line.includes('Failed') || line.includes('!!!!!!!!!!')) {
        div.classList.add('text-error');
    } else if (line.includes('WARNING') || line.includes('anomaly')) {
        div.classList.add('text-warning');
    } else if (line.startsWith('[Client]') || line.startsWith('[Scanlight]')) {
        div.classList.add('text-muted');
    }

    div.textContent = line;
    target.appendChild(div);
    target.scrollTop = target.scrollHeight;
}

export function clearLogs() {
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

export function runBatchJob() {
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
        base_ratios: window.activeFilmBaseRatios,
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

// Expose to window for inline HTML handlers and cross-module access
Object.assign(window, {
    toggleTooltips,
    connectSSE,
    fetchInitialStatus,
    fetchInitialLogs,
    switchTab,
    updateSliderVal,
    toggleBatchSettings,
    toggleScannerSettings,
    updateBatchTaskDesc,
    updateScannerModeUI,
    toggleUiSection,
    toggleSidebar,
    updateStatusUI,
    updateLastFrameCard,
    setLastFrameViewMode,
    openLastFrameInLightbox,
    disableInputs,
    syncConfigToUI,
    toggleMonitor,
    appendLogLine,
    appendLineToTarget,
    clearLogs,
    runBatchJob,
});
