// ============================================================
// HISTOGRAM, ETTR AUTO-TUNE & MINI SCANLIGHT MODULE
// ============================================================

export let activeHistogramMode = localStorage.getItem('histogram-mode') || 'all';
export let isAutoTuningLEDs = false;

let histogramCanvas = null;
let histogramCtx = null;

export function initHistogramCanvas() {
    histogramCanvas = document.getElementById('camera-histogram-canvas');
    if (histogramCanvas) {
        histogramCtx = histogramCanvas.getContext('2d');
    }
}

export function computeAndRenderHistogram(offscreenCtx, sampleW, sampleH) {
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
}

export function renderRGBHistogram(rHist, gHist, bHist, stats = null) {
    if (!histogramCanvas) {
        initHistogramCanvas();
    }
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

        const activeArrays = channels.map(c => c.data);
        if (activeArrays.length > 0) {
            const merged = [].concat(...activeArrays);
            maxVal = Math.max(...merged) || 1;
        }

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

            histogramCtx.fillStyle = ch.fill;
            histogramCtx.fill();

            histogramCtx.lineWidth = 1.25;
            histogramCtx.strokeStyle = ch.stroke;
            histogramCtx.stroke();
        });

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

export function setHistogramMode(mode) {
    activeHistogramMode = mode;
    window.activeHistogramMode = mode;

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

export function cycleHistogramMode() {
    const modes = ['all', 'r', 'g', 'b', 'split'];
    const currentMode = activeHistogramMode || 'all';
    const currentIdx = modes.indexOf(currentMode);
    const nextMode = modes[(currentIdx + 1) % modes.length];
    setHistogramMode(nextMode);
}

export function clearHistogramCanvas() {
    if (histogramCtx && histogramCanvas) {
        histogramCtx.clearRect(0, 0, histogramCanvas.width, histogramCanvas.height);
    }
}

// Sync Mini-Scanlight UI with active global ScanlightController
export function syncMiniScanlightUI() {
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

export function toggleMiniScanlightConnection() {
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

export function updateMiniScanlightColor(channel, value) {
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

        const mainSlider = document.getElementById(`scanlight-${channel}-slider`);
        const mainInput = document.getElementById(`scanlight-${channel}-val`);
        if (mainSlider) mainSlider.value = intVal;
        if (mainInput) mainInput.value = intVal;

        window.scanlightController.updateColor();
    }

    fetch('/api/camera/update_mock_leds', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ red: r, green: g, blue: b })
    }).catch(() => {});
}

export function applyMiniScanlightPreset(channels) {
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

    fetch('/api/camera/update_mock_leds', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ red: r, green: g, blue: b })
    }).catch(() => {});
}

export function runMiniScanlightSequence(sequenceName) {
    if (!window.scanlightController) {
        alert('Scanlight controller script is not loaded.');
        return;
    }
    if (!window.scanlightController.connected) {
        alert('Scanlight is not connected. Please connect the light first.');
        return;
    }
    if (window.scanlightController.isSequenceRunning) {
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

export async function autoTuneLEDLevels() {
    if (isAutoTuningLEDs) return;

    const btn = document.getElementById('btn-auto-tune-leds');
    const statusEl = document.getElementById('auto-tune-status');
    const headroomSelect = document.getElementById('auto-tune-headroom');
    const targetCeiling = headroomSelect ? parseInt(headroomSelect.value, 10) : 242;
    const tolerance = 3;

    const scannerModeSelect = document.getElementById('scanner-mode');
    const isTripletMode = !scannerModeSelect || scannerModeSelect.value === 'triplet';

    if (!window.isLiveviewActive) {
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

    window.appendLogLine?.(`[Auto-Tune] Starting closed-loop LED ETTR calibration (Mode: ${isTripletMode ? 'Triplet (Sequential R->G->B)' : 'Single-Shot (Parallel)'}, Target: ${targetCeiling})...`);

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
            for (const chObj of channelList) {
                const ch = chObj.name;
                const statKey = chObj.statKey;
                const symbol = chObj.symbol;

                if (statusEl) {
                    statusEl.textContent = `Tuning isolated ${symbol} channel... (Target: ${targetCeiling})`;
                }
                window.appendLogLine?.(`[Auto-Tune] Isolating ${symbol} LED (G=0, B=0) for narrowband ETTR calibration...`);

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

                    window.appendLogLine?.(`[Auto-Tune] ${symbol} Pass ${pass}: Measured peak=${peak}, current PWM=${currentVal}, error=${error}`);

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
                    applyMiniScanlightPreset(chObj.mask);
                }
            }

            applyMiniScanlightPreset([1, 1, 1, 0, 0]);
            updateMiniScanlightColor('red', optimalPWM.red);
            updateMiniScanlightColor('green', optimalPWM.green);
            updateMiniScanlightColor('blue', optimalPWM.blue);

        } else {
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

                window.appendLogLine?.(`[Auto-Tune] Pass ${iter}/${maxIterations}: Measured P99 Peaks -> R:${currentPeaks.red}, G:${currentPeaks.green}, B:${currentPeaks.blue}`);
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

        const rVal = optimalPWM.red;
        const gVal = optimalPWM.green;
        const bVal = optimalPWM.blue;

        if (statusEl) {
            statusEl.className = 'auto-tune-status status-success';
            const modeLabel = isTripletMode ? 'Triplet Narrowband' : 'Single-Shot';
            statusEl.innerHTML = `✓ <strong>Optimal ${modeLabel} ETTR Locked:</strong> R: ${rVal}, G: ${gVal}, B: ${bVal} (Target: ~${Math.round(targetCeiling / 2.55)}%).`;
        }
        window.appendLogLine?.(`[Auto-Tune] Successfully calibrated optimal LED levels for ${isTripletMode ? 'Triplet' : 'Single-Shot'} mode: R=${rVal}, G=${gVal}, B=${bVal}`);

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

// Expose to window for inline HTML handlers
Object.assign(window, {
    renderRGBHistogram,
    setHistogramMode,
    cycleHistogramMode,
    clearHistogramCanvas,
    syncMiniScanlightUI,
    toggleMiniScanlightConnection,
    updateMiniScanlightColor,
    applyMiniScanlightPreset,
    runMiniScanlightSequence,
    autoTuneLEDLevels,
});
