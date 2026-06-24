// Global State
let currentTab = 'scanner';
let systemStatus = 'idle';
window.systemStatus = 'idle';
let activeSessionDirs = {};
let eventSource = null;

// On Load Initialization
document.addEventListener('DOMContentLoaded', () => {
    // Set default scanner path
    const rootInput = document.getElementById('scanner-root-dir');
    if (rootInput && !rootInput.value) {
        rootInput.value = '~/Pictures/Scans';
    }

    // Load initial logs
    fetchInitialLogs();
    
    // Connect to real-time event stream
    connectSSE();
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
}

// Freeze forms when running
function disableInputs(disabled) {
    document.getElementById('scanner-root-dir').disabled = disabled;
    document.getElementById('scanner-stock').disabled = disabled;
    document.getElementById('scanner-format').disabled = disabled;
    document.getElementById('scanner-roll').disabled = disabled;
    document.getElementsByName('scanner-mode').forEach(rad => rad.disabled = disabled);
    
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
        
        let mode = 'triplet';
        document.getElementsByName('scanner-mode').forEach(rad => {
            if (rad.checked) mode = rad.value;
        });

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
            compress_tiff: document.getElementById('config-compress').checked,
            align_channels: document.getElementById('config-align-channels').checked
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
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                appendLogLine(`[Client] Session monitor initialized successfully.`);
            } else {
                alert(`Error starting scanner: ${data.message}`);
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
        compress_tiff: document.getElementById('batch-compress').checked,
        align_channels: document.getElementById('batch-align-channels').checked
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
    
    // Clear preview image while loading new one to avoid layout jumps
    img.src = '';
    title.textContent = name;
    desc.textContent = `Size: ${formatBytes(size)} | Path: ${path}`;
    
    // Set dynamic source link
    dlLink.href = `/api/preview?path=${encodeURIComponent(path)}`;
    
    box.classList.add('active');
    
    // Load high-resolution preview (800px width limit for responsive viewport loading)
    img.src = `/api/preview?path=${encodeURIComponent(path)}&w=900`;
}

function closeLightbox() {
    const box = document.getElementById('image-lightbox');
    if (box) box.classList.remove('active');
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
