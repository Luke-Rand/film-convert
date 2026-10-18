// ============================================================
// GALLERY, CONTACT SHEET, LIGHTBOX & FOLDER BROWSER MODULE
// ============================================================

export let currentLightboxPath = '';

let browserActiveInputId = '';
let browserCurrentPath = '';
let browserSelectedPath = '';
let browserAllFolders = [];

export function formatBytes(bytes, decimals = 1) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

export function refreshFiles() {
    const grid = document.getElementById('gallery-grid');
    const countInfo = document.getElementById('gallery-count-info');
    const csBanner = document.getElementById('contact-sheet-banner');
    const csActions = document.getElementById('cs-banner-actions');

    if (!grid) return;

    grid.innerHTML = '<div class="empty-gallery-msg">Loading session files...</div>';

    fetch('/api/files')
        .then(res => res.json())
        .then(data => {
            if (!data.success) {
                grid.innerHTML = `<div class="empty-gallery-msg">${data.message || 'No active session. Please start monitoring or run a batch task.'}</div>`;
                countInfo.textContent = '0 Positive Frames found in session';
                if (csBanner) csBanner.style.display = 'none';
                return;
            }

            const positives = data.positives || [];
            const contactSheets = data.contact_sheets || [];
            countInfo.textContent = `${positives.length} Positive Frame(s) found in session`;

            if (csBanner && csActions) {
                if (contactSheets.length > 0) {
                    csBanner.style.display = 'flex';
                    csActions.innerHTML = '';

                    const pdfs = contactSheets.filter(cs => cs.type === 'pdf');
                    const jpegs = contactSheets.filter(cs => cs.type === 'jpeg');

                    if (pdfs.length > 0) {
                        const pdf = pdfs[0];
                        const btnDl = document.createElement('a');
                        btnDl.className = 'btn btn-primary btn-small';
                        btnDl.href = `/api/contact_sheet/download?path=${encodeURIComponent(pdf.path)}&download=1`;
                        btnDl.innerHTML = `<span>📥</span> Download PDF (${formatBytes(pdf.size)})`;
                        csActions.appendChild(btnDl);

                        const btnView = document.createElement('a');
                        btnView.className = 'btn btn-secondary btn-small';
                        btnView.href = `/api/contact_sheet/download?path=${encodeURIComponent(pdf.path)}`;
                        btnView.target = '_blank';
                        btnView.innerHTML = `<span>👁️</span> View PDF`;
                        csActions.appendChild(btnView);
                    }

                    if (jpegs.length > 0) {
                        jpegs.forEach((jpg, i) => {
                            const btnJpg = document.createElement('a');
                            btnJpg.className = 'btn btn-secondary btn-small';
                            btnJpg.href = `/api/contact_sheet/download?path=${encodeURIComponent(jpg.path)}&download=1`;
                            btnJpg.innerHTML = `<span>🖼️</span> JPG${jpegs.length > 1 ? ' #' + (i + 1) : ''} (${formatBytes(jpg.size)})`;
                            csActions.appendChild(btnJpg);
                        });
                    }

                    const btnRegen = document.createElement('button');
                    btnRegen.className = 'btn btn-secondary btn-small';
                    btnRegen.onclick = openContactSheetModal;
                    btnRegen.innerHTML = `<span>⚙️</span> Options`;
                    csActions.appendChild(btnRegen);
                } else {
                    csBanner.style.display = 'none';
                }
            }

            if (positives.length === 0) {
                grid.innerHTML = '<div class="empty-gallery-msg">No positive TIFF scans found yet. Triplets will appear here once processed.</div>';
                return;
            }

            grid.innerHTML = '';

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

export function openContactSheetModal() {
    const modal = document.getElementById('contact-sheet-modal');
    if (!modal) return;

    const stockInput = document.getElementById('cs-stock-input');
    const formatInput = document.getElementById('cs-format-input');
    const rollInput = document.getElementById('cs-roll-input');
    const statusDiv = document.getElementById('cs-generate-status');

    if (stockInput) stockInput.value = document.getElementById('scanner-stock')?.value || '';
    if (formatInput) formatInput.value = document.getElementById('scanner-format')?.value || '135';
    if (rollInput) rollInput.value = document.getElementById('scanner-roll')?.value || '01';
    if (statusDiv) {
        statusDiv.style.display = 'none';
        statusDiv.textContent = '';
    }

    modal.classList.add('active');
}

export function closeContactSheetModal() {
    const modal = document.getElementById('contact-sheet-modal');
    if (modal) modal.classList.remove('active');
}

export function submitContactSheetGeneration() {
    const folderInput = document.getElementById('cs-session-folder');
    const stockInput = document.getElementById('cs-stock-input');
    const formatInput = document.getElementById('cs-format-input');
    const rollInput = document.getElementById('cs-roll-input');
    const colsSelect = document.getElementById('cs-columns-select');
    const themeSelect = document.getElementById('cs-theme-select');
    const exportPdfCheck = document.getElementById('cs-export-pdf');
    const exportJpegCheck = document.getElementById('cs-export-jpeg');
    const submitBtn = document.getElementById('btn-submit-contact-sheet');
    const statusDiv = document.getElementById('cs-generate-status');

    const payload = {
        session_dir: folderInput?.value.trim() || null,
        stock: stockInput?.value.trim() || '',
        format: formatInput?.value.trim() || '',
        roll: rollInput?.value.trim() || '',
        columns: parseInt(colsSelect?.value || '6', 10),
        theme: themeSelect?.value || 'dark',
        export_pdf: exportPdfCheck ? exportPdfCheck.checked : true,
        export_jpeg: exportJpegCheck ? exportJpegCheck.checked : true
    };

    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = '⏳ Rendering Contact Sheet...';
    }
    if (statusDiv) {
        statusDiv.style.display = 'block';
        statusDiv.className = 'text-muted';
        statusDiv.textContent = 'Generating high-resolution archival sheets...';
    }

    fetch('/api/contact_sheet/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(async res => {
        const data = await res.json().catch(() => null);
        if (!res.ok) {
            const err = (data && data.message) || `Error ${res.status}: ${res.statusText}`;
            throw new Error(err);
        }
        return data;
    })
    .then(data => {
        if (statusDiv) {
            statusDiv.className = 'text-success';
            statusDiv.textContent = `✅ ${data.message}`;
        }
        window.appendLogLine?.(`[Archival Contact Sheet] ${data.message}`);
        setTimeout(() => {
            closeContactSheetModal();
            refreshFiles();
        }, 1200);
    })
    .catch(err => {
        if (statusDiv) {
            statusDiv.className = 'text-error';
            statusDiv.textContent = `❌ ${err.message || err}`;
        }
        window.appendLogLine?.(`[Client Error] Contact sheet failed: ${err.message || err}`);
    })
    .finally(() => {
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.textContent = '⚡ Generate Contact Sheet';
        }
    });
}

export function openLightbox(name, path, size) {
    const box = document.getElementById('image-lightbox');
    const img = document.getElementById('lightbox-img');
    const title = document.getElementById('lightbox-title');
    const desc = document.getElementById('lightbox-desc');
    const dlLink = document.getElementById('lightbox-download-link');

    currentLightboxPath = path;
    window.currentLightboxPath = path;

    img.src = '';
    title.textContent = name;
    desc.textContent = `Size: ${formatBytes(size)} | Path: ${path}`;

    dlLink.href = `/api/preview?path=${encodeURIComponent(path)}`;

    box.classList.add('active');

    img.src = `/api/preview?path=${encodeURIComponent(path)}&w=900`;

    window.initLightboxEyedropperListeners?.();
}

export function closeLightbox() {
    if (window.isEyedropperActive && window.eyedropperSource === 'lightbox') {
        window.cancelFilmBaseEyedropper?.();
    }
    const box = document.getElementById('image-lightbox');
    if (box) box.classList.remove('active');
}

export function openLoadSessionBrowser() {
    if (window.systemStatus && window.systemStatus !== 'idle') {
        alert("Please stop the active scanning session before loading another session.");
        return;
    }
    browserActiveInputId = '__load_session__';
    const currentRoot = document.getElementById('scanner-root-dir')?.value.trim() || '';
    browserSelectedPath = '';
    const filterInput = document.getElementById('browser-filter-input');
    if (filterInput) filterInput.value = '';
    const modal = document.getElementById('folder-browser-modal');
    if (modal) modal.classList.add('active');
    loadDirectory(currentRoot);
}

export function loadSessionFromPath(sessionPath) {
    if (!sessionPath) return;
    fetch('/api/session/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_dir: sessionPath })
    })
    .then(async res => {
        const data = await res.json().catch(() => null);
        if (!res.ok || !data || !data.success) {
            const err = (data && data.message) || `Failed to load session (${res.status})`;
            throw new Error(err);
        }
        return data;
    })
    .then(data => {
        const m = data.manifest;
        if (m) {
            if (m.root_dir) {
                const rootEl = document.getElementById('scanner-root-dir');
                if (rootEl) rootEl.value = m.root_dir;
            }
            if (m.stock) {
                const stockEl = document.getElementById('scanner-stock');
                if (stockEl) stockEl.value = m.stock;
            }
            if (m.format) {
                const fmtEl = document.getElementById('scanner-format');
                if (fmtEl) fmtEl.value = m.format;
            }
            if (m.roll) {
                const rollEl = document.getElementById('scanner-roll');
                if (rollEl) rollEl.value = m.roll;
            }
            if (m.mode) {
                const modeEl = document.getElementById('scanner-mode');
                if (modeEl) modeEl.value = m.mode;
                if (typeof window.updateScannerModeUI === 'function') {
                    window.updateScannerModeUI(m.mode);
                }
            }
            window.activeFilmBaseRGB = m.sampled_rgb || null;
            if (m.config && typeof window.syncConfigToUI === 'function') {
                window.syncConfigToUI(m.config);
            }
            if (typeof window.updateLastFrameCard === 'function') {
                window.updateLastFrameCard(m.last_frame || (data.status_data && data.status_data.last_frame) || null, false);
            }
        }
        window.appendLogLine?.(`[Client] ${data.message}`);
        refreshFiles();
    })
    .catch(err => {
        window.appendLogLine?.(`[Client Error] Could not load session: ${err.message || err}`);
        alert(`Error loading session: ${err.message || err}`);
    });
}

export function openFolderBrowser(inputId) {
    browserActiveInputId = inputId;
    const currentVal = document.getElementById(inputId).value.trim();

    browserSelectedPath = '';

    const filterInput = document.getElementById('browser-filter-input');
    if (filterInput) filterInput.value = '';

    const modal = document.getElementById('folder-browser-modal');
    if (modal) modal.classList.add('active');

    loadDirectory(currentVal);
}

export function closeFolderBrowser() {
    const modal = document.getElementById('folder-browser-modal');
    if (modal) modal.classList.remove('active');
}

export function loadDirectory(path) {
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
                if (path) {
                    window.appendLogLine?.(`[Folder Browser] Failed to read '${path}'. Falling back to Home folder.`);
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
            browserSelectedPath = data.current;
            browserAllFolders = data.folders || [];

            renderBrowser(data);
        })
        .catch(err => {
            listContainer.innerHTML = `<div class="browser-folder-empty text-error">Error: ${err.message}</div>`;
        });
}

export function renderBrowser(data) {
    renderBreadcrumbs(data.current);

    const listContainer = document.getElementById('browser-folder-list');
    if (!listContainer) return;
    listContainer.innerHTML = '';

    if (data.current === 'root' && data.drives && data.drives.length > 0) {
        data.drives.forEach(drive => {
            const item = document.createElement('div');
            item.className = 'browser-folder-item';
            item.onclick = () => selectFolderItem(item, drive);
            item.ondblclick = () => traverseToFolder(drive);

            item.innerHTML = `
                <span class="browser-folder-icon">💾</span>
                <span class="browser-folder-name">${drive}</span>
            `;
            listContainer.appendChild(item);
        });
        return;
    }

    if (data.parent) {
        const item = document.createElement('div');
        item.className = 'browser-folder-item';
        item.onclick = () => selectFolderItem(item, data.parent);
        item.ondblclick = () => traverseToFolder(data.parent);

        const label = data.parent === 'root' ? '.. [System Drives]' : '.. (Parent Directory)';
        item.innerHTML = `
            <span class="browser-folder-icon">⬆️</span>
            <span class="browser-folder-name" style="font-weight: 500;">${label}</span>
        `;
        listContainer.appendChild(item);
    }

    renderFolderListItems(data.folders);
}

export function renderFolderListItems(folders) {
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
        item.onclick = () => selectFolderItem(item, folder.path);
        item.ondblclick = () => traverseToFolder(folder.path);

        const badgeHtml = folder.has_manifest
            ? `<span style="margin-left: auto; font-size: 0.7rem; padding: 0.1rem 0.45rem; border-radius: 4px; background: rgba(14, 165, 233, 0.15); color: var(--accent-blue); border: 1px solid rgba(14, 165, 233, 0.3);">🎞️ Session</span>`
            : '';

        item.innerHTML = `
            <span class="browser-folder-icon">${folder.has_manifest ? '🎞️' : '📁'}</span>
            <span class="browser-folder-name">${folder.name}</span>
            ${badgeHtml}
        `;
        listContainer.appendChild(item);
    });
}

export function renderBreadcrumbs(pathStr) {
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
            runningPath = (index === 0) ? (part + '\\') : (runningPath + part + '\\');
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

export function selectFolderItem(itemEl, path) {
    document.querySelectorAll('.browser-folder-item').forEach(el => el.classList.remove('selected'));
    itemEl.classList.add('selected');
    browserSelectedPath = path;
}

export function traverseToFolder(path) {
    const filterInput = document.getElementById('browser-filter-input');
    if (filterInput) filterInput.value = '';
    loadDirectory(path);
}

export function filterBrowserFolders() {
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

export function getParentPath(pathStr) {
    if (pathStr === 'root') return '';
    const parts = pathStr.split(/[\\/]/).filter(p => p !== '');
    if (parts.length <= 1) {
        return 'root';
    }
    const sep = pathStr.includes('\\') ? '\\' : '/';
    return pathStr.substring(0, pathStr.lastIndexOf(sep));
}

export function confirmFolderSelection() {
    if (!browserSelectedPath) {
        alert("Please select a folder first.");
        return;
    }

    if (browserActiveInputId === '__load_session__') {
        const chosen = browserSelectedPath;
        closeFolderBrowser();
        loadSessionFromPath(chosen);
        return;
    }

    const input = document.getElementById(browserActiveInputId);
    if (input) {
        input.value = browserSelectedPath;
        input.dispatchEvent(new Event('change', { bubbles: true }));
    }
    if (browserActiveInputId === 'cs-session-folder') {
        fetch(`/api/session/manifest?path=${encodeURIComponent(browserSelectedPath)}`)
            .then(res => res.ok ? res.json() : null)
            .then(data => {
                if (data && data.success && data.manifest) {
                    const m = data.manifest;
                    if (m.stock && document.getElementById('cs-stock-input')) document.getElementById('cs-stock-input').value = m.stock;
                    if (m.format && document.getElementById('cs-format-input')) document.getElementById('cs-format-input').value = m.format;
                    if (m.roll && document.getElementById('cs-roll-input')) document.getElementById('cs-roll-input').value = m.roll;
                }
            })
            .catch(() => {});
    }
    closeFolderBrowser();
}

// Expose to window for inline HTML handlers
Object.assign(window, {
    formatBytes,
    refreshFiles,
    openContactSheetModal,
    closeContactSheetModal,
    submitContactSheetGeneration,
    openLightbox,
    closeLightbox,
    openLoadSessionBrowser,
    loadSessionFromPath,
    openFolderBrowser,
    closeFolderBrowser,
    loadDirectory,
    renderBrowser,
    renderFolderListItems,
    renderBreadcrumbs,
    selectFolderItem,
    traverseToFolder,
    filterBrowserFolders,
    getParentPath,
    confirmFolderSelection,
});
