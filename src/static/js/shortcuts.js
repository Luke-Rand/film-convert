// ============================================================
// KEYBOARD SHORTCUTS MODULE
// ============================================================

import {
    cycleHistogramMode,
    autoTuneLEDLevels,
    runMiniScanlightSequence,
} from './histogram.js';

import {
    closeFolderBrowser,
    closeLightbox,
} from './gallery.js';

import {
    cancelFilmBaseEyedropper,
    toggleLiveviewFullscreen,
    triggerCameraCapture,
    triggerAutofocus,
    toggleCameraLiveview,
    toggleLiveviewRotation,
    toggleMarginOverlay,
    toggleFilmBaseEyedropper,
    toggleFocusPeaking,
    stepCameraShutter,
    driveFocus,
    setZoomMagnification,
} from './liveview.js';

export function toggleShortcutsModal() {
    const modal = document.getElementById('shortcuts-modal');
    if (modal) {
        if (modal.classList.contains('active')) {
            closeShortcutsModal();
        } else {
            openShortcutsModal();
        }
    }
}

export function openShortcutsModal() {
    const modal = document.getElementById('shortcuts-modal');
    if (modal) {
        modal.classList.add('active');
    }
}

export function closeShortcutsModal() {
    const modal = document.getElementById('shortcuts-modal');
    if (modal) {
        modal.classList.remove('active');
    }
}

export function isTypingInInput(e) {
    const target = e.target;
    if (!target) return false;
    const tagName = target.tagName ? target.tagName.toUpperCase() : '';
    if (tagName === 'INPUT' || tagName === 'TEXTAREA' || tagName === 'SELECT' || target.isContentEditable) {
        return true;
    }
    return false;
}

export function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        const isInput = isTypingInInput(e);

        // Escape always cancels overlays / modals / eyedropper / fullscreen
        if (e.key === 'Escape') {
            const shortcutsModal = document.getElementById('shortcuts-modal');
            if (shortcutsModal && shortcutsModal.classList.contains('active')) {
                closeShortcutsModal();
                e.preventDefault();
                return;
            }
            if (window.isEyedropperActive) {
                cancelFilmBaseEyedropper();
                e.preventDefault();
                return;
            }
            if (window.isLiveviewFullscreen) {
                toggleLiveviewFullscreen();
                e.preventDefault();
                return;
            }
            const folderModal = document.getElementById('folder-browser-modal');
            if (folderModal && folderModal.classList.contains('active')) {
                closeFolderBrowser();
                e.preventDefault();
                return;
            }
            const lightbox = document.getElementById('image-lightbox');
            if (lightbox && lightbox.classList.contains('active')) {
                closeLightbox();
                e.preventDefault();
                return;
            }
            const trimModal = document.getElementById('scanlight-trim-modal');
            if (trimModal && trimModal.classList.contains('active')) {
                if (typeof window.scanlightController !== 'undefined' && window.scanlightController.closeTrimModal) {
                    window.scanlightController.closeTrimModal();
                }
                e.preventDefault();
                return;
            }
            return;
        }

        // Shift + / or ? toggles keyboard shortcuts modal (from anywhere unless in input)
        if (e.key === '?' || (e.shiftKey && (e.key === '/' || e.code === 'Slash'))) {
            if (!isInput) {
                e.preventDefault();
                toggleShortcutsModal();
                return;
            }
        }

        // Ignore single key shortcuts if user is typing into an input field or textarea
        if (isInput) return;

        // Alt + 1..4 (or Option + 1..4 / Cmd + 1..4) Tab Navigation
        if ((e.altKey || e.metaKey) && !e.ctrlKey) {
            if (e.code === 'Digit1' || e.code === 'Numpad1' || e.key === '1' || e.key === '¡') {
                e.preventDefault();
                window.switchTab?.('scanner');
                return;
            }
            if (e.code === 'Digit2' || e.code === 'Numpad2' || e.key === '2' || e.key === '™') {
                e.preventDefault();
                window.switchTab?.('batch');
                return;
            }
            if (e.code === 'Digit3' || e.code === 'Numpad3' || e.key === '3' || e.key === '£') {
                e.preventDefault();
                window.switchTab?.('gallery');
                return;
            }
            if (e.code === 'Digit4' || e.code === 'Numpad4' || e.key === '4' || e.key === '¢') {
                e.preventDefault();
                window.switchTab?.('scanlight');
                return;
            }
        }

        // If shortcuts modal is currently open, don't execute background action shortcuts
        const shortcutsModal = document.getElementById('shortcuts-modal');
        if (shortcutsModal && shortcutsModal.classList.contains('active')) {
            return;
        }

        // Trigger Camera RAW Capture: Space or C
        if ((e.code === 'Space' && !e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey) ||
            (e.key.toLowerCase() === 'c' && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey)) {
            e.preventDefault();
            triggerCameraCapture();
            return;
        }

        // Auto R-G-B Sequence: A (or Shift + A for Autofocus)
        if (e.key.toLowerCase() === 'a' && !e.ctrlKey && !e.metaKey && !e.altKey) {
            e.preventDefault();
            if (e.shiftKey) {
                triggerAutofocus();
            } else {
                runMiniScanlightSequence('SequenceRGB');
            }
            return;
        }

        // Autofocus Trigger: U
        if (e.key.toLowerCase() === 'u' && !e.ctrlKey && !e.metaKey && !e.altKey) {
            e.preventDefault();
            triggerAutofocus();
            return;
        }

        // Toggle Hot Folder Monitor: M
        if (e.key.toLowerCase() === 'm' && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
            e.preventDefault();
            window.toggleMonitor?.();
            return;
        }

        // Toggle Live View Feed: V or L
        if ((e.key.toLowerCase() === 'v' || e.key.toLowerCase() === 'l') && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
            e.preventDefault();
            const toggle = document.getElementById('camera-liveview-toggle');
            if (toggle) {
                toggle.checked = !toggle.checked;
                toggleCameraLiveview(toggle.checked);
            }
            return;
        }

        // Toggle Fullscreen Live View: F
        if (e.key.toLowerCase() === 'f' && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
            e.preventDefault();
            toggleLiveviewFullscreen();
            return;
        }

        // Rotate Live View 180°: R
        if (e.key.toLowerCase() === 'r' && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
            e.preventDefault();
            toggleLiveviewRotation();
            return;
        }

        // Toggle Margins Overlay: O
        if (e.key.toLowerCase() === 'o' && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
            e.preventDefault();
            toggleMarginOverlay();
            return;
        }

        // Sample Film Rebate Eyedropper: E
        if (e.key.toLowerCase() === 'e' && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
            e.preventDefault();
            toggleFilmBaseEyedropper('liveview');
            return;
        }

        // Toggle Focus Peaking: P
        if (e.key.toLowerCase() === 'p' && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
            e.preventDefault();
            const peakToggle = document.getElementById('focus-peaking-toggle');
            if (peakToggle) {
                peakToggle.checked = !peakToggle.checked;
                toggleFocusPeaking(peakToggle.checked);
            }
            return;
        }

        // Auto-Tune Optimal LEDs (ETTR): T
        if (e.key.toLowerCase() === 't' && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
            e.preventDefault();
            autoTuneLEDLevels();
            return;
        }

        // Cycle Histogram Modes: H
        if (e.key.toLowerCase() === 'h' && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
            e.preventDefault();
            cycleHistogramMode();
            return;
        }

        // Step Camera Shutter Speed: - / _ / , (slower), = / + / . (faster)
        if ((e.key === '-' || e.key === '_' || e.code === 'Minus' || e.code === 'NumpadSubtract' || e.key === ',') && !e.ctrlKey && !e.metaKey && !e.altKey) {
            e.preventDefault();
            stepCameraShutter('slower');
            return;
        }
        if ((e.key === '=' || e.key === '+' || e.code === 'Equal' || e.code === 'NumpadAdd' || e.key === '.') && !e.ctrlKey && !e.metaKey && !e.altKey) {
            e.preventDefault();
            stepCameraShutter('faster');
            return;
        }

        // Focus Stepping: [ or Left (Near), ] or Right (Far)
        if (e.key === '[' || e.key === '{' || e.key === 'ArrowLeft' || e.code === 'BracketLeft') {
            e.preventDefault();
            const step = (e.shiftKey || e.key === '{') ? 3 : 1;
            driveFocus('near', step);
            return;
        }
        if (e.key === ']' || e.key === '}' || e.key === 'ArrowRight' || e.code === 'BracketRight') {
            e.preventDefault();
            const step = (e.shiftKey || e.key === '}') ? 3 : 1;
            driveFocus('far', step);
            return;
        }

        // Magnification Zoom Levels: 1 (1x), 2 (3x), 3 (5x) / 5 (5x)
        if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
            if (e.key === '1') {
                e.preventDefault();
                setZoomMagnification(1);
                return;
            }
            if (e.key === '2') {
                e.preventDefault();
                setZoomMagnification(3);
                return;
            }
            if (e.key === '3' || e.key === '5') {
                e.preventDefault();
                setZoomMagnification(5);
                return;
            }
        }
    });
}

// Expose to window for inline HTML handlers
Object.assign(window, {
    toggleShortcutsModal,
    openShortcutsModal,
    closeShortcutsModal,
    isTypingInInput,
});
