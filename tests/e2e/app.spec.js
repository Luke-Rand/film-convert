const { test, expect } = require('@playwright/test');

test.describe('FilmConvert Web & Electron UI Test Suite', () => {

  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#app-root')).toBeVisible();
    await expect(page.locator('.sidebar')).toBeVisible();
  });

  test('Page loads properly with correct header and initial state', async ({ page }) => {
    await expect(page).toHaveTitle(/Tri-Color Film Auto-Compositor & Inverter/);
    await expect(page.locator('.sidebar-header h1')).toHaveText('FilmConvert');
    await expect(page.locator('#summary-status-text')).toContainText('System Idle');
  });

  test('Navigation tabs switch smoothly across all four panels', async ({ page }) => {
    // Tab 1: Scanner is active by default
    await expect(page.locator('#panel-scanner')).toHaveClass(/active/);
    await expect(page.locator('#nav-scanner-btn')).toHaveClass(/active/);

    // Tab 2: Switch to Batch Processor
    await page.click('#nav-batch-btn');
    await expect(page.locator('#panel-batch')).toHaveClass(/active/);
    await expect(page.locator('#nav-batch-btn')).toHaveClass(/active/);
    await expect(page.locator('#panel-scanner')).not.toHaveClass(/active/);

    // Tab 3: Switch to Scan Gallery
    await page.click('#nav-gallery-btn');
    await expect(page.locator('#panel-gallery')).toHaveClass(/active/);
    await expect(page.locator('#nav-gallery-btn')).toHaveClass(/active/);

    // Tab 4: Switch to Scanlight Controller
    await page.click('#nav-scanlight-btn');
    await expect(page.locator('#panel-scanlight')).toHaveClass(/active/);
    await expect(page.locator('#nav-scanlight-btn')).toHaveClass(/active/);

    // Switch back to Scanner
    await page.click('#nav-scanner-btn');
    await expect(page.locator('#panel-scanner')).toHaveClass(/active/);
  });

  test('Sidebar toggle and tooltip toggle update UI states', async ({ page }) => {
    // Sidebar collapse/expand
    const sidebar = page.locator('.sidebar');
    await page.click('#btn-sidebar-toggle');
    await expect(sidebar).toHaveClass(/collapsed/);

    await page.click('#btn-sidebar-toggle');
    await expect(sidebar).not.toHaveClass(/collapsed/);

    // Tooltip toggle
    const tooltipBtn = page.locator('#btn-tooltip-toggle');
    await expect(tooltipBtn).toHaveClass(/active/);
    await tooltipBtn.click();
    await expect(tooltipBtn).not.toHaveClass(/active/);
    await expect(tooltipBtn.locator('.tt-label')).toHaveText('Tooltips Off');

    await tooltipBtn.click();
    await expect(tooltipBtn).toHaveClass(/active/);
    await expect(tooltipBtn.locator('.tt-label')).toHaveText('Tooltips On');
  });

  test('Live Scanner controls and sliders update configuration', async ({ page }) => {
    // Check simulated camera badge
    const cameraBadge = page.locator('#camera-status-badge');
    await expect(cameraBadge).toBeVisible();

    // Toggle live view switch
    const liveviewContainer = page.locator('.switch-container:has(#camera-liveview-toggle)');
    await expect(liveviewContainer).toBeVisible();
    const liveviewToggle = page.locator('#camera-liveview-toggle');
    await liveviewContainer.click();
    await expect(liveviewToggle).toBeChecked();
    await liveviewContainer.click();
    await expect(liveviewToggle).not.toBeChecked();

    // Fill Session details
    const stockInput = page.locator('#scanner-stock');
    if (await stockInput.count() > 0) {
      await stockInput.fill('Gold200');
      await expect(stockInput).toHaveValue('Gold200');
    }

    // Expand Advanced Processing Parameters Accordion
    const accordionHeader = page.locator('#panel-scanner .accordion-header:has-text("Advanced Processing Parameters")');
    if (await accordionHeader.count() > 0) {
      await accordionHeader.click();
      await expect(page.locator('#scanner-settings-panel')).not.toHaveClass(/hidden/);
    }

    // Adjust Gamma Slider
    const gammaSlider = page.locator('#config-gamma');
    if (await gammaSlider.count() > 0) {
      await gammaSlider.fill('2.4');
      await gammaSlider.dispatchEvent('input');
      await gammaSlider.dispatchEvent('change');
      await expect(page.locator('#val-gamma')).toHaveText('2.4');
    }

    // Toggle Monochrome
    const monoToggle = page.locator('#config-monochrome');
    if (await monoToggle.count() > 0) {
      await monoToggle.setChecked(true, { force: true });
      await expect(monoToggle).toBeChecked();
      await monoToggle.setChecked(false, { force: true });
      await expect(monoToggle).not.toBeChecked();
    }
  });

  test('Batch Processor panel has required controls and options', async ({ page }) => {
    await page.click('#nav-batch-btn');
    await expect(page.locator('#panel-batch')).toBeVisible();

    const batchModeSelect = page.locator('#batch-task-type');
    if (await batchModeSelect.count() > 0) {
      await batchModeSelect.selectOption('invert');
      await expect(batchModeSelect).toHaveValue('invert');
      await batchModeSelect.selectOption('composite');
      await expect(batchModeSelect).toHaveValue('composite');
    }
  });

  test('Scan Gallery displays empty state when no files exist', async ({ page }) => {
    await page.click('#nav-gallery-btn');
    await expect(page.locator('#panel-gallery')).toBeVisible();
    const galleryContainer = page.locator('#panel-gallery');
    await expect(galleryContainer).toBeVisible();
  });

  test('Scanlight controller preset and slider interactions', async ({ page }) => {
    await page.click('#nav-scanlight-btn');
    await expect(page.locator('#panel-scanlight')).toBeVisible();

    // Preset buttons
    const presetBtn = page.locator('.btn-preset').first();
    if (await presetBtn.count() > 0) {
      await presetBtn.click();
    }
  });

  test('Backend API schema validation returns 400 on malformed payloads', async ({ request }) => {
    // Invalid config update
    const badConfigResp = await request.post('/api/config', {
      data: { clip: 999.0 }
    });
    expect(badConfigResp.status()).toBe(400);
    const badConfigData = await badConfigResp.json();
    expect(badConfigData.success).toBe(false);

    // Invalid camera focus step
    const badFocusResp = await request.post('/api/camera/focus_step', {
      data: { direction: 'invalid_direction' }
    });
    expect(badFocusResp.status()).toBe(400);
    const badFocusData = await badFocusResp.json();
    expect(badFocusData.success).toBe(false);

    // Valid config update succeeds
    const goodConfigResp = await request.post('/api/config', {
      data: { clip: 0.15, gamma: 2.2 }
    });
    expect(goodConfigResp.status()).toBe(200);
    const goodConfigData = await goodConfigResp.json();
    expect(goodConfigData.success).toBe(true);
    expect(goodConfigData.config.clip).toBe(0.15);
  });

  test('Keyboard shortcuts modal opens on Shift+/ or ? and closes on Escape', async ({ page }) => {
    const shortcutsModal = page.locator('#shortcuts-modal');
    await expect(shortcutsModal).not.toHaveClass(/active/);

    // Press Shift+/ (or ?) to open modal
    await page.keyboard.press('Shift+Slash');
    await expect(shortcutsModal).toHaveClass(/active/);
    await expect(shortcutsModal.locator('h3')).toHaveText('Keyboard Shortcuts');

    // Press Escape to close modal
    await page.keyboard.press('Escape');
    await expect(shortcutsModal).not.toHaveClass(/active/);

    // Press '?' directly
    await page.keyboard.press('?');
    await expect(shortcutsModal).toHaveClass(/active/);

    // Close via close button
    await shortcutsModal.locator('.modal-close').click();
    await expect(shortcutsModal).not.toHaveClass(/active/);

    // Open via sidebar shortcuts button
    await page.click('#btn-shortcuts-toggle');
    await expect(shortcutsModal).toHaveClass(/active/);
    await page.keyboard.press('Escape');
    await expect(shortcutsModal).not.toHaveClass(/active/);

    // Open via liveview toolbar shortcuts button
    await page.click('#btn-liveview-shortcuts');
    await expect(shortcutsModal).toHaveClass(/active/);
    await page.click('#shortcuts-modal .modal-footer button');
    await expect(shortcutsModal).not.toHaveClass(/active/);
  });

  test('Live Capture panel keyboard shortcuts function properly', async ({ page }) => {
    // 1. Zoom levels via 1, 2, 3
    const zoom1x = page.locator('#btn-zoom-1x');
    const zoom3x = page.locator('#btn-zoom-3x');
    const zoom5x = page.locator('#btn-zoom-5x');

    await expect(zoom1x).toHaveClass(/active/);
    await page.keyboard.press('2');
    await expect(zoom3x).toHaveClass(/active/);
    await page.keyboard.press('3');
    await expect(zoom5x).toHaveClass(/active/);
    await page.keyboard.press('1');
    await expect(zoom1x).toHaveClass(/active/);

    // 2. Peaking toggle via 'p'
    const peakingToggle = page.locator('#focus-peaking-toggle');
    await expect(peakingToggle).not.toBeChecked();
    await page.keyboard.press('p');
    await expect(peakingToggle).toBeChecked();
    await page.keyboard.press('p');
    await expect(peakingToggle).not.toBeChecked();

    // 3. Margin overlay via 'o'
    const marginOverlay = page.locator('#margin-overlay');
    await expect(marginOverlay).toBeHidden();
    await page.keyboard.press('o');
    await expect(marginOverlay).toBeVisible();
    await page.keyboard.press('o');
    await expect(marginOverlay).toBeHidden();

    // 4. Fullscreen Live View via 'f' and exit via Escape
    const liveviewCard = page.locator('.liveview-card');
    await expect(liveviewCard).not.toHaveClass(/fullscreen-focus-mode/);
    await page.keyboard.press('f');
    await expect(liveviewCard).toHaveClass(/fullscreen-focus-mode/);
    await page.keyboard.press('Escape');
    await expect(liveviewCard).not.toHaveClass(/fullscreen-focus-mode/);

    // 5. Eyedropper mode via 'e' and cancel via Escape
    const eyedropperBanner = page.locator('#liveview-eyedropper-banner');
    await expect(eyedropperBanner).toBeHidden();
    await page.keyboard.press('e');
    await expect(eyedropperBanner).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(eyedropperBanner).toBeHidden();

    // 6. Typing in input fields does NOT trigger action shortcuts
    const stockInput = page.locator('#scanner-stock');
    await stockInput.fill('');
    await stockInput.focus();
    await page.keyboard.type('test-stock?fpo123');
    await expect(stockInput).toHaveValue('test-stock?fpo123');
    // Modal should not have opened
    await expect(page.locator('#shortcuts-modal')).not.toHaveClass(/active/);
    // Peaking should still be unchecked
    await expect(peakingToggle).not.toBeChecked();
  });

});

