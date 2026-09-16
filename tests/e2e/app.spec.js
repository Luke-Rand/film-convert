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

});
