const { defineConfig, devices } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests/e2e',
  timeout: 30000,
  expect: {
    timeout: 5000
  },
  fullyParallel: false,
  workers: 1,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: 'http://127.0.0.1:5055',
    trace: 'on-first-retry',
    viewport: { width: 1400, height: 900 }
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] }
    }
  ],
  webServer: {
    command: 'PORT=5055 .venv/bin/python src/web_ui.py',
    url: 'http://127.0.0.1:5055/',
    reuseExistingServer: !process.env.CI,
    timeout: 15000
  }
});
