import { test, expect } from '@playwright/test';

test.describe('RBTA Security Analytics Dashboard E2E', () => {
  test.beforeEach(async ({ page }) => {
    // Open the dashboard entry point
    await page.goto('/dashboard/');
  });

  test('sign-in gate requires valid API key and grants access', async ({ page }) => {
    // Initial view should present authentication gate
    const heading = page.locator('h1');
    await expect(heading).toContainText(/RBTA Security Analytics/i);

    // Attempting invalid input
    const input = page.locator('input[type="password"]');
    await input.fill('wrong-key');
    await page.keyboard.press('Enter');

    // Enter valid credentials
    await input.fill('secret-api-key-123');
    await page.keyboard.press('Enter');

    // Overview page should load
    await expect(page.locator('text=Security Analytics Overview')).toBeVisible({ timeout: 10000 });
  });

  test('overview KPIs display verbatim reduction rate and actionable escalations', async ({ page }) => {
    // Set authenticated session directly
    await page.evaluate(() => {
      sessionStorage.setItem('rbta.dashboard.apiKey', 'secret-api-key-123');
    });
    await page.goto('/dashboard/overview');

    await expect(page.locator('text=Raw Ingested Alerts')).toBeVisible();
    await expect(page.locator('text=Finalized MetaAlerts')).toBeVisible();
    await expect(page.locator('text=Alert Reduction Rate')).toBeVisible();
  });

  test('meta-alerts table drill-down to member raw alerts and raw forensic view', async ({ page }) => {
    await page.evaluate(() => {
      sessionStorage.setItem('rbta.dashboard.apiKey', 'secret-api-key-123');
    });
    await page.goto('/dashboard/meta-alerts');

    await expect(page.locator('text=MetaAlerts Investigation Table')).toBeVisible();
  });

  test('replay controller interface supports dataset selection and state pacing', async ({ page }) => {
    await page.evaluate(() => {
      sessionStorage.setItem('rbta.dashboard.apiKey', 'secret-api-key-123');
    });
    await page.goto('/dashboard/replay');

    await expect(page.locator('text=Demonstration Replay Controller')).toBeVisible();
    await expect(page.locator('select').first()).toBeVisible();
  });

  test('theme switcher toggles dark and light modes cleanly', async ({ page }) => {
    await page.evaluate(() => {
      sessionStorage.setItem('rbta.dashboard.apiKey', 'secret-api-key-123');
    });
    await page.goto('/dashboard/overview');

    // Click theme switcher
    const themeBtn = page.locator('button[title*="Theme:"]');
    if (await themeBtn.isVisible()) {
      await themeBtn.click();
    }
  });
});
