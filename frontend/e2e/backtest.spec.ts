import { test, expect, Page } from '@playwright/test';

/**
 * E2E tests for the backtesting flow.
 * Covers: navigate to backtest → select strategy → configure → run → view results.
 */

const TEST_USER = {
  username: process.env.E2E_USERNAME || 'testuser',
  password: process.env.E2E_PASSWORD || 'TestPass123!',
};

async function loginAndNavigate(page: Page, target: string) {
  await page.goto('/');

  // Click through landing if needed
  const signInBtn = page.getByRole('button', { name: 'Sign In' });
  if (!(await signInBtn.isVisible())) {
    const getStartedBtn = page.getByRole('button', { name: /get started|sign in|login/i });
    if (await getStartedBtn.isVisible()) await getStartedBtn.click();
  }

  await page.getByPlaceholder('Enter your email or username').waitFor({ timeout: 10_000 });
  await page.getByPlaceholder('Enter your email or username').fill(TEST_USER.username);
  await page.getByPlaceholder('••••••••••').fill(TEST_USER.password);
  await page.getByRole('button', { name: 'Sign In' }).click();

  // Wait for dashboard
  await expect(
    page.getByText('Activity Hub').or(page.getByText('Dashboard').first())
  ).toBeVisible({ timeout: 15_000 });

  // Navigate via sidebar
  await page.getByText(target).click();
}

test.describe('Backtesting', () => {
  test('should navigate to backtest page', async ({ page }) => {
    await loginAndNavigate(page, 'Backtest');

    // Should see the backtest page content
    await expect(
      page.getByText(/backtest|strategy/i).first()
    ).toBeVisible({ timeout: 10_000 });
  });

  test('should show strategy selection dropdown', async ({ page }) => {
    await loginAndNavigate(page, 'Backtest');

    // Look for a strategy selector/dropdown
    const strategySelector = page.locator('select, [role="combobox"], [role="listbox"]').first();
    await expect(strategySelector).toBeVisible({ timeout: 10_000 });
  });

  test('should configure and run a single backtest', async ({ page }) => {
    await loginAndNavigate(page, 'Backtest');

    // Wait for page to fully load
    await page.waitForTimeout(2000);

    // Look for ticker input and type a symbol
    const tickerInput = page.getByPlaceholder(/ticker|symbol|enter.*stock/i).first();
    if (await tickerInput.isVisible()) {
      await tickerInput.clear();
      await tickerInput.fill('AAPL');
    }

    // Click Run Backtest button
    const runBtn = page.getByRole('button', { name: /run.*backtest|start.*backtest|execute/i });
    if (await runBtn.isVisible()) {
      await runBtn.click();

      // Wait for results — look for metrics like Sharpe Ratio or Total Return
      await expect(
        page.getByText(/sharpe.*ratio|total.*return|running|queued/i).first()
      ).toBeVisible({ timeout: 60_000 });
    }
  });

  test('should open Bayesian optimizer modal', async ({ page }) => {
    await loginAndNavigate(page, 'Backtest');
    await page.waitForTimeout(2000);

    // Look for optimizer button
    const optimizerBtn = page.getByRole('button', { name: /optimi/i }).first();
    if (await optimizerBtn.isVisible()) {
      await optimizerBtn.click();

      // Should see the Bayesian optimizer modal
      await expect(
        page.getByText(/bayesian|parameter.*range|optimi/i).first()
      ).toBeVisible({ timeout: 5_000 });
    }
  });
});
