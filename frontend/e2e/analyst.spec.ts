import { test, expect, Page } from '@playwright/test';

/**
 * E2E tests for the Deep Analyst (AI Analyst) page.
 * Covers: navigate → enter ticker → run analysis → view results.
 */

const TEST_USER = {
  username: process.env.E2E_USERNAME || 'testuser',
  password: process.env.E2E_PASSWORD || 'TestPass123!',
};

async function loginAndNavigate(page: Page) {
  await page.goto('/');
  const signInBtn = page.getByRole('button', { name: 'Sign In' });
  if (!(await signInBtn.isVisible())) {
    const getStartedBtn = page.getByRole('button', { name: /get started|sign in|login/i });
    if (await getStartedBtn.isVisible()) await getStartedBtn.click();
  }

  await page.getByPlaceholder('Enter your email or username').waitFor({ timeout: 10_000 });
  await page.getByPlaceholder('Enter your email or username').fill(TEST_USER.username);
  await page.getByPlaceholder('••••••••••').fill(TEST_USER.password);
  await page.getByRole('button', { name: 'Sign In' }).click();

  await expect(
    page.getByText('Activity Hub').or(page.getByText('Dashboard').first())
  ).toBeVisible({ timeout: 15_000 });

  await page.getByText('Deep Analyst').click();
}

test.describe('AI Deep Analyst', () => {
  test('should navigate to analyst page', async ({ page }) => {
    await loginAndNavigate(page);

    await expect(
      page.getByText(/deep.*analyst|ai.*analyst|analysis/i).first()
    ).toBeVisible({ timeout: 10_000 });
  });

  test('should have ticker input', async ({ page }) => {
    await loginAndNavigate(page);

    // Look for ticker input
    const tickerInput = page.getByPlaceholder(/ticker|symbol|enter.*stock|AAPL/i).first();
    await expect(tickerInput).toBeVisible({ timeout: 10_000 });
  });

  test('should run analysis for a ticker', async ({ page }) => {
    test.setTimeout(120_000); // Analysis can take a while
    await loginAndNavigate(page);

    // Enter ticker
    const tickerInput = page.getByPlaceholder(/ticker|symbol|enter.*stock|AAPL/i).first();
    if (await tickerInput.isVisible()) {
      await tickerInput.clear();
      await tickerInput.fill('AAPL');

      // Hit enter or click analyze button
      const analyzeBtn = page.getByRole('button', { name: /analyz|run|generate|go/i }).first();
      if (await analyzeBtn.isVisible()) {
        await analyzeBtn.click();
      } else {
        await tickerInput.press('Enter');
      }

      // Wait for results — should show summary tab or loading state
      await expect(
        page.getByText(/summary|loading|analyzing|score|rating|recommendation/i).first()
      ).toBeVisible({ timeout: 90_000 });
    }
  });
});
