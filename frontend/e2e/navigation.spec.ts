import { test, expect, Page } from '@playwright/test';

/**
 * E2E tests for core navigation.
 * Verifies all major pages load without crashes.
 */

const TEST_USER = {
  username: process.env.E2E_USERNAME || 'testuser',
  password: process.env.E2E_PASSWORD || 'TestPass123!',
};

async function login(page: Page) {
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
}

const PAGES_TO_TEST = [
  { sidebar: 'Activity Hub', expects: /activity|dashboard|hub/i },
  { sidebar: 'Watchlist', expects: /watchlist|portfolio|my.*list/i },
  { sidebar: 'Backtest', expects: /backtest|strategy|lab/i },
  { sidebar: 'Deep Analyst', expects: /analyst|deep|analysis/i },
  { sidebar: 'Options Desk', expects: /option|desk|chain/i },
  { sidebar: 'Sector Scanner', expects: /sector|scanner/i },
  { sidebar: 'Marketplace', expects: /marketplace|strateg/i },
  { sidebar: 'Leaderboard', expects: /leaderboard|ranking|top/i },
  { sidebar: 'Getting Started', expects: /getting.*started|quick.*start|guide/i },
  { sidebar: 'Settings', expects: /settings|preferences|account/i },
];

test.describe('Navigation - All Pages Load', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  for (const { sidebar, expects } of PAGES_TO_TEST) {
    test(`should navigate to "${sidebar}" without errors`, async ({ page }) => {
      // Click sidebar item
      const navItem = page.getByText(sidebar, { exact: false }).first();
      if (await navItem.isVisible()) {
        await navItem.click();
        await page.waitForTimeout(2000);

        // Page should render without unhandled errors
        const consoleErrors: string[] = [];
        page.on('console', (msg) => {
          if (msg.type() === 'error') consoleErrors.push(msg.text());
        });

        // Verify some expected content is visible
        await expect(page.getByText(expects).first()).toBeVisible({ timeout: 10_000 });

        // No fatal unhandled promise rejections
        const criticalErrors = consoleErrors.filter(
          (e) => e.includes('Unhandled') || e.includes('chunk') || e.includes('FATAL')
        );
        expect(criticalErrors).toHaveLength(0);
      }
    });
  }
});
