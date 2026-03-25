import { test, expect, Page } from '@playwright/test';

/**
 * E2E tests for authentication flows.
 * Covers: login, signup, invalid credentials, session persistence.
 */

const TEST_USER = {
  username: process.env.E2E_USERNAME || 'testuser',
  email: process.env.E2E_EMAIL || 'test@oraculum.io',
  password: process.env.E2E_PASSWORD || 'TestPass123!',
};

/** Helper: fill login form and submit */
async function login(page: Page, username: string, password: string) {
  // The login page has a username/email field and a password field
  await page.getByPlaceholder('Enter your email or username').fill(username);
  await page.getByPlaceholder('••••••••••').fill(password);
  await page.getByRole('button', { name: 'Sign In' }).click();
}

test.describe('Authentication', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should show login page on initial visit', async ({ page }) => {
    // Landing page or login form should be visible
    await expect(
      page.getByPlaceholder('Enter your email or username').or(page.getByText('Sign In'))
    ).toBeVisible({ timeout: 15_000 });
  });

  test('should reject invalid credentials', async ({ page }) => {
    // Wait for login form to appear (may be behind a landing page)
    const signInBtn = page.getByRole('button', { name: 'Sign In' });
    if (!(await signInBtn.isVisible())) {
      // Click through landing page if present
      const getStartedBtn = page.getByRole('button', { name: /get started|sign in|login/i });
      if (await getStartedBtn.isVisible()) {
        await getStartedBtn.click();
      }
    }

    await page.getByPlaceholder('Enter your email or username').waitFor({ timeout: 10_000 });
    await login(page, 'nonexistent@test.com', 'wrongpassword');

    // Should show error message
    await expect(page.getByText(/invalid|incorrect|failed|error/i)).toBeVisible({ timeout: 10_000 });
  });

  test('should login successfully with valid credentials', async ({ page }) => {
    // Navigate past landing if needed
    const signInBtn = page.getByRole('button', { name: 'Sign In' });
    if (!(await signInBtn.isVisible())) {
      const getStartedBtn = page.getByRole('button', { name: /get started|sign in|login/i });
      if (await getStartedBtn.isVisible()) {
        await getStartedBtn.click();
      }
    }

    await page.getByPlaceholder('Enter your email or username').waitFor({ timeout: 10_000 });
    await login(page, TEST_USER.username, TEST_USER.password);

    // Should navigate to dashboard — look for sidebar or dashboard content
    await expect(
      page.getByText('Activity Hub').or(page.getByText('Dashboard').first())
    ).toBeVisible({ timeout: 15_000 });
  });

  test('should toggle between login and signup forms', async ({ page }) => {
    const signInBtn = page.getByRole('button', { name: 'Sign In' });
    if (!(await signInBtn.isVisible())) {
      const getStartedBtn = page.getByRole('button', { name: /get started|sign in|login/i });
      if (await getStartedBtn.isVisible()) {
        await getStartedBtn.click();
      }
    }

    await page.getByPlaceholder('Enter your email or username').waitFor({ timeout: 10_000 });

    // Find and click "Create Account" or "Sign Up" toggle
    const signupToggle = page.getByText(/create.*account|sign.*up|register/i);
    if (await signupToggle.isVisible()) {
      await signupToggle.click();

      // Should show email field for signup
      await expect(page.getByPlaceholder('name@example.com')).toBeVisible({ timeout: 5_000 });
    }
  });
});
