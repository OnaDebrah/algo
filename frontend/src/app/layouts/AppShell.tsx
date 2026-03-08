'use client'

import React, { lazy, Suspense, useState, useEffect, useCallback } from 'react';
import LoginPage from "@/components/auth/LoginPage";
import LandingPage from "@/components/landing/LandingPage";
import { TickerTape } from "@/components/widgets/TickerTape";
import { CommandPalette } from "@/components/common/CommandPalette";
import ErrorBoundary from "@/components/common/ErrorBoundary";
import Sidebar from "@/app/layouts/Sidebar";
import Header from "@/app/layouts/Header";
import { User } from "@/types/all_types";
import { api } from '@/utils/api';
import { useNavigationStore } from '@/store/useNavigationStore';

// ── Lazy-loaded page components (each gets its own chunk) ─────────
const Dashboard = lazy(() => import("@/components/dashboard/Dashboard"));
const BacktestPage = lazy(() => import("@/components/backtest/BacktestPage"));
const MLStudio = lazy(() => import("@/components/mlstudio/MLStudio"));
const LiveExecution = lazy(() => import("@/components/live/LiveExecution"));
const AIAdvisor = lazy(() => import("@/components/advisor/AIAdvisor"));
const RegimeDetector = lazy(() => import("@/components/regime/RegimeDetector"));
const CrashPredictionDashboard = lazy(() => import("@/components/crash/CrashPredictionDashboard"));
const AIAnalyst = lazy(() => import("@/components/aianalyst/AIAnalyst"));
const OptionsDesk = lazy(() => import("@/components/optionsdesk/OptionsDesk"));
const PortfolioOptimization = lazy(() => import("@/components/portfolio/PortfolioOptimisation"));
const SectorScanner = lazy(() => import("@/components/sector/SectorScanner"));
const StrategyBuilder = lazy(() => import("@/components/strategies/StrategyBuilder"));
const Marketplace = lazy(() => import("@/components/marketplace/Marketplace"));
const SettingsPage = lazy(() => import("@/components/settings/SettingsPage"));

export type PageKey =
  | 'dashboard'
  | 'backtest'
  | 'ml-studio'
  | 'live'
  | 'advisor'
  | 'regime'
  | 'crash-prediction'
  | 'analyst'
  | 'options'
  | 'portfolio'
  | 'sector-scanner'
  | 'strategy-builder'
  | 'marketplace'
  | 'settings';

const PAGE_COMPONENTS: Record<PageKey, React.ReactNode> = {
  dashboard: <Dashboard />,
  backtest: <BacktestPage />,
  'ml-studio': <MLStudio />,
  live: <LiveExecution />,
  advisor: <AIAdvisor />,
  regime: <RegimeDetector />,
  'crash-prediction': <CrashPredictionDashboard />,
  analyst: <AIAnalyst />,
  options: <OptionsDesk />,
  portfolio: <PortfolioOptimization />,
  'sector-scanner': <SectorScanner />,
  'strategy-builder': <StrategyBuilder />,
  marketplace: <Marketplace />,
  settings: <SettingsPage />,
};

type AppShellProps = object

const AppShell: React.FC<AppShellProps> = () => {
  const [currentPage, setCurrentPage] = useState<PageKey>('dashboard');
  const [user, setUser] = useState<User | null>(null);
  const [showLanding, setShowLanding] = useState(true);
  const [serverStatus, setServerStatus] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // Session restoration — cookie is sent automatically with the request
  const restoreSession = useCallback(async () => {
    try {
      // Call /auth/me — the httpOnly cookie is sent automatically.
      // No need to read localStorage for the token.
      const userData = await api.auth.getMe();

      setUser({
        id: userData.id,
        username: userData.username,
        email: userData.email,
        tier: userData.tier,
        is_active: userData.is_active,
        created_at: userData.created_at,
        last_login: userData.last_login,
        country: userData.country ?? null,
        investor_type: userData.investor_type ?? null,
        risk_profile: userData.risk_profile ?? null,
      });
    } catch {
      // No valid session — user will see login page
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Health check
  const checkHealth = useCallback(async () => {
    try {
      const healthData = await api.health.check();
      setServerStatus(healthData.status === 'ok' || healthData.status === '1');
    } catch (error) {
      console.error("Health check failed:", error);
      setServerStatus(false);
    }
  }, []);

  // Initial mount effects
  useEffect(() => {
    restoreSession().then();
  }, [restoreSession]);

  useEffect(() => {
    if (user) {
      // Initial health check
      checkHealth().then();

      // Set up polling interval
      const intervalId = setInterval(checkHealth, 30000); // Every 30 seconds

      return () => {
        clearInterval(intervalId);
      };
    }
  }, [user, checkHealth]);

  // Handle login — httpOnly cookie is set by the server response automatically
  const handleLogin = useCallback((userData: User, _token: string) => {
    setUser(userData);

    if ((userData?.tier ?? 'FREE') === 'FREE') {
      setCurrentPage('dashboard');
    }
  }, []);

  // Handle logout — server clears the httpOnly cookie
  const handleLogout = useCallback(() => {
    setUser(null);

    // Call logout endpoint to clear httpOnly cookies
    api.auth.logout().catch(console.error);

    // Redirect to login page
    setCurrentPage('dashboard');
  }, []);

  // Handle page change
  const handlePageChange = useCallback((page: PageKey) => {
    setCurrentPage(page);
  }, []);

  // Listen for programmatic navigation from child components via store
  const pendingPage = useNavigationStore(state => state.pendingPage);
  const clearPending = useNavigationStore(state => state.clearPending);

  useEffect(() => {
    if (pendingPage) {
      setCurrentPage(pendingPage);
      clearPending();
    }
  }, [pendingPage, clearPending]);

  // Show loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-slate-950">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-violet-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
          <p className="mt-4 text-slate-300">Restoring session...</p>
        </div>
      </div>
    );
  }

  // Show landing page or login page if no user
  if (!user) {
    if (showLanding) {
      return (
        <LandingPage
          onGetStarted={() => setShowLanding(false)}
          onSignIn={() => setShowLanding(false)}
        />
      );
    }
    return (
      <LoginPage
        onLogin={handleLogin}
        setCurrentPage={(page: string) => setCurrentPage(page as PageKey)}
        onBackToLanding={() => setShowLanding(true)}
      />
    );
  }

  return (
    <div className="flex min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-violet-500/30">
      <CommandPalette />
      <Sidebar
        currentPage={currentPage}
        setCurrentPage={handlePageChange}
        user={user}
        onLogout={handleLogout}
        serverStatus={serverStatus}
      />

      <div className="flex-1 ml-64 min-h-screen flex flex-col">
        {/* Global Ticker */}
        <div className="w-full border-b border-slate-800/80 bg-slate-900/50">
          <TickerTape />
        </div>

        <div className="max-w-[1600px] mx-auto w-full px-8 py-6 flex-1">
          <Header
            user={user}
            currentPage={currentPage}
            serverStatus={serverStatus}
            onLogout={handleLogout}
          />

          <main className="mt-6 animate-fade-in">
            <Suspense
              fallback={
                <div className="flex items-center justify-center min-h-[60vh]">
                  <div className="text-center">
                    <div className="w-10 h-10 border-3 border-violet-500 border-t-transparent rounded-full animate-spin mx-auto" />
                    <p className="mt-3 text-sm text-slate-400">Loading…</p>
                  </div>
                </div>
              }
            >
              <ErrorBoundary key={currentPage} onReset={() => setCurrentPage('dashboard')}>
                {PAGE_COMPONENTS[currentPage]}
              </ErrorBoundary>
            </Suspense>
          </main>

          {/* Status indicator */}
          {!serverStatus && (
            <div className="fixed bottom-4 right-4 bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-2 flex items-center gap-2">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-red-300">
                API Server Offline</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AppShell;
