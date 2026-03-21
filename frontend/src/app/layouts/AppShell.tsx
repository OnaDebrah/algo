'use client'

import React, {lazy, Suspense, useCallback, useEffect, useMemo, useState} from 'react';
import {Activity, BarChart3, Briefcase, Menu, Settings, X, Zap} from 'lucide-react';
import LoginPage from "@/components/auth/LoginPage";
import ResetPasswordPage from "@/components/auth/ResetPasswordPage";
import LandingPage from "@/components/landing/LandingPage";
import {TickerTape} from "@/components/widgets/TickerTape";
import {CommandPalette} from "@/components/common/CommandPalette";
import ErrorBoundary from "@/components/common/ErrorBoundary";
import Sidebar from "@/app/layouts/Sidebar";
import Header from "@/app/layouts/Header";
import {User} from "@/types/all_types";
import {api} from '@/utils/api';
import {useNavigationStore} from '@/store/useNavigationStore';
import {UserProvider} from '@/contexts/UserContext';
import TourOverlay, {isTourCompleted} from '@/components/onboarding/TourOverlay';

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
const WatchlistPage = lazy(() => import("@/components/watchlist/WatchlistPage"));
const Leaderboard = lazy(() => import("@/components/marketplace/Leaderboard"));
const AdminDashboard = lazy(() => import("@/components/admin/AdminDashboard"));
const PricingPage = lazy(() => import("@/components/pricing/PricingPage"));
const SettingsPage = lazy(() => import("@/components/settings/SettingsPage"));
const PaperTradingDashboard = lazy(() => import("@/components/paper/PaperTradingDashboard"));
const TradeJournal = lazy(() => import("@/components/journal/TradeJournal"));
const EconomicCalendar = lazy(() => import("@/components/calendar/EconomicCalendar"));
const TeamDashboard = lazy(() => import("@/components/teams/TeamDashboard"));

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
  | 'watchlist'
  | 'leaderboard'
  | 'admin'
  | 'pricing'
  | 'settings'
  | 'paper-trading'
  | 'trade-journal'
  | 'economic-calendar'
  | 'teams';

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
  watchlist: <WatchlistPage />,
  leaderboard: <Leaderboard />,
  admin: <AdminDashboard />,
  pricing: <PricingPage />,
  settings: <SettingsPage />,
  'paper-trading': <PaperTradingDashboard />,
  'trade-journal': <TradeJournal />,
  'economic-calendar': <EconomicCalendar />,
  teams: <TeamDashboard />,
};

type AppShellProps = object

const AppShell: React.FC<AppShellProps> = () => {
  const [currentPage, setCurrentPage] = useState<PageKey>('dashboard');
  const [user, setUser] = useState<User | null>(null);
  const [showLanding, setShowLanding] = useState(true);
  const [authMode, setAuthMode] = useState<'login' | 'signup'>('login');
  const [serverStatus, setServerStatus] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [resetToken, setResetToken] = useState<string | null>(null);
  const [showTour, setShowTour] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

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
        is_superuser: userData.is_superuser ?? false,
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

  // Check for password reset token in URL
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const token = params.get('reset_token');
    if (token) {
      setResetToken(token);
      setShowLanding(false);
      // Clean up the URL without reloading
      window.history.replaceState({}, '', window.location.pathname);
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

    // Trigger onboarding tour for first-time users
    if (!isTourCompleted()) {
      setTimeout(() => setShowTour(true), 1000);
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

  const mobileNavItems = useMemo(() => [
    { id: 'dashboard' as PageKey, icon: Activity, label: 'Activity' },
    { id: 'backtest' as PageKey, icon: BarChart3, label: 'Backtest' },
    { id: 'live' as PageKey, icon: Zap, label: 'Live' },
    { id: 'portfolio' as PageKey, icon: Briefcase, label: 'Portfolio' },
    { id: 'settings' as PageKey, icon: Settings, label: 'Settings' },
  ], []);

  // Show loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-violet-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
          <p className="mt-4 text-muted-foreground">Restoring session...</p>
        </div>
      </div>
    );
  }

  // Show reset password page if token present
  if (resetToken) {
    return (
      <ResetPasswordPage
        token={resetToken}
        onBackToLogin={() => { setResetToken(null); setShowLanding(false); }}
      />
    );
  }

  // Show landing page or login page if no user
  if (!user) {
    if (showLanding) {
      return (
        <LandingPage
          onGetStarted={() => { setAuthMode('signup'); setShowLanding(false); }}
          onSignIn={() => { setAuthMode('login'); setShowLanding(false); }}
        />
      );
    }
    return (
      <LoginPage
        onLogin={handleLogin}
        setCurrentPage={(page: string) => setCurrentPage(page as PageKey)}
        onBackToLanding={() => setShowLanding(true)}
        initialMode={authMode}
      />
    );
  }

  return (
    <div className="flex min-h-screen bg-background text-foreground font-sans selection:bg-violet-500/30">
      <CommandPalette />

      {/* Mobile sidebar backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar — hidden on mobile, slide-in drawer */}
      <div className={`
        fixed inset-y-0 left-0 z-50 transform transition-transform duration-300 ease-in-out
        md:translate-x-0
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <Sidebar
          currentPage={currentPage}
          setCurrentPage={(page: PageKey) => { handlePageChange(page); setSidebarOpen(false); }}
          user={user}
          onLogout={handleLogout}
          serverStatus={serverStatus}
        />
      </div>

      <div className="flex-1 md:ml-64 min-h-screen flex flex-col pb-16 md:pb-0">
        {/* Mobile header bar */}
        <div className="md:hidden flex items-center justify-between px-4 py-3 border-b border-border bg-card/80 backdrop-blur-sm sticky top-0 z-30">
          <button onClick={() => setSidebarOpen(true)} className="p-1.5 rounded-lg hover:bg-accent text-foreground">
            <Menu size={22} />
          </button>
          <span className="text-sm font-bold tracking-tight text-foreground">ORACULUM <span className="text-xs bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">AI</span></span>
          <button onClick={() => setSidebarOpen(false)} className="p-1.5 rounded-lg opacity-0 pointer-events-none">
            <X size={22} />
          </button>
        </div>

        {/* Global Ticker — hidden on mobile for space */}
        <div className="hidden md:block w-full border-b border-border bg-card/50">
          <TickerTape />
        </div>

        <div className="max-w-[1600px] mx-auto w-full px-4 md:px-8 py-4 md:py-6 flex-1">
          <div className="hidden md:block">
            <Header
              user={user}
              currentPage={currentPage}
              serverStatus={serverStatus}
              onLogout={handleLogout}
            />
          </div>

          <main className="mt-2 md:mt-6 animate-fade-in">
            <UserProvider user={user}>
              <Suspense
                fallback={
                  <div className="flex items-center justify-center min-h-[60vh]">
                    <div className="text-center">
                      <div className="w-10 h-10 border-3 border-violet-500 border-t-transparent rounded-full animate-spin mx-auto" />
                      <p className="mt-3 text-sm text-slate-400">Loading...</p>
                    </div>
                  </div>
                }
              >
                <ErrorBoundary key={currentPage} onReset={() => setCurrentPage('dashboard')}>
                  {PAGE_COMPONENTS[currentPage]}
                </ErrorBoundary>
              </Suspense>
            </UserProvider>
          </main>

          {/* Status indicator */}
          {!serverStatus && (
            <div className="fixed bottom-20 md:bottom-4 right-4 bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-2 flex items-center gap-2">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-red-300">API Server Offline</span>
            </div>
          )}
        </div>
      </div>

      {/* Mobile Bottom Nav */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-card/95 backdrop-blur-lg border-t border-border z-40 px-2 py-1 safe-area-pb">
        <div className="flex items-center justify-around">
          {mobileNavItems.map(item => {
            const isActive = currentPage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => handlePageChange(item.id)}
                className={`flex flex-col items-center py-1.5 px-3 rounded-lg transition-colors ${
                  isActive ? 'text-violet-400' : 'text-muted-foreground'
                }`}
              >
                <item.icon size={20} strokeWidth={isActive ? 2.5 : 2} />
                <span className="text-[10px] mt-0.5 font-medium">{item.label}</span>
              </button>
            );
          })}
        </div>
      </nav>

      {/* Onboarding Tour */}
      {showTour && (
        <TourOverlay
          onComplete={() => setShowTour(false)}
          onNavigate={(page) => setCurrentPage(page as PageKey)}
        />
      )}
    </div>
  );
};

export default AppShell;
