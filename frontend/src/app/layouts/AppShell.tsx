'use client'

import React, { useState, useEffect, useCallback } from 'react';
import LoginPage from "@/components/auth/LoginPage";
import Dashboard from "@/components/dashboard/Dashboard";
import PortfolioOptimization from "@/components/portfolio/PortfolioOptimisation";
import SettingsPage from "@/components/settings/SettingsPage";
import Marketplace from "@/components/marketplace/Marketplace";
import StrategyBuilder from "@/components/strategies/StrategyBuilder";
import MLStudio from "@/components/mlstudio/MLStudio";
import OptionsDesk from "@/components/optionsdesk/OptionsDesk";
import BacktestPage from "@/components/backtest/BacktestPage";
import AIAnalyst from "@/components/aianalyst/AIAnalyst";
import RegimeDetector from "@/components/regime/RegimeDetector";
import AIAdvisor from "@/components/advisor/AIAdvisor";
import LiveExecution from "@/components/live/LiveExecution";
import Sidebar from "@/app/layouts/Sidebar";
import Header from "@/app/layouts/Header";
import { User } from "@/types/all_types";
import { api } from '@/utils/api';

export type PageKey =
  | 'dashboard'
  | 'backtest'
  | 'ml-studio'
  | 'live'
  | 'advisor'
  | 'regime'
  | 'analyst'
  | 'options'
  | 'portfolio'
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
  analyst: <AIAnalyst />,
  options: <OptionsDesk />,
  portfolio: <PortfolioOptimization />,
  'strategy-builder': <StrategyBuilder />,
  marketplace: <Marketplace />,
  settings: <SettingsPage />,
};

type AppShellProps = object

const AppShell: React.FC<AppShellProps> = () => {
  const [currentPage, setCurrentPage] = useState<PageKey>('dashboard');
  const [user, setUser] = useState<User | null>(null);
  const [serverStatus, setServerStatus] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // Session restoration
  const restoreSession = useCallback(async () => {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        setIsLoading(false);
        return;
      }

      // Set token in API client
      api.setAuthToken(token);

      // Get user info
      const userData = await api.auth.getMe();

      setUser({
        id: userData.id,
        username: userData.username,
        email: userData.email,
        tier: userData.tier,
        is_active: userData.is_active,
        created_at: userData.created_at,
        last_login: userData.last_login,
      });

      // Store user in localStorage for quick access
      localStorage.setItem('user', JSON.stringify(userData));
    } catch (error) {
      console.error("Session restoration failed:", error);
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      localStorage.removeItem('user');
      api.clearAuthToken();
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

  // Handle login
  const handleLogin = useCallback((userData: User, token: string) => {
    setUser(userData);
    api.setAuthToken(token);

    // Store tokens and user data
    localStorage.setItem('access_token', token);
    localStorage.setItem('user', JSON.stringify(userData));

    if (userData.tier === 'FREE') {
      setCurrentPage('dashboard');
    }
  }, []);

  // Handle logout
  const handleLogout = useCallback(() => {
    setUser(null);
    api.clearAuthToken();
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');

    // Optional: call logout endpoint
    api.auth.logout().catch(console.error);

    // Redirect to login page
    setCurrentPage('dashboard');
  }, []);

  // Handle page change
  const handlePageChange = useCallback((page: PageKey) => {
    setCurrentPage(page);
  }, []);

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

  // Show login page if no user
  if (!user) {
    return (
      <LoginPage
        onLogin={handleLogin}
        setCurrentPage={(page: string) => setCurrentPage(page as PageKey)}
      />
    );
  }

  return (
    <div className="flex min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-violet-500/30">
      <Sidebar
        currentPage={currentPage}
        setCurrentPage={handlePageChange}
        user={user}
        onLogout={handleLogout}
        serverStatus={serverStatus}
      />

      <div className="flex-1 ml-64 min-h-screen">
        <div className="max-w-[1600px] mx-auto px-8 py-6">
          <Header
            user={user}
            currentPage={currentPage}
            serverStatus={serverStatus}
            onLogout={handleLogout}
          />

          <main className="mt-6 animate-fade-in">
            {PAGE_COMPONENTS[currentPage]}
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
