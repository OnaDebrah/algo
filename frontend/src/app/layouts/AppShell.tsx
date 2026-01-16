'use client'

import React, { useState } from 'react';
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
import { User } from "@/types/user";

const PAGE_COMPONENTS: Record<string, React.ReactNode> = {
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
    default: <Dashboard />

};
const AppShell = () => {
    const [currentPage, setCurrentPage] = useState('dashboard');
    const [user, setUser] = useState<User | null>(null);
    const [serverStatus, setServerStatus] = useState<boolean>(false);

    // 1. Session Restoration - Run once on mount
    React.useEffect(() => {
        const restoreSession = async () => {
            const token = localStorage.getItem('token');
            if (token) {
                try {
                    const { auth } = await import('@/utils/api');
                    const meRes = await auth.getMe();
                    if (meRes.data) {
                        setUser({
                            name: meRes.data.username,
                            email: meRes.data.email,
                            tier: meRes.data.tier || 'FREE',
                            token: token
                        });
                    }
                } catch (e) {
                    console.error("Session restoration failed:", e);
                    localStorage.removeItem('token');
                }
            }
        };
        restoreSession();
    }, []);

    // 2. Health Check Polling - Run on mount
    React.useEffect(() => {
        const checkHealth = async () => {
            try {
                fetch('http://localhost:8000/health')
                    .then(res => setServerStatus(res.ok))
                    .catch(() => setServerStatus(false));
            } catch (e) {
                setServerStatus(false);
            }
        };
        checkHealth();
        const interval = setInterval(checkHealth, 30000);
        return () => clearInterval(interval);
    }, []);

    if (!user) return <LoginPage onLogin={setUser} setCurrentPage={setCurrentPage} />;

    return (
        <div className="flex min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-violet-500/30">
            <Sidebar
                currentPage={currentPage}
                setCurrentPage={setCurrentPage}
                user={user}
                onLogout={() => setUser(null)}
            />

            <div className="flex-1 ml-64 min-h-screen">
                <div className="max-w-[1600px] mx-auto px-8 py-6">
                    <Header user={user} currentPage={currentPage} serverStatus={serverStatus} />
                    <main className="mt-6">
                        {PAGE_COMPONENTS[currentPage] || <Dashboard />}
                    </main>
                </div>
            </div>
        </div>
    );
};

export default AppShell;