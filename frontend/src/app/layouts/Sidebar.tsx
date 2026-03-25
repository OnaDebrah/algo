/* eslint-disable @typescript-eslint/no-explicit-any */

import React, {useState} from "react";
import {
    Activity,
    Banknote,
    BarChart3,
    Bitcoin,
    BookOpen,
    ChevronDown,
    Cpu,
    CreditCard,
    DollarSign,
    Eye,
    LogOut,
    Package,
    Settings,
    Shield,
    ShieldCheck,
    Target,
    TrendingUp,
    Trophy,
    Users,
    Zap
} from "lucide-react";

const Sidebar = ({currentPage, setCurrentPage, user, onLogout}: any) => {
    const [expandedSections, setExpandedSections] = useState(['MONITOR', 'ANALYZE', 'RESEARCH', 'BUILD']);
    const [showUserMenu, setShowUserMenu] = useState(false);

    const menuSections = [
        {
            id: 'monitor',
            title: 'MONITOR',
            items: [
                {id: 'dashboard', icon: Activity, label: 'Activity Hub'},
                {id: 'watchlist', icon: Eye, label: 'Watchlist'},
                {id: 'live', icon: Zap, label: 'Live Execution'},
                {id: 'paper-trading', icon: Banknote, label: 'Paper Trading'},
                {id: 'trade-journal', icon: BookOpen, label: 'Trade Journal'}
            ]
        },
        {
            id: 'analyze',
            title: 'ANALYZE',
            items: [
                {id: 'advisor', icon: Zap, label: 'AI Strategy Advisor'},
                {id: 'regime', icon: TrendingUp, label: 'Market Regime'},
                {id: 'crash-prediction', icon: Shield, label: 'Crash Prediction'},
                {id: 'analyst', icon: Activity, label: 'Deep Analyst'},
                {id: 'sector-scanner', icon: Target, label: 'Sector Scanner'},
                {id: 'crypto', icon: Bitcoin, label: 'Crypto Markets'},
                {id: 'economic-calendar', icon: Activity, label: 'Economic Calendar'}
            ]
        },
        {
            id: 'research',
            title: 'RESEARCH',
            items: [
                {id: 'backtest', icon: BarChart3, label: 'Backtesting Lab'},
                {id: 'options', icon: DollarSign, label: 'Options Desk'},
                {id: 'portfolio', icon: TrendingUp, label: 'Portfolio Optimization'}
            ]
        },
        {
            id: 'build',
            title: 'BUILD',
            items: [
                {id: 'ml-studio', icon: Cpu, label: 'ML Strategy Studio'},
                {id: 'strategy-builder', icon: Target, label: 'Strategy Builder'},
                {id: 'research', icon: BookOpen, label: 'Research Lab'}
            ]
        }
    ];

    const bottomItems = [
        {id: 'marketplace', icon: Package, label: 'Marketplace'},
        {id: 'leaderboard', icon: Trophy, label: 'Leaderboard'},
        {id: 'teams', icon: Users, label: 'Teams'},
        {id: 'getting-started', icon: BookOpen, label: 'Getting Started'},
    ];

    const toggleSection = (sectionTitle: string) => {
        setExpandedSections(prev =>
            prev.includes(sectionTitle) ? prev.filter(s => s !== sectionTitle) : [...prev, sectionTitle]
        );
    };

    return (
        <aside
            data-tour="sidebar"
            className="fixed left-0 top-0 h-full w-64 bg-sidebar backdrop-blur-xl border-r border-sidebar-border flex flex-col z-50">
            {/* Brand Section */}
            <div className="p-6 border-b border-sidebar-border">
                <div className="flex items-center space-x-3">
                    <div
                        className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center shadow-lg shadow-violet-500/30">
                        <Activity size={20} className="text-white" strokeWidth={2.5}/>
                    </div>
                    <div>
                        <span className="text-lg font-bold tracking-tight flex items-center gap-2">
                        <span className="text-sidebar-foreground">ORACULUM</span>
                        <span
                            className="text-xs font-semibold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
                        AI
                        </span>
                        </span>
                        <div
                            className="h-0.5 w-10 bg-gradient-to-r from-violet-500 to-fuchsia-500 rounded-full"/>
                    </div>
                </div>
            </div>

            {/* Navigation */}
            <div className="flex-1 overflow-y-auto px-3 py-4 space-y-6">
                <nav className="space-y-6">
                    {menuSections.map((section) => {
                        const isExpanded = expandedSections.includes(section.title);
                        return (
                            <div key={section.id} className="space-y-1">
                                <button
                                    onClick={() => toggleSection(section.title)}
                                    className="w-full flex items-center justify-between px-3 py-2 text-[11px] font-bold text-muted-foreground hover:text-sidebar-foreground transition-colors tracking-wider"
                                >
                                    <span>{section.title}</span>
                                    <ChevronDown
                                        size={14}
                                        className={`transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`}
                                    />
                                </button>

                                {isExpanded && (
                                    <div className="space-y-0.5">
                                        {section.items.map((item) => {
                                            const isActive = currentPage === item.id;
                                            return (
                                                <button
                                                    key={item.id}
                                                    onClick={() => setCurrentPage(item.id)}
                                                    data-tour={
                                                        item.id === 'dashboard' ? 'activity-hub' :
                                                        item.id === 'strategy-builder' ? 'strategy-builder' :
                                                        item.id === 'backtest' ? 'backtest-lab' :
                                                        item.id === 'options' ? 'options-desk' :
                                                        item.id === 'live' ? 'live-execution' :
                                                        item.id === 'watchlist' ? 'watchlist' :
                                                        undefined
                                                    }
                                                    className={`w-full group flex items-center px-3 py-2.5 rounded-lg text-sm transition-all relative ${isActive
                                                        ? 'text-sidebar-foreground bg-sidebar-accent'
                                                        : 'text-sidebar-accent-foreground hover:text-sidebar-foreground hover:bg-sidebar-accent/50'
                                                    }`}
                                                >
                                                    {isActive && (
                                                        <div
                                                            className="absolute left-0 w-1 h-5 bg-violet-500 rounded-r-full"/>
                                                    )}
                                                    <item.icon
                                                        size={18}
                                                        className={`mr-3 transition-colors ${isActive ? 'text-sidebar-primary' : 'text-muted-foreground group-hover:text-sidebar-accent-foreground'
                                                        }`}
                                                        strokeWidth={2}
                                                    />
                                                    <span className="font-medium">{item.label}</span>
                                                </button>
                                            );
                                        })}
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </nav>

                {/* System Section */}
                <div className="space-y-1 pt-4 border-t border-sidebar-border">
                    <p className="px-3 text-[11px] font-bold text-muted-foreground tracking-wider mb-2">COMMUNITY</p>
                    {bottomItems.map((item) => (
                        <button
                            key={item.id}
                            onClick={() => setCurrentPage(item.id)}
                            data-tour={item.id === 'marketplace' ? 'marketplace' : item.id === 'leaderboard' ? 'leaderboard' : undefined}
                            className={`w-full flex items-center px-3 py-2.5 rounded-lg text-sm transition-all ${currentPage === item.id
                                ? 'text-sidebar-foreground bg-sidebar-accent'
                                : 'text-sidebar-accent-foreground hover:text-sidebar-foreground hover:bg-sidebar-accent/50'
                            }`}
                        >
                            <item.icon size={18} className="mr-3 text-muted-foreground" strokeWidth={2}/>
                            <span className="font-medium">{item.label}</span>
                        </button>
                    ))}
                </div>
            </div>

            {/* User Profile Footer */}
            <div className="p-3 border-t border-sidebar-border relative">
                {showUserMenu && (
                    <div
                        className="absolute bottom-full left-3 right-3 mb-2 glass-panel rounded-xl overflow-hidden animate-in slide-in-from-bottom-2 fade-in duration-200">
                        <div className="p-3 space-y-1">
                            <button
                                onClick={() => { setCurrentPage('settings'); setShowUserMenu(false); }}
                                className="w-full flex items-center space-x-3 px-3 py-2 text-sm text-slate-300 hover:bg-slate-700/50 rounded-lg transition-colors">
                                <Settings size={16} className="text-violet-400"/>
                                <span>Account Settings</span>
                            </button>
                            <button
                                onClick={() => { setCurrentPage('pricing'); setShowUserMenu(false); }}
                                className="w-full flex items-center space-x-3 px-3 py-2 text-sm text-slate-300 hover:bg-slate-700/50 rounded-lg transition-colors">
                                <CreditCard size={16} className="text-emerald-400"/>
                                <span>Billing</span>
                            </button>
                            {user?.is_superuser && (
                                <button
                                    onClick={() => { setCurrentPage('admin'); setShowUserMenu(false); }}
                                    className="w-full flex items-center space-x-3 px-3 py-2 text-sm text-slate-300 hover:bg-slate-700/50 rounded-lg transition-colors">
                                    <ShieldCheck size={16} className="text-amber-400"/>
                                    <span>Admin</span>
                                </button>
                            )}
                        </div>
                        <div className="border-t border-slate-700/80 p-2">
                            <button
                                onClick={onLogout}
                                className="w-full flex items-center space-x-3 px-3 py-2 text-sm text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
                            >
                                <LogOut size={16}/>
                                <span className="font-medium">Sign Out</span>
                            </button>
                        </div>
                    </div>
                )}

                <button
                    onClick={() => setShowUserMenu(!showUserMenu)}
                    className={`w-full rounded-xl p-3 transition-all border backdrop-blur-md group ${showUserMenu
                        ? 'bg-white/10 border-primary-500/50'
                        : 'bg-white/5 border-white/5 hover:bg-white/10 hover:border-white/10'
                    }`}
                >
                    <div className="flex items-center space-x-3">
                        <div className="relative">
                            <div
                                className="w-10 h-10 rounded-full bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center text-white font-bold text-sm border-2 border-slate-900">
                                {user?.username?.charAt(0)?.toUpperCase() || 'U'}
                            </div>
                            <div
                                className="absolute bottom-0 right-0 w-3 h-3 bg-emerald-500 border-2 border-slate-900 rounded-full"/>
                        </div>
                        <div className="flex-1 text-left overflow-hidden">
                            <div className="flex items-center justify-between">
                                <p className="text-sm font-semibold text-slate-200 truncate">{user?.username || 'Trader'}</p>
                                <ChevronDown
                                    size={14}
                                    className={`text-slate-500 transition-transform ${showUserMenu ? 'rotate-180' : ''}`}
                                />
                            </div>
                            <p className="text-[10px] text-violet-400 font-medium tracking-wider">{user?.tier || 'INSTITUTIONAL'}</p>
                        </div>
                    </div>
                </button>
            </div>
        </aside>
    );
};

export default Sidebar;
