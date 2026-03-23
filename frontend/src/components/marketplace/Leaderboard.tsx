/* eslint-disable @typescript-eslint/no-explicit-any */
'use client'

import React, {useCallback, useEffect, useState} from "react";
import {
    Award,
    BarChart3,
    ChevronDown,
    Crown,
    Download,
    Loader2,
    Medal,
    Star,
    TrendingUp,
    Trophy,
} from "lucide-react";
import {marketplace} from "@/utils/api";
import {LeaderboardEntry} from "@/types/all_types";

type Metric = 'sharpe_ratio' | 'total_return' | 'downloads' | 'rating' | 'win_rate';

const metricTabs: { key: Metric; label: string; icon: any }[] = [
    {key: 'sharpe_ratio', label: 'Best Sharpe', icon: BarChart3},
    {key: 'total_return', label: 'Best Return', icon: TrendingUp},
    {key: 'downloads', label: 'Most Downloaded', icon: Download},
    {key: 'rating', label: 'Top Rated', icon: Star},
    {key: 'win_rate', label: 'Highest Win Rate', icon: Award},
];

const rankBadge = (rank: number) => {
    if (rank === 1) return <Crown size={18} className="text-amber-400"/>;
    if (rank === 2) return <Medal size={18} className="text-slate-300"/>;
    if (rank === 3) return <Medal size={18} className="text-amber-600"/>;
    return <span className="text-sm font-bold text-slate-500 w-5 text-center">{rank}</span>;
};

const rankBg = (rank: number) => {
    if (rank === 1) return 'bg-amber-500/5 border-amber-500/20';
    if (rank === 2) return 'bg-slate-400/5 border-slate-400/20';
    if (rank === 3) return 'bg-amber-700/5 border-amber-700/20';
    return 'bg-slate-800/20 border-slate-800/30';
};

const Leaderboard = () => {
    const [metric, setMetric] = useState<Metric>('sharpe_ratio');
    const [category, setCategory] = useState<string>('All');
    const [entries, setEntries] = useState<LeaderboardEntry[]>([]);
    const [loading, setLoading] = useState(true);

    const categories = ['All', 'Momentum', 'Mean Reversion', 'Trend Following', 'Volatility', 'Statistical Arbitrage', 'Machine Learning'];

    const fetchLeaderboard = useCallback(async () => {
        setLoading(true);
        try {
            const res = await marketplace.leaderboard({
                metric,
                category: category === 'All' ? undefined : category,
                limit: 25,
            });
            setEntries(Array.isArray(res) ? res : []);
        } catch {
            setEntries([]);
        } finally {
            setLoading(false);
        }
    }, [metric, category]);

    useEffect(() => {
        fetchLeaderboard();
    }, [fetchLeaderboard]);

    const getMetricValue = (strategy: any) => {
        switch (metric) {
            case 'sharpe_ratio': return strategy.sharpe_ratio?.toFixed(2) ?? '—';
            case 'total_return': return (strategy.total_return?.toFixed(1) ?? '0') + '%';
            case 'downloads': return strategy.total_downloads?.toLocaleString() ?? '0';
            case 'rating': return strategy.rating?.toFixed(1) ?? '—';
            case 'win_rate': return (strategy.win_rate != null ? (strategy.win_rate * 100).toFixed(1) : '—') + '%';
            default: return '—';
        }
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center shadow-lg shadow-amber-500/20">
                        <Trophy size={24} className="text-white"/>
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-slate-100">Strategy Leaderboard</h1>
                        <p className="text-sm text-slate-500">Top performing strategies ranked by key metrics</p>
                    </div>
                </div>

                {/* Category filter */}
                <div className="relative">
                    <select
                        value={category}
                        onChange={(e) => setCategory(e.target.value)}
                        className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-4 py-2 text-sm text-slate-200 focus:border-violet-500 outline-none appearance-none pr-8"
                    >
                        {categories.map(c => (
                            <option key={c} value={c}>{c}</option>
                        ))}
                    </select>
                    <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none"/>
                </div>
            </div>

            {/* Metric tabs */}
            <div className="flex gap-2 bg-slate-800/30 rounded-xl p-1.5">
                {metricTabs.map(tab => {
                    const Icon = tab.icon;
                    const active = metric === tab.key;
                    return (
                        <button
                            key={tab.key}
                            onClick={() => setMetric(tab.key)}
                            className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
                                active
                                    ? 'bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white shadow-lg shadow-violet-500/20'
                                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                            }`}
                        >
                            <Icon size={14}/> {tab.label}
                        </button>
                    );
                })}
            </div>

            {/* Leaderboard table */}
            {loading ? (
                <div className="flex items-center justify-center py-20">
                    <Loader2 size={32} className="animate-spin text-violet-400"/>
                </div>
            ) : entries.length > 0 ? (
                <div className="space-y-2">
                    {entries.map((entry) => {
                        const s = entry.strategy;
                        return (
                            <div
                                key={`${entry.rank}-${s.id}`}
                                className={`flex items-center gap-4 p-4 border rounded-xl transition-all hover:scale-[1.005] ${rankBg(entry.rank)}`}
                            >
                                {/* Rank */}
                                <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-slate-800/50">
                                    {rankBadge(entry.rank)}
                                </div>

                                {/* Strategy info */}
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2">
                                        <p className="text-sm font-bold text-slate-200 truncate">{s.name}</p>
                                        {s.is_verified && (
                                            <span className="text-xs px-1.5 py-0.5 bg-blue-500/20 text-blue-400 rounded-full font-medium">Verified</span>
                                        )}
                                    </div>
                                    <p className="text-xs text-slate-500 mt-0.5">
                                        by {s.creator} · {s.category}
                                    </p>
                                </div>

                                {/* Key stats */}
                                <div className="flex items-center gap-6 text-right">
                                    <div>
                                        <p className="text-xs text-slate-500">Sharpe</p>
                                        <p className="text-sm font-semibold text-slate-200">{s.sharpe_ratio?.toFixed(2)}</p>
                                    </div>
                                    <div>
                                        <p className="text-xs text-slate-500">Return</p>
                                        <p className={`text-sm font-semibold ${(s.total_return ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                            {s.total_return?.toFixed(1)}%
                                        </p>
                                    </div>
                                    <div>
                                        <p className="text-xs text-slate-500">Drawdown</p>
                                        <p className="text-sm font-semibold text-red-400">-{Math.abs(s.max_drawdown ?? 0).toFixed(1)}%</p>
                                    </div>
                                    <div>
                                        <p className="text-xs text-slate-500">Downloads</p>
                                        <p className="text-sm font-semibold text-slate-200">{s.total_downloads?.toLocaleString()}</p>
                                    </div>
                                    <div>
                                        <p className="text-xs text-slate-500">Rating</p>
                                        <div className="flex items-center gap-1">
                                            <Star size={12} className="text-amber-400 fill-amber-400"/>
                                            <p className="text-sm font-semibold text-slate-200">{s.rating?.toFixed(1)}</p>
                                        </div>
                                    </div>

                                    {/* Primary metric highlighted */}
                                    <div className="ml-2 px-4 py-2 bg-violet-500/10 rounded-lg min-w-[80px]">
                                        <p className="text-xs text-violet-400 font-medium">{metricTabs.find(t => t.key === metric)?.label}</p>
                                        <p className="text-lg font-bold text-violet-300">{getMetricValue(s)}</p>
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            ) : (
                <div className="p-12 text-center bg-slate-800/20 border border-slate-800/30 rounded-xl">
                    <Trophy size={36} className="text-slate-700 mx-auto mb-3"/>
                    <p className="text-sm text-slate-500">No strategies found for this category</p>
                </div>
            )}
        </div>
    );
};

export default Leaderboard;
