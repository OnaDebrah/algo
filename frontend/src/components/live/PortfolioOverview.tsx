/**
 * Portfolio Overview
 * Compare performance across multiple live strategies
 */

'use client'

import React, {useEffect, useState} from 'react';
import {
    Activity,
    AlertCircle,
    CheckCircle,
    Clock,
    DollarSign,
    Download,
    PieChart,
    Target,
    TrendingUp
} from 'lucide-react';
import {PortfolioMetrics, StrategyPerformance} from "@/types/live";


export default function PortfolioOverview() {
    const [strategies, setStrategies] = useState<StrategyPerformance[]>([]);
    const [metrics, setMetrics] = useState<PortfolioMetrics | null>(null);
    const [timeframe, setTimeframe] = useState<'1D' | '1W' | '1M' | 'ALL'>('1D');
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadPortfolio();

        // Refresh every 30 seconds
        const interval = setInterval(loadPortfolio, 30000);
        return () => clearInterval(interval);
    }, []);

    const loadPortfolio = async () => {
        try {
            const response = await fetch('/api/live/strategies', {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            });

            if (response.ok) {
                const data = await response.json();
                setStrategies(data);
                calculateMetrics(data);
            }
        } catch (error) {
            console.error('Error loading portfolio:', error);
        } finally {
            setLoading(false);
        }
    };

    const calculateMetrics = (strats: StrategyPerformance[]) => {
        const total_equity = strats.reduce((sum, s) => sum + s.current_equity, 0);
        const total_invested = strats.reduce((sum, s) => sum + s.initial_capital, 0);
        const total_pnl = total_equity - total_invested;
        const total_pnl_pct = (total_pnl / total_invested) * 100;
        const active_strategies = strats.filter(s => s.status === 'running').length;

        const best = strats.reduce((best, curr) =>
                curr.total_return_pct > (best?.total_return_pct || -Infinity) ? curr : best
            , null as StrategyPerformance | null);

        const worst = strats.reduce((worst, curr) =>
                curr.total_return_pct < (worst?.total_return_pct || Infinity) ? curr : worst
            , null as StrategyPerformance | null);

        setMetrics({
            total_equity,
            total_invested,
            total_pnl,
            total_pnl_pct,
            active_strategies,
            total_strategies: strats.length,
            best_performer: best,
            worst_performer: worst
        });
    };

    const exportData = () => {
        const csv = [
            ['Name', 'Status', 'Mode', 'Equity', 'Return %', 'Sharpe', 'Trades', 'Win Rate'].join(','),
            ...strategies.map(s => [
                s.name,
                s.status,
                s.deployment_mode,
                s.current_equity,
                s.total_return_pct.toFixed(2),
                s.sharpe_ratio?.toFixed(2) || 'N/A',
                s.total_trades,
                s.win_rate.toFixed(1)
            ].join(','))
        ].join('\n');

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `portfolio_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-screen">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-violet-500"></div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-slate-950 p-6">
            <div className="max-w-7xl mx-auto space-y-6">

                {/* Header */}
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-3xl font-bold text-slate-100 flex items-center gap-2">
                            <PieChart size={32} className="text-violet-500" />
                            Portfolio Overview
                        </h1>
                        <p className="text-slate-400 mt-1">Monitor all your live trading strategies</p>
                    </div>

                    <div className="flex items-center gap-3">
                        {/* Timeframe Selector */}
                        <div className="flex bg-slate-900 rounded-lg border border-slate-800 p-1">
                            {(['1D', '1W', '1M', 'ALL'] as const).map(tf => (
                                <button
                                    key={tf}
                                    onClick={() => setTimeframe(tf)}
                                    className={`px-4 py-2 rounded-md text-sm font-semibold transition-all ${
                                        timeframe === tf
                                            ? 'bg-violet-600 text-white'
                                            : 'text-slate-400 hover:text-slate-300'
                                    }`}
                                >
                                    {tf}
                                </button>
                            ))}
                        </div>

                        <button
                            onClick={exportData}
                            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg border border-slate-700 text-slate-300 flex items-center gap-2"
                        >
                            <Download size={16} />
                            Export
                        </button>
                    </div>
                </div>

                {/* Portfolio Summary Cards */}
                {metrics && (
                    <>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                            {/* Total Equity */}
                            <div className="bg-gradient-to-br from-slate-900 to-slate-950 rounded-xl p-6 border border-slate-800">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-xs font-bold text-slate-500 uppercase">Total Equity</span>
                                    <DollarSign size={16} className="text-emerald-500" />
                                </div>
                                <div className="text-2xl font-black text-slate-100">
                                    ${metrics.total_equity.toLocaleString()}
                                </div>
                                <div className="text-xs text-slate-400 mt-1">
                                    Invested: ${metrics.total_invested.toLocaleString()}
                                </div>
                            </div>

                            {/* Total P&L */}
                            <div className="bg-gradient-to-br from-slate-900 to-slate-950 rounded-xl p-6 border border-slate-800">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-xs font-bold text-slate-500 uppercase">Total P&L</span>
                                    <TrendingUp size={16} className={metrics.total_pnl >= 0 ? 'text-emerald-500' : 'text-red-500'} />
                                </div>
                                <div className={`text-2xl font-black ${metrics.total_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                    {metrics.total_pnl >= 0 ? '+' : ''}${metrics.total_pnl.toFixed(2)}
                                </div>
                                <div className={`text-sm font-semibold mt-1 ${metrics.total_pnl_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                    {metrics.total_pnl_pct >= 0 ? '+' : ''}{metrics.total_pnl_pct.toFixed(2)}%
                                </div>
                            </div>

                            {/* Active Strategies */}
                            <div className="bg-gradient-to-br from-slate-900 to-slate-950 rounded-xl p-6 border border-slate-800">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-xs font-bold text-slate-500 uppercase">Active Strategies</span>
                                    <Activity size={16} className="text-blue-500" />
                                </div>
                                <div className="text-2xl font-black text-slate-100">
                                    {metrics.active_strategies}/{metrics.total_strategies}
                                </div>
                                <div className="text-xs text-slate-400 mt-1">
                                    {((metrics.active_strategies / metrics.total_strategies) * 100).toFixed(0)}% Running
                                </div>
                            </div>

                            {/* Best Performer */}
                            <div className="bg-gradient-to-br from-slate-900 to-slate-950 rounded-xl p-6 border border-slate-800">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-xs font-bold text-slate-500 uppercase">Best Performer</span>
                                    <Target size={16} className="text-violet-500" />
                                </div>
                                {metrics.best_performer ? (
                                    <>
                                        <div className="text-sm font-semibold text-slate-100 truncate">
                                            {metrics.best_performer.name}
                                        </div>
                                        <div className="text-lg font-bold text-emerald-400 mt-1">
                                            +{metrics.best_performer.total_return_pct.toFixed(2)}%
                                        </div>
                                    </>
                                ) : (
                                    <div className="text-sm text-slate-500">No data</div>
                                )}
                            </div>
                        </div>

                        {/* Performance Breakdown */}
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                            {/* Allocation Pie Chart Placeholder */}
                            <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-800">
                                <h3 className="text-lg font-bold text-slate-100 mb-4">Allocation</h3>
                                <div className="aspect-square flex items-center justify-center">
                                    <PieChart size={120} className="text-slate-700" />
                                </div>
                            </div>

                            {/* Top Performers */}
                            <div className="lg:col-span-2 bg-slate-900/50 rounded-xl p-6 border border-slate-800">
                                <h3 className="text-lg font-bold text-slate-100 mb-4">Performance Rankings</h3>
                                <div className="space-y-3">
                                    {strategies
                                        .sort((a, b) => b.total_return_pct - a.total_return_pct)
                                        .slice(0, 5)
                                        .map((strategy, idx) => (
                                            <div
                                                key={strategy.id}
                                                className="flex items-center gap-4 p-3 bg-slate-800/50 rounded-lg hover:bg-slate-800 transition-colors"
                                            >
                                                <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${
                                                    idx === 0 ? 'bg-amber-500/20 text-amber-400' :
                                                        idx === 1 ? 'bg-slate-500/20 text-slate-400' :
                                                            idx === 2 ? 'bg-orange-700/20 text-orange-600' :
                                                                'bg-slate-700 text-slate-500'
                                                }`}>
                                                    {idx + 1}
                                                </div>

                                                <div className="flex-1">
                                                    <div className="font-semibold text-slate-200">{strategy.name}</div>
                                                    <div className="text-xs text-slate-500">
                                                        {strategy.total_trades} trades • {strategy.win_rate.toFixed(1)}% win rate
                                                    </div>
                                                </div>

                                                <div className="text-right">
                                                    <div className={`font-bold ${
                                                        strategy.total_return_pct >= 0 ? 'text-emerald-400' : 'text-red-400'
                                                    }`}>
                                                        {strategy.total_return_pct >= 0 ? '+' : ''}{strategy.total_return_pct.toFixed(2)}%
                                                    </div>
                                                    <div className="text-xs text-slate-500">
                                                        ${strategy.current_equity.toLocaleString()}
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                </div>
                            </div>
                        </div>
                    </>
                )}

                {/* Strategy Comparison Table */}
                <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-800">
                    <h3 className="text-lg font-bold text-slate-100 mb-6">All Strategies</h3>

                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                            <tr className="border-b border-slate-800">
                                <th className="text-left pb-3 text-xs font-bold text-slate-500 uppercase">Strategy</th>
                                <th className="text-left pb-3 text-xs font-bold text-slate-500 uppercase">Status</th>
                                <th className="text-right pb-3 text-xs font-bold text-slate-500 uppercase">Equity</th>
                                <th className="text-right pb-3 text-xs font-bold text-slate-500 uppercase">Return</th>
                                <th className="text-right pb-3 text-xs font-bold text-slate-500 uppercase">Daily P&L</th>
                                <th className="text-right pb-3 text-xs font-bold text-slate-500 uppercase">Sharpe</th>
                                <th className="text-right pb-3 text-xs font-bold text-slate-500 uppercase">Trades</th>
                                <th className="text-right pb-3 text-xs font-bold text-slate-500 uppercase">Win Rate</th>
                                <th className="text-right pb-3 text-xs font-bold text-slate-500 uppercase">Max DD</th>
                            </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-800/50">
                            {strategies.map(strategy => (
                                <tr key={strategy.id} className="group hover:bg-white/5">
                                    <td className="py-4">
                                        <div className="font-semibold text-slate-200">{strategy.name}</div>
                                        <div className="text-xs text-slate-500 mt-0.5">
                                            {strategy.deployment_mode === 'paper' ? '📋 Paper' : '🚀 Live'}
                                        </div>
                                    </td>
                                    <td className="py-4">
                      <span className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-semibold ${
                          strategy.status === 'running' ? 'bg-emerald-500/20 text-emerald-400' :
                              strategy.status === 'paused' ? 'bg-amber-500/20 text-amber-400' :
                                  strategy.status === 'error' ? 'bg-red-500/20 text-red-400' :
                                      'bg-slate-700 text-slate-400'
                      }`}>
                        {strategy.status === 'running' && <CheckCircle size={12} />}
                          {strategy.status === 'paused' && <Clock size={12} />}
                          {strategy.status === 'error' && <AlertCircle size={12} />}
                          {strategy.status.toUpperCase()}
                      </span>
                                    </td>
                                    <td className="py-4 text-right font-mono text-sm text-slate-200">
                                        ${strategy.current_equity.toLocaleString()}
                                    </td>
                                    <td className="py-4 text-right">
                                        <div className={`font-semibold ${
                                            strategy.total_return_pct >= 0 ? 'text-emerald-400' : 'text-red-400'
                                        }`}>
                                            {strategy.total_return_pct >= 0 ? '+' : ''}{strategy.total_return_pct.toFixed(2)}%
                                        </div>
                                    </td>
                                    <td className="py-4 text-right">
                                        <div className={`font-semibold ${
                                            strategy.daily_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'
                                        }`}>
                                            {strategy.daily_pnl >= 0 ? '+' : ''}${strategy.daily_pnl.toFixed(2)}
                                        </div>
                                    </td>
                                    <td className="py-4 text-right font-mono text-sm text-slate-200">
                                        {strategy.sharpe_ratio?.toFixed(2) || 'N/A'}
                                    </td>
                                    <td className="py-4 text-right font-mono text-sm text-slate-200">
                                        {strategy.total_trades}
                                    </td>
                                    <td className="py-4 text-right">
                      <span className={`font-semibold ${
                          strategy.win_rate >= 50 ? 'text-emerald-400' : 'text-amber-400'
                      }`}>
                        {strategy.win_rate.toFixed(1)}%
                      </span>
                                    </td>
                                    <td className="py-4 text-right font-semibold text-red-400">
                                        {strategy.max_drawdown.toFixed(2)}%
                                    </td>
                                </tr>
                            ))}
                            </tbody>
                        </table>
                    </div>
                </div>

            </div>
        </div>
    );
}
