'use client'
import React, { useState, useMemo } from 'react';
import { Area, AreaChart, CartesianGrid, ComposedChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Line, BarChart, Bar, Cell } from 'recharts';
import { Activity, Target, TrendingDown, TrendingUp, Calendar, AlertTriangle, Info, Shield, List, Download, Zap } from 'lucide-react';
import MetricCard from "@/components/backtest/MetricCard";
import BenchmarkComparison from "@/components/backtest/BenchmarkComparison";
import RiskAnalysisModal from "@/components/backtest/RiskAnalysisModal";
import PerformanceHeatmap from "@/components/backtest/PerformanceHeatmap";
import FactorAttribution from "@/components/backtest/FactorAttribution";
import TradeChart from "@/components/backtest/TradeChart";
import { formatCurrency, formatPercent } from "@/utils/formatters";
import { BacktestResult, EquityCurvePoint, SymbolStats, Trade } from "@/types/all_types";

interface MultiBacktestResultsProps {
    results: BacktestResult;
}

const MultiBacktestResults: React.FC<MultiBacktestResultsProps> = ({ results }) => {
    const [activeTab, setActiveTab] = useState<'overview' | 'tearsheet' | 'trades'>('overview');
    const [tradeFilter, setTradeFilter] = useState('all');
    const [showRiskAnalysis, setShowRiskAnalysis] = useState(false);

    const trades = results.trades || [];
    const equityCurve = results.equity_curve || [];

    // ============================================================
    // MONTHLY RETURNS CALCULATION
    // ============================================================
    const monthlyReturns = useMemo(() => {
        if (!equityCurve || equityCurve.length === 0) return [];

        const monthlyData: { [key: string]: { start: number; end: number; dates: string[] } } = {};

        equityCurve.forEach((point) => {
            if (!point.timestamp) return;
            const date = new Date(point.timestamp);
            const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;

            if (!monthlyData[monthKey]) {
                monthlyData[monthKey] = {
                    start: point.equity,
                    end: point.equity,
                    dates: [point.timestamp]
                };
            } else {
                monthlyData[monthKey].end = point.equity;
                monthlyData[monthKey].dates.push(point.timestamp);
            }
        });

        return Object.entries(monthlyData)
            .map(([month, data]) => {
                const returnPct = ((data.end - data.start) / data.start) * 100;
                return {
                    month,
                    return: returnPct,
                    startEquity: data.start,
                    endEquity: data.end
                };
            })
            .sort((a, b) => a.month.localeCompare(b.month));
    }, [equityCurve]);

    // ============================================================
    // DRAWDOWN CALCULATION
    // ============================================================
    const drawdownData = useMemo(() => {
        if (!equityCurve || equityCurve.length === 0) return [];

        let peak = equityCurve[0].equity;
        return equityCurve.map((point) => {
            peak = Math.max(peak, point.equity);
            const drawdown = ((point.equity - peak) / peak) * 100;

            return {
                timestamp: point.timestamp,
                drawdown: drawdown,
                peak: peak
            };
        });
    }, [equityCurve]);

    // ============================================================
    // ROLLING SHARPE RATIO CALCULATION (30-day window)
    // ============================================================
    const rollingSharpe = useMemo(() => {
        if (!equityCurve || equityCurve.length < 30) return [];

        const window = 30;
        const riskFreeRate = 0;

        return equityCurve.map((point, index) => {
            if (index < window) {
                return {
                    timestamp: point.timestamp,
                    sharpe: null
                };
            }

            const windowData = equityCurve.slice(index - window, index + 1);
            const returns = windowData.slice(1).map((p, i) =>
                ((p.equity - windowData[i].equity) / windowData[i].equity) * 100
            );

            const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
            const stdDev = Math.sqrt(
                returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
            );

            const sharpe = stdDev !== 0 ? (avgReturn - riskFreeRate) / stdDev : 0;

            return {
                timestamp: point.timestamp,
                sharpe: sharpe * Math.sqrt(252) // Annualized
            };
        }).filter(d => d.sharpe !== null);
    }, [equityCurve]);

    // ============================================================
    // EQUITY CURVE DATA (with benchmark)
    // ============================================================
    const equityChartData = useMemo(() => {
        const strategyData = equityCurve;
        const benchmarkData = results.benchmark?.equity_curve || [];

        return strategyData.map((point) => {
            const benchmarkPoint = benchmarkData.find((bp) => bp.timestamp === point.timestamp);

            return {
                timestamp: point.timestamp,
                strategy_equity: point.equity,
                benchmark_equity: benchmarkPoint?.equity || null
            };
        });
    }, [equityCurve, results.benchmark]);

    return (
        <div className="space-y-6">
            {/* Tabs Header */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                <div className="flex space-x-1 bg-slate-900/50 p-1 rounded-xl border border-slate-800">
                    <button
                        onClick={() => setActiveTab('overview')}
                        className={`px-6 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center gap-2 ${activeTab === 'overview'
                            ? 'bg-violet-600 text-white shadow-lg shadow-violet-600/20'
                            : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                            }`}
                    >
                        <Info size={16} />
                        Portfolio Overview
                    </button>
                    <button
                        onClick={() => setActiveTab('tearsheet')}
                        className={`px-6 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center gap-2 ${activeTab === 'tearsheet'
                            ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20'
                            : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                            }`}
                    >
                        <Shield size={16} />
                        Quant Tear Sheet
                    </button>
                    <button
                        onClick={() => setActiveTab('trades')}
                        className={`px-6 py-2.5 rounded-lg text-sm font-bold transition-all flex items-center gap-2 ${activeTab === 'trades'
                            ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-600/20'
                            : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                            }`}
                    >
                        <List size={16} />
                        Trade History
                    </button>
                </div>

                <div className="flex items-center gap-4">
                    <button
                        onClick={() => setShowRiskAnalysis(true)}
                        className="flex items-center gap-2 px-4 py-2 bg-amber-500/10 hover:bg-amber-500/20 text-amber-500 rounded-lg text-xs font-black uppercase tracking-widest transition-all border border-amber-500/20"
                    >
                        <AlertTriangle size={14} />
                        Portfolio Risk Analysis
                    </button>
                    <button className="p-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-400 transition-colors border border-slate-700">
                        <Download size={18} />
                    </button>
                </div>
            </div>

            {/* Content per Tab */}
            {activeTab === 'overview' && (
                <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-300">
                    <div className="grid grid-cols-4 gap-6">
                        <MetricCard
                            title="Total Return"
                            value={formatPercent(results.total_return)}
                            icon={TrendingUp}
                            trend="up"
                            color="emerald"
                        />
                        <MetricCard
                            title="Win Rate"
                            value={`${results.win_rate.toFixed(1)}%`}
                            icon={Target}
                            trend="up"
                            color="blue"
                        />
                        <MetricCard
                            title="Sharpe Ratio"
                            value={results.sharpe_ratio.toFixed(2)}
                            icon={Activity}
                            trend="up"
                            color="violet"
                        />
                        <MetricCard
                            title="Max Drawdown"
                            value={formatPercent(results.max_drawdown)}
                            icon={TrendingDown}
                            trend="down"
                            color="red"
                        />
                    </div>

                    {/* Per-Symbol Performance Table */}
                    <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                        <div className="flex items-center justify-between mb-6">
                            <h3 className="text-xl font-semibold text-slate-100 flex items-center gap-3">
                                <Zap size={20} className="text-amber-400" />
                                Asset Allocation Performance
                            </h3>
                            <span className="text-[10px] text-slate-500 font-black uppercase tracking-widest bg-slate-800/50 px-3 py-1 rounded-full border border-slate-700/50">
                                {Object.keys(results.symbol_stats || {}).length} Active Assets
                            </span>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {Object.entries(results.symbol_stats || {}).map(([symbol, stats]: [string, SymbolStats]) => (
                                <div
                                    key={symbol}
                                    className="p-4 bg-slate-800/40 border border-slate-700/50 rounded-xl hover:bg-slate-800/60 transition-all border-l-4 border-l-violet-500"
                                >
                                    <div className="flex justify-between items-start mb-3">
                                        <div>
                                            <p className="font-black text-lg text-slate-100 uppercase tracking-tighter">{symbol}</p>
                                            <p className="text-[10px] text-slate-500 font-bold uppercase">{stats.total_trades} Executed Trades</p>
                                        </div>
                                        <div className={`px-2 py-1 rounded text-[10px] font-black ${stats.win_rate >= 50 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                                            {stats.win_rate.toFixed(1)}% WR
                                        </div>
                                    </div>
                                    <div className="flex justify-between items-end">
                                        <div>
                                            <p className="text-[9px] text-slate-600 font-black uppercase">Net P&L</p>
                                            <p className={`text-base font-bold ${stats.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                {formatCurrency(stats.total_return)}
                                            </p>
                                        </div>
                                        <TrendingUp size={16} className={stats.total_return >= 0 ? 'text-emerald-500/30' : 'text-red-500/30'} />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                        <div className="flex justify-between items-center mb-6">
                            <div>
                                <h3 className="text-xl font-semibold text-slate-100">Portfolio Equity Curve</h3>
                                {results.benchmark && (
                                    <p className="text-xs text-slate-500 mt-1 uppercase font-bold tracking-widest">
                                        Multi-Asset Strategy (Violet) vs {results.benchmark.symbol} (Blue)
                                    </p>
                                )}
                            </div>
                        </div>
                        <ResponsiveContainer width="100%" height={400}>
                            <ComposedChart data={equityChartData}>
                                <defs>
                                    <linearGradient id="colorStrategyEquity" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.4} />
                                        <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                                    </linearGradient>
                                    <linearGradient id="colorBenchmarkEquity" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2} />
                                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                                <XAxis dataKey="timestamp" stroke="#64748b" style={{ fontSize: '10px' }} />
                                <YAxis stroke="#64748b" tickFormatter={(value) => formatCurrency(value)} style={{ fontSize: '10px' }} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }}
                                    formatter={(value: any, name: any) => {
                                        if (name === 'strategy_equity') return [formatCurrency(value), 'Portfolio'];
                                        if (name === 'benchmark_equity') return [formatCurrency(value), 'Benchmark'];
                                        return [formatCurrency(value), name];
                                    }}
                                />
                                {results.benchmark && (
                                    <Area
                                        type="monotone"
                                        dataKey="benchmark_equity"
                                        stroke="#3b82f6"
                                        strokeWidth={2}
                                        fillOpacity={1}
                                        fill="url(#colorBenchmarkEquity)"
                                        strokeDasharray="5 5"
                                    />
                                )}
                                <Area
                                    type="monotone"
                                    dataKey="strategy_equity"
                                    stroke="#8b5cf6"
                                    strokeWidth={3}
                                    fillOpacity={1}
                                    fill="url(#colorStrategyEquity)"
                                />
                            </ComposedChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Price Action & Trade Signals Chart */}
                    {results.price_data && results.price_data.length > 0 && trades.length > 0 && (
                        <TradeChart priceData={results.price_data} trades={trades} />
                    )}

                    {results.benchmark && <BenchmarkComparison benchmark={results.benchmark} />}
                </div>
            )}

            {activeTab === 'tearsheet' && (
                <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-300">
                    {/* Factor Attribution (Market Alpha/Beta) */}
                    {results && (
                        <FactorAttribution
                            alpha={results.alpha || 0}
                            beta={results.beta || 0}
                            rSquared={results.rSquared}
                        />
                    )}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl">
                            <div className="flex items-center gap-2 mb-2">
                                <Shield size={14} className="text-violet-400" />
                                <p className="text-[10px] text-slate-500 font-black uppercase tracking-widest">Sortino Ratio</p>
                            </div>
                            <p className="text-2xl font-bold text-slate-200">
                                {(results?.sortino_ratio || 0).toFixed(2)}
                            </p>
                        </div>
                        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl">
                            <div className="flex items-center gap-2 mb-2">
                                <Zap size={14} className="text-blue-400" />
                                <p className="text-[10px] text-slate-500 font-black uppercase tracking-widest">Calmar Ratio</p>
                            </div>
                            <p className="text-2xl font-bold text-slate-200">
                                {(results?.calmar_ratio || 0).toFixed(2)}
                            </p>
                        </div>
                        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl">
                            <div className="flex items-center gap-2 mb-2">
                                <TrendingDown size={14} className="text-red-400" />
                                <p className="text-[10px] text-slate-500 font-black uppercase tracking-widest">VaR (95%)</p>
                            </div>
                            <p className="text-2xl font-bold text-red-500">
                                {formatPercent(Math.abs(results?.var_95 || 0))}
                            </p>
                        </div>
                        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl">
                            <div className="flex items-center gap-2 mb-2">
                                <Activity size={14} className="text-amber-400" />
                                <p className="text-[10px] text-slate-500 font-black uppercase tracking-widest">Portfolio Vol</p>
                            </div>
                            <p className="text-2xl font-bold text-slate-200">
                                {formatPercent(results?.volatility || 0)}
                            </p>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl">
                            <h4 className="text-sm font-bold text-slate-300 mb-6 flex items-center gap-2 uppercase tracking-widest">
                                <TrendingDown size={16} className="text-red-400" />
                                Portfolio Drawdowns
                            </h4>
                            <ResponsiveContainer width="100%" height={200}>
                                <AreaChart data={drawdownData}>
                                    <defs>
                                        <linearGradient id="colorDrawdown" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4} />
                                            <stop offset="95%" stopColor="#ef4444" stopOpacity={0.05} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} vertical={false} />
                                    <XAxis dataKey="timestamp" hide />
                                    <YAxis stroke="#64748b" style={{ fontSize: '10px' }} tickFormatter={(val) => `${val.toFixed(1)}%`} />
                                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }} />
                                    <Area type="monotone" dataKey="drawdown" stroke="#ef4444" fill="url(#colorDrawdown)" />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl">
                            <h4 className="text-sm font-bold text-slate-300 mb-6 flex items-center gap-2 uppercase tracking-widest">
                                <Activity size={16} className="text-violet-400" />
                                30D Rolling Sharpe
                            </h4>
                            <ResponsiveContainer width="100%" height={200}>
                                <ComposedChart data={rollingSharpe}>
                                    <defs>
                                        <linearGradient id="colorSharpe" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} vertical={false} />
                                    <XAxis dataKey="timestamp" hide />
                                    <YAxis stroke="#64748b" style={{ fontSize: '10px' }} />
                                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }} />
                                    <Line type="monotone" dataKey={() => 0} stroke="#475569" strokeDasharray="5 5" dot={false} />
                                    <Area type="monotone" dataKey="sharpe" stroke="#8b5cf6" fill="url(#colorSharpe)" />
                                </ComposedChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Monthly Returns Heatmap */}
                    <PerformanceHeatmap monthlyReturns={results.monthly_returns_matrix} />
                </div>
            )}

            {activeTab === 'trades' && (
                <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-300">
                    <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
                        <div className="p-6 border-b border-slate-800 flex justify-between items-center bg-white/[0.01]">
                            <h3 className="text-xl font-semibold text-slate-100 italic">Portfolio Trade History</h3>
                            <div className="flex bg-slate-800/50 p-1 rounded-lg border border-slate-700">
                                {['all', 'wins', 'losses'].map((f) => (
                                    <button
                                        key={f}
                                        onClick={() => setTradeFilter(f)}
                                        className={`px-4 py-1.5 rounded text-[10px] font-black uppercase tracking-widest transition-all ${tradeFilter === f
                                            ? 'bg-violet-600 text-white'
                                            : 'text-slate-500 hover:text-slate-300'
                                            }`}
                                    >
                                        {f}
                                    </button>
                                ))}
                            </div>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="w-full">
                                <thead>
                                    <tr className="border-b border-slate-800 bg-black/20">
                                        <th className="text-left p-4 text-[10px] font-black text-slate-500 uppercase tracking-widest">Symbol</th>
                                        <th className="text-left p-4 text-[10px] font-black text-slate-500 uppercase tracking-widest">Type</th>
                                        <th className="text-right p-4 text-[10px] font-black text-slate-500 uppercase tracking-widest">Quantity</th>
                                        <th className="text-right p-4 text-[10px] font-black text-slate-500 uppercase tracking-widest">Price</th>
                                        <th className="text-right p-4 text-[10px] font-black text-slate-500 uppercase tracking-widest">P/L</th>
                                        <th className="text-right p-4 text-[10px] font-black text-slate-500 uppercase tracking-widest">Executed At</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {trades
                                        .filter((t) => {
                                            if (tradeFilter === 'wins') return (t.profit || 0) > 0;
                                            if (tradeFilter === 'losses') return (t.profit || 0) < 0;
                                            return true;
                                        })
                                        .map((trade, idx) => (
                                            <tr key={idx} className="border-b border-slate-800/50 hover:bg-slate-800/20 transition-colors">
                                                <td className="p-4 text-xs font-black text-slate-200 uppercase">{trade.symbol}</td>
                                                <td className="p-4">
                                                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${trade.order_type === 'BUY' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                                                        {trade.order_type}
                                                    </span>
                                                </td>
                                                <td className="p-4 text-right text-xs font-mono text-slate-400">{trade.quantity}</td>
                                                <td className="p-4 text-right text-xs font-mono text-slate-300">{formatCurrency(trade.price)}</td>
                                                <td className={`p-4 text-right text-xs font-bold ${(trade.profit || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                    {trade.profit ? formatCurrency(trade.profit) : '-'}
                                                </td>
                                                <td className="p-4 text-right text-[10px] text-slate-600 uppercase font-bold">
                                                    {trade.executed_at}
                                                </td>
                                            </tr>
                                        ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            )}

            {showRiskAnalysis && (
                <RiskAnalysisModal results={results} onClose={() => setShowRiskAnalysis(false)} />
            )}
        </div>
    );
};

export default MultiBacktestResults;
