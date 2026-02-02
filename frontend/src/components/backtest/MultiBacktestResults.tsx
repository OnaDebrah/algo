'use client'
import React, { useState, useMemo } from 'react';
import { Area, AreaChart, CartesianGrid, ComposedChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Line, BarChart, Bar, Cell } from 'recharts';
import { Activity, Target, TrendingDown, TrendingUp, Calendar, AlertTriangle } from 'lucide-react';
import MetricCard from "@/components/backtest/MetricCard";
import BenchmarkComparison from "@/components/backtest/BenchmarkComparison";
import RiskAnalysisModal from "@/components/backtest/RiskAnalysisModal";
import { formatCurrency, formatPercent } from "@/utils/formatters";
import { BacktestResult, EquityCurvePoint, SymbolStats, Trade } from "@/types/all_types";

const MultiBacktestResults = ({ results }: { results: BacktestResult }) => {
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
            {/* Metrics Cards */}
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

            {/* Per-Symbol Performance */}
            <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                <h3 className="text-xl font-semibold text-slate-100 mb-6">Per-Symbol Performance</h3>
                <div className="grid grid-cols-1 gap-3">
                    {Object.entries(results.symbol_stats || {}).map(([symbol, stats]: [string, SymbolStats]) => (
                        <div
                            key={symbol}
                            className="flex items-center justify-between p-4 bg-slate-800/40 border border-slate-700/50 rounded-xl hover:bg-slate-800/60 transition-all"
                        >
                            <div className="flex items-center space-x-4">
                                <div className="w-12 h-12 bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 border border-violet-500/30 rounded-lg flex items-center justify-center">
                                    <span className="font-bold text-sm text-violet-300">{symbol.slice(0, 3)}</span>
                                </div>
                                <div>
                                    <p className="font-semibold text-slate-200">{symbol}</p>
                                </div>
                            </div>
                            <div className="flex items-center space-x-8">
                                <div className="text-right">
                                    <p className="text-xs text-slate-500 font-medium mb-0.5">Profit</p>
                                    <p className={`font-bold ${stats.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                        {formatCurrency(stats.total_return)}
                                    </p>
                                </div>
                                <div className="text-right">
                                    <p className="text-xs text-slate-500 font-medium mb-0.5">Trades</p>
                                    <p className="font-bold text-slate-200">{stats.total_trades}</p>
                                </div>
                                <div className="text-right">
                                    <p className="text-xs text-slate-500 font-medium mb-0.5">Win Rate</p>
                                    <p className="font-bold text-blue-400">{stats.win_rate.toFixed(1)}%</p>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Portfolio Equity Curve */}
            <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                <div className="mb-6">
                    <h3 className="text-xl font-semibold text-slate-100">Portfolio Equity Curve</h3>
                    {results.benchmark && (
                        <p className="text-xs text-slate-500 mt-1">
                            Strategy (purple) vs Benchmark (blue)
                        </p>
                    )}
                </div>
                <ResponsiveContainer width="100%" height={320}>
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
                        <XAxis dataKey="timestamp" stroke="#64748b" style={{ fontSize: '12px' }} />
                        <YAxis stroke="#64748b" tickFormatter={(value) => formatCurrency(value)} style={{ fontSize: '12px' }} />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#1e293b',
                                border: '1px solid #334155',
                                borderRadius: '12px',
                                padding: '12px'
                            }}
                            formatter={(value: any, name: any) => {
                                if (name === 'strategy_equity') return [formatCurrency(value), 'Strategy'];
                                if (name === 'benchmark_equity') return [formatCurrency(value), 'Benchmark'];
                                return [formatCurrency(value), name];
                            }}
                            labelStyle={{ color: '#94a3b8', fontWeight: 600, marginBottom: '4px' }}
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

            {/* ⭐ NEW: DRAWDOWN CHART */}
            <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                <div className="flex justify-between items-center mb-6">
                    <div>
                        <h3 className="text-xl font-semibold text-slate-100">Drawdown Analysis</h3>
                        <p className="text-xs text-slate-500 mt-1">
                            Underwater equity chart • Max DD: {formatPercent(Math.abs(results.max_drawdown))}
                        </p>
                    </div>
                </div>
                <ResponsiveContainer width="100%" height={250}>
                    <AreaChart data={drawdownData}>
                        <defs>
                            <linearGradient id="colorDrawdown" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4} />
                                <stop offset="95%" stopColor="#ef4444" stopOpacity={0.05} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                        <XAxis dataKey="timestamp" stroke="#64748b" style={{ fontSize: '12px' }} />
                        <YAxis
                            stroke="#64748b"
                            tickFormatter={(value) => `${value.toFixed(1)}%`}
                            style={{ fontSize: '12px' }}
                            domain={['dataMin', 0]}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#1e293b',
                                border: '1px solid #334155',
                                borderRadius: '12px',
                                padding: '12px'
                            }}
                            formatter={(value: any) => [`${Number(value).toFixed(2)}%`, 'Drawdown']}
                            labelStyle={{ color: '#94a3b8', fontWeight: 600, marginBottom: '4px' }}
                        />
                        <Area
                            type="monotone"
                            dataKey="drawdown"
                            stroke="#ef4444"
                            strokeWidth={2}
                            fillOpacity={1}
                            fill="url(#colorDrawdown)"
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>

            {/* ⭐ NEW: ROLLING SHARPE RATIO */}
            <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                <div className="flex justify-between items-center mb-6">
                    <div>
                        <h3 className="text-xl font-semibold text-slate-100">Rolling Sharpe Ratio (30-day)</h3>
                        <p className="text-xs text-slate-500 mt-1">
                            Annualized risk-adjusted return over time
                        </p>
                    </div>
                </div>
                <ResponsiveContainer width="100%" height={250}>
                    <ComposedChart data={rollingSharpe}>
                        <defs>
                            <linearGradient id="colorSharpe" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                        <XAxis dataKey="timestamp" stroke="#64748b" style={{ fontSize: '12px' }} />
                        <YAxis
                            stroke="#64748b"
                            tickFormatter={(value) => value.toFixed(1)}
                            style={{ fontSize: '12px' }}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#1e293b',
                                border: '1px solid #334155',
                                borderRadius: '12px',
                                padding: '12px'
                            }}
                            formatter={(value: any) => [Number(value).toFixed(2), 'Sharpe Ratio']}
                            labelStyle={{ color: '#94a3b8', fontWeight: 600, marginBottom: '4px' }}
                        />
                        <Line
                            type="monotone"
                            dataKey={() => 0}
                            stroke="#64748b"
                            strokeWidth={1}
                            strokeDasharray="5 5"
                            dot={false}
                        />
                        <Area
                            type="monotone"
                            dataKey="sharpe"
                            stroke="#8b5cf6"
                            strokeWidth={2}
                            fillOpacity={1}
                            fill="url(#colorSharpe)"
                        />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>

            {/* ⭐ NEW: MONTHLY RETURNS TABLE */}
            <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                <div className="flex items-center gap-3 mb-6">
                    <Calendar className="text-violet-400" size={24} />
                    <div>
                        <h3 className="text-xl font-semibold text-slate-100">Monthly Returns</h3>
                        <p className="text-xs text-slate-500 mt-1">
                            Month-by-month performance breakdown
                        </p>
                    </div>
                </div>

                {monthlyReturns.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                        {monthlyReturns.map((monthData) => {
                            const [year, month] = monthData.month.split('-');
                            const monthName = new Date(parseInt(year), parseInt(month) - 1).toLocaleDateString('en-US', { month: 'short', year: 'numeric' });

                            return (
                                <div
                                    key={monthData.month}
                                    className="p-4 bg-slate-800/40 border border-slate-700/50 rounded-xl hover:bg-slate-800/60 transition-all"
                                >
                                    <div className="flex justify-between items-center">
                                        <div>
                                            <p className="text-xs text-slate-500 font-medium mb-1">{monthName}</p>
                                            <p className={`text-xl font-bold ${monthData.return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                {monthData.return >= 0 ? '+' : ''}{monthData.return.toFixed(2)}%
                                            </p>
                                        </div>
                                        <div className={`p-2 rounded-lg ${monthData.return >= 0 ? 'bg-emerald-500/10' : 'bg-red-500/10'}`}>
                                            {monthData.return >= 0 ? (
                                                <TrendingUp className="text-emerald-400" size={20} />
                                            ) : (
                                                <TrendingDown className="text-red-400" size={20} />
                                            )}
                                        </div>
                                    </div>
                                    <div className="mt-2 pt-2 border-t border-slate-700/30">
                                        <p className="text-xs text-slate-600">
                                            {formatCurrency(monthData.startEquity)} → {formatCurrency(monthData.endEquity)}
                                        </p>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                ) : (
                    <div className="text-center py-12">
                        <Calendar className="mx-auto text-slate-600 mb-3" size={48} />
                        <p className="text-slate-500">No monthly data available</p>
                    </div>
                )}

                {/* Monthly Stats Summary */}
                {monthlyReturns.length > 0 && (
                    <div className="grid grid-cols-4 gap-4 mt-6 pt-6 border-t border-slate-700/50">
                        <div className="text-center">
                            <p className="text-xs text-slate-500 mb-1">Best Month</p>
                            <p className="text-lg font-bold text-emerald-400">
                                +{Math.max(...monthlyReturns.map(m => m.return)).toFixed(2)}%
                            </p>
                        </div>
                        <div className="text-center">
                            <p className="text-xs text-slate-500 mb-1">Worst Month</p>
                            <p className="text-lg font-bold text-red-400">
                                {Math.min(...monthlyReturns.map(m => m.return)).toFixed(2)}%
                            </p>
                        </div>
                        <div className="text-center">
                            <p className="text-xs text-slate-500 mb-1">Avg Month</p>
                            <p className="text-lg font-bold text-slate-300">
                                {(monthlyReturns.reduce((sum, m) => sum + m.return, 0) / monthlyReturns.length).toFixed(2)}%
                            </p>
                        </div>
                        <div className="text-center">
                            <p className="text-xs text-slate-500 mb-1">Positive Months</p>
                            <p className="text-lg font-bold text-blue-400">
                                {monthlyReturns.filter(m => m.return > 0).length}/{monthlyReturns.length}
                            </p>
                        </div>
                    </div>
                )}
            </div>

            {/* Benchmark Comparison */}
            {results.benchmark && <BenchmarkComparison benchmark={results.benchmark} />}

            {/* Trades Table */}
            <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                <div className="flex justify-between items-center mb-6">
                    <h3 className="text-xl font-semibold text-slate-100">Trade History</h3>
                    <div className="flex space-x-2">
                        {['all', 'wins', 'losses'].map((filter) => (
                            <button
                                key={filter}
                                onClick={() => setTradeFilter(filter)}
                                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${tradeFilter === filter
                                    ? 'bg-violet-500 text-white shadow-lg shadow-violet-500/20'
                                    : 'bg-slate-800/60 text-slate-400 hover:text-slate-200'
                                }`}
                            >
                                {filter.charAt(0).toUpperCase() + filter.slice(1)}
                            </button>
                        ))}
                        <button
                            onClick={() => setShowRiskAnalysis(true)}
                            className="flex items-center gap-2 px-4 py-2 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 rounded-lg text-sm font-medium transition-all border border-blue-500/30"
                        >
                            <AlertTriangle size={16} />
                            Risk Analysis
                        </button>
                    </div>
                </div>

                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead>
                        <tr className="border-b border-slate-700/50">
                            <th className="text-left p-3 text-xs font-bold text-slate-500 uppercase">Symbol</th>
                            <th className="text-left p-3 text-xs font-bold text-slate-500 uppercase">Type</th>
                            <th className="text-right p-3 text-xs font-bold text-slate-500 uppercase">Quantity</th>
                            <th className="text-right p-3 text-xs font-bold text-slate-500 uppercase">Price</th>
                            <th className="text-right p-3 text-xs font-bold text-slate-500 uppercase">Profit/Loss</th>
                            <th className="text-right p-3 text-xs font-bold text-slate-500 uppercase">Return %</th>
                            <th className="text-left p-3 text-xs font-bold text-slate-500 uppercase">Date</th>
                        </tr>
                        </thead>
                        <tbody>
                        {trades
                            .filter((trade) => {
                                if (tradeFilter === 'wins') return (trade.profit || 0) > 0;
                                if (tradeFilter === 'losses') return (trade.profit || 0) < 0;
                                return true;
                            })
                            .map((trade, index) => (
                                <tr key={index} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                                    <td className="p-3 font-medium text-slate-200">{trade.symbol}</td>
                                    <td className="p-3">
                                            <span className={`px-2 py-1 rounded text-xs font-bold ${trade.order_type === 'BUY'
                                                ? 'bg-emerald-500/20 text-emerald-400'
                                                : 'bg-red-500/20 text-red-400'
                                            }`}>
                                                {trade.order_type}
                                            </span>
                                    </td>
                                    <td className="p-3 text-right text-slate-300">{trade.quantity}</td>
                                    <td className="p-3 text-right font-mono text-slate-300">{formatCurrency(trade.price)}</td>
                                    <td className={`p-3 text-right font-bold ${(trade.profit || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
                                    }`}>
                                        {trade.profit ? formatCurrency(trade.profit) : '-'}
                                    </td>
                                    <td className={`p-3 text-right font-mono ${(trade.profit_pct || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
                                    }`}>
                                        {trade.profit_pct ? `${trade.profit_pct.toFixed(2)}%` : '-'}
                                    </td>
                                    <td className="p-3 text-slate-400 text-sm">
                                        {trade.executed_at}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Risk Analysis Modal */}
            {showRiskAnalysis && (
                <RiskAnalysisModal results={results} onClose={() => setShowRiskAnalysis(false)} />
            )}
        </div>
    );
};

export default MultiBacktestResults;
