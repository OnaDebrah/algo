// 'use client'
// import React, { useState } from 'react';
// import { Area, CartesianGrid, ComposedChart, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from 'recharts';
// import { Activity, Download, Target, TrendingDown, TrendingUp } from 'lucide-react';
// import MetricCard from "@/components/backtest/MetricCard";
// import BenchmarkComparison from "@/components/backtest/BenchmarkComparison";
// import { formatCurrency, formatPercent, formatTimeZone, toPrecision } from "@/utils/formatters";
// import { BacktestResult, EquityCurvePoint, Trade } from "@/types/all_types";
//
// const SingleBacktestResults: ({ results }: { results: BacktestResult }) => React.JSX.Element = ({ results }: {
//     results: BacktestResult
// }) => {
//     const [tradeFilter, setTradeFilter] = useState('all');
//     const trades = results.trades || [];
//
//     return (
//         <div className="space-y-6">
//             <div className="grid grid-cols-4 gap-6">
//                 <MetricCard title="Total Return" value={formatPercent(results.total_return)} icon={TrendingUp}
//                     trend="up" color="emerald" />
//                 <MetricCard title="Win Rate" value={`${results.win_rate.toFixed(1)}%`} icon={Target} trend="up"
//                     color="blue" />
//                 <MetricCard title="Sharpe Ratio" value={results.sharpe_ratio.toFixed(2)} icon={Activity} trend="up"
//                     color="violet" />
//                 <MetricCard title="Max Drawdown" value={formatPercent(results.max_drawdown)} icon={TrendingDown}
//                     trend="down" color="red" />
//             </div>
//
//             <div
//                 className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
//                 <div className="flex justify-between items-center mb-6">
//                     <div>
//                         <h3 className="text-xl font-semibold text-slate-100">Equity Curve</h3>
//                         {results.benchmark && (
//                             <p className="text-xs text-slate-500 mt-1">
//                                 Strategy (green) vs Benchmark (blue)
//                             </p>
//                         )}
//                     </div>
//                     <button
//                         className="flex items-center space-x-2 px-4 py-2 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50 rounded-lg transition-all text-sm font-medium text-slate-300">
//                         <Download size={18} strokeWidth={2} />
//                         <span>Export</span>
//                     </button>
//                 </div>
//                 <ResponsiveContainer width="100%" height={320}>
//                     <ComposedChart data={(() => {
//
//                         const strategyData: EquityCurvePoint[] = results?.equity_curve || [];
//                         const benchmarkData: EquityCurvePoint[] = results.benchmark?.equity_curve || [];
//
//                         const merged: {
//                             timestamp: string;
//                             strategy_equity: number;
//                             benchmark_equity: number | null
//                         }[] = strategyData.map((point: EquityCurvePoint) => {
//                             // Both timestamps are already formatted the same way (via formatDate in BacktestPage)
//                             const benchmarkPoint: EquityCurvePoint | undefined = benchmarkData.find((bp: EquityCurvePoint) =>
//                                 bp.timestamp === point.timestamp
//                             );
//
//                             return {
//                                 timestamp: point.timestamp,
//                                 strategy_equity: point.equity,
//                                 benchmark_equity: benchmarkPoint?.equity || null
//                             };
//                         });
//
//                         return merged;
//                     })()}>
//                         <defs>
//                             <linearGradient id="colorStrategyEquity" x1="0" y1="0" x2="0" y2="1">
//                                 <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
//                                 <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
//                             </linearGradient>
//                             <linearGradient id="colorBenchmarkEquity" x1="0" y1="0" x2="0" y2="1">
//                                 <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2} />
//                                 <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
//                             </linearGradient>
//                         </defs>
//                         <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
//                         <XAxis dataKey="timestamp" stroke="#64748b" style={{ fontSize: '12px', fontWeight: 500 }} />
//                         <YAxis stroke="#64748b" tickFormatter={(value) => formatCurrency(value)}
//                             style={{ fontSize: '12px', fontWeight: 500 }} />
//                         <Tooltip
//                             contentStyle={{
//                                 backgroundColor: '#1e293b',
//                                 border: '1px solid #334155',
//                                 borderRadius: '12px',
//                                 padding: '12px'
//                             }}
//                             formatter={(value: number | undefined, name: string | undefined) => {
//                                 if (name === 'strategy_equity') return [formatCurrency(Number(value || 0)), 'Strategy'];
//                                 if (name === 'benchmark_equity') return [formatCurrency(Number(value || 0)), 'Benchmark'];
//                                 return [formatCurrency(Number(value || 0)), name];
//                             }}
//                             labelStyle={{ color: '#94a3b8', fontWeight: 600, marginBottom: '4px' }}
//                         />
//
//                         {/* Benchmark line (below) */}
//                         {results.benchmark && (
//                             <Area
//                                 type="monotone"
//                                 dataKey="benchmark_equity"
//                                 stroke="#3b82f6"
//                                 strokeWidth={2}
//                                 fillOpacity={1}
//                                 fill="url(#colorBenchmarkEquity)"
//                                 strokeDasharray="5 5"
//                             />
//                         )}
//
//                         {/* Strategy line (on top) */}
//                         <Area
//                             type="monotone"
//                             dataKey="strategy_equity"
//                             stroke="#10b981"
//                             strokeWidth={2}
//                             fillOpacity={1}
//                             fill="url(#colorStrategyEquity)"
//                         />
//                     </ComposedChart>
//                 </ResponsiveContainer>
//
//                 {/* Legend */}
//                 {results.benchmark && (
//                     <div className="flex items-center justify-center space-x-6 mt-4">
//                         <div className="flex items-center space-x-2">
//                             <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
//                             <span className="text-xs text-slate-400 font-medium">Strategy</span>
//                         </div>
//                         <div className="flex items-center space-x-2">
//                             <div className="w-3 h-3 rounded-full bg-blue-500"></div>
//                             <span
//                                 className="text-xs text-slate-400 font-medium">Benchmark ({results.benchmark.symbol || 'SPY'})</span>
//                         </div>
//                     </div>
//                 )}
//             </div>
//
//             {/* Benchmark Comparison */}
//             {results.benchmark && (
//                 <BenchmarkComparison benchmark={results.benchmark} />
//             )}
//
//             {/* Price Chart with Trade Markers */}
//             {results.price_data && results.price_data.length > 0 && (
//                 <div
//                     className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
//                     <div className="flex justify-between items-center mb-6">
//                         <div>
//                             <h3 className="text-xl font-semibold text-slate-100">Price Action with Trade Signals</h3>
//                             <p className="text-xs text-slate-500 mt-1">Entry (🟢) and Exit (🔴) points overlaid on price
//                                 movement</p>
//                         </div>
//                     </div>
//                     <ResponsiveContainer width="100%" height={400}>
//                         <ComposedChart data={(() => {
//                             // Prepare price data with trade markers
//                             const priceData = results.price_data.map((point) => {
//                                 const timestamp: string = new Date(point.timestamp).toLocaleDateString();
//                                 const buyTrades: Trade[] = trades.filter((t: Trade) =>
//                                     t.order_type === 'BUY' &&
//                                     new Date(t.timestamp).toLocaleDateString() === timestamp
//                                 );
//                                 const sellTrades: Trade[] = trades.filter((t: Trade) =>
//                                     t.order_type === 'SELL' &&
//                                     new Date(t.timestamp).toLocaleDateString() === timestamp
//                                 );
//
//                                 return {
//                                     timestamp,
//                                     close: point.close,
//                                     buyPrice: buyTrades.length > 0 ? buyTrades[0].price : null,
//                                     sellPrice: sellTrades.length > 0 ? sellTrades[0].price : null,
//                                 };
//                             });
//                             return priceData;
//                         })()}>
//                             <defs>
//                                 <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
//                                     <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
//                                     <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
//                                 </linearGradient>
//                             </defs>
//                             <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
//                             <XAxis
//                                 dataKey="timestamp"
//                                 stroke="#64748b"
//                                 style={{ fontSize: '11px', fontWeight: 500 }}
//                                 interval="preserveStartEnd"
//                                 minTickGap={50}
//                             />
//                             <YAxis
//                                 stroke="#64748b"
//                                 tickFormatter={(value) => `$${value.toFixed(2)}`}
//                                 style={{ fontSize: '12px', fontWeight: 500 }}
//                                 domain={['auto', 'auto']}
//                             />
//                             <Tooltip
//                                 contentStyle={{
//                                     backgroundColor: '#1e293b',
//                                     border: '1px solid #334155',
//                                     borderRadius: '12px',
//                                     padding: '12px'
//                                 }}
//                                 formatter={(value: number | undefined, name: string | undefined) => {
//                                     if (value === undefined) return ['', name];
//
//                                     if (name === 'close') return [formatCurrency(value), 'Close Price'];
//                                     if (name === 'buyPrice') return [formatCurrency(value), 'BUY'];
//                                     if (name === 'sellPrice') return [formatCurrency(value), 'SELL'];
//
//                                     return [value.toString(), name];
//                                 }}
//                                 labelStyle={{ color: '#94a3b8', fontWeight: 600, marginBottom: '4px' }}
//                             />
//
//                             {/* Price line */}
//                             <Area
//                                 type="monotone"
//                                 dataKey="close"
//                                 stroke="#3b82f6"
//                                 strokeWidth={2}
//                                 fillOpacity={1}
//                                 fill="url(#colorPrice)"
//                             />
//
//                             {/* Buy signals (green) */}
//                             <Scatter
//                                 dataKey="buyPrice"
//                                 fill="#10b981"
//                                 shape="circle"
//                                 r={6}
//                             />
//
//                             {/* Sell signals (red) */}
//                             <Scatter
//                                 dataKey="sellPrice"
//                                 fill="#ef4444"
//                                 shape="circle"
//                                 r={6}
//                             />
//                         </ComposedChart>
//                     </ResponsiveContainer>
//
//                     {/* Legend */}
//                     <div className="flex items-center justify-center space-x-6 mt-4">
//                         <div className="flex items-center space-x-2">
//                             <div className="w-3 h-3 rounded-full bg-blue-500"></div>
//                             <span className="text-xs text-slate-400 font-medium">Price</span>
//                         </div>
//                         <div className="flex items-center space-x-2">
//                             <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
//                             <span className="text-xs text-slate-400 font-medium">Buy Signal</span>
//                         </div>
//                         <div className="flex items-center space-x-2">
//                             <div className="w-3 h-3 rounded-full bg-red-500"></div>
//                             <span className="text-xs text-slate-400 font-medium">Sell Signal</span>
//                         </div>
//                     </div>
//                 </div>
//             )}
//
//             {/* 3. Detailed Trade Ledger Table */}
//             <div
//                 className="bg-slate-900/50 backdrop-blur-xl border border-slate-800/50 rounded-2xl overflow-hidden shadow-2xl">
//                 <div
//                     className="px-6 py-5 border-b border-slate-800/50 flex justify-between items-center bg-white/[0.02]">
//                     <div>
//                         <h3 className="text-lg font-bold text-slate-100 tracking-tight">Trade Ledger</h3>
//                         <p className="text-[10px] text-slate-500 font-black uppercase tracking-[0.2em] mt-1">Transaction
//                             History & Execution Details</p>
//                     </div>
//                     <div className="flex bg-black/40 p-1 rounded-lg border border-white/5">
//                         {['all', 'profitable', 'loss'].map((f) => (
//                             <button
//                                 key={f}
//                                 onClick={() => setTradeFilter(f)}
//                                 className={`px-3 py-1.5 rounded-md text-[10px] font-black uppercase tracking-widest transition-all ${tradeFilter === f ? 'bg-violet-600 text-white' : 'text-slate-500 hover:text-slate-300'
//                                     }`}
//                             >
//                                 {f}
//                             </button>
//                         ))}
//                     </div>
//                 </div>
//
//                 <div className="overflow-x-auto">
//                     <table className="w-full text-left border-collapse">
//                         <thead>
//                             <tr className="bg-white/[0.01] text-[10px] font-black text-slate-500 uppercase tracking-widest border-b border-slate-800/50">
//                                 <th className="px-6 py-4">Asset</th>
//                                 <th className="px-6 py-4">Type</th>
//                                 <th className="px-6 py-4">Timestamp</th>
//                                 <th className="px-6 py-4">Strategy</th>
//                                 <th className="px-6 py-4 text-right">Price</th>
//                                 <th className="px-6 py-4 text-right">Quantity</th>
//                                 <th className="px-6 py-4 text-right">Commission</th>
//                                 <th className="px-6 py-4 text-right">Profit</th>
//                             </tr>
//                         </thead>
//                         <tbody className="divide-y divide-slate-800/30">
//                             {trades.length > 0 ? trades.map((trade, idx: number) => {
//                                 const isWin = trade.profit && trade.profit >= 0;
//                                 const isClosed = trade.profit !== null && trade.profit !== undefined;
//                                 if (tradeFilter === 'profitable' && (!isClosed || !isWin)) return null;
//                                 if (tradeFilter === 'loss' && (!isClosed || isWin)) return null;
//
//                                 return (
//                                     <tr key={idx} className="hover:bg-white/[0.02] transition-colors group">
//                                         <td className="px-6 py-4">
//                                             <div className="flex items-center space-x-3">
//                                                 <span
//                                                     className="w-8 h-8 rounded bg-slate-800 flex items-center justify-center font-mono text-[10px] font-bold text-violet-400 border border-slate-700">
//                                                     {trade.symbol.slice(0, 2)}
//                                                 </span>
//                                                 <span className="font-bold text-slate-200 text-xs">{trade.symbol}</span>
//                                             </div>
//                                         </td>
//                                         <td className="px-6 py-4">
//                                             <span
//                                                 className={`px-2 py-1 rounded text-[9px] font-black uppercase tracking-tighter ${trade.order_type === 'BUY' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'
//                                                     }`}>
//                                                 {trade.order_type}
//                                             </span>
//                                         </td>
//                                         <td className="px-6 py-4 text-[11px] text-slate-400 font-medium">
//                                             {trade.timestamp}
//                                         </td>
//                                         <td className="px-6 py-4 text-[11px] text-slate-400 font-medium">
//                                             {trade.strategy}
//                                         </td>
//                                         <td className="px-6 py-4 text-right font-mono text-xs text-slate-300">
//                                             {formatCurrency(trade.price)}
//                                         </td>
//                                         <td className="px-6 py-4 text-right font-mono text-xs text-slate-500">
//                                             {toPrecision(trade.quantity)}
//                                         </td>
//                                         <td className="px-6 py-4 text-right font-mono text-xs text-slate-500">
//                                             {formatCurrency(trade.commission)}
//                                         </td>
//                                         <td className={`px-6 py-4 text-right font-bold text-xs ${isClosed ? (isWin ? 'text-emerald-400' : 'text-red-400') : 'text-slate-500'}`}>
//                                             <div className="flex flex-col items-end">
//                                                 {isClosed ? (
//                                                     <>
//                                                         <span>{isWin ? '+' : ''}{formatCurrency(trade.profit || 0)}</span>
//                                                         <span className="text-[9px] font-medium opacity-60">
//                                                             ({trade.profit_pct?.toFixed(2) || '0.00'}%)
//                                                         </span>
//                                                     </>
//                                                 ) : (
//                                                     <span className="text-[10px] uppercase">Open</span>
//                                                 )}
//                                             </div>
//                                         </td>
//                                     </tr>
//                                 );
//                             }) : (
//                                 <tr>
//                                     <td colSpan={8} className="px-6 py-12 text-center text-slate-600 text-sm">
//                                         No transaction data available for this backtest.
//                                     </td>
//                                 </tr>
//                             )}
//                         </tbody>
//                     </table>
//                 </div>
//
//                 {/* Table Footer / Summary */}
//                 <div
//                     className="px-6 py-4 bg-white/[0.01] border-t border-slate-800/50 flex justify-between items-center">
//                     <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">
//                         Total Captured Segments: {trades.length}
//                     </span>
//                     <div className="flex items-center space-x-4">
//                         <div className="flex items-center space-x-1">
//                             <div className="w-2 h-2 rounded-full bg-emerald-500" />
//                             <span
//                                 className="text-[10px] text-slate-400 font-bold uppercase">Average Winner: {formatCurrency(results.avg_win || 0)}</span>
//                         </div>
//                         <div className="flex items-center space-x-1">
//                             <div className="w-2 h-2 rounded-full bg-red-500" />
//                             <span
//                                 className="text-[10px] text-slate-400 font-bold uppercase">Average Loser: {formatCurrency(results.avg_loss || 0)}</span>
//                         </div>
//                     </div>
//                 </div>
//             </div>
//         </div>
//     );
// };
//
// export default SingleBacktestResults;

'use client'
import React, {useMemo, useState} from 'react';
import {Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis, AreaChart} from 'recharts';
import {Activity, AlertTriangle, Calendar, Download, Target, TrendingDown, TrendingUp} from 'lucide-react';
import MetricCard from "@/components/backtest/MetricCard";
import BenchmarkComparison from "@/components/backtest/BenchmarkComparison";
import RiskAnalysisModal from "@/components/backtest/RiskAnalysisModal";
import {formatCurrency, formatPercent} from "@/utils/formatters";
import {BacktestResult, EquityCurvePoint} from "@/types/all_types";

const SingleBacktestResults = ({ results }: { results: BacktestResult }) => {
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

        equityCurve.forEach((point: EquityCurvePoint) => {
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
        const riskFreeRate = 0; // Assume 0 for simplicity

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

            {/* Equity Curve */}
            <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                <div className="flex justify-between items-center mb-6">
                    <div>
                        <h3 className="text-xl font-semibold text-slate-100">Equity Curve</h3>
                        {results.benchmark && (
                            <p className="text-xs text-slate-500 mt-1">
                                Strategy (green) vs Benchmark (blue)
                            </p>
                        )}
                    </div>
                    <button className="flex items-center space-x-2 px-4 py-2 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50 rounded-lg transition-all text-sm font-medium text-slate-300">
                        <Download size={18} strokeWidth={2} />
                        <span>Export</span>
                    </button>
                </div>
                <ResponsiveContainer width="100%" height={320}>
                    <ComposedChart data={equityChartData}>
                        <defs>
                            <linearGradient id="colorStrategyEquity" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
                                <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                            </linearGradient>
                            <linearGradient id="colorBenchmarkEquity" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2} />
                                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                        <XAxis dataKey="timestamp" stroke="#64748b" style={{ fontSize: '12px', fontWeight: 500 }} />
                        <YAxis stroke="#64748b" tickFormatter={(value) => formatCurrency(value)} style={{ fontSize: '12px', fontWeight: 500 }} />
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
                            stroke="#10b981"
                            strokeWidth={3}
                            fillOpacity={1}
                            fill="url(#colorStrategyEquity)"
                        />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>

            {/* ⭐ DRAWDOWN CHART */}
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

            {/* ⭐ ROLLING SHARPE RATIO */}
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
                        {/* Reference line at 0 */}
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
                                        {new Date(trade.executed_at).toLocaleString()}
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

export default SingleBacktestResults;
