'use client'
import React, {useState} from 'react';
import {Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis} from 'recharts';
import {Activity, Download, Target, TrendingDown, TrendingUp} from 'lucide-react';
import MetricCard from "@/components/backtest/MetricCard";
import {formatCurrency, formatDate, formatPercent, toPrecision} from "@/utils/formatters";

const SingleBacktestResults = ({results}: any) => {
    const [tradeFilter, setTradeFilter] = useState('all');
    const trades = results.trades || [];

    return (
        <div className="space-y-6">
            <div className="grid grid-cols-4 gap-6">
                <MetricCard title="Total Return" value={formatPercent(results.total_return)} icon={TrendingUp}
                            trend="up" color="emerald"/>
                <MetricCard title="Win Rate" value={`${results.win_rate.toFixed(1)}%`} icon={Target} trend="up"
                            color="blue"/>
                <MetricCard title="Sharpe Ratio" value={results.sharpe_ratio.toFixed(2)} icon={Activity} trend="up"
                            color="violet"/>
                <MetricCard title="Max Drawdown" value={formatPercent(results.max_drawdown)} icon={TrendingDown}
                            trend="down" color="red"/>
            </div>

            <div
                className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                <div className="flex justify-between items-center mb-6">
                    <h3 className="text-xl font-semibold text-slate-100">Equity Curve</h3>
                    <button
                        className="flex items-center space-x-2 px-4 py-2 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50 rounded-lg transition-all text-sm font-medium text-slate-300">
                        <Download size={18} strokeWidth={2}/>
                        <span>Export</span>
                    </button>
                </div>
                <ResponsiveContainer width="100%" height={320}>
                    <AreaChart data={results.equity_curve}>
                        <defs>
                            <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#10b981" stopOpacity={0.4}/>
                                <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3}/>
                        <XAxis dataKey="timestamp" stroke="#64748b" style={{fontSize: '12px', fontWeight: 500}}/>
                        <YAxis stroke="#64748b" tickFormatter={(value) => formatCurrency(value)}
                               style={{fontSize: '12px', fontWeight: 500}}/>
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#1e293b',
                                border: '1px solid #334155',
                                borderRadius: '12px',
                                padding: '12px'
                            }}
                            formatter={(value) => [formatCurrency(Number(value || 0)), 'Equity']}
                            labelStyle={{color: '#94a3b8', fontWeight: 600, marginBottom: '4px'}}
                        />
                        <Area type="monotone" dataKey="equity" stroke="#10b981" strokeWidth={2} fillOpacity={1}
                              fill="url(#colorEquity)"/>
                    </AreaChart>
                </ResponsiveContainer>
            </div>
            {/* 3. Detailed Trade Ledger Table */}
            <div
                className="bg-slate-900/50 backdrop-blur-xl border border-slate-800/50 rounded-2xl overflow-hidden shadow-2xl">
                <div
                    className="px-6 py-5 border-b border-slate-800/50 flex justify-between items-center bg-white/[0.02]">
                    <div>
                        <h3 className="text-lg font-bold text-slate-100 tracking-tight">Trade Ledger</h3>
                        <p className="text-[10px] text-slate-500 font-black uppercase tracking-[0.2em] mt-1">Transaction
                            History & Execution Details</p>
                    </div>
                    <div className="flex bg-black/40 p-1 rounded-lg border border-white/5">
                        {['all', 'profitable', 'loss'].map((f) => (
                            <button
                                key={f}
                                onClick={() => setTradeFilter(f)}
                                className={`px-3 py-1.5 rounded-md text-[10px] font-black uppercase tracking-widest transition-all ${tradeFilter === f ? 'bg-violet-600 text-white' : 'text-slate-500 hover:text-slate-300'
                                }`}
                            >
                                {f}
                            </button>
                        ))}
                    </div>
                </div>

                <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                        <thead>
                        <tr className="bg-white/[0.01] text-[10px] font-black text-slate-500 uppercase tracking-widest border-b border-slate-800/50">
                            <th className="px-6 py-4">Asset</th>
                            <th className="px-6 py-4">Side</th>
                            <th className="px-6 py-4">Entry Date</th>
                            <th className="px-6 py-4">Exit Date</th>
                            <th className="px-6 py-4 text-right">Entry Price</th>
                            <th className="px-6 py-4 text-right">Exit Price</th>
                            <th className="px-6 py-4 text-right">Size</th>
                            <th className="px-6 py-4 text-right">P&L</th>
                        </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/30">
                        {trades.length > 0 ? trades.map((trade: any, idx: number) => {
                            const isWin = trade.pnl >= 0;
                            // Basic filter logic
                            if (tradeFilter === 'profitable' && !isWin) return null;
                            if (tradeFilter === 'loss' && isWin) return null;

                            return (
                                <tr key={idx} className="hover:bg-white/[0.02] transition-colors group">
                                    <td className="px-6 py-4">
                                        <div className="flex items-center space-x-3">
                                                <span
                                                    className="w-8 h-8 rounded bg-slate-800 flex items-center justify-center font-mono text-[10px] font-bold text-violet-400 border border-slate-700">
                                                    {trade.symbol.slice(0, 2)}
                                                </span>
                                            <span className="font-bold text-slate-200 text-xs">{trade.symbol}</span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4">
                                            <span
                                                className={`px-2 py-1 rounded text-[9px] font-black uppercase tracking-tighter ${trade.side === 'LONG' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'
                                                }`}>
                                                {trade.side}
                                            </span>
                                    </td>
                                    <td className="px-6 py-4 text-[11px] text-slate-400 font-medium">
                                        {formatDate(trade.entry_time)}
                                    </td>
                                    <td className="px-6 py-4 text-[11px] text-slate-400 font-medium">
                                        {formatDate(trade.exit_time)}
                                    </td>
                                    <td className="px-6 py-4 text-right font-mono text-xs text-slate-300">
                                        {formatCurrency(trade.entry_price)}
                                    </td>
                                    <td className="px-6 py-4 text-right font-mono text-xs text-slate-300">
                                        {formatCurrency(trade.exit_price)}
                                    </td>
                                    <td className="px-6 py-4 text-right font-mono text-xs text-slate-500">
                                        {toPrecision(trade.quantity)}
                                    </td>
                                    <td className={`px-6 py-4 text-right font-bold text-xs ${isWin ? 'text-emerald-400' : 'text-red-400'}`}>
                                        <div className="flex flex-col items-end">
                                            <span>{isWin ? '+' : ''}{formatCurrency(trade.pnl)}</span>
                                            <span className="text-[9px] font-medium opacity-60">
                                                    ({formatPercent(trade.pnl)})
                                                </span>
                                        </div>
                                    </td>
                                </tr>
                            );
                        }) : (
                            <tr>
                                <td colSpan={8} className="px-6 py-12 text-center text-slate-600 text-sm">
                                    No transaction data available for this backtest.
                                </td>
                            </tr>
                        )}
                        </tbody>
                    </table>
                </div>

                {/* Table Footer / Summary */}
                <div
                    className="px-6 py-4 bg-white/[0.01] border-t border-slate-800/50 flex justify-between items-center">
                    <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                        Total Captured Segments: {trades.length}
                    </span>
                    <div className="flex items-center space-x-4">
                        <div className="flex items-center space-x-1">
                            <div className="w-2 h-2 rounded-full bg-emerald-500"/>
                            <span
                                className="text-[10px] text-slate-400 font-bold uppercase">Average Winner: {formatCurrency(results.avg_win || 0)}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                            <div className="w-2 h-2 rounded-full bg-red-500"/>
                            <span
                                className="text-[10px] text-slate-400 font-bold uppercase">Average Loser: {formatCurrency(results.avg_loss || 0)}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default SingleBacktestResults;