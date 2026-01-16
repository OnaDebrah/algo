'use client'
import React from 'react';
import {Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis} from 'recharts';
import {Activity, Download, Target, TrendingDown, TrendingUp} from 'lucide-react';
import MetricCard from "@/components/backtest/MetricCard";
import {formatCurrency, formatPercent} from "@/utils/formatters";

const BacktestResults = ({ results }: any) => {
    return (
        <div className="space-y-6">
            <div className="grid grid-cols-4 gap-6">
                <MetricCard title="Total Return" value={formatPercent(results.total_return)} icon={TrendingUp} trend="up" color="emerald" />
                <MetricCard title="Win Rate" value={`${results.win_rate.toFixed(1)}%`} icon={Target} trend="up" color="blue" />
                <MetricCard title="Sharpe Ratio" value={results.sharpe_ratio.toFixed(2)} icon={Activity} trend="up" color="violet" />
                <MetricCard title="Max Drawdown" value={formatPercent(results.max_drawdown)} icon={TrendingDown} trend="down" color="red" />
            </div>

            <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                <div className="flex justify-between items-center mb-6">
                    <h3 className="text-xl font-semibold text-slate-100">Equity Curve</h3>
                    <button className="flex items-center space-x-2 px-4 py-2 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50 rounded-lg transition-all text-sm font-medium text-slate-300">
                        <Download size={18} strokeWidth={2} />
                        <span>Export</span>
                    </button>
                </div>
                <ResponsiveContainer width="100%" height={320}>
                    <AreaChart data={results.equity_curve}>
                        <defs>
                            <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
                                <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                        <XAxis dataKey="timestamp" stroke="#64748b" style={{ fontSize: '12px', fontWeight: 500 }} />
                        <YAxis stroke="#64748b" tickFormatter={(value) => formatCurrency(value)} style={{ fontSize: '12px', fontWeight: 500 }} />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '12px', padding: '12px' }}
                            formatter={(value) => [formatCurrency(Number(value || 0)), 'Equity']}
                            labelStyle={{ color: '#94a3b8', fontWeight: 600, marginBottom: '4px' }}
                        />
                        <Area type="monotone" dataKey="equity" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#colorEquity)" />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default BacktestResults;