import {DollarSign, LineChart as LineChartIcon, Loader2, Play, Settings} from "lucide-react";
import {Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis} from "recharts";
import React from "react";
import {STRATEGY_TEMPLATES} from "@/components/optionsdesk/contants/strategyTemplates";
import {BacktestConfig} from "@/types/all_types";

interface BackTestTabProps {
    backtestConfig: BacktestConfig,
    setBacktestConfig:  React.Dispatch<React.SetStateAction<BacktestConfig>>,
    runStrategyBacktest: () => Promise<void> ,
    backtestResults: any,
    equityData: any,
    recentTrades: any,
    isLoading: boolean,
}

const BackTestTab: React.FC<BackTestTabProps> = ({
                                                     backtestConfig,
                                                     setBacktestConfig,
                                                     runStrategyBacktest,
                                                     backtestResults,
                                                     equityData,
                                                     recentTrades,
                                                     isLoading,
                                                 }: BackTestTabProps) => {
    return (
        <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6 h-fit">
                    <h3 className="text-lg font-bold text-slate-200 mb-6 flex items-center gap-2">
                        <Settings size={20} className="text-amber-400"/>
                        Parameters
                    </h3>

                    <div className="space-y-4">
                        <div>
                            <label className="block text-xs font-bold text-slate-500 uppercase mb-2">Strategy</label>
                            <select
                                value={backtestConfig.strategy_type}
                                onChange={(e) => setBacktestConfig({...backtestConfig, strategy_type: e.target.value})}
                                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 outline-none focus:border-amber-500"
                            >
                                <option value="">Select Strategy</option>
                                {STRATEGY_TEMPLATES.map(s => (
                                    <option key={s.id} value={s.id}>{s.name}</option>
                                ))}
                            </select>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-xs font-bold text-slate-500 uppercase mb-2">Start</label>
                                <input
                                    type="date"
                                    value={backtestConfig.start_date}
                                    onChange={(e) => setBacktestConfig({...backtestConfig, start_date: e.target.value})}
                                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2 py-2 text-xs text-slate-200"
                                />
                            </div>
                            <div>
                                <label className="block text-xs font-bold text-slate-500 uppercase mb-2">End</label>
                                <input
                                    type="date"
                                    value={backtestConfig.end_date}
                                    onChange={(e) => setBacktestConfig({...backtestConfig, end_date: e.target.value})}
                                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2 py-2 text-xs text-slate-200"
                                />
                            </div>
                        </div>

                        <div>
                            <label className="block text-xs font-bold text-slate-500 uppercase mb-2">Capital</label>
                            <div className="relative">
                                <DollarSign className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
                                            size={14}/>
                                <input
                                    type="number"
                                    value={backtestConfig.initial_capital}
                                    onChange={(e) => setBacktestConfig({
                                        ...backtestConfig,
                                        initial_capital: parseFloat(e.target.value)
                                    })}
                                    className="w-full pl-9 pr-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-200"
                                />
                            </div>
                        </div>

                        <button
                            onClick={runStrategyBacktest}
                            disabled={isLoading || !backtestConfig.strategy_type}
                            className="w-full py-3 mt-4 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold rounded-xl flex items-center justify-center gap-2 shadow-lg shadow-emerald-900/20 disabled:opacity-50"
                        >
                            {isLoading ? <Loader2 size={18} className="animate-spin"/> : <Play size={18}/>}
                            Execute Backtest
                        </button>
                    </div>
                </div>

                <div className="lg:col-span-2 space-y-6">
                    {backtestResults ? (
                        <>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                {[
                                    {
                                        label: 'Return',
                                        val: `${backtestResults.total_return.toFixed(2)}%`,
                                        color: backtestResults.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'
                                    },
                                    {
                                        label: 'Win Rate',
                                        val: `${backtestResults.win_rate.toFixed(1)}%`,
                                        color: 'text-amber-400'
                                    },
                                    {
                                        label: 'Profit Factor',
                                        val: backtestResults.profit_factor.toFixed(2),
                                        color: 'text-blue-400'
                                    },
                                    {
                                        label: 'Sharpe',
                                        val: backtestResults.sharpe_ratio.toFixed(2),
                                        color: 'text-purple-400'
                                    }
                                ].map((kpi, idx) => (
                                    <div key={idx}
                                         className="bg-slate-900/50 p-4 rounded-xl border border-slate-800/50">
                                        <div
                                            className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">{kpi.label}</div>
                                        <div className={`text-xl font-bold mt-1 ${kpi.color}`}>{kpi.val}</div>
                                    </div>
                                ))}
                            </div>

                            <div className="bg-slate-900/50 p-6 rounded-xl border border-slate-800/50">
                                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-6">Performance
                                    &
                                    Drawdown</h4>
                                <div className="space-y-2">
                                    <div className="h-[280px]">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <AreaChart data={equityData}
                                                       margin={{top: 0, right: 10, left: 0, bottom: 0}}>
                                                <defs>
                                                    <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                                                        <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.2}/>
                                                        <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                                                    </linearGradient>
                                                </defs>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false}/>
                                                <XAxis dataKey="date" hide/>
                                                <YAxis stroke="#475569" fontSize={10}
                                                       tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}/>
                                                <Tooltip
                                                    contentStyle={{
                                                        backgroundColor: '#0f172a',
                                                        border: '1px solid #1e293b'
                                                    }}/>
                                                <Area type="monotone" dataKey="equity" stroke="#f59e0b" strokeWidth={2}
                                                      fill="url(#colorEquity)"/>
                                            </AreaChart>
                                        </ResponsiveContainer>
                                    </div>
                                    <div className="h-[100px] border-t border-slate-800 pt-2">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <AreaChart data={equityData}
                                                       margin={{top: 0, right: 10, left: 0, bottom: 0}}>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false}/>
                                                <XAxis dataKey="date" stroke="#475569" fontSize={9}/>
                                                <YAxis stroke="#475569" fontSize={9} domain={['auto', 0]}
                                                       tickFormatter={(v) => `${v}%`}/>
                                                <Tooltip
                                                    contentStyle={{
                                                        backgroundColor: '#0f172a',
                                                        border: '1px solid #334155'
                                                    }}/>
                                                <Area type="monotone" dataKey="drawdown" stroke="#ef4444" fill="#ef4444"
                                                      fillOpacity={0.1}/>
                                            </AreaChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>
                            </div>

                            <div className="bg-slate-900/50 rounded-xl border border-slate-800/50 overflow-hidden">
                                <div className="p-4 border-b border-slate-800 flex justify-between items-center">
                                    <h4 className="text-xs font-bold text-slate-400 uppercase">Recent Trades</h4>
                                    <span
                                        className="text-[10px] text-slate-500 font-medium">{recentTrades.length} Total Trades</span>
                                </div>
                                <div className="max-h-[300px] overflow-y-auto">
                                    <table className="w-full text-left text-[11px]">
                                        <thead className="sticky top-0 bg-slate-900 text-slate-500 font-bold uppercase">
                                        <tr>
                                            <th className="px-4 py-3">Date</th>
                                            <th className="px-4 py-3">Symbol</th>
                                            <th className="px-4 py-3">Strategy</th>
                                            <th className="px-4 py-3 text-right">Net P&L</th>
                                        </tr>
                                        </thead>
                                        <tbody className="divide-y divide-slate-800/50">
                                        {recentTrades.map((trade, i) => (
                                            <tr key={i} className="hover:bg-slate-800/30 transition-colors">
                                                <td className="px-4 py-3 text-slate-400">{trade.time}</td>
                                                <td className="px-4 py-3 font-bold text-slate-200">{trade.symbol}</td>
                                                <td className="px-4 py-3 text-slate-500">{trade.strategy}</td>
                                                <td className={`px-4 py-3 text-right font-bold ${trade.profit >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                    {trade.profit >= 0 ? '+' : ''}{trade.profit.toFixed(2)}
                                                </td>
                                            </tr>
                                        ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </>
                    ) : (
                        <div
                            className="h-full flex flex-col items-center justify-center text-slate-600 bg-slate-900/20 border-2 border-dashed border-slate-800/50 rounded-2xl py-20">
                            <LineChartIcon size={48} className="mb-4 opacity-10"/>
                            <p className="text-sm font-medium">Configure parameters and run backtest to view
                                performance</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

export default BackTestTab;