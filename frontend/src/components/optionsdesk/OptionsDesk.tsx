'use client'
import React, { useState, useMemo, useEffect } from 'react';
import {
    Activity,
    ArrowRight,
    ArrowUpRight,
    BarChart2,
    Calendar,
    ChevronDown,
    ChevronRight,
    Clock,
    DollarSign,
    Filter,
    LineChart,
    Play,
    RefreshCw,
    Search,
    Settings,
    Shield,
    Target,
    Zap,
    Loader2
} from "lucide-react";
import {
    Area,
    AreaChart,
    Bar,
    BarChart,
    CartesianGrid,
    Cell,
    Line,
    LineChart as RechartsLineChart,
    ReferenceLine,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from 'recharts';
import { options } from '@/utils/api';
import { OptionContract, ChainResponse, OptionsBacktestResult } from '@/types/api.types';

// Strategy Templates (Frontend helper)
const STRATEGY_TEMPLATES = [
    {
        id: 'iron_condor',
        name: 'Iron Condor',
        description: 'Neutral strategy profiting from low volatility',
        risk: 'Defined',
        sentiment: 'Neutral',
        icon: Shield
    },
    {
        id: 'covered_call',
        name: 'Covered Call',
        description: 'Generate income from long stock positions',
        risk: 'Low',
        sentiment: 'Bullish',
        icon: DollarSign
    },
    {
        id: 'long_straddle',
        name: 'Long Straddle',
        description: 'Profit from massive volatility in either direction',
        risk: 'Defined',
        sentiment: 'Volatile',
        icon: Activity
    },
    {
        id: 'bull_put_spread',
        name: 'Bull Put Spread',
        description: 'Bullish strategy with defined risk/reward',
        risk: 'Defined',
        sentiment: 'Bullish',
        icon: ArrowUpRight
    }
];

const OptionsDesk = () => {
    const [selectedSymbol, setSelectedSymbol] = useState('SPY');
    const [selectedExpiry, setSelectedExpiry] = useState<string>('');
    const [chainData, setChainData] = useState<ChainResponse | null>(null);
    const [isLoadingChain, setIsLoadingChain] = useState(false);

    // Backtest State
    const [activeTab, setActiveTab] = useState('chain');
    const [selectedStrategy, setSelectedStrategy] = useState(STRATEGY_TEMPLATES[0]);
    const [isBacktesting, setIsBacktesting] = useState(false);
    const [backtestResults, setBacktestResults] = useState<OptionsBacktestResult | null>(null);

    // Config for backtest
    const [backtestConfig, setBacktestConfig] = useState({
        startDate: '2023-01-01',
        endDate: '2023-12-31',
        capital: 100000,
        entryRules: {
            signal: 'regular',
            entry_frequency: 30,
            days_to_expiration: 45
        },
        exitRules: {
            profit_target: 0.50,
            loss_limit: -0.25,
            dte_exit: 21
        }
    });

    // Fetch Chain Data
    useEffect(() => {
        fetchChain();
    }, [selectedSymbol]);

    const fetchChain = async () => {
        setIsLoadingChain(true);
        try {
            const response = await options.getChain({
                symbol: selectedSymbol,
                expiration: selectedExpiry || undefined
            });
            if (response.data) {
                setChainData(response.data);
                if (!selectedExpiry && response.data.expiration_dates.length > 0) {
                    setSelectedExpiry(response.data.expiration_dates[0]);
                }
            }
        } catch (err) {
            console.error("Failed to fetch chain:", err);
        } finally {
            setIsLoadingChain(false);
        }
    };

    const handleRunBacktest = async () => {
        setIsBacktesting(true);
        try {
            const response = await options.backtest({
                symbol: selectedSymbol,
                strategy_type: selectedStrategy.id.toUpperCase(), // Backend expects uppercase enum logic often
                start_date: backtestConfig.startDate,
                end_date: backtestConfig.endDate,
                initial_capital: backtestConfig.capital,
                risk_free_rate: 0.04,
                entry_rules: backtestConfig.entryRules,
                exit_rules: backtestConfig.exitRules
            });
            if (response.data) {
                setBacktestResults(response.data);
            }
        } catch (err) {
            console.error("Backtest failed:", err);
        } finally {
            setIsBacktesting(false);
        }
    };

    // Helper to format currency
    const formatCurrency = (val: number) =>
        new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val);

    return (
        <div className="space-y-6 animate-in fade-in duration-700">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div className="flex items-center gap-4">
                    <div className="p-3 bg-gradient-to-br from-amber-500/20 to-orange-500/20 rounded-2xl border border-amber-500/30">
                        <Zap className="text-amber-400" size={32} />
                    </div>
                    <div>
                        <h1 className="text-3xl font-bold text-slate-100 tracking-tight">Options Desk</h1>
                        <p className="text-sm text-slate-500 font-medium">Advanced volatility analysis and strategy backtesting</p>
                    </div>
                </div>

                <div className="flex bg-slate-900/60 p-1 rounded-xl border border-slate-700/50 backdrop-blur-sm">
                    {['chain', 'backtest', 'volatility'].map((tab) => (
                        <button
                            key={tab}
                            onClick={() => setActiveTab(tab)}
                            className={`px-6 py-2.5 rounded-lg text-xs font-bold uppercase tracking-wider transition-all ${activeTab === tab
                                    ? 'bg-gradient-to-r from-amber-500 to-orange-500 text-white shadow-lg'
                                    : 'text-slate-500 hover:text-slate-300'
                                }`}
                        >
                            {tab}
                        </button>
                    ))}
                </div>
            </div>

            {/* Main Content Area */}
            {activeTab === 'chain' && (
                <div className="space-y-6">
                    {/* Controls */}
                    <div className="flex gap-4 bg-slate-900/50 p-4 rounded-xl border border-slate-800/50">
                        <div className="relative">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={16} />
                            <input
                                type="text"
                                value={selectedSymbol}
                                onChange={(e) => setSelectedSymbol(e.target.value.toUpperCase())}
                                className="pl-9 pr-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-200 outline-none focus:border-amber-500 w-32"
                            />
                        </div>
                        <select
                            value={selectedExpiry}
                            onChange={(e) => setSelectedExpiry(e.target.value)}
                            className="bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-200 px-4 outline-none focus:border-amber-500"
                        >
                            {chainData?.expiration_dates.map(date => (
                                <option key={date} value={date}>{date}</option>
                            ))}
                        </select>
                        <button onClick={fetchChain} className="p-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-amber-400 transition-colors">
                            <RefreshCw size={18} className={isLoadingChain ? "animate-spin" : ""} />
                        </button>
                    </div>

                    {/* Chain Table */}
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl overflow-hidden">
                        <div className="grid grid-cols-11 bg-slate-950/50 p-3 text-[10px] font-bold text-slate-500 uppercase tracking-wider text-center border-b border-slate-800">
                            <div className="col-span-5 grid grid-cols-5">
                                <span>Delta</span><span>Bid</span><span>Ask</span><span>Vol</span><span>OI</span>
                            </div>
                            <div className="col-span-1 text-slate-200 font-extrabold">Strike</div>
                            <div className="col-span-5 grid grid-cols-5">
                                <span>Bid</span><span>Ask</span><span>Vol</span><span>OI</span><span>Delta</span>
                            </div>
                        </div>

                        <div className="max-h-[600px] overflow-y-auto">
                            {chainData?.calls.map((call, i) => {
                                const put = chainData.puts[i]; // Assuming symmetric data for simplicity
                                if (!put) return null;

                                return (
                                    <div key={call.strike} className="grid grid-cols-11 hover:bg-slate-800/50 transition-colors border-b border-slate-800/30 text-xs py-2 text-center items-center">
                                        {/* CALLS */}
                                        <div className="col-span-1 text-slate-400">{call.delta.toFixed(2)}</div>
                                        <div className="col-span-1 text-emerald-400">{call.bid.toFixed(2)}</div>
                                        <div className="col-span-1 text-red-400">{call.ask.toFixed(2)}</div>
                                        <div className="col-span-1 text-slate-300">{call.volume}</div>
                                        <div className="col-span-1 text-slate-500">{call.openInterest}</div>

                                        {/* STRIKE */}
                                        <div className="col-span-1 bg-slate-800/50 py-1 rounded font-bold text-slate-200">
                                            {call.strike.toFixed(1)}
                                        </div>

                                        {/* PUTS */}
                                        <div className="col-span-1 text-emerald-400">{put.bid.toFixed(2)}</div>
                                        <div className="col-span-1 text-red-400">{put.ask.toFixed(2)}</div>
                                        <div className="col-span-1 text-slate-300">{put.volume}</div>
                                        <div className="col-span-1 text-slate-500">{put.openInterest}</div>
                                        <div className="col-span-1 text-slate-400">{put.delta.toFixed(2)}</div>
                                    </div>
                                )
                            })}
                        </div>
                    </div>
                </div>
            )}

            {activeTab === 'backtest' && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Sidebar Configuration */}
                    <div className="space-y-6">
                        <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6 space-y-4">
                            <h3 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                                <Settings size={16} className="text-amber-400" /> Backtest Config
                            </h3>

                            <div className="space-y-2">
                                <label className="text-xs font-medium text-slate-500">Strategy</label>
                                <div className="grid grid-cols-1 gap-2">
                                    {STRATEGY_TEMPLATES.map(s => (
                                        <button
                                            key={s.id}
                                            onClick={() => setSelectedStrategy(s)}
                                            className={`p-3 rounded-lg border text-left flex items-center gap-3 transition-colors ${selectedStrategy.id === s.id ? 'bg-amber-500/10 border-amber-500/50 text-amber-400' : 'bg-slate-800 border-slate-700 text-slate-400'}`}
                                        >
                                            <s.icon size={16} />
                                            <div>
                                                <div className="text-xs font-bold">{s.name}</div>
                                                <div className="text-[10px] opacity-70">{s.sentiment}</div>
                                            </div>
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="text-xs font-medium text-slate-500">Start Date</label>
                                    <input
                                        type="date"
                                        value={backtestConfig.startDate}
                                        onChange={(e) => setBacktestConfig({ ...backtestConfig, startDate: e.target.value })}
                                        className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1.5 text-xs text-slate-200 mt-1"
                                    />
                                </div>
                                <div>
                                    <label className="text-xs font-medium text-slate-500">End Date</label>
                                    <input
                                        type="date"
                                        value={backtestConfig.endDate}
                                        onChange={(e) => setBacktestConfig({ ...backtestConfig, endDate: e.target.value })}
                                        className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1.5 text-xs text-slate-200 mt-1"
                                    />
                                </div>
                            </div>

                            <button
                                onClick={handleRunBacktest}
                                disabled={isBacktesting}
                                className="w-full py-3 bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 text-white font-bold rounded-xl flex items-center justify-center gap-2 transition-all shadow-lg shadow-amber-500/20"
                            >
                                {isBacktesting ? <Loader2 className="animate-spin" size={18} /> : <Play size={18} />}
                                Run Backtest
                            </button>
                        </div>
                    </div>

                    {/* Results Area */}
                    <div className="lg:col-span-2 space-y-6">
                        {backtestResults ? (
                            <div className="space-y-6 animate-in slide-in-from-bottom-4">
                                {/* KPIs */}
                                <div className="grid grid-cols-4 gap-4">
                                    <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800/50">
                                        <div className="text-xs text-slate-500 mb-1">Total Return</div>
                                        <div className={`text-xl font-bold ${backtestResults.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                            {backtestResults.total_return > 0 && '+'}{backtestResults.total_return.toFixed(2)}%
                                        </div>
                                    </div>
                                    <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800/50">
                                        <div className="text-xs text-slate-500 mb-1">Win Rate</div>
                                        <div className="text-xl font-bold text-amber-400">
                                            {backtestResults.win_rate.toFixed(1)}%
                                        </div>
                                    </div>
                                    <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800/50">
                                        <div className="text-xs text-slate-500 mb-1">Profit Factor</div>
                                        <div className="text-xl font-bold text-blue-400">
                                            {backtestResults.profit_factor.toFixed(2)}
                                        </div>
                                    </div>
                                    <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800/50">
                                        <div className="text-xs text-slate-500 mb-1">Total Trades</div>
                                        <div className="text-xl font-bold text-slate-200">
                                            {backtestResults.total_trades}
                                        </div>
                                    </div>
                                </div>

                                {/* Equity Curve */}
                                <div className="bg-slate-900/50 p-6 rounded-xl border border-slate-800/50">
                                    <h4 className="text-sm font-bold text-slate-300 mb-4">Equity Curve</h4>
                                    <div className="h-[300px]">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <AreaChart data={backtestResults.equity_curve}>
                                                <defs>
                                                    <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                                                        <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.1} />
                                                        <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                                                    </linearGradient>
                                                </defs>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                                <XAxis dataKey="date" stroke="#64748b" fontSize={10} tickFormatter={(val) => val.split('T')[0]} />
                                                <YAxis stroke="#64748b" fontSize={10} />
                                                <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }} />
                                                <Area type="monotone" dataKey="equity" stroke="#f59e0b" fillOpacity={1} fill="url(#colorEquity)" />
                                            </AreaChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-slate-500">
                                <Activity size={48} className="mb-4 opacity-20" />
                                <p>Select a strategy and run backtest to see results</p>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default OptionsDesk;