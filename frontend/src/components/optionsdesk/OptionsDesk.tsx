'use client'
import React, {useEffect, useState} from 'react';
import {
    Activity,
    ArrowUpRight,
    DollarSign,
    Loader2,
    Play,
    RefreshCw,
    Search,
    Settings,
    Shield,
    Zap
} from "lucide-react";
import {
    Area,
    AreaChart,
    Bar,
    BarChart,
    CartesianGrid,
    Line, LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from 'recharts';
import {options} from '@/utils/api';
import {ChainResponse, OptionsBacktestResult} from '@/types/all_types';

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
    const [isLoadingExpirations, setIsLoadingExpirations] = useState(false);

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


    const fetchExpirations = async () => {
        setIsLoadingExpirations(true);
        try {
            console.log('📅 Fetching expiration dates for:', selectedSymbol);

            // ✅ CRITICAL FIX: Pass expiration as null/undefined to get all dates
            const response = await options.getChain({
                symbol: selectedSymbol,
                expiration: undefined  // or null - backend should handle this
            });

            console.log('📅 Expiration response:', response);

            if (response && response.expiration_dates && response.expiration_dates.length > 0) {
                // ✅ FIX: Only update chainData with expiration dates, don't overwrite chain data
                setChainData(prevData => ({
                    ...prevData,
                    symbol: response.symbol,
                    current_price: response.current_price,
                    expiration_dates: response.expiration_dates,
                    calls: prevData?.calls || [],
                    puts: prevData?.puts || []
                } as ChainResponse));

                // ✅ FIX: Auto-select first expiration date
                setSelectedExpiry(response.expiration_dates[0]);
            }
        } catch (err) {
            console.error("❌ Failed to fetch expirations:", err);
        } finally {
            setIsLoadingExpirations(false);
        }
    };

    // ✅ FIX 2: Fetch full chain for specific expiration
    const fetchChain = async () => {
        if (!selectedExpiry) {
            console.log('⚠️ No expiry selected, skipping chain fetch');
            return;
        }

        setIsLoadingChain(true);
        try {
            console.log('📊 Fetching chain for:', selectedSymbol, selectedExpiry);

            const response = await options.getChain({
                symbol: selectedSymbol,
                expiration: selectedExpiry  // ✅ Pass specific date
            });

            console.log('📊 Chain response:', response);

            if (response) {
                setChainData(response);
            }
        } catch (err) {
            console.error("❌ Failed to fetch chain:", err);
            // Show error to user
            alert(`Failed to fetch option chain: ${err instanceof Error ? err.message : 'Unknown error'}`);
        } finally {
            setIsLoadingChain(false);
        }
    };

    useEffect(() => {
        if (selectedSymbol) {
            fetchExpirations();
        }
    }, [selectedSymbol]); // ✅ Added dependency

    useEffect(() => {
        if (selectedExpiry && selectedSymbol) {
            fetchChain();
        }
    }, [selectedExpiry]);

    const handleRunBacktest = async () => {
        setIsBacktesting(true);
        try {
            console.log('🧪 Running backtest with config:', {
                symbol: selectedSymbol,
                strategy: selectedStrategy.id,
                ...backtestConfig
            });

            const response = await options.runBacktest({
                symbol: selectedSymbol,
                strategy_type: selectedStrategy.id, // Backend might expect lowercase
                start_date: backtestConfig.startDate,
                end_date: backtestConfig.endDate,
                initial_capital: backtestConfig.capital,
                risk_free_rate: 0.04,
                entry_rules: backtestConfig.entryRules,
                exit_rules: backtestConfig.exitRules
            });

            console.log('🧪 Backtest results:', response);

            if (response) {
                setBacktestResults(response);
            }
        } catch (err) {
            console.error("❌ Backtest failed:", err);
            alert(`Backtest failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
        } finally {
            setIsBacktesting(false);
        }
    };

    return (
        <div className="space-y-6 animate-in fade-in duration-700">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div className="flex items-center gap-4">
                    <div
                        className="p-3 bg-gradient-to-br from-amber-500/20 to-orange-500/20 rounded-2xl border border-amber-500/30">
                        <Zap className="text-amber-400" size={32}/>
                    </div>
                    <div>
                        <h1 className="text-3xl font-bold text-slate-100 tracking-tight">Options Desk</h1>
                        <p className="text-sm text-slate-500 font-medium">Advanced volatility analysis and strategy
                            backtesting</p>
                    </div>
                </div>

                <div className="flex bg-slate-900/60 p-1 rounded-xl border border-slate-700/50 backdrop-blur-sm">
                    {['chain', 'backtest', 'volatility'].map((tab) => (
                        <button
                            key={tab}
                            onClick={() => setActiveTab(tab)}
                            className={`px-6 py-2.5 rounded-lg text-xs font-bold uppercase tracking-wider transition-all ${
                                activeTab === tab
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
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={16}/>
                            <input
                                type="text"
                                value={selectedSymbol}
                                onChange={(e) => setSelectedSymbol(e.target.value.toUpperCase())}
                                onKeyDown={(e) => e.key === 'Enter' && fetchExpirations()}
                                className="pl-9 pr-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-200 outline-none focus:border-amber-500 w-32"
                                placeholder="Symbol..."
                            />
                        </div>

                        <div className="flex items-center gap-2">
                            <label className="text-xs font-bold text-slate-500 uppercase">Expiry:</label>
                            <select
                                value={selectedExpiry}
                                onChange={(e) => setSelectedExpiry(e.target.value)}
                                disabled={isLoadingExpirations}
                                className="bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-200 px-4 py-2 outline-none focus:border-amber-500 min-w-[140px] disabled:opacity-50"
                            >
                                {isLoadingExpirations && <option>Loading...</option>}
                                {!isLoadingExpirations && !chainData?.expiration_dates?.length && (
                                    <option>No dates available</option>
                                )}
                                {chainData?.expiration_dates?.map(date => (
                                    <option key={date} value={date}>{date}</option>
                                ))}
                            </select>
                        </div>

                        <button
                            onClick={fetchChain}
                            disabled={isLoadingChain || !selectedExpiry}
                            className="p-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-amber-400 transition-colors flex items-center gap-2 px-3 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <RefreshCw size={18} className={isLoadingChain ? "animate-spin" : ""}/>
                            <span className="text-xs font-bold">Refresh</span>
                        </button>
                    </div>

                    {/* Chain Table */}
                    {isLoadingChain ? (
                        <div
                            className="flex items-center justify-center h-64 bg-slate-900/50 border border-slate-800/50 rounded-xl">
                            <div className="flex flex-col items-center gap-3">
                                <Loader2 size={32} className="animate-spin text-amber-400"/>
                                <p className="text-slate-400">Loading option chain...</p>
                            </div>
                        </div>
                    ) : chainData?.calls && chainData.calls.length > 0 ? (
                        <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl overflow-hidden">
                            <div
                                className="grid grid-cols-11 bg-slate-950/50 p-3 text-[10px] font-bold text-slate-500 uppercase tracking-wider text-center border-b border-slate-800">
                                <div className="col-span-5 grid grid-cols-5">
                                    <span>Delta</span><span>Bid</span><span>Ask</span><span>Vol</span><span>OI</span>
                                </div>
                                <div className="col-span-1 text-slate-200 font-extrabold">Strike</div>
                                <div className="col-span-5 grid grid-cols-5">
                                    <span>Bid</span><span>Ask</span><span>Vol</span><span>OI</span><span>Delta</span>
                                </div>
                            </div>

                            <div className="max-h-[600px] overflow-y-auto">
                                {chainData.calls.map((call, i) => {
                                    const put = chainData.puts[i];
                                    if (!put) return null;

                                    return (
                                        <div key={call.strike}
                                             className="grid grid-cols-11 hover:bg-slate-800/50 transition-colors border-b border-slate-800/30 text-xs py-2 text-center items-center">
                                            {/* CALLS */}
                                            <div
                                                className="col-span-1 text-slate-400">{call.delta?.toFixed(2) ?? 'N/A'}</div>
                                            <div
                                                className="col-span-1 text-emerald-400">{call.bid?.toFixed(2) ?? '0.00'}</div>
                                            <div
                                                className="col-span-1 text-red-400">{call.ask?.toFixed(2) ?? '0.00'}</div>
                                            <div className="col-span-1 text-slate-300">{call.volume ?? 0}</div>
                                            <div className="col-span-1 text-slate-500">{call.openInterest ?? 0}</div>

                                            {/* STRIKE */}
                                            <div
                                                className="col-span-1 bg-slate-800/50 py-1 rounded font-bold text-slate-200">
                                                {call.strike.toFixed(1)}
                                            </div>

                                            {/* PUTS */}
                                            <div
                                                className="col-span-1 text-emerald-400">{put.bid?.toFixed(2) ?? '0.00'}</div>
                                            <div
                                                className="col-span-1 text-red-400">{put.ask?.toFixed(2) ?? '0.00'}</div>
                                            <div className="col-span-1 text-slate-300">{put.volume ?? 0}</div>
                                            <div className="col-span-1 text-slate-500">{put.openInterest ?? 0}</div>
                                            <div
                                                className="col-span-1 text-slate-400">{put.delta?.toFixed(2) ?? 'N/A'}</div>
                                        </div>
                                    )
                                })}
                            </div>
                        </div>
                    ) : (
                        <div
                            className="flex items-center justify-center h-64 bg-slate-900/50 border border-slate-800/50 rounded-xl">
                            <div className="text-center text-slate-500">
                                <Activity size={48} className="mx-auto mb-4 opacity-20"/>
                                <p>No option chain data available</p>
                                <p className="text-sm mt-2">Select a symbol and expiration date</p>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {activeTab === 'backtest' && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Sidebar Configuration */}
                    <div className="space-y-6">
                        <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6 space-y-4">
                            <h3 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                                <Settings size={16} className="text-amber-400"/> Backtest Config
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
                                            <s.icon size={16}/>
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
                                        onChange={(e) => setBacktestConfig({
                                            ...backtestConfig,
                                            startDate: e.target.value
                                        })}
                                        className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1.5 text-xs text-slate-200 mt-1"
                                    />
                                </div>
                                <div>
                                    <label className="text-xs font-medium text-slate-500">End Date</label>
                                    <input
                                        type="date"
                                        value={backtestConfig.endDate}
                                        onChange={(e) => setBacktestConfig({
                                            ...backtestConfig,
                                            endDate: e.target.value
                                        })}
                                        className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1.5 text-xs text-slate-200 mt-1"
                                    />
                                </div>
                            </div>

                            <button
                                onClick={handleRunBacktest}
                                disabled={isBacktesting}
                                className="w-full py-3 bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 text-white font-bold rounded-xl flex items-center justify-center gap-2 transition-all shadow-lg shadow-amber-500/20"
                            >
                                {isBacktesting ? <Loader2 className="animate-spin" size={18}/> : <Play size={18}/>}
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
                                        <div
                                            className={`text-xl font-bold ${backtestResults.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
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
                                                        <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.1}/>
                                                        <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                                                    </linearGradient>
                                                </defs>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                                                <XAxis dataKey="date" stroke="#64748b" fontSize={10}
                                                       tickFormatter={(val) => val.split('T')[0]}/>
                                                <YAxis stroke="#64748b" fontSize={10}/>
                                                <Tooltip contentStyle={{
                                                    backgroundColor: '#0f172a',
                                                    border: '1px solid #1e293b'
                                                }}/>
                                                <Area type="monotone" dataKey="equity" stroke="#f59e0b" fillOpacity={1}
                                                      fill="url(#colorEquity)"/>
                                            </AreaChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div
                                className="flex flex-col items-center justify-center h-full min-h-[400px] text-slate-500">
                                <Activity size={48} className="mb-4 opacity-20"/>
                                <p>Select a strategy and run backtest to see results</p>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {activeTab === 'volatility' && (
                <div className="space-y-6">
                    {chainData && selectedExpiry ? (
                        <>
                            {/* Implied Volatility Surface */}
                            <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                                <h3 className="text-sm font-bold text-slate-300 mb-6 flex items-center gap-2">
                                    <Activity size={16} className="text-amber-400"/>
                                    Implied Volatility Surface
                                </h3>
                                <div className="h-[400px]">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart
                                            data={chainData.calls.map((call, idx) => ({
                                                strike: call.strike,
                                                callIV: call.impliedVolatility ? call.impliedVolatility * 100 : 0,
                                                putIV: chainData.puts[idx]?.impliedVolatility
                                                    ? chainData.puts[idx].impliedVolatility * 100
                                                    : 0
                                            }))}
                                        >
                                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                                            <XAxis
                                                dataKey="strike"
                                                stroke="#64748b"
                                                fontSize={10}
                                                label={{value: 'Strike Price', position: 'insideBottom', offset: -5}}
                                            />
                                            <YAxis
                                                stroke="#64748b"
                                                fontSize={10}
                                                label={{value: 'IV (%)', angle: -90, position: 'insideLeft'}}
                                            />
                                            <Tooltip
                                                contentStyle={{
                                                    backgroundColor: '#0f172a',
                                                    border: '1px solid #1e293b'
                                                }}
                                                formatter={(value: any) => `${Number(value).toFixed(2)}%`}
                                            />
                                            <Line
                                                type="monotone"
                                                dataKey="callIV"
                                                stroke="#10b981"
                                                strokeWidth={2}
                                                name="Call IV"
                                                dot={false}
                                            />
                                            <Line
                                                type="monotone"
                                                dataKey="putIV"
                                                stroke="#ef4444"
                                                strokeWidth={2}
                                                name="Put IV"
                                                dot={false}
                                            />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            {/* IV Stats Grid */}
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                {/* ATM IV */}
                                <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                                    <h4 className="text-xs font-bold text-slate-500 uppercase mb-4">ATM Implied
                                        Volatility</h4>
                                    {(() => {
                                        // Find ATM strike (closest to current price)
                                        const atmStrike = chainData.calls.reduce((prev, curr) =>
                                            Math.abs(curr.strike - (chainData.current_price || 0)) <
                                            Math.abs(prev.strike - (chainData.current_price || 0)) ? curr : prev
                                        );
                                        const atmIV = atmStrike.impliedVolatility || 0;

                                        return (
                                            <>
                                                <div className="text-3xl font-bold text-amber-400 mb-2">
                                                    {(atmIV * 100).toFixed(2)}%
                                                </div>
                                                <div className="text-sm text-slate-400">
                                                    Strike: ${atmStrike.strike.toFixed(2)}
                                                </div>
                                            </>
                                        );
                                    })()}
                                </div>

                                {/* IV Range */}
                                <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                                    <h4 className="text-xs font-bold text-slate-500 uppercase mb-4">IV Range</h4>
                                    {(() => {
                                        const allIVs = [
                                            ...chainData.calls.map(c => c.impliedVolatility || 0),
                                            ...chainData.puts.map(p => p.impliedVolatility || 0)
                                        ].filter(iv => iv > 0);

                                        const minIV = Math.min(...allIVs);
                                        const maxIV = Math.max(...allIVs);

                                        return (
                                            <>
                                                <div className="flex items-center justify-between mb-2">
                                                    <span className="text-sm text-slate-400">Low</span>
                                                    <span className="text-lg font-bold text-emerald-400">
                                            {(minIV * 100).toFixed(2)}%
                                        </span>
                                                </div>
                                                <div className="flex items-center justify-between">
                                                    <span className="text-sm text-slate-400">High</span>
                                                    <span className="text-lg font-bold text-red-400">
                                            {(maxIV * 100).toFixed(2)}%
                                        </span>
                                                </div>
                                            </>
                                        );
                                    })()}
                                </div>

                                {/* Volatility Skew */}
                                <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                                    <h4 className="text-xs font-bold text-slate-500 uppercase mb-4">Volatility Skew</h4>
                                    {(() => {
                                        // Calculate skew (put IV - call IV for same strike)
                                        const skews = chainData.calls.map((call, idx) => {
                                            const put = chainData.puts[idx];
                                            if (!put || !call.impliedVolatility || !put.impliedVolatility) return 0;
                                            return (put.impliedVolatility - call.impliedVolatility) * 100;
                                        });

                                        const avgSkew = skews.reduce((a, b) => a + b, 0) / skews.length;

                                        return (
                                            <>
                                                <div className="text-3xl font-bold text-blue-400 mb-2">
                                                    {avgSkew > 0 ? '+' : ''}{avgSkew.toFixed(2)}%
                                                </div>
                                                <div className="text-sm text-slate-400">
                                                    {avgSkew > 0 ? 'Put skew (bearish)' : 'Call skew (bullish)'}
                                                </div>
                                            </>
                                        );
                                    })()}
                                </div>
                            </div>

                            {/* Volume & Open Interest Analysis */}
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                {/* Call Volume */}
                                <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                                    <h4 className="text-sm font-bold text-slate-300 mb-4">Call Volume by Strike</h4>
                                    <div className="h-[250px]">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <BarChart data={chainData.calls.slice(0, 20)}>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                                                <XAxis dataKey="strike" stroke="#64748b" fontSize={10}/>
                                                <YAxis stroke="#64748b" fontSize={10}/>
                                                <Tooltip
                                                    contentStyle={{
                                                        backgroundColor: '#0f172a',
                                                        border: '1px solid #1e293b'
                                                    }}
                                                />
                                                <Bar dataKey="volume" fill="#10b981" radius={[4, 4, 0, 0]}/>
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>

                                {/* Put Volume */}
                                <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                                    <h4 className="text-sm font-bold text-slate-300 mb-4">Put Volume by Strike</h4>
                                    <div className="h-[250px]">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <BarChart data={chainData.puts.slice(0, 20)}>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                                                <XAxis dataKey="strike" stroke="#64748b" fontSize={10}/>
                                                <YAxis stroke="#64748b" fontSize={10}/>
                                                <Tooltip
                                                    contentStyle={{
                                                        backgroundColor: '#0f172a',
                                                        border: '1px solid #1e293b'
                                                    }}
                                                />
                                                <Bar dataKey="volume" fill="#ef4444" radius={[4, 4, 0, 0]}/>
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>
                            </div>
                        </>
                    ) : (
                        <div className="flex flex-col items-center justify-center h-[400px] text-slate-500">
                            <Activity size={48} className="mb-4 opacity-20"/>
                            <p>Load an options chain to view volatility analysis</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default OptionsDesk;