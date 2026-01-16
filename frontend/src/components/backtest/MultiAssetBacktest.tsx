

'use client'

import {
    Activity, AlertCircle, BarChart3, Calendar, Check, ChevronDown, ChevronRight,
    Clock, Cpu, Database, Download, Filter, HelpCircle, Info, Layers, Lock,
    Play, PlusCircle, RefreshCw, Save, Search, Settings, Shield, Sliders,
    TrendingUp, TrendingDown, Users, X, Zap, Target, TrendingUp as TrendUp,
    PieChart, DollarSign, BarChart, LineChart, Grid, List, Eye, EyeOff, Copy
} from "lucide-react";
import React, { useMemo, useState } from "react";
import StrategyParameterForm from "@/components/backtest/StrategyParameterForm";
import MultiBacktestResults from "@/components/backtest/MultiBacktestResults";
import { BacktestResult, Strategy } from "@/types/backtest";

interface MultiAssetBacktestProps {
    config: any;
    setConfig: (config: any) => void;
    strategies: Strategy[];
    runBacktest: () => Promise<void>;
    isRunning: boolean;
    results: BacktestResult | null;
    addSymbol: () => void;
    removeSymbol: (symbol: string) => void;
}

const MultiAssetBacktest: React.FC<MultiAssetBacktestProps> = ({
    config,
    setConfig,
    strategies,
    runBacktest,
    isRunning,
    results,
    addSymbol,
    removeSymbol
}) => {
    const [activeCategory, setActiveCategory] = useState<string>('All');
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [hasRunBacktest, setHasRunBacktest] = useState(false);
    const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
    const [showParameters, setShowParameters] = useState(true);
    const [allocationMode, setAllocationMode] = useState<'equal' | 'manual' | 'optimized'>('equal');

    // Find the currently active strategy
    const selectedStrategy = useMemo(() =>
        strategies.find((s: Strategy) => s.id === config.strategy),
        [config.strategy, strategies]
    );

    // Get unique categories
    const categories = useMemo(() =>
        ['All', ...Array.from(new Set(strategies.map((s: Strategy) => s.category)))],
        [strategies]
    );

    // Filter strategies by category
    const filteredStrategies = useMemo(() =>
        activeCategory === 'All'
            ? strategies
            : strategies.filter((s: Strategy) => s.category === activeCategory),
        [activeCategory, strategies]
    );

    // Update parameters
    const handleParamChange = (key: string, val: any) => {
        setConfig({
            ...config,
            params: { ...(config.params || {}), [key]: val }
        });
    };

    // Handle backtest execution
    const handleRunBacktest = async () => {
        setHasRunBacktest(false);
        await runBacktest();
        setHasRunBacktest(true);
    };

    // Asset suggestions
    const assetSuggestions = [
        { symbol: 'AAPL', name: 'Apple Inc.', sector: 'Technology', color: 'from-gray-500 to-slate-500' },
        { symbol: 'MSFT', name: 'Microsoft', sector: 'Technology', color: 'from-blue-500 to-cyan-500' },
        { symbol: 'GOOGL', name: 'Alphabet', sector: 'Technology', color: 'from-red-500 to-orange-500' },
        { symbol: 'AMZN', name: 'Amazon', sector: 'Consumer', color: 'from-amber-500 to-yellow-500' },
        { symbol: 'TSLA', name: 'Tesla', sector: 'Automotive', color: 'from-emerald-500 to-green-500' },
        { symbol: 'NVDA', name: 'NVIDIA', sector: 'Semiconductors', color: 'from-green-500 to-emerald-500' },
        { symbol: 'JPM', name: 'JPMorgan', sector: 'Financial', color: 'from-blue-500 to-indigo-500' },
        { symbol: 'BTC-USD', name: 'Bitcoin', sector: 'Crypto', color: 'from-orange-500 to-amber-500' },
    ];

    return (
        <div className="space-y-6">
            {/* Header Section */}
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end gap-4 pb-6 border-b border-slate-800/80">
                <div className="flex items-start gap-4">
                    <div className="p-3 bg-gradient-to-br from-violet-500/20 to-purple-500/20 rounded-2xl border border-violet-500/30 shadow-xl shadow-violet-500/10">
                        <BarChart3 className="text-violet-400" size={28} strokeWidth={2} />
                    </div>
                    <div>
                        <h3 className="text-2xl font-bold text-slate-100 tracking-tight">
                            Multi-Asset <span className="text-slate-400 font-normal">Portfolio</span>
                        </h3>
                        <p className="text-sm text-slate-500 font-medium mt-1">Configure and backtest diversified trading strategies</p>
                    </div>
                </div>

                <div className="flex flex-col sm:flex-row gap-3">
                    <button
                        onClick={handleRunBacktest}
                        disabled={isRunning || config.symbols.length < 2}
                        className="group relative overflow-hidden flex items-center space-x-3 px-7 py-3.5 bg-gradient-to-r from-violet-600 via-purple-600 to-fuchsia-600 hover:from-violet-500 hover:via-purple-500 hover:to-fuchsia-500 disabled:from-slate-700 disabled:via-slate-700 disabled:to-slate-700 disabled:cursor-not-allowed rounded-xl font-bold transition-all shadow-xl shadow-violet-500/30 disabled:shadow-none text-white"
                    >
                        <div className="absolute inset-0 bg-gradient-to-r from-white/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                        {isRunning ? (
                            <>
                                <RefreshCw size={20} className="animate-spin relative z-10" strokeWidth={2.5} />
                                <span className="relative z-10">Processing...</span>
                            </>
                        ) : (
                            <>
                                <Play size={20} strokeWidth={2.5} className="relative z-10" />
                                <span className="relative z-10">Run Backtest</span>
                            </>
                        )}
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-12 gap-6">
                {/* LEFT: Asset & Strategy Selection */}
                <div className="col-span-12 lg:col-span-8 space-y-6">
                    {/* Enhanced Asset Management Card */}
                    <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                        <div className="flex justify-between items-center mb-4">
                            <div className="flex items-center gap-3">
                                <div className="p-2 bg-violet-500/20 rounded-lg border border-violet-500/30">
                                    <Database className="text-violet-400" size={20} strokeWidth={2} />
                                </div>
                                <div>
                                    <h4 className="text-sm font-bold text-slate-300">Portfolio Assets</h4>
                                    <p className="text-xs text-slate-500 font-medium mt-0.5">
                                        {config.symbols.length} selected {config.symbols.length < 2 && '(minimum 2 required)'}
                                    </p>
                                </div>
                            </div>
                            <div className="flex gap-2">
                                {config.symbols.length > 0 && (
                                    <button
                                        onClick={() => setConfig({ ...config, symbols: [] })}
                                        className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold text-red-400 hover:text-red-300 transition-colors bg-red-500/10 rounded-lg border border-red-500/20"
                                    >
                                        <X size={14} />
                                        <span>Clear All</span>
                                    </button>
                                )}
                                <button className="p-2 hover:bg-slate-800 rounded-lg transition-colors text-slate-500 hover:text-violet-400">
                                    <HelpCircle size={18} />
                                </button>
                            </div>
                        </div>

                        {/* Enhanced Asset Input with Suggestions */}
                        <div className="space-y-4 mb-6">
                            <div className="flex space-x-3">
                                <div className="relative flex-1 group">
                                    <Search
                                        className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-600 group-focus-within:text-violet-500 transition-colors"
                                        size={18}
                                    />
                                    <input
                                        type="text"
                                        placeholder="Search ticker (e.g. AAPL, TSLA, BTC-USD)"
                                        value={config.symbolInput}
                                        onChange={(e) => setConfig({ ...config, symbolInput: e.target.value.toUpperCase() })}
                                        onKeyPress={(e) => e.key === 'Enter' && addSymbol()}
                                        className="w-full pl-12 pr-4 py-3.5 bg-slate-800/50 border border-slate-700/50 rounded-xl focus:border-violet-500 focus:ring-2 focus:ring-violet-500/20 outline-none transition-all placeholder:text-slate-600 font-mono text-sm text-slate-200"
                                    />
                                </div>
                                <button
                                    onClick={addSymbol}
                                    className="group relative overflow-hidden px-6 py-3.5 bg-gradient-to-r from-slate-800/60 to-slate-900/60 border border-slate-700/50 hover:border-slate-600/50 rounded-xl font-semibold text-sm transition-all text-slate-200 flex items-center space-x-2"
                                >
                                    <div className="absolute inset-0 bg-gradient-to-r from-violet-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                                    <PlusCircle size={18} strokeWidth={2} className="relative z-10" />
                                    <span className="relative z-10">Add Asset</span>
                                </button>
                            </div>

                            {/* Asset Suggestions */}
                            <div className="flex flex-wrap gap-2">
                                {assetSuggestions.map((asset) => (
                                    <button
                                        key={asset.symbol}
                                        onClick={() => {
                                            setConfig({
                                                ...config,
                                                symbolInput: asset.symbol,
                                            });
                                            setTimeout(addSymbol, 100);
                                        }}
                                        className="group relative overflow-hidden px-3 py-2 bg-gradient-to-br from-slate-800/50 to-slate-900/50 border border-slate-700/50 hover:border-violet-500/50 rounded-lg transition-all"
                                    >
                                        <div className={`absolute inset-0 bg-gradient-to-br ${asset.color} opacity-0 group-hover:opacity-10 transition-opacity`} />
                                        <div className="relative flex items-center gap-2">
                                            <div className="w-6 h-6 rounded bg-gradient-to-br from-slate-700 to-slate-800 flex items-center justify-center">
                                                <span className="text-xs font-bold text-slate-400">{asset.symbol.charAt(0)}</span>
                                            </div>
                                            <div className="text-left">
                                                <p className="text-xs font-bold text-slate-300">{asset.symbol}</p>
                                                <p className="text-[10px] text-slate-500">{asset.sector}</p>
                                            </div>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Enhanced Asset Pills */}
                        <div className="min-h-[100px] p-4 bg-slate-900/30 border border-slate-700/30 rounded-xl">
                            {config.symbols.length > 0 ? (
                                <div className="flex flex-wrap gap-3">
                                    {config.symbols.map((symbol: string, index: number) => {
                                        const suggestion = assetSuggestions.find(a => a.symbol === symbol);
                                        return (
                                            <div
                                                key={symbol}
                                                className="group relative overflow-hidden flex items-center space-x-3 pl-4 pr-3 py-3 bg-gradient-to-br from-slate-800/50 to-slate-900/50 border border-slate-700/50 hover:border-violet-500/50 rounded-xl transition-all shadow-lg"
                                            >
                                                <div className={`absolute inset-0 bg-gradient-to-br ${suggestion?.color || 'from-violet-500/20 to-purple-500/20'} opacity-5 group-hover:opacity-10`} />
                                                <div className="relative flex items-center space-x-3">
                                                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-slate-700 to-slate-800 border border-slate-600/50 flex items-center justify-center">
                                                        <span className="font-mono font-bold text-lg text-violet-300">{symbol.slice(0, 2)}</span>
                                                    </div>
                                                    <div>
                                                        <p className="font-mono font-bold text-sm text-slate-200">{symbol}</p>
                                                        {suggestion && (
                                                            <p className="text-xs text-slate-500">{suggestion.name}</p>
                                                        )}
                                                    </div>
                                                    <button
                                                        onClick={() => removeSymbol(symbol)}
                                                        className="p-1.5 text-slate-500 hover:text-red-400 transition-colors rounded-lg hover:bg-red-500/10"
                                                    >
                                                        <X size={16} strokeWidth={2.5} />
                                                    </button>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            ) : (
                                <div className="w-full flex flex-col items-center justify-center py-8">
                                    <div className="w-16 h-16 rounded-full bg-gradient-to-br from-slate-800 to-slate-900 border-2 border-dashed border-slate-700/50 flex items-center justify-center mb-4">
                                        <PlusCircle size={28} className="text-slate-700" strokeWidth={1.5} />
                                    </div>
                                    <p className="text-sm font-semibold text-slate-600">No assets selected</p>
                                    <p className="text-xs text-slate-700 text-center mt-1 max-w-md">
                                        Add at least 2 symbols to configure your portfolio. <br />
                                        Use suggestions above or type custom tickers.
                                    </p>
                                </div>
                            )}
                        </div>

                        {/* Allocation Controls */}
                        {config.symbols.length >= 2 && (
                            <div className="mt-6 pt-6 border-t border-slate-700/50">
                                <div className="flex justify-between items-center mb-4">
                                    <h5 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                                        <PieChart size={16} className="text-violet-400" />
                                        Portfolio Allocation
                                    </h5>
                                    <div className="flex bg-slate-800/60 p-1 rounded-lg border border-slate-700/50">
                                        {['equal', 'manual', 'optimized'].map((mode) => (
                                            <button
                                                key={mode}
                                                onClick={() => setAllocationMode(mode as any)}
                                                className={`px-3 py-1.5 rounded-md text-xs font-bold uppercase transition-all ${allocationMode === mode
                                                    ? 'bg-gradient-to-r from-violet-600 to-purple-600 text-white'
                                                    : 'text-slate-500 hover:text-slate-300'
                                                    }`}
                                            >
                                                {mode}
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {allocationMode === 'manual' && (
                                    <div className="space-y-3">
                                        {config.symbols.map((symbol: string) => (
                                            <div key={symbol} className="flex items-center justify-between">
                                                <span className="text-sm text-slate-400 font-medium">{symbol}</span>
                                                <div className="flex items-center gap-3">
                                                    <input
                                                        type="range"
                                                        min="0"
                                                        max="100"
                                                        value={config.allocations[symbol] || 100 / config.symbols.length}
                                                        onChange={(e) => setConfig({
                                                            ...config,
                                                            allocations: {
                                                                ...config.allocations,
                                                                [symbol]: parseInt(e.target.value)
                                                            }
                                                        })}
                                                        className="w-32 accent-violet-500"
                                                    />
                                                    <span className="text-sm font-mono text-violet-400 w-12">
                                                        {config.allocations[symbol] || 100 / config.symbols.length}%
                                                    </span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}

                                {allocationMode === 'equal' && (
                                    <div className="p-4 bg-gradient-to-br from-violet-500/5 to-purple-500/5 border border-violet-500/20 rounded-xl">
                                        <p className="text-xs text-slate-400">
                                            <span className="font-bold text-violet-400">Equal allocation:</span> Each asset receives {(100 / config.symbols.length).toFixed(1)}% of the portfolio capital.
                                        </p>
                                    </div>
                                )}

                                {allocationMode === 'optimized' && (
                                    <div className="p-4 bg-gradient-to-br from-emerald-500/5 to-green-500/5 border border-emerald-500/20 rounded-xl">
                                        <p className="text-xs text-slate-400">
                                            <span className="font-bold text-emerald-400">Optimized allocation:</span> Capital allocation will be optimized based on risk-adjusted returns and correlation analysis.
                                        </p>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Enhanced Strategy Selector Card */}
                    <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
                            <div className="flex items-center gap-3">
                                <div className="p-2 bg-purple-500/20 rounded-lg border border-purple-500/30">
                                    <Layers className="text-purple-400" size={20} strokeWidth={2} />
                                </div>
                                <div>
                                    <h4 className="text-sm font-bold text-slate-300">Strategy Library</h4>
                                    <p className="text-xs text-slate-500 font-medium mt-0.5">
                                        {filteredStrategies.length} of {strategies.length} strategies
                                    </p>
                                </div>
                            </div>

                            <div className="flex gap-3">
                                {/* Category Filter */}
                                <div className="flex items-center space-x-2">
                                    <Filter size={14} className="text-slate-600" />
                                    <select
                                        value={activeCategory}
                                        onChange={(e) => setActiveCategory(e.target.value)}
                                        className="px-4 py-2 bg-slate-800/60 border border-slate-700/50 rounded-xl text-xs font-bold text-slate-300 focus:border-violet-500 outline-none cursor-pointer"
                                    >
                                        {categories.map((cat) => (
                                            <option key={cat} value={cat}>{cat}</option>
                                        ))}
                                    </select>
                                </div>

                                {/* View Toggle */}
                                <div className="flex bg-slate-800/60 border border-slate-700/50 rounded-xl overflow-hidden">
                                    <button
                                        onClick={() => setViewMode('grid')}
                                        className={`px-3 py-2 transition-all ${viewMode === 'grid' ? 'bg-slate-800 text-slate-200' : 'text-slate-500 hover:text-slate-300'
                                            }`}
                                    >
                                        <Grid size={16} />
                                    </button>
                                    <button
                                        onClick={() => setViewMode('list')}
                                        className={`px-3 py-2 transition-all ${viewMode === 'list' ? 'bg-slate-800 text-slate-200' : 'text-slate-500 hover:text-slate-300'
                                            }`}
                                    >
                                        <List size={16} />
                                    </button>
                                </div>
                            </div>
                        </div>

                        {/* Strategy Grid/List */}
                        <div className={`
                            ${viewMode === 'grid' ? 'grid grid-cols-2 md:grid-cols-3 gap-3' : 'space-y-2'} 
                            max-h-[400px] overflow-y-auto pr-2 custom-scrollbar
                        `}>
                            {filteredStrategies.map((strat: any) => {
                                const isSelected = config.strategy === strat.id;
                                return (
                                    <button
                                        key={strat.id}
                                        onClick={() => setConfig({ ...config, strategy: strat.id, params: strat.params })}
                                        className={`group relative overflow-hidden p-4 rounded-xl border transition-all text-left ${isSelected
                                            ? 'border-violet-500 bg-gradient-to-br from-violet-500/10 to-purple-500/10 shadow-xl shadow-violet-500/20'
                                            : 'border-slate-700/50 bg-slate-800/40 hover:border-slate-600/50 hover:bg-slate-800/60'
                                            }`}
                                    >
                                        {isSelected && (
                                            <div className="absolute inset-0 bg-gradient-to-br from-violet-500/5 to-purple-500/5" />
                                        )}
                                        <div className="relative">
                                            <div className="flex items-center justify-between mb-2">
                                                <span className={`text-[9px] font-bold uppercase tracking-wider px-2 py-1 rounded ${isSelected
                                                    ? 'bg-violet-500/20 text-violet-300 border border-violet-500/30'
                                                    : 'bg-slate-700/50 text-slate-400'
                                                    }`}>
                                                    {strat.category}
                                                </span>
                                                {isSelected && (
                                                    <div className="w-6 h-6 rounded-full bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center">
                                                        <Check size={12} className="text-white" strokeWidth={3} />
                                                    </div>
                                                )}
                                            </div>
                                            <p className="text-sm font-bold text-slate-200 mb-2 group-hover:text-violet-300 transition-colors">
                                                {strat.name}
                                            </p>
                                            <p className="text-xs text-slate-400 line-clamp-2 mb-3">{strat.description}</p>
                                            <div className="flex items-center justify-between">
                                                <span className={`text-[10px] px-2.5 py-1 rounded-lg font-medium ${strat.complexity === 'Advanced'
                                                    ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                                                    : strat.complexity === 'Intermediate'
                                                        ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                                                        : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                                                    }`}>
                                                    {strat.complexity}
                                                </span>
                                                <div className="flex items-center gap-1 text-amber-400">
                                                    <Star size={10} className="fill-current" />
                                                    <span className="text-xs font-bold">{strat.rating || 4.5}</span>
                                                </div>
                                            </div>
                                        </div>
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    {/* Enhanced Backtest Configuration */}
                    {config.symbols.length >= 2 && (
                        <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                            <div className="flex items-center gap-3 mb-6">
                                <div className="p-2 bg-gradient-to-br from-purple-500/20 to-violet-500/20 rounded-lg border border-purple-500/30">
                                    <Sliders className="text-purple-400" size={20} strokeWidth={2} />
                                </div>
                                <h4 className="text-sm font-bold text-slate-300">Backtest Configuration</h4>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                                <div className="space-y-3">
                                    <label className="text-xs font-bold text-slate-400 tracking-wide flex items-center gap-2">
                                        <Calendar size={14} className="text-violet-400" />
                                        Time Period
                                    </label>
                                    <select
                                        value={config.period}
                                        onChange={(e) => setConfig({ ...config, period: e.target.value })}
                                        className="w-full px-4 py-3.5 bg-slate-800/50 border border-slate-700/50 rounded-xl focus:border-violet-500 focus:ring-2 focus:ring-violet-500/20 outline-none text-sm text-slate-200"
                                    >
                                        <option value="1mo">1 Month</option>
                                        <option value="3mo">3 Months</option>
                                        <option value="6mo">6 Months</option>
                                        <option value="1y">1 Year</option>
                                        <option value="2y">2 Years</option>
                                        <option value="5y">5 Years</option>
                                    </select>
                                </div>
                                <div className="space-y-3">
                                    <label className="text-xs font-bold text-slate-400 tracking-wide flex items-center gap-2">
                                        <Activity size={14} className="text-violet-400" />
                                        Data Interval
                                    </label>
                                    <select
                                        value={config.interval}
                                        onChange={(e) => setConfig({ ...config, interval: e.target.value })}
                                        className="w-full px-4 py-3.5 bg-slate-800/50 border border-slate-700/50 rounded-xl focus:border-violet-500 focus:ring-2 focus:ring-violet-500/20 outline-none text-sm text-slate-200"
                                    >
                                        <option value="1h">1 Hour</option>
                                        <option value="4h">4 Hours</option>
                                        <option value="1d">1 Day</option>
                                        <option value="1wk">1 Week</option>
                                    </select>
                                </div>
                            </div>

                            {/* Strategy Mode Toggle */}
                            <div className="space-y-4 mb-6 pt-6 border-t border-slate-700/50">
                                <div className="flex justify-between items-center">
                                    <div className="flex items-center gap-2">
                                        <div className="p-1.5 bg-violet-500/20 rounded border border-violet-500/30">
                                            <Cpu size={14} className="text-violet-400" />
                                        </div>
                                        <label className="text-xs font-bold text-slate-400 tracking-wide">Strategy Assignment</label>
                                    </div>
                                    <div className="flex bg-slate-800/60 p-1 rounded-xl border border-slate-700/50">
                                        {[
                                            { id: 'same', label: 'Same Strategy', icon: Copy },
                                            { id: 'different', label: 'Different Per Asset', icon: Layers },
                                            { id: 'portfolio', label: 'Portfolio Strategy', icon: PieChart }
                                        ].map((mode) => {
                                            const Icon = mode.icon;
                                            return (
                                                <button
                                                    key={mode.id}
                                                    onClick={() => setConfig({ ...config, strategyMode: mode.id })}
                                                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold uppercase transition-all ${config.strategyMode === mode.id
                                                        ? 'bg-gradient-to-r from-violet-600 to-purple-600 text-white shadow-lg'
                                                        : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800'
                                                        }`}
                                                >
                                                    <Icon size={12} />
                                                    <span>{mode.label}</span>
                                                </button>
                                            );
                                        })}
                                    </div>
                                </div>

                                {/* Different Strategy Mode - Enhanced Per Asset Table */}
                                {config.strategyMode === 'different' && (
                                    <div className="bg-slate-800/40 rounded-xl border border-slate-700/50 overflow-hidden">
                                        <div className="px-4 py-3 bg-slate-800/60 border-b border-slate-700/50">
                                            <div className="flex items-center justify-between">
                                                <span className="text-xs font-bold text-slate-400">Custom Strategy Assignment</span>
                                                <span className="text-[10px] text-slate-600">Select unique strategy for each asset</span>
                                            </div>
                                        </div>
                                        <div className="divide-y divide-slate-700/50">
                                            {config.symbols.map((symbol: string, index: number) => (
                                                <div key={symbol} className="p-4 hover:bg-slate-800/20 transition-colors">
                                                    <div className="flex items-center justify-between">
                                                        <div className="flex items-center gap-3">
                                                            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-slate-700 to-slate-800 flex items-center justify-center">
                                                                <span className="font-mono font-bold text-violet-400">{symbol.slice(0, 2)}</span>
                                                            </div>
                                                            <div>
                                                                <p className="font-bold text-slate-200">{symbol}</p>
                                                                <p className="text-xs text-slate-500">Asset #{index + 1}</p>
                                                            </div>
                                                        </div>
                                                        <select
                                                            className="bg-slate-800/50 border border-slate-700/50 rounded-xl px-4 py-2.5 text-sm text-slate-300 focus:border-violet-500 outline-none min-w-[180px]"
                                                            defaultValue={strategies[0]?.id}
                                                        >
                                                            {strategies.map((st: any) => (
                                                                <option key={st.id} value={st.id}>{st.name}</option>
                                                            ))}
                                                        </select>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {config.strategyMode === 'same' && (
                                    <div className="p-4 bg-gradient-to-br from-violet-500/5 to-purple-500/5 border border-violet-500/20 rounded-xl">
                                        <div className="flex items-center gap-3">
                                            <Copy size={16} className="text-violet-400" />
                                            <div>
                                                <p className="text-xs font-bold text-violet-400">Same Strategy Mode</p>
                                                <p className="text-xs text-slate-400 mt-1">
                                                    The selected strategy will be applied to all assets in your portfolio.
                                                    This creates consistency across your multi-asset portfolio.
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Advanced Options */}
                            <div className="pt-6 border-t border-slate-700/50">
                                <button
                                    onClick={() => setShowAdvanced(!showAdvanced)}
                                    className="flex items-center gap-2 text-xs font-bold text-slate-400 hover:text-violet-400 transition-colors mb-4"
                                >
                                    <ChevronDown
                                        size={14}
                                        className={`transition-transform ${showAdvanced ? 'rotate-180' : ''}`}
                                    />
                                    <span>Advanced Configuration</span>
                                </button>

                                {showAdvanced && (
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 animate-in fade-in">
                                        <div className="space-y-3">
                                            <label className="text-xs font-bold text-slate-400 tracking-wide flex items-center gap-2">
                                                <DollarSign size={14} className="text-emerald-400" />
                                                Initial Capital
                                            </label>
                                            <div className="relative">
                                                <span className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 font-bold">$</span>
                                                <input
                                                    type="number"
                                                    value={config.initialCapital}
                                                    onChange={(e) => setConfig({ ...config, initialCapital: parseInt(e.target.value) })}
                                                    className="w-full pl-8 pr-4 py-3.5 bg-slate-800/50 border border-slate-700/50 rounded-xl focus:border-violet-500 focus:ring-2 focus:ring-violet-500/20 outline-none text-slate-200 font-mono"
                                                    min="1000"
                                                    step="1000"
                                                />
                                            </div>
                                        </div>
                                        <div className="space-y-3">
                                            <label className="text-xs font-bold text-slate-400 tracking-wide flex items-center gap-2">
                                                <Target size={14} className="text-amber-400" />
                                                Max Position %
                                            </label>
                                            <input
                                                type="number"
                                                value={config.maxPositionPct}
                                                onChange={(e) => setConfig({ ...config, maxPositionPct: parseInt(e.target.value) })}
                                                className="w-full px-4 py-3.5 bg-slate-800/50 border border-slate-700/50 rounded-xl focus:border-violet-500 focus:ring-2 focus:ring-violet-500/20 outline-none text-slate-200"
                                                min="1"
                                                max="100"
                                            />
                                        </div>
                                        <div className="space-y-3">
                                            <label className="text-xs font-bold text-slate-400 tracking-wide flex items-center gap-2">
                                                <Shield size={14} className="text-blue-400" />
                                                Risk Level
                                            </label>
                                            <select
                                                value={config.riskLevel || 'medium'}
                                                onChange={(e) => setConfig({ ...config, riskLevel: e.target.value })}
                                                className="w-full px-4 py-3.5 bg-slate-800/50 border border-slate-700/50 rounded-xl focus:border-violet-500 focus:ring-2 focus:ring-violet-500/20 outline-none text-sm text-slate-200"
                                            >
                                                <option value="low">Conservative</option>
                                                <option value="medium">Moderate</option>
                                                <option value="high">Aggressive</option>
                                            </select>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>

                {/* RIGHT: Enhanced Strategy Intelligence & Parameters */}
                <div className="col-span-12 lg:col-span-4">
                    <div className="sticky top-6 space-y-6">
                        {/* Strategy Intelligence Card */}
                        {selectedStrategy ? (
                            <div className="bg-gradient-to-br from-violet-900/40 via-purple-900/40 to-slate-900/90 backdrop-blur-xl border border-violet-500/30 rounded-2xl p-6 shadow-2xl relative overflow-hidden">
                                <div className="absolute inset-0 bg-gradient-to-br from-violet-500/5 via-transparent to-purple-500/5" />
                                <div className="relative">
                                    <div className="flex items-center justify-between mb-4">
                                        <div className="flex items-center gap-2">
                                            <div className="p-1.5 bg-gradient-to-br from-violet-500/30 to-purple-500/30 rounded border border-violet-500/50">
                                                <Zap size={16} className="text-violet-300" strokeWidth={2} />
                                            </div>
                                            <h4 className="text-xs font-bold text-violet-400 uppercase tracking-wider">Strategy Intelligence</h4>
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                                            <span className="text-[10px] text-emerald-400 font-bold">LIVE</span>
                                        </div>
                                    </div>

                                    <h2 className="text-xl font-bold text-slate-100 mb-3">{selectedStrategy.name}</h2>
                                    <p className="text-sm text-slate-400 leading-relaxed mb-6">
                                        {selectedStrategy.description}
                                    </p>

                                    <div className="grid grid-cols-2 gap-3 mb-6">
                                        <div className="bg-slate-800/40 p-3 rounded-xl border border-slate-700/50">
                                            <p className="text-[10px] text-slate-500 font-bold uppercase mb-1">Horizon</p>
                                            <p className="text-sm font-bold text-slate-200">{selectedStrategy.time_horizon}</p>
                                        </div>
                                        <div className="bg-slate-800/40 p-3 rounded-xl border border-slate-700/50">
                                            <p className="text-[10px] text-slate-500 font-bold uppercase mb-1">Complexity</p>
                                            <p className={`text-sm font-bold ${selectedStrategy.complexity === 'Advanced'
                                                ? 'text-red-400'
                                                : selectedStrategy.complexity === 'Intermediate'
                                                    ? 'text-amber-400'
                                                    : 'text-emerald-400'
                                                }`}>
                                                {selectedStrategy.complexity}
                                            </p>
                                        </div>
                                    </div>

                                    {selectedStrategy.best_for && selectedStrategy.best_for.length > 0 && (
                                        <div className="mb-6">
                                            <p className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
                                                <Target size={14} className="text-violet-400" />
                                                Optimal Market Conditions
                                            </p>
                                            <div className="flex flex-wrap gap-2">
                                                {selectedStrategy.best_for.map((tag: string) => (
                                                    <span
                                                        key={tag}
                                                        className="text-[10px] font-bold bg-gradient-to-r from-violet-500/10 to-purple-500/10 text-violet-300 px-3 py-1.5 rounded-lg border border-violet-500/20"
                                                    >
                                                        {tag}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Performance Metrics */}
                                    <div className="pt-6 border-t border-violet-500/20">
                                        <p className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Expected Performance</p>
                                        <div className="grid grid-cols-3 gap-2">
                                            <div className="text-center">
                                                <p className="text-lg font-bold text-emerald-400">+{selectedStrategy.monthly_return || 4.2}%</p>
                                                <p className="text-[9px] text-slate-600 font-bold">Monthly</p>
                                            </div>
                                            <div className="text-center">
                                                <p className="text-lg font-bold text-red-400">{selectedStrategy.drawdown || 8.5}%</p>
                                                <p className="text-[9px] text-slate-600 font-bold">Drawdown</p>
                                            </div>
                                            <div className="text-center">
                                                <p className="text-lg font-bold text-blue-400">{selectedStrategy.sharpe_ratio?.toFixed(2) || '1.45'}</p>
                                                <p className="text-[9px] text-slate-600 font-bold">Sharpe</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="h-[400px] flex flex-col items-center justify-center p-12 border-2 border-dashed border-slate-700/50 rounded-2xl bg-gradient-to-br from-slate-900/30 to-slate-800/30 backdrop-blur-sm">
                                <div className="w-20 h-20 rounded-full bg-gradient-to-br from-slate-800 to-slate-900 border-2 border-dashed border-slate-700/50 flex items-center justify-center mb-6">
                                    <Info size={32} className="text-slate-700" strokeWidth={1.5} />
                                </div>
                                <p className="text-sm font-bold text-slate-600 text-center">Select a Strategy</p>
                                <p className="text-xs text-slate-700 text-center mt-2 max-w-[200px]">
                                    Choose a strategy from the library to view detailed intelligence and configure parameters
                                </p>
                            </div>
                        )}

                        {/* Enhanced Parameter Editor Card */}
                        {selectedStrategy && selectedStrategy.params && Object.keys(selectedStrategy.params).length > 0 && (
                            <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                                <div className="flex items-center justify-between mb-6">
                                    <div className="flex items-center gap-3">
                                        <div className="p-1.5 bg-purple-500/20 rounded border border-purple-500/30">
                                            <Settings size={16} className="text-purple-400" strokeWidth={2} />
                                        </div>
                                        <div>
                                            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Strategy Parameters</h4>
                                            <p className="text-[10px] text-slate-600 font-medium mt-0.5">
                                                {Object.keys(selectedStrategy.params).length} parameters
                                            </p>
                                        </div>
                                    </div>
                                    <div className="flex gap-2">
                                        <button
                                            onClick={() => setShowParameters(!showParameters)}
                                            className="p-2 hover:bg-slate-800 rounded-lg transition-colors text-slate-500 hover:text-violet-400"
                                        >
                                            {showParameters ? <Eye size={16} /> : <EyeOff size={16} />}
                                        </button>
                                        <button
                                            onClick={() => setConfig({ ...config, params: selectedStrategy.params })}
                                            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold text-slate-500 hover:text-violet-400 transition-colors bg-slate-800/50 rounded-lg border border-slate-700/50"
                                        >
                                            <RefreshCw size={12} strokeWidth={2.5} />
                                            <span>Reset</span>
                                        </button>
                                    </div>
                                </div>

                                {showParameters && (
                                    <div className="animate-in fade-in">
                                        <StrategyParameterForm
                                            params={selectedStrategy.params}
                                            values={config.params || {}}
                                            onChange={handleParamChange}
                                        />

                                        <div className="mt-8 pt-6 border-t border-slate-700/50">
                                            <div className="flex items-center justify-between">
                                                <div className="flex items-center gap-2">
                                                    <div className="w-2 h-2 rounded-full bg-emerald-500" />
                                                    <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Validated</span>
                                                </div>
                                                <div className="flex items-center gap-1 text-xs text-slate-600">
                                                    <Lock size={12} />
                                                    <span>Safe to run</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Quick Actions Panel */}
                        <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">Quick Actions</h4>
                            <div className="space-y-3">
                                <button className="w-full flex items-center justify-between p-3 bg-slate-800/40 hover:bg-slate-800/60 border border-slate-700/50 rounded-xl transition-all group">
                                    <div className="flex items-center gap-3">
                                        <div className="p-2 bg-emerald-500/20 rounded-lg border border-emerald-500/30">
                                            <Save size={16} className="text-emerald-400" />
                                        </div>
                                        <div className="text-left">
                                            <p className="text-sm font-bold text-slate-200">Save Portfolio</p>
                                            <p className="text-xs text-slate-500">Store current configuration</p>
                                        </div>
                                    </div>
                                    <ChevronRight size={16} className="text-slate-500 group-hover:text-violet-400" />
                                </button>
                                <button className="w-full flex items-center justify-between p-3 bg-slate-800/40 hover:bg-slate-800/60 border border-slate-700/50 rounded-xl transition-all group">
                                    <div className="flex items-center gap-3">
                                        <div className="p-2 bg-blue-500/20 rounded-lg border border-blue-500/30">
                                            <Download size={16} className="text-blue-400" />
                                        </div>
                                        <div className="text-left">
                                            <p className="text-sm font-bold text-slate-200">Export Config</p>
                                            <p className="text-xs text-slate-500">Download JSON file</p>
                                        </div>
                                    </div>
                                    <ChevronRight size={16} className="text-slate-500 group-hover:text-violet-400" />
                                </button>
                                <button className="w-full flex items-center justify-between p-3 bg-slate-800/40 hover:bg-slate-800/60 border border-slate-700/50 rounded-xl transition-all group">
                                    <div className="flex items-center gap-3">
                                        <div className="p-2 bg-amber-500/20 rounded-lg border border-amber-500/30">
                                            <AlertCircle size={16} className="text-amber-400" />
                                        </div>
                                        <div className="text-left">
                                            <p className="text-sm font-bold text-slate-200">Risk Analysis</p>
                                            <p className="text-xs text-slate-500">Run risk assessment</p>
                                        </div>
                                    </div>
                                    <ChevronRight size={16} className="text-slate-500 group-hover:text-violet-400" />
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Results Section - Only show after backtest has been run */}
            {hasRunBacktest && results && results.type === 'multi' && (
                <div className="pt-6 border-t border-slate-800/80 animate-in fade-in">
                    <MultiBacktestResults results={results} />
                </div>
            )}
        </div>
    );
};

// Star component for ratings
const Star = ({ size, className }: { size: number; className: string }) => (
    <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="currentColor"
        className={className}
    >
        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
    </svg>
);

// Custom scrollbar styles
// const style = document.createElement('style');
// style.textContent = `
//     .custom-scrollbar::-webkit-scrollbar {
//         width: 8px;
//         height: 8px;
//     }
//     .custom-scrollbar::-webkit-scrollbar-track {
//         background: rgba(30, 41, 59, 0.3);
//         border-radius: 4px;
//     }
//     .custom-scrollbar::-webkit-scrollbar-thumb {
//         background: linear-gradient(to bottom, #8b5cf6, #7c3aed);
//         border-radius: 4px;
//     }
//     .custom-scrollbar::-webkit-scrollbar-thumb:hover {
//         background: linear-gradient(to bottom, #7c3aed, #6d28d9);
//     }
// `;
// if (typeof document !== 'undefined') {
//     document.head.appendChild(style);
// }

export default MultiAssetBacktest;