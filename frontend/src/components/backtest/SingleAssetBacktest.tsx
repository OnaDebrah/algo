'use client'
import React, {useMemo, useState} from 'react';
import {
    Activity,
    AlertCircle,
    BarChart3,
    Calendar,
    Check,
    ChevronDown,
    Clock,
    Copy,
    DollarSign,
    Download,
    Eye,
    EyeOff,
    Filter,
    Grid,
    Info,
    List,
    Play,
    RefreshCw,
    Save,
    Search,
    Settings,
    Star,
    Target,
    TrendingUp,
    X,
    Zap
} from 'lucide-react';
import SingleBacktestResults from "@/components/backtest/SingleBacktestResults";
import StrategyParameterForm from "@/components/backtest/StrategyParameterForm";
import {BacktestResult, ScalarParam, SingleAssetConfig, Strategy} from "@/types/all_types";
import {formatCurrency} from "@/utils/formatters";

interface SingleAssetBacktestProps {
    config: SingleAssetConfig;
    setConfig: (config: SingleAssetConfig) => void;
    strategies: Strategy[];
    runBacktest: () => Promise<void>;
    isRunning: boolean;
    results: BacktestResult | null;
}

const SingleAssetBacktest: React.FC<SingleAssetBacktestProps> = ({
    config,
    setConfig,
    strategies,
    runBacktest,
    isRunning,
    results
}: SingleAssetBacktestProps) => {
    const [selectedCategory, setSelectedCategory] = useState('All');
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [showParameters, setShowParameters] = useState(false);
    const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
    const [searchQuery, setSearchQuery] = useState('');

    // Asset suggestions
    const assetSuggestions = [
        { symbol: 'AAPL', name: 'Apple Inc.', sector: 'Technology', color: 'from-gray-500 to-slate-500' },
        { symbol: 'MSFT', name: 'Microsoft', sector: 'Technology', color: 'from-blue-500 to-cyan-500' },
        { symbol: 'GOOGL', name: 'Alphabet', sector: 'Technology', color: 'from-red-500 to-orange-500' },
        { symbol: 'AMZN', name: 'Amazon', sector: 'Consumer', color: 'from-amber-500 to-yellow-500' },
        { symbol: 'TSLA', name: 'Tesla', sector: 'Automotive', color: 'from-emerald-500 to-green-500' },
        { symbol: 'NVDA', name: 'NVIDIA', sector: 'Semiconductors', color: 'from-green-500 to-emerald-500' },
        { symbol: 'BTC-USD', name: 'Bitcoin', sector: 'Crypto', color: 'from-orange-500 to-amber-500' },
        { symbol: 'SPY', name: 'S&P 500 ETF', sector: 'ETF', color: 'from-indigo-500 to-violet-500' },
    ];

    // Get unique categories
    const categories = useMemo(() =>
        ['All', ...Array.from(new Set(strategies.map((s: Strategy) => s.category)))],
        [strategies]
    );

    // Find selected strategy
    const selectedStrategy = useMemo(() =>
        strategies.find((s) => s.id === config.strategy),
        [config.strategy, strategies]
    );

    // Filter strategies by category and search
    const filteredStrategies = useMemo(() => {
        let filtered = selectedCategory === 'All'
            ? strategies
            : strategies.filter((s: Strategy) => s.category === selectedCategory);

        if (searchQuery) {
            const query = searchQuery.toLowerCase();
            filtered = filtered.filter(s =>
                s.name.toLowerCase().includes(query) ||
                s.description.toLowerCase().includes(query) ||
                s.category.toLowerCase().includes(query)
            );
        }

        return filtered;
    }, [strategies, selectedCategory, searchQuery]);

    // Update parameters
    const handleParamChange = (key: string, val: any) => {
        setConfig({
            ...config,
            params: { ...(config.params || {}), [key]: val }
        });
    };

    // Quick suggestions
    const quickSuggestions = [
        { label: '1 Month', period: '1mo', interval: '1d', capital: 10000 },
        { label: '3 Months', period: '3mo', interval: '1d', capital: 25000 },
        { label: '1 Year', period: '1y', interval: '1wk', capital: 50000 },
        { label: 'Full History', period: '5y', interval: '1wk', capital: 100000 },
    ];

    return (
        <div className="space-y-6">
            {/* Header with Stats */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div className="flex items-center gap-4">
                    <div className="p-3 bg-gradient-to-br from-violet-500/20 to-purple-500/20 rounded-2xl border border-violet-500/30 shadow-xl shadow-violet-500/10">
                        <BarChart3 className="text-violet-400" size={28} strokeWidth={2} />
                    </div>
                    <div>
                        <h2 className="text-2xl font-bold text-slate-100 tracking-tight">
                            Single Asset <span className="text-slate-400 font-normal">Backtest</span>
                        </h2>
                        <p className="text-sm text-slate-500 font-medium mt-1">Test strategies on individual assets with precision</p>
                    </div>
                </div>

                <div className="flex gap-3">
                    <button className="flex items-center gap-2 px-5 py-2.5 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50 text-slate-300 rounded-xl text-sm font-bold transition-all">
                        <Save size={16} />
                        <span>Save Setup</span>
                    </button>
                    <button className="flex items-center gap-2 px-5 py-2.5 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50 text-slate-300 rounded-xl text-sm font-bold transition-all">
                        <Copy size={16} />
                        <span>Duplicate</span>
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-12 gap-6">
                {/* LEFT: Configuration Panel */}
                <div className="col-span-12 lg:col-span-8 space-y-6">
                    {/* Asset & Basic Configuration Card */}
                    <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                        <div className="flex items-center gap-3 mb-6">
                            <div className="p-2 bg-violet-500/20 rounded-lg border border-violet-500/30">
                                <Target className="text-violet-400" size={20} strokeWidth={2} />
                            </div>
                            <h3 className="text-sm font-bold text-slate-300">Asset & Basic Configuration</h3>
                        </div>

                        {/* Enhanced Symbol Input with Suggestions */}
                        <div className="space-y-4 mb-6">
                            <label className="text-xs font-bold text-slate-400 tracking-wide flex items-center gap-2">
                                <Zap size={14} className="text-violet-400" />
                                Asset Selection
                            </label>
                            <div className="flex gap-3">
                                <div className="relative flex-1 group">
                                    <Search
                                        className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-600 group-focus-within:text-violet-500 transition-colors"
                                        size={18}
                                    />
                                    <input
                                        type="text"
                                        value={config.symbol}
                                        onChange={(e) => setConfig({ ...config, symbol: e.target.value.toUpperCase() })}
                                        className="w-full pl-12 pr-4 py-3.5 bg-slate-800/50 border border-slate-700/50 rounded-xl focus:border-violet-500 focus:ring-2 focus:ring-violet-500/20 outline-none transition-all text-slate-200 font-mono"
                                        placeholder="Enter ticker (e.g. AAPL)"
                                    />
                                    {config.symbol && (
                                        <button
                                            onClick={() => setConfig({ ...config, symbol: '' })}
                                            className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-600 hover:text-red-400 transition-colors"
                                        >
                                            <X size={18} />
                                        </button>
                                    )}
                                </div>
                                <div className="text-center">
                                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500/20 to-purple-500/20 border border-violet-500/30 flex items-center justify-center">
                                        <span className="font-mono font-bold text-violet-300">
                                            {config.symbol.slice(0, 2) || '??'}
                                        </span>
                                    </div>
                                    <p className="text-xs text-slate-600 mt-1 font-medium">Symbol</p>
                                </div>
                            </div>

                            {/* Asset Suggestions */}
                            <div className="flex flex-wrap gap-2">
                                {assetSuggestions.map((asset) => (
                                    <button
                                        key={asset.symbol}
                                        onClick={() => setConfig({ ...config, symbol: asset.symbol })}
                                        className={`group relative overflow-hidden px-3 py-2 bg-gradient-to-br from-slate-800/50 to-slate-900/50 border rounded-lg transition-all ${config.symbol === asset.symbol
                                                ? 'border-violet-500 bg-violet-500/10'
                                                : 'border-slate-700/50 hover:border-violet-500/50'
                                            }`}
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
                                            {config.symbol === asset.symbol && (
                                                <Check size={12} className="text-violet-400 ml-2" />
                                            )}
                                        </div>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Quick Setup Suggestions */}
                        <div className="mb-6">
                            <label className="text-xs font-bold text-slate-400 tracking-wide flex items-center gap-2 mb-3">
                                <Clock size={14} className="text-amber-400" />
                                Quick Setup
                            </label>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                {quickSuggestions.map((suggestion) => (
                                    <button
                                        key={suggestion.label}
                                        onClick={() => setConfig({
                                            ...config,
                                            period: suggestion.period,
                                            interval: suggestion.interval,
                                            initialCapital: suggestion.capital
                                        })}
                                        className="group p-3 bg-gradient-to-br from-slate-800/50 to-slate-900/50 border border-slate-700/50 hover:border-violet-500/50 rounded-xl transition-all text-left"
                                    >
                                        <p className="text-xs font-bold text-slate-300 group-hover:text-violet-300 transition-colors">
                                            {suggestion.label}
                                        </p>
                                        <p className="text-[10px] text-slate-500 mt-1">
                                            {suggestion.period} • {suggestion.interval}
                                        </p>
                                        <p className="text-[10px] text-slate-600 font-mono mt-1">
                                            {formatCurrency(suggestion.capital)}
                                        </p>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Time & Data Configuration */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
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
                                    <option value="max">Max History</option>
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
                                    <option value="1m">1 Minute</option>
                                    <option value="5m">5 Minutes</option>
                                    <option value="15m">15 Minutes</option>
                                    <option value="1h">1 Hour</option>
                                    <option value="1d">1 Day</option>
                                    <option value="1wk">1 Week</option>
                                </select>
                            </div>
                            <div className="space-y-3">
                                <label className="text-xs font-bold text-slate-400 tracking-wide flex items-center gap-2">
                                    <DollarSign size={14} className="text-emerald-400" />
                                    Capital
                                </label>
                                <div className="relative">
                                    <span className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 font-bold">$</span>
                                    <input
                                        type="number"
                                        value={config.initialCapital}
                                        onChange={(e) => setConfig({ ...config, initialCapital: parseInt(e.target.value) })}
                                        className="w-full pl-8 pr-4 py-3.5 bg-slate-800/50 border border-slate-700/50 rounded-xl focus:border-violet-500 focus:ring-2 focus:ring-violet-500/20 outline-none text-slate-200 font-mono"
                                        min="100"
                                        step="100"
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Advanced Options Toggle */}
                        <button
                            onClick={() => setShowAdvanced(!showAdvanced)}
                            className="flex items-center gap-2 text-xs font-bold text-slate-400 hover:text-violet-400 transition-colors mb-4"
                        >
                            <ChevronDown
                                size={14}
                                className={`transition-transform ${showAdvanced ? 'rotate-180' : ''}`}
                            />
                            <span>Advanced Options</span>
                        </button>

                        {showAdvanced && (
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4 border-t border-slate-700/50 animate-in fade-in">
                                <div className="space-y-3">
                                    <label className="text-xs font-bold text-slate-400 tracking-wide">Max Position %</label>
                                    <input
                                        type="number"
                                        value={config.maxPositionPct || 100}
                                        onChange={(e) => setConfig({ ...config, maxPositionPct: parseInt(e.target.value) })}
                                        className="w-full px-4 py-3.5 bg-slate-800/50 border border-slate-700/50 rounded-xl focus:border-violet-500 outline-none text-slate-200"
                                        min="1"
                                        max="100"
                                    />
                                </div>
                                <div className="space-y-3">
                                    <label className="text-xs font-bold text-slate-400 tracking-wide">Risk Level</label>
                                    <select
                                        value={config.riskLevel || 'medium'}
                                        onChange={(e) => setConfig({ ...config, riskLevel: e.target.value })}
                                        className="w-full px-4 py-3.5 bg-slate-800/50 border border-slate-700/50 rounded-xl focus:border-violet-500 outline-none text-sm text-slate-200"
                                    >
                                        <option value="low">Conservative</option>
                                        <option value="medium">Moderate</option>
                                        <option value="high">Aggressive</option>
                                    </select>
                                </div>
                                <div className="space-y-3">
                                    <label className="text-xs font-bold text-slate-400 tracking-wide">Commission</label>
                                    <div className="relative">
                                        <span className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 font-bold">$</span>
                                        <input
                                            type="number"
                                            value={config.commission || 0}
                                            onChange={(e) => setConfig({ ...config, commission: parseFloat(e.target.value) })}
                                            className="w-full pl-8 pr-4 py-3.5 bg-slate-800/50 border border-slate-700/50 rounded-xl focus:border-violet-500 outline-none text-slate-200"
                                            min="0"
                                            step="0.01"
                                        />
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Strategy Selection Card */}
                    <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
                            <div className="flex items-center gap-3">
                                <div className="p-2 bg-purple-500/20 rounded-lg border border-purple-500/30">
                                    <TrendingUp className="text-purple-400" size={20} strokeWidth={2} />
                                </div>
                                <div>
                                    <h4 className="text-sm font-bold text-slate-300">Strategy Selection</h4>
                                    <p className="text-xs text-slate-500 font-medium mt-0.5">
                                        {filteredStrategies.length} of {strategies.length} strategies
                                    </p>
                                </div>
                            </div>

                            <div className="flex gap-3">
                                {/* Search Bar */}
                                <div className="relative group">
                                    <Search
                                        className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-600 group-focus-within:text-violet-500 transition-colors"
                                        size={16}
                                    />
                                    <input
                                        type="text"
                                        placeholder="Search strategies..."
                                        value={searchQuery}
                                        onChange={(e) => setSearchQuery(e.target.value)}
                                        className="pl-10 pr-4 py-2.5 bg-slate-800/60 border border-slate-700/50 rounded-xl focus:border-violet-500 outline-none text-sm text-slate-200 w-48"
                                    />
                                </div>

                                {/* Category Filter */}
                                <div className="flex items-center space-x-2">
                                    <Filter size={14} className="text-slate-600" />
                                    <select
                                        value={selectedCategory}
                                        onChange={(e) => setSelectedCategory(e.target.value)}
                                        className="px-4 py-2.5 bg-slate-800/60 border border-slate-700/50 rounded-xl text-xs font-bold text-slate-300 focus:border-violet-500 outline-none cursor-pointer"
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
                                        className={`px-3 py-2.5 transition-all ${viewMode === 'grid' ? 'bg-slate-800 text-slate-200' : 'text-slate-500 hover:text-slate-300'
                                            }`}
                                    >
                                        <Grid size={16} />
                                    </button>
                                    <button
                                        onClick={() => setViewMode('list')}
                                        className={`px-3 py-2.5 transition-all ${viewMode === 'list' ? 'bg-slate-800 text-slate-200' : 'text-slate-500 hover:text-slate-300'
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
                            {filteredStrategies.map((strategy: Strategy) => {
                                const isSelected = config.strategy === strategy.id;
                                return (
                                    <button
                                        key={strategy.id}
                                        onClick={() => setConfig({ ...config, strategy: strategy.id })}
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
                                                    {strategy.category}
                                                </span>
                                                {isSelected && (
                                                    <div className="w-6 h-6 rounded-full bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center">
                                                        <Check size={12} className="text-white" strokeWidth={3} />
                                                    </div>
                                                )}
                                            </div>
                                            <p className="text-sm font-bold text-slate-200 mb-2 group-hover:text-violet-300 transition-colors">
                                                {strategy.name}
                                            </p>
                                            <p className="text-xs text-slate-400 line-clamp-2 mb-3">{strategy.description}</p>
                                            <div className="flex items-center justify-between">
                                                <span className={`text-[10px] px-2.5 py-1 rounded-lg font-medium ${strategy.complexity === 'Advanced'
                                                        ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                                                        : strategy.complexity === 'Intermediate'
                                                            ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                                                            : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                                                    }`}>
                                                    {strategy.complexity}
                                                </span>
                                                <div className="flex items-center gap-1 text-amber-400">
                                                    <Star size={10} fill="currentColor" />
                                                    <span className="text-xs font-bold">{strategy.rating || 4.5}</span>
                                                </div>
                                            </div>
                                        </div>
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    {/* Parameters Card */}
                    {selectedStrategy && selectedStrategy.parameters && Object.keys(selectedStrategy.parameters).length > 0 && (
                        <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                            <div className="flex items-center justify-between mb-6">
                                <div className="flex items-center gap-3">
                                    <div className="p-2 bg-purple-500/20 rounded-lg border border-purple-500/30">
                                        <Settings className="text-purple-400" size={20} strokeWidth={2} />
                                    </div>
                                    <div>
                                        <h4 className="text-sm font-bold text-slate-300">Strategy Parameters</h4>
                                        <p className="text-xs text-slate-500 font-medium mt-0.5">
                                            Fine-tune {Object.keys(selectedStrategy.parameters).length} parameters
                                        </p>
                                    </div>
                                </div>
                                <div className="flex gap-2">
                                    <button
                                        onClick={() => setShowParameters(!showParameters)}
                                        className="flex items-center gap-2 px-3 py-2 text-xs font-bold text-slate-500 hover:text-violet-400 transition-colors bg-slate-800/50 rounded-lg border border-slate-700/50"
                                    >
                                        {showParameters ? <Eye size={14} /> : <EyeOff size={14} />}
                                        <span>{showParameters ? 'Hide' : 'Show'}</span>
                                    </button>
                                    <button
                                        onClick={() => setConfig({ ...config, params: selectedStrategy?.parameters })}
                                        className="flex items-center gap-2 px-3 py-2 text-xs font-bold text-slate-500 hover:text-violet-400 transition-colors bg-slate-800/50 rounded-lg border border-slate-700/50"
                                    >
                                        <RefreshCw size={14} />
                                        <span>Reset</span>
                                    </button>
                                </div>
                            </div>

                            {showParameters && (
                                <div className="animate-in fade-in">
                                    <StrategyParameterForm
                                        params={selectedStrategy.parameters}
                                        values={config.params || {}}
                                        onChange={handleParamChange}
                                    />
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* RIGHT: Strategy Intelligence & Action Panel */}
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
                                            <span className="text-[10px] text-emerald-400 font-bold">READY</span>
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
                                    Choose a strategy from the library to view detailed intelligence
                                </p>
                            </div>
                        )}

                        {/* Action Panel */}
                        <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">Backtest Actions</h4>

                            <button
                                onClick={runBacktest}
                                disabled={isRunning || !config.symbol}
                                className="group relative overflow-hidden w-full flex items-center justify-center gap-3 px-6 py-4 mb-4 bg-gradient-to-r from-violet-600 via-purple-600 to-fuchsia-600 hover:from-violet-500 hover:via-purple-500 hover:to-fuchsia-500 disabled:from-slate-700 disabled:via-slate-700 disabled:to-slate-700 disabled:cursor-not-allowed rounded-xl font-bold transition-all shadow-xl shadow-violet-500/30 disabled:shadow-none text-white"
                            >
                                <div className="absolute inset-0 bg-gradient-to-r from-white/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                                {isRunning ? (
                                    <>
                                        <RefreshCw size={20} className="animate-spin relative z-10" strokeWidth={2.5} />
                                        <span className="relative z-10">Running Analysis...</span>
                                    </>
                                ) : (
                                    <>
                                        <Play size={20} strokeWidth={2.5} className="relative z-10" />
                                        <span className="relative z-10">Execute Backtest</span>
                                    </>
                                )}
                            </button>

                            <div className="space-y-3">
                                <button className="w-full flex items-center justify-between p-3 bg-slate-800/40 hover:bg-slate-800/60 border border-slate-700/50 rounded-xl transition-all group">
                                    <div className="flex items-center gap-3">
                                        <div className="p-2 bg-emerald-500/20 rounded-lg border border-emerald-500/30">
                                            <Download size={16} className="text-emerald-400" />
                                        </div>
                                        <div className="text-left">
                                            <p className="text-sm font-bold text-slate-200">Export Results</p>
                                            <p className="text-xs text-slate-500">Download CSV/PDF</p>
                                        </div>
                                    </div>
                                    <ChevronDown size={16} className="text-slate-500 group-hover:text-violet-400" />
                                </button>
                                <button className="w-full flex items-center justify-between p-3 bg-slate-800/40 hover:bg-slate-800/60 border border-slate-700/50 rounded-xl transition-all group">
                                    <div className="flex items-center gap-3">
                                        <div className="p-2 bg-blue-500/20 rounded-lg border border-blue-500/30">
                                            <AlertCircle size={16} className="text-blue-400" />
                                        </div>
                                        <div className="text-left">
                                            <p className="text-sm font-bold text-slate-200">Risk Analysis</p>
                                            <p className="text-xs text-slate-500">Detailed risk metrics</p>
                                        </div>
                                    </div>
                                    <ChevronDown size={16} className="text-slate-500 group-hover:text-violet-400" />
                                </button>
                            </div>

                            {/* Status Indicators */}
                            <div className="mt-6 pt-6 border-t border-slate-700/50 space-y-2">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                        <div className={`w-2 h-2 rounded-full ${config.symbol ? 'bg-emerald-500' : 'bg-red-500'
                                            }`} />
                                        <span className="text-xs text-slate-500">Asset Selected</span>
                                    </div>
                                    <span className="text-xs font-bold text-slate-300">{config.symbol || 'None'}</span>
                                </div>
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                        <div className={`w-2 h-2 rounded-full ${config.strategy ? 'bg-emerald-500' : 'bg-red-500'
                                            }`} />
                                        <span className="text-xs text-slate-500">Strategy Ready</span>
                                    </div>
                                    <span className="text-xs font-bold text-slate-300">
                                        {config.strategy ? 'Selected' : 'Pending'}
                                    </span>
                                </div>
                            </div>
                        </div>

                        {/* Quick Stats */}
                        <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">Quick Stats</h4>
                            <div className="space-y-3">
                                <div className="flex items-center justify-between">
                                    <span className="text-xs text-slate-500">Capital at Risk</span>
                                    <span className="text-sm font-bold text-slate-200">
                                        {formatCurrency(config.initialCapital * ((config.maxPositionPct || 100) / 100))}
                                    </span>
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-xs text-slate-500">Expected Return</span>
                                    <span className="text-sm font-bold text-emerald-400">
                                        +{selectedStrategy?.monthly_return || 4.2}%
                                    </span>
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-xs text-slate-500">Risk Level</span>
                                    <span className={`text-sm font-bold ${config.riskLevel === 'high' ? 'text-red-400' :
                                            config.riskLevel === 'medium' ? 'text-amber-400' :
                                                'text-emerald-400'
                                        }`}>
                                        {config.riskLevel || 'medium'}
                                    </span>
                                </div>
                                <div className="flex items-center justify-between">
                                    <span className="text-xs text-slate-500">Data Points</span>
                                    <span className="text-sm font-bold text-blue-400">~1,250</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Results Section */}
            {results && (
                <div className="pt-6 border-t border-slate-800/80 animate-in fade-in">
                    <SingleBacktestResults results={results} />
                </div>
            )}
        </div>
    );
};

export default SingleAssetBacktest;