'use client'

import {
    Activity,
    AlertCircle,
    BarChart3,
    Calendar,
    Check,
    ChevronDown,
    ChevronRight,
    Copy,
    Cpu,
    Database,
    DollarSign,
    Download,
    Eye,
    EyeOff,
    Filter,
    Grid,
    HelpCircle,
    Info,
    Layers,
    List,
    Lock,
    PieChart,
    Play,
    PlusCircle,
    RefreshCw,
    Save,
    Search,
    Settings,
    Shield,
    Sliders,
    Star,
    Target,
    X,
    Zap
} from "lucide-react";
import React, { useMemo, useState } from "react";
import StrategyParameterForm from "@/components/backtest/StrategyParameterForm";
import MultiBacktestResults from "@/components/backtest/MultiBacktestResults";
import RiskAnalysisModal from "@/components/backtest/RiskAnalysisModal";
import { BacktestResult, MultiAssetConfig, Strategy, PortfolioCreate } from "@/types/all_types";
import { portfolio } from "@/utils/api";

interface MultiAssetBacktestProps {
    config: MultiAssetConfig;
    setConfig: (config: MultiAssetConfig) => void;
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
}: MultiAssetBacktestProps) => {
    const [activeCategory, setActiveCategory] = useState('All');
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [hasRunBacktest, setHasRunBacktest] = useState(false);
    const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
    const [showParameters, setShowParameters] = useState(true);
    const [allocationMode, setAllocationMode] = useState<'equal' | 'manual' | 'optimized'>('equal');
    const [searchQuery, setSearchQuery] = useState('');
    const [showRiskAnalysis, setShowRiskAnalysis] = useState(false);
    const [isSaving, setIsSaving] = useState(false);

    // Find the currently active strategy
    const selectedStrategy = useMemo(() =>
        strategies.find((s) => s.id === config.strategy),
        [config.strategy, strategies]
    );

    // Get unique categories
    const categories = useMemo(() =>
        ['All', ...Array.from(new Set(strategies.map((s) => s.category)))],
        [strategies]
    );

    // Filter strategies by category
    const filteredStrategies = useMemo(() => {
        let filtered = activeCategory === 'All'
            ? strategies
            : strategies.filter((s) => s.category === activeCategory);

        if (searchQuery) {
            const query = searchQuery.toLowerCase();
            filtered = filtered.filter(s =>
                s.name.toLowerCase().includes(query) ||
                s.description.toLowerCase().includes(query) ||
                s.category.toLowerCase().includes(query)
            );
        }

        return filtered;
    }, [strategies, activeCategory, searchQuery]);


    const handleParamChange = (key: string, val: any) => {
        setConfig({
            ...config,
            params: { ...(config.params || {}), [key]: val }
        });
    };

    const handleRunBacktest = async () => {
        await runBacktest();
        setHasRunBacktest(true);
    };

    const handleExportResults = () => {
        console.log('Export Results clicked');
        if (!results) {
            console.error('No results to export');
            return;
        }

        try {
            console.log('Generating CSV for results:', results);
            const rows: (string | number)[][] = [];

            // 1. Portfolio Summary
            rows.push(['PORTFOLIO SUMMARY']);
            rows.push(['Metric', 'Value']);
            rows.push(['Total Return', `${((results.total_return || 0) * 100).toFixed(2)}%`]);
            rows.push(['Sharpe Ratio', (results.sharpe_ratio || 0).toFixed(2)]);
            rows.push(['Max Drawdown', `${((results.max_drawdown || 0) * 100).toFixed(2)}%`]);
            rows.push(['Win Rate', `${((results.win_rate || 0)).toFixed(2)}%`]);
            rows.push(['Profit Factor', (results.profit_factor || 0).toFixed(2)]);
            rows.push(['Total Trades', results.total_trades || 0]);
            rows.push(['Final Equity', (results.final_equity || 0).toFixed(2)]);
            rows.push([]);

            // 2. Asset Performance Breakdown
            rows.push(['ASSET PERFORMANCE']);
            rows.push(['Symbol', 'Strategy', 'Total Return', 'Win Rate', 'Trades', 'Avg Profit', 'Loss Rate']);

            if (results.symbol_stats) {
                Object.entries(results.symbol_stats).forEach(([symbol, stats]) => {
                    const totalTrades = stats.total_trades || 0;
                    const losingTrades = stats.losing_trades || 0;
                    const lossRate = totalTrades > 0 ? (losingTrades / totalTrades) * 100 : 0;

                    rows.push([
                        symbol || 'Unknown',
                        'Multi Asset',
                        `${((stats.total_return || 0) * 100).toFixed(2)}%`,
                        `${(stats.win_rate || 0).toFixed(2)}%`,
                        totalTrades,
                        (stats.avg_profit || 0).toFixed(2),
                        lossRate.toFixed(2) + '%'
                    ]);
                });
            } else {
                console.warn('No symbol_stats found in results');
            }
            rows.push([]);

            // 3. Trade Ledger
            rows.push(['TRADE LEDGER']);
            rows.push(['ID', 'Symbol', 'Side', 'Date', 'Qty', 'Price', 'Commission', 'Profit', 'Status']);

            if (results.trades && Array.isArray(results.trades)) {
                results.trades.forEach(t => {
                    rows.push([
                        t.id || '',
                        t.symbol || '',
                        t.order_type || '',
                        t.timestamp || '',
                        t.quantity || 0,
                        t.price || 0,
                        t.commission || 0,
                        typeof t.profit === 'number' ? t.profit.toFixed(2) : '',
                        typeof t.profit === 'number' ? 'Closed' : 'Open'
                    ]);
                });
            }

            // Convert to CSV using Blob
            const csvContent = rows.map(e => e.join(",")).join("\n");
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const url = URL.createObjectURL(blob);

            const link = document.createElement("a");
            link.setAttribute("href", url);
            const filename = `multi_asset_backtest_${new Date().toISOString().slice(0, 10)}.csv`;
            link.setAttribute("download", filename);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            // Clean up
            setTimeout(() => URL.revokeObjectURL(url), 100);

            console.log('Export complete');
        } catch (error) {
            console.error('Error exporting results:', error);
            alert('Failed to export results. See console for details.');
        }
    };

    const handleSavePortfolio = async () => {
        try {
            setIsSaving(true);
            const portfolioName = `Backtest Allocation ${new Date().toLocaleDateString()}`;

            // We store the configuration JSON in the description field
            // This is a workaround to persist the backtest config
            const description = JSON.stringify({
                strategy: config.strategy,
                symbols: config.symbols,
                allocations: config.allocations,
                params: config.params,
                period: config.period,
                interval: config.interval
            });

            const data: PortfolioCreate = {
                name: portfolioName,
                initial_capital: config.initialCapital,
                description: description
            };

            await portfolio.create(data);

            alert('Portfolio Configuration Saved Successfully!');
        } catch (error) {
            console.error('Failed to save portfolio:', error);
            alert('Failed to save portfolio. Please try again.');
        } finally {
            setIsSaving(false);
        }
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
            <div
                className="flex flex-col sm:flex-row justify-between items-start sm:items-end gap-4 pb-6 border-b border-slate-800/80">
                <div className="flex items-start gap-4">
                    <div
                        className="p-3 bg-gradient-to-br from-violet-500/20 to-purple-500/20 rounded-2xl border border-violet-500/30 shadow-xl shadow-violet-500/10">
                        <BarChart3 className="text-violet-400" size={28} strokeWidth={2} />
                    </div>
                    <div>
                        <h3 className="text-2xl font-bold text-slate-100 tracking-tight">
                            Multi-Asset <span className="text-slate-400 font-normal">Portfolio</span>
                        </h3>
                        <p className="text-sm text-slate-500 font-medium mt-1">Configure and backtest diversified
                            trading strategies</p>
                    </div>
                </div>

                <div className="flex flex-col sm:flex-row gap-3">
                    <button
                        onClick={handleRunBacktest}
                        disabled={isRunning || config.symbols.length < 2}
                        className="group relative overflow-hidden flex items-center space-x-3 px-7 py-3.5 bg-gradient-to-r from-violet-600 via-purple-600 to-fuchsia-600 hover:from-violet-500 hover:via-purple-500 hover:to-fuchsia-500 disabled:from-slate-700 disabled:via-slate-700 disabled:to-slate-700 disabled:cursor-not-allowed rounded-xl font-bold transition-all shadow-xl shadow-violet-500/30 disabled:shadow-none text-white"
                    >
                        <div
                            className="absolute inset-0 bg-gradient-to-r from-white/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
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
                    <div
                        className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
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
                                <button
                                    className="p-2 hover:bg-slate-800 rounded-lg transition-colors text-slate-500 hover:text-violet-400">
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
                                        onChange={(e) => setConfig({
                                            ...config,
                                            symbolInput: e.target.value.toUpperCase()
                                        })}
                                        onKeyUp={(e) => e.key === 'Enter' && addSymbol()}
                                        className="w-full pl-12 pr-4 py-3.5 bg-slate-800/50 border border-slate-700/50 rounded-xl focus:border-violet-500 focus:ring-2 focus:ring-violet-500/20 outline-none transition-all placeholder:text-slate-600 font-mono text-sm text-slate-200"
                                    />
                                </div>
                                <button
                                    onClick={addSymbol}
                                    className="group relative overflow-hidden px-6 py-3.5 bg-gradient-to-r from-slate-800/60 to-slate-900/60 border border-slate-700/50 hover:border-slate-600/50 rounded-xl font-semibold text-sm transition-all text-slate-200 flex items-center space-x-2"
                                >
                                    <div
                                        className="absolute inset-0 bg-gradient-to-r from-violet-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
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
                                        <div
                                            className={`absolute inset-0 bg-gradient-to-br ${asset.color} opacity-0 group-hover:opacity-10 transition-opacity`} />
                                        <div className="relative flex items-center gap-2">
                                            <div
                                                className="w-6 h-6 rounded bg-gradient-to-br from-slate-700 to-slate-800 flex items-center justify-center">
                                                <span
                                                    className="text-xs font-bold text-slate-400">{asset.symbol.charAt(0)}</span>
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
                                                <div
                                                    className={`absolute inset-0 bg-gradient-to-br ${suggestion?.color || 'from-violet-500/20 to-purple-500/20'} opacity-5 group-hover:opacity-10`} />
                                                <div className="relative flex items-center space-x-3">
                                                    <div
                                                        className="w-10 h-10 rounded-lg bg-gradient-to-br from-slate-700 to-slate-800 border border-slate-600/50 flex items-center justify-center">
                                                        <span
                                                            className="font-mono font-bold text-lg text-violet-300">{symbol.slice(0, 2)}</span>
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
                                    <div
                                        className="w-16 h-16 rounded-full bg-gradient-to-br from-slate-800 to-slate-900 border-2 border-dashed border-slate-700/50 flex items-center justify-center mb-4">
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
                                                onClick={() => setAllocationMode(mode as never)}
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
                                    <div
                                        className="p-4 bg-gradient-to-br from-violet-500/5 to-purple-500/5 border border-violet-500/20 rounded-xl">
                                        <p className="text-xs text-slate-400">
                                            <span className="font-bold text-violet-400">Equal allocation:</span> Each
                                            asset receives {(100 / config.symbols.length).toFixed(1)}% of the portfolio
                                            capital.
                                        </p>
                                    </div>
                                )}

                                {allocationMode === 'optimized' && (
                                    <div
                                        className="p-4 bg-gradient-to-br from-emerald-500/5 to-green-500/5 border border-emerald-500/20 rounded-xl">
                                        <p className="text-xs text-slate-400">
                                            <span
                                                className="font-bold text-emerald-400">Optimized allocation:</span> Capital
                                            allocation will be optimized based on risk-adjusted returns and correlation
                                            analysis.
                                        </p>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Enhanced Strategy Selector Card */}
                    <div
                        className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                        <div
                            className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
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
                                <div
                                    className="flex bg-slate-800/60 border border-slate-700/50 rounded-xl overflow-hidden">
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
                            {filteredStrategies.map((strat) => {
                                const isSelected = config.strategy === strat.id;
                                return (
                                    <button
                                        key={strat.id}
                                        onClick={() => setConfig({
                                            ...config,
                                            strategy: strat.id,
                                            params: typeof strat.parameters === 'object' && !Array.isArray(strat.parameters)
                                                ? strat.parameters
                                                : {}
                                        })}
                                        className={`group relative overflow-hidden p-4 rounded-xl border transition-all text-left ${isSelected
                                            ? 'border-violet-500 bg-gradient-to-br from-violet-500/10 to-purple-500/10 shadow-xl shadow-violet-500/20'
                                            : 'border-slate-700/50 bg-slate-800/40 hover:border-slate-600/50 hover:bg-slate-800/60'
                                            }`}
                                    >
                                        {isSelected && (
                                            <div
                                                className="absolute inset-0 bg-gradient-to-br from-violet-500/5 to-purple-500/5" />
                                        )}
                                        <div className="relative">
                                            <div className="flex items-center justify-between mb-2">
                                                <span
                                                    className={`text-[9px] font-bold uppercase tracking-wider px-2 py-1 rounded ${isSelected
                                                        ? 'bg-violet-500/20 text-violet-300 border border-violet-500/30'
                                                        : 'bg-slate-700/50 text-slate-400'
                                                        }`}>
                                                    {strat.category}
                                                </span>
                                                {isSelected && (
                                                    <div
                                                        className="w-6 h-6 rounded-full bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center">
                                                        <Check size={12} className="text-white" strokeWidth={3} />
                                                    </div>
                                                )}
                                            </div>
                                            <p className="text-sm font-bold text-slate-200 mb-2 group-hover:text-violet-300 transition-colors">
                                                {strat.name}
                                            </p>
                                            <p className="text-xs text-slate-400 line-clamp-2 mb-3">{strat.description}</p>
                                            <div className="flex items-center justify-between">
                                                <span
                                                    className={`text-[10px] px-2.5 py-1 rounded-lg font-medium ${strat.complexity === 'Advanced'
                                                        ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                                                        : strat.complexity === 'Intermediate'
                                                            ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                                                            : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                                                        }`}>
                                                    {strat.complexity}
                                                </span>
                                                <div className="flex items-center gap-1 text-amber-400">
                                                    <Star size={10} className="fill-current" />
                                                    <span className="text-xs font-bold">{strat.drawdown || 4.5}</span>
                                                </div>
                                            </div>
                                        </div>
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    {/* Backtest Configuration */}
                    {config.symbols.length >= 2 && (
                        <div
                            className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                            <div className="flex items-center gap-3 mb-6">
                                <div
                                    className="p-2 bg-gradient-to-br from-purple-500/20 to-violet-500/20 rounded-lg border border-purple-500/30">
                                    <Sliders className="text-purple-400" size={20} strokeWidth={2} />
                                </div>
                                <h4 className="text-sm font-bold text-slate-300">Backtest Configuration</h4>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                                <div className="space-y-3">
                                    <label
                                        className="text-xs font-bold text-slate-400 tracking-wide flex items-center gap-2">
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
                                    <label
                                        className="text-xs font-bold text-slate-400 tracking-wide flex items-center gap-2">
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
                                        <label className="text-xs font-bold text-slate-400 tracking-wide">Strategy
                                            Assignment</label>
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
                                                    onClick={() => setConfig({ ...config, strategyMode: mode.id as "same" | "different" | "portfolio" })}
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

                                {/* Different Strategy Mode - Per Asset Table */}
                                {config.strategyMode === 'different' && (
                                    <div
                                        className="bg-slate-800/40 rounded-xl border border-slate-700/50 overflow-hidden">
                                        <div className="px-4 py-3 bg-slate-800/60 border-b border-slate-700/50">
                                            <div className="flex items-center justify-between">
                                                <span className="text-xs font-bold text-slate-400">Custom Strategy Assignment</span>
                                                <span className="text-[10px] text-slate-600">Select unique strategy for each asset</span>
                                            </div>
                                        </div>
                                        <div className="divide-y divide-slate-700/50">
                                            {config.symbols.map((symbol: string, index: number) => (
                                                <div key={symbol}
                                                    className="p-4 hover:bg-slate-800/20 transition-colors">
                                                    <div className="flex items-center justify-between">
                                                        <div className="flex items-center gap-3">
                                                            <div
                                                                className="w-10 h-10 rounded-lg bg-gradient-to-br from-slate-700 to-slate-800 flex items-center justify-center">
                                                                <span
                                                                    className="font-mono font-bold text-violet-400">{symbol.slice(0, 2)}</span>
                                                            </div>
                                                            <div>
                                                                <p className="font-bold text-slate-200">{symbol}</p>
                                                                <p className="text-xs text-slate-500">Asset
                                                                    #{index + 1}</p>
                                                            </div>
                                                        </div>
                                                        <select
                                                            value={config.strategies[symbol] || config.strategy}
                                                            onChange={(e) => setConfig({
                                                                ...config,
                                                                strategies: {
                                                                    ...config.strategies,
                                                                    [symbol]: e.target.value
                                                                }
                                                            })}
                                                            className="bg-slate-800/50 border border-slate-700/50 rounded-xl px-4 py-2.5 text-sm text-slate-300 focus:border-violet-500 outline-none min-w-[180px]"
                                                        >
                                                            {strategies.map((st) => (
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
                                    <div
                                        className="p-4 bg-gradient-to-br from-violet-500/5 to-purple-500/5 border border-violet-500/20 rounded-xl">
                                        <div className="flex items-center gap-3">
                                            <Copy size={16} className="text-violet-400" />
                                            <div>
                                                <p className="text-xs font-bold text-violet-400">Same Strategy Mode</p>
                                                <p className="text-xs text-slate-400 mt-1">
                                                    The selected strategy will be applied to all assets in your
                                                    portfolio.
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
                                            <label
                                                className="text-xs font-bold text-slate-400 tracking-wide flex items-center gap-2">
                                                <DollarSign size={14} className="text-emerald-400" />
                                                Initial Capital
                                            </label>
                                            <div className="relative">
                                                <span
                                                    className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 font-bold">$</span>
                                                <input
                                                    type="number"
                                                    value={config.initialCapital}
                                                    onChange={(e) => setConfig({
                                                        ...config,
                                                        initialCapital: parseInt(e.target.value)
                                                    })}
                                                    className="w-full pl-8 pr-4 py-3.5 bg-slate-800/50 border border-slate-700/50 rounded-xl focus:border-violet-500 focus:ring-2 focus:ring-violet-500/20 outline-none text-slate-200 font-mono"
                                                    min="1000"
                                                    step="1000"
                                                />
                                            </div>
                                        </div>
                                        <div className="space-y-3">
                                            <label
                                                className="text-xs font-bold text-slate-400 tracking-wide flex items-center gap-2">
                                                <Target size={14} className="text-amber-400" />
                                                Max Position %
                                            </label>
                                            <input
                                                type="number"
                                                value={config.maxPositionPct}
                                                onChange={(e) => setConfig({
                                                    ...config,
                                                    maxPositionPct: parseInt(e.target.value)
                                                })}
                                                className="w-full px-4 py-3.5 bg-slate-800/50 border border-slate-700/50 rounded-xl focus:border-violet-500 focus:ring-2 focus:ring-violet-500/20 outline-none text-slate-200"
                                                min="1"
                                                max="100"
                                            />
                                        </div>
                                        <div className="space-y-3">
                                            <label
                                                className="text-xs font-bold text-slate-400 tracking-wide flex items-center gap-2">
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
                            <div
                                className="bg-gradient-to-br from-violet-900/40 via-purple-900/40 to-slate-900/90 backdrop-blur-xl border border-violet-500/30 rounded-2xl p-6 shadow-2xl relative overflow-hidden">
                                <div
                                    className="absolute inset-0 bg-gradient-to-br from-violet-500/5 via-transparent to-purple-500/5" />
                                <div className="relative">
                                    <div className="flex items-center justify-between mb-4">
                                        <div className="flex items-center gap-2">
                                            <div
                                                className="p-1.5 bg-gradient-to-br from-violet-500/30 to-purple-500/30 rounded border border-violet-500/50">
                                                <Zap size={16} className="text-violet-300" strokeWidth={2} />
                                            </div>
                                            <h4 className="text-xs font-bold text-violet-400 uppercase tracking-wider">Strategy
                                                Intelligence</h4>
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
                                        <p className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Expected
                                            Performance</p>
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
                            <div
                                className="h-[400px] flex flex-col items-center justify-center p-12 border-2 border-dashed border-slate-700/50 rounded-2xl bg-gradient-to-br from-slate-900/30 to-slate-800/30 backdrop-blur-sm">
                                <div
                                    className="w-20 h-20 rounded-full bg-gradient-to-br from-slate-800 to-slate-900 border-2 border-dashed border-slate-700/50 flex items-center justify-center mb-6">
                                    <Info size={32} className="text-slate-700" strokeWidth={1.5} />
                                </div>
                                <p className="text-sm font-bold text-slate-600 text-center">Select a Strategy</p>
                                <p className="text-xs text-slate-700 text-center mt-2 max-w-[200px]">
                                    Choose a strategy from the library to view detailed intelligence and configure
                                    parameters
                                </p>
                            </div>
                        )}

                        {/* Enhanced Parameter Editor Card */}
                        {selectedStrategy && selectedStrategy.parameters !== undefined && (

                            <div
                                className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                                <div className="flex items-center justify-between mb-6">
                                    <div className="flex items-center gap-3">
                                        <div className="p-1.5 bg-purple-500/20 rounded border border-purple-500/30">
                                            <Settings size={16} className="text-purple-400" strokeWidth={2} />
                                        </div>
                                        <div>
                                            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Strategy
                                                Parameters</h4>
                                            <p className="text-[10px] text-slate-600 font-medium mt-0.5">
                                                {Object.keys(selectedStrategy.parameters).length} parameters
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
                                            onClick={() => setConfig({ ...config, params: selectedStrategy?.parameters })}
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
                                            params={selectedStrategy.parameters}
                                            values={config.params || {}}
                                            onChange={handleParamChange}
                                        />

                                        <div className="mt-8 pt-6 border-t border-slate-700/50">
                                            <div className="flex items-center justify-between">
                                                <div className="flex items-center gap-2">
                                                    <div className="w-2 h-2 rounded-full bg-emerald-500" />
                                                    <span
                                                        className="text-xs font-bold text-slate-500 uppercase tracking-wider">Validated</span>
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
                        <div
                            className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">Quick
                                Actions</h4>
                            <div className="space-y-3">
                                <button
                                    onClick={handleSavePortfolio}
                                    disabled={isSaving}
                                    className="w-full flex items-center justify-between p-3 bg-slate-800/40 hover:bg-slate-800/60 border border-slate-700/50 rounded-xl transition-all group disabled:opacity-50">
                                    <div className="flex items-center gap-3">
                                        <div className="p-2 bg-emerald-500/20 rounded-lg border border-emerald-500/30">
                                            {isSaving ? (
                                                <RefreshCw size={16} className="text-emerald-400 animate-spin" />
                                            ) : (
                                                <Save size={16} className="text-emerald-400" />
                                            )}
                                        </div>
                                        <div className="text-left">
                                            <p className="text-sm font-bold text-slate-200">
                                                {isSaving ? 'Saving...' : 'Save Portfolio'}
                                            </p>
                                            <p className="text-xs text-slate-500">Store current configuration</p>
                                        </div>
                                    </div>
                                    <ChevronRight size={16} className="text-slate-500 group-hover:text-violet-400" />
                                </button>
                                <button
                                    onClick={handleExportResults}
                                    disabled={!results}
                                    className="w-full flex items-center justify-between p-3 bg-slate-800/40 hover:bg-slate-800/60 border border-slate-700/50 rounded-xl transition-all group disabled:opacity-50 disabled:cursor-not-allowed">
                                    <div className="flex items-center gap-3">
                                        <div className="p-2 bg-blue-500/20 rounded-lg border border-blue-500/30">
                                            <Download size={16} className="text-blue-400" />
                                        </div>
                                        <div className="text-left">
                                            <p className="text-sm font-bold text-slate-200">Export Results</p>
                                            <p className="text-xs text-slate-500">Download CSV report</p>
                                        </div>
                                    </div>
                                    <ChevronRight size={16} className="text-slate-500 group-hover:text-violet-400" />
                                </button>
                                <button
                                    onClick={() => setShowRiskAnalysis(true)}
                                    disabled={!results}
                                    className="w-full flex items-center justify-between p-3 bg-slate-800/40 hover:bg-slate-800/60 border border-slate-700/50 rounded-xl transition-all group disabled:opacity-50 disabled:cursor-not-allowed">
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
            {hasRunBacktest && results && (
                <div className="pt-6 border-t border-slate-800/80 animate-in fade-in">
                    <MultiBacktestResults results={results} />
                </div>
            )}

            {/* Risk Analysis Modal */}
            {showRiskAnalysis && results && (
                <RiskAnalysisModal
                    results={results}
                    onClose={() => setShowRiskAnalysis(false)}
                />
            )}
        </div>
    );
};

export default MultiAssetBacktest;