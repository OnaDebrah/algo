'use client'

import {
    Activity,
    AlertCircle,
    BarChart3,
    Calendar,
    Check,
    ChevronDown,
    ChevronRight,
    ChevronUp,
    Copy,
    Cpu,
    Database,
    DollarSign,
    Download,
    Eye,
    EyeOff,
    Filter,
    FolderOpen,
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
    Sparkles,
    Star,
    Target,
    X,
    Zap
} from "lucide-react";
import React, { useMemo, useState } from "react";
import StrategyParameterForm from "@/components/backtest/StrategyParameterForm";
import MultiBacktestResults from "@/components/backtest/MultiBacktestResults";
import RiskAnalysisModal from "@/components/backtest/RiskAnalysisModal";
import LoadConfigModal from "@/components/backtest/LoadConfigModal";
import { BacktestResult, MultiAssetConfig, Strategy, PortfolioCreate } from "@/types/all_types";
import { portfolio } from "@/utils/api";
import BayesianOptimizerModal from "@/components/backtest/BayesianOptimizerModal";
import { assetSuggestions } from "@/utils/suggestions";
import KalmanFilterParameters from "@/components/backtest/KalmanFilterParameters";
import StrategyInfoPopover from "@/components/backtest/StrategyInfoPopover";

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
    const [showParameters, setShowParameters] = useState(true);
    const [allocationMode, setAllocationMode] = useState<'equal' | 'manual' | 'optimized'>('equal');
    const [searchQuery, setSearchQuery] = useState('');
    const [showRiskAnalysis, setShowRiskAnalysis] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [showLoadModal, setShowLoadModal] = useState(false);
    const [showOptimizer, setShowOptimizer] = useState(false);
    const [expandedConfig, setExpandedConfig] = useState(!results);

    const selectedStrategy = useMemo(() =>
        strategies.find((s) => s.id === config.strategy),
        [config.strategy, strategies]
    );

    const categories = useMemo(() =>
        ['All', ...Array.from(new Set(strategies.map((s) => s.category)))],
        [strategies]
    );

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
    };

    const handleExportResults = () => {
        if (!results) return;

        try {
            const rows: (string | number)[][] = [];
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
            rows.push(['ASSET PERFORMANCE']);
            rows.push(['Symbol', 'Strategy', 'Total Return', 'Win Rate', 'Trades', 'Avg Profit', 'Loss Rate']);
            if (results.symbol_stats) {
                Object.entries(results.symbol_stats).forEach(([symbol, stats]) => {
                    const totalTrades = stats.total_trades || 0;
                    const losingTrades = stats.losing_trades || 0;
                    const lossRate = totalTrades > 0 ? (losingTrades / totalTrades) * 100 : 0;
                    rows.push([
                        symbol || 'Unknown', 'Multi Asset',
                        `${((stats.total_return || 0) * 100).toFixed(2)}%`,
                        `${(stats.win_rate || 0).toFixed(2)}%`,
                        totalTrades, (stats.avg_profit || 0).toFixed(2),
                        lossRate.toFixed(2) + '%'
                    ]);
                });
            }
            rows.push([]);
            rows.push(['TRADE LEDGER']);
            rows.push(['ID', 'Symbol', 'Side', 'Date', 'Qty', 'Price', 'Commission', 'Profit', 'Status']);
            if (results.trades && Array.isArray(results.trades)) {
                results.trades.forEach(t => {
                    rows.push([
                        t.id || '', t.symbol || '', t.order_type || '', t.executed_at || '',
                        t.quantity || 0, t.price || 0, t.commission || 0,
                        typeof t.profit === 'number' ? t.profit.toFixed(2) : '',
                        typeof t.profit === 'number' ? 'Closed' : 'Open'
                    ]);
                });
            }
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
            setTimeout(() => URL.revokeObjectURL(url), 100);
        } catch (error) {
            console.error('Error exporting results:', error);
            alert('Failed to export results. See console for details.');
        }
    };

    const handleSavePortfolio = async () => {
        try {
            setIsSaving(true);
            const portfolioName = `Backtest Allocation ${new Date().toLocaleDateString()}`;
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

    const isPairsStrategy = useMemo(() => {
        const pairsStrategies = ['kalman_filter', 'kalman_filter_hft', 'pairs_trading', 'cointegration'];
        return pairsStrategies.includes(config.strategy);
    }, [config.strategy]);

    return (
        <div className="space-y-4">
            {/* ════════════════════════════════════════════════════════════
                COMPACT CONFIGURATION BAR
               ════════════════════════════════════════════════════════════ */}
            <div className="bg-gradient-to-br from-slate-900/95 to-slate-800/95 backdrop-blur-xl border border-slate-700/50 rounded-2xl shadow-xl overflow-hidden">
                <div className="p-5">
                    <div className="flex items-end gap-4">
                        {/* Assets pill display */}
                        <div className="flex-[2] min-w-[200px]">
                            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1.5 block">
                                Portfolio Assets ({config.symbols.length})
                            </label>
                            <div className="flex items-center gap-2 min-h-[40px] px-3 py-1.5 bg-slate-800/60 border border-slate-700/50 rounded-xl overflow-x-auto">
                                {config.symbols.length > 0 ? (
                                    <div className="flex items-center gap-1.5 flex-nowrap">
                                        {config.symbols.map((s: string) => (
                                            <span key={s} className="flex items-center gap-1 px-2 py-1 bg-violet-500/10 border border-violet-500/30 rounded-lg text-xs font-mono font-bold text-violet-300 whitespace-nowrap">
                                                {s}
                                                <button onClick={() => removeSymbol(s)} className="text-slate-500 hover:text-red-400 transition-colors">
                                                    <X size={10} />
                                                </button>
                                            </span>
                                        ))}
                                    </div>
                                ) : (
                                    <span className="text-xs text-slate-600">Add 2+ assets below</span>
                                )}
                            </div>
                        </div>

                        {/* Strategy dropdown */}
                        <div className="flex-1 min-w-[180px]">
                            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1.5 block">Strategy</label>
                            <select
                                value={config.strategy}
                                onChange={(e) => {
                                    const strat = strategies.find(s => s.id === e.target.value);
                                    setConfig({
                                        ...config,
                                        strategy: e.target.value,
                                        params: typeof strat?.parameters === 'object' && !Array.isArray(strat?.parameters) ? strat.parameters : {}
                                    });
                                }}
                                className="w-full px-3 py-2.5 bg-slate-800/60 border border-slate-700/50 rounded-xl focus:border-violet-500 focus:ring-2 focus:ring-violet-500/20 outline-none text-sm text-slate-200 cursor-pointer"
                            >
                                {strategies.map(s => (
                                    <option key={s.id} value={s.id}>{s.name}</option>
                                ))}
                            </select>
                        </div>

                        {/* Period */}
                        <div className="w-[120px]">
                            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1.5 block">Period</label>
                            <select
                                value={config.period}
                                onChange={(e) => setConfig({ ...config, period: e.target.value })}
                                className="w-full px-3 py-2.5 bg-slate-800/60 border border-slate-700/50 rounded-xl focus:border-violet-500 outline-none text-sm text-slate-200"
                            >
                                <option value="1mo">1 Month</option>
                                <option value="3mo">3 Months</option>
                                <option value="6mo">6 Months</option>
                                <option value="1y">1 Year</option>
                                <option value="2y">2 Years</option>
                                <option value="5y">5 Years</option>
                            </select>
                        </div>

                        {/* Interval */}
                        <div className="w-[100px]">
                            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1.5 block">Interval</label>
                            <select
                                value={config.interval}
                                onChange={(e) => setConfig({ ...config, interval: e.target.value })}
                                className="w-full px-3 py-2.5 bg-slate-800/60 border border-slate-700/50 rounded-xl focus:border-violet-500 outline-none text-sm text-slate-200"
                            >
                                <option value="1h">1h</option>
                                <option value="4h">4h</option>
                                <option value="1d">1D</option>
                                <option value="1wk">1W</option>
                            </select>
                        </div>

                        {/* Run Button */}
                        <button
                            onClick={handleRunBacktest}
                            disabled={isRunning || config.symbols.length < 2}
                            className="group relative overflow-hidden flex items-center gap-2.5 px-7 py-2.5 bg-gradient-to-r from-violet-600 via-purple-600 to-fuchsia-600 hover:from-violet-500 hover:via-purple-500 hover:to-fuchsia-500 disabled:from-slate-700 disabled:via-slate-700 disabled:to-slate-700 disabled:cursor-not-allowed rounded-xl font-bold transition-all shadow-xl shadow-violet-500/30 disabled:shadow-none text-white whitespace-nowrap"
                        >
                            <div className="absolute inset-0 bg-gradient-to-r from-white/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                            {isRunning ? (
                                <>
                                    <RefreshCw size={18} className="animate-spin relative z-10" strokeWidth={2.5} />
                                    <span className="relative z-10">Running...</span>
                                </>
                            ) : (
                                <>
                                    <Play size={18} strokeWidth={2.5} className="relative z-10" />
                                    <span className="relative z-10">Run Backtest</span>
                                </>
                            )}
                        </button>
                    </div>

                    {/* Quick actions row */}
                    <div className="flex items-center justify-between mt-3 pt-3 border-t border-slate-700/30">
                        <div className="flex items-center gap-2 flex-wrap">
                            {selectedStrategy && (
                                <span className={`text-[10px] font-bold px-2.5 py-1 rounded-lg ${
                                    selectedStrategy.complexity === 'Advanced' || selectedStrategy.complexity === 'Expert'
                                        ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                                        : selectedStrategy.complexity === 'Intermediate'
                                            ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                                            : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                                }`}>
                                    {selectedStrategy.complexity}
                                </span>
                            )}
                            {config.symbols.length < 2 && (
                                <span className="text-[10px] font-bold px-2.5 py-1 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20">
                                    Need 2+ assets
                                </span>
                            )}
                            <span className="text-[10px] font-mono text-slate-500 bg-slate-800/40 px-2.5 py-1 rounded-lg border border-slate-700/50">
                                Allocation: {allocationMode}
                            </span>
                        </div>

                        <div className="flex items-center gap-2">
                            <button
                                onClick={handleSavePortfolio}
                                disabled={isSaving}
                                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold text-slate-400 hover:text-slate-200 bg-slate-800/40 hover:bg-slate-800/60 border border-slate-700/50 rounded-lg transition-all disabled:opacity-50"
                            >
                                {isSaving ? <RefreshCw size={12} className="animate-spin" /> : <Save size={12} />}
                                Save
                            </button>
                            <button
                                onClick={() => setShowLoadModal(true)}
                                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold text-slate-400 hover:text-slate-200 bg-slate-800/40 hover:bg-slate-800/60 border border-slate-700/50 rounded-lg transition-all"
                            >
                                <FolderOpen size={12} />
                                Load
                            </button>
                            <button
                                onClick={handleExportResults}
                                disabled={!results}
                                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold text-slate-400 hover:text-slate-200 bg-slate-800/40 hover:bg-slate-800/60 border border-slate-700/50 rounded-lg transition-all disabled:opacity-50"
                            >
                                <Download size={12} />
                                Export
                            </button>
                            <div className="w-px h-5 bg-slate-700/50 mx-1" />
                            <button
                                onClick={() => setExpandedConfig(!expandedConfig)}
                                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold text-violet-400 hover:text-violet-300 bg-violet-500/10 hover:bg-violet-500/20 border border-violet-500/20 rounded-lg transition-all"
                            >
                                {expandedConfig ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                                {expandedConfig ? 'Collapse' : 'Full Config'}
                            </button>
                        </div>
                    </div>
                </div>

                {/* ════════════════════════════════════════════════════════════
                    EXPANDED CONFIGURATION
                   ════════════════════════════════════════════════════════════ */}
                {expandedConfig && (
                    <div className="border-t border-slate-700/50 animate-in fade-in slide-in-from-top-2 duration-300">
                        <div className="grid grid-cols-12 gap-6 p-6">
                            {/* LEFT: Assets + Allocation */}
                            <div className="col-span-12 lg:col-span-6 space-y-5">
                                {/* Asset Input */}
                                <div>
                                    <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                                        <Database size={14} className="text-violet-400" />
                                        Portfolio Assets
                                    </h4>
                                    <div className="flex gap-2 mb-3">
                                        <div className="relative flex-1 group">
                                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-600 group-focus-within:text-violet-500 transition-colors" size={14} />
                                            <input
                                                type="text"
                                                placeholder="Search ticker (e.g. AAPL)"
                                                value={config.symbolInput}
                                                onChange={(e) => setConfig({ ...config, symbolInput: e.target.value.toUpperCase() })}
                                                onKeyUp={(e) => e.key === 'Enter' && addSymbol()}
                                                className="w-full pl-9 pr-3 py-2.5 bg-slate-800/50 border border-slate-700/50 rounded-xl focus:border-violet-500 focus:ring-2 focus:ring-violet-500/20 outline-none text-sm text-slate-200 font-mono"
                                            />
                                        </div>
                                        <button
                                            onClick={addSymbol}
                                            className="flex items-center gap-1.5 px-4 py-2.5 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50 rounded-xl text-sm font-bold text-slate-200 transition-all"
                                        >
                                            <PlusCircle size={16} />
                                            Add
                                        </button>
                                    </div>

                                    {/* Quick add */}
                                    <div className="flex flex-wrap gap-1.5 mb-3">
                                        {assetSuggestions.map((asset) => (
                                            <button
                                                key={asset.symbol}
                                                onClick={() => {
                                                    setConfig({ ...config, symbolInput: asset.symbol });
                                                    setTimeout(addSymbol, 100);
                                                }}
                                                className={`px-2 py-1 text-[10px] font-bold rounded-md border transition-all ${
                                                    config.symbols.includes(asset.symbol)
                                                        ? 'bg-violet-500/10 border-violet-500/30 text-violet-300'
                                                        : 'bg-slate-800/40 border-slate-700/50 text-slate-400 hover:border-violet-500/50'
                                                }`}
                                            >
                                                {asset.symbol}
                                            </button>
                                        ))}
                                    </div>

                                    {/* Selected assets */}
                                    <div className="min-h-[60px] p-3 bg-slate-900/30 border border-slate-700/30 rounded-xl">
                                        {config.symbols.length > 0 ? (
                                            <div className="flex flex-wrap gap-2">
                                                {config.symbols.map((symbol: string) => (
                                                    <div key={symbol} className="flex items-center gap-2 pl-3 pr-2 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg">
                                                        <span className="font-mono font-bold text-sm text-violet-300">{symbol}</span>
                                                        <button onClick={() => removeSymbol(symbol)} className="p-1 text-slate-500 hover:text-red-400 transition-colors rounded">
                                                            <X size={12} />
                                                        </button>
                                                    </div>
                                                ))}
                                            </div>
                                        ) : (
                                            <p className="text-xs text-slate-600 text-center py-3">Add at least 2 symbols</p>
                                        )}
                                    </div>
                                </div>

                                {/* Allocation Controls */}
                                {config.symbols.length >= 2 && (
                                    <div>
                                        <div className="flex justify-between items-center mb-3">
                                            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                                                <PieChart size={14} className="text-violet-400" />
                                                Allocation
                                            </h4>
                                            <div className="flex bg-slate-800/60 p-0.5 rounded-lg border border-slate-700/50">
                                                {['equal', 'manual', 'optimized'].map((mode) => (
                                                    <button
                                                        key={mode}
                                                        onClick={() => setAllocationMode(mode as never)}
                                                        className={`px-2.5 py-1 rounded-md text-[10px] font-bold uppercase transition-all ${
                                                            allocationMode === mode
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
                                            <div className="space-y-2">
                                                {config.symbols.map((symbol: string) => (
                                                    <div key={symbol} className="flex items-center justify-between">
                                                        <span className="text-xs text-slate-400 font-mono font-medium">{symbol}</span>
                                                        <div className="flex items-center gap-2">
                                                            <input
                                                                type="range" min="0" max="100"
                                                                value={config.allocations[symbol] || 100 / config.symbols.length}
                                                                onChange={(e) => setConfig({
                                                                    ...config,
                                                                    allocations: { ...config.allocations, [symbol]: parseInt(e.target.value) }
                                                                })}
                                                                className="w-28 accent-violet-500"
                                                            />
                                                            <span className="text-xs font-mono text-violet-400 w-10 text-right">
                                                                {config.allocations[symbol] || Math.round(100 / config.symbols.length)}%
                                                            </span>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                        {allocationMode === 'equal' && (
                                            <p className="text-xs text-slate-400 bg-violet-500/5 border border-violet-500/20 rounded-lg p-3">
                                                <span className="font-bold text-violet-400">Equal:</span> Each asset receives {(100 / config.symbols.length).toFixed(1)}% of capital.
                                            </p>
                                        )}
                                        {allocationMode === 'optimized' && (
                                            <p className="text-xs text-slate-400 bg-emerald-500/5 border border-emerald-500/20 rounded-lg p-3">
                                                <span className="font-bold text-emerald-400">Optimized:</span> Allocation will be optimized based on risk-adjusted returns.
                                            </p>
                                        )}
                                    </div>
                                )}

                                {/* Strategy Mode */}
                                {config.symbols.length >= 2 && (
                                    <div>
                                        <div className="flex justify-between items-center mb-3">
                                            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                                                <Cpu size={14} className="text-violet-400" />
                                                Strategy Assignment
                                            </h4>
                                            <div className="flex bg-slate-800/60 p-0.5 rounded-lg border border-slate-700/50">
                                                {[
                                                    { id: 'same', label: 'Same' },
                                                    { id: 'different', label: 'Per Asset' },
                                                    { id: 'portfolio', label: 'Portfolio' }
                                                ].map((mode) => (
                                                    <button
                                                        key={mode.id}
                                                        onClick={() => setConfig({ ...config, strategyMode: mode.id as "same" | "different" | "portfolio" })}
                                                        className={`px-2.5 py-1 rounded-md text-[10px] font-bold uppercase transition-all ${
                                                            config.strategyMode === mode.id
                                                                ? 'bg-gradient-to-r from-violet-600 to-purple-600 text-white'
                                                                : 'text-slate-500 hover:text-slate-300'
                                                        }`}
                                                    >
                                                        {mode.label}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>

                                        {config.strategyMode === 'different' && (
                                            <div className="space-y-2">
                                                {config.symbols.map((symbol: string) => (
                                                    <div key={symbol} className="flex items-center justify-between p-2 bg-slate-800/30 rounded-lg">
                                                        <span className="text-xs font-mono font-bold text-slate-300">{symbol}</span>
                                                        <select
                                                            value={config.strategies[symbol] || config.strategy}
                                                            onChange={(e) => setConfig({
                                                                ...config,
                                                                strategies: { ...config.strategies, [symbol]: e.target.value }
                                                            })}
                                                            className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-1.5 text-xs text-slate-300 focus:border-violet-500 outline-none"
                                                        >
                                                            {strategies.map((st) => (
                                                                <option key={st.id} value={st.id}>{st.name}</option>
                                                            ))}
                                                        </select>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                )}

                                {/* Advanced Options */}
                                <div>
                                    <button
                                        onClick={() => setShowAdvanced(!showAdvanced)}
                                        className="flex items-center gap-2 text-xs font-bold text-slate-400 hover:text-violet-400 transition-colors"
                                    >
                                        <ChevronDown size={14} className={`transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
                                        Advanced Options
                                    </button>
                                    {showAdvanced && (
                                        <div className="grid grid-cols-3 gap-4 mt-3 animate-in fade-in">
                                            <div className="space-y-2">
                                                <label className="text-[10px] font-bold text-slate-500">Initial Capital</label>
                                                <div className="relative">
                                                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 text-xs">$</span>
                                                    <input
                                                        type="number" value={config.initialCapital}
                                                        onChange={(e) => setConfig({ ...config, initialCapital: parseInt(e.target.value) })}
                                                        className="w-full pl-6 pr-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg focus:border-violet-500 outline-none text-sm text-slate-200 font-mono"
                                                        min="1000" step="1000"
                                                    />
                                                </div>
                                            </div>
                                            <div className="space-y-2">
                                                <label className="text-[10px] font-bold text-slate-500">Max Position %</label>
                                                <input
                                                    type="number" value={config.maxPositionPct}
                                                    onChange={(e) => setConfig({ ...config, maxPositionPct: parseInt(e.target.value) })}
                                                    className="w-full px-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg focus:border-violet-500 outline-none text-sm text-slate-200"
                                                    min="1" max="100"
                                                />
                                            </div>
                                            <div className="space-y-2">
                                                <label className="text-[10px] font-bold text-slate-500">Risk Level</label>
                                                <select
                                                    value={config.riskLevel || 'medium'}
                                                    onChange={(e) => setConfig({ ...config, riskLevel: e.target.value })}
                                                    className="w-full px-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg focus:border-violet-500 outline-none text-sm text-slate-200"
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

                            {/* RIGHT: Strategy Library + Parameters */}
                            <div className="col-span-12 lg:col-span-6 space-y-5">
                                {/* Strategy Library */}
                                <div>
                                    <div className="flex items-center justify-between mb-3">
                                        <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                                            <Layers size={14} className="text-purple-400" />
                                            Strategy Library
                                            <span className="text-slate-600 font-medium normal-case tracking-normal">
                                                ({filteredStrategies.length})
                                            </span>
                                        </h4>
                                        <div className="flex gap-2">
                                            <div className="relative group">
                                                <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-600 group-focus-within:text-violet-500 transition-colors" size={13} />
                                                <input
                                                    type="text" placeholder="Search..."
                                                    value={searchQuery}
                                                    onChange={(e) => setSearchQuery(e.target.value)}
                                                    className="pl-8 pr-3 py-1.5 bg-slate-800/60 border border-slate-700/50 rounded-lg focus:border-violet-500 outline-none text-xs text-slate-200 w-36"
                                                />
                                            </div>
                                            <select
                                                value={activeCategory}
                                                onChange={(e) => setActiveCategory(e.target.value)}
                                                className="px-3 py-1.5 bg-slate-800/60 border border-slate-700/50 rounded-lg text-xs font-bold text-slate-300 focus:border-violet-500 outline-none cursor-pointer"
                                            >
                                                {categories.map((cat) => (
                                                    <option key={cat} value={cat}>{cat}</option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-3 gap-2 max-h-[250px] overflow-y-auto pr-1 custom-scrollbar">
                                        {filteredStrategies.map((strat) => {
                                            const isSelected = config.strategy === strat.id;
                                            return (
                                                <button
                                                    key={strat.id}
                                                    onClick={() => setConfig({
                                                        ...config,
                                                        strategy: strat.id,
                                                        params: typeof strat.parameters === 'object' && !Array.isArray(strat.parameters)
                                                            ? strat.parameters : {}
                                                    })}
                                                    className={`group relative overflow-hidden p-3 rounded-xl border transition-all text-left ${isSelected
                                                        ? 'border-violet-500 bg-gradient-to-br from-violet-500/10 to-purple-500/10 shadow-lg shadow-violet-500/20'
                                                        : 'border-slate-700/50 bg-slate-800/40 hover:border-slate-600/50 hover:bg-slate-800/60'
                                                    }`}
                                                >
                                                    {isSelected && <div className="absolute inset-0 bg-gradient-to-br from-violet-500/5 to-purple-500/5" />}
                                                    <div className="relative">
                                                        <div className="flex items-center justify-between mb-1.5">
                                                            <span className={`text-[8px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded ${isSelected
                                                                ? 'bg-violet-500/20 text-violet-300' : 'bg-slate-700/50 text-slate-500'
                                                            }`}>{strat.category}</span>
                                                            <div className="flex items-center gap-1">
                                                                <StrategyInfoPopover strategy={strat} />
                                                                {isSelected && (
                                                                    <div className="w-5 h-5 rounded-full bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center">
                                                                        <Check size={10} className="text-white" strokeWidth={3} />
                                                                    </div>
                                                                )}
                                                            </div>
                                                        </div>
                                                        <p className="text-xs font-bold text-slate-200 mb-1 group-hover:text-violet-300 transition-colors line-clamp-1">{strat.name}</p>
                                                        <p className="text-[10px] text-slate-500 line-clamp-1">{strat.description}</p>
                                                    </div>
                                                </button>
                                            );
                                        })}
                                    </div>
                                </div>

                                {/* Parameters + Optimizer */}
                                <div className="flex gap-4">
                                    {selectedStrategy && selectedStrategy.parameters !== undefined && (
                                        <div className="flex-1 bg-slate-800/30 border border-slate-700/30 rounded-xl p-4">
                                            <div className="flex items-center justify-between mb-3">
                                                <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                                                    <Settings size={12} className="text-purple-400" />
                                                    Parameters
                                                </h4>
                                                <div className="flex gap-1.5">
                                                    <button onClick={() => setShowParameters(!showParameters)}
                                                        className="p-1.5 hover:bg-slate-700/50 rounded-lg text-slate-500 hover:text-violet-400 transition-all">
                                                        {showParameters ? <EyeOff size={12} /> : <Eye size={12} />}
                                                    </button>
                                                    <button onClick={() => setConfig({ ...config, params: selectedStrategy?.parameters })}
                                                        className="p-1.5 hover:bg-slate-700/50 rounded-lg text-slate-500 hover:text-violet-400 transition-all">
                                                        <RefreshCw size={12} />
                                                    </button>
                                                </div>
                                            </div>
                                            {showParameters && (
                                                <div className="max-h-[200px] overflow-y-auto custom-scrollbar">
                                                    {isPairsStrategy ? (
                                                        <KalmanFilterParameters config={config} setConfig={setConfig} />
                                                    ) : (
                                                        <StrategyParameterForm
                                                            params={selectedStrategy.parameters}
                                                            values={config.params || {}}
                                                            onChange={handleParamChange}
                                                        />
                                                    )}
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    {selectedStrategy && (
                                        <div className="w-[180px] bg-gradient-to-br from-indigo-900/30 to-slate-900/30 border border-indigo-500/20 rounded-xl p-4 flex flex-col justify-between">
                                            <div>
                                                <h4 className="text-[10px] font-bold text-indigo-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                                                    <Sparkles size={12} />
                                                    Optimizer
                                                </h4>
                                                <p className="text-[10px] text-slate-500 leading-relaxed">
                                                    Find optimal parameters across all assets
                                                </p>
                                            </div>
                                            <button
                                                onClick={() => setShowOptimizer(true)}
                                                className="mt-3 w-full py-2 bg-indigo-600/20 hover:bg-indigo-600/30 border border-indigo-500/50 text-indigo-300 rounded-lg text-[10px] font-black transition-all flex items-center justify-center gap-1.5"
                                            >
                                                <Zap size={12} fill="currentColor" />
                                                LAUNCH
                                            </button>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* ════════════════════════════════════════════════════════════
                RESULTS SECTION
               ════════════════════════════════════════════════════════════ */}
            {results && (
                <div className="animate-in fade-in slide-in-from-bottom-3 duration-500">
                    <MultiBacktestResults results={results} />
                </div>
            )}

            {/* Empty state */}
            {!results && !isRunning && (
                <div className="flex flex-col items-center justify-center py-20 border-2 border-dashed border-slate-800/50 rounded-2xl bg-slate-900/20">
                    <div className="w-20 h-20 rounded-full bg-gradient-to-br from-slate-800 to-slate-900 border-2 border-dashed border-slate-700/50 flex items-center justify-center mb-6">
                        <BarChart3 size={36} className="text-slate-700" strokeWidth={1.5} />
                    </div>
                    <p className="text-lg font-bold text-slate-600">Ready to Backtest</p>
                    <p className="text-sm text-slate-700 text-center mt-2 max-w-md">
                        Add 2+ assets, choose a strategy, then click <strong className="text-violet-400">Run Backtest</strong>
                    </p>
                </div>
            )}

            {/* Running state */}
            {isRunning && !results && (
                <div className="flex flex-col items-center justify-center py-20 rounded-2xl bg-slate-900/20">
                    <div className="w-16 h-16 rounded-full bg-gradient-to-br from-violet-500/20 to-purple-500/20 border border-violet-500/30 flex items-center justify-center mb-6 animate-pulse">
                        <RefreshCw size={28} className="text-violet-400 animate-spin" />
                    </div>
                    <p className="text-lg font-bold text-slate-300">Running Multi-Asset Backtest</p>
                    <p className="text-sm text-slate-500 mt-2">Analyzing {config.symbols.join(', ')} with {selectedStrategy?.name || config.strategy}...</p>
                </div>
            )}

            {/* ═══ MODALS ═══ */}
            {showRiskAnalysis && results && (
                <RiskAnalysisModal results={results} onClose={() => setShowRiskAnalysis(false)} />
            )}

            {showLoadModal && (
                <LoadConfigModal
                    mode="multi"
                    onClose={() => setShowLoadModal(false)}
                    onSelect={(savedConfig: any) => {
                        setConfig({
                            ...config,
                            strategy: savedConfig.strategy || config.strategy,
                            symbols: savedConfig.symbols || config.symbols,
                            allocations: savedConfig.allocations || config.allocations,
                            params: savedConfig.params || config.params,
                            period: savedConfig.period || config.period,
                            interval: savedConfig.interval || config.interval,
                        });
                    }}
                />
            )}

            {showOptimizer && selectedStrategy && (
                <BayesianOptimizerModal
                    symbols={config.symbols}
                    strategy={selectedStrategy}
                    onClose={() => setShowOptimizer(false)}
                    onApply={(bestParams) => {
                        setConfig({ ...config, params: bestParams });
                        setShowOptimizer(false);
                    }}
                />
            )}
        </div>
    );
};

export default MultiAssetBacktest;
