'use client'
import React, { useEffect, useMemo, useState } from 'react';
import {
    Activity,
    AlertCircle,
    BarChart3,
    Brain,
    Calendar,
    Check,
    ChevronDown,
    ChevronRight,
    ChevronUp,
    Clock,
    Copy,
    DollarSign,
    Download,
    Eye,
    EyeOff,
    Filter,
    FolderOpen,
    Grid,
    Info,
    List,
    Play,
    RefreshCw,
    Save,
    Search,
    Settings,
    Sparkles,
    Star,
    Target,
    TrendingUp,
    X,
    Zap
} from 'lucide-react';
import SingleBacktestResults from "@/components/backtest/SingleBacktestResults";
import StrategyParameterForm from "@/components/backtest/StrategyParameterForm";
import RiskAnalysisModal from "@/components/backtest/RiskAnalysisModal";
import LoadConfigModal from "@/components/backtest/LoadConfigModal";
import BayesianOptimizerModal from "@/components/backtest/BayesianOptimizerModal";
import { BacktestResult, SingleAssetConfig, Strategy, PortfolioCreate, DeployedMLModel } from "@/types/all_types";
import { formatCSVCell, formatCurrency } from "@/utils/formatters";
import { portfolio, mlstudio } from "@/utils/api";
import { assetSuggestions, quickSuggestions } from "@/utils/suggestions";
import { useBacktestStore } from "@/store/useBacktestStore";
import { useNavigationStore } from "@/store/useNavigationStore";

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
    const [showParameters, setShowParameters] = useState(true);
    const [showRiskAnalysis, setShowRiskAnalysis] = useState(false);
    const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
    const [searchQuery, setSearchQuery] = useState('');
    const [isSaving, setIsSaving] = useState(false);
    const [showLoadModal, setShowLoadModal] = useState(false);
    const [showOptimizer, setShowOptimizer] = useState(false);
    const [deployedModels, setDeployedModels] = useState<DeployedMLModel[]>([]);
    const [loadingModels, setLoadingModels] = useState(false);
    const [expandedConfig, setExpandedConfig] = useState(!results);

    const clearVisualStrategy = useBacktestStore(state => state.clearVisualStrategy);
    const navigateTo = useNavigationStore(state => state.navigateTo);
    const isVisualBuilder = config.strategy === 'visual_builder';
    const visualBlockCount = isVisualBuilder ? (config.params?.blocks?.length || 0) : 0;

    const ML_STRATEGY_KEYS = ['ml_random_forest', 'ml_gradient_boosting', 'ml_svm', 'ml_logistic', 'ml_lstm', 'mc_ml_sentiment'];

    const categories = useMemo(() =>
        ['All', ...Array.from(new Set(strategies.map((s: Strategy) => s.category)))],
        [strategies]
    );

    const selectedStrategy = useMemo(() =>
        strategies.find((s: Strategy) => s.id === config.strategy),
        [config.strategy, strategies]
    );

    const isMLStrategy = useMemo(() =>
        ML_STRATEGY_KEYS.includes(config.strategy),
        [config.strategy]
    );

    useEffect(() => {
        if (isMLStrategy) {
            setLoadingModels(true);
            mlstudio.getDeployedModels()
                .then((models) => {
                    setDeployedModels(models || []);
                })
                .catch((err) => {
                    console.error('Failed to fetch deployed models:', err);
                    setDeployedModels([]);
                })
                .finally(() => setLoadingModels(false));
        } else {
            if (config.ml_model_id) {
                setConfig({ ...config, ml_model_id: undefined });
            }
            setDeployedModels([]);
        }
    }, [isMLStrategy]);

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

    const handleParamChange = (key: string, val: any) => {
        setConfig({
            ...config,
            params: { ...(config.params || {}), [key]: val }
        });
    };

    const handleExport = () => {
        if (!results) return;

        const rows = [];
        rows.push(['Metric', 'Value'].map(formatCSVCell));
        rows.push(['Total Return', `${(results.total_return * 100).toFixed(2)}%`].map(formatCSVCell));
        rows.push(['Win Rate', `${results.win_rate.toFixed(2)}%`].map(formatCSVCell));
        rows.push(['Sharpe Ratio', results.sharpe_ratio.toFixed(2)].map(formatCSVCell));
        rows.push(['Max Drawdown', `${(results.max_drawdown).toFixed(2)}%`].map(formatCSVCell));
        rows.push(['Profit Factor', results.profit_factor?.toFixed(2) || 'N/A'].map(formatCSVCell));
        rows.push(['Total Trades', results.total_trades].map(formatCSVCell));
        rows.push([]);
        rows.push(['Symbol', 'Side', 'Date', 'Qty', 'Price', 'Commission', 'Profit', 'Profit %', 'Status'].map(formatCSVCell));
        if (results.trades) {
            results.trades.forEach(t => {
                rows.push([
                    t.symbol, t.order_type, t.executed_at, t.quantity,
                    t.price.toFixed(2), t.commission.toFixed(2),
                    t.profit !== null && t.profit !== undefined ? t.profit.toFixed(2) : '0',
                    t.profit_pct !== null && t.profit_pct !== undefined ? `${t.profit_pct.toFixed(2)}%` : '0%',
                    t.profit !== null ? 'Closed' : 'Open'
                ].map(formatCSVCell));
            });
        }
        const csvString = rows.map(r => r.join(",")).join("\n");
        const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        const filename = `backtest_${config.symbol}_${new Date().toISOString().slice(0, 10)}.csv`;
        link.href = url;
        link.setAttribute("download", filename);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    };

    const handleSavePortfolio = async () => {
        try {
            setIsSaving(true);
            const portfolioName = `Single Backtest: ${config.symbol || 'Unnamed'} ${new Date().toLocaleDateString()}`;
            const description = JSON.stringify({
                type: 'single',
                strategy: config.strategy,
                symbol: config.symbol,
                params: config.params,
                period: config.period,
                interval: config.interval,
                riskLevel: config.riskLevel,
                initialCapital: config.initialCapital
            });
            const data: PortfolioCreate = {
                name: portfolioName,
                initial_capital: config.initialCapital,
                description: description
            };
            await portfolio.create(data);
            alert('Backtest Configuration Saved Successfully!');
        } catch (error) {
            console.error('Failed to save portfolio:', error);
            alert('Failed to save portfolio. Please try again.');
        } finally {
            setIsSaving(false);
        }
    };

    return (
        <div className="space-y-4">
            {/* Visual Strategy Builder info card */}
            {isVisualBuilder && (
                <div className="bg-gradient-to-r from-fuchsia-900/30 via-violet-900/30 to-slate-900/30 border border-fuchsia-500/30 rounded-2xl p-4 flex items-center justify-between animate-in fade-in">
                    <div className="flex items-center gap-4">
                        <div className="p-2.5 bg-fuchsia-500/20 rounded-xl border border-fuchsia-500/30">
                            <Brain size={20} className="text-fuchsia-400" />
                        </div>
                        <div>
                            <h3 className="text-sm font-bold text-fuchsia-300 flex items-center gap-2">
                                Visual Strategy Builder
                                <span className="text-[10px] bg-fuchsia-500/20 text-fuchsia-400 px-2 py-0.5 rounded-full uppercase font-black">Active</span>
                            </h3>
                            <p className="text-xs text-slate-400 mt-0.5">
                                {visualBlockCount > 0 ? `${visualBlockCount} blocks configured` : 'No blocks configured'}
                            </p>
                        </div>
                    </div>
                    <div className="flex items-center gap-3">
                        <button
                            onClick={() => navigateTo('ml-studio')}
                            className="px-4 py-2 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50 text-slate-300 rounded-xl text-xs font-bold transition-all"
                        >
                            Edit in ML Studio
                        </button>
                        <button
                            onClick={() => clearVisualStrategy()}
                            className="px-4 py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 text-red-400 rounded-xl text-xs font-bold transition-all"
                        >
                            <X size={14} className="inline mr-1" />
                            Clear
                        </button>
                    </div>
                </div>
            )}

            {/* ════════════════════════════════════════════════════════════
                COMPACT CONFIGURATION BAR
               ════════════════════════════════════════════════════════════ */}
            <div className="bg-gradient-to-br from-slate-900/95 to-slate-800/95 backdrop-blur-xl border border-slate-700/50 rounded-2xl shadow-xl overflow-hidden">
                {/* Top row: key inputs + run button */}
                <div className="p-5">
                    <div className="flex items-end gap-4">
                        {/* Asset */}
                        <div className="flex-1 min-w-[140px] max-w-[200px]">
                            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1.5 block">Asset</label>
                            <div className="relative group">
                                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-600 group-focus-within:text-violet-500 transition-colors" size={14} />
                                <input
                                    type="text"
                                    value={config.symbol}
                                    onChange={(e) => setConfig({ ...config, symbol: e.target.value.toUpperCase() })}
                                    className="w-full pl-9 pr-3 py-2.5 bg-slate-800/60 border border-slate-700/50 rounded-xl focus:border-violet-500 focus:ring-2 focus:ring-violet-500/20 outline-none text-sm text-slate-200 font-mono"
                                    placeholder="AAPL"
                                />
                            </div>
                        </div>

                        {/* Strategy dropdown */}
                        <div className="flex-[2] min-w-[200px]">
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
                                <option value="max">Max</option>
                            </select>
                        </div>

                        {/* Interval */}
                        <div className="w-[110px]">
                            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1.5 block">Interval</label>
                            <select
                                value={config.interval}
                                onChange={(e) => setConfig({ ...config, interval: e.target.value })}
                                className="w-full px-3 py-2.5 bg-slate-800/60 border border-slate-700/50 rounded-xl focus:border-violet-500 outline-none text-sm text-slate-200"
                            >
                                <option value="1m">1m</option>
                                <option value="5m">5m</option>
                                <option value="15m">15m</option>
                                <option value="1h">1h</option>
                                <option value="1d">1D</option>
                                <option value="1wk">1W</option>
                            </select>
                        </div>

                        {/* Capital */}
                        <div className="w-[140px]">
                            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1.5 block">Capital</label>
                            <div className="relative">
                                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 text-sm font-bold">$</span>
                                <input
                                    type="number"
                                    value={config.initialCapital}
                                    onChange={(e) => setConfig({ ...config, initialCapital: parseInt(e.target.value) })}
                                    className="w-full pl-7 pr-3 py-2.5 bg-slate-800/60 border border-slate-700/50 rounded-xl focus:border-violet-500 outline-none text-sm text-slate-200 font-mono"
                                    min="100"
                                    step="100"
                                />
                            </div>
                        </div>

                        {/* Run Button */}
                        <button
                            onClick={runBacktest}
                            disabled={isRunning || !config.symbol}
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

                    {/* Param chips + quick actions row */}
                    <div className="flex items-center justify-between mt-3 pt-3 border-t border-slate-700/30">
                        <div className="flex items-center gap-2 flex-wrap">
                            {/* Show key params as chips */}
                            {selectedStrategy && config.params && Object.entries(config.params).slice(0, 4).map(([key, val]) => (
                                <span key={key} className="text-[10px] font-mono font-bold text-slate-400 bg-slate-800/60 px-2.5 py-1 rounded-lg border border-slate-700/50">
                                    {key}: {typeof val === 'number' ? val : String(val)}
                                </span>
                            ))}
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
                                onClick={handleExport}
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
                    EXPANDED CONFIGURATION (collapsible)
                   ════════════════════════════════════════════════════════════ */}
                {expandedConfig && (
                    <div className="border-t border-slate-700/50 animate-in fade-in slide-in-from-top-2 duration-300">
                        <div className="grid grid-cols-12 gap-6 p-6">
                            {/* LEFT: Asset & Strategy Selection */}
                            <div className="col-span-12 lg:col-span-6 space-y-6">
                                {/* Asset Selection */}
                                <div>
                                    <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                                        <Target size={14} className="text-violet-400" />
                                        Quick Asset Selection
                                    </h4>
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
                                                    <span className="text-xs font-bold text-slate-300">{asset.symbol}</span>
                                                    <span className="text-[10px] text-slate-500">{asset.sector}</span>
                                                    {config.symbol === asset.symbol && <Check size={12} className="text-violet-400" />}
                                                </div>
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {/* Quick Setup Presets */}
                                <div>
                                    <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                                        <Clock size={14} className="text-amber-400" />
                                        Quick Setup
                                    </h4>
                                    <div className="grid grid-cols-4 gap-2">
                                        {quickSuggestions.map((suggestion) => (
                                            <button
                                                key={suggestion.label}
                                                onClick={() => setConfig({
                                                    ...config,
                                                    period: suggestion.period,
                                                    interval: suggestion.interval,
                                                    initialCapital: suggestion.capital
                                                })}
                                                className="group p-2.5 bg-slate-800/40 border border-slate-700/50 hover:border-violet-500/50 rounded-lg transition-all text-left"
                                            >
                                                <p className="text-xs font-bold text-slate-300 group-hover:text-violet-300">{suggestion.label}</p>
                                                <p className="text-[10px] text-slate-500 mt-0.5">{suggestion.period} &bull; {formatCurrency(suggestion.capital)}</p>
                                            </button>
                                        ))}
                                    </div>
                                </div>

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
                                                <label className="text-[10px] font-bold text-slate-500">Max Position %</label>
                                                <input
                                                    type="number"
                                                    value={config.maxPositionPct || 100}
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
                                            <div className="space-y-2">
                                                <label className="text-[10px] font-bold text-slate-500">Commission</label>
                                                <div className="relative">
                                                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 text-xs">$</span>
                                                    <input
                                                        type="number"
                                                        value={config.commission || 0}
                                                        onChange={(e) => setConfig({ ...config, commission: parseFloat(e.target.value) })}
                                                        className="w-full pl-6 pr-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg focus:border-violet-500 outline-none text-sm text-slate-200"
                                                        min="0" step="0.01"
                                                    />
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* RIGHT: Strategy Library + Parameters */}
                            <div className="col-span-12 lg:col-span-6 space-y-6">
                                {/* Strategy Library */}
                                <div>
                                    <div className="flex items-center justify-between mb-3">
                                        <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                                            <TrendingUp size={14} className="text-purple-400" />
                                            Strategy Library
                                            <span className="text-slate-600 font-medium normal-case tracking-normal">
                                                ({filteredStrategies.length})
                                            </span>
                                        </h4>
                                        <div className="flex gap-2">
                                            <div className="relative group">
                                                <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-600 group-focus-within:text-violet-500 transition-colors" size={13} />
                                                <input
                                                    type="text"
                                                    placeholder="Search..."
                                                    value={searchQuery}
                                                    onChange={(e) => setSearchQuery(e.target.value)}
                                                    className="pl-8 pr-3 py-1.5 bg-slate-800/60 border border-slate-700/50 rounded-lg focus:border-violet-500 outline-none text-xs text-slate-200 w-36"
                                                />
                                            </div>
                                            <select
                                                value={selectedCategory}
                                                onChange={(e) => setSelectedCategory(e.target.value)}
                                                className="px-3 py-1.5 bg-slate-800/60 border border-slate-700/50 rounded-lg text-xs font-bold text-slate-300 focus:border-violet-500 outline-none cursor-pointer"
                                            >
                                                {categories.map((cat) => (
                                                    <option key={cat} value={cat}>{cat}</option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-3 gap-2 max-h-[250px] overflow-y-auto pr-1 custom-scrollbar">
                                        {filteredStrategies.map((strategy: Strategy) => {
                                            const isSelected = config.strategy === strategy.id;
                                            return (
                                                <button
                                                    key={strategy.id}
                                                    onClick={() => setConfig({
                                                        ...config,
                                                        strategy: strategy.id,
                                                        params: typeof strategy.parameters === 'object' && !Array.isArray(strategy.parameters) ? strategy.parameters : {}
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
                                                            }`}>{strategy.category}</span>
                                                            {isSelected && (
                                                                <div className="w-5 h-5 rounded-full bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center">
                                                                    <Check size={10} className="text-white" strokeWidth={3} />
                                                                </div>
                                                            )}
                                                        </div>
                                                        <p className="text-xs font-bold text-slate-200 mb-1 group-hover:text-violet-300 transition-colors line-clamp-1">{strategy.name}</p>
                                                        <p className="text-[10px] text-slate-500 line-clamp-1">{strategy.description}</p>
                                                    </div>
                                                </button>
                                            );
                                        })}
                                    </div>
                                </div>

                                {/* Strategy Parameters + Optimizer in one row */}
                                <div className="flex gap-4">
                                    {/* Parameters */}
                                    {selectedStrategy && selectedStrategy.parameters && Object.keys(selectedStrategy.parameters).length > 0 && (
                                        <div className="flex-1 bg-slate-800/30 border border-slate-700/30 rounded-xl p-4">
                                            <div className="flex items-center justify-between mb-3">
                                                <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                                                    <Settings size={12} className="text-purple-400" />
                                                    Parameters
                                                </h4>
                                                <div className="flex gap-1.5">
                                                    <button
                                                        onClick={() => setShowParameters(!showParameters)}
                                                        className="p-1.5 hover:bg-slate-700/50 rounded-lg text-slate-500 hover:text-violet-400 transition-all"
                                                    >
                                                        {showParameters ? <EyeOff size={12} /> : <Eye size={12} />}
                                                    </button>
                                                    <button
                                                        onClick={() => setConfig({ ...config, params: selectedStrategy?.parameters })}
                                                        className="p-1.5 hover:bg-slate-700/50 rounded-lg text-slate-500 hover:text-violet-400 transition-all"
                                                    >
                                                        <RefreshCw size={12} />
                                                    </button>
                                                </div>
                                            </div>
                                            {showParameters && (
                                                <div className="max-h-[200px] overflow-y-auto custom-scrollbar">
                                                    <StrategyParameterForm
                                                        params={selectedStrategy.parameters}
                                                        values={config.params || {}}
                                                        onChange={handleParamChange}
                                                    />
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    {/* ML Model Selector */}
                                    {isMLStrategy && (
                                        <div className="w-[240px] bg-slate-800/30 border border-slate-700/30 rounded-xl p-4">
                                            <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                                                <Brain size={12} className="text-purple-400" />
                                                Model
                                            </h4>
                                            {loadingModels ? (
                                                <div className="flex items-center gap-2 text-xs text-slate-500">
                                                    <RefreshCw size={14} className="animate-spin" />
                                                    Loading...
                                                </div>
                                            ) : (
                                                <div className="space-y-1.5 max-h-[180px] overflow-y-auto">
                                                    <button
                                                        onClick={() => setConfig({ ...config, ml_model_id: undefined })}
                                                        className={`w-full p-2 rounded-lg border text-left text-xs transition-all ${
                                                            !config.ml_model_id
                                                                ? 'border-emerald-500 bg-emerald-500/10'
                                                                : 'border-slate-700/50 bg-slate-800/40 hover:border-slate-600/50'
                                                        }`}
                                                    >
                                                        <p className="font-bold text-slate-200">Auto-Train</p>
                                                        <p className="text-[10px] text-slate-500">Train fresh on data</p>
                                                    </button>
                                                    {deployedModels.map((model) => (
                                                        <button
                                                            key={model.id}
                                                            onClick={() => setConfig({ ...config, ml_model_id: model.id })}
                                                            className={`w-full p-2 rounded-lg border text-left text-xs transition-all ${
                                                                config.ml_model_id === model.id
                                                                    ? 'border-purple-500 bg-purple-500/10'
                                                                    : 'border-slate-700/50 bg-slate-800/40 hover:border-slate-600/50'
                                                            }`}
                                                        >
                                                            <p className="font-bold text-slate-200">{model.name}</p>
                                                            <p className="text-[10px] text-slate-500">{(model.test_accuracy * 100).toFixed(1)}% acc</p>
                                                        </button>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    {/* Optimizer button */}
                                    {selectedStrategy && !isMLStrategy && (
                                        <div className="w-[180px] bg-gradient-to-br from-indigo-900/30 to-slate-900/30 border border-indigo-500/20 rounded-xl p-4 flex flex-col justify-between">
                                            <div>
                                                <h4 className="text-[10px] font-bold text-indigo-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                                                    <Sparkles size={12} />
                                                    Optimizer
                                                </h4>
                                                <p className="text-[10px] text-slate-500 leading-relaxed">
                                                    Find optimal parameters with Bayesian search
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
                    <SingleBacktestResults results={results} />
                </div>
            )}

            {/* Empty state when no results */}
            {!results && !isRunning && (
                <div className="flex flex-col items-center justify-center py-20 border-2 border-dashed border-slate-800/50 rounded-2xl bg-slate-900/20">
                    <div className="w-20 h-20 rounded-full bg-gradient-to-br from-slate-800 to-slate-900 border-2 border-dashed border-slate-700/50 flex items-center justify-center mb-6">
                        <BarChart3 size={36} className="text-slate-700" strokeWidth={1.5} />
                    </div>
                    <p className="text-lg font-bold text-slate-600">Ready to Backtest</p>
                    <p className="text-sm text-slate-700 text-center mt-2 max-w-md">
                        Configure your asset, strategy, and parameters above, then click <strong className="text-violet-400">Run Backtest</strong> to analyze performance
                    </p>
                </div>
            )}

            {/* Running state */}
            {isRunning && !results && (
                <div className="flex flex-col items-center justify-center py-20 rounded-2xl bg-slate-900/20">
                    <div className="w-16 h-16 rounded-full bg-gradient-to-br from-violet-500/20 to-purple-500/20 border border-violet-500/30 flex items-center justify-center mb-6 animate-pulse">
                        <RefreshCw size={28} className="text-violet-400 animate-spin" />
                    </div>
                    <p className="text-lg font-bold text-slate-300">Running Backtest</p>
                    <p className="text-sm text-slate-500 mt-2">Analyzing {config.symbol} with {selectedStrategy?.name || config.strategy}...</p>
                </div>
            )}

            {/* ═══ MODALS ═══ */}
            {showRiskAnalysis && results && (
                <RiskAnalysisModal results={results} onClose={() => setShowRiskAnalysis(false)} />
            )}

            {showLoadModal && (
                <LoadConfigModal
                    mode="single"
                    onClose={() => setShowLoadModal(false)}
                    onSelect={(savedConfig: any) => {
                        setConfig({
                            ...config,
                            strategy: savedConfig.strategy || config.strategy,
                            symbol: savedConfig.symbol || config.symbol,
                            params: savedConfig.params || config.params,
                            period: savedConfig.period || config.period,
                            interval: savedConfig.interval || config.interval,
                            riskLevel: savedConfig.riskLevel || config.riskLevel,
                            initialCapital: savedConfig.initialCapital || config.initialCapital,
                        });
                    }}
                />
            )}

            {showOptimizer && selectedStrategy && (
                <BayesianOptimizerModal
                    symbols={[config.symbol]}
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

export default SingleAssetBacktest;
