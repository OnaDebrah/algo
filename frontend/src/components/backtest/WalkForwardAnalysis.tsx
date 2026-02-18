import React, { useMemo, useState } from 'react';
import {
    Activity,
    AlertCircle,
    BarChart3,
    Calendar,
    ChevronRight,
    Layers,
    Play,
    Settings,
    ShieldCheck,
    TrendingUp,
    Zap
} from 'lucide-react';
import { ParamRange, Strategy, WFARequest, WFAResponse } from "@/types/all_types";
import { backtest as backtestApi } from "@/utils/api";
import { formatPercent } from "@/utils/formatters";
import SingleBacktestResults from "@/components/backtest/SingleBacktestResults";

interface WalkForwardAnalysisProps {
    strategies: Strategy[];
}

const WalkForwardAnalysis: React.FC<WalkForwardAnalysisProps> = ({ strategies }: WalkForwardAnalysisProps) => {
    const [config, setConfig] = useState<Partial<WFARequest>>({
        symbol: 'AAPL',
        strategy_key: strategies[0]?.id || '',
        period: '2y',
        interval: '1d',
        initial_capital: 100000,
        is_window_days: 180,
        oos_window_days: 60,
        step_days: 60,
        anchored: false,
        metric: 'sharpe_ratio',
        n_trials: 20
    });

    const [results, setResults] = useState<WFAResponse | null>(null);
    const [isRunning, setIsRunning] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const selectedStrategy = useMemo(() =>
        strategies.find(s => s.id === config.strategy_key),
        [config.strategy_key, strategies]
    );

    const handleRunWFA = async () => {
        setIsRunning(true);
        setError(null);
        try {
            // Prepare param ranges based on selected strategy
            const param_ranges: Record<string, ParamRange> = {};

            // If strategy has metadata, use it. Otherwise, infer from current parameters
            const metadata = selectedStrategy?.parameterMetadata ||
                (selectedStrategy?.parameters ? Object.entries(selectedStrategy.parameters).map(([name, value]) => ({
                    name,
                    label: name,
                    type: typeof value === 'number' ? (Number.isInteger(value) ? 'int' : 'float') : 'string',
                    default: value
                })) : []);

            metadata.forEach((p: any) => {
                if (p.type === 'number' || p.type === 'int' || p.type === 'float') {
                    // Avoid 0 or negative mins for windows/periods
                    const isWindow = p.name.toLowerCase().includes('window') ||
                        p.name.toLowerCase().includes('period') ||
                        p.name.toLowerCase().includes('lookback');

                    let min = p.min ?? (p.default * 0.5);
                    if (isWindow && min < 1) min = 1;

                    param_ranges[p.name] = {
                        min: min,
                        max: p.max ?? (p.default * 2.0),
                        step: p.type === 'int' ? 1 : 0.1,
                        type: p.type === 'int' ? 'int' : 'float'
                    };
                }
            });

            const request: WFARequest = {
                ...config as WFARequest,
                param_ranges
            };

            const response = await backtestApi.walkForward(request);
            setResults(response);
        } catch (err: any) {
            setError(err.message || "Failed to run Walk-Forward Analysis");
        } finally {
            setIsRunning(false);
        }
    };

    return (
        <div className="space-y-6">
            {/* Configuration Card */}
            <div className="bg-slate-900/60 border border-slate-700/50 rounded-3xl p-8 backdrop-blur-xl shadow-2xl">
                <div className="flex items-center justify-between mb-8">
                    <div className="flex items-center space-x-4">
                        <div className="p-3 bg-violet-500/10 rounded-2xl">
                            <Layers className="text-violet-400" size={28} />
                        </div>
                        <div>
                            <h3 className="text-2xl font-bold text-white tracking-tight">Walk-Forward Configuration</h3>
                            <p className="text-slate-400 text-sm font-medium">Define IS/OOS windows and optimization parameters</p>
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
                    {/* Basic Settings */}
                    <div className="space-y-6">
                        <div className="space-y-2">
                            <label className="text-slate-400 text-sm font-bold flex items-center space-x-2 px-1">
                                <Activity size={14} />
                                <span>Ticker Symbol</span>
                            </label>
                            <input
                                type="text"
                                value={config.symbol}
                                onChange={e => setConfig({ ...config, symbol: e.target.value.toUpperCase() })}
                                className="w-full bg-slate-800/50 border border-slate-700/50 rounded-2xl px-5 py-4 text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50 transition-all font-bold placeholder-slate-600"
                                placeholder="e.g. BTCUSDT"
                            />
                        </div>

                        <div className="space-y-2">
                            <label className="text-slate-400 text-sm font-bold flex items-center space-x-2 px-1">
                                <Settings size={14} />
                                <span>Strategy</span>
                            </label>
                            <select
                                value={config.strategy_key}
                                onChange={e => setConfig({ ...config, strategy_key: e.target.value })}
                                className="w-full bg-slate-800/50 border border-slate-700/50 rounded-2xl px-5 py-4 text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50 transition-all font-bold bg-no-repeat appearance-none"
                            >
                                {strategies.map(s => (
                                    <option key={s.id} value={s.id}>{s.name}</option>
                                ))}
                            </select>
                        </div>
                    </div>

                    {/* Window Settings */}
                    <div className="space-y-6">
                        <div className="space-y-2">
                            <label className="text-slate-400 text-sm font-bold flex items-center space-x-2 px-1">
                                <Calendar size={14} />
                                <span>In-Sample Window (Days)</span>
                            </label>
                            <input
                                type="number"
                                value={config.is_window_days}
                                onChange={e => setConfig({ ...config, is_window_days: parseInt(e.target.value) })}
                                className="w-full bg-slate-800/50 border border-slate-700/50 rounded-2xl px-5 py-4 text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50 transition-all font-medium"
                            />
                        </div>

                        <div className="space-y-2">
                            <label className="text-slate-400 text-sm font-bold flex items-center space-x-2 px-1">
                                <Zap size={14} />
                                <span>Out-of-Sample Window (Days)</span>
                            </label>
                            <input
                                type="number"
                                value={config.oos_window_days}
                                onChange={e => setConfig({ ...config, oos_window_days: parseInt(e.target.value) })}
                                className="w-full bg-slate-800/50 border border-slate-700/50 rounded-2xl px-5 py-4 text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50 transition-all font-medium"
                            />
                        </div>
                    </div>

                    {/* Advanced Logic */}
                    <div className="space-y-6">
                        <div className="space-y-2">
                            <label className="text-slate-400 text-sm font-bold flex items-center space-x-2 px-1">
                                <TrendingUp size={14} />
                                <span>Step Size (Days)</span>
                            </label>
                            <input
                                type="number"
                                value={config.step_days}
                                onChange={e => setConfig({ ...config, step_days: parseInt(e.target.value) })}
                                className="w-full bg-slate-800/50 border border-slate-700/50 rounded-2xl px-5 py-4 text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50 transition-all font-medium"
                            />
                        </div>

                        <div className="flex items-center justify-between p-4 bg-slate-800/30 border border-slate-700/30 rounded-2xl mt-8">
                            <div className="flex flex-col">
                                <span className="text-white text-sm font-bold">Anchored Window</span>
                                <span className="text-slate-500 text-xs">Expand IS window over time</span>
                            </div>
                            <button
                                onClick={() => setConfig({ ...config, anchored: !config.anchored })}
                                className={`w-12 h-6 rounded-full transition-all flex items-center px-1 ${config.anchored ? 'bg-violet-600' : 'bg-slate-700'}`}
                            >
                                <div className={`w-4 h-4 bg-white rounded-full transition-transform ${config.anchored ? 'translate-x-6' : 'translate-x-0'}`} />
                            </button>
                        </div>
                    </div>

                    {/* Action Block */}
                    <div className="flex flex-col justify-end">
                        {error && (
                            <div className="mb-4 p-4 bg-red-500/10 border border-red-500/20 rounded-2xl flex items-center space-x-3 text-red-400 text-sm animate-pulse">
                                <AlertCircle size={18} />
                                <span>{error}</span>
                            </div>
                        )}
                        <button
                            onClick={handleRunWFA}
                            disabled={isRunning}
                            className={`w-full py-5 rounded-2xl flex items-center justify-center space-x-3 text-lg font-black transition-all transform active:scale-95 ${isRunning
                                ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                                : 'bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white shadow-xl shadow-violet-600/25 hover:shadow-violet-600/40 hover:-translate-y-1'
                                }`}
                        >
                            {isRunning ? <Activity className="animate-spin" size={24} /> : <Play size={24} fill="currentColor" />}
                            <span>{isRunning ? 'Analyzing Folds...' : 'Start Walk-Forward'}</span>
                        </button>
                    </div>
                </div>
            </div>

            {/* Results Section */}
            {results && (
                <div className="animate-in fade-in slide-in-from-bottom-6 duration-700">
                    <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-6">
                        <ResultMetricCard
                            icon={<ShieldCheck className="text-emerald-400" />}
                            label="Walk-Forward Efficiency"
                            value={`${(results.wfe * 100).toFixed(1)}%`}
                            description="Ratio of OOS vs IS performance"
                            highlight={results.wfe > 0.5 ? 'emerald' : 'rose'}
                        />
                        <ResultMetricCard
                            icon={<TrendingUp className="text-blue-400" />}
                            label="OOS Total Return"
                            value={formatPercent(results.aggregated_oos_metrics.total_return_pct)}
                            description="Consolidated Out-of-Sample return"
                        />
                        <ResultMetricCard
                            icon={<Zap className="text-amber-400" />}
                            label="Total Folds"
                            value={results.folds.length.toString()}
                            description="Individual validation steps"
                        />
                        <ResultMetricCard
                            icon={<BarChart3 className="text-fuchsia-400" />}
                            label="OOS Sharpe"
                            value={results.aggregated_oos_metrics.sharpe_ratio.toFixed(2)}
                            description="Consolidated Risk-adjusted return"
                        />
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        {/* Fold List */}
                        <div className="lg:col-span-1 space-y-4">
                            <h4 className="text-white font-bold px-4 flex items-center space-x-2">
                                <List size={18} className="text-slate-400" />
                                <span>Validation Folds</span>
                            </h4>
                            <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2 custom-scrollbar">
                                {results.folds.map((fold, idx) => (
                                    <FoldCard key={idx} fold={fold} />
                                ))}
                            </div>
                        </div>

                        {/* Aggregated Performance Visualization */}
                        <div className="lg:col-span-2 space-y-4">
                            <SingleBacktestResults
                                wfaResponse={results}
                            />
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

// Sub-components
const ResultMetricCard = ({ icon, label, value, description, highlight = 'slate' }: any) => (
    <div className="bg-slate-900/60 border border-slate-700/40 p-6 rounded-3xl backdrop-blur-md">
        <div className="flex items-center space-x-3 mb-4">
            <div className="p-2 bg-slate-800/50 rounded-xl">{icon}</div>
            <span className="text-slate-400 text-sm font-bold uppercase tracking-wider">{label}</span>
        </div>
        <div className={`text-4xl font-black mb-1 ${highlight === 'emerald' ? 'text-emerald-400' :
            highlight === 'rose' ? 'text-rose-400' : 'text-white'
            }`}>
            {value}
        </div>
        <p className="text-slate-500 text-xs font-medium">{description}</p>
    </div>
);

const FoldCard = ({ fold }: { fold: any }) => (
    <div className="bg-slate-900/40 border border-slate-700/30 p-5 rounded-2xl hover:bg-slate-800/50 transition-all group">
        <div className="flex justify-between items-start mb-3">
            <span className="px-3 py-1 bg-violet-500/10 text-violet-400 text-[10px] font-black uppercase rounded-lg">Fold #{fold.fold_index + 1}</span>
            <div className="flex items-center space-x-1 text-[10px] text-slate-500 font-bold uppercase">
                <Calendar size={10} />
                <span>OOS: {new Date(fold.oos_start).toLocaleDateString()}</span>
            </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
            <div>
                <span className="block text-[10px] text-slate-500 font-bold uppercase mb-1">IS Return</span>
                <span className={`text-sm font-black ${fold.is_metrics.total_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {formatPercent(fold.is_metrics.total_return_pct)}
                </span>
            </div>
            <div>
                <span className="block text-[10px] text-slate-500 font-bold uppercase mb-1">OOS Return</span>
                <span className={`text-sm font-black ${fold.oos_metrics.total_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {formatPercent(fold.oos_metrics.total_return_pct)}
                </span>
            </div>
        </div>

        <div className="mt-4 pt-4 border-t border-slate-700/50 flex justify-between items-center opacity-0 group-hover:opacity-100 transition-opacity">
            <span className="text-[10px] text-slate-400 font-medium">View detailed parameters</span>
            <ChevronRight size={14} className="text-slate-500" />
        </div>
    </div>
);

const List = ({ ...props }) => (
    <svg
        {...props}
        xmlns="http://www.w3.org/2000/svg"
        width="24" height="24" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" strokeWidth="2"
        strokeLinecap="round" strokeLinejoin="round"
    >
        <line x1="8" y1="6" x2="21" y2="6"></line>
        <line x1="8" y1="12" x2="21" y2="12"></line>
        <line x1="8" y1="18" x2="21" y2="18"></line>
        <line x1="3" y1="6" x2="3.01" y2="6"></line>
        <line x1="3" y1="12" x2="3.01" y2="12"></line>
        <line x1="3" y1="18" x2="3.01" y2="18"></line>
    </svg>
);

export default WalkForwardAnalysis;
