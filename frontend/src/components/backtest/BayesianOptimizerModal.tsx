'use client'
import React, { useState, useMemo } from 'react';
import { X, Play, RefreshCw, Check, Zap, Target, Activity, TrendingUp, AlertCircle, Sparkles } from 'lucide-react';
import { Strategy, StrategyParameter } from "@/types/all_types";
import { formatPercent } from "@/utils/formatters";
import { backtest } from "@/utils/api";

interface BayesianOptimizerModalProps {
    symbols: string[];
    strategy: Strategy;
    onApply: (params: Record<string, any>) => void;
    onClose: () => void;
}

const BayesianOptimizerModal: React.FC<BayesianOptimizerModalProps> = ({ symbols, strategy, onApply, onClose }) => {
    const [isOptimizing, setIsOptimizing] = useState(false);
    const [trials, setTrials] = useState(20);
    const [metric, setMetric] = useState('sharpe_ratio');
    const [paramRanges, setParamRanges] = useState<Record<string, any>>({});
    const [bestValue, setBestValue] = useState<number | null>(null);
    const [bestParams, setBestParams] = useState<Record<string, any> | null>(null);
    const [optimizationHistory, setOptimizationHistory] = useState<any[]>([]);

    // Initialize ranges from strategy parameters
    useMemo(() => {
        const initialRanges: Record<string, any> = {};

        // Try to get parameter definitions (with min/max/type)
        const paramDefinitions = strategy.parameterMetadata || (Array.isArray(strategy.parameters) ? strategy.parameters : []);

        if (paramDefinitions.length > 0) {
            paramDefinitions.forEach((param: any) => {
                if (param.type === 'number' || param.type === 'int' || param.type === 'float') {
                    initialRanges[param.name] = {
                        min: param.min ?? (param.default * 0.5),
                        max: param.max ?? (param.default * 2),
                        type: param.type === 'int' ? 'int' : 'float',
                        step: param.type === 'int' ? 1 : 0.1
                    };
                }
            });
        } else if (typeof strategy.parameters === 'object' && strategy.parameters !== null) {
            // Fallback: If we only have parameter values, create default ranges
            Object.entries(strategy.parameters).forEach(([name, value]) => {
                if (typeof value === 'number') {
                    initialRanges[name] = {
                        min: value * 0.5,
                        max: value * 2,
                        type: Number.isInteger(value) ? 'int' : 'float',
                        step: Number.isInteger(value) ? 1 : 0.1
                    };
                }
            });
        }

        setParamRanges(initialRanges);
    }, [strategy]);

    const handleRunOptimization = async () => {
        setIsOptimizing(true);
        setBestValue(null);
        setBestParams(null);

        try {
            // Ideally call the API here
            // const response = await optimization.bayesian({
            //     ticker: symbol,
            //     strategy_key: strategy.id,
            //     param_ranges: paramRanges,
            //     n_trials: trials,
            //     metric: metric
            // });

            // For now, let's mock the API call if the backend isn't ready
            // or just assume the backend is ready and call it.

            const request = {
                tickers: symbols,
                strategy_key: strategy.id,
                param_ranges: paramRanges,
                n_trials: trials,
                metric: metric
            };

            const result = await backtest.bayesian(request);
            setBestValue(result.best_value);
            setBestParams(result.best_params);
            setOptimizationHistory(result.trials || []);

        } catch (error) {
            console.error("Optimization failed", error);
        } finally {
            setIsOptimizing(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-300">
            <div className="bg-slate-900 border border-slate-800 w-full max-w-4xl max-h-[90vh] rounded-3xl shadow-2xl overflow-hidden flex flex-col">
                {/* Header */}
                <div className="p-6 border-b border-slate-800 flex justify-between items-center bg-violet-600/5">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-violet-500/20 rounded-xl border border-violet-500/30">
                            <Sparkles className="text-violet-400" size={24} />
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-slate-100 italic">Bayesian Strategy Optimizer</h2>
                            <p className="text-xs text-slate-500 font-medium">Fine-tuning {strategy.name} for {symbols.length === 1 ? symbols[0] : `${symbols.length} Assets`}</p>
                        </div>
                    </div>
                    <button onClick={onClose} className="p-2 hover:bg-slate-800 rounded-full transition-colors">
                        <X size={20} className="text-slate-500" />
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto p-6 space-y-8">
                    {/* Settings Section */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="space-y-4 col-span-2">
                            <h3 className="text-sm font-black text-slate-400 uppercase tracking-widest flex items-center gap-2">
                                <Zap size={14} className="text-amber-400" />
                                Parameter Ranges
                            </h3>
                            <div className="grid grid-cols-1 gap-3">
                                {Object.entries(paramRanges).map(([name, range]) => (
                                    <div key={name} className="p-4 bg-slate-800/40 border border-slate-700/50 rounded-2xl flex items-center justify-between">
                                        <span className="text-sm font-bold text-slate-200 uppercase tracking-tighter w-24">{name}</span>
                                        <div className="flex items-center gap-4 flex-1 justify-end">
                                            <div className="flex flex-col">
                                                <span className="text-[10px] text-slate-500 font-black uppercase mb-1">Min</span>
                                                <input
                                                    type="number"
                                                    value={range.min}
                                                    onChange={(e) => setParamRanges({ ...paramRanges, [name]: { ...range, min: parseFloat(e.target.value) } })}
                                                    className="w-20 px-2 py-1 bg-slate-900 border border-slate-700 rounded-lg text-xs font-mono text-slate-300"
                                                />
                                            </div>
                                            <div className="flex flex-col">
                                                <span className="text-[10px] text-slate-500 font-black uppercase mb-1">Max</span>
                                                <input
                                                    type="number"
                                                    value={range.max}
                                                    onChange={(e) => setParamRanges({ ...paramRanges, [name]: { ...range, max: parseFloat(e.target.value) } })}
                                                    className="w-20 px-2 py-1 bg-slate-900 border border-slate-700 rounded-lg text-xs font-mono text-slate-300"
                                                />
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="space-y-6">
                            <div className="space-y-4">
                                <h3 className="text-sm font-black text-slate-400 uppercase tracking-widest">Target Metric</h3>
                                <select
                                    value={metric}
                                    onChange={(e) => setMetric(e.target.value)}
                                    className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-sm font-bold text-slate-200"
                                >
                                    <option value="sharpe_ratio">Maximize Sharpe Ratio</option>
                                    <option value="total_return_pct">Maximize Total Return</option>
                                    <option value="win_rate">Maximize Win Rate</option>
                                    <option value="sortino_ratio">Maximize Sortino Ratio</option>
                                </select>
                            </div>

                            <div className="space-y-4">
                                <h3 className="text-sm font-black text-slate-400 uppercase tracking-widest">Trials Count</h3>
                                <div className="flex items-center gap-4">
                                    <input
                                        type="range"
                                        min="5"
                                        max="50"
                                        step="5"
                                        value={trials}
                                        onChange={(e) => setTrials(parseInt(e.target.value))}
                                        className="flex-1 accent-violet-500"
                                    />
                                    <span className="text-xl font-black text-violet-400">{trials}</span>
                                </div>
                                <p className="text-[10px] text-slate-600 font-bold uppercase italic">Higher trials = better results, longer compute</p>
                            </div>

                            <button
                                onClick={handleRunOptimization}
                                disabled={isOptimizing}
                                className="w-full py-4 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white font-black rounded-2xl shadow-xl shadow-violet-600/20 transition-all flex items-center justify-center gap-3 disabled:opacity-50"
                            >
                                {isOptimizing ? (
                                    <>
                                        <RefreshCw size={20} className="animate-spin" />
                                        <span>OPTIMIZING...</span>
                                    </>
                                ) : (
                                    <>
                                        <Play size={20} fill="currentColor" />
                                        <span>START OPTIMIZATION</span>
                                    </>
                                )}
                            </button>
                        </div>
                    </div>

                    {/* Results Section */}
                    {bestParams && (
                        <div className="space-y-6 animate-in slide-in-from-bottom-4 duration-500">
                            <div className="p-6 bg-emerald-500/10 border border-emerald-500/30 rounded-3xl flex flex-col md:flex-row items-center justify-between gap-6">
                                <div>
                                    <div className="flex items-center gap-2 mb-2">
                                        <div className="w-8 h-8 bg-emerald-500 rounded-full flex items-center justify-center">
                                            <Check size={18} className="text-slate-900" strokeWidth={3} />
                                        </div>
                                        <h3 className="text-xl font-bold text-emerald-400 italic">Optimal Parameters Discovered!</h3>
                                    </div>
                                    <p className="text-sm text-emerald-500/70 font-medium">
                                        Max achieved {metric.replace('_', ' ')}: <span className="text-emerald-300 font-bold">{bestValue?.toFixed(4)}</span>
                                    </p>
                                </div>
                                <div className="flex gap-3">
                                    <button
                                        onClick={() => onApply(bestParams)}
                                        className="px-8 py-3 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-black rounded-xl transition-all shadow-lg shadow-emerald-500/20"
                                    >
                                        APPLY TO STRATEGY
                                    </button>
                                </div>
                            </div>

                            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                                {Object.entries(bestParams).map(([key, val]) => (
                                    <div key={key} className="p-4 bg-slate-800/60 border border-slate-700/50 rounded-2xl text-center">
                                        <p className="text-[10px] text-slate-500 font-black uppercase tracking-widest mb-1">{key}</p>
                                        <p className="text-lg font-bold text-slate-100">{typeof val === 'number' ? val.toFixed(2) : val}</p>
                                    </div>
                                ))}
                            </div>

                            {/* Trials Chart (Optional/Decorative) */}
                            <div className="h-24 bg-slate-900/50 rounded-2xl border border-slate-800 flex items-end p-2 gap-1 overflow-hidden">
                                {optimizationHistory.map((trial, i) => (
                                    <div
                                        key={i}
                                        className="flex-1 bg-violet-600/30 hover:bg-violet-500 transition-all rounded-t-sm"
                                        style={{ height: `${(trial.value / (bestValue || 1)) * 100}%` }}
                                        title={`Trial ${i}: ${trial.value}`}
                                    />
                                ))}
                            </div>
                        </div>
                    )}

                    {isOptimizing && (
                        <div className="flex flex-col items-center justify-center py-20 animate-pulse">
                            <Activity className="text-violet-500 mb-4" size={48} />
                            <p className="text-lg font-black text-slate-400 italic">MOCKING UNIVERSES... COLLAPSING PROBABILITIES...</p>
                            <p className="text-[10px] text-slate-600 font-bold uppercase mt-2 tracking-tighter">Running backtests across the probability cloud</p>
                        </div>
                    )}

                    {!bestParams && !isOptimizing && (
                        <div className="flex flex-col items-center justify-center py-20 text-slate-700">
                            <Target size={48} className="mb-4 opacity-20" />
                            <p className="text-sm font-black uppercase tracking-widest italic">Ready for Alpha Generation</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default BayesianOptimizerModal;
