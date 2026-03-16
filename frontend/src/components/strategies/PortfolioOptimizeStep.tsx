/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useEffect, useState } from 'react';
import {
    BarChart3,
    Check,
    Loader2,
    Scale,
    Shield,
    Target,
    TrendingUp,
    X,
    Zap,
} from 'lucide-react';
import { api } from '@/utils/api';
import { OptimizationResult, OptimizePreviewResponse } from '@/types/all_types';

interface PortfolioOptimizeStepProps {
    symbols: string[];
    onSelect: (weights: Record<string, number>, method: string) => void;
    onSkip: () => void;
    onClose: () => void;
}

const METHOD_META: Record<string, { label: string; description: string; icon: any; accent: string }> = {
    max_sharpe: {
        label: 'Max Sharpe',
        description: 'Maximize risk-adjusted returns',
        icon: TrendingUp,
        accent: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
    },
    min_volatility: {
        label: 'Min Volatility',
        description: 'Minimize portfolio risk',
        icon: Shield,
        accent: 'text-blue-400 bg-blue-500/10 border-blue-500/30',
    },
    risk_parity: {
        label: 'Risk Parity',
        description: 'Equal risk contribution',
        icon: Scale,
        accent: 'text-violet-400 bg-violet-500/10 border-violet-500/30',
    },
    equal_weight: {
        label: 'Equal Weight',
        description: 'Simple 1/N allocation',
        icon: BarChart3,
        accent: 'text-slate-400 bg-slate-500/10 border-slate-500/30',
    },
    black_litterman: {
        label: 'Black-Litterman',
        description: 'Market equilibrium + views',
        icon: Target,
        accent: 'text-amber-400 bg-amber-500/10 border-amber-500/30',
    },
    target_return: {
        label: 'Target Return',
        description: 'Min vol for 15% target',
        icon: Zap,
        accent: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/30',
    },
};

const PortfolioOptimizeStep: React.FC<PortfolioOptimizeStepProps> = ({
    symbols,
    onSelect,
    onSkip,
    onClose,
}) => {
    const [preview, setPreview] = useState<OptimizePreviewResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [selected, setSelected] = useState<string | null>(null);

    useEffect(() => {
        const load = async () => {
            try {
                const data = await api.deployOptimize.preview(symbols);
                setPreview(data);

                // Default-select the best sharpe method
                if (data.methods) {
                    let bestMethod = '';
                    let bestSharpe = -Infinity;
                    for (const [method, result] of Object.entries(data.methods) as [string, OptimizationResult][]) {
                        if (result.sharpe > bestSharpe) {
                            bestSharpe = result.sharpe;
                            bestMethod = method;
                        }
                    }
                    setSelected(bestMethod);
                }
            } catch (err: any) {
                setError(err?.response?.data?.detail || 'Failed to compute optimization preview');
            } finally {
                setLoading(false);
            }
        };
        load();
    }, [symbols]);

    const handleApply = () => {
        if (!selected || !preview) return;
        const result = preview.methods[selected];
        if (result) {
            onSelect(result.weights, selected);
        }
    };

    return (
        <div className="fixed inset-0 z-[90] flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <div className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl max-w-5xl w-full max-h-[90vh] overflow-y-auto relative animate-in zoom-in-95 fade-in duration-200">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-slate-800">
                    <div>
                        <h2 className="text-xl font-bold text-slate-100">
                            Optimize Portfolio Weights
                        </h2>
                        <p className="text-sm text-slate-500 mt-1">
                            Choose an optimization method before deploying to live
                        </p>
                    </div>
                    <button
                        onClick={onClose}
                        className="text-slate-500 hover:text-slate-300 transition-colors"
                    >
                        <X size={20} />
                    </button>
                </div>

                {/* Body */}
                <div className="p-6">
                    {loading && (
                        <div className="flex flex-col items-center justify-center py-16">
                            <Loader2 size={40} className="text-violet-400 animate-spin mb-4" />
                            <p className="text-slate-400">Computing optimization for {symbols.length} assets...</p>
                        </div>
                    )}

                    {error && (
                        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-center">
                            <p className="text-red-400">{error}</p>
                        </div>
                    )}

                    {preview && !loading && (
                        <div className="space-y-6">
                            {/* Method Cards */}
                            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                                {Object.entries(preview.methods).map(([method, result]) => {
                                    const meta = METHOD_META[method] || METHOD_META.equal_weight;
                                    const Icon = meta.icon;
                                    const isSelected = selected === method;

                                    return (
                                        <button
                                            key={method}
                                            onClick={() => setSelected(method)}
                                            className={`text-left p-4 rounded-xl border-2 transition-all ${
                                                isSelected
                                                    ? 'border-violet-500 bg-violet-500/5 ring-2 ring-violet-500/20'
                                                    : 'border-slate-700/60 hover:border-slate-600 bg-slate-800/30'
                                            }`}
                                        >
                                            <div className="flex items-center justify-between mb-3">
                                                <div className="flex items-center gap-2">
                                                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center border ${meta.accent}`}>
                                                        <Icon size={16} />
                                                    </div>
                                                    <div>
                                                        <p className="font-semibold text-slate-100 text-sm">{meta.label}</p>
                                                        <p className="text-[10px] text-slate-500">{meta.description}</p>
                                                    </div>
                                                </div>
                                                {isSelected && (
                                                    <div className="w-6 h-6 rounded-full bg-violet-500 flex items-center justify-center">
                                                        <Check size={14} className="text-white" />
                                                    </div>
                                                )}
                                            </div>

                                            {/* Metrics */}
                                            <div className="grid grid-cols-3 gap-2 text-xs">
                                                <div>
                                                    <p className="text-slate-500">Sharpe</p>
                                                    <p className={`font-bold ${result.sharpe >= 1 ? 'text-emerald-400' : 'text-slate-300'}`}>
                                                        {result.sharpe.toFixed(2)}
                                                    </p>
                                                </div>
                                                <div>
                                                    <p className="text-slate-500">Return</p>
                                                    <p className="text-slate-300 font-bold">{(result.expected_return * 100).toFixed(1)}%</p>
                                                </div>
                                                <div>
                                                    <p className="text-slate-500">Vol</p>
                                                    <p className="text-slate-300 font-bold">{(result.volatility * 100).toFixed(1)}%</p>
                                                </div>
                                            </div>

                                            {/* Weights */}
                                            <div className="mt-3 space-y-1">
                                                {Object.entries(result.weights)
                                                    .sort(([, a], [, b]) => b - a)
                                                    .map(([sym, w]) => (
                                                        <div key={sym} className="flex items-center gap-2 text-xs">
                                                            <span className="text-slate-500 w-12 shrink-0">{sym}</span>
                                                            <div className="flex-1 h-1.5 bg-slate-700/50 rounded-full overflow-hidden">
                                                                <div
                                                                    className="h-full bg-violet-500/60 rounded-full"
                                                                    style={{ width: `${Math.max(w * 100, 1)}%` }}
                                                                />
                                                            </div>
                                                            <span className="text-slate-400 w-10 text-right">
                                                                {(w * 100).toFixed(1)}%
                                                            </span>
                                                        </div>
                                                    ))}
                                            </div>
                                        </button>
                                    );
                                })}
                            </div>
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between p-6 border-t border-slate-800">
                    <button
                        onClick={onSkip}
                        className="text-sm text-slate-500 hover:text-slate-300 transition-colors"
                    >
                        Skip (use equal weights)
                    </button>
                    <div className="flex items-center gap-3">
                        <button
                            onClick={onClose}
                            className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm text-slate-300 transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleApply}
                            disabled={!selected || loading}
                            className="px-6 py-2.5 bg-violet-500 hover:bg-violet-600 disabled:bg-slate-700 disabled:text-slate-500 rounded-lg text-sm font-semibold text-white transition-colors shadow-lg shadow-violet-500/30"
                        >
                            Apply & Continue
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PortfolioOptimizeStep;
