'use client'
import React from 'react';
import {Activity, ArrowDownRight, ArrowUpRight, Target, TrendingUp} from 'lucide-react';
import {formatPercent} from "@/utils/formatters";
import {BenchmarkInfo} from "@/types/all_types";

interface BenchmarkComparisonProps {
    benchmark: BenchmarkInfo;
}

const BenchmarkComparison: React.FC<BenchmarkComparisonProps> = ({ benchmark }: BenchmarkComparisonProps) => {
    if (!benchmark || !benchmark.comparison) {
        return null;
    }

    const { comparison } = benchmark;
    const outperforming = comparison.outperformance > 0;

    return (
        <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h3 className="text-xl font-semibold text-slate-100">Benchmark Comparison</h3>
                    <p className="text-xs text-slate-500 mt-1">
                        Performance vs {benchmark.symbol || 'Portfolio'} Buy & Hold
                    </p>
                </div>
                <div className={`flex items-center gap-2 px-4 py-2 rounded-xl ${
                    outperforming
                        ? 'bg-emerald-500/10 border border-emerald-500/30'
                        : 'bg-red-500/10 border border-red-500/30'
                }`}>
                    {outperforming ? (
                        <ArrowUpRight className="text-emerald-400" size={20} strokeWidth={2.5} />
                    ) : (
                        <ArrowDownRight className="text-red-400" size={20} strokeWidth={2.5} />
                    )}
                    <div>
                        <p className={`text-xs font-bold ${outperforming ? 'text-emerald-400' : 'text-red-400'}`}>
                            {outperforming ? 'Outperforming' : 'Underperforming'}
                        </p>
                        <p className={`text-lg font-bold ${outperforming ? 'text-emerald-400' : 'text-red-400'}`}>
                            {outperforming ? '+' : ''}{comparison.outperformance.toFixed(2)}%
                        </p>
                    </div>
                </div>
            </div>

            {/* Comparison Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {/* Strategy Return */}
                <div className="bg-slate-800/40 p-4 rounded-xl border border-slate-700/50">
                    <div className="flex items-center gap-2 mb-2">
                        <TrendingUp size={16} className="text-violet-400" />
                        <p className="text-xs font-bold text-slate-500 uppercase">Strategy</p>
                    </div>
                    <p className="text-2xl font-bold text-slate-100">
                        {formatPercent(comparison.strategy_return)}
                    </p>
                    <p className="text-xs text-slate-500 mt-1">Total Return</p>
                </div>

                {/* Benchmark Return */}
                <div className="bg-slate-800/40 p-4 rounded-xl border border-slate-700/50">
                    <div className="flex items-center gap-2 mb-2">
                        <Target size={16} className="text-blue-400" />
                        <p className="text-xs font-bold text-slate-500 uppercase">Benchmark</p>
                    </div>
                    <p className="text-2xl font-bold text-slate-100">
                        {formatPercent(comparison.benchmark_return)}
                    </p>
                    <p className="text-xs text-slate-500 mt-1">Total Return</p>
                </div>

                {/* Alpha */}
                <div className="bg-slate-800/40 p-4 rounded-xl border border-slate-700/50">
                    <div className="flex items-center gap-2 mb-2">
                        <Activity size={16} className="text-emerald-400" />
                        <p className="text-xs font-bold text-slate-500 uppercase">Alpha</p>
                    </div>
                    <p className={`text-2xl font-bold ${comparison.alpha >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {comparison.alpha >= 0 ? '+' : ''}{comparison.alpha.toFixed(2)}%
                    </p>
                    <p className="text-xs text-slate-500 mt-1">Excess Return</p>
                </div>

                {/* Sharpe Difference */}
                <div className="bg-slate-800/40 p-4 rounded-xl border border-slate-700/50">
                    <div className="flex items-center gap-2 mb-2">
                        <Activity size={16} className="text-violet-400" />
                        <p className="text-xs font-bold text-slate-500 uppercase">Sharpe Δ</p>
                    </div>
                    <p className={`text-2xl font-bold ${comparison.sharpe_ratio_diff >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {comparison.sharpe_ratio_diff >= 0 ? '+' : ''}{comparison.sharpe_ratio_diff.toFixed(3)}
                    </p>
                    <p className="text-xs text-slate-500 mt-1">Risk-Adj. Diff</p>
                </div>
            </div>

            {/* Detailed Comparison Table */}
            <div className="mt-6 overflow-hidden rounded-xl border border-slate-700/50">
                <table className="w-full">
                    <thead className="bg-slate-800/60">
                        <tr className="text-xs font-bold text-slate-400 uppercase">
                            <th className="px-4 py-3 text-left">Metric</th>
                            <th className="px-4 py-3 text-right">Strategy</th>
                            <th className="px-4 py-3 text-right">Benchmark</th>
                            <th className="px-4 py-3 text-right">Difference</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-700/30">
                        <tr className="hover:bg-slate-800/20 transition-colors">
                            <td className="px-4 py-3 text-sm text-slate-300">Total Return</td>
                            <td className="px-4 py-3 text-right font-mono text-sm text-slate-200">
                                {formatPercent(comparison.strategy_return)}
                            </td>
                            <td className="px-4 py-3 text-right font-mono text-sm text-slate-200">
                                {formatPercent(comparison.benchmark_return)}
                            </td>
                            <td className={`px-4 py-3 text-right font-mono text-sm font-bold ${
                                comparison.outperformance >= 0 ? 'text-emerald-400' : 'text-red-400'
                            }`}>
                                {comparison.outperformance >= 0 ? '+' : ''}{comparison.outperformance.toFixed(2)}%
                            </td>
                        </tr>
                        <tr className="hover:bg-slate-800/20 transition-colors">
                            <td className="px-4 py-3 text-sm text-slate-300">Sharpe Ratio</td>
                            <td className="px-4 py-3 text-right font-mono text-sm text-slate-200">
                                {comparison.strategy_sharpe.toFixed(3)}
                            </td>
                            <td className="px-4 py-3 text-right font-mono text-sm text-slate-200">
                                {comparison.benchmark_sharpe.toFixed(3)}
                            </td>
                            <td className={`px-4 py-3 text-right font-mono text-sm font-bold ${
                                comparison.sharpe_ratio_diff >= 0 ? 'text-emerald-400' : 'text-red-400'
                            }`}>
                                {comparison.sharpe_ratio_diff >= 0 ? '+' : ''}{comparison.sharpe_ratio_diff.toFixed(3)}
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>

            {/* Info Note */}
            <div className="mt-4 p-3 bg-blue-500/5 border border-blue-500/20 rounded-lg">
                <p className="text-xs text-blue-400">
                    <span className="font-bold">ℹ️ Note:</span> Benchmark represents a simple buy-and-hold strategy
                    with the same initial capital and time period. Alpha shows the excess return generated by your strategy.
                </p>
            </div>
        </div>
    );
};

export default BenchmarkComparison;
