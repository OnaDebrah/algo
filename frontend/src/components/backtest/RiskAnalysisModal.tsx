import React, { useMemo } from 'react';
import { BacktestResult, Trade } from "@/types/all_types";
import { formatCurrency, formatPercent } from "@/utils/formatters";
import { X, AlertTriangle, TrendingDown, Activity, ArrowUp, ArrowDown } from 'lucide-react';

interface RiskAnalysisModalProps {
    results: BacktestResult;
    onClose: () => void;
}

const RiskAnalysisModal: React.FC<RiskAnalysisModalProps> = ({ results, onClose }) => {
    // Calculate Advanced Metrics
    const metrics = useMemo(() => {
        if (!results || !results.trades) return null;

        const trades = results.trades;
        const profits = trades.map(t => t.profit || 0).filter(p => p !== 0);

        // 1. Sortino Ratio (needs downside deviation)
        // Assuming 0 as target return for downside deviation
        const negativeReturns = profits.filter(p => p < 0);
        const downsideDeviation = Math.sqrt(
            negativeReturns.reduce((acc, val) => acc + Math.pow(val, 2), 0) / (trades.length || 1)
        );
        const sortinoRatio = downsideDeviation !== 0 ? results.avg_profit / downsideDeviation : 0;

        // 2. Calmar Ratio (Annualized Return / Max Drawdown)
        // Approximate annualized return based on simple total return for now
        // Ideally we'd know the exact time period duration
        const calmarRatio = results.max_drawdown !== 0 ? results.total_return / Math.abs(results.max_drawdown) : 0;

        // 3. Consecutive Wins/Losses
        let maxConsecutiveWins = 0;
        let maxConsecutiveLosses = 0;
        let currentWins = 0;
        let currentLosses = 0;

        trades.forEach(t => {
            if ((t.profit || 0) > 0) {
                currentWins++;
                currentLosses = 0;
                maxConsecutiveWins = Math.max(maxConsecutiveWins, currentWins);
            } else if ((t.profit || 0) < 0) {
                currentLosses++;
                currentWins = 0;
                maxConsecutiveLosses = Math.max(maxConsecutiveLosses, currentLosses);
            }
        });

        // 4. Best/Worst Trade
        const bestTrade = Math.max(...profits);
        const worstTrade = Math.min(...profits);

        // 5. Value at Risk (VaR) - 95% Confidence
        // Sort returns ascending
        const sortedProfits = [...profits].sort((a, b) => a - b);
        const varIndex = Math.floor(sortedProfits.length * 0.05);
        const valueAtRisk = sortedProfits[varIndex] || 0;

        return {
            sortinoRatio,
            calmarRatio,
            maxConsecutiveWins,
            maxConsecutiveLosses,
            bestTrade,
            worstTrade,
            valueAtRisk,
            downsideDeviation
        };
    }, [results]);

    if (!metrics) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="bg-slate-900 border border-slate-700/50 rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
                <div className="flex items-center justify-between p-6 border-b border-slate-800">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-blue-500/20 rounded-lg border border-blue-500/30">
                            <AlertTriangle size={20} className="text-blue-400" />
                        </div>
                        <div>
                            <h3 className="text-lg font-bold text-slate-100">Risk Analysis</h3>
                            <p className="text-xs text-slate-500">Advanced risk metrics & distribution</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-slate-800 rounded-lg text-slate-500 hover:text-slate-300 transition-colors"
                    >
                        <X size={20} />
                    </button>
                </div>

                <div className="p-6 space-y-6">
                    {/* Primary Risk Ratios */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                            <p className="text-xs text-slate-500 font-bold uppercase mb-1">Sortino Ratio</p>
                            <p className={`text-xl font-bold ${metrics.sortinoRatio > 1 ? 'text-emerald-400' : 'text-slate-200'}`}>
                                {metrics.sortinoRatio.toFixed(2)}
                            </p>
                            <p className="text-[10px] text-slate-500 mt-1">Return / Downside Dev</p>
                        </div>
                        <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                            <p className="text-xs text-slate-500 font-bold uppercase mb-1">Calmar Ratio</p>
                            <p className={`text-xl font-bold ${metrics.calmarRatio > 1 ? 'text-emerald-400' : 'text-slate-200'}`}>
                                {metrics.calmarRatio.toFixed(2)}
                            </p>
                            <p className="text-[10px] text-slate-500 mt-1">Return / Max Drawdown</p>
                        </div>
                        <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                            <p className="text-xs text-slate-500 font-bold uppercase mb-1">Profit Factor</p>
                            <p className={`text-xl font-bold ${results.profit_factor > 1.5 ? 'text-emerald-400' : results.profit_factor > 1 ? 'text-blue-400' : 'text-red-400'}`}>
                                {results.profit_factor.toFixed(2)}
                            </p>
                            <p className="text-[10px] text-slate-500 mt-1">Gross Profit / Gross Loss</p>
                        </div>
                        <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                            <p className="text-xs text-slate-500 font-bold uppercase mb-1">Value at Risk (95%)</p>
                            <p className="text-xl font-bold text-red-400">
                                {formatCurrency(metrics.valueAtRisk)}
                            </p>
                            <p className="text-[10px] text-slate-500 mt-1">Worst 5% Trade Exp.</p>
                        </div>
                    </div>

                    {/* Streak Analysis */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="p-5 bg-gradient-to-br from-slate-800/40 to-slate-900/40 rounded-xl border border-slate-700/50">
                            <div className="flex items-center gap-2 mb-4">
                                <Activity size={16} className="text-violet-400" />
                                <h4 className="text-sm font-bold text-slate-300">Streak Analysis</h4>
                            </div>
                            <div className="flex justify-between items-center mb-3">
                                <span className="text-xs text-slate-500">Max Consecutive Wins</span>
                                <span className="text-sm font-bold text-emerald-400 flex items-center gap-1">
                                    <ArrowUp size={12} /> {metrics.maxConsecutiveWins}
                                </span>
                            </div>
                            <div className="w-full bg-slate-700/30 h-1.5 rounded-full mb-4">
                                <div
                                    className="bg-emerald-500 h-1.5 rounded-full"
                                    style={{ width: `${Math.min((metrics.maxConsecutiveWins / 10) * 100, 100)}%` }}
                                />
                            </div>

                            <div className="flex justify-between items-center mb-3">
                                <span className="text-xs text-slate-500">Max Consecutive Losses</span>
                                <span className="text-sm font-bold text-red-400 flex items-center gap-1">
                                    <ArrowDown size={12} /> {metrics.maxConsecutiveLosses}
                                </span>
                            </div>
                            <div className="w-full bg-slate-700/30 h-1.5 rounded-full">
                                <div
                                    className="bg-red-500 h-1.5 rounded-full"
                                    style={{ width: `${Math.min((metrics.maxConsecutiveLosses / 10) * 100, 100)}%` }}
                                />
                            </div>
                        </div>

                        {/* Extreme Trades */}
                        <div className="p-5 bg-gradient-to-br from-slate-800/40 to-slate-900/40 rounded-xl border border-slate-700/50">
                            <div className="flex items-center gap-2 mb-4">
                                <TrendingDown size={16} className="text-amber-400" />
                                <h4 className="text-sm font-bold text-slate-300">Extreme Events</h4>
                            </div>

                            <div className="space-y-4">
                                <div className="flex justify-between items-center bg-white/[0.02] p-2.5 rounded-lg border border-white/5">
                                    <span className="text-xs text-slate-500">Best Trade</span>
                                    <span className="text-sm font-mono font-bold text-emerald-400">
                                        +{formatCurrency(metrics.bestTrade)}
                                    </span>
                                </div>
                                <div className="flex justify-between items-center bg-white/[0.02] p-2.5 rounded-lg border border-white/5">
                                    <span className="text-xs text-slate-500">Worst Trade</span>
                                    <span className="text-sm font-mono font-bold text-red-400">
                                        {formatCurrency(metrics.worstTrade)}
                                    </span>
                                </div>
                                <div className="flex justify-between items-center bg-white/[0.02] p-2.5 rounded-lg border border-white/5">
                                    <span className="text-xs text-slate-500">Avg Trade</span>
                                    <span className={`text-sm font-mono font-bold ${results.avg_profit >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                        {results.avg_profit >= 0 ? '+' : ''}{formatCurrency(results.avg_profit)}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="p-6 border-t border-slate-800 bg-slate-900/50 flex justify-end">
                    <button
                        onClick={onClose}
                        className="px-6 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-bold rounded-xl transition-colors"
                    >
                        Close Analysis
                    </button>
                </div>
            </div>
        </div>
    );
};

export default RiskAnalysisModal;
