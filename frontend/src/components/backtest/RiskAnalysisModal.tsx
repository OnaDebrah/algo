import React, {useMemo} from 'react';
import {BacktestResult} from "@/types/all_types";
import {formatCurrency, formatPercent} from "@/utils/formatters";
import {Activity, AlertTriangle, BarChart3, Gauge, PieChart, Target, TrendingDown, X} from 'lucide-react';

interface RiskAnalysisModalProps {
    results: BacktestResult;
    onClose: () => void;
}

const RiskAnalysisModal: React.FC<RiskAnalysisModalProps> = ({results, onClose}) => {

    const metrics = useMemo(() => {
        if (!results || !results.trades) return null;

        const trades = results.trades;
        const result = results;

        const closedTrades = trades.filter(t => t.profit !== null && t.profit !== undefined);
        const profits = closedTrades.map(t => t.profit || 0);

        const profitFactor = result.profit_factor;

        // Gain to Pain Ratio
        const totalProfit = profits.reduce((sum, p) => sum + Math.max(p, 0), 0);
        const totalLoss = Math.abs(profits.reduce((sum, p) => sum + Math.min(p, 0), 0));
        const gainToPainRatio = totalLoss > 0 ? totalProfit / totalLoss : totalProfit > 0 ? Infinity : 0;

        // Risk-Reward Ratio
        const avgWin = result.avg_win || 0;
        const avgLoss = Math.abs(result.avg_loss || 0);
        const riskRewardRatio = avgLoss > 0 ? avgWin / avgLoss : 0;

        // Ulcer Index (drawdown severity)
        const equityCurve = results.equity_curve || [];
        let sumSquaredDrawdown = 0;
        equityCurve.forEach(point => {
            if ((point.drawdown || 0) < 0) {
                sumSquaredDrawdown += Math.pow((point.drawdown || 0), 2);
            }
        });
        const ulcerIndex = Math.sqrt(sumSquaredDrawdown / (equityCurve.length || 1)) * 100;

        // Recovery Factor
        const totalReturn = result.total_return || 0;
        const maxDrawdown = Math.abs(result.max_drawdown || 0);
        const recoveryFactor = maxDrawdown > 0 ? totalReturn / maxDrawdown : totalReturn > 0 ? Infinity : 0;

        // Tail Ratio (95th percentile / 5th percentile)
        if (profits.length > 0) {
            const sortedProfits = [...profits].sort((a, b) => a - b);
            const percentile5 = sortedProfits[Math.floor(sortedProfits.length * 0.05)] || 0;
            const percentile95 = sortedProfits[Math.floor(sortedProfits.length * 0.95)] || 0;
            const tailRatio = percentile5 !== 0 ? Math.abs(percentile95 / percentile5) : 0;

            const n = profits.length;
            const mean = profits.reduce((a, b) => a + b, 0) / n;

            // Skewness (asymmetry)
            const skewNumerator = profits.reduce((sum, p) => sum + Math.pow(p - mean, 3), 0);
            const skewDenominator = Math.pow(profits.reduce((sum, p) => sum + Math.pow(p - mean, 2), 0) / n, 1.5);
            const skewness = skewDenominator !== 0 ? skewNumerator / (n * skewDenominator) : 0;

            // Kurtosis (tail heaviness)
            const kurtNumerator = profits.reduce((sum, p) => sum + Math.pow(p - mean, 4), 0);
            const kurtDenominator = Math.pow(profits.reduce((sum, p) => sum + Math.pow(p - mean, 2), 0) / n, 2);
            const kurtosis = kurtDenominator !== 0 ? kurtNumerator / (n * kurtDenominator) - 3 : 0; // Excess kurtosis

            const winCount = profits.filter(p => p > 0).length;
            const lossCount = profits.filter(p => p < 0).length;
            const winLossRatio = lossCount > 0 ? winCount / lossCount : winCount > 0 ? Infinity : 0;

            const maxAdverseExcursion = Math.min(...profits);

            const maxFavorableExcursion = Math.max(...profits);

            let tradeFrequency = 0;
            if (equityCurve.length > 1) {
                const firstDate = new Date(equityCurve[0].timestamp);
                const lastDate = new Date(equityCurve[equityCurve.length - 1].timestamp);
                const totalDays = Math.ceil((lastDate.getTime() - firstDate.getTime()) / (1000 * 60 * 60 * 24));

                const parseDate = (timestamp: string): Date => {
                    try {
                        // Handle format like "2025-05-07T00:00:00-04:00"
                        return new Date(timestamp);
                    } catch {
                        return new Date();
                    }
                };

                const positionDays = (() => {
                    const openPositions: Map<string, Date> = new Map();
                    let totalDays = 0;

                    const sortedTrades = [...closedTrades].sort((a, b) =>
                        parseDate(a.executed_at).getTime() - parseDate(b.executed_at).getTime()
                    );

                    sortedTrades.forEach(trade => {
                        const tradeDate = parseDate(trade.executed_at);
                        const symbol = trade.symbol;

                        if (trade.order_type === 'BUY') {
                            openPositions.set(symbol, tradeDate);
                        } else if (trade.order_type === 'SELL' && openPositions.has(symbol)) {
                            const entryDate = openPositions.get(symbol)!;
                            const daysInPosition = Math.ceil(
                                (tradeDate.getTime() - entryDate.getTime()) / (1000 * 60 * 60 * 24)
                            );

                            totalDays += Math.max(1, daysInPosition); // At least 1 day
                            openPositions.delete(symbol);
                        }
                    });

                    // Handle any positions still open at the end
                    if (openPositions.size > 0 && equityCurve.length > 0) {
                        const lastDate = parseDate(equityCurve[equityCurve.length - 1].timestamp);

                        openPositions.forEach((entryDate, symbol) => {
                            const daysInPosition = Math.ceil(
                                (lastDate.getTime() - entryDate.getTime()) / (1000 * 60 * 60 * 24)
                            );
                            totalDays += Math.max(1, daysInPosition);
                        });
                    }

                    return totalDays;
                })();

                tradeFrequency = totalDays > 0 ? (positionDays / totalDays) * 100 : 0;
            }
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

            return {
                sortinoRatio: result.sortino_ratio,
                calmarRatio: result.calmar_ratio,
                profitFactor,
                valueAtRisk: result.var_95,
                maxConsecutiveWins: maxConsecutiveWins,
                maxConsecutiveLosses: maxConsecutiveLosses,
                bestTrade: Math.max(...profits),
                worstTrade: Math.min(...profits),
                gainToPainRatio,
                riskRewardRatio,
                ulcerIndex,
                recoveryFactor,
                tailRatio,
                skewness,
                kurtosis,
                winLossRatio,
                avgWin: result.avg_win,
                avgLoss: result.avg_loss,
                maxAdverseExcursion,
                maxFavorableExcursion,
                tradeFrequency: tradeFrequency,
                cvar: result.cvar_95,
                expectancy: result.expectancy,
                volatility: result.volatility,
                alpha: result.alpha,
                beta: result.beta,
                rSquared: result.r_squared
            };
        }

        return null;
    }, [results]);

    if (!metrics) return null;

    return (
        <div
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200 overflow-y-auto">
            <div
                className="bg-slate-900 border border-slate-700/50 rounded-2xl shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-y-auto">
                <div
                    className="flex items-center justify-between p-6 border-b border-slate-800 sticky top-0 bg-slate-900 z-10">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-blue-500/20 rounded-lg border border-blue-500/30">
                            <AlertTriangle size={20} className="text-blue-400"/>
                        </div>
                        <div>
                            <h3 className="text-lg font-bold text-slate-100">Advanced Risk Analysis</h3>
                            <p className="text-xs text-slate-500">Comprehensive risk metrics & statistical analysis</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-slate-800 rounded-lg text-slate-500 hover:text-slate-300 transition-colors"
                    >
                        <X size={20}/>
                    </button>
                </div>

                <div className="p-6 space-y-6">
                    {/* Primary Risk Ratios */}
                    <div>
                        <h4 className="text-sm font-bold text-slate-400 mb-3 flex items-center gap-2">
                            <Gauge size={16} className="text-blue-400"/>
                            Risk-Adjusted Return Metrics
                        </h4>
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
                                <p className="text-[10px] text-slate-500 mt-1">Return / Max DD</p>
                            </div>
                            <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                                <p className="text-xs text-slate-500 font-bold uppercase mb-1">Recovery Factor</p>
                                <p className={`text-xl font-bold ${metrics.recoveryFactor > 2 ? 'text-emerald-400' : 'text-slate-200'}`}>
                                    {metrics.recoveryFactor === Infinity ? '∞' : metrics.recoveryFactor.toFixed(2)}
                                </p>
                                <p className="text-[10px] text-slate-500 mt-1">Return / Max DD</p>
                            </div>
                            <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                                <p className="text-xs text-slate-500 font-bold uppercase mb-1">Gain/Pain Ratio</p>
                                <p className={`text-xl font-bold ${metrics.gainToPainRatio > 1 ? 'text-emerald-400' : 'text-slate-200'}`}>
                                    {metrics.gainToPainRatio === Infinity ? '∞' : metrics.gainToPainRatio.toFixed(2)}
                                </p>
                                <p className="text-[10px] text-slate-500 mt-1">Total Profit / Total Loss</p>
                            </div>
                        </div>
                    </div>

                    {/* Trade Efficiency Metrics */}
                    <div>
                        <h4 className="text-sm font-bold text-slate-400 mb-3 flex items-center gap-2">
                            <Target size={16} className="text-emerald-400"/>
                            Trade Efficiency
                        </h4>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                                <p className="text-xs text-slate-500 font-bold uppercase mb-1">Profit Factor</p>
                                <p className={`text-xl font-bold ${
                                    metrics.profitFactor > 1.5 ? 'text-emerald-400' :
                                        metrics.profitFactor > 1 ? 'text-blue-400' : 'text-red-400'
                                }`}>
                                    {metrics.profitFactor.toFixed(2)}
                                </p>
                                <p className="text-[10px] text-slate-500 mt-1">Gross Profit/Loss</p>
                            </div>
                            <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                                <p className="text-xs text-slate-500 font-bold uppercase mb-1">Risk/Reward Ratio</p>
                                <p className={`text-xl font-bold ${metrics.riskRewardRatio > 1.5 ? 'text-emerald-400' : 'text-slate-200'}`}>
                                    {metrics.riskRewardRatio.toFixed(2)}
                                </p>
                                <p className="text-[10px] text-slate-500 mt-1">Avg Win / Avg Loss</p>
                            </div>
                            <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                                <p className="text-xs text-slate-500 font-bold uppercase mb-1">Win/Loss Ratio</p>
                                <p className={`text-xl font-bold ${metrics.winLossRatio > 1 ? 'text-emerald-400' : 'text-slate-200'}`}>
                                    {metrics.winLossRatio === Infinity ? '∞' : metrics.winLossRatio.toFixed(2)}
                                </p>
                                <p className="text-[10px] text-slate-500 mt-1">Win Count / Loss Count</p>
                            </div>
                            <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                                <p className="text-xs text-slate-500 font-bold uppercase mb-1">Expectancy</p>
                                <p className={`text-xl font-bold ${metrics.expectancy > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                    {formatCurrency(metrics.expectancy)}
                                </p>
                                <p className="text-[10px] text-slate-500 mt-1">Avg Profit per Trade</p>
                            </div>
                        </div>
                    </div>

                    {/* Drawdown & Risk Metrics */}
                    <div>
                        <h4 className="text-sm font-bold text-slate-400 mb-3 flex items-center gap-2">
                            <TrendingDown size={16} className="text-red-400"/>
                            Drawdown & Risk Metrics
                        </h4>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                                <p className="text-xs text-slate-500 font-bold uppercase mb-1">Ulcer Index</p>
                                <p className="text-xl font-bold text-slate-200">
                                    {metrics.ulcerIndex.toFixed(2)}%
                                </p>
                                <p className="text-[10px] text-slate-500 mt-1">Drawdown Severity</p>
                            </div>
                            <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                                <p className="text-xs text-slate-500 font-bold uppercase mb-1">VaR (95%)</p>
                                <p className="text-xl font-bold text-red-400">
                                    {formatCurrency(metrics.valueAtRisk)}
                                </p>
                                <p className="text-[10px] text-slate-500 mt-1">Worst 5% Trade</p>
                            </div>
                            <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                                <p className="text-xs text-slate-500 font-bold uppercase mb-1">CVaR (95%)</p>
                                <p className="text-xl font-bold text-red-400">
                                    {formatCurrency(metrics.cvar)}
                                </p>
                                <p className="text-[10px] text-slate-500 mt-1">Expected Shortfall</p>
                            </div>
                            <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                                <p className="text-xs text-slate-500 font-bold uppercase mb-1">Volatility</p>
                                <p className="text-xl font-bold text-slate-200">
                                    {formatPercent(metrics.volatility * 100)}
                                </p>
                                <p className="text-[10px] text-slate-500 mt-1">Annualized</p>
                            </div>
                        </div>
                    </div>

                    {/* Statistical Distribution */}
                    <div>
                        <h4 className="text-sm font-bold text-slate-400 mb-3 flex items-center gap-2">
                            <BarChart3 size={16} className="text-violet-400"/>
                            Return Distribution
                        </h4>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                                <p className="text-xs text-slate-500 font-bold uppercase mb-1">Skewness</p>
                                <p className={`text-xl font-bold ${
                                    metrics.skewness > 0 ? 'text-emerald-400' :
                                        metrics.skewness < 0 ? 'text-red-400' : 'text-slate-200'
                                }`}>
                                    {metrics.skewness.toFixed(2)}
                                </p>
                                <p className="text-[10px] text-slate-500 mt-1">Asymmetry</p>
                            </div>
                            <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                                <p className="text-xs text-slate-500 font-bold uppercase mb-1">Kurtosis</p>
                                <p className={`text-xl font-bold ${
                                    metrics.kurtosis > 0 ? 'text-yellow-400' : 'text-slate-200'
                                }`}>
                                    {metrics.kurtosis.toFixed(2)}
                                </p>
                                <p className="text-[10px] text-slate-500 mt-1">Tail Risk</p>
                            </div>
                            <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                                <p className="text-xs text-slate-500 font-bold uppercase mb-1">Tail Ratio</p>
                                <p className="text-xl font-bold text-slate-200">
                                    {metrics.tailRatio.toFixed(2)}
                                </p>
                                <p className="text-[10px] text-slate-500 mt-1">95% / 5%</p>
                            </div>
                            <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                                <p className="text-xs text-slate-500 font-bold uppercase mb-1">Trade Frequency</p>
                                <p className="text-xl font-bold text-slate-200">
                                    {metrics.tradeFrequency.toFixed(1)}%
                                </p>
                                <p className="text-[10px] text-slate-500 mt-1">Exposure Time</p>
                            </div>
                        </div>
                    </div>

                    {/* Market Correlation */}
                    {metrics.alpha !== undefined && (
                        <div>
                            <h4 className="text-sm font-bold text-slate-400 mb-3 flex items-center gap-2">
                                <PieChart size={16} className="text-indigo-400"/>
                                Market Correlation
                            </h4>
                            <div className="grid grid-cols-3 gap-4">
                                <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                                    <p className="text-xs text-slate-500 font-bold uppercase mb-1">Alpha</p>
                                    <p className={`text-xl font-bold ${metrics.alpha > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                        {metrics.alpha.toFixed(2)}%
                                    </p>
                                    <p className="text-[10px] text-slate-500 mt-1">Excess Return</p>
                                </div>
                                <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                                    <p className="text-xs text-slate-500 font-bold uppercase mb-1">Beta</p>
                                    <p className={`text-xl font-bold ${
                                        (metrics.beta || 0) < 1 ? 'text-emerald-400' : 'text-yellow-400'
                                    }`}>
                                        {(metrics.beta || 0).toFixed(2)}
                                    </p>
                                    <p className="text-[10px] text-slate-500 mt-1">Market Sensitivity</p>
                                </div>
                                <div className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50">
                                    <p className="text-xs text-slate-500 font-bold uppercase mb-1">R-Squared</p>
                                    <p className="text-xl font-bold text-slate-200">
                                        {((metrics.rSquared || 0) * 100).toFixed(1)}%
                                    </p>
                                    <p className="text-[10px] text-slate-500 mt-1">Market Correlation</p>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Extreme Trades & Streaks */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div
                            className="p-5 bg-gradient-to-br from-slate-800/40 to-slate-900/40 rounded-xl border border-slate-700/50">
                            <div className="flex items-center gap-2 mb-4">
                                <Activity size={16} className="text-violet-400"/>
                                <h4 className="text-sm font-bold text-slate-300">Streak Analysis</h4>
                            </div>

                            <div className="space-y-4">
                                <div>
                                    <div className="flex justify-between items-center mb-2">
                                        <span className="text-xs text-slate-500">Max Consecutive Wins</span>
                                        <span className="text-sm font-bold text-emerald-400">
                                            {metrics.maxConsecutiveWins}
                                        </span>
                                    </div>
                                    <div className="w-full bg-slate-700/30 h-2 rounded-full">
                                        <div
                                            className="bg-emerald-500 h-2 rounded-full"
                                            style={{width: `${Math.min((metrics.maxConsecutiveWins / 10) * 100, 100)}%`}}
                                        />
                                    </div>
                                </div>

                                <div>
                                    <div className="flex justify-between items-center mb-2">
                                        <span className="text-xs text-slate-500">Max Consecutive Losses</span>
                                        <span className="text-sm font-bold text-red-400">
                                            {metrics.maxConsecutiveLosses}
                                        </span>
                                    </div>
                                    <div className="w-full bg-slate-700/30 h-2 rounded-full">
                                        <div
                                            className="bg-red-500 h-2 rounded-full"
                                            style={{width: `${Math.min((metrics.maxConsecutiveLosses / 10) * 100, 100)}%`}}
                                        />
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div
                            className="p-5 bg-gradient-to-br from-slate-800/40 to-slate-900/40 rounded-xl border border-slate-700/50">
                            <div className="flex items-center gap-2 mb-4">
                                <TrendingDown size={16} className="text-amber-400"/>
                                <h4 className="text-sm font-bold text-slate-300">Extreme Events</h4>
                            </div>

                            <div className="space-y-3">
                                <div
                                    className="flex justify-between items-center bg-white/[0.02] p-3 rounded-lg border border-white/5">
                                    <span className="text-xs text-slate-500">Best Trade</span>
                                    <span className="text-sm font-mono font-bold text-emerald-400">
                                        +{formatCurrency(metrics.bestTrade)}
                                    </span>
                                </div>
                                <div
                                    className="flex justify-between items-center bg-white/[0.02] p-3 rounded-lg border border-white/5">
                                    <span className="text-xs text-slate-500">Worst Trade</span>
                                    <span className="text-sm font-mono font-bold text-red-400">
                                        {formatCurrency(metrics.worstTrade)}
                                    </span>
                                </div>
                                <div
                                    className="flex justify-between items-center bg-white/[0.02] p-3 rounded-lg border border-white/5">
                                    <span className="text-xs text-slate-500">Max Favorable Excursion</span>
                                    <span className="text-sm font-mono font-bold text-emerald-400">
                                        +{formatCurrency(metrics.maxFavorableExcursion)}
                                    </span>
                                </div>
                                <div
                                    className="flex justify-between items-center bg-white/[0.02] p-3 rounded-lg border border-white/5">
                                    <span className="text-xs text-slate-500">Max Adverse Excursion</span>
                                    <span className="text-sm font-mono font-bold text-red-400">
                                        {formatCurrency(metrics.maxAdverseExcursion)}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="p-6 border-t border-slate-800 bg-slate-900/50 flex justify-end sticky bottom-0">
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
