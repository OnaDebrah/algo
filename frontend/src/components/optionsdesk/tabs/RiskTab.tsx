import {BarChart2, Shield} from "lucide-react";
import React from "react";
import {formatCurrency, formatPercent} from "@/utils/formatters";

interface RiskTabProps {
    portfolioStats: any,
    riskMetrics: any
}

const RiskTab: React.FC<RiskTabProps> = ({
                                             portfolioStats,
                                             riskMetrics
                                         }: RiskTabProps) => {

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {portfolioStats && (
                <div className="lg:col-span-2 bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                    <h3 className="text-lg font-bold text-slate-200 mb-6 flex items-center gap-2">
                        <BarChart2 size={20} className="text-amber-400"/>
                        Portfolio Statistics
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="p-4 bg-slate-800/50 rounded-lg">
                            <div className="text-xs text-slate-500">Win Rate</div>
                            <div className="text-xl font-bold text-emerald-400">
                                {portfolioStats.win_rate.toFixed(1)}%
                            </div>
                        </div>
                        <div className="p-4 bg-slate-800/50 rounded-lg">
                            <div className="text-xs text-slate-500">Profit Factor</div>
                            <div className="text-xl font-bold text-blue-400">
                                {portfolioStats.profit_factor.toFixed(2)}
                            </div>
                        </div>
                        <div className="p-4 bg-slate-800/50 rounded-lg">
                            <div className="text-xs text-slate-500">Avg Win</div>
                            <div className="text-xl font-bold text-emerald-400">
                                {formatCurrency(portfolioStats.avg_win)}
                            </div>
                        </div>
                        <div className="p-4 bg-slate-800/50 rounded-lg">
                            <div className="text-xs text-slate-500">Avg Loss</div>
                            <div className="text-xl font-bold text-red-400">
                                {formatCurrency(Math.abs(portfolioStats.avg_loss))}
                            </div>
                        </div>
                        <div className="p-4 bg-slate-800/50 rounded-lg">
                            <div className="text-xs text-slate-500">Expectancy</div>
                            <div className="text-xl font-bold text-amber-400">
                                {formatCurrency(portfolioStats.expectancy)}
                            </div>
                        </div>
                        <div className="p-4 bg-slate-800/50 rounded-lg">
                            <div className="text-xs text-slate-500">Sharpe Ratio</div>
                            <div className="text-xl font-bold text-purple-400">
                                {portfolioStats.kelly_fraction?.toFixed(3) || 'N/A'}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            <div className="space-y-6">
                {riskMetrics && (
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                        <h4 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
                            <Shield size={16} className="text-amber-400"/>
                            Risk Recommendations
                        </h4>
                        <div className="space-y-3">
                            <div className="p-3 bg-slate-800/50 rounded-lg">
                                <div className="text-xs text-slate-500 mb-1">Value at Risk (95%)</div>
                                <div className="text-sm font-bold text-red-400">
                                    {formatCurrency(riskMetrics.var_95)}
                                </div>
                            </div>
                            <div className="p-3 bg-slate-800/50 rounded-lg">
                                <div className="text-xs text-slate-500 mb-1">Conditional VaR</div>
                                <div className="text-sm font-bold text-amber-400">
                                    {formatCurrency(riskMetrics.cvar_95)}
                                </div>
                            </div>
                            <div className="p-3 bg-slate-800/50 rounded-lg">
                                <div className="text-xs text-slate-500 mb-1">Kelly Criterion</div>
                                <div className="text-sm font-bold text-blue-400">
                                    {riskMetrics.kelly_fraction ? formatPercent(riskMetrics.kelly_fraction) : 'N/A'}
                                </div>
                            </div>
                            {riskMetrics.recommendation && (
                                <div className="p-3 bg-slate-800/50 rounded-lg border border-emerald-500/30">
                                    <div className="text-xs text-slate-500 mb-1">AI Recommendation</div>
                                    <div
                                        className="text-sm font-bold text-emerald-400">{riskMetrics.recommendation}</div>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}

export default RiskTab;