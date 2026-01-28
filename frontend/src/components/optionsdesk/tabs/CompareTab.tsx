import {GitCompare, X} from "lucide-react";
import {Legend, PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer} from "recharts";
import React from "react";
import {getStrategyColor} from "@/components/optionsdesk/utils/colors";
import {formatCurrency, formatPercent} from "@/utils/formatters";
import {STRATEGY_TEMPLATES} from "@/components/optionsdesk/contants/strategyTemplates";
import {StrategyAnalysis, StrategyTemplate} from "@/types/all_types";

interface CompareTabProps {
    selectedStrategies: StrategyTemplate[],
    strategyAnalysis: StrategyAnalysis[],
    addStrategyToCompare: any,
    removeStrategyFromCompare: any,
    profitLossData: any,
    monteCarloDistribution: any
}

const CompareTabs: React.FC<CompareTabProps> = ({
                                                    selectedStrategies,
                                                    strategyAnalysis,
                                                    addStrategyToCompare,
                                                    removeStrategyFromCompare,
                                                    profitLossData,
                                                    monteCarloDistribution
                                                }: CompareTabProps) => {
    return (
        <div className="space-y-6">
            <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                <h3 className="text-lg font-bold text-slate-200 mb-4 flex items-center gap-2">
                    <GitCompare size={20} className="text-amber-400"/>
                    Select Strategies to Compare (Max 4)
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {STRATEGY_TEMPLATES.map((strategy: StrategyTemplate) => {
                        const isSelected = selectedStrategies.find(s => s.id === strategy.id);
                        const Icon = strategy.icon;

                        return (
                            <button
                                key={strategy.id}
                                onClick={() => isSelected ? removeStrategyFromCompare(strategy.id) : addStrategyToCompare(strategy)}
                                disabled={selectedStrategies.length >= 4 && !isSelected}
                                className={`p-4 rounded-xl border text-left transition-all ${
                                    isSelected
                                        ? 'bg-amber-500/10 border-amber-500/50'
                                        : 'bg-slate-800 border-slate-700 hover:border-slate-600'
                                } disabled:opacity-50 disabled:cursor-not-allowed`}
                            >
                                <div className="flex items-start justify-between mb-2">
                                    <Icon size={20} className={isSelected ? 'text-amber-400' : 'text-slate-500'}/>
                                    {isSelected && <div className="w-2 h-2 rounded-full bg-amber-400"/>}
                                </div>
                                <div className="text-sm font-bold text-slate-200 mb-1">{strategy.name}</div>
                                <div
                                    className={`text-xs ${getStrategyColor(strategy.sentiment)}`}>{strategy.sentiment}</div>
                            </button>
                        );
                    })}
                </div>
            </div>

            {strategyAnalysis.length > 0 && (
                <>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        {strategyAnalysis.map((analysis: StrategyAnalysis) => (
                            <div key={analysis.id}
                                 className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6 relative">
                                <button
                                    onClick={() => removeStrategyFromCompare(analysis.id)}
                                    className="absolute top-4 right-4 p-1 hover:bg-red-500/10 rounded text-red-400"
                                >
                                    <X size={14}/>
                                </button>
                                <h4 className="font-bold text-slate-200 mb-4 pr-6">{analysis.name}</h4>
                                <div className="space-y-3">
                                    <div className="flex justify-between items-center">
                                        <span className="text-xs text-slate-400">Max Profit</span>
                                        <span className="text-sm font-bold text-emerald-400">
                                                        {formatCurrency(analysis.analysis?.max_profit ?? 0)}
                                                    </span>
                                    </div>
                                    <div className="flex justify-between items-center">
                                        <span className="text-xs text-slate-400">Max Loss</span>
                                        <span className="text-sm font-bold text-red-400">
                                                        {formatCurrency(Math.abs(analysis.analysis?.max_loss ?? 0))}
                                                    </span>
                                    </div>
                                    <div className="flex justify-between items-center">
                                        <span className="text-xs text-slate-400">Win Prob</span>
                                        <span className="text-sm font-bold text-amber-400">
                                                        {formatPercent(analysis.analysis?.probability_of_profit ?? 0)}
                                                    </span>
                                    </div>
                                    <div className="flex justify-between items-center">
                                        <span className="text-xs text-slate-400">Initial Cost</span>
                                        <span className="text-sm font-bold text-blue-400">
                                                        {formatCurrency(analysis.analysis?.initial_cost ?? 0)}
                                                    </span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>

                    {profitLossData.length > 0 && (
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                                <h4 className="text-sm font-bold text-slate-300 mb-6">Greeks Comparison</h4>
                                <div className="h-64">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <RadarChart data={strategyAnalysis.map(s => ({
                                            subject: s.name,
                                            delta: Math.abs(s.greeks?.delta ?? 0),
                                            gamma: Math.abs(s.greeks?.gamma ?? 0),
                                            theta: Math.abs(s.greeks?.theta ?? 0),
                                            vega: Math.abs(s.greeks?.vega ?? 0),
                                            rho: Math.abs(s.greeks?.rho ?? 0)
                                        }))}>
                                            <PolarGrid/>
                                            <PolarAngleAxis dataKey="subject"/>
                                            <PolarRadiusAxis/>
                                            <Radar name="Delta" dataKey="delta" stroke="#10b981" fill="#10b981"
                                                   fillOpacity={0.6}/>
                                            <Radar name="Gamma" dataKey="gamma" stroke="#8b5cf6" fill="#8b5cf6"
                                                   fillOpacity={0.6}/>
                                            <Legend/>
                                        </RadarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            {monteCarloDistribution && (
                                <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                                    <h4 className="text-sm font-bold text-slate-300 mb-6">Monte Carlo Analysis</h4>
                                    <div className="space-y-3">
                                        <div className="grid grid-cols-2 gap-3">
                                            <div className="p-3 bg-slate-800/50 rounded-lg">
                                                <div className="text-xs text-slate-500">Mean Price</div>
                                                <div className="text-sm font-bold text-emerald-400">
                                                    {formatCurrency(monteCarloDistribution.mean_final_price)}
                                                </div>
                                            </div>
                                            <div className="p-3 bg-slate-800/50 rounded-lg">
                                                <div className="text-xs text-slate-500">Prob Above</div>
                                                <div className="text-sm font-bold text-amber-400">
                                                    {formatPercent(monteCarloDistribution.probability_above_current)}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </>
            )}
        </div>
    )
}

export default CompareTabs;