import {BrainCircuit, Loader2} from "lucide-react";
import React from "react";
import {STRATEGY_TEMPLATES} from "@/components/optionsdesk/contants/strategyTemplates";
import {MLForecast, StrategyTemplate} from "@/types/all_types";

interface MLForecastPanelProps {
    mlForecast: MLForecast,
    selectedSymbol: string,
    selectedStrategies: StrategyTemplate[],
    addStrategyToCompare: (strategy: StrategyTemplate) => Promise<void>,
    isLoading: boolean
}

const MLForecastPanel: React.FC<MLForecastPanelProps> = ({
                                                             mlForecast,
                                                             selectedSymbol,
                                                             selectedStrategies,
                                                             addStrategyToCompare
                                                             ,
                                                             isLoading
                                                         }) => {
    return (

        <div className="bg-gradient-to-br from-purple-900/20 to-pink-900/20 border border-purple-500/30 rounded-xl p-6">
            <div className="flex items-start justify-between">
                <div className="flex-1">
                    <h3 className="text-sm font-bold text-slate-300 mb-3 flex items-center gap-2">
                        <BrainCircuit size={16} className="text-purple-400"/>
                        AI Market Forecast for {selectedSymbol}
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-4">
                        <div className={`text-3xl font-bold ${
                            mlForecast.direction === 'bullish' ? 'text-emerald-400' :
                                mlForecast.direction === 'bearish' ? 'text-red-400' :
                                    'text-slate-400'
                        }`}>
                            {mlForecast.direction === 'bullish' ? '📈 Bullish' :
                                mlForecast.direction === 'bearish' ? '📉 Bearish' :
                                    '➖ Neutral'}
                        </div>
                        <div>
                            <div className="text-sm text-slate-500">AI Confidence</div>
                            <div className="text-xl font-bold text-purple-400">
                                {(mlForecast.confidence * 100).toFixed(1)}%
                            </div>
                        </div>
                        {mlForecast.priceTargets && (
                            <div>
                                <div className="text-sm text-slate-500">Price Targets</div>
                                <div className="text-sm text-slate-300">
                                    ${mlForecast.priceTargets.low.toFixed(2)} -
                                    ${mlForecast.priceTargets.high.toFixed(2)}
                                </div>
                            </div>
                        )}
                    </div>
                    <div className="space-y-2">
                        <p className="text-xs font-bold text-slate-500 uppercase">Recommended Strategies:</p>
                        <div className="flex flex-wrap gap-2">
                            {mlForecast.suggestedStrategies.map((strat: string, idx: number) => {
                                const strategy = STRATEGY_TEMPLATES.find(s => s.name === strat);
                                return strategy ? (
                                    <button
                                        key={idx}
                                        onClick={() => addStrategyToCompare(strategy)}
                                        disabled={selectedStrategies.length >= 4}
                                        className="px-3 py-2 bg-purple-500/10 hover:bg-purple-500/20 border border-purple-500/30 rounded-lg text-sm text-purple-300 transition-colors disabled:opacity-50"
                                    >
                                        {strat}
                                    </button>
                                ) : null;
                            })}
                        </div>
                    </div>
                </div>
                {isLoading && (
                    <Loader2 className="animate-spin text-purple-400" size={24}/>
                )}
            </div>
        </div>
    )
};

export default MLForecastPanel;