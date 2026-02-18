import React from 'react';
import {Activity, Info, Target, Zap} from 'lucide-react';

interface FactorAttributionProps {
    alpha?: number;
    beta?: number;
    rSquared?: number;
}

const FactorAttribution: React.FC<FactorAttributionProps> = ({
                                                                 alpha = 0,
                                                                 beta = 0,
                                                                 rSquared = 0
                                                             }: FactorAttributionProps) => {
    return (
        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
            <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
                <Target className="w-5 h-5 text-indigo-400"/>
                Factor Attribution
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Alpha */}
                <div className="space-y-2">
                    <div className="flex items-center justify-between">
                        <span className="text-slate-400 text-sm flex items-center gap-1">
                            Alpha (Annual)
                            <div className="group relative">
                                <Info className="w-3 h-3 cursor-help"/>
                                <div
                                    className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 p-2 bg-slate-800 text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 border border-slate-700 shadow-xl">
                                    The strategy's performance relative to the market, adjusted for risk. Positive alpha means outperformance.
                                </div>
                            </div>
                        </span>
                        <Zap className="w-4 h-4 text-yellow-400"/>
                    </div>
                    <div
                        className={`text-2xl font-bold ${alpha >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {(alpha || 0) >= 0 ? '+' : ''}{alpha.toFixed(2)}%
                    </div>
                    <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                        <div
                            className={`h-full ${(alpha || 0) >= 0 ? 'bg-green-500' : 'bg-red-500'}`}
                            style={{width: `${Math.min(Math.abs(alpha || 0) * 5, 100)}%`}}
                        ></div>
                    </div>
                </div>

                {/* Beta */}
                <div className="space-y-2">
                    <div className="flex items-center justify-between">
                        <span className="text-slate-400 text-sm flex items-center gap-1">
                            Beta
                            <div className="group relative">
                                <Info className="w-3 h-3 cursor-help"/>
                                <div
                                    className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 p-2 bg-slate-800 text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 border border-slate-700 shadow-xl">
                                    Measures sensitivity to market movements. Beta &gt; 1 is more volatile than market; Beta &lt; 1 is less volatile.
                                </div>
                            </div>
                        </span>
                        <Activity className="w-4 h-4 text-blue-400"/>
                    </div>
                    <div className="text-2xl font-bold text-white">
                        {(beta || 0).toFixed(2)}
                    </div>
                    <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-blue-500"
                            style={{width: `${Math.min((beta || 0) * 50, 100)}%`}}
                        ></div>
                    </div>
                </div>

                {/* R-Squared */}
                <div className="space-y-2">
                    <div className="flex items-center justify-between">
                        <span className="text-slate-400 text-sm flex items-center gap-1">
                            R-Squared
                            <div className="group relative">
                                <Info className="w-3 h-3 cursor-help"/>
                                <div
                                    className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 p-2 bg-slate-800 text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 border border-slate-700 shadow-xl">
                                    The percentage of a strategy's movements explained by movements in its benchmark.
                                </div>
                            </div>
                        </span>
                        <Target className="w-4 h-4 text-purple-400"/>
                    </div>
                    <div className="text-2xl font-bold text-white">
                        {((rSquared || 0) * 100).toFixed(1)}%
                    </div>
                    <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-purple-500"
                            style={{width: `${(rSquared || 0) * 100}%`}}
                        ></div>
                    </div>
                </div>
            </div>

            <div className="mt-6 pt-4 border-t border-slate-800 flex items-center gap-2 text-xs text-slate-500">
                <Info className="w-3 h-3"/>
                Benchmark: SPY (S&P 500 ETF Trust)
            </div>
        </div>
    );
};

export default FactorAttribution;
