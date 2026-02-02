import {Activity, Calculator, Layers, Loader2, Plus, Settings, Target, X} from "lucide-react";
import {CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis} from "recharts";
import React from "react";
import {formatCurrency, formatPercent} from "@/utils/formatters";
import {getGreekColor} from "@/components/optionsdesk/utils/colors";
import {GreeksChartData, OptionLeg} from "@/types/all_types";

interface BuilderTabProps {
    customLegs: OptionLeg[],
    newLeg:  Partial<OptionLeg>,
    setNewLeg: React.Dispatch<React.SetStateAction<Partial<OptionLeg>>>,
    addCustomLeg: () => void ,
    removeLeg: (id: string) => void,
    analyzeCustomStrategy: () => Promise<void>,
    profitLossData: any,
    riskMetrics: any,
    strikeOptimizer: any,
    greeksChartData: GreeksChartData[],
    currentPrice: number,
    isLoading: boolean,
}

const BuilderTab: React.FC<BuilderTabProps> = ({
                                                   customLegs,
                                                   newLeg,
                                                   setNewLeg,
                                                   addCustomLeg,
                                                   removeLeg,
                                                   analyzeCustomStrategy,
                                                   profitLossData,
                                                   riskMetrics,
                                                   strikeOptimizer,
                                                   greeksChartData,
                                                   currentPrice,
                                                   isLoading,
                                               }: BuilderTabProps) => {
    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-6">
                <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                    <h3 className="text-lg font-bold text-slate-200 mb-6 flex items-center gap-2">
                        <Settings size={20} className="text-amber-400"/>
                        Build Custom Strategy
                    </h3>

                    <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
                        <select
                            value={newLeg.type}
                            onChange={(e) => setNewLeg({...newLeg, type: e.target.value as any})}
                            className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200"
                        >
                            <option value="call">Call</option>
                            <option value="put">Put</option>
                        </select>
                        <select
                            value={newLeg.position}
                            onChange={(e) => setNewLeg({...newLeg, position: e.target.value as any})}
                            className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200"
                        >
                            <option value="long">Long</option>
                            <option value="short">Short</option>
                        </select>
                        <input
                            type="number"
                            value={newLeg.strike || ''}
                            onChange={(e) => setNewLeg({...newLeg, strike: parseFloat(e.target.value) || 0})}
                            placeholder="Strike"
                            className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200"
                        />
                        <input
                            type="number"
                            value={newLeg.quantity || ''}
                            onChange={(e) => setNewLeg({...newLeg, quantity: parseInt(e.target.value) || 1})}
                            placeholder="Qty"
                            className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200"
                        />
                        <button
                            onClick={addCustomLeg}
                            className="bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white rounded-lg flex items-center justify-center gap-2 transition-all"
                        >
                            <Plus size={16}/>
                            Add Leg
                        </button>
                    </div>

                    <div className="mb-4">
                        <label className="block text-xs font-bold text-slate-400 uppercase mb-2">Expiration Date</label>
                        <input
                            type="date"
                            value={newLeg.expiration || ''}
                            onChange={(e) => setNewLeg({...newLeg, expiration: e.target.value})}
                            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200"
                        />
                    </div>

                    <div className="space-y-2">
                        <h4 className="text-sm font-bold text-slate-400 uppercase mb-3">Current Legs
                            ({customLegs.length})</h4>
                        {customLegs.length === 0 ? (
                            <div className="text-center py-12 text-slate-500">
                                <Layers size={48} className="mx-auto mb-4 opacity-20"/>
                                <p>No legs added yet. Build your custom strategy above.</p>
                            </div>
                        ) : (
                            customLegs.map((leg) => (
                                <div key={leg.id}
                                     className="flex items-center justify-between p-4 bg-slate-800/50 rounded-lg border border-slate-700/50">
                                    <div className="flex items-center gap-4 flex-1">
                                        <div
                                            className={`w-2 h-2 rounded-full ${leg.position === 'long' ? 'bg-emerald-500' : 'bg-red-500'}`}/>
                                        <div className="flex-1">
                                            <div className="text-sm font-bold text-slate-200 mb-1">
                                                {leg.position === 'long' ? '🟢' : '🔴'} {leg.quantity}x {leg.type.toUpperCase()} @
                                                ${leg.strike}
                                            </div>
                                            <div className="text-xs text-slate-500 mb-2">
                                                Expires: {leg.expiration} | Premium: ${leg.premium?.toFixed(2)}
                                            </div>
                                            <div className="grid grid-cols-4 gap-2 text-xs">
                                                <div>
                                                    <span className="text-slate-600">Δ:</span>
                                                    <span
                                                        className={`ml-1 font-bold ${leg.delta && leg.delta > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                                    {leg.delta?.toFixed(3)}
                                                                </span>
                                                </div>
                                                <div>
                                                    <span className="text-slate-600">Γ:</span>
                                                    <span
                                                        className="ml-1 font-bold text-purple-400">{leg.gamma?.toFixed(3)}</span>
                                                </div>
                                                <div>
                                                    <span className="text-slate-600">Θ:</span>
                                                    <span
                                                        className="ml-1 font-bold text-amber-400">{leg.theta?.toFixed(3)}</span>
                                                </div>
                                                <div>
                                                    <span className="text-slate-600">ν:</span>
                                                    <span
                                                        className="ml-1 font-bold text-blue-400">{leg.vega?.toFixed(3)}</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <button
                                        onClick={() => removeLeg(leg.id)}
                                        className="p-2 hover:bg-red-500/10 rounded-lg text-red-400 transition-colors"
                                    >
                                        <X size={16}/>
                                    </button>
                                </div>
                            ))
                        )}
                    </div>

                    {customLegs.length > 0 && (
                        <button
                            onClick={analyzeCustomStrategy}
                            disabled={isLoading}
                            className="w-full mt-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-bold rounded-xl flex items-center justify-center gap-2 disabled:opacity-50"
                        >
                            {isLoading ? <Loader2 size={20} className="animate-spin"/> : <Calculator size={20}/>}
                            Analyze Strategy
                        </button>
                    )}
                </div>

                {profitLossData.length > 0 && (
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                        <h4 className="text-sm font-bold text-slate-300 mb-4">Payoff Diagram</h4>
                        <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={profitLossData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#374151"/>
                                    <XAxis dataKey="price" stroke="#9ca3af"
                                           label={{value: 'Underlying Price', position: 'insideBottom', offset: -5}}/>
                                    <YAxis stroke="#9ca3af"
                                           label={{value: 'Profit/Loss', angle: -90, position: 'insideLeft'}}/>
                                    <Tooltip formatter={(value) => [formatCurrency(Number(value)), 'Profit']}
                                             contentStyle={{backgroundColor: '#1f2937', borderColor: '#4b5563'}}
                                             labelStyle={{color: '#d1d5db'}}/>
                                    <ReferenceLine x={currentPrice} stroke="#f59e0b" strokeDasharray="3 3"/>
                                    <ReferenceLine y={0} stroke="#6b7280"/>
                                    <Line type="monotone" dataKey="profit" stroke="#8b5cf6" strokeWidth={2}
                                          dot={false}/>
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                )}
            </div>

            <div className="space-y-6">
                {riskMetrics && (
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                        <h4 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
                            <Target size={16} className="text-amber-400"/>
                            Risk Metrics
                        </h4>
                        <div className="space-y-3">
                            <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                                <span className="text-xs text-slate-400">VaR (95%)</span>
                                <span className="text-sm font-bold text-red-400">
                                                {formatCurrency(Math.abs(riskMetrics.var_95))}
                                            </span>
                            </div>
                            <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                                <span className="text-xs text-slate-400">CVaR (95%)</span>
                                <span className="text-sm font-bold text-amber-400">
                                                {formatCurrency(Math.abs(riskMetrics.cvar_95))}
                                            </span>
                            </div>
                            {riskMetrics.kelly_fraction && (
                                <div className="flex justify-between items-center p-3 bg-slate-800/50 rounded-lg">
                                    <span className="text-xs text-slate-400">Kelly Fraction</span>
                                    <span className="text-sm font-bold text-blue-400">
                                                    {formatPercent(riskMetrics.kelly_fraction)}
                                                </span>
                                </div>
                            )}
                            {riskMetrics.recommendation && (
                                <div className="p-3 bg-slate-800/50 rounded-lg">
                                    <div className="text-xs text-slate-400 mb-1">Recommendation</div>
                                    <div
                                        className="text-sm font-bold text-emerald-400">{riskMetrics.recommendation}</div>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* Portfolio Greeks Summary */}
                {customLegs.length > 0 && (
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                        <h4 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
                            <Activity size={16} className="text-purple-400"/>
                            Portfolio Greeks
                        </h4>
                        <div className="space-y-3">
                            {(() => {
                                const totalDelta = customLegs.reduce((sum, leg) => sum + (leg.delta || 0) * leg.quantity, 0);
                                const totalGamma = customLegs.reduce((sum, leg) => sum + (leg.gamma || 0) * leg.quantity, 0);
                                const totalTheta = customLegs.reduce((sum, leg) => sum + (leg.theta || 0) * leg.quantity, 0);
                                const totalVega = customLegs.reduce((sum, leg) => sum + (leg.vega || 0) * leg.quantity, 0);

                                return (
                                    <>
                                        <div className="p-3 bg-slate-800/50 rounded-lg">
                                            <div className="flex justify-between items-center mb-1">
                                                <span className="text-xs text-slate-400">Delta (Δ)</span>
                                                <span
                                                    className={`text-lg font-bold ${totalDelta > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                                {totalDelta.toFixed(3)}
                                                            </span>
                                            </div>
                                            <div className="text-xs text-slate-500">
                                                Directional exposure: ${(totalDelta * currentPrice).toFixed(2)} per $1
                                                move
                                            </div>
                                        </div>
                                        <div className="p-3 bg-slate-800/50 rounded-lg">
                                            <div className="flex justify-between items-center mb-1">
                                                <span className="text-xs text-slate-400">Gamma (Γ)</span>
                                                <span className="text-lg font-bold text-purple-400">
                                                                {totalGamma.toFixed(4)}
                                                            </span>
                                            </div>
                                            <div className="text-xs text-slate-500">
                                                Delta change per $1 move
                                            </div>
                                        </div>
                                        <div className="p-3 bg-slate-800/50 rounded-lg">
                                            <div className="flex justify-between items-center mb-1">
                                                <span className="text-xs text-slate-400">Theta (Θ)</span>
                                                <span
                                                    className={`text-lg font-bold ${totalTheta < 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                                                                {totalTheta.toFixed(3)}
                                                            </span>
                                            </div>
                                            <div className="text-xs text-slate-500">
                                                Time decay: ${(totalTheta * 100).toFixed(2)}/day
                                            </div>
                                        </div>
                                        <div className="p-3 bg-slate-800/50 rounded-lg">
                                            <div className="flex justify-between items-center mb-1">
                                                <span className="text-xs text-slate-400">Vega (ν)</span>
                                                <span className="text-lg font-bold text-blue-400">
                                                                {totalVega.toFixed(3)}
                                                            </span>
                                            </div>
                                            <div className="text-xs text-slate-500">
                                                IV sensitivity: ${(totalVega * 100).toFixed(2)} per 1% IV change
                                            </div>
                                        </div>
                                    </>
                                );
                            })()}
                        </div>
                    </div>
                )}

                {/* Break-even Analysis */}
                {customLegs.length > 0 && (
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                        <h4 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
                            <Target size={16} className="text-emerald-400"/>
                            Break-even Analysis
                        </h4>
                        <div className="space-y-3">
                            {(() => {
                                const totalCost = customLegs.reduce((sum, leg) => {
                                    const multiplier = leg.position === 'long' ? 1 : -1;
                                    return sum + (leg.premium || 0) * leg.quantity * multiplier * 100;
                                }, 0);

                                const avgStrike = customLegs.reduce((sum, leg) => sum + leg.strike, 0) / customLegs.length;
                                const breakEvenUp = avgStrike + Math.abs(totalCost) / 100;
                                const breakEvenDown = avgStrike - Math.abs(totalCost) / 100;

                                return (
                                    <>
                                        <div className="p-3 bg-slate-800/50 rounded-lg">
                                            <div className="text-xs text-slate-400 mb-1">Net Cost/Credit</div>
                                            <div
                                                className={`text-xl font-bold ${totalCost < 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                                                {totalCost < 0 ? '-' : '+'}{formatCurrency(Math.abs(totalCost))}
                                            </div>
                                        </div>
                                        <div className="p-3 bg-slate-800/50 rounded-lg">
                                            <div className="text-xs text-slate-400 mb-1">Upper Break-even</div>
                                            <div className="text-lg font-bold text-blue-400">
                                                ${breakEvenUp.toFixed(2)}
                                            </div>
                                            <div className="text-xs text-slate-500">
                                                {((breakEvenUp - currentPrice) / currentPrice * 100).toFixed(2)}% from
                                                current
                                            </div>
                                        </div>
                                        <div className="p-3 bg-slate-800/50 rounded-lg">
                                            <div className="text-xs text-slate-400 mb-1">Lower Break-even</div>
                                            <div className="text-lg font-bold text-blue-400">
                                                ${breakEvenDown.toFixed(2)}
                                            </div>
                                            <div className="text-xs text-slate-500">
                                                {((breakEvenDown - currentPrice) / currentPrice * 100).toFixed(2)}% from
                                                current
                                            </div>
                                        </div>
                                        <div className="p-3 bg-emerald-900/20 border border-emerald-500/30 rounded-lg">
                                            <div className="text-xs text-slate-400 mb-1">Profitable Range</div>
                                            <div className="text-sm font-bold text-emerald-400">
                                                ${breakEvenDown.toFixed(2)} - ${breakEvenUp.toFixed(2)}
                                            </div>
                                        </div>
                                    </>
                                );
                            })()}
                        </div>
                    </div>
                )}

                {greeksChartData.length > 0 && (
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                        <h4 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
                            <Activity size={16} className="text-purple-400"/>
                            Greeks Analysis
                        </h4>
                        <div className="space-y-3">
                            {greeksChartData.map((data, idx: number) => (
                                <div key={idx} className="p-3 bg-slate-800/50 rounded-lg">
                                    <div className="text-xs font-bold text-slate-400 mb-2">{data.name}</div>
                                    <div className="grid grid-cols-2 gap-2 text-sm">
                                        <div className="flex justify-between">
                                            <span className="text-slate-500">Delta:</span>
                                            <span className={`font-bold ${getGreekColor(data.delta, 'delta')}`}>
                                                            {data.delta.toFixed(3)}
                                                        </span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-slate-500">Gamma:</span>
                                            <span className={`font-bold ${getGreekColor(data.gamma, 'gamma')}`}>
                                                            {data.gamma.toFixed(3)}
                                                        </span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-slate-500">Theta:</span>
                                            <span className={`font-bold ${getGreekColor(data.theta, 'theta')}`}>
                                                            {data.theta.toFixed(3)}
                                                        </span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-slate-500">Vega:</span>
                                            <span className={`font-bold ${getGreekColor(data.vega, 'vega')}`}>
                                                            {data.vega.toFixed(3)}
                                                        </span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-slate-500">Rho:</span>
                                            <span className={`font-bold ${getGreekColor(data.rho, 'rho')}`}>
                                                            {data.rho.toFixed(3)}
                                                        </span>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {strikeOptimizer && (
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                        <h4 className="text-sm font-bold text-slate-300 mb-4">Strike Optimization</h4>
                        <div className="space-y-2">
                            {strikeOptimizer.strikes.slice(0, 3).map((strike: any, idx: number) => (
                                <div key={idx} className="p-2 bg-slate-800/50 rounded-lg">
                                    <div className="flex justify-between items-center">
                                        <span className="text-sm text-slate-300">${strike.strike}</span>
                                        <span className="text-xs text-emerald-400">
                                                        {formatPercent(strike.prob_itm)} ITM
                                                    </span>
                                    </div>
                                    <div className="text-xs text-slate-500">
                                        Est. Premium: {formatCurrency(strike.premium_estimate)}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}

export default BuilderTab;
