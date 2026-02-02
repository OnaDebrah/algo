import {Activity} from "lucide-react";
import {Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis} from "recharts";
import React from "react";

interface VolatilityTabProps {
    optionsChain: any,
    selectedExpiry: string,
    currentPrice: number
}

const VolatilityTab: React.FC<VolatilityTabProps> = ({
                                                         optionsChain,
                                                         selectedExpiry,
                                                         currentPrice
                                                     }) => {
    return (
        <div className="space-y-6">
            {optionsChain && selectedExpiry ? (
                <>
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                        <h3 className="text-sm font-bold text-slate-300 mb-6 flex items-center gap-2">
                            <Activity size={16} className="text-amber-400"/>
                            Implied Volatility Surface
                        </h3>
                        <div className="h-[400px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={optionsChain.calls.map((call: any, idx: number) => ({
                                    strike: call.strike,
                                    callIV: call.impliedVolatility ? call.impliedVolatility * 100 : 0,
                                    putIV: optionsChain.puts[idx]?.impliedVolatility ? optionsChain.puts[idx].impliedVolatility * 100 : 0
                                }))}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                                    <XAxis dataKey="strike" stroke="#64748b" fontSize={10}
                                           label={{value: 'Strike Price', position: 'insideBottom', offset: -5}}/>
                                    <YAxis stroke="#64748b" fontSize={10}
                                           label={{value: 'IV (%)', angle: -90, position: 'insideLeft'}}/>
                                    <Tooltip contentStyle={{backgroundColor: '#0f172a', border: '1px solid #1e293b'}}
                                             formatter={(value: any) => `${Number(value).toFixed(2)}%`}/>
                                    <Line type="monotone" dataKey="callIV" stroke="#10b981" strokeWidth={2}
                                          name="Call IV" dot={false}/>
                                    <Line type="monotone" dataKey="putIV" stroke="#ef4444" strokeWidth={2} name="Put IV"
                                          dot={false}/>
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                            <h4 className="text-xs font-bold text-slate-500 uppercase mb-4">ATM Implied Volatility</h4>
                            {(() => {
                                if (!optionsChain.calls || optionsChain.calls.length === 0) {
                                    return <span>Loading strikes...</span>;
                                }
                                const atmStrike = optionsChain.calls.reduce((prev: any, curr: any) =>
                                        Math.abs(curr.strike - (optionsChain.underlying_price || 0)) < Math.abs(prev.strike - (optionsChain.underlying_price || 0)) ? curr : prev,
                                    optionsChain.calls[0]
                                );
                                const atmIV = atmStrike.impliedVolatility || 0;

                                return (
                                    <>
                                        <div className="text-3xl font-bold text-amber-400 mb-2">
                                            {(atmIV * 100).toFixed(2)}%
                                        </div>
                                        <div className="text-sm text-slate-400">
                                            Strike: ${atmStrike.strike.toFixed(2)}
                                        </div>
                                    </>
                                );
                            })()}
                        </div>

                        <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                            <h4 className="text-xs font-bold text-slate-500 uppercase mb-4">IV Range</h4>
                            {(() => {
                                const allIVs = [
                                    ...optionsChain.calls.map((c: any) => c.impliedVolatility || 0),
                                    ...optionsChain.puts.map((p: any) => p.impliedVolatility || 0)
                                ].filter((iv: number) => iv > 0);

                                const minIV = Math.min(...allIVs);
                                const maxIV = Math.max(...allIVs);

                                return (
                                    <>
                                        <div className="flex items-center justify-between mb-2">
                                            <span className="text-sm text-slate-400">Low</span>
                                            <span className="text-lg font-bold text-emerald-400">
                                                            {(minIV * 100).toFixed(2)}%
                                                        </span>
                                        </div>
                                        <div className="flex items-center justify-between">
                                            <span className="text-sm text-slate-400">High</span>
                                            <span className="text-lg font-bold text-red-400">
                                                            {(maxIV * 100).toFixed(2)}%
                                                        </span>
                                        </div>
                                    </>
                                );
                            })()}
                        </div>

                        <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                            <h4 className="text-xs font-bold text-slate-500 uppercase mb-4">Volatility Skew</h4>
                            {(() => {
                                const skews = optionsChain.calls.map((call: any, idx: number) => {
                                    const put = optionsChain.puts[idx];
                                    if (!put || !call.impliedVolatility || !put.impliedVolatility) return 0;
                                    return (put.impliedVolatility - call.impliedVolatility) * 100;
                                });

                                const avgSkew = skews.reduce((a: number, b: number) => a + b, 0) / skews.length;

                                return (
                                    <>
                                        <div className="text-3xl font-bold text-blue-400 mb-2">
                                            {avgSkew > 0 ? '+' : ''}{avgSkew.toFixed(2)}%
                                        </div>
                                        <div className="text-sm text-slate-400">
                                            {avgSkew > 0 ? 'Put skew (bearish)' : 'Call skew (bullish)'}
                                        </div>
                                    </>
                                );
                            })()}
                        </div>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                            <h4 className="text-sm font-bold text-slate-300 mb-4">Call Volume by Strike</h4>
                            <div className="h-[250px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={optionsChain.calls.slice(0, 20)}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                                        <XAxis dataKey="strike" stroke="#64748b" fontSize={10}/>
                                        <YAxis stroke="#64748b" fontSize={10}/>
                                        <Tooltip
                                            contentStyle={{backgroundColor: '#0f172a', border: '1px solid #1e293b'}}/>
                                        <Bar dataKey="volume" fill="#10b981" radius={[4, 4, 0, 0]}/>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                            <h4 className="text-sm font-bold text-slate-300 mb-4">Put Volume by Strike</h4>
                            <div className="h-[250px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={optionsChain.puts.slice(0, 20)}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                                        <XAxis dataKey="strike" stroke="#64748b" fontSize={10}/>
                                        <YAxis stroke="#64748b" fontSize={10}/>
                                        <Tooltip
                                            contentStyle={{backgroundColor: '#0f172a', border: '1px solid #1e293b'}}/>
                                        <Bar dataKey="volume" fill="#ef4444" radius={[4, 4, 0, 0]}/>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>
                </>
            ) : (
                <div className="flex flex-col items-center justify-center h-[400px] text-slate-500">
                    <Activity size={48} className="mb-4 opacity-20"/>
                    <p>Load an options chain to view volatility analysis</p>
                </div>
            )}
        </div>
    )
}

export default VolatilityTab;
