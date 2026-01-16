'use client'

import React from "react";

const StrategyParameterForm = ({ params, values, onChange }: {
    params: Record<string, any>,
    values: Record<string, any>,
    onChange: (key: string, val: any) => void
}) => {
    return (
        <div className="space-y-4">
            {Object.entries(params).map(([key, defaultValue]) => {
                const currentVal = values[key] ?? defaultValue;
                const label = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

                return (
                    <div key={key} className="space-y-2">
                        <label className="text-xs font-semibold text-slate-400 tracking-wide flex items-center justify-between">
                            <span>{label}</span>
                            <span className="text-[10px] text-slate-600 font-mono">{typeof defaultValue}</span>
                        </label>
                        {typeof defaultValue === 'boolean' ? (
                            <button
                                onClick={() => onChange(key, !currentVal)}
                                className={`w-full py-3 px-4 rounded-xl border text-sm font-semibold transition-all ${
                                    currentVal
                                        ? 'bg-violet-500/20 border-violet-500/50 text-violet-300'
                                        : 'bg-slate-800/50 border-slate-700/50 text-slate-400 hover:border-slate-600'
                                }`}
                            >
                                {currentVal ? '✓ ENABLED' : '✗ DISABLED'}
                            </button>
                        ) : (
                            <div className="relative">
                                <input
                                    type="number"
                                    step="any"
                                    value={currentVal}
                                    onChange={(e) => onChange(key, parseFloat(e.target.value))}
                                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-xl px-4 py-3 text-sm text-slate-200 font-mono focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none transition-all"
                                />
                                <div className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-slate-600">
                                    {typeof currentVal === 'number' && currentVal % 1 !== 0 ? 'float' : 'int'}
                                </div>
                            </div>
                        )}
                    </div>
                );
            })}
        </div>
    );
};

export default StrategyParameterForm;