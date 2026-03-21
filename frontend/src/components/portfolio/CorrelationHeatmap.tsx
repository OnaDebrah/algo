'use client'

import React, { useState } from 'react';
import { AlertCircle, Grid3X3, Loader2 } from 'lucide-react';
import { analytics } from '@/utils/api';

interface CorrelationData {
    symbols: string[];
    matrix: number[][];
    data_points: number;
    errors: string[];
}

const PERIODS = [
    { value: '1M', label: '1 Month' },
    { value: '3M', label: '3 Months' },
    { value: '6M', label: '6 Months' },
    { value: '1Y', label: '1 Year' },
    { value: '2Y', label: '2 Years' },
    { value: '5Y', label: '5 Years' },
];

function getCorrelationColor(value: number): string {
    // Blue (-1) -> White (0) -> Red (+1)
    if (value >= 0) {
        const intensity = Math.min(value, 1);
        const r = 220 + Math.round(35 * intensity);
        const g = Math.round(220 * (1 - intensity * 0.7));
        const b = Math.round(220 * (1 - intensity * 0.7));
        return `rgb(${r}, ${g}, ${b})`;
    } else {
        const intensity = Math.min(Math.abs(value), 1);
        const r = Math.round(220 * (1 - intensity * 0.7));
        const g = Math.round(220 * (1 - intensity * 0.5));
        const b = 220 + Math.round(35 * intensity);
        return `rgb(${r}, ${g}, ${b})`;
    }
}

function getTextColor(value: number): string {
    const abs = Math.abs(value);
    if (abs > 0.7) return 'text-white';
    return 'text-slate-800';
}

const CorrelationHeatmap: React.FC = () => {
    const [symbolsInput, setSymbolsInput] = useState('AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA');
    const [period, setPeriod] = useState('1Y');
    const [loading, setLoading] = useState(false);
    const [data, setData] = useState<CorrelationData | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleCompute = async () => {
        const symbols = symbolsInput.split(',').map(s => s.trim().toUpperCase()).filter(Boolean);
        if (symbols.length < 2) {
            setError('Enter at least 2 symbols');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const result = await analytics.correlation(symbols, period);
            setData(result);
        } catch (err: any) {
            const detail = err?.response?.data?.detail ?? err?.message ?? 'Failed to compute correlation';
            setError(typeof detail === 'string' ? detail : JSON.stringify(detail));
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="space-y-6">
            {/* Controls */}
            <div className="flex flex-col md:flex-row gap-4 items-end">
                <div className="flex-1">
                    <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest block mb-2">
                        Symbols (comma-separated)
                    </label>
                    <input
                        type="text"
                        value={symbolsInput}
                        onChange={(e) => setSymbolsInput(e.target.value)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-xl py-3 px-4 text-xs font-mono text-emerald-400 focus:border-emerald-500 outline-none transition-all"
                        placeholder="AAPL, MSFT, GOOGL, AMZN"
                    />
                </div>
                <div>
                    <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest block mb-2">
                        Period
                    </label>
                    <select
                        value={period}
                        onChange={(e) => setPeriod(e.target.value)}
                        className="bg-slate-950 border border-slate-800 rounded-xl py-3 px-4 text-xs text-slate-200 focus:border-emerald-500 outline-none transition-all"
                    >
                        {PERIODS.map(p => (
                            <option key={p.value} value={p.value}>{p.label}</option>
                        ))}
                    </select>
                </div>
                <button
                    onClick={handleCompute}
                    disabled={loading}
                    className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white rounded-xl font-bold text-xs transition-all shadow-lg shadow-emerald-500/20 disabled:opacity-50"
                >
                    {loading ? <Loader2 size={14} className="animate-spin" /> : <Grid3X3 size={14} />}
                    Compute
                </button>
            </div>

            {/* Error */}
            {error && (
                <div className="bg-red-500/10 border border-red-500/50 rounded-xl p-3 flex items-center gap-2">
                    <AlertCircle className="text-red-400" size={16} />
                    <p className="text-red-400 text-sm">{error}</p>
                </div>
            )}

            {/* Heatmap */}
            {data && (
                <div className="bg-slate-900/50 border border-slate-800 rounded-2xl overflow-hidden">
                    <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
                        <h3 className="text-lg font-semibold text-white">
                            Correlation Matrix
                        </h3>
                        <span className="text-xs text-slate-500">
                            {data.data_points} data points
                            {data.errors.length > 0 && ` · Failed: ${data.errors.join(', ')}`}
                        </span>
                    </div>
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-slate-800">
                                    <th className="px-4 py-3 text-left font-medium text-slate-400 sticky left-0 bg-slate-900/90 z-10" />
                                    {data.symbols.map(sym => (
                                        <th key={sym} className="px-3 py-3 text-center font-bold text-slate-300 text-xs min-w-[64px]">
                                            {sym}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {data.symbols.map((rowSym, ri) => (
                                    <tr key={rowSym} className="border-b border-slate-800/30">
                                        <td className="px-4 py-3 font-bold text-slate-300 text-xs sticky left-0 bg-slate-900/90 z-10">
                                            {rowSym}
                                        </td>
                                        {data.matrix[ri].map((val, ci) => {
                                            const isDiagonal = ri === ci;
                                            return (
                                                <td
                                                    key={ci}
                                                    className="px-1 py-3 text-center"
                                                    title={`${rowSym} × ${data.symbols[ci]}: ${val.toFixed(4)}`}
                                                >
                                                    <div
                                                        className={`mx-auto rounded-lg py-1.5 px-2 text-xs font-bold ${getTextColor(val)} ${isDiagonal ? 'ring-1 ring-slate-600' : ''}`}
                                                        style={{ backgroundColor: isDiagonal ? 'rgba(100,116,139,0.3)' : getCorrelationColor(val) }}
                                                    >
                                                        {val.toFixed(2)}
                                                    </div>
                                                </td>
                                            );
                                        })}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    {/* Legend */}
                    <div className="px-6 py-3 border-t border-slate-800 flex items-center justify-center gap-2">
                        <span className="text-[10px] text-slate-500 font-bold">-1.0</span>
                        <div className="flex h-3 w-48 rounded-full overflow-hidden">
                            {Array.from({ length: 20 }).map((_, i) => (
                                <div
                                    key={i}
                                    className="flex-1"
                                    style={{ backgroundColor: getCorrelationColor((i - 10) / 10) }}
                                />
                            ))}
                        </div>
                        <span className="text-[10px] text-slate-500 font-bold">+1.0</span>
                    </div>
                </div>
            )}

            {/* Empty state */}
            {!data && !loading && !error && (
                <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-12 text-center">
                    <Grid3X3 className="mx-auto text-slate-600 mb-3" size={32} />
                    <p className="text-slate-400 text-sm">Enter symbols and click Compute to see the correlation matrix</p>
                </div>
            )}
        </div>
    );
};

export default CorrelationHeatmap;
