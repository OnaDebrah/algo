/* eslint-disable @typescript-eslint/no-explicit-any */

'use client'
import React, {useState} from 'react';
import {
    AlertTriangle,
    ArrowDown,
    ArrowUp,
    CheckCircle2,
    Loader2,
    RefreshCw,
    Search,
    Shield,
    TrendingUp,
    Zap
} from 'lucide-react';
import {options} from '@/utils/api';

interface ArbitrageOpportunity {
    type: string;
    asset: string;
    strike: number | null;
    direction: number;
    confidence: number;
    mispricing: number | null;
    entry_price: number | null;
    size: number | null;
    details: Record<string, any> | null;
}

interface ArbitrageScanResult {
    symbol: string;
    spot_price: number;
    opportunities: ArbitrageOpportunity[];
    vol_surface_summary: Record<string, any> | null;
    greek_exposure: Record<string, any> | null;
    scanned_at: string;
}

const ARB_TYPE_LABELS: Record<string, { label: string; color: string; icon: any }> = {
    volatility: {label: 'Volatility Arb', color: 'text-violet-400', icon: Zap},
    put_call: {label: 'Put-Call Parity', color: 'text-blue-400', icon: Shield},
    term_structure: {label: 'Term Structure', color: 'text-cyan-400', icon: TrendingUp},
    skew: {label: 'Skew Arb', color: 'text-amber-400', icon: AlertTriangle},
    butterfly: {label: 'Butterfly', color: 'text-emerald-400', icon: CheckCircle2},
    box_spread: {label: 'Box Spread', color: 'text-rose-400', icon: Shield},
    calendar: {label: 'Calendar Spread', color: 'text-orange-400', icon: RefreshCw},
};

const ALL_ARB_TYPES = Object.keys(ARB_TYPE_LABELS);

interface ArbitrageTabProps {
    selectedSymbol: string;
    currentPrice: number;
}

const ArbitrageTab: React.FC<ArbitrageTabProps> = ({selectedSymbol, currentPrice}) => {
    const [isScanning, setIsScanning] = useState(false);
    const [scanResult, setScanResult] = useState<ArbitrageScanResult | null>(null);
    const [selectedArbTypes, setSelectedArbTypes] = useState<string[]>(ALL_ARB_TYPES);
    const [volModel, setVolModel] = useState('garch');
    const [entryThreshold, setEntryThreshold] = useState(2.0);
    const [minLiquidity, setMinLiquidity] = useState(1.0);
    const [error, setError] = useState<string | null>(null);

    const toggleArbType = (type: string) => {
        setSelectedArbTypes(prev =>
            prev.includes(type) ? prev.filter(t => t !== type) : [...prev, type]
        );
    };

    const runScan = async () => {
        if (selectedArbTypes.length === 0) {
            setError('Select at least one arbitrage type');
            return;
        }
        setIsScanning(true);
        setError(null);
        try {
            const result = await options.scanArbitrage({
                symbol: selectedSymbol,
                arb_types: selectedArbTypes,
                volatility_model: volModel,
                entry_threshold: entryThreshold,
                min_liquidity: minLiquidity,
            });
            setScanResult(result);
        } catch (err: any) {
            const detail = err?.response?.data?.detail || err?.message || 'Scan failed';
            setError(detail);
        } finally {
            setIsScanning(false);
        }
    };

    const getConfidenceColor = (conf: number) => {
        if (conf >= 0.8) return 'text-emerald-400';
        if (conf >= 0.5) return 'text-amber-400';
        return 'text-red-400';
    };

    const getConfidenceLabel = (conf: number) => {
        if (conf >= 0.8) return 'High';
        if (conf >= 0.5) return 'Medium';
        return 'Low';
    };

    return (
        <div className="space-y-6">
            {/* Scanner Controls */}
            <div className="bg-slate-900/80 border border-slate-700/50 rounded-2xl p-6">
                <div className="flex items-center justify-between mb-6">
                    <div>
                        <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
                            <Search size={22} className="text-amber-400"/>
                            Derivative Arbitrage Scanner
                        </h2>
                        <p className="text-sm text-slate-400 mt-1">
                            Scan {selectedSymbol} options for mispricing and arbitrage opportunities
                        </p>
                    </div>
                    <button
                        onClick={runScan}
                        disabled={isScanning}
                        className="px-6 py-3 bg-gradient-to-r from-amber-500 to-orange-500 text-white font-bold rounded-xl hover:from-amber-600 hover:to-orange-600 transition-all disabled:opacity-50 flex items-center gap-2"
                    >
                        {isScanning ? (
                            <>
                                <Loader2 size={18} className="animate-spin"/>
                                Scanning...
                            </>
                        ) : (
                            <>
                                <Search size={18}/>
                                Scan Now
                            </>
                        )}
                    </button>
                </div>

                {/* Arbitrage Type Filters */}
                <div className="mb-4">
                    <p className="text-xs text-slate-500 font-bold uppercase tracking-wider mb-2">Arbitrage Types</p>
                    <div className="flex flex-wrap gap-2">
                        {ALL_ARB_TYPES.map(type => {
                            const meta = ARB_TYPE_LABELS[type];
                            const isSelected = selectedArbTypes.includes(type);
                            return (
                                <button
                                    key={type}
                                    onClick={() => toggleArbType(type)}
                                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all border ${
                                        isSelected
                                            ? 'bg-slate-700/80 border-slate-600 text-slate-200'
                                            : 'bg-slate-800/50 border-slate-700/30 text-slate-500 hover:text-slate-300'
                                    }`}
                                >
                                    <span className={isSelected ? meta.color : ''}>{meta.label}</span>
                                </button>
                            );
                        })}
                    </div>
                </div>

                {/* Advanced Settings */}
                <div className="grid grid-cols-3 gap-4">
                    <div>
                        <label className="text-xs text-slate-500 font-medium block mb-1">Volatility Model</label>
                        <select
                            value={volModel}
                            onChange={e => setVolModel(e.target.value)}
                            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200"
                        >
                            <option value="garch">GARCH(1,1)</option>
                            <option value="historical">Historical</option>
                            <option value="ewma">EWMA</option>
                            <option value="har">HAR</option>
                            <option value="jump_diffusion">Jump Diffusion</option>
                        </select>
                    </div>
                    <div>
                        <label className="text-xs text-slate-500 font-medium block mb-1">Entry Threshold (sigma)</label>
                        <input
                            type="number"
                            value={entryThreshold}
                            onChange={e => setEntryThreshold(parseFloat(e.target.value) || 2.0)}
                            step={0.5}
                            min={0.5}
                            max={5}
                            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200"
                        />
                    </div>
                    <div>
                        <label className="text-xs text-slate-500 font-medium block mb-1">Min Liquidity Score</label>
                        <input
                            type="number"
                            value={minLiquidity}
                            onChange={e => setMinLiquidity(parseFloat(e.target.value) || 1.0)}
                            step={0.5}
                            min={0}
                            max={10}
                            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200"
                        />
                    </div>
                </div>

                {error && (
                    <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-sm text-red-400">
                        {error}
                    </div>
                )}
            </div>

            {/* Results */}
            {scanResult && (
                <>
                    {/* Summary Cards */}
                    <div className="grid grid-cols-4 gap-4">
                        <div className="bg-slate-900/80 border border-slate-700/50 rounded-xl p-4">
                            <p className="text-xs text-slate-500 mb-1">Spot Price</p>
                            <p className="text-2xl font-bold text-slate-100">${scanResult.spot_price.toFixed(2)}</p>
                        </div>
                        <div className="bg-slate-900/80 border border-slate-700/50 rounded-xl p-4">
                            <p className="text-xs text-slate-500 mb-1">Opportunities Found</p>
                            <p className="text-2xl font-bold text-amber-400">{scanResult.opportunities.length}</p>
                        </div>
                        <div className="bg-slate-900/80 border border-slate-700/50 rounded-xl p-4">
                            <p className="text-xs text-slate-500 mb-1">High Confidence</p>
                            <p className="text-2xl font-bold text-emerald-400">
                                {scanResult.opportunities.filter(o => o.confidence >= 0.8).length}
                            </p>
                        </div>
                        <div className="bg-slate-900/80 border border-slate-700/50 rounded-xl p-4">
                            <p className="text-xs text-slate-500 mb-1">Scanned At</p>
                            <p className="text-sm font-medium text-slate-300">
                                {new Date(scanResult.scanned_at).toLocaleTimeString()}
                            </p>
                        </div>
                    </div>

                    {/* Vol Surface Summary */}
                    {scanResult.vol_surface_summary && (
                        <div className="bg-slate-900/80 border border-slate-700/50 rounded-xl p-4">
                            <h3 className="text-sm font-bold text-slate-300 mb-3">Volatility Surface</h3>
                            <div className="flex gap-6">
                                <div>
                                    <span className="text-xs text-slate-500">ATM IV: </span>
                                    <span className="text-sm font-bold text-violet-400">
                                        {((scanResult.vol_surface_summary.atm_iv || 0) * 100).toFixed(1)}%
                                    </span>
                                </div>
                                <div>
                                    <span className="text-xs text-slate-500">Skew: </span>
                                    <span className="text-sm font-bold text-cyan-400">
                                        {(scanResult.vol_surface_summary.skew || 0).toFixed(4)}
                                    </span>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Opportunities Table */}
                    {scanResult.opportunities.length > 0 ? (
                        <div className="bg-slate-900/80 border border-slate-700/50 rounded-2xl overflow-hidden">
                            <div className="p-4 border-b border-slate-700/50">
                                <h3 className="text-lg font-bold text-slate-100">Detected Opportunities</h3>
                            </div>
                            <div className="overflow-x-auto">
                                <table className="w-full">
                                    <thead>
                                    <tr className="text-xs text-slate-500 uppercase tracking-wider border-b border-slate-700/50">
                                        <th className="text-left p-4">Type</th>
                                        <th className="text-left p-4">Strike</th>
                                        <th className="text-left p-4">Direction</th>
                                        <th className="text-left p-4">Confidence</th>
                                        <th className="text-right p-4">Mispricing</th>
                                        <th className="text-right p-4">Entry Price</th>
                                        <th className="text-left p-4">Details</th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    {scanResult.opportunities.map((opp, idx) => {
                                        const meta = ARB_TYPE_LABELS[opp.type] || {
                                            label: opp.type,
                                            color: 'text-slate-400'
                                        };
                                        return (
                                            <tr key={idx}
                                                className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                                                <td className="p-4">
                                                    <span className={`text-sm font-semibold ${meta.color}`}>
                                                        {meta.label}
                                                    </span>
                                                </td>
                                                <td className="p-4 text-sm text-slate-300">
                                                    {opp.strike != null ? `$${opp.strike.toFixed(2)}` : '-'}
                                                </td>
                                                <td className="p-4">
                                                    {opp.direction > 0 ? (
                                                        <span
                                                            className="flex items-center gap-1 text-emerald-400 text-sm font-medium">
                                                            <ArrowUp size={14}/> Long
                                                        </span>
                                                    ) : opp.direction < 0 ? (
                                                        <span
                                                            className="flex items-center gap-1 text-red-400 text-sm font-medium">
                                                            <ArrowDown size={14}/> Short
                                                        </span>
                                                    ) : (
                                                        <span className="text-slate-500 text-sm">Neutral</span>
                                                    )}
                                                </td>
                                                <td className="p-4">
                                                    <span
                                                        className={`text-sm font-bold ${getConfidenceColor(opp.confidence)}`}>
                                                        {(opp.confidence * 100).toFixed(0)}% {getConfidenceLabel(opp.confidence)}
                                                    </span>
                                                </td>
                                                <td className="p-4 text-right">
                                                    {opp.mispricing != null ? (
                                                        <span className={`text-sm font-medium ${
                                                            opp.mispricing > 0 ? 'text-emerald-400' : 'text-red-400'
                                                        }`}>
                                                            {opp.mispricing > 0 ? '+' : ''}{opp.mispricing.toFixed(4)}
                                                        </span>
                                                    ) : '-'}
                                                </td>
                                                <td className="p-4 text-right text-sm text-slate-300">
                                                    {opp.entry_price != null ? `$${opp.entry_price.toFixed(2)}` : '-'}
                                                </td>
                                                <td className="p-4 text-xs text-slate-500 max-w-[200px] truncate">
                                                    {opp.details ? Object.entries(opp.details).slice(0, 3).map(([k, v]) =>
                                                        `${k}: ${typeof v === 'number' ? v.toFixed(3) : v}`
                                                    ).join(' | ') : '-'}
                                                </td>
                                            </tr>
                                        );
                                    })}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    ) : (
                        <div className="bg-slate-900/80 border border-slate-700/50 rounded-2xl p-12 text-center">
                            <CheckCircle2 size={48} className="text-emerald-400 mx-auto mb-4"/>
                            <h3 className="text-lg font-bold text-slate-100 mb-2">No Arbitrage Detected</h3>
                            <p className="text-sm text-slate-400">
                                The {selectedSymbol} options market appears efficiently priced. Try adjusting the
                                entry threshold or liquidity filter.
                            </p>
                        </div>
                    )}
                </>
            )}

            {/* Empty State */}
            {!scanResult && !isScanning && (
                <div className="bg-slate-900/80 border border-slate-700/50 rounded-2xl p-16 text-center">
                    <div className="w-20 h-20 bg-gradient-to-br from-amber-500/20 to-orange-500/20 rounded-2xl flex items-center justify-center mx-auto mb-6">
                        <Zap size={40} className="text-amber-400"/>
                    </div>
                    <h3 className="text-xl font-bold text-slate-100 mb-3">Derivative Arbitrage Scanner</h3>
                    <p className="text-sm text-slate-400 max-w-md mx-auto mb-6">
                        Scan real-time options chains for mispricing opportunities including volatility arbitrage,
                        put-call parity violations, term structure anomalies, skew trades, butterfly/box spread
                        inefficiencies, and calendar spread opportunities.
                    </p>
                    <button
                        onClick={runScan}
                        className="px-8 py-3 bg-gradient-to-r from-amber-500 to-orange-500 text-white font-bold rounded-xl hover:from-amber-600 hover:to-orange-600 transition-all"
                    >
                        Run First Scan
                    </button>
                </div>
            )}
        </div>
    );
};

export default ArbitrageTab;
