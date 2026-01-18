'use client'
import React, { useEffect, useState } from 'react';
import {
    Activity,
    AlertTriangle,
    ArrowRight,
    BrainCircuit,
    CheckCircle2,
    Gauge,
    History,
    LineChart,
    RefreshCw,
    Search,
    ShieldAlert,
    TrendingUp,
    Zap,
    Loader2
} from "lucide-react";
import {
    Area,
    AreaChart,
    CartesianGrid,
    Cell,
    Line,
    LineChart as RechartsLineChart,
    Pie,
    PieChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from 'recharts';
import { regime } from '@/utils/api';
import { CurrentRegimeResponse, RegimeData } from '@/types/api';

const RegimeDetector = () => {
    const [selectedSymbol, setSelectedSymbol] = useState('SPY');
    const [isLoading, setIsLoading] = useState(false);
    const [regimeData, setRegimeData] = useState<CurrentRegimeResponse | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        fetchRegimeData();
    }, []); // Initial load

    const fetchRegimeData = async () => {
        setIsLoading(true);
        setError(null);
        try {
            const response = await regime.detect(selectedSymbol);
            if (response.data) {
                setRegimeData(response.data);
            }
        } catch (err) {
            console.error("Failed to fetch regime data:", err);
            setError("Failed to analyze market regime. Please try again.");
        } finally {
            setIsLoading(false);
        }
    };

    const getRegimeColor = (regimeName: string) => {
        const lower = regimeName.toLowerCase();
        if (lower.includes('bull')) return 'text-emerald-400';
        if (lower.includes('bear')) return 'text-red-400';
        if (lower.includes('volat')) return 'text-amber-400';
        return 'text-blue-400';
    };

    const getRegimeBg = (regimeName: string) => {
        const lower = regimeName.toLowerCase();
        if (lower.includes('bull')) return 'bg-emerald-500/10 border-emerald-500/30';
        if (lower.includes('bear')) return 'bg-red-500/10 border-red-500/30';
        if (lower.includes('volat')) return 'bg-amber-500/10 border-amber-500/30';
        return 'bg-blue-500/10 border-blue-500/30';
    };

    return (
        <div className="space-y-6 animate-in fade-in duration-700">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div className="flex items-center gap-4">
                    <div className="p-3 bg-gradient-to-br from-indigo-500/20 to-violet-500/20 rounded-2xl border border-indigo-500/30">
                        <Activity className="text-indigo-400" size={32} />
                    </div>
                    <div>
                        <h1 className="text-3xl font-bold text-slate-100 tracking-tight">Market Regime Detector</h1>
                        <p className="text-sm text-slate-500 font-medium">AI-powered trend classification and anomaly detection</p>
                    </div>
                </div>

                <div className="flex items-center gap-2 bg-slate-900/50 p-1.5 rounded-xl border border-slate-800/50">
                    <div className="relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={16} />
                        <input
                            type="text"
                            value={selectedSymbol}
                            onChange={(e) => setSelectedSymbol(e.target.value.toUpperCase())}
                            onKeyDown={(e) => e.key === 'Enter' && fetchRegimeData()}
                            className="pl-9 pr-4 py-2 bg-slate-950 border border-slate-800 rounded-lg text-sm text-slate-200 outline-none focus:border-indigo-500 w-32 transition-all"
                            placeholder="Symbol"
                        />
                    </div>
                    <button
                        onClick={fetchRegimeData}
                        disabled={isLoading}
                        className="p-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-white transition-colors disabled:opacity-50"
                    >
                        {isLoading ? <Loader2 size={18} className="animate-spin" /> : <RefreshCw size={18} />}
                    </button>
                </div>
            </div>

            {error && (
                <div className="bg-red-500/10 border border-red-500/30 p-4 rounded-xl text-red-400 text-sm flex items-center gap-2">
                    <AlertTriangle size={16} />
                    {error}
                </div>
            )}

            {/* Main Analysis */}
            {regimeData ? (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Current Regime Card */}
                    <div className="lg:col-span-2 space-y-6">
                        <div className={`p-8 rounded-3xl border ${getRegimeBg(regimeData.current_regime.name)} relative overflow-hidden transition-all duration-500`}>
                            {/* Background Elements */}
                            <div className="absolute top-0 right-0 p-8 opacity-10">
                                <Activity size={120} />
                            </div>

                            <div className="relative z-10">
                                <div className="flex items-center gap-2 mb-2 text-sm font-bold uppercase tracking-wider opacity-70">
                                    <div className="w-2 h-2 rounded-full bg-current animate-pulse" />
                                    Detected Regime
                                </div>
                                <h2 className={`text-5xl font-black ${getRegimeColor(regimeData.current_regime.name)} mb-4 tracking-tight`}>
                                    {regimeData.current_regime.name}
                                </h2>
                                <p className="text-slate-300 text-lg max-w-xl leading-relaxed">
                                    {regimeData.current_regime.description}
                                </p>

                                <div className="mt-8 flex gap-6">
                                    <div>
                                        <div className="text-xs text-slate-500 font-bold uppercase mb-1">Confidence Score</div>
                                        <div className="text-2xl font-bold text-slate-100 flex items-center gap-1">
                                            {(regimeData.current_regime.confidence * 100).toFixed(1)}%
                                            <ShieldAlert className="text-emerald-500" size={16} />
                                        </div>
                                    </div>
                                    <div>
                                        <div className="text-xs text-slate-500 font-bold uppercase mb-1">Duration Est.</div>
                                        <div className="text-2xl font-bold text-slate-100 flex items-center gap-1">
                                            12 Days
                                            <History className="text-indigo-500" size={16} />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Metrics Grid */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="bg-slate-900/50 p-5 rounded-2xl border border-slate-800/50 hover:border-indigo-500/30 transition-colors">
                                <div className="flex items-center justify-between mb-3">
                                    <div className="p-2 bg-indigo-500/10 rounded-lg">
                                        <Zap className="text-indigo-400" size={18} />
                                    </div>
                                    <span className="text-xs font-bold text-slate-500">Volatility</span>
                                </div>
                                <div className="text-2xl font-bold text-slate-200">{regimeData.current_regime.metrics.volatility.toFixed(2)}</div>
                                <div className="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
                                    <div className="h-full bg-indigo-500 rounded-full" style={{ width: `${regimeData.current_regime.metrics.volatility * 100}%` }} />
                                </div>
                            </div>
                            <div className="bg-slate-900/50 p-5 rounded-2xl border border-slate-800/50 hover:border-emerald-500/30 transition-colors">
                                <div className="flex items-center justify-between mb-3">
                                    <div className="p-2 bg-emerald-500/10 rounded-lg">
                                        <TrendingUp className="text-emerald-400" size={18} />
                                    </div>
                                    <span className="text-xs font-bold text-slate-500">Trend Strength</span>
                                </div>
                                <div className="text-2xl font-bold text-slate-200">{regimeData.current_regime.metrics.trend_strength.toFixed(2)}</div>
                                <div className="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
                                    <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${regimeData.current_regime.metrics.trend_strength * 100}%` }} />
                                </div>
                            </div>
                            <div className="bg-slate-900/50 p-5 rounded-2xl border border-slate-800/50 hover:border-blue-500/30 transition-colors">
                                <div className="flex items-center justify-between mb-3">
                                    <div className="p-2 bg-blue-500/10 rounded-lg">
                                        <Activity className="text-blue-400" size={18} />
                                    </div>
                                    <span className="text-xs font-bold text-slate-500">Liquidity Score</span>
                                </div>
                                <div className="text-2xl font-bold text-slate-200">{regimeData.current_regime.metrics.liquidity_score.toFixed(2)}</div>
                                <div className="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
                                    <div className="h-full bg-blue-500 rounded-full" style={{ width: `${regimeData.current_regime.metrics.liquidity_score * 100}%` }} />
                                </div>
                            </div>
                            <div className="bg-slate-900/50 p-5 rounded-2xl border border-slate-800/50 hover:border-amber-500/30 transition-colors">
                                <div className="flex items-center justify-between mb-3">
                                    <div className="p-2 bg-amber-500/10 rounded-lg">
                                        <BrainCircuit className="text-amber-400" size={18} />
                                    </div>
                                    <span className="text-xs font-bold text-slate-500">Correlation</span>
                                </div>
                                <div className="text-2xl font-bold text-slate-200">{regimeData.current_regime.metrics.correlation_index.toFixed(2)}</div>
                                <div className="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
                                    <div className="h-full bg-amber-500 rounded-full" style={{ width: `${regimeData.current_regime.metrics.correlation_index * 100}%` }} />
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Sidebar Stats */}
                    <div className="space-y-6">
                        <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                            <h3 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
                                <Gauge size={16} className="text-indigo-400" />
                                Market Health Score
                            </h3>
                            <div className="relative flex items-center justify-center p-6">
                                <div className="absolute inset-0 flex items-center justify-center">
                                    <div className="text-4xl font-black text-slate-100">{regimeData.market_health_score.toFixed(0)}</div>
                                </div>
                                {/* Simple circular progress using conic-gradient as simple visualization */}
                                <div
                                    className="w-32 h-32 rounded-full"
                                    style={{
                                        background: `conic-gradient(#6366f1 ${regimeData.market_health_score}%, #1e293b 0)`
                                    }}
                                >

                                </div>
                                <div className="absolute w-28 h-28 bg-slate-900/90 rounded-full" />
                            </div>
                            <div className="text-center mt-2">
                                <div className="text-xs font-bold text-slate-500 uppercase tracking-wider">Overall Condition</div>
                                <div className="text-sm text-indigo-400 mt-1 font-medium">Stable Growth</div>
                            </div>
                        </div>

                        <div className="bg-gradient-to-br from-indigo-900/20 to-purple-900/20 border border-indigo-500/20 rounded-2xl p-6">
                            <h3 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
                                <BrainCircuit size={16} className="text-purple-400" />
                                AI Insight
                            </h3>
                            <p className="text-xs text-slate-300 leading-relaxed">
                                Our models detect a shift towards {regimeData.current_regime.name}. Volatility is stabilizing, suggesting a favorable environment for trend-following strategies. Consider reducing hedging positions.
                            </p>
                        </div>
                    </div>
                </div>
            ) : (
                <div className="flex flex-col items-center justify-center h-[400px] text-slate-500">
                    <Activity size={48} className="mb-4 opacity-20" />
                    <p>Enter a symbol and scan to detect market regime</p>
                </div>
            )}
        </div>
    );
};

export default RegimeDetector;