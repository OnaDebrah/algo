'use client'
import React, {useEffect, useState} from 'react';
import {
    Activity,
    AlertTriangle,
    BrainCircuit,
    Calendar,
    ChartNoAxesCombined,
    Gauge,
    History,
    Layers,
    Loader2,
    PieChart,
    RefreshCw,
    Search,
    ShieldAlert,
    Target,
    TrendingUp,
    Zap
} from "lucide-react";
import {regime} from '@/utils/api';
import {
    AllocationResponse,
    CurrentRegimeResponse,
    FeatureImportance,
    FeaturesResponse, RegimeBatchResponse, RegimeData, RegimeHistoryResponse, RegimeReportResponse,
    RegimeStrengthResponse, RegimeTrainResponse, RegimeWarning, RegimeWarningResponse,
    StrategyAllocation,
    TransitionProbability,
    TransitionResponse
} from '@/types/all_types';
import {formatPercent} from "@/utils/formatters";

const RegimeDetector = () => {
    const [selectedSymbol, setSelectedSymbol] = useState('SPY');
    const [isLoading, setIsLoading] = useState(false);
    const [isLoadingHistory, setIsLoadingHistory] = useState(false);
    const [isLoadingBatch, setIsLoadingBatch] = useState(false);
    const [isTraining, setIsTraining] = useState(false);
    const [activeTab, setActiveTab] = useState<'detect' | 'history' | 'analysis' | 'advanced'>('detect');

    const [regimeData, setRegimeData] = useState<CurrentRegimeResponse | null>(null);
    const [regimeHistory, setRegimeHistory] = useState<RegimeHistoryResponse['history']>([]);
    const [regimeReport, setRegimeReport] = useState<RegimeReportResponse['report'] | null>(null);
    const [regimeWarning, setRegimeWarning] = useState<RegimeWarning | null>(null);
    const [allocationData, setAllocationData] = useState<AllocationResponse | null>(null);
    const [strengthData, setStrengthData] = useState<RegimeStrengthResponse | null>(null);
    const [transitionData, setTransitionData] = useState<TransitionResponse | null>(null);
    const [featuresData, setFeaturesData] = useState<FeaturesResponse | null>(null);
    const [batchResults, setBatchResults] = useState<RegimeBatchResponse['results']>([]);

    const [symbolsInput, setSymbolsInput] = useState('SPY,QQQ,IWM');
    const [error, setError] = useState<string | null>(null);
    const [period, setPeriod] = useState('2y');

    // Initial load
    useEffect(() => {
        fetchAllData();
    }, []);

    const fetchAllData = async () => {
        await fetchRegimeData();
        await fetchRegimeHistory();
        await fetchRegimeReport();
        await fetchRegimeWarning();
        await fetchAllocation();
        await fetchStrength();
        await fetchTransitions();
        await fetchFeatures();
    };

    const fetchRegimeData = async () => {
        setIsLoading(true);
        setError(null);
        try {
            const response = await regime.detect(selectedSymbol, { period });
            setRegimeData(response);
        } catch (err) {
            console.error("Failed to fetch regime data:", err);
            setError("Failed to detect market regime.");
        } finally {
            setIsLoading(false);
        }
    };

    const fetchRegimeHistory = async () => {
        setIsLoadingHistory(true);
        try {
            const response = await regime.getHistory(selectedSymbol, { period });
            setRegimeHistory(response.history || []);
        } catch (err) {
            console.error("Failed to fetch regime history:", err);
        } finally {
            setIsLoadingHistory(false);
        }
    };

    const fetchRegimeReport = async () => {
        try {
            const response = await regime.getReport(selectedSymbol, { period });
            setRegimeReport(response.report);
        } catch (err) {
            console.error("Failed to fetch regime report:", err);
        }
    };

    const fetchRegimeWarning = async () => {
        try {
            const response = await regime.getWarning(selectedSymbol, {period}) as unknown as RegimeWarningResponse;
            setRegimeWarning(response.warning);
        } catch (err) {
            console.error("Failed to fetch regime warnings:", err);
        }
    };

    const fetchAllocation = async () => {
        try {
            const response = await regime.getAllocation(selectedSymbol, { period });
            setAllocationData(response);
        } catch (err) {
            console.error("Failed to fetch allocation data:", err);
        }
    };

    const fetchStrength = async () => {
        try {
            const response = await regime.getStrength(selectedSymbol, { period });
            setStrengthData(response);
        } catch (err) {
            console.error("Failed to fetch regime strength:", err);
        }
    };

    const fetchTransitions = async () => {
        try {
            const response = await regime.getTransitions(selectedSymbol, { period });
            setTransitionData(response);
        } catch (err) {
            console.error("Failed to fetch transition data:", err);
        }
    };

    const fetchFeatures = async () => {
        try {
            const response = await regime.getFeatures(selectedSymbol, { period });
            setFeaturesData(response);
        } catch (err) {
            console.error("Failed to fetch feature analysis:", err);
        }
    };

    const detectBatchRegimes = async () => {
        setIsLoadingBatch(true);
        setError(null);
        try {
            const symbols = symbolsInput.split(',').map(s => s.trim().toUpperCase()).filter(s => s);
            const response = await regime.detectBatch(symbols, {period}) as unknown as RegimeBatchResponse;
            setBatchResults(response.results);
        } catch (err) {
            console.error("Failed to detect batch regimes:", err);
            setError("Failed to detect batch regimes.");
        } finally {
            setIsLoadingBatch(false);
        }
    };

    const trainModel = async () => {
        setIsTraining(true);
        try {
            const response = await regime.trainModel(selectedSymbol, {period: '5y'}) as unknown as RegimeTrainResponse;
            if (response.success) {
                alert(`Model trained successfully! Model ID: ${response.model_id}`);
            }
        } catch (err) {
            console.error("Failed to train model:", err);
            alert("Failed to train model.");
        } finally {
            setIsTraining(false);
        }
    };

    const clearCache = async () => {
        try {
            await regime.clearDetectorCache(selectedSymbol);
            alert(`Cache cleared for ${selectedSymbol}`);
        } catch (err) {
            console.error("Failed to clear cache:", err);
        }
    };

    const clearAllCache = async () => {
        try {
            await regime.clearAllCache();
            alert("All cache cleared successfully");
        } catch (err) {
            console.error("Failed to clear all cache:", err);
        }
    };

    const getRegimeColor = (regimeName: string) => {
        const lower = regimeName.toLowerCase();
        if (lower.includes('bull') || lower.includes('recovery')) return 'text-emerald-400';
        if (lower.includes('bear') || lower.includes('crisis')) return 'text-red-400';
        if (lower.includes('volat')) return 'text-amber-400';
        if (lower.includes('mean') || lower.includes('revert')) return 'text-blue-400';
        if (lower.includes('low')) return 'text-slate-400';
        return 'text-indigo-400';
    };

    const getRegimeBg = (regimeName: string) => {
        const lower = regimeName.toLowerCase();
        if (lower.includes('bull') || lower.includes('recovery')) return 'bg-emerald-500/10 border-emerald-500/30';
        if (lower.includes('bear') || lower.includes('crisis')) return 'bg-red-500/10 border-red-500/30';
        if (lower.includes('volat')) return 'bg-amber-500/10 border-amber-500/30';
        if (lower.includes('mean') || lower.includes('revert')) return 'bg-blue-500/10 border-blue-500/30';
        if (lower.includes('low')) return 'bg-slate-500/10 border-slate-500/30';
        return 'bg-indigo-500/10 border-indigo-500/30';
    };

    const getWarningColor = (level: string) => {
        switch(level) {
            case 'high': return 'text-red-400 bg-red-500/10';
            case 'medium': return 'text-amber-400 bg-amber-500/10';
            case 'low': return 'text-yellow-400 bg-yellow-500/10';
            default: return 'text-slate-400 bg-slate-500/10';
        }
    };

    const formatPercentage = (value: number) => {
        return (value * 100).toFixed(1) + '%';
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
                        <h1 className="text-3xl font-bold text-slate-100 tracking-tight">Market Regime Intelligence</h1>
                        <p className="text-sm text-slate-500 font-medium">Advanced AI-powered market state analysis</p>
                    </div>
                </div>

                <div className="flex flex-col md:flex-row gap-3">
                    <div className="flex items-center gap-2 bg-slate-900/50 p-1.5 rounded-xl border border-slate-800/50">
                        <div className="relative">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={16} />
                            <input
                                type="text"
                                value={selectedSymbol}
                                onChange={(e) => setSelectedSymbol(e.target.value.toUpperCase())}
                                onKeyDown={(e) => e.key === 'Enter' && fetchAllData()}
                                className="pl-9 pr-4 py-2 bg-slate-950 border border-slate-800 rounded-lg text-sm text-slate-200 outline-none focus:border-indigo-500 w-32 transition-all"
                                placeholder="Symbol"
                            />
                        </div>
                        <select
                            value={period}
                            onChange={(e) => setPeriod(e.target.value)}
                            className="px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-sm text-slate-200 outline-none focus:border-indigo-500"
                        >
                            <option value="3mo">3 Months</option>
                            <option value="6mo">6 Months</option>
                            <option value="1y">1 Year</option>
                            <option value="2y">2 Years</option>
                            <option value="5y">5 Years</option>
                        </select>
                        <button
                            onClick={fetchAllData}
                            disabled={isLoading}
                            className="p-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-white transition-colors disabled:opacity-50"
                        >
                            {isLoading ? <Loader2 size={18} className="animate-spin" /> : <RefreshCw size={18} />}
                        </button>
                    </div>

                    <div className="flex gap-2">
                        <button
                            onClick={trainModel}
                            disabled={isTraining}
                            className="px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 rounded-lg text-white text-sm font-medium transition-all disabled:opacity-50 flex items-center gap-2"
                        >
                            {isTraining ? <Loader2 size={16} className="animate-spin" /> : <BrainCircuit size={16} />}
                            Train AI Model
                        </button>
                        <button
                            onClick={clearCache}
                            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 text-sm font-medium transition-colors"
                        >
                            Clear Cache
                        </button>
                    </div>
                </div>
            </div>

            {/* Error Display */}
            {error && (
                <div className="bg-red-500/10 border border-red-500/30 p-4 rounded-xl text-red-400 text-sm flex items-center gap-2">
                    <AlertTriangle size={16} />
                    {error}
                </div>
            )}

            {/* Tabs.tsx */}
            <div className="flex gap-2 p-1 bg-slate-900/50 rounded-xl border border-slate-800/50">
                <button
                    onClick={() => setActiveTab('detect')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'detect' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'}`}
                >
                    <div className="flex items-center gap-2">
                        <Activity size={16} />
                        Current Regime
                    </div>
                </button>
                <button
                    onClick={() => setActiveTab('history')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'history' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'}`}
                >
                    <div className="flex items-center gap-2">
                        <History size={16} />
                        History & Report
                    </div>
                </button>
                <button
                    onClick={() => setActiveTab('analysis')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'analysis' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'}`}
                >
                    <div className="flex items-center gap-2">
                        <ChartNoAxesCombined size={16} />
                        Advanced Analysis
                    </div>
                </button>
                <button
                    onClick={() => setActiveTab('advanced')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'advanced' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'}`}
                >
                    <div className="flex items-center gap-2">
                        <BrainCircuit size={16} />
                        Batch & ML
                    </div>
                </button>
            </div>

            {/* Main Content */}
            {activeTab === 'detect' && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Current Regime Card */}
                    <div className="lg:col-span-2 space-y-6">
                        {regimeData ? (
                            <>
                                <div className={`p-8 rounded-3xl border ${getRegimeBg(regimeData.current_regime.name)} relative overflow-hidden transition-all duration-500`}>
                                    <div className="absolute top-0 right-0 p-8 opacity-10">
                                        <Activity size={120} />
                                    </div>

                                    <div className="relative z-10">
                                        <div className="flex items-center justify-between mb-2">
                                            <div className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider opacity-70">
                                                <div className="w-2 h-2 rounded-full bg-current animate-pulse" />
                                                Detected Regime
                                            </div>
                                            {regimeWarning?.has_warning && (
                                                <div className={`px-3 py-1 rounded-full text-xs font-bold ${getWarningColor(regimeWarning.warning_level)}`}>
                                                    ⚠️ {regimeWarning.warning_level.toUpperCase()} WARNING
                                                </div>
                                            )}
                                        </div>
                                        <h2 className={`text-5xl font-black ${getRegimeColor(regimeData.current_regime.name)} mb-4 tracking-tight`}>
                                            {regimeData.current_regime.name}
                                        </h2>
                                        <p className="text-slate-300 text-lg max-w-xl leading-relaxed">
                                            {regimeData.current_regime.description}
                                        </p>

                                        <div className="mt-8 flex flex-wrap gap-6">
                                            <div>
                                                <div className="text-xs text-slate-500 font-bold uppercase mb-1">Confidence Score</div>
                                                <div className="text-2xl font-bold text-slate-100 flex items-center gap-1">
                                                    {formatPercentage(regimeData.current_regime.confidence)}
                                                    <ShieldAlert className="text-emerald-500" size={16} />
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-xs text-slate-500 font-bold uppercase mb-1">Regime Strength</div>
                                                <div className="text-2xl font-bold text-slate-100 flex items-center gap-1">
                                                    {strengthData ? formatPercentage(strengthData.strength) : '0%'}
                                                    <Gauge className="text-indigo-500" size={16} />
                                                </div>
                                            </div>
                                            {transitionData && (
                                                <div>
                                                    <div className="text-xs text-slate-500 font-bold uppercase mb-1">Expected Duration</div>
                                                    <div className="text-2xl font-bold text-slate-100 flex items-center gap-1">
                                                        {Math.round(transitionData.expected_duration)} Days
                                                        <Calendar className="text-blue-500" size={16} />
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                {/* Warning Details */}
                                {regimeWarning?.has_warning && (
                                    <div className="bg-gradient-to-r from-amber-900/20 to-red-900/20 border border-amber-500/30 rounded-2xl p-6">
                                        <div className="flex items-center gap-3 mb-4">
                                            <AlertTriangle className="text-amber-400" size={20} />
                                            <h3 className="text-lg font-bold text-slate-200">Regime Change Warning</h3>
                                        </div>
                                        <div className="space-y-3">
                                            <p className="text-slate-300">
                                                Expected transition to: <span className="font-bold">{regimeWarning.expected_transition}</span>
                                            </p>
                                            <p className="text-sm text-slate-400">Timeframe: {regimeWarning.timeframe}</p>
                                            {regimeWarning.reasons.length > 0 && (
                                                <div>
                                                    <p className="text-sm text-slate-400 mb-1">Reasons:</p>
                                                    <ul className="text-sm text-slate-300 space-y-1">
                                                        {regimeWarning.reasons.map((reason, idx) => (
                                                            <li key={idx}>• {reason}</li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            )}
                                            {regimeWarning.suggested_actions.length > 0 && (
                                                <div>
                                                    <p className="text-sm text-slate-400 mb-1">Suggested Actions:</p>
                                                    <ul className="text-sm text-slate-300 space-y-1">
                                                        {regimeWarning.suggested_actions.map((action, idx) => (
                                                            <li key={idx}>• {action}</li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )}

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
                                            <div className="h-full bg-indigo-500 rounded-full" style={{ width: `${Math.min(regimeData.current_regime.metrics.volatility * 100, 100)}%` }} />
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
                                            <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${Math.min(regimeData.current_regime.metrics.trend_strength * 100, 100)}%` }} />
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
                                            <div className="h-full bg-blue-500 rounded-full" style={{ width: `${Math.min(regimeData.current_regime.metrics.liquidity_score * 100, 100)}%` }} />
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
                                            <div className="h-full bg-amber-500 rounded-full" style={{ width: `${Math.min(regimeData.current_regime.metrics.correlation_index * 100, 100)}%` }} />
                                        </div>
                                    </div>
                                </div>
                            </>
                        ) : (
                            <div className="flex flex-col items-center justify-center h-[400px] text-slate-500">
                                <Activity size={48} className="mb-4 opacity-20" />
                                <p>Enter a symbol and scan to detect market regime</p>
                            </div>
                        )}
                    </div>

                    {/* Sidebar Stats */}
                    <div className="space-y-6">
                        {/* Market Health Score */}
                        <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                            <h3 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
                                <Gauge size={16} className="text-indigo-400" />
                                Market Health Score
                            </h3>
                            <div className="relative flex items-center justify-center p-6">
                                <div className="absolute inset-0 flex items-center justify-center">
                                    <div className="text-4xl font-black text-slate-100">{regimeData?.market_health_score.toFixed(0) || '0'}</div>
                                </div>
                                <div
                                    className="w-32 h-32 rounded-full"
                                    style={{
                                        background: `conic-gradient(#6366f1 ${regimeData?.market_health_score || 0}%, #1e293b 0)`
                                    }}
                                />
                                <div className="absolute w-28 h-28 bg-slate-900/90 rounded-full" />
                            </div>
                            <div className="text-center mt-2">
                                <div className="text-xs font-bold text-slate-500 uppercase tracking-wider">Overall Condition</div>
                                <div className="text-sm text-indigo-400 mt-1 font-medium">
                                    {regimeData?.market_health_score && regimeData.market_health_score > 80 ? 'Excellent' :
                                     regimeData?.market_health_score && regimeData.market_health_score > 60 ? 'Good' :
                                     regimeData?.market_health_score && regimeData.market_health_score > 40 ? 'Moderate' :
                                     'Poor'}
                                </div>
                            </div>
                        </div>

                        {/* Regime Strength */}
                        {strengthData && (
                            <div className="bg-gradient-to-br from-indigo-900/20 to-purple-900/20 border border-indigo-500/20 rounded-2xl p-6">
                                <h3 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
                                    <Target size={16} className="text-purple-400" />
                                    Regime Strength Analysis
                                </h3>
                                <div className="space-y-3">
                                    <div>
                                        <div className="flex justify-between text-sm mb-1">
                                            <span className="text-slate-400">Strength Score</span>
                                            <span className="text-slate-200 font-bold">{formatPercentage(strengthData.strength)}</span>
                                        </div>
                                        <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                                            <div className="h-full bg-gradient-to-r from-indigo-500 to-purple-500" style={{ width: `${strengthData.strength * 100}%` }} />
                                        </div>
                                    </div>
                                    <div>
                                        <div className="flex justify-between text-sm mb-1">
                                            <span className="text-slate-400">Confirming Signals</span>
                                            <span className="text-slate-200 font-bold">{strengthData.confirming_signals}/{strengthData.total_signals}</span>
                                        </div>
                                        <div className="text-xs text-slate-400 mt-1">{strengthData.description}</div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Quick Actions */}
                        <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                            <h3 className="text-sm font-bold text-slate-300 mb-4">Quick Actions</h3>
                            <div className="space-y-2">
                                <button
                                    onClick={fetchRegimeHistory}
                                    disabled={isLoadingHistory}
                                    className="w-full px-3 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 text-sm font-medium transition-colors flex items-center justify-center gap-2"
                                >
                                    {isLoadingHistory ? <Loader2 size={14} className="animate-spin" /> : <History size={14} />}
                                    Refresh History
                                </button>
                                <button
                                    onClick={() => setActiveTab('analysis')}
                                    className="w-full px-3 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 rounded-lg text-white text-sm font-medium transition-all flex items-center justify-center gap-2"
                                >
                                    <ChartNoAxesCombined size={14} />
                                    View Full Analysis
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* History & Report Tab */}
            {activeTab === 'history' && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Regime History */}
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                        <div className="flex items-center justify-between mb-6">
                            <h3 className="text-lg font-bold text-slate-200 flex items-center gap-2">
                                <History size={20} className="text-indigo-400" />
                                Regime History
                            </h3>
                            <span className="text-sm text-slate-500">
                                {regimeHistory.length} entries
                            </span>
                        </div>
                        {isLoadingHistory ? (
                            <div className="flex items-center justify-center h-64">
                                <Loader2 size={24} className="animate-spin text-indigo-400" />
                            </div>
                        ) : regimeHistory.length > 0 ? (
                            <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2">
                                {regimeHistory.slice().reverse().map((entry, idx) => (
                                    <div key={idx} className={`p-4 rounded-xl border ${getRegimeBg(entry.regime)}`}>
                                        <div className="flex justify-between items-start mb-2">
                                            <span className={`font-bold ${getRegimeColor(entry.regime)}`}>
                                                {entry.regime}
                                            </span>
                                            <span className="text-sm text-slate-400">
                                                {new Date(entry.timestamp).toLocaleDateString()}
                                            </span>
                                        </div>
                                        <div className="flex justify-between text-sm">
                                            <span className="text-slate-400">Confidence: {formatPercentage(entry.confidence)}</span>
                                            <span className="text-slate-400">Strength: {formatPercentage(entry.strength)}</span>
                                        </div>
                                        <div className="text-xs text-slate-500 mt-2">
                                            Method: {entry.method}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="text-center py-12 text-slate-500">
                                No history data available
                            </div>
                        )}
                    </div>

                    {/* Regime Report */}
                    <div className="space-y-6">
                        {regimeReport ? (
                            <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                                <h3 className="text-lg font-bold text-slate-200 mb-6 flex items-center gap-2">
                                    <BrainCircuit size={20} className="text-purple-400" />
                                    Comprehensive Report
                                </h3>
                                <div className="space-y-4">
                                    <div className="grid grid-cols-2 gap-3 mb-4">
                                        <div className="bg-slate-800/50 p-3 rounded-lg">
                                            <div className="text-xs text-slate-500">Current Strength</div>
                                            <div className="text-lg font-bold text-slate-200">
                                                {formatPercent(regimeReport.current_strength)}
                                            </div>
                                        </div>
                                        <div className="bg-slate-800/50 p-3 rounded-lg">
                                            <div className="text-xs text-slate-500">History Length</div>
                                            <div className="text-lg font-bold text-slate-200">
                                                {regimeReport.history_length}
                                            </div>
                                        </div>
                                    </div>

                                    {regimeReport.regime_statistics?.counts && (
                                        <div>
                                            <h4 className="text-sm font-bold text-slate-400 mb-2">Regime Distribution</h4>
                                            <div className="space-y-2">
                                                {Object.entries(regimeReport.regime_statistics.counts).map(([regime, count]) => (
                                                    <div key={regime} className="flex items-center justify-between">
                                                        <span className="text-sm text-slate-300">{regime}</span>
                                                        <div className="flex items-center gap-2">
                                                            <span className="text-sm font-bold text-slate-200">{count}</span>
                                                            <div className="w-24 bg-slate-800 h-2 rounded-full overflow-hidden">
                                                                <div className={`h-full ${getRegimeBg(regime).split(' ')[0]}`} style={{
                                                                    width: `${((count as number) / regimeReport.history_length) * 100}%`
                                                                }} />
                                                            </div>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {regimeReport.duration_prediction && (
                                        <div>
                                            <h4 className="text-sm font-bold text-slate-400 mb-2">Duration Predictions</h4>
                                            <div className="grid grid-cols-2 gap-3">
                                                <div className="bg-slate-800/50 p-3 rounded-lg">
                                                    <div className="text-xs text-slate-500">Expected</div>
                                                    <div className="text-lg font-bold text-slate-200">
                                                        {Math.round(regimeReport.duration_prediction.expected_duration)} days
                                                    </div>
                                                </div>
                                                <div className="bg-slate-800/50 p-3 rounded-lg">
                                                    <div className="text-xs text-slate-500">End Probability</div>
                                                    <div className="text-lg font-bold text-slate-200">
                                                        {formatPercent(regimeReport.duration_prediction.probability_end_next_week)}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        ) : (
                            <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                                <div className="text-center py-12 text-slate-500">
                                    No report data available
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Advanced Analysis Tab */}
            {activeTab === 'analysis' && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Strategy Allocation */}
                    {allocationData && (
                        <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                            <div className="flex items-center justify-between mb-6">
                                <h3 className="text-lg font-bold text-slate-200 flex items-center gap-2">
                                    <PieChart size={20} className="text-emerald-400" />
                                    Strategy Allocation
                                </h3>
                                <div className="text-right">
                                    <div className="text-sm text-slate-500">{allocationData.current_regime}</div>
                                    <div className="text-xs text-slate-400">Confidence: {formatPercent(allocationData.confidence)}</div>
                                </div>
                            </div>
                            <div className="space-y-4">
                                {allocationData.allocation && Object.entries(allocationData.allocation).map(([strategy, percentage]) => (
                                    <div key={strategy} className="space-y-1">
                                        <div className="flex justify-between text-sm">
                                            <span className="text-slate-300 capitalize">{strategy.replace(/_/g, ' ')}</span>
                                            <span className="text-slate-200 font-bold">{formatPercent(percentage)}</span>
                                        </div>
                                        <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                                            <div className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500" style={{ width: `${percentage * 100}%` }} />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Transition Probabilities */}
                    {transitionData && (
                        <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                            <h3 className="text-lg font-bold text-slate-200 mb-6 flex items-center gap-2">
                                <Layers size={20} className="text-blue-400" />
                                Transition Probabilities
                            </h3>
                            <div className="space-y-4">
                                <div className="grid grid-cols-2 gap-3 mb-4">
                                    <div className="bg-slate-800/50 p-3 rounded-lg">
                                        <div className="text-xs text-slate-500">Expected Duration</div>
                                        <div className="text-lg font-bold text-slate-200">
                                            {Math.round(transitionData.expected_duration)} days
                                        </div>
                                    </div>
                                    <div className="bg-slate-800/50 p-3 rounded-lg">
                                        <div className="text-xs text-slate-500">End Next Week</div>
                                        <div className="text-lg font-bold text-slate-200">
                                            {formatPercentage(transitionData.probability_end_next_week)}
                                        </div>
                                    </div>
                                </div>
                                {transitionData.likely_transitions?.map((transition: TransitionProbability, idx) => (
                                    <div key={idx} className="bg-slate-800/30 p-4 rounded-xl border border-slate-700/50">
                                        <div className="flex justify-between items-center mb-2">
                                            <div className="flex items-center gap-2">
                                                <span className="text-slate-300">{transition.from_regime}</span>
                                                <span className="text-slate-500">→</span>
                                                <span className={`font-bold ${getRegimeColor(transition.to_regime)}`}>
                                                    {transition.to_regime}
                                                </span>
                                            </div>
                                            <span className="text-lg font-bold text-slate-200">
                                                {formatPercentage(transition.probability)}
                                            </span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Feature Analysis */}
                    {featuresData && (
                        <div className="lg:col-span-2 bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                            <h3 className="text-lg font-bold text-slate-200 mb-6 flex items-center gap-2">
                                <Target size={20} className="text-amber-400" />
                                Feature Importance Analysis
                            </h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                {featuresData.top_features?.map((feature: FeatureImportance, idx: number) => (
                                    <div key={idx} className="bg-slate-800/30 p-4 rounded-xl border border-slate-700/50">
                                        <div className="flex justify-between items-start mb-2">
                                            <span className="text-sm font-bold text-slate-200">
                                                {feature.feature.replace(/_/g, ' ')}
                                            </span>
                                            <span className={`text-xs px-2 py-1 rounded-full ${
                                                feature.importance > 0.7 ? 'bg-emerald-500/20 text-emerald-400' :
                                                feature.importance > 0.4 ? 'bg-amber-500/20 text-amber-400' :
                                                'bg-slate-500/20 text-slate-400'
                                            }`}>
                                                {formatPercentage(feature.importance)}
                                            </span>
                                        </div>
                                        <div className="text-sm text-slate-400 mb-2">
                                            Value: {feature.current_value.toFixed(3)}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Batch & ML Tab */}
            {activeTab === 'advanced' && (
                <div className="space-y-6">
                    {/* Batch Detection */}
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                        <h3 className="text-lg font-bold text-slate-200 mb-6 flex items-center gap-2">
                            <Activity size={20} className="text-indigo-400" />
                            Batch Regime Detection
                        </h3>
                        <div className="space-y-4">
                            <div className="flex flex-col md:flex-row gap-3">
                                <div className="flex-1">
                                    <input
                                        type="text"
                                        value={symbolsInput}
                                        onChange={(e) => setSymbolsInput(e.target.value)}
                                        placeholder="Enter symbols separated by commas (e.g., SPY,QQQ,IWM)"
                                        className="w-full px-4 py-3 bg-slate-950 border border-slate-800 rounded-lg text-slate-200 outline-none focus:border-indigo-500 transition-all"
                                    />
                                </div>
                                <button
                                    onClick={detectBatchRegimes}
                                    disabled={isLoadingBatch}
                                    className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 rounded-lg text-white font-medium transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                                >
                                    {isLoadingBatch ? <Loader2 size={18} className="animate-spin" /> : <BrainCircuit size={18} />}
                                    Detect All
                                </button>
                            </div>

                            {batchResults.length > 0 && (
                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-6">
                                    {batchResults.map((result, idx) => (
                                        <div key={idx} className={`p-4 rounded-xl border ${getRegimeBg(result.regime)}`}>
                                            <div className="flex justify-between items-start mb-2">
                                                <span className="font-bold text-slate-200">{result.symbol}</span>
                                                <span className={`text-sm font-bold ${getRegimeColor(result.regime)}`}>
                                                    {result.regime}
                                                </span>
                                            </div>
                                            <div className="space-y-1 text-sm">
                                                <div className="flex justify-between">
                                                    <span className="text-slate-500">Confidence</span>
                                                    <span className="text-slate-300">{formatPercentage(result.confidence)}</span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-slate-500">Strength</span>
                                                    <span className="text-slate-300">{formatPercentage(result.regime_strength)}</span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-slate-500">Method</span>
                                                    <span className="text-slate-300 capitalize">{result.method}</span>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>

                    {/* ML Training Status */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div className="bg-gradient-to-br from-purple-900/20 to-pink-900/20 border border-purple-500/20 rounded-2xl p-6">
                            <h3 className="text-lg font-bold text-slate-200 mb-4 flex items-center gap-2">
                                <BrainCircuit size={20} className="text-purple-400" />
                                AI Model Status
                            </h3>
                            <div className="space-y-3">
                                <div className="bg-purple-900/30 p-4 rounded-xl">
                                    <div className="text-sm text-slate-400 mb-1">Current Model</div>
                                    <div className="text-lg font-bold text-slate-200">
                                        Random Forest Classifier
                                    </div>
                                </div>
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="bg-slate-800/50 p-3 rounded-lg">
                                        <div className="text-xs text-slate-500">Features Used</div>
                                        <div className="text-lg font-bold text-slate-200">
                                            {featuresData?.top_features?.length || 'N/A'}
                                        </div>
                                    </div>
                                    <div className="bg-slate-800/50 p-3 rounded-lg">
                                        <div className="text-xs text-slate-500">ML Confidence</div>
                                        <div className="text-lg font-bold text-slate-200">
                                            {regimeData?.current_regime.confidence ? formatPercentage(regimeData.current_regime.confidence) : 'N/A'}
                                        </div>
                                    </div>
                                </div>
                                <button
                                    onClick={trainModel}
                                    disabled={isTraining}
                                    className="w-full mt-4 px-4 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 rounded-lg text-white font-medium transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                                >
                                    {isTraining ? <Loader2 size={18} className="animate-spin" /> : <BrainCircuit size={18} />}
                                    {isTraining ? 'Training Model...' : 'Retrain ML Model'}
                                </button>
                            </div>
                        </div>

                        {/* System Information */}
                        <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                            <h3 className="text-lg font-bold text-slate-200 mb-4">System Information</h3>
                            <div className="space-y-3">
                                <div className="flex justify-between items-center py-2 border-b border-slate-800/50">
                                    <span className="text-slate-400">Lookback Period</span>
                                    <span className="text-slate-200 font-mono">252 days</span>
                                </div>
                                <div className="flex justify-between items-center py-2 border-b border-slate-800/50">
                                    <span className="text-slate-400">Confidence Threshold</span>
                                    <span className="text-slate-200 font-mono">70%</span>
                                </div>
                                <div className="flex justify-between items-center py-2 border-b border-slate-800/50">
                                    <span className="text-slate-400">Regime Types</span>
                                    <span className="text-slate-200 font-mono">8</span>
                                </div>
                                <div className="flex justify-between items-center py-2 border-b border-slate-800/50">
                                    <span className="text-slate-400">Cache Status</span>
                                    <span className="text-slate-200 font-mono">Active</span>
                                </div>
                                <div className="flex gap-2 mt-4">
                                    <button
                                        onClick={clearCache}
                                        className="flex-1 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 font-medium transition-colors"
                                    >
                                        Clear Symbol Cache
                                    </button>
                                    <button
                                        onClick={clearAllCache}
                                        className="flex-1 px-4 py-2 bg-red-600/20 hover:bg-red-600/30 text-red-400 font-medium transition-colors rounded-lg"
                                    >
                                        Clear All Cache
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default RegimeDetector;
