'use client';
import React, { useEffect, useState, useCallback } from 'react';
import {
    Shield, AlertTriangle, TrendingDown, Activity,
    Loader2, RefreshCw, Search, Bell, Settings2,
    ArrowUp, ArrowDown, Minus, ShieldAlert, Zap,
    Target, BarChart3, Clock, CheckCircle2, XCircle
} from 'lucide-react';
import AccuracyTab from './AccuracyTab';
import { api } from '@/utils/api';
import {
    CrashDashboardData,
    CrashPredictionHistoryItem,
    CrashAlertConfig,
} from '@/types/all_types';
import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, ReferenceLine,
    RadialBarChart, RadialBar, BarChart, Bar, Cell, Legend,
} from 'recharts';

// ==================== HELPERS ====================

const getIntensityColor = (intensity: string) => {
    switch (intensity) {
        case 'severe': return { text: 'text-red-400', bg: 'bg-red-500/20', border: 'border-red-500/30', fill: '#ef4444' };
        case 'moderate': return { text: 'text-amber-400', bg: 'bg-amber-500/20', border: 'border-amber-500/30', fill: '#f59e0b' };
        case 'mild': return { text: 'text-emerald-400', bg: 'bg-emerald-500/20', border: 'border-emerald-500/30', fill: '#10b981' };
        default: return { text: 'text-slate-400', bg: 'bg-slate-500/20', border: 'border-slate-500/30', fill: '#64748b' };
    }
};

const getProbabilityColor = (prob: number) => {
    if (prob >= 0.66) return '#ef4444';
    if (prob >= 0.33) return '#f59e0b';
    return '#10b981';
};

const getTrendIcon = (trend: string) => {
    switch (trend) {
        case 'increasing': return <ArrowUp size={16} className="text-red-400" />;
        case 'decreasing': return <ArrowDown size={16} className="text-emerald-400" />;
        default: return <Minus size={16} className="text-slate-400" />;
    }
};

const formatPct = (v: number | null | undefined) => {
    if (v == null) return 'N/A';
    return `${(v * 100).toFixed(1)}%`;
};

// ==================== SUB-COMPONENTS ====================

const MetricCard = ({ label, value, icon: Icon, color = 'violet' }: {
    label: string; value: string; icon: React.ElementType; color?: string;
}) => {
    const colorMap: Record<string, string> = {
        violet: 'from-violet-500/20 to-violet-500/5 text-violet-400',
        red: 'from-red-500/20 to-red-500/5 text-red-400',
        amber: 'from-amber-500/20 to-amber-500/5 text-amber-400',
        emerald: 'from-emerald-500/20 to-emerald-500/5 text-emerald-400',
        blue: 'from-blue-500/20 to-blue-500/5 text-blue-400',
    };
    return (
        <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
                <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${colorMap[color]} flex items-center justify-center`}>
                    <Icon size={16} />
                </div>
                <span className="text-xs text-slate-500 uppercase tracking-wider">{label}</span>
            </div>
            <div className="text-2xl font-bold text-slate-100">{value}</div>
        </div>
    );
};

// ==================== PROBABILITY GAUGE ====================

const ProbabilityGauge = ({ probability, intensity }: { probability: number; intensity: string }) => {
    const pct = Math.round(probability * 100);
    const color = getProbabilityColor(probability);
    const intensityColors = getIntensityColor(intensity);
    const gaugeData = [{ value: pct, fill: color }];

    return (
        <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6 flex flex-col items-center">
            <h3 className="text-sm font-medium text-slate-400 mb-4">Crash Probability</h3>
            <div className="relative w-48 h-48">
                <ResponsiveContainer width="100%" height="100%">
                    <RadialBarChart
                        cx="50%" cy="50%"
                        innerRadius="70%" outerRadius="90%"
                        startAngle={180} endAngle={0}
                        data={gaugeData}
                        barSize={12}
                    >
                        <RadialBar
                            dataKey="value"
                            cornerRadius={6}
                            background={{ fill: '#1e293b' }}
                        />
                    </RadialBarChart>
                </ResponsiveContainer>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-4xl font-bold text-slate-100">{pct}%</span>
                    <span className={`text-xs font-semibold uppercase mt-1 px-2 py-0.5 rounded ${intensityColors.bg} ${intensityColors.text}`}>
                        {intensity}
                    </span>
                </div>
            </div>
        </div>
    );
};

// ==================== TABS ====================

type TabKey = 'overview' | 'models' | 'hedging' | 'alerts' | 'accuracy';

const TABS: { key: TabKey; label: string; icon: React.ElementType }[] = [
    { key: 'overview', label: 'Overview', icon: Activity },
    { key: 'models', label: 'ML Models', icon: Zap },
    { key: 'accuracy', label: 'Accuracy', icon: BarChart3 },
    { key: 'hedging', label: 'Hedging', icon: Shield },
    { key: 'alerts', label: 'Alerts', icon: Bell },
];

// ==================== OVERVIEW TAB ====================

const OverviewTab = ({ data }: { data: CrashDashboardData }) => {
    const { prediction, stress, history } = data;
    const prob = prediction?.crash_probability ?? 0;
    const intensity = prediction?.intensity ?? 'unknown';

    // Build timeline data from history
    const timelineData = [...(history || [])]
        .reverse()
        .map((h) => ({
            date: h.timestamp ? new Date(h.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : '',
            probability: h.crash_probability ?? 0,
            stress: h.lstm_stress_index ?? 0,
        }));

    return (
        <div className="space-y-6">
            {/* Top row: Gauge + Metrics */}
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
                <div className="lg:col-span-1">
                    <ProbabilityGauge probability={prob} intensity={intensity} />
                </div>
                <div className="lg:col-span-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                    <MetricCard
                        label="Crash Probability"
                        value={formatPct(prob)}
                        icon={AlertTriangle}
                        color={prob >= 0.66 ? 'red' : prob >= 0.33 ? 'amber' : 'emerald'}
                    />
                    <MetricCard
                        label="Stress Index"
                        value={formatPct(stress?.stress_index)}
                        icon={Activity}
                        color={(stress?.stress_index ?? 0) >= 0.7 ? 'red' : 'blue'}
                    />
                    <MetricCard
                        label="LPPLS Confidence"
                        value={formatPct(prediction?.lppls?.confidence)}
                        icon={Target}
                        color="violet"
                    />
                    <MetricCard
                        label="Combined Score"
                        value={formatPct(prediction?.combined_score)}
                        icon={BarChart3}
                        color={prob >= 0.5 ? 'red' : 'emerald'}
                    />
                </div>
            </div>

            {/* Probability Timeline */}
            {timelineData.length > 0 && (
                <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
                    <h3 className="text-sm font-medium text-slate-400 mb-4">Crash Probability Timeline</h3>
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={timelineData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                <XAxis dataKey="date" stroke="#64748b" fontSize={11} />
                                <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} stroke="#64748b" fontSize={11} />
                                <Tooltip
                                    contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                                    labelStyle={{ color: '#94a3b8' }}
                                    formatter={(v: number | undefined) => [`${((v ?? 0) * 100).toFixed(1)}%`]}
                                />
                                <ReferenceLine y={0.66} stroke="#ef4444" strokeDasharray="5 5" label={{ value: 'Severe', fill: '#ef4444', fontSize: 10 }} />
                                <ReferenceLine y={0.33} stroke="#f59e0b" strokeDasharray="5 5" label={{ value: 'Moderate', fill: '#f59e0b', fontSize: 10 }} />
                                <Area type="monotone" dataKey="probability" stroke="#8b5cf6" fill="url(#probGradient)" strokeWidth={2} name="Crash Prob" />
                                <Area type="monotone" dataKey="stress" stroke="#3b82f6" fill="url(#stressGradient)" strokeWidth={1.5} name="Stress" />
                                <defs>
                                    <linearGradient id="probGradient" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.3} />
                                        <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0.05} />
                                    </linearGradient>
                                    <linearGradient id="stressGradient" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.2} />
                                        <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.02} />
                                    </linearGradient>
                                </defs>
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}

            {/* Quick Status Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* LPPLS Status */}
                <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-3">
                        <Target size={16} className="text-violet-400" />
                        <span className="text-sm font-medium text-slate-300">LPPLS Bubble</span>
                    </div>
                    <div className="flex items-center gap-2">
                        {prediction?.lppls?.bubble_detected ? (
                            <span className="flex items-center gap-1 text-red-400 text-sm font-semibold">
                                <AlertTriangle size={14} /> Bubble Detected
                            </span>
                        ) : (
                            <span className="flex items-center gap-1 text-emerald-400 text-sm font-semibold">
                                <CheckCircle2 size={14} /> No Bubble
                            </span>
                        )}
                    </div>
                    {prediction?.lppls?.critical_date && (
                        <p className="text-xs text-slate-500 mt-2">
                            Critical date: {new Date(prediction.lppls.critical_date).toLocaleDateString()}
                        </p>
                    )}
                </div>

                {/* LSTM Stress */}
                <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-3">
                        <Activity size={16} className="text-blue-400" />
                        <span className="text-sm font-medium text-slate-300">Market Stress</span>
                    </div>
                    <div className="flex items-center gap-3">
                        <span className="text-xl font-bold text-slate-100">{formatPct(stress?.stress_index)}</span>
                        <div className="flex items-center gap-1">
                            {getTrendIcon(stress?.trend ?? 'stable')}
                            <span className="text-xs text-slate-500 capitalize">{stress?.trend ?? 'stable'}</span>
                        </div>
                    </div>
                    {stress?.forecast && (
                        <p className="text-xs text-slate-500 mt-2">
                            60-day outlook: <span className="capitalize">{stress.forecast.outlook}</span>
                        </p>
                    )}
                </div>

                {/* Hedge Status */}
                <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-3">
                        <Shield size={16} className="text-emerald-400" />
                        <span className="text-sm font-medium text-slate-300">Hedge Recommendation</span>
                    </div>
                    <div className="text-lg font-bold text-slate-100 capitalize">
                        {data.hedge_recommendation?.strategy?.replace(/_/g, ' ') ?? 'None'}
                    </div>
                    {data.hedge_recommendation?.monitoring?.rebalance_frequency && (
                        <p className="text-xs text-slate-500 mt-2">
                            Rebalance: {data.hedge_recommendation.monitoring.rebalance_frequency}
                        </p>
                    )}
                </div>
            </div>
        </div>
    );
};

// ==================== ML MODELS TAB ====================

const ModelsTab = ({ data }: { data: CrashDashboardData }) => {
    const { prediction, stress } = data;
    const lppls = prediction?.lppls;
    const lstm = prediction?.lstm;

    const modelComparisonData = [
        { name: 'LPPLS', probability: (lppls?.crash_probability ?? 0) * 100, confidence: (lppls?.confidence ?? 0) * 100 },
        { name: 'LSTM', probability: (lstm?.stress_index ?? 0) * 100, confidence: (lstm?.confidence ?? 0) * 100 },
        { name: 'Combined', probability: (prediction?.combined_score ?? 0) * 100, confidence: (prediction?.confidence ?? 0) * 100 },
    ];

    return (
        <div className="space-y-6">
            {/* Model Comparison Chart */}
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
                <h3 className="text-sm font-medium text-slate-400 mb-4">Model Comparison</h3>
                <div className="h-56">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={modelComparisonData} layout="vertical" barGap={4}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                            <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`} stroke="#64748b" fontSize={11} />
                            <YAxis type="category" dataKey="name" stroke="#64748b" fontSize={12} width={80} />
                            <Tooltip
                                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                                formatter={(v: number | undefined) => [`${(v ?? 0).toFixed(1)}%`]}
                            />
                            <Legend />
                            <Bar dataKey="probability" name="Risk Signal" radius={[0, 4, 4, 0]} barSize={16}>
                                {modelComparisonData.map((entry, index) => (
                                    <Cell key={index} fill={getProbabilityColor(entry.probability / 100)} />
                                ))}
                            </Bar>
                            <Bar dataKey="confidence" name="Confidence" fill="#8b5cf6" radius={[0, 4, 4, 0]} barSize={16} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* LPPLS Detail Panel */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
                    <div className="flex items-center gap-2 mb-4">
                        <Target size={18} className="text-violet-400" />
                        <h3 className="text-sm font-semibold text-slate-200">LPPLS Bubble Detection</h3>
                    </div>
                    <div className="space-y-3">
                        <div className="flex justify-between items-center">
                            <span className="text-xs text-slate-500">Bubble Status</span>
                            {lppls?.bubble_detected ? (
                                <span className="text-xs font-semibold text-red-400 bg-red-500/20 px-2 py-0.5 rounded">DETECTED</span>
                            ) : (
                                <span className="text-xs font-semibold text-emerald-400 bg-emerald-500/20 px-2 py-0.5 rounded">CLEAR</span>
                            )}
                        </div>
                        <div className="flex justify-between">
                            <span className="text-xs text-slate-500">Confidence</span>
                            <span className="text-sm text-slate-200">{formatPct(lppls?.confidence)}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-xs text-slate-500">Crash Probability</span>
                            <span className="text-sm text-slate-200">{formatPct(lppls?.crash_probability)}</span>
                        </div>
                        {lppls?.critical_date && (
                            <div className="flex justify-between">
                                <span className="text-xs text-slate-500">Critical Date</span>
                                <span className="text-sm text-slate-200">{new Date(lppls.critical_date).toLocaleDateString()}</span>
                            </div>
                        )}
                        {/* Confidence Bar */}
                        <div className="mt-2">
                            <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-violet-500 rounded-full transition-all"
                                    style={{ width: `${(lppls?.confidence ?? 0) * 100}%` }}
                                />
                            </div>
                        </div>
                        {/* Reasons */}
                        {lppls?.reasons && lppls.reasons.length > 0 && (
                            <div className="mt-3 pt-3 border-t border-slate-700/50">
                                <span className="text-xs text-slate-500 block mb-2">Analysis Notes</span>
                                <ul className="space-y-1">
                                    {lppls.reasons.map((r, i) => (
                                        <li key={i} className="text-xs text-slate-400 flex items-start gap-1">
                                            <span className="text-violet-400 mt-0.5">&#8226;</span> {r}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                </div>

                {/* LSTM Detail Panel */}
                <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
                    <div className="flex items-center gap-2 mb-4">
                        <Activity size={18} className="text-blue-400" />
                        <h3 className="text-sm font-semibold text-slate-200">LSTM Stress Prediction</h3>
                    </div>
                    <div className="space-y-3">
                        <div className="flex justify-between items-center">
                            <span className="text-xs text-slate-500">Stress Index</span>
                            <span className="text-lg font-bold text-slate-100">{formatPct(lstm?.stress_index ?? stress?.stress_index)}</span>
                        </div>
                        <div className="flex justify-between items-center">
                            <span className="text-xs text-slate-500">Trend</span>
                            <div className="flex items-center gap-1">
                                {getTrendIcon(lstm?.stress_trend ?? stress?.trend ?? 'stable')}
                                <span className="text-sm text-slate-200 capitalize">{lstm?.stress_trend ?? stress?.trend ?? 'stable'}</span>
                            </div>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-xs text-slate-500">Confidence</span>
                            <span className="text-sm text-slate-200">{formatPct(lstm?.confidence ?? stress?.confidence)}</span>
                        </div>
                        {stress?.tap_deviation !== undefined && (
                            <div className="flex justify-between">
                                <span className="text-xs text-slate-500">TAP Deviation</span>
                                <span className="text-sm text-slate-200">{(stress.tap_deviation * 100).toFixed(2)}%</span>
                            </div>
                        )}
                        {/* Stress Bar */}
                        <div className="mt-2">
                            <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-blue-500 rounded-full transition-all"
                                    style={{ width: `${((lstm?.stress_index ?? stress?.stress_index ?? 0)) * 100}%` }}
                                />
                            </div>
                        </div>
                        {/* 60-Day Forecast */}
                        {stress?.forecast && (
                            <div className="mt-3 pt-3 border-t border-slate-700/50">
                                <span className="text-xs text-slate-500 block mb-2">60-Day Forecast</span>
                                <div className="grid grid-cols-2 gap-2">
                                    <div>
                                        <span className="text-xs text-slate-500">Outlook</span>
                                        <p className="text-sm text-slate-200 capitalize">{stress.forecast.outlook}</p>
                                    </div>
                                    <div>
                                        <span className="text-xs text-slate-500">Projected Stress</span>
                                        <p className="text-sm text-slate-200">{formatPct(stress.forecast.projected_stress_60d)}</p>
                                    </div>
                                    <div>
                                        <span className="text-xs text-slate-500">Volatility</span>
                                        <p className="text-sm text-slate-200 capitalize">{stress.forecast.key_indicators.volatility_regime}</p>
                                    </div>
                                    <div>
                                        <span className="text-xs text-slate-500">Liquidity</span>
                                        <p className="text-sm text-slate-200 capitalize">{stress.forecast.key_indicators.liquidity_conditions}</p>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Model Consensus */}
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
                <h3 className="text-sm font-medium text-slate-400 mb-3">Model Consensus</h3>
                <div className="flex items-center gap-4">
                    {(() => {
                        const lp = lppls?.crash_probability ?? 0;
                        const ls = lstm?.stress_index ?? 0;
                        const bothHigh = lp >= 0.5 && ls >= 0.5;
                        const bothLow = lp < 0.33 && ls < 0.33;
                        const agreement = bothHigh || bothLow;
                        return (
                            <>
                                <div className={`flex items-center gap-2 px-3 py-2 rounded-lg ${agreement ? 'bg-violet-500/20 text-violet-300' : 'bg-amber-500/20 text-amber-300'}`}>
                                    {agreement ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}
                                    <span className="text-sm font-medium">{agreement ? 'Models Agree' : 'Models Diverge'}</span>
                                </div>
                                <p className="text-xs text-slate-500">
                                    {bothHigh
                                        ? 'Both LPPLS and LSTM indicate elevated risk. High-confidence signal.'
                                        : bothLow
                                        ? 'Both models indicate low risk. Market conditions appear stable.'
                                        : 'Models show mixed signals. Monitor closely and use combined score for decisions.'}
                                </p>
                            </>
                        );
                    })()}
                </div>
            </div>
        </div>
    );
};

// ==================== HEDGING TAB ====================

const HedgingTab = ({ data }: { data: CrashDashboardData }) => {
    const hedge = data.hedge_recommendation;
    if (!hedge || hedge.error) {
        return (
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6 text-center">
                <Shield size={32} className="text-slate-600 mx-auto mb-3" />
                <p className="text-slate-400 text-sm">
                    {hedge?.error ? `Hedge recommendation unavailable: ${hedge.error}` : 'No hedge recommendation available. Run a crash prediction first.'}
                </p>
            </div>
        );
    }

    const strategyDescriptions: Record<string, { title: string; desc: string; risk: string }> = {
        covered_calls: {
            title: 'Covered Calls',
            desc: 'Sell out-of-the-money calls on existing positions to generate premium income that offsets minor losses.',
            risk: 'Caps upside but generates income. Best for mild correction expectations.',
        },
        put_spread: {
            title: 'Bear Put Spread',
            desc: 'Buy ATM put and sell OTM put to create cost-effective downside protection with a defined risk/reward.',
            risk: 'Limited cost, limited protection. Best for moderate crash expectations.',
        },
        tail_risk: {
            title: 'Tail Risk Hedge',
            desc: 'Buy deep OTM puts and VIX calls for maximum protection during severe systemic events.',
            risk: 'Higher premium cost but unlimited protection. Best for severe crash expectations.',
        },
        collar: {
            title: 'Collar Strategy',
            desc: 'Buy protective put and sell covered call. Premium from call offsets put cost for near-zero cost protection.',
            risk: 'Caps both upside and downside. Best for moderate risk with zero-cost objective.',
        },
    };

    const strategyInfo = strategyDescriptions[hedge.strategy] || {
        title: hedge.strategy?.replace(/_/g, ' ') ?? 'Unknown',
        desc: hedge.description ?? '',
        risk: '',
    };

    return (
        <div className="space-y-6">
            {/* Main Recommendation */}
            <div className="bg-gradient-to-r from-violet-500/10 to-fuchsia-500/10 border border-violet-500/20 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 rounded-xl bg-violet-500/20 flex items-center justify-center">
                        <Shield size={20} className="text-violet-400" />
                    </div>
                    <div>
                        <h3 className="text-lg font-semibold text-slate-100">{strategyInfo.title}</h3>
                        <span className="text-xs text-violet-400 uppercase tracking-wider">Recommended Strategy</span>
                    </div>
                </div>
                <p className="text-sm text-slate-300 mb-4">{strategyInfo.desc}</p>
                {strategyInfo.risk && (
                    <p className="text-xs text-slate-400 italic">{strategyInfo.risk}</p>
                )}
            </div>

            {/* Cost & Coverage */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {hedge.cost !== undefined && (
                    <MetricCard label="Estimated Cost" value={`$${(hedge.cost ?? 0).toLocaleString()}`} icon={TrendingDown} color="amber" />
                )}
                {hedge.protection && (
                    <MetricCard label="Protection Level" value={hedge.protection} icon={ShieldAlert} color="emerald" />
                )}
                {hedge.coverage !== undefined && (
                    <MetricCard label="Coverage" value={formatPct(hedge.coverage)} icon={Target} color="blue" />
                )}
            </div>

            {/* ML Signals Driving Recommendation */}
            {hedge.ml_signals && (
                <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
                    <h3 className="text-sm font-medium text-slate-400 mb-3">ML Signals Driving This Recommendation</h3>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                        <div>
                            <span className="text-xs text-slate-500">LPPLS Bubble</span>
                            <p className={`text-sm font-medium ${hedge.ml_signals.lppls_bubble ? 'text-red-400' : 'text-emerald-400'}`}>
                                {hedge.ml_signals.lppls_bubble ? 'Detected' : 'Clear'}
                            </p>
                        </div>
                        <div>
                            <span className="text-xs text-slate-500">LPPLS Crash Prob</span>
                            <p className="text-sm font-medium text-slate-200">{formatPct(hedge.ml_signals.lppls_crash_prob)}</p>
                        </div>
                        <div>
                            <span className="text-xs text-slate-500">LSTM Stress</span>
                            <p className="text-sm font-medium text-slate-200">{formatPct(hedge.ml_signals.lstm_stress)}</p>
                        </div>
                        <div>
                            <span className="text-xs text-slate-500">LPPLS Confidence</span>
                            <p className="text-sm font-medium text-slate-200">{formatPct(hedge.ml_signals.lppls_confidence)}</p>
                        </div>
                        <div>
                            <span className="text-xs text-slate-500">LSTM Confidence</span>
                            <p className="text-sm font-medium text-slate-200">{formatPct(hedge.ml_signals.lstm_confidence)}</p>
                        </div>
                        <div>
                            <span className="text-xs text-slate-500">Combined Probability</span>
                            <p className="text-sm font-bold" style={{ color: getProbabilityColor(hedge.ml_signals.combined_probability) }}>
                                {formatPct(hedge.ml_signals.combined_probability)}
                            </p>
                        </div>
                    </div>
                </div>
            )}

            {/* Monitoring Instructions */}
            {hedge.monitoring && (
                <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
                    <h3 className="text-sm font-medium text-slate-400 mb-3">Monitoring Instructions</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Alert Triggers */}
                        <div>
                            <span className="text-xs text-slate-500 uppercase tracking-wider">Alert Triggers</span>
                            {hedge.monitoring.alert_triggers?.length > 0 ? (
                                <ul className="mt-2 space-y-2">
                                    {hedge.monitoring.alert_triggers.map((t, i) => (
                                        <li key={i} className="flex items-start gap-2 text-sm">
                                            <Bell size={14} className="text-amber-400 mt-0.5 shrink-0" />
                                            <div>
                                                <span className="text-slate-300">{t.condition}</span>
                                                <span className="text-slate-500"> &rarr; </span>
                                                <span className="text-slate-400">{t.action}</span>
                                            </div>
                                        </li>
                                    ))}
                                </ul>
                            ) : (
                                <p className="text-xs text-slate-500 mt-2">No active triggers</p>
                            )}
                        </div>

                        {/* Stop-Loss Levels */}
                        <div>
                            <span className="text-xs text-slate-500 uppercase tracking-wider">Stop-Loss Levels</span>
                            <div className="mt-2 space-y-2">
                                <div className="flex justify-between">
                                    <span className="text-xs text-slate-500">Portfolio Stop</span>
                                    <span className="text-sm text-slate-200">{hedge.monitoring.stop_loss_levels.portfolio_stop}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-xs text-slate-500">Hedge Trigger</span>
                                    <span className="text-sm text-slate-200">{hedge.monitoring.stop_loss_levels.hedge_trigger}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-xs text-slate-500">Cash Level</span>
                                    <span className="text-sm text-slate-200">{hedge.monitoring.stop_loss_levels.cash_level}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div className="mt-4 pt-3 border-t border-slate-700/50 flex items-center gap-2">
                        <Clock size={14} className="text-violet-400" />
                        <span className="text-xs text-slate-400">
                            Rebalance Frequency: <span className="text-slate-200 font-medium">{hedge.monitoring.rebalance_frequency}</span>
                        </span>
                    </div>
                </div>
            )}
        </div>
    );
};

// ==================== ALERTS TAB ====================

const AlertsTab = ({ data }: { data: CrashDashboardData }) => {
    const [config, setConfig] = useState<CrashAlertConfig>({
        crash_threshold: 0.33,
        stress_threshold: 0.7,
        email_enabled: true,
        sms_enabled: false,
    });
    const [isSaving, setIsSaving] = useState(false);
    const [saveMessage, setSaveMessage] = useState<string | null>(null);

    const handleSave = async () => {
        setIsSaving(true);
        setSaveMessage(null);
        try {
            await api.crashPrediction.configureAlerts(config);
            setSaveMessage('Alert preferences saved successfully');
        } catch (err) {
            console.error('Failed to save alert config:', err);
            setSaveMessage('Failed to save preferences');
        } finally {
            setIsSaving(false);
            setTimeout(() => setSaveMessage(null), 3000);
        }
    };

    // Get recent history as proxy for alert history
    const recentAlerts = (data.history || [])
        .filter((h) => h.crash_probability >= 0.33 || (h.lstm_stress_index ?? 0) >= 0.7)
        .slice(0, 10);

    return (
        <div className="space-y-6">
            {/* Alert Configuration */}
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
                <div className="flex items-center gap-2 mb-4">
                    <Settings2 size={18} className="text-violet-400" />
                    <h3 className="text-sm font-semibold text-slate-200">Alert Configuration</h3>
                </div>

                <div className="space-y-5">
                    {/* Crash Threshold */}
                    <div>
                        <div className="flex justify-between mb-1">
                            <label className="text-xs text-slate-400">Crash Probability Threshold</label>
                            <span className="text-xs text-slate-300 font-mono">{(config.crash_threshold * 100).toFixed(0)}%</span>
                        </div>
                        <input
                            type="range" min={0} max={100} step={5}
                            value={config.crash_threshold * 100}
                            onChange={(e) => setConfig({ ...config, crash_threshold: parseInt(e.target.value) / 100 })}
                            className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-violet-500"
                        />
                        <div className="flex justify-between text-[10px] text-slate-600 mt-0.5">
                            <span>0%</span><span>33% (default)</span><span>100%</span>
                        </div>
                    </div>

                    {/* Stress Threshold */}
                    <div>
                        <div className="flex justify-between mb-1">
                            <label className="text-xs text-slate-400">Stress Index Threshold</label>
                            <span className="text-xs text-slate-300 font-mono">{(config.stress_threshold * 100).toFixed(0)}%</span>
                        </div>
                        <input
                            type="range" min={0} max={100} step={5}
                            value={config.stress_threshold * 100}
                            onChange={(e) => setConfig({ ...config, stress_threshold: parseInt(e.target.value) / 100 })}
                            className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                        />
                        <div className="flex justify-between text-[10px] text-slate-600 mt-0.5">
                            <span>0%</span><span>70% (default)</span><span>100%</span>
                        </div>
                    </div>

                    {/* Channel Toggles */}
                    <div className="flex gap-6">
                        <label className="flex items-center gap-2 cursor-pointer">
                            <input
                                type="checkbox"
                                checked={config.email_enabled}
                                onChange={(e) => setConfig({ ...config, email_enabled: e.target.checked })}
                                className="w-4 h-4 rounded bg-slate-700 border-slate-600 text-violet-500 focus:ring-violet-500"
                            />
                            <span className="text-sm text-slate-300">Email Alerts</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                            <input
                                type="checkbox"
                                checked={config.sms_enabled}
                                onChange={(e) => setConfig({ ...config, sms_enabled: e.target.checked })}
                                className="w-4 h-4 rounded bg-slate-700 border-slate-600 text-violet-500 focus:ring-violet-500"
                            />
                            <span className="text-sm text-slate-300">SMS Alerts</span>
                        </label>
                    </div>

                    {/* Save Button */}
                    <div className="flex items-center gap-3">
                        <button
                            onClick={handleSave}
                            disabled={isSaving}
                            className="px-4 py-2 bg-violet-600 hover:bg-violet-500 disabled:opacity-50 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-2"
                        >
                            {isSaving ? <Loader2 size={14} className="animate-spin" /> : <CheckCircle2 size={14} />}
                            Save Preferences
                        </button>
                        {saveMessage && (
                            <span className={`text-xs ${saveMessage.includes('success') ? 'text-emerald-400' : 'text-red-400'}`}>
                                {saveMessage}
                            </span>
                        )}
                    </div>
                </div>
            </div>

            {/* Alert History */}
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
                <h3 className="text-sm font-medium text-slate-400 mb-4">Recent High-Risk Predictions</h3>
                {recentAlerts.length > 0 ? (
                    <div className="space-y-2">
                        {recentAlerts.map((a) => {
                            const colors = getIntensityColor(a.intensity);
                            return (
                                <div key={a.id} className="flex items-center gap-3 p-3 bg-slate-900/50 rounded-lg">
                                    <div className={`w-2 h-2 rounded-full ${colors.bg.replace('/20', '')}`} />
                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center gap-2">
                                            <span className="text-sm font-medium text-slate-200">{a.symbol}</span>
                                            <span className={`text-xs px-1.5 py-0.5 rounded ${colors.bg} ${colors.text} uppercase`}>
                                                {a.intensity}
                                            </span>
                                        </div>
                                        <p className="text-xs text-slate-500 mt-0.5">
                                            Crash Prob: {formatPct(a.crash_probability)} | Stress: {formatPct(a.lstm_stress_index)}
                                            {a.hedge_strategy && ` | Hedge: ${a.hedge_strategy.replace(/_/g, ' ')}`}
                                        </p>
                                    </div>
                                    <span className="text-xs text-slate-600 whitespace-nowrap">
                                        {a.timestamp ? new Date(a.timestamp).toLocaleString() : ''}
                                    </span>
                                </div>
                            );
                        })}
                    </div>
                ) : (
                    <div className="text-center py-8">
                        <Bell size={24} className="text-slate-600 mx-auto mb-2" />
                        <p className="text-sm text-slate-500">No high-risk predictions recorded yet</p>
                    </div>
                )}
            </div>
        </div>
    );
};

// ==================== MAIN COMPONENT ====================

const CrashPredictionDashboard: React.FC = () => {
    const [selectedSymbol, setSelectedSymbol] = useState('SPY');
    const [symbolInput, setSymbolInput] = useState('SPY');
    const [activeTab, setActiveTab] = useState<TabKey>('overview');
    const [isLoading, setIsLoading] = useState(false);
    const [dashboardData, setDashboardData] = useState<CrashDashboardData | null>(null);
    const [error, setError] = useState<string | null>(null);

    const fetchDashboard = useCallback(async (symbol: string) => {
        setIsLoading(true);
        setError(null);
        try {
            const data = await api.crashPrediction.getDashboard(symbol) as unknown as CrashDashboardData;
            setDashboardData(data);
        } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : 'Failed to load crash prediction data';
            console.error('Dashboard fetch error:', err);
            setError(msg);
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchDashboard(selectedSymbol);
    }, [selectedSymbol, fetchDashboard]);

    const handleAnalyze = () => {
        const sym = symbolInput.trim().toUpperCase();
        if (sym) {
            setSelectedSymbol(sym);
            setSymbolInput(sym);
        }
    };

    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                    <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-red-500 to-amber-500 flex items-center justify-center shadow-lg shadow-red-500/20">
                            <ShieldAlert size={22} className="text-white" />
                        </div>
                        Crash Prediction
                    </h1>
                    <p className="text-sm text-slate-500 mt-1">ML-powered crash detection, intensity classification &amp; hedge recommendations</p>
                </div>

                {/* Symbol Input */}
                <div className="flex items-center gap-2">
                    <div className="relative">
                        <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                        <input
                            type="text"
                            value={symbolInput}
                            onChange={(e) => setSymbolInput(e.target.value.toUpperCase())}
                            onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
                            placeholder="Symbol"
                            className="pl-8 pr-3 py-2 w-32 bg-slate-800/70 border border-slate-700/50 rounded-lg text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:ring-1 focus:ring-violet-500/50"
                        />
                    </div>
                    <button
                        onClick={handleAnalyze}
                        disabled={isLoading}
                        className="px-4 py-2 bg-violet-600 hover:bg-violet-500 disabled:opacity-50 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-2"
                    >
                        {isLoading ? <Loader2 size={14} className="animate-spin" /> : <Activity size={14} />}
                        Analyze
                    </button>
                    <button
                        onClick={() => fetchDashboard(selectedSymbol)}
                        disabled={isLoading}
                        className="p-2 bg-slate-800/70 hover:bg-slate-700/70 border border-slate-700/50 rounded-lg transition-colors disabled:opacity-50"
                        title="Refresh"
                    >
                        <RefreshCw size={14} className={`text-slate-400 ${isLoading ? 'animate-spin' : ''}`} />
                    </button>
                </div>
            </div>

            {/* Tabs */}
            <div className="flex gap-1 bg-slate-800/30 p-1 rounded-xl border border-slate-700/30">
                {TABS.map(({ key, label, icon: Icon }) => (
                    <button
                        key={key}
                        onClick={() => setActiveTab(key)}
                        className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                            activeTab === key
                                ? 'bg-slate-800/70 text-slate-100 shadow-sm'
                                : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/30'
                        }`}
                    >
                        <Icon size={14} />
                        {label}
                    </button>
                ))}
            </div>

            {/* Error State */}
            {error && (
                <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 flex items-center gap-3">
                    <XCircle size={18} className="text-red-400 shrink-0" />
                    <p className="text-sm text-red-300">{error}</p>
                </div>
            )}

            {/* Loading State */}
            {isLoading && (
                <div className="flex flex-col items-center justify-center py-16">
                    <Loader2 size={32} className="text-violet-400 animate-spin mb-4" />
                    <p className="text-sm text-slate-500">Running ML models for {selectedSymbol}...</p>
                    <p className="text-xs text-slate-600 mt-1">LPPLS + LSTM + Ensemble analysis</p>
                </div>
            )}

            {/* Content */}
            {!isLoading && dashboardData && (
                <>
                    {activeTab === 'overview' && <OverviewTab data={dashboardData} />}
                    {activeTab === 'models' && <ModelsTab data={dashboardData} />}
                    {activeTab === 'hedging' && <HedgingTab data={dashboardData} />}
                    {activeTab === 'accuracy' && <AccuracyTab symbol={selectedSymbol} />}
                    {activeTab === 'alerts' && <AlertsTab data={dashboardData} />}
                </>
            )}

            {/* Empty State */}
            {!isLoading && !dashboardData && !error && (
                <div className="flex flex-col items-center justify-center py-16">
                    <Shield size={40} className="text-slate-600 mb-4" />
                    <p className="text-slate-400 text-sm">Enter a symbol and click Analyze to start crash prediction</p>
                </div>
            )}
        </div>
    );
};

export default CrashPredictionDashboard;
