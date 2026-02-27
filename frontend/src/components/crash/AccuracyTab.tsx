'use client';
import React, { useEffect, useState, useMemo } from 'react';
import {
    Loader2, CheckCircle2, XCircle, Target, Clock,
    ShieldAlert, TrendingDown, BarChart3, AlertTriangle,
} from 'lucide-react';
import { api } from '@/utils/api';
import {
    HistoricalAccuracyData,
    CrashEventAccuracy,
} from '@/types/all_types';
import {
    ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, ReferenceLine, ReferenceArea,
    BarChart, Bar, Cell, Legend,
} from 'recharts';

// ==================== HELPERS ====================

const MetricCard = ({ label, value, icon: Icon, color, subtitle }: {
    label: string; value: string; icon: React.ElementType; color: string; subtitle?: string;
}) => {
    const colorMap: Record<string, string> = {
        violet: 'from-violet-500/10 to-violet-500/5 border-violet-500/20 text-violet-400',
        emerald: 'from-emerald-500/10 to-emerald-500/5 border-emerald-500/20 text-emerald-400',
        red: 'from-red-500/10 to-red-500/5 border-red-500/20 text-red-400',
        amber: 'from-amber-500/10 to-amber-500/5 border-amber-500/20 text-amber-400',
        blue: 'from-blue-500/10 to-blue-500/5 border-blue-500/20 text-blue-400',
        cyan: 'from-cyan-500/10 to-cyan-500/5 border-cyan-500/20 text-cyan-400',
    };
    const cls = colorMap[color] || colorMap.violet;

    return (
        <div className={`bg-gradient-to-br ${cls} border rounded-xl p-4`}>
            <div className="flex items-center gap-2 mb-2">
                <Icon className="w-4 h-4" />
                <span className="text-xs font-medium text-slate-400">{label}</span>
            </div>
            <div className="text-2xl font-bold text-white">{value}</div>
            {subtitle && <div className="text-xs text-slate-500 mt-1">{subtitle}</div>}
        </div>
    );
};

// ==================== MAIN COMPONENT ====================

interface AccuracyTabProps {
    symbol: string;
}

export default function AccuracyTab({ symbol }: AccuracyTabProps) {
    const [data, setData] = useState<HistoricalAccuracyData | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [fetched, setFetched] = useState(false);

    useEffect(() => {
        if (fetched) return;
        const fetchData = async () => {
            setLoading(true);
            setError(null);
            try {
                const result = await api.crashPrediction.getHistoricalAccuracy(symbol, 20, 0.33) as unknown as HistoricalAccuracyData;
                setData(result);
            } catch (e: unknown) {
                setError(e instanceof Error ? e.message : 'Failed to load accuracy data');
            } finally {
                setLoading(false);
                setFetched(true);
            }
        };
        fetchData();
    }, [symbol, fetched]);

    // ── Chart data ──────────────────────────────────────────────────
    const chartData = useMemo(() => {
        if (!data?.timeseries) return [];
        // Downsample for rendering if too many points
        const ts = data.timeseries;
        const step = ts.length > 500 ? Math.ceil(ts.length / 500) : 1;
        return ts.filter((_, i) => i % step === 0).map(p => ({
            ...p,
            dateLabel: new Date(p.date).toLocaleDateString('en-US', { year: '2-digit', month: 'short' }),
        }));
    }, [data]);

    // ── Model comparison chart data ─────────────────────────────────
    const modelCompData = useMemo(() => {
        if (!data?.model_comparison) return [];
        const mc = data.model_comparison;
        return [
            { name: 'LPPLS', sensitivity: mc.lppls.sensitivity * 100, specificity: mc.lppls.specificity * 100, lead_time: mc.lppls.avg_lead_time },
            { name: 'LSTM', sensitivity: mc.lstm.sensitivity * 100, specificity: mc.lstm.specificity * 100, lead_time: mc.lstm.avg_lead_time },
            { name: 'Combined', sensitivity: mc.combined.sensitivity * 100, specificity: mc.combined.specificity * 100, lead_time: mc.combined.avg_lead_time },
        ];
    }, [data]);

    // ── Loading state ───────────────────────────────────────────────
    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center py-24 gap-4">
                <Loader2 className="w-12 h-12 text-violet-400 animate-spin" />
                <div className="text-center">
                    <h3 className="text-lg font-semibold text-white mb-2">Running Historical Backtest</h3>
                    <p className="text-sm text-slate-400 max-w-md">
                        Evaluating LPPLS + LSTM models across 18 years of market data
                        against 7 known crash events. This typically takes 15-30 seconds...
                    </p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex flex-col items-center justify-center py-24 gap-4">
                <AlertTriangle className="w-12 h-12 text-red-400" />
                <div className="text-center">
                    <h3 className="text-lg font-semibold text-white mb-2">Backtest Failed</h3>
                    <p className="text-sm text-red-400">{error}</p>
                </div>
            </div>
        );
    }

    if (!data) return null;

    const { metrics, crash_events } = data;

    return (
        <div className="space-y-6">
            {/* ── Section A: Time-Series Overlay Chart ────────────────── */}
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
                <h3 className="text-sm font-medium text-slate-400 mb-1">
                    Historical Crash Prediction vs Actual Events
                </h3>
                <p className="text-xs text-slate-500 mb-4">
                    {data.period.start} to {data.period.end} &middot; {data.timeseries.length} evaluation points &middot; Threshold: {(data.threshold * 100).toFixed(0)}%
                </p>
                <div className="h-[420px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={chartData}>
                            <defs>
                                <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%" stopColor="#64748b" stopOpacity={0.15} />
                                    <stop offset="100%" stopColor="#64748b" stopOpacity={0.02} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                            <XAxis
                                dataKey="dateLabel"
                                stroke="#64748b"
                                fontSize={10}
                                interval={Math.floor(chartData.length / 10)}
                                tickLine={false}
                            />
                            <YAxis
                                yAxisId="price"
                                orientation="left"
                                stroke="#64748b"
                                fontSize={10}
                                tickFormatter={(v: number) => `$${v.toFixed(0)}`}
                                domain={['auto', 'auto']}
                            />
                            <YAxis
                                yAxisId="prob"
                                orientation="right"
                                stroke="#64748b"
                                fontSize={10}
                                domain={[0, 1]}
                                tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                            />
                            <Tooltip
                                contentStyle={{
                                    background: '#1e293b',
                                    border: '1px solid #334155',
                                    borderRadius: '8px',
                                    fontSize: '12px',
                                }}
                                labelFormatter={(label: string) => label}
                                formatter={(value: number | undefined, name?: string) => {
                                    const v = value ?? 0;
                                    const n = name ?? '';
                                    if (n === 'Price') return [`$${v.toFixed(2)}`, n];
                                    return [`${(v * 100).toFixed(1)}%`, n];
                                }}
                            />
                            <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />

                            {/* Crash zone shading */}
                            {crash_events.map((event) => {
                                const peakIdx = chartData.findIndex(p => p.date >= event.peak_date);
                                const troughIdx = chartData.findIndex(p => p.date >= event.trough_date);
                                if (peakIdx < 0) return null;
                                const endIdx = troughIdx >= 0 ? troughIdx : chartData.length - 1;
                                return (
                                    <ReferenceArea
                                        key={event.name}
                                        x1={chartData[peakIdx]?.dateLabel}
                                        x2={chartData[endIdx]?.dateLabel}
                                        yAxisId="price"
                                        fill="#ef4444"
                                        fillOpacity={0.08}
                                        strokeOpacity={0}
                                    />
                                );
                            })}

                            {/* Threshold line */}
                            <ReferenceLine
                                yAxisId="prob"
                                y={data.threshold}
                                stroke="#f59e0b"
                                strokeDasharray="5 5"
                                label={{ value: `Threshold ${(data.threshold * 100).toFixed(0)}%`, position: 'right', fill: '#f59e0b', fontSize: 10 }}
                            />

                            {/* Price area */}
                            <Area
                                yAxisId="price"
                                type="monotone"
                                dataKey="price_normalized"
                                stroke="#64748b"
                                fill="url(#priceGradient)"
                                strokeWidth={1}
                                name="Price"
                                dot={false}
                            />

                            {/* Model probability lines */}
                            <Line yAxisId="prob" type="monotone" dataKey="lppls_prob" stroke="#8b5cf6" strokeWidth={1.5} dot={false} name="LPPLS" />
                            <Line yAxisId="prob" type="monotone" dataKey="lstm_stress" stroke="#3b82f6" strokeWidth={1.5} dot={false} name="LSTM Stress" />
                            <Line yAxisId="prob" type="monotone" dataKey="combined_prob" stroke="#ef4444" strokeWidth={2} dot={false} name="Combined" />
                        </ComposedChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* ── Section B: Accuracy Metric Cards ────────────────────── */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                <MetricCard
                    label="Sensitivity"
                    value={`${(metrics.sensitivity * 100).toFixed(1)}%`}
                    icon={Target}
                    color="emerald"
                    subtitle="True Positive Rate"
                />
                <MetricCard
                    label="Specificity"
                    value={`${(metrics.specificity * 100).toFixed(1)}%`}
                    icon={ShieldAlert}
                    color="blue"
                    subtitle="True Negative Rate"
                />
                <MetricCard
                    label="Precision"
                    value={`${(metrics.precision * 100).toFixed(1)}%`}
                    icon={Target}
                    color="violet"
                    subtitle="Signal Accuracy"
                />
                <MetricCard
                    label="False Positive Rate"
                    value={`${(metrics.false_positive_rate * 100).toFixed(1)}%`}
                    icon={AlertTriangle}
                    color="red"
                    subtitle="False Alarms"
                />
                <MetricCard
                    label="Avg Lead Time"
                    value={`${metrics.avg_lead_time_days.toFixed(0)}d`}
                    icon={Clock}
                    color="cyan"
                    subtitle="Days Before Crash"
                />
                <MetricCard
                    label="Margin of Error"
                    value={`\u00B1${metrics.lead_time_std_days.toFixed(0)}d`}
                    icon={BarChart3}
                    color="amber"
                    subtitle="Lead Time Std Dev"
                />
            </div>

            {/* ── Section C: Model Comparison + Section D: Event Table ── */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Model Comparison Bar Chart */}
                <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
                    <h3 className="text-sm font-medium text-slate-400 mb-4">Model Comparison</h3>
                    <div className="h-[220px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={modelCompData} layout="vertical" barGap={2}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                <XAxis
                                    type="number"
                                    domain={[0, 100]}
                                    tickFormatter={(v: number) => `${v}%`}
                                    stroke="#64748b"
                                    fontSize={10}
                                />
                                <YAxis
                                    type="category"
                                    dataKey="name"
                                    stroke="#64748b"
                                    fontSize={11}
                                    width={70}
                                />
                                <Tooltip
                                    contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', fontSize: '12px' }}
                                    formatter={(v: number | undefined) => [`${(v ?? 0).toFixed(1)}%`]}
                                />
                                <Legend wrapperStyle={{ fontSize: '11px' }} />
                                <Bar dataKey="sensitivity" name="Sensitivity" fill="#10b981" radius={[0, 4, 4, 0]} />
                                <Bar dataKey="specificity" name="Specificity" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="mt-3 flex items-center gap-4 text-xs text-slate-500">
                        {modelCompData.map(m => (
                            <span key={m.name}>{m.name}: {m.lead_time.toFixed(0)}d avg lead</span>
                        ))}
                    </div>
                </div>

                {/* Crash Event Breakdown Table */}
                <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
                    <h3 className="text-sm font-medium text-slate-400 mb-4">Per-Crash Event Analysis</h3>
                    <div className="overflow-x-auto">
                        <table className="w-full text-xs">
                            <thead>
                                <tr className="border-b border-slate-700">
                                    <th className="text-left text-slate-500 py-2 pr-3">Event</th>
                                    <th className="text-center text-slate-500 py-2 px-2">Drawdown</th>
                                    <th className="text-center text-slate-500 py-2 px-2">Detected</th>
                                    <th className="text-center text-slate-500 py-2 px-2">Lead</th>
                                    <th className="text-center text-slate-500 py-2 px-2">Peak Prob</th>
                                    <th className="text-center text-slate-500 py-2 px-1">L</th>
                                    <th className="text-center text-slate-500 py-2 px-1">S</th>
                                </tr>
                            </thead>
                            <tbody>
                                {crash_events.map((event: CrashEventAccuracy) => (
                                    <tr key={event.name} className="border-b border-slate-700/50 hover:bg-slate-700/20">
                                        <td className="py-2 pr-3">
                                            <div className="text-white font-medium">{event.name}</div>
                                            <div className="text-slate-500">{event.peak_date}</div>
                                        </td>
                                        <td className="text-center text-red-400 font-mono py-2 px-2">
                                            {event.drawdown_pct.toFixed(1)}%
                                        </td>
                                        <td className="text-center py-2 px-2">
                                            {event.detected
                                                ? <CheckCircle2 className="w-4 h-4 text-emerald-400 inline" />
                                                : <XCircle className="w-4 h-4 text-red-400 inline" />
                                            }
                                        </td>
                                        <td className="text-center text-slate-300 font-mono py-2 px-2">
                                            {event.lead_time_days != null ? `${event.lead_time_days}d` : '-'}
                                        </td>
                                        <td className="text-center font-mono py-2 px-2">
                                            <span className={event.peak_probability >= 0.5 ? 'text-red-400' : event.peak_probability >= 0.33 ? 'text-amber-400' : 'text-slate-400'}>
                                                {(event.peak_probability * 100).toFixed(0)}%
                                            </span>
                                        </td>
                                        <td className="text-center py-2 px-1">
                                            {event.lppls_detected
                                                ? <CheckCircle2 className="w-3 h-3 text-violet-400 inline" />
                                                : <XCircle className="w-3 h-3 text-slate-600 inline" />
                                            }
                                        </td>
                                        <td className="text-center py-2 px-1">
                                            {event.lstm_detected
                                                ? <CheckCircle2 className="w-3 h-3 text-blue-400 inline" />
                                                : <XCircle className="w-3 h-3 text-slate-600 inline" />
                                            }
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                    <div className="mt-2 flex items-center gap-3 text-[10px] text-slate-600">
                        <span>L = LPPLS</span>
                        <span>S = LSTM Stress</span>
                    </div>
                </div>
            </div>

            {/* ── F1 Score and Confusion Matrix Summary ───────────────── */}
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-medium text-slate-400">Classification Summary</h3>
                    <span className="text-xs text-slate-500">
                        F1 Score: <span className="text-white font-mono">{(metrics.f1_score * 100).toFixed(1)}%</span>
                    </span>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center p-3 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
                        <div className="text-2xl font-bold text-emerald-400">{metrics.true_positives}</div>
                        <div className="text-xs text-slate-400 mt-1">True Positives</div>
                    </div>
                    <div className="text-center p-3 bg-red-500/10 rounded-lg border border-red-500/20">
                        <div className="text-2xl font-bold text-red-400">{metrics.false_positives}</div>
                        <div className="text-xs text-slate-400 mt-1">False Positives</div>
                    </div>
                    <div className="text-center p-3 bg-blue-500/10 rounded-lg border border-blue-500/20">
                        <div className="text-2xl font-bold text-blue-400">{metrics.true_negatives}</div>
                        <div className="text-xs text-slate-400 mt-1">True Negatives</div>
                    </div>
                    <div className="text-center p-3 bg-amber-500/10 rounded-lg border border-amber-500/20">
                        <div className="text-2xl font-bold text-amber-400">{metrics.false_negatives}</div>
                        <div className="text-xs text-slate-400 mt-1">False Negatives</div>
                    </div>
                </div>
            </div>
        </div>
    );
}
