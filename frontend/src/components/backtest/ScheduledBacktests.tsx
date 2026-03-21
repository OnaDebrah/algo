/* eslint-disable @typescript-eslint/no-explicit-any */

'use client'

import { useEffect, useState } from 'react';
import { Calendar, ChevronDown, Clock, Loader2, Pause, Play, Plus, Trash2, X } from 'lucide-react';
import { client } from '@/utils/api';

interface ScheduledItem {
    id: number;
    name: string;
    strategy_key: string;
    strategy_params: Record<string, any>;
    symbols: string[];
    interval: string;
    period: string;
    initial_capital: number;
    schedule_cron: string;
    is_active: boolean;
    last_run_at: string | null;
    next_run_at: string | null;
    created_at: string;
}

interface RunItem {
    id: number;
    status: string;
    result_summary: Record<string, any> | null;
    error_message: string | null;
    started_at: string | null;
    completed_at: string | null;
}

const CRON_PRESETS = [
    { label: 'Daily at 9 AM', value: '0 9 * * *' },
    { label: 'Weekly (Monday 9 AM)', value: '0 9 * * 1' },
    { label: 'Every 6 hours', value: '0 */6 * * *' },
    { label: 'Monthly (1st at 9 AM)', value: '0 9 1 * *' },
];

const ScheduledBacktests = () => {
    const [schedules, setSchedules] = useState<ScheduledItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [showCreate, setShowCreate] = useState(false);
    const [creating, setCreating] = useState(false);
    const [expandedId, setExpandedId] = useState<number | null>(null);
    const [runs, setRuns] = useState<RunItem[]>([]);
    const [loadingRuns, setLoadingRuns] = useState(false);

    // Create form
    const [name, setName] = useState('');
    const [strategyKey, setStrategyKey] = useState('');
    const [symbols, setSymbols] = useState('AAPL');
    const [interval, setInterval_] = useState('1d');
    const [cron, setCron] = useState('0 9 * * 1');

    const fetchSchedules = async () => {
        try {
            const data = await client.get('/scheduled-backtests/');
            setSchedules(data as any[]);
        } catch {
            console.error('Failed to fetch schedules');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { fetchSchedules(); }, []);

    const handleCreate = async () => {
        if (!name.trim() || !strategyKey.trim()) return;
        setCreating(true);
        try {
            await client.post('/scheduled-backtests/', {
                name: name.trim(),
                strategy_key: strategyKey.trim(),
                symbols: symbols.split(',').map(s => s.trim()).filter(Boolean),
                interval,
                schedule_cron: cron,
            });
            setShowCreate(false);
            setName('');
            setStrategyKey('');
            await fetchSchedules();
        } catch {
            alert('Failed to create schedule');
        } finally {
            setCreating(false);
        }
    };

    const toggleActive = async (id: number, currentActive: boolean) => {
        try {
            await client.patch(`/scheduled-backtests/${id}`, { is_active: !currentActive });
            await fetchSchedules();
        } catch {
            alert('Failed to update schedule');
        }
    };

    const deleteSchedule = async (id: number) => {
        if (!confirm('Delete this scheduled backtest?')) return;
        try {
            await client.delete(`/scheduled-backtests/${id}`);
            await fetchSchedules();
        } catch {
            alert('Failed to delete');
        }
    };

    const loadRuns = async (id: number) => {
        if (expandedId === id) {
            setExpandedId(null);
            return;
        }
        setExpandedId(id);
        setLoadingRuns(true);
        try {
            const data = await client.get(`/scheduled-backtests/${id}/runs`);
            setRuns(data as RunItem[]);
        } catch {
            console.error('Failed to fetch runs');
        } finally {
            setLoadingRuns(false);
        }
    };

    return (
        <div className="space-y-4">
            <div className="flex items-center justify-between">
                <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                    <Calendar size={20} className="text-indigo-400" />
                    Scheduled Backtests
                </h3>
                <button
                    onClick={() => setShowCreate(true)}
                    className="flex items-center gap-2 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-semibold transition-all"
                >
                    <Plus size={14} /> New Schedule
                </button>
            </div>

            {/* Create form */}
            {showCreate && (
                <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4 space-y-3">
                    <div className="flex items-center justify-between">
                        <h4 className="text-sm font-semibold text-slate-200">Create Schedule</h4>
                        <button onClick={() => setShowCreate(false)}><X size={16} className="text-slate-500" /></button>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                        <input
                            type="text"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            placeholder="Schedule name"
                            className="px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:border-indigo-500 outline-none"
                        />
                        <input
                            type="text"
                            value={strategyKey}
                            onChange={(e) => setStrategyKey(e.target.value)}
                            placeholder="Strategy key (e.g. rsi)"
                            className="px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:border-indigo-500 outline-none"
                        />
                        <input
                            type="text"
                            value={symbols}
                            onChange={(e) => setSymbols(e.target.value)}
                            placeholder="Symbols (comma-separated)"
                            className="px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:border-indigo-500 outline-none"
                        />
                        <select
                            value={cron}
                            onChange={(e) => setCron(e.target.value)}
                            className="px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-sm text-slate-300 outline-none"
                        >
                            {CRON_PRESETS.map(p => (
                                <option key={p.value} value={p.value}>{p.label}</option>
                            ))}
                        </select>
                    </div>
                    <button
                        onClick={handleCreate}
                        disabled={creating || !name.trim() || !strategyKey.trim()}
                        className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-semibold transition-all disabled:opacity-50 flex items-center gap-2"
                    >
                        {creating ? <Loader2 size={14} className="animate-spin" /> : <Calendar size={14} />}
                        Create
                    </button>
                </div>
            )}

            {/* Schedule list */}
            {loading ? (
                <div className="flex items-center justify-center py-12">
                    <Loader2 className="animate-spin text-slate-500" size={24} />
                </div>
            ) : schedules.length === 0 ? (
                <div className="text-center py-12">
                    <Calendar className="mx-auto text-slate-600 mb-3" size={40} />
                    <p className="text-slate-400">No scheduled backtests</p>
                    <p className="text-xs text-slate-500 mt-1">Create a schedule to auto-run backtests on a recurring basis</p>
                </div>
            ) : (
                <div className="space-y-2">
                    {schedules.map((s) => (
                        <div key={s.id} className="bg-slate-900/50 border border-slate-800/50 rounded-xl overflow-hidden">
                            <div
                                className="p-4 flex items-center justify-between cursor-pointer hover:bg-slate-800/30 transition-all"
                                onClick={() => loadRuns(s.id)}
                            >
                                <div className="flex items-center gap-3 min-w-0">
                                    <div className={`w-2 h-2 rounded-full ${s.is_active ? 'bg-emerald-400' : 'bg-slate-600'}`} />
                                    <div className="min-w-0">
                                        <span className="text-sm font-semibold text-slate-200 truncate block">{s.name}</span>
                                        <div className="flex items-center gap-2 text-xs text-slate-500 mt-0.5">
                                            <span>{s.strategy_key}</span>
                                            <span>&middot;</span>
                                            <span>{s.symbols.join(', ')}</span>
                                            <span>&middot;</span>
                                            <span className="flex items-center gap-1"><Clock size={10} />{s.schedule_cron}</span>
                                        </div>
                                    </div>
                                </div>
                                <div className="flex items-center gap-1">
                                    <button
                                        onClick={(e) => { e.stopPropagation(); toggleActive(s.id, s.is_active); }}
                                        className="p-2 hover:bg-slate-700 rounded-lg transition-all"
                                        title={s.is_active ? 'Pause' : 'Resume'}
                                    >
                                        {s.is_active ? <Pause size={14} className="text-amber-400" /> : <Play size={14} className="text-emerald-400" />}
                                    </button>
                                    <button
                                        onClick={(e) => { e.stopPropagation(); deleteSchedule(s.id); }}
                                        className="p-2 hover:bg-red-900/30 rounded-lg transition-all"
                                    >
                                        <Trash2 size={14} className="text-red-400" />
                                    </button>
                                    <ChevronDown size={14} className={`text-slate-500 transition-transform ${expandedId === s.id ? 'rotate-180' : ''}`} />
                                </div>
                            </div>

                            {/* Run history */}
                            {expandedId === s.id && (
                                <div className="border-t border-slate-800/50 p-4">
                                    {loadingRuns ? (
                                        <div className="flex items-center justify-center py-4">
                                            <Loader2 size={16} className="animate-spin text-slate-500" />
                                        </div>
                                    ) : runs.length === 0 ? (
                                        <p className="text-xs text-slate-500 text-center py-4">No runs yet</p>
                                    ) : (
                                        <div className="space-y-1">
                                            {runs.map((r) => (
                                                <div key={r.id} className="flex items-center justify-between text-xs py-1.5">
                                                    <div className="flex items-center gap-2">
                                                        <span className={`px-1.5 py-0.5 rounded text-xs font-semibold ${
                                                            r.status === 'completed' ? 'bg-emerald-900/30 text-emerald-400' :
                                                            r.status === 'failed' ? 'bg-red-900/30 text-red-400' :
                                                            'bg-amber-900/30 text-amber-400'
                                                        }`}>{r.status}</span>
                                                        <span className="text-slate-500">{r.started_at ? new Date(r.started_at).toLocaleString() : 'Pending'}</span>
                                                    </div>
                                                    {r.result_summary && (
                                                        <div className="flex gap-3 text-slate-400">
                                                            {r.result_summary.sharpe_ratio != null && <span>Sharpe: {Number(r.result_summary.sharpe_ratio).toFixed(2)}</span>}
                                                            {r.result_summary.total_return != null && <span>Return: {Number(r.result_summary.total_return).toFixed(1)}%</span>}
                                                        </div>
                                                    )}
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default ScheduledBacktests;
