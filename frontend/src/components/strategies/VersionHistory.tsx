/* eslint-disable @typescript-eslint/no-explicit-any */

'use client'

import { useEffect, useState } from 'react';
import { ArrowLeftRight, Clock, GitBranch, Loader2, RotateCcw } from 'lucide-react';

interface StrategyVersionItem {
    id: number;
    version_number: number;
    version_label: string;
    parameters_snapshot: Record<string, any>;
    performance_snapshot?: Record<string, any>;
    change_description?: string;
    created_at: string;
}

import { client } from '@/utils/api';

const VersionHistory = ({ strategyId, strategyType = 'marketplace', onRollback }: {
    strategyId: number;
    strategyType?: string;
    onRollback?: () => void;
}) => {
    const [versions, setVersions] = useState<StrategyVersionItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [compareIds, setCompareIds] = useState<[number | null, number | null]>([null, null]);
    const [comparison, setComparison] = useState<any>(null);
    const [comparing, setComparing] = useState(false);

    const fetchVersions = async () => {
        try {
            const data = await client.get(`/strategies/${strategyId}/versions`, { params: { strategy_type: strategyType } });
            setVersions(data as any[]);
        } catch {
            console.error('Failed to fetch versions');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { fetchVersions(); }, [strategyId]);

    const handleCompare = async () => {
        const [v1, v2] = compareIds;
        if (!v1 || !v2) return;
        setComparing(true);
        try {
            const data = await client.get(`/strategies/${strategyId}/versions/${v1}/compare/${v2}`);
            setComparison(data);
        } catch {
            alert('Failed to compare versions');
        } finally {
            setComparing(false);
        }
    };

    const handleRollback = async (versionId: number) => {
        if (!confirm('Rollback strategy to this version? Current parameters will be overwritten.')) return;
        try {
            await client.post(`/strategies/${strategyId}/versions/${versionId}/rollback`, null, {
                params: { strategy_type: strategyType }
            });
            alert('Strategy rolled back successfully');
            onRollback?.();
        } catch {
            alert('Failed to rollback');
        }
    };

    const toggleCompare = (id: number) => {
        setCompareIds(([a, b]) => {
            if (a === id) return [null, b];
            if (b === id) return [a, null];
            if (!a) return [id, b];
            return [a, id];
        });
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center py-8">
                <Loader2 className="animate-spin text-slate-500" size={24} />
            </div>
        );
    }

    if (versions.length === 0) {
        return (
            <div className="text-center py-8">
                <GitBranch className="mx-auto text-slate-600 mb-3" size={32} />
                <p className="text-slate-400 text-sm">No version history yet</p>
            </div>
        );
    }

    return (
        <div className="space-y-4">
            <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                    <GitBranch size={16} /> Version History
                </h3>
                {compareIds[0] && compareIds[1] && (
                    <button
                        onClick={handleCompare}
                        disabled={comparing}
                        className="flex items-center gap-1 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold transition-all disabled:opacity-50"
                    >
                        {comparing ? <Loader2 size={12} className="animate-spin" /> : <ArrowLeftRight size={12} />}
                        Compare
                    </button>
                )}
            </div>

            {/* Version list */}
            <div className="space-y-2">
                {versions.map((v) => (
                    <div
                        key={v.id}
                        className={`p-3 rounded-xl border transition-all cursor-pointer ${
                            compareIds.includes(v.id)
                                ? 'bg-indigo-900/20 border-indigo-700/50'
                                : 'bg-slate-800/30 border-slate-700/30 hover:border-slate-600/50'
                        }`}
                        onClick={() => toggleCompare(v.id)}
                    >
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <span className="text-sm font-bold text-indigo-400">v{v.version_label}</span>
                                {v.change_description && (
                                    <span className="text-xs text-slate-400">{v.change_description}</span>
                                )}
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="text-xs text-slate-500 flex items-center gap-1">
                                    <Clock size={10} />
                                    {new Date(v.created_at).toLocaleDateString()}
                                </span>
                                <button
                                    onClick={(e) => { e.stopPropagation(); handleRollback(v.id); }}
                                    className="p-1 hover:bg-slate-700 rounded transition-all"
                                    title="Rollback to this version"
                                >
                                    <RotateCcw size={12} className="text-slate-500" />
                                </button>
                            </div>
                        </div>
                        {v.performance_snapshot && (
                            <div className="flex gap-3 mt-2 text-xs text-slate-500">
                                {v.performance_snapshot.sharpe_ratio != null && (
                                    <span>Sharpe: {Number(v.performance_snapshot.sharpe_ratio).toFixed(2)}</span>
                                )}
                                {v.performance_snapshot.total_return != null && (
                                    <span>Return: {Number(v.performance_snapshot.total_return).toFixed(1)}%</span>
                                )}
                            </div>
                        )}
                    </div>
                ))}
            </div>

            {/* Comparison panel */}
            {comparison && (
                <div className="bg-slate-800/30 border border-slate-700/30 rounded-xl p-4 space-y-3">
                    <div className="flex items-center justify-between">
                        <h4 className="text-sm font-semibold text-slate-200">
                            Comparing v{comparison.v1.version} vs v{comparison.v2.version}
                        </h4>
                        <button onClick={() => setComparison(null)} className="text-xs text-slate-500 hover:text-slate-300">
                            Close
                        </button>
                    </div>
                    {Object.keys(comparison.parameter_diffs || {}).length === 0 ? (
                        <p className="text-xs text-slate-400">No parameter differences</p>
                    ) : (
                        <div className="space-y-1">
                            {Object.entries(comparison.parameter_diffs).map(([key, diff]: [string, any]) => (
                                <div key={key} className="flex items-center justify-between text-xs">
                                    <span className="text-slate-400 font-mono">{key}</span>
                                    <div className="flex items-center gap-2">
                                        <span className="text-red-400 line-through">{JSON.stringify(diff.v1)}</span>
                                        <span className="text-slate-500">&rarr;</span>
                                        <span className="text-emerald-400">{JSON.stringify(diff.v2)}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default VersionHistory;
