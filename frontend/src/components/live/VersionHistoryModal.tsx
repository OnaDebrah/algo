'use client'

import React, { useEffect, useState } from 'react';
import { History, RotateCcw, X, Clock, FileJson, CheckCircle2 } from 'lucide-react';
import { live } from '@/utils/api';

interface VersionHistoryModalProps {
    strategyId: number;
    strategyName: string;
    isOpen: boolean;
    onClose: () => void;
    onRollbackSuccess: () => void;
}

const VersionHistoryModal = ({ strategyId, strategyName, isOpen, onClose, onRollbackSuccess }: VersionHistoryModalProps) => {
    const [versions, setVersions] = useState<any[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [rollbackLoading, setRollbackLoading] = useState<number | null>(null);

    useEffect(() => {
        if (isOpen && strategyId) {
            fetchVersions();
        }
    }, [isOpen, strategyId]);

    const fetchVersions = async () => {
        setIsLoading(true);
        setError(null);
        try {
            const data = await live.getVersions(strategyId);
            setVersions(data);
        } catch (err) {
            console.error("Failed to fetch versions:", err);
            setError("Failed to load version history");
        } finally {
            setIsLoading(false);
        }
    };

    const handleRollback = async (versionId: number, versionNum: number) => {
        if (!window.confirm(`Are you sure you want to rollback to version ${versionNum}?`)) {
            return;
        }

        setRollbackLoading(versionId);
        try {
            await live.rollback(strategyId, versionId);
            onRollbackSuccess();
            onClose();
        } catch (err) {
            console.error("Rollback failed:", err);
            alert("Failed to rollback strategy parameters");
        } finally {
            setRollbackLoading(null);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-300">
            <div className="bg-slate-900 border border-slate-800 w-full max-w-2xl rounded-3xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-300">
                {/* Header */}
                <div className="px-6 py-5 border-b border-slate-800 flex items-center justify-between bg-gradient-to-r from-slate-900 to-slate-800">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-violet-500/20 rounded-xl">
                            <History className="text-violet-400" size={20} />
                        </div>
                        <div>
                            <h3 className="text-lg font-bold text-slate-100 uppercase tracking-tight">Version History</h3>
                            <p className="text-xs text-slate-500">{strategyName}</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-slate-800 rounded-full transition-colors text-slate-500 hover:text-slate-200"
                    >
                        <X size={20} />
                    </button>
                </div>

                {/* Body */}
                <div className="p-6 max-h-[60vh] overflow-y-auto custom-scrollbar">
                    {isLoading ? (
                        <div className="flex flex-col items-center justify-center py-12 space-y-4">
                            <Clock className="text-slate-700 animate-pulse" size={48} />
                            <p className="text-sm text-slate-500">Loading snapshots...</p>
                        </div>
                    ) : error ? (
                        <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-center text-red-400 text-sm">
                            {error}
                        </div>
                    ) : versions.length === 0 ? (
                        <div className="text-center py-12">
                            <div className="p-4 bg-slate-800/30 rounded-full w-fit mx-auto mb-4">
                                <History className="text-slate-600" size={32} />
                            </div>
                            <p className="text-slate-400 font-medium">No previous versions found</p>
                            <p className="text-xs text-slate-600 mt-1">Snapshots are created automatically when you update parameters.</p>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            {versions.map((v) => (
                                <div
                                    key={v.id}
                                    className="group p-5 bg-slate-800/30 border border-slate-800/50 rounded-2xl hover:border-slate-700 transition-all"
                                >
                                    <div className="flex items-start justify-between gap-4">
                                        <div className="flex-1">
                                            <div className="flex items-center gap-2 mb-1">
                                                <span className="text-xs font-black px-2 py-0.5 bg-violet-500/20 text-violet-400 rounded uppercase">
                                                    v{v.version}
                                                </span>
                                                <span className="text-xs text-slate-500 font-mono">
                                                    {new Date(v.created_at).toLocaleString()}
                                                </span>
                                            </div>
                                            <p className="text-sm text-slate-300 line-clamp-2 italic mb-3">
                                                "{v.notes || 'No notes provided'}"
                                            </p>

                                            {/* Parameters Preview */}
                                            <div className="bg-slate-950/50 rounded-lg p-3 border border-slate-800/50">
                                                <div className="flex items-center gap-2 mb-2 text-[10px] font-black text-slate-600 uppercase">
                                                    <FileJson size={12} /> Parameter Snapshot
                                                </div>
                                                <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                                                    {Object.entries(v.parameters || {}).slice(0, 4).map(([key, val]) => (
                                                        <div key={key} className="flex justify-between items-center text-[10px] font-mono">
                                                            <span className="text-slate-500 truncate mr-2">{key}:</span>
                                                            <span className="text-slate-300 font-bold">{String(val)}</span>
                                                        </div>
                                                    ))}
                                                    {Object.keys(v.parameters || {}).length > 4 && (
                                                        <div className="text-[10px] text-slate-600 italic">
                                                            +{Object.keys(v.parameters || {}).length - 4} more...
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        </div>

                                        <button
                                            onClick={() => handleRollback(v.id, v.version)}
                                            disabled={rollbackLoading !== null}
                                            className="px-4 py-2 bg-emerald-500/10 border border-emerald-500/30 hover:bg-emerald-500/20 text-emerald-400 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all flex items-center gap-2 shrink-0 self-center group-hover:scale-105"
                                        >
                                            {rollbackLoading === v.id ? (
                                                <div className="w-3 h-3 border-2 border-emerald-400/30 border-t-emerald-400 rounded-full animate-spin" />
                                            ) : (
                                                <RotateCcw size={12} />
                                            )}
                                            Rollback
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="px-6 py-4 bg-slate-950 flex items-center justify-between border-t border-slate-800">
                    <p className="text-[10px] text-slate-500 flex items-center gap-1.5 uppercase tracking-wider font-bold">
                        <CheckCircle2 size={12} className="text-emerald-500" /> Auto-snapshotting enabled
                    </p>
                    <button
                        onClick={onClose}
                        className="px-6 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-[10px] font-black uppercase tracking-widest rounded-xl transition-all"
                    >
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
};

export default VersionHistoryModal;
