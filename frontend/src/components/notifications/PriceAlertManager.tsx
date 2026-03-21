/* eslint-disable @typescript-eslint/no-explicit-any */
'use client'

import React, {useCallback, useEffect, useState} from "react";
import {
    AlertCircle,
    Bell,
    Check,
    ChevronDown,
    DollarSign,
    Loader2,
    Plus,
    Trash2,
    TrendingDown,
    TrendingUp,
} from "lucide-react";
import {priceAlerts} from "@/utils/api";
import {PriceAlert} from "@/types/all_types";

const PriceAlertManager = () => {
    const [alerts, setAlerts] = useState<PriceAlert[]>([]);
    const [loading, setLoading] = useState(true);
    const [showForm, setShowForm] = useState(false);
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState('');

    // Form state
    const [symbol, setSymbol] = useState('');
    const [condition, setCondition] = useState<'above' | 'below'>('above');
    const [targetPrice, setTargetPrice] = useState('');

    const fetchAlerts = useCallback(async () => {
        try {
            const res = await priceAlerts.list();
            setAlerts(Array.isArray(res) ? res : []);
        } catch {
            // silent
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchAlerts();
    }, [fetchAlerts]);

    const handleCreate = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!symbol.trim() || !targetPrice) return;
        setSubmitting(true);
        setError('');
        try {
            await priceAlerts.create({
                symbol: symbol.trim().toUpperCase(),
                condition,
                target_price: parseFloat(targetPrice),
            });
            setSymbol('');
            setTargetPrice('');
            setShowForm(false);
            await fetchAlerts();
        } catch (err: any) {
            setError(err?.response?.data?.detail || 'Failed to create alert');
        } finally {
            setSubmitting(false);
        }
    };

    const handleDelete = async (id: number) => {
        try {
            await priceAlerts.delete(id);
            setAlerts(prev => prev.filter(a => a.id !== id));
        } catch {
            // silent
        }
    };

    const activeAlerts = alerts.filter(a => a.is_active);
    const triggeredAlerts = alerts.filter(a => !a.is_active);

    if (loading) {
        return (
            <div className="flex items-center justify-center py-8">
                <Loader2 size={24} className="animate-spin text-violet-400"/>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center shadow-lg shadow-amber-500/20">
                        <DollarSign size={20} className="text-white"/>
                    </div>
                    <div>
                        <h3 className="text-lg font-bold text-slate-200">Price Alerts</h3>
                        <p className="text-xs text-slate-500">{activeAlerts.length} active alert{activeAlerts.length !== 1 ? 's' : ''}</p>
                    </div>
                </div>
                <button
                    onClick={() => setShowForm(!showForm)}
                    className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white text-sm font-medium rounded-lg transition-all shadow-lg shadow-violet-500/20"
                >
                    <Plus size={16}/>
                    Add Alert
                </button>
            </div>

            {/* Create form */}
            {showForm && (
                <form onSubmit={handleCreate} className="p-4 bg-slate-800/30 border border-slate-700/50 rounded-xl space-y-4">
                    <div className="grid grid-cols-3 gap-3">
                        <div>
                            <label className="text-xs font-semibold text-slate-400 mb-1 block">Symbol</label>
                            <input
                                type="text"
                                value={symbol}
                                onChange={(e) => setSymbol(e.target.value)}
                                placeholder="AAPL"
                                className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-violet-500 outline-none"
                                required
                            />
                        </div>
                        <div>
                            <label className="text-xs font-semibold text-slate-400 mb-1 block">Condition</label>
                            <div className="relative">
                                <select
                                    value={condition}
                                    onChange={(e) => setCondition(e.target.value as 'above' | 'below')}
                                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm text-slate-200 focus:border-violet-500 outline-none appearance-none"
                                >
                                    <option value="above">Above</option>
                                    <option value="below">Below</option>
                                </select>
                                <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none"/>
                            </div>
                        </div>
                        <div>
                            <label className="text-xs font-semibold text-slate-400 mb-1 block">Target Price</label>
                            <input
                                type="number"
                                step="0.01"
                                min="0"
                                value={targetPrice}
                                onChange={(e) => setTargetPrice(e.target.value)}
                                placeholder="250.00"
                                className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-violet-500 outline-none"
                                required
                            />
                        </div>
                    </div>
                    {error && (
                        <p className="text-xs text-red-400 flex items-center gap-1">
                            <AlertCircle size={12}/> {error}
                        </p>
                    )}
                    <div className="flex items-center gap-2">
                        <button
                            type="submit"
                            disabled={submitting}
                            className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-500 disabled:bg-slate-700 text-white text-sm font-medium rounded-lg transition-all"
                        >
                            {submitting ? <Loader2 size={14} className="animate-spin"/> : <Check size={14}/>}
                            Create Alert
                        </button>
                        <button
                            type="button"
                            onClick={() => setShowForm(false)}
                            className="px-4 py-2 text-sm text-slate-400 hover:text-slate-200 transition-colors"
                        >
                            Cancel
                        </button>
                    </div>
                </form>
            )}

            {/* Active alerts */}
            {activeAlerts.length > 0 ? (
                <div className="space-y-2">
                    <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Active Alerts</h4>
                    {activeAlerts.map(alert => (
                        <div
                            key={alert.id}
                            className="flex items-center justify-between p-3 bg-slate-800/30 border border-slate-700/30 rounded-xl hover:border-slate-700/50 transition-all group"
                        >
                            <div className="flex items-center gap-3">
                                <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                                    alert.condition === 'above' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                                }`}>
                                    {alert.condition === 'above' ? <TrendingUp size={16}/> : <TrendingDown size={16}/>}
                                </div>
                                <div>
                                    <p className="text-sm font-semibold text-slate-200">
                                        {alert.symbol}
                                        <span className="text-slate-500 font-normal ml-2">
                                            {alert.condition === 'above' ? '>' : '<'} ${alert.target_price.toFixed(2)}
                                        </span>
                                    </p>
                                    <p className="text-xs text-slate-500">
                                        Created {new Date(alert.created_at).toLocaleDateString()}
                                    </p>
                                </div>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="text-xs px-2 py-1 bg-emerald-500/10 text-emerald-400 rounded-full font-medium">
                                    Active
                                </span>
                                <button
                                    onClick={() => handleDelete(alert.id)}
                                    className="text-slate-600 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
                                >
                                    <Trash2 size={14}/>
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            ) : (
                <div className="p-6 text-center bg-slate-800/20 border border-slate-800/30 rounded-xl">
                    <Bell size={28} className="text-slate-700 mx-auto mb-2"/>
                    <p className="text-sm text-slate-500">No active price alerts</p>
                    <p className="text-xs text-slate-600 mt-1">Create one to get notified when a price target is reached</p>
                </div>
            )}

            {/* Triggered alerts (history) */}
            {triggeredAlerts.length > 0 && (
                <div className="space-y-2">
                    <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Triggered</h4>
                    {triggeredAlerts.map(alert => (
                        <div
                            key={alert.id}
                            className="flex items-center justify-between p-3 bg-slate-800/10 border border-slate-800/20 rounded-xl opacity-60 group"
                        >
                            <div className="flex items-center gap-3">
                                <div className="w-8 h-8 rounded-lg bg-slate-800/50 text-slate-500 flex items-center justify-center">
                                    <Check size={16}/>
                                </div>
                                <div>
                                    <p className="text-sm text-slate-400">
                                        {alert.symbol} {alert.condition === 'above' ? '>' : '<'} ${alert.target_price.toFixed(2)}
                                    </p>
                                    <p className="text-xs text-slate-600">
                                        Triggered {alert.triggered_at ? new Date(alert.triggered_at).toLocaleDateString() : ''}
                                    </p>
                                </div>
                            </div>
                            <button
                                onClick={() => handleDelete(alert.id)}
                                className="text-slate-600 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
                            >
                                <Trash2 size={14}/>
                            </button>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default PriceAlertManager;
