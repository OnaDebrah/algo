/* eslint-disable @typescript-eslint/no-explicit-any */

'use client'

import { useEffect, useState } from 'react';
import {
    BookOpen,
    Check,
    ChevronLeft,
    ChevronRight,
    Download,
    Edit3,
    Filter,
    Loader2,
    Search,
    Tag,
    X,
} from 'lucide-react';
import { auditApi } from '@/utils/api';

interface AuditEvent {
    id: number;
    user_id: number;
    event_type: string;
    category: string;
    title: string;
    description?: string;
    metadata?: Record<string, any>;
    tags?: string[];
    notes?: string;
    created_at: string;
}

const EVENT_TYPE_COLORS: Record<string, string> = {
    trade: 'bg-emerald-900/30 text-emerald-400 border-emerald-700/50',
    login: 'bg-blue-900/30 text-blue-400 border-blue-700/50',
    strategy_deploy: 'bg-violet-900/30 text-violet-400 border-violet-700/50',
    settings_change: 'bg-amber-900/30 text-amber-400 border-amber-700/50',
};

const EVENT_TYPES = [
    { value: '', label: 'All Types' },
    { value: 'trade', label: 'Trades' },
    { value: 'login', label: 'Logins' },
    { value: 'strategy_deploy', label: 'Deployments' },
    { value: 'settings_change', label: 'Settings' },
];

const CATEGORIES = [
    { value: '', label: 'All Categories' },
    { value: 'trade_journal', label: 'Trade Journal' },
    { value: 'system', label: 'System' },
    { value: 'security', label: 'Security' },
];

const TradeJournal = () => {
    const [events, setEvents] = useState<AuditEvent[]>([]);
    const [total, setTotal] = useState(0);
    const [page, setPage] = useState(1);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');
    const [filterType, setFilterType] = useState('');
    const [filterCategory, setFilterCategory] = useState('');
    const [editingId, setEditingId] = useState<number | null>(null);
    const [editNotes, setEditNotes] = useState('');
    const [editTags, setEditTags] = useState('');
    const [saving, setSaving] = useState(false);

    const pageSize = 25;

    const fetchEvents = async () => {
        setLoading(true);
        try {
            const params: any = { page, page_size: pageSize };
            if (filterType) params.event_type = filterType;
            if (filterCategory) params.category = filterCategory;
            if (searchQuery) params.search = searchQuery;

            const data = await auditApi.getEvents(params) as any;
            setEvents(data.events || []);
            setTotal(data.total || 0);
        } catch {
            console.error('Failed to fetch audit events');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { fetchEvents(); }, [page, filterType, filterCategory]);

    const handleSearch = () => {
        setPage(1);
        fetchEvents();
    };

    const startEdit = (event: AuditEvent) => {
        setEditingId(event.id);
        setEditNotes(event.notes || '');
        setEditTags((event.tags || []).join(', '));
    };

    const saveNotes = async () => {
        if (editingId === null) return;
        setSaving(true);
        try {
            const tags = editTags.split(',').map(t => t.trim()).filter(Boolean);
            await auditApi.updateNotes(editingId, { notes: editNotes, tags });
            setEditingId(null);
            await fetchEvents();
        } catch {
            alert('Failed to save notes');
        } finally {
            setSaving(false);
        }
    };

    const totalPages = Math.ceil(total / pageSize);

    return (
        <div className="p-6 space-y-6 max-w-7xl mx-auto">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-indigo-500/10 flex items-center justify-center border border-indigo-500/20">
                        <BookOpen className="text-indigo-400" size={20} />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-slate-100">Trade Journal</h1>
                        <p className="text-sm text-slate-500">{total} events recorded</p>
                    </div>
                </div>
                <button
                    onClick={() => auditApi.exportCsv(filterType || undefined)}
                    className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700/50 text-slate-300 rounded-lg text-sm transition-all"
                >
                    <Download size={16} /> Export CSV
                </button>
            </div>

            {/* Filters */}
            <div className="flex flex-wrap items-center gap-3">
                <div className="relative flex-1 min-w-[200px]">
                    <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                    <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                        placeholder="Search events..."
                        className="w-full pl-10 pr-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:border-indigo-500 outline-none"
                    />
                </div>
                <select
                    value={filterType}
                    onChange={(e) => { setFilterType(e.target.value); setPage(1); }}
                    className="px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-sm text-slate-300 outline-none"
                >
                    {EVENT_TYPES.map(t => (
                        <option key={t.value} value={t.value}>{t.label}</option>
                    ))}
                </select>
                <select
                    value={filterCategory}
                    onChange={(e) => { setFilterCategory(e.target.value); setPage(1); }}
                    className="px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-sm text-slate-300 outline-none"
                >
                    {CATEGORIES.map(c => (
                        <option key={c.value} value={c.value}>{c.label}</option>
                    ))}
                </select>
            </div>

            {/* Event Timeline */}
            {loading ? (
                <div className="flex items-center justify-center py-20">
                    <Loader2 className="animate-spin text-slate-500" size={32} />
                </div>
            ) : events.length === 0 ? (
                <div className="text-center py-20">
                    <BookOpen className="mx-auto text-slate-600 mb-4" size={48} />
                    <p className="text-slate-400 text-lg">No events found</p>
                    <p className="text-sm text-slate-500 mt-1">Trade and system events will appear here</p>
                </div>
            ) : (
                <div className="space-y-2">
                    {events.map((event) => (
                        <div
                            key={event.id}
                            className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-4 hover:border-slate-700/50 transition-all"
                        >
                            <div className="flex items-start justify-between gap-4">
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2 flex-wrap">
                                        <span className={`px-2 py-0.5 rounded text-xs font-semibold border ${EVENT_TYPE_COLORS[event.event_type] || 'bg-slate-800 text-slate-400 border-slate-700'}`}>
                                            {event.event_type.replace('_', ' ')}
                                        </span>
                                        <span className="text-sm font-semibold text-slate-200">{event.title}</span>
                                    </div>
                                    {event.description && (
                                        <p className="text-xs text-slate-400 mt-1">{event.description}</p>
                                    )}
                                    {/* Metadata preview */}
                                    {event.metadata && Object.keys(event.metadata).length > 0 && (
                                        <div className="flex flex-wrap gap-2 mt-2">
                                            {Object.entries(event.metadata).slice(0, 4).map(([k, v]) => (
                                                <span key={k} className="text-xs bg-slate-800 text-slate-400 px-2 py-0.5 rounded">
                                                    {k}: {typeof v === 'number' ? v.toFixed(2) : String(v)}
                                                </span>
                                            ))}
                                        </div>
                                    )}
                                    {/* Tags */}
                                    {event.tags && event.tags.length > 0 && (
                                        <div className="flex items-center gap-1 mt-2">
                                            <Tag size={12} className="text-slate-500" />
                                            {event.tags.map((tag) => (
                                                <span key={tag} className="text-xs bg-indigo-900/30 text-indigo-400 px-2 py-0.5 rounded-full border border-indigo-700/30">
                                                    {tag}
                                                </span>
                                            ))}
                                        </div>
                                    )}
                                    {/* Notes */}
                                    {editingId === event.id ? (
                                        <div className="mt-3 space-y-2">
                                            <textarea
                                                value={editNotes}
                                                onChange={(e) => setEditNotes(e.target.value)}
                                                placeholder="Add journal notes..."
                                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:border-indigo-500 outline-none resize-none"
                                                rows={3}
                                            />
                                            <input
                                                type="text"
                                                value={editTags}
                                                onChange={(e) => setEditTags(e.target.value)}
                                                placeholder="Tags (comma-separated)"
                                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:border-indigo-500 outline-none"
                                            />
                                            <div className="flex gap-2">
                                                <button
                                                    onClick={saveNotes}
                                                    disabled={saving}
                                                    className="flex items-center gap-1 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold transition-all disabled:opacity-50"
                                                >
                                                    {saving ? <Loader2 size={12} className="animate-spin" /> : <Check size={12} />}
                                                    Save
                                                </button>
                                                <button
                                                    onClick={() => setEditingId(null)}
                                                    className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-xs transition-all"
                                                >
                                                    Cancel
                                                </button>
                                            </div>
                                        </div>
                                    ) : event.notes ? (
                                        <div className="mt-2 flex items-start gap-2">
                                            <p className="text-xs text-slate-400 italic bg-slate-800/50 rounded px-2 py-1 flex-1">{event.notes}</p>
                                            <button onClick={() => startEdit(event)} className="p-1 hover:bg-slate-800 rounded transition-all">
                                                <Edit3 size={12} className="text-slate-500" />
                                            </button>
                                        </div>
                                    ) : null}
                                </div>
                                <div className="flex items-center gap-2 flex-shrink-0">
                                    <span className="text-xs text-slate-500">
                                        {new Date(event.created_at).toLocaleString()}
                                    </span>
                                    {editingId !== event.id && (
                                        <button
                                            onClick={() => startEdit(event)}
                                            className="p-1.5 hover:bg-slate-800 rounded-lg transition-all"
                                            title="Add notes"
                                        >
                                            <Edit3 size={14} className="text-slate-500" />
                                        </button>
                                    )}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Pagination */}
            {totalPages > 1 && (
                <div className="flex items-center justify-center gap-2">
                    <button
                        onClick={() => setPage(p => Math.max(1, p - 1))}
                        disabled={page === 1}
                        className="p-2 hover:bg-slate-800 rounded-lg transition-all disabled:opacity-30"
                    >
                        <ChevronLeft size={16} className="text-slate-400" />
                    </button>
                    <span className="text-sm text-slate-400">
                        Page {page} of {totalPages}
                    </span>
                    <button
                        onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                        disabled={page === totalPages}
                        className="p-2 hover:bg-slate-800 rounded-lg transition-all disabled:opacity-30"
                    >
                        <ChevronRight size={16} className="text-slate-400" />
                    </button>
                </div>
            )}
        </div>
    );
};

export default TradeJournal;
