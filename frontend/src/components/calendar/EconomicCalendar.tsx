/* eslint-disable @typescript-eslint/no-explicit-any */

'use client'

import { useEffect, useState } from 'react';
import { Calendar, Loader2, RefreshCw } from 'lucide-react';
import { client } from '@/utils/api';

interface EconomicEventItem {
    id: number;
    event_name: string;
    country: string;
    event_date: string;
    impact: string;
    previous_value?: string;
    forecast_value?: string;
    actual_value?: string;
    category?: string;
}

const IMPACT_COLORS: Record<string, string> = {
    high: 'bg-red-900/30 text-red-400 border-red-700/50',
    medium: 'bg-amber-900/30 text-amber-400 border-amber-700/50',
    low: 'bg-emerald-900/30 text-emerald-400 border-emerald-700/50',
};

const IMPACT_DOT: Record<string, string> = {
    high: 'bg-red-400',
    medium: 'bg-amber-400',
    low: 'bg-emerald-400',
};

const EconomicCalendar = () => {
    const [events, setEvents] = useState<EconomicEventItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [viewDays, setViewDays] = useState(7);
    const [impactFilter, setImpactFilter] = useState('');
    const [syncing, setSyncing] = useState(false);

    const handleSync = async () => {
        setSyncing(true);
        try {
            await client.post('/calendar/sync', null, { params: { months: 6 } });
            await fetchEvents();
        } catch {
            console.error('Failed to sync calendar');
        } finally {
            setSyncing(false);
        }
    };

    const fetchEvents = async () => {
        setLoading(true);
        try {
            const params: any = { days: viewDays };
            if (impactFilter) params.impact = impactFilter;
            const data = await client.get('/calendar/upcoming', { params });
            setEvents(data as EconomicEventItem[]);
        } catch {
            console.error('Failed to fetch calendar events');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { fetchEvents(); }, [viewDays, impactFilter]);

    // Group events by date
    const groupedEvents: Record<string, EconomicEventItem[]> = {};
    events.forEach((e) => {
        const date = e.event_date ? new Date(e.event_date).toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' }) : 'Unknown';
        if (!groupedEvents[date]) groupedEvents[date] = [];
        groupedEvents[date].push(e);
    });

    return (
        <div className="p-6 space-y-6 max-w-5xl mx-auto">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center border border-amber-500/20">
                        <Calendar className="text-amber-400" size={20} />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-slate-100">Economic Calendar</h1>
                        <p className="text-sm text-slate-500">Upcoming market-moving events</p>
                    </div>
                </div>
                <button
                    onClick={handleSync}
                    disabled={syncing}
                    className="flex items-center gap-2 px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
                >
                    <RefreshCw size={14} className={syncing ? 'animate-spin' : ''} />
                    {syncing ? 'Syncing...' : 'Sync Events'}
                </button>
            </div>

            {/* Filters */}
            <div className="flex items-center gap-3">
                <div className="flex bg-slate-800 rounded-lg p-0.5">
                    {[7, 14, 30].map((d) => (
                        <button
                            key={d}
                            onClick={() => setViewDays(d)}
                            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                                viewDays === d ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'
                            }`}
                        >
                            {d}D
                        </button>
                    ))}
                </div>
                <div className="flex bg-slate-800 rounded-lg p-0.5">
                    {['', 'high', 'medium', 'low'].map((imp) => (
                        <button
                            key={imp}
                            onClick={() => setImpactFilter(imp)}
                            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                                impactFilter === imp ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'
                            }`}
                        >
                            {imp || 'All'}
                        </button>
                    ))}
                </div>
            </div>

            {/* Events */}
            {loading ? (
                <div className="flex items-center justify-center py-20">
                    <Loader2 className="animate-spin text-slate-500" size={32} />
                </div>
            ) : events.length === 0 ? (
                <div className="text-center py-20">
                    <Calendar className="mx-auto text-slate-600 mb-4" size={48} />
                    <p className="text-slate-400 text-lg">No upcoming events</p>
                    <p className="text-sm text-slate-500 mt-1">Check back later or expand the date range</p>
                </div>
            ) : (
                <div className="space-y-6">
                    {Object.entries(groupedEvents).map(([date, dayEvents]) => (
                        <div key={date}>
                            <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-3 border-b border-slate-800 pb-2">
                                {date}
                            </h3>
                            <div className="space-y-2">
                                {dayEvents.map((e) => (
                                    <div
                                        key={e.id}
                                        className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-4 hover:border-slate-700/50 transition-all"
                                    >
                                        <div className="flex items-center justify-between">
                                            <div className="flex items-center gap-3">
                                                <div className={`w-2.5 h-2.5 rounded-full ${IMPACT_DOT[e.impact] || IMPACT_DOT.low}`} />
                                                <div>
                                                    <span className="text-sm font-semibold text-slate-200">{e.event_name}</span>
                                                    <div className="flex items-center gap-2 mt-0.5">
                                                        <span className={`px-1.5 py-0.5 rounded text-xs font-semibold border ${IMPACT_COLORS[e.impact] || IMPACT_COLORS.low}`}>
                                                            {e.impact}
                                                        </span>
                                                        {e.category && (
                                                            <span className="text-xs text-slate-500">{e.category}</span>
                                                        )}
                                                        <span className="text-xs text-slate-500">{e.country}</span>
                                                    </div>
                                                </div>
                                            </div>
                                            <div className="flex items-center gap-4 text-xs">
                                                {e.previous_value && (
                                                    <div className="text-center">
                                                        <p className="text-slate-500">Previous</p>
                                                        <p className="text-slate-300 font-semibold">{e.previous_value}</p>
                                                    </div>
                                                )}
                                                {e.forecast_value && (
                                                    <div className="text-center">
                                                        <p className="text-slate-500">Forecast</p>
                                                        <p className="text-indigo-400 font-semibold">{e.forecast_value}</p>
                                                    </div>
                                                )}
                                                {e.actual_value && (
                                                    <div className="text-center">
                                                        <p className="text-slate-500">Actual</p>
                                                        <p className="text-emerald-400 font-semibold">{e.actual_value}</p>
                                                    </div>
                                                )}
                                                <span className="text-slate-500">
                                                    {new Date(e.event_date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default EconomicCalendar;
