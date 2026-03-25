/* eslint-disable @typescript-eslint/no-explicit-any */
'use client'

import React, {useCallback, useEffect, useState} from "react";
import {
    AlertCircle,
    ArrowDown,
    ArrowUp,
    BarChart3,
    ChevronDown,
    Eye,
    Filter,
    Loader2,
    Plus,
    RefreshCw,
    Search,
    Star,
    Trash2,
    TrendingDown,
    TrendingUp,
    X,
} from "lucide-react";
import {watchlists as watchlistsApi, market as marketApi} from "@/utils/api";
import {Watchlist, WatchlistQuote, ScreenerResult} from "@/types/all_types";

type Tab = 'watchlist' | 'screener';

const WatchlistPage = () => {
    const [tab, setTab] = useState<Tab>('watchlist');
    const [lists, setLists] = useState<Watchlist[]>([]);
    const [activeListId, setActiveListId] = useState<number | null>(null);
    const [quotes, setQuotes] = useState<WatchlistQuote[]>([]);
    const [loading, setLoading] = useState(true);
    const [quotesLoading, setQuotesLoading] = useState(false);
    const [showCreateForm, setShowCreateForm] = useState(false);
    const [newListName, setNewListName] = useState('');
    const [addSymbol, setAddSymbol] = useState('');
    const [searchResults, setSearchResults] = useState<{symbol: string; name?: string}[]>([]);
    const [searchOpen, setSearchOpen] = useState(false);
    const [error, setError] = useState('');

    // Screener state
    const [screenerResults, setScreenerResults] = useState<ScreenerResult[]>([]);
    const [screenerLoading, setScreenerLoading] = useState(false);
    const [filters, setFilters] = useState({
        min_price: '',
        max_price: '',
        min_change_pct: '',
        max_change_pct: '',
        min_volume: '',
    });

    const fetchLists = useCallback(async () => {
        try {
            const res = await watchlistsApi.list();
            const data = Array.isArray(res) ? res : [];
            setLists(data);
            if (data.length > 0 && !activeListId) {
                setActiveListId(data[0].id);
            }
        } catch {
            // silent
        } finally {
            setLoading(false);
        }
    }, [activeListId]);

    const fetchQuotes = useCallback(async () => {
        if (!activeListId) return;
        setQuotesLoading(true);
        try {
            const res = await watchlistsApi.getQuotes(activeListId);
            setQuotes(Array.isArray(res) ? res : []);
        } catch {
            setQuotes([]);
        } finally {
            setQuotesLoading(false);
        }
    }, [activeListId]);

    useEffect(() => {
        fetchLists();
    }, [fetchLists]);

    useEffect(() => {
        if (activeListId) {
            fetchQuotes();
            const interval = setInterval(fetchQuotes, 30000);
            return () => clearInterval(interval);
        }
    }, [activeListId, fetchQuotes]);

    const handleCreateList = async () => {
        if (!newListName.trim()) return;
        try {
            const res = await watchlistsApi.create(newListName.trim());
            setLists(prev => [...prev, res]);
            setActiveListId(res.id);
            setNewListName('');
            setShowCreateForm(false);
        } catch {
            setError('Failed to create watchlist');
        }
    };

    const handleDeleteList = async (id: number) => {
        try {
            await watchlistsApi.remove(id);
            setLists(prev => prev.filter(l => l.id !== id));
            if (activeListId === id) {
                const remaining = lists.filter(l => l.id !== id);
                setActiveListId(remaining.length > 0 ? remaining[0].id : null);
            }
        } catch {
            // silent
        }
    };

    const handleAddSymbol = async (symbolOverride?: string) => {
        const sym = (symbolOverride || addSymbol).trim().toUpperCase();
        if (!sym || !activeListId) return;
        setError('');
        setSearchOpen(false);
        setSearchResults([]);
        try {
            await watchlistsApi.addSymbol(activeListId, sym);
            setAddSymbol('');
            await fetchLists();
            await fetchQuotes();
        } catch (err: any) {
            setError(err?.response?.data?.detail || 'Failed to add symbol');
        }
    };

    // Debounced symbol search
    useEffect(() => {
        if (addSymbol.trim().length < 1) {
            setSearchResults([]);
            setSearchOpen(false);
            return;
        }
        const timer = setTimeout(async () => {
            try {
                const results = await marketApi.search(addSymbol.trim(), 8);
                setSearchResults(results as any[]);
                setSearchOpen(results.length > 0);
            } catch {
                setSearchResults([]);
            }
        }, 300);
        return () => clearTimeout(timer);
    }, [addSymbol]);

    const handleRemoveSymbol = async (symbol: string) => {
        if (!activeListId) return;
        try {
            await watchlistsApi.removeSymbol(activeListId, symbol);
            setQuotes(prev => prev.filter(q => q.symbol !== symbol));
            await fetchLists();
        } catch {
            // silent
        }
    };

    const handleScreen = async () => {
        setScreenerLoading(true);
        try {
            const f: any = {};
            if (filters.min_price) f.min_price = parseFloat(filters.min_price);
            if (filters.max_price) f.max_price = parseFloat(filters.max_price);
            if (filters.min_change_pct) f.min_change_pct = parseFloat(filters.min_change_pct);
            if (filters.max_change_pct) f.max_change_pct = parseFloat(filters.max_change_pct);
            if (filters.min_volume) f.min_volume = parseInt(filters.min_volume);
            const res = await watchlistsApi.screen(f);
            setScreenerResults(Array.isArray(res) ? res : []);
        } catch {
            setScreenerResults([]);
        } finally {
            setScreenerLoading(false);
        }
    };

    const handleAddScreenerToWatchlist = async (symbol: string) => {
        if (!activeListId) {
            setError('Select or create a watchlist first');
            return;
        }
        try {
            await watchlistsApi.addSymbol(activeListId, symbol);
            await fetchLists();
        } catch {
            // silent
        }
    };

    const activeList = lists.find(l => l.id === activeListId);

    if (loading) {
        return (
            <div className="flex items-center justify-center py-20">
                <Loader2 size={32} className="animate-spin text-violet-400"/>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
                        <Eye size={24} className="text-white"/>
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-slate-100">Watchlist & Screener</h1>
                        <p className="text-sm text-slate-500">Track symbols and discover opportunities</p>
                    </div>
                </div>
                {/* Tab switcher */}
                <div className="flex bg-slate-800/50 rounded-lg p-1">
                    <button
                        onClick={() => setTab('watchlist')}
                        className={`px-4 py-2 text-sm font-medium rounded-lg transition-all ${tab === 'watchlist' ? 'bg-violet-600 text-white' : 'text-slate-400 hover:text-slate-200'}`}
                    >
                        <Star size={14} className="inline mr-2"/>Watchlists
                    </button>
                    <button
                        onClick={() => setTab('screener')}
                        className={`px-4 py-2 text-sm font-medium rounded-lg transition-all ${tab === 'screener' ? 'bg-violet-600 text-white' : 'text-slate-400 hover:text-slate-200'}`}
                    >
                        <Filter size={14} className="inline mr-2"/>Screener
                    </button>
                </div>
            </div>

            {error && (
                <div className="flex items-center gap-2 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-sm text-red-400">
                    <AlertCircle size={16}/> {error}
                    <button onClick={() => setError('')} className="ml-auto"><X size={14}/></button>
                </div>
            )}

            {tab === 'watchlist' ? (
                <div className="grid grid-cols-12 gap-6">
                    {/* Sidebar — watchlist list */}
                    <div className="col-span-3 space-y-3">
                        <div className="flex items-center justify-between mb-2">
                            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">My Watchlists</h3>
                            <button
                                onClick={() => setShowCreateForm(!showCreateForm)}
                                className="text-violet-400 hover:text-violet-300 transition-colors"
                            >
                                <Plus size={16}/>
                            </button>
                        </div>

                        {showCreateForm && (
                            <div className="flex gap-2">
                                <input
                                    type="text"
                                    value={newListName}
                                    onChange={(e) => setNewListName(e.target.value)}
                                    onKeyDown={(e) => e.key === 'Enter' && handleCreateList()}
                                    placeholder="Watchlist name"
                                    className="flex-1 bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-violet-500 outline-none"
                                />
                                <button onClick={handleCreateList} className="px-3 py-2 bg-violet-600 text-white text-sm rounded-lg hover:bg-violet-500">
                                    Add
                                </button>
                            </div>
                        )}

                        {lists.map(list => (
                            <button
                                key={list.id}
                                onClick={() => setActiveListId(list.id)}
                                className={`w-full flex items-center justify-between p-3 rounded-xl text-left transition-all group ${
                                    activeListId === list.id
                                        ? 'bg-slate-800/70 border border-violet-500/30'
                                        : 'bg-slate-800/20 border border-slate-800/30 hover:bg-slate-800/40'
                                }`}
                            >
                                <div>
                                    <p className="text-sm font-semibold text-slate-200">{list.name}</p>
                                    <p className="text-xs text-slate-500">{list.items?.length || 0} symbols</p>
                                </div>
                                <button
                                    onClick={(e) => { e.stopPropagation(); handleDeleteList(list.id); }}
                                    className="text-slate-600 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
                                >
                                    <Trash2 size={14}/>
                                </button>
                            </button>
                        ))}

                        {lists.length === 0 && (
                            <p className="text-sm text-slate-600 text-center py-4">No watchlists yet</p>
                        )}
                    </div>

                    {/* Main — quotes table */}
                    <div className="col-span-9 space-y-4">
                        {activeList ? (
                            <>
                                <div className="flex items-center justify-between">
                                    <h2 className="text-lg font-bold text-slate-200">{activeList.name}</h2>
                                    <div className="flex items-center gap-3">
                                        <div className="flex gap-2 relative">
                                            <div className="relative">
                                                <div className="flex items-center bg-slate-800/50 border border-slate-700/50 rounded-lg focus-within:border-violet-500 w-56">
                                                    <Search size={14} className="ml-3 text-slate-500 shrink-0"/>
                                                    <input
                                                        type="text"
                                                        value={addSymbol}
                                                        onChange={(e) => setAddSymbol(e.target.value)}
                                                        onKeyDown={(e) => {
                                                            if (e.key === 'Enter') handleAddSymbol();
                                                            if (e.key === 'Escape') { setSearchOpen(false); setSearchResults([]); }
                                                        }}
                                                        onFocus={() => searchResults.length > 0 && setSearchOpen(true)}
                                                        onBlur={() => setTimeout(() => setSearchOpen(false), 200)}
                                                        placeholder="Search & add symbol..."
                                                        className="flex-1 bg-transparent px-2 py-2 text-sm text-slate-200 placeholder:text-slate-600 outline-none"
                                                    />
                                                </div>
                                                {/* Search dropdown */}
                                                {searchOpen && searchResults.length > 0 && (
                                                    <div className="absolute top-full left-0 right-0 mt-1 bg-slate-800 border border-slate-700 rounded-lg shadow-xl z-50 max-h-64 overflow-y-auto">
                                                        {searchResults.map((r) => (
                                                            <button
                                                                key={r.symbol}
                                                                onMouseDown={(e) => {
                                                                    e.preventDefault();
                                                                    handleAddSymbol(r.symbol);
                                                                }}
                                                                className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-slate-700/50 transition-colors"
                                                            >
                                                                <div>
                                                                    <span className="text-sm font-bold text-slate-200">{r.symbol}</span>
                                                                    {r.name && (
                                                                        <span className="text-xs text-slate-500 ml-2 truncate">{r.name}</span>
                                                                    )}
                                                                </div>
                                                                <Plus size={14} className="text-violet-400 shrink-0"/>
                                                            </button>
                                                        ))}
                                                    </div>
                                                )}
                                            </div>
                                            <button onClick={() => handleAddSymbol()} className="px-3 py-2 bg-violet-600 text-white text-sm rounded-lg hover:bg-violet-500 flex items-center gap-1">
                                                <Plus size={14}/> Add
                                            </button>
                                        </div>
                                        <button onClick={fetchQuotes} className="text-slate-400 hover:text-slate-200 transition-colors p-2">
                                            <RefreshCw size={16} className={quotesLoading ? 'animate-spin' : ''}/>
                                        </button>
                                    </div>
                                </div>

                                {quotes.length > 0 ? (
                                    <div className="bg-slate-800/20 border border-slate-800/30 rounded-xl overflow-hidden">
                                        <table className="w-full">
                                            <thead>
                                                <tr className="text-xs font-semibold text-slate-500 uppercase tracking-wider border-b border-slate-800/30">
                                                    <th className="text-left px-4 py-3">Symbol</th>
                                                    <th className="text-right px-4 py-3">Price</th>
                                                    <th className="text-right px-4 py-3">Change</th>
                                                    <th className="text-right px-4 py-3">Change %</th>
                                                    <th className="text-right px-4 py-3">Volume</th>
                                                    <th className="text-right px-4 py-3">Day Range</th>
                                                    <th className="text-right px-4 py-3"></th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {quotes.map((q) => {
                                                    const changePct = Number.isFinite(q.change_percent) ? q.change_percent : 0;
                                                    const isUp = changePct >= 0;
                                                    return (
                                                        <tr key={q.symbol} className="border-b border-slate-800/20 hover:bg-slate-800/30 transition-colors group">
                                                            <td className="px-4 py-3">
                                                                <div className="flex items-center gap-2">
                                                                    <div className={`w-6 h-6 rounded flex items-center justify-center ${isUp ? 'bg-emerald-500/20' : 'bg-red-500/20'}`}>
                                                                        {isUp ? <TrendingUp size={12} className="text-emerald-400"/> : <TrendingDown size={12} className="text-red-400"/>}
                                                                    </div>
                                                                    <span className="text-sm font-bold text-slate-200">{q.symbol}</span>
                                                                </div>
                                                            </td>
                                                            <td className="px-4 py-3 text-right text-sm font-semibold text-slate-200">${q.price?.toFixed(2)}</td>
                                                            <td className={`px-4 py-3 text-right text-sm font-medium ${isUp ? 'text-emerald-400' : 'text-red-400'}`}>
                                                                {isUp ? '+' : ''}{q.change?.toFixed(2)}
                                                            </td>
                                                            <td className="px-4 py-3 text-right">
                                                                <span className={`inline-flex items-center gap-1 text-sm font-medium px-2 py-0.5 rounded-full ${isUp ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                                                                    {isUp ? <ArrowUp size={10}/> : <ArrowDown size={10}/>}
                                                                    {Math.abs(changePct).toFixed(2)}%
                                                                </span>
                                                            </td>
                                                            <td className="px-4 py-3 text-right text-sm text-slate-400">
                                                                {q.volume ? (q.volume / 1e6).toFixed(1) + 'M' : '—'}
                                                            </td>
                                                            <td className="px-4 py-3 text-right text-xs text-slate-500">
                                                                {q.day_low?.toFixed(2)} — {q.day_high?.toFixed(2)}
                                                            </td>
                                                            <td className="px-4 py-3 text-right">
                                                                <button
                                                                    onClick={() => handleRemoveSymbol(q.symbol)}
                                                                    className="text-slate-600 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
                                                                >
                                                                    <Trash2 size={14}/>
                                                                </button>
                                                            </td>
                                                        </tr>
                                                    );
                                                })}
                                            </tbody>
                                        </table>
                                    </div>
                                ) : (
                                    <div className="p-12 text-center bg-slate-800/20 border border-slate-800/30 rounded-xl">
                                        <BarChart3 size={36} className="text-slate-700 mx-auto mb-3"/>
                                        <p className="text-sm text-slate-500">No symbols in this watchlist</p>
                                        <p className="text-xs text-slate-600 mt-1">Add symbols above to start tracking</p>
                                    </div>
                                )}
                            </>
                        ) : (
                            <div className="p-12 text-center bg-slate-800/20 border border-slate-800/30 rounded-xl">
                                <Star size={36} className="text-slate-700 mx-auto mb-3"/>
                                <p className="text-sm text-slate-500">Create a watchlist to get started</p>
                            </div>
                        )}
                    </div>
                </div>
            ) : (
                /* Screener Tab */
                <div className="space-y-6">
                    {/* Filter panel */}
                    <div className="p-4 bg-slate-800/30 border border-slate-700/50 rounded-xl">
                        <h3 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
                            <Filter size={16} className="text-violet-400"/>
                            Screening Filters
                        </h3>
                        <div className="grid grid-cols-5 gap-4">
                            <div>
                                <label className="text-xs font-semibold text-slate-500 mb-1 block">Min Price</label>
                                <input
                                    type="number" step="0.01" placeholder="0"
                                    value={filters.min_price}
                                    onChange={(e) => setFilters(f => ({...f, min_price: e.target.value}))}
                                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-violet-500 outline-none"
                                />
                            </div>
                            <div>
                                <label className="text-xs font-semibold text-slate-500 mb-1 block">Max Price</label>
                                <input
                                    type="number" step="0.01" placeholder="∞"
                                    value={filters.max_price}
                                    onChange={(e) => setFilters(f => ({...f, max_price: e.target.value}))}
                                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-violet-500 outline-none"
                                />
                            </div>
                            <div>
                                <label className="text-xs font-semibold text-slate-500 mb-1 block">Min Change %</label>
                                <input
                                    type="number" step="0.1" placeholder="-∞"
                                    value={filters.min_change_pct}
                                    onChange={(e) => setFilters(f => ({...f, min_change_pct: e.target.value}))}
                                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-violet-500 outline-none"
                                />
                            </div>
                            <div>
                                <label className="text-xs font-semibold text-slate-500 mb-1 block">Max Change %</label>
                                <input
                                    type="number" step="0.1" placeholder="∞"
                                    value={filters.max_change_pct}
                                    onChange={(e) => setFilters(f => ({...f, max_change_pct: e.target.value}))}
                                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-violet-500 outline-none"
                                />
                            </div>
                            <div>
                                <label className="text-xs font-semibold text-slate-500 mb-1 block">Min Volume</label>
                                <input
                                    type="number" step="1" placeholder="0"
                                    value={filters.min_volume}
                                    onChange={(e) => setFilters(f => ({...f, min_volume: e.target.value}))}
                                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-violet-500 outline-none"
                                />
                            </div>
                        </div>
                        <div className="mt-4 flex items-center gap-3">
                            <button
                                onClick={handleScreen}
                                disabled={screenerLoading}
                                className="flex items-center gap-2 px-5 py-2 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white text-sm font-medium rounded-lg transition-all shadow-lg shadow-violet-500/20"
                            >
                                {screenerLoading ? <Loader2 size={14} className="animate-spin"/> : <Search size={14}/>}
                                Screen
                            </button>
                            <button
                                onClick={() => { setFilters({min_price: '', max_price: '', min_change_pct: '', max_change_pct: '', min_volume: ''}); setScreenerResults([]); }}
                                className="text-sm text-slate-400 hover:text-slate-200 transition-colors"
                            >
                                Clear filters
                            </button>
                        </div>
                    </div>

                    {/* Screener results */}
                    {screenerResults.length > 0 ? (
                        <div className="bg-slate-800/20 border border-slate-800/30 rounded-xl overflow-hidden">
                            <div className="px-4 py-3 border-b border-slate-800/30 flex items-center justify-between">
                                <p className="text-sm text-slate-400">{screenerResults.length} results</p>
                                {activeList && (
                                    <p className="text-xs text-slate-500">Adding to: <span className="text-violet-400">{activeList.name}</span></p>
                                )}
                            </div>
                            <table className="w-full">
                                <thead>
                                    <tr className="text-xs font-semibold text-slate-500 uppercase tracking-wider border-b border-slate-800/30">
                                        <th className="text-left px-4 py-3">Symbol</th>
                                        <th className="text-right px-4 py-3">Price</th>
                                        <th className="text-right px-4 py-3">Change %</th>
                                        <th className="text-right px-4 py-3">Volume</th>
                                        <th className="text-right px-4 py-3"></th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {screenerResults.map((r) => {
                                        const pct = Number.isFinite(r.change_percent) ? r.change_percent : 0;
                                        const isUp = pct >= 0;
                                        return (
                                            <tr key={r.symbol} className="border-b border-slate-800/20 hover:bg-slate-800/30 transition-colors">
                                                <td className="px-4 py-3 text-sm font-bold text-slate-200">{r.symbol}</td>
                                                <td className="px-4 py-3 text-right text-sm text-slate-200">${r.price?.toFixed(2)}</td>
                                                <td className={`px-4 py-3 text-right text-sm font-medium ${isUp ? 'text-emerald-400' : 'text-red-400'}`}>
                                                    {isUp ? '+' : ''}{pct.toFixed(2)}%
                                                </td>
                                                <td className="px-4 py-3 text-right text-sm text-slate-400">
                                                    {r.volume ? (r.volume / 1e6).toFixed(1) + 'M' : '—'}
                                                </td>
                                                <td className="px-4 py-3 text-right">
                                                    <button
                                                        onClick={() => handleAddScreenerToWatchlist(r.symbol)}
                                                        className="text-xs px-3 py-1 bg-violet-600/20 text-violet-400 rounded-full hover:bg-violet-600/30 transition-colors"
                                                    >
                                                        + Watchlist
                                                    </button>
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    ) : !screenerLoading && (
                        <div className="p-12 text-center bg-slate-800/20 border border-slate-800/30 rounded-xl">
                            <Search size={36} className="text-slate-700 mx-auto mb-3"/>
                            <p className="text-sm text-slate-500">Set your filters and click Screen to discover symbols</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default WatchlistPage;
