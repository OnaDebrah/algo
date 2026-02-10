'use client'

import React, {useEffect, useMemo, useState} from 'react';
import {ChevronDown, Filter, LayoutGrid, List, Loader2, Package, Search, Shield, Users} from "lucide-react";
import {marketplace} from '@/utils/api';
import StrategyDetailsModal from '@/components/marketplace/StrategyDetailsModal';
import {StrategyCard} from "@/components/marketplace/StrategyCard";
import {StrategyListItem} from "@/components/marketplace/StrategyListItem";
import {StrategyListing} from "@/types/all_types";


const CATEGORIES = [
    {value: "all", label: "All Categories"},
    {value: "momentum", label: "Momentum"},
    {value: "mean_reversion", label: "Mean Reversion"},
    {value: "trend_following", label: "Trend Following"},
    {value: "breakout", label: "Breakout"},
    {value: "ml", label: "Machine Learning"},
    {value: "options", label: "Options"},
    {value: "arbitrage", label: "Arbitrage"},
];

const SORT_OPTIONS = [
    {value: "downloads", label: "Most Popular"},
    {value: "rating", label: "Highest Rated"},
    {value: "return", label: "Best Return"},
    {value: "newest", label: "Newest"},
];

const Marketplace = () => {
    const [strategies, setStrategies] = useState<StrategyListing[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const [selectedCategory, setSelectedCategory] = useState('all');
    const [searchQuery, setSearchQuery] = useState('');
    const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
    const [sortBy, setSortBy] = useState('downloads');
    const [showFilters, setShowFilters] = useState(false);
    const [selectedStrategyId, setSelectedStrategyId] = useState<number | null>(null);

    const [priceFilter, setPriceFilter] = useState<'all' | 'free' | 'paid'>('all');

    const [showVerifiedOnly, setShowVerifiedOnly] = useState(false);

    useEffect(() => {
        fetchStrategies();
    }, []);

    const fetchStrategies = async () => {
        setIsLoading(true);
        setError(null);

        try {
            const params: any = {};

            // Add filters
            if (selectedCategory !== 'all') {
                params.category = selectedCategory;
            }
            if (sortBy) {
                params.sort_by = sortBy;
            }
            if (priceFilter === 'free') {
                params.max_price = 0;
            }

            const response = await marketplace.browse(params);
            setStrategies(response || []);
        } catch (err: any) {
            console.error("Failed to fetch strategies:", err);
            setError(err?.message || "Failed to load marketplace");
            // Set mock data for testing
            setStrategies(getMockStrategies());
        } finally {
            setIsLoading(false);
        }
    };

    const toggleFavorite = async (e: React.MouseEvent, strategyId: number, isCurrentlyFavorite: boolean) => {
        e.stopPropagation();
        try {
            if (isCurrentlyFavorite) {
                await marketplace.unfavorite(strategyId);
            } else {
                await marketplace.favorite(strategyId);
            }

            setStrategies((prev: StrategyListing[]) => prev.map((s: StrategyListing) =>
                s.id === strategyId ? {...s, is_favorite: !isCurrentlyFavorite} : s
            ));
        } catch (err) {
            console.error("Failed to toggle favorite:", err);
        }
    };

    const handleDownload = async (e: React.MouseEvent, strategyId: number) => {
        e.stopPropagation();
        try {
            await marketplace.recordDownload(strategyId);

            setStrategies((prev: StrategyListing[]) => prev.map((s: StrategyListing) =>
                s.id === strategyId ? {...s, total_downloads: (s.total_downloads || 0) + 1} : s
            ));

            alert("Strategy download recorded! Check your strategies page.");
        } catch (err) {
            console.error("Failed to record download:", err);
        }
    };

    // Filter and sort strategies
    const filteredStrategies = useMemo(() => {
        let filtered = strategies;

        // Search filter
        if (searchQuery) {
            const query = searchQuery.toLowerCase();
            filtered = filtered.filter(s =>
                s.name.toLowerCase().includes(query) ||
                s.description.toLowerCase().includes(query) ||
                s.creator.toLowerCase().includes(query) ||
                s.tags.some(tag => tag.toLowerCase().includes(query))
            );
        }

        // Category filter
        if (selectedCategory !== 'all') {
            filtered = filtered.filter(s => s.category === selectedCategory);
        }

        // Price filter
        if (priceFilter === 'free') {
            filtered = filtered.filter(s => s.price === 0);
        } else if (priceFilter === 'paid') {
            filtered = filtered.filter(s => s.price > 0);
        }

        // Verified filter
        if (showVerifiedOnly) {
            filtered = filtered.filter(s => s.is_verified);
        }

        // Sort
        filtered.sort((a, b) => {
            switch (sortBy) {
                case 'rating':
                    return b.rating - a.rating;
                case 'return':
                    return (b?.total_return || 0) - (a?.total_return || 0);
                case 'downloads':
                default:
                    return (b.total_downloads || 0) - (a.total_downloads || 0);
            }
        });

        return filtered;
    }, [strategies, searchQuery, selectedCategory, sortBy, priceFilter, showVerifiedOnly]);

    if (isLoading) {
        return (
            <div className="flex items-center justify-center h-screen">
                <div className="text-center">
                    <Loader2 className="animate-spin h-12 w-12 text-indigo-500 mx-auto mb-4"/>
                    <p className="text-slate-400">Loading marketplace...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
            {/* Header */}
            <div className="sticky top-0 z-40 bg-slate-900/80 backdrop-blur-xl border-b border-slate-700/50">
                <div className="max-w-7xl mx-auto px-6 py-6">
                    <div className="flex items-center justify-between mb-6">
                        <div>
                            <h1 className="text-4xl font-black text-slate-100 mb-2">Strategy Marketplace</h1>
                            <p className="text-slate-400">Discover and deploy proven trading strategies</p>
                        </div>
                        <p className="text-sm text-slate-500 mt-1">
                            Publish strategies from the Dashboard after backtesting
                        </p>
                    </div>

                    {/* Search & Filters */}
                    <div className="flex flex-col md:flex-row gap-4">
                        {/* Search */}
                        <div className="flex-1 relative">
                            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" size={20}/>
                            <input
                                type="text"
                                placeholder="Search strategies, creators, tags..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="w-full pl-12 pr-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-slate-200 placeholder-slate-500 focus:border-indigo-500 focus:outline-none"
                            />
                        </div>

                        {/* View Mode */}
                        <div className="flex gap-2 bg-slate-800 border border-slate-700 rounded-xl p-1">
                            <button
                                onClick={() => setViewMode('grid')}
                                className={`px-4 py-2 rounded-lg transition-all ${
                                    viewMode === 'grid'
                                        ? 'bg-indigo-600 text-white'
                                        : 'text-slate-400 hover:text-slate-200'
                                }`}
                            >
                                <LayoutGrid size={20}/>
                            </button>
                            <button
                                onClick={() => setViewMode('list')}
                                className={`px-4 py-2 rounded-lg transition-all ${
                                    viewMode === 'list'
                                        ? 'bg-indigo-600 text-white'
                                        : 'text-slate-400 hover:text-slate-200'
                                }`}
                            >
                                <List size={20}/>
                            </button>
                        </div>

                        {/* Filters Toggle */}
                        <button
                            onClick={() => setShowFilters(!showFilters)}
                            className="flex items-center gap-2 px-6 py-3 bg-slate-800 border border-slate-700 hover:bg-slate-700 rounded-xl text-slate-200 transition-all"
                        >
                            <Filter size={20}/>
                            Filters
                            <ChevronDown
                                size={16}
                                className={`transition-transform ${showFilters ? 'rotate-180' : ''}`}
                            />
                        </button>
                    </div>

                    {/* Filter Panel */}
                    {showFilters && (
                        <div className="mt-4 p-6 bg-slate-800/50 border border-slate-700 rounded-xl">
                            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                                {/* Category */}
                                <div>
                                    <label className="block text-sm font-bold text-slate-300 mb-2">Category</label>
                                    <select
                                        value={selectedCategory}
                                        onChange={(e) => setSelectedCategory(e.target.value)}
                                        className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-200 focus:border-indigo-500 outline-none"
                                    >
                                        {CATEGORIES.map(cat => (
                                            <option key={cat.value} value={cat.value}>{cat.label}</option>
                                        ))}
                                    </select>
                                </div>

                                {/* Sort */}
                                <div>
                                    <label className="block text-sm font-bold text-slate-300 mb-2">Sort By</label>
                                    <select
                                        value={sortBy}
                                        onChange={(e) => setSortBy(e.target.value)}
                                        className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-200 focus:border-indigo-500 outline-none"
                                    >
                                        {SORT_OPTIONS.map(opt => (
                                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                                        ))}
                                    </select>
                                </div>

                                {/* Price */}
                                <div>
                                    <label className="block text-sm font-bold text-slate-300 mb-2">Price</label>
                                    <select
                                        value={priceFilter}
                                        onChange={(e) => setPriceFilter(e.target.value as any)}
                                        className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-200 focus:border-indigo-500 outline-none"
                                    >
                                        <option value="all">All Prices</option>
                                        <option value="free">Free Only</option>
                                        <option value="paid">Paid Only</option>
                                    </select>
                                </div>

                                {/* Verified */}
                                <div>
                                    <label className="block text-sm font-bold text-slate-300 mb-2">Filter</label>
                                    <button
                                        onClick={() => setShowVerifiedOnly(!showVerifiedOnly)}
                                        className={`w-full px-4 py-2 rounded-lg transition-all flex items-center justify-center gap-2 ${
                                            showVerifiedOnly
                                                ? 'bg-emerald-600 text-white'
                                                : 'bg-slate-900 border border-slate-700 text-slate-400'
                                        }`}
                                    >
                                        <Shield size={16}/>
                                        Verified Only
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Main Content */}
            <div className="max-w-7xl mx-auto px-6 py-8">
                {/* Stats Bar */}
                <div className="mb-8 flex items-center justify-between">
                    <p className="text-slate-400">
                        Showing <span className="font-bold text-slate-200">{filteredStrategies.length}</span> strategies
                    </p>
                    <div className="flex items-center gap-4 text-sm text-slate-500">
                        <div className="flex items-center gap-2">
                            <Users size={16}/>
                            {strategies.reduce((sum, s) => sum + (s.total_downloads || 0), 0)} downloads
                        </div>
                        <div className="flex items-center gap-2">
                            <Shield size={16}/>
                            {strategies.filter(s => s.is_verified).length} verified
                        </div>
                    </div>
                </div>

                {/* Strategies Grid/List */}
                {filteredStrategies.length === 0 ? (
                    <div className="text-center py-20">
                        <Package className="mx-auto text-slate-700 mb-4" size={64}/>
                        <h3 className="text-xl font-bold text-slate-300 mb-2">No strategies found</h3>
                        <p className="text-slate-500">Try adjusting your filters or search query</p>
                    </div>
                ) : viewMode === 'grid' ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {filteredStrategies.map(strategy => (
                            <StrategyCard
                                key={strategy.id}
                                strategy={strategy}
                                onSelect={() => setSelectedStrategyId(strategy.id)}
                                onFavorite={(e: any) => toggleFavorite(e, strategy.id, strategy.is_favorite || false)}
                                onDownload={(e: any) => handleDownload(e, strategy.id)}
                            />
                        ))}
                    </div>
                ) : (
                    <div className="space-y-4">
                        {filteredStrategies.map(strategy => (
                            <StrategyListItem
                                key={strategy.id}
                                strategy={strategy}
                                onSelect={() => setSelectedStrategyId(strategy.id)}
                                onFavorite={(e: any) => toggleFavorite(e, strategy.id, strategy.is_favorite || false)}
                                onDownload={(e: any) => handleDownload(e, strategy.id)}
                            />
                        ))}
                    </div>
                )}
            </div>

            {/* Details Modal */}
            {selectedStrategyId && (
                <StrategyDetailsModal
                    strategyId={selectedStrategyId}
                    onClose={() => setSelectedStrategyId(null)}
                />
            )}

        </div>
    );
};

// Mock data for testing
const getMockStrategies = (): StrategyListing[] => [
    {
        id: 1,
        name: "Momentum Breakout Pro",
        description: "High-performance momentum strategy with proven results",
        creator: "AlgoTrader",
        category: "momentum",
        tags: ["momentum", "breakout", "stocks"],
        price: 0,
        rating: 4.8,
        reviews: 127,
        total_downloads: 1543,
        is_verified: true,
        total_return: 45.3,
        sharpe_ratio: 2.1,
        max_drawdown: -12.4,
        win_rate: 68,
        num_trades: 234,
        avg_win: 2.3,
        avg_loss: 1.1,
        profit_factor: 2.8,
        volatility: 18.5,
        sortino_ratio: 3.2,
        calmar_ratio: 3.6,
        var_95: -2.1,
        initial_capital: 10000,

        pros: ["High win rate", "Low drawdown"],
        cons: ["Requires active monitoring"]
    },
    // Add more mock strategies as needed
];

export default Marketplace;
