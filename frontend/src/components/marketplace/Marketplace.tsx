'use client'

import React, { useState, useMemo, useEffect } from 'react';
import {
    Package, Search, Filter, Star, Download,
    TrendingUp, Shield, Clock, Users, ExternalLink,
    Heart, Upload, LayoutGrid, List, Sparkles,
    CheckCircle2, ArrowUpRight, TrendingDown, Activity,
    Zap, Target, DollarSign, ChevronDown, X, Info,
    Loader2
} from "lucide-react";
import { marketplace } from '@/utils/api';
import { StrategyListing } from '@/types/api.types';
import StrategyDetailsModal from './StrategyDetailsModal';

const CATEGORIES = [
    "All",
    "Trend Following",
    "Momentum",
    "Mean Reversion",
    "Volatility",
    "Machine Learning",
    "Options",
    "Pairs Trading",
    "Statistical Arbitrage"
];

const COMPLEXITY_LEVELS = ["All", "Beginner", "Intermediate", "Advanced", "Expert"];

const Marketplace = () => {
    const [strategies, setStrategies] = useState<StrategyListing[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const [activeTab, setActiveTab] = useState('browse');
    const [selectedCategory, setSelectedCategory] = useState('All');
    const [selectedComplexity, setSelectedComplexity] = useState('All');
    const [searchQuery, setSearchQuery] = useState('');
    const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
    const [sortBy, setSortBy] = useState<'popular' | 'rating' | 'price' | 'newest'>('popular');
    const [showFilters, setShowFilters] = useState(false);
    const [selectedStrategy, setSelectedStrategy] = useState<StrategyListing | null>(null);

    useEffect(() => {
        fetchStrategies();
    }, []);

    const fetchStrategies = async (params: any = {}) => {
        setIsLoading(true);
        try {
            const response = await marketplace.getAll(params);
            if (response.data) {
                setStrategies(response.data);
            }
        } catch (err) {
            console.error("Failed to fetch strategies:", err);
            setError("Failed to load marketplace content");
        } finally {
            setIsLoading(false);
        }
    };

    const toggleFavorite = async (e: React.MouseEvent, strategyId: string | number, isCurrentlyFavorite: boolean) => {
        e.stopPropagation();
        try {
            if (isCurrentlyFavorite) {
                await marketplace.unfavorite(Number(strategyId));
            } else {
                await marketplace.favorite(Number(strategyId));
            }

            // Update local state
            setStrategies(prev => prev.map(s =>
                s.id === strategyId ? { ...s, is_favorite: !isCurrentlyFavorite } : s
            ));
        } catch (err) {
            console.error("Failed to toggle favorite:", err);
        }
    };

    const handleDownload = async (e: React.MouseEvent, strategyId: string | number) => {
        e.stopPropagation();
        try {
            await marketplace.recordDownload(Number(strategyId));

            // Update local state increment
            setStrategies(prev => prev.map(s =>
                s.id === strategyId ? { ...s, total_downloads: (s.total_downloads || 0) + 1 } : s
            ));

            // Optional: Trigger actual file download or show confirmation
            alert("Strategy added to your downloads!");
        } catch (err) {
            console.error("Failed to record download:", err);
        }
    };

    // Filter and sort strategies
    const filteredStrategies = useMemo(() => {
        let filtered = strategies;

        // Filter by tab
        if (activeTab === 'favorites') {
            filtered = filtered.filter(s => s.is_favorite);
        }

        // Filter by category
        if (selectedCategory !== 'All') {
            filtered = filtered.filter(s => s.category === selectedCategory);
        }

        // Filter by complexity
        if (selectedComplexity !== 'All') {
            filtered = filtered.filter(s => s.complexity === selectedComplexity);
        }

        // Filter by search
        if (searchQuery) {
            const query = searchQuery.toLowerCase();
            filtered = filtered.filter(s =>
                s.name.toLowerCase().includes(query) ||
                s.description.toLowerCase().includes(query) ||
                s.creator.toLowerCase().includes(query) ||
                s.tags.some(tag => tag.toLowerCase().includes(query))
            );
        }

        // Sort
        switch (sortBy) {
            case 'popular':
                filtered.sort((a, b) => b.total_downloads - a.total_downloads);
                break;
            case 'rating':
                filtered.sort((a, b) => b.rating - a.rating);
                break;
            case 'price':
                filtered.sort((a, b) => a.price - b.price);
                break;
            case 'newest':
                filtered.sort((a, b) => new Date(b.publish_date).getTime() - new Date(a.publish_date).getTime());
                break;
        }

        return filtered;
    }, [strategies, activeTab, selectedCategory, selectedComplexity, searchQuery, sortBy]);

    const getCategoryIcon = (category: string) => {
        const icons: any = {
            'Trend Following': TrendingUp,
            'Momentum': Zap,
            'Mean Reversion': Activity,
            'Volatility': TrendingDown,
            'Machine Learning': Sparkles,
            'Options': DollarSign,
            'Pairs Trading': Target,
            'Statistical Arbitrage': Shield
        };
        return icons[category] || TrendingUp;
    };

    if (isLoading) {
        return (
            <div className="flex items-center justify-center min-h-[400px]">
                <div className="text-center space-y-4">
                    <Loader2 size={32} className="animate-spin text-indigo-500 mx-auto" />
                    <p className="text-sm text-slate-400">Loading marketplace...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex items-center justify-center min-h-[400px]">
                <div className="text-center space-y-4">
                    <Shield size={32} className="text-red-500 mx-auto" />
                    <p className="text-sm text-red-400">{error}</p>
                    <button
                        onClick={fetchStrategies}
                        className="text-xs text-indigo-400 hover:text-indigo-300 underline"
                    >
                        Try Again
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-6 animate-in fade-in duration-700">
            {/* Header */}
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
                <div className="flex items-center gap-4">
                    <div className="p-3 bg-gradient-to-br from-indigo-500/20 to-purple-500/20 rounded-2xl border border-indigo-500/30 shadow-xl shadow-indigo-500/10">
                        <Package className="text-indigo-400" size={32} strokeWidth={2} />
                    </div>
                    <div>
                        <h2 className="text-3xl font-bold text-slate-100 tracking-tight">
                            Strategy <span className="text-slate-400 font-normal">Marketplace</span>
                        </h2>
                        <p className="text-sm text-slate-500 font-medium mt-1">
                            {filteredStrategies.length} battle-tested strategies from quant researchers
                        </p>
                    </div>
                </div>

                {/* Tab Navigation */}
                <div className="flex bg-slate-900/60 p-1 rounded-xl border border-slate-700/50 backdrop-blur-sm">
                    {['browse', 'favorites', 'downloads'].map((tab) => (
                        <button
                            key={tab}
                            onClick={() => setActiveTab(tab)}
                            className={`px-5 py-2.5 rounded-lg text-xs font-bold uppercase tracking-wider transition-all ${activeTab === tab
                                ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-lg'
                                : 'text-slate-500 hover:text-slate-300'
                                }`}
                        >
                            {tab}
                        </button>
                    ))}
                </div>
            </div>

            {/* Search & Filters */}
            <div className="flex flex-col gap-4">
                <div className="flex flex-col md:flex-row gap-4">
                    {/* Search Bar */}
                    <div className="relative flex-1">
                        <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" size={20} />
                        <input
                            type="text"
                            placeholder="Search strategies, creators, or keywords..."
                            className="w-full bg-slate-900/60 border border-slate-700/50 rounded-xl py-4 pl-12 pr-4 text-sm text-slate-200 placeholder:text-slate-600 focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none transition-all"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                        />
                        {searchQuery && (
                            <button
                                onClick={() => setSearchQuery('')}
                                className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                            >
                                <X size={18} />
                            </button>
                        )}
                    </div>

                    {/* View Toggle & Filters */}
                    <div className="flex gap-2">
                        <button
                            onClick={() => setShowFilters(!showFilters)}
                            className={`px-4 py-3 rounded-xl border font-semibold text-sm transition-all flex items-center space-x-2 ${showFilters
                                ? 'bg-violet-500/20 border-violet-500/50 text-violet-300'
                                : 'bg-slate-900/60 border-slate-700/50 text-slate-400 hover:border-slate-600'
                                }`}
                        >
                            <Filter size={18} strokeWidth={2} />
                            <span>Filters</span>
                        </button>

                        <div className="flex bg-slate-900/60 border border-slate-700/50 rounded-xl overflow-hidden">
                            <button
                                onClick={() => setViewMode('grid')}
                                className={`px-4 py-3 transition-all ${viewMode === 'grid' ? 'bg-slate-800 text-slate-200' : 'text-slate-500 hover:text-slate-300'
                                    }`}
                            >
                                <LayoutGrid size={18} strokeWidth={2} />
                            </button>
                            <button
                                onClick={() => setViewMode('list')}
                                className={`px-4 py-3 transition-all ${viewMode === 'list' ? 'bg-slate-800 text-slate-200' : 'text-slate-500 hover:text-slate-300'
                                    }`}
                            >
                                <List size={18} strokeWidth={2} />
                            </button>
                        </div>
                    </div>
                </div>

                {/* Filter Panel */}
                {showFilters && (
                    <div className="bg-slate-900/60 border border-slate-700/50 rounded-xl p-6 space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            {/* Category Filter */}
                            <div className="space-y-3">
                                <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Category</label>
                                <select
                                    value={selectedCategory}
                                    onChange={(e) => setSelectedCategory(e.target.value)}
                                    className="w-full bg-slate-800/60 border border-slate-700/50 rounded-xl px-4 py-3 text-sm text-slate-200 focus:border-violet-500 outline-none"
                                >
                                    {CATEGORIES.map(cat => (
                                        <option key={cat} value={cat}>{cat}</option>
                                    ))}
                                </select>
                            </div>

                            {/* Complexity Filter */}
                            <div className="space-y-3">
                                <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Complexity</label>
                                <select
                                    value={selectedComplexity}
                                    onChange={(e) => setSelectedComplexity(e.target.value)}
                                    className="w-full bg-slate-800/60 border border-slate-700/50 rounded-xl px-4 py-3 text-sm text-slate-200 focus:border-violet-500 outline-none"
                                >
                                    {COMPLEXITY_LEVELS.map(level => (
                                        <option key={level} value={level}>{level}</option>
                                    ))}
                                </select>
                            </div>

                            {/* Sort By */}
                            <div className="space-y-3">
                                <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Sort By</label>
                                <select
                                    value={sortBy}
                                    onChange={(e) => setSortBy(e.target.value as any)}
                                    className="w-full bg-slate-800/60 border border-slate-700/50 rounded-xl px-4 py-3 text-sm text-slate-200 focus:border-violet-500 outline-none"
                                >
                                    <option value="popular">Most Popular</option>
                                    <option value="rating">Highest Rated</option>
                                    <option value="price">Price: Low to High</option>
                                    <option value="newest">Newest First</option>
                                </select>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Strategy Grid/List */}
            <div className={viewMode === 'grid' ? 'grid grid-cols-1 xl:grid-cols-2 gap-6' : 'space-y-4'}>
                {filteredStrategies.map((strategy) => {
                    const CategoryIcon = getCategoryIcon(strategy.category);

                    return (
                        <div
                            key={strategy.id}
                            className="group bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 hover:border-violet-500/50 transition-all duration-300 relative overflow-hidden shadow-xl cursor-pointer"
                            onClick={() => setSelectedStrategy(strategy)}
                        >
                            {/* Verified Badge */}
                            {strategy.is_verified && (
                                <div className="absolute top-4 right-4">
                                    <span className="flex items-center gap-1.5 px-3 py-1 bg-emerald-500/20 text-emerald-400 rounded-full text-[10px] font-black uppercase tracking-wider border border-emerald-500/30">
                                        <Sparkles size={10} /> Verified
                                    </span>
                                </div>
                            )}

                            <div className="flex gap-6">
                                {/* Strategy Icon */}
                                <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700/50 flex items-center justify-center shrink-0 shadow-inner group-hover:scale-105 transition-transform">
                                    <CategoryIcon className="text-indigo-400" size={32} strokeWidth={2} />
                                </div>

                                <div className="flex-1 space-y-3">
                                    <div className="flex justify-between items-start">
                                        <div className="flex-1 pr-12">
                                            <h3 className="text-lg font-bold text-slate-100 group-hover:text-violet-300 transition-colors">
                                                {strategy.name}
                                            </h3>
                                            <p className="text-xs text-slate-500 font-medium tracking-tight mt-0.5">
                                                by {strategy.creator}
                                            </p>
                                        </div>
                                        <div className="text-right">
                                            <p className="text-2xl font-black text-slate-100">${strategy.price}</p>
                                            <p className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">One-time</p>
                                        </div>
                                    </div>

                                    <p className="text-sm text-slate-400 leading-relaxed line-clamp-2">
                                        {strategy.description}
                                    </p>

                                    <div className="flex items-center gap-4 pt-2">
                                        <div className="flex items-center gap-1.5 text-amber-400">
                                            <Star size={14} fill="currentColor" strokeWidth={2} />
                                            <span className="text-sm font-bold">{strategy.rating}</span>
                                            <span className="text-xs text-slate-600">({strategy.reviews})</span>
                                        </div>
                                        <span className="text-xs text-slate-700">•</span>
                                        <span className="px-2.5 py-1 bg-slate-800/60 border border-slate-700/50 text-slate-400 rounded-lg text-[10px] font-bold uppercase tracking-wider">
                                            {strategy.category}
                                        </span>
                                        <span className="text-xs text-slate-700">•</span>
                                        <span className="text-xs text-slate-500 font-medium">
                                            {strategy.total_downloads.toLocaleString()} downloads
                                        </span>
                                    </div>
                                </div>
                            </div>

                            {/* Performance Metrics */}
                            <div className="grid grid-cols-4 gap-4 mt-6 pt-6 border-t border-slate-700/50">
                                <div className="space-y-1">
                                    <p className="text-[9px] font-black text-slate-600 uppercase tracking-widest">Monthly Ret</p>
                                    <p className="text-sm font-bold text-emerald-400">+{strategy.monthly_return}%</p>
                                </div>
                                <div className="space-y-1">
                                    <p className="text-[9px] font-black text-slate-600 uppercase tracking-widest">Max DD</p>
                                    <p className="text-sm font-bold text-red-400">{strategy.drawdown}%</p>
                                </div>
                                <div className="space-y-1">
                                    <p className="text-[9px] font-black text-slate-600 uppercase tracking-widest">Sharpe</p>
                                    <p className="text-sm font-bold text-blue-400">{strategy.sharpe_ratio.toFixed(2)}</p>
                                </div>
                                <div className="space-y-1">
                                    <p className="text-[9px] font-black text-slate-600 uppercase tracking-widest">Level</p>
                                    <p className="text-sm font-bold text-violet-400">{strategy.complexity}</p>
                                </div>
                            </div>

                            {/* Action Bar */}
                            <div className="flex gap-2 mt-6">
                                <button
                                    onClick={(e) => handleDownload(e, strategy.id)}
                                    className="flex-1 py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-black uppercase tracking-widest transition-all flex items-center justify-center gap-2 shadow-lg shadow-indigo-600/20"
                                >
                                    <Download size={14} /> Get Strategy
                                </button>
                                <button
                                    onClick={(e) => toggleFavorite(e, strategy.id, !!strategy.is_favorite)}
                                    className="px-4 py-3 bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-red-400 rounded-xl transition-all border border-slate-700/50"
                                >
                                    <Heart size={16} fill={strategy.is_favorite ? "currentColor" : "none"} className={strategy.is_favorite ? "text-red-400" : ""} />
                                </button>
                                <button className="px-4 py-3 bg-slate-800 hover:bg-slate-700 text-slate-400 rounded-xl transition-all border border-slate-700/50">
                                    <ExternalLink size={16} />
                                </button>
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Creator Stats Bar (Mocked for now as backend support not requested yet) */}
            <div className="bg-gradient-to-r from-indigo-900/20 to-purple-900/20 border border-indigo-500/20 rounded-2xl p-6 flex flex-col md:flex-row items-center justify-between gap-6">
                <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-full bg-indigo-500/20 flex items-center justify-center">
                        <Upload className="text-indigo-400" size={24} />
                    </div>
                    <div>
                        <h4 className="text-sm font-bold text-slate-100">Ready to publish your alpha?</h4>
                        <p className="text-xs text-slate-400">Join 500+ quantitative researchers earning from their strategies.</p>
                    </div>
                </div>
                <div className="flex gap-8">
                    <div className="text-center">
                        <p className="text-lg font-black text-indigo-400">1.2k</p>
                        <p className="text-[9px] font-bold text-slate-500 uppercase tracking-tighter">Daily Users</p>
                    </div>
                    <div className="text-center">
                        <p className="text-lg font-black text-purple-400">$2.4M</p>
                        <p className="text-[9px] font-bold text-slate-500 uppercase tracking-tighter">Total Payouts</p>
                    </div>
                    <button className="px-6 py-3 bg-white text-indigo-900 rounded-xl text-xs font-black uppercase tracking-widest hover:bg-indigo-50 transition-all">
                        Apply to Publish
                    </button>
                </div>
            </div>

            {/* Detailed View Modal */}
            {selectedStrategy && (
                <StrategyDetailsModal
                    strategyId={Number(selectedStrategy.id)}
                    onClose={() => setSelectedStrategy(null)}
                />
            )}
        </div>
    );
};

export default Marketplace