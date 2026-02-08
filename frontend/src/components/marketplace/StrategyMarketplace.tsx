/**
 * Strategy Marketplace
 * Browse, search, and deploy verified strategies
 */

'use client'

import React, { useEffect, useState } from 'react';
import { BarChart3, Search, Shield, Star, Users, Award, Trophy, Zap, AlertTriangle } from 'lucide-react';

interface MarketplaceStrategy {
    id: number;
    name: string;
    description: string;
    strategy_key: string;
    category: string;
    creator_name: string;
    is_verified: boolean;
    verification_badge?: string;

    // Performance
    avg_return_pct: number;
    avg_sharpe: number;
    avg_win_rate: number;
    max_drawdown: number;
    total_deployments: number;

    // Pricing
    price_monthly: number;
    has_free_trial: boolean;
    trial_days: number;

    // Rating
    rating: number;
    num_ratings: number;
}

interface StrategyDetailsModal {
    strategy: MarketplaceStrategy | null;
    backtestResults: any[];
    reviews: any[];
}

export default function StrategyMarketplace() {
    const [strategies, setStrategies] = useState<MarketplaceStrategy[]>([]);
    const [filteredStrategies, setFilteredStrategies] = useState<MarketplaceStrategy[]>([]);
    const [selectedStrategy, setSelectedStrategy] = useState<MarketplaceStrategy | null>(null);
    const [loading, setLoading] = useState(true);

    // Filters
    const [searchQuery, setSearchQuery] = useState('');
    const [categoryFilter, setCategoryFilter] = useState('all');
    const [sortBy, setSortBy] = useState<'rating' | 'return' | 'sharpe' | 'deployments'>('rating');
    const [showVerifiedOnly, setShowVerifiedOnly] = useState(false);

    useEffect(() => {
        loadStrategies();
    }, []);

    useEffect(() => {
        applyFilters();
    }, [strategies, searchQuery, categoryFilter, sortBy, showVerifiedOnly]);

    const loadStrategies = async () => {
        try {
            const response = await fetch('/api/marketplace/strategies');
            if (response.ok) {
                const data = await response.json();
                setStrategies(data);
            }
        } catch (error) {
            console.error('Error loading strategies:', error);
        } finally {
            setLoading(false);
        }
    };

    const applyFilters = () => {
        let filtered = [...strategies];

        // Search
        if (searchQuery) {
            filtered = filtered.filter(s =>
                s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                s.description.toLowerCase().includes(searchQuery.toLowerCase())
            );
        }

        // Category
        if (categoryFilter !== 'all') {
            filtered = filtered.filter(s => s.category === categoryFilter);
        }

        // Verified only
        if (showVerifiedOnly) {
            filtered = filtered.filter(s => s.is_verified);
        }

        // Sort
        filtered.sort((a, b) => {
            switch (sortBy) {
                case 'rating':
                    return b.rating - a.rating;
                case 'return':
                    return b.avg_return_pct - a.avg_return_pct;
                case 'sharpe':
                    return b.avg_sharpe - a.avg_sharpe;
                case 'deployments':
                    return b.total_deployments - a.total_deployments;
                default:
                    return 0;
            }
        });

        setFilteredStrategies(filtered);
    };

    const categories = [
        'all',
        'trend_following',
        'mean_reversion',
        'momentum',
        'statistical_arbitrage',
        'pairs_trading',
        'options',
        'multi_asset'
    ];

    if (loading) {
        return (
            <div className="flex items-center justify-center h-screen">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-violet-500"></div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-slate-950 p-6">
            <div className="max-w-7xl mx-auto space-y-6">

                {/* Header */}
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-3xl font-bold text-slate-100 flex items-center gap-2">
                            🏪 Strategy Marketplace
                        </h1>
                        <p className="text-slate-400 mt-1">
                            Browse and deploy verified trading strategies
                        </p>
                    </div>

                    <div className="text-right">
                        <div className="text-2xl font-bold text-violet-400">{filteredStrategies.length}</div>
                        <div className="text-sm text-slate-500">Strategies Available</div>
                    </div>
                </div>

                {/* Search & Filters */}
                <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-800 space-y-4">

                    {/* Search Bar */}
                    <div className="relative">
                        <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" size={20} />
                        <input
                            type="text"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            placeholder="Search strategies..."
                            className="w-full pl-12 pr-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:border-violet-500 outline-none"
                        />
                    </div>

                    {/* Filters Row */}
                    <div className="flex flex-wrap gap-4">
                        {/* Category */}
                        <select
                            value={categoryFilter}
                            onChange={(e) => setCategoryFilter(e.target.value)}
                            className="px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:border-violet-500 outline-none"
                        >
                            {categories.map(cat => (
                                <option key={cat} value={cat}>
                                    {cat.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                </option>
                            ))}
                        </select>

                        {/* Sort */}
                        <select
                            value={sortBy}
                            onChange={(e) => setSortBy(e.target.value as any)}
                            className="px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:border-violet-500 outline-none"
                        >
                            <option value="rating">Sort by Rating</option>
                            <option value="return">Sort by Return</option>
                            <option value="sharpe">Sort by Sharpe</option>
                            <option value="deployments">Sort by Popularity</option>
                        </select>

                        {/* Verified Only */}
                        <label className="flex items-center gap-2 px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg cursor-pointer hover:bg-slate-750">
                            <input
                                type="checkbox"
                                checked={showVerifiedOnly}
                                onChange={(e) => setShowVerifiedOnly(e.target.checked)}
                                className="w-4 h-4"
                            />
                            <Shield size={16} className="text-emerald-500" />
                            <span className="text-sm text-slate-300">Verified Only</span>
                        </label>
                    </div>
                </div>

                {/* Strategy Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {filteredStrategies.map(strategy => (
                        <div
                            key={strategy.id}
                            onClick={() => setSelectedStrategy(strategy)}
                            className="group bg-gradient-to-br from-slate-900 to-slate-950 rounded-xl p-6 border border-slate-800 hover:border-violet-500/50 transition-all cursor-pointer"
                        >
                            {/* Header */}
                            <div className="flex items-start justify-between mb-4">
                                <div className="flex-1">
                                    <h3 className="text-lg font-bold text-slate-100 group-hover:text-violet-400 transition-colors">
                                        {strategy.name}
                                    </h3>
                                    <div className="flex items-center gap-2 mt-1 flex-wrap">
                                        <span className="text-xs text-slate-500">by {strategy.creator_name}</span>
                                        {strategy.is_verified && (
                                            <div className="flex items-center gap-1 px-2 py-0.5 bg-emerald-500/20 rounded text-emerald-400 text-[10px] font-black uppercase tracking-tight">
                                                <Shield size={10} />
                                                Verified
                                            </div>
                                        )}
                                        {strategy.verification_badge === 'INSTITUTIONAL' && (
                                            <div className="flex items-center gap-1 px-2 py-0.5 bg-violet-500/20 rounded text-violet-400 text-[10px] font-black uppercase tracking-tight border border-violet-500/30 shadow-[0_0_10px_rgba(139,92,246,0.3)]">
                                                <Trophy size={10} />
                                                Institutional
                                            </div>
                                        )}
                                        {strategy.verification_badge === 'CONSISTENT' && (
                                            <div className="flex items-center gap-1 px-2 py-0.5 bg-blue-500/20 rounded text-blue-400 text-[10px] font-black uppercase tracking-tight">
                                                <Award size={10} />
                                                Consistent
                                            </div>
                                        )}
                                        {strategy.verification_badge === 'DRIFTING' && (
                                            <div className="flex items-center gap-1 px-2 py-0.5 bg-amber-500/20 rounded text-amber-400 text-[10px] font-black uppercase tracking-tight">
                                                <AlertTriangle size={10} />
                                                Drifting
                                            </div>
                                        )}
                                    </div>
                                </div>

                                {/* Rating */}
                                <div className="flex items-center gap-1 px-2 py-1 bg-amber-500/20 rounded-lg">
                                    <Star size={14} className="fill-amber-400 text-amber-400" />
                                    <span className="text-sm font-bold text-amber-400">{strategy.rating.toFixed(1)}</span>
                                </div>
                            </div>

                            {/* Description */}
                            <p className="text-sm text-slate-400 mb-4 line-clamp-2">
                                {strategy.description}
                            </p>

                            {/* Stats Grid */}
                            <div className="grid grid-cols-2 gap-3 mb-4">
                                <div className="p-3 bg-slate-800/50 rounded-lg">
                                    <div className="text-xs text-slate-500 mb-1">Avg Return</div>
                                    <div className="text-sm font-bold text-emerald-400">
                                        +{strategy.avg_return_pct.toFixed(1)}%
                                    </div>
                                </div>

                                <div className="p-3 bg-slate-800/50 rounded-lg">
                                    <div className="text-xs text-slate-500 mb-1">Sharpe Ratio</div>
                                    <div className="text-sm font-bold text-slate-200">
                                        {strategy.avg_sharpe.toFixed(2)}
                                    </div>
                                </div>

                                <div className="p-3 bg-slate-800/50 rounded-lg">
                                    <div className="text-xs text-slate-500 mb-1">Win Rate</div>
                                    <div className="text-sm font-bold text-blue-400">
                                        {(strategy.avg_win_rate * 100).toFixed(1)}%
                                    </div>
                                </div>

                                <div className="p-3 bg-slate-800/50 rounded-lg">
                                    <div className="text-xs text-slate-500 mb-1">Max DD</div>
                                    <div className="text-sm font-bold text-red-400">
                                        {strategy.max_drawdown.toFixed(1)}%
                                    </div>
                                </div>
                            </div>

                            {/* Footer */}
                            <div className="flex items-center justify-between pt-4 border-t border-slate-800">
                                <div className="flex items-center gap-1 text-xs text-slate-500">
                                    <Users size={14} />
                                    {strategy.total_deployments} deployments
                                </div>

                                <div className="text-right">
                                    {strategy.price_monthly > 0 ? (
                                        <div>
                                            <div className="text-lg font-bold text-violet-400">
                                                ${strategy.price_monthly}/mo
                                            </div>
                                            {strategy.has_free_trial && (
                                                <div className="text-xs text-emerald-400">
                                                    {strategy.trial_days} day trial
                                                </div>
                                            )}
                                        </div>
                                    ) : (
                                        <div className="text-lg font-bold text-emerald-400">
                                            FREE
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Empty State */}
                {filteredStrategies.length === 0 && (
                    <div className="text-center py-12">
                        <div className="text-slate-500 mb-2">No strategies found</div>
                        <button
                            onClick={() => {
                                setSearchQuery('');
                                setCategoryFilter('all');
                                setShowVerifiedOnly(false);
                            }}
                            className="text-violet-400 hover:text-violet-300"
                        >
                            Clear filters
                        </button>
                    </div>
                )}

                {/* Strategy Details Modal */}
                {selectedStrategy && (
                    <StrategyDetailsModal
                        strategy={selectedStrategy}
                        onClose={() => setSelectedStrategy(null)}
                        onDeploy={(strategy) => {
                            // Open deployment modal
                            console.log('Deploy strategy:', strategy);
                        }}
                    />
                )}

            </div>
        </div>
    );
}

function StrategyDetailsModal({
    strategy,
    onClose,
    onDeploy
}: {
    strategy: MarketplaceStrategy;
    onClose: () => void;
    onDeploy: (strategy: MarketplaceStrategy) => void;
}) {
    const [activeTab, setActiveTab] = useState<'overview' | 'performance' | 'reviews'>('overview');

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
            <div className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl max-w-4xl w-full mx-4 max-h-[90vh] overflow-hidden flex flex-col">

                {/* Header */}
                <div className="p-6 border-b border-slate-700">
                    <div className="flex items-start justify-between">
                        <div className="flex-1">
                            <div className="flex items-center gap-3 mb-2">
                                <h2 className="text-2xl font-bold text-slate-100">{strategy.name}</h2>
                                {strategy.is_verified && (
                                    <div className="flex items-center gap-1 px-3 py-1 bg-emerald-500/20 rounded-lg text-emerald-400 text-sm font-semibold">
                                        <Shield size={16} />
                                        Verified
                                    </div>
                                )}
                                {strategy.verification_badge === 'INSTITUTIONAL' && (
                                    <div className="flex items-center gap-1 px-3 py-1 bg-violet-500/20 border border-violet-500/30 rounded-lg text-violet-400 text-sm font-bold shadow-[0_0_15px_rgba(139,92,246,0.2)]">
                                        <Trophy size={16} />
                                        Institutional Grade
                                    </div>
                                )}
                            </div>
                            <div className="flex items-center gap-4 text-sm text-slate-400">
                                <span>by {strategy.creator_name}</span>
                                <span>•</span>
                                <div className="flex items-center gap-1">
                                    <Star size={14} className="fill-amber-400 text-amber-400" />
                                    <span className="font-semibold text-amber-400">{strategy.rating.toFixed(1)}</span>
                                    <span>({strategy.num_ratings} reviews)</span>
                                </div>
                                <span>•</span>
                                <div className="flex items-center gap-1">
                                    <Users size={14} />
                                    {strategy.total_deployments} deployments
                                </div>
                            </div>
                        </div>

                        <button
                            onClick={onClose}
                            className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
                        >
                            <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M6 6l8 8M14 6l-8 8" />
                            </svg>
                        </button>
                    </div>
                </div>

                {/* Tabs */}
                <div className="flex border-b border-slate-800 px-6">
                    {[
                        { id: 'overview', label: 'Overview' },
                        { id: 'performance', label: 'Performance' },
                        { id: 'reviews', label: 'Reviews' }
                    ].map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id as any)}
                            className={`px-6 py-3 font-semibold text-sm border-b-2 transition-colors ${activeTab === tab.id
                                    ? 'border-violet-500 text-violet-400'
                                    : 'border-transparent text-slate-400 hover:text-slate-300'
                                }`}
                        >
                            {tab.label}
                        </button>
                    ))}
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-6">
                    {activeTab === 'overview' && (
                        <div className="space-y-6">
                            <div>
                                <h3 className="text-lg font-bold text-slate-100 mb-2">Description</h3>
                                <p className="text-slate-300">{strategy.description}</p>
                            </div>

                            <div>
                                <h3 className="text-lg font-bold text-slate-100 mb-4">Key Metrics</h3>
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="p-4 bg-slate-800/50 rounded-xl">
                                        <div className="text-sm text-slate-500 mb-1">Average Return</div>
                                        <div className="text-2xl font-bold text-emerald-400">
                                            +{strategy.avg_return_pct.toFixed(1)}%
                                        </div>
                                    </div>

                                    <div className="p-4 bg-slate-800/50 rounded-xl">
                                        <div className="text-sm text-slate-500 mb-1">Sharpe Ratio</div>
                                        <div className="text-2xl font-bold text-slate-200">
                                            {strategy.avg_sharpe.toFixed(2)}
                                        </div>
                                    </div>

                                    <div className="p-4 bg-slate-800/50 rounded-xl">
                                        <div className="text-sm text-slate-500 mb-1">Win Rate</div>
                                        <div className="text-2xl font-bold text-blue-400">
                                            {(strategy.avg_win_rate * 100).toFixed(1)}%
                                        </div>
                                    </div>

                                    <div className="p-4 bg-slate-800/50 rounded-xl">
                                        <div className="text-sm text-slate-500 mb-1">Max Drawdown</div>
                                        <div className="text-2xl font-bold text-red-400">
                                            {strategy.max_drawdown.toFixed(1)}%
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div>
                                <h3 className="text-lg font-bold text-slate-100 mb-2">Category</h3>
                                <div className="inline-block px-4 py-2 bg-violet-500/20 rounded-lg text-violet-400 font-semibold">
                                    {strategy.category.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                </div>
                            </div>
                        </div>
                    )}

                    {activeTab === 'performance' && (
                        <div className="space-y-6">
                            <div className="text-center py-8 text-slate-500">
                                <BarChart3 size={48} className="mx-auto mb-3 opacity-50" />
                                <p>Performance charts and backtest results would be displayed here</p>
                            </div>
                        </div>
                    )}

                    {activeTab === 'reviews' && (
                        <div className="space-y-4">
                            <div className="text-center py-8 text-slate-500">
                                <Star size={48} className="mx-auto mb-3 opacity-50" />
                                <p>User reviews and ratings would be displayed here</p>
                            </div>
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="p-6 border-t border-slate-700 flex items-center justify-between">
                    <div>
                        {strategy.price_monthly > 0 ? (
                            <div>
                                <div className="text-2xl font-bold text-violet-400">
                                    ${strategy.price_monthly}/month
                                </div>
                                {strategy.has_free_trial && (
                                    <div className="text-sm text-emerald-400">
                                        {strategy.trial_days} day free trial available
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className="text-2xl font-bold text-emerald-400">
                                FREE
                            </div>
                        )}
                    </div>

                    <div className="flex gap-3">
                        <button
                            onClick={onClose}
                            className="px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg text-slate-200 font-semibold"
                        >
                            Cancel
                        </button>

                        <button
                            onClick={() => onDeploy(strategy)}
                            className="px-6 py-3 bg-violet-600 hover:bg-violet-500 rounded-lg text-white font-semibold flex items-center gap-2"
                        >
                            🚀 Deploy Strategy
                        </button>
                    </div>
                </div>

            </div>
        </div>
    );
}
