'use client'

import React, { useEffect, useState } from 'react';
import {
    Activity, ArrowUpRight, CheckCircle2, Clock, Download,
    Rocket, Shield, Star, Target, X, Zap, Loader2,
    TrendingUp, AlertTriangle, Send, MessageSquare
} from 'lucide-react';
import { live, marketplace } from '@/utils/api';

import { DeploymentConfig } from '@/types/all_types';
import DeploymentModal from "@/components/strategies/DeploymentModel";

interface StrategyDetailsModalProps {
    strategyId: number;
    onClose: () => void;
}

const StrategyDetailsModal = ({ strategyId, onClose }: StrategyDetailsModalProps) => {
    const [strategy, setStrategy] = useState<any>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [showDeployModal, setShowDeployModal] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Review state
    const [reviewRating, setReviewRating] = useState(5);
    const [reviewText, setReviewText] = useState('');
    const [isSubmittingReview, setIsSubmittingReview] = useState(false);

    const fetchDetails = async () => {
        try {
            const response = await marketplace.getDetails(strategyId);
            setStrategy(response);
        } catch (err: any) {
            console.error("Failed to fetch strategy details:", err);
            setError(err?.message || "Failed to load strategy");
            setStrategy(getMockStrategy(strategyId));
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchDetails();
    }, [strategyId]);

    const handleDeploy = async (config: DeploymentConfig) => {
        try {
            await live.deploy({
                ...config,
                source: 'marketplace',
                marketplace_id: strategyId
            });

            alert('Strategy deployed successfully! Check Live Execution page.');
            setShowDeployModal(false);
            onClose();
        } catch (err: any) {
            console.error('Failed to deploy strategy:', err);
            alert(err?.message || 'Failed to deploy strategy');
        }
    };

    const handleSubmitReview = async () => {
        if (!reviewText.trim()) return;

        setIsSubmittingReview(true);
        try {
            await marketplace.createReview(strategyId, {
                rating: reviewRating,
                review_text: reviewText.trim(),
            });
            setReviewText('');
            setReviewRating(5);
            // Refresh strategy details to show new review
            const response = await marketplace.getDetails(strategyId);
            setStrategy(response);
        } catch (err: any) {
            console.error('Failed to submit review:', err);
            alert(err?.message || 'Failed to submit review');
        } finally {
            setIsSubmittingReview(false);
        }
    };

    // Convert marketplace strategy to backtest-like format for DeploymentModal
    const backtestData = strategy ? {
        id: String(strategy.id),
        strategy: strategy.name,
        symbols: strategy.backtest_results?.symbols || strategy.symbols || ['SPY'],
        total_return_pct: strategy.backtest_results?.total_return || 0,
        win_rate: strategy.backtest_results?.win_rate || 0,
        sharpe_ratio: strategy.backtest_results?.sharpe_ratio || 0,
        max_drawdown: strategy.backtest_results?.max_drawdown || 0,
        total_trades: strategy.backtest_results?.num_trades || 0,
        initial_capital: strategy.backtest_results?.initial_capital || 10000,
        parameters: strategy.parameters || {}
    } : null;

    if (isLoading) {
        return (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm">
                <div className="text-center">
                    <Loader2 className="animate-spin h-12 w-12 text-indigo-500 mx-auto mb-4" />
                    <p className="text-slate-400">Loading strategy details...</p>
                </div>
            </div>
        );
    }

    if (error || !strategy) {
        return (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm">
                <div className="bg-slate-900 border border-slate-700 rounded-2xl p-8 max-w-md text-center">
                    <AlertTriangle className="text-red-400 mx-auto mb-4" size={48} />
                    <h3 className="text-xl font-bold text-slate-100 mb-2">Failed to Load Strategy</h3>
                    <p className="text-slate-400 mb-4">{error || 'Strategy not found'}</p>
                    <button
                        onClick={onClose}
                        className="px-6 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg transition-all"
                    >
                        Close
                    </button>
                </div>
            </div>
        );
    }

    const results = strategy.backtest_results || {};
    const reviews = strategy.reviews_list || [];

    return (
        <>
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/90 backdrop-blur-md p-4 overflow-y-auto">
                <div className="bg-slate-900 border border-slate-700/50 w-full max-w-6xl rounded-3xl shadow-2xl relative my-8">
                    {/* Close Button */}
                    <button
                        onClick={onClose}
                        className="absolute top-6 right-6 p-2 bg-slate-800 hover:bg-slate-700 text-slate-400 rounded-xl transition-all z-10"
                    >
                        <X size={24} />
                    </button>

                    <div className="grid grid-cols-1 lg:grid-cols-12">
                        {/* Left Sidebar - Summary */}
                        <div className="lg:col-span-4 p-8 border-r border-slate-700/30 space-y-8">
                            <div className="space-y-4">
                                <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 flex items-center justify-center border border-indigo-500/20">
                                    <Zap className="text-indigo-400" size={32} />
                                </div>
                                <div>
                                    <h2 className="text-2xl font-bold text-slate-100">{strategy.name}</h2>
                                    <p className="text-slate-400">by {strategy.creator || 'Unknown'}</p>
                                </div>
                            </div>

                            <div className="space-y-4">
                                <div className="flex items-center gap-2">
                                    <Star className="text-amber-400 fill-amber-400" size={16} />
                                    <span className="font-bold text-slate-200">{(strategy.rating || 0).toFixed(1)}</span>
                                    <span className="text-slate-500 text-sm">({strategy.reviews || 0} reviews)</span>
                                </div>
                                {strategy.is_verified && (
                                    <div className="flex items-center gap-2 text-emerald-400">
                                        <Shield size={16} />
                                        <span className="text-sm font-medium">Verified Strategy</span>
                                    </div>
                                )}
                                <div className="flex flex-wrap gap-2">
                                    {(strategy.tags || []).map((tag: string) => (
                                        <span
                                            key={tag}
                                            className="px-3 py-1 bg-slate-800 border border-slate-700/50 text-slate-400 rounded-full text-[10px] font-bold uppercase tracking-wider"
                                        >
                                            {tag}
                                        </span>
                                    ))}
                                </div>
                            </div>

                            <div className="space-y-2">
                                <p className="text-sm text-slate-400 leading-relaxed">
                                    {strategy.description || 'No description available'}
                                </p>
                            </div>

                            <div className="pt-6 border-t border-slate-700/30">
                                <div className="flex items-center justify-between mb-4">
                                    <span className="text-slate-500 text-sm">Price</span>
                                    <span className="text-2xl font-black text-slate-100">
                                        {strategy.price === 0 ? 'Free' : `$${strategy.price}`}
                                    </span>
                                </div>

                                {strategy.price > 0 && (
                                    <button
                                        disabled
                                        className="w-full py-4 bg-slate-700 text-slate-400 rounded-2xl font-bold flex items-center justify-center gap-2 mb-2 cursor-not-allowed"
                                        title="Purchases coming soon"
                                    >
                                        <Download size={20} /> Purchase - Coming Soon
                                    </button>
                                )}

                                <button
                                    onClick={() => setShowDeployModal(true)}
                                    className="w-full py-4 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white rounded-2xl font-bold transition-all shadow-lg shadow-emerald-600/20 flex items-center justify-center gap-2"
                                >
                                    <Rocket size={20} /> Deploy to {strategy.price === 0 ? 'Paper' : 'Live'}
                                </button>
                            </div>
                        </div>

                        {/* Right Content - Performance */}
                        <div className="lg:col-span-8 p-8 space-y-8 overflow-y-auto max-h-[85vh]">
                            {/* Highlights Grid */}
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                {[
                                    {
                                        label: 'Total Return',
                                        val: `+${results.total_return || 0}%`,
                                        icon: ArrowUpRight,
                                        color: 'text-emerald-400'
                                    },
                                    {
                                        label: 'Sharpe Ratio',
                                        val: (results.sharpe_ratio || 0).toFixed(2),
                                        icon: Activity,
                                        color: 'text-blue-400'
                                    },
                                    {
                                        label: 'Max Drawdown',
                                        val: `${results.max_drawdown || 0}%`,
                                        icon: Shield,
                                        color: 'text-red-400'
                                    },
                                    {
                                        label: 'Win Rate',
                                        val: `${results.win_rate || 0}%`,
                                        icon: Target,
                                        color: 'text-indigo-400'
                                    },
                                ].map((item, i) => (
                                    <div key={i} className="bg-slate-800/40 border border-slate-700/30 p-4 rounded-2xl">
                                        <div className="flex items-center justify-between mb-2">
                                            <item.icon size={16} className="text-slate-500" />
                                            <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">
                                                {item.label}
                                            </p>
                                        </div>
                                        <p className={`text-xl font-black ${item.color}`}>{item.val}</p>
                                    </div>
                                ))}
                            </div>

                            {/* Performance Description */}
                            <div className="bg-slate-800/20 border border-slate-700/30 rounded-2xl p-6">
                                <h3 className="text-lg font-bold text-slate-100 mb-4">Performance Overview</h3>
                                <p className="text-slate-400 leading-relaxed">
                                    This strategy has been backtested over {strategy.period || '1 year'} with {results.num_trades || 0} trades,
                                    achieving a {results.win_rate || 0}% win rate and {results.total_return || 0}% total return.
                                    With a Sharpe ratio of {(results.sharpe_ratio || 0).toFixed(2)} and maximum drawdown of {results.max_drawdown || 0}%,
                                    it demonstrates {results.sharpe_ratio > 1.5 ? 'excellent' : 'good'} risk-adjusted returns.
                                </p>
                            </div>

                            {/* Detailed Metrics */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                <div className="space-y-4">
                                    <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest border-b border-slate-700/30 pb-2">
                                        Trading Stats
                                    </h3>
                                    <div className="space-y-3">
                                        {[
                                            { label: 'Total Trades', val: results.num_trades || 0 },
                                            { label: 'Avg Win', val: `$${results.avg_win || 0}` },
                                            { label: 'Avg Loss', val: `-$${results.avg_loss || 0}` },
                                            { label: 'Profit Factor', val: (results.profit_factor || 0).toFixed(2) },
                                        ].map((stat, i) => (
                                            <div key={i} className="flex justify-between text-sm">
                                                <span className="text-slate-500">{stat.label}</span>
                                                <span className="text-slate-200 font-medium">{stat.val}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                                <div className="space-y-4">
                                    <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest border-b border-slate-700/30 pb-2">
                                        Risk Analytics
                                    </h3>
                                    <div className="space-y-3">
                                        {[
                                            { label: 'Volatility (Ann.)', val: `${results.volatility || 0}%` },
                                            { label: 'Sortino Ratio', val: (results.sortino_ratio || 0).toFixed(2) },
                                            { label: 'Calmar Ratio', val: (results.calmar_ratio || 0).toFixed(2) },
                                            { label: 'VaR (95%)', val: `${results.var_95 || 0}%` },
                                        ].map((stat, i) => (
                                            <div key={i} className="flex justify-between text-sm">
                                                <span className="text-slate-500">{stat.label}</span>
                                                <span className="text-slate-200 font-medium">{stat.val}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>

                            {/* Pros & Cons */}
                            {(strategy.pros?.length > 0 || strategy.cons?.length > 0) && (
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                    {strategy.pros?.length > 0 && (
                                        <div className="space-y-4">
                                            <h3 className="text-sm font-bold text-emerald-400/70 uppercase tracking-widest border-b border-emerald-900/30 pb-2">
                                                Strengths
                                            </h3>
                                            <ul className="space-y-2">
                                                {strategy.pros.map((pro: string, i: number) => (
                                                    <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                                                        <CheckCircle2 size={14} className="text-emerald-500 mt-0.5 flex-shrink-0" />
                                                        <span>{pro}</span>
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                    {strategy.cons?.length > 0 && (
                                        <div className="space-y-4">
                                            <h3 className="text-sm font-bold text-red-400/70 uppercase tracking-widest border-b border-red-900/30 pb-2">
                                                Considerations
                                            </h3>
                                            <ul className="space-y-2">
                                                {strategy.cons.map((con: string, i: number) => (
                                                    <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                                                        <Clock size={14} className="text-red-400 mt-0.5 flex-shrink-0" />
                                                        <span>{con}</span>
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Reviews Section */}
                            <div className="space-y-6">
                                <div className="flex items-center justify-between">
                                    <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest border-b border-slate-700/30 pb-2 flex items-center gap-2">
                                        <MessageSquare size={16} />
                                        Reviews ({reviews.length})
                                    </h3>
                                </div>

                                {/* Write a Review */}
                                <div className="bg-slate-800/30 border border-slate-700/30 rounded-2xl p-5 space-y-4">
                                    <p className="text-sm font-semibold text-slate-300">Write a Review</p>
                                    <div className="flex items-center gap-1">
                                        {[1, 2, 3, 4, 5].map((star) => (
                                            <button
                                                key={star}
                                                onClick={() => setReviewRating(star)}
                                                className="p-0.5 transition-all"
                                            >
                                                <Star
                                                    size={20}
                                                    className={star <= reviewRating
                                                        ? 'fill-amber-400 text-amber-400'
                                                        : 'text-slate-600'
                                                    }
                                                />
                                            </button>
                                        ))}
                                        <span className="text-xs text-slate-500 ml-2">{reviewRating}/5</span>
                                    </div>
                                    <div className="flex gap-2">
                                        <input
                                            type="text"
                                            value={reviewText}
                                            onChange={(e) => setReviewText(e.target.value)}
                                            onKeyDown={(e) => e.key === 'Enter' && handleSubmitReview()}
                                            placeholder="Share your experience with this strategy..."
                                            className="flex-1 px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-sm text-slate-200 placeholder-slate-500 focus:border-indigo-500 outline-none"
                                        />
                                        <button
                                            onClick={handleSubmitReview}
                                            disabled={isSubmittingReview || !reviewText.trim()}
                                            className="px-4 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl font-semibold text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                                        >
                                            {isSubmittingReview ? (
                                                <Loader2 size={16} className="animate-spin" />
                                            ) : (
                                                <Send size={16} />
                                            )}
                                        </button>
                                    </div>
                                </div>

                                {/* Existing Reviews */}
                                {reviews.length > 0 ? (
                                    <div className="space-y-3">
                                        {reviews.map((review: any) => (
                                            <div key={review.id} className="bg-slate-800/20 border border-slate-700/20 rounded-xl p-4">
                                                <div className="flex items-center justify-between mb-2">
                                                    <div className="flex items-center gap-2">
                                                        <span className="text-sm font-semibold text-slate-200">{review.username}</span>
                                                        <div className="flex items-center gap-0.5">
                                                            {[1, 2, 3, 4, 5].map((star) => (
                                                                <Star
                                                                    key={star}
                                                                    size={12}
                                                                    className={star <= review.rating
                                                                        ? 'fill-amber-400 text-amber-400'
                                                                        : 'text-slate-700'
                                                                    }
                                                                />
                                                            ))}
                                                        </div>
                                                    </div>
                                                    <span className="text-xs text-slate-600">
                                                        {review.created_at ? new Date(review.created_at).toLocaleDateString() : ''}
                                                    </span>
                                                </div>
                                                <p className="text-sm text-slate-400">{review.review_text}</p>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <p className="text-sm text-slate-600 text-center py-4">
                                        No reviews yet. Be the first to review this strategy.
                                    </p>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Deployment Modal */}
            {showDeployModal && backtestData && (
                <DeploymentModal
                    backtest={backtestData}
                    onClose={() => setShowDeployModal(false)}
                    onDeploy={handleDeploy}
                />
            )}
        </>
    );
};

// Mock data for testing
const getMockStrategy = (id: number) => ({
    id,
    name: "Momentum Breakout Pro",
    description: "A sophisticated momentum-based strategy that identifies breakout opportunities with high win rates and controlled risk.",
    creator: "AlgoTrader",
    category: "momentum",
    tags: ["momentum", "breakout", "stocks", "verified"],
    price: 0,
    rating: 4.8,
    reviews: 127,
    total_downloads: 1543,
    is_verified: true,
    reviews_list: [],
    backtest_results: {
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
        symbols: ['SPY', 'QQQ']
    },
    pros: [
        "High win rate in trending markets",
        "Excellent risk-adjusted returns",
        "Low drawdown during normal conditions"
    ],
    cons: [
        "Requires active monitoring during market hours",
        "Performance may degrade in sideways markets",
        "Best suited for experienced traders"
    ],
    period: "2 years",
    symbols: ['SPY', 'QQQ', 'IWM'],
    parameters: {
        lookback_period: 20,
        breakout_threshold: 2.0,
        stop_loss_pct: 2.0
    }
});

export default StrategyDetailsModal;
