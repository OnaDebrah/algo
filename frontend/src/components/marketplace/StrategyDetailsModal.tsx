/* eslint-disable @typescript-eslint/no-explicit-any */

'use client'

import React, {useEffect, useState} from 'react';
import {
    Activity,
    AlertTriangle,
    ArrowUpRight,
    CheckCircle2,
    Clock,
    Download,
    GitFork,
    Loader2,
    Lock,
    MessageSquare,
    Rocket,
    Send,
    Shield,
    Star,
    Target,
    Unlock,
    X,
    Zap
} from 'lucide-react';
import {live, marketplace, payments} from '@/utils/api';

import {DeploymentConfig} from '@/types/all_types';
import DeploymentModal from "@/components/strategies/DeploymentModel";
import {toPrecision} from "@/utils/formatters";

interface StrategyDetailsModalProps {
    strategyId: number;
    onClose: () => void;
}

const StrategyDetailsModal = ({ strategyId, onClose }: StrategyDetailsModalProps) => {
    const [strategy, setStrategy] = useState<any>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [showDeployModal, setShowDeployModal] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Clone state
    const [isCloning, setIsCloning] = useState(false);

    // Review state
    const [reviewRating, setReviewRating] = useState(5);
    const [reviewText, setReviewText] = useState('');
    const [isSubmittingReview, setIsSubmittingReview] = useState(false);

    // Discussion state
    const [comments, setComments] = useState<any[]>([]);
    const [commentText, setCommentText] = useState('');
    const [replyTo, setReplyTo] = useState<{ id: number; username: string } | null>(null);
    const [isSubmittingComment, setIsSubmittingComment] = useState(false);
    const [isLoadingComments, setIsLoadingComments] = useState(false);

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

    const fetchComments = async () => {
        setIsLoadingComments(true);
        try {
            const data = await marketplace.getComments(strategyId);
            setComments(Array.isArray(data) ? data : []);
        } catch {
            // Silently fail — comments are non-critical
        } finally {
            setIsLoadingComments(false);
        }
    };

    const handleSubmitComment = async () => {
        if (!commentText.trim()) return;
        setIsSubmittingComment(true);
        try {
            await marketplace.createComment(strategyId, {
                content: commentText.trim(),
                parent_comment_id: replyTo?.id,
            });
            setCommentText('');
            setReplyTo(null);
            fetchComments();
        } catch (err: any) {
            console.error('Failed to post comment:', err);
        } finally {
            setIsSubmittingComment(false);
        }
    };

    const handleDeleteComment = async (commentId: number) => {
        try {
            await marketplace.deleteComment(commentId);
            fetchComments();
        } catch (err: any) {
            console.error('Failed to delete comment:', err);
        }
    };

    useEffect(() => {
        fetchDetails();
        fetchComments();
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

    const handleClone = async () => {
        setIsCloning(true);
        try {
            const res = await marketplace.cloneStrategy(strategyId);
            alert(res.message || 'Strategy forked successfully!');
        } catch (err: any) {
            console.error('Failed to clone strategy:', err);
            alert(err?.response?.data?.detail || 'Failed to fork strategy');
        } finally {
            setIsCloning(false);
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

                                {strategy.price > 0 && !strategy.is_purchased ? (
                                    <button
                                        onClick={async () => {
                                            try {
                                                const res = await payments.createCheckoutSession(strategy.id as number);
                                                window.location.href = res.checkout_url;
                                            } catch (err: any) {
                                                alert(err?.response?.data?.detail || 'Failed to start checkout');
                                            }
                                        }}
                                        className="w-full py-4 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 text-white rounded-2xl font-bold transition-all shadow-lg shadow-amber-600/20 flex items-center justify-center gap-2 mb-2"
                                    >
                                        <Unlock size={20} /> Unlock for ${strategy.price}
                                    </button>
                                ) : (
                                    <button
                                        onClick={() => setShowDeployModal(true)}
                                        className="w-full py-4 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white rounded-2xl font-bold transition-all shadow-lg shadow-emerald-600/20 flex items-center justify-center gap-2"
                                    >
                                        <Rocket size={20} /> Deploy to {strategy.price === 0 ? 'Paper' : 'Live'}
                                    </button>
                                )}

                                <button
                                    onClick={handleClone}
                                    disabled={isCloning}
                                    className="w-full py-3 mt-2 bg-slate-800 hover:bg-slate-700 border border-slate-700/50 text-slate-200 rounded-2xl font-semibold transition-all flex items-center justify-center gap-2 text-sm disabled:opacity-50"
                                >
                                    {isCloning ? (
                                        <Loader2 size={16} className="animate-spin" />
                                    ) : (
                                        <GitFork size={16} />
                                    )}
                                    Fork Strategy
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
                                        val: `+${toPrecision(results.total_return) || 0}%`,
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
                                        val: `${toPrecision(results.max_drawdown) || 0}%`,
                                        icon: Shield,
                                        color: 'text-red-400'
                                    },
                                    {
                                        label: 'Win Rate',
                                        val: `${toPrecision(results.win_rate) || 0}%`,
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
                                    achieving a {toPrecision(results.win_rate) || 0}% win rate and {toPrecision(results.total_return) || 0}% total return.
                                    With a Sharpe ratio of {(results.sharpe_ratio || 0).toFixed(2)} and maximum drawdown of {toPrecision(results.max_drawdown) || 0}%,
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

                            {/* Discussion Section */}
                            <div className="space-y-6">
                                <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest border-b border-slate-700/30 pb-2 flex items-center gap-2">
                                    <MessageSquare size={16} />
                                    Discussion ({comments.reduce((acc: number, c: any) => acc + 1 + (c.replies?.length || 0), 0)})
                                </h3>

                                {/* Write a Comment */}
                                <div className="bg-slate-800/30 border border-slate-700/30 rounded-2xl p-5 space-y-3">
                                    {replyTo && (
                                        <div className="flex items-center gap-2 text-xs text-violet-400">
                                            <span>Replying to {replyTo.username}</span>
                                            <button onClick={() => setReplyTo(null)} className="text-slate-500 hover:text-slate-300">
                                                <X size={12} />
                                            </button>
                                        </div>
                                    )}
                                    <div className="flex gap-2">
                                        <input
                                            type="text"
                                            value={commentText}
                                            onChange={(e) => setCommentText(e.target.value)}
                                            onKeyDown={(e) => e.key === 'Enter' && handleSubmitComment()}
                                            placeholder={replyTo ? `Reply to ${replyTo.username}...` : "Join the discussion..."}
                                            className="flex-1 px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-sm text-slate-200 placeholder-slate-500 focus:border-violet-500 outline-none"
                                        />
                                        <button
                                            onClick={handleSubmitComment}
                                            disabled={isSubmittingComment || !commentText.trim()}
                                            className="px-4 py-2.5 bg-violet-600 hover:bg-violet-500 text-white rounded-xl font-semibold text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                                        >
                                            {isSubmittingComment ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
                                        </button>
                                    </div>
                                </div>

                                {/* Comments Thread */}
                                {isLoadingComments ? (
                                    <div className="flex justify-center py-4">
                                        <Loader2 size={20} className="animate-spin text-slate-500" />
                                    </div>
                                ) : comments.length > 0 ? (
                                    <div className="space-y-3">
                                        {comments.map((comment: any) => (
                                            <div key={comment.id} className="space-y-2">
                                                {/* Top-level comment */}
                                                <div className="bg-slate-800/20 border border-slate-700/20 rounded-xl p-4">
                                                    <div className="flex items-center justify-between mb-2">
                                                        <div className="flex items-center gap-2">
                                                            <span className="text-sm font-semibold text-slate-200">{comment.username}</span>
                                                            {comment.is_edited && <span className="text-[10px] text-slate-600">(edited)</span>}
                                                        </div>
                                                        <div className="flex items-center gap-2">
                                                            <span className="text-xs text-slate-600">
                                                                {comment.created_at ? new Date(comment.created_at).toLocaleDateString() : ''}
                                                            </span>
                                                            <button
                                                                onClick={() => setReplyTo({ id: comment.id, username: comment.username })}
                                                                className="text-xs text-violet-400 hover:text-violet-300"
                                                            >
                                                                Reply
                                                            </button>
                                                        </div>
                                                    </div>
                                                    <p className="text-sm text-slate-400">{comment.content}</p>
                                                </div>

                                                {/* Replies */}
                                                {comment.replies?.length > 0 && (
                                                    <div className="ml-6 space-y-2">
                                                        {comment.replies.map((reply: any) => (
                                                            <div key={reply.id} className="bg-slate-800/10 border border-slate-700/15 rounded-lg p-3">
                                                                <div className="flex items-center justify-between mb-1">
                                                                    <div className="flex items-center gap-2">
                                                                        <span className="text-xs font-semibold text-slate-300">{reply.username}</span>
                                                                        {reply.is_edited && <span className="text-[10px] text-slate-600">(edited)</span>}
                                                                    </div>
                                                                    <span className="text-[10px] text-slate-600">
                                                                        {reply.created_at ? new Date(reply.created_at).toLocaleDateString() : ''}
                                                                    </span>
                                                                </div>
                                                                <p className="text-xs text-slate-400">{reply.content}</p>
                                                            </div>
                                                        ))}
                                                    </div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <p className="text-sm text-slate-600 text-center py-4">
                                        No comments yet. Start the discussion!
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
