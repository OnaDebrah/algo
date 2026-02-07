'use client'

import React, {useEffect, useState} from 'react';
import {
    Activity,
    ArrowUpRight,
    CheckCircle2,
    Clock,
    Download,
    Rocket,
    Shield,
    Star,
    Target,
    X,
    Zap
} from 'lucide-react';
import {live, marketplace} from '@/utils/api';
import DeploymentModal from '@/components/strategies/DeploymentModel';
import {DeploymentConfig} from '@/types/all_types';

interface StrategyDetailsModalProps {
    strategyId: number;
    onClose: () => void;
}

const StrategyDetailsModal = ({ strategyId, onClose }: StrategyDetailsModalProps) => {
    const [strategy, setStrategy] = useState<any>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [showDeployModal, setShowDeployModal] = useState(false);

    useEffect(() => {
        const fetchDetails = async () => {
            try {
                const response = await marketplace.getDetails(strategyId);
                setStrategy(response);
            } catch (err) {
                console.error("Failed to fetch strategy details:", err);
            } finally {
                setIsLoading(false);
            }
        };
        fetchDetails();
    }, [strategyId]);

    const handleDeploy = async (config: DeploymentConfig) => {
        try {
            await live.deploy(config);
            setShowDeployModal(false);
            onClose();
        } catch (err) {
            console.error('Failed to deploy strategy:', err);
        }
    };

    // Convert marketplace strategy to backtest-like format for DeploymentModal
    const backtestData = strategy ? {
        id: String(strategy.id),
        strategy: strategy.name,
        symbols: strategy.backtest_results?.symbols || ['MULTI'],
        total_return_pct: strategy.backtest_results?.total_return || 0,
        win_rate: strategy.backtest_results?.win_rate || 0,
        sharpe_ratio: strategy.backtest_results?.sharpe_ratio || 0,
        max_drawdown: strategy.backtest_results?.max_drawdown || 0,
        total_trades: strategy.backtest_results?.num_trades || 0,
        initial_capital: strategy.backtest_results?.initial_capital || 10000,
        parameters: {}
    } : null;

    if (isLoading) {
        return (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-500 mx-auto"></div>
                    <p className="mt-4 text-slate-400">Loading metrics...</p>
                </div>
            </div>
        );
    }

    if (!strategy) return null;

    const results = strategy.backtest_results;

    return (
        <>
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/90 backdrop-blur-md p-4 md:p-10 overflow-y-auto">
                <div className="bg-slate-900 border border-slate-700/50 w-full max-w-6xl rounded-3xl shadow-2xl relative animate-in zoom-in-95 duration-300">
                    {/* Close Button */}
                    <button
                        onClick={onClose}
                        className="absolute top-6 right-6 p-2 bg-slate-800 hover:bg-slate-700 text-slate-400 rounded-xl transition-all"
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
                                    <p className="text-slate-400">by {strategy.creator}</p>
                                </div>
                            </div>

                            <div className="space-y-4">
                                <div className="flex items-center gap-2">
                                    <Star className="text-amber-400 fill-amber-400" size={16} />
                                    <span className="font-bold text-slate-200">{strategy.rating}</span>
                                    <span className="text-slate-500 text-sm">({strategy.reviews} reviews)</span>
                                </div>
                                <div className="flex flex-wrap gap-2">
                                    {strategy.tags.map((tag: string) => (
                                        <span key={tag} className="px-3 py-1 bg-slate-800 border border-slate-700/50 text-slate-400 rounded-full text-[10px] font-bold uppercase tracking-wider">
                                            {tag}
                                        </span>
                                    ))}
                                </div>
                            </div>

                            <div className="space-y-2">
                                <p className="text-sm text-slate-400 leading-relaxed">
                                    {strategy.description}
                                </p>
                            </div>

                            <div className="pt-6 border-t border-slate-700/30">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-slate-500 text-sm">Price</span>
                                    <span className="text-2xl font-black text-slate-100">${strategy.price}</span>
                                </div>
                                <button className="w-full py-4 bg-indigo-600 hover:bg-indigo-500 text-white rounded-2xl font-bold transition-all shadow-lg shadow-indigo-600/20 flex items-center justify-center gap-2 mb-2">
                                    <Download size={20} /> Get Full Strategy
                                </button>
                                <button
                                    onClick={() => setShowDeployModal(true)}
                                    className="w-full py-4 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white rounded-2xl font-bold transition-all shadow-lg shadow-emerald-600/20 flex items-center justify-center gap-2"
                                >
                                    <Rocket size={20} /> Deploy Live
                                </button>
                            </div>
                        </div>

                        {/* Right Content - Performance */}
                        <div className="lg:col-span-8 p-8 space-y-8 overflow-y-auto max-h-[90vh]">
                            {/* Highlights Grid */}
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                {[
                                    { label: 'Total Return', val: `+${results.total_return}%`, icon: ArrowUpRight, color: 'text-emerald-400' },
                                    { label: 'Sharpe Ratio', val: results.sharpe_ratio.toFixed(2), icon: Activity, color: 'text-blue-400' },
                                    { label: 'Max Drawdown', val: `${results.max_drawdown}%`, icon: Shield, color: 'text-red-400' },
                                    { label: 'Win Rate', val: `${results.win_rate}%`, icon: Target, color: 'text-indigo-400' },
                                ].map((item, i) => (
                                    <div key={i} className="bg-slate-800/40 border border-slate-700/30 p-4 rounded-2xl">
                                        <div className="flex items-center justify-between mb-2">
                                            <item.icon size={16} className="text-slate-500" />
                                            <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">{item.label}</p>
                                        </div>
                                        <p className={`text-xl font-black ${item.color}`}>{item.val}</p>
                                    </div>
                                ))}
                            </div>

                            {/* Chart Area (Mock for now) */}
                            <div className="bg-slate-800/20 border border-slate-700/30 rounded-3xl p-6 h-64 flex items-center justify-center">
                                <div className="text-center text-slate-600">
                                    <Activity size={48} className="mx-auto mb-4 opacity-20" />
                                    <p className="text-sm italic">Detailed Equity Curve Plotting...</p>
                                </div>
                            </div>

                            {/* Detailed Metrics */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                <div className="space-y-4">
                                    <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest border-b border-slate-700/30 pb-2">Trading Stats</h3>
                                    <div className="space-y-3">
                                        {[
                                            { label: 'Total Trades', val: results.num_trades },
                                            { label: 'Avg Win', val: `$${results.avg_win}` },
                                            { label: 'Avg Loss', val: `-$${results.avg_loss}` },
                                            { label: 'Profit Factor', val: results.profit_factor },
                                        ].map((stat, i) => (
                                            <div key={i} className="flex justify-between text-sm">
                                                <span className="text-slate-500">{stat.label}</span>
                                                <span className="text-slate-200 font-medium">{stat.val}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                                <div className="space-y-4">
                                    <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest border-b border-slate-700/30 pb-2">Risk Analytics</h3>
                                    <div className="space-y-3">
                                        {[
                                            { label: 'Volatility (Ann.)', val: `${results.volatility}%` },
                                            { label: 'Sortino Ratio', val: results.sortino_ratio },
                                            { label: 'Calmar Ratio', val: results.calmar_ratio },
                                            { label: 'VaR (95%)', val: `${results.var_95}%` },
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
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                <div className="space-y-4">
                                    <h3 className="text-sm font-bold text-emerald-400/70 uppercase tracking-widest border-b border-emerald-900/30 pb-2">Strengths</h3>
                                    <ul className="space-y-2">
                                        {strategy.pros.map((pro: string, i: number) => (
                                            <li key={i} className="flex items-center gap-2 text-sm text-slate-300">
                                                <CheckCircle2 size={14} className="text-emerald-500" /> {pro}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                                <div className="space-y-4">
                                    <h3 className="text-sm font-bold text-red-400/70 uppercase tracking-widest border-b border-red-900/30 pb-2">Considerations</h3>
                                    <ul className="space-y-2">
                                        {strategy.cons.map((con: string, i: number) => (
                                            <li key={i} className="flex items-center gap-2 text-sm text-slate-300">
                                                <Clock size={14} className="text-red-400" /> {con}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
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

export default StrategyDetailsModal;
