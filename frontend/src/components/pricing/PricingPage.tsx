/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useEffect, useState } from 'react';
import { Check, Crown, Rocket, Sparkles, Star, X, Zap } from 'lucide-react';
import { api } from '@/utils/api';
import { PricingData, QuotaStatus } from '@/types/all_types';

const tierIcons: Record<string, React.ReactNode> = {
    FREE: <Zap size={24} className="text-slate-400" />,
    BASIC: <Star size={24} className="text-blue-400" />,
    PRO: <Sparkles size={24} className="text-violet-400" />,
    ENTERPRISE: <Crown size={24} className="text-amber-400" />,
};

const tierAccents: Record<string, string> = {
    FREE: 'border-slate-700',
    BASIC: 'border-blue-500/50',
    PRO: 'border-violet-500/50 ring-2 ring-violet-500/20',
    ENTERPRISE: 'border-amber-500/50',
};

const tierBg: Record<string, string> = {
    FREE: 'from-slate-800/50 to-slate-900/50',
    BASIC: 'from-blue-900/20 to-slate-900/50',
    PRO: 'from-violet-900/20 to-slate-900/50',
    ENTERPRISE: 'from-amber-900/20 to-slate-900/50',
};

const PricingPage: React.FC = () => {
    const [data, setData] = useState<PricingData | null>(null);
    const [quota, setQuota] = useState<QuotaStatus | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const load = async () => {
            try {
                const [tierData, quotaData] = await Promise.all([
                    api.pricing.getTiers(),
                    api.pricing.getQuota().catch(() => null),
                ]);
                setData(tierData);
                setQuota(quotaData);
            } catch (err) {
                console.error('Failed to load pricing', err);
            } finally {
                setLoading(false);
            }
        };
        load();
    }, []);

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-[60vh]">
                <div className="w-10 h-10 border-3 border-violet-500 border-t-transparent rounded-full animate-spin" />
            </div>
        );
    }

    if (!data) return <p className="text-center text-slate-400 mt-20">Failed to load pricing.</p>;

    const currentTier = quota?.tier || 'FREE';

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="text-center space-y-3">
                <h1 className="text-3xl font-bold text-slate-100">Choose Your Plan</h1>
                <p className="text-slate-400 max-w-lg mx-auto">
                    Scale your algorithmic trading from research to production with the right tier.
                </p>
            </div>

            {/* Promo Banner */}
            {data.promo.active && (
                <div className="mx-auto max-w-3xl bg-gradient-to-r from-violet-500/10 via-fuchsia-500/10 to-violet-500/10 border border-violet-500/30 rounded-2xl p-5 text-center">
                    <div className="flex items-center justify-center gap-2 mb-1">
                        <Rocket size={20} className="text-violet-400" />
                        <span className="font-bold text-violet-300 text-lg">Launch Special</span>
                    </div>
                    <p className="text-slate-300">
                        All features unlocked free for 6 months! Only backtest limits apply.
                    </p>
                    {data.promo.ends_at && (
                        <p className="text-xs text-slate-500 mt-2">
                            Promo ends {new Date(data.promo.ends_at).toLocaleDateString()}
                        </p>
                    )}
                </div>
            )}

            {/* Quota Indicator */}
            {quota && quota.limit !== null && (
                <div className="mx-auto max-w-md">
                    <div className="flex items-center justify-between text-sm mb-2">
                        <span className="text-slate-400">Monthly Backtests</span>
                        <span className="text-slate-300 font-medium">
                            {quota.used} / {quota.limit}
                        </span>
                    </div>
                    <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div
                            className={`h-full rounded-full transition-all ${
                                (quota.used / quota.limit) >= 0.9
                                    ? 'bg-red-500'
                                    : (quota.used / quota.limit) >= 0.7
                                      ? 'bg-amber-500'
                                      : 'bg-emerald-500'
                            }`}
                            style={{ width: `${Math.min((quota.used / quota.limit) * 100, 100)}%` }}
                        />
                    </div>
                    {quota.remaining !== null && quota.remaining <= 5 && quota.remaining > 0 && (
                        <p className="text-xs text-amber-400 mt-1">
                            {quota.remaining} backtest{quota.remaining !== 1 ? 's' : ''} remaining this month
                        </p>
                    )}
                </div>
            )}

            {/* Pricing Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
                {data.tiers.map((tier) => {
                    const isCurrent = tier.tier === currentTier;
                    return (
                        <div
                            key={tier.tier}
                            className={`relative bg-gradient-to-b ${tierBg[tier.tier]} border ${tierAccents[tier.tier]} rounded-2xl p-6 flex flex-col transition-all hover:scale-[1.02]`}
                        >
                            {/* Popular badge */}
                            {tier.tier === 'PRO' && (
                                <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-violet-500 text-white text-xs font-bold px-4 py-1 rounded-full">
                                    MOST POPULAR
                                </div>
                            )}

                            {/* Current plan badge */}
                            {isCurrent && (
                                <div className="absolute top-4 right-4 bg-emerald-500/20 text-emerald-400 text-xs font-bold px-3 py-1 rounded-full border border-emerald-500/30">
                                    CURRENT
                                </div>
                            )}

                            {/* Tier header */}
                            <div className="flex items-center gap-3 mb-4">
                                {tierIcons[tier.tier]}
                                <div>
                                    <h3 className="text-lg font-bold text-slate-100">{tier.label}</h3>
                                    <p className="text-xs text-slate-500">{tier.tier}</p>
                                </div>
                            </div>

                            {/* Price */}
                            <div className="mb-6">
                                <span className="text-4xl font-bold text-slate-100">
                                    ${tier.price}
                                </span>
                                {tier.price > 0 && (
                                    <span className="text-slate-500 text-sm"> / month</span>
                                )}
                                {tier.price === 0 && (
                                    <span className="text-slate-500 text-sm ml-2">forever</span>
                                )}
                            </div>

                            {/* Features */}
                            <ul className="space-y-3 flex-1 mb-6">
                                {tier.features.map((feature, i) => (
                                    <li key={i} className="flex items-start gap-2 text-sm">
                                        <Check size={16} className="text-emerald-400 mt-0.5 shrink-0" />
                                        <span className="text-slate-300">{feature}</span>
                                    </li>
                                ))}
                            </ul>

                            {/* CTA */}
                            <button
                                disabled={isCurrent}
                                className={`w-full py-3 rounded-xl font-semibold text-sm transition-all ${
                                    isCurrent
                                        ? 'bg-slate-800 text-slate-500 cursor-default'
                                        : tier.tier === 'PRO'
                                          ? 'bg-violet-500 hover:bg-violet-600 text-white shadow-lg shadow-violet-500/30'
                                          : 'bg-slate-800 hover:bg-slate-700 text-slate-200'
                                }`}
                            >
                                {isCurrent
                                    ? 'Current Plan'
                                    : data.promo.active
                                      ? 'Coming Soon'
                                      : tier.price === 0
                                        ? 'Get Started'
                                        : 'Upgrade'}
                            </button>
                        </div>
                    );
                })}
            </div>

            {/* FAQ / Fine Print */}
            <div className="text-center text-sm text-slate-500 space-y-1 pt-6 border-t border-slate-800/80">
                <p>All plans include access to the performance dashboard and market data.</p>
                <p>Backtest limits reset on the 1st of each month (UTC).</p>
                {data.promo.active && (
                    <p className="text-violet-400">
                        During the launch promo, all features are unlocked regardless of tier.
                    </p>
                )}
            </div>
        </div>
    );
};

export default PricingPage;
