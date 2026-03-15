import React from 'react';
import { ArrowUpCircle, X } from 'lucide-react';
import { useNavigationStore } from '@/store/useNavigationStore';

interface UpgradePromptProps {
    used: number;
    limit: number;
    tier: string;
    onClose: () => void;
}

const TIER_ORDER = ['FREE', 'BASIC', 'PRO', 'ENTERPRISE'];

const UpgradePrompt: React.FC<UpgradePromptProps> = ({ used, limit, tier, onClose }) => {
    const navigateTo = useNavigationStore((s) => s.navigateTo);
    const currentIdx = TIER_ORDER.indexOf(tier);
    const nextTier = currentIdx < TIER_ORDER.length - 1 ? TIER_ORDER[currentIdx + 1] : null;

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <div className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl max-w-md w-full p-8 relative animate-in zoom-in-95 fade-in duration-200">
                {/* Close */}
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 text-slate-500 hover:text-slate-300 transition-colors"
                >
                    <X size={20} />
                </button>

                {/* Icon */}
                <div className="flex justify-center mb-5">
                    <div className="w-16 h-16 rounded-2xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center">
                        <ArrowUpCircle size={32} className="text-amber-400" />
                    </div>
                </div>

                {/* Content */}
                <h2 className="text-xl font-bold text-slate-100 text-center mb-2">
                    Backtest Limit Reached
                </h2>
                <p className="text-slate-400 text-center text-sm mb-6">
                    You&apos;ve used <span className="text-slate-200 font-semibold">{used}</span> of{' '}
                    <span className="text-slate-200 font-semibold">{limit}</span> backtests this month
                    on the <span className="text-violet-400 font-semibold">{tier}</span> plan.
                </p>

                {/* Actions */}
                <div className="space-y-3">
                    {nextTier && (
                        <button
                            onClick={() => {
                                onClose();
                                navigateTo('pricing' as any);
                            }}
                            className="w-full py-3 bg-violet-500 hover:bg-violet-600 text-white font-semibold rounded-xl transition-colors shadow-lg shadow-violet-500/30"
                        >
                            Upgrade to {nextTier}
                        </button>
                    )}
                    <button
                        onClick={() => {
                            onClose();
                            navigateTo('pricing' as any);
                        }}
                        className="w-full py-3 bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium rounded-xl transition-colors"
                    >
                        View All Plans
                    </button>
                    <button
                        onClick={onClose}
                        className="w-full py-2 text-slate-500 hover:text-slate-300 text-sm transition-colors"
                    >
                        Maybe Later
                    </button>
                </div>

                <p className="text-xs text-slate-600 text-center mt-4">
                    Limits reset on the 1st of each month.
                </p>
            </div>
        </div>
    );
};

export default UpgradePrompt;
