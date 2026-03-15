import React, { useEffect, useState } from 'react';
import { BarChart3 } from 'lucide-react';
import { api } from '@/utils/api';
import { QuotaStatus } from '@/types/all_types';

interface QuotaIndicatorProps {
    className?: string;
}

const QuotaIndicator: React.FC<QuotaIndicatorProps> = ({ className = '' }) => {
    const [quota, setQuota] = useState<QuotaStatus | null>(null);

    useEffect(() => {
        api.pricing.getQuota().then(setQuota).catch(() => {});
    }, []);

    if (!quota || quota.limit === null) return null;

    const pct = Math.min((quota.used / quota.limit) * 100, 100);
    const color =
        pct >= 90 ? 'bg-red-500' : pct >= 70 ? 'bg-amber-500' : 'bg-emerald-500';
    const textColor =
        pct >= 90 ? 'text-red-400' : pct >= 70 ? 'text-amber-400' : 'text-emerald-400';

    return (
        <div className={`flex items-center gap-3 ${className}`}>
            <BarChart3 size={14} className="text-slate-500" />
            <div className="flex-1 min-w-[120px]">
                <div className="flex items-center justify-between text-xs mb-1">
                    <span className="text-slate-500">Backtests</span>
                    <span className={`font-medium ${textColor}`}>
                        {quota.used}/{quota.limit}
                    </span>
                </div>
                <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                        className={`h-full rounded-full transition-all ${color}`}
                        style={{ width: `${pct}%` }}
                    />
                </div>
            </div>
        </div>
    );
};

export default QuotaIndicator;
