import React from 'react';
import {AlertCircle, AlertTriangle, CheckCircle, XCircle} from 'lucide-react';

interface ValidationCheckProps {
    title: string;
    description: string;
    status: 'pass' | 'warn' | 'fail' | 'pending';
    value: string;
    threshold?: string;
}

const ValidationCheck: React.FC<ValidationCheckProps> = ({ title, description, status, value, threshold }) => {
    const statusConfig = {
        pass: { icon: CheckCircle, color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30' },
        warn: { icon: AlertTriangle, color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/30' },
        fail: { icon: XCircle, color: 'text-red-400', bg: 'bg-red-500/10', border: 'border-red-500/30' },
        pending: { icon: AlertCircle, color: 'text-slate-400', bg: 'bg-slate-800/50', border: 'border-slate-700' }
    };

    const config = statusConfig[status];
    const Icon = config.icon;

    return (
        <div className={`p-4 rounded-lg border ${config.bg} ${config.border}`}>
            <div className="flex items-start gap-3">
                <Icon className={config.color} size={20} />
                <div className="flex-1">
                    <h5 className="text-sm font-bold text-slate-200 mb-1">{title}</h5>
                    <p className="text-xs text-slate-400 mb-2">{description}</p>
                    <div className="flex items-center justify-between">
                        <span className={`text-sm font-mono ${config.color}`}>{value}</span>
                        {threshold && <span className="text-xs text-slate-500">{threshold}</span>}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ValidationCheck;
