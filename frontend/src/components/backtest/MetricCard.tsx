'use client'

import React from "react";
import {Activity, AlertCircle, TrendingUp} from "lucide-react";

const MetricCard = ({ title, value, trend, color }: any) => {
    const themes: any = {
        emerald: 'from-emerald-500/10 to-transparent border-emerald-500/30 text-emerald-400',
        red: 'from-red-500/10 to-transparent border-red-500/30 text-red-400',
        blue: 'from-blue-500/10 to-transparent border-blue-500/30 text-blue-400',
        violet: 'from-violet-500/10 to-transparent border-violet-500/30 text-violet-400'
    };

    const icons: any = {
        emerald: TrendingUp,
        red: AlertCircle,
        blue: Activity,
        violet: Activity
    };

    const Icon = icons[color];

    return (
        <div className={`relative group bg-gradient-to-br border rounded-2xl p-6 transition-all hover:scale-[1.02] overflow-hidden shadow-xl ${themes[color]}`}>
            <div className="absolute -right-6 -top-6 opacity-5 group-hover:opacity-10 transition-opacity">
                <Icon size={140} />
            </div>

            <div className="relative z-10">
                <div className="flex justify-between items-start mb-6">
                    <div className="p-2.5 bg-slate-900/60 border border-slate-700/50 rounded-xl backdrop-blur-sm">
                        <Icon size={22} className="text-slate-300" strokeWidth={2} />
                    </div>
                    <div className={`flex items-center space-x-1 text-xs font-bold px-2 py-1 rounded-full ${
                        trend === 'up' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                    }`}>
                        <span>{trend === 'up' ? '↑' : '↓'}</span>
                    </div>
                </div>

                <p className="text-xs font-semibold text-slate-500 tracking-wider mb-2">
                    {title.toUpperCase()}
                </p>

                <h3 className="text-3xl font-bold text-slate-100 tracking-tight">
                    {value}
                </h3>
            </div>
        </div>
    );
};

export default MetricCard;