'use client'

import {Download, Heart, Shield, Star} from "lucide-react";

export const StrategyCard = ({ strategy, onSelect, onFavorite, onDownload }: any) => (
    <div
        onClick={onSelect}
        className="group bg-slate-900 border border-slate-700/50 rounded-2xl p-6 hover:border-indigo-500/50 transition-all cursor-pointer hover:shadow-xl hover:shadow-indigo-500/10"
    >
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
            <div className="flex-1">
                <div className="flex items-center gap-2 mb-2">
                    <h3 className="font-bold text-slate-100 text-lg truncate">{strategy.name}</h3>
                    {strategy.is_verified && (
                        <Shield className="text-emerald-400 flex-shrink-0" size={16} />
                    )}
                </div>
                <p className="text-xs text-slate-500">by {strategy.creator}</p>
            </div>
            <button
                onClick={onFavorite}
                className="p-2 hover:bg-slate-800 rounded-lg transition-all"
            >
                <Heart
                    size={18}
                    className={strategy.is_favorite ? 'fill-red-500 text-red-500' : 'text-slate-500'}
                />
            </button>
        </div>

        {/* Tags */}
        <div className="flex flex-wrap gap-2 mb-4">
            {strategy.tags.slice(0, 3).map((tag: string) => (
                <span
                    key={tag}
                    className="px-2 py-1 bg-slate-800 text-slate-400 rounded text-xs font-medium"
                >
                    {tag}
                </span>
            ))}
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-2 gap-3 mb-4">
            <div className="p-3 bg-slate-800/50 rounded-lg">
                <p className="text-xs text-slate-500 mb-1">Return</p>
                <p className="text-sm font-bold text-emerald-400">
                    +{strategy.total_return || 0}%
                </p>
            </div>
            <div className="p-3 bg-slate-800/50 rounded-lg">
                <p className="text-xs text-slate-500 mb-1">Sharpe</p>
                <p className="text-sm font-bold text-blue-400">
                    {(strategy.sharpe_ratio || 0).toFixed(2)}
                </p>
            </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between pt-4 border-t border-slate-700/50">
            <div className="flex items-center gap-3 text-xs text-slate-500">
                <div className="flex items-center gap-1">
                    <Star className="fill-amber-400 text-amber-400" size={14} />
                    <span>{strategy.rating.toFixed(1)}</span>
                </div>
                <div className="flex items-center gap-1">
                    <Download size={14} />
                    <span>{strategy.total_downloads}</span>
                </div>
            </div>
            <div className="text-lg font-bold text-slate-100">
                {strategy.price === 0 ? 'Free' : `$${strategy.price}`}
            </div>
        </div>
    </div>
);
