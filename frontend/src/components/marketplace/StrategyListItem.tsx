import {Download, Heart, Shield, Star, Zap} from "lucide-react";

export const StrategyListItem = ({ strategy, onSelect, onFavorite, onDownload }: any) => (
    <div
        onClick={onSelect}
        className="bg-slate-900 border border-slate-700/50 rounded-xl p-6 hover:border-indigo-500/50 transition-all cursor-pointer flex items-center gap-6"
    >
        <div className="w-16 h-16 bg-gradient-to-br from-indigo-500/20 to-violet-500/20 rounded-xl flex items-center justify-center">
            <Zap className="text-indigo-400" size={32} />
        </div>

        <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
                <h3 className="font-bold text-slate-100 text-lg">{strategy.name}</h3>
                {strategy.is_verified && <Shield className="text-emerald-400" size={16} />}
            </div>
            <p className="text-sm text-slate-400 mb-2">{strategy.description}</p>
            <div className="flex items-center gap-4 text-xs text-slate-500">
                <span>by {strategy.creator}</span>
                <div className="flex items-center gap-1">
                    <Star className="fill-amber-400 text-amber-400" size={12} />
                    {strategy.rating.toFixed(1)} ({strategy.reviews})
                </div>
                <div className="flex items-center gap-1">
                    <Download size={12} />
                    {strategy.total_downloads}
                </div>
            </div>
        </div>

        <div className="flex items-center gap-4">
            <div className="text-right">
                <p className="text-xs text-slate-500 mb-1">Return</p>
                <p className="text-lg font-bold text-emerald-400">
                    +{strategy.total_return || 0}%
                </p>
            </div>
            <div className="text-right">
                <p className="text-xs text-slate-500 mb-1">Sharpe</p>
                <p className="text-lg font-bold text-blue-400">
                    {(strategy.sharpe_ratio || 0).toFixed(2)}
                </p>
            </div>
            <div className="text-2xl font-bold text-slate-100 min-w-[80px] text-right">
                {strategy.price === 0 ? 'Free' : `$${strategy.price}`}
            </div>
        </div>

        <button
            onClick={(e) => { e.stopPropagation(); onFavorite?.(e); }}
            className="p-3 hover:bg-slate-800 rounded-lg transition-all"
        >
            <Heart
                size={20}
                className={strategy.is_favorite ? 'fill-red-500 text-red-500' : 'text-slate-500'}
            />
        </button>
    </div>
);
