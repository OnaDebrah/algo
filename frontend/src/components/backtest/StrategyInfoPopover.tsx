'use client'
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import {
    Activity,
    BarChart3,
    Clock,
    Info,
    Shield,
    Star,
    Target,
    TrendingDown,
    TrendingUp,
    X,
    Zap
} from 'lucide-react';
import { Strategy } from '@/types/all_types';

interface StrategyInfoPopoverProps {
    strategy: Strategy;
}

const POPOVER_WIDTH = 340;

const StrategyInfoPopover: React.FC<StrategyInfoPopoverProps> = ({ strategy }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [position, setPosition] = useState<{ top: number; left: number } | null>(null);
    const popoverRef = useRef<HTMLDivElement>(null);
    const triggerRef = useRef<HTMLButtonElement>(null);

    // Calculate position based on trigger button location
    const updatePosition = useCallback(() => {
        if (!triggerRef.current) return;
        const rect = triggerRef.current.getBoundingClientRect();
        const viewportHeight = window.innerHeight;
        const viewportWidth = window.innerWidth;

        // Prefer opening above; if not enough space, open below
        const spaceAbove = rect.top;
        const spaceBelow = viewportHeight - rect.bottom;
        const popoverEstimatedHeight = 380;

        let top: number;
        if (spaceAbove >= popoverEstimatedHeight || spaceAbove > spaceBelow) {
            // Position above the trigger
            top = rect.top + window.scrollY - 8; // 8px gap
        } else {
            // Position below the trigger
            top = rect.bottom + window.scrollY + 8;
        }

        // Horizontal: align right edge to trigger right, but keep within viewport
        let left = rect.right + window.scrollX - POPOVER_WIDTH;
        if (left < 8) left = 8;
        if (left + POPOVER_WIDTH > viewportWidth - 8) {
            left = viewportWidth - POPOVER_WIDTH - 8;
        }

        setPosition({ top, left });
    }, []);

    // Open handler
    const handleToggle = useCallback((e: React.MouseEvent) => {
        e.stopPropagation();
        e.preventDefault();
        if (!isOpen) {
            updatePosition();
        }
        setIsOpen(prev => !prev);
    }, [isOpen, updatePosition]);

    // Close on click outside
    useEffect(() => {
        if (!isOpen) return;
        const handleClickOutside = (e: MouseEvent) => {
            if (
                popoverRef.current && !popoverRef.current.contains(e.target as Node) &&
                triggerRef.current && !triggerRef.current.contains(e.target as Node)
            ) {
                setIsOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [isOpen]);

    // Close on Escape
    useEffect(() => {
        if (!isOpen) return;
        const handleEscape = (e: KeyboardEvent) => {
            if (e.key === 'Escape') setIsOpen(false);
        };
        document.addEventListener('keydown', handleEscape);
        return () => document.removeEventListener('keydown', handleEscape);
    }, [isOpen]);

    // Reposition on scroll or resize
    useEffect(() => {
        if (!isOpen) return;
        const handleReposition = () => updatePosition();
        window.addEventListener('scroll', handleReposition, true);
        window.addEventListener('resize', handleReposition);
        return () => {
            window.removeEventListener('scroll', handleReposition, true);
            window.removeEventListener('resize', handleReposition);
        };
    }, [isOpen, updatePosition]);

    // Adjust position once popover is rendered so it doesn't overflow viewport
    useEffect(() => {
        if (!isOpen || !popoverRef.current || !triggerRef.current || !position) return;
        const popoverRect = popoverRef.current.getBoundingClientRect();
        const triggerRect = triggerRef.current.getBoundingClientRect();
        const viewportHeight = window.innerHeight;

        // If popover overflows top, flip to below
        if (popoverRect.top < 0) {
            setPosition(prev => prev ? {
                ...prev,
                top: triggerRect.bottom + window.scrollY + 8
            } : prev);
        }
        // If popover overflows bottom, flip to above
        else if (popoverRect.bottom > viewportHeight) {
            setPosition(prev => prev ? {
                ...prev,
                top: triggerRect.top + window.scrollY - popoverRect.height - 8
            } : prev);
        }
    }, [isOpen, position]);

    const complexityConfig: Record<string, { color: string; bg: string; border: string; level: number }> = {
        'Beginner': { color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/20', level: 1 },
        'Intermediate': { color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/20', level: 2 },
        'Advanced': { color: 'text-orange-400', bg: 'bg-orange-500/10', border: 'border-orange-500/20', level: 3 },
        'Expert': { color: 'text-red-400', bg: 'bg-red-500/10', border: 'border-red-500/20', level: 4 },
        'Institutional': { color: 'text-purple-400', bg: 'bg-purple-500/10', border: 'border-purple-500/20', level: 5 },
    };

    const comp = complexityConfig[strategy.complexity] || complexityConfig['Beginner'];

    const popoverContent = position ? (
        <div
            ref={popoverRef}
            style={{
                position: 'absolute',
                top: position.top,
                left: position.left,
                width: POPOVER_WIDTH,
                transform: 'translateY(-100%)',
            }}
            className="z-[9999] bg-gradient-to-br from-slate-900 via-slate-900 to-slate-800 border border-slate-700/60 rounded-2xl shadow-2xl shadow-black/50 animate-in fade-in zoom-in-95 duration-200"
            onClick={(e) => e.stopPropagation()}
        >
            {/* Header */}
            <div className="p-4 pb-3 border-b border-slate-700/40">
                <div className="flex items-start justify-between gap-3">
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1.5">
                            <span className={`text-[9px] font-black uppercase tracking-wider px-2 py-0.5 rounded-md ${comp.bg} ${comp.color} ${comp.border} border`}>
                                {strategy.category}
                            </span>
                            {strategy.rating && strategy.rating > 0 && (
                                <span className="flex items-center gap-0.5 text-amber-400">
                                    <Star size={10} fill="currentColor" />
                                    <span className="text-[10px] font-bold">{strategy.rating.toFixed(1)}</span>
                                </span>
                            )}
                        </div>
                        <h3 className="text-sm font-bold text-slate-100 leading-tight">{strategy.name}</h3>
                        <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">{strategy.description}</p>
                    </div>
                    <button
                        onClick={(e) => { e.stopPropagation(); setIsOpen(false); }}
                        className="p-1 rounded-lg text-slate-600 hover:text-slate-300 hover:bg-slate-800/80 transition-all shrink-0"
                    >
                        <X size={14} />
                    </button>
                </div>
            </div>

            {/* Strategy Intelligence Grid */}
            <div className="p-4 pt-3 space-y-3">
                {/* Meta info row */}
                <div className="grid grid-cols-3 gap-2">
                    {/* Complexity */}
                    <div className={`p-2 rounded-lg ${comp.bg} border ${comp.border}`}>
                        <div className="flex items-center gap-1.5 mb-1">
                            <Shield size={10} className={comp.color} />
                            <span className="text-[9px] font-bold text-slate-500 uppercase">Complexity</span>
                        </div>
                        <p className={`text-xs font-bold ${comp.color}`}>{strategy.complexity}</p>
                        <div className="flex gap-0.5 mt-1">
                            {[1, 2, 3, 4, 5].map(i => (
                                <div
                                    key={i}
                                    className={`h-1 flex-1 rounded-full ${i <= comp.level
                                        ? `${comp.color.replace('text-', 'bg-').replace('-400', '-500')}`
                                        : 'bg-slate-800'
                                    }`}
                                />
                            ))}
                        </div>
                    </div>

                    {/* Time Horizon */}
                    <div className="p-2 rounded-lg bg-blue-500/5 border border-blue-500/20">
                        <div className="flex items-center gap-1.5 mb-1">
                            <Clock size={10} className="text-blue-400" />
                            <span className="text-[9px] font-bold text-slate-500 uppercase">Horizon</span>
                        </div>
                        <p className="text-xs font-bold text-blue-400">{strategy.time_horizon || 'Any'}</p>
                    </div>

                    {/* Category */}
                    <div className="p-2 rounded-lg bg-violet-500/5 border border-violet-500/20">
                        <div className="flex items-center gap-1.5 mb-1">
                            <Target size={10} className="text-violet-400" />
                            <span className="text-[9px] font-bold text-slate-500 uppercase">Type</span>
                        </div>
                        <p className="text-xs font-bold text-violet-400 truncate">{strategy.category}</p>
                    </div>
                </div>

                {/* Performance metrics (if available) */}
                {(strategy.monthly_return !== undefined || strategy.sharpe_ratio !== undefined || strategy.drawdown !== undefined) && (
                    <div>
                        <h4 className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                            <BarChart3 size={10} className="text-slate-600" />
                            Expected Performance
                        </h4>
                        <div className="grid grid-cols-3 gap-2">
                            {strategy.monthly_return !== undefined && (
                                <div className="p-2 bg-slate-800/40 rounded-lg border border-slate-700/30">
                                    <div className="flex items-center gap-1 mb-0.5">
                                        <TrendingUp size={9} className="text-emerald-400" />
                                        <span className="text-[9px] text-slate-500 font-bold">Return/mo</span>
                                    </div>
                                    <p className={`text-xs font-black ${strategy.monthly_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                        {strategy.monthly_return >= 0 ? '+' : ''}{strategy.monthly_return.toFixed(1)}%
                                    </p>
                                </div>
                            )}
                            {strategy.sharpe_ratio !== undefined && (
                                <div className="p-2 bg-slate-800/40 rounded-lg border border-slate-700/30">
                                    <div className="flex items-center gap-1 mb-0.5">
                                        <Activity size={9} className="text-blue-400" />
                                        <span className="text-[9px] text-slate-500 font-bold">Sharpe</span>
                                    </div>
                                    <p className={`text-xs font-black ${(strategy.sharpe_ratio || 0) >= 1 ? 'text-emerald-400' : (strategy.sharpe_ratio || 0) >= 0.5 ? 'text-amber-400' : 'text-red-400'}`}>
                                        {strategy.sharpe_ratio?.toFixed(2) || 'N/A'}
                                    </p>
                                </div>
                            )}
                            {strategy.drawdown !== undefined && (
                                <div className="p-2 bg-slate-800/40 rounded-lg border border-slate-700/30">
                                    <div className="flex items-center gap-1 mb-0.5">
                                        <TrendingDown size={9} className="text-red-400" />
                                        <span className="text-[9px] text-slate-500 font-bold">Drawdown</span>
                                    </div>
                                    <p className="text-xs font-black text-red-400">
                                        {strategy.drawdown?.toFixed(1)}%
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* Best For tags */}
                {strategy.best_for && strategy.best_for.length > 0 && (
                    <div>
                        <h4 className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                            <Zap size={10} className="text-slate-600" />
                            Best For
                        </h4>
                        <div className="flex flex-wrap gap-1.5">
                            {strategy.best_for.map((tag, i) => (
                                <span
                                    key={i}
                                    className="text-[10px] font-bold text-slate-300 bg-slate-800/60 border border-slate-700/40 px-2 py-0.5 rounded-md"
                                >
                                    {tag}
                                </span>
                            ))}
                        </div>
                    </div>
                )}

                {/* Parameters preview */}
                {strategy.parameterMetadata && strategy.parameterMetadata.length > 0 && (
                    <div>
                        <h4 className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                            <Activity size={10} className="text-slate-600" />
                            Parameters ({strategy.parameterMetadata.length})
                        </h4>
                        <div className="space-y-1 max-h-[100px] overflow-y-auto custom-scrollbar">
                            {strategy.parameterMetadata.map((param) => (
                                <div
                                    key={param.name}
                                    className="flex items-center justify-between py-1 px-2 bg-slate-800/30 rounded-md"
                                >
                                    <span className="text-[10px] font-mono font-bold text-slate-400">{param.name}</span>
                                    <div className="flex items-center gap-2">
                                        <span className="text-[9px] text-slate-600">{param.type}</span>
                                        <span className="text-[10px] font-mono font-bold text-violet-400">
                                            {param.default !== null && param.default !== undefined ? String(param.default) : '—'}
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    ) : null;

    return (
        <>
            <span
                ref={triggerRef}
                onClick={handleToggle}
                className="p-1 rounded-md text-slate-600 hover:text-violet-400 hover:bg-violet-500/10 transition-all"
                title="Strategy details"
            >
                <Info size={12} />
            </span>

            {isOpen && popoverContent && createPortal(popoverContent, document.body)}
        </>
    );
};

export default StrategyInfoPopover;
