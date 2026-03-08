'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Search, Loader2 } from 'lucide-react';
import { api } from '@/utils/api';

interface SearchResult {
    symbol: string;
    name: string;
    type: string;
    exchange: string;
}

interface TickerSearchProps {
    /** Current value displayed in the input */
    value: string;
    /** Fired on every keystroke (uppercased) */
    onChange: (symbol: string) => void;
    /** Fired when a result is picked from the dropdown */
    onSelect?: (symbol: string, name: string) => void;
    placeholder?: string;
    className?: string;
    /** Extra Tailwind classes for the outer wrapper */
    wrapperClassName?: string;
    /** Size variant */
    size?: 'sm' | 'md';
}

export default function TickerSearch({
    value,
    onChange,
    onSelect,
    placeholder = 'Search ticker…',
    className = '',
    wrapperClassName = '',
    size = 'md',
}: TickerSearchProps) {
    const [results, setResults] = useState<SearchResult[]>([]);
    const [open, setOpen] = useState(false);
    const [loading, setLoading] = useState(false);
    const [activeIdx, setActiveIdx] = useState(-1);
    const wrapperRef = useRef<HTMLDivElement>(null);
    const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);

    // Debounced search
    const doSearch = useCallback(async (q: string) => {
        if (q.length < 1) {
            setResults([]);
            setOpen(false);
            return;
        }
        setLoading(true);
        try {
            const data = await api.market.search(q, 8) as unknown as SearchResult[];
            setResults(data);
            setOpen(data.length > 0);
            setActiveIdx(-1);
        } catch {
            setResults([]);
        } finally {
            setLoading(false);
        }
    }, []);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const val = e.target.value.toUpperCase();
        onChange(val);

        if (debounceRef.current) clearTimeout(debounceRef.current);
        debounceRef.current = setTimeout(() => doSearch(val), 300);
    };

    const pick = (r: SearchResult) => {
        onChange(r.symbol);
        onSelect?.(r.symbol, r.name);
        setOpen(false);
        setResults([]);
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (!open || results.length === 0) {
            if (e.key === 'Enter') {
                // Let parent handle raw Enter on the input value
                onSelect?.(value, '');
            }
            return;
        }

        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                setActiveIdx(i => (i < results.length - 1 ? i + 1 : 0));
                break;
            case 'ArrowUp':
                e.preventDefault();
                setActiveIdx(i => (i > 0 ? i - 1 : results.length - 1));
                break;
            case 'Enter':
                e.preventDefault();
                if (activeIdx >= 0 && activeIdx < results.length) {
                    pick(results[activeIdx]);
                } else if (results.length > 0) {
                    pick(results[0]);
                }
                break;
            case 'Escape':
                setOpen(false);
                break;
        }
    };

    // Close on outside click
    useEffect(() => {
        const handler = (e: MouseEvent) => {
            if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
                setOpen(false);
            }
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, []);

    const sizeClasses = size === 'sm'
        ? 'pl-8 pr-3 py-2 text-sm'
        : 'pl-9 pr-3 py-2.5 text-sm';

    const iconSize = size === 'sm' ? 12 : 14;

    return (
        <div ref={wrapperRef} className={`relative ${wrapperClassName}`}>
            {/* Input */}
            <div className="relative group">
                <Search
                    className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-600 group-focus-within:text-violet-500 transition-colors"
                    size={iconSize}
                />
                <input
                    type="text"
                    value={value}
                    onChange={handleChange}
                    onKeyDown={handleKeyDown}
                    onFocus={() => { if (results.length > 0) setOpen(true); }}
                    placeholder={placeholder}
                    className={`w-full ${sizeClasses} bg-slate-800/60 border border-slate-700/50 rounded-xl focus:border-violet-500 focus:ring-2 focus:ring-violet-500/20 outline-none text-slate-200 font-mono ${className}`}
                />
                {loading && (
                    <Loader2
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-violet-400 animate-spin"
                        size={iconSize}
                    />
                )}
            </div>

            {/* Dropdown */}
            {open && results.length > 0 && (
                <div className="absolute z-50 mt-1 w-full max-h-64 overflow-y-auto rounded-xl bg-slate-900 border border-slate-700/50 shadow-2xl backdrop-blur-xl">
                    {results.map((r, i) => (
                        <button
                            key={`${r.symbol}-${r.exchange}`}
                            type="button"
                            onMouseDown={() => pick(r)}
                            onMouseEnter={() => setActiveIdx(i)}
                            className={`w-full text-left px-4 py-2.5 flex items-center justify-between gap-3 transition-colors ${
                                i === activeIdx
                                    ? 'bg-violet-500/15 text-slate-100'
                                    : 'text-slate-300 hover:bg-slate-800/60'
                            }`}
                        >
                            <div className="flex items-center gap-3 min-w-0">
                                <span className="font-mono font-bold text-sm shrink-0">
                                    {r.symbol}
                                </span>
                                <span className="text-slate-400 text-xs truncate">
                                    {r.name}
                                </span>
                            </div>
                            <span className="text-[10px] font-semibold text-slate-600 uppercase shrink-0">
                                {r.type === 'EQUITY' ? r.exchange : r.type}
                            </span>
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}
