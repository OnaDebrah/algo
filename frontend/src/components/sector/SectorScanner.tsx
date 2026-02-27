'use client';

import React, {useCallback, useEffect, useState} from 'react';
import {ArrowDown, ArrowUp, BarChart3, ChevronRight, Loader2, RefreshCw, Target,} from 'lucide-react';
import {sector} from '@/utils/api';
import {SectorScanResult, SectorSummary, StockRanking, StrategyRecommendation} from '@/types/all_types';
import {useNavigationStore} from '@/store/useNavigationStore';

// ── Regime badge colours ───────────────────────────────────────────
const REGIME_COLORS: Record<string, string> = {
    bull: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    bear: 'bg-red-500/20 text-red-400 border-red-500/30',
    neutral: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
    high_volatility: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    low_volatility: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
};

const REGIME_LABELS: Record<string, string> = {
    bull: 'Bullish',
    bear: 'Bearish',
    neutral: 'Neutral',
    high_volatility: 'High Volatility',
    low_volatility: 'Low Volatility',
};

// ── Score → colour interpolation ───────────────────────────────────
function scoreColor(score: number): string {
    if (score > 0.5) return 'text-emerald-400';
    if (score > 0.1) return 'text-emerald-300';
    if (score > -0.1) return 'text-slate-400';
    if (score > -0.5) return 'text-red-300';
    return 'text-red-400';
}

function scoreBg(score: number): string {
    if (score > 0.5) return 'from-emerald-900/30 to-emerald-800/10 border-emerald-500/30';
    if (score > 0.1) return 'from-emerald-900/20 to-slate-900/10 border-emerald-500/20';
    if (score > -0.1) return 'from-slate-800/30 to-slate-900/10 border-slate-600/30';
    if (score > -0.5) return 'from-red-900/20 to-slate-900/10 border-red-500/20';
    return 'from-red-900/30 to-red-800/10 border-red-500/30';
}

function pctStr(v: number): string {
    return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`;
}

const SectorScanner: React.FC = () => {
    const navigateTo = useNavigationStore(state => state.navigateTo);

    // Data state
    const [scanResult, setScanResult] = useState<SectorScanResult | null>(null);
    const [stockRankings, setStockRankings] = useState<StockRanking[]>([]);
    const [recommendations, setRecommendations] = useState<StrategyRecommendation[]>([]);

    // Selection state
    const [selectedSector, setSelectedSector] = useState<string | null>(null);
    const [selectedStock, setSelectedStock] = useState<string | null>(null);

    // Loading state
    const [scanLoading, setScanLoading] = useState(false);
    const [stocksLoading, setStocksLoading] = useState(false);
    const [recsLoading, setRecsLoading] = useState(false);

    // Sort state for stock table
    const [sortField, setSortField] = useState<string>('rank');
    const [sortAsc, setSortAsc] = useState(true);

    // ── Scan all sectors ────────────────────────────────────────────
    const handleScan = useCallback(async () => {
        setScanLoading(true);
        setSelectedSector(null);
        setSelectedStock(null);
        setStockRankings([]);
        setRecommendations([]);
        try {
            const result = await sector.scan('6mo') as unknown as SectorScanResult;
            setScanResult(result);
        } catch (e) {
            console.error('Sector scan failed:', e);
        } finally {
            setScanLoading(false);
        }
    }, []);

    // ── Fetch stock rankings for a sector ───────────────────────────
    const handleSectorClick = useCallback(async (sectorName: string) => {
        setSelectedSector(sectorName);
        setSelectedStock(null);
        setRecommendations([]);
        setStocksLoading(true);
        try {
            const ranked = await sector.stocks(sectorName, 10) as unknown as StockRanking[];
            setStockRankings(ranked);
        } catch (e) {
            console.error('Stock ranking failed:', e);
            setStockRankings([]);
        } finally {
            setStocksLoading(false);
        }
    }, []);

    // ── Fetch strategy recommendations for a stock ──────────────────
    const handleStockClick = useCallback(async (symbol: string) => {
        setSelectedStock(symbol);
        setRecsLoading(true);
        try {
            const recs = await sector.recommend(symbol) as unknown as StrategyRecommendation[];
            setRecommendations(recs);
        } catch (e) {
            console.error('Strategy recommendation failed:', e);
            setRecommendations([]);
        } finally {
            setRecsLoading(false);
        }
    }, []);

    // Auto-scan on mount
    useEffect(() => {
        handleScan();
    }, [handleScan]);

    // ── Sort stocks ─────────────────────────────────────────────────
    const handleSort = (field: string) => {
        if (sortField === field) {
            setSortAsc(!sortAsc);
        } else {
            setSortField(field);
            setSortAsc(field === 'rank');
        }
    };

    const sortedStocks = [...stockRankings].sort((a, b) => {
        const aVal = (a as Record<string, any>)[sortField] ?? 0;
        const bVal = (b as Record<string, any>)[sortField] ?? 0;
        return sortAsc ? aVal - bVal : bVal - aVal;
    });

    // ================================================================
    // RENDER
    // ================================================================
    return (
        <div className="space-y-6 p-6 max-w-7xl mx-auto">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
                            <Target size={22} className="text-white"/>
                        </div>
                        Sector Scanner
                    </h1>
                    <p className="text-sm text-slate-500 mt-1">
                        ML-powered sector analysis, stock ranking &amp; strategy recommendations
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    {scanResult && (
                        <span
                            className={`px-3 py-1 text-xs font-semibold rounded-full border ${REGIME_COLORS[scanResult.market_regime] || REGIME_COLORS.neutral}`}>
                            Market: {REGIME_LABELS[scanResult.market_regime] || scanResult.market_regime}
                        </span>
                    )}
                    <button
                        onClick={handleScan}
                        disabled={scanLoading}
                        className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 rounded-lg text-sm font-medium text-white transition-colors"
                    >
                        {scanLoading ? <Loader2 size={16} className="animate-spin"/> :
                            <RefreshCw size={16}/>}
                        Scan Sectors
                    </button>
                </div>
            </div>

            {/* ─── Panel 1: Sector Heatmap ─────────────────────────────── */}
            <div>
                <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">
                    Sector Rankings
                </h2>
                {scanLoading && !scanResult ? (
                    <div className="flex items-center justify-center py-20">
                        <Loader2 size={32} className="animate-spin text-cyan-400"/>
                        <span className="ml-3 text-slate-400">Scanning 10 sectors...</span>
                    </div>
                ) : scanResult ? (
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
                        {scanResult.sectors.map((s: SectorSummary) => (
                            <button
                                key={s.name}
                                onClick={() => handleSectorClick(s.name)}
                                className={`relative p-4 rounded-xl border bg-gradient-to-br transition-all duration-200 text-left
                                    ${scoreBg(s.composite_score)}
                                    ${selectedSector === s.name
                                    ? 'ring-2 ring-cyan-500 shadow-lg shadow-cyan-500/10'
                                    : 'hover:ring-1 hover:ring-slate-600'}`}
                            >
                                {/* Rank badge */}
                                <div
                                    className="absolute top-2 right-2 w-6 h-6 rounded-full bg-slate-800/80 flex items-center justify-center">
                                    <span className="text-xs font-bold text-slate-300">
                                        {s.rank}
                                    </span>
                                </div>

                                <div className="text-xs text-slate-500 font-medium">
                                    {s.etf}
                                </div>
                                <div className="text-sm font-bold text-slate-200 mt-1 truncate">
                                    {s.name}
                                </div>

                                <div className="mt-3 space-y-1">
                                    <div className="flex justify-between text-xs">
                                        <span className="text-slate-500">Momentum</span>
                                        <span className={scoreColor(s.momentum_score)}>
                                            {pctStr(s.momentum_score)}
                                        </span>
                                    </div>
                                    <div className="flex justify-between text-xs">
                                        <span className="text-slate-500">Return</span>
                                        <span className={scoreColor(s.total_return)}>
                                            {pctStr(s.total_return)}
                                        </span>
                                    </div>
                                    <div className="flex justify-between text-xs">
                                        <span className="text-slate-500">Volatility</span>
                                        <span className="text-slate-400">
                                            {(s.volatility * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                </div>
                            </button>
                        ))}
                    </div>
                ) : null}
            </div>

            {/* ─── Panel 2: Stock Rankings Table ───────────────────────── */}
            {selectedSector && (
                <div>
                    <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                        <ChevronRight size={14}/>
                        Top Stocks in {selectedSector}
                    </h2>
                    {stocksLoading ? (
                        <div className="flex items-center justify-center py-16">
                            <Loader2 size={24} className="animate-spin text-cyan-400"/>
                            <span className="ml-3 text-slate-400">Ranking stocks...</span>
                        </div>
                    ) : stockRankings.length > 0 ? (
                        <div className="bg-slate-900/50 border border-slate-800/60 rounded-xl overflow-hidden">
                            <table className="w-full text-sm">
                                <thead>
                                <tr className="border-b border-slate-800/60">
                                    {[
                                        {key: 'rank', label: '#'},
                                        {key: 'symbol', label: 'Symbol'},
                                        {key: 'composite_score', label: 'Score'},
                                        {key: 'percentile', label: 'Percentile'},
                                        {key: 'confidence', label: 'Confidence'},
                                    ].map(col => (
                                        <th
                                            key={col.key}
                                            onClick={() => handleSort(col.key)}
                                            className="px-4 py-3 text-left text-xs font-semibold text-slate-500 uppercase cursor-pointer hover:text-slate-300 transition-colors"
                                        >
                                            <span className="flex items-center gap-1">
                                                {col.label}
                                                {sortField === col.key && (
                                                    sortAsc ? <ArrowUp size={12}/> :
                                                        <ArrowDown size={12}/>
                                                )}
                                            </span>
                                        </th>
                                    ))}
                                </tr>
                                </thead>
                                <tbody>
                                {sortedStocks.map((stock) => (
                                    <tr
                                        key={stock.symbol}
                                        onClick={() => handleStockClick(stock.symbol)}
                                        className={`border-b border-slate-800/30 cursor-pointer transition-colors
                                            ${selectedStock === stock.symbol
                                            ? 'bg-cyan-500/10'
                                            : 'hover:bg-slate-800/30'}`}
                                    >
                                        <td className="px-4 py-3 text-slate-500 font-mono">
                                            {stock.rank}
                                        </td>
                                        <td className="px-4 py-3 font-bold text-slate-200">
                                            {stock.symbol}
                                        </td>
                                        <td className={`px-4 py-3 font-mono ${scoreColor(stock.composite_score)}`}>
                                            {stock.composite_score.toFixed(3)}
                                        </td>
                                        <td className="px-4 py-3">
                                            <div className="flex items-center gap-2">
                                                <div
                                                    className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                                    <div
                                                        className="h-full bg-cyan-500 rounded-full"
                                                        style={{width: `${stock.percentile * 100}%`}}
                                                    />
                                                </div>
                                                <span className="text-slate-400 text-xs">
                                                    {(stock.percentile * 100).toFixed(0)}%
                                                </span>
                                            </div>
                                        </td>
                                        <td className="px-4 py-3 text-slate-400">
                                            {(stock.confidence * 100).toFixed(0)}%
                                        </td>
                                    </tr>
                                ))}
                                </tbody>
                            </table>
                        </div>
                    ) : (
                        <div className="text-center py-10 text-slate-500">
                            No stock rankings available for this sector.
                        </div>
                    )}

                    {/* Factor breakdown for selected stock */}
                    {selectedStock && stockRankings.length > 0 && (
                        <div className="mt-3 p-4 bg-slate-900/30 border border-slate-800/40 rounded-xl">
                            <h3 className="text-xs font-semibold text-slate-500 uppercase mb-2">
                                Factor Breakdown — {selectedStock}
                            </h3>
                            <div className="flex flex-wrap gap-2">
                                {(() => {
                                    const stock = stockRankings.find(s => s.symbol === selectedStock);
                                    if (!stock) return null;
                                    return Object.entries(stock.factor_scores)
                                        .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                                        .slice(0, 8)
                                        .map(([factor, value]) => (
                                            <span key={factor}
                                                  className="px-2 py-1 bg-slate-800/50 rounded text-xs">
                                                <span className="text-slate-500">
                                                    {factor.replace(/_/g, ' ')}
                                                </span>
                                                {' '}
                                                <span className={scoreColor(value)}>
                                                    {value.toFixed(2)}
                                                </span>
                                            </span>
                                        ));
                                })()}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* ─── Panel 3: Strategy Recommendations ───────────────────── */}
            {selectedStock && (
                <div>
                    <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                        <ChevronRight size={14}/>
                        Strategy Recommendations for {selectedStock}
                    </h2>
                    {recsLoading ? (
                        <div className="flex items-center justify-center py-12">
                            <Loader2 size={24} className="animate-spin text-cyan-400"/>
                            <span className="ml-3 text-slate-400">Analysing regime &amp; matching strategies...</span>
                        </div>
                    ) : recommendations.length > 0 ? (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {recommendations.map((rec, idx) => (
                                <div
                                    key={idx}
                                    className="p-5 bg-slate-900/50 border border-slate-800/60 rounded-xl hover:border-slate-700/60 transition-colors"
                                >
                                    <div className="flex items-start justify-between mb-3">
                                        <div>
                                            <h3 className="font-bold text-slate-200 text-sm">
                                                {rec.strategy_name}
                                            </h3>
                                            <div className="flex items-center gap-2 mt-1">
                                                <span
                                                    className="px-2 py-0.5 text-[10px] font-semibold uppercase rounded bg-slate-800 text-slate-400">
                                                    {rec.category}
                                                </span>
                                                <span
                                                    className={`px-2 py-0.5 text-[10px] font-semibold uppercase rounded border ${REGIME_COLORS[rec.regime] || REGIME_COLORS.neutral}`}>
                                                    {REGIME_LABELS[rec.regime] || rec.regime}
                                                </span>
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <div className="text-lg font-bold text-cyan-400">
                                                {(rec.suitability_score * 100).toFixed(0)}
                                            </div>
                                            <div className="text-[10px] text-slate-500 uppercase">
                                                Fit Score
                                            </div>
                                        </div>
                                    </div>

                                    {/* Suitability bar */}
                                    <div className="w-full h-1.5 bg-slate-800 rounded-full mb-3">
                                        <div
                                            className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all"
                                            style={{width: `${rec.suitability_score * 100}%`}}
                                        />
                                    </div>

                                    <p className="text-xs text-slate-500 mb-4 line-clamp-2">
                                        {rec.reason}
                                    </p>

                                    <div className="flex items-center justify-between">
                                        <div className="flex gap-2">
                                            <span
                                                className="text-[10px] text-slate-600">
                                                {rec.complexity}
                                            </span>
                                            <span className="text-slate-700">|</span>
                                            <span
                                                className="text-[10px] text-slate-600">
                                                {rec.time_horizon}
                                            </span>
                                        </div>
                                        <button
                                            onClick={() => navigateTo('backtest')}
                                            className="flex items-center gap-1 px-3 py-1.5 bg-cyan-600/20 hover:bg-cyan-600/30 border border-cyan-500/30 rounded-lg text-xs text-cyan-400 font-medium transition-colors"
                                        >
                                            <BarChart3 size={12}/>
                                            Backtest
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="text-center py-10 text-slate-500">
                            No strategy recommendations available.
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default SectorScanner;
