/* eslint-disable @typescript-eslint/no-explicit-any */
'use client'

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
    ArrowDownRight,
    ArrowUpRight,
    Banknote,
    Bot,
    Clock,
    Loader2,
    Pause,
    Play,
    Plus,
    RefreshCw,
    ShoppingCart,
    TrendingUp,
    Trash2,
    Unlink,
    X,
    Zap,
} from 'lucide-react';
import {
    Area,
    AreaChart,
    CartesianGrid,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';

import { paper } from '@/utils/api';
import { useNavigationStore } from '@/store/useNavigationStore';
import { formatCurrency } from '@/utils/formatters';
import {
    MarketStatus,
    PaperEquitySnapshot,
    PaperPerformance,
    PaperPortfolio,
    PaperStrategyInfo,
    PaperTrade,
    StrategySignalResult,
} from '@/types/all_types';

const AUTO_RUN_INTERVALS = [
    { label: '1 min', ms: 60_000 },
    { label: '5 min', ms: 300_000 },
    { label: '15 min', ms: 900_000 },
    { label: '1 hour', ms: 3_600_000 },
    { label: 'Daily', ms: 86_400_000 },
];

const DATA_INTERVALS = [
    { value: '1m', label: '1 Minute' },
    { value: '5m', label: '5 Minutes' },
    { value: '15m', label: '15 Minutes' },
    { value: '30m', label: '30 Minutes' },
    { value: '1h', label: '1 Hour' },
    { value: '4h', label: '4 Hours' },
    { value: '1d', label: '1 Day' },
    { value: '1wk', label: '1 Week' },
];

// Suggest appropriate auto-run polling interval based on data interval
const INTERVAL_AUTORUN_SUGGESTION: Record<string, number> = {
    '1m': 60_000,      // poll every 1 min
    '5m': 300_000,     // 5 min
    '15m': 900_000,    // 15 min
    '30m': 900_000,    // 15 min (half a candle)
    '1h': 3_600_000,   // 1 hour
    '4h': 3_600_000,   // 1 hour
    '1d': 86_400_000,  // daily
    '1wk': 86_400_000, // daily
};

const PaperTradingDashboard = () => {
    // Portfolio state
    const [portfolios, setPortfolios] = useState<PaperPortfolio[]>([]);
    const [activePortfolio, setActivePortfolio] = useState<PaperPortfolio | null>(null);
    const [activeId, setActiveId] = useState<number | null>(null);
    const [loading, setLoading] = useState(true);

    // Trade form
    const [tradeSymbol, setTradeSymbol] = useState('');
    const [tradeQty, setTradeQty] = useState('');
    const [tradeSide, setTradeSide] = useState<'buy' | 'sell'>('buy');
    const [tradingLoading, setTradingLoading] = useState(false);
    const [tradeError, setTradeError] = useState<string | null>(null);

    // Data
    const [trades, setTrades] = useState<PaperTrade[]>([]);
    const [equity, setEquity] = useState<PaperEquitySnapshot[]>([]);
    const [performance, setPerformance] = useState<PaperPerformance | null>(null);

    // Create portfolio modal
    const [showCreate, setShowCreate] = useState(false);
    const [newName, setNewName] = useState('');
    const [newCash, setNewCash] = useState(100000);

    // Strategy state
    const [strategies, setStrategies] = useState<PaperStrategyInfo[]>([]);
    const [showStrategyPicker, setShowStrategyPicker] = useState(false);
    const [selectedStratKey, setSelectedStratKey] = useState('');
    const [stratSymbol, setStratSymbol] = useState('');
    const [stratQty, setStratQty] = useState(100);
    const [stratInterval, setStratInterval] = useState('1d');
    const [stratCategory, setStratCategory] = useState('all');
    const [attachingStrategy, setAttachingStrategy] = useState(false);
    const [signalLoading, setSignalLoading] = useState(false);
    const [lastSignal, setLastSignal] = useState<StrategySignalResult | null>(null);
    const [refreshingEquity, setRefreshingEquity] = useState(false);

    // Auto-run & market status
    const [autoRunActive, setAutoRunActive] = useState(false);
    const [autoRunInterval, setAutoRunInterval] = useState(300_000); // default 5 min
    const [marketStatus, setMarketStatus] = useState<MarketStatus | null>(null);
    const [signalHistory, setSignalHistory] = useState<StrategySignalResult[]>([]);
    const autoRunRef = useRef<ReturnType<typeof setInterval> | null>(null);

    const fetchPortfolios = useCallback(async () => {
        try {
            const data = await paper.listPortfolios();
            setPortfolios(data);
            if (data.length > 0 && !activeId) {
                setActiveId(data[0].id);
            }
        } catch {
            // ignore
        }
    }, [activeId]);

    const fetchPortfolioData = useCallback(async (id: number) => {
        try {
            const [portfolioData, tradesData, equityData, perfData] = await Promise.all([
                paper.getPortfolio(id),
                paper.getTrades(id),
                paper.getEquity(id),
                paper.getPerformance(id),
            ]);
            setActivePortfolio(portfolioData);
            setTrades(tradesData);
            setEquity(equityData);
            setPerformance(perfData);
        } catch {
            // ignore
        }
    }, []);

    useEffect(() => {
        setLoading(true);
        Promise.all([
            fetchPortfolios(),
            paper.listStrategies().then(setStrategies).catch(() => {}),
        ]).finally(() => setLoading(false));
    }, [fetchPortfolios]);

    useEffect(() => {
        if (activeId) {
            fetchPortfolioData(activeId);
            const interval = setInterval(() => fetchPortfolioData(activeId), 30000);
            return () => clearInterval(interval);
        }
    }, [activeId, fetchPortfolioData]);

    // Market status polling (every 60s)
    useEffect(() => {
        paper.marketStatus().then(setMarketStatus).catch(() => {});
        const iv = setInterval(() => {
            paper.marketStatus().then(setMarketStatus).catch(() => {});
        }, 60_000);
        return () => clearInterval(iv);
    }, []);

    // Auto-create portfolio from navigation payload (e.g., from Backtest → Paper Trade)
    const payloadHandled = useRef(false);
    useEffect(() => {
        if (payloadHandled.current || loading) return;
        const payload = useNavigationStore.getState().pendingPayload;
        if (payload?.fromBacktest) {
            payloadHandled.current = true;
            useNavigationStore.getState().clearPending();

            const stratKey = payload.strategy_key as string;
            const stratSym = (payload.strategy_symbol as string) || '';
            const interval = (payload.data_interval as string) || '1d';
            const capital = (payload.initial_capital as number) || 100000;
            const qty = (payload.trade_quantity as number) || 100;
            const params = (payload.strategy_params as Record<string, unknown>) || undefined;
            const autoName = `${stratSym} ${stratKey} ${interval} Paper`.slice(0, 100);

            (async () => {
                try {
                    const created = await paper.createPortfolio(
                        autoName, capital, stratKey, stratSym, params, qty, interval,
                    );
                    setPortfolios(prev => [created, ...prev]);
                    setActiveId(created.id);
                } catch {
                    // If creation fails (e.g., duplicate name), just navigate — user can create manually
                }
            })();
        }
    }, [loading]);

    // Auto-run strategy on interval
    useEffect(() => {
        if (autoRunRef.current) {
            clearInterval(autoRunRef.current);
            autoRunRef.current = null;
        }
        if (autoRunActive && activeId && activePortfolio?.strategy_key) {
            const tick = async () => {
                try {
                    const result = await paper.runSignal(activeId, true);
                    setLastSignal(result);
                    setSignalHistory(prev => [result, ...prev].slice(0, 50));
                    if (result.trade_executed) {
                        await fetchPortfolioData(activeId);
                    }
                } catch {
                    // keep running
                }
            };
            // Run immediately on start, then on interval
            tick();
            autoRunRef.current = setInterval(tick, autoRunInterval);
        }
        return () => {
            if (autoRunRef.current) {
                clearInterval(autoRunRef.current);
                autoRunRef.current = null;
            }
        };
    }, [autoRunActive, autoRunInterval, activeId, activePortfolio?.strategy_key]);

    // Stop auto-run when strategy detached or portfolio changed
    useEffect(() => {
        if (!activePortfolio?.strategy_key) {
            setAutoRunActive(false);
        }
    }, [activePortfolio?.strategy_key]);

    const handleCreatePortfolio = async () => {
        if (!newName.trim()) return;
        try {
            const created = await paper.createPortfolio(newName, newCash);
            setPortfolios(prev => [created, ...prev]);
            setActiveId(created.id);
            setShowCreate(false);
            setNewName('');
            setNewCash(100000);
        } catch {
            // ignore
        }
    };

    const handlePlaceTrade = async () => {
        if (!activeId || !tradeSymbol.trim() || !tradeQty) return;
        setTradingLoading(true);
        setTradeError(null);

        try {
            await paper.placeTrade(activeId, tradeSymbol.toUpperCase(), tradeSide, parseFloat(tradeQty));
            setTradeSymbol('');
            setTradeQty('');
            await fetchPortfolioData(activeId);
        } catch (err: any) {
            const detail = err?.response?.data?.detail ?? err?.message ?? 'Trade failed';
            setTradeError(typeof detail === 'string' ? detail : JSON.stringify(detail));
        } finally {
            setTradingLoading(false);
        }
    };

    const handleAttachStrategy = async () => {
        if (!activeId || !selectedStratKey || !stratSymbol.trim()) return;
        setAttachingStrategy(true);
        try {
            const updated = await paper.attachStrategy(activeId, selectedStratKey, stratSymbol.toUpperCase(), undefined, stratQty, stratInterval);
            setActivePortfolio(updated);
            setPortfolios(prev => prev.map(p => p.id === updated.id ? updated : p));
            setShowStrategyPicker(false);
            setSelectedStratKey('');
            setStratSymbol('');
        } catch (err: any) {
            setTradeError(err?.response?.data?.detail ?? 'Failed to attach strategy');
        } finally {
            setAttachingStrategy(false);
        }
    };

    const handleDetachStrategy = async () => {
        if (!activeId) return;
        try {
            const updated = await paper.detachStrategy(activeId);
            setActivePortfolio(updated);
            setPortfolios(prev => prev.map(p => p.id === updated.id ? updated : p));
            setLastSignal(null);
        } catch {
            // ignore
        }
    };

    const handleRunSignal = async (autoExecute: boolean = true) => {
        if (!activeId) return;
        setSignalLoading(true);
        setLastSignal(null);
        try {
            const result = await paper.runSignal(activeId, autoExecute);
            setLastSignal(result);
            if (result.trade_executed) {
                await fetchPortfolioData(activeId);
            }
        } catch (err: any) {
            setTradeError(err?.response?.data?.detail ?? 'Signal execution failed');
        } finally {
            setSignalLoading(false);
        }
    };

    const handleRefreshEquity = async () => {
        if (!activeId) return;
        setRefreshingEquity(true);
        try {
            await paper.refreshEquity(activeId);
            await fetchPortfolioData(activeId);
        } catch {
            // ignore
        } finally {
            setRefreshingEquity(false);
        }
    };

    const handleDelete = async (id: number) => {
        await paper.deactivate(id);
        setPortfolios(prev => prev.filter(p => p.id !== id));
        if (activeId === id) {
            setActiveId(portfolios.find(p => p.id !== id)?.id ?? null);
            setActivePortfolio(null);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-[60vh]">
                <Loader2 className="animate-spin text-violet-500" size={32} />
            </div>
        );
    }

    const equityChartData = equity.map(s => ({
        time: s.timestamp ? new Date(s.timestamp).toLocaleDateString() : '',
        equity: s.equity,
    }));

    const hasStrategy = !!activePortfolio?.strategy_key;

    // Get unique strategy categories for filter
    const categories = ['all', ...Array.from(new Set(strategies.map(s => s.category)))];
    const filteredStrategies = stratCategory === 'all'
        ? strategies
        : strategies.filter(s => s.category === stratCategory);

    return (
        <div className="space-y-6 animate-in fade-in duration-700">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-3xl font-bold text-foreground tracking-tight flex items-center gap-3">
                        <Banknote className="text-emerald-400" size={32} />
                        Paper <span className="text-muted-foreground font-normal">Trading</span>
                    </h2>
                    <p className="text-muted-foreground text-sm mt-1">Test strategies with virtual money in real-time market conditions</p>
                </div>
                <div className="flex items-center gap-3">
                    {/* Market status */}
                    {marketStatus && (
                        <div className={`flex items-center gap-1.5 px-3 py-2 rounded-xl text-[10px] font-bold border ${
                            marketStatus.market_open
                                ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                                : 'bg-amber-500/10 border-amber-500/20 text-amber-400'
                        }`}>
                            <div className={`w-1.5 h-1.5 rounded-full ${marketStatus.market_open ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
                            {marketStatus.market_open ? 'MARKET OPEN' : 'MARKET CLOSED'}
                        </div>
                    )}
                    {/* Portfolio selector */}
                    <select
                        value={activeId ?? ''}
                        onChange={(e) => setActiveId(Number(e.target.value))}
                        className="bg-card border border-border rounded-xl py-2.5 px-4 text-sm text-foreground focus:border-emerald-500 outline-none"
                    >
                        {portfolios.map(p => (
                            <option key={p.id} value={p.id}>{p.name}</option>
                        ))}
                    </select>
                    <button
                        onClick={() => setShowCreate(true)}
                        className="flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white rounded-xl text-sm font-bold transition-all shadow-lg shadow-emerald-500/20"
                    >
                        <Plus size={16} /> New Portfolio
                    </button>
                </div>
            </div>

            {/* Create modal */}
            {showCreate && (
                <div className="bg-card border border-border rounded-2xl p-6 space-y-4">
                    <div className="flex items-center justify-between">
                        <h3 className="text-lg font-bold text-foreground">Create Paper Portfolio</h3>
                        <button onClick={() => setShowCreate(false)} className="text-muted-foreground hover:text-foreground">
                            <X size={18} />
                        </button>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="text-xs font-bold text-muted-foreground block mb-1">Portfolio Name</label>
                            <input
                                value={newName}
                                onChange={e => setNewName(e.target.value)}
                                className="w-full bg-background border border-border rounded-xl py-2.5 px-4 text-sm text-foreground outline-none focus:border-emerald-500"
                                placeholder="My Paper Portfolio"
                            />
                        </div>
                        <div>
                            <label className="text-xs font-bold text-muted-foreground block mb-1">Starting Cash</label>
                            <input
                                type="number"
                                value={newCash}
                                onChange={e => setNewCash(Number(e.target.value))}
                                className="w-full bg-background border border-border rounded-xl py-2.5 px-4 text-sm text-foreground outline-none focus:border-emerald-500"
                            />
                        </div>
                    </div>
                    <button
                        onClick={handleCreatePortfolio}
                        className="px-6 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl text-sm font-bold transition-all"
                    >
                        Create
                    </button>
                </div>
            )}

            {!activePortfolio && !showCreate && (
                <div className="bg-card border border-border rounded-2xl p-12 text-center">
                    <Banknote className="mx-auto text-muted-foreground mb-3" size={48} />
                    <p className="text-muted-foreground">No paper portfolios yet. Create one to start trading!</p>
                </div>
            )}

            {activePortfolio && (
                <>
                    {/* Strategy Banner */}
                    {hasStrategy ? (
                        <div className="bg-gradient-to-r from-violet-500/10 to-fuchsia-500/10 border border-violet-500/30 rounded-2xl p-4 space-y-3">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <Bot className="text-violet-400" size={20} />
                                    <div>
                                        <p className="text-sm font-bold text-foreground flex items-center gap-2">
                                            Strategy: <span className="text-violet-400">{activePortfolio.strategy_key}</span>
                                            <span className="text-muted-foreground font-normal">·</span>
                                            <span className="text-emerald-400 font-mono">{activePortfolio.strategy_symbol}</span>
                                            <span className="text-muted-foreground font-normal">·</span>
                                            <span className="text-amber-400 font-mono text-xs">{activePortfolio.data_interval || '1d'}</span>
                                            <span className="text-muted-foreground font-normal text-xs">({activePortfolio.trade_quantity} shares/signal)</span>
                                        </p>
                                        {lastSignal && (
                                            <p className="text-xs text-muted-foreground mt-0.5">
                                                Last signal: <span className={`font-bold ${
                                                    lastSignal.signal === 1 ? 'text-emerald-400' :
                                                    lastSignal.signal === -1 ? 'text-red-400' : 'text-amber-400'
                                                }`}>{lastSignal.signal_label}</span>
                                                {lastSignal.trade_detail && (
                                                    <span className="ml-2">&mdash; {lastSignal.trade_detail}</span>
                                                )}
                                                {lastSignal.data_as_of && (
                                                    <span className="ml-2 text-muted-foreground/60">
                                                        (data as of {new Date(lastSignal.data_as_of).toLocaleDateString()})
                                                    </span>
                                                )}
                                                {!lastSignal.market_open && (
                                                    <span className="ml-2 text-amber-400/80">
                                                        <Clock className="inline" size={10} /> using last close price
                                                    </span>
                                                )}
                                            </p>
                                        )}
                                    </div>
                                </div>
                                <div className="flex items-center gap-2">
                                    <button
                                        onClick={() => handleRunSignal(false)}
                                        disabled={signalLoading || autoRunActive}
                                        className="flex items-center gap-1.5 px-3 py-2 bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 rounded-xl text-xs font-bold transition-all border border-amber-500/20 disabled:opacity-50"
                                        title="Check signal without trading"
                                    >
                                        {signalLoading ? <Loader2 size={12} className="animate-spin" /> : <Zap size={12} />}
                                        Check
                                    </button>
                                    <button
                                        onClick={() => handleRunSignal(true)}
                                        disabled={signalLoading || autoRunActive}
                                        className="flex items-center gap-1.5 px-3 py-2 bg-violet-600 hover:bg-violet-500 text-white rounded-xl text-xs font-bold transition-all shadow-lg shadow-violet-500/20 disabled:opacity-50"
                                    >
                                        {signalLoading ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
                                        Run Once
                                    </button>
                                    <button
                                        onClick={handleDetachStrategy}
                                        disabled={autoRunActive}
                                        className="flex items-center gap-1.5 px-3 py-2 text-muted-foreground hover:text-red-400 rounded-xl text-xs font-bold transition-all hover:bg-red-500/10 disabled:opacity-50"
                                        title="Remove strategy"
                                    >
                                        <Unlink size={12} />
                                    </button>
                                </div>
                            </div>

                            {/* Auto-Run Controls */}
                            <div className="flex items-center gap-3 pt-2 border-t border-violet-500/15">
                                <button
                                    onClick={() => setAutoRunActive(!autoRunActive)}
                                    className={`flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-bold transition-all ${
                                        autoRunActive
                                            ? 'bg-emerald-500 text-white shadow-lg shadow-emerald-500/30 animate-pulse'
                                            : 'bg-card border border-border text-muted-foreground hover:text-foreground hover:border-violet-500/40'
                                    }`}
                                >
                                    {autoRunActive ? <Pause size={12} /> : <Play size={12} />}
                                    {autoRunActive ? 'Stop Auto-Run' : 'Start Auto-Run'}
                                </button>

                                <select
                                    value={autoRunInterval}
                                    onChange={e => setAutoRunInterval(Number(e.target.value))}
                                    disabled={autoRunActive}
                                    className="bg-card border border-border rounded-xl py-2 px-3 text-xs text-foreground outline-none disabled:opacity-50"
                                >
                                    {AUTO_RUN_INTERVALS.map(iv => (
                                        <option key={iv.ms} value={iv.ms}>Every {iv.label}</option>
                                    ))}
                                </select>

                                {autoRunActive && (
                                    <span className="text-[10px] text-emerald-400 font-bold flex items-center gap-1">
                                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                                        RUNNING — strategy checks every {AUTO_RUN_INTERVALS.find(iv => iv.ms === autoRunInterval)?.label}
                                        {signalHistory.length > 0 && (
                                            <span className="text-muted-foreground ml-1">
                                                ({signalHistory.filter(s => s.trade_executed).length} trades from {signalHistory.length} checks)
                                            </span>
                                        )}
                                    </span>
                                )}
                            </div>
                        </div>
                    ) : (
                        <button
                            onClick={() => setShowStrategyPicker(true)}
                            className="w-full bg-card border border-dashed border-violet-500/30 rounded-2xl p-4 flex items-center justify-center gap-3 hover:border-violet-500/60 hover:bg-violet-500/5 transition-all group"
                        >
                            <Bot className="text-muted-foreground group-hover:text-violet-400 transition-colors" size={20} />
                            <span className="text-sm font-bold text-muted-foreground group-hover:text-foreground transition-colors">
                                Attach a Strategy to auto-trade with signals
                            </span>
                        </button>
                    )}

                    {/* Strategy Picker Modal */}
                    {showStrategyPicker && (
                        <div className="bg-card border border-violet-500/30 rounded-2xl p-6 space-y-4">
                            <div className="flex items-center justify-between">
                                <h3 className="text-lg font-bold text-foreground flex items-center gap-2">
                                    <Bot className="text-violet-400" size={20} />
                                    Attach Strategy
                                </h3>
                                <button onClick={() => setShowStrategyPicker(false)} className="text-muted-foreground hover:text-foreground">
                                    <X size={18} />
                                </button>
                            </div>

                            {/* Category filter */}
                            <div className="flex gap-2 flex-wrap">
                                {categories.map(cat => (
                                    <button
                                        key={cat}
                                        onClick={() => setStratCategory(cat)}
                                        className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                                            stratCategory === cat
                                                ? 'bg-violet-600 text-white'
                                                : 'bg-background border border-border text-muted-foreground hover:text-foreground'
                                        }`}
                                    >
                                        {cat === 'all' ? 'All' : cat}
                                    </button>
                                ))}
                            </div>

                            {/* Strategy list */}
                            <div className="max-h-[200px] overflow-y-auto space-y-1">
                                {filteredStrategies.map(s => (
                                    <button
                                        key={s.key}
                                        onClick={() => setSelectedStratKey(s.key)}
                                        className={`w-full text-left px-4 py-2.5 rounded-xl text-sm transition-all flex items-center justify-between ${
                                            selectedStratKey === s.key
                                                ? 'bg-violet-500/15 border border-violet-500/40 text-foreground'
                                                : 'hover:bg-accent/30 text-muted-foreground hover:text-foreground'
                                        }`}
                                    >
                                        <div>
                                            <span className="font-bold">{s.name}</span>
                                            <span className="ml-2 text-[10px] text-muted-foreground uppercase">{s.category}</span>
                                        </div>
                                        <span className="text-[10px] font-mono text-muted-foreground">{s.key}</span>
                                    </button>
                                ))}
                            </div>

                            {/* Configuration */}
                            {selectedStratKey && (
                                <div className="grid grid-cols-3 gap-4 pt-2 border-t border-border">
                                    <div>
                                        <label className="text-xs font-bold text-muted-foreground block mb-1">Symbol to Trade</label>
                                        <input
                                            value={stratSymbol}
                                            onChange={e => setStratSymbol(e.target.value.toUpperCase())}
                                            className="w-full bg-background border border-border rounded-xl py-2.5 px-4 text-sm font-mono text-emerald-400 outline-none focus:border-violet-500"
                                            placeholder="e.g. AAPL"
                                        />
                                    </div>
                                    <div>
                                        <label className="text-xs font-bold text-muted-foreground block mb-1">Data Interval</label>
                                        <select
                                            value={stratInterval}
                                            onChange={e => {
                                                setStratInterval(e.target.value);
                                                // Auto-suggest matching polling interval
                                                const suggested = INTERVAL_AUTORUN_SUGGESTION[e.target.value];
                                                if (suggested) setAutoRunInterval(suggested);
                                            }}
                                            className="w-full bg-background border border-border rounded-xl py-2.5 px-4 text-sm text-foreground outline-none focus:border-violet-500"
                                        >
                                            {DATA_INTERVALS.map(iv => (
                                                <option key={iv.value} value={iv.value}>{iv.label}</option>
                                            ))}
                                        </select>
                                    </div>
                                    <div>
                                        <label className="text-xs font-bold text-muted-foreground block mb-1">Shares per Signal</label>
                                        <input
                                            type="number"
                                            value={stratQty}
                                            onChange={e => setStratQty(Number(e.target.value))}
                                            className="w-full bg-background border border-border rounded-xl py-2.5 px-4 text-sm text-foreground outline-none focus:border-violet-500"
                                            min="1"
                                        />
                                    </div>
                                </div>
                            )}

                            <button
                                onClick={handleAttachStrategy}
                                disabled={!selectedStratKey || !stratSymbol.trim() || attachingStrategy}
                                className="flex items-center gap-2 px-6 py-2.5 bg-violet-600 hover:bg-violet-500 text-white rounded-xl text-sm font-bold transition-all disabled:opacity-50"
                            >
                                {attachingStrategy ? <Loader2 size={14} className="animate-spin" /> : <Bot size={14} />}
                                Attach Strategy
                            </button>
                        </div>
                    )}

                    {/* Portfolio Overview Cards */}
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                        {[
                            { label: 'Equity', value: formatCurrency(activePortfolio.equity ?? 0), color: 'text-foreground' },
                            { label: 'Cash', value: formatCurrency(activePortfolio.current_cash), color: 'text-blue-400' },
                            { label: 'Total Return', value: `${(activePortfolio.total_return_pct ?? 0) >= 0 ? '+' : ''}${(activePortfolio.total_return_pct ?? 0).toFixed(2)}%`, color: (activePortfolio.total_return_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400' },
                            { label: 'Win Rate', value: performance ? `${performance.win_rate}%` : '--', color: 'text-amber-400' },
                            { label: 'Sharpe', value: performance?.sharpe_ratio?.toFixed(2) ?? '--', color: 'text-violet-400' },
                        ].map((card, i) => (
                            <div key={i} className="bg-card border border-border rounded-2xl p-4">
                                <p className="text-[10px] font-black text-muted-foreground uppercase tracking-widest mb-1">{card.label}</p>
                                <p className={`text-xl font-bold ${card.color}`}>{card.value}</p>
                            </div>
                        ))}
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        {/* Left: Equity Chart + Trade History */}
                        <div className="lg:col-span-2 space-y-6">
                            {/* Equity Chart */}
                            <div className="bg-card border border-border rounded-2xl p-6">
                                <div className="flex items-center justify-between mb-4">
                                    <h3 className="text-xs font-black text-muted-foreground uppercase tracking-widest flex items-center gap-2">
                                        <TrendingUp size={14} className="text-emerald-400" /> Equity Curve
                                    </h3>
                                    <button
                                        onClick={handleRefreshEquity}
                                        disabled={refreshingEquity}
                                        className="flex items-center gap-1.5 px-3 py-1.5 text-muted-foreground hover:text-foreground text-[10px] font-bold rounded-lg hover:bg-accent/30 transition-all"
                                        title="Snapshot current equity with live prices"
                                    >
                                        <RefreshCw size={10} className={refreshingEquity ? 'animate-spin' : ''} />
                                        Snapshot
                                    </button>
                                </div>
                                <div className="h-[280px]">
                                    {equityChartData.length > 1 ? (
                                        <ResponsiveContainer width="100%" height="100%">
                                            <AreaChart data={equityChartData}>
                                                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                                                <XAxis dataKey="time" stroke="var(--muted-foreground)" fontSize={10} tickLine={false} />
                                                <YAxis stroke="var(--muted-foreground)" fontSize={10} tickLine={false} tickFormatter={v => `$${(v / 1000).toFixed(0)}k`} />
                                                <Tooltip
                                                    contentStyle={{ backgroundColor: 'var(--card)', border: '1px solid var(--border)', borderRadius: '12px' }}
                                                    formatter={(val: number | undefined) => {
                                                        if (val === undefined) return ["$0.00", "Equity"];
                                                        return [formatCurrency(val), "Equity"] as [string, string];
                                                    }}
                                                />
                                                <Area type="monotone" dataKey="equity" stroke="#10b981" fill="#10b981" fillOpacity={0.1} strokeWidth={2} />
                                            </AreaChart>
                                        </ResponsiveContainer>
                                    ) : (
                                        <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
                                            Place trades or click Snapshot to track equity over time
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Trade History */}
                            <div className="bg-card border border-border rounded-2xl overflow-hidden">
                                <div className="px-6 py-4 border-b border-border">
                                    <h3 className="text-xs font-black text-muted-foreground uppercase tracking-widest flex items-center gap-2">
                                        <ShoppingCart size={14} className="text-blue-400" /> Trade History
                                    </h3>
                                </div>
                                <div className="overflow-x-auto max-h-[300px] overflow-y-auto">
                                    <table className="w-full text-sm">
                                        <thead className="sticky top-0 bg-card">
                                            <tr className="border-b border-border">
                                                <th className="px-4 py-3 text-left font-bold text-muted-foreground text-[10px] uppercase">Time</th>
                                                <th className="px-4 py-3 text-left font-bold text-muted-foreground text-[10px] uppercase">Symbol</th>
                                                <th className="px-4 py-3 text-center font-bold text-muted-foreground text-[10px] uppercase">Side</th>
                                                <th className="px-4 py-3 text-center font-bold text-muted-foreground text-[10px] uppercase">Source</th>
                                                <th className="px-4 py-3 text-right font-bold text-muted-foreground text-[10px] uppercase">Qty</th>
                                                <th className="px-4 py-3 text-right font-bold text-muted-foreground text-[10px] uppercase">Price</th>
                                                <th className="px-4 py-3 text-right font-bold text-muted-foreground text-[10px] uppercase">P&L</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {trades.length === 0 ? (
                                                <tr>
                                                    <td colSpan={7} className="px-4 py-8 text-center text-muted-foreground">No trades yet</td>
                                                </tr>
                                            ) : (
                                                trades.map(t => (
                                                    <tr key={t.id} className="border-b border-border/50 hover:bg-accent/30 transition-colors">
                                                        <td className="px-4 py-2.5 text-xs text-muted-foreground">
                                                            {t.executed_at ? new Date(t.executed_at).toLocaleString() : '--'}
                                                        </td>
                                                        <td className="px-4 py-2.5 font-bold text-foreground">{t.symbol}</td>
                                                        <td className="px-4 py-2.5 text-center">
                                                            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold ${
                                                                t.side === 'buy'
                                                                    ? 'bg-emerald-500/15 text-emerald-400'
                                                                    : 'bg-red-500/15 text-red-400'
                                                            }`}>
                                                                {t.side === 'buy' ? <ArrowUpRight size={10} /> : <ArrowDownRight size={10} />}
                                                                {t.side.toUpperCase()}
                                                            </span>
                                                        </td>
                                                        <td className="px-4 py-2.5 text-center">
                                                            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold ${
                                                                t.source === 'strategy'
                                                                    ? 'bg-violet-500/15 text-violet-400'
                                                                    : 'bg-slate-500/15 text-muted-foreground'
                                                            }`}>
                                                                {t.source === 'strategy' ? <Bot size={10} /> : null}
                                                                {t.source === 'strategy' ? 'Strategy' : 'Manual'}
                                                            </span>
                                                        </td>
                                                        <td className="px-4 py-2.5 text-right font-mono text-foreground">{t.quantity}</td>
                                                        <td className="px-4 py-2.5 text-right font-mono text-foreground">{formatCurrency(t.price)}</td>
                                                        <td className={`px-4 py-2.5 text-right font-mono font-bold ${
                                                            t.realized_pnl === null ? 'text-muted-foreground' :
                                                            t.realized_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'
                                                        }`}>
                                                            {t.realized_pnl !== null ? formatCurrency(t.realized_pnl) : '--'}
                                                        </td>
                                                    </tr>
                                                ))
                                            )}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>

                        {/* Right: Trade Panel + Positions */}
                        <div className="space-y-6">
                            {/* Quick Trade Panel */}
                            <div className="bg-card border border-border rounded-2xl p-6 space-y-4">
                                <h3 className="text-xs font-black text-muted-foreground uppercase tracking-widest">Manual Trade</h3>

                                <input
                                    value={tradeSymbol}
                                    onChange={e => setTradeSymbol(e.target.value.toUpperCase())}
                                    className="w-full bg-background border border-border rounded-xl py-2.5 px-4 text-sm font-mono text-emerald-400 outline-none focus:border-emerald-500"
                                    placeholder="Symbol (e.g. AAPL)"
                                />

                                <input
                                    type="number"
                                    value={tradeQty}
                                    onChange={e => setTradeQty(e.target.value)}
                                    className="w-full bg-background border border-border rounded-xl py-2.5 px-4 text-sm text-foreground outline-none focus:border-emerald-500"
                                    placeholder="Quantity"
                                    min="1"
                                />

                                <div className="grid grid-cols-2 gap-2">
                                    <button
                                        onClick={() => { setTradeSide('buy'); handlePlaceTrade(); }}
                                        disabled={tradingLoading || !tradeSymbol || !tradeQty}
                                        className="flex items-center justify-center gap-2 py-3 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl font-bold text-sm transition-all disabled:opacity-50"
                                    >
                                        {tradingLoading && tradeSide === 'buy' ? <Loader2 size={14} className="animate-spin" /> : <ArrowUpRight size={14} />}
                                        BUY
                                    </button>
                                    <button
                                        onClick={() => { setTradeSide('sell'); handlePlaceTrade(); }}
                                        disabled={tradingLoading || !tradeSymbol || !tradeQty}
                                        className="flex items-center justify-center gap-2 py-3 bg-red-600 hover:bg-red-500 text-white rounded-xl font-bold text-sm transition-all disabled:opacity-50"
                                    >
                                        {tradingLoading && tradeSide === 'sell' ? <Loader2 size={14} className="animate-spin" /> : <ArrowDownRight size={14} />}
                                        SELL
                                    </button>
                                </div>

                                {tradeError && (
                                    <p className="text-red-400 text-xs bg-red-500/10 rounded-lg p-2">{tradeError}</p>
                                )}
                            </div>

                            {/* Positions */}
                            <div className="bg-card border border-border rounded-2xl overflow-hidden">
                                <div className="px-6 py-4 border-b border-border flex items-center justify-between">
                                    <h3 className="text-xs font-black text-muted-foreground uppercase tracking-widest">Positions</h3>
                                    <span className="text-[10px] font-bold text-muted-foreground">{activePortfolio.positions.length} open</span>
                                </div>
                                <div className="divide-y divide-border/50">
                                    {activePortfolio.positions.length === 0 ? (
                                        <div className="p-6 text-center text-muted-foreground text-sm">No open positions</div>
                                    ) : (
                                        activePortfolio.positions.map(pos => (
                                            <div key={pos.id} className="px-4 py-3 flex items-center justify-between hover:bg-accent/20 transition-colors">
                                                <div>
                                                    <p className="font-bold text-foreground text-sm">{pos.symbol}</p>
                                                    <p className="text-[10px] text-muted-foreground">
                                                        {pos.quantity} shares @ {formatCurrency(pos.avg_entry_price)}
                                                    </p>
                                                </div>
                                                <div className="text-right">
                                                    <p className="text-sm font-mono text-foreground">
                                                        {pos.current_price ? formatCurrency(pos.current_price) : '--'}
                                                    </p>
                                                    <p className={`text-[10px] font-bold ${
                                                        (pos.unrealized_pnl ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
                                                    }`}>
                                                        {pos.unrealized_pnl !== null ? (
                                                            <>
                                                                {pos.unrealized_pnl >= 0 ? '+' : ''}{formatCurrency(pos.unrealized_pnl)}
                                                                {' '}({pos.unrealized_pnl_pct?.toFixed(1)}%)
                                                            </>
                                                        ) : '--'}
                                                    </p>
                                                </div>
                                            </div>
                                        ))
                                    )}
                                </div>
                            </div>

                            {/* Performance Cards */}
                            {performance && (
                                <div className="bg-card border border-border rounded-2xl p-4 space-y-3">
                                    <h3 className="text-xs font-black text-muted-foreground uppercase tracking-widest">Performance</h3>
                                    <div className="grid grid-cols-2 gap-3 text-xs">
                                        {[
                                            { label: 'Total Trades', value: performance.total_trades },
                                            { label: 'Winning', value: performance.winning_trades },
                                            { label: 'Losing', value: performance.losing_trades },
                                            { label: 'Avg Win', value: formatCurrency(performance.avg_win) },
                                            { label: 'Avg Loss', value: formatCurrency(performance.avg_loss) },
                                            { label: 'Max DD', value: `${performance.max_drawdown.toFixed(1)}%` },
                                        ].map((item, i) => (
                                            <div key={i} className="flex justify-between">
                                                <span className="text-muted-foreground">{item.label}</span>
                                                <span className="font-bold text-foreground">{item.value}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Delete button */}
                            <button
                                onClick={() => activeId && handleDelete(activeId)}
                                className="w-full flex items-center justify-center gap-2 py-2.5 text-red-400 hover:bg-red-500/10 rounded-xl text-xs font-bold transition-all border border-red-500/20"
                            >
                                <Trash2 size={14} /> Delete Portfolio
                            </button>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
};

export default PaperTradingDashboard;
