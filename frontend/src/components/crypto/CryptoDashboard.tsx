/* eslint-disable @typescript-eslint/no-explicit-any */

'use client'
import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
    Activity,
    ArrowDown,
    ArrowUp,
    BarChart3,
    Bitcoin,
    Brain,
    Briefcase,
    Globe,
    Grid3X3,
    Layers,
    Loader2,
    Minus,
    Plus,
    RefreshCw,
    Search,
    Shield,
    Target,
    Trash2,
    TrendingDown,
    TrendingUp,
    X,
    ExternalLink,
    Github,
    Twitter,
    Users,
    Code2,
    Calendar,
    Tag,
    Zap,
} from 'lucide-react';
import { cryptoApi } from '@/utils/api';

// ─── Shared interfaces ───────────────────────────────────────────
interface MarketOverview {
    total_market_cap_usd: number;
    total_volume_24h_usd: number;
    bitcoin_dominance: number;
    ethereum_dominance: number;
    active_cryptocurrencies: number;
    market_cap_change_24h_pct: number;
}

interface Coin {
    id: string;
    symbol: string;
    name: string;
    image: string;
    price: number;
    market_cap: number;
    market_cap_rank: number;
    volume_24h: number;
    change_1h: number;
    change_24h: number;
    change_7d: number;
    change_30d: number;
    circulating_supply: number;
    total_supply: number;
    ath: number;
    ath_change_pct: number;
    sparkline: number[];
}

// ─── Helpers ─────────────────────────────────────────────────────
const formatNumber = (n: number, decimals = 2): string => {
    if (!Number.isFinite(n)) return '0';
    if (Math.abs(n) >= 1e12) return `$${(n / 1e12).toFixed(decimals)}T`;
    if (Math.abs(n) >= 1e9) return `$${(n / 1e9).toFixed(decimals)}B`;
    if (Math.abs(n) >= 1e6) return `$${(n / 1e6).toFixed(decimals)}M`;
    if (Math.abs(n) >= 1e3) return `$${(n / 1e3).toFixed(decimals)}K`;
    return `$${n.toFixed(decimals)}`;
};

const formatPrice = (price: number): string => {
    if (!Number.isFinite(price)) return '$0';
    if (price >= 1) return `$${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    if (price >= 0.01) return `$${price.toFixed(4)}`;
    return `$${price.toFixed(8)}`;
};

const ChangeCell = ({ value }: { value: number }) => {
    const v = Number.isFinite(value) ? value : 0;
    const isUp = v >= 0;
    return (
        <span className={`text-xs font-bold flex items-center gap-0.5 ${isUp ? 'text-emerald-400' : 'text-red-400'}`}>
            {isUp ? <ArrowUp size={10} /> : <ArrowDown size={10} />}
            {Math.abs(v).toFixed(2)}%
        </span>
    );
};

const MiniSparkline = ({ data, positive }: { data: number[]; positive: boolean }) => {
    if (!data || data.length < 2) return null;
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const h = 28;
    const w = 80;
    const step = w / (data.length - 1);
    const points = data.map((v, i) => `${i * step},${h - ((v - min) / range) * h}`).join(' ');

    return (
        <svg width={w} height={h} className="shrink-0">
            <polyline
                points={points}
                fill="none"
                stroke={positive ? '#34d399' : '#f87171'}
                strokeWidth="1.5"
            />
        </svg>
    );
};

// ─── Tab config ──────────────────────────────────────────────────
type TabKey = 'overview' | 'signals' | 'sectors' | 'predictions' | 'volatility' | 'correlation' | 'btc-dominance' | 'portfolio';

const TABS: { key: TabKey; label: string; icon: React.ElementType }[] = [
    { key: 'overview', label: 'Overview', icon: Globe },
    { key: 'signals', label: 'Signals', icon: Zap },
    { key: 'sectors', label: 'Sectors', icon: Layers },
    { key: 'predictions', label: 'Predictions', icon: Brain },
    { key: 'volatility', label: 'Volatility', icon: Activity },
    { key: 'correlation', label: 'Correlation', icon: Grid3X3 },
    { key: 'btc-dominance', label: 'BTC Dominance', icon: Bitcoin },
    { key: 'portfolio', label: 'Portfolio', icon: Briefcase },
];

// ─── Reusable card wrapper ───────────────────────────────────────
const Card = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
    <div className={`bg-slate-800/40 border border-slate-700/40 rounded-2xl p-5 ${className}`}>
        {children}
    </div>
);

const SectionLoader = () => (
    <div className="flex justify-center py-20">
        <Loader2 size={32} className="animate-spin text-amber-400" />
    </div>
);

const ErrorMsg = ({ msg }: { msg: string }) => (
    <div className="flex justify-center py-16 text-sm text-red-400">{msg}</div>
);

// ─── Signal badge helper ─────────────────────────────────────────
const signalColor = (signal: string): string => {
    const s = (signal || '').toUpperCase();
    if (s === 'STRONG_BUY' || s === 'BULLISH') return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    if (s === 'BUY') return 'bg-emerald-500/10 text-emerald-300 border-emerald-500/20';
    if (s === 'STRONG_SELL' || s === 'BEARISH') return 'bg-red-500/20 text-red-400 border-red-500/30';
    if (s === 'SELL') return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
    return 'bg-slate-700/40 text-slate-400 border-slate-600/40';
};

const SignalBadge = ({ signal }: { signal: string }) => (
    <span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded-md border ${signalColor(signal)}`}>
        {(signal || 'NEUTRAL').replace('_', ' ')}
    </span>
);

// ─── Fear & Greed Gauge ──────────────────────────────────────────
const FearGreedGauge = ({ value, label, size = 160 }: { value: number; label: string; size?: number }) => {
    const v = Math.max(0, Math.min(100, value || 0));
    const radius = size / 2 - 16;
    const cx = size / 2;
    const cy = size / 2 + 10;
    // Semi-circle from PI to 0 (left to right)
    const startAngle = Math.PI;
    const endAngle = 0;
    const sweepAngle = startAngle - (startAngle - endAngle) * (v / 100);

    const needleX = cx + radius * Math.cos(sweepAngle);
    const needleY = cy - radius * Math.sin(sweepAngle);

    const gaugeColor = v <= 25 ? '#ef4444' : v <= 45 ? '#f97316' : v <= 55 ? '#eab308' : v <= 75 ? '#84cc16' : '#22c55e';

    // Arc path for background
    const arcPath = (startA: number, endA: number, r: number) => {
        const x1 = cx + r * Math.cos(startA);
        const y1 = cy - r * Math.sin(startA);
        const x2 = cx + r * Math.cos(endA);
        const y2 = cy - r * Math.sin(endA);
        return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`;
    };

    return (
        <div className="flex flex-col items-center">
            <svg width={size} height={size / 2 + 30} viewBox={`0 0 ${size} ${size / 2 + 30}`}>
                {/* Background arc */}
                <path d={arcPath(Math.PI, 0, radius)} fill="none" stroke="#334155" strokeWidth="12" strokeLinecap="round" />
                {/* Value arc */}
                {v > 0 && (
                    <path d={arcPath(Math.PI, sweepAngle, radius)} fill="none" stroke={gaugeColor} strokeWidth="12" strokeLinecap="round" />
                )}
                {/* Needle */}
                <line x1={cx} y1={cy} x2={needleX} y2={needleY} stroke="#f8fafc" strokeWidth="2" strokeLinecap="round" />
                <circle cx={cx} cy={cy} r="4" fill="#f8fafc" />
                {/* Value text */}
                <text x={cx} y={cy - radius / 3} textAnchor="middle" className="text-2xl font-bold" fill={gaugeColor} fontSize="24">
                    {v}
                </text>
            </svg>
            <p className="text-sm font-bold mt-1" style={{ color: gaugeColor }}>{label}</p>
        </div>
    );
};

// ─── Score Bar (horizontal, -100 to 100 or 0 to 100) ────────────
const ScoreBar = ({ value, min = -100, max = 100 }: { value: number; min?: number; max?: number }) => {
    const range = max - min;
    const pct = Math.max(0, Math.min(100, ((value - min) / range) * 100));
    const color = value > 30 ? 'bg-emerald-400' : value > 0 ? 'bg-emerald-400/60' : value > -30 ? 'bg-orange-400' : 'bg-red-400';
    return (
        <div className="w-full h-2 bg-slate-700/60 rounded-full overflow-hidden">
            <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
        </div>
    );
};

// ─── RSI color ───────────────────────────────────────────────────
const rsiColor = (rsi: number): string => {
    if (rsi < 30) return 'text-emerald-400';
    if (rsi > 70) return 'text-red-400';
    return 'text-yellow-400';
};

// ─── Risk Level badge ────────────────────────────────────────────
const riskColor = (level: string): string => {
    const l = (level || '').toUpperCase();
    if (l === 'LOW') return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    if (l === 'MEDIUM') return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
    if (l === 'HIGH') return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
    if (l === 'EXTREME') return 'bg-red-500/20 text-red-400 border-red-500/30';
    return 'bg-slate-700/40 text-slate-400 border-slate-600/40';
};

// ─── Correlation cell color ──────────────────────────────────────
const corrColor = (v: number): string => {
    if (v >= 0.7) return 'bg-emerald-500/60';
    if (v >= 0.3) return 'bg-emerald-500/30';
    if (v > -0.3) return 'bg-slate-700/40';
    if (v > -0.7) return 'bg-red-500/30';
    return 'bg-red-500/60';
};

// =================================================================
// MAIN DASHBOARD COMPONENT
// =================================================================
const CryptoDashboard = () => {
    const [activeTab, setActiveTab] = useState<TabKey>('overview');

    // ─── Overview state ──────────────────────────────────────────
    const [overview, setOverview] = useState<MarketOverview | null>(null);
    const [coins, setCoins] = useState<Coin[]>([]);
    const [trending, setTrending] = useState<any[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');
    const [sortBy, setSortBy] = useState<'market_cap_rank' | 'change_24h' | 'volume_24h'>('market_cap_rank');
    const [overviewFearGreed, setOverviewFearGreed] = useState<any>(null);

    // ─── Tab data cache ──────────────────────────────────────────
    const cache = useRef<Record<string, any>>({});
    const [tabData, setTabData] = useState<Record<string, any>>({});
    const [tabLoading, setTabLoading] = useState<Record<string, boolean>>({});
    const [tabError, setTabError] = useState<Record<string, string>>({});

    // ─── Portfolio state ─────────────────────────────────────────
    const [holdings, setHoldings] = useState<{ coin_id: string; amount_usd: number }[]>([
        { coin_id: 'bitcoin', amount_usd: 5000 },
        { coin_id: 'ethereum', amount_usd: 3000 },
        { coin_id: 'solana', amount_usd: 2000 },
    ]);
    const [newCoinId, setNewCoinId] = useState('');
    const [newAmount, setNewAmount] = useState('');
    const [portfolioResult, setPortfolioResult] = useState<any>(null);
    const [portfolioLoading, setPortfolioLoading] = useState(false);
    const [portfolioError, setPortfolioError] = useState('');

    // ─── Coin detail modal ──────────────────────────────────────
    const [selectedCoin, setSelectedCoin] = useState<any>(null);
    const [coinDetailLoading, setCoinDetailLoading] = useState(false);

    const openCoinDetail = useCallback(async (coinId: string) => {
        setCoinDetailLoading(true);
        setSelectedCoin({ id: coinId, loading: true });
        try {
            const data = await cryptoApi.coinDetail(coinId);
            setSelectedCoin(data);
        } catch (err) {
            console.error('Failed to fetch coin detail:', err);
            setSelectedCoin({ id: coinId, error: 'Failed to load details' });
        } finally {
            setCoinDetailLoading(false);
        }
    }, []);

    // ─── Correlation period ──────────────────────────────────────
    const [corrPeriod, setCorrPeriod] = useState('30d');

    // ─── Overview fetch ──────────────────────────────────────────
    const fetchOverview = useCallback(async () => {
        setIsLoading(true);
        try {
            const [overviewData, coinsData, trendingData, fgData] = await Promise.allSettled([
                cryptoApi.marketOverview(),
                cryptoApi.topCoins(100),
                cryptoApi.trending(),
                cryptoApi.fearGreed(),
            ]);
            if (overviewData.status === 'fulfilled') setOverview(overviewData.value as any);
            if (coinsData.status === 'fulfilled') setCoins((coinsData.value as any)?.coins || []);
            if (trendingData.status === 'fulfilled') setTrending((trendingData.value as any)?.coins || []);
            if (fgData.status === 'fulfilled') setOverviewFearGreed(fgData.value as any);
        } catch (err) {
            console.error('Failed to fetch crypto data:', err);
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchOverview();
    }, [fetchOverview]);

    // ─── Generic tab data fetcher with cache ─────────────────────
    const fetchTabData = useCallback(async (tab: TabKey) => {
        if (cache.current[tab]) {
            setTabData(prev => ({ ...prev, [tab]: cache.current[tab] }));
            return;
        }
        setTabLoading(prev => ({ ...prev, [tab]: true }));
        setTabError(prev => ({ ...prev, [tab]: '' }));
        try {
            let data: any;
            switch (tab) {
                case 'signals': {
                    const [m, t] = await Promise.allSettled([cryptoApi.momentum(), cryptoApi.technicalSignals()]);
                    data = {
                        momentum: m.status === 'fulfilled' ? m.value : [],
                        technical: t.status === 'fulfilled' ? t.value : [],
                    };
                    break;
                }
                case 'sectors':
                    data = await cryptoApi.sectors();
                    break;
                case 'predictions': {
                    const [p, fg] = await Promise.allSettled([cryptoApi.predictions(), cryptoApi.fearGreed()]);
                    data = {
                        predictions: p.status === 'fulfilled' ? p.value : [],
                        fearGreed: fg.status === 'fulfilled' ? fg.value : null,
                    };
                    break;
                }
                case 'volatility':
                    data = await cryptoApi.volatility();
                    break;
                case 'correlation':
                    data = await cryptoApi.correlation(undefined, corrPeriod);
                    break;
                case 'btc-dominance':
                    data = await cryptoApi.btcDominance();
                    break;
                default:
                    return;
            }
            cache.current[tab] = data;
            setTabData(prev => ({ ...prev, [tab]: data }));
        } catch (err: any) {
            setTabError(prev => ({ ...prev, [tab]: err?.message || 'Failed to load data' }));
        } finally {
            setTabLoading(prev => ({ ...prev, [tab]: false }));
        }
    }, [corrPeriod]);

    // Fetch tab data when tab changes
    useEffect(() => {
        if (activeTab !== 'overview' && activeTab !== 'portfolio') {
            fetchTabData(activeTab);
        }
    }, [activeTab, fetchTabData]);

    // Refetch correlation when period changes
    useEffect(() => {
        if (activeTab === 'correlation') {
            delete cache.current['correlation'];
            fetchTabData('correlation');
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [corrPeriod]);

    // ─── Filtered & sorted coins ─────────────────────────────────
    const filteredCoins = coins
        .filter(c => {
            if (!searchQuery) return true;
            const q = searchQuery.toLowerCase();
            return c.name.toLowerCase().includes(q) || c.symbol.toLowerCase().includes(q);
        })
        .sort((a, b) => {
            if (sortBy === 'change_24h') return (b.change_24h || 0) - (a.change_24h || 0);
            if (sortBy === 'volume_24h') return (b.volume_24h || 0) - (a.volume_24h || 0);
            return (a.market_cap_rank || 999) - (b.market_cap_rank || 999);
        });

    // ─── Portfolio helpers ────────────────────────────────────────
    const addHolding = () => {
        if (!newCoinId.trim() || !newAmount.trim()) return;
        const amt = parseFloat(newAmount);
        if (!Number.isFinite(amt) || amt <= 0) return;
        setHoldings(prev => [...prev, { coin_id: newCoinId.trim().toLowerCase(), amount_usd: amt }]);
        setNewCoinId('');
        setNewAmount('');
    };

    const removeHolding = (idx: number) => {
        setHoldings(prev => prev.filter((_, i) => i !== idx));
    };

    const optimizePortfolio = async () => {
        if (holdings.length === 0) return;
        setPortfolioLoading(true);
        setPortfolioError('');
        try {
            const result = await cryptoApi.portfolioOptimize(holdings);
            setPortfolioResult(result);
        } catch (err: any) {
            setPortfolioError(err?.message || 'Failed to optimize portfolio');
        } finally {
            setPortfolioLoading(false);
        }
    };

    // =============================================================
    // RENDER
    // =============================================================
    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center shadow-lg shadow-amber-500/20">
                        <Bitcoin size={24} className="text-white" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-slate-100">Crypto Markets</h1>
                        <p className="text-sm text-slate-500">Real-time cryptocurrency data &amp; analytics</p>
                    </div>
                </div>
                <button onClick={fetchOverview} className="p-2 hover:bg-slate-800 rounded-xl transition-colors" title="Refresh">
                    <RefreshCw size={18} className={`text-slate-400 ${isLoading ? 'animate-spin' : ''}`} />
                </button>
            </div>

            {/* Tab Bar */}
            <div className="flex gap-1 overflow-x-auto pb-1">
                {TABS.map(({ key, label, icon: Icon }) => (
                    <button
                        key={key}
                        onClick={() => setActiveTab(key)}
                        className={`px-4 py-2.5 rounded-xl text-xs font-bold flex items-center gap-1.5 shrink-0 transition-all ${
                            activeTab === key
                                ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                                : 'bg-slate-800/40 text-slate-400 border border-slate-700/40 hover:border-slate-600 hover:text-slate-300'
                        }`}
                    >
                        <Icon size={14} />
                        {label}
                    </button>
                ))}
            </div>

            {/* ═══════════════ TAB: OVERVIEW ═══════════════════════ */}
            {activeTab === 'overview' && (
                <div className="space-y-6">
                    {/* Fear & Greed mini + Market Overview */}
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                        {overviewFearGreed && (
                            <Card className="flex flex-col items-center justify-center col-span-1">
                                <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-2">Fear &amp; Greed</p>
                                <FearGreedGauge value={overviewFearGreed.value} label={overviewFearGreed.label || ''} size={100} />
                            </Card>
                        )}
                        {overview && (
                            <>
                                <Card>
                                    <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Total Market Cap</p>
                                    <p className="text-lg font-bold text-slate-100 mt-1">{formatNumber(overview.total_market_cap_usd)}</p>
                                    <ChangeCell value={overview.market_cap_change_24h_pct} />
                                </Card>
                                <Card>
                                    <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">24h Volume</p>
                                    <p className="text-lg font-bold text-slate-100 mt-1">{formatNumber(overview.total_volume_24h_usd)}</p>
                                </Card>
                                <Card>
                                    <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">BTC Dominance</p>
                                    <p className="text-lg font-bold text-amber-400 mt-1">{overview.bitcoin_dominance?.toFixed(1)}%</p>
                                </Card>
                                <Card>
                                    <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">ETH Dominance</p>
                                    <p className="text-lg font-bold text-violet-400 mt-1">{overview.ethereum_dominance?.toFixed(1)}%</p>
                                </Card>
                            </>
                        )}
                    </div>

                    {/* Trending */}
                    {trending.length > 0 && (
                        <div className="bg-slate-800/30 border border-slate-700/30 rounded-2xl p-5">
                            <div className="flex items-center gap-2 mb-3">
                                <TrendingUp size={16} className="text-amber-400" />
                                <h3 className="text-sm font-bold text-slate-300 uppercase tracking-widest">Trending</h3>
                            </div>
                            <div className="flex gap-3 overflow-x-auto pb-1">
                                {trending.map((coin: any) => (
                                    <div key={coin.id} className="flex items-center gap-2 px-3 py-2 bg-slate-800/60 border border-slate-700/40 rounded-xl shrink-0">
                                        {coin.thumb && <img src={coin.thumb} alt="" className="w-5 h-5 rounded-full" />}
                                        <span className="text-xs font-bold text-slate-200">{coin.symbol}</span>
                                        <span className="text-[10px] text-slate-500">#{coin.market_cap_rank}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Search and Sort */}
                    <div className="flex items-center gap-3">
                        <div className="relative flex-1 max-w-sm">
                            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                            <input
                                type="text"
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                placeholder="Search coins..."
                                className="w-full pl-10 pr-4 py-2.5 bg-slate-800/40 border border-slate-700/40 rounded-xl text-sm text-slate-200 placeholder-slate-500 focus:border-amber-500/50 outline-none"
                            />
                        </div>
                        <div className="flex gap-1">
                            {([
                                { key: 'market_cap_rank' as const, label: 'Rank', icon: Globe },
                                { key: 'change_24h' as const, label: '24h %', icon: TrendingUp },
                                { key: 'volume_24h' as const, label: 'Volume', icon: BarChart3 },
                            ]).map(({ key, label, icon: Icon }) => (
                                <button
                                    key={key}
                                    onClick={() => setSortBy(key)}
                                    className={`px-3 py-2 rounded-lg text-xs font-bold flex items-center gap-1 transition-all ${
                                        sortBy === key
                                            ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                                            : 'bg-slate-800/40 text-slate-400 border border-slate-700/40 hover:border-slate-600'
                                    }`}
                                >
                                    <Icon size={12} />
                                    {label}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Coins Table */}
                    {isLoading ? (
                        <SectionLoader />
                    ) : (
                        <div className="bg-slate-800/20 border border-slate-700/30 rounded-2xl overflow-hidden">
                            <div className="overflow-x-auto">
                                <table className="w-full">
                                    <thead>
                                        <tr className="border-b border-slate-700/30 text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                                            <th className="text-left px-4 py-3">#</th>
                                            <th className="text-left px-4 py-3">Coin</th>
                                            <th className="text-right px-4 py-3">Price</th>
                                            <th className="text-right px-4 py-3">1h</th>
                                            <th className="text-right px-4 py-3">24h</th>
                                            <th className="text-right px-4 py-3">7d</th>
                                            <th className="text-right px-4 py-3">Market Cap</th>
                                            <th className="text-right px-4 py-3">Volume (24h)</th>
                                            <th className="text-right px-4 py-3">7d Chart</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {filteredCoins.map((coin) => (
                                            <tr key={coin.id} className="border-b border-slate-800/30 hover:bg-slate-800/20 transition-colors cursor-pointer" onClick={() => openCoinDetail(coin.id)}>
                                                <td className="px-4 py-3 text-xs text-slate-500 font-bold">{coin.market_cap_rank}</td>
                                                <td className="px-4 py-3">
                                                    <div className="flex items-center gap-2">
                                                        {coin.image && <img src={coin.image} alt="" className="w-6 h-6 rounded-full" />}
                                                        <div>
                                                            <p className="text-sm font-bold text-slate-200 group-hover:text-amber-400">{coin.name}</p>
                                                            <p className="text-[10px] text-slate-500 uppercase">{coin.symbol}</p>
                                                        </div>
                                                    </div>
                                                </td>
                                                <td className="px-4 py-3 text-right text-sm font-bold text-slate-200">{formatPrice(coin.price)}</td>
                                                <td className="px-4 py-3 text-right"><ChangeCell value={coin.change_1h} /></td>
                                                <td className="px-4 py-3 text-right"><ChangeCell value={coin.change_24h} /></td>
                                                <td className="px-4 py-3 text-right"><ChangeCell value={coin.change_7d} /></td>
                                                <td className="px-4 py-3 text-right text-xs text-slate-400">{formatNumber(coin.market_cap, 1)}</td>
                                                <td className="px-4 py-3 text-right text-xs text-slate-400">{formatNumber(coin.volume_24h, 1)}</td>
                                                <td className="px-4 py-3 text-right">
                                                    <MiniSparkline data={coin.sparkline} positive={(coin.change_7d || 0) >= 0} />
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* ═══════════════ TAB: SIGNALS ════════════════════════ */}
            {activeTab === 'signals' && (
                <div className="space-y-6">
                    {tabLoading['signals'] ? <SectionLoader /> : tabError['signals'] ? <ErrorMsg msg={tabError['signals']} /> : (
                        <>
                            {/* Momentum Scanner */}
                            <div>
                                <h2 className="text-sm font-bold text-slate-300 uppercase tracking-widest mb-4 flex items-center gap-2">
                                    <Target size={16} className="text-amber-400" /> Momentum Scanner
                                </h2>
                                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                                    {(tabData['signals']?.momentum || []).map((coin: any) => (
                                        <Card key={coin.coin_id || coin.symbol}>
                                            <div className="flex items-center justify-between mb-3">
                                                <div className="flex items-center gap-2">
                                                    <div className="w-8 h-8 rounded-full bg-slate-700/60 flex items-center justify-center text-xs font-bold text-slate-300 uppercase">
                                                        {(coin.symbol || '?').slice(0, 3)}
                                                    </div>
                                                    <div>
                                                        <p className="text-sm font-bold text-slate-200">{coin.name || coin.coin_id}</p>
                                                        <p className="text-[10px] text-slate-500 uppercase">{coin.symbol}</p>
                                                    </div>
                                                </div>
                                                <SignalBadge signal={coin.signal} />
                                            </div>
                                            <div className="flex items-center justify-between mb-2">
                                                <span className="text-[10px] text-slate-500 uppercase font-bold">RSI</span>
                                                <span className={`text-sm font-bold ${rsiColor(coin.rsi || 50)}`}>
                                                    {(coin.rsi || 0).toFixed(1)}
                                                </span>
                                            </div>
                                            <div>
                                                <div className="flex items-center justify-between mb-1">
                                                    <span className="text-[10px] text-slate-500 uppercase font-bold">Score</span>
                                                    <span className="text-[10px] text-slate-400">{(coin.score || 0).toFixed(0)}</span>
                                                </div>
                                                <ScoreBar value={coin.score || 0} />
                                            </div>
                                        </Card>
                                    ))}
                                    {(tabData['signals']?.momentum || []).length === 0 && (
                                        <p className="text-sm text-slate-500 col-span-full text-center py-8">No momentum data available</p>
                                    )}
                                </div>
                            </div>

                            {/* Technical Signals Grid */}
                            <div>
                                <h2 className="text-sm font-bold text-slate-300 uppercase tracking-widest mb-4 flex items-center gap-2">
                                    <Zap size={16} className="text-amber-400" /> Technical Signals
                                </h2>
                                <div className="bg-slate-800/20 border border-slate-700/30 rounded-2xl overflow-hidden">
                                    <div className="overflow-x-auto">
                                        <table className="w-full">
                                            <thead>
                                                <tr className="border-b border-slate-700/30 text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                                                    <th className="text-left px-4 py-3">Coin</th>
                                                    <th className="text-center px-4 py-3">RSI</th>
                                                    <th className="text-center px-4 py-3">SMA Cross</th>
                                                    <th className="text-center px-4 py-3">MACD</th>
                                                    <th className="text-center px-4 py-3">Bollinger</th>
                                                    <th className="text-center px-4 py-3">Overall</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {(tabData['signals']?.technical || []).map((coin: any) => {
                                                    const indicators = coin.indicators || {};
                                                    const signals = [
                                                        indicators.rsi?.signal,
                                                        indicators.sma_cross?.signal,
                                                        indicators.macd?.signal,
                                                        indicators.bollinger?.signal,
                                                    ];
                                                    const buyCount = signals.filter((s: string) => s === 'BUY' || s === 'STRONG_BUY').length;
                                                    const sellCount = signals.filter((s: string) => s === 'SELL' || s === 'STRONG_SELL').length;
                                                    return (
                                                        <tr key={coin.coin_id || coin.symbol} className="border-b border-slate-800/30 hover:bg-slate-800/20">
                                                            <td className="px-4 py-3">
                                                                <span className="text-sm font-bold text-slate-200">{coin.name || coin.coin_id}</span>
                                                                <span className="text-[10px] text-slate-500 ml-2 uppercase">{coin.symbol}</span>
                                                            </td>
                                                            <td className="px-4 py-3 text-center"><SignalBadge signal={indicators.rsi?.signal || 'NEUTRAL'} /></td>
                                                            <td className="px-4 py-3 text-center"><SignalBadge signal={indicators.sma_cross?.signal || 'NEUTRAL'} /></td>
                                                            <td className="px-4 py-3 text-center"><SignalBadge signal={indicators.macd?.signal || 'NEUTRAL'} /></td>
                                                            <td className="px-4 py-3 text-center"><SignalBadge signal={indicators.bollinger?.signal || 'NEUTRAL'} /></td>
                                                            <td className="px-4 py-3 text-center">
                                                                <span className="text-xs font-bold">
                                                                    <span className="text-emerald-400">{buyCount}B</span>
                                                                    <span className="text-slate-600 mx-1">/</span>
                                                                    <span className="text-red-400">{sellCount}S</span>
                                                                </span>
                                                            </td>
                                                        </tr>
                                                    );
                                                })}
                                            </tbody>
                                        </table>
                                    </div>
                                    {(tabData['signals']?.technical || []).length === 0 && (
                                        <p className="text-sm text-slate-500 text-center py-8">No technical signal data available</p>
                                    )}
                                </div>
                            </div>
                        </>
                    )}
                </div>
            )}

            {/* ═══════════════ TAB: SECTORS ════════════════════════ */}
            {activeTab === 'sectors' && (
                <div>
                    {tabLoading['sectors'] ? <SectionLoader /> : tabError['sectors'] ? <ErrorMsg msg={tabError['sectors']} /> : (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {(Array.isArray(tabData['sectors']) ? tabData['sectors'] : []).map((sector: any) => {
                                const avg24h = sector.avg_change_24h ?? sector.avg_performance_24h ?? 0;
                                const avg7d = sector.avg_change_7d ?? sector.avg_performance_7d ?? 0;
                                const isUp = avg24h >= 0;
                                const topCoin = sector.top_performer || sector.coins?.[0];
                                return (
                                    <Card key={sector.name || sector.sector}>
                                        <div className="flex items-center justify-between mb-3">
                                            <div>
                                                <h3 className="text-sm font-bold text-slate-200">{sector.name || sector.sector}</h3>
                                                <p className="text-[10px] text-slate-500">{sector.coin_count || sector.coins?.length || 0} coins</p>
                                            </div>
                                            <div className={`text-xl font-bold ${isUp ? 'text-emerald-400' : 'text-red-400'}`}>
                                                {isUp ? '+' : ''}{avg24h.toFixed(2)}%
                                            </div>
                                        </div>
                                        <div className="flex items-center justify-between text-xs text-slate-400 mb-3">
                                            <span>7d avg: <span className={avg7d >= 0 ? 'text-emerald-400' : 'text-red-400'}>{avg7d >= 0 ? '+' : ''}{avg7d.toFixed(2)}%</span></span>
                                            {sector.total_market_cap && (
                                                <span>MCap: {formatNumber(sector.total_market_cap, 1)}</span>
                                            )}
                                        </div>
                                        {topCoin && (
                                            <div className="border-t border-slate-700/30 pt-2 mt-2 flex items-center justify-between">
                                                <span className="text-[10px] text-slate-500 uppercase font-bold">Top Performer</span>
                                                <span className="text-xs font-bold text-slate-300">
                                                    {topCoin.name || topCoin.coin_id}
                                                    {topCoin.change_24h != null && (
                                                        <span className={`ml-2 ${topCoin.change_24h >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                            {topCoin.change_24h >= 0 ? '+' : ''}{topCoin.change_24h.toFixed(2)}%
                                                        </span>
                                                    )}
                                                </span>
                                            </div>
                                        )}
                                        {/* Mini bar chart of coin performances */}
                                        {sector.coins && sector.coins.length > 0 && (
                                            <div className="flex items-end gap-1 mt-3 h-10">
                                                {sector.coins.slice(0, 10).map((c: any, i: number) => {
                                                    const chg = c.change_24h ?? 0;
                                                    const maxAbs = Math.max(...sector.coins.slice(0, 10).map((cc: any) => Math.abs(cc.change_24h ?? 0)), 1);
                                                    const barH = Math.max(4, (Math.abs(chg) / maxAbs) * 40);
                                                    return (
                                                        <div key={i} className="flex flex-col items-center flex-1" title={`${c.name || c.coin_id}: ${chg.toFixed(2)}%`}>
                                                            <div
                                                                className={`w-full rounded-sm ${chg >= 0 ? 'bg-emerald-500/60' : 'bg-red-500/60'}`}
                                                                style={{ height: `${barH}px` }}
                                                            />
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        )}
                                    </Card>
                                );
                            })}
                            {(Array.isArray(tabData['sectors']) ? tabData['sectors'] : []).length === 0 && (
                                <p className="text-sm text-slate-500 col-span-full text-center py-8">No sector data available</p>
                            )}
                        </div>
                    )}
                </div>
            )}

            {/* ═══════════════ TAB: PREDICTIONS ═══════════════════ */}
            {activeTab === 'predictions' && (
                <div className="space-y-6">
                    {tabLoading['predictions'] ? <SectionLoader /> : tabError['predictions'] ? <ErrorMsg msg={tabError['predictions']} /> : (
                        <>
                            {/* Fear & Greed large gauge */}
                            {tabData['predictions']?.fearGreed && (
                                <Card className="flex flex-col items-center">
                                    <h2 className="text-sm font-bold text-slate-300 uppercase tracking-widest mb-4">Fear &amp; Greed Index</h2>
                                    <FearGreedGauge
                                        value={tabData['predictions'].fearGreed.value}
                                        label={tabData['predictions'].fearGreed.label || ''}
                                        size={200}
                                    />
                                    {tabData['predictions'].fearGreed.description && (
                                        <p className="text-xs text-slate-400 mt-3 text-center max-w-md">
                                            {tabData['predictions'].fearGreed.description}
                                        </p>
                                    )}
                                    {/* Components breakdown */}
                                    {tabData['predictions'].fearGreed.components && (
                                        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mt-4 w-full max-w-lg">
                                            {Object.entries(tabData['predictions'].fearGreed.components).map(([key, val]: [string, any]) => (
                                                <div key={key} className="bg-slate-800/60 rounded-lg px-3 py-2">
                                                    <p className="text-[10px] text-slate-500 uppercase font-bold">{key.replace(/_/g, ' ')}</p>
                                                    <p className="text-sm font-bold text-slate-200">{typeof val === 'object' && val !== null ? ((val as any).score ?? 0).toFixed(1) : typeof val === 'number' ? val.toFixed(1) : String(val)}</p>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </Card>
                            )}

                            {/* Predictions table */}
                            <div className="bg-slate-800/20 border border-slate-700/30 rounded-2xl overflow-hidden">
                                <div className="overflow-x-auto">
                                    <table className="w-full">
                                        <thead>
                                            <tr className="border-b border-slate-700/30 text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                                                <th className="text-left px-4 py-3">#</th>
                                                <th className="text-left px-4 py-3">Coin</th>
                                                <th className="text-center px-4 py-3">Prediction</th>
                                                <th className="text-center px-4 py-3">Confidence</th>
                                                <th className="text-center px-4 py-3">Score</th>
                                                <th className="text-right px-4 py-3">Projected 7d</th>
                                                <th className="text-left px-4 py-3">Key Factors</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {(tabData['predictions']?.predictions || []).map((coin: any, idx: number) => (
                                                <tr key={coin.coin_id || idx} className="border-b border-slate-800/30 hover:bg-slate-800/20">
                                                    <td className="px-4 py-3 text-xs text-slate-500 font-bold">{idx + 1}</td>
                                                    <td className="px-4 py-3">
                                                        <span className="text-sm font-bold text-slate-200">{coin.name || coin.coin_id}</span>
                                                        <span className="text-[10px] text-slate-500 ml-2 uppercase">{coin.symbol}</span>
                                                    </td>
                                                    <td className="px-4 py-3 text-center">
                                                        <SignalBadge signal={coin.prediction} />
                                                    </td>
                                                    <td className="px-4 py-3">
                                                        <div className="flex items-center gap-2">
                                                            <div className="flex-1 h-2 bg-slate-700/60 rounded-full overflow-hidden">
                                                                <div
                                                                    className="h-full rounded-full bg-amber-400"
                                                                    style={{ width: `${Math.min(100, coin.confidence || 0)}%` }}
                                                                />
                                                            </div>
                                                            <span className="text-[10px] text-slate-400 w-8 text-right">{(coin.confidence || 0).toFixed(0)}%</span>
                                                        </div>
                                                    </td>
                                                    <td className="px-4 py-3 w-28">
                                                        <ScoreBar value={coin.score || 0} />
                                                    </td>
                                                    <td className="px-4 py-3 text-right">
                                                        {coin.projected_7d != null ? (
                                                            <span className={`text-xs font-bold ${coin.projected_7d >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                                {coin.projected_7d >= 0 ? '+' : ''}{coin.projected_7d.toFixed(2)}%
                                                            </span>
                                                        ) : (
                                                            <span className="text-xs text-slate-500">-</span>
                                                        )}
                                                    </td>
                                                    <td className="px-4 py-3">
                                                        <div className="flex flex-wrap gap-1">
                                                            {(coin.key_factors || []).slice(0, 3).map((f: string, fi: number) => (
                                                                <span key={fi} className="text-[10px] bg-slate-700/60 text-slate-400 px-1.5 py-0.5 rounded">
                                                                    {f}
                                                                </span>
                                                            ))}
                                                        </div>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                                {(tabData['predictions']?.predictions || []).length === 0 && (
                                    <p className="text-sm text-slate-500 text-center py-8">No prediction data available</p>
                                )}
                            </div>
                        </>
                    )}
                </div>
            )}

            {/* ═══════════════ TAB: VOLATILITY ════════════════════ */}
            {activeTab === 'volatility' && (
                <div className="space-y-6">
                    {tabLoading['volatility'] ? <SectionLoader /> : tabError['volatility'] ? <ErrorMsg msg={tabError['volatility']} /> : (
                        <>
                            {/* Summary cards */}
                            {(() => {
                                const items = Array.isArray(tabData['volatility']) ? tabData['volatility'] : [];
                                if (items.length === 0) return null;
                                const sorted = [...items].sort((a: any, b: any) => (b.annual_volatility || 0) - (a.annual_volatility || 0));
                                const avgVol = items.reduce((s: number, c: any) => s + (c.annual_volatility || 0), 0) / items.length;
                                return (
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                                        <Card>
                                            <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Most Volatile</p>
                                            <p className="text-lg font-bold text-red-400 mt-1">{sorted[0]?.name || sorted[0]?.coin_id}</p>
                                            <p className="text-xs text-slate-400">{(sorted[0]?.annual_volatility || 0).toFixed(1)}% annual</p>
                                        </Card>
                                        <Card>
                                            <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Least Volatile</p>
                                            <p className="text-lg font-bold text-emerald-400 mt-1">{sorted[sorted.length - 1]?.name || sorted[sorted.length - 1]?.coin_id}</p>
                                            <p className="text-xs text-slate-400">{(sorted[sorted.length - 1]?.annual_volatility || 0).toFixed(1)}% annual</p>
                                        </Card>
                                        <Card>
                                            <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Average Volatility</p>
                                            <p className="text-lg font-bold text-amber-400 mt-1">{avgVol.toFixed(1)}%</p>
                                            <p className="text-xs text-slate-400">annual average</p>
                                        </Card>
                                    </div>
                                );
                            })()}

                            {/* Volatility ranking table */}
                            <div className="bg-slate-800/20 border border-slate-700/30 rounded-2xl overflow-hidden">
                                <div className="overflow-x-auto">
                                    <table className="w-full">
                                        <thead>
                                            <tr className="border-b border-slate-700/30 text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                                                <th className="text-left px-4 py-3">#</th>
                                                <th className="text-left px-4 py-3">Coin</th>
                                                <th className="text-center px-4 py-3">Annual Volatility</th>
                                                <th className="text-center px-4 py-3">Bollinger Width</th>
                                                <th className="text-center px-4 py-3">Risk Level</th>
                                                <th className="px-4 py-3 w-32">Visual</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {(Array.isArray(tabData['volatility']) ? tabData['volatility'] : []).map((coin: any, idx: number) => {
                                                const maxVol = Math.max(
                                                    ...(Array.isArray(tabData['volatility']) ? tabData['volatility'] : []).map((c: any) => c.annual_volatility || 0),
                                                    1
                                                );
                                                const barPct = Math.min(100, ((coin.annual_volatility || 0) / maxVol) * 100);
                                                return (
                                                    <tr key={coin.coin_id || idx} className="border-b border-slate-800/30 hover:bg-slate-800/20">
                                                        <td className="px-4 py-3 text-xs text-slate-500 font-bold">{idx + 1}</td>
                                                        <td className="px-4 py-3">
                                                            <span className="text-sm font-bold text-slate-200">{coin.name || coin.coin_id}</span>
                                                            <span className="text-[10px] text-slate-500 ml-2 uppercase">{coin.symbol}</span>
                                                        </td>
                                                        <td className="px-4 py-3 text-center text-sm font-bold text-slate-200">
                                                            {(coin.annual_volatility || 0).toFixed(1)}%
                                                        </td>
                                                        <td className="px-4 py-3 text-center text-xs text-slate-400">
                                                            {(coin.bollinger_width || 0).toFixed(3)}
                                                        </td>
                                                        <td className="px-4 py-3 text-center">
                                                            <span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded-md border ${riskColor(coin.risk_level || '')}`}>
                                                                {coin.risk_level || 'N/A'}
                                                            </span>
                                                        </td>
                                                        <td className="px-4 py-3">
                                                            <div className="w-full h-2 bg-slate-700/60 rounded-full overflow-hidden">
                                                                <div
                                                                    className="h-full rounded-full bg-gradient-to-r from-emerald-400 via-yellow-400 to-red-400"
                                                                    style={{ width: `${barPct}%` }}
                                                                />
                                                            </div>
                                                        </td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>
                                {(Array.isArray(tabData['volatility']) ? tabData['volatility'] : []).length === 0 && (
                                    <p className="text-sm text-slate-500 text-center py-8">No volatility data available</p>
                                )}
                            </div>
                        </>
                    )}
                </div>
            )}

            {/* ═══════════════ TAB: CORRELATION ═══════════════════ */}
            {activeTab === 'correlation' && (
                <div className="space-y-4">
                    {/* Period selector */}
                    <div className="flex items-center gap-2">
                        <span className="text-xs text-slate-500 font-bold uppercase">Period:</span>
                        {['7d', '30d', '90d'].map(p => (
                            <button
                                key={p}
                                onClick={() => setCorrPeriod(p)}
                                className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                                    corrPeriod === p
                                        ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                                        : 'bg-slate-800/40 text-slate-400 border border-slate-700/40 hover:border-slate-600'
                                }`}
                            >
                                {p}
                            </button>
                        ))}
                    </div>

                    {tabLoading['correlation'] ? <SectionLoader /> : tabError['correlation'] ? <ErrorMsg msg={tabError['correlation']} /> : (
                        (() => {
                            const corrData = tabData['correlation'];
                            const coinsList: string[] = corrData?.coins || [];
                            const matrix: number[][] = corrData?.matrix || [];
                            if (coinsList.length === 0) return <p className="text-sm text-slate-500 text-center py-8">No correlation data available</p>;
                            return (
                                <Card className="overflow-x-auto">
                                    <table className="min-w-max">
                                        <thead>
                                            <tr>
                                                <th className="px-2 py-2" />
                                                {coinsList.map(c => (
                                                    <th key={c} className="px-2 py-2 text-[10px] text-slate-400 font-bold uppercase text-center w-14">
                                                        {c.slice(0, 4)}
                                                    </th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {coinsList.map((row, ri) => (
                                                <tr key={row}>
                                                    <td className="px-2 py-1 text-[10px] text-slate-400 font-bold uppercase text-right pr-3 w-16">
                                                        {row.slice(0, 4)}
                                                    </td>
                                                    {coinsList.map((col, ci) => {
                                                        const val = matrix[ri]?.[ci] ?? 0;
                                                        return (
                                                            <td key={col} className="px-0 py-0">
                                                                <div
                                                                    className={`w-14 h-10 flex items-center justify-center text-[10px] font-bold ${
                                                                        ri === ci ? 'bg-amber-500/20 text-amber-400' : `${corrColor(val)} text-slate-200`
                                                                    }`}
                                                                    title={`${row} / ${col}: ${val.toFixed(2)}`}
                                                                >
                                                                    {val.toFixed(2)}
                                                                </div>
                                                            </td>
                                                        );
                                                    })}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </Card>
                            );
                        })()
                    )}
                </div>
            )}

            {/* ═══════════════ TAB: BTC DOMINANCE ═════════════════ */}
            {activeTab === 'btc-dominance' && (
                <div className="space-y-6">
                    {tabLoading['btc-dominance'] ? <SectionLoader /> : tabError['btc-dominance'] ? <ErrorMsg msg={tabError['btc-dominance']} /> : (
                        (() => {
                            const d = tabData['btc-dominance'];
                            if (!d) return <p className="text-sm text-slate-500 text-center py-8">No BTC dominance data available</p>;
                            const altIdx = d.altcoin_season_index ?? 0;
                            const altLabel = altIdx < 25 ? 'Bitcoin Season' : altIdx > 75 ? 'Altcoin Season' : 'Neutral';
                            const altColor = altIdx < 25 ? '#f97316' : altIdx > 75 ? '#22c55e' : '#eab308';
                            return (
                                <>
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                        {/* BTC Dominance */}
                                        <Card className="flex flex-col items-center justify-center">
                                            <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-2">BTC Dominance</p>
                                            <p className="text-4xl font-bold text-amber-400">{(d.btc_dominance || 0).toFixed(1)}%</p>
                                            <div className="flex items-center gap-1 mt-2">
                                                {(d.trend || '').toLowerCase() === 'up' ? (
                                                    <TrendingUp size={16} className="text-emerald-400" />
                                                ) : (d.trend || '').toLowerCase() === 'down' ? (
                                                    <TrendingDown size={16} className="text-red-400" />
                                                ) : (
                                                    <Minus size={16} className="text-slate-400" />
                                                )}
                                                <span className="text-xs text-slate-400 capitalize">{d.trend || 'stable'}</span>
                                            </div>
                                        </Card>

                                        {/* Altcoin Season Index */}
                                        <Card className="flex flex-col items-center justify-center">
                                            <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-2">Altcoin Season Index</p>
                                            <FearGreedGauge value={altIdx} label={altLabel} size={160} />
                                        </Card>

                                        {/* Season Indicator */}
                                        <Card className="flex flex-col items-center justify-center">
                                            <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-3">Current Season</p>
                                            <div
                                                className="w-20 h-20 rounded-full flex items-center justify-center border-4"
                                                style={{ borderColor: altColor, backgroundColor: `${altColor}20` }}
                                            >
                                                {d.is_altcoin_season ? (
                                                    <Layers size={28} style={{ color: altColor }} />
                                                ) : (
                                                    <Bitcoin size={28} style={{ color: altColor }} />
                                                )}
                                            </div>
                                            <p className="text-sm font-bold mt-3" style={{ color: altColor }}>
                                                {d.is_altcoin_season ? 'Altcoin Season' : 'Bitcoin Season'}
                                            </p>
                                        </Card>
                                    </div>

                                    {/* Analysis */}
                                    {d.analysis && (
                                        <Card>
                                            <h3 className="text-sm font-bold text-slate-300 uppercase tracking-widest mb-2 flex items-center gap-2">
                                                <Shield size={14} className="text-amber-400" /> Analysis
                                            </h3>
                                            <p className="text-sm text-slate-400 leading-relaxed">{d.analysis}</p>
                                        </Card>
                                    )}
                                </>
                            );
                        })()
                    )}
                </div>
            )}

            {/* ═══════════════ TAB: PORTFOLIO ═════════════════════ */}
            {activeTab === 'portfolio' && (
                <div className="space-y-6">
                    {/* Holdings input */}
                    <Card>
                        <h3 className="text-sm font-bold text-slate-300 uppercase tracking-widest mb-4 flex items-center gap-2">
                            <Briefcase size={14} className="text-amber-400" /> Your Holdings
                        </h3>
                        {/* Current holdings list */}
                        <div className="space-y-2 mb-4">
                            {holdings.map((h, i) => (
                                <div key={i} className="flex items-center gap-3 bg-slate-800/60 rounded-xl px-3 py-2">
                                    <span className="text-sm font-bold text-slate-200 flex-1">{h.coin_id}</span>
                                    <span className="text-sm text-slate-400">${h.amount_usd.toLocaleString()}</span>
                                    <button
                                        onClick={() => removeHolding(i)}
                                        className="p-1 hover:bg-red-500/20 rounded-lg transition-colors"
                                    >
                                        <Trash2 size={14} className="text-red-400" />
                                    </button>
                                </div>
                            ))}
                            {holdings.length === 0 && (
                                <p className="text-sm text-slate-500 text-center py-4">No holdings added yet</p>
                            )}
                        </div>
                        {/* Add holding form */}
                        <div className="flex items-center gap-2">
                            <input
                                type="text"
                                value={newCoinId}
                                onChange={e => setNewCoinId(e.target.value)}
                                placeholder="coin id (e.g. bitcoin)"
                                className="flex-1 px-3 py-2 bg-slate-800/60 border border-slate-700/40 rounded-xl text-sm text-slate-200 placeholder-slate-500 focus:border-amber-500/50 outline-none"
                            />
                            <input
                                type="number"
                                value={newAmount}
                                onChange={e => setNewAmount(e.target.value)}
                                placeholder="USD amount"
                                className="w-32 px-3 py-2 bg-slate-800/60 border border-slate-700/40 rounded-xl text-sm text-slate-200 placeholder-slate-500 focus:border-amber-500/50 outline-none"
                            />
                            <button
                                onClick={addHolding}
                                className="p-2 bg-amber-500/20 text-amber-400 border border-amber-500/30 rounded-xl hover:bg-amber-500/30 transition-colors"
                            >
                                <Plus size={16} />
                            </button>
                        </div>
                        {/* Optimize button */}
                        <button
                            onClick={optimizePortfolio}
                            disabled={holdings.length === 0 || portfolioLoading}
                            className="mt-4 w-full py-3 bg-gradient-to-r from-amber-500 to-orange-500 text-white font-bold rounded-xl hover:from-amber-600 hover:to-orange-600 disabled:opacity-40 transition-all flex items-center justify-center gap-2"
                        >
                            {portfolioLoading ? <Loader2 size={16} className="animate-spin" /> : <Zap size={16} />}
                            Optimize Portfolio
                        </button>
                        {portfolioError && <p className="text-xs text-red-400 mt-2">{portfolioError}</p>}
                    </Card>

                    {/* Portfolio Results */}
                    {portfolioResult && (
                        <>
                            {/* Allocation comparison */}
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                {/* Current allocation */}
                                {portfolioResult.current_allocation && (
                                    <Card>
                                        <h4 className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-3">Current Allocation</h4>
                                        <PieChart data={portfolioResult.current_allocation} />
                                    </Card>
                                )}
                                {/* Optimized allocation */}
                                {portfolioResult.optimized_allocation && (
                                    <Card>
                                        <h4 className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-3">Optimized Allocation</h4>
                                        <PieChart data={portfolioResult.optimized_allocation} />
                                    </Card>
                                )}
                                {/* Equal weight */}
                                {portfolioResult.equal_weight && (
                                    <Card>
                                        <h4 className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-3">Equal Weight</h4>
                                        <PieChart data={portfolioResult.equal_weight} />
                                    </Card>
                                )}
                            </div>

                            {/* Metrics */}
                            {portfolioResult.metrics && (
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                                    <Card>
                                        <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Expected Return</p>
                                        <p className="text-xl font-bold text-emerald-400 mt-1">
                                            {(portfolioResult.metrics.expected_return || 0).toFixed(2)}%
                                        </p>
                                    </Card>
                                    <Card>
                                        <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Volatility</p>
                                        <p className="text-xl font-bold text-amber-400 mt-1">
                                            {(portfolioResult.metrics.volatility || 0).toFixed(2)}%
                                        </p>
                                    </Card>
                                    <Card>
                                        <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Sharpe Ratio</p>
                                        <p className="text-xl font-bold text-violet-400 mt-1">
                                            {(portfolioResult.metrics.sharpe_ratio || 0).toFixed(3)}
                                        </p>
                                    </Card>
                                </div>
                            )}
                        </>
                    )}
                </div>
            )}
            {/* ═══════════════ COIN DETAIL MODAL ═══════════════════ */}
            {selectedCoin && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setSelectedCoin(null)}>
                    <div
                        className="bg-slate-900 border border-slate-700/60 rounded-2xl w-full max-w-3xl max-h-[85vh] overflow-y-auto mx-4 shadow-2xl"
                        onClick={(e) => e.stopPropagation()}
                    >
                        {/* Header */}
                        <div className="sticky top-0 bg-slate-900/95 backdrop-blur border-b border-slate-800 px-6 py-4 flex items-center justify-between rounded-t-2xl z-10">
                            <div className="flex items-center gap-3">
                                {selectedCoin.image && <img src={selectedCoin.image} alt="" className="w-10 h-10 rounded-full" />}
                                <div>
                                    <h2 className="text-lg font-bold text-slate-100">{selectedCoin.name || selectedCoin.id}</h2>
                                    <div className="flex items-center gap-2">
                                        <span className="text-xs text-slate-500 uppercase font-bold">{selectedCoin.symbol}</span>
                                        {selectedCoin.coingecko_rank && <span className="text-[10px] bg-amber-500/20 text-amber-400 px-1.5 py-0.5 rounded font-bold">Rank #{selectedCoin.coingecko_rank}</span>}
                                    </div>
                                </div>
                            </div>
                            <button onClick={() => setSelectedCoin(null)} className="text-slate-500 hover:text-slate-300 transition-colors p-1">
                                <X size={20} />
                            </button>
                        </div>

                        {selectedCoin.loading ? (
                            <div className="py-20 flex justify-center"><Loader2 size={32} className="animate-spin text-amber-400" /></div>
                        ) : selectedCoin.error ? (
                            <div className="py-16 text-center text-red-400 text-sm">{selectedCoin.error}</div>
                        ) : (
                            <div className="p-6 space-y-6">
                                {/* Price & Market Data */}
                                {selectedCoin.market_data && (
                                    <>
                                        <div className="flex items-baseline gap-3 flex-wrap">
                                            <span className="text-3xl font-bold text-slate-100">{formatPrice(selectedCoin.market_data.current_price)}</span>
                                            <ChangeCell value={selectedCoin.market_data.price_change_pct_24h} />
                                        </div>

                                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                                            <Card>
                                                <p className="text-[10px] text-slate-500 font-bold uppercase">Market Cap</p>
                                                <p className="text-sm font-bold text-slate-200 mt-1">{formatNumber(selectedCoin.market_data.market_cap)}</p>
                                            </Card>
                                            <Card>
                                                <p className="text-[10px] text-slate-500 font-bold uppercase">24h Volume</p>
                                                <p className="text-sm font-bold text-slate-200 mt-1">{formatNumber(selectedCoin.market_data.total_volume)}</p>
                                            </Card>
                                            <Card>
                                                <p className="text-[10px] text-slate-500 font-bold uppercase">24h Range</p>
                                                <p className="text-xs text-slate-300 mt-1">{formatPrice(selectedCoin.market_data.low_24h)} — {formatPrice(selectedCoin.market_data.high_24h)}</p>
                                            </Card>
                                            <Card>
                                                <p className="text-[10px] text-slate-500 font-bold uppercase">ATH</p>
                                                <p className="text-sm font-bold text-slate-200 mt-1">{formatPrice(selectedCoin.market_data.ath)}</p>
                                                <p className="text-[10px] text-red-400">{(selectedCoin.market_data.ath_change_pct || 0).toFixed(1)}%</p>
                                            </Card>
                                        </div>

                                        {/* Price Changes */}
                                        <div className="grid grid-cols-4 gap-3">
                                            {[
                                                { label: '24h', value: selectedCoin.market_data.price_change_pct_24h },
                                                { label: '7d', value: selectedCoin.market_data.price_change_pct_7d },
                                                { label: '30d', value: selectedCoin.market_data.price_change_pct_30d },
                                                { label: '1y', value: selectedCoin.market_data.price_change_pct_1y },
                                            ].map(({ label, value }) => (
                                                <div key={label} className="bg-slate-800/30 rounded-lg p-2.5 text-center">
                                                    <p className="text-[10px] text-slate-500 font-bold">{label}</p>
                                                    <p className={`text-sm font-bold mt-0.5 ${(value || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                        {(value || 0) >= 0 ? '+' : ''}{(value || 0).toFixed(2)}%
                                                    </p>
                                                </div>
                                            ))}
                                        </div>

                                        {/* Supply Info */}
                                        <div className="grid grid-cols-3 gap-3">
                                            <div className="bg-slate-800/30 rounded-lg p-3">
                                                <p className="text-[10px] text-slate-500 font-bold">Circulating Supply</p>
                                                <p className="text-sm text-slate-300 mt-1">{(selectedCoin.market_data.circulating_supply || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}</p>
                                            </div>
                                            <div className="bg-slate-800/30 rounded-lg p-3">
                                                <p className="text-[10px] text-slate-500 font-bold">Total Supply</p>
                                                <p className="text-sm text-slate-300 mt-1">{selectedCoin.market_data.total_supply ? selectedCoin.market_data.total_supply.toLocaleString(undefined, { maximumFractionDigits: 0 }) : '∞'}</p>
                                            </div>
                                            <div className="bg-slate-800/30 rounded-lg p-3">
                                                <p className="text-[10px] text-slate-500 font-bold">Max Supply</p>
                                                <p className="text-sm text-slate-300 mt-1">{selectedCoin.market_data.max_supply ? selectedCoin.market_data.max_supply.toLocaleString(undefined, { maximumFractionDigits: 0 }) : '∞'}</p>
                                            </div>
                                        </div>

                                        {/* 7d Sparkline */}
                                        {selectedCoin.market_data.sparkline_7d?.length > 2 && (
                                            <div className="bg-slate-800/30 rounded-lg p-4">
                                                <p className="text-[10px] text-slate-500 font-bold uppercase mb-2">7-Day Price</p>
                                                <MiniSparkline data={selectedCoin.market_data.sparkline_7d} positive={(selectedCoin.market_data.price_change_pct_7d || 0) >= 0} />
                                            </div>
                                        )}
                                    </>
                                )}

                                {/* Categories */}
                                {selectedCoin.categories?.length > 0 && (
                                    <div className="flex flex-wrap gap-1.5">
                                        <Tag size={12} className="text-slate-500 mt-0.5" />
                                        {selectedCoin.categories.filter(Boolean).slice(0, 8).map((cat: string) => (
                                            <span key={cat} className="text-[10px] bg-slate-800/60 text-slate-400 px-2 py-0.5 rounded-full border border-slate-700/40">{cat}</span>
                                        ))}
                                    </div>
                                )}

                                {/* Description */}
                                {selectedCoin.description && (
                                    <div>
                                        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">About</h3>
                                        <p className="text-sm text-slate-300 leading-relaxed line-clamp-6">{selectedCoin.description}</p>
                                    </div>
                                )}

                                {/* Genesis & Algorithm */}
                                <div className="flex gap-4 text-xs text-slate-400">
                                    {selectedCoin.genesis_date && (
                                        <span className="flex items-center gap-1"><Calendar size={12} /> Launched: {selectedCoin.genesis_date}</span>
                                    )}
                                    {selectedCoin.hashing_algorithm && (
                                        <span className="flex items-center gap-1"><Code2 size={12} /> {selectedCoin.hashing_algorithm}</span>
                                    )}
                                </div>

                                {/* Community & Developer Stats */}
                                {(selectedCoin.community || selectedCoin.developer) && (
                                    <div className="grid grid-cols-2 gap-4">
                                        {selectedCoin.community && (
                                            <div>
                                                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2 flex items-center gap-1">
                                                    <Users size={12} /> Community
                                                </h3>
                                                <div className="space-y-1.5">
                                                    {selectedCoin.community.twitter_followers && (
                                                        <div className="flex justify-between text-xs">
                                                            <span className="text-slate-500">Twitter Followers</span>
                                                            <span className="text-slate-300 font-bold">{selectedCoin.community.twitter_followers.toLocaleString()}</span>
                                                        </div>
                                                    )}
                                                    {selectedCoin.community.reddit_subscribers && (
                                                        <div className="flex justify-between text-xs">
                                                            <span className="text-slate-500">Reddit Subscribers</span>
                                                            <span className="text-slate-300 font-bold">{selectedCoin.community.reddit_subscribers.toLocaleString()}</span>
                                                        </div>
                                                    )}
                                                    {selectedCoin.community.reddit_active_accounts && (
                                                        <div className="flex justify-between text-xs">
                                                            <span className="text-slate-500">Reddit Active (48h)</span>
                                                            <span className="text-slate-300 font-bold">{selectedCoin.community.reddit_active_accounts.toLocaleString()}</span>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        )}
                                        {selectedCoin.developer && (
                                            <div>
                                                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2 flex items-center gap-1">
                                                    <Code2 size={12} /> Developer
                                                </h3>
                                                <div className="space-y-1.5">
                                                    {selectedCoin.developer.github_stars != null && (
                                                        <div className="flex justify-between text-xs">
                                                            <span className="text-slate-500">GitHub Stars</span>
                                                            <span className="text-slate-300 font-bold">{(selectedCoin.developer.github_stars || 0).toLocaleString()}</span>
                                                        </div>
                                                    )}
                                                    {selectedCoin.developer.github_forks != null && (
                                                        <div className="flex justify-between text-xs">
                                                            <span className="text-slate-500">GitHub Forks</span>
                                                            <span className="text-slate-300 font-bold">{(selectedCoin.developer.github_forks || 0).toLocaleString()}</span>
                                                        </div>
                                                    )}
                                                    {selectedCoin.developer.commit_count_4_weeks != null && (
                                                        <div className="flex justify-between text-xs">
                                                            <span className="text-slate-500">Commits (4 wks)</span>
                                                            <span className="text-slate-300 font-bold">{(selectedCoin.developer.commit_count_4_weeks || 0).toLocaleString()}</span>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}

                                {/* Sentiment */}
                                {selectedCoin.sentiment_votes_up_pct != null && (
                                    <div className="bg-slate-800/30 rounded-lg p-3">
                                        <p className="text-[10px] text-slate-500 font-bold uppercase mb-2">Sentiment</p>
                                        <div className="flex items-center gap-2">
                                            <div className="flex-1 h-3 bg-red-500/30 rounded-full overflow-hidden">
                                                <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${selectedCoin.sentiment_votes_up_pct || 50}%` }} />
                                            </div>
                                            <span className="text-xs text-emerald-400 font-bold">{(selectedCoin.sentiment_votes_up_pct || 0).toFixed(0)}%</span>
                                            <TrendingUp size={12} className="text-emerald-400" />
                                        </div>
                                    </div>
                                )}

                                {/* Links */}
                                {selectedCoin.links && (
                                    <div>
                                        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">Links</h3>
                                        <div className="flex flex-wrap gap-2">
                                            {selectedCoin.links.homepage && (
                                                <a href={selectedCoin.links.homepage} target="_blank" rel="noopener noreferrer"
                                                    className="flex items-center gap-1 text-xs bg-slate-800/60 hover:bg-slate-700/60 text-blue-400 px-3 py-1.5 rounded-lg border border-slate-700/40 transition-colors">
                                                    <ExternalLink size={11} /> Website
                                                </a>
                                            )}
                                            {selectedCoin.links.twitter && !selectedCoin.links.twitter.endsWith('/') && (
                                                <a href={selectedCoin.links.twitter} target="_blank" rel="noopener noreferrer"
                                                    className="flex items-center gap-1 text-xs bg-slate-800/60 hover:bg-slate-700/60 text-blue-400 px-3 py-1.5 rounded-lg border border-slate-700/40 transition-colors">
                                                    <Twitter size={11} /> Twitter
                                                </a>
                                            )}
                                            {selectedCoin.links.subreddit && (
                                                <a href={selectedCoin.links.subreddit} target="_blank" rel="noopener noreferrer"
                                                    className="flex items-center gap-1 text-xs bg-slate-800/60 hover:bg-slate-700/60 text-orange-400 px-3 py-1.5 rounded-lg border border-slate-700/40 transition-colors">
                                                    <Users size={11} /> Reddit
                                                </a>
                                            )}
                                            {selectedCoin.links.github?.length > 0 && selectedCoin.links.github[0] && (
                                                <a href={selectedCoin.links.github[0]} target="_blank" rel="noopener noreferrer"
                                                    className="flex items-center gap-1 text-xs bg-slate-800/60 hover:bg-slate-700/60 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700/40 transition-colors">
                                                    <Github size={11} /> GitHub
                                                </a>
                                            )}
                                            {selectedCoin.links.blockchain_site?.length > 0 && (
                                                <a href={selectedCoin.links.blockchain_site[0]} target="_blank" rel="noopener noreferrer"
                                                    className="flex items-center gap-1 text-xs bg-slate-800/60 hover:bg-slate-700/60 text-violet-400 px-3 py-1.5 rounded-lg border border-slate-700/40 transition-colors">
                                                    <Globe size={11} /> Explorer
                                                </a>
                                            )}
                                        </div>
                                    </div>
                                )}

                                {/* Scores */}
                                {(selectedCoin.coingecko_score || selectedCoin.liquidity_score) && (
                                    <div className="flex gap-4 text-xs border-t border-slate-800/40 pt-4">
                                        {selectedCoin.coingecko_score && (
                                            <span className="text-slate-500">CoinGecko Score: <span className="text-amber-400 font-bold">{selectedCoin.coingecko_score.toFixed(1)}</span></span>
                                        )}
                                        {selectedCoin.liquidity_score && (
                                            <span className="text-slate-500">Liquidity Score: <span className="text-blue-400 font-bold">{selectedCoin.liquidity_score.toFixed(1)}</span></span>
                                        )}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

// ─── CSS-based Pie Chart ─────────────────────────────────────────
const PIE_COLORS = ['#f59e0b', '#8b5cf6', '#22c55e', '#3b82f6', '#ef4444', '#ec4899', '#14b8a6', '#f97316'];

const PieChart = ({ data }: { data: Record<string, number> | Array<{ coin_id: string; weight: number }> }) => {
    // Normalize to array
    const items: { label: string; value: number }[] = Array.isArray(data)
        ? data.map(d => ({ label: d.coin_id, value: d.weight * 100 }))
        : Object.entries(data).map(([k, v]) => ({ label: k, value: typeof v === 'number' ? v * 100 : 0 }));

    const total = items.reduce((s, i) => s + i.value, 0) || 1;
    let cumulative = 0;
    const segments = items.map((item, idx) => {
        const pct = (item.value / total) * 100;
        const start = cumulative;
        cumulative += pct;
        return { ...item, pct, start, color: PIE_COLORS[idx % PIE_COLORS.length] };
    });

    // Build conic gradient
    const gradientStops = segments.map(s => `${s.color} ${s.start}% ${s.start + s.pct}%`).join(', ');

    return (
        <div className="flex flex-col items-center gap-3">
            <div
                className="w-28 h-28 rounded-full"
                style={{
                    background: `conic-gradient(${gradientStops})`,
                }}
            />
            <div className="space-y-1 w-full">
                {segments.map(s => (
                    <div key={s.label} className="flex items-center gap-2 text-xs">
                        <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: s.color }} />
                        <span className="text-slate-400 flex-1 capitalize">{s.label}</span>
                        <span className="text-slate-300 font-bold">{s.pct.toFixed(1)}%</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default CryptoDashboard;
