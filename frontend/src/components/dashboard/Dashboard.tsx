import MetricCard from "@/components/backtest/MetricCard";
import {
    Activity,
    AlertTriangle,
    ArrowDownRight,
    ArrowUpRight,
    Award,
    Calendar,
    Download,
    Eye,
    History,
    List,
    PlayCircle,
    RefreshCw,
    Search,
    Sparkles,
    Target,
    TrendingDown,
    TrendingUp
} from "lucide-react";
import {
    Area,
    AreaChart,
    CartesianGrid,
    Cell,
    ComposedChart,
    Line,
    Pie,
    PieChart as RechartsPieChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from "recharts";
import {formatCurrency, formatPercent, toPrecision} from "@/utils/formatters";

import {useEffect, useMemo, useState} from "react";
import {analytics, backtest, portfolio, strategy} from "@/utils/api";
import {BacktestHistoryItem, Portfolio, StrategyInfo} from "@/types/all_types";
import RiskAnalysisModal from "@/components/backtest/RiskAnalysisModal";

interface PerformanceMetrics {
    total_return: number;
    total_return_pct: number;
    win_rate: number;
    sharpe_ratio: number;
    max_drawdown: number;
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    avg_profit: number;
    avg_win: number;
    avg_loss: number;
    total_profit: number;
    profit_factor: number;
    final_equity: number;
    initial_capital: number;
}

const Dashboard = () => {
    const [metrics, setMetrics] = useState<PerformanceMetrics>({
        total_return: 0,
        total_return_pct: 0,
        win_rate: 0,
        sharpe_ratio: 0,
        max_drawdown: 0,
        total_trades: 0,
        winning_trades: 0,
        losing_trades: 0,
        avg_profit: 0,
        avg_win: 0,
        avg_loss: 0,
        total_profit: 0,
        profit_factor: 0,
        final_equity: 0,
        initial_capital: 0
    });
    const [equityData, setEquityData] = useState<any[]>([]);
    const [recentTrades, setRecentTrades] = useState<any[]>([]);
    const [backtestHistory, setBacktestHistory] = useState<BacktestHistoryItem[]>([]);
    const [activeStrategies, setActiveStrategies] = useState<any[]>([]);
    const [activeHistoryTab, setActiveHistoryTab] = useState<'trades' | 'backtests'>('backtests');
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [selectedBacktest, setSelectedBacktest] = useState<BacktestHistoryItem | null>(null);
    const [performanceDistribution, setPerformanceDistribution] = useState<any[]>([]);
    const [selectedPeriod, setSelectedPeriod] = useState('1M');
    const [tradeDistribution, setTradeDistribution] = useState<any[]>([]);
    const [usePortfolioData, setUsePortfolioData] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [filterType, setFilterType] = useState<'all' | 'single' | 'multi'>('all');
    const [showRiskAnalysis, setShowRiskAnalysis] = useState(false);

    // ============================================================
    // Monthly Returns Calculation
    // ============================================================
    const monthlyReturns = useMemo(() => {
        if (!equityData || equityData.length === 0) return [];

        const monthlyData: { [key: string]: { start: number; end: number } } = {};

        equityData.forEach((point) => {
            const timestamp = point.timestamp || point.date;
            const date = new Date(timestamp);
            const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
            const value = point.equity || point.value || 0;

            if (!monthlyData[monthKey]) {
                monthlyData[monthKey] = { start: value, end: value };
            } else {
                monthlyData[monthKey].end = value;
            }
        });

        return Object.entries(monthlyData)
            .map(([month, data]) => {
                const returnPct = ((data.end - data.start) / data.start) * 100;
                return {
                    month,
                    return: returnPct,
                    startEquity: data.start,
                    endEquity: data.end
                };
            })
            .sort((a, b) => a.month.localeCompare(b.month));
    }, [equityData]);

    // ============================================================
    // Drawdown Calculation
    // ============================================================
    const drawdownData = useMemo(() => {
        if (!equityData || equityData.length === 0) return [];

        let peak = equityData[0].equity || equityData[0].value || 0;
        return equityData.map((point) => {
            const value = point.equity || point.value || 0;
            peak = Math.max(peak, value);
            const drawdown = ((value - peak) / peak) * 100;

            return {
                timestamp: point.timestamp || point.date,
                drawdown: drawdown,
                peak: peak
            };
        });
    }, [equityData]);

    // ============================================================
    // Rolling Sharpe Ratio (30-day window)
    // ============================================================
    const rollingSharpe = useMemo(() => {
        if (!equityData || equityData.length < 30) return [];

        const window = 30;
        const riskFreeRate = 0;

        return equityData.map((point, index) => {
            if (index < window) return null;

            const windowData = equityData.slice(index - window, index + 1);
            const returns = windowData.slice(1).map((p, i) => {
                const current = p.equity || p.value || 0;
                const previous = windowData[i].equity || windowData[i].value || 1;
                return ((current - previous) / previous) * 100;
            });

            const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
            const stdDev = Math.sqrt(
                returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
            );

            const sharpe = stdDev !== 0 ? (avgReturn - riskFreeRate) / stdDev : 0;

            return {
                timestamp: point.timestamp || point.date,
                sharpe: sharpe * Math.sqrt(252) // Annualized
            };
        }).filter(d => d !== null);
    }, [equityData]);

    // ============================================================
    // Load Backtest Data (for replay/review)
    // ============================================================
    const loadBacktestData = async (backtestHistoryItem: BacktestHistoryItem) => {
        try {
            setLoading(true);
            console.log('📊 Loading backtest:', backtestHistoryItem.id);

            // Fetch detailed data if not present
            let detailedData = backtestHistoryItem;
            if (!backtestHistoryItem.equity_curve || !backtestHistoryItem.trades) {
                detailedData = await backtest.getDetails(backtestHistoryItem.id);
            }

            setSelectedBacktest(detailedData);

            // Set metrics
            if (detailedData.equity_curve && detailedData.trades) {
                setMetrics({
                    total_return: detailedData.total_return_pct || 0,
                    total_return_pct: detailedData.total_return_pct || 0,
                    win_rate: detailedData.win_rate || 0,
                    sharpe_ratio: detailedData.sharpe_ratio || 0,
                    max_drawdown: detailedData.max_drawdown || 0,
                    total_trades: detailedData.total_trades || 0,
                    winning_trades: detailedData.trades?.filter((t: any) => t.profit > 0).length || 0,
                    losing_trades: detailedData.trades?.filter((t: any) => t.profit < 0).length || 0,
                    avg_profit: detailedData.trades?.reduce((sum: number, t: any) => sum + (t.profit || 0), 0) / (detailedData.total_trades || 1),
                    avg_win: detailedData.trades?.filter((t: any) => t.profit > 0).reduce((sum: number, t: any) => sum + t.profit, 0) / (detailedData.trades?.filter((t: any) => t.profit > 0).length || 1),
                    avg_loss: detailedData.trades?.filter((t: any) => t.profit < 0).reduce((sum: number, t: any) => sum + t.profit, 0) / (detailedData.trades?.filter((t: any) => t.profit < 0).length || 1),
                    total_profit: detailedData.trades?.reduce((sum: number, t: any) => sum + (t.profit || 0), 0) || 0,
                    profit_factor: 0,
                    final_equity: detailedData.final_equity || 0,
                    initial_capital: detailedData.initial_capital || 0
                });

                // Calculate profit factor
                const totalWins = detailedData.trades?.filter((t: any) => t.profit > 0).reduce((sum: number, t: any) => sum + t.profit, 0) || 0;
                const totalLosses = Math.abs(detailedData.trades?.filter((t: any) => t.profit < 0).reduce((sum: number, t: any) => sum + t.profit, 0) || 1);
                setMetrics(prev => ({
                    ...prev,
                    profit_factor: totalLosses !== 0 ? totalWins / totalLosses : 0
                }));

                // Set equity curve
                const curve = detailedData.equity_curve.map((p: any) => ({
                    date: new Date(p.timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }),
                    timestamp: p.timestamp,
                    value: p.equity,
                    equity: p.equity,
                    cash: p.cash || 0
                }));
                setEquityData(curve);

                // Set trade distribution
                const winning = detailedData.trades?.filter((t: any) => t.profit > 0).length || 0;
                const losing = detailedData.trades?.filter((t: any) => t.profit < 0).length || 0;
                setTradeDistribution([
                    { name: 'Winning', value: winning, color: '#10b981' },
                    { name: 'Losing', value: losing, color: '#ef4444' }
                ]);

                // Set trades for trade history
                setRecentTrades(detailedData.trades?.slice(0, 10).map((t: any) => ({
                    symbol: t.symbol,
                    strategy: t.strategy || detailedData.strategy_config?.strategy_key || 'Unknown',
                    profit: t.profit || 0,
                    time: t.executed_at ? new Date(t.executed_at).toLocaleTimeString() : 'N/A',
                    date: t.executed_at ? new Date(t.executed_at).toLocaleDateString() : 'N/A',
                    status: 'closed'
                })) || []);
            }

            setLoading(false);
        } catch (error) {
            console.error('Failed to load backtest:', error);
            setLoading(false);
        }
    };

    const fetchDashboardData = async () => {
        try {
            setRefreshing(true);

            const historyRes = await backtest.getHistory({ limit: 50 });
            console.log('📊 Backtest History:', historyRes);

            if (historyRes && historyRes.length > 0) {
                setBacktestHistory(historyRes);

                if (!selectedBacktest) {
                    await loadBacktestData(historyRes[0]);
                }

                if (historyRes.length > 1) {
                    const distribution = historyRes
                        .filter(b => b.total_return_pct !== null)
                        .map(b => ({
                            name: b.name || `BT #${b.id}`,
                            return: b.total_return_pct || 0,
                            sharpe: b.sharpe_ratio || 0,
                            id: b.id
                        }))
                        .slice(0, 6);
                    setPerformanceDistribution(distribution);
                }

                setUsePortfolioData(false);
            } else {
                console.log('⚠️ No backtest history, using portfolio data');
                setUsePortfolioData(true);
            }

            // Fetch Portfolio Analytics (fallback)
            if (usePortfolioData || historyRes.length === 0) {
                const portfoliosRes: Portfolio[] = await portfolio.list();
                if (portfoliosRes && portfoliosRes.length > 0) {
                    const pid = portfoliosRes[0].id;

                    try {
                        const perfRes = await analytics.getPerformance(pid, { period: selectedPeriod });

                        if (perfRes.metrics) {
                            setMetrics(perfRes.metrics as PerformanceMetrics);
                        }

                        if (perfRes.equity_curve && perfRes.equity_curve.length > 0) {
                            const curve = perfRes.equity_curve.map((p: any) => ({
                                date: new Date(p.timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }),
                                timestamp: p.timestamp,
                                value: p.equity,
                                equity: p.equity,
                                cash: p.cash || 0
                            }));
                            setEquityData(curve);
                        }
                    } catch (e) {
                        console.error('Failed to fetch portfolio analytics:', e);
                    }
                }
            }

            // 3. Get Strategies
            const strategiesRes = await strategy.list();
            if (strategiesRes) {
                setActiveStrategies(strategiesRes.slice(0, 4).map((s: StrategyInfo) => ({
                    name: s.name,
                    performance: s.historical_return || 0,
                    trades: s.total_trades || 0,
                    status: 'active',
                    description: s.description || ''
                })));
            }

            setRefreshing(false);
        } catch (error) {
            console.error("Failed to load dashboard data", error);
            setRefreshing(false);
        }
    };

    useEffect(() => {
        fetchDashboardData();
    }, [selectedPeriod]);

    // Filtered backtest history
    const filteredHistory = useMemo(() => {
        return backtestHistory.filter(bt => {
            const matchesSearch = searchQuery === '' ||
                bt.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
                bt.symbols.some(s => s.toLowerCase().includes(searchQuery.toLowerCase()));

            const matchesFilter = filterType === 'all' || bt.backtest_type === filterType;

            return matchesSearch && matchesFilter;
        });
    }, [backtestHistory, searchQuery, filterType]);

    if (loading && !selectedBacktest) return (
        <div className="flex flex-col items-center justify-center h-[60vh]">
            <div className="relative">
                <div className="w-16 h-16 border-4 border-violet-500/20 border-t-violet-500 rounded-full animate-spin"></div>
                <Sparkles className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-violet-400" size={24} />
            </div>
            <p className="text-lg font-semibold text-slate-300 mt-6">Loading Backtest Analytics</p>
            <p className="text-sm text-slate-500 mt-2">Preparing replay environment...</p>
        </div>
    );

    const hasData = equityData.length > 0;
    const hasIncrease = equityData.length > 1 && (equityData[equityData.length - 1].value || equityData[equityData.length - 1].equity) > (equityData[0].value || equityData[0].equity);
    const totalGain = hasData ? (equityData[equityData.length - 1].value || equityData[equityData.length - 1].equity) - (equityData[0].value || equityData[0].equity) : 0;
    const totalGainPct = hasData && (equityData[0].value || equityData[0].equity) > 0
        ? (((equityData[equityData.length - 1].value || equityData[equityData.length - 1].equity) - (equityData[0].value || equityData[0].equity)) / (equityData[0].value || equityData[0].equity)) * 100
        : 0;

    return (
        <div className="space-y-6">
            {/* Enhanced Header with Backtest Info */}
            <div className="flex justify-between items-center">
                <div>
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center shadow-lg shadow-violet-500/20">
                            <PlayCircle className="text-white" size={22} strokeWidth={2.5} />
                        </div>
                        <div>
                            <h1 className="text-3xl font-bold text-slate-100">Backtest Replay Center</h1>
                            <p className="text-sm text-slate-500 mt-1">
                                {selectedBacktest
                                    ? `${selectedBacktest.name || `Backtest #${selectedBacktest.id}`} • ${selectedBacktest.symbols.join(', ')} • ${new Date(selectedBacktest.created_at || '').toLocaleDateString()}`
                                    : 'Select a backtest to review detailed analytics'
                                }
                            </p>
                        </div>
                    </div>
                </div>
                <div className="flex items-center gap-3">
                    {usePortfolioData && (
                        <div className="flex gap-1 bg-slate-800/60 p-1 rounded-lg border border-slate-700/50">
                            {['1D', '1W', '1M', '3M', '6M', '1Y'].map((period) => (
                                <button
                                    key={period}
                                    onClick={() => setSelectedPeriod(period)}
                                    className={`px-3 py-1.5 rounded-md text-xs font-bold transition-all ${selectedPeriod === period
                                        ? 'bg-violet-500 text-white shadow-lg shadow-violet-500/20'
                                        : 'text-slate-400 hover:text-slate-200'
                                    }`}
                                >
                                    {period}
                                </button>
                            ))}
                        </div>
                    )}
                    <button
                        onClick={fetchDashboardData}
                        disabled={refreshing}
                        className="flex items-center gap-2 px-4 py-2.5 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50 rounded-xl transition-all text-sm font-medium text-slate-300 disabled:opacity-50"
                    >
                        <RefreshCw className={refreshing ? 'animate-spin' : ''} size={16} />
                        <span>{refreshing ? 'Refreshing...' : 'Refresh'}</span>
                    </button>
                </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-6 gap-4">
                <MetricCard title="Total Return" value={formatPercent(metrics.total_return_pct)} icon={TrendingUp} trend={metrics.total_return_pct >= 0 ? "up" : "down"} color="emerald" />
                <MetricCard title="Win Rate" value={`${toPrecision(metrics.win_rate)}%`} icon={Target} trend="up" color="blue" />
                <MetricCard title="Sharpe Ratio" value={toPrecision(metrics.sharpe_ratio)} icon={Activity} trend="up" color="violet" />
                <MetricCard title="Max Drawdown" value={formatPercent(Math.abs(metrics.max_drawdown))} icon={TrendingDown} trend="down" color="red" />
                <MetricCard title="Profit Factor" value={toPrecision(metrics.profit_factor)} icon={Award} trend="up" color="amber" />
                <MetricCard title="Total Trades" value={metrics.total_trades} icon={List} trend="neutral" color="slate" />
            </div>

            {/* Main Content Grid */}
            <div className="grid grid-cols-3 gap-6">
                {/* Left Column: Backtest Library (1/3 width) */}
                <div className="col-span-1 space-y-6">
                    {/* Search and Filter */}
                    <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-4 shadow-xl">
                        <div className="relative mb-3">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={16} />
                            <input
                                type="text"
                                placeholder="Search backtests..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="w-full pl-10 pr-4 py-2 bg-slate-800/60 border border-slate-700/50 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                            />
                        </div>
                        <div className="flex gap-2">
                            {(['all', 'single', 'multi'] as const).map((type) => (
                                <button
                                    key={type}
                                    onClick={() => setFilterType(type)}
                                    className={`flex-1 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${filterType === type
                                        ? 'bg-violet-500 text-white'
                                        : 'bg-slate-800/60 text-slate-400 hover:text-slate-200'
                                    }`}
                                >
                                    {type.charAt(0).toUpperCase() + type.slice(1)}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Backtest Library */}
                    <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl shadow-xl overflow-hidden">
                        <div className="p-4 border-b border-slate-700/50">
                            <div className="flex items-center gap-2">
                                <History className="text-violet-400" size={20} />
                                <h3 className="text-lg font-semibold text-slate-100">Backtest Library</h3>
                                <span className="ml-auto text-xs font-bold text-slate-500">{filteredHistory.length} total</span>
                            </div>
                        </div>
                        <div className="max-h-[800px] overflow-y-auto custom-scrollbar">
                            {filteredHistory.map((bt) => (
                                <button
                                    key={bt.id}
                                    onClick={() => loadBacktestData(bt)}
                                    className={`w-full text-left p-4 border-b border-slate-800/50 transition-all ${selectedBacktest?.id === bt.id
                                        ? 'bg-violet-500/20 border-l-4 border-l-violet-500'
                                        : 'hover:bg-slate-800/30'
                                    }`}
                                >
                                    <div className="flex items-start justify-between gap-2">
                                        <div className="flex-1 min-w-0">
                                            <p className="font-semibold text-slate-200 truncate">{bt.name || `Backtest #${bt.id}`}</p>
                                            <div className="flex items-center gap-2 mt-1">
                                                <span className={`text-xs px-2 py-0.5 rounded ${bt.backtest_type === 'multi' ? 'bg-blue-500/20 text-blue-400' : 'bg-amber-500/20 text-amber-400'
                                                }`}>
                                                    {bt.backtest_type}
                                                </span>
                                                <span className="text-xs text-slate-500">{bt.symbols.slice(0, 2).join(', ')}</span>
                                                {bt.symbols.length > 2 && <span className="text-xs text-slate-500">+{bt.symbols.length - 2}</span>}
                                            </div>
                                            <p className="text-xs text-slate-600 mt-1">{new Date(bt.created_at || '').toLocaleDateString()}</p>
                                        </div>
                                        <div className="text-right flex-shrink-0">
                                            <p className={`text-sm font-bold ${(bt.total_return_pct || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                {formatPercent(bt.total_return_pct || 0)}
                                            </p>
                                            <p className="text-xs text-slate-500">{bt.total_trades || 0} trades</p>
                                        </div>
                                    </div>
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Right Column: Analytics (2/3 width) */}
                <div className="col-span-2 space-y-6">
                    {hasData ? (
                        <>
                            {/* Equity Curve */}
                            <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                                <div className="flex justify-between items-start mb-6">
                                    <div>
                                        <h3 className="text-xl font-semibold text-slate-100">Equity Curve</h3>
                                        <p className="text-sm text-slate-500 mt-1">
                                            {metrics.initial_capital > 0 && `Initial: ${formatCurrency(metrics.initial_capital)} • `}
                                            Final: {formatCurrency(metrics.final_equity)} • {equityData.length} points
                                        </p>
                                        <div className="flex items-center gap-4 mt-2">
                                            <div className={`flex items-center gap-1.5 text-sm font-semibold ${totalGain >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                {totalGain >= 0 ? <ArrowUpRight size={16} /> : <ArrowDownRight size={16} />}
                                                <span>{formatCurrency(Math.abs(totalGain))}</span>
                                            </div>
                                            <div className="text-sm text-slate-500">
                                                ({totalGainPct >= 0 ? '+' : ''}{totalGainPct.toFixed(2)}%)
                                            </div>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <button
                                            onClick={() => setShowRiskAnalysis(true)}
                                            className="flex items-center gap-2 px-4 py-2 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 rounded-lg text-sm font-medium transition-all border border-blue-500/30"
                                        >
                                            <AlertTriangle size={16} />
                                            Risk Analysis
                                        </button>
                                        <button className="flex items-center gap-2 px-4 py-2 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50 rounded-lg transition-all text-sm font-medium text-slate-300">
                                            <Download size={16} />
                                            Export
                                        </button>
                                    </div>
                                </div>
                                <ResponsiveContainer width="100%" height={280}>
                                    <ComposedChart data={equityData}>
                                        <defs>
                                            <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor={hasIncrease ? "#10b981" : "#ef4444"} stopOpacity={0.3} />
                                                <stop offset="95%" stopColor={hasIncrease ? "#10b981" : "#ef4444"} stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
                                        <XAxis dataKey="date" stroke="#64748b" style={{ fontSize: '11px' }} />
                                        <YAxis stroke="#64748b" tickFormatter={(value) => formatCurrency(value)} style={{ fontSize: '11px' }} />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '12px', padding: '12px' }}
                                            formatter={(value: any) => [formatCurrency(value), 'Equity']}
                                        />
                                        <Area type="monotone" dataKey="value" stroke={hasIncrease ? "#10b981" : "#ef4444"} strokeWidth={3} fillOpacity={1} fill="url(#colorEquity)" />
                                    </ComposedChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Drawdown Chart */}
                            <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                                <div className="mb-6">
                                    <h3 className="text-xl font-semibold text-slate-100">Drawdown Analysis</h3>
                                    <p className="text-xs text-slate-500 mt-1">
                                        Underwater equity • Max DD: {formatPercent(Math.abs(metrics.max_drawdown))}
                                    </p>
                                </div>
                                <ResponsiveContainer width="100%" height={200}>
                                    <AreaChart data={drawdownData}>
                                        <defs>
                                            <linearGradient id="colorDrawdown" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4} />
                                                <stop offset="95%" stopColor="#ef4444" stopOpacity={0.05} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
                                        <XAxis dataKey="timestamp" stroke="#64748b" style={{ fontSize: '11px' }} />
                                        <YAxis stroke="#64748b" tickFormatter={(value) => `${value.toFixed(1)}%`} style={{ fontSize: '11px' }} domain={['dataMin', 0]} />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '12px', padding: '12px' }}
                                            formatter={(value: any) => [`${Number(value).toFixed(2)}%`, 'Drawdown']}
                                        />
                                        <Area type="monotone" dataKey="drawdown" stroke="#ef4444" strokeWidth={2} fillOpacity={1} fill="url(#colorDrawdown)" />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Rolling Sharpe */}
                            <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                                <div className="mb-6">
                                    <h3 className="text-xl font-semibold text-slate-100">Rolling Sharpe Ratio (30-day)</h3>
                                    <p className="text-xs text-slate-500 mt-1">Annualized risk-adjusted return over time</p>
                                </div>
                                <ResponsiveContainer width="100%" height={200}>
                                    <ComposedChart data={rollingSharpe}>
                                        <defs>
                                            <linearGradient id="colorSharpe" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                                                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
                                        <XAxis dataKey="timestamp" stroke="#64748b" style={{ fontSize: '11px' }} />
                                        <YAxis stroke="#64748b" tickFormatter={(value) => value.toFixed(1)} style={{ fontSize: '11px' }} />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '12px', padding: '12px' }}
                                            formatter={(value: any) => [Number(value).toFixed(2), 'Sharpe Ratio']}
                                        />
                                        <Line type="monotone" dataKey={() => 0} stroke="#64748b" strokeWidth={1} strokeDasharray="5 5" dot={false} />
                                        <Area type="monotone" dataKey="sharpe" stroke="#8b5cf6" strokeWidth={2} fillOpacity={1} fill="url(#colorSharpe)" />
                                    </ComposedChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Monthly Returns */}
                            {monthlyReturns.length > 0 && (
                                <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                                    <div className="flex items-center gap-3 mb-6">
                                        <Calendar className="text-violet-400" size={24} />
                                        <div>
                                            <h3 className="text-xl font-semibold text-slate-100">Monthly Returns</h3>
                                            <p className="text-xs text-slate-500 mt-1">Month-by-month performance breakdown</p>
                                        </div>
                                    </div>
                                    <div className="grid grid-cols-2 gap-3">
                                        {monthlyReturns.map((monthData) => {
                                            const [year, month] = monthData.month.split('-');
                                            const monthName = new Date(parseInt(year), parseInt(month) - 1).toLocaleDateString('en-US', { month: 'short', year: 'numeric' });

                                            return (
                                                <div key={monthData.month} className="p-3 bg-slate-800/40 border border-slate-700/50 rounded-xl hover:bg-slate-800/60 transition-all">
                                                    <div className="flex justify-between items-center">
                                                        <div>
                                                            <p className="text-xs text-slate-500 font-medium mb-1">{monthName}</p>
                                                            <p className={`text-lg font-bold ${monthData.return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                                {monthData.return >= 0 ? '+' : ''}{monthData.return.toFixed(2)}%
                                                            </p>
                                                        </div>
                                                        <div className={`p-2 rounded-lg ${monthData.return >= 0 ? 'bg-emerald-500/10' : 'bg-red-500/10'}`}>
                                                            {monthData.return >= 0 ? <TrendingUp className="text-emerald-400" size={16} /> : <TrendingDown className="text-red-400" size={16} />}
                                                        </div>
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                    {/* Monthly Stats */}
                                    <div className="grid grid-cols-4 gap-3 mt-4 pt-4 border-t border-slate-700/50">
                                        <div className="text-center">
                                            <p className="text-xs text-slate-500 mb-1">Best Month</p>
                                            <p className="text-sm font-bold text-emerald-400">+{Math.max(...monthlyReturns.map(m => m.return)).toFixed(2)}%</p>
                                        </div>
                                        <div className="text-center">
                                            <p className="text-xs text-slate-500 mb-1">Worst Month</p>
                                            <p className="text-sm font-bold text-red-400">{Math.min(...monthlyReturns.map(m => m.return)).toFixed(2)}%</p>
                                        </div>
                                        <div className="text-center">
                                            <p className="text-xs text-slate-500 mb-1">Avg Month</p>
                                            <p className="text-sm font-bold text-slate-300">{(monthlyReturns.reduce((sum, m) => sum + m.return, 0) / monthlyReturns.length).toFixed(2)}%</p>
                                        </div>
                                        <div className="text-center">
                                            <p className="text-xs text-slate-500 mb-1">Positive</p>
                                            <p className="text-sm font-bold text-blue-400">{monthlyReturns.filter(m => m.return > 0).length}/{monthlyReturns.length}</p>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Trade Distribution */}
                            {tradeDistribution.length > 0 && (
                                <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                                    <div className="mb-6">
                                        <h3 className="text-xl font-semibold text-slate-100">Trade Distribution</h3>
                                        <p className="text-sm text-slate-500 mt-1">Win/Loss breakdown</p>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <ResponsiveContainer width="55%" height={180}>
                                            <RechartsPieChart>
                                                <Pie data={tradeDistribution} cx="50%" cy="50%" innerRadius={50} outerRadius={70} paddingAngle={5} dataKey="value">
                                                    {tradeDistribution.map((entry, index) => <Cell key={`cell-${index}`} fill={entry.color} />)}
                                                </Pie>
                                                <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '12px', padding: '12px' }} />
                                            </RechartsPieChart>
                                        </ResponsiveContainer>
                                        <div className="space-y-4">
                                            <div>
                                                <div className="flex items-center gap-2 mb-1">
                                                    <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
                                                    <span className="text-sm font-medium text-slate-300">Winning</span>
                                                </div>
                                                <p className="text-2xl font-bold text-emerald-400">{metrics.winning_trades}</p>
                                                <p className="text-xs text-slate-500">Avg: {formatCurrency(metrics.avg_win)}</p>
                                            </div>
                                            <div>
                                                <div className="flex items-center gap-2 mb-1">
                                                    <div className="w-3 h-3 rounded-full bg-red-500"></div>
                                                    <span className="text-sm font-medium text-slate-300">Losing</span>
                                                </div>
                                                <p className="text-2xl font-bold text-red-400">{metrics.losing_trades}</p>
                                                <p className="text-xs text-slate-500">Avg: {formatCurrency(metrics.avg_loss)}</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </>
                    ) : (
                        <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-12 shadow-xl text-center">
                            <Eye className="mx-auto text-slate-600 mb-4" size={48} />
                            <h3 className="text-xl font-semibold text-slate-300 mb-2">No Backtest Selected</h3>
                            <p className="text-slate-500">Select a backtest from the library to view detailed analytics</p>
                        </div>
                    )}
                </div>
            </div>

            {/* Risk Analysis Modal */}
            {showRiskAnalysis && selectedBacktest && (
                <RiskAnalysisModal
                    results={{
                        ...metrics,
                        trades: selectedBacktest.trades || [],
                        equity_curve: selectedBacktest.equity_curve || []
                    } as any}
                    onClose={() => setShowRiskAnalysis(false)}
                />
            )}
        </div>
    );
};

export default Dashboard;
