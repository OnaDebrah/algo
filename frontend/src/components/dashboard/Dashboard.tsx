import MetricCard from "@/components/backtest/MetricCard";
import {
    Activity,
    ArrowDownRight,
    ArrowUpRight,
    Award,
    BarChart3,
    Calendar,
    Clock,
    Download,
    History,
    List,
    RefreshCw,
    Sparkles,
    Target,
    TrendingDown,
    TrendingUp,
    TrendingUp as TrendingUpIcon
} from "lucide-react";
import {
    Area,
    Bar,
    BarChart,
    CartesianGrid,
    Cell,
    ComposedChart,
    Pie,
    PieChart as RechartsPieChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from "recharts";
import {formatCurrency, formatPercent, toPrecision} from "@/utils/formatters";

import {useEffect, useState} from "react";
import {analytics, backtest, portfolio, strategy} from "@/utils/api";
import {BacktestHistoryItem, Portfolio, PortfolioTrade, StrategyInfo} from "@/types/all_types";

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
    const [latestBacktest, setLatestBacktest] = useState<BacktestHistoryItem | null>(null);
    const [performanceDistribution, setPerformanceDistribution] = useState<any[]>([]);
    const [selectedPeriod, setSelectedPeriod] = useState('1M');
    const [tradeDistribution, setTradeDistribution] = useState<any[]>([]);
    const [usePortfolioData, setUsePortfolioData] = useState(false);

    const fetchDashboardData = async () => {
        try {
            setRefreshing(true);

            // 1. Fetch Backtest History
            const historyRes = await backtest.getHistory({ limit: 10 });
            console.log('📊 Backtest History:', historyRes);

            if (historyRes && historyRes.length > 0) {
                setBacktestHistory(historyRes);
                const latest = historyRes[0];
                setLatestBacktest(latest);

                // Try to get detailed data
                let detailedData = latest;
                if (!latest.equity_curve || !latest.trades) {
                    console.log('📥 Fetching detailed backtest data...');
                    try {
                        detailedData = await backtest.getDetails(latest.id);
                    } catch (e) {
                        console.error('Failed to fetch details:', e);
                    }
                }

                // Set metrics from backtest
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
                        profit_factor: 0, // Calculate below
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
                        value: p.equity,
                        cash: p.cash || 0
                    }));
                    setEquityData(curve);

                    // Set trade distribution for pie chart
                    const winning = detailedData.trades?.filter((t: any) => t.profit > 0).length || 0;
                    const losing = detailedData.trades?.filter((t: any) => t.profit < 0).length || 0;
                    setTradeDistribution([
                        { name: 'Winning', value: winning, color: '#10b981' },
                        { name: 'Losing', value: losing, color: '#ef4444' }
                    ]);
                }

                // Performance distribution
                if (historyRes.length > 1) {
                    const distribution = historyRes
                        .filter(b => b.total_return_pct !== null)
                        .map(b => ({
                            name: b.name || `BT #${b.id}`,
                            return: b.total_return_pct || 0,
                            sharpe: b.sharpe_ratio || 0
                        }))
                        .slice(0, 6);
                    setPerformanceDistribution(distribution);
                }

                setUsePortfolioData(false);
            } else {
                console.log('⚠️ No backtest history, using portfolio data');
                setUsePortfolioData(true);
            }

            // 2. Fetch Portfolio Analytics (fallback or supplementary)
            const portfoliosRes: Portfolio[] = await portfolio.list();
            if (portfoliosRes && portfoliosRes.length > 0) {
                const pid = portfoliosRes[0].id;

                // Use portfolio analytics if no backtest data
                if (usePortfolioData || historyRes.length === 0) {
                    console.log('📈 Fetching portfolio analytics...');
                    try {
                        const perfRes = await analytics.getPerformance(pid, { period: selectedPeriod });
                        console.log('Portfolio Performance:', perfRes);

                        if (perfRes.metrics) {
                            setMetrics(perfRes.metrics as PerformanceMetrics);
                        }

                        if (perfRes.equity_curve && perfRes.equity_curve.length > 0) {
                            const curve = perfRes.equity_curve.map((p: any) => ({
                                date: new Date(p.timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }),
                                value: p.equity,
                                cash: p.cash || 0
                            }));
                            setEquityData(curve);
                        }
                    } catch (e) {
                        console.error('Failed to fetch portfolio analytics:', e);
                    }
                }

                // Always fetch trades
                const tradesRes = await portfolio.getTrades(pid);
                if (tradesRes) {
                    setRecentTrades(tradesRes.slice(0, 10).map((t: PortfolioTrade) => ({
                        symbol: t.symbol,
                        strategy: t.strategy || 'Manual',
                        profit: t.profit || 0,
                        time: t.executed_at ? new Date(t.executed_at).toLocaleTimeString() : 'Just now',
                        date: t.executed_at ? new Date(t.executed_at).toLocaleDateString() : 'Today',
                        status: 'closed'
                    })));
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

            setLoading(false);
            setRefreshing(false);
        } catch (error) {
            console.error("Failed to load dashboard data", error);
            setLoading(false);
            setRefreshing(false);
        }
    };

    useEffect(() => {
        fetchDashboardData();
    }, [selectedPeriod]);

    if (loading) return (
        <div className="flex flex-col items-center justify-center h-[60vh]">
            <div className="relative">
                <div className="w-16 h-16 border-4 border-violet-500/20 border-t-violet-500 rounded-full animate-spin"></div>
                <Sparkles className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-violet-400" size={24} />
            </div>
            <p className="text-lg font-semibold text-slate-300 mt-6">Synchronizing Performance Data</p>
            <p className="text-sm text-slate-500 mt-2">Analyzing market insights...</p>
        </div>
    );

    const hasData = equityData.length > 0;
    const hasIncrease = equityData.length > 1 && equityData[equityData.length - 1].value > equityData[0].value;
    const totalGain = hasData ? equityData[equityData.length - 1].value - equityData[0].value : 0;
    const totalGainPct = hasData && equityData[0].value > 0
        ? ((equityData[equityData.length - 1].value - equityData[0].value) / equityData[0].value) * 100
        : 0;

    return (
        <div className="space-y-6">
            {/* Header with Refresh and Period Selector */}
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-slate-100 flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center shadow-lg shadow-violet-500/20">
                            <BarChart3 className="text-white" size={22} strokeWidth={2.5} />
                        </div>
                        Performance Hub
                    </h1>
                    <p className="text-sm text-slate-500 mt-2 ml-13">
                        {latestBacktest
                            ? `Latest: ${latestBacktest.name || 'Unnamed Backtest'} • ${latestBacktest.symbols.join(', ')}`
                            : usePortfolioData
                                ? 'Portfolio Performance Analytics'
                                : 'Real-time analytics and strategy performance'
                        }
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    {usePortfolioData && (
                        <div className="flex gap-1 bg-slate-800/60 p-1 rounded-lg border border-slate-700/50">
                            {['1D', '1W', '1M', '3M', '6M', '1Y'].map((period) => (
                                <button
                                    key={period}
                                    onClick={() => setSelectedPeriod(period)}
                                    className={`px-3 py-1.5 rounded-md text-xs font-bold transition-all ${
                                        selectedPeriod === period
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
                <MetricCard
                    title="Total Return"
                    value={formatPercent(metrics.total_return_pct)}
                    icon={TrendingUp}
                    trend={metrics.total_return_pct >= 0 ? "up" : "down"}
                    color="emerald"
                />
                <MetricCard
                    title="Win Rate"
                    value={`${toPrecision(metrics.win_rate)}%`}
                    icon={Target}
                    trend="up"
                    color="blue"
                />
                <MetricCard
                    title="Sharpe Ratio"
                    value={toPrecision(metrics.sharpe_ratio)}
                    trend="up"
                    color="violet"
                />
                <MetricCard
                    title="Max Drawdown"
                    value={formatPercent(Math.abs(metrics.max_drawdown))}
                    trend="down"
                    color="red"
                />
                <MetricCard
                    title="Profit Factor"
                    value={toPrecision(metrics.profit_factor)}
                    trend="up"
                    color="amber"
                />
                <MetricCard
                    title="Total Trades"
                    value={metrics.total_trades}
                    trend="neutral"
                    color="slate"
                />
            </div>

            {/* Main Performance Chart */}
            {hasData ? (
                <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                    <div className="flex justify-between items-start mb-6">
                        <div>
                            <h3 className="text-xl font-semibold text-slate-100">Portfolio Equity Curve</h3>
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
                        <div className="flex items-center gap-4">
                            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${
                                hasIncrease
                                    ? 'bg-emerald-500/10 border border-emerald-500/30'
                                    : 'bg-red-500/10 border border-red-500/30'
                            }`}>
                                {hasIncrease ? (
                                    <ArrowUpRight className="text-emerald-400" size={16} strokeWidth={2.5} />
                                ) : (
                                    <ArrowDownRight className="text-red-400" size={16} strokeWidth={2.5} />
                                )}
                                <span className={`text-sm font-bold ${hasIncrease ? 'text-emerald-400' : 'text-red-400'}`}>
                                    {hasIncrease ? 'Trending Up' : 'Trending Down'}
                                </span>
                            </div>
                            <button className="flex items-center space-x-2 px-4 py-2 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50 rounded-lg transition-all text-sm font-medium text-slate-300">
                                <Download size={16} strokeWidth={2} />
                                <span>Export</span>
                            </button>
                        </div>
                    </div>
                    <ResponsiveContainer width="100%" height={320}>
                        <ComposedChart data={equityData}>
                            <defs>
                                <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor={hasIncrease ? "#10b981" : "#ef4444"} stopOpacity={0.3} />
                                    <stop offset="95%" stopColor={hasIncrease ? "#10b981" : "#ef4444"} stopOpacity={0} />
                                </linearGradient>
                                <linearGradient id="colorCash" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2} />
                                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
                            <XAxis
                                dataKey="date"
                                stroke="#64748b"
                                style={{ fontSize: '12px', fontWeight: 500 }}
                                tick={{ fill: '#94a3b8' }}
                            />
                            <YAxis
                                stroke="#64748b"
                                tickFormatter={(value) => formatCurrency(value)}
                                style={{ fontSize: '12px', fontWeight: 500 }}
                                tick={{ fill: '#94a3b8' }}
                            />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: '#1e293b',
                                    border: '1px solid #334155',
                                    borderRadius: '12px',
                                    padding: '12px',
                                    boxShadow: '0 10px 40px rgba(0, 0, 0, 0.3)'
                                }}
                                formatter={(value: number | undefined, name: string | undefined) => {
                                    if (name === 'value') return [formatCurrency(Number(value || 0)), 'Portfolio Value'];
                                    if (name === 'cash') return [formatCurrency(Number(value || 0)), 'Cash'];
                                    return [value, name];
                                }}
                                labelStyle={{ color: '#94a3b8', fontWeight: 600, marginBottom: '4px' }}
                            />
                            <Area
                                type="monotone"
                                dataKey="value"
                                stroke={hasIncrease ? "#10b981" : "#ef4444"}
                                strokeWidth={3}
                                fillOpacity={1}
                                fill="url(#colorEquity)"
                            />
                            {equityData[0]?.cash !== undefined && (
                                <Area
                                    type="monotone"
                                    dataKey="cash"
                                    stroke="#3b82f6"
                                    strokeWidth={2}
                                    fillOpacity={1}
                                    fill="url(#colorCash)"
                                    strokeDasharray="5 5"
                                />
                            )}
                        </ComposedChart>
                    </ResponsiveContainer>
                    <div className="flex items-center justify-center gap-6 mt-4">
                        <div className="flex items-center gap-2">
                            <div className={`w-3 h-3 rounded-full ${hasIncrease ? 'bg-emerald-500' : 'bg-red-500'}`}></div>
                            <span className="text-xs text-slate-400 font-medium">Portfolio Value</span>
                        </div>
                        {equityData[0]?.cash !== undefined && (
                            <div className="flex items-center gap-2">
                                <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                                <span className="text-xs text-slate-400 font-medium">Cash</span>
                            </div>
                        )}
                    </div>
                </div>
            ) : (
                <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-12 shadow-xl">
                    <div className="text-center">
                        <div className="w-20 h-20 rounded-full bg-slate-800/50 border-2 border-dashed border-slate-700 mx-auto flex items-center justify-center mb-4">
                            <BarChart3 className="text-slate-600" size={32} />
                        </div>
                        <h3 className="text-xl font-semibold text-slate-300 mb-2">No Performance Data Yet</h3>
                        <p className="text-slate-500 max-w-md mx-auto">
                            Run a backtest or execute trades to populate your performance hub visualization.
                        </p>
                    </div>
                </div>
            )}

            {/* Analytics Grid: Performance Distribution + Trade Distribution */}
            <div className="grid grid-cols-2 gap-6">
                {/* Performance Distribution */}
                {performanceDistribution.length > 0 && (
                    <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                        <div className="mb-6">
                            <h3 className="text-xl font-semibold text-slate-100">Backtest Performance</h3>
                            <p className="text-sm text-slate-500 mt-1">Returns across recent backtests</p>
                        </div>
                        <ResponsiveContainer width="100%" height={250}>
                            <BarChart data={performanceDistribution}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
                                <XAxis
                                    dataKey="name"
                                    stroke="#64748b"
                                    style={{ fontSize: '11px', fontWeight: 500 }}
                                    tick={{ fill: '#94a3b8' }}
                                />
                                <YAxis
                                    stroke="#64748b"
                                    tickFormatter={(value) => `${value}%`}
                                    style={{ fontSize: '12px', fontWeight: 500 }}
                                    tick={{ fill: '#94a3b8' }}
                                />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#1e293b',
                                        border: '1px solid #334155',
                                        borderRadius: '12px',
                                        padding: '12px'
                                    }}
                                    formatter={(value: any) => [`${value.toFixed(2)}%`, 'Return']}
                                    labelStyle={{ color: '#94a3b8', fontWeight: 600, marginBottom: '4px' }}
                                />
                                <Bar dataKey="return" radius={[8, 8, 0, 0]}>
                                    {performanceDistribution.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.return >= 0 ? '#10b981' : '#ef4444'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                )}

                {/* Trade Win/Loss Distribution */}
                {tradeDistribution.length > 0 && (
                    <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                        <div className="mb-6">
                            <h3 className="text-xl font-semibold text-slate-100">Trade Distribution</h3>
                            <p className="text-sm text-slate-500 mt-1">Winning vs Losing trades</p>
                        </div>
                        <div className="flex items-center justify-between">
                            <ResponsiveContainer width="60%" height={200}>
                                <RechartsPieChart>
                                    <Pie
                                        data={tradeDistribution}
                                        cx="50%"
                                        cy="50%"
                                        innerRadius={60}
                                        outerRadius={80}
                                        paddingAngle={5}
                                        dataKey="value"
                                    >
                                        {tradeDistribution.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.color} />
                                        ))}
                                    </Pie>
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: '#1e293b',
                                            border: '1px solid #334155',
                                            borderRadius: '12px',
                                            padding: '12px'
                                        }}
                                    />
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
            </div>

            {/* History and Strategies Grid */}
            <div className="grid grid-cols-2 gap-6">
                {/* History Panel */}
                <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl shadow-xl overflow-hidden">
                    <div className="p-6 pb-4 border-b border-slate-700/50">
                        <div className="flex items-center justify-between mb-4">
                            <div className="flex items-center gap-2">
                                <History className="text-violet-400" size={20} />
                                <h3 className="text-lg font-semibold text-slate-100">Activity History</h3>
                            </div>
                            <div className="flex items-center gap-2 px-2.5 py-1 bg-slate-800/50 rounded-lg border border-slate-700/30">
                                <Clock size={12} className="text-slate-500" />
                                <span className="text-xs font-medium text-slate-400">Last 10</span>
                            </div>
                        </div>

                        <div className="flex gap-2 bg-slate-800/40 p-1 rounded-lg">
                            <button
                                onClick={() => setActiveHistoryTab('backtests')}
                                className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-md font-medium text-sm transition-all ${
                                    activeHistoryTab === 'backtests'
                                        ? 'bg-violet-500 text-white shadow-lg shadow-violet-500/20'
                                        : 'text-slate-400 hover:text-slate-200'
                                }`}
                            >
                                <List size={16} />
                                Backtests
                                <span className="text-xs bg-white/10 px-1.5 py-0.5 rounded">{backtestHistory.length}</span>
                            </button>
                            <button
                                onClick={() => setActiveHistoryTab('trades')}
                                className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-md font-medium text-sm transition-all ${
                                    activeHistoryTab === 'trades'
                                        ? 'bg-violet-500 text-white shadow-lg shadow-violet-500/20'
                                        : 'text-slate-400 hover:text-slate-200'
                                }`}
                            >
                                <TrendingUpIcon size={16} />
                                Trades
                                <span className="text-xs bg-white/10 px-1.5 py-0.5 rounded">{recentTrades.length}</span>
                            </button>
                        </div>
                    </div>

                    <div className="p-6 pt-4">
                        <div className="space-y-2.5 max-h-[400px] overflow-y-auto pr-2 custom-scrollbar">
                            {activeHistoryTab === 'trades' ? (
                                recentTrades.length > 0 ? (
                                    recentTrades.map((trade, i) => (
                                        <div
                                            key={i}
                                            className="group relative flex justify-between items-center p-4 bg-slate-800/30 hover:bg-slate-800/50 border border-slate-700/30 hover:border-slate-600/50 rounded-xl transition-all cursor-pointer"
                                        >
                                            <div className="flex items-center space-x-3">
                                                <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 border border-violet-500/30 flex items-center justify-center">
                                                    <span className="font-bold text-sm text-violet-300">{trade.symbol.slice(0, 2)}</span>
                                                </div>
                                                <div>
                                                    <p className="font-semibold text-slate-200 flex items-center gap-2">
                                                        {trade.symbol}
                                                        <span className="text-xs text-slate-600">•</span>
                                                        <span className="text-xs text-slate-500 font-normal">{trade.strategy}</span>
                                                    </p>
                                                    <div className="flex items-center gap-1.5 mt-0.5">
                                                        <Calendar size={10} className="text-slate-600" />
                                                        <p className="text-xs text-slate-500 font-medium">{trade.date}</p>
                                                    </div>
                                                </div>
                                            </div>
                                            <div className="text-right">
                                                <p className={`font-bold text-base ${trade.profit >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                    {trade.profit >= 0 ? '+' : ''}{formatCurrency(trade.profit)}
                                                </p>
                                            </div>
                                        </div>
                                    ))
                                ) : (
                                    <div className="p-12 text-center">
                                        <TrendingUpIcon className="text-slate-600 mx-auto mb-3" size={32} />
                                        <p className="text-slate-600 font-medium">No recent trades</p>
                                    </div>
                                )
                            ) : (
                                backtestHistory.length > 0 ? (
                                    backtestHistory.map((item) => (
                                        <div
                                            key={item.id}
                                            className="group relative flex justify-between items-center p-4 bg-slate-800/30 hover:bg-slate-800/50 border border-slate-700/30 hover:border-slate-600/50 rounded-xl transition-all cursor-pointer"
                                        >
                                            <div className="flex items-center space-x-3">
                                                <div className={`w-11 h-11 rounded-xl bg-gradient-to-br flex items-center justify-center border ${
                                                    item.backtest_type === 'multi'
                                                        ? 'from-blue-500/20 to-cyan-500/20 border-blue-500/30'
                                                        : 'from-amber-500/20 to-orange-500/20 border-amber-500/30'
                                                }`}>
                                                    <List className={item.backtest_type === 'multi' ? 'text-blue-300' : 'text-amber-300'} size={20} />
                                                </div>
                                                <div>
                                                    <p className="font-semibold text-slate-200">{item.name || `Backtest #${item.id}`}</p>
                                                    <div className="flex items-center gap-1.5 mt-0.5">
                                                        <span className="text-xs font-medium text-slate-500 capitalize">{item.backtest_type}</span>
                                                        <span className="text-xs text-slate-700">•</span>
                                                        <span className="text-xs text-slate-500">{item.symbols.slice(0, 2).join(', ')}</span>
                                                        {item.symbols.length > 2 && <span className="text-xs text-slate-500">+{item.symbols.length - 2}</span>}
                                                    </div>
                                                </div>
                                            </div>
                                            <div className="text-right">
                                                <p className={`font-bold text-base ${(item.total_return_pct || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                    {formatPercent(item.total_return_pct || 0)}
                                                </p>
                                                <p className="text-xs text-slate-500 font-medium">{item.total_trades || 0} trades</p>
                                            </div>
                                        </div>
                                    ))
                                ) : (
                                    <div className="p-12 text-center">
                                        <List className="text-slate-600 mx-auto mb-3" size={32} />
                                        <p className="text-slate-600 font-medium">No backtest history</p>
                                    </div>
                                )
                            )}
                        </div>
                    </div>
                </div>

                {/* Strategies */}
                <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl shadow-xl overflow-hidden">
                    <div className="p-6 border-b border-slate-700/50">
                        <div className="flex justify-between items-center">
                            <div className="flex items-center gap-2">
                                <Activity className="text-emerald-400" size={20} />
                                <h3 className="text-lg font-semibold text-slate-100">Available Strategies</h3>
                            </div>
                            <div className="flex items-center gap-2 px-2.5 py-1 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
                                <Award size={12} className="text-emerald-400" />
                                <span className="text-xs font-bold text-emerald-400">{activeStrategies.length} TOTAL</span>
                            </div>
                        </div>
                    </div>

                    <div className="p-6 pt-4">
                        <div className="space-y-2.5">
                            {activeStrategies.length > 0 ? (
                                activeStrategies.map((strat, i) => (
                                    <div
                                        key={i}
                                        className="group relative flex justify-between items-center p-4 bg-slate-800/30 hover:bg-slate-800/50 border border-slate-700/30 hover:border-slate-600/50 rounded-xl transition-all"
                                    >
                                        <div className="flex-1">
                                            <div className="flex items-center space-x-2 mb-1.5">
                                                <p className="font-semibold text-slate-200">{strat.name}</p>
                                                <span className={`text-xs font-bold px-2.5 py-0.5 rounded-full ${
                                                    strat.status === 'active'
                                                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                                                        : 'bg-slate-700/50 text-slate-400 border border-slate-600/30'
                                                }`}>
                                                    {strat.status.toUpperCase()}
                                                </span>
                                            </div>
                                            {strat.description && (
                                                <p className="text-xs text-slate-500 mt-1">{strat.description}</p>
                                            )}
                                        </div>
                                        <div className="text-right ml-4">
                                            <div className="flex items-center justify-end gap-1.5">
                                                {strat.performance >= 0 ? (
                                                    <ArrowUpRight className="text-emerald-400" size={16} strokeWidth={2.5} />
                                                ) : (
                                                    <ArrowDownRight className="text-red-400" size={16} strokeWidth={2.5} />
                                                )}
                                                <p className={`font-bold text-lg ${strat.performance >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                    {formatPercent(strat.performance)}
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                ))
                            ) : (
                                <div className="p-12 text-center">
                                    <Activity className="text-slate-600 mx-auto mb-3" size={32} />
                                    <p className="text-slate-600 font-medium">No strategies available</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
