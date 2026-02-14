'use client';

import MetricCard from "@/components/backtest/MetricCard";
import {
    AlertTriangle,
    ArrowDownRight,
    ArrowUpRight,
    Calendar,
    Download,
    Eye,
    History,
    List,
    RefreshCw,
    Rocket,
    Search,
    TrendingDown,
    TrendingUp,
    Upload
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
import {backtest, live, marketplace} from "@/utils/api";
import {BacktestHistoryItem, DeploymentConfig, EquityCurvePoint} from "@/types/all_types";
import RiskAnalysisModal from "@/components/backtest/RiskAnalysisModal";
import SocialFeed from "@/components/dashboard/SocialFeed";
import LightweightChart from "@/components/charts/LightweightChart";
import {motion} from "framer-motion";
import DeploymentModal from "@/components/strategies/DeploymentModel";
import PublishModal from "@/components/strategies/PublishModel";
import {PublishData} from "@/types/publish";
import {
    calculateAvgLoss,
    calculateAvgProfit,
    calculateAvgWin,
    calculateLosingTrades,
    calculateProfitFactor,
    calculateWinningTrades
} from "@/components/dashboard/MetricsCalculator";

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
    const [backtestHistory, setBacktestHistory] = useState<BacktestHistoryItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [selectedBacktest, setSelectedBacktest] = useState<BacktestHistoryItem | null>(null);
    const [tradeDistribution, setTradeDistribution] = useState<any[]>([]);
    const [searchQuery, setSearchQuery] = useState('');
    const [filterType, setFilterType] = useState<'all' | 'single' | 'multi'>('all');
    const [showRiskAnalysis, setShowRiskAnalysis] = useState(false);

    const [showDeployModal, setShowDeployModal] = useState(false);
    const [showPublishModal, setShowPublishModal] = useState(false);
    const [selectedBacktestForAction, setSelectedBacktestForAction] = useState<BacktestHistoryItem | null>(null);

    const monthlyReturns = useMemo(() => {
        if (!equityData || equityData.length === 0) return [];

        const monthlyData: { [key: string]: { start: number; end: number } } = {};

        equityData.forEach((point) => {
            const timestamp = point.timestamp || point.date;
            const date = new Date(timestamp);
            const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
            const value = point.equity || point.value || 0;

            if (!monthlyData[monthKey]) {
                monthlyData[monthKey] = {start: value, end: value};
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
                value: value
            };
        });
    }, [equityData]);

    const rollingSharpe = useMemo(() => {
        if (!equityData || equityData.length < 30) return [];

        const returns: number[] = [];
        for (let i = 1; i < equityData.length; i++) {
            const prevValue = equityData[i - 1].equity || equityData[i - 1].value || 0;
            const currentValue = equityData[i].equity || equityData[i].value || 0;
            const dailyReturn = (currentValue - prevValue) / prevValue;
            returns.push(dailyReturn);
        }

        const rollingData: any[] = [];
        const windowSize = 30;

        for (let i = windowSize; i < returns.length; i++) {
            const windowReturns = returns.slice(i - windowSize, i);
            const mean = windowReturns.reduce((a, b) => a + b, 0) / windowSize;
            const variance = windowReturns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / windowSize;
            const std = Math.sqrt(variance);
            const sharpe = std !== 0 ? (mean / std) * Math.sqrt(252) : 0;

            rollingData.push({
                timestamp: equityData[i].timestamp || equityData[i].date,
                sharpe: sharpe
            });
        }

        return rollingData;
    }, [equityData]);

    const handleDeploy = (backtest: BacktestHistoryItem) => {
        setSelectedBacktestForAction(backtest);
        setShowDeployModal(true);
    };

    const handlePublish = (backtest: BacktestHistoryItem) => {
        if ((backtest.sharpe_ratio || 0) < 1.0) {
            alert('Strategy must have Sharpe Ratio ≥ 1.0 to publish to marketplace');
            return;
        }

        if ((backtest.total_return_pct || 0) < 10) {
            alert('Strategy must have return ≥ 10% to publish to marketplace');
            return;
        }

        setSelectedBacktestForAction(backtest);
        setShowPublishModal(true);
    };

    const onDeployConfirm = async (config: DeploymentConfig) => {
        try {
            await live.deploy(config);

            alert('Strategy deployed successfully! Check Live Execution page.');

            setShowDeployModal(false);
            setSelectedBacktestForAction(null);

        } catch (error: any) {
            console.error('Failed to deploy strategy:', error);
            alert(error.response?.data?.detail || 'Failed to deploy strategy');
        }
    };

    const onPublishConfirm = async (publishData: PublishData) => {
        if (!selectedBacktestForAction) return;

        try {
            await marketplace.publish({
                backtest_id: selectedBacktestForAction.id,
                strategy_key: selectedBacktestForAction.strategy_key || selectedBacktestForAction.strategy_config?.strategy_key || '',
                ...publishData
            });

            alert('Strategy submitted for review! You\'ll be notified when approved.');

            setShowPublishModal(false);
            setSelectedBacktestForAction(null);
        } catch (error: any) {
            console.error('Failed to publish strategy:', error);
            alert(error.response?.data?.detail || 'Failed to publish strategy');
        }
    };

    useEffect(() => {
        loadDashboardData();
    }, []);

    const loadDashboardData = async () => {
        setLoading(true);
        try {
            const backtestHistoryData = await backtest.getHistory({
                limit: 100,
                status: 'completed'
            });

            if (backtestHistoryData && backtestHistoryData.length > 0) {
                setBacktestHistory(backtestHistoryData);

                // Auto-select most recent backtest
                const mostRecent = backtestHistoryData[0];
                setSelectedBacktest(mostRecent);

                await loadBacktestDetails(mostRecent);
            } else {
                setBacktestHistory([]);

                setMetrics({
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
            }

        } catch (error) {
            console.error('Error loading dashboard data:', error);
            setBacktestHistory([]);
        } finally {
            setLoading(false);
        }
    };

    const loadBacktestDetails = async (backtestItem: BacktestHistoryItem) => {
        try {
            const details = await backtest.getDetails(backtestItem.id);

            if (details) {
                setMetrics({
                    total_return: (details.final_equity || details.initial_capital) - details.initial_capital,
                    total_return_pct: details.total_return_pct || 0,
                    win_rate: details.win_rate || 0,
                    sharpe_ratio: details.sharpe_ratio || 0,
                    max_drawdown: details.max_drawdown || 0,
                    total_trades: details.total_trades || 0,

                    winning_trades: details.winning_trades || calculateWinningTrades(details.trades),
                    losing_trades: details.losing_trades || calculateLosingTrades(details.trades),
                    avg_win: details.avg_win || calculateAvgWin(details.trades),
                    avg_loss: details.avg_loss || calculateAvgLoss(details.trades),

                    avg_profit: calculateAvgProfit(details.trades),
                    total_profit: (details.final_equity || 0) - details.initial_capital,
                    profit_factor: details.profit_factor || calculateProfitFactor(details.trades),
                    final_equity: details.final_equity || details.initial_capital,
                    initial_capital: details.initial_capital
                });

                const curve = (details.equity_curve || []).map((p: EquityCurvePoint) => ({
                    date: new Date(p.timestamp).toLocaleDateString(undefined, {month: 'short', day: 'numeric'}),
                    timestamp: p.timestamp,
                    value: p.equity,
                    equity: p.equity,
                    cash: p.cash || 0
                }));
                setEquityData(curve);

                const winningCount = details.winning_trades || calculateWinningTrades(details.trades);
                const losingCount = details.losing_trades || calculateLosingTrades(details.trades);

                if (winningCount > 0 || losingCount > 0) {
                    setTradeDistribution([
                        {name: 'Winning', value: winningCount, color: '#10b981'},
                        {name: 'Losing', value: losingCount, color: '#ef4444'}
                    ]);
                }
            }
        } catch (error) {
            console.error('Error loading backtest details:', error);
        }
    };

    const handleBacktestSelect = async (backtest: BacktestHistoryItem) => {
        setSelectedBacktest(backtest);
        await loadBacktestDetails(backtest);
    };

    const handleRefresh = async () => {
        setRefreshing(true);
        await loadDashboardData();
        setRefreshing(false);
    };

    const filteredBacktests = useMemo(() => {
        let filtered = backtestHistory;

        if (searchQuery) {
            filtered = filtered.filter((bt: BacktestHistoryItem) =>
                (bt.name?.toLowerCase() || '').includes(searchQuery.toLowerCase()) ||
                bt.symbols.some(s => s.toLowerCase().includes(searchQuery.toLowerCase()))
            );
        }

        if (filterType !== 'all') {
            filtered = filtered.filter((bt: BacktestHistoryItem) => bt.backtest_type === filterType);
        }

        return filtered;
    }, [backtestHistory, searchQuery, filterType]);

    const hasData = equityData.length > 0;
    const totalGain = metrics.final_equity - metrics.initial_capital;
    const totalGainPct = metrics.initial_capital > 0 ? (totalGain / metrics.initial_capital) * 100 : 0;
    const hasIncrease = totalGain >= 0;

    if (loading) {
        return (
            <div className="flex items-center justify-center h-screen">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-500 mx-auto"></div>
                    <p className="mt-4 text-slate-400">Loading dashboard...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-8 pb-12">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-4xl font-black text-slate-100 tracking-tight">Dashboard</h1>
                    <p className="text-slate-400 mt-1">Overview of your trading performance and analytics</p>
                </div>
                <button
                    onClick={handleRefresh}
                    disabled={refreshing}
                    className="flex items-center gap-2 px-4 py-2 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50 rounded-lg transition-all text-sm font-medium text-slate-300"
                >
                    <RefreshCw size={16} className={refreshing ? 'animate-spin' : ''}/>
                    {refreshing ? 'Refreshing...' : 'Refresh'}
                </button>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <MetricCard
                    title="Total Return"
                    value={formatPercent(metrics.total_return_pct)}
                    trend={metrics.total_return_pct >= 0 ? 'up' : 'down'}
                    color="emerald"
                />
                <MetricCard
                    title="Win Rate"
                    value={formatPercent(metrics.win_rate)}
                    trend={metrics.win_rate >= 50 ? 'up' : 'down'}
                    color="blue"
                />
                <MetricCard
                    title="Sharpe Ratio"
                    value={toPrecision(metrics.sharpe_ratio, 2)}
                    trend={metrics.sharpe_ratio >= 1.0 ? 'up' : 'down'}
                    color="violet"
                />
                <MetricCard
                    title="Max Drawdown"
                    value={formatPercent(Math.abs(metrics.max_drawdown))}
                    trend={Math.abs(metrics.max_drawdown) <= 10 ? 'up' : 'down'}
                    color="red"
                />
            </div>

            {/* Main Content Grid */}
            <div className="grid grid-cols-12 gap-6">
                {/* Left Column: Backtest Library (3/12 width) */}
                <div className="col-span-12 lg:col-span-3 h-[calc(100vh-140px)] sticky top-24">
                    <motion.div
                        initial={{opacity: 0, x: -20}}
                        animate={{opacity: 1, x: 0}}
                        transition={{duration: 0.5}}
                        className="glass-panel rounded-2xl h-full flex flex-col overflow-hidden"
                    >
                        {/* Header */}
                        <div className="p-6 border-b border-slate-700/50">
                            <div className="flex items-center gap-3 mb-4">
                                <div className="p-2 bg-indigo-500/10 rounded-xl">
                                    <History className="text-indigo-400" size={20}/>
                                </div>
                                <div>
                                    <h3 className="text-lg font-bold text-slate-100">Backtest Library</h3>
                                    <p className="text-xs text-slate-500">{backtestHistory.length} total</p>
                                </div>
                            </div>

                            {/* Search */}
                            <div className="relative">
                                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={16}/>
                                <input
                                    type="text"
                                    placeholder="Search backtests..."
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    className="w-full pl-10 pr-4 py-2 bg-slate-800/60 border border-slate-700/50 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:border-indigo-500 focus:outline-none"
                                />
                            </div>

                            {/* Filter */}
                            <div className="flex gap-2 mt-3">
                                {['all', 'single', 'multi'].map((type) => (
                                    <button
                                        key={type}
                                        onClick={() => setFilterType(type as any)}
                                        className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                                            filterType === type
                                                ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/30'
                                                : 'bg-slate-800/40 text-slate-500 hover:bg-slate-700/40'
                                        }`}
                                    >
                                        {type === 'all' ? 'All' : type === 'single' ? 'Single' : 'Multi'}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Backtest List */}
                        <div className="flex-1 overflow-y-auto p-3 space-y-2">
                            {filteredBacktests.length === 0 ? (
                                <div className="text-center py-12 text-slate-500">
                                    <List className="mx-auto mb-3" size={32}/>
                                    <p className="text-sm">No backtests found</p>
                                </div>
                            ) : (
                                filteredBacktests.map((bt) => (
                                    <div
                                        key={bt.id}
                                        className={`p-4 rounded-xl transition-all cursor-pointer ${
                                            selectedBacktest?.id === bt.id
                                                ? 'bg-violet-500/20 border-2 border-violet-500/50'
                                                : 'bg-slate-800/40 border border-slate-700/50 hover:bg-slate-700/40'
                                        }`}
                                    >
                                        {/* Existing backtest info */}
                                        <div
                                            onClick={() => handleBacktestSelect(bt)}
                                            className="flex items-start justify-between gap-2 mb-3"
                                        >
                                            <div className="flex-1 min-w-0">
                                                <p className="font-semibold text-slate-200 truncate">
                                                    {bt.name || `Backtest #${bt.id}`}
                                                </p>
                                                <div className="flex items-center gap-2 mt-1">
                                                    <span className={`text-xs px-2 py-0.5 rounded ${
                                                        bt.backtest_type === 'multi'
                                                            ? 'bg-blue-500/20 text-blue-400'
                                                            : 'bg-amber-500/20 text-amber-400'
                                                    }`}>
                                                        {bt.backtest_type}
                                                    </span>
                                                    <span className="text-xs text-slate-500">
                                                        {bt.symbols.slice(0, 2).join(', ')}
                                                    </span>
                                                    {bt.symbols.length > 2 && (
                                                        <span className="text-xs text-slate-500">
                                                            +{bt.symbols.length - 2}
                                                        </span>
                                                    )}
                                                </div>
                                                <p className="text-xs text-slate-600 mt-1">
                                                    {new Date(bt.created_at || '').toLocaleDateString()}
                                                </p>
                                            </div>
                                            <div className="text-right flex-shrink-0">
                                                <p className={`text-sm font-bold ${
                                                    (bt.total_return_pct || 0) >= 0
                                                        ? 'text-emerald-400'
                                                        : 'text-red-400'
                                                }`}>
                                                    {formatPercent(bt.total_return_pct || 0)}
                                                </p>
                                                <p className="text-xs text-slate-500">
                                                    {bt.total_trades || 0} trades
                                                </p>
                                            </div>
                                        </div>

                                        {/* ✅ NEW: Action Buttons */}
                                        <div className="flex gap-2 pt-3 border-t border-slate-700/30">
                                            {/* Deploy Button - Always available */}
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    handleDeploy(bt);
                                                }}
                                                className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white rounded-lg font-semibold text-xs transition-all shadow-lg shadow-emerald-600/20"
                                                title="Deploy to live or paper trading"
                                            >
                                                <Rocket size={14}/>
                                                Deploy
                                            </button>

                                            {/* Publish Button - Only for good strategies */}
                                            {(bt.sharpe_ratio || 0) >= 1.0 && (bt.total_return_pct || 0) >= 10 && (
                                                <button
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        handlePublish(bt);
                                                    }}
                                                    className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 bg-violet-600 hover:bg-violet-500 text-white rounded-lg font-semibold text-xs transition-all shadow-lg shadow-violet-600/20"
                                                    title="Publish to marketplace"
                                                >
                                                    <Upload size={14}/>
                                                    Publish
                                                </button>
                                            )}

                                            {/* View Details - Keep existing functionality */}
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    handleBacktestSelect(bt);
                                                }}
                                                className="px-3 py-2 bg-slate-700/50 hover:bg-slate-600/50 text-slate-300 rounded-lg text-xs font-medium transition-all"
                                                title="View details"
                                            >
                                                <Eye size={14}/>
                                            </button>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </motion.div>
                </div>

                {/* Middle Column: Analytics (6/12 width) */}
                <div className="col-span-12 lg:col-span-6 space-y-6">
                    {hasData ? (
                        <>
                            {/* Equity Curve */}
                            <motion.div
                                initial={{opacity: 0, scale: 0.95}}
                                animate={{opacity: 1, scale: 1}}
                                transition={{duration: 0.5}}
                                className="glass-panel rounded-2xl p-6 relative overflow-hidden"
                            >
                                <div
                                    className="absolute top-0 right-0 w-64 h-64 bg-violet-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none"></div>
                                <div className="flex justify-between items-start mb-6">
                                    <div>
                                        <h3 className="text-xl font-semibold text-slate-100">Equity Curve</h3>
                                        <p className="text-sm text-slate-500 mt-1">
                                            {metrics.initial_capital > 0 && `Initial: ${formatCurrency(metrics.initial_capital)} • `}
                                            Final: {formatCurrency(metrics.final_equity)} • {equityData.length} points
                                        </p>
                                        <div className="flex items-center gap-4 mt-2">
                                            <div
                                                className={`flex items-center gap-1.5 text-sm font-semibold ${totalGain >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                {totalGain >= 0 ? <ArrowUpRight size={16}/> :
                                                    <ArrowDownRight size={16}/>}
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
                                            <AlertTriangle size={16}/>
                                            Risk Analysis
                                        </button>
                                        <button
                                            className="flex items-center gap-2 px-4 py-2 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50 rounded-lg transition-all text-sm font-medium text-slate-300">
                                            <Download size={16}/>
                                            Export
                                        </button>
                                    </div>
                                </div>
                                <div className="h-[320px] w-full">
                                    <LightweightChart
                                        data={equityData.map(d => ({
                                            time: d.timestamp,
                                            value: d.value
                                        }))}
                                        type="area"
                                        height={320}
                                        colors={{
                                            lineColor: hasIncrease ? '#10b981' : '#ef4444',
                                            areaTopColor: hasIncrease ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                                            areaBottomColor: 'rgba(0, 0, 0, 0)',
                                        }}
                                    />
                                </div>
                            </motion.div>

                            {/* Drawdown Chart */}
                            <motion.div
                                initial={{opacity: 0, y: 20}}
                                animate={{opacity: 1, y: 0}}
                                transition={{duration: 0.5, delay: 0.2}}
                                className="glass-panel rounded-2xl p-6"
                            >
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
                                                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4}/>
                                                <stop offset="95%" stopColor="#ef4444" stopOpacity={0.05}/>
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2}/>
                                        <XAxis dataKey="timestamp" stroke="#64748b" style={{fontSize: '11px'}}/>
                                        <YAxis stroke="#64748b" tickFormatter={(value) => `${value.toFixed(1)}%`}
                                               style={{fontSize: '11px'}} domain={['dataMin', 0]}/>
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: '#1e293b',
                                                border: '1px solid #334155',
                                                borderRadius: '12px',
                                                padding: '12px'
                                            }}
                                            formatter={(value: any) => [`${Number(value).toFixed(2)}%`, 'Drawdown']}
                                        />
                                        <Area type="monotone" dataKey="drawdown" stroke="#ef4444" strokeWidth={2}
                                              fillOpacity={1} fill="url(#colorDrawdown)"/>
                                    </AreaChart>
                                </ResponsiveContainer>
                            </motion.div>

                            {/* Rolling Sharpe */}
                            <motion.div
                                initial={{opacity: 0, y: 20}}
                                animate={{opacity: 1, y: 0}}
                                transition={{duration: 0.5, delay: 0.3}}
                                className="glass-panel rounded-2xl p-6"
                            >
                                <div className="mb-6">
                                    <h3 className="text-xl font-semibold text-slate-100">Rolling Sharpe Ratio
                                        (30-day)</h3>
                                    <p className="text-xs text-slate-500 mt-1">Annualized risk-adjusted return over
                                        time</p>
                                </div>
                                <ResponsiveContainer width="100%" height={200}>
                                    <ComposedChart data={rollingSharpe}>
                                        <defs>
                                            <linearGradient id="colorSharpe" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                                                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2}/>
                                        <XAxis dataKey="timestamp" stroke="#64748b" style={{fontSize: '11px'}}/>
                                        <YAxis stroke="#64748b" tickFormatter={(value) => value.toFixed(1)}
                                               style={{fontSize: '11px'}}/>
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: '#1e293b',
                                                border: '1px solid #334155',
                                                borderRadius: '12px',
                                                padding: '12px'
                                            }}
                                            formatter={(value: any) => [Number(value).toFixed(2), 'Sharpe Ratio']}
                                        />
                                        <Line type="monotone" dataKey={() => 0} stroke="#64748b" strokeWidth={1}
                                              strokeDasharray="5 5" dot={false}/>
                                        <Area type="monotone" dataKey="sharpe" stroke="#8b5cf6" strokeWidth={2}
                                              fillOpacity={1} fill="url(#colorSharpe)"/>
                                    </ComposedChart>
                                </ResponsiveContainer>
                            </motion.div>

                            {/* Monthly Returns */}
                            {monthlyReturns.length > 0 && (
                                <motion.div
                                    initial={{opacity: 0, y: 20}}
                                    animate={{opacity: 1, y: 0}}
                                    transition={{duration: 0.5, delay: 0.4}}
                                    className="glass-panel rounded-2xl p-6"
                                >
                                    <div className="flex items-center gap-3 mb-6">
                                        <Calendar className="text-violet-400" size={24}/>
                                        <div>
                                            <h3 className="text-xl font-semibold text-slate-100">Monthly Returns</h3>
                                            <p className="text-xs text-slate-500 mt-1">Month-by-month performance
                                                breakdown</p>
                                        </div>
                                    </div>
                                    <div className="grid grid-cols-2 gap-3">
                                        {monthlyReturns.map((monthData) => {
                                            const [year, month] = monthData.month.split('-');
                                            const monthName = new Date(parseInt(year), parseInt(month) - 1).toLocaleDateString('en-US', {
                                                month: 'short',
                                                year: 'numeric'
                                            });

                                            return (
                                                <div key={monthData.month}
                                                     className="p-3 bg-slate-800/40 border border-slate-700/50 rounded-xl hover:bg-slate-800/60 transition-all">
                                                    <div className="flex justify-between items-center">
                                                        <div>
                                                            <p className="text-xs text-slate-500 font-medium mb-1">{monthName}</p>
                                                            <p className={`text-lg font-bold ${monthData.return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                                {monthData.return >= 0 ? '+' : ''}{monthData.return.toFixed(2)}%
                                                            </p>
                                                        </div>
                                                        <div
                                                            className={`p-2 rounded-lg ${monthData.return >= 0 ? 'bg-emerald-500/10' : 'bg-red-500/10'}`}>
                                                            {monthData.return >= 0 ?
                                                                <TrendingUp className="text-emerald-400" size={16}/> :
                                                                <TrendingDown className="text-red-400" size={16}/>}
                                                        </div>
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
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
                                </motion.div>
                            )}

                            {/* Trade Distribution */}
                            {tradeDistribution.length > 0 && (
                                <motion.div
                                    initial={{opacity: 0, y: 20}}
                                    animate={{opacity: 1, y: 0}}
                                    transition={{duration: 0.5, delay: 0.5}}
                                    className="glass-panel rounded-2xl p-6"
                                >
                                    <div className="mb-6">
                                        <h3 className="text-xl font-semibold text-slate-100">Trade Distribution</h3>
                                        <p className="text-sm text-slate-500 mt-1">Win/Loss breakdown</p>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <ResponsiveContainer width="55%" height={180}>
                                            <RechartsPieChart>
                                                <Pie data={tradeDistribution} cx="50%" cy="50%" innerRadius={50}
                                                     outerRadius={70} paddingAngle={5} dataKey="value">
                                                    {tradeDistribution.map((entry, index) => <Cell key={`cell-${index}`}
                                                                                                   fill={entry.color}/>)}
                                                </Pie>
                                                <Tooltip contentStyle={{
                                                    backgroundColor: '#1e293b',
                                                    border: '1px solid #334155',
                                                    borderRadius: '12px',
                                                    padding: '12px'
                                                }}/>
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
                                </motion.div>
                            )}
                        </>
                    ) : (
                        <div className="glass-panel rounded-2xl p-12 text-center">
                            <Eye className="mx-auto text-slate-600 mb-4" size={48}/>
                            <h3 className="text-xl font-semibold text-slate-300 mb-2">No Backtest Selected</h3>
                            <p className="text-slate-500">Select a backtest from the library to view detailed
                                analytics</p>
                        </div>
                    )}
                </div>

                {/* Right Column: Social Feed (3/12 width) */}
                <div className="col-span-12 lg:col-span-3 h-[calc(100vh-140px)] sticky top-24">
                    <SocialFeed/>
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

            {/* Deploy Modal */}
            {showDeployModal && selectedBacktestForAction && (
                <DeploymentModal
                    backtest={{
                        id: String(selectedBacktestForAction.id),
                        strategy: selectedBacktestForAction.strategy_name || selectedBacktestForAction.name || '',
                        symbols: selectedBacktestForAction.symbols,
                        total_return_pct: selectedBacktestForAction.total_return_pct || 0,
                        win_rate: selectedBacktestForAction.win_rate || 0,
                        sharpe_ratio: selectedBacktestForAction.sharpe_ratio || 0,
                        max_drawdown: selectedBacktestForAction.max_drawdown || 0,
                        total_trades: selectedBacktestForAction.total_trades || 0,
                        initial_capital: selectedBacktestForAction.initial_capital || 10000,
                        parameters: selectedBacktestForAction.parameters || {}
                    }}
                    onClose={() => {
                        setShowDeployModal(false);
                        setSelectedBacktestForAction(null);
                    }}
                    onDeploy={onDeployConfirm}
                />
            )}

            {/* Publish Modal */}
            {showPublishModal && selectedBacktestForAction && (
                <PublishModal
                    backtest={{
                        id: selectedBacktestForAction.id,
                        name: selectedBacktestForAction.name || selectedBacktestForAction.strategy_name || '',
                        strategy_key: selectedBacktestForAction.strategy_key || selectedBacktestForAction.strategy_name || '',
                        symbols: selectedBacktestForAction.symbols,
                        total_return_pct: selectedBacktestForAction.total_return_pct || 0,
                        sharpe_ratio: selectedBacktestForAction.sharpe_ratio || 0,
                        max_drawdown: selectedBacktestForAction.max_drawdown || 0,
                        win_rate: selectedBacktestForAction.win_rate || 0,
                        total_trades: selectedBacktestForAction.total_trades || 0,
                        period: selectedBacktestForAction.period || '1y',
                        parameters: selectedBacktestForAction.parameters || {}
                    }}
                    onClose={() => {
                        setShowPublishModal(false);
                        setSelectedBacktestForAction(null);
                    }}
                    onPublish={onPublishConfirm}
                />
            )}
        </div>
    );
};

export default Dashboard;
