import MetricCard from "@/components/backtest/MetricCard";
import {Activity, Download, Target, TrendingDown, TrendingUp} from "lucide-react";
import {Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis} from "recharts";
import {formatCurrency, formatPercent} from "@/utils/formatters";

import {useEffect, useState} from "react";
import {analytics, portfolio, strategy} from "@/utils/api";
import {PortfolioTrade, StrategyInfo} from "@/types/all_types";

const Dashboard = () => {
    const [metrics, setMetrics] = useState({
        totalReturn: 0,
        winRate: 0,
        sharpeRatio: 0,
        maxDrawdown: 0
    });
    const [equityData, setEquityData] = useState<any[]>([]);
    const [recentTrades, setRecentTrades] = useState<any[]>([]);
    const [activeStrategies, setActiveStrategies] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchDashboardData = async () => {
            try {
                setLoading(true);
                const portfoliosRes = await portfolio.list();

                if (portfoliosRes.data && portfoliosRes.data.length > 0) {
                    const pid = portfoliosRes.data[0].id;

                    // 1. Get Performance metrics
                    const perfRes = await analytics.getPerformance(pid, {period: "1M"});

                    // Map the real equity curve data
                    // Assuming the backend returns { dates: string[], values: number[] }
                    if (perfRes.data && perfRes.data.dates && perfRes.data.values) {
                        const realEquity = perfRes.data.dates.map((date: string, index: number) => ({
                            date: new Date(date).toLocaleDateString(undefined, {month: 'short', day: 'numeric'}),
                            value: perfRes.data.values[index]
                        }));
                        setEquityData(realEquity);
                    }

                    // 2. Get Recent Trades
                    const tradesRes = await portfolio.getTrades(pid);
                    if (tradesRes.data) {
                        setRecentTrades(tradesRes.data.slice(0, 5).map((t: PortfolioTrade) => ({
                            symbol: t.symbol,
                            strategy: t.strategy || 'Manual',
                            profit: t.profit || 0,
                            time: t.executed_at ? new Date(t.executed_at).toLocaleTimeString() : 'Just now',
                            status: 'closed'
                        })));
                    }

                    // 3. Get Strategies
                    const strategiesRes = await strategy.list();
                    if (strategiesRes.data) {
                        setActiveStrategies(strategiesRes.data.slice(0, 4).map((s: StrategyInfo) => ({
                            name: s.name,
                            performance: s.historical_return || 0,
                            trades: s.total_trades || 0,
                            status: 'active'
                        })));
                    }
                }

                setLoading(false);
            } catch (error) {
                console.error("Failed to load dashboard data", error);
                setLoading(false);
            }
        };
        fetchDashboardData();
    }, []);

    if (loading) return <div className="p-10 text-center text-slate-500">Loading Dashboard...</div>;

    return (
        <div className="space-y-6">
            {/* Key Metrics */}
            <div className="grid grid-cols-4 gap-6">
                <MetricCard
                    title="Total Return"
                    value={formatPercent(metrics.totalReturn)}
                    icon={TrendingUp}
                    trend="up"
                    color="emerald"
                />
                <MetricCard
                    title="Win Rate"
                    value={`${metrics.winRate.toFixed(1)}%`}
                    icon={Target}
                    trend="up"
                    color="blue"
                />
                <MetricCard
                    title="Sharpe Ratio"
                    value={metrics.sharpeRatio.toFixed(2)}
                    icon={Activity}
                    trend="up"
                    color="violet"
                />
                <MetricCard
                    title="Max Drawdown"
                    value={formatPercent(metrics.maxDrawdown)}
                    icon={TrendingDown}
                    trend="down"
                    color="red"
                />
            </div>

            {/* Equity Curve */}
            <div
                className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                <div className="flex justify-between items-center mb-6">
                    <h3 className="text-xl font-semibold text-slate-100">Portfolio Equity Curve</h3>
                    <button
                        className="flex items-center space-x-2 px-4 py-2 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50 rounded-lg transition-all text-sm font-medium text-slate-300">
                        <Download size={18} strokeWidth={2}/>
                        <span>Export Data</span>
                    </button>
                </div>
                <ResponsiveContainer width="100%" height={320}>
                    <AreaChart data={equityData}>
                        <defs>
                            <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.4}/>
                                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3}/>
                        <XAxis
                            dataKey="date"
                            stroke="#64748b"
                            style={{fontSize: '12px', fontWeight: 500}}
                        />
                        <YAxis
                            stroke="#64748b"
                            tickFormatter={(value) => `${(value / 1000).toFixed(0)}k`}
                            style={{fontSize: '12px', fontWeight: 500}}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#1e293b',
                                border: '1px solid #334155',
                                borderRadius: '12px',
                                padding: '12px'
                            }}
                            formatter={(value) => [formatCurrency(Number(value || 0)), 'Equity']}
                            labelStyle={{color: '#94a3b8', fontWeight: 600, marginBottom: '4px'}}
                        />
                        <Area
                            type="monotone"
                            dataKey="value"
                            stroke="#8b5cf6"
                            strokeWidth={2}
                            fillOpacity={1}
                            fill="url(#colorValue)"
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>

            {/* Bottom Grid */}
            <div className="grid grid-cols-2 gap-6">
                {/* Recent Trades */}
                <div
                    className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                    <div className="flex justify-between items-center mb-6">
                        <h3 className="text-xl font-semibold text-slate-100">Recent Trades</h3>
                        <span className="text-xs font-semibold text-slate-500 tracking-wider">LAST 24H</span>
                    </div>
                    <div className="space-y-3">
                        {recentTrades.map((trade, i) => (
                            <div
                                key={i}
                                className="flex justify-between items-center p-4 bg-slate-800/40 hover:bg-slate-800/60 border border-slate-700/30 rounded-xl transition-all group"
                            >
                                <div className="flex items-center space-x-4">
                                    <div
                                        className="w-10 h-10 rounded-lg bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 border border-violet-500/30 flex items-center justify-center">
                                        <span
                                            className="font-bold text-sm text-violet-300">{trade.symbol.slice(0, 2)}</span>
                                    </div>
                                    <div>
                                        <p className="font-semibold text-slate-200">{trade.symbol}</p>
                                        <p className="text-xs text-slate-500 font-medium">{trade.strategy}</p>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <p className={`font-bold ${trade.profit >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                        {trade.profit >= 0 ? '+' : ''}{formatCurrency(trade.profit)}
                                    </p>
                                    <p className="text-xs text-slate-500 font-medium">{trade.time}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Active Strategies */}
                <div
                    className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                    <div className="flex justify-between items-center mb-6">
                        <h3 className="text-xl font-semibold text-slate-100">Active Strategies</h3>
                        <span className="text-xs font-semibold text-emerald-500 tracking-wider">LIVE</span>
                    </div>
                    <div className="space-y-3">
                        {activeStrategies.map((strategy, i) => (
                            <div
                                key={i}
                                className="flex justify-between items-center p-4 bg-slate-800/40 hover:bg-slate-800/60 border border-slate-700/30 rounded-xl transition-all"
                            >
                                <div>
                                    <div className="flex items-center space-x-2 mb-1">
                                        <p className="font-semibold text-slate-200">{strategy.name}</p>
                                        <span
                                            className={`text-xs font-bold px-2 py-0.5 rounded-full ${strategy.status === 'active'
                                                ? 'bg-emerald-500/20 text-emerald-400'
                                                : 'bg-slate-700/50 text-slate-400'
                                            }`}>
                                            {strategy.status.toUpperCase()}
                                        </span>
                                    </div>
                                    <p className="text-xs text-slate-500 font-medium">{strategy.trades} trades
                                        executed</p>
                                </div>
                                <div className="text-right">
                                    <p className={`font-bold text-lg ${strategy.performance >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                        {formatPercent(strategy.performance)}
                                    </p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;