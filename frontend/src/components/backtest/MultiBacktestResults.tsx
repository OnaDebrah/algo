'use client'
import React, {useState} from 'react';
import {
    Area,
    AreaChart,
    CartesianGrid,
    ComposedChart,
    ResponsiveContainer,
    Scatter,
    Tooltip,
    XAxis,
    YAxis
} from 'recharts';
import {Activity, Target, TrendingDown, TrendingUp} from 'lucide-react';
import MetricCard from "@/components/backtest/MetricCard";
import {formatCurrency, formatPercent, toPrecision} from "@/utils/formatters";
import {BacktestResult, SymbolStats, Trade} from "@/types/all_types";

const MultiBacktestResults: ({results}: { results: BacktestResult }) => React.JSX.Element = ({results}: { results: BacktestResult }) => {
    const [tradeFilter, setTradeFilter] = useState('all');
    const trades = results.trades || [];

    return (
        <div className="space-y-6">
            <div className="grid grid-cols-4 gap-6">
                <MetricCard title="Total Return" value={formatPercent(results.total_return)} icon={TrendingUp}
                            trend="up" color="emerald"/>
                <MetricCard title="Win Rate" value={`${results.win_rate.toFixed(1)}%`} icon={Target} trend="up"
                            color="blue"/>
                <MetricCard title="Sharpe Ratio" value={results.sharpe_ratio.toFixed(2)} icon={Activity} trend="up"
                            color="violet"/>
                <MetricCard title="Max Drawdown" value={formatPercent(results.max_drawdown)} icon={TrendingDown}
                            trend="down" color="red"/>
            </div>

            <div
                className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                <h3 className="text-xl font-semibold text-slate-100 mb-6">Per-Symbol Performance</h3>
                <div className="grid grid-cols-1 gap-3">
                    {Object.entries(results.symbol_stats).map(([symbol, stats]: [string, SymbolStats]) => (
                        <div key={symbol}
                             className="flex items-center justify-between p-4 bg-slate-800/40 border border-slate-700/50 rounded-xl hover:bg-slate-800/60 transition-all">
                            <div className="flex items-center space-x-4">
                                <div
                                    className="w-12 h-12 bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 border border-violet-500/30 rounded-lg flex items-center justify-center">
                                    <span className="font-bold text-sm text-violet-300">{symbol.slice(0, 3)}</span>
                                </div>
                                <div>
                                    <p className="font-semibold text-slate-200">{symbol}</p>
                                </div>
                            </div>
                            <div className="flex items-center space-x-8">
                                <div className="text-right">
                                    <p className="text-xs text-slate-500 font-medium mb-0.5">Profit</p>
                                    <p className={`font-bold ${stats.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                        {formatCurrency(stats.total_return)}
                                    </p>
                                </div>
                                <div className="text-right">
                                    <p className="text-xs text-slate-500 font-medium mb-0.5">Trades</p>
                                    <p className="font-bold text-slate-200">{stats.total_trades}</p>
                                </div>
                                <div className="text-right">
                                    <p className="text-xs text-slate-500 font-medium mb-0.5">Win Rate</p>
                                    <p className="font-bold text-blue-400">{stats.win_rate.toFixed(1)}%</p>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            <div
                className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
                <h3 className="text-xl font-semibold text-slate-100 mb-6">Portfolio Equity Curve</h3>
                <ResponsiveContainer width="100%" height={320}>
                    <AreaChart data={results.equity_curve}>
                        <defs>
                            <linearGradient id="colorMultiEquity" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.4}/>
                                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3}/>
                        <XAxis dataKey="timestamp" stroke="#64748b" style={{fontSize: '12px', fontWeight: 500}}/>
                        <YAxis stroke="#64748b" tickFormatter={(value) => formatCurrency(value)}
                               style={{fontSize: '12px', fontWeight: 500}}/>
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
                        <Area type="monotone" dataKey="equity" stroke="#8b5cf6" strokeWidth={2} fillOpacity={1}
                              fill="url(#colorMultiEquity)"/>
                    </AreaChart>
                </ResponsiveContainer>
            </div>

            {/* Price Charts with Trade Markers for Each Symbol */}
            {results.price_data && Object.keys(results.price_data).length > 0 && (
                <div className="space-y-6">
                    {Object.entries(results.price_data).map(([symbol, prices]: [string, prices: Record<string, any>]) => {
                        const pricesArray = Object.values(prices) as any[];
                        const symbolTrades: Trade[] = trades.filter((t: Trade) => t.symbol === symbol);

                        return (
                            <div
                                key={symbol}
                                className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl"
                            >
                                <div className="flex justify-between items-center mb-6">
                                    <div>
                                        <h3 className="text-xl font-semibold text-slate-100">{symbol} - Price Action
                                            with Trade Signals</h3>
                                        <p className="text-xs text-slate-500 mt-1">Entry (🟢) and Exit (🔴) points
                                            overlaid on price movement</p>
                                    </div>
                                </div>
                                <ResponsiveContainer width="100%" height={350}>
                                    <ComposedChart data={(() => {
                                        // Prepare price data with trade markers
                                        const priceData = pricesArray.map((point) => {
                                            const timestamp: string = new Date(point.timestamp).toLocaleDateString();
                                            const buyTrades: Trade[] = symbolTrades.filter((trade: Trade) =>
                                                trade.order_type === 'BUY' &&
                                                new Date(trade.timestamp).toLocaleDateString() === timestamp
                                            );
                                            const sellTrades: Trade[] = symbolTrades.filter((trade: Trade) =>
                                                trade.order_type === 'SELL' &&
                                                new Date(trade.timestamp).toLocaleDateString() === timestamp
                                            );

                                            return {
                                                timestamp,
                                                close: point.close,
                                                buyPrice: buyTrades.length > 0 ? buyTrades[0].price : null,
                                                sellPrice: sellTrades.length > 0 ? sellTrades[0].price : null,
                                            };
                                        });
                                        return priceData;
                                    })()}>
                                        <defs>
                                            <linearGradient id={`colorPrice-${symbol}`} x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                                                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3}/>
                                        <XAxis
                                            dataKey="timestamp"
                                            stroke="#64748b"
                                            style={{fontSize: '11px', fontWeight: 500}}
                                            interval="preserveStartEnd"
                                            minTickGap={50}
                                        />
                                        <YAxis
                                            stroke="#64748b"
                                            tickFormatter={(value) => `$${value.toFixed(2)}`}
                                            style={{fontSize: '12px', fontWeight: 500}}
                                            domain={['auto', 'auto']}
                                        />
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: '#1e293b',
                                                border: '1px solid #334155',
                                                borderRadius: '12px',
                                                padding: '12px'
                                            }}
                                            formatter={(value: number | undefined, name: string | undefined) => {
                                                if (value === undefined) return ['', name];

                                                if (name === 'close') return [formatCurrency(value), 'Close Price'];
                                                if (name === 'buyPrice') return [formatCurrency(value), 'BUY'];
                                                if (name === 'sellPrice') return [formatCurrency(value), 'SELL'];
                                                return [value, name];
                                            }}
                                            labelStyle={{color: '#94a3b8', fontWeight: 600, marginBottom: '4px'}}
                                        />

                                        {/* Price line */}
                                        <Area
                                            type="monotone"
                                            dataKey="close"
                                            stroke="#8b5cf6"
                                            strokeWidth={2}
                                            fillOpacity={1}
                                            fill={`url(#colorPrice-${symbol})`}
                                        />

                                        {/* Buy signals (green) */}
                                        <Scatter
                                            dataKey="buyPrice"
                                            fill="#10b981"
                                            shape="circle"
                                            r={6}
                                        />

                                        {/* Sell signals (red) */}
                                        <Scatter
                                            dataKey="sellPrice"
                                            fill="#ef4444"
                                            shape="circle"
                                            r={6}
                                        />
                                    </ComposedChart>
                                </ResponsiveContainer>

                                {/* Legend */}
                                <div className="flex items-center justify-center space-x-6 mt-4">
                                    <div className="flex items-center space-x-2">
                                        <div className="w-3 h-3 rounded-full bg-violet-500"></div>
                                        <span className="text-xs text-slate-400 font-medium">Price</span>
                                    </div>
                                    <div className="flex items-center space-x-2">
                                        <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
                                        <span className="text-xs text-slate-400 font-medium">Buy Signal</span>
                                    </div>
                                    <div className="flex items-center space-x-2">
                                        <div className="w-3 h-3 rounded-full bg-red-500"></div>
                                        <span className="text-xs text-slate-400 font-medium">Sell Signal</span>
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}

            {/* 3. Detailed Trade Ledger Table */}
            <div
                className="bg-slate-900/50 backdrop-blur-xl border border-slate-800/50 rounded-2xl overflow-hidden shadow-2xl">
                <div
                    className="px-6 py-5 border-b border-slate-800/50 flex justify-between items-center bg-white/[0.02]">
                    <div>
                        <h3 className="text-lg font-bold text-slate-100 tracking-tight">Trade Ledger</h3>
                        <p className="text-[10px] text-slate-500 font-black uppercase tracking-[0.2em] mt-1">Transaction
                            History & Execution Details</p>
                    </div>
                    <div className="flex bg-black/40 p-1 rounded-lg border border-white/5">
                        {['all', 'profitable', 'loss'].map((f) => (
                            <button
                                key={f}
                                onClick={() => setTradeFilter(f)}
                                className={`px-3 py-1.5 rounded-md text-[10px] font-black uppercase tracking-widest transition-all ${tradeFilter === f ? 'bg-violet-600 text-white' : 'text-slate-500 hover:text-slate-300'
                                }`}
                            >
                                {f}
                            </button>
                        ))}
                    </div>
                </div>

                <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                        <thead>
                        <tr className="bg-white/[0.01] text-[10px] font-black text-slate-500 uppercase tracking-widest border-b border-slate-800/50">
                            <th className="px-6 py-4">Asset</th>
                            <th className="px-6 py-4">Type</th>
                            <th className="px-6 py-4">Timestamp</th>
                            <th className="px-6 py-4">Strategy</th>
                            <th className="px-6 py-4 text-right">Price</th>
                            <th className="px-6 py-4 text-right">Quantity</th>
                            <th className="px-6 py-4 text-right">Commission</th>
                            <th className="px-6 py-4 text-right">Profit</th>
                        </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/30">
                        {trades.length > 0 ? trades.map((trade: Trade, idx: number) => {
                            const isWin = trade.profit && trade.profit >= 0;
                            const isClosed = trade.profit !== null && trade.profit !== undefined;
                            // Basic filter logic
                            if (tradeFilter === 'profitable' && (!isClosed || !isWin)) return null;
                            if (tradeFilter === 'loss' && (!isClosed || isWin)) return null;

                            return (
                                <tr key={idx} className="hover:bg-white/[0.02] transition-colors group">
                                    <td className="px-6 py-4">
                                        <div className="flex items-center space-x-3">
                                                <span
                                                    className="w-8 h-8 rounded bg-slate-800 flex items-center justify-center font-mono text-[10px] font-bold text-violet-400 border border-slate-700">
                                                    {trade.symbol.slice(0, 2)}
                                                </span>
                                            <span className="font-bold text-slate-200 text-xs">{trade.symbol}</span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4">
                                            <span
                                                className={`px-2 py-1 rounded text-[9px] font-black uppercase tracking-tighter ${trade.order_type === 'BUY' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'
                                                }`}>
                                                {trade.order_type}
                                            </span>
                                    </td>
                                    <td className="px-6 py-4 text-[11px] text-slate-400 font-medium">
                                        {trade.timestamp}
                                    </td>
                                    <td className="px-6 py-4 text-[11px] text-slate-400 font-medium">
                                        {trade.strategy}
                                    </td>
                                    <td className="px-6 py-4 text-right font-mono text-xs text-slate-300">
                                        {formatCurrency(trade.price)}
                                    </td>
                                    <td className="px-6 py-4 text-right font-mono text-xs text-slate-500">
                                        {toPrecision(trade.quantity)}
                                    </td>
                                    <td className="px-6 py-4 text-right font-mono text-xs text-slate-500">
                                        {formatCurrency(trade.commission)}
                                    </td>
                                    <td className={`px-6 py-4 text-right font-bold text-xs ${isClosed ? (isWin ? 'text-emerald-400' : 'text-red-400') : 'text-slate-500'}`}>
                                        <div className="flex flex-col items-end">
                                            {isClosed ? (
                                                <>
                                                    <span>{isWin ? '+' : ''}{formatCurrency(trade.profit || 0)}</span>
                                                    <span className="text-[9px] font-medium opacity-60">
                                                            ({trade.profit_pct?.toFixed(2) || '0.00'}%)
                                                        </span>
                                                </>
                                            ) : (
                                                <span className="text-[10px] uppercase">Open</span>
                                            )}
                                        </div>
                                    </td>
                                </tr>
                            );
                        }) : (
                            <tr>
                                <td colSpan={8} className="px-6 py-12 text-center text-slate-600 text-sm">
                                    No transaction data available for this backtest.
                                </td>
                            </tr>
                        )}
                        </tbody>
                    </table>
                </div>

                {/* Table Footer / Summary */}
                <div
                    className="px-6 py-4 bg-white/[0.01] border-t border-slate-800/50 flex justify-between items-center">
                    <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                        Total Captured Segments: {trades.length}
                    </span>
                    <div className="flex items-center space-x-4">
                        <div className="flex items-center space-x-1">
                            <div className="w-2 h-2 rounded-full bg-emerald-500"/>
                            <span
                                className="text-[10px] text-slate-400 font-bold uppercase">Average Winner: {formatCurrency(results.avg_win || 0)}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                            <div className="w-2 h-2 rounded-full bg-red-500"/>
                            <span
                                className="text-[10px] text-slate-400 font-bold uppercase">Average Loser: {formatCurrency(results.avg_loss || 0)}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default MultiBacktestResults;