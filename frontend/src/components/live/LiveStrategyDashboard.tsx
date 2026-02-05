/**
 * Live Strategy Dashboard
 * Complete monitoring interface for live trading strategies
 */

'use client'

import React, {useEffect, useState} from 'react';
import {
    Activity,
    ArrowDownRight,
    ArrowUpRight,
    BarChart3,
    Clock,
    DollarSign,
    Pause,
    Play,
    RefreshCw,
    Square,
    TrendingUp
} from 'lucide-react';
import {LiveEquityPoint, LiveStrategy, LiveTrade} from "@/types/all_types";
import {live} from "@/utils/api";

interface WebSocketMessage {
    type: 'equity_update' | 'trade_executed' | 'status_change' | 'error';
    strategy_id: number;
    timestamp: string;
    data: any;
}

export default function LiveStrategyDashboard() {
    const [strategies, setStrategies] = useState<LiveStrategy[]>([]);
    const [selectedStrategy, setSelectedStrategy] = useState<number | null>(null);
    const [equityCurve, setEquityCurve] = useState<LiveEquityPoint[]>([]);
    const [trades, setTrades] = useState<LiveTrade[]>([]);
    const [loading, setLoading] = useState(true);
    const [wsConnected, setWsConnected] = useState(false);

    // WebSocket connection
    const [ws, setWs] = useState<WebSocket | null>(null);

    // Load strategies on mount
    useEffect(() => {
        loadStrategies();

        // Refresh every 30 seconds
        const interval = setInterval(loadStrategies, 30000);

        return () => clearInterval(interval);
    }, []);

    // Connect to WebSocket when strategy selected
    useEffect(() => {
        if (!selectedStrategy) return;

        connectWebSocket(selectedStrategy);
        loadStrategyDetails(selectedStrategy);

        return () => {
            if (ws) {
                ws.close();
                setWs(null);
            }
        };
    }, [selectedStrategy]);

    const loadStrategies = async () => {
        setLoading(true);
        try {
            const response = await live.list();
            setStrategies(response);

            if (!selectedStrategy && response.length > 0) {
                setSelectedStrategy(response[0].id);
            }
        } catch (error) {
            console.error('Error loading strategies:', error);
        } finally {
            setLoading(false);
        }
    };

    const loadStrategyDetails = async (strategyId: number) => {
        try {
            const response = await live.getDetails(strategyId);
            setEquityCurve(response.equity_curve || []);
            setTrades(response.trades || []);
        } catch (error) {
            console.error('Error loading strategy details:', error);
        }
    };

    const connectWebSocket = (strategyId: number) => {
        const wsUrl = `ws://localhost:8000/ws/strategy/${strategyId}`;
        const websocket = new WebSocket(wsUrl);

        websocket.onopen = () => {
            console.log('WebSocket connected');
            setWsConnected(true);
        };

        websocket.onmessage = (event) => {
            const message: WebSocketMessage = JSON.parse(event.data);
            handleWebSocketMessage(message);
        };

        websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            setWsConnected(false);
        };

        websocket.onclose = () => {
            console.log('WebSocket disconnected');
            setWsConnected(false);
        };

        setWs(websocket);
    };

    const handleWebSocketMessage = (message: WebSocketMessage) => {
        switch (message.type) {
            case 'equity_update':
                // Add new equity point
                setEquityCurve(prev => [...prev, {
                    timestamp: message.timestamp,
                    equity: message.data.equity,
                    cash: message.data.cash,
                    daily_pnl: message.data.daily_pnl,
                    total_pnl: message.data.total_pnl,
                    drawdown_pct: message.data.drawdown_pct
                }]);

                // Update strategy equity
                setStrategies(prev => prev.map(s =>
                    s.id === message.strategy_id
                        ? { ...s, current_equity: message.data.equity, daily_pnl: message.data.daily_pnl }
                        : s
                ));
                break;

            case 'trade_executed':
                // Add new trade
                loadStrategyDetails(message.strategy_id);

                // Show notification
                showNotification('Trade Executed', `${message.data.side} ${message.data.quantity} ${message.data.symbol} @ $${message.data.price}`);
                break;

            case 'status_change':
                // Update strategy status
                setStrategies(prev => prev.map(s =>
                    s.id === message.strategy_id
                        ? { ...s, status: message.data.new_status }
                        : s
                ));

                showNotification('Status Change', `Strategy ${message.data.new_status}`);
                break;

            case 'error':
                showNotification('Error', message.data.error, 'error');
                break;
        }
    };

    const controlStrategy = async (strategyId: number, action: 'start' | 'pause' | 'stop') => {
        try {
            await live.control(strategyId, action);
            await loadStrategies();
            showNotification('Success', `Strategy ${action}ed`);
        } catch (error) {
            console.error('Error controlling strategy:', error);
            showNotification('Error', 'Failed to control strategy', 'error');
        }
    };

    const showNotification = (title: string, message: string, type: 'success' | 'error' = 'success') => {
        // Could integrate with toast library
        console.log(`${type.toUpperCase()}: ${title} - ${message}`);
    };

    const currentStrategy = strategies.find(s => s.id === selectedStrategy);

    if (loading) {
        return (
            <div className="flex items-center justify-center h-screen">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-500"></div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-slate-950 p-6">
            <div className="max-w-7xl mx-auto space-y-6">

                {/* Header */}
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-3xl font-bold text-slate-100">Live Trading Dashboard</h1>
                        <p className="text-slate-400 mt-1">Monitor and control your active strategies</p>
                    </div>

                    <div className="flex items-center gap-3">
                        {/* WebSocket Status */}
                        <div className="flex items-center gap-2 px-4 py-2 bg-slate-900 rounded-lg border border-slate-800">
                            <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`} />
                            <span className="text-xs text-slate-400">
                {wsConnected ? 'Live' : 'Disconnected'}
              </span>
                        </div>

                        <button
                            onClick={loadStrategies}
                            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg border border-slate-700 text-slate-300 flex items-center gap-2"
                        >
                            <RefreshCw size={16} />
                            Refresh
                        </button>
                    </div>
                </div>

                {/* Strategy Tabs */}
                <div className="flex gap-2 overflow-x-auto pb-2">
                    {strategies.map(strategy => (
                        <button
                            key={strategy.id}
                            onClick={() => setSelectedStrategy(strategy.id)}
                            className={`px-6 py-3 rounded-xl flex-shrink-0 transition-all ${
                                selectedStrategy === strategy.id
                                    ? 'bg-violet-600 text-white shadow-lg shadow-violet-600/20'
                                    : 'bg-slate-900 text-slate-400 hover:bg-slate-800 border border-slate-800'
                            }`}
                        >
                            <div className="flex items-center gap-3">
                                <div className={`w-2 h-2 rounded-full ${
                                    strategy.status === 'running' ? 'bg-emerald-500 animate-pulse' :
                                        strategy.status === 'paused' ? 'bg-amber-500' :
                                            strategy.status === 'error' ? 'bg-red-500' :
                                                'bg-slate-600'
                                }`} />
                                <div className="text-left">
                                    <div className="font-semibold text-sm">{strategy.name}</div>
                                    <div className="text-xs opacity-70">{strategy.symbols.join(', ')}</div>
                                </div>
                            </div>
                        </button>
                    ))}
                </div>

                {currentStrategy && (
                    <>
                        {/* Performance Cards */}
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                            {/* Current Equity */}
                            <div className="bg-gradient-to-br from-slate-900 to-slate-950 rounded-xl p-6 border border-slate-800">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-xs font-bold text-slate-500 uppercase">Current Equity</span>
                                    <DollarSign size={16} className="text-emerald-500" />
                                </div>
                                <div className="text-2xl font-black text-slate-100">
                                    ${currentStrategy.current_equity.toLocaleString()}
                                </div>
                                <div className={`text-sm font-semibold mt-1 flex items-center gap-1 ${
                                    currentStrategy.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'
                                }`}>
                                    {currentStrategy.total_return >= 0 ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
                                    {currentStrategy.total_return_pct >= 0 ? '+' : ''}
                                    {currentStrategy.total_return_pct.toFixed(2)}%
                                </div>
                            </div>

                            {/* Daily P&L */}
                            <div className="bg-gradient-to-br from-slate-900 to-slate-950 rounded-xl p-6 border border-slate-800">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-xs font-bold text-slate-500 uppercase">Today's P&L</span>
                                    <TrendingUp size={16} className={currentStrategy.daily_pnl >= 0 ? 'text-emerald-500' : 'text-red-500'} />
                                </div>
                                <div className={`text-2xl font-black ${
                                    currentStrategy.daily_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'
                                }`}>
                                    {currentStrategy.daily_pnl >= 0 ? '+' : ''}${currentStrategy.daily_pnl.toFixed(2)}
                                </div>
                                <div className="text-sm text-slate-400 mt-1">
                                    Since {new Date().toLocaleDateString()}
                                </div>
                            </div>

                            {/* Total Trades */}
                            <div className="bg-gradient-to-br from-slate-900 to-slate-950 rounded-xl p-6 border border-slate-800">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-xs font-bold text-slate-500 uppercase">Total Trades</span>
                                    <BarChart3 size={16} className="text-blue-500" />
                                </div>
                                <div className="text-2xl font-black text-slate-100">
                                    {currentStrategy.total_trades}
                                </div>
                                <div className="text-sm text-slate-400 mt-1">
                                    Win Rate: {currentStrategy.total_trades > 0
                                    ? ((currentStrategy.winning_trades / currentStrategy.total_trades) * 100).toFixed(1)
                                    : '0.0'}%
                                </div>
                            </div>

                            {/* Sharpe Ratio */}
                            <div className="bg-gradient-to-br from-slate-900 to-slate-950 rounded-xl p-6 border border-slate-800">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-xs font-bold text-slate-500 uppercase">Sharpe Ratio</span>
                                    <Activity size={16} className="text-purple-500" />
                                </div>
                                <div className="text-2xl font-black text-slate-100">
                                    {currentStrategy.sharpe_ratio?.toFixed(2) || 'N/A'}
                                </div>
                                <div className="text-sm text-slate-400 mt-1">
                                    Max DD: {currentStrategy.max_drawdown.toFixed(2)}%
                                </div>
                            </div>
                        </div>

                        {/* Main Content Grid */}
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                            {/* Equity Curve */}
                            <div className="lg:col-span-2 bg-slate-900/50 rounded-xl p-6 border border-slate-800">
                                <div className="flex items-center justify-between mb-6">
                                    <h3 className="text-lg font-bold text-slate-100">Real-Time Equity Curve</h3>
                                    <div className="flex items-center gap-2 text-xs text-slate-400">
                                        <Clock size={14} />
                                        Updates every 60s
                                    </div>
                                </div>

                                {/* Simple equity chart visualization */}
                                <div className="h-64 relative">
                                    {equityCurve.length > 0 ? (
                                        <div className="w-full h-full flex items-end justify-between gap-1">
                                            {equityCurve.slice(-50).map((point, idx) => {
                                                const maxEquity = Math.max(...equityCurve.map(p => p.equity));
                                                const minEquity = Math.min(...equityCurve.map(p => p.equity));
                                                const range = maxEquity - minEquity || 1;
                                                const height = ((point.equity - minEquity) / range) * 100;

                                                return (
                                                    <div
                                                        key={idx}
                                                        className="flex-1 bg-gradient-to-t from-emerald-500/50 to-emerald-400 rounded-t"
                                                        style={{ height: `${height}%` }}
                                                        title={`$${point.equity.toFixed(2)}`}
                                                    />
                                                );
                                            })}
                                        </div>
                                    ) : (
                                        <div className="flex items-center justify-center h-full text-slate-500">
                                            No equity data yet
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Strategy Controls */}
                            <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-800">
                                <h3 className="text-lg font-bold text-slate-100 mb-6">Strategy Controls</h3>

                                <div className="space-y-4">
                                    {/* Status */}
                                    <div>
                                        <label className="text-xs font-bold text-slate-500 uppercase block mb-2">Status</label>
                                        <div className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
                                            currentStrategy.status === 'running' ? 'bg-emerald-500/20 text-emerald-400' :
                                                currentStrategy.status === 'paused' ? 'bg-amber-500/20 text-amber-400' :
                                                    currentStrategy.status === 'error' ? 'bg-red-500/20 text-red-400' :
                                                        'bg-slate-800 text-slate-400'
                                        }`}>
                                            {currentStrategy.status === 'running' && <Play size={16} className="fill-current" />}
                                            {currentStrategy.status === 'paused' && <Pause size={16} />}
                                            {currentStrategy.status === 'stopped' && <Square size={16} />}
                                            <span className="font-semibold uppercase text-sm">{currentStrategy.status}</span>
                                        </div>
                                    </div>

                                    {/* Action Buttons */}
                                    <div className="space-y-2">
                                        {currentStrategy.status === 'running' && (
                                            <>
                                                <button
                                                    onClick={() => controlStrategy(currentStrategy.id, 'pause')}
                                                    className="w-full px-4 py-3 bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/50 rounded-lg text-amber-400 font-semibold flex items-center justify-center gap-2"
                                                >
                                                    <Pause size={16} />
                                                    Pause Strategy
                                                </button>
                                                <button
                                                    onClick={() => controlStrategy(currentStrategy.id, 'stop')}
                                                    className="w-full px-4 py-3 bg-red-500/20 hover:bg-red-500/30 border border-red-500/50 rounded-lg text-red-400 font-semibold flex items-center justify-center gap-2"
                                                >
                                                    <Square size={16} />
                                                    Stop Strategy
                                                </button>
                                            </>
                                        )}

                                        {currentStrategy.status === 'paused' && (
                                            <button
                                                onClick={() => controlStrategy(currentStrategy.id, 'start')}
                                                className="w-full px-4 py-3 bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/50 rounded-lg text-emerald-400 font-semibold flex items-center justify-center gap-2"
                                            >
                                                <Play size={16} className="fill-current" />
                                                Resume Strategy
                                            </button>
                                        )}
                                    </div>

                                    {/* Strategy Info */}
                                    <div className="pt-4 border-t border-slate-800 space-y-3">
                                        <div className="flex justify-between text-sm">
                                            <span className="text-slate-500">Mode:</span>
                                            <span className="text-slate-300 font-semibold uppercase">{currentStrategy.deployment_mode}</span>
                                        </div>
                                        <div className="flex justify-between text-sm">
                                            <span className="text-slate-500">Broker:</span>
                                            <span className="text-slate-300">{currentStrategy.broker || 'Paper'}</span>
                                        </div>
                                        <div className="flex justify-between text-sm">
                                            <span className="text-slate-500">Deployed:</span>
                                            <span className="text-slate-300">
                        {new Date(currentStrategy.deployed_at).toLocaleDateString()}
                      </span>
                                        </div>
                                        <div className="flex justify-between text-sm">
                                            <span className="text-slate-500">Last Trade:</span>
                                            <span className="text-slate-300">
                        {currentStrategy.last_trade_at
                            ? new Date(currentStrategy.last_trade_at).toLocaleTimeString()
                            : 'None'}
                      </span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Recent Trades */}
                        <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-800">
                            <h3 className="text-lg font-bold text-slate-100 mb-6">Recent Trades</h3>

                            <div className="overflow-x-auto">
                                <table className="w-full">
                                    <thead>
                                    <tr className="border-b border-slate-800">
                                        <th className="text-left pb-3 text-xs font-bold text-slate-500 uppercase">Time</th>
                                        <th className="text-left pb-3 text-xs font-bold text-slate-500 uppercase">Symbol</th>
                                        <th className="text-left pb-3 text-xs font-bold text-slate-500 uppercase">Side</th>
                                        <th className="text-right pb-3 text-xs font-bold text-slate-500 uppercase">Qty</th>
                                        <th className="text-right pb-3 text-xs font-bold text-slate-500 uppercase">Price</th>
                                        <th className="text-right pb-3 text-xs font-bold text-slate-500 uppercase">P&L</th>
                                        <th className="text-left pb-3 text-xs font-bold text-slate-500 uppercase">Status</th>
                                    </tr>
                                    </thead>
                                    <tbody className="divide-y divide-slate-800/50">
                                    {trades.slice(0, 10).map(trade => (
                                        <tr key={trade.id} className="group hover:bg-white/5">
                                            <td className="py-3 text-sm text-slate-400">
                                                {new Date(trade.opened_at).toLocaleTimeString()}
                                            </td>
                                            <td className="py-3 text-sm font-semibold text-slate-200">
                                                {trade.symbol}
                                            </td>
                                            <td className="py-3">
                          <span className={`text-xs font-bold px-2 py-1 rounded ${
                              trade.side === 'BUY'
                                  ? 'bg-emerald-500/20 text-emerald-400'
                                  : 'bg-red-500/20 text-red-400'
                          }`}>
                            {trade.side}
                          </span>
                                            </td>
                                            <td className="py-3 text-sm text-right text-slate-300 font-mono">
                                                {trade.quantity}
                                            </td>
                                            <td className="py-3 text-sm text-right text-slate-300 font-mono">
                                                ${trade.entry_price?.toFixed(2) || '-'}
                                            </td>
                                            <td className="py-3 text-sm text-right font-semibold">
                                                {trade.profit !== null ? (
                                                    <span className={trade.profit >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                              {trade.profit >= 0 ? '+' : ''}${trade.profit.toFixed(2)}
                            </span>
                                                ) : (
                                                    <span className="text-slate-500">-</span>
                                                )}
                                            </td>
                                            <td className="py-3">
                          <span className={`text-xs font-semibold ${
                              trade.status === 'closed' ? 'text-slate-500' : 'text-blue-400'
                          }`}>
                            {trade.status.toUpperCase()}
                          </span>
                                            </td>
                                        </tr>
                                    ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </>
                )}

            </div>
        </div>
    );
}
