// 'use client'
// import React, {useState, useEffect} from 'react';
// import {
//     Zap, ShieldAlert, Link, Link2Off, Activity,
//     RefreshCcw, Play, Square, Settings,
//     History, Wallet, ArrowUpRight, ArrowDownRight,
//     Search, Filter, CheckCircle2, AlertCircle,
//     Loader2
// } from "lucide-react";
// import {live} from '@/utils/api';
// import {BrokerType, ConnectRequest, EngineStatus, ExecutionOrder, LiveStatus} from '@/types/all_types';
//
// const LiveExecution = () => {
//     const [isConnected, setIsConnected] = useState(false);
//     const [engineStatus, setEngineStatus] = useState<EngineStatus>(EngineStatus.IDLE);
//     const [activeBroker, setActiveBroker] = useState<BrokerType>(BrokerType.PAPER_TRADING);
//     const [orders, setOrders] = useState<ExecutionOrder[]>([]);
//     const [isLoading, setIsLoading] = useState(false);
//     const [error, setError] = useState<string | null>(null);
//
//     // Initial load and polling
//     useEffect(() => {
//         fetchStatus();
//         const interval = setInterval(() => {
//             fetchStatus();
//             if (isConnected) fetchOrders();
//         }, 3000); // Poll every 3 seconds
//
//         return () => clearInterval(interval);
//     }, [isConnected]);
//
//     const fetchStatus = async () => {
//         try {
//             const response = await live.getStatus();
//             if (response) {
//                 setIsConnected(response.is_connected);
//                 setEngineStatus(response.engine_status);
//                 setActiveBroker(response.active_broker);
//             }
//         } catch (err) {
//             console.error("Failed to fetch live status:", err);
//         }
//     };
//
//     const fetchOrders = async () => {
//         try {
//             const response = await live.getOrders();
//             if (response) {
//                 setOrders(response);
//             }
//         } catch (err) {
//             console.error("Failed to fetch orders:", err);
//         }
//     };
//
//     const handleConnect = async () => {
//         setIsLoading(true);
//         setError(null);
//
//         const payload: ConnectRequest = {
//             broker: activeBroker,
//             api_key: null,
//             api_secret: null
//         };
//
//         try {
//             if (isConnected) {
//                 await live.disconnect();
//             } else {
//                 await live.connect(payload);
//             }
//             await fetchStatus();
//         } catch (err) {
//             setError("Failed to change connection status");
//             console.error(err);
//         } finally {
//             setIsLoading(false);
//         }
//     };
//
//     const handleEngineToggle = async () => {
//         if (!isConnected) return;
//
//         try {
//             if (engineStatus === EngineStatus.RUNNING) {
//                 await live.stopEngine();
//             } else {
//                 await live.startEngine();
//             }
//             await fetchStatus();
//         } catch (err) {
//             console.error("Failed to toggle engine:", err);
//             setError("Failed to toggle engine state");
//         }
//     };
//
//     return (
//         <div className="space-y-6 animate-in fade-in duration-700">
//             {/* 1. Critical Warning */}
//             <div className="bg-amber-500/10 border border-amber-500/20 rounded-2xl p-4 flex gap-4 items-start">
//                 <ShieldAlert className="text-amber-500 shrink-0" size={24}/>
//                 <div>
//                     <h4 className="text-sm font-bold text-amber-500 uppercase tracking-tight">Real Money Risk
//                         Warning</h4>
//                     <p className="text-xs text-slate-400 leading-relaxed">
//                         Live trading involves real capital risk. Ensure stop-losses are active and monitor positions
//                         regularly.
//                         Start with <b>Paper Trading</b> to verify connectivity and execution logic.
//                     </p>
//                 </div>
//             </div>
//
//             <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
//                 {/* 2. Broker Configuration */}
//                 <div className="lg:col-span-1 space-y-6">
//                     <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
//                         <div className="flex items-center justify-between mb-6">
//                             <h3 className="text-sm font-black text-slate-300 uppercase tracking-widest flex items-center gap-2">
//                                 <Link size={16} className="text-emerald-500"/> Connectivity
//                             </h3>
//                             <span
//                                 className={`px-2 py-1 rounded text-[10px] font-black uppercase ${isConnected ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-800 text-slate-500'}`}>
//                                 {isConnected ? 'Connected' : 'Disconnected'}
//                             </span>
//                         </div>
//
//                         <div className="space-y-4">
//                             <div className="space-y-2">
//                                 <label className="text-[10px] font-bold text-slate-500 uppercase px-1">Active
//                                     Broker</label>
//                                 <select
//                                     className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-sm text-slate-300 focus:border-emerald-500 outline-none"
//                                     value={activeBroker}
//                                     onChange={(e) => setActiveBroker(e.target.value as BrokerType)}
//                                     disabled={isConnected}
//                                 >
//                                     {Object.values(BrokerType).map(broker => (
//                                         <option key={broker} value={broker}>{broker}</option>
//                                     ))}
//                                 </select>
//                             </div>
//
//                             <button
//                                 onClick={handleConnect}
//                                 disabled={isLoading}
//                                 className={`w-full py-3 rounded-xl text-xs font-black uppercase tracking-widest transition-all flex items-center justify-center gap-2 ${isConnected
//                                     ? 'bg-slate-800 text-red-400 border border-red-500/20 hover:bg-red-500/10'
//                                     : 'bg-emerald-600 text-white shadow-lg shadow-emerald-600/20 hover:bg-emerald-500'
//                                 }`}
//                             >
//                                 {isLoading ? <Loader2 size={14} className="animate-spin"/> :
//                                     isConnected ? <Link2Off size={14}/> : <Link size={14}/>}
//                                 {isConnected ? 'Disconnect Broker' : 'Connect to Gateway'}
//                             </button>
//                             {error && <p className="text-xs text-red-400 text-center">{error}</p>}
//                         </div>
//                     </div>
//
//                     {/* 3. Account Snapshot (Mock for now, could be wired to portfolio endpoint later) */}
//                     <div
//                         className="bg-gradient-to-br from-slate-900 to-slate-950 border border-slate-800/50 rounded-2xl p-6">
//                         <div className="flex items-center gap-2 mb-4">
//                             <Wallet className="text-emerald-500" size={16}/>
//                             <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Equity
//                                 Snapshot</h4>
//                         </div>
//                         <div className="space-y-4">
//                             <div>
//                                 <p className="text-3xl font-black text-slate-100">$42,850.12</p>
//                                 <p className="text-[10px] font-bold text-emerald-500 uppercase flex items-center gap-1">
//                                     <ArrowUpRight size={10}/> +$1,240.50 (2.4%)
//                                 </p>
//                             </div>
//                             <div className="grid grid-cols-2 gap-4 pt-4 border-t border-slate-800">
//                                 <div>
//                                     <p className="text-[9px] font-black text-slate-600 uppercase">Buying Power</p>
//                                     <p className="text-sm font-bold text-slate-300">$171,400</p>
//                                 </div>
//                                 <div>
//                                     <p className="text-[9px] font-black text-slate-600 uppercase">Margin Used</p>
//                                     <p className="text-sm font-bold text-slate-300">$0.00</p>
//                                 </div>
//                             </div>
//                         </div>
//                     </div>
//                 </div>
//
//                 {/* 4. Engine Controls & Active Orders */}
//                 <div className="lg:col-span-2 space-y-6">
//                     <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
//                         <div
//                             className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
//                             <div>
//                                 <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
//                                     <Activity className="text-emerald-500" size={20}/> Execution Engine
//                                 </h3>
//                                 <p className="text-xs text-slate-500">
//                                     Status: <span
//                                     className={`font-bold ${engineStatus === EngineStatus.RUNNING ? 'text-emerald-400' : 'text-slate-400'}`}>{engineStatus.toUpperCase()}</span>
//                                 </p>
//                             </div>
//
//                             <div className="flex gap-2">
//                                 <button
//                                     onClick={handleEngineToggle}
//                                     disabled={!isConnected}
//                                     className={`px-6 py-2.5 rounded-xl text-[10px] font-black uppercase tracking-widest flex items-center gap-2 transition-all ${engineStatus === EngineStatus.RUNNING
//                                         ? 'bg-emerald-500 text-white shadow-lg shadow-emerald-500/20'
//                                         : 'bg-slate-800 text-slate-400 hover:bg-slate-700 disabled:opacity-50'
//                                     }`}
//                                 >
//                                     {engineStatus === EngineStatus.RUNNING
//                                         ? <><Square size={14} className="fill-current"/> STOP ENGINE</>
//                                         : <><Play size={14} className="fill-current"/> START ENGINE</>
//                                     }
//                                 </button>
//                             </div>
//                         </div>
//
//                         {/* Active Orders Table */}
//                         <div className="space-y-4">
//                             <div className="flex items-center justify-between">
//                                 <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Live
//                                     Order Bliss</h4>
//                                 <RefreshCcw
//                                     size={14}
//                                     className="text-slate-600 cursor-pointer hover:text-emerald-400 transition-colors"
//                                     onClick={fetchOrders}
//                                 />
//                             </div>
//
//                             <div className="overflow-x-auto">
//                                 <table className="w-full text-left">
//                                     <thead>
//                                     <tr className="border-b border-slate-800">
//                                         <th className="pb-3 text-[10px] font-black text-slate-600 uppercase">Symbol</th>
//                                         <th className="pb-3 text-[10px] font-black text-slate-600 uppercase">Side</th>
//                                         <th className="pb-3 text-[10px] font-black text-slate-600 uppercase">Qty</th>
//                                         <th className="pb-3 text-[10px] font-black text-slate-600 uppercase">Status</th>
//                                         <th className="pb-3 text-[10px] font-black text-slate-600 uppercase text-right">Price</th>
//                                     </tr>
//                                     </thead>
//                                     <tbody className="divide-y divide-slate-800/50">
//                                     {orders.length === 0 ? (
//                                         <tr>
//                                             <td colSpan={5} className="py-8 text-center text-slate-500 text-xs italic">
//                                                 No active orders found
//                                             </td>
//                                         </tr>
//                                     ) : orders.map((order) => (
//                                         <tr key={order.id} className="group hover:bg-white/5 transition-colors">
//                                             <td className="py-4">
//                                                 <span className="text-xs font-bold text-slate-200">{order.symbol}</span>
//                                                 <p className="text-[9px] text-slate-600 font-mono">{order.time}</p>
//                                             </td>
//                                             <td className="py-4">
//                                                     <span
//                                                         className={`text-[10px] font-black px-2 py-0.5 rounded ${order.side === 'BUY' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
//                                                         {order.side}
//                                                     </span>
//                                             </td>
//                                             <td className="py-4 text-xs text-slate-400 font-mono">{order.qty}</td>
//                                             <td className="py-4">
//                                                 <div className="flex items-center gap-1.5">
//                                                     <div
//                                                         className={`w-1.5 h-1.5 rounded-full ${order.status === 'FILLED' ? 'bg-emerald-500' : 'bg-amber-500 animate-pulse'}`}/>
//                                                     <span
//                                                         className="text-[10px] font-bold text-slate-300 uppercase tracking-tight">{order.status}</span>
//                                                 </div>
//                                             </td>
//                                             <td className="py-4 text-xs font-bold text-slate-200 text-right font-mono">
//                                                 ${order.price?.toFixed(2) || 'MKT'}
//                                             </td>
//                                         </tr>
//                                     ))}
//                                     </tbody>
//                                 </table>
//                             </div>
//                         </div>
//                     </div>
//
//                     {/* 5. Execution Logic & Safety Settings */}
//                     <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
//                         <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-6 flex items-center gap-2">
//                             <Settings size={16} className="text-emerald-500"/> Execution Guardrails
//                         </h4>
//                         <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
//                             <div className="space-y-4">
//                                 <div className="flex justify-between items-center">
//                                     <span
//                                         className="text-xs text-slate-500 font-bold uppercase">Max Position Size</span>
//                                     <span className="text-xs font-mono text-slate-200">10% Portfolio</span>
//                                 </div>
//                                 <div className="flex justify-between items-center">
//                                     <span className="text-xs text-slate-500 font-bold uppercase">Hard Stop Loss</span>
//                                     <span className="text-xs font-mono text-red-400">2.5% per Trade</span>
//                                 </div>
//                             </div>
//                             <div className="space-y-4 border-l border-slate-800 pl-6">
//                                 <div className="flex justify-between items-center">
//                                     <span className="text-xs text-slate-500 font-bold uppercase">Cooldown Period</span>
//                                     <span className="text-xs font-mono text-slate-200">60 Seconds</span>
//                                 </div>
//                                 <div className="flex justify-between items-center">
//                                     <span
//                                         className="text-xs text-slate-500 font-bold uppercase">Slippage Tolerance</span>
//                                     <span className="text-xs font-mono text-amber-400">0.05%</span>
//                                 </div>
//                             </div>
//                         </div>
//                     </div>
//                 </div>
//             </div>
//         </div>
//     );
// };
//
// export default LiveExecution;

'use client'
import React, {useEffect, useMemo, useState} from 'react';
import {
    Activity,
    AlertCircle,
    Eye,
    Link,
    Link2Off,
    Loader2,
    PieChart,
    Play,
    RefreshCcw,
    Settings,
    ShieldAlert,
    Square,
    TrendingUp,
    Wallet
} from "lucide-react";
import {live} from '@/utils/api';
import {BrokerType, ConnectRequest, EngineStatus, ExecutionOrder, LiveStrategy} from '@/types/live';
import {Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis} from 'recharts';

const LiveExecution = () => {
    const [isConnected, setIsConnected] = useState(false);
    const [engineStatus, setEngineStatus] = useState<EngineStatus>(EngineStatus.IDLE);
    const [activeBroker, setActiveBroker] = useState<BrokerType>(BrokerType.PAPER_TRADING);
    const [orders, setOrders] = useState<ExecutionOrder[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Live strategies management
    const [liveStrategies, setLiveStrategies] = useState<LiveStrategy[]>([]);
    const [selectedStrategyIds, setSelectedStrategyIds] = useState<number[]>([]);
    const [strategyPerformance, setStrategyPerformance] = useState<Map<number, any>>(new Map());

    // Portfolio metrics
    const [portfolioEquity, setPortfolioEquity] = useState<any[]>([]);
    const [totalPnL, setTotalPnL] = useState(0);
    const [totalPnLPct, setTotalPnLPct] = useState(0);

    // Initial load and polling
    useEffect(() => {
        fetchStatus();
        loadLiveStrategies();

        const interval = setInterval(() => {
            fetchStatus();
            if (isConnected) {
                fetchOrders();
            }
            if (selectedStrategyIds.length > 0) {
                updateStrategyPerformance();
            }
        }, 3000); // Poll every 3 seconds

        return () => clearInterval(interval);
    }, [isConnected, selectedStrategyIds]);

    const fetchStatus = async () => {
        try {
            const response = await live.getStatus();
            if (response) {
                setIsConnected(response.is_connected);
                setEngineStatus(response.engine_status);
                setActiveBroker(response.active_broker);
            }
        } catch (err) {
            console.error("Failed to fetch live status:", err);
        }
    };

    const fetchOrders = async () => {
        try {
            const response = await live.getOrders();
            if (response) {
                setOrders(response);
            }
        } catch (err) {
            console.error("Failed to fetch orders:", err);
        }
    };

    const loadLiveStrategies = async () => {
        try {
            const response = await live.list();
            setLiveStrategies(response);

            // Auto-select RUNNING strategies
            const runningIds = (response || [])
                .filter((s: LiveStrategy) => s.status === 'RUNNING')
                .map((s: LiveStrategy) => s.id);

            if (runningIds.length > 0) {
                setSelectedStrategyIds(runningIds);
            }
        } catch (err) {
            console.error("Failed to load live strategies:", err);
        }
    };

    const updateStrategyPerformance = async () => {
        if (selectedStrategyIds.length === 0) return;

        try {
            const performanceMap = new Map();
            let totalEquity = 0;
            let totalInitial = 0;
            const equityPoints: any[] = [];

            for (const strategyId of selectedStrategyIds) {
                const details = await live.getDetails(strategyId);
                performanceMap.set(strategyId, details);

                const strategy = details.strategy;
                totalEquity += strategy.current_equity || 0;
                totalInitial += strategy.initial_capital || 0;

                // Aggregate equity curves
                if (details.equity_curve && details.equity_curve.length > 0) {
                    details.equity_curve.forEach((point: any, idx: number) => {
                        if (!equityPoints[idx]) {
                            equityPoints[idx] = {
                                timestamp: point.timestamp,
                                equity: 0
                            };
                        }
                        equityPoints[idx].equity += point.equity;
                    });
                }
            }

            setStrategyPerformance(performanceMap);
            setPortfolioEquity(equityPoints);
            setTotalPnL(totalEquity - totalInitial);
            setTotalPnLPct(totalInitial > 0 ? ((totalEquity - totalInitial) / totalInitial) * 100 : 0);
        } catch (err) {
            console.error("Failed to update strategy performance:", err);
        }
    };

    const handleConnect = async () => {
        setIsLoading(true);
        setError(null);

        const payload: ConnectRequest = {
            broker: activeBroker,
            api_key: null,
            api_secret: null
        };

        try {
            if (isConnected) {
                await live.disconnect();
            } else {
                await live.connect(payload);
            }
            await fetchStatus();
        } catch (err) {
            setError("Failed to change connection status");
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    const handleEngineToggle = async () => {
        if (!isConnected) {
            setError("Please connect to a broker first");
            return;
        }

        try {
            if (engineStatus === EngineStatus.RUNNING) {
                await live.stopEngine();
            } else {
                // When starting, pass selected strategy IDs
                if (selectedStrategyIds.length === 0) {
                    setError("Please select at least one strategy to run");
                    return;
                }

                // Start engine with selected strategies
                await live.startEngine();

                // Start each selected strategy
                for (const strategyId of selectedStrategyIds) {
                    await live.controlStrategy(strategyId, 'start');
                }
            }
            await fetchStatus();
            await loadLiveStrategies();
            setError(null);
        } catch (err) {
            console.error("Failed to toggle engine:", err);
            setError("Failed to toggle engine state");
        }
    };

    const toggleStrategySelection = (strategyId: number) => {
        setSelectedStrategyIds(prev => {
            if (prev.includes(strategyId)) {
                return prev.filter(id => id !== strategyId);
            } else {
                return [...prev, strategyId];
            }
        });
    };

    const runningStrategies = useMemo(() => {
        return liveStrategies.filter(s => s.status === 'RUNNING');
    }, [liveStrategies]);

    const selectedStrategies = useMemo(() => {
        return liveStrategies.filter(s => selectedStrategyIds.includes(s.id));
    }, [liveStrategies, selectedStrategyIds]);

    return (
        <div className="space-y-6 animate-in fade-in duration-700">
            {/* Critical Warning */}
            <div className="bg-amber-500/10 border border-amber-500/20 rounded-2xl p-4 flex gap-4 items-start">
                <ShieldAlert className="text-amber-500 shrink-0" size={24}/>
                <div>
                    <h4 className="text-sm font-bold text-amber-500 uppercase tracking-tight">Real Money Risk Warning</h4>
                    <p className="text-xs text-slate-400 leading-relaxed">
                        Live trading involves real capital risk. Ensure stop-losses are active and monitor positions regularly.
                        Start with <b>Paper Trading</b> to verify connectivity and execution logic.
                    </p>
                </div>
            </div>

            {/* Portfolio Overview - Shows when strategies are selected */}
            {selectedStrategyIds.length > 0 && portfolioEquity.length > 0 && (
                <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 border border-slate-700/50 rounded-2xl p-6">
                    <div className="flex items-center justify-between mb-6">
                        <div>
                            <h3 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
                                <TrendingUp className="text-violet-500" size={28} />
                                Live Portfolio Performance
                            </h3>
                            <p className="text-sm text-slate-400 mt-1">
                                Tracking {selectedStrategyIds.length} {selectedStrategyIds.length === 1 ? 'strategy' : 'strategies'}
                            </p>
                        </div>
                        <div className="text-right">
                            <div className="text-xs text-slate-500 font-bold uppercase mb-1">Total P&L</div>
                            <div className={`text-3xl font-black ${totalPnL >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                {totalPnL >= 0 ? '+' : ''}${totalPnL.toFixed(2)}
                            </div>
                            <div className={`text-lg font-bold ${totalPnLPct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                {totalPnLPct >= 0 ? '+' : ''}{totalPnLPct.toFixed(2)}%
                            </div>
                        </div>
                    </div>

                    {/* Equity Curve */}
                    <div className="bg-slate-950/50 rounded-xl p-4">
                        <ResponsiveContainer width="100%" height={250}>
                            <AreaChart data={portfolioEquity}>
                                <defs>
                                    <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                                        <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />
                                <XAxis
                                    dataKey="timestamp"
                                    stroke="#64748b"
                                    style={{ fontSize: '11px' }}
                                    tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                                />
                                <YAxis
                                    stroke="#64748b"
                                    style={{ fontSize: '11px' }}
                                    tickFormatter={(value) => `$${value.toLocaleString()}`}
                                />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#1e293b',
                                        border: '1px solid #334155',
                                        borderRadius: '12px',
                                        padding: '12px'
                                    }}
                                    formatter={(value: any) => [`$${Number(value).toFixed(2)}`, 'Equity']}
                                    labelFormatter={(label) => new Date(label).toLocaleString()}
                                />
                                <Area
                                    type="monotone"
                                    dataKey="equity"
                                    stroke="#8b5cf6"
                                    strokeWidth={2}
                                    fillOpacity={1}
                                    fill="url(#equityGradient)"
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Strategy Breakdown */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                        {selectedStrategies.map(strategy => {
                            const perf = strategyPerformance.get(strategy.id);
                            return (
                                <div key={strategy.id} className="bg-slate-800/50 rounded-xl p-4">
                                    <div className="flex items-start justify-between mb-3">
                                        <div>
                                            <h4 className="font-bold text-slate-200 text-sm">{strategy.name}</h4>
                                            <p className="text-xs text-slate-500 mt-0.5">
                                                {strategy.deployment_mode === 'paper' ? '📋 Paper' : '🚀 Live'}
                                            </p>
                                        </div>
                                        <span className={`px-2 py-1 rounded text-[10px] font-black uppercase ${
                                            strategy.status === 'RUNNING' ? 'bg-emerald-500/20 text-emerald-400' :
                                                strategy.status === 'PAUSED' ? 'bg-amber-500/20 text-amber-400' :
                                                    'bg-slate-700 text-slate-400'
                                        }`}>
                                            {strategy.status}
                                        </span>
                                    </div>
                                    {perf && (
                                        <div className="space-y-2">
                                            <div className="flex justify-between text-sm">
                                                <span className="text-slate-500">Equity:</span>
                                                <span className="text-slate-200 font-mono font-semibold">
                                                    ${perf.strategy.current_equity?.toFixed(2) || '0.00'}
                                                </span>
                                            </div>
                                            <div className="flex justify-between text-sm">
                                                <span className="text-slate-500">Return:</span>
                                                <span className={`font-semibold ${
                                                    (perf.strategy.total_return_pct || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
                                                }`}>
                                                    {(perf.strategy.total_return_pct || 0) >= 0 ? '+' : ''}{(perf.strategy.total_return_pct || 0).toFixed(2)}%
                                                </span>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column - Broker & Strategy Selection */}
                <div className="lg:col-span-1 space-y-6">
                    {/* Broker Configuration */}
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                        <div className="flex items-center justify-between mb-6">
                            <h3 className="text-sm font-black text-slate-300 uppercase tracking-widest flex items-center gap-2">
                                <Link size={16} className="text-emerald-500"/> Connectivity
                            </h3>
                            <span className={`px-2 py-1 rounded text-[10px] font-black uppercase ${
                                isConnected ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-800 text-slate-500'
                            }`}>
                                {isConnected ? 'Connected' : 'Disconnected'}
                            </span>
                        </div>

                        <div className="space-y-4">
                            <div className="space-y-2">
                                <label className="text-[10px] font-bold text-slate-500 uppercase px-1">Active Broker</label>
                                <select
                                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-sm text-slate-300 focus:border-emerald-500 outline-none"
                                    value={activeBroker}
                                    onChange={(e) => setActiveBroker(e.target.value as BrokerType)}
                                    disabled={isConnected}
                                >
                                    {Object.values(BrokerType).map(broker => (
                                        <option key={broker} value={broker}>{broker}</option>
                                    ))}
                                </select>
                            </div>

                            <button
                                onClick={handleConnect}
                                disabled={isLoading}
                                className={`w-full py-3 rounded-xl text-xs font-black uppercase tracking-widest transition-all flex items-center justify-center gap-2 ${
                                    isConnected
                                        ? 'bg-red-500/20 border border-red-500/50 text-red-400 hover:bg-red-500/30'
                                        : 'bg-emerald-500/20 border border-emerald-500/50 text-emerald-400 hover:bg-emerald-500/30'
                                }`}
                            >
                                {isLoading ? (
                                    <Loader2 size={14} className="animate-spin" />
                                ) : isConnected ? (
                                    <><Link2Off size={14} /> Disconnect</>
                                ) : (
                                    <><Link size={14} /> Connect</>
                                )}
                            </button>
                        </div>
                    </div>

                    {/* Strategy Selection */}
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-sm font-black text-slate-300 uppercase tracking-widest flex items-center gap-2">
                                <PieChart size={16} className="text-violet-500"/> Active Strategies
                            </h3>
                            <button
                                onClick={loadLiveStrategies}
                                className="p-1 hover:bg-slate-800 rounded transition-colors"
                            >
                                <RefreshCcw size={14} className="text-slate-500" />
                            </button>
                        </div>

                        <div className="space-y-2 max-h-80 overflow-y-auto">
                            {liveStrategies.length === 0 ? (
                                <div className="text-center py-8 text-slate-500 text-xs">
                                    <Eye size={32} className="mx-auto mb-2 opacity-50" />
                                    <p>No deployed strategies</p>
                                    <p className="text-[10px] mt-1">Deploy from Backtest or Marketplace</p>
                                </div>
                            ) : (
                                liveStrategies.map(strategy => (
                                    <label
                                        key={strategy.id}
                                        className={`flex items-start gap-3 p-3 rounded-xl cursor-pointer transition-all ${
                                            selectedStrategyIds.includes(strategy.id)
                                                ? 'bg-violet-500/20 border border-violet-500/50'
                                                : 'bg-slate-800/30 border border-slate-700/50 hover:bg-slate-800/50'
                                        }`}
                                    >
                                        <input
                                            type="checkbox"
                                            checked={selectedStrategyIds.includes(strategy.id)}
                                            onChange={() => toggleStrategySelection(strategy.id)}
                                            className="mt-1"
                                        />
                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-center justify-between gap-2">
                                                <div className="font-semibold text-slate-200 text-sm truncate">
                                                    {strategy.name}
                                                </div>
                                                <span className={`px-1.5 py-0.5 rounded text-[9px] font-black uppercase shrink-0 ${
                                                    strategy.status === 'RUNNING' ? 'bg-emerald-500/20 text-emerald-400' :
                                                        strategy.status === 'PAUSED' ? 'bg-amber-500/20 text-amber-400' :
                                                            'bg-slate-700 text-slate-400'
                                                }`}>
                                                    {strategy.status}
                                                </span>
                                            </div>
                                            <div className="text-[10px] text-slate-500 mt-0.5">
                                                {strategy.symbols?.join(', ')} • {strategy.deployment_mode}
                                            </div>
                                        </div>
                                    </label>
                                ))
                            )}
                        </div>
                    </div>

                    {/* Account Summary */}
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                        <div className="flex items-center justify-between mb-6">
                            <h3 className="text-sm font-black text-slate-300 uppercase tracking-widest flex items-center gap-2">
                                <Wallet size={16} className="text-emerald-500"/> Account
                            </h3>
                        </div>
                        <div className="space-y-4">
                            <div>
                                <p className="text-[9px] font-black text-slate-600 uppercase">Balance</p>
                                <p className="text-2xl font-bold text-emerald-400">$50,000.00</p>
                            </div>
                            <div className="grid grid-cols-2 gap-4 pt-4 border-t border-slate-800">
                                <div>
                                    <p className="text-[9px] font-black text-slate-600 uppercase">Buying Power</p>
                                    <p className="text-sm font-bold text-slate-300">$171,400</p>
                                </div>
                                <div>
                                    <p className="text-[9px] font-black text-slate-600 uppercase">Margin Used</p>
                                    <p className="text-sm font-bold text-slate-300">$0.00</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Right Column - Engine Controls & Active Orders */}
                <div className="lg:col-span-2 space-y-6">
                    {/* Engine Controls */}
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
                            <div>
                                <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                                    <Activity className="text-emerald-500" size={20}/> Execution Engine
                                </h3>
                                <p className="text-xs text-slate-500">
                                    Status: <span className={`font-bold ${
                                    engineStatus === EngineStatus.RUNNING ? 'text-emerald-400' : 'text-slate-400'
                                }`}>{engineStatus.toUpperCase()}</span>
                                </p>
                                {selectedStrategyIds.length > 0 && (
                                    <p className="text-xs text-violet-400 mt-1">
                                        {selectedStrategyIds.length} {selectedStrategyIds.length === 1 ? 'strategy' : 'strategies'} ready
                                    </p>
                                )}
                            </div>

                            <div className="flex gap-2">
                                <button
                                    onClick={handleEngineToggle}
                                    disabled={!isConnected}
                                    className={`px-6 py-2.5 rounded-xl text-[10px] font-black uppercase tracking-widest flex items-center gap-2 transition-all ${
                                        engineStatus === EngineStatus.RUNNING
                                            ? 'bg-red-500 text-white shadow-lg shadow-red-500/20'
                                            : 'bg-emerald-500 text-white shadow-lg shadow-emerald-500/20 hover:bg-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed'
                                    }`}
                                >
                                    {engineStatus === EngineStatus.RUNNING
                                        ? <><Square size={14} className="fill-current"/> STOP ENGINE</>
                                        : <><Play size={14} className="fill-current"/> START ENGINE</>
                                    }
                                </button>
                            </div>
                        </div>

                        {/* Error Display */}
                        {error && (
                            <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-2 text-red-400 text-sm">
                                <AlertCircle size={16} />
                                {error}
                            </div>
                        )}

                        {/* Active Orders Table */}
                        <div className="space-y-4">
                            <div className="flex items-center justify-between">
                                <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest">
                                    Live Order Book
                                </h4>
                                <RefreshCcw
                                    size={14}
                                    className="text-slate-600 cursor-pointer hover:text-emerald-400 transition-colors"
                                    onClick={fetchOrders}
                                />
                            </div>

                            <div className="overflow-x-auto">
                                <table className="w-full text-left">
                                    <thead>
                                    <tr className="border-b border-slate-800">
                                        <th className="pb-3 text-[10px] font-black text-slate-600 uppercase">Symbol</th>
                                        <th className="pb-3 text-[10px] font-black text-slate-600 uppercase">Side</th>
                                        <th className="pb-3 text-[10px] font-black text-slate-600 uppercase">Qty</th>
                                        <th className="pb-3 text-[10px] font-black text-slate-600 uppercase">Status</th>
                                        <th className="pb-3 text-[10px] font-black text-slate-600 uppercase text-right">Price</th>
                                    </tr>
                                    </thead>
                                    <tbody className="divide-y divide-slate-800/50">
                                    {orders.length === 0 ? (
                                        <tr>
                                            <td colSpan={5} className="py-8 text-center text-slate-500 text-xs italic">
                                                {engineStatus === EngineStatus.RUNNING
                                                    ? 'No active orders yet - waiting for signals...'
                                                    : 'No active orders. Start the engine to begin trading.'}
                                            </td>
                                        </tr>
                                    ) : orders.map((order) => (
                                        <tr key={order.id} className="group hover:bg-white/5 transition-colors">
                                            <td className="py-4">
                                                <span className="text-xs font-bold text-slate-200">{order.symbol}</span>
                                                <p className="text-[9px] text-slate-600 font-mono">{order.time}</p>
                                            </td>
                                            <td className="py-4">
                                                <span className={`text-[10px] font-black px-2 py-0.5 rounded ${
                                                    order.side === 'BUY' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'
                                                }`}>
                                                    {order.side}
                                                </span>
                                            </td>
                                            <td className="py-4 text-xs text-slate-400 font-mono">{order.qty}</td>
                                            <td className="py-4">
                                                <div className="flex items-center gap-1.5">
                                                    <div className={`w-1.5 h-1.5 rounded-full ${
                                                        order.status === 'FILLED' ? 'bg-emerald-500' : 'bg-amber-500 animate-pulse'
                                                    }`}/>
                                                    <span className="text-[10px] font-bold text-slate-300 uppercase tracking-tight">
                                                        {order.status}
                                                    </span>
                                                </div>
                                            </td>
                                            <td className="py-4 text-xs font-bold text-slate-200 text-right font-mono">
                                                ${order.price?.toFixed(2) || 'MKT'}
                                            </td>
                                        </tr>
                                    ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>

                    {/* Execution Guardrails */}
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                        <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-6 flex items-center gap-2">
                            <Settings size={16} className="text-emerald-500"/> Execution Guardrails
                        </h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-4">
                                <div className="flex justify-between items-center">
                                    <span className="text-xs text-slate-500 font-bold uppercase">Max Position Size</span>
                                    <span className="text-xs font-mono text-slate-200">10% Portfolio</span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-xs text-slate-500 font-bold uppercase">Hard Stop Loss</span>
                                    <span className="text-xs font-mono text-red-400">2.5% per Trade</span>
                                </div>
                            </div>
                            <div className="space-y-4 border-l border-slate-800 pl-6">
                                <div className="flex justify-between items-center">
                                    <span className="text-xs text-slate-500 font-bold uppercase">Cooldown Period</span>
                                    <span className="text-xs font-mono text-slate-200">60 Seconds</span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-xs text-slate-500 font-bold uppercase">Slippage Tolerance</span>
                                    <span className="text-xs font-mono text-amber-400">0.05%</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default LiveExecution;
