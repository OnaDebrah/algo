'use client'
import React, {useState, useEffect} from 'react';
import {
    Zap, ShieldAlert, Link, Link2Off, Activity,
    RefreshCcw, Play, Square, Settings,
    History, Wallet, ArrowUpRight, ArrowDownRight,
    Search, Filter, CheckCircle2, AlertCircle,
    Loader2
} from "lucide-react";
import {live} from '@/utils/api';
import {BrokerType, ConnectRequest, EngineStatus, ExecutionOrder, LiveStatus} from '@/types/all_types';

const LiveExecution = () => {
    const [isConnected, setIsConnected] = useState(false);
    const [engineStatus, setEngineStatus] = useState<EngineStatus>(EngineStatus.IDLE);
    const [activeBroker, setActiveBroker] = useState<BrokerType>(BrokerType.PAPER_TRADING);
    const [orders, setOrders] = useState<ExecutionOrder[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Initial load and polling
    useEffect(() => {
        fetchStatus();
        const interval = setInterval(() => {
            fetchStatus();
            if (isConnected) fetchOrders();
        }, 3000); // Poll every 3 seconds

        return () => clearInterval(interval);
    }, [isConnected]);

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
        if (!isConnected) return;

        try {
            if (engineStatus === EngineStatus.RUNNING) {
                await live.stopEngine();
            } else {
                await live.startEngine();
            }
            await fetchStatus();
        } catch (err) {
            console.error("Failed to toggle engine:", err);
            setError("Failed to toggle engine state");
        }
    };

    return (
        <div className="space-y-6 animate-in fade-in duration-700">
            {/* 1. Critical Warning */}
            <div className="bg-amber-500/10 border border-amber-500/20 rounded-2xl p-4 flex gap-4 items-start">
                <ShieldAlert className="text-amber-500 shrink-0" size={24}/>
                <div>
                    <h4 className="text-sm font-bold text-amber-500 uppercase tracking-tight">Real Money Risk
                        Warning</h4>
                    <p className="text-xs text-slate-400 leading-relaxed">
                        Live trading involves real capital risk. Ensure stop-losses are active and monitor positions
                        regularly.
                        Start with <b>Paper Trading</b> to verify connectivity and execution logic.
                    </p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* 2. Broker Configuration */}
                <div className="lg:col-span-1 space-y-6">
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                        <div className="flex items-center justify-between mb-6">
                            <h3 className="text-sm font-black text-slate-300 uppercase tracking-widest flex items-center gap-2">
                                <Link size={16} className="text-emerald-500"/> Connectivity
                            </h3>
                            <span
                                className={`px-2 py-1 rounded text-[10px] font-black uppercase ${isConnected ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-800 text-slate-500'}`}>
                                {isConnected ? 'Connected' : 'Disconnected'}
                            </span>
                        </div>

                        <div className="space-y-4">
                            <div className="space-y-2">
                                <label className="text-[10px] font-bold text-slate-500 uppercase px-1">Active
                                    Broker</label>
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
                                className={`w-full py-3 rounded-xl text-xs font-black uppercase tracking-widest transition-all flex items-center justify-center gap-2 ${isConnected
                                    ? 'bg-slate-800 text-red-400 border border-red-500/20 hover:bg-red-500/10'
                                    : 'bg-emerald-600 text-white shadow-lg shadow-emerald-600/20 hover:bg-emerald-500'
                                }`}
                            >
                                {isLoading ? <Loader2 size={14} className="animate-spin"/> :
                                    isConnected ? <Link2Off size={14}/> : <Link size={14}/>}
                                {isConnected ? 'Disconnect Broker' : 'Connect to Gateway'}
                            </button>
                            {error && <p className="text-xs text-red-400 text-center">{error}</p>}
                        </div>
                    </div>

                    {/* 3. Account Snapshot (Mock for now, could be wired to portfolio endpoint later) */}
                    <div
                        className="bg-gradient-to-br from-slate-900 to-slate-950 border border-slate-800/50 rounded-2xl p-6">
                        <div className="flex items-center gap-2 mb-4">
                            <Wallet className="text-emerald-500" size={16}/>
                            <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Equity
                                Snapshot</h4>
                        </div>
                        <div className="space-y-4">
                            <div>
                                <p className="text-3xl font-black text-slate-100">$42,850.12</p>
                                <p className="text-[10px] font-bold text-emerald-500 uppercase flex items-center gap-1">
                                    <ArrowUpRight size={10}/> +$1,240.50 (2.4%)
                                </p>
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

                {/* 4. Engine Controls & Active Orders */}
                <div className="lg:col-span-2 space-y-6">
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                        <div
                            className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
                            <div>
                                <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                                    <Activity className="text-emerald-500" size={20}/> Execution Engine
                                </h3>
                                <p className="text-xs text-slate-500">
                                    Status: <span
                                    className={`font-bold ${engineStatus === EngineStatus.RUNNING ? 'text-emerald-400' : 'text-slate-400'}`}>{engineStatus.toUpperCase()}</span>
                                </p>
                            </div>

                            <div className="flex gap-2">
                                <button
                                    onClick={handleEngineToggle}
                                    disabled={!isConnected}
                                    className={`px-6 py-2.5 rounded-xl text-[10px] font-black uppercase tracking-widest flex items-center gap-2 transition-all ${engineStatus === EngineStatus.RUNNING
                                        ? 'bg-emerald-500 text-white shadow-lg shadow-emerald-500/20'
                                        : 'bg-slate-800 text-slate-400 hover:bg-slate-700 disabled:opacity-50'
                                    }`}
                                >
                                    {engineStatus === EngineStatus.RUNNING
                                        ? <><Square size={14} className="fill-current"/> STOP ENGINE</>
                                        : <><Play size={14} className="fill-current"/> START ENGINE</>
                                    }
                                </button>
                            </div>
                        </div>

                        {/* Active Orders Table */}
                        <div className="space-y-4">
                            <div className="flex items-center justify-between">
                                <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Live
                                    Order Bliss</h4>
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
                                                No active orders found
                                            </td>
                                        </tr>
                                    ) : orders.map((order) => (
                                        <tr key={order.id} className="group hover:bg-white/5 transition-colors">
                                            <td className="py-4">
                                                <span className="text-xs font-bold text-slate-200">{order.symbol}</span>
                                                <p className="text-[9px] text-slate-600 font-mono">{order.time}</p>
                                            </td>
                                            <td className="py-4">
                                                    <span
                                                        className={`text-[10px] font-black px-2 py-0.5 rounded ${order.side === 'BUY' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                                                        {order.side}
                                                    </span>
                                            </td>
                                            <td className="py-4 text-xs text-slate-400 font-mono">{order.qty}</td>
                                            <td className="py-4">
                                                <div className="flex items-center gap-1.5">
                                                    <div
                                                        className={`w-1.5 h-1.5 rounded-full ${order.status === 'FILLED' ? 'bg-emerald-500' : 'bg-amber-500 animate-pulse'}`}/>
                                                    <span
                                                        className="text-[10px] font-bold text-slate-300 uppercase tracking-tight">{order.status}</span>
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

                    {/* 5. Execution Logic & Safety Settings */}
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                        <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-6 flex items-center gap-2">
                            <Settings size={16} className="text-emerald-500"/> Execution Guardrails
                        </h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-4">
                                <div className="flex justify-between items-center">
                                    <span
                                        className="text-xs text-slate-500 font-bold uppercase">Max Position Size</span>
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
                                    <span
                                        className="text-xs text-slate-500 font-bold uppercase">Slippage Tolerance</span>
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