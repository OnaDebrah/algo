'use client'
import React, { useState } from 'react';
import {
    TrendingUp, Calculator, PieChart, BarChart3,
    Settings, Play, Target, ShieldCheck,
    Layers, DollarSign, ArrowRight, Info
} from "lucide-react";
import {
    ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis,
    ZAxis, Tooltip, CartesianGrid, Cell, PieChart as RePie, Pie,
    LineChart, Line, AreaChart, Area
} from 'recharts';

const OPTIMIZATION_METHODS = [
    { id: 'sharpe', label: 'Maximum Sharpe Ratio', desc: 'Risk-adjusted return focus' },
    { id: 'min_vol', label: 'Minimum Volatility', desc: 'Lowest possible risk' },
    { id: 'equal', label: 'Equal Weight', desc: '1/N diversification' },
    { id: 'risk_parity', label: 'Risk Parity', desc: 'Equalize risk contribution' },
    { id: 'black_litterman', label: 'Black-Litterman', desc: 'Market views + MPT' }
];

import { market } from "@/utils/api";

const PortfolioOptimization = () => {
    const [isOptimizing, setIsOptimizing] = useState(false);
    const [method, setMethod] = useState('sharpe');
    const [symbolsInput, setSymbolsInput] = useState('AAPL, MSFT, GOOGL, AMZN, TSLA');
    const [detectedAssets, setDetectedAssets] = useState(5);
    const [assetPrices, setAssetPrices] = useState<Record<string, number>>({});

    // Mock data derived from PortfolioOptimizer logic
    const results = {
        metrics: { return: 0.185, volatility: 0.124, sharpe: 1.49 },
        allocation: [
            { symbol: 'AAPL', weight: 0.35, color: '#10b981' },
            { symbol: 'MSFT', weight: 0.25, color: '#3b82f6' },
            { symbol: 'GOOGL', weight: 0.20, color: '#8b5cf6' },
            { symbol: 'AMZN', weight: 0.15, color: '#f59e0b' },
            { symbol: 'TSLA', weight: 0.05, color: '#ef4444' },
        ],
        frontier: Array.from({ length: 20 }, (_, i) => ({
            risk: 8 + i * 1.5,
            return: 10 + Math.sqrt(i) * 5
        }))
    };

    const handleOptimize = async () => {
        setIsOptimizing(true);
        // Validate symbols and get prices
        const symbols = symbolsInput.split(',').map(s => s.trim().toUpperCase()).filter(s => s);
        if (symbols.length > 0) {
            try {
                const res = await market.getQuotes(symbols);
                if (res.data) {
                    const priceMap: Record<string, number> = {};
                    res.data.forEach((q: any) => {
                        priceMap[q.symbol] = q.price;
                    });
                    setAssetPrices(priceMap);
                    setDetectedAssets(res.data.length);
                }
            } catch (e) {
                console.error("Failed to fetch quotes", e);
            }
        }

        setTimeout(() => setIsOptimizing(false), 2000);
    };

    return (
        <div className="space-y-8 animate-in fade-in duration-700">
            {/* 1. Header Section */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h2 className="text-3xl font-bold text-slate-100 tracking-tight flex items-center gap-3">
                        <Calculator className="text-emerald-400" size={32} />
                        Portfolio <span className="text-slate-400 font-normal">Optimizer</span>
                    </h2>
                    <p className="text-slate-400 text-sm mt-1">Modern Portfolio Theory & Markowitz Efficient Frontier Analysis</p>
                </div>
                <button
                    onClick={handleOptimize}
                    className="flex items-center space-x-2 px-8 py-4 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white rounded-2xl font-bold text-sm transition-all shadow-xl shadow-emerald-500/20"
                >
                    {isOptimizing ? <Settings className="animate-spin" size={18} /> : <Play size={18} />}
                    <span>{isOptimizing ? "Running Solver..." : "Optimize Portfolio"}</span>
                </button>
            </div>

            {/* 2. Strategy & Parameters (Selection Section) */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                <div className="lg:col-span-1 space-y-4">
                    <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest px-2">Asset Selection</label>
                    <div className="p-5 bg-slate-900/50 border border-slate-800/50 rounded-2xl space-y-4">
                        <input
                            type="text"
                            value={symbolsInput}
                            onChange={(e) => setSymbolsInput(e.target.value)}
                            className="w-full bg-slate-950 border border-slate-800 rounded-xl py-3 px-4 text-xs font-mono text-emerald-400 focus:border-emerald-500 outline-none transition-all"
                            placeholder="Symbols (comma separated)"
                        />
                        <div className="flex items-center justify-between text-[10px] text-slate-500 font-bold uppercase">
                            <span>{detectedAssets} Assets Detected</span>
                            <span className="text-emerald-500">Ready</span>
                        </div>
                    </div>

                    <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest px-2">Optimization Method</label>
                    <div className="space-y-2">
                        {OPTIMIZATION_METHODS.map((m) => (
                            <button
                                key={m.id}
                                onClick={() => setMethod(m.id)}
                                className={`w-full text-left p-4 rounded-2xl border transition-all ${method === m.id
                                        ? 'bg-emerald-500/10 border-emerald-500/50 shadow-lg shadow-emerald-500/5'
                                        : 'bg-slate-900/30 border-slate-800/50 hover:border-slate-700'
                                    }`}
                            >
                                <p className={`text-xs font-bold ${method === m.id ? 'text-emerald-400' : 'text-slate-300'}`}>{m.label}</p>
                                <p className="text-[10px] text-slate-500 mt-0.5">{m.desc}</p>
                            </button>
                        ))}
                    </div>
                </div>

                {/* 3. Results Dashboard */}
                <div className="lg:col-span-3 space-y-6">
                    {/* Key Metrics Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {[
                            { label: 'Expected Annual Return', value: '18.52%', icon: TrendingUp, color: 'text-emerald-400' },
                            { label: 'Annual Volatility', value: '12.41%', icon: ShieldCheck, color: 'text-blue-400' },
                            { label: 'Portfolio Sharpe Ratio', value: '1.49', icon: Target, color: 'text-amber-400' },
                        ].map((metric, i) => (
                            <div key={i} className="bg-slate-900/40 border border-slate-800/50 p-6 rounded-2xl relative overflow-hidden group">
                                <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                                    <metric.icon size={48} />
                                </div>
                                <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">{metric.label}</p>
                                <p className={`text-2xl font-bold ${metric.color}`}>{metric.value}</p>
                            </div>
                        ))}
                    </div>

                    {/* Charts Row */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Allocation Chart */}
                        <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                            <div className="flex items-center justify-between mb-6">
                                <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest flex items-center gap-2">
                                    <PieChart size={14} className="text-emerald-500" />
                                    Optimal Allocation
                                </h4>
                            </div>
                            <div className="h-[280px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <RePie>
                                        <Pie
                                            data={results.allocation}
                                            innerRadius={70}
                                            outerRadius={100}
                                            paddingAngle={8}
                                            dataKey="weight"
                                            stroke="none"
                                        >
                                            {results.allocation.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={entry.color} />
                                            ))}
                                        </Pie>
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }}
                                            formatter={(val: number) => [`${(val * 100).toFixed(2)}%`, 'Allocation']}
                                        />
                                    </RePie>
                                </ResponsiveContainer>
                            </div>
                            <div className="grid grid-cols-3 gap-2 mt-4">
                                {results.allocation.map((item, i) => (
                                    <div key={i} className="flex flex-col items-center p-2 bg-white/5 rounded-xl">
                                        <span className="text-[10px] font-black text-slate-200">{item.symbol}</span>
                                        <span className="text-[10px] font-mono text-emerald-400">{(item.weight * 100).toFixed(0)}%</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Efficient Frontier */}
                        <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                            <div className="flex items-center justify-between mb-6">
                                <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest flex items-center gap-2">
                                    <BarChart3 size={14} className="text-blue-500" />
                                    Efficient Frontier
                                </h4>
                                <Info size={14} className="text-slate-600 cursor-help" />
                            </div>
                            <div className="h-[280px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: -20 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                                        <XAxis type="number" dataKey="risk" name="Risk" unit="%" stroke="#64748b" fontSize={10} tickLine={false} axisLine={false} />
                                        <YAxis type="number" dataKey="return" name="Return" unit="%" stroke="#64748b" fontSize={10} tickLine={false} axisLine={false} />
                                        <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }} />
                                        <Scatter name="Portfolios" data={results.frontier} fill="#10b981" fillOpacity={0.6} line={{ stroke: '#10b981', strokeWidth: 2 }} />
                                    </ScatterChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>

                    {/* Capital Allocation Tool */}
                    <div className="bg-emerald-500/5 border border-emerald-500/10 rounded-2xl p-6 flex flex-col md:flex-row items-center justify-between gap-6">
                        <div className="space-y-1">
                            <p className="text-xs font-bold text-emerald-400 uppercase tracking-wider">Capital Allocation Tool</p>
                            <p className="text-sm text-slate-400">Calculate share counts for your investment amount</p>
                        </div>
                        <div className="flex items-center gap-4 bg-slate-950 p-2 rounded-2xl border border-white/5 w-full md:w-auto">
                            <div className="flex items-center gap-2 px-4">
                                <DollarSign size={16} className="text-slate-500" />
                                <input type="number" defaultValue="100000" className="bg-transparent text-slate-100 font-bold outline-none w-32" />
                            </div>
                            <button className="px-6 py-2 bg-emerald-600 text-white rounded-xl text-xs font-bold hover:bg-emerald-500 transition-all">
                                Generate Trade List
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PortfolioOptimization;