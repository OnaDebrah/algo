'use client'
import React, {useState} from 'react';
import {
    AlertCircle,
    BarChart3,
    Calculator,
    DollarSign, Download, FileText,
    Info,
    PieChart,
    Play,
    Settings,
    ShieldCheck,
    Target,
    TrendingUp
} from "lucide-react";
import {
    CartesianGrid,
    Cell,
    Pie,
    PieChart as RePie,
    ResponsiveContainer,
    Scatter,
    ScatterChart,
    Tooltip,
    XAxis,
    YAxis
} from 'recharts';

import {market, optimization, portfolioHelpers} from "@/utils/api";
import {QuoteData} from "@/types/all_types";
import {FrontierPortfolio, OptimizationResponse} from "@/types/optimise";
import {formatCurrency, formatPercent} from "@/utils/formatters";

const OPTIMIZATION_METHODS = [
    {id: 'sharpe', label: 'Maximum Sharpe Ratio', desc: 'Risk-adjusted return focus', params: ['lookback', 'riskFree']},
    {id: 'min_vol', label: 'Minimum Volatility', desc: 'Lowest possible risk', params: ['lookback']},
    {id: 'equal', label: 'Equal Weight', desc: '1/N diversification', params: ['lookback']},
    {id: 'risk_parity', label: 'Risk Parity', desc: 'Equalize risk contribution', params: ['lookback']},
    {id: 'target', label: 'Target Return', desc: 'Achieve specific return', params: ['lookback', 'targetReturn']},
    {
        id: 'litterman',
        label: 'Black-Litterman',
        desc: 'Market views + MPT',
        params: ['lookback', 'confidence', 'views']
    },
];

interface AllocationItem {
    symbol: string;
    weight: number;
    color: string;
}

interface FrontierPoint {
    risk: number;
    return: number;
    isCurrent?: boolean;
    label?: string;
}

const COLORS = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#84cc16'];

const PortfolioOptimization = () => {
    const [isOptimizing, setIsOptimizing] = useState(false);
    const [method, setMethod] = useState('sharpe');
    const [symbolsInput, setSymbolsInput] = useState('AAPL, MSFT, GOOGL, AMZN, TSLA');
    const [detectedAssets, setDetectedAssets] = useState(0);
    const [assetPrices, setAssetPrices] = useState<Record<string, number>>({});
    const [capitalAmount, setCapitalAmount] = useState(100000);
    const [trades, setTrades] = useState<any[] | null>(null);

    // Parameter states
    const [lookbackDays, setLookbackDays] = useState(252);
    const [riskFreeRate, setRiskFreeRate] = useState(0.02);
    const [targetReturn, setTargetReturn] = useState(0.15);
    const [confidence, setConfidence] = useState(0.5);
    const [viewsInput, setViewsInput] = useState('');

    // API Response States
    const [optimizationResult, setOptimizationResult] = useState<OptimizationResponse | null>(null);
    const [frontierData, setFrontierData] = useState<FrontierPoint[]>([]);
    const [allocation, setAllocation] = useState<AllocationItem[]>([]);
    const [error, setError] = useState<string | null>(null);

    const selectedMethod = OPTIMIZATION_METHODS.find(m => m.id === method);

    const handleOptimize = async () => {
        setIsOptimizing(true);
        setError(null);

        try {
            // Parse and validate symbols
            const symbols = symbolsInput.split(',').map(s => s.trim().toUpperCase()).filter(s => s);

            if (symbols.length < 2) {
                setError('Please enter at least 2 symbols');
                setIsOptimizing(false);
                return;
            }

            // Validate symbols
            portfolioHelpers.validateSymbols(symbols);

            // Fetch current prices
            const quotes = await market.getQuotes(symbols);
            if (quotes) {
                const priceMap: Record<string, number> = {};
                quotes.forEach((q: QuoteData) => {
                    priceMap[q.symbol] = q.price;
                });
                setAssetPrices(priceMap);
                setDetectedAssets(quotes.length);
            }

            // Run optimization based on selected method
            let result: OptimizationResponse;

            switch (method) {
                case 'sharpe':
                    result = await optimization.sharpe({
                        symbols,
                        lookback_days: lookbackDays,
                        risk_free_rate: riskFreeRate
                    });
                    break;

                case 'min_vol':
                    result = await optimization.minVolatility({
                        symbols,
                        lookback_days: lookbackDays
                    });
                    break;

                case 'equal':
                    result = await optimization.equalWeight({
                        symbols,
                        lookback_days: lookbackDays
                    });
                    break;

                case 'risk_parity':
                    result = await optimization.riskParity({
                        symbols,
                        lookback_days: lookbackDays
                    });
                    break;

                case 'litterman':
                    // Parse views from input (format: "AAPL:0.12,MSFT:0.08")
                    const views: Record<string, number> = {};
                    if (viewsInput.trim()) {
                        viewsInput.split(',').forEach(view => {
                            const [symbol, returnValue] = view.split(':').map(s => s.trim());
                            if (symbol && returnValue) {
                                views[symbol.toUpperCase()] = parseFloat(returnValue);
                            }
                        });
                    }

                    result = await optimization.blackLitterman({
                        symbols,
                        lookback_days: lookbackDays,
                        views,
                        confidence
                    });
                    break;

                case 'target':
                    result = await optimization.targetReturn({
                        symbols,
                        lookback_days: lookbackDays,
                        target_return: targetReturn
                    });
                    break;

                default:
                    result = await optimization.sharpe({
                        symbols,
                        lookback_days: lookbackDays
                    });
            }

            setOptimizationResult(result);

            // Transform weights into allocation array with colors
            const allocationData: AllocationItem[] = Object.entries(result.weights)
                .map(([symbol, weight], index) => ({
                    symbol,
                    weight,
                    color: COLORS[index % COLORS.length]
                }))
                .sort((a, b) => b.weight - a.weight);

            setAllocation(allocationData);

            // Fetch efficient frontier data
            const frontierResult = await optimization.efficientFrontier({
                symbols,
                num_portfolios: Math.max(50, symbols.length * 5),
                lookback_days: lookbackDays
            });

            // Transform frontier data and add current portfolio point
            const frontierPoints: FrontierPoint[] = frontierResult.portfolios.map((p: FrontierPortfolio) => ({
                risk: p.volatility * 100,
                return: p['return'] * 100,
                isCurrent: false
            }));

            // Add the current optimized portfolio as a highlighted point
            const currentPoint: FrontierPoint = {
                risk: result.volatility * 100,
                return: result.expected_return * 100,
                isCurrent: true,
                label: result.method
            };

            // Insert current point into the frontier data
            setFrontierData([...frontierPoints, currentPoint]);

        } catch (err: any) {
            console.error('Optimization error:', err);

            let errorMessage = 'Failed to optimize portfolio';

            if (err.message && typeof err.message === 'string') {
                errorMessage = err.message;
            } else if (err.response?.data?.detail) {
                const detail = err.response.data.detail;
                errorMessage = Array.isArray(detail)
                    ? detail.map((e: any) => e.msg).join(', ')
                    : typeof detail === 'string' ? detail : JSON.stringify(detail);
            } else if (typeof err === 'string') {
                errorMessage = err;
            }

            setError(String(errorMessage));
        } finally {
            setIsOptimizing(false);
        }
    };

    const handleGenerateTradeList = () => {
        if (!optimizationResult || !assetPrices) return;

        let totalSpent = 0;
        const list = Object.entries(optimizationResult.weights).map(([symbol, weight]) => {
            const targetValue = capitalAmount * weight;
            const price = assetPrices[symbol] || 0;
            const shares = price > 0 ? Math.floor(targetValue / price) : 0;
            const actualCost = shares * price;
            totalSpent += actualCost;

            return {
                symbol,
                targetWeight: weight,
                targetAllocation: targetValue,
                price: price,
                shares,
                actualCost: actualCost,
                drift: ((actualCost / capitalAmount) - weight) // Difference between target and actual due to rounding
            };
        }).sort((a, b) => b.actualCost - a.actualCost);

        setTrades(list);
    };

// 4. Utility to export to CSV
    const downloadCSV = () => {
        if (!trades) return;
        const headers = ["Symbol", "Shares", "Price", "Cost", "Target Weight"];
        const rows = trades.map(t => [
            t.symbol,
            t.shares,
            t.price.toFixed(2),
            t.actualCost.toFixed(2),
            (t.targetWeight * 100).toFixed(2) + "%"
        ]);

        const csvContent = [headers, ...rows].map(e => e.join(",")).join("\n");
        const blob = new Blob([csvContent], {type: 'text/csv'});
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.setAttribute('href', url);
        a.setAttribute('download', `trades_${new Date().toISOString().split('T')[0]}.csv`);
        a.click();
    };

    // Display metrics
    const metrics = optimizationResult ? {
        return: formatPercent(optimizationResult.expected_return),
        volatility: formatPercent(optimizationResult.volatility),
        sharpe: optimizationResult.sharpe_ratio.toFixed(2)
    } : {
        return: '--',
        volatility: '--',
        sharpe: '--'
    };

    return (
        <div className="space-y-8 animate-in fade-in duration-700">
            {/* Error Banner */}
            {error && (
                <div className="bg-red-500/10 border border-red-500/50 rounded-2xl p-4 flex items-center gap-3">
                    <AlertCircle className="text-red-400" size={20}/>
                    <p className="text-red-400 text-sm font-medium">{error}</p>
                </div>
            )}

            {/* 1. Header Section */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h2 className="text-3xl font-bold text-slate-100 tracking-tight flex items-center gap-3">
                        <Calculator className="text-emerald-400" size={32}/>
                        Portfolio <span className="text-slate-400 font-normal">Optimizer</span>
                    </h2>
                    <p className="text-slate-400 text-sm mt-1">Modern Portfolio Theory & Markowitz Efficient Frontier
                        Analysis</p>
                </div>
                <button
                    onClick={handleOptimize}
                    disabled={isOptimizing}
                    className="flex items-center space-x-2 px-8 py-4 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white rounded-2xl font-bold text-sm transition-all shadow-xl shadow-emerald-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {isOptimizing ? <Settings className="animate-spin" size={18}/> : <Play size={18}/>}
                    <span>{isOptimizing ? "Running Solver..." : "Optimize Portfolio"}</span>
                </button>
            </div>

            {/* 2. Strategy & Parameters (Selection Section) */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                <div className="lg:col-span-1 space-y-4">
                    {/* Asset Selection */}
                    <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest px-2">Asset
                        Selection</label>
                    <div className="p-5 bg-slate-900/50 border border-slate-800/50 rounded-2xl space-y-4">
                        <input
                            type="text"
                            value={symbolsInput}
                            onChange={(e) => setSymbolsInput(e.target.value)}
                            className="w-full bg-slate-950 border border-slate-800 rounded-xl py-3 px-4 text-xs font-mono text-emerald-400 focus:border-emerald-500 outline-none transition-all"
                            placeholder="Symbols (comma separated)"
                        />
                        <div
                            className="flex items-center justify-between text-[10px] text-slate-500 font-bold uppercase">
                            <span>{detectedAssets} Assets Detected</span>
                            <span className={detectedAssets >= 2 ? "text-emerald-500" : "text-amber-500"}>
                                {detectedAssets >= 2 ? 'Ready' : 'Min 2 Required'}
                            </span>
                        </div>
                    </div>

                    {/* Optimization Method */}
                    <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest px-2">Optimization
                        Method</label>
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

                    {/* Dynamic Parameters */}
                    <label
                        className="text-[10px] font-black text-slate-500 uppercase tracking-widest px-2">Parameters</label>
                    <div className="p-5 bg-slate-900/50 border border-slate-800/50 rounded-2xl space-y-4">
                        {/* Lookback Days - Always shown */}
                        {selectedMethod?.params.includes('lookback') && (
                            <div>
                                <label className="text-[10px] text-slate-400 font-semibold block mb-2">
                                    Lookback Period (Days)
                                </label>
                                <input
                                    type="number"
                                    value={lookbackDays}
                                    onChange={(e) => setLookbackDays(Number(e.target.value))}
                                    min="30"
                                    max="1000"
                                    className="w-full bg-slate-950 border border-slate-800 rounded-xl py-2 px-3 text-xs text-slate-200 focus:border-emerald-500 outline-none transition-all"
                                />
                                <p className="text-[9px] text-slate-600 mt-1">252 days ≈ 1 year of trading</p>
                            </div>
                        )}

                        {/* Risk-Free Rate */}
                        {selectedMethod?.params.includes('riskFree') && (
                            <div>
                                <label className="text-[10px] text-slate-400 font-semibold block mb-2">
                                    Risk-Free Rate (Annual)
                                </label>
                                <input
                                    type="number"
                                    value={riskFreeRate}
                                    onChange={(e) => setRiskFreeRate(Number(e.target.value))}
                                    min="0"
                                    max="0.1"
                                    step="0.001"
                                    className="w-full bg-slate-950 border border-slate-800 rounded-xl py-2 px-3 text-xs text-slate-200 focus:border-emerald-500 outline-none transition-all"
                                />
                                <p className="text-[9px] text-slate-600 mt-1">e.g., 0.02 = 2% (T-Bill rate)</p>
                            </div>
                        )}

                        {/* Target Return */}
                        {selectedMethod?.params.includes('targetReturn') && (
                            <div>
                                <label className="text-[10px] text-slate-400 font-semibold block mb-2">
                                    Target Annual Return
                                </label>
                                <input
                                    type="number"
                                    value={targetReturn}
                                    onChange={(e) => setTargetReturn(Number(e.target.value))}
                                    min="-1"
                                    max="5"
                                    step="0.01"
                                    className="w-full bg-slate-950 border border-slate-800 rounded-xl py-2 px-3 text-xs text-slate-200 focus:border-emerald-500 outline-none transition-all"
                                />
                                <p className="text-[9px] text-slate-600 mt-1">e.g., 0.15 = 15% target return</p>
                            </div>
                        )}

                        {/* Confidence (Black-Litterman) */}
                        {selectedMethod?.params.includes('confidence') && (
                            <div>
                                <label className="text-[10px] text-slate-400 font-semibold block mb-2">
                                    View Confidence (0-1)
                                </label>
                                <input
                                    type="number"
                                    value={confidence}
                                    onChange={(e) => setConfidence(Number(e.target.value))}
                                    min="0"
                                    max="1"
                                    step="0.1"
                                    className="w-full bg-slate-950 border border-slate-800 rounded-xl py-2 px-3 text-xs text-slate-200 focus:border-emerald-500 outline-none transition-all"
                                />
                                <p className="text-[9px] text-slate-600 mt-1">Higher = more confidence in views</p>
                            </div>
                        )}

                        {/* Views (Black-Litterman) */}
                        {selectedMethod?.params.includes('views') && (
                            <div>
                                <label className="text-[10px] text-slate-400 font-semibold block mb-2">
                                    Market Views (Optional)
                                </label>
                                <textarea
                                    value={viewsInput}
                                    onChange={(e) => setViewsInput(e.target.value)}
                                    placeholder="AAPL:0.12,MSFT:0.08"
                                    rows={3}
                                    className="w-full bg-slate-950 border border-slate-800 rounded-xl py-2 px-3 text-xs text-slate-200 focus:border-emerald-500 outline-none transition-all resize-none font-mono"
                                />
                                <p className="text-[9px] text-slate-600 mt-1">Format: SYMBOL:RETURN (e.g., AAPL:0.12 for
                                    12%)</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* 3. Results Dashboard */}
                <div className="lg:col-span-3 space-y-6">
                    {/* Key Metrics Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {[
                            {
                                label: 'Expected Annual Return',
                                value: metrics.return,
                                icon: TrendingUp,
                                color: 'text-emerald-400'
                            },
                            {
                                label: 'Annual Volatility',
                                value: metrics.volatility,
                                icon: ShieldCheck,
                                color: 'text-blue-400'
                            },
                            {
                                label: 'Portfolio Sharpe Ratio',
                                value: metrics.sharpe,
                                icon: Target,
                                color: 'text-amber-400'
                            },
                        ].map((metric, i) => (
                            <div key={i}
                                 className="bg-slate-900/40 border border-slate-800/50 p-6 rounded-2xl relative overflow-hidden group">
                                <div
                                    className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                                    <metric.icon size={48}/>
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
                                    <PieChart size={14} className="text-emerald-500"/>
                                    Optimal Allocation
                                </h4>
                            </div>
                            <div className="h-[280px]">
                                {allocation.length > 0 ? (
                                    <ResponsiveContainer width="100%" height="100%">
                                        <RePie>
                                            <Pie
                                                data={allocation}
                                                innerRadius={70}
                                                outerRadius={100}
                                                paddingAngle={8}
                                                dataKey="weight"
                                                stroke="none"
                                            >
                                                {allocation.map((entry, index) => (
                                                    <Cell key={`cell-${index}`} fill={entry.color}/>
                                                ))}
                                            </Pie>
                                            <Tooltip
                                                contentStyle={{
                                                    backgroundColor: '#0f172a',
                                                    border: '1px solid #1e293b',
                                                    borderRadius: '12px'
                                                }}
                                                formatter={(val: number) => [`${(val * 100).toFixed(2)}%`, 'Allocation']}
                                            />
                                        </RePie>
                                    </ResponsiveContainer>
                                ) : (
                                    <div className="h-full flex items-center justify-center text-slate-500 text-sm">
                                        Run optimization to see allocation
                                    </div>
                                )}
                            </div>
                            <div className="grid grid-cols-3 gap-2 mt-4">
                                {allocation.map((item, i) => (
                                    <div key={i} className="flex flex-col items-center p-2 bg-white/5 rounded-xl">
                                        <span className="text-[10px] font-black text-slate-200">{item.symbol}</span>
                                        <span
                                            className="text-[10px] font-mono text-emerald-400">{(item.weight * 100).toFixed(1)}%</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Efficient Frontier */}
                        <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                            <div className="flex items-center justify-between mb-6">
                                <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest flex items-center gap-2">
                                    <BarChart3 size={14} className="text-blue-500"/>
                                    Efficient Frontier
                                </h4>
                                <Info size={14} className="text-slate-600 cursor-help"
                                      title="The efficient frontier shows optimal portfolios. Your selected portfolio is highlighted in orange."/>
                            </div>
                            <div className="h-[280px]">
                                {frontierData.length > 0 ? (
                                    <ResponsiveContainer width="100%" height="100%">
                                        <ScatterChart margin={{top: 20, right: 20, bottom: 20, left: -20}}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false}/>
                                            <XAxis type="number" dataKey="risk" name="Risk" unit="%" stroke="#64748b"
                                                   fontSize={10} tickLine={false} axisLine={false}/>
                                            <YAxis type="number" dataKey="return" name="Return" unit="%"
                                                   stroke="#64748b" fontSize={10} tickLine={false} axisLine={false}/>
                                            <Tooltip
                                                cursor={{strokeDasharray: '3 3'}}
                                                contentStyle={{
                                                    backgroundColor: '#0f172a',
                                                    border: '1px solid #1e293b',
                                                    borderRadius: '12px'
                                                }}
                                                content={({active, payload}) => {
                                                    if (active && payload && payload.length) {
                                                        const data = payload[0].payload;
                                                        return (
                                                            <div
                                                                className="bg-slate-900 border border-slate-700 rounded-xl p-3 shadow-lg">
                                                                {data.isCurrent && (
                                                                    <p className="text-[10px] font-bold text-amber-400 mb-1">{data.label}</p>
                                                                )}
                                                                <p className="text-xs text-slate-300">Return: <span
                                                                    className="font-bold text-emerald-400">{data.return.toFixed(2)}%</span>
                                                                </p>
                                                                <p className="text-xs text-slate-300">Risk: <span
                                                                    className="font-bold text-blue-400">{data.risk.toFixed(2)}%</span>
                                                                </p>
                                                            </div>
                                                        );
                                                    }
                                                    return null;
                                                }}
                                            />
                                            {/* Efficient frontier line */}
                                            <Scatter
                                                name="Efficient Frontier"
                                                data={frontierData.filter(p => !p.isCurrent)}
                                                fill="#10b981"
                                                fillOpacity={0.6}
                                                line={{stroke: '#10b981', strokeWidth: 2}}
                                            />
                                            {/* Current portfolio highlight */}
                                            <Scatter
                                                name="Your Portfolio"
                                                data={frontierData.filter(p => p.isCurrent)}
                                                fill="#f59e0b"
                                                shape="star"
                                            />
                                        </ScatterChart>
                                    </ResponsiveContainer>
                                ) : (
                                    <div className="h-full flex items-center justify-center text-slate-500 text-sm">
                                        Run optimization to see frontier
                                    </div>
                                )}
                            </div>
                            {optimizationResult && (
                                <div className="mt-4 flex items-center justify-center gap-6 text-[10px]">
                                    <div className="flex items-center gap-2">
                                        <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
                                        <span className="text-slate-400">Efficient Frontier</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <div className="w-3 h-3 bg-amber-500"
                                             style={{clipPath: 'polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%)'}}></div>
                                        <span className="text-slate-400">{optimizationResult.method}</span>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Capital Allocation Tool */}
                    <div
                        className="bg-emerald-500/5 border border-emerald-500/10 rounded-2xl p-6 flex flex-col md:flex-row items-center justify-between gap-6">
                        <div className="space-y-1">
                            <p className="text-xs font-bold text-emerald-400 uppercase tracking-wider">Capital
                                Allocation Tool</p>
                            <p className="text-sm text-slate-400">Calculate share counts for your investment amount</p>
                        </div>
                        <div
                            className="flex items-center gap-4 bg-slate-950 p-2 rounded-2xl border border-white/5 w-full md:w-auto">
                            <div className="flex items-center gap-2 px-4">
                                <DollarSign size={16} className="text-slate-500"/>
                                <input
                                    type="number"
                                    value={capitalAmount}
                                    onChange={(e) => setCapitalAmount(Number(e.target.value))}
                                    className="bg-transparent text-slate-100 font-bold outline-none w-32"
                                />
                            </div>
                            <button
                                onClick={handleGenerateTradeList}
                                disabled={!optimizationResult}
                                className="px-6 py-2 bg-emerald-600 text-white rounded-xl text-xs font-bold hover:bg-emerald-500 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                Generate Trade List
                            </button>
                        </div>
                    </div>
                    {trades && (
                        <div
                            className="bg-slate-900/80 border border-slate-800 rounded-2xl overflow-hidden animate-in slide-in-from-bottom-4 duration-500">
                            <div
                                className="p-6 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
                                <div className="flex items-center gap-3">
                                    <div className="p-2 bg-emerald-500/10 rounded-lg">
                                        <FileText className="text-emerald-400" size={18}/>
                                    </div>
                                    <div>
                                        <h4 className="text-sm font-bold text-slate-200">Execution Trade List</h4>
                                        <p className="text-[10px] text-slate-500 uppercase tracking-tight">Rounded to
                                            nearest whole share</p>
                                    </div>
                                </div>
                                <button
                                    onClick={downloadCSV}
                                    className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-xs font-bold transition-all border border-slate-700"
                                >
                                    <Download size={14}/>
                                    Export CSV
                                </button>
                            </div>

                            <div className="overflow-x-auto">
                                <table className="w-full text-left border-collapse">
                                    <thead>
                                    <tr className="bg-slate-950/50">
                                        <th className="p-4 text-[10px] font-black text-slate-500 uppercase tracking-widest">Asset</th>
                                        <th className="p-4 text-[10px] font-black text-slate-500 uppercase tracking-widest text-right">Action</th>
                                        <th className="p-4 text-[10px] font-black text-slate-500 uppercase tracking-widest text-right">Price</th>
                                        <th className="p-4 text-[10px] font-black text-slate-500 uppercase tracking-widest text-right">Target
                                            Value
                                        </th>
                                        <th className="p-4 text-[10px] font-black text-slate-500 uppercase tracking-widest text-right">Actual
                                            Cost
                                        </th>
                                    </tr>
                                    </thead>
                                    <tbody className="divide-y divide-slate-800/50">
                                    {trades.map((trade, i) => (
                                        <tr key={i} className="hover:bg-white/5 transition-colors group">
                                            <td className="p-4">
                                                <div className="flex flex-col">
                                                    <span
                                                        className="text-sm font-bold text-slate-200">{trade.symbol}</span>
                                                    <span
                                                        className="text-[10px] text-slate-500">{(trade.targetWeight * 100).toFixed(1)}% weight</span>
                                                </div>
                                            </td>
                                            <td className="p-4 text-right">
                                                <div
                                                    className="inline-flex items-center gap-2 px-3 py-1 bg-emerald-500/10 text-emerald-400 rounded-full text-xs font-mono font-bold">
                                                    BUY {trade.shares}
                                                </div>
                                            </td>
                                            <td className="p-4 text-right text-xs font-mono text-slate-400">{formatCurrency(trade.price)}</td>
                                            <td className="p-4 text-right text-xs font-mono text-slate-400">{formatCurrency(trade.targetAllocation)}</td>
                                            <td className="p-4 text-right text-sm font-bold text-emerald-400 font-mono">{formatCurrency(trade.actualCost)}</td>
                                        </tr>
                                    ))}
                                    </tbody>
                                </table>
                            </div>

                            {/* Summary Footer */}
                            <div
                                className="p-4 bg-slate-950/80 border-t border-slate-800 flex justify-between items-center">
                                <div className="flex gap-6">
                                    <div>
                                        <span className="text-[9px] text-slate-500 uppercase font-black block">Total Invested</span>
                                        <span className="text-sm font-bold text-slate-200">
                        {formatCurrency(trades.reduce((acc, t) => acc + t.actualCost, 0))}
                    </span>
                                    </div>
                                    <div>
                                        <span className="text-[9px] text-slate-500 uppercase font-black block">Residual Cash</span>
                                        <span className="text-sm font-bold text-amber-400">
                        {formatCurrency(capitalAmount - trades.reduce((acc, t) => acc + t.actualCost, 0))}
                    </span>
                                    </div>
                                </div>
                                <div className="text-[10px] text-slate-600 italic">
                                    * Prices are based on last fetched quote. Market orders may vary.
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default PortfolioOptimization;