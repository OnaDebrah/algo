/* eslint-disable @typescript-eslint/no-explicit-any */

'use client'
import React, {useEffect, useMemo, useState} from 'react';
import {
    AlertCircle,
    BarChart3,
    BookOpen,
    Brain,
    CheckCircle2,
    Code2,
    Cog,
    Copy,
    Cpu,
    DollarSign,
    Download,
    Eye,
    FileCode,
    Filter,
    FlaskConical,
    GitBranch,
    GitMerge,
    History,
    Info,
    LineChart,
    Maximize2,
    Minimize2,
    MousePointerClick,
    Play,
    Puzzle,
    RefreshCw,
    Save,
    Settings2,
    Share2,
    Shield,
    Sparkles,
    Target,
    Timer,
    Trash2,
    TrendingUp,
    Variable,
    Zap
} from "lucide-react";
import {
    Area,
    AreaChart,
    CartesianGrid,
    ResponsiveContainer,
    Scatter,
    ScatterChart,
    Tooltip,
    XAxis,
    YAxis,
    ZAxis
} from 'recharts';
import type {CustomStrategy as CustomStrategyType, StrategyGenerateResponse} from '@/utils/api';
import {strategy as strategyApi} from '@/utils/api';
import TickerSearch from "@/components/common/TickerSearch";

const STRATEGY_TEMPLATES = [
    {
        name: "Quantum Momentum AI",
        desc: "Neural network momentum detection",
        icon: Brain,
        color: "from-purple-600 to-pink-600",
        complexity: "Advanced",
        signals: 5
    },
    {
        name: "RSI Mean Reversion Pro",
        desc: "Multi-timeframe RSI with volume confirmation",
        icon: Cpu,
        color: "from-blue-600 to-cyan-600",
        complexity: "Intermediate",
        signals: 3
    },
    {
        name: "Volatility Breakout AI",
        desc: "Bollinger Bands + ATR dynamic breakout",
        icon: Zap,
        color: "from-orange-600 to-red-600",
        complexity: "Advanced",
        signals: 4
    },
    {
        name: "Market Maker Lite",
        desc: "Statistical arbitrage with risk controls",
        icon: Cog,
        color: "from-emerald-600 to-teal-600",
        complexity: "Expert",
        signals: 6
    }
];

const STRATEGY_COMPONENTS = [
    { name: "RSI Indicator", type: "Indicator", color: "bg-blue-500", icon: Variable },
    { name: "MACD Cross", type: "Signal", color: "bg-purple-500", icon: GitMerge },
    { name: "Stop Loss", type: "Risk", color: "bg-red-500", icon: Shield },
    { name: "Take Profit", type: "Risk", color: "bg-green-500", icon: Target },
    { name: "Volume Filter", type: "Filter", color: "bg-amber-500", icon: Filter },
    { name: "Trend Filter", type: "Filter", color: "bg-indigo-500", icon: TrendingUp },
];

const StrategyBuilder = () => {
    const [activeTab, setActiveTab] = useState('ai');
    const [prompt, setPrompt] = useState("");
    const [code, setCode] = useState(`import pandas as pd
import numpy as np
from typing import Dict, List

class QuantumMomentumStrategy:
    """
    Advanced neural momentum strategy with dynamic risk management
    """

    def __init__(self, params: Dict):
        self.lookback_period = params.get('lookback', 20)
        self.volatility_threshold = params.get('volatility_thresh', 0.02)
        self.risk_per_trade = params.get('risk_per_trade', 0.02)

    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical features for ML model
        """
        # Price momentum features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))

        # Volatility features
        data['volatility'] = data['returns'].rolling(self.lookback_period).std()
        data['atr'] = self.calculate_atr(data)

        # Volume features
        data['volume_sma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']

        return data.dropna()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals using multi-factor analysis
        """
        signals = pd.Series(0, index=data.index)

        # Momentum condition
        momentum_up = data['close'] > data['close'].rolling(20).mean()

        # Volume confirmation
        volume_confirmed = data['volume_ratio'] > 1.2

        # Low volatility environment
        low_vol = data['volatility'] < self.volatility_threshold

        # Generate buy signals
        buy_signals = momentum_up & volume_confirmed & low_vol
        signals[buy_signals] = 1

        # Generate sell signals
        momentum_down = data['close'] < data['close'].rolling(20).mean()
        signals[momentum_down] = -1

        return signals

    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Average True Range for volatility
        """
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean()

        return atr

    def calculate_position_size(self, capital: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk management
        """
        risk_amount = capital * self.risk_per_trade
        position_size = risk_amount / stop_loss

        return position_size`);

    const [isGenerating, setIsGenerating] = useState(false);
    const [isBacktesting, setIsBacktesting] = useState(false);
    const [strategyName, setStrategyName] = useState("Quantum Momentum AI v2.0");
    const [draggedComponent, setDraggedComponent] = useState<string | null>(null);
    const [canvasComponents, setCanvasComponents] = useState<string[]>([
        "RSI Indicator", "MACD Cross", "Stop Loss", "Take Profit"
    ]);
    const [isMaximized, setIsMaximized] = useState(false);

    // API-driven state
    const [generationResult, setGenerationResult] = useState<StrategyGenerateResponse | null>(null);
    const [validationResult, setValidationResult] = useState<{is_valid: boolean; errors: string[]; warnings: string[]} | null>(null);
    const [backtestResult, setBacktestResult] = useState<any>(null);
    const [savedStrategies, setSavedStrategies] = useState<CustomStrategyType[]>([]);
    const [currentStrategyId, setCurrentStrategyId] = useState<number | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [strategyType, setStrategyType] = useState('technical');
    const [timeframe, setTimeframe] = useState('1d');
    const [backtestSymbol, setBacktestSymbol] = useState('AAPL');
    const [backtestPeriod, setBacktestPeriod] = useState('1y');
    const [backtestCapital, setBacktestCapital] = useState(10000);
    const [isSaving, setIsSaving] = useState(false);
    const [isValidating, setIsValidating] = useState(false);

    // Derived chart data from backtest results
    const equityChartData = useMemo(() => {
        if (!backtestResult?.equity_curve?.length) return [];
        return backtestResult.equity_curve.map((pt: any, i: number) => ({
            time: pt.date || `Day ${i+1}`,
            strategy: pt.equity ?? pt.value ?? 0,
            benchmark: backtestResult.benchmark?.equity_curve?.[i]?.equity ?? null,
        }));
    }, [backtestResult]);

    const tradeChartData = useMemo(() => {
        if (!backtestResult?.trades?.length) return [];
        return backtestResult.trades.map((t: any, i: number) => ({
            trade: i + 1,
            pnl: t.pnl ?? 0,
            duration: t.duration_days ?? 1,
            size: Math.abs(t.pnl ?? 0),
        }));
    }, [backtestResult]);

    // Load library when switching to library tab
    useEffect(() => {
        if (activeTab === 'library') loadLibrary();
    }, [activeTab]);

    // --- API handlers ---

    const handleGenerate = async () => {
        setIsGenerating(true);
        setError(null);
        try {
            const result = await strategyApi.generate({
                prompt,
                style: strategyType,
                timeframe,
            });
            setCode(result.code);
            setGenerationResult(result);
            setActiveTab('editor');
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to generate strategy');
        } finally {
            setIsGenerating(false);
        }
    };

    const handleDragStart = (component: string) => {
        setDraggedComponent(component);
    };

    const handleCanvasDrop = () => {
        if (draggedComponent && !canvasComponents.includes(draggedComponent)) {
            setCanvasComponents([...canvasComponents, draggedComponent]);
        }
        setDraggedComponent(null);
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
    };

    const removeComponent = (component: string) => {
        setCanvasComponents(canvasComponents.filter(c => c !== component));
    };

    const runBacktest = async () => {
        setIsBacktesting(true);
        setError(null);
        setBacktestResult(null);
        try {
            const result = await strategyApi.backtestCustom({
                code,
                symbol: backtestSymbol,
                period: backtestPeriod,
                initial_capital: backtestCapital,
                strategy_name: strategyName,
                custom_strategy_id: currentStrategyId || undefined,
            });
            setBacktestResult(result);
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Backtest failed');
        } finally {
            setIsBacktesting(false);
        }
    };

    const handleSave = async () => {
        setIsSaving(true);
        setError(null);
        try {
            if (currentStrategyId) {
                await strategyApi.updateCustom(currentStrategyId, {
                    name: strategyName,
                    code,
                    strategy_type: strategyType,
                });
            } else {
                const saved = await strategyApi.createCustom({
                    name: strategyName,
                    code,
                    strategy_type: strategyType,
                    ai_generated: generationResult !== null,
                    ai_explanation: generationResult?.explanation,
                });
                setCurrentStrategyId(saved.id);
            }
            await loadLibrary();
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to save');
        } finally {
            setIsSaving(false);
        }
    };

    const handleValidate = async () => {
        setIsValidating(true);
        setValidationResult(null);
        try {
            const result = await strategyApi.validate({ code });
            setValidationResult(result);
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Validation failed');
        } finally {
            setIsValidating(false);
        }
    };

    const loadLibrary = async () => {
        try {
            const strategies = await strategyApi.listCustom();
            setSavedStrategies(strategies);
        } catch (err) {
            console.error('Failed to load library:', err);
        }
    };

    const loadStrategy = (s: CustomStrategyType) => {
        setCode(s.code);
        setStrategyName(s.name);
        setCurrentStrategyId(s.id);
        if (s.ai_explanation) {
            setGenerationResult({ code: s.code, explanation: s.ai_explanation, example_usage: '', provider: 'saved' });
        }
        setActiveTab('editor');
    };

    const deleteStrategy = async (id: number) => {
        try {
            await strategyApi.deleteCustom(id);
            await loadLibrary();
            if (currentStrategyId === id) setCurrentStrategyId(null);
        } catch (err: any) {
            setError(err?.response?.data?.detail || err?.message || 'Failed to delete');
        }
    };

    const handleCopyCode = () => {
        navigator.clipboard.writeText(code);
    };

    return (
        <div className="space-y-8 animate-in fade-in duration-700">
            {/* Header with Strategy Name */}
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-gradient-to-br from-purple-600/20 to-indigo-600/20 rounded-xl border border-purple-500/30">
                        <Target className="text-purple-400" size={24} />
                    </div>
                    <div>
                        <input
                            type="text"
                            value={strategyName}
                            onChange={(e) => setStrategyName(e.target.value)}
                            className="text-2xl font-bold bg-transparent border-none outline-none text-slate-100 placeholder-slate-400"
                            placeholder="Strategy Name"
                        />
                        <p className="text-sm text-slate-400">Visual Strategy Designer & AI Code Generator</p>
                    </div>
                </div>
                <div className="flex items-center gap-3">
                    <button className="flex items-center gap-2 px-4 py-2 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700/50 rounded-xl text-sm font-medium text-slate-300 transition-all">
                        <Share2 size={16} />
                        Share
                    </button>
                    <button className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white rounded-xl text-sm font-medium transition-all shadow-lg shadow-purple-600/20">
                        <Download size={16} />
                        Export
                    </button>
                </div>
            </div>

            {/* Main Tabs.tsx Navigation */}
            <div className="flex flex-col lg:flex-row gap-6">
                {/* Left Sidebar - Components Library */}
                <div className="lg:w-64 space-y-6">
                    <div className="bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-800/50 rounded-2xl p-6">
                        <h4 className="text-xs font-black text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                            <Puzzle size={14} />
                            Strategy Components
                        </h4>
                        <div className="space-y-2">
                            {STRATEGY_COMPONENTS.map((component, idx) => {
                                const Icon = component.icon;
                                return (
                                    <div
                                        key={idx}
                                        draggable
                                        onDragStart={() => handleDragStart(component.name)}
                                        className="flex items-center gap-3 p-3 bg-slate-900/50 border border-slate-800/50 rounded-xl cursor-move hover:border-purple-500/30 transition-all group"
                                    >
                                        <div className={`w-2 h-2 rounded-full ${component.color}`} />
                                        <Icon size={14} className="text-slate-400" />
                                        <div className="flex-1">
                                            <p className="text-xs font-medium text-slate-300">{component.name}</p>
                                            <p className="text-[10px] text-slate-500">{component.type}</p>
                                        </div>
                                        <MousePointerClick size={12} className="text-slate-600 opacity-0 group-hover:opacity-100 transition-opacity" />
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* Quick Stats */}
                    <div className="bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-800/50 rounded-2xl p-6">
                        <h4 className="text-xs font-black text-slate-500 uppercase tracking-widest mb-4">Stats</h4>
                        <div className="space-y-3">
                            <div className="flex items-center justify-between">
                                <span className="text-xs text-slate-400">Components</span>
                                <span className="text-sm font-semibold text-slate-100">{canvasComponents.length}</span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-xs text-slate-400">Complexity</span>
                                <span className="text-sm font-semibold text-amber-400">Advanced</span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-xs text-slate-400">Lines of Code</span>
                                <span className="text-sm font-semibold text-slate-100">{code.split('\n').length}</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Main Content Area */}
                <div className="flex-1">
                    {/* Tabs.tsx Navigation */}
                    <div className="flex flex-col md:flex-row items-center justify-between border-b border-slate-800 pb-2 gap-4 mb-6">
                        <div className="flex bg-slate-900/50 p-1 rounded-xl border border-slate-800/50 w-full overflow-x-auto">
                            {[
                                { id: 'ai', label: 'AI Generator', icon: Brain },
                                { id: 'editor', label: 'Code Editor', icon: Code2 },
                                { id: 'canvas', label: 'Visual Canvas', icon: GitBranch },
                                { id: 'backtest', label: 'Backtest', icon: FlaskConical },
                                { id: 'library', label: 'Library', icon: BookOpen },
                            ].map((tab) => {
                                const Icon = tab.icon;
                                return (
                                    <button
                                        key={tab.id}
                                        onClick={() => setActiveTab(tab.id)}
                                        className={`flex items-center gap-2 px-6 py-2.5 rounded-lg text-xs font-bold uppercase tracking-widest transition-all whitespace-nowrap ${
                                            activeTab === tab.id
                                                ? 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-lg shadow-purple-600/20'
                                                : 'text-slate-500 hover:text-slate-300'
                                        }`}
                                    >
                                        <Icon size={16} /> {tab.label}
                                    </button>
                                );
                            })}
                        </div>

                        <div className="flex items-center gap-2">
                            <button onClick={handleSave} disabled={isSaving} className="p-2.5 bg-slate-800/50 hover:bg-slate-700/50 rounded-xl text-slate-400 hover:text-white transition-all">
                                {isSaving ? <RefreshCw className="animate-spin" size={18} /> : <Save size={18} />}
                            </button>
                            <button onClick={handleCopyCode} className="p-2.5 bg-slate-800/50 hover:bg-slate-700/50 rounded-xl text-slate-400 hover:text-green-400 transition-all">
                                <Copy size={18} />
                            </button>
                            <button className="p-2.5 bg-slate-800/50 hover:bg-slate-700/50 rounded-xl text-slate-400 hover:text-red-400 transition-all">
                                <Trash2 size={18} />
                            </button>
                            <button
                                onClick={() => setIsMaximized(!isMaximized)}
                                className="p-2.5 bg-slate-800/50 hover:bg-slate-700/50 rounded-xl text-slate-400 hover:text-white transition-all"
                            >
                                {isMaximized ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
                            </button>
                        </div>
                    </div>

                    {/* Tab Content */}
                    <div className={`${isMaximized ? 'fixed inset-4 z-50 bg-slate-950 rounded-2xl p-6 overflow-auto' : ''}`}>

                        {/* Error Banner */}
                        {error && (
                            <div className="mb-4 p-4 bg-red-500/10 border border-red-500/30 rounded-xl flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <AlertCircle className="text-red-400" size={16} />
                                    <span className="text-sm text-red-400">{error}</span>
                                </div>
                                <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300 text-xs">&#x2715;</button>
                            </div>
                        )}

                        {/* AI Generator Tab */}
                        {activeTab === 'ai' && (
                            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 animate-in slide-in-from-left-4">
                                <div className="lg:col-span-2 space-y-6">
                                    <div className="bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-800/50 rounded-2xl p-8">
                                        <div className="flex items-center gap-3 mb-6">
                                            <div className="p-2 bg-gradient-to-br from-purple-600/20 to-pink-600/20 rounded-xl border border-purple-500/30">
                                                <Brain className="text-purple-400" size={24} />
                                            </div>
                                            <div>
                                                <h3 className="text-xl font-bold text-slate-100">Neural Strategy Generator</h3>
                                                <p className="text-sm text-slate-400">Describe your strategy in plain English</p>
                                            </div>
                                        </div>

                                        <div className="space-y-4">
                                            <textarea
                                                value={prompt}
                                                onChange={(e) => setPrompt(e.target.value)}
                                                placeholder="Example: Create a momentum strategy that buys when price crosses above 50-day SMA with RSI confirmation below 70, and sells when price drops below 20-day EMA with stop loss at 2% ATR..."
                                                className="w-full h-64 bg-slate-950/50 border border-slate-800 rounded-2xl p-5 text-sm text-slate-300 focus:border-purple-500 outline-none transition-all resize-none leading-relaxed font-medium"
                                            />

                                            <div className="grid grid-cols-2 gap-4">
                                                <div className="space-y-2">
                                                    <label className="text-xs font-semibold text-slate-400">Strategy Type</label>
                                                    <select value={strategyType} onChange={(e) => setStrategyType(e.target.value)} className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-300">
                                                        <option value="momentum">Momentum</option>
                                                        <option value="mean_reversion">Mean Reversion</option>
                                                        <option value="breakout">Breakout</option>
                                                        <option value="technical">Technical</option>
                                                        <option value="hybrid">Hybrid</option>
                                                    </select>
                                                </div>
                                                <div className="space-y-2">
                                                    <label className="text-xs font-semibold text-slate-400">Timeframe</label>
                                                    <select value={timeframe} onChange={(e) => setTimeframe(e.target.value)} className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-300">
                                                        <option value="1m">1 Minute</option>
                                                        <option value="5m">5 Minutes</option>
                                                        <option value="1h">1 Hour</option>
                                                        <option value="4h">4 Hours</option>
                                                        <option value="1d">1 Day</option>
                                                    </select>
                                                </div>
                                            </div>
                                        </div>

                                        <button
                                            onClick={handleGenerate}
                                            disabled={isGenerating || !prompt.trim()}
                                            className="w-full mt-6 py-4 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-xl font-bold text-sm uppercase tracking-widest flex items-center justify-center gap-3 transition-all shadow-xl shadow-purple-600/20"
                                        >
                                            {isGenerating ? (
                                                <>
                                                    <Brain className="animate-spin" size={20} />
                                                    Generating AI Code...
                                                </>
                                            ) : (
                                                <>
                                                    <Sparkles size={20} />
                                                    Generate Strategy
                                                </>
                                            )}
                                        </button>

                                        {generationResult && (
                                            <div className="mt-4 p-4 bg-slate-900/50 border border-slate-800/50 rounded-xl">
                                                <div className="flex items-center gap-2 mb-2">
                                                    <CheckCircle2 className="text-emerald-400" size={14} />
                                                    <span className="text-xs font-semibold text-emerald-400">Generated via {generationResult.provider}</span>
                                                </div>
                                                <p className="text-xs text-slate-400 leading-relaxed">{generationResult.explanation.slice(0, 300)}...</p>
                                            </div>
                                        )}
                                    </div>
                                </div>

                                <div className="space-y-6">
                                    <div className="bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-800/50 rounded-2xl p-6">
                                        <h4 className="text-xs font-black text-slate-500 uppercase tracking-widest mb-4">AI Templates</h4>
                                        <div className="space-y-3">
                                            {STRATEGY_TEMPLATES.map((t, i) => {
                                                const Icon = t.icon;
                                                return (
                                                    <button
                                                        key={i}
                                                        onClick={() => setPrompt(`Create a ${t.name.toLowerCase()} strategy`)}
                                                        className="w-full text-left p-4 bg-slate-950/50 border border-slate-800/50 rounded-xl hover:border-purple-500/50 group transition-all hover:scale-[1.02]"
                                                    >
                                                        <div className="flex items-center gap-3 mb-2">
                                                            <div className={`p-2 bg-gradient-to-br ${t.color} rounded-lg`}>
                                                                <Icon className="text-white" size={16} />
                                                            </div>
                                                            <div>
                                                                <p className="text-sm font-bold text-slate-300 group-hover:text-purple-400">{t.name}</p>
                                                                <p className="text-[10px] text-slate-600">{t.complexity}</p>
                                                            </div>
                                                        </div>
                                                        <p className="text-xs text-slate-500">{t.desc}</p>
                                                        <div className="flex items-center gap-2 mt-3">
                                                            <div className="flex items-center gap-1 text-[10px] text-slate-600">
                                                                <Zap size={10} />
                                                                {t.signals} Signals
                                                            </div>
                                                        </div>
                                                    </button>
                                                );
                                            })}
                                        </div>
                                    </div>

                                    <div className="bg-gradient-to-br from-purple-900/20 to-indigo-900/20 border border-purple-800/30 rounded-2xl p-6">
                                        <h4 className="text-xs font-black text-purple-500 uppercase tracking-widest mb-3 flex items-center gap-2">
                                            <Info size={14} />
                                            AI Tips
                                        </h4>
                                        <ul className="space-y-2">
                                            <li className="text-xs text-purple-400/80">&#x2022; Be specific about entry/exit conditions</li>
                                            <li className="text-xs text-purple-400/80">&#x2022; Include risk management parameters</li>
                                            <li className="text-xs text-purple-400/80">&#x2022; Mention preferred indicators</li>
                                            <li className="text-xs text-purple-400/80">&#x2022; Specify timeframe and asset class</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Code Editor Tab */}
                        {activeTab === 'editor' && (
                            <div className="bg-gradient-to-br from-slate-950/50 to-slate-900/30 border border-slate-800/50 rounded-2xl overflow-hidden animate-in slide-in-from-right-4">
                                <div className="bg-slate-900/80 px-6 py-3 flex items-center justify-between border-b border-slate-800">
                                    <div className="flex items-center gap-4">
                                        <div className="flex items-center gap-2">
                                            <FileCode className="text-purple-400" size={18} />
                                            <span className="text-sm font-mono text-slate-300">quantum_momentum.py</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <span className="px-2 py-1 bg-emerald-500/10 text-emerald-400 text-xs rounded-lg">AI Generated</span>
                                            <span className="px-2 py-1 bg-blue-500/10 text-blue-400 text-xs rounded-lg">Python 3.10</span>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <button className="text-xs text-slate-400 hover:text-slate-200">
                                            <Eye size={14} />
                                        </button>
                                        <button onClick={handleCopyCode} className="text-xs text-slate-400 hover:text-green-400">
                                            <Copy size={14} />
                                        </button>
                                        <button className="text-xs text-slate-400 hover:text-red-400">
                                            <Trash2 size={14} />
                                        </button>
                                    </div>
                                </div>

                                <div className="flex">
                                    <div className="w-12 bg-slate-900/30 border-r border-slate-800/50 py-4 flex flex-col items-center gap-1 font-mono text-xs text-slate-700 select-none">
                                        {Array.from({length: 25}, (_, i) => <span key={i}>{i+1}</span>)}
                                    </div>
                                    <textarea
                                        value={code}
                                        onChange={(e) => setCode(e.target.value)}
                                        className="flex-1 bg-transparent p-6 font-mono text-sm text-slate-300 outline-none h-[500px] resize-none leading-relaxed scrollbar-thin"
                                        spellCheck={false}
                                    />
                                </div>

                                <div className="bg-slate-900/50 px-6 py-3 border-t border-slate-800 flex items-center justify-between">
                                    <div className="flex items-center text-xs text-slate-500">
                                        <span>Lines: {code.split('\n').length} &bull; Characters: {code.length}</span>
                                        {validationResult && (
                                            <span className={`ml-3 ${validationResult.is_valid ? 'text-emerald-400' : 'text-red-400'}`}>
                                                {validationResult.is_valid ? '\u2713 Valid code' : `\u2717 ${validationResult.errors.join(', ')}`}
                                                {validationResult.warnings.length > 0 && ` (${validationResult.warnings.length} warnings)`}
                                            </span>
                                        )}
                                    </div>
                                    <button
                                        onClick={handleValidate}
                                        disabled={isValidating}
                                        className="px-4 py-2 bg-gradient-to-r from-purple-600 to-indigo-600 text-white text-xs font-semibold rounded-lg hover:opacity-90 transition-all disabled:opacity-50"
                                    >
                                        {isValidating ? 'Validating...' : 'Validate Syntax'}
                                    </button>
                                </div>
                            </div>
                        )}

                        {/* Visual Canvas Tab */}
                        {activeTab === 'canvas' && (
                            <div className="space-y-6 animate-in slide-in-from-bottom-4">
                                <div className="bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-800/50 rounded-2xl p-6">
                                    <div className="flex items-center justify-between mb-6">
                                        <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                                            <GitBranch className="text-purple-400" size={20} />
                                            Visual Strategy Canvas
                                        </h3>
                                        <div className="flex items-center gap-2">
                                            <button className="px-4 py-2 bg-slate-800/50 text-slate-300 text-xs rounded-lg hover:bg-slate-700/50">
                                                Clear All
                                            </button>
                                            <button className="px-4 py-2 bg-gradient-to-r from-purple-600 to-indigo-600 text-white text-xs rounded-lg">
                                                Auto-Layout
                                            </button>
                                        </div>
                                    </div>

                                    {/* Canvas Area */}
                                    <div
                                        className="min-h-[400px] bg-slate-950/30 border-2 border-dashed border-slate-800/50 rounded-xl p-6"
                                        onDrop={handleCanvasDrop}
                                        onDragOver={handleDragOver}
                                    >
                                        {canvasComponents.length === 0 ? (
                                            <div className="flex flex-col items-center justify-center h-full text-center">
                                                <GitBranch className="text-slate-700 mb-3" size={48} />
                                                <p className="text-slate-600 font-medium">Drag & Drop Components Here</p>
                                                <p className="text-sm text-slate-700 mt-1">Build your strategy visually</p>
                                            </div>
                                        ) : (
                                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                                {canvasComponents.map((component, idx) => {
                                                    const comp = STRATEGY_COMPONENTS.find(c => c.name === component);
                                                    const Icon = comp?.icon;
                                                    return (
                                                        <div
                                                            key={idx}
                                                            className="p-4 bg-slate-900/50 border border-slate-800 rounded-xl flex items-center justify-between group"
                                                        >
                                                            <div className="flex items-center gap-3">
                                                                <div className={`w-3 h-3 rounded-full ${comp?.color}`} />
                                                                <div>
                                                                    <p className="text-sm font-medium text-slate-300">{component}</p>
                                                                    <p className="text-xs text-slate-500">{comp?.type}</p>
                                                                </div>
                                                            </div>
                                                            <button
                                                                onClick={() => removeComponent(component)}
                                                                className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-red-500/20 rounded"
                                                            >
                                                                <Trash2 size={14} className="text-red-400" />
                                                            </button>
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        )}

                                        {/* Connection Lines */}
                                        <div className="mt-6">
                                            <div className="h-px bg-gradient-to-r from-transparent via-purple-500/30 to-transparent" />
                                            <div className="flex justify-center mt-2">
                                                <div className="px-3 py-1 bg-purple-500/10 text-purple-400 text-xs rounded-lg">
                                                    {canvasComponents.length} Components Connected
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Backtest Tab */}
                        {activeTab === 'backtest' && (
                            <div className="space-y-6 animate-in slide-in-from-bottom-4">
                                <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                                    {/* Backtest Controls */}
                                    <div className="bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-800/50 rounded-2xl p-6 space-y-6">
                                        <h4 className="text-xs font-black text-slate-500 uppercase tracking-widest flex items-center gap-2">
                                            <Settings2 size={14} /> Test Parameters
                                        </h4>

                                        <div className="space-y-4">
                                            <div className="space-y-2">
                                                <label className="text-xs font-semibold text-slate-400">Asset</label>
                                                <TickerSearch
                                                    value={backtestSymbol}
                                                    onChange={setBacktestSymbol}
                                                    placeholder="Search asset..."
                                                    size="sm"
                                                />
                                            </div>

                                            <div className="space-y-2">
                                                <label className="text-xs font-semibold text-slate-400">Time Period</label>
                                                <select value={backtestPeriod} onChange={(e) => setBacktestPeriod(e.target.value)} className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-300">
                                                    <option value="1mo">Last 30 Days</option>
                                                    <option value="3mo">Last 90 Days</option>
                                                    <option value="1y">Last 1 Year</option>
                                                    <option value="2y">Last 2 Years</option>
                                                    <option value="5y">Last 5 Years</option>
                                                </select>
                                            </div>

                                            <div className="space-y-2">
                                                <label className="text-xs font-semibold text-slate-400">Initial Capital</label>
                                                <div className="relative">
                                                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500">$</span>
                                                    <input
                                                        type="number"
                                                        value={backtestCapital}
                                                        onChange={(e) => setBacktestCapital(Number(e.target.value) || 10000)}
                                                        className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-8 pr-4 py-3 text-sm text-slate-300"
                                                    />
                                                </div>
                                            </div>

                                            <button
                                                onClick={runBacktest}
                                                disabled={isBacktesting}
                                                className="w-full py-3.5 bg-gradient-to-r from-purple-600 to-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-xl font-bold text-sm uppercase tracking-widest hover:from-purple-500 hover:to-indigo-500 transition-all flex items-center justify-center gap-3 shadow-lg shadow-purple-600/20"
                                            >
                                                {isBacktesting ? (
                                                    <>
                                                        <History className="animate-spin" size={18} />
                                                        Running Backtest...
                                                    </>
                                                ) : (
                                                    <>
                                                        <Play size={18} />
                                                        Run Full Backtest
                                                    </>
                                                )}
                                            </button>
                                        </div>
                                    </div>

                                    {/* Performance Metrics */}
                                    <div className="lg:col-span-3">
                                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                                            {[
                                                {
                                                    label: 'Total Return',
                                                    value: backtestResult ? `${(backtestResult.result.total_return_pct ?? 0).toFixed(1)}%` : '--',
                                                    sub: backtestResult ? `Final: $${(backtestResult.result.final_equity ?? 0).toLocaleString()}` : 'Run backtest',
                                                    icon: TrendingUp,
                                                    color: backtestResult && backtestResult.result.total_return_pct > 0 ? 'text-emerald-400' : 'text-red-400',
                                                    bg: 'from-emerald-900/20 to-teal-900/20'
                                                },
                                                {
                                                    label: 'Sharpe Ratio',
                                                    value: backtestResult ? (backtestResult.result.sharpe_ratio ?? 0).toFixed(2) : '--',
                                                    sub: backtestResult ? (backtestResult.result.sharpe_ratio > 1 ? 'Good Risk-Adjusted' : 'Below Average') : 'Run backtest',
                                                    icon: Target,
                                                    color: 'text-purple-400',
                                                    bg: 'from-purple-900/20 to-indigo-900/20'
                                                },
                                                {
                                                    label: 'Max Drawdown',
                                                    value: backtestResult ? `${(backtestResult.result.max_drawdown ?? 0).toFixed(1)}%` : '--',
                                                    sub: 'Peak-to-trough decline',
                                                    icon: AlertCircle,
                                                    color: 'text-amber-400',
                                                    bg: 'from-amber-900/20 to-orange-900/20'
                                                },
                                                {
                                                    label: 'Win Rate',
                                                    value: backtestResult ? `${(backtestResult.result.win_rate ?? 0).toFixed(1)}%` : '--',
                                                    sub: backtestResult ? `${backtestResult.result.total_trades ?? 0} trades` : 'Run backtest',
                                                    icon: CheckCircle2,
                                                    color: 'text-blue-400',
                                                    bg: 'from-blue-900/20 to-cyan-900/20'
                                                },
                                                {
                                                    label: 'Profit Factor',
                                                    value: backtestResult ? (backtestResult.result.profit_factor ?? 0).toFixed(2) : '--',
                                                    sub: backtestResult ? 'Gross profits / losses' : 'Run backtest',
                                                    icon: DollarSign,
                                                    color: 'text-green-400',
                                                    bg: 'from-green-900/20 to-emerald-900/20'
                                                },
                                                {
                                                    label: 'Total Trades',
                                                    value: backtestResult ? `${backtestResult.result.total_trades ?? 0}` : '--',
                                                    sub: backtestResult ? `Win: ${backtestResult.result.winning_trades ?? 0} / Loss: ${backtestResult.result.losing_trades ?? 0}` : 'Run backtest',
                                                    icon: Timer,
                                                    color: 'text-slate-400',
                                                    bg: 'from-slate-900/20 to-slate-800/20'
                                                },
                                            ].map((m, i) => {
                                                const Icon = m.icon;
                                                return (
                                                    <div key={i} className={`bg-gradient-to-br ${m.bg} border border-slate-800/50 p-5 rounded-2xl`}>
                                                        <div className="flex items-center justify-between mb-2">
                                                            <p className="text-xs font-semibold text-slate-400 uppercase">{m.label}</p>
                                                            <Icon size={18} className="text-slate-600" />
                                                        </div>
                                                        <span className={`text-2xl font-bold ${m.color}`}>{m.value}</span>
                                                        <p className="text-[11px] text-slate-600 font-medium mt-1">{m.sub}</p>
                                                    </div>
                                                );
                                            })}
                                        </div>

                                        {/* Equity Curve Chart */}
                                        <div className="bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-800/50 rounded-2xl p-6">
                                            <div className="flex items-center justify-between mb-6">
                                                <h4 className="text-sm font-semibold text-slate-200">Equity Growth Curve</h4>
                                                <div className="flex items-center gap-2">
                                                    <div className="flex items-center gap-1 text-xs text-slate-400">
                                                        <div className="w-2 h-2 rounded-full bg-purple-500" />
                                                        My Strategy
                                                    </div>
                                                    <div className="flex items-center gap-1 text-xs text-slate-400">
                                                        <div className="w-2 h-2 rounded-full bg-slate-600" />
                                                        Buy & Hold
                                                    </div>
                                                </div>
                                            </div>
                                            <div className="h-[300px]">
                                                {equityChartData.length > 0 ? (
                                                    <ResponsiveContainer width="100%" height="100%">
                                                        <AreaChart data={equityChartData}>
                                                            <defs>
                                                                <linearGradient id="colorStrategy" x1="0" y1="0" x2="0" y2="1">
                                                                    <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                                                                    <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                                                                </linearGradient>
                                                            </defs>
                                                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                                                            <XAxis
                                                                dataKey="time"
                                                                stroke="#64748b"
                                                                fontSize={10}
                                                                axisLine={false}
                                                                tickLine={false}
                                                            />
                                                            <YAxis
                                                                stroke="#64748b"
                                                                fontSize={10}
                                                                axisLine={false}
                                                                tickLine={false}
                                                                tickFormatter={(value) => `$${value.toLocaleString()}`}
                                                            />
                                                            <Tooltip
                                                                contentStyle={{
                                                                    backgroundColor: '#0f172a',
                                                                    border: '1px solid #1e293b',
                                                                    borderRadius: '8px',
                                                                    color: '#e2e8f0'
                                                                }}
                                                                formatter={(value) => [`$${(value ?? 0).toLocaleString()}`, 'Value']}
                                                            />
                                                            <Area
                                                                type="monotone"
                                                                dataKey="strategy"
                                                                stroke="#8b5cf6"
                                                                fill="url(#colorStrategy)"
                                                                strokeWidth={2}
                                                                name="My Strategy"
                                                            />
                                                            <Area
                                                                type="monotone"
                                                                dataKey="benchmark"
                                                                stroke="#475569"
                                                                fill="transparent"
                                                                strokeWidth={2}
                                                                strokeDasharray="5 5"
                                                                name="Buy & Hold"
                                                            />
                                                        </AreaChart>
                                                    </ResponsiveContainer>
                                                ) : (
                                                    <div className="flex flex-col items-center justify-center h-full text-center">
                                                        <LineChart className="text-slate-700 mb-3" size={48} />
                                                        <p className="text-slate-600 font-medium">No backtest data yet</p>
                                                        <p className="text-sm text-slate-700 mt-1">Run a backtest to see the equity curve</p>
                                                    </div>
                                                )}
                                            </div>
                                        </div>

                                        {/* Trade Analysis */}
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                                            <div className="bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-800/50 rounded-2xl p-6">
                                                <h4 className="text-sm font-semibold text-slate-200 mb-6">Trade Distribution</h4>
                                                <div className="h-[200px]">
                                                    {tradeChartData.length > 0 ? (
                                                        <ResponsiveContainer width="100%" height="100%">
                                                            <ScatterChart>
                                                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                                                <XAxis
                                                                    dataKey="duration"
                                                                    name="Duration"
                                                                    stroke="#64748b"
                                                                    fontSize={10}
                                                                />
                                                                <YAxis
                                                                    dataKey="pnl"
                                                                    name="PnL"
                                                                    stroke="#64748b"
                                                                    fontSize={10}
                                                                    tickFormatter={(value) => `$${value}`}
                                                                />
                                                                <ZAxis range={[50, 1000]} />
                                                                <Tooltip
                                                                    contentStyle={{
                                                                        backgroundColor: '#0f172a',
                                                                        border: '1px solid #1e293b'
                                                                    }}
                                                                    formatter={(value, name) => [
                                                                        name === 'pnl' ? `$${value}` : `${value} days`,
                                                                        name === 'pnl' ? 'Profit' : 'Duration'
                                                                    ]}
                                                                />
                                                                <Scatter
                                                                    name="Trades"
                                                                    data={tradeChartData}
                                                                    fill="#8b5cf6"
                                                                    shape="circle"
                                                                />
                                                            </ScatterChart>
                                                        </ResponsiveContainer>
                                                    ) : (
                                                        <div className="flex flex-col items-center justify-center h-full text-center">
                                                            <BarChart3 className="text-slate-700 mb-3" size={36} />
                                                            <p className="text-sm text-slate-600">Run a backtest to see trades</p>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>

                                            <div className="bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-800/50 rounded-2xl p-6">
                                                <h4 className="text-sm font-semibold text-slate-200 mb-6">Performance Insights</h4>
                                                <div className="space-y-4">
                                                    <div className="flex items-center justify-between p-3 bg-slate-900/30 rounded-xl">
                                                        <span className="text-sm text-slate-400">Best Trade</span>
                                                        <span className="text-lg font-bold text-emerald-400">
                                                            {backtestResult ? `$${Math.max(...(backtestResult.trades || []).map((t: any) => t.pnl ?? 0)).toLocaleString()}` : '--'}
                                                        </span>
                                                    </div>
                                                    <div className="flex items-center justify-between p-3 bg-slate-900/30 rounded-xl">
                                                        <span className="text-sm text-slate-400">Worst Trade</span>
                                                        <span className="text-lg font-bold text-red-400">
                                                            {backtestResult ? `$${Math.min(...(backtestResult.trades || []).map((t: any) => t.pnl ?? 0)).toLocaleString()}` : '--'}
                                                        </span>
                                                    </div>
                                                    <div className="flex items-center justify-between p-3 bg-slate-900/30 rounded-xl">
                                                        <span className="text-sm text-slate-400">Avg Win / Loss</span>
                                                        <span className="text-lg font-bold text-slate-300">
                                                            {backtestResult ? `$${(backtestResult.result.avg_win ?? 0).toFixed(0)} / $${Math.abs(backtestResult.result.avg_loss ?? 0).toFixed(0)}` : '--'}
                                                        </span>
                                                    </div>
                                                    <div className="flex items-center justify-between p-3 bg-slate-900/30 rounded-xl">
                                                        <span className="text-sm text-slate-400">Sortino Ratio</span>
                                                        <span className="text-lg font-bold text-amber-400">
                                                            {backtestResult ? (backtestResult.result.sortino_ratio ?? 0).toFixed(2) : '--'}
                                                        </span>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Library Tab */}
                        {activeTab === 'library' && (
                            <div className="space-y-6 animate-in slide-in-from-bottom-4">
                                <div className="flex items-center justify-between">
                                    <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                                        <BookOpen className="text-purple-400" size={20} />
                                        My Strategy Library
                                    </h3>
                                    <span className="text-sm text-slate-500">{savedStrategies.length} saved</span>
                                </div>

                                {savedStrategies.length === 0 ? (
                                    <div className="bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-800/50 rounded-2xl p-12 text-center">
                                        <BookOpen className="mx-auto text-slate-700 mb-4" size={48} />
                                        <p className="text-slate-400 font-medium mb-2">No saved strategies yet</p>
                                        <p className="text-sm text-slate-600">Generate or write a strategy and save it to build your library</p>
                                    </div>
                                ) : (
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                        {savedStrategies.map((s) => (
                                            <div key={s.id} className="bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-800/50 rounded-2xl p-5 hover:border-purple-500/30 transition-all">
                                                <div className="flex items-center justify-between mb-3">
                                                    <h4 className="text-sm font-bold text-slate-200 truncate">{s.name}</h4>
                                                    <div className="flex gap-1.5 flex-shrink-0">
                                                        {s.ai_generated && (
                                                            <span className="px-2 py-0.5 bg-purple-500/10 text-purple-400 text-[10px] font-semibold rounded-lg">AI</span>
                                                        )}
                                                        {s.is_validated && (
                                                            <span className="px-2 py-0.5 bg-emerald-500/10 text-emerald-400 text-[10px] font-semibold rounded-lg">Valid</span>
                                                        )}
                                                    </div>
                                                </div>
                                                <p className="text-xs text-slate-500 mb-1">{s.strategy_type}</p>
                                                <p className="text-[10px] text-slate-600 mb-4">
                                                    Created {new Date(s.created_at).toLocaleDateString()} &middot; {s.code.split('\n').length} lines
                                                </p>
                                                <div className="flex gap-2">
                                                    <button
                                                        onClick={() => loadStrategy(s)}
                                                        className="flex-1 px-3 py-2 bg-purple-600/20 text-purple-400 text-xs font-semibold rounded-lg hover:bg-purple-600/30 transition-all"
                                                    >
                                                        Load
                                                    </button>
                                                    <button
                                                        onClick={() => deleteStrategy(s.id)}
                                                        className="px-3 py-2 bg-red-500/10 text-red-400 text-xs font-semibold rounded-lg hover:bg-red-500/20 transition-all"
                                                    >
                                                        <Trash2 size={14} />
                                                    </button>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default StrategyBuilder;
