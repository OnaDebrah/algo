'use client'
import React, { useState } from 'react';
import {
    Target, Wand2, Code2, FlaskConical, BookOpen,
    Play, Save, Download, ChevronRight, Sparkles,
    Trash2, Copy, History, Terminal, LineChart,
    Settings2, Info, AlertCircle, TrendingUp, CheckCircle2,
    Brain, Cpu, Zap, FileCode, GitBranch, Cog,
    Database, Shield, Timer, DollarSign, BarChart3,
    Maximize2, Minimize2, Eye, EyeOff, Layers,
    Puzzle, Rocket, RefreshCw, Upload, Share2,
    Variable, Filter, GitMerge, GitPullRequest,
    MousePointerClick, MousePointer
} from "lucide-react";
import {
    ResponsiveContainer, AreaChart, Area, XAxis, YAxis,
    CartesianGrid, Tooltip, Legend, Line,
    BarChart, Bar, ScatterChart, Scatter, ZAxis
} from 'recharts';

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

    // Enhanced Backtest Data
    const equityData = Array.from({ length: 50 }, (_, i) => ({
        time: `Day ${i+1}`,
        strategy: 10000 + (Math.random() * 400 * i * (0.8 + Math.random() * 0.4)),
        benchmark: 10000 + (Math.random() * 200 * i * (0.9 + Math.random() * 0.2)),
        drawdown: Math.random() * 15,
        volatility: 10 + Math.random() * 20
    }));

    const tradeData = Array.from({ length: 30 }, (_, i) => ({
        trade: i + 1,
        pnl: (Math.random() - 0.45) * 2000,
        duration: Math.floor(Math.random() * 10) + 1,
        size: Math.random() * 10000
    }));

    const handleGenerate = () => {
        setIsGenerating(true);
        setTimeout(() => {
            setActiveTab('editor');
            setIsGenerating(false);
            // Update code with generated strategy
            setCode(`# Generated Strategy: ${prompt.slice(0, 50)}...
import pandas as pd
import numpy as np

def generated_strategy(data):
    """
    AI-Generated Strategy based on: "${prompt}"
    """
    # Advanced signal generation...
    signals = pd.Series(0, index=data.index)
    
    # Your AI logic here
    # ...
    
    return signals`);
        }, 1500);
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

    const runBacktest = () => {
        setIsBacktesting(true);
        setTimeout(() => setIsBacktesting(false), 2500);
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

            {/* Main Tabs Navigation */}
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
                    {/* Tabs Navigation */}
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
                            <button className="p-2.5 bg-slate-800/50 hover:bg-slate-700/50 rounded-xl text-slate-400 hover:text-white transition-all">
                                <Save size={18} />
                            </button>
                            <button className="p-2.5 bg-slate-800/50 hover:bg-slate-700/50 rounded-xl text-slate-400 hover:text-green-400 transition-all">
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
                                                    <select className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-300">
                                                        <option>Momentum</option>
                                                        <option>Mean Reversion</option>
                                                        <option>Breakout</option>
                                                        <option>Arbitrage</option>
                                                    </select>
                                                </div>
                                                <div className="space-y-2">
                                                    <label className="text-xs font-semibold text-slate-400">Timeframe</label>
                                                    <select className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-300">
                                                        <option>1 Minute</option>
                                                        <option>5 Minutes</option>
                                                        <option>1 Hour</option>
                                                        <option>4 Hours</option>
                                                        <option>1 Day</option>
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
                                            <li className="text-xs text-purple-400/80">• Be specific about entry/exit conditions</li>
                                            <li className="text-xs text-purple-400/80">• Include risk management parameters</li>
                                            <li className="text-xs text-purple-400/80">• Mention preferred indicators</li>
                                            <li className="text-xs text-purple-400/80">• Specify timeframe and asset class</li>
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
                                        <button className="text-xs text-slate-400 hover:text-green-400">
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
                                    <div className="text-xs text-slate-500">
                                        Lines: {code.split('\n').length} • Characters: {code.length}
                                    </div>
                                    <button className="px-4 py-2 bg-gradient-to-r from-purple-600 to-indigo-600 text-white text-xs font-semibold rounded-lg hover:opacity-90 transition-all">
                                        Validate Syntax
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
                                                <select className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-300">
                                                    <option>BTC-USD (Bitcoin)</option>
                                                    <option>ETH-USD (Ethereum)</option>
                                                    <option>AAPL (Apple)</option>
                                                    <option>TSLA (Tesla)</option>
                                                </select>
                                            </div>

                                            <div className="space-y-2">
                                                <label className="text-xs font-semibold text-slate-400">Time Period</label>
                                                <select className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-300">
                                                    <option>Last 30 Days</option>
                                                    <option>Last 90 Days</option>
                                                    <option>Last 1 Year</option>
                                                    <option>Last 3 Years</option>
                                                </select>
                                            </div>

                                            <div className="space-y-2">
                                                <label className="text-xs font-semibold text-slate-400">Initial Capital</label>
                                                <div className="relative">
                                                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500">$</span>
                                                    <input
                                                        type="text"
                                                        defaultValue="10,000"
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
                                                    value: '+42.8%',
                                                    sub: 'vs +18.2% Benchmark',
                                                    icon: TrendingUp,
                                                    color: 'text-emerald-400',
                                                    bg: 'from-emerald-900/20 to-teal-900/20'
                                                },
                                                {
                                                    label: 'Sharpe Ratio',
                                                    value: '2.15',
                                                    sub: 'Excellent Risk-Adjusted',
                                                    icon: Target,
                                                    color: 'text-purple-400',
                                                    bg: 'from-purple-900/20 to-indigo-900/20'
                                                },
                                                {
                                                    label: 'Max Drawdown',
                                                    value: '-12.4%',
                                                    sub: 'During Market Volatility',
                                                    icon: AlertCircle,
                                                    color: 'text-amber-400',
                                                    bg: 'from-amber-900/20 to-orange-900/20'
                                                },
                                                {
                                                    label: 'Win Rate',
                                                    value: '68.3%',
                                                    sub: '143 Winning Trades',
                                                    icon: CheckCircle2,
                                                    color: 'text-blue-400',
                                                    bg: 'from-blue-900/20 to-cyan-900/20'
                                                },
                                                {
                                                    label: 'Profit Factor',
                                                    value: '2.84',
                                                    sub: 'High Profit Efficiency',
                                                    icon: DollarSign,
                                                    color: 'text-green-400',
                                                    bg: 'from-green-900/20 to-emerald-900/20'
                                                },
                                                {
                                                    label: 'Avg Trade Duration',
                                                    value: '2.4 Days',
                                                    sub: 'Medium-Term Strategy',
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
                                                <ResponsiveContainer width="100%" height="100%">
                                                    <AreaChart data={equityData}>
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
                                                            formatter={(value) => [`$${value.toLocaleString()}`, 'Value']}
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
                                            </div>
                                        </div>

                                        {/* Trade Analysis */}
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                                            <div className="bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-800/50 rounded-2xl p-6">
                                                <h4 className="text-sm font-semibold text-slate-200 mb-6">Trade Distribution</h4>
                                                <div className="h-[200px]">
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
                                                                data={tradeData}
                                                                fill="#8b5cf6"
                                                                shape="circle"
                                                            />
                                                        </ScatterChart>
                                                    </ResponsiveContainer>
                                                </div>
                                            </div>

                                            <div className="bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-slate-800/50 rounded-2xl p-6">
                                                <h4 className="text-sm font-semibold text-slate-200 mb-6">Performance Insights</h4>
                                                <div className="space-y-4">
                                                    <div className="flex items-center justify-between p-3 bg-slate-900/30 rounded-xl">
                                                        <span className="text-sm text-slate-400">Best Trade</span>
                                                        <span className="text-lg font-bold text-emerald-400">+$2,450</span>
                                                    </div>
                                                    <div className="flex items-center justify-between p-3 bg-slate-900/30 rounded-xl">
                                                        <span className="text-sm text-slate-400">Worst Trade</span>
                                                        <span className="text-lg font-bold text-red-400">-$890</span>
                                                    </div>
                                                    <div className="flex items-center justify-between p-3 bg-slate-900/30 rounded-xl">
                                                        <span className="text-sm text-slate-400">Avg Win / Loss</span>
                                                        <span className="text-lg font-bold text-slate-300">$420 / $210</span>
                                                    </div>
                                                    <div className="flex items-center justify-between p-3 bg-slate-900/30 rounded-xl">
                                                        <span className="text-sm text-slate-400">Longest Drawdown</span>
                                                        <span className="text-lg font-bold text-amber-400">14 days</span>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default StrategyBuilder;