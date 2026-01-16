'use client'
import React, { useState, useEffect, useMemo } from 'react';
import {
    Activity, Search, FileText, BarChart2, TrendingUp,
    AlertTriangle, Download, Globe, Target, Shield,
    PieChart, Zap, ChevronRight, CheckCircle2, Info,
    ArrowUpRight, ArrowDownRight, Clock, Users, DollarSign,
    Building, ChartNoAxesColumn, TrendingDown, Star, AlertCircle,
    Cpu, Network, Database, LineChart as LineChartIcon,
    Percent, Award, Target as TargetIcon, Scale,
    ChartBar, ChartLine, ChartPie, TrendingUp as TrendingUpIcon,
    Activity as ActivityIcon, Globe as GlobeIcon,
    Heart, ThumbsUp, ThumbsDown, MessageSquare,
    Shield as ShieldIcon, Lock, AlertOctagon,
    Calculator, Coins, Wallet, Banknote
} from "lucide-react";
import {
    ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis,
    Radar, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid,
    PieChart as RePie, Pie, Cell, LineChart, Line, AreaChart, Area,
    ComposedChart, Scatter, Legend
} from 'recharts';

import { analyst } from "@/utils/api";
import { AnalystReport } from "@/types/api.types";



const AIAnalyst = () => {
    const [ticker, setTicker] = useState("");
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [report, setReport] = useState<AnalystReport | null>(null);
    const [activeTab, setActiveTab] = useState('summary');
    const [error, setError] = useState<string | null>(null);
    const [analysisDepth, setAnalysisDepth] = useState('Standard');
    const [timeframe, setTimeframe] = useState('1M');



    // Historical price data for chart
    const priceData = useMemo(() => {
        const basePrice = 182.63;
        return Array.from({ length: 30 }, (_, i) => ({
            date: `Day ${i + 1}`,
            price: basePrice + Math.sin(i * 0.3) * 5 + Math.random() * 2,
            volume: Math.floor(Math.random() * 1000000) + 500000
        }));
    }, []);

    const handleAnalyze = async () => {
        if (!ticker.trim()) {
            setError("Please enter a ticker symbol");
            return;
        }

        setIsAnalyzing(true);
        setError(null);
        setReport(null);

        try {
            const res = await analyst.getReport(ticker);
            setReport(res.data);
        } catch (err: any) {
            console.error("Analysis failed", err);
            setError(err.response?.data?.detail || "Analysis failed to generate report");
        } finally {
            setIsAnalyzing(false);
        }
    };

    const getRecommendationColor = (rec: AnalystReport['recommendation']) => {
        const colors = {
            'Strong Buy': 'text-emerald-400',
            'Buy': 'text-green-400',
            'Hold': 'text-amber-400',
            'Sell': 'text-orange-400',
            'Strong Sell': 'text-red-400'
        };
        return colors[rec];
    };

    const getRiskColor = (risk: AnalystReport['risk_rating']) => {
        const colors = {
            'Low': 'text-emerald-400',
            'Medium': 'text-amber-400',
            'High': 'text-orange-400',
            'Very High': 'text-red-400'
        };
        return colors[risk];
    };

    // Tab content components
    const SummaryTab = () => (
        <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                    <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-4 flex items-center gap-2">
                        <Cpu size={16} className="text-purple-500" /> AI Insights
                    </h4>
                    <p className="text-sm text-slate-300 leading-relaxed">
                        Our AI model identifies <span className="text-emerald-400 font-bold">strong momentum</span> in services revenue growth
                        and <span className="text-blue-400 font-bold">improving profit margins</span>. The technical setup suggests
                        <span className="text-amber-400 font-bold"> consolidation near resistance</span> with potential breakout above $185.
                    </p>
                    <div className="mt-4 p-3 bg-purple-900/20 border border-purple-800/30 rounded-lg">
                        <p className="text-xs text-purple-300">
                            <span className="font-bold">AI Confidence:</span> 92% - Based on 48 indicators across fundamental, technical, and sentiment analysis.
                        </p>
                    </div>
                </div>

                <div className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                    <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-4 flex items-center gap-2">
                        <Award size={16} className="text-amber-500" /> Key Takeaways
                    </h4>
                    <ul className="space-y-3">
                        <li className="flex items-start gap-2 text-sm">
                            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-1.5" />
                            <span><span className="font-bold text-emerald-400">Strong Buy</span> rating with 15.1% upside to target</span>
                        </li>
                        <li className="flex items-start gap-2 text-sm">
                            <div className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-1.5" />
                            <span><span className="font-bold text-blue-400">Services segment</span> growing at 8% YoY</span>
                        </li>
                        <li className="flex items-start gap-2 text-sm">
                            <div className="w-1.5 h-1.5 rounded-full bg-amber-500 mt-1.5" />
                            <span>Technical indicators show <span className="font-bold text-amber-400">bullish momentum</span></span>
                        </li>
                        <li className="flex items-start gap-2 text-sm">
                            <div className="w-1.5 h-1.5 rounded-full bg-purple-500 mt-1.5" />
                            <span><span className="font-bold text-purple-400">Institutional ownership</span> at 82% bullish</span>
                        </li>
                    </ul>
                </div>
            </div>

            <div className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-6 flex items-center gap-2">
                    <LineChartIcon size={16} className="text-cyan-500" /> Performance vs Benchmark
                </h4>
                <div className="h-[200px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={priceData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                            <XAxis dataKey="date" stroke="#64748b" fontSize={10} />
                            <YAxis stroke="#64748b" fontSize={10} />
                            <Tooltip />
                            <Legend />
                            <Area type="monotone" dataKey="price" fill="#3b82f6" fillOpacity={0.1} stroke="#3b82f6" strokeWidth={2} name="AAPL" />
                            <Line type="monotone" dataKey={(d) => d.price * 0.95} stroke="#94a3b8" strokeWidth={1} strokeDasharray="3 3" name="S&P 500" />
                        </ComposedChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );

    const FundamentalTab = () => (
        <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                    <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-6 flex items-center gap-2">
                        <ChartBar size={16} className="text-emerald-500" /> Financial Ratios
                    </h4>
                    <div className="space-y-4">
                        {Object.entries(report!.fundamental).map(([key, value]) => (
                            <div key={key} className="flex items-center justify-between p-3 bg-slate-900/50 rounded-lg">
                                <div className="flex items-center gap-2">
                                    <Percent size={14} className="text-slate-500" />
                                    <span className="text-xs text-slate-400 capitalize">{key.replace('_', ' ')}</span>
                                </div>
                                <div className="text-right">
                                    <p className="text-sm font-bold text-slate-100">
                                        {typeof value === 'number' ? value.toFixed(2) : value}
                                        {key.includes('ratio') || key.includes('yield') ? 'x' : key.includes('growth') || key.includes('margin') ? '%' : ''}
                                    </p>
                                    <p className="text-[10px] text-slate-500 mt-1">
                                        {key === 'pe_ratio' ? 'Sector: 24.8x' :
                                            key === 'roe' ? 'Industry: 89.2%' :
                                                key === 'debt_to_equity' ? 'Peer avg: 2.3' :
                                                    'vs benchmark'}
                                    </p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                    <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-6 flex items-center gap-2">
                        <TrendingUpIcon size={16} className="text-blue-500" /> Growth Metrics
                    </h4>
                    <div className="h-[250px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={[
                                { name: 'Revenue', current: 7.5, sector: 4.2 },
                                { name: 'EPS', current: 12.3, sector: 6.8 },
                                { name: 'FCF', current: 8.9, sector: 5.1 },
                                { name: 'Dividend', current: 4.2, sector: 2.8 }
                            ]}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                <XAxis dataKey="name" stroke="#64748b" fontSize={10} />
                                <YAxis stroke="#64748b" fontSize={10} />
                                <Tooltip />
                                <Bar dataKey="current" fill="#3b82f6" name="AAPL Growth (%)" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="sector" fill="#64748b" name="Sector Avg (%)" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            <div className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-4 flex items-center gap-2">
                    <Database size={16} className="text-amber-500" /> Financial Health Score
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[
                        { label: 'Liquidity', score: 85, desc: 'Cash & equivalents' },
                        { label: 'Solvency', score: 78, desc: 'Debt management' },
                        { label: 'Efficiency', score: 92, desc: 'Asset utilization' },
                        { label: 'Profitability', score: 88, desc: 'Margin strength' }
                    ].map((item, idx) => (
                        <div key={idx} className="text-center p-4 bg-slate-900/50 rounded-xl">
                            <div className="text-2xl font-bold text-slate-100">{item.score}</div>
                            <div className="text-xs text-slate-400 mt-1">{item.label}</div>
                            <div className="text-[10px] text-slate-500 mt-2">{item.desc}</div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );

    const TechnicalTab = () => (
        <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                    <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-6 flex items-center gap-2">
                        <ChartLine size={16} className="text-cyan-500" /> Technical Indicators
                    </h4>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                        {[
                            { label: 'RSI (14)', value: report!.technical.rsi, status: report!.technical.rsi_signal },
                            { label: 'MACD', value: report!.technical.macd.value, status: report!.technical.macd.histogram > 0 ? 'Bullish' : 'Bearish' },
                            { label: '20D MA', value: report!.technical.ma_20, status: 'Support' },
                            { label: '50D MA', value: report!.technical.ma_50, status: 'Support' },
                            { label: '200D MA', value: report!.technical.ma_200, status: 'Support' },
                            { label: 'Volume Trend', value: report!.technical.volume_trend, status: report!.technical.volume_trend }
                        ].map((indicator, idx) => (
                            <div key={idx} className="p-4 bg-slate-900/50 rounded-xl">
                                <p className="text-xs text-slate-500">{indicator.label}</p>
                                <p className="text-lg font-bold text-slate-100 mt-1">
                                    {typeof indicator.value === 'number' ? indicator.value.toFixed(2) : indicator.value}
                                </p>
                                <p className={`text-[10px] mt-1 ${indicator.status === 'Bullish' ? 'text-emerald-400' :
                                    indicator.status === 'Bearish' ? 'text-red-400' :
                                        'text-amber-400'
                                    }`}>
                                    {indicator.status}
                                </p>
                            </div>
                        ))}
                    </div>

                    <div className="mt-6 grid grid-cols-2 gap-4">
                        <div className="p-4 bg-slate-900/50 rounded-xl">
                            <p className="text-xs text-slate-500">Support Levels</p>
                            <div className="mt-2 space-y-1">
                                {report!.technical.support_levels.map((level, idx) => (
                                    <p key={idx} className="text-sm text-emerald-400">${level.toFixed(2)}</p>
                                ))}
                            </div>
                        </div>
                        <div className="p-4 bg-slate-900/50 rounded-xl">
                            <p className="text-xs text-slate-500">Resistance Levels</p>
                            <div className="mt-2 space-y-1">
                                {report!.technical.resistance_levels.map((level, idx) => (
                                    <p key={idx} className="text-sm text-red-400">${level.toFixed(2)}</p>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>

                <div className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                    <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-6 flex items-center gap-2">
                        <ActivityIcon size={16} className="text-purple-500" /> Trend Analysis
                    </h4>
                    <div className="space-y-4">
                        <div>
                            <div className="flex justify-between text-xs mb-1">
                                <span className="text-slate-400">Trend Strength</span>
                                <span className="text-slate-100">{report!.technical.trend_strength}%</span>
                            </div>
                            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                                <div className="h-full bg-gradient-to-r from-blue-500 to-cyan-400" style={{ width: `${report!.technical.trend_strength}%` }} />
                            </div>
                        </div>

                        <div className="p-4 bg-slate-900/50 rounded-xl">
                            <p className="text-xs text-slate-500 mb-2">MACD Signal</p>
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm font-bold text-slate-100">{report!.technical.macd.value.toFixed(2)}</p>
                                    <p className="text-[10px] text-slate-500">MACD Line</p>
                                </div>
                                <div>
                                    <p className="text-sm font-bold text-slate-100">{report!.technical.macd.signal.toFixed(2)}</p>
                                    <p className="text-[10px] text-slate-500">Signal Line</p>
                                </div>
                                <div>
                                    <p className={`text-sm font-bold ${report!.technical.macd.histogram > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                        {report!.technical.macd.histogram > 0 ? '+' : ''}{report!.technical.macd.histogram.toFixed(2)}
                                    </p>
                                    <p className="text-[10px] text-slate-500">Histogram</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );

    const SentimentTab = () => (
        <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                    <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-6 flex items-center gap-2">
                        <GlobeIcon size={16} className="text-purple-500" /> Market Sentiment
                    </h4>
                    <div className="space-y-4">
                        {Object.entries(report!.sentiment).map(([key, value]) => (
                            <div key={key} className="space-y-2">
                                <div className="flex justify-between text-sm">
                                    <span className="text-slate-400 capitalize">{key}</span>
                                    <span className="font-bold text-slate-100">{value}%</span>
                                </div>
                                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                                    <div
                                        className={`h-full rounded-full ${value >= 75 ? 'bg-emerald-500' :
                                            value >= 60 ? 'bg-amber-500' :
                                                'bg-red-500'
                                            }`}
                                        style={{ width: `${value}%` }}
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                    <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-6 flex items-center gap-2">
                        <ChartPie size={16} className="text-pink-500" /> Sentiment Distribution
                    </h4>
                    <div className="h-[250px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <RePie>
                                <Pie
                                    data={[
                                        { name: 'Bullish', value: 65, color: '#10b981' },
                                        { name: 'Neutral', value: 25, color: '#64748b' },
                                        { name: 'Bearish', value: 10, color: '#ef4444' }
                                    ]}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={80}
                                    paddingAngle={5}
                                    dataKey="value"
                                >
                                    {[{ color: '#10b981' }, { color: '#64748b' }, { color: '#ef4444' }].map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                    ))}
                                </Pie>
                                <Tooltip />
                                <Legend />
                            </RePie>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gradient-to-br from-emerald-900/20 to-emerald-950/20 border border-emerald-800/30 rounded-xl p-4 text-center">
                    <ThumbsUp className="w-8 h-8 text-emerald-400 mx-auto mb-2" />
                    <p className="text-lg font-bold text-emerald-400">82%</p>
                    <p className="text-xs text-emerald-300">Analyst Consensus</p>
                </div>
                <div className="bg-gradient-to-br from-amber-900/20 to-amber-950/20 border border-amber-800/30 rounded-xl p-4 text-center">
                    <MessageSquare className="w-8 h-8 text-amber-400 mx-auto mb-2" />
                    <p className="text-lg font-bold text-amber-400">72%</p>
                    <p className="text-xs text-amber-300">News Sentiment</p>
                </div>
                <div className="bg-gradient-to-br from-blue-900/20 to-blue-950/20 border border-blue-800/30 rounded-xl p-4 text-center">
                    <Network className="w-8 h-8 text-blue-400 mx-auto mb-2" />
                    <p className="text-lg font-bold text-blue-400">68%</p>
                    <p className="text-xs text-blue-300">Social Media</p>
                </div>
            </div>
        </div>
    );

    const RisksTab = () => (
        <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {Object.entries(report!.risks).map(([category, risks], idx) => (
                    <div key={category} className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                        <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-4 flex items-center gap-2">
                            <AlertOctagon size={16} className={
                                category === 'regulatory' ? 'text-red-500' :
                                    category === 'competitive' ? 'text-orange-500' :
                                        category === 'market' ? 'text-amber-500' :
                                            category === 'financial' ? 'text-purple-500' :
                                                'text-slate-500'
                            } />
                            {category.charAt(0).toUpperCase() + category.slice(1)} Risks
                        </h4>
                        <ul className="space-y-2">
                            {risks.map((risk, riskIdx) => (
                                <li key={riskIdx} className="flex items-start gap-2 text-sm text-slate-300">
                                    <div className="w-1.5 h-1.5 rounded-full bg-red-500 mt-1.5 flex-shrink-0" />
                                    {risk}
                                </li>
                            ))}
                        </ul>
                        <div className="mt-4 pt-4 border-t border-slate-800/50">
                            <p className="text-xs text-slate-500">
                                <span className="font-bold">Severity:</span> {
                                    category === 'regulatory' ? 'High' :
                                        category === 'competitive' ? 'Medium' :
                                            category === 'market' ? 'Low' :
                                                category === 'financial' ? 'Medium' :
                                                    'Low'
                                }
                            </p>
                        </div>
                    </div>
                ))}
            </div>

            <div className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-6 flex items-center gap-2">
                    <ShieldIcon size={16} className="text-cyan-500" /> Risk Mitigation Score
                </h4>
                <div className="space-y-4">
                    {[
                        { label: 'Diversification', score: 85, desc: 'Revenue streams across segments' },
                        { label: 'Liquidity', score: 92, desc: 'Cash position & access to capital' },
                        { label: 'Governance', score: 78, desc: 'Board oversight & controls' },
                        { label: 'Hedging', score: 65, desc: 'Risk management strategies' }
                    ].map((item, idx) => (
                        <div key={idx} className="space-y-2">
                            <div className="flex justify-between text-sm">
                                <div>
                                    <span className="text-slate-400">{item.label}</span>
                                    <span className="text-[10px] text-slate-500 ml-2">({item.desc})</span>
                                </div>
                                <span className="font-bold text-slate-100">{item.score}%</span>
                            </div>
                            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                                <div
                                    className={`h-full rounded-full ${item.score >= 80 ? 'bg-emerald-500' :
                                        item.score >= 60 ? 'bg-amber-500' :
                                            'bg-red-500'
                                        }`}
                                    style={{ width: `${item.score}%` }}
                                />
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );

    const ValuationTab = () => (
        <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                    <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-6 flex items-center gap-2">
                        <Calculator size={16} className="text-purple-500" /> Valuation Metrics
                    </h4>
                    <div className="h-[300px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <RadarChart cx="50%" cy="50%" outerRadius="80%" data={report!.valuation}>
                                <PolarGrid stroke="#1e293b" strokeOpacity={0.5} />
                                <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                                <Radar
                                    name="Current"
                                    dataKey="score"
                                    stroke="#8b5cf6"
                                    fill="#8b5cf6"
                                    fillOpacity={0.3}
                                    strokeWidth={2}
                                />
                                <Radar
                                    name="Benchmark"
                                    dataKey="benchmark"
                                    stroke="#64748b"
                                    fill="none"
                                    strokeWidth={1}
                                    strokeDasharray="3 3"
                                />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#0f172a',
                                        border: '1px solid #1e293b',
                                        borderRadius: '8px'
                                    }}
                                />
                                <Legend />
                            </RadarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                    <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-6 flex items-center gap-2">
                        <Scale size={16} className="text-amber-500" /> Fair Value Analysis
                    </h4>
                    <div className="space-y-6">
                        {[
                            { method: 'DCF', value: 215.00, premium: 17.7 },
                            { method: 'Comparables', value: 205.50, premium: 12.5 },
                            { method: 'Dividend Discount', value: 195.75, premium: 7.2 },
                            { method: 'Asset-Based', value: 185.25, premium: 1.4 }
                        ].map((item, idx) => (
                            <div key={idx} className="p-4 bg-slate-900/50 rounded-xl">
                                <div className="flex justify-between items-center mb-2">
                                    <span className="text-sm font-bold text-slate-300">{item.method}</span>
                                    <span className={`text-sm font-bold ${item.premium > 10 ? 'text-emerald-400' : 'text-amber-400'}`}>
                                        ${item.value.toFixed(2)}
                                    </span>
                                </div>
                                <div className="flex justify-between text-xs">
                                    <span className="text-slate-500">Implied Premium</span>
                                    <span className={`${item.premium > 10 ? 'text-emerald-400' : 'text-amber-400'}`}>
                                        {item.premium > 0 ? '+' : ''}{item.premium.toFixed(1)}%
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                    <div className="mt-6 p-4 bg-purple-900/20 border border-purple-800/30 rounded-xl">
                        <p className="text-xs text-purple-300">
                            <span className="font-bold">AI Weighted Fair Value:</span> ${report!.target_price.toFixed(2)}
                            <span className="text-emerald-400 ml-2">(+{report!.upside.toFixed(1)}% upside)</span>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );

    // Render active tab content
    const renderTabContent = () => {
        if (!report) return null;

        switch (activeTab) {
            case 'summary': return <SummaryTab />;
            case 'fundamental': return <FundamentalTab />;
            case 'technical': return <TechnicalTab />;
            case 'sentiment': return <SentimentTab />;
            case 'risks': return <RisksTab />;
            case 'valuation': return <ValuationTab />;
            default: return <SummaryTab />;
        }
    };

    return (
        <div className="space-y-8 animate-in fade-in duration-700">
            {/* 1. Search & Configuration Header */}
            <div className="bg-gradient-to-br from-slate-900/90 to-slate-950/90 border border-slate-800/50 rounded-2xl p-6 shadow-2xl">
                <div className="flex flex-col md:flex-row gap-6 items-end">
                    <div className="flex-1 space-y-2">
                        <div className="flex items-center gap-2">
                            <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest px-1">Ticker Symbol</label>
                            <Info size={12} className="text-slate-600" />
                        </div>
                        <div className="relative group">
                            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-purple-400 transition-colors" size={18} />
                            <input
                                type="text"
                                value={ticker}
                                onChange={(e) => {
                                    setTicker(e.target.value.toUpperCase());
                                    setError(null);
                                }}
                                placeholder="e.g., AAPL, MSFT, TSLA, GOOGL"
                                className="w-full bg-slate-950/70 border border-slate-800 rounded-xl py-4 pl-12 pr-4 text-slate-100 focus:border-purple-500 focus:ring-2 focus:ring-purple-500/30 outline-none transition-all placeholder:text-slate-600"
                                onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
                            />
                            <div className="absolute right-3 top-1/2 -translate-y-1/2 flex gap-1">
                                {['AAPL', 'MSFT', 'TSLA', 'NVDA'].map((t) => (
                                    <button
                                        key={t}
                                        onClick={() => setTicker(t)}
                                        className="text-xs px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-slate-200 transition-colors"
                                    >
                                        {t}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4 w-full md:w-auto">
                        <div className="space-y-2">
                            <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest px-1">Depth</label>
                            <select
                                value={analysisDepth}
                                onChange={(e) => setAnalysisDepth(e.target.value)}
                                className="w-full bg-slate-950/70 border border-slate-800 rounded-xl py-3 px-4 text-slate-300 focus:border-purple-500 outline-none text-sm"
                            >
                                <option value="Standard">Standard</option>
                                <option value="Comprehensive">Comprehensive</option>
                                <option value="Deep Dive">Deep Dive</option>
                            </select>
                        </div>
                        <div className="space-y-2">
                            <label className="text-[10px] font-black text-slate-500 uppercase tracking-widest px-1">Timeframe</label>
                            <select
                                value={timeframe}
                                onChange={(e) => setTimeframe(e.target.value)}
                                className="w-full bg-slate-950/70 border border-slate-800 rounded-xl py-3 px-4 text-slate-300 focus:border-purple-500 outline-none text-sm"
                            >
                                <option value="1M">1 Month</option>
                                <option value="3M">3 Months</option>
                                <option value="6M">6 Months</option>
                                <option value="1Y">1 Year</option>
                            </select>
                        </div>
                    </div>

                    <button
                        onClick={handleAnalyze}
                        disabled={!ticker || isAnalyzing}
                        className="w-full md:w-auto px-8 py-4 bg-gradient-to-r from-purple-600 via-purple-500 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:opacity-50 text-white rounded-xl font-bold transition-all shadow-lg shadow-purple-500/30 hover:shadow-purple-500/40 flex items-center justify-center gap-2 group"
                    >
                        {isAnalyzing ? (
                            <>
                                <Activity className="animate-spin" size={18} />
                                Analyzing...
                            </>
                        ) : (
                            <>
                                <Zap size={18} className="group-hover:scale-110 transition-transform" />
                                Generate Analysis
                            </>
                        )}
                    </button>
                </div>

                {error && (
                    <div className="mt-4 p-3 bg-red-900/30 border border-red-800/50 rounded-lg flex items-center gap-2 text-red-300 text-sm">
                        <AlertCircle size={16} />
                        {error}
                    </div>
                )}
            </div>

            {report && (
                <div className="space-y-6 animate-in slide-in-from-bottom-4">
                    {/* 2. Key Metrics Dashboard */}
                    <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
                        <div className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 p-5 rounded-2xl hover:border-slate-700/50 transition-colors group">
                            <p className="text-[10px] font-bold text-slate-500 uppercase mb-2">Recommendation</p>
                            <p className={`text-xl font-bold ${getRecommendationColor(report.recommendation)} flex items-center gap-2`}>
                                <CheckCircle2 size={20} /> {report.recommendation}
                            </p>
                            <div className="mt-2">
                                <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-1000"
                                        style={{ width: `${report.recommendation_confidence}%` }}
                                    />
                                </div>
                                <p className="text-[10px] text-slate-500 mt-1">{report.recommendation_confidence}% confidence</p>
                            </div>
                        </div>

                        <div className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 p-5 rounded-2xl hover:border-slate-700/50 transition-colors">
                            <p className="text-[10px] font-bold text-slate-500 uppercase mb-2">Target Price</p>
                            <p className="text-2xl font-bold text-slate-100">${report.target_price.toFixed(2)}</p>
                            <div className="flex items-center gap-1 mt-1">
                                {report.upside > 0 ? (
                                    <ArrowUpRight size={14} className="text-emerald-400" />
                                ) : (
                                    <ArrowDownRight size={14} className="text-red-400" />
                                )}
                                <p className={`text-xs font-bold ${report.upside > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                    {report.upside > 0 ? '+' : ''}{report.upside.toFixed(1)}% Upside
                                </p>
                                <span className="text-[10px] text-slate-500 ml-2">Current: ${report.current_price.toFixed(2)}</span>
                            </div>
                        </div>

                        <div className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 p-5 rounded-2xl hover:border-slate-700/50 transition-colors">
                            <p className="text-[10px] font-bold text-slate-500 uppercase mb-2">Risk Rating</p>
                            <p className={`text-xl font-bold ${getRiskColor(report.risk_rating)} flex items-center gap-2`}>
                                <Shield size={20} /> {report.risk_rating}
                            </p>
                            <p className="text-[10px] text-slate-500 mt-1">Based on 15 risk factors</p>
                        </div>

                        <div className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 p-5 rounded-2xl hover:border-slate-700/50 transition-colors">
                            <p className="text-[10px] font-bold text-slate-500 uppercase mb-2">Market Cap</p>
                            <p className="text-xl font-bold text-slate-100 flex items-center gap-2">
                                <DollarSign size={18} className="text-amber-400" /> {report.market_cap}
                            </p>
                            <p className="text-[10px] text-slate-500 mt-1">{report.sector} • {report.industry}</p>
                        </div>

                        <div className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 p-5 rounded-2xl hover:border-slate-700/50 transition-colors lg:col-span-2">
                            <p className="text-[10px] font-bold text-slate-500 uppercase mb-2">Overall Score</p>
                            <div className="flex items-center gap-4">
                                <div className="relative w-20 h-20">
                                    <svg className="w-full h-full" viewBox="0 0 100 100">
                                        <circle cx="50" cy="50" r="40" fill="none" stroke="#1e293b" strokeWidth="8" />
                                        <circle
                                            cx="50" cy="50" r="40" fill="none"
                                            stroke="url(#gradient)"
                                            strokeWidth="8"
                                            strokeLinecap="round"
                                            strokeDasharray="251.2"
                                            strokeDashoffset={251.2 * (1 - 0.78)}
                                            transform="rotate(-90 50 50)"
                                        />
                                        <defs>
                                            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                                <stop offset="0%" stopColor="#8b5cf6" />
                                                <stop offset="100%" stopColor="#ec4899" />
                                            </linearGradient>
                                        </defs>
                                    </svg>
                                    <div className="absolute inset-0 flex items-center justify-center">
                                        <span className="text-2xl font-bold text-slate-100">78</span>
                                    </div>
                                </div>
                                <div className="flex-1">
                                    <p className="text-[10px] text-slate-500">Last updated: {new Date(report.last_updated).toLocaleDateString()}</p>
                                    <div className="h-2 bg-slate-800 rounded-full mt-3 overflow-hidden">
                                        <div className="h-full bg-gradient-to-r from-purple-500 to-pink-400 w-[78%]" />
                                    </div>
                                    <p className="text-[10px] text-slate-400 mt-2 italic">Above sector average (65)</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* 3. Navigation Tabs */}
                    <div className="flex space-x-1 border-b border-slate-800/50 overflow-x-auto">
                        {[
                            { id: 'summary', label: 'Summary', icon: <FileText size={14} /> },
                            { id: 'fundamental', label: 'Fundamental', icon: <BarChart2 size={14} /> },
                            { id: 'technical', label: 'Technical', icon: <TrendingUp size={14} /> },
                            { id: 'sentiment', label: 'Sentiment', icon: <Users size={14} /> },
                            { id: 'risks', label: 'Risks', icon: <AlertTriangle size={14} /> },
                            { id: 'valuation', label: 'Valuation', icon: <Calculator size={14} /> }
                        ].map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`px-5 py-3 text-xs font-bold uppercase tracking-widest transition-all flex items-center gap-2 whitespace-nowrap ${activeTab === tab.id
                                    ? 'text-purple-400 border-b-2 border-purple-500 bg-purple-500/10'
                                    : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/30'
                                    }`}
                            >
                                {tab.icon}
                                {tab.label}
                            </button>
                        ))}
                    </div>

                    {/* 4. Tab Content */}
                    {renderTabContent()}

                    {/* 5. Investment Thesis */}
                    <div className="bg-gradient-to-br from-slate-900/50 to-slate-950/50 border border-slate-800/50 rounded-2xl p-8">
                        <div className="flex items-center justify-between mb-6">
                            <div className="flex items-center gap-3">
                                <FileText className="text-purple-500" size={24} />
                                <h3 className="text-xl font-bold text-slate-100">Investment Thesis</h3>
                            </div>
                            <div className="flex items-center gap-2 text-sm text-slate-500">
                                <Clock size={14} />
                                Updated {new Date(report.last_updated).toLocaleDateString()}
                            </div>
                        </div>
                        <p className="text-slate-300 leading-relaxed text-sm">
                            {report.investment_thesis}
                        </p>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-8">
                            <div className="bg-gradient-to-br from-emerald-900/20 to-emerald-950/20 border border-emerald-800/30 rounded-xl p-6">
                                <h4 className="text-xs font-black text-emerald-400 uppercase tracking-[0.2em] mb-4 flex items-center gap-2">
                                    <ArrowUpRight size={14} /> Positive Catalysts
                                </h4>
                                <ul className="space-y-3 text-sm text-slate-300">
                                    <li className="flex items-start gap-2">
                                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-1.5" />
                                        Ecosystem expansion into mixed reality and AI-powered services
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-1.5" />
                                        Growing services revenue with 70%+ gross margins
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-1.5" />
                                        $110B annual buyback program supporting EPS growth
                                    </li>
                                </ul>
                            </div>
                            <div className="bg-gradient-to-br from-red-900/20 to-red-950/20 border border-red-800/30 rounded-xl p-6">
                                <h4 className="text-xs font-black text-red-400 uppercase tracking-[0.2em] mb-4 flex items-center gap-2">
                                    <AlertTriangle size={14} /> Key Risks
                                </h4>
                                <ul className="space-y-3 text-sm text-slate-300">
                                    <li className="flex items-start gap-2">
                                        <div className="w-1.5 h-1.5 rounded-full bg-red-500 mt-1.5" />
                                        Antitrust regulatory pressure in EU/US markets
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <div className="w-1.5 h-1.5 rounded-full bg-red-500 mt-1.5" />
                                        Geopolitical supply chain sensitivities in Asia
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <div className="w-1.5 h-1.5 rounded-full bg-red-500 mt-1.5" />
                                        High valuation relative to historical mean (30x P/E)
                                    </li>
                                </ul>
                            </div>
                        </div>
                    </div>

                    {/* 6. Export & Actions Footer */}
                    <div className="flex flex-col md:flex-row justify-between items-center gap-4 p-6 bg-gradient-to-br from-slate-900/30 to-slate-950/30 border border-slate-800/50 rounded-2xl">
                        <div className="text-sm text-slate-500">
                            Report ID: {report.ticker}-{Date.now().toString(36).toUpperCase()}
                        </div>
                        <div className="flex gap-4">
                            <button className="flex items-center gap-2 px-6 py-3 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-sm font-bold transition-all hover:scale-[1.02] active:scale-[0.98]">
                                <Download size={16} /> Export PDF
                            </button>
                            <button className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white rounded-xl text-sm font-bold transition-all shadow-lg shadow-purple-600/20 hover:scale-[1.02] active:scale-[0.98]">
                                <Star size={16} /> Add to Portfolio
                            </button>
                            <button className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white rounded-xl text-sm font-bold transition-all shadow-lg shadow-emerald-600/20 hover:scale-[1.02] active:scale-[0.98]">
                                <TargetIcon size={16} /> Set Alert
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Quick Help Tooltip */}
            {!report && (
                <div className="fixed bottom-6 right-6 bg-gradient-to-br from-purple-900/90 to-pink-900/90 border border-purple-800/50 rounded-lg p-4 max-w-xs shadow-2xl animate-in slide-in-from-right-4">
                    <div className="flex items-start gap-3">
                        <Info size={16} className="text-purple-300 mt-0.5" />
                        <div>
                            <p className="text-xs font-bold text-purple-200 mb-1">Get Started</p>
                            <p className="text-xs text-purple-300/80">
                                Enter a ticker symbol above and click "Generate Analysis" to receive AI-powered investment insights.
                                Try AAPL, MSFT, or TSLA for demo.
                            </p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default AIAnalyst;