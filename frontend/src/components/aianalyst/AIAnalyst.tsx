'use client'
import React, {JSX, useEffect, useState} from 'react';
import {
    Activity as ActivityIcon,
    AlertCircle,
    AlertOctagon,
    Award,
    Calculator,
    ChartBar,
    ChartLine,
    ChartPie,
    Cpu,
    Database,
    Download,
    FileText,
    Globe as GlobeIcon,
    LineChart as LineChartIcon,
    MessageSquare,
    Network,
    Percent,
    Shield as ShieldIcon,
    Target as TargetIcon,
    ThumbsUp,
    TrendingUp as TrendingUpIcon,
    Zap
} from "lucide-react";
import {
    Area,
    Bar,
    BarChart,
    CartesianGrid,
    Cell,
    ComposedChart,
    Legend,
    Line,
    Pie,
    PieChart as RePie,
    PolarAngleAxis,
    PolarGrid,
    Radar,
    RadarChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from 'recharts';

import {api} from "@/utils/api";
import {
    AnalystReport,
    FundamentalData, HistoricalDataPoint,
    RisksData,
    SentimentData,
    TechnicalData,
    ValuationMetric
} from "@/types/all_types";
import {toPrecision} from "@/utils/formatters";
import TickerSearch from "@/components/common/TickerSearch";

interface PriceDataPoint {
    date: string;
    price: number;
    volume: number;
    benchmark?: number;
}

const AIAnalyst = () => {
    const [ticker, setTicker] = useState<string>("");
    const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
    const [report, setReport] = useState<AnalystReport | null>(null);
    const [activeTab, setActiveTab] = useState<string>('summary');
    const [error, setError] = useState<string | null>(null);
    const [analysisDepth, setAnalysisDepth] = useState<string>('standard');
    const [timeframe, setTimeframe] = useState<string>('1M');

    const [priceData, setPriceData] = useState<PriceDataPoint[]>([]);
    const [isLoadingPriceData, setIsLoadingPriceData] = useState<boolean>(false);

    useEffect(() => {
        const fetchHistoricalData = async () => {
            if (!report || !ticker) return;

            setIsLoadingPriceData(true);
            try {
                // Map timeframe to API period
                const periodMap: Record<string, string> = {
                    '1D': '1d',
                    '1W': '5d',
                    '1M': '1mo',
                    '3M': '3mo',
                    '6M': '6mo',
                    '1Y': '1y',
                    'YTD': 'ytd',
                    'ALL': 'max'
                };

                const intervalMap: Record<string, string> = {
                    '1D': '5m',
                    '1W': '30m',
                    '1M': '1d',
                    '3M': '1d',
                    '6M': '1d',
                    '1Y': '1d',
                    'YTD': '1d',
                    'ALL': '1wk'
                };

                const period = periodMap[timeframe] || '1mo';
                const interval = intervalMap[timeframe] || '1d';

                // Fetch stock data first, then benchmark sequentially to avoid rate limiting
                const rawData: HistoricalDataPoint[] = await api.market.getHistorical(
                    ticker, {period, interval}
                ).catch((err) => {
                    console.warn(`[AIAnalyst] Historical data fetch failed for ${ticker}:`, err);
                    return [] as HistoricalDataPoint[];
                });

                let rawBenchmark: HistoricalDataPoint[] = [];
                if (ticker.toUpperCase() !== 'SPY' && rawData.length > 0) {
                    rawBenchmark = await api.market.getHistorical(
                        'SPY', {period, interval}
                    ).catch((err) => {
                        console.warn('[AIAnalyst] Benchmark (SPY) fetch failed:', err);
                        return [] as HistoricalDataPoint[];
                    });
                }

                console.log(`[AIAnalyst] ${ticker} data: ${rawData.length} pts, SPY benchmark: ${rawBenchmark.length} pts`);

                if (rawData.length > 0) {
                    const stockStart = Number(rawData[0]?.close) || 1;

                    // Build benchmark values aligned to stock data
                    // Primary: index-based alignment (both arrays cover same period/interval)
                    // Fallback: date-key matching for mismatched lengths
                    let benchmarkValues: (number | undefined)[];

                    if (rawBenchmark.length > 0) {
                        const benchStart = Number(rawBenchmark[0].close) || 1;

                        if (Math.abs(rawData.length - rawBenchmark.length) <= Math.max(5, rawData.length * 0.1)) {
                            // Lengths are close — align by index (most reliable)
                            benchmarkValues = rawData.map((_, idx) => {
                                const benchItem = rawBenchmark[Math.min(idx, rawBenchmark.length - 1)];
                                return benchItem ? (Number(benchItem.close) / benchStart) * stockStart : undefined;
                            });
                        } else {
                            // Lengths differ significantly — fall back to date-key matching
                            const toDateKey = (ts: string): string => {
                                try {
                                    return String(ts).slice(0, 10); // YYYY-MM-DD from ISO string
                                } catch {
                                    return '';
                                }
                            };
                            const benchmarkMap = new Map<string, number>();
                            rawBenchmark.forEach((item: HistoricalDataPoint) => {
                                const key = toDateKey(item.timestamp || (item as any).date);
                                benchmarkMap.set(key, (Number(item.close) / benchStart) * stockStart);
                            });
                            benchmarkValues = rawData.map((item: HistoricalDataPoint) => {
                                const key = toDateKey(item.timestamp || (item as any).date);
                                return benchmarkMap.get(key);
                            });
                        }
                    } else {
                        benchmarkValues = rawData.map(() => undefined);
                    }

                    const formattedData: PriceDataPoint[] = rawData.map((item: HistoricalDataPoint, idx: number) => {
                        let dateStr: string;
                        const rawDate = item.timestamp || (item as any).date;

                        try {
                            const dateObj = new Date(rawDate);
                            if (isNaN(dateObj.getTime())) throw new Error('Invalid date');
                            if (timeframe === '1D') {
                                dateStr = dateObj.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
                            } else if (['1W', '1M'].includes(timeframe)) {
                                dateStr = dateObj.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                            } else {
                                dateStr = dateObj.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
                            }
                        } catch {
                            dateStr = String(rawDate).slice(0, 10);
                        }

                        return {
                            date: dateStr,
                            price: Number(item.close || 0),
                            volume: Number(item.volume || 0),
                            benchmark: benchmarkValues[idx],
                        };
                    });

                    setPriceData(formattedData);
                } else {
                    console.warn('[AIAnalyst] No historical data returned for', ticker, 'period:', period, 'interval:', interval);
                    setPriceData([]);
                }
            } catch (err) {
                console.error("❌ Failed to fetch historical data:", err);
            } finally {
                setIsLoadingPriceData(false);
            }
        };

        fetchHistoricalData();
    }, [report, ticker, timeframe]);

    const handleAnalyze = async (): Promise<void> => {
        if (!ticker.trim()) {
            setError("Please enter a ticker symbol");
            return;
        }

        setIsAnalyzing(true);
        setError(null);
        setReport(null);
        setPriceData([]); // ✅ Clear previous price data

        try {
            const reportData = await api.analyst.getReport(ticker, {depth: analysisDepth});
            setReport(reportData);  // ✅ Fixed - no .data access
            console.log("Analysis report received:", reportData);
        } catch (err: unknown) {
            console.error("Analysis failed", err);
            if (typeof err === 'object' && err !== null) {
                if ('message' in err) {
                    setError((err as { message: string }).message);
                } else if ('detail' in err) {
                    setError((err as { detail: string }).detail);
                } else {
                    setError("Analysis failed to generate report");
                }
            } else {
                setError("Analysis failed to generate report");
            }
        } finally {
            setIsAnalyzing(false);
        }
    };

    const getRecommendationColor = (recommendation: string): string => {
        const colors: Record<string, string> = {
            'Strong Buy': 'text-emerald-400',
            'Buy': 'text-green-400',
            'Hold': 'text-amber-400',
            'Sell': 'text-orange-400',
            'Strong Sell': 'text-red-400'
        };
        return colors[recommendation] || 'text-slate-400';
    };

    const getRiskColor = (risk: string): string => {
        const colors: Record<string, string> = {
            'Low': 'text-emerald-400',
            'Medium': 'text-amber-400',
            'High': 'text-orange-400',
            'Very High': 'text-red-400'
        };
        return colors[risk] || 'text-slate-400';
    };

    // Tab content components
    const SummaryTab = (): JSX.Element => (
        <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div
                    className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                    <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-4 flex items-center gap-2">
                        <Cpu size={16} className="text-purple-500"/> AI Insights
                    </h4>
                    {report ? (() => {
                        const tech = report.technical;
                        const rsiLabel = tech.rsi > 70 ? 'overbought conditions' : tech.rsi < 30 ? 'oversold conditions' : 'neutral RSI territory';
                        const trendLabel = tech.trend_strength > 60 ? 'bullish momentum' : tech.trend_strength < 40 ? 'bearish pressure' : 'a consolidation pattern';
                        const macdLabel = tech.macd.histogram > 0 ? 'positive MACD divergence' : 'negative MACD convergence';
                        const marginLabel = report.fundamental.profit_margin > 15 ? 'strong profit margins' : report.fundamental.profit_margin > 5 ? 'moderate profit margins' : 'thin profit margins';
                        const growthLabel = report.fundamental.revenue_growth > 10 ? 'robust revenue growth' : report.fundamental.revenue_growth > 0 ? 'modest revenue growth' : 'declining revenues';

                        return (
                            <>
                                <p className="text-sm text-slate-300 leading-relaxed">
                                    Our AI model identifies <span className="text-emerald-400 font-bold">{growthLabel}</span> with
                                    <span className="text-blue-400 font-bold"> {marginLabel}</span>. Technical indicators show
                                    <span className="text-amber-400 font-bold"> {trendLabel}</span> with {rsiLabel} and {macdLabel}.
                                    Target price of <span className="text-cyan-400 font-bold">${toPrecision(report.target_price)}</span> implies
                                    {' '}{(report.upside).toFixed(1)}% {report.upside >= 0 ? 'upside' : 'downside'}.
                                </p>
                                <div className="mt-4 p-3 bg-purple-900/20 border border-purple-800/30 rounded-lg">
                                    <p className="text-xs text-purple-300">
                                        <span className="font-bold">AI Confidence:</span> {report.recommendation_confidence}% — Based on
                                        technical, fundamental, and sentiment analysis.
                                    </p>
                                </div>
                            </>
                        );
                    })() : (
                        <p className="text-sm text-slate-300">Loading analysis data...</p>
                    )}
                </div>

                <div
                    className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                    <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-4 flex items-center gap-2">
                        <Award size={16} className="text-amber-500"/> Key Takeaways
                    </h4>
                    {report ? (
                        <ul className="space-y-3">
                            <li className="flex items-start gap-2 text-sm">
                                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-1.5"/>
                                <span><span
                                    className={`font-bold ${getRecommendationColor(report.recommendation)}`}>{report.recommendation}</span>
                                rating with {(report.upside).toFixed(1)}% upside to target</span>
                            </li>
                            <li className="flex items-start gap-2 text-sm">
                                <div className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-1.5"/>
                                <span><span className="font-bold text-blue-400">{report.sector}</span>
                                sector with strong growth potential</span>
                            </li>
                            <li className="flex items-start gap-2 text-sm">
                                <div className="w-1.5 h-1.5 rounded-full bg-amber-500 mt-1.5"/>
                                <span>Technical indicators show <span className="font-bold text-amber-400">
                                    {report.technical.trend_strength > 0.6 ? 'bullish momentum' : 'consolidation pattern'}
                                </span></span>
                            </li>
                            <li className="flex items-start gap-2 text-sm">
                                <div className="w-1.5 h-1.5 rounded-full bg-purple-500 mt-1.5"/>
                                <span><span className="font-bold text-purple-400">Market Cap:</span> {report.market_cap}</span>
                            </li>
                        </ul>
                    ) : (
                        <p className="text-sm text-slate-300">Loading key takeaways...</p>
                    )}
                </div>
            </div>

            {/* ✅ UPDATED: Performance chart with real data and timeframe selector */}
            <div
                className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                <div className="flex items-center justify-between mb-6">
                    <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest flex items-center gap-2">
                        <LineChartIcon size={16} className="text-cyan-500"/> Performance vs Benchmark
                    </h4>

                    {/* ✅ Timeframe selector */}
                    <div className="flex gap-1 bg-slate-800/50 rounded-lg p-1">
                        {['1D', '1W', '1M', '3M', '6M', '1Y', 'YTD', 'ALL'].map((tf) => (
                            <button
                                key={tf}
                                onClick={() => setTimeframe(tf)}
                                className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
                                    timeframe === tf
                                        ? 'bg-violet-600 text-white'
                                        : 'text-slate-400 hover:text-slate-300'
                                }`}
                            >
                                {tf}
                            </button>
                        ))}
                    </div>
                </div>

                <div className="h-[200px] relative">
                    {isLoadingPriceData && (
                        <div
                            className="absolute inset-0 flex items-center justify-center bg-slate-900/50 backdrop-blur-sm rounded-lg z-10">
                            <div className="flex items-center gap-2 text-slate-400">
                                <div
                                    className="w-4 h-4 border-2 border-violet-500 border-t-transparent rounded-full animate-spin"/>
                                <span className="text-sm">Loading price data...</span>
                            </div>
                        </div>
                    )}

                    {priceData.length > 0 ? (
                        <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart data={priceData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                                <XAxis
                                    dataKey="date"
                                    stroke="#64748b"
                                    fontSize={10}
                                    angle={-45}
                                    textAnchor="end"
                                    height={60}
                                />
                                <YAxis stroke="#64748b" fontSize={10}/>
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#1e293b',
                                        border: '1px solid #334155',
                                        borderRadius: '8px'
                                    }}
                                    formatter={(value) => [`$${Number(value).toFixed(2)}`, '']}
                                />
                                <Legend/>
                                <Area
                                    type="monotone"
                                    dataKey="price"
                                    fill="#3b82f6"
                                    fillOpacity={0.1}
                                    stroke="#3b82f6"
                                    strokeWidth={2}
                                    name={ticker || "Stock"}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="benchmark"
                                    stroke="#94a3b8"
                                    strokeWidth={1}
                                    strokeDasharray="3 3"
                                    name="S&P 500"
                                    dot={false}
                                    connectNulls
                                />
                            </ComposedChart>
                        </ResponsiveContainer>
                    ) : (
                        <div className="flex items-center justify-center h-full text-slate-500">
                            <p className="text-sm">No price data available</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );


    const FundamentalTab = (): JSX.Element => {
        if (!report) {
            return <div>Loading fundamental data...</div>;
        }

        const fundamental = report.fundamental as FundamentalData;

        return (
            <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div
                        className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                        <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-6 flex items-center gap-2">
                            <ChartBar size={16} className="text-emerald-500"/> Financial Ratios
                        </h4>
                        <div className="space-y-4">
                            {Object.entries(fundamental).map(([key, value]) => {
                                const formattedKey = key.replace(/_/g, ' ');
                                const displayValue = typeof value === 'number'
                                    ? key.includes('yield') || key.includes('growth') || key.includes('margin')
                                        ? `${value.toFixed(2)}%`
                                        : value.toFixed(2)
                                    : value;

                                const suffix = key.includes('ratio') ? 'x' :
                                    key.includes('yield') || key.includes('growth') || key.includes('margin') ? '%' : '';

                                return (
                                    <div key={key}
                                         className="flex items-center justify-between p-3 bg-slate-900/50 rounded-lg">
                                        <div className="flex items-center gap-2">
                                            <Percent size={14} className="text-slate-500"/>
                                            <span className="text-xs text-slate-400 capitalize">{formattedKey}</span>
                                        </div>
                                        <div className="text-right">
                                            <p className="text-sm font-bold text-slate-100">
                                                {displayValue}{suffix}
                                            </p>
                                            <p className="text-[10px] text-slate-500 mt-1">
                                                {typeof value === 'number' && value !== 0
                                                    ? key.includes('ratio') || key === 'pe_ratio'
                                                        ? value > 25 ? 'Above average' : value > 15 ? 'Near average' : 'Below average'
                                                    : key === 'roe'
                                                        ? value > 20 ? 'Strong' : value > 10 ? 'Adequate' : 'Weak'
                                                    : key.includes('growth')
                                                        ? value > 10 ? 'High growth' : value > 0 ? 'Moderate' : 'Declining'
                                                    : key.includes('margin')
                                                        ? value > 20 ? 'High margin' : value > 10 ? 'Moderate margin' : 'Low margin'
                                                    : ''
                                                : ''}
                                            </p>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    <div
                        className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                        <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-6 flex items-center gap-2">
                            <TrendingUpIcon size={16} className="text-blue-500"/> Growth Metrics
                        </h4>
                        <div className="h-[250px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={[
                                    {name: 'Revenue Growth', current: fundamental.revenue_growth},
                                    {name: 'EPS Growth', current: fundamental.eps_growth},
                                    {name: 'Profit Margin', current: fundamental.profit_margin},
                                    {name: 'ROE', current: fundamental.roe},
                                    {name: 'Div Yield', current: fundamental.dividend_yield},
                                ]}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                                    <XAxis dataKey="name" stroke="#64748b" fontSize={10}/>
                                    <YAxis stroke="#64748b" fontSize={10}/>
                                    <Tooltip/>
                                    <Bar dataKey="current" fill="#3b82f6" name="Value (%)"
                                         radius={[4, 4, 0, 0]}/>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>

                <div
                    className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                    <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-4 flex items-center gap-2">
                        <Database size={16} className="text-amber-500"/> Financial Health Score
                    </h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        {[
                            {label: 'Solvency', score: Math.max(0, Math.min(100, Math.round(100 - (fundamental.debt_to_equity || 0) / 2))), desc: `D/E: ${(fundamental.debt_to_equity || 0).toFixed(1)}%`},
                            {label: 'Profitability', score: Math.max(0, Math.min(100, Math.round((fundamental.profit_margin || 0) * 3))), desc: `Margin: ${(fundamental.profit_margin || 0).toFixed(1)}%`},
                            {label: 'Returns', score: Math.max(0, Math.min(100, Math.round(fundamental.roe || 0))), desc: `ROE: ${(fundamental.roe || 0).toFixed(1)}%`},
                            {label: 'Growth', score: Math.max(0, Math.min(100, Math.round(50 + (fundamental.revenue_growth || 0) * 2))), desc: `Rev Growth: ${(fundamental.revenue_growth || 0).toFixed(1)}%`}
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
    };

    const TechnicalTab = (): JSX.Element => {
        if (!report) {
            return <div>Loading technical data...</div>;
        }

        const technical = report.technical as TechnicalData;

        return (
            <div className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div
                        className="lg:col-span-2 bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                        <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-6 flex items-center gap-2">
                            <ChartLine size={16} className="text-cyan-500"/> Technical Indicators
                        </h4>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                            {[
                                {label: 'RSI (14)', value: technical.rsi, status: technical.rsi_signal},
                                {
                                    label: 'MACD',
                                    value: technical.macd.value,
                                    status: technical.macd.histogram > 0 ? 'Bullish' : 'Bearish'
                                },
                                {label: '20D MA', value: technical.ma_20, status: 'Support'},
                                {label: '50D MA', value: technical.ma_50, status: 'Support'},
                                {label: '200D MA', value: technical.ma_200, status: 'Support'},
                                {label: 'Volume Trend', value: technical.volume_trend, status: technical.volume_trend}
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
                                    {technical.support_levels.map((level, idx) => (
                                        <p key={idx} className="text-sm text-emerald-400">${level.toFixed(2)}</p>
                                    ))}
                                </div>
                            </div>
                            <div className="p-4 bg-slate-900/50 rounded-xl">
                                <p className="text-xs text-slate-500">Resistance Levels</p>
                                <div className="mt-2 space-y-1">
                                    {technical.resistance_levels.map((level, idx) => (
                                        <p key={idx} className="text-sm text-red-400">${level.toFixed(2)}</p>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>

                    <div
                        className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                        <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-6 flex items-center gap-2">
                            <ActivityIcon size={16} className="text-purple-500"/> Trend Analysis
                        </h4>
                        <div className="space-y-4">
                            <div>
                                <div className="flex justify-between text-xs mb-1">
                                    <span className="text-slate-400">Trend Strength</span>
                                    <span
                                        className="text-slate-100">{(technical.trend_strength * 100).toFixed(1)}%</span>
                                </div>
                                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                                    <div className="h-full bg-gradient-to-r from-blue-500 to-cyan-400"
                                         style={{width: `${technical.trend_strength * 100}%`}}/>
                                </div>
                            </div>

                            <div className="p-4 bg-slate-900/50 rounded-xl">
                                <p className="text-xs text-slate-500 mb-2">MACD Signal</p>
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm font-bold text-slate-100">{technical.macd.value.toFixed(2)}</p>
                                        <p className="text-[10px] text-slate-500">MACD Line</p>
                                    </div>
                                    <div>
                                        <p className="text-sm font-bold text-slate-100">{technical.macd.signal.toFixed(2)}</p>
                                        <p className="text-[10px] text-slate-500">Signal Line</p>
                                    </div>
                                    <div>
                                        <p className={`text-sm font-bold ${technical.macd.histogram > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                            {technical.macd.histogram > 0 ? '+' : ''}{technical.macd.histogram.toFixed(2)}
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
    };

    const SentimentTab = (): JSX.Element => {
        if (!report) {
            return <div>Loading sentiment data...</div>;
        }

        const sentiment = report.sentiment as SentimentData;

        return (
            <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div
                        className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                        <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-6 flex items-center gap-2">
                            <GlobeIcon size={16} className="text-purple-500"/> Market Sentiment
                        </h4>
                        <div className="space-y-4">
                            {Object.entries(sentiment).map(([key, value]) => (
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
                                            style={{width: `${value}%`}}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div
                        className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                        <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-6 flex items-center gap-2">
                            <ChartPie size={16} className="text-pink-500"/> Sentiment Distribution
                        </h4>
                        <div className="h-[250px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <RePie>
                                    <Pie
                                        data={[
                                            {name: 'Bullish', value: sentiment.institutional, color: '#10b981'},
                                            {name: 'Neutral', value: sentiment.retail, color: '#64748b'},
                                            {
                                                name: 'Bearish',
                                                value: 100 - sentiment.institutional - sentiment.retail,
                                                color: '#ef4444'
                                            }
                                        ]}
                                        cx="50%"
                                        cy="50%"
                                        innerRadius={60}
                                        outerRadius={80}
                                        paddingAngle={5}
                                        dataKey="value"
                                    >
                                        {[{color: '#10b981'}, {color: '#64748b'}, {color: '#ef4444'}].map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.color}/>
                                        ))}
                                    </Pie>
                                    <Tooltip/>
                                    <Legend/>
                                </RePie>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div
                        className="bg-gradient-to-br from-emerald-900/20 to-emerald-950/20 border border-emerald-800/30 rounded-xl p-4 text-center">
                        <ThumbsUp className="w-8 h-8 text-emerald-400 mx-auto mb-2"/>
                        <p className="text-lg font-bold text-emerald-400">{sentiment.analyst}%</p>
                        <p className="text-xs text-emerald-300">Analyst Consensus</p>
                    </div>
                    <div
                        className="bg-gradient-to-br from-amber-900/20 to-amber-950/20 border border-amber-800/30 rounded-xl p-4 text-center">
                        <MessageSquare className="w-8 h-8 text-amber-400 mx-auto mb-2"/>
                        <p className="text-lg font-bold text-amber-400">{sentiment.news}%</p>
                        <p className="text-xs text-amber-300">News Sentiment</p>
                    </div>
                    <div
                        className="bg-gradient-to-br from-blue-900/20 to-blue-950/20 border border-blue-800/30 rounded-xl p-4 text-center">
                        <Network className="w-8 h-8 text-blue-400 mx-auto mb-2"/>
                        <p className="text-lg font-bold text-blue-400">{sentiment.social}%</p>
                        <p className="text-xs text-blue-300">Social Media</p>
                    </div>
                </div>
            </div>
        );
    };

    const RisksTab = (): JSX.Element => {
        if (!report) {
            return <div>Loading risk data...</div>;
        }

        const risks = report.risks as RisksData;

        return (
            <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {Object.entries(risks).map(([category, riskList]) => (
                        <div key={category}
                             className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                            <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-4 flex items-center gap-2">
                                <AlertOctagon size={16} className={
                                    category === 'regulatory' ? 'text-red-500' :
                                        category === 'competitive' ? 'text-orange-500' :
                                            category === 'market' ? 'text-amber-500' :
                                                category === 'financial' ? 'text-purple-500' :
                                                    'text-slate-500'
                                }/>
                                {category.charAt(0).toUpperCase() + category.slice(1)} Risks
                            </h4>
                            <ul className="space-y-2">
                                {riskList.map((risk: string, riskIdx: number) => (
                                    <li key={riskIdx} className="flex items-start gap-2 text-sm text-slate-300">
                                        <div className="w-1.5 h-1.5 rounded-full bg-red-500 mt-1.5 flex-shrink-0"/>
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

                <div
                    className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                    <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-6 flex items-center gap-2">
                        <ShieldIcon size={16} className="text-cyan-500"/> Risk Mitigation Score
                    </h4>
                    <div className="space-y-4">
                        {[
                            {label: 'Diversification', score: 85, desc: 'Revenue streams across segments'},
                            {label: 'Liquidity', score: 92, desc: 'Cash position & access to capital'},
                            {label: 'Governance', score: 78, desc: 'Board oversight & controls'},
                            {label: 'Hedging', score: 65, desc: 'Risk management strategies'}
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
                                        style={{width: `${item.score}%`}}
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        );
    };

    const ValuationTab = (): JSX.Element => {
        if (!report) {
            return <div>Loading valuation data...</div>;
        }

        return (
            <div className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div
                        className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                        <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-6 flex items-center gap-2">
                            <Calculator size={16} className="text-purple-500"/> Valuation Metrics
                        </h4>
                        <div className="h-[300px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={report.valuation}>
                                    <PolarGrid stroke="#1e293b" strokeOpacity={0.5}/>
                                    <PolarAngleAxis dataKey="subject" tick={{fill: '#94a3b8', fontSize: 10}}/>
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
                                        fill="#64748b"
                                        fillOpacity={0.2}
                                        strokeWidth={1}
                                        strokeDasharray="3 3"
                                    />
                                    <Tooltip/>
                                    <Legend/>
                                </RadarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    <div className="space-y-4">
                        {report.valuation.map((metric: ValuationMetric, idx: number) => (
                            <div key={idx} className="p-4 bg-slate-900/50 rounded-xl">
                                <div className="flex justify-between items-center">
                                    <span className="text-sm font-medium text-slate-300">{metric.subject}</span>
                                    <div className="flex items-center gap-2">
                                        <span
                                            className={`text-lg font-bold ${metric.score >= metric.benchmark ? 'text-emerald-400' : 'text-red-400'}`}>
                                            {metric.score.toFixed(1)}
                                        </span>
                                        <span className="text-xs text-slate-500">vs {metric.benchmark.toFixed(1)}</span>
                                    </div>
                                </div>
                                <p className="text-xs text-slate-400 mt-2">{metric.description}</p>
                            </div>
                        ))}
                    </div>
                </div>

                <div
                    className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                    <h4 className="text-xs font-black text-slate-300 uppercase tracking-widest mb-4 flex items-center gap-2">
                        <TargetIcon size={16} className="text-emerald-500"/> Valuation Summary
                    </h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="text-center p-4 bg-slate-900/50 rounded-xl">
                            <p className="text-2xl font-bold text-emerald-400">{(report.upside).toFixed(1)}%</p>
                            <p className="text-xs text-slate-400 mt-1">Upside Potential</p>
                        </div>
                        <div className="text-center p-4 bg-slate-900/50 rounded-xl">
                            <p className="text-2xl font-bold text-slate-100">${report.current_price.toFixed(2)}</p>
                            <p className="text-xs text-slate-400 mt-1">Current Price</p>
                        </div>
                        <div className="text-center p-4 bg-slate-900/50 rounded-xl">
                            <p className="text-2xl font-bold text-amber-400">${report.target_price.toFixed(2)}</p>
                            <p className="text-xs text-slate-400 mt-1">Target Price</p>
                        </div>
                        <div className="text-center p-4 bg-slate-900/50 rounded-xl">
                            <p className="text-2xl font-bold text-purple-400">{
                                toPrecision(report.valuation.reduce((acc, metric) => acc + metric.score, 0) / report.valuation.length)
                            }</p>
                            <p className="text-xs text-slate-400 mt-1">Avg Score</p>
                        </div>
                    </div>
                </div>
            </div>
        );
    };

    const renderActiveTab = (): JSX.Element => {
        switch (activeTab) {
            case 'summary':
                return <SummaryTab/>;
            case 'fundamental':
                return <FundamentalTab/>;
            case 'technical':
                return <TechnicalTab/>;
            case 'sentiment':
                return <SentimentTab/>;
            case 'risks':
                return <RisksTab/>;
            case 'valuation':
                return <ValuationTab/>;
            default:
                return <SummaryTab/>;
        }
    };

    return (
        <div className="p-6 max-w-7xl mx-auto">
            {/* Header */}
            <div className="mb-8">
                <div className="flex items-center justify-between mb-6">
                    <div>
                        <h1 className="text-3xl font-bold text-slate-100">AI Analyst</h1>
                        <p className="text-slate-400 mt-2">Advanced AI-powered stock analysis and insights</p>
                    </div>
                    <div className="flex items-center gap-4">
                        <div
                            className="bg-gradient-to-br from-violet-900/20 to-fuchsia-900/20 border border-violet-800/30 rounded-xl px-4 py-2">
                            <p className="text-xs text-violet-300">Powered by GPT-4 & Custom Models</p>
                        </div>
                    </div>
                </div>

                {/* Search Bar */}
                <div
                    className="bg-gradient-to-br from-slate-900/60 to-slate-950/60 border border-slate-800/50 rounded-2xl p-6">
                    <div className="flex items-center gap-4">
                        <div className="flex-1">
                            <TickerSearch
                                value={ticker}
                                onChange={setTicker}
                                onSelect={() => handleAnalyze()}
                                placeholder="Enter ticker symbol (e.g., AAPL, GOOGL, TSLA)"
                            />
                        </div>
                        <div className="flex items-center gap-3">
                            <select
                                value={analysisDepth}
                                onChange={(e) => setAnalysisDepth(e.target.value)}
                                className="bg-slate-800/50 border border-slate-700/50 rounded-xl px-4 py-4 text-slate-200 focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none transition-all"
                            >
                                <option value="quick">Quick Analysis</option>
                                <option value="standard">Standard</option>
                                <option value="comprehensive">Comprehensive</option>
                                <option value="deep_dive">Deep Dive</option>
                            </select>
                            <button
                                onClick={handleAnalyze}
                                disabled={isAnalyzing || !ticker.trim()}
                                className="bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 disabled:from-slate-700 disabled:to-slate-700 text-white px-8 py-4 rounded-xl font-semibold transition-all shadow-xl shadow-violet-500/20 disabled:shadow-none flex items-center gap-2 disabled:opacity-70"
                            >
                                {isAnalyzing ? (
                                    <>
                                        <div
                                            className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"/>
                                        Analyzing...
                                    </>
                                ) : (
                                    <>
                                        <Zap size={20}/>
                                        Analyze Stock
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                    {error && (
                        <div
                            className="mt-4 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-sm">
                            <AlertCircle className="inline-block mr-2" size={16}/>
                            {error}
                        </div>
                    )}
                </div>
            </div>

            {/* Main Content */}
            {report ? (
                <>
                    {/* Report Header */}
                    <div
                        className="mb-8 bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 rounded-2xl p-6">
                        <div className="flex items-center justify-between">
                            <div>
                                <div className="flex items-center gap-4 mb-2">
                                    <h2 className="text-2xl font-bold text-slate-100">{report.company_name} ({report.ticker})</h2>
                                    <span
                                        className={`px-3 py-1 rounded-full text-xs font-bold ${getRecommendationColor(report.recommendation)} bg-slate-800/50`}>
                                        {report.recommendation}
                                    </span>
                                </div>
                                <div className="flex items-center gap-6 text-sm text-slate-400">
                                    <span>{report.sector} • {report.industry}</span>
                                    <span>Market Cap: {report.market_cap}</span>
                                    <span>Last Updated: {new Date(report.last_updated).toLocaleDateString()}</span>
                                </div>
                            </div>
                            <div className="text-right">
                                <div className="flex items-baseline gap-2">
                                    <span
                                        className="text-3xl font-bold text-slate-100">${report.current_price.toFixed(2)}</span>
                                    <span
                                        className={`text-lg font-bold ${report.upside >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                        {report.upside >= 0 ? '+' : ''}{(report.upside).toFixed(1)}%
                                    </span>
                                </div>
                                <p className="text-sm text-slate-400 mt-1">Target: ${report.target_price.toFixed(2)}</p>
                            </div>
                        </div>

                        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div className="p-4 bg-slate-900/50 rounded-xl">
                                <p className="text-xs text-slate-500">Investment Thesis</p>
                                <p className="text-sm text-slate-300 mt-1">{report.investment_thesis}</p>
                            </div>
                            <div className="p-4 bg-slate-900/50 rounded-xl">
                                <p className="text-xs text-slate-500">Risk Rating</p>
                                <p className={`text-lg font-bold mt-1 ${getRiskColor(report.risk_rating)}`}>{report.risk_rating}</p>
                                <p className="text-xs text-slate-500 mt-1">Confidence: {(report.recommendation_confidence).toFixed(1)}%</p>
                            </div>
                            <div className="p-4 bg-slate-900/50 rounded-xl">
                                <p className="text-xs text-slate-500">Quick Actions</p>
                                <div className="mt-2 flex gap-2">
                                    <button
                                        className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-300 py-2 rounded-lg text-sm transition-colors">
                                        <Download size={16} className="inline-block mr-2"/>
                                        Export
                                    </button>
                                    <button
                                        className="flex-1 bg-violet-600 hover:bg-violet-500 text-white py-2 rounded-lg text-sm transition-colors">
                                        <FileText size={16} className="inline-block mr-2"/>
                                        Full Report
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Tabs.tsx */}
                    <div className="mb-6">
                        <div className="flex space-x-1 border-b border-slate-800">
                            {['summary', 'fundamental', 'technical', 'sentiment', 'risks', 'valuation'].map((tab) => (
                                <button
                                    key={tab}
                                    onClick={() => setActiveTab(tab)}
                                    className={`px-6 py-3 text-sm font-medium rounded-t-lg transition-colors ${activeTab === tab
                                        ? 'text-violet-400 border-b-2 border-violet-400 bg-violet-400/10'
                                        : 'text-slate-400 hover:text-slate-300 hover:bg-slate-800/50'
                                    }`}
                                >
                                    {tab.charAt(0).toUpperCase() + tab.slice(1)}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Tab Content */}
                    <div className="animate-fade-in">
                        {renderActiveTab()}
                    </div>
                </>
            ) : (
                <div className="text-center py-16">
                    <div
                        className="w-20 h-20 border-4 border-violet-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"/>
                    <h3 className="text-xl font-bold text-slate-100 mb-2">Enter a ticker symbol to begin analysis</h3>
                    <p className="text-slate-400">Our AI will analyze fundamental data, technical indicators, market
                        sentiment, and more.</p>
                </div>
            )}
        </div>
    );
};

export default AIAnalyst;
