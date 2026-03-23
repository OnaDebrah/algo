/* eslint-disable @typescript-eslint/no-explicit-any */
'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
    createChart,
    ColorType,
    ISeriesApi,
    UTCTimestamp,
    CrosshairMode,
    CandlestickSeries,
    HistogramSeries,
} from 'lightweight-charts';
import { useChartDrawings, Drawing } from '@/hooks/useChartDrawings';
import DrawingToolbar, { DrawingTool } from './DrawingToolbar';
import { market } from '@/utils/api';

interface AdvancedChartProps {
    symbol: string;
    height?: number;
}

const TIMEFRAMES = [
    { label: '1m', interval: '1m', period: '1d' },
    { label: '5m', interval: '5m', period: '5d' },
    { label: '15m', interval: '15m', period: '5d' },
    { label: '1h', interval: '1h', period: '1mo' },
    { label: '4h', interval: '4h', period: '3mo' },
    { label: '1D', interval: '1d', period: '1y' },
    { label: '1W', interval: '1wk', period: '5y' },
] as const;

type CandleData = {
    time: UTCTimestamp;
    open: number;
    high: number;
    low: number;
    close: number;
};

type VolumeData = {
    time: UTCTimestamp;
    value: number;
    color: string;
};

const AdvancedChart: React.FC<AdvancedChartProps> = ({ symbol, height = 500 }) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<ReturnType<typeof createChart> | null>(null);
    const candleSeriesRef = useRef<ISeriesApi<any> | null>(null);
    const volumeSeriesRef = useRef<ISeriesApi<any> | null>(null);
    const overlayRef = useRef<HTMLCanvasElement>(null);

    const [activeTimeframe, setActiveTimeframe] = useState(5); // default 1D
    const [activeTool, setActiveTool] = useState<DrawingTool>('select');
    const [activeColor, setActiveColor] = useState('#6366f1');
    const [loading, setLoading] = useState(false);
    const [pendingPoints, setPendingPoints] = useState<{ time: number; price: number }[]>([]);

    const tf = TIMEFRAMES[activeTimeframe];
    const { drawings, addDrawing, clearAll, undo, loadForTimeframe } = useChartDrawings(symbol, tf.interval);

    const fetchData = useCallback(async () => {
        setLoading(true);
        try {
            const quotes = await market.getHistorical(symbol, { period: tf.period, interval: tf.interval });
            if (!candleSeriesRef.current || !volumeSeriesRef.current || !quotes?.length) return;

            const candles: CandleData[] = [];
            const volumes: VolumeData[] = [];

            for (const q of quotes as any[]) {
                const t = (new Date(q.timestamp || q.date || q.time).getTime() / 1000) as UTCTimestamp;
                candles.push({ time: t, open: q.open, high: q.high, low: q.low, close: q.close });
                volumes.push({
                    time: t,
                    value: q.volume ?? 0,
                    color: q.close >= q.open ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)',
                });
            }

            // Sort and deduplicate
            const sortUniq = <T extends { time: UTCTimestamp }>(arr: T[]): T[] => {
                const sorted = arr.sort((a, b) => (a.time as number) - (b.time as number));
                return sorted.filter((item, i, self) => i === 0 || item.time !== self[i - 1].time);
            };

            candleSeriesRef.current.setData(sortUniq(candles));
            volumeSeriesRef.current.setData(sortUniq(volumes));
            chartRef.current?.timeScale().fitContent();
        } catch (err) {
            console.error('Failed to fetch chart data', err);
        } finally {
            setLoading(false);
        }
    }, [symbol, tf.period, tf.interval]);

    // Initialize chart
    useEffect(() => {
        if (!containerRef.current) return;

        const chart = createChart(containerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: 'transparent' },
                textColor: '#94a3b8',
                fontFamily: "'Inter', sans-serif",
                attributionLogo: false,
            },
            grid: {
                vertLines: { color: 'rgba(51,65,85,0.2)' },
                horzLines: { color: 'rgba(51,65,85,0.2)' },
            },
            width: containerRef.current.clientWidth,
            height,
            crosshair: { mode: CrosshairMode.Normal },
            timeScale: { borderColor: 'rgba(51,65,85,0.3)', timeVisible: true },
            rightPriceScale: { borderColor: 'rgba(51,65,85,0.3)' },
        });

        chartRef.current = chart;

        const candleSeries = chart.addSeries(CandlestickSeries, {
            upColor: '#10b981',
            downColor: '#ef4444',
            wickUpColor: '#10b981',
            wickDownColor: '#ef4444',
            borderVisible: false,
        });
        candleSeriesRef.current = candleSeries;

        const volumeSeries = chart.addSeries(HistogramSeries, {
            color: 'rgba(99,102,241,0.3)',
            priceFormat: { type: 'volume' },
            priceScaleId: '',
        });
        volumeSeries.priceScale().applyOptions({
            scaleMargins: { top: 0.85, bottom: 0 },
        });
        volumeSeriesRef.current = volumeSeries;

        const handleResize = () => {
            if (containerRef.current) {
                chart.applyOptions({ width: containerRef.current.clientWidth });
            }
        };
        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
            chartRef.current = null;
            candleSeriesRef.current = null;
            volumeSeriesRef.current = null;
        };
    }, [height]);

    // Fetch data on symbol/timeframe change
    useEffect(() => { fetchData(); }, [fetchData]);

    // Load drawings for new timeframe
    useEffect(() => { loadForTimeframe(symbol, tf.interval); }, [symbol, tf.interval, loadForTimeframe]);

    // Draw overlay for drawings
    useEffect(() => {
        const canvas = overlayRef.current;
        const chart = chartRef.current;
        const series = candleSeriesRef.current;
        if (!canvas || !chart || !series) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const render = () => {
            canvas.width = canvas.offsetWidth * window.devicePixelRatio;
            canvas.height = canvas.offsetHeight * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
            ctx.clearRect(0, 0, canvas.offsetWidth, canvas.offsetHeight);

            const timeScale = chart.timeScale();

            for (const d of drawings) {
                ctx.strokeStyle = d.color;
                ctx.lineWidth = d.lineWidth;
                ctx.fillStyle = d.color;

                if (d.type === 'horizontal' && d.points.length >= 1) {
                    const y = series.priceToCoordinate(d.points[0].price);
                    if (y !== null) {
                        ctx.setLineDash([5, 3]);
                        ctx.beginPath();
                        ctx.moveTo(0, y);
                        ctx.lineTo(canvas.offsetWidth, y);
                        ctx.stroke();
                        ctx.setLineDash([]);
                        ctx.font = '10px Inter';
                        ctx.fillText(`${d.points[0].price.toFixed(2)}`, 4, y - 4);
                    }
                } else if (d.type === 'trendline' && d.points.length >= 2) {
                    const x1 = timeScale.timeToCoordinate(d.points[0].time as any);
                    const y1 = series.priceToCoordinate(d.points[0].price);
                    const x2 = timeScale.timeToCoordinate(d.points[1].time as any);
                    const y2 = series.priceToCoordinate(d.points[1].price);
                    if (x1 !== null && y1 !== null && x2 !== null && y2 !== null) {
                        ctx.beginPath();
                        ctx.moveTo(x1, y1);
                        ctx.lineTo(x2, y2);
                        ctx.stroke();
                    }
                } else if (d.type === 'rectangle' && d.points.length >= 2) {
                    const x1 = timeScale.timeToCoordinate(d.points[0].time as any);
                    const y1 = series.priceToCoordinate(d.points[0].price);
                    const x2 = timeScale.timeToCoordinate(d.points[1].time as any);
                    const y2 = series.priceToCoordinate(d.points[1].price);
                    if (x1 !== null && y1 !== null && x2 !== null && y2 !== null) {
                        ctx.globalAlpha = 0.1;
                        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
                        ctx.globalAlpha = 1;
                        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                    }
                } else if (d.type === 'fibonacci' && d.points.length >= 2) {
                    const y1 = series.priceToCoordinate(d.points[0].price);
                    const y2 = series.priceToCoordinate(d.points[1].price);
                    if (y1 !== null && y2 !== null) {
                        const levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
                        const priceDiff = d.points[1].price - d.points[0].price;
                        ctx.setLineDash([3, 3]);
                        ctx.font = '10px Inter';
                        for (const level of levels) {
                            const price = d.points[0].price + priceDiff * level;
                            const y = series.priceToCoordinate(price);
                            if (y !== null) {
                                ctx.globalAlpha = level === 0 || level === 1 ? 0.8 : 0.4;
                                ctx.beginPath();
                                ctx.moveTo(0, y);
                                ctx.lineTo(canvas.offsetWidth, y);
                                ctx.stroke();
                                ctx.fillText(`${(level * 100).toFixed(1)}% (${price.toFixed(2)})`, 4, y - 4);
                            }
                        }
                        ctx.globalAlpha = 1;
                        ctx.setLineDash([]);
                    }
                } else if (d.type === 'text' && d.points.length >= 1 && d.label) {
                    const x = timeScale.timeToCoordinate(d.points[0].time as any);
                    const y = series.priceToCoordinate(d.points[0].price);
                    if (x !== null && y !== null) {
                        ctx.font = 'bold 12px Inter';
                        ctx.fillText(d.label, x, y);
                    }
                }
            }

            // Draw pending points
            for (const pt of pendingPoints) {
                const x = timeScale.timeToCoordinate(pt.time as any);
                const y = series.priceToCoordinate(pt.price);
                if (x !== null && y !== null) {
                    ctx.fillStyle = activeColor;
                    ctx.beginPath();
                    ctx.arc(x, y, 4, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
        };

        render();
        chart.timeScale().subscribeVisibleLogicalRangeChange(render);
        return () => { chart.timeScale().unsubscribeVisibleLogicalRangeChange(render); };
    }, [drawings, pendingPoints, activeColor]);

    // Handle click for drawing
    const handleChartClick = useCallback((e: React.MouseEvent) => {
        if (activeTool === 'select' || !chartRef.current || !candleSeriesRef.current || !containerRef.current) return;

        const rect = containerRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const time = chartRef.current.timeScale().coordinateToTime(x);
        const price = candleSeriesRef.current.coordinateToPrice(y);
        if (time === null || price === null) return;

        const point = { time: time as number, price: price as number };
        const needsTwoPoints = activeTool === 'trendline' || activeTool === 'fibonacci' || activeTool === 'rectangle';

        if (activeTool === 'horizontal') {
            addDrawing({ type: 'horizontal', points: [point], color: activeColor, lineWidth: 1 });
        } else if (activeTool === 'text') {
            const label = prompt('Enter text:');
            if (label) {
                addDrawing({ type: 'text', points: [point], color: activeColor, lineWidth: 1, label });
            }
        } else if (needsTwoPoints) {
            if (pendingPoints.length === 0) {
                setPendingPoints([point]);
            } else {
                addDrawing({ type: activeTool, points: [...pendingPoints, point], color: activeColor, lineWidth: 1 });
                setPendingPoints([]);
            }
        }
    }, [activeTool, activeColor, pendingPoints, addDrawing]);

    const handleTimeframeChange = (idx: number) => {
        setPendingPoints([]);
        setActiveTimeframe(idx);
    };

    return (
        <div className="space-y-2">
            {/* Toolbar Row */}
            <div className="flex items-center justify-between flex-wrap gap-2">
                {/* Timeframe Selector */}
                <div className="flex items-center gap-0.5 bg-card border border-border rounded-lg p-1">
                    {TIMEFRAMES.map((t, i) => (
                        <button
                            key={t.label}
                            onClick={() => handleTimeframeChange(i)}
                            className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                                activeTimeframe === i
                                    ? 'bg-violet-600 text-white'
                                    : 'text-muted-foreground hover:text-foreground hover:bg-accent'
                            }`}
                        >
                            {t.label}
                        </button>
                    ))}
                </div>

                <DrawingToolbar
                    activeTool={activeTool}
                    onToolChange={setActiveTool}
                    activeColor={activeColor}
                    onColorChange={setActiveColor}
                    onUndo={undo}
                    onClearAll={clearAll}
                    drawingCount={drawings.length}
                />
            </div>

            {/* Chart Container */}
            <div className="relative bg-card border border-border rounded-xl overflow-hidden" style={{ height }}>
                {loading && (
                    <div className="absolute inset-0 flex items-center justify-center bg-background/50 z-20">
                        <div className="w-6 h-6 border-2 border-violet-500 border-t-transparent rounded-full animate-spin" />
                    </div>
                )}
                <div ref={containerRef} className="w-full h-full" />
                <canvas
                    ref={overlayRef}
                    onClick={handleChartClick}
                    className="absolute inset-0 w-full h-full"
                    style={{ pointerEvents: activeTool === 'select' ? 'none' : 'auto', cursor: activeTool !== 'select' ? 'crosshair' : undefined }}
                />
            </div>
        </div>
    );
};

export default AdvancedChart;
