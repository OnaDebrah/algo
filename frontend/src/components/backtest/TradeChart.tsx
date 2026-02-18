'use client'
import React, { useEffect, useRef, useMemo } from 'react';
import { createChart, ColorType, CrosshairMode, CandlestickSeries, HistogramSeries, createSeriesMarkers } from 'lightweight-charts';
import type { IChartApi, Time } from 'lightweight-charts';
import { Trade } from '@/types/all_types';

interface OHLCPoint {
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

interface TradeChartProps {
    priceData: Record<string, any>[];
    trades: Trade[];
    height?: number;
}

/**
 * Parse a date string to a YYYY-MM-DD format for lightweight-charts.
 */
function toChartTime(dateStr: string): string {
    try {
        const d = new Date(dateStr);
        if (isNaN(d.getTime())) return dateStr;
        return d.toISOString().split('T')[0];
    } catch {
        return dateStr;
    }
}

const TradeChart: React.FC<TradeChartProps> = ({ priceData, trades, height = 450 }) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);

    // Normalize raw price data to typed OHLC
    const ohlcData: OHLCPoint[] = useMemo(() => {
        if (!priceData || priceData.length === 0) return [];
        return priceData.map((d) => ({
            timestamp: d.timestamp || '',
            open: Number(d.open) || 0,
            high: Number(d.high) || 0,
            low: Number(d.low) || 0,
            close: Number(d.close) || 0,
            volume: Number(d.volume) || 0,
        }));
    }, [priceData]);

    // Process OHLC data for the chart
    const candleData = useMemo(() => {
        if (ohlcData.length === 0) return [];

        return ohlcData
            .map((d) => ({
                time: toChartTime(d.timestamp) as Time,
                open: d.open,
                high: d.high,
                low: d.low,
                close: d.close,
            }))
            .filter((d, i, arr) => {
                // Remove duplicates by time
                if (i === 0) return true;
                return d.time !== arr[i - 1].time;
            });
    }, [ohlcData]);

    // Process volume data
    const volumeData = useMemo(() => {
        if (ohlcData.length === 0) return [];

        return ohlcData
            .map((d) => ({
                time: toChartTime(d.timestamp) as Time,
                value: d.volume,
                color: d.close >= d.open ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
            }))
            .filter((d, i, arr) => {
                if (i === 0) return true;
                return d.time !== arr[i - 1].time;
            });
    }, [ohlcData]);

    // Process trade markers
    const markers = useMemo(() => {
        if (!trades || trades.length === 0 || candleData.length === 0) return [];

        // Build a set of valid chart dates for snapping
        const validDates = new Set(candleData.map(d => d.time as string));

        const tradeMarkers = trades
            .map((trade) => {
                const tradeDate = toChartTime(trade.executed_at || '');
                const isBuy = trade.order_type === 'BUY';

                // Find closest valid date if exact match not found
                let matchDate = tradeDate;
                if (!validDates.has(tradeDate) && candleData.length > 0) {
                    let minDiff = Infinity;
                    for (const cd of candleData) {
                        const diff = Math.abs(new Date(cd.time as string).getTime() - new Date(tradeDate).getTime());
                        if (diff < minDiff) {
                            minDiff = diff;
                            matchDate = cd.time as string;
                        }
                    }
                }

                if (!validDates.has(matchDate)) return null;

                return {
                    time: matchDate as Time,
                    position: isBuy ? 'belowBar' as const : 'aboveBar' as const,
                    color: isBuy ? '#10b981' : '#ef4444',
                    shape: isBuy ? 'arrowUp' as const : 'arrowDown' as const,
                    text: `${isBuy ? 'BUY' : 'SELL'} @ $${trade.price.toFixed(2)}${trade.profit !== null && trade.profit !== undefined ? ` (${trade.profit >= 0 ? '+' : ''}$${trade.profit.toFixed(2)})` : ''}`,
                    size: 1.5,
                };
            })
            .filter(Boolean);

        // Sort by time (required by lightweight-charts)
        tradeMarkers.sort((a, b) => {
            if (!a || !b) return 0;
            return (a.time as string).localeCompare(b.time as string);
        });

        return tradeMarkers;
    }, [trades, candleData]);

    // Create and manage chart
    useEffect(() => {
        if (!chartContainerRef.current || candleData.length === 0) return;

        // Clean up existing chart
        if (chartRef.current) {
            chartRef.current.remove();
            chartRef.current = null;
        }

        const chart = createChart(chartContainerRef.current, {
            width: chartContainerRef.current.clientWidth,
            height: height,
            layout: {
                background: { type: ColorType.Solid, color: 'transparent' },
                textColor: '#94a3b8',
                fontSize: 11,
                fontFamily: "'Inter', -apple-system, system-ui, sans-serif",
                attributionLogo: false,
            },
            grid: {
                vertLines: { color: 'rgba(51, 65, 85, 0.3)' },
                horzLines: { color: 'rgba(51, 65, 85, 0.3)' },
            },
            crosshair: {
                mode: CrosshairMode.Normal,
                vertLine: {
                    color: 'rgba(139, 92, 246, 0.3)',
                    width: 1,
                    style: 3,
                    labelBackgroundColor: '#6d28d9',
                },
                horzLine: {
                    color: 'rgba(139, 92, 246, 0.3)',
                    width: 1,
                    style: 3,
                    labelBackgroundColor: '#6d28d9',
                },
            },
            rightPriceScale: {
                borderColor: 'rgba(51, 65, 85, 0.5)',
                scaleMargins: { top: 0.05, bottom: 0.25 },
            },
            timeScale: {
                borderColor: 'rgba(51, 65, 85, 0.5)',
                timeVisible: false,
                fixLeftEdge: true,
                fixRightEdge: true,
            },
        });

        chartRef.current = chart;

        // Add candlestick series (v5 API)
        const candleSeries = chart.addSeries(CandlestickSeries, {
            upColor: '#10b981',
            downColor: '#ef4444',
            borderUpColor: '#10b981',
            borderDownColor: '#ef4444',
            wickUpColor: '#10b981',
            wickDownColor: '#ef4444',
        });

        candleSeries.setData(candleData as any);

        // Add volume histogram (v5 API)
        const volumeSeries = chart.addSeries(HistogramSeries, {
            priceFormat: { type: 'volume' },
            priceScaleId: 'volume',
        });

        chart.priceScale('volume').applyOptions({
            scaleMargins: { top: 0.8, bottom: 0 },
        });

        volumeSeries.setData(volumeData as any);

        // Add trade markers (v5 API: createSeriesMarkers)
        if (markers.length > 0) {
            createSeriesMarkers(candleSeries, markers as any);
        }

        // Fit content
        chart.timeScale().fitContent();

        // Handle resize
        const handleResize = () => {
            if (chartContainerRef.current && chartRef.current) {
                chartRef.current.applyOptions({
                    width: chartContainerRef.current.clientWidth,
                });
            }
        };

        const resizeObserver = new ResizeObserver(handleResize);
        resizeObserver.observe(chartContainerRef.current);

        return () => {
            resizeObserver.disconnect();
            if (chartRef.current) {
                chartRef.current.remove();
                chartRef.current = null;
            }
        };
    }, [candleData, volumeData, markers, height]);

    if (!priceData || priceData.length === 0) {
        return null;
    }

    // Count buy/sell trades
    const buyCount = trades.filter(t => t.order_type === 'BUY').length;
    const sellCount = trades.filter(t => t.order_type === 'SELL').length;

    return (
        <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-xl">
            <div className="flex justify-between items-center mb-4">
                <div>
                    <h3 className="text-xl font-semibold text-slate-100 italic">Price Action & Trade Signals</h3>
                    <p className="text-xs text-slate-500 mt-1 uppercase font-bold tracking-widest">
                        Candlestick chart with entry/exit markers
                    </p>
                </div>
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <div className="flex items-center gap-1.5">
                            <svg width="12" height="12" viewBox="0 0 12 12">
                                <polygon points="6,0 12,12 0,12" fill="#10b981" />
                            </svg>
                            <span className="text-[10px] font-bold text-emerald-400">BUY ({buyCount})</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                            <svg width="12" height="12" viewBox="0 0 12 12">
                                <polygon points="6,12 12,0 0,0" fill="#ef4444" />
                            </svg>
                            <span className="text-[10px] font-bold text-red-400">SELL ({sellCount})</span>
                        </div>
                    </div>
                </div>
            </div>
            <div
                ref={chartContainerRef}
                className="w-full rounded-xl overflow-hidden"
            />
        </div>
    );
};

export default TradeChart;
