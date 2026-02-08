'use client';

import React, { useEffect, useRef } from 'react';
import {
    createChart,
    ColorType,
    ISeriesApi,
    Time,
    UTCTimestamp,
    CrosshairMode,
    ChartOptions,
    DeepPartial,
    SeriesMarker,
    AreaSeries,
    CandlestickSeries,
    HistogramSeries,
} from 'lightweight-charts';

export interface ChartDataPoint {
    time: string | number;
    value?: number;
    open?: number;
    high?: number;
    low?: number;
    close?: number;
    color?: string;
}

interface LightweightChartProps {
    data: ChartDataPoint[];
    type?: 'area' | 'candlestick' | 'histogram';
    height?: number;
    colors?: {
        backgroundColor?: string;
        lineColor?: string;
        textColor?: string;
        areaTopColor?: string;
        areaBottomColor?: string;
        upColor?: string;
        downColor?: string;
        wickUpColor?: string;
        wickDownColor?: string;
    };
    markers?: SeriesMarker<Time>[];
    onCrosshairMove?: (param: any) => void;
}

const LightweightChart: React.FC<LightweightChartProps> = ({
    data,
    type = 'area',
    height = 300,
    colors = {},
    markers = [],
    onCrosshairMove
}) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<ReturnType<typeof createChart> | null>(null);
    const seriesRef = useRef<ISeriesApi<any> | null>(null);

    // Default Theme Colors (Bloomberg/Midnight Style)
    const defaultColors = {
        backgroundColor: 'transparent',
        lineColor: '#6366f1', // Indigo 500
        textColor: '#94a3b8', // Slate 400
        areaTopColor: 'rgba(99, 102, 241, 0.4)',
        areaBottomColor: 'rgba(99, 102, 241, 0.0)',
        upColor: '#10b981', // Emerald 500
        downColor: '#ef4444', // Red 500
        wickUpColor: '#10b981',
        wickDownColor: '#ef4444',
        gridColor: 'rgba(51, 65, 85, 0.2)', // Slate 700 with opacity
    };

    const theme = { ...defaultColors, ...colors };

    // Initialize Chart
    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chartOptions: DeepPartial<ChartOptions> = {
            layout: {
                background: { type: ColorType.Solid, color: theme.backgroundColor },
                textColor: theme.textColor,
                fontFamily: "'Inter', sans-serif",
            },
            grid: {
                vertLines: { color: theme.gridColor },
                horzLines: { color: theme.gridColor },
            },
            width: chartContainerRef.current.clientWidth,
            height: height,
            crosshair: {
                mode: CrosshairMode.Normal,
                vertLine: {
                    width: 1,
                    color: '#94a3b8',
                    labelBackgroundColor: '#6366f1',
                },
                horzLine: {
                    width: 1,
                    color: '#94a3b8',
                    labelBackgroundColor: '#6366f1',
                },
            },
            timeScale: {
                borderColor: theme.gridColor,
                timeVisible: true,
                secondsVisible: false,
            },
            rightPriceScale: {
                borderColor: theme.gridColor,
                scaleMargins: {
                    top: 0.2,
                    bottom: 0.1,
                },
            },
        };

        const chart = createChart(chartContainerRef.current, chartOptions);
        chartRef.current = chart;

        // Add Series based on type
        let series: ISeriesApi<any>;

        if (type === 'area') {
            series = chart.addSeries(AreaSeries, {
                lineColor: theme.lineColor,
                topColor: theme.areaTopColor,
                bottomColor: theme.areaBottomColor,
                lineWidth: 2,
            });
        } else if (type === 'candlestick') {
            series = chart.addSeries(CandlestickSeries, {
                upColor: theme.upColor,
                downColor: theme.downColor,
                wickUpColor: theme.wickUpColor,
                wickDownColor: theme.wickDownColor,
                borderVisible: false,
            });
        } else if (type === 'histogram') {
            series = chart.addSeries(HistogramSeries, {
                color: theme.lineColor,
            });
        }

        seriesRef.current = series!;

        // Initial markers set
        if (markers.length > 0) {
            (series! as any).setMarkers(markers);
        }

        // Handle Crosshair Move
        if (onCrosshairMove) {
            chart.subscribeCrosshairMove(onCrosshairMove);
        }

        // Cleanup
        return () => {
            if (onCrosshairMove) chart.unsubscribeCrosshairMove(onCrosshairMove);
            chart.remove();
            chartRef.current = null;
            seriesRef.current = null;
        };
    }, [type, height]);

    // Update Series Options when theme changes (without destroying chart)
    useEffect(() => {
        if (!seriesRef.current || !chartRef.current) return;

        chartRef.current.applyOptions({
            layout: {
                background: { type: ColorType.Solid, color: theme.backgroundColor },
                textColor: theme.textColor,
            },
            grid: {
                vertLines: { color: theme.gridColor },
                horzLines: { color: theme.gridColor },
            }
        });

        if (type === 'area') {
            seriesRef.current.applyOptions({
                lineColor: theme.lineColor,
                topColor: theme.areaTopColor,
                bottomColor: theme.areaBottomColor,
            });
        } else if (type === 'candlestick') {
            seriesRef.current.applyOptions({
                upColor: theme.upColor,
                downColor: theme.downColor,
                wickUpColor: theme.wickUpColor,
                wickDownColor: theme.wickDownColor,
            });
        } else if (type === 'histogram') {
            seriesRef.current.applyOptions({
                color: theme.lineColor,
            });
        }
    }, [
        theme.backgroundColor,
        theme.lineColor,
        theme.textColor,
        theme.areaTopColor,
        theme.areaBottomColor,
        theme.upColor,
        theme.downColor,
        type
    ]);

    // Update Data
    useEffect(() => {
        if (!seriesRef.current || !data) return;

        // Ensure data is sorted by time and in correct format
        const sortedData = [...data].sort((a, b) => {
            const timeA = typeof a.time === 'string' ? new Date(a.time).getTime() : a.time as number;
            const timeB = typeof b.time === 'string' ? new Date(b.time).getTime() : b.time as number;
            return timeA - timeB;
        }).map(d => ({
            ...d,
            time: (typeof d.time === 'string'
                ? (new Date(d.time).getTime() / 1000) as UTCTimestamp
                : d.time as UTCTimestamp)
        }));

        // Filter duplicates
        const uniqueData = sortedData.filter((item, index, self) =>
            index === self.findIndex((t) => t.time === item.time)
        );

        if (uniqueData.length > 0) {
            seriesRef.current.setData(uniqueData);
            chartRef.current?.timeScale().fitContent();
        }
    }, [data]);

    // Update Markers
    useEffect(() => {
        if (!seriesRef.current) return;
        // Defensive check to avoid runtime errors
        if (typeof (seriesRef.current as any).setMarkers === 'function') {
            try {
                (seriesRef.current as any).setMarkers(markers);
            } catch (e) {
                console.error("Error setting markers:", e);
            }
        }
    }, [markers]);

    // Resize Handler
    useEffect(() => {
        const handleResize = () => {
            if (chartContainerRef.current && chartRef.current) {
                chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
        };

        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    return (
        <div ref={chartContainerRef} className="w-full relative" />
    );
};

export default LightweightChart;
