'use client';

import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, IChartApi, Time, CandlestickSeries, HistogramSeries } from 'lightweight-charts';

export interface CandlestickData {
    time: Time;
    open: number;
    high: number;
    low: number;
    close: number;
}

export interface VolumeData {
    time: Time;
    value: number;
    color: string;
}

interface FinancialChartProps {
    data: CandlestickData[];
    volumeData?: VolumeData[];
    width?: number;
    height?: number;
    colors?: {
        backgroundColor?: string;
        textColor?: string;
        upColor?: string;
        downColor?: string;
    };
}

export const FinancialChart: React.FC<FinancialChartProps> = ({
    data,
    volumeData,
    width,
    height = 400,
    colors: {
        backgroundColor = 'transparent',
        textColor = '#94a3b8',
        upColor = '#10b981',
        downColor = '#ef4444',
    } = {},
}) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);

    useEffect(() => {
        if (!chartContainerRef.current) return;

        const handleResize = () => {
            if (chartRef.current && chartContainerRef.current) {
                chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
        };

        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: backgroundColor },
                textColor,
            },
            grid: {
                vertLines: { color: 'rgba(30, 41, 59, 0.4)', style: 1 },
                horzLines: { color: 'rgba(30, 41, 59, 0.4)', style: 1 },
            },
            width: width || chartContainerRef.current.clientWidth,
            height,
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
                borderColor: 'rgba(30, 41, 59, 1)',
            },
            rightPriceScale: {
                borderColor: 'rgba(30, 41, 59, 1)',
            },
            crosshair: {
                mode: 0, // Normal mode
                vertLine: {
                    color: '#64748b',
                    width: 1,
                    style: 3,
                    labelBackgroundColor: '#1e293b',
                },
                horzLine: {
                    color: '#64748b',
                    width: 1,
                    style: 3,
                    labelBackgroundColor: '#1e293b',
                },
            },
        });

        chartRef.current = chart;

        const candlestickSeries = chart.addSeries(CandlestickSeries, {
            upColor,
            downColor,
            borderVisible: false,
            wickUpColor: upColor,
            wickDownColor: downColor,
        });

        candlestickSeries.setData(data);

        if (volumeData && volumeData.length > 0) {
            const volumeSeries = chart.addSeries(HistogramSeries, {
                priceFormat: {
                    type: 'volume',
                },
                priceScaleId: '', // overlaying the volume on the same chart
            });
            volumeSeries.priceScale().applyOptions({
                scaleMargins: {
                    top: 0.8, // highest point of the series will be at 80% from top
                    bottom: 0,
                },
            });
            volumeSeries.setData(volumeData);
        }

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, [data, volumeData, backgroundColor, textColor, upColor, downColor, height, width]);

    return (
        <div
            ref={chartContainerRef}
            className="w-full rounded-xl overflow-hidden border border-slate-800/80 bg-slate-900/50 shadow-inner"
        />
    );
};
