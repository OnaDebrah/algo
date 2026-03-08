'use client';
import React, { useEffect, useRef, memo } from 'react';

interface MiniChartProps {
    symbol?: string;
    dateRange?: '1D' | '1M' | '3M' | '12M' | '60M' | 'ALL';
    height?: number;
    width?: string;
}

function MiniChartComponent({
    symbol = 'NASDAQ:AAPL',
    dateRange = '1M',
    height = 220,
    width = '100%',
}: MiniChartProps) {
    const container = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!container.current) return;

        container.current.innerHTML = '';

        const wrapper = document.createElement('div');
        wrapper.className = 'tradingview-widget-container__widget';
        wrapper.style.width = width;
        wrapper.style.height = `${height}px`;
        container.current.appendChild(wrapper);

        const script = document.createElement('script');
        script.src =
            'https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js';
        script.type = 'text/javascript';
        script.async = true;
        script.innerHTML = JSON.stringify({
            symbol,
            width: '100%',
            height,
            locale: 'en',
            dateRange,
            colorTheme: 'dark',
            isTransparent: true,
            autosize: false,
            largeChartUrl: '',
            noTimeScale: false,
        });

        container.current.appendChild(script);
    }, [symbol, dateRange, height, width]);

    return (
        <div
            className="tradingview-widget-container"
            ref={container}
            style={{ height: `${height}px`, width }}
        />
    );
}

export const MiniChart = memo(MiniChartComponent);
