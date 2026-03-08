'use client';
import React, { useEffect, useRef, memo } from 'react';

interface TechnicalAnalysisProps {
    symbol?: string;
    interval?: string;
    height?: number;
}

function TechnicalAnalysisComponent({
    symbol = 'NASDAQ:AAPL',
    interval = '1D',
    height = 400,
}: TechnicalAnalysisProps) {
    const container = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!container.current) return;

        // Clear previous widget
        container.current.innerHTML = '';

        const wrapper = document.createElement('div');
        wrapper.className = 'tradingview-widget-container__widget';
        wrapper.style.width = '100%';
        wrapper.style.height = `${height}px`;
        container.current.appendChild(wrapper);

        const script = document.createElement('script');
        script.src =
            'https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js';
        script.type = 'text/javascript';
        script.async = true;
        script.innerHTML = JSON.stringify({
            interval,
            width: '100%',
            isTransparent: true,
            height,
            symbol,
            showIntervalTabs: true,
            displayMode: 'single',
            locale: 'en',
            colorTheme: 'dark',
        });

        container.current.appendChild(script);
    }, [symbol, interval, height]);

    return (
        <div
            className="tradingview-widget-container"
            ref={container}
            style={{ height: `${height}px`, width: '100%' }}
        />
    );
}

export const TechnicalAnalysis = memo(TechnicalAnalysisComponent);
