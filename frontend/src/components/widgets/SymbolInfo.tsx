'use client';
import React, { useEffect, useRef, memo } from 'react';

interface SymbolInfoProps {
    symbol?: string;
    width?: string;
}

function SymbolInfoComponent({
    symbol = 'NASDAQ:AAPL',
    width = '100%',
}: SymbolInfoProps) {
    const container = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!container.current) return;

        container.current.innerHTML = '';

        const wrapper = document.createElement('div');
        wrapper.className = 'tradingview-widget-container__widget';
        container.current.appendChild(wrapper);

        const script = document.createElement('script');
        script.src =
            'https://s3.tradingview.com/external-embedding/embed-widget-symbol-info.js';
        script.type = 'text/javascript';
        script.async = true;
        script.innerHTML = JSON.stringify({
            symbol,
            width,
            locale: 'en',
            colorTheme: 'dark',
            isTransparent: true,
        });

        container.current.appendChild(script);
    }, [symbol, width]);

    return (
        <div className="tradingview-widget-container" ref={container} />
    );
}

export const SymbolInfo = memo(SymbolInfoComponent);
