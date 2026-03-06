'use client';
import React, { useEffect, useRef, memo } from 'react';

function TickerTapeComponent() {
    const container = useRef<HTMLDivElement>(null);

    useEffect(() => {
        // Prevent multiple script injections in React StrictMode
        if (container.current && container.current.innerHTML !== '') {
            return;
        }

        const script = document.createElement("script");
        script.src = "https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js";
        script.type = "text/javascript";
        script.async = true;
        script.innerHTML = `
      {
        "symbols": [
          { "proName": "FOREXCOM:SPXUSD", "title": "S&P 500" },
          { "proName": "FOREXCOM:NSXUSD", "title": "Nasdaq 100" },
          { "proName": "FX_IDC:EURUSD", "title": "EUR/USD" },
          { "proName": "BITSTAMP:BTCUSD", "title": "Bitcoin" },
          { "proName": "BITSTAMP:ETHUSD", "title": "Ethereum" },
          { "description": "Apple", "proName": "NASDAQ:AAPL" },
          { "description": "Tesla", "proName": "NASDAQ:TSLA" },
          { "description": "Nvidia", "proName": "NASDAQ:NVDA" }
        ],
        "showSymbolLogo": true,
        "isTransparent": true,
        "displayMode": "adaptive",
        "colorTheme": "dark",
        "locale": "en"
      }
    `;
        if (container.current) {
            container.current.appendChild(script);
        }
    }, []);

    return (
        <div className="tradingview-widget-container h-[42px] overflow-hidden flex items-center" ref={container}>
            <div className="tradingview-widget-container__widget h-full w-full"></div>
        </div>
    );
}

export const TickerTape = memo(TickerTapeComponent);
