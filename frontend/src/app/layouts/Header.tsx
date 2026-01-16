import React from "react";
import { ChevronRight, Clock, Globe, Shield, TrendingUp, Wifi } from "lucide-react";
import { formatCurrency } from "@/utils/formatters";

import { market } from "@/utils/api";
import { Quote } from "@/types/api.types";
import { useEffect, useState } from "react";

interface HeaderProps {
    user: any;
    currentPage: string;
    serverStatus?: boolean;
}

const Header = ({ user, currentPage, serverStatus = false }: HeaderProps) => {
    const [marketData, setMarketData] = useState<Quote[]>([]);

    useEffect(() => {
        const fetchMarketData = async () => {
            // Example symbols to fetch
            const symbols = ['SPY', 'QQQ', 'BTC-USD'];
            try {
                // We'll mock this for now if the endpoint doesn't support batch well, 
                // or use individual calls if getQuotes isn't fully implemented in backend yet.
                // Assuming market.getQuotes exists and works as defined in api.ts
                const response = await market.getQuotes(symbols);
                if (response.data) {
                    setMarketData(response.data);
                }
            } catch (error) {
                console.error("Failed to fetch market data", error);
                // Fallback or leave empty
            }
        };

        if (serverStatus) {
            fetchMarketData();
            const interval = setInterval(fetchMarketData, 60000); // Update every minute
            return () => clearInterval(interval);
        }
    }, [serverStatus]);

    // Helper to format market data for display if needed, 
    // or just map directly if Quote matches the UI expectation.
    // The UI expects: { symbol, price, change, up }
    // Our Quote type has: { symbol, price, change, changePct }

    const displayData = marketData.length > 0 ? marketData.map(q => ({
        symbol: q.symbol,
        price: formatCurrency(q.price),
        change: `${q.changePct >= 0 ? '+' : ''}${q.changePct.toFixed(2)}%`,
        up: q.changePct >= 0
    })) : [
        // Keep initial state empty or loading? Let's show placeholders if empty but connected?
        // Or just show nothing.
        { symbol: 'SPY', price: '---', change: '---', up: true },
        { symbol: 'QQQ', price: '---', change: '---', up: true },
        { symbol: 'BTC', price: '---', change: '---', up: false }
    ];

    const showMainHeader = currentPage === 'live' || currentPage === 'dashboard';

    return (
        <header className="space-y-4">
            {/* Status Bar - Always Visible */}
            <div className="flex items-center justify-between py-3 border-b border-slate-800/80 text-xs">
                <div className="flex items-center space-x-6 text-slate-500">
                    <div className="flex items-center space-x-2">
                        <div className={`w-2 h-2 rounded-full ${serverStatus ? 'bg-emerald-500' : 'bg-red-500'} animate-pulse`} />
                        <span className="font-medium">System <span className="text-slate-400">{serverStatus ? 'Online' : 'Offline'}</span></span>
                    </div>
                    <div className="flex items-center space-x-2 border-l border-slate-800 pl-6">
                        <Globe size={14} className={serverStatus ? "text-blue-500" : "text-slate-600"} />
                        <span className="font-medium">Backend <span className="text-slate-400">{serverStatus ? 'Connected' : 'Disconnected'}</span></span>
                    </div>
                </div>

                <div className="flex items-center space-x-6">
                    {displayData.map((m, i) => (
                        <div key={i} className="flex items-center space-x-2 font-medium">
                            <span className="text-slate-500">{m.symbol}</span>
                            <span className="text-slate-300">{m.price}</span>
                            <span className={m.up ? 'text-emerald-400' : 'text-red-400'}>{m.change}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Main Header - Conditionally Rendered */}
            {showMainHeader && (
                <div className="flex justify-between items-end animate-in fade-in slide-in-from-top-2 duration-300">
                    <div>
                        <div className="flex items-center space-x-2 text-sm text-violet-400 mb-2">
                            <Shield size={16} strokeWidth={2} />
                            <span className="font-semibold tracking-wide">INSTITUTIONAL TERMINAL</span>
                        </div>
                        <h1 className="text-3xl font-bold text-slate-100 tracking-tight">
                            {currentPage === 'live' ? 'Live' : 'Performance'} <span className="text-slate-400 font-normal">Dashboard</span>
                        </h1>
                    </div>

                    <div className="flex items-center space-x-4">
                        {/* Account Balance */}
                        <div className="bg-gradient-to-br from-slate-800/90 to-slate-900/90 backdrop-blur-xl border border-slate-700/50 rounded-xl px-6 py-4 shadow-xl relative overflow-hidden group">
                            <div className="absolute top-0 right-0 p-2 opacity-5 group-hover:opacity-10 transition-opacity">
                                <TrendingUp size={48} />
                            </div>
                            <div className="relative z-10">
                                <p className="text-xs font-semibold text-slate-500 tracking-wider mb-1">BUYING POWER</p>
                                <div className="flex items-baseline space-x-2">
                                    <span className="text-2xl font-bold text-slate-100">
                                        {formatCurrency(124500.00)}
                                    </span>
                                    <span className="text-sm font-semibold text-emerald-400">+2.4%</span>
                                </div>
                            </div>
                        </div>

                        {/* Quick Action */}
                        <button className="bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white px-5 py-4 rounded-xl transition-all shadow-xl shadow-violet-500/20 flex items-center group">
                            <div className="bg-white/20 p-2 rounded-lg mr-3">
                                <Clock size={20} strokeWidth={2} />
                            </div>
                            <div className="text-left">
                                <p className="text-xs font-bold opacity-80 leading-none">EXECUTE</p>
                                <p className="text-sm font-bold leading-none mt-1">New Order</p>
                            </div>
                            <ChevronRight size={20} className="ml-2 group-hover:translate-x-1 transition-transform" strokeWidth={2} />
                        </button>
                    </div>
                </div>
            )}
        </header>
    );
};

export default Header;