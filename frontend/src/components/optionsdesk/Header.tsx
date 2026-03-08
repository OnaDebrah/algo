import {BrainCircuit, Loader2, RefreshCw, Zap} from "lucide-react";
import React from "react";
import TickerSearch from "@/components/common/TickerSearch";

interface HeaderProps {
    selectedSymbol: string,
    setSelectedSymbol:  React.Dispatch<React.SetStateAction<string>>,
    fetchMLForecast: () => Promise<void>,
    fetchOptionsChain: () => Promise<void>,
    isLoading: boolean
}

const Header: React.FC<HeaderProps> = ({
                                           selectedSymbol,
                                           setSelectedSymbol,
                                           fetchMLForecast,
                                           fetchOptionsChain,
                                           isLoading
                                       }: HeaderProps) => {

    return (
        <div className="flex items-center justify-between">
            < div
                className="flex items-center gap-4">
                < div
                    className="p-3 bg-gradient-to-br from-amber-500/20 to-orange-500/20 rounded-2xl border border-amber-500/30">
                    < Zap
                        className="text-amber-400"
                        size={32}
                    />
                </div>
                <div>
                    <h1 className="text-3xl font-bold text-slate-100">Advanced Options Desk</h1>
                    <p className="text-sm text-slate-500">Professional options analysis & strategy builder</p>
                </div>
            </div>

            <div className="flex gap-3">
                <div>
                    <TickerSearch
                        value={selectedSymbol}
                        onChange={setSelectedSymbol}
                        placeholder="Symbol..."
                        size="sm"
                    />
                </div>
                <button
                    onClick={fetchMLForecast}
                    className="px-4 py-2 bg-purple-600 hover:bg-purple-500 rounded-lg text-white text-sm font-medium flex items-center gap-2"
                    disabled={isLoading}
                >
                    {isLoading ? <Loader2 size={16} className="animate-spin"/> : <BrainCircuit size={16}/>}
                    AI Forecast
                </button>
                <button
                    onClick={fetchOptionsChain}
                    className="px-4 py-2 bg-amber-600 hover:bg-amber-500 rounded-lg text-white text-sm font-medium flex items-center gap-2"
                    disabled={isLoading}
                >
                    {isLoading ? <Loader2 size={16} className="animate-spin"/> : <RefreshCw size={16}/>}
                    Refresh
                </button>
            </div>
        </div>
    )};

export default Header;
