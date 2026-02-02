import React from 'react';
import { Layers } from "lucide-react";
import {getGreekColor} from "@/components/optionsdesk/utils/colors";
import {ChainResponse} from "@/types/all_types";

interface ChainTabProps {
    selectedSymbol: string;
    currentPrice: number;
    selectedExpiry: string;
    setSelectedExpiry: (expiry: string) => void;
    optionsChain: ChainResponse;
}

const ChainTab: React.FC<ChainTabProps> = ({
    selectedSymbol,
    currentPrice,
    selectedExpiry,
    setSelectedExpiry,
    optionsChain
}:ChainTabProps) => {
    return (
        <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
            <div className="flex justify-between items-center mb-6">
                <h3 className="text-lg font-bold text-slate-200 flex items-center gap-2">
                    <Layers size={20} className="text-amber-400" />
                    Options Chain - {selectedSymbol}
                </h3>
                <div className="flex items-center gap-3">
                    <div className="text-sm text-slate-300">
                        Current Price: <span className="font-bold text-emerald-400">${currentPrice.toFixed(2)}</span>
                    </div>
                    <select
                        value={selectedExpiry}
                        onChange={(e) => setSelectedExpiry(e.target.value)}
                        className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-1 text-sm text-slate-200"
                    >
                        {optionsChain.expiration_dates?.map((date: string) => (
                            <option key={date} value={date}>
                                {date}
                            </option>
                        ))}
                    </select>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                    <h4 className="text-sm font-bold text-emerald-400 mb-3">CALLS</h4>
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-slate-700">
                                    <th className="py-2 text-left text-slate-400">Strike</th>
                                    <th className="py-2 text-left text-slate-400">Bid</th>
                                    <th className="py-2 text-left text-slate-400">Ask</th>
                                    <th className="py-2 text-left text-slate-400">IV</th>
                                    <th className="py-2 text-left text-slate-400">Delta</th>
                                </tr>
                            </thead>
                            <tbody>
                                {optionsChain.calls?.slice(0, 10).map((call, idx: number) => (
                                    <tr key={idx} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                                        <td className="py-2 text-slate-300">${call.strike}</td>
                                        <td className="py-2 text-emerald-400">${call.bid?.toFixed(2)}</td>
                                        <td className="py-2 text-red-400">${call.ask?.toFixed(2)}</td>
                                        <td className="py-2 text-amber-400">
                                            {call.impliedVolatility ? (call.impliedVolatility * 100).toFixed(1) + '%' : 'N/A'}
                                        </td>
                                        <td className={`py-2 ${getGreekColor(call.delta || 0, 'delta')}`}>
                                            {call.delta?.toFixed(3) || 'N/A'}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                <div>
                    <h4 className="text-sm font-bold text-red-400 mb-3">PUTS</h4>
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-slate-700">
                                    <th className="py-2 text-left text-slate-400">Strike</th>
                                    <th className="py-2 text-left text-slate-400">Bid</th>
                                    <th className="py-2 text-left text-slate-400">Ask</th>
                                    <th className="py-2 text-left text-slate-400">IV</th>
                                    <th className="py-2 text-left text-slate-400">Delta</th>
                                </tr>
                            </thead>
                            <tbody>
                                {optionsChain.puts?.slice(0, 10).map((put, idx: number) => (
                                    <tr key={idx} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                                        <td className="py-2 text-slate-300">${put.strike}</td>
                                        <td className="py-2 text-emerald-400">${put.bid?.toFixed(2)}</td>
                                        <td className="py-2 text-red-400">${put.ask?.toFixed(2)}</td>
                                        <td className="py-2 text-amber-400">
                                            {put.impliedVolatility ? (put.impliedVolatility * 100).toFixed(1) + '%' : 'N/A'}
                                        </td>
                                        <td className={`py-2 ${getGreekColor(put.delta || 0, 'delta')}`}>
                                            {put.delta?.toFixed(3) || 'N/A'}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ChainTab;
