import React from 'react';
import { formatPercent } from "@/utils/formatters";

interface PerformanceHeatmapProps {
    monthlyReturns?: Record<string, Record<string, number>>;
}

const PerformanceHeatmap: React.FC<PerformanceHeatmapProps> = ({ monthlyReturns }) => {
    if (!monthlyReturns || Object.keys(monthlyReturns).length === 0) {
        return (
            <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 text-center text-slate-400">
                No monthly returns data available.
            </div>
        );
    }

    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const years = Object.keys(monthlyReturns).sort((a, b) => b.localeCompare(a));

    const getBgColor = (value: number) => {
        if (value > 0) {
            const opacity = Math.min(value / 10, 1);
            return `rgba(34, 197, 94, ${0.1 + opacity * 0.4})`; // Green
        } else if (value < 0) {
            const opacity = Math.min(Math.abs(value) / 10, 1);
            return `rgba(239, 68, 68, ${0.1 + opacity * 0.4})`; // Red
        }
        return 'transparent';
    };

    const getTextColor = (value: number) => {
        if (value > 5) return 'text-green-300';
        if (value > 0) return 'text-green-400';
        if (value < -5) return 'text-red-300';
        if (value < 0) return 'text-red-400';
        return 'text-slate-400';
    };

    return (
        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
                <h3 className="text-lg font-semibold text-white">Monthly Returns Heatmap (%)</h3>
            </div>
            <div className="overflow-x-auto">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b border-slate-800">
                            <th className="px-4 py-3 text-left font-medium text-slate-400">Year</th>
                            {months.map(m => (
                                <th key={m} className="px-2 py-3 text-center font-medium text-slate-400 w-16">{m}</th>
                            ))}
                            <th className="px-4 py-3 text-right font-medium text-slate-400">YTD</th>
                        </tr>
                    </thead>
                    <tbody>
                        {years.map(year => {
                            const yearData = monthlyReturns[year];
                            let ytd = 1;
                            months.forEach(m => {
                                if (yearData[m] !== undefined) {
                                    ytd *= (1 + yearData[m] / 100);
                                }
                            });
                            const ytdPct = (ytd - 1) * 100;

                            return (
                                <tr key={year} className="border-b border-slate-800/50 hover:bg-slate-800/20 transition-colors">
                                    <td className="px-4 py-3 font-semibold text-slate-300">{year}</td>
                                    {months.map(month => {
                                        const value = yearData[month];
                                        return (
                                            <td
                                                key={month}
                                                className="px-1 py-3 text-center"
                                                style={{ backgroundColor: value !== undefined ? getBgColor(value) : 'transparent' }}
                                            >
                                                {value !== undefined ? (
                                                    <span className={`font-medium ${getTextColor(value)}`}>
                                                        {value.toFixed(1)}
                                                    </span>
                                                ) : (
                                                    <span className="text-slate-700">-</span>
                                                )}
                                            </td>
                                        );
                                    })}
                                    <td className={`px-4 py-3 text-right font-bold ${ytdPct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                        {ytdPct.toFixed(1)}%
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default PerformanceHeatmap;
