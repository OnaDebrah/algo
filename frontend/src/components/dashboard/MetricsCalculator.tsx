import {Trade} from "@/types/all_types";

export const calculateWinningTrades = (trades: Trade[] | null | undefined): number => {
    if (!trades || !Array.isArray(trades)) return 0;
    return trades.filter(t => (t.profit || 0) > 0).length;
};

export const calculateLosingTrades = (trades: Trade[] | null | undefined): number => {
    if (!trades || !Array.isArray(trades)) return 0;
    return trades.filter(t => (t.profit || 0) <= 0).length;
};

export const calculateAvgWin = (trades: Trade[] | null | undefined): number => {
    if (!trades || !Array.isArray(trades)) return 0;
    const wins = trades.filter(t => (t.profit || 0) > 0);
    if (wins.length === 0) return 0;
    return wins.reduce((sum, t) => sum + (t.profit || 0), 0) / wins.length;
};

export const calculateAvgLoss = (trades: Trade[] | null | undefined): number => {
    if (!trades || !Array.isArray(trades)) return 0;
    const losses = trades.filter(t => (t.profit || 0) <= 0);
    if (losses.length === 0) return 0;
    return Math.abs(losses.reduce((sum, t) => sum + (t.profit || 0), 0) / losses.length);
};

export const calculateAvgProfit = (trades: Trade[] | null | undefined): number => {
    if (!trades || !Array.isArray(trades) || trades.length === 0) return 0;
    return trades.reduce((sum, t) => sum + (t.profit || 0), 0) / trades.length;
};

export const calculateProfitFactor = (trades: Trade[] | null | undefined): number => {
    if (!trades || !Array.isArray(trades)) return 0;
    const totalWins = trades.filter(t => (t.profit || 0) > 0).reduce((sum, t) => sum + (t.profit || 0), 0);
    const totalLosses = Math.abs(trades.filter(t => (t.profit || 0) <= 0).reduce((sum, t) => sum + (t.profit || 0), 0));
    return totalLosses > 0 ? totalWins / totalLosses : 0;
};
