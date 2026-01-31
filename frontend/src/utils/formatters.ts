import {formatInTimeZone} from "date-fns-tz";

export const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
};

export const formatPercent = (value: number) => {
    if (value === undefined || value === null) return 0;
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
};

export const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
};

export const toPrecision = (value: number | undefined | null, precision: number = 4): number => {
    if (value === undefined || value === null) return 0;

    const factor = Math.pow(10, precision);
    return Math.round(value * factor) / factor;
};

export const formatTimeZone: (input: string) => string = (input: string) => {
    return formatInTimeZone(
        new Date(input),
        'America/New_York',
        'yyyy-MM-dd HH:mm:ssXXX'
    )
};

/**
 * Calculates a strategy rating from 0-100
 * @param monthlyReturn - Expected monthly return as a decimal (e.g., 0.02 for 2%)
 * @param drawdown - Max drawdown as a decimal (e.g., 0.15 for 15%)
 * @param sharpeRatio - The annualized Sharpe Ratio
 */
export const calculateStrategyRating = (
    monthlyReturn: number,
    drawdown: number,
    sharpeRatio: number
): number => {
    // 1. Normalize Sharpe Ratio (Target: 2.0+ for a high score)
    // A Sharpe of 1.0 gives 50 points, 2.0 gives 100 points
    const sharpeScore = Math.min(Math.max(sharpeRatio * 50, 0), 100);

    // 2. Normalize Monthly Return (Target: 2% per month)
    // 2% monthly return gives 100 points
    const returnScore = Math.min(Math.max((monthlyReturn / 0.02) * 100, 0), 100);

    // 3. Normalize Drawdown Penalty (Target: Keep below 10%)
    // 10% drawdown = 20 point penalty, 30% drawdown = 60 point penalty
    const absDrawdown = Math.abs(drawdown);
    const drawdownPenalty = Math.min(absDrawdown * 200, 100);

    // 4. Apply Weights
    // Sharpe (50%) + Return (30%) - Drawdown Penalty (20%)
    const rawScore = (sharpeScore * 0.5) + (returnScore * 0.3) - (drawdownPenalty * 0.2);

    // Final clamp between 0 and 100
    return Math.round(Math.max(0, Math.min(100, rawScore)));
};
