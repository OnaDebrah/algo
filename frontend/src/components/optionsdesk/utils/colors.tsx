
export const getGreekColor = (value: number, greek: string) => {
        if (greek === 'theta') {
            return value < 0 ? 'text-red-400' : 'text-emerald-400';
        }
        if (greek === 'delta') {
            return Math.abs(value) > 0.5 ? 'text-amber-400' : 'text-slate-400';
        }
        return Math.abs(value) > 0.5 ? 'text-amber-400' : 'text-slate-400';
    };

export const getStrategyColor = (sentiment: string) => {
        switch (sentiment.toLowerCase()) {
            case 'bullish':
                return 'text-emerald-400';
            case 'bearish':
                return 'text-red-400';
            case 'neutral':
                return 'text-blue-400';
            case 'volatile':
                return 'text-purple-400';
            default:
                return 'text-slate-400';
        }
    };
