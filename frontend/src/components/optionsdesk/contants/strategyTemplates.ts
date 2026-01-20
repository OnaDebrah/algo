
import {
    DollarSign,
    ArrowUpRight,
    Shield,
    Activity
} from "lucide-react";
import {StrategyTemplate} from "@/types/all_types";

export const STRATEGY_TEMPLATES: StrategyTemplate[] = [
    {
        id: 'covered_call',
        name: 'Covered Call',
        description: 'Generate income from long stock positions',
        risk: 'Low',
        sentiment: 'Neutral-Bullish',
        icon: DollarSign,
        legs: [
            { id: '1', type: 'call', position: 'short', strike: 0, quantity: 1, expiration: '' },
            { id: '2', type: 'call', position: 'long', strike: 0, quantity: 100, expiration: '' }
        ],
        typicalSetup: {
            strikes: [0, 5],
            expirations: ['30-45 DTE']
        }
    },
    {
        id: 'bull_put_spread',
        name: 'Bull Put Spread',
        description: 'Bullish strategy with defined risk/reward',
        risk: 'Defined',
        sentiment: 'Bullish',
        icon: ArrowUpRight,
        legs: [
            { id: '1', type: 'put', position: 'short', strike: 0, quantity: 1, expiration: '' },
            { id: '2', type: 'put', position: 'long', strike: 0, quantity: 1, expiration: '' }
        ]
    },
    {
        id: 'iron_condor',
        name: 'Iron Condor',
        description: 'Neutral strategy profiting from low volatility',
        risk: 'Defined',
        sentiment: 'Neutral',
        icon: Shield,
        legs: [
            { id: '1', type: 'call', position: 'short', strike: 0, quantity: 1, expiration: '' },
            { id: '2', type: 'call', position: 'long', strike: 0, quantity: 1, expiration: '' },
            { id: '3', type: 'put', position: 'short', strike: 0, quantity: 1, expiration: '' },
            { id: '4', type: 'put', position: 'long', strike: 0, quantity: 1, expiration: '' }
        ]
    },
    {
        id: 'long_straddle',
        name: 'Long Straddle',
        description: 'Profit from massive volatility in either direction',
        risk: 'Defined',
        sentiment: 'Volatile',
        icon: Activity,
        legs: [
            { id: '1', type: 'call', position: 'long', strike: 0, quantity: 1, expiration: '' },
            { id: '2', type: 'put', position: 'long', strike: 0, quantity: 1, expiration: '' }
        ]
    }
];