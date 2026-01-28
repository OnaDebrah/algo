import { Activity, DollarSign, Shield, TrendingUp, TrendingDown, BarChart3, Zap, Calendar, Layers, Lock, PieChart } from "lucide-react";
import { StrategyTemplate } from "@/types/all_types";

export const STRATEGY_TEMPLATES: StrategyTemplate[] = [
    {
        id: 'covered_call',
        name: 'Covered Call',
        description: 'Generate income from long stock positions',
        risk: 'Low',
        sentiment: 'Neutral-Bullish',
        icon: DollarSign,
        legs: [
            { id: '1', type: 'stock', position: 'short', strike: 0, quantity: 1, expiration: '' },
            { id: '2', type: 'call', position: 'long', strike: 0, quantity: 100, expiration: '' }
        ],
        typicalSetup: {
            strikes: [0, 5],
            expirations: ['30-45 DTE']
        }
    },
    {
        id: 'cash_secured_put',
        name: 'Cash-Secured Put',
        description: 'Generate income while willing to buy stock at lower price',
        risk: 'Moderate',
        sentiment: 'Bullish',
        icon: Lock,
        legs: [
            { id: '1', type: 'put', position: 'short', strike: 0, quantity: 1, expiration: '' }
        ],
        typicalSetup: {
            strikes: [0],
            expirations: ['30-45 DTE']
        }
    },
    {
        id: 'protective_put',
        name: 'Protective Put',
        description: 'Protect long stock position from downside risk',
        risk: 'Low',
        sentiment: 'Bullish with Protection',
        icon: Shield,
        legs: [
            { id: '1', type: 'stock', position: 'long', strike: 0, quantity: 100, expiration: '' },
            { id: '2', type: 'put', position: 'long', strike: 0, quantity: 1, expiration: '' }
        ]
    },
    {
        id: 'vertical_call_spread',
        name: 'Vertical Call Spread',
        description: 'Bullish strategy with defined risk and reward',
        risk: 'Defined',
        sentiment: 'Bullish',
        icon: TrendingUp,
        legs: [
            { id: '1', type: 'call', position: 'long', strike: 0, quantity: 1, expiration: '' },
            { id: '2', type: 'call', position: 'short', strike: 0, quantity: 1, expiration: '' }
        ],
        typicalSetup: {
            strikes: [0, 5],
            expirations: ['30-60 DTE']
        }
    },
    {
        id: 'vertical_put_spread',
        name: 'Vertical Put Spread',
        description: 'Bearish strategy with defined risk and reward',
        risk: 'Defined',
        sentiment: 'Bearish',
        icon: TrendingDown,
        legs: [
            { id: '1', type: 'put', position: 'long', strike: 0, quantity: 1, expiration: '' },
            { id: '2', type: 'put', position: 'short', strike: 0, quantity: 1, expiration: '' }
        ],
        typicalSetup: {
            strikes: [0, 5],
            expirations: ['30-60 DTE']
        }
    },
    {
        id: 'iron_condor',
        name: 'Iron Condor',
        description: 'Neutral strategy profiting from low volatility',
        risk: 'Defined',
        sentiment: 'Neutral',
        icon: Shield,
        legs: [
            { id: '1', type: 'put', position: 'long', strike: 0, quantity: 1, expiration: '' },  // Wing
            { id: '2', type: 'put', position: 'short', strike: 0, quantity: 1, expiration: '' }, // Body
            { id: '3', type: 'call', position: 'short', strike: 0, quantity: 1, expiration: '' },// Body
            { id: '4', type: 'call', position: 'long', strike: 0, quantity: 1, expiration: '' }  // Wing
        ],
        typicalSetup: {
            strikes: [-10, -5, 5, 10],
            expirations: ['45-60 DTE']
        }
    },
    {
        id: 'butterfly_spread',
        name: 'Butterfly Spread',
        description: 'Neutral strategy with high reward potential',
        risk: 'Defined',
        sentiment: 'Neutral',
        icon: BarChart3,
        legs: [
            { id: '1', type: 'call', position: 'long', strike: -5, quantity: 1, expiration: '' },
            { id: '2', type: 'call', position: 'short', strike: 0, quantity: 2, expiration: '' },
            { id: '3', type: 'call', position: 'long', strike: 5, quantity: 1, expiration: '' }
        ],
        typicalSetup: {
            strikes: [-5, 0, 5],
            expirations: ['30-60 DTE']
        }
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
        ],
        typicalSetup: {
            strikes: [0, 0],
            expirations: ['30-60 DTE']
        }
    },
    {
        id: 'long_strangle',
        name: 'Long Strangle',
        description: 'Profit from large price movements with lower cost than straddle',
        risk: 'Defined',
        sentiment: 'Volatile',
        icon: Zap,
        legs: [
            { id: '1', type: 'call', position: 'long', strike: 0, quantity: 1, expiration: '' },
            { id: '2', type: 'put', position: 'long', strike: 0, quantity: 1, expiration: '' }
        ],
        typicalSetup: {
            strikes: [0, 0],
            expirations: ['30-60 DTE']
        }
    },
    {
        id: 'calendar_spread',
        name: 'Calendar Spread',
        description: 'Profit from time decay differences between options',
        risk: 'Defined',
        sentiment: 'Neutral',
        icon: Calendar,
        legs: [
            { id: '1', type: 'call', position: 'short', strike: 0, quantity: 1, expiration: '' },
            { id: '2', type: 'call', position: 'long', strike: 0, quantity: 1, expiration: '' }
        ],
        typicalSetup: {
            strikes: [0, 0],
            expirations: ['30 DTE', '60 DTE']
        }
    },
    {
        id: 'diagonal_spread',
        name: 'Diagonal Spread',
        description: 'Combination of vertical and calendar spreads',
        risk: 'Defined',
        sentiment: 'Directional',
        icon: Layers,
        legs: [
            { id: '1', type: 'call', position: 'short', strike: 0, quantity: 1, expiration: '' },
            { id: '2', type: 'call', position: 'long', strike: 0, quantity: 1, expiration: '' }
        ],
        typicalSetup: {
            strikes: [0, 0],
            expirations: ['30 DTE', '90 DTE']
        }
    },
    {
        id: 'collar',
        name: 'Collar',
        description: 'Protect long stock position while generating income',
        risk: 'Limited',
        sentiment: 'Neutral',
        icon: Lock,
        legs: [
            { id: '1', type: 'stock', position: 'long', strike: 0, quantity: 100, expiration: '' },
            { id: '2', type: 'call', position: 'short', strike: 0, quantity: 1, expiration: '' },
            { id: '3', type: 'put', position: 'long', strike: 0, quantity: 1, expiration: '' }
        ],
        typicalSetup: {
            strikes: [0, 5],
            expirations: ['30-60 DTE']
        }
    },
    {
        id: 'ratio_spread',
        name: 'Ratio Spread',
        description: 'Profit from directional move with reduced cost',
        risk: 'Unlimited',
        sentiment: 'Directional',
        icon: PieChart,
        legs: [
            { id: '1', type: 'call', position: 'long', strike: 0, quantity: 1, expiration: '' },
            { id: '2', type: 'call', position: 'short', strike: 0, quantity: 2, expiration: '' }
        ],
        typicalSetup: {
            strikes: [0, 0],
            expirations: ['30-60 DTE']
        }
    }
];