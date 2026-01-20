export const mockAPI = {
    getChain: async (request: any) => ({
        current_price: 450,
        expiration_dates: ['2024-02-16', '2024-03-15', '2024-04-19'],
        calls: Array.from({ length: 20 }, (_, i) => ({
            strike: 440 + i * 5,
            bid: 5 + Math.random() * 10,
            ask: 6 + Math.random() * 10,
            impliedVolatility: 0.15 + Math.random() * 0.1,
            delta: 0.3 + Math.random() * 0.4,
            volume: Math.floor(Math.random() * 1000)
        })),
        puts: Array.from({ length: 20 }, (_, i) => ({
            strike: 440 + i * 5,
            bid: 5 + Math.random() * 10,
            ask: 6 + Math.random() * 10,
            impliedVolatility: 0.15 + Math.random() * 0.1,
            delta: -(0.3 + Math.random() * 0.4),
            volume: Math.floor(Math.random() * 1000)
        }))
    }),
    calculateGreeks: async (request: any) => ({
        delta: 0.5,
        gamma: 0.02,
        theta: -0.05,
        vega: 0.15,
        rho: 0.03
    }),
    analyzeStrategy: async (request: any) => ({
        max_profit: 500,
        max_loss: -200,
        probability_of_profit: 65,
        initial_cost: 100,
        payoff_diagram: Array.from({ length: 50 }, (_, i) => ({
            price: 400 + i * 5,
            payoff: Math.random() * 200 - 100
        }))
    }),
    calculateRiskMetrics: async (request: any) => ({
        var_95: -150,
        cvar_95: -200,
        kelly_fraction: 0.15,
        recommendation: 'Moderate position size recommended'
    }),
    runMonteCarlo: async (request: any) => ({
        mean_final_price: 455,
        probability_above_current: 52
    }),
    compareStrategies: async (request: any) => ({
        comparisons: []
    }),
    optimizeStrike: async (request: any) => ({
        strikes: Array.from({ length: 5 }, (_, i) => ({
            strike: 445 + i * 5,
            prob_itm: 0.5 - i * 0.1,
            premium_estimate: 8 - i
        }))
    }),
    calculateProbability: async (request: any) => ({
        probability: 0.5
    }),
    calculatePortfolioStats: async (request: any) => ({
        win_rate: 65,
        profit_factor: 1.8,
        avg_win: 150,
        avg_loss: -80,
        expectancy: 50,
        kelly_fraction: 0.15
    }),
    runBacktest: async (request: any) => ({
        data: {
            result: {
                total_return: 15.5,
                win_rate: 68,
                profit_factor: 2.1,
                sharpe_ratio: 1.4
            },
            equity_curve: Array.from({ length: 20 }, (_, i) => ({
                timestamp: new Date(2024, 0, i + 1).toISOString(),
                equity: 10000 + i * 100,
                drawdown: -Math.random() * 5,
                cash: 10000 + i * 100
            })),
            trades: Array.from({ length: 10 }, (_, i) => ({
                symbol: 'SPY',
                strategy: 'COVERED_CALL',
                profit: Math.random() * 200 - 50,
                timestamp: new Date(2024, 0, i + 1).toISOString()
            }))
        }
    })
};