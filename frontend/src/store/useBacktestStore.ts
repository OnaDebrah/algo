
import { create } from 'zustand';
import {
    BacktestResult,
    MultiAssetConfig,
    SingleAssetConfig,
    EquityCurvePoint,
    Trade
} from '@/types/all_types';
import { backtest } from '@/utils/api';
import { formatDate } from '@/utils/formatters';

interface BacktestState {
    // Mode
    backtestMode: 'single' | 'multi';
    setBacktestMode: (mode: 'single' | 'multi') => void;

    // Configs
    singleConfig: SingleAssetConfig;
    setSingleConfig: (config: SingleAssetConfig | ((prev: SingleAssetConfig) => SingleAssetConfig)) => void;

    multiConfig: MultiAssetConfig;
    setMultiConfig: (config: MultiAssetConfig | ((prev: MultiAssetConfig) => MultiAssetConfig)) => void;

    // Execution
    isRunning: boolean;
    results: BacktestResult | null;
    runBacktest: () => Promise<void>;
    resetResults: () => void;
}

const DEFAULT_SINGLE_CONFIG: SingleAssetConfig = {
    symbol: 'AAPL',
    period: '1y',
    interval: '1d',
    strategy: 'sma_crossover',
    initialCapital: 1000,
    maxPositionPct: 20,
    params: {}
};

const DEFAULT_MULTI_CONFIG: MultiAssetConfig = {
    symbols: [],
    symbolInput: '',
    period: '1y',
    interval: '1d',
    strategyMode: 'same',
    params: {},
    strategy: 'sma_crossover',
    strategies: {},
    allocationMethod: 'equal',
    allocations: {},
    initialCapital: 1000,
    maxPositionPct: 20,
    riskLevel: 'Intermediate'
};

export const useBacktestStore = create<BacktestState>((set, get) => ({
    backtestMode: 'single',
    setBacktestMode: (mode) => set({ backtestMode: mode }),

    singleConfig: DEFAULT_SINGLE_CONFIG,
    setSingleConfig: (config) => set((state) => ({
        singleConfig: typeof config === 'function' ? config(state.singleConfig) : config
    })),

    multiConfig: DEFAULT_MULTI_CONFIG,
    setMultiConfig: (config) => set((state) => ({
        multiConfig: typeof config === 'function' ? config(state.multiConfig) : config
    })),

    isRunning: false,
    results: null,
    resetResults: () => set({ results: null }),

    runBacktest: async () => {
        const { backtestMode, singleConfig, multiConfig } = get();

        set({ isRunning: true, results: null });

        try {
            if (backtestMode === 'single') {
                const request: any = {
                    symbol: singleConfig.symbol,
                    strategy_key: singleConfig.strategy,
                    parameters: singleConfig.params || {},
                    period: singleConfig.period,
                    interval: singleConfig.interval,
                    initial_capital: singleConfig.initialCapital
                };

                // Pass ml_model_id for ML strategies with a deployed model selected
                if (singleConfig.ml_model_id) {
                    request.ml_model_id = singleConfig.ml_model_id;
                }

                const response = await backtest.runSingle(request);

                if (response) {
                    const result: BacktestResult = response.result;
                    const formattedBenchmark = response.benchmark ? {
                        ...response.benchmark,
                        equity_curve: response.benchmark.equity_curve.map((bp: EquityCurvePoint) => ({
                            ...bp,
                            timestamp: formatDate(bp.timestamp)
                        }))
                    } : undefined;

                    set({
                        results: {
                            ...result,
                            equity_curve: response.equity_curve ? response.equity_curve.map((equityCurvePoint: EquityCurvePoint) => ({
                                timestamp: formatDate(equityCurvePoint.timestamp),
                                equity: equityCurvePoint.equity,
                                cash: equityCurvePoint.cash,
                                drawdown: equityCurvePoint.drawdown
                            })) : [],
                            benchmark: formattedBenchmark,
                            trades: (response.trades || []).map((trade: Trade) => ({
                                id: trade.id || Math.random(),
                                executed_at: formatDate(trade.executed_at),
                                symbol: trade.symbol,
                                order_type: trade.order_type,
                                strategy: trade.strategy,
                                quantity: trade.quantity,
                                price: trade.price,
                                commission: trade.commission,
                                total: trade.price * trade.quantity,
                                profit: trade.profit || 0,
                                profit_pct: trade.profit_pct,
                                status: trade.profit !== null ? 'closed' : 'open'
                            })),
                            price_data: response.price_data,
                        }
                    });
                }
            } else {
                // Multi-Asset Logic
                const isPairsStrategy = (key: string) => ['kalman_filter', 'pairs_trading', 'cointegration'].includes(key);
                const isPairs = isPairsStrategy(multiConfig.strategy);
                let strategyConfigs: any = {};

                if (isPairs) {
                    if (multiConfig.symbols.length !== 2) {
                        alert('Pairs trading strategies require exactly 2 symbols');
                        set({ isRunning: false });
                        return;
                    }
                    strategyConfigs = {
                        [multiConfig.symbols[0]]: {
                            strategy_key: multiConfig.strategy,
                            parameters: typeof multiConfig.params === 'object' && !Array.isArray(multiConfig.params)
                                ? { ...multiConfig.params, pair: multiConfig.symbols[1] }
                                : { pair: multiConfig.symbols[1] }
                        }
                    };
                } else {
                    if (multiConfig.strategyMode === 'same') {
                        strategyConfigs = multiConfig.symbols.reduce((acc, sym: string) => ({
                            ...acc,
                            [sym]: {
                                strategy_key: multiConfig.strategy,
                                parameters: typeof multiConfig.params === 'object' && !Array.isArray(multiConfig.params)
                                    ? multiConfig.params
                                    : {}
                            }
                        }), {});
                    } else if (multiConfig.strategyMode === 'different') {
                        strategyConfigs = multiConfig.symbols.reduce((acc, sym: string) => ({
                            ...acc,
                            [sym]: {
                                strategy_key: multiConfig.strategies[sym] || multiConfig.strategy,
                                parameters: typeof multiConfig.params === 'object' && !Array.isArray(multiConfig.params)
                                    ? multiConfig.params
                                    : {}
                            }
                        }), {});
                    } else {
                        // Portfolio mode default
                        strategyConfigs = multiConfig.symbols.reduce((acc, sym: string) => ({
                            ...acc,
                            [sym]: {
                                strategy_key: multiConfig.strategy,
                                parameters: typeof multiConfig.params === 'object' && !Array.isArray(multiConfig.params)
                                    ? multiConfig.params
                                    : {}
                            }
                        }), {});
                    }
                }

                const request: any = {
                    symbols: multiConfig.symbols,
                    strategy_configs: strategyConfigs,
                    period: multiConfig.period,
                    interval: multiConfig.interval,
                    initial_capital: multiConfig.initialCapital
                };

                if (multiConfig.allocationMethod === 'custom') {
                    request.allocation_method = 'custom';
                    request.custom_allocations = multiConfig.allocations;
                }

                const response = await backtest.runMulti(request);

                if (response) {
                    const result = response.result;
                    set({
                        results: {
                            ...result,
                            symbol_stats: result.symbol_stats || {},
                            equity_curve: response.equity_curve ? response.equity_curve.map((equityCurvePoint: EquityCurvePoint) => ({
                                timestamp: formatDate(equityCurvePoint.timestamp),
                                equity: equityCurvePoint.equity,
                                num_positions: 0,
                                cash: equityCurvePoint.cash,
                                drawdown: equityCurvePoint.drawdown
                            })) : [],
                            benchmark: response.benchmark ? {
                                ...response.benchmark,
                                equity_curve: response.benchmark.equity_curve.map((p: EquityCurvePoint) => ({
                                    ...p,
                                    timestamp: formatDate(p.timestamp)
                                }))
                            } : undefined,
                            trades: (response.trades || []).map((trade: Trade) => ({
                                id: trade.id || Math.random(),
                                executed_at: new Date(trade.executed_at).toLocaleString(),
                                symbol: trade.symbol,
                                order_type: trade.order_type,
                                strategy: trade.strategy,
                                quantity: trade.quantity,
                                price: trade.price,
                                commission: trade.commission,
                                total: trade.price * trade.quantity,
                                profit: trade.profit || 0,
                                profit_pct: trade.profit_pct,
                                status: trade.profit !== null ? 'closed' : 'open'
                            })),
                            // Polyfill missing advanced metrics for Multi-Asset
                            sortino_ratio: 0,
                            calmar_ratio: 0,
                            var_95: 0,
                            cvar_95: 0,
                            volatility: 0,
                            expectancy: 0,
                            total_commission: 0
                        }
                    });
                }
            }
        } catch (error) {
            console.error("Backtest failed", error);
        } finally {
            set({ isRunning: false });
        }
    }
}));
