'use client'
import React, { useEffect, useState } from 'react';
import { BarChart3, TrendingUp } from 'lucide-react';
import MultiAssetBacktest from "@/components/backtest/MultiAssetBacktest";
import SingleAssetBacktest from "@/components/backtest/SingleAssetBacktest";
import { strategies } from "@/components/strategies/Strategies";
import {
    BacktestResult,
    EquityCurvePoint,
    MultiAssetConfig,
    SingleAssetConfig,
    Strategy,
    StrategyInfo,
    Trade
} from "@/types/all_types";

import { backtest, strategy as strategyApi } from "@/utils/api";
import { formatDate } from "@/utils/formatters";

const BacktestPage = () => {
    const [backtestMode, setBacktestMode] = useState('single');
    const [strategiesList, setStrategiesList] = useState<Strategy[]>(strategies); // Initialize with static then fetch

    useEffect(() => {
        const fetchStrategies: () => Promise<void> = async () => {
            try {
                const response = await strategyApi.list();
                if (response) {
                    const mapped = response.map((s: StrategyInfo) => ({
                        ...s,
                        id: s.key,
                        complexity: 'Intermediate' as const,
                        // Convert parameters array to object format for StrategyParameterForm
                        // API returns: [{name: "short_window", default: 10}, ...]
                        // Form expects: {short_window: 10, ...}
                        parameters: Array.isArray(s.parameters)
                            ? s.parameters.reduce((acc: Record<string, any>, p: any) => {
                                acc[p.name] = p.default;
                                return acc;
                            }, {})
                            : s.parameters,
                        time_horizon: s.time_horizon,
                    }));
                    setStrategiesList(mapped);
                }
            } catch (e) {
                console.error("Failed to fetch strategies", e);
            }
        };
        fetchStrategies();
    }, []);

    const [singleAssetConfig, setSingleAssetConfig] = useState<SingleAssetConfig>({
        symbol: 'AAPL',
        period: '1y',
        interval: '1d',
        strategy: 'sma_crossover',
        initialCapital: 1000,
        maxPositionPct: 20,
        params: {}
    });

    const [multiConfig, setMultiConfig] = useState<MultiAssetConfig>({
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

    });
    const [results, setResults] = useState<BacktestResult | null>(null);
    const [isRunning, setIsRunning] = useState(false);


    const addSymbol = () => {
        const symbol = multiConfig.symbolInput.trim().toUpperCase();
        if (symbol && !multiConfig.symbols.includes(symbol)) {
            setMultiConfig({
                ...multiConfig,
                symbols: [...multiConfig.symbols, symbol],
                symbolInput: '',
                allocations: {
                    ...multiConfig.allocations,
                    [symbol]: 100 / (multiConfig.symbols.length + 1)
                }
            });
        }
    };

    const removeSymbol = (symbolToRemove: string) => {
        const newSymbols = multiConfig.symbols.filter(s => s !== symbolToRemove);
        const newAllocations = { ...multiConfig.allocations };
        delete newAllocations[symbolToRemove];
        setMultiConfig({
            ...multiConfig,
            symbols: newSymbols,
            allocations: newAllocations
        });
    };

    const runBacktest: () => Promise<void> = async () => {
        setIsRunning(true);
        setResults(null);

        try {
            if (backtestMode === 'single') {
                const request = {
                    symbol: singleAssetConfig.symbol,
                    strategy_key: singleAssetConfig.strategy,
                    parameters: singleAssetConfig.params || {},
                    period: singleAssetConfig.period,
                    interval: singleAssetConfig.interval,
                    initial_capital: singleAssetConfig.initialCapital
                };

                const response = await backtest.runSingle(request);

                if (response) {
                    const result: BacktestResult = response.result;

                    const formattedBenchmark = response.benchmark ? {
                        ...response.benchmark,
                        // We MUST format the benchmark's internal curve to match the strategy's curve
                        equity_curve: response.benchmark.equity_curve.map((bp: EquityCurvePoint) => ({
                            ...bp,
                            timestamp: formatDate(bp.timestamp)
                        }))
                    } : undefined;

                    setResults({
                        total_return: result.total_return,
                        total_return_pct: result.total_return_pct,
                        winning_trades: result.winning_trades,
                        losing_trades: result.losing_trades,
                        avg_profit: result.avg_profit,
                        win_rate: result.win_rate * 100,
                        sharpe_ratio: result.sharpe_ratio,
                        max_drawdown: result.max_drawdown * 100,
                        total_trades: result.total_trades,
                        final_equity: result.final_equity,
                        avg_win: result.avg_win,
                        avg_loss: result.avg_loss,
                        profit_factor: result.profit_factor,
                        initial_capital: result.initial_capital,
                        symbol_stats: result.symbol_stats,
                        equity_curve: response.equity_curve ? response.equity_curve.map((equityCurvePoint: EquityCurvePoint) => ({
                            timestamp: formatDate(equityCurvePoint.timestamp),
                            equity: equityCurvePoint.equity,
                            cash: equityCurvePoint.cash,
                            drawdown: equityCurvePoint.drawdown
                        })) : [],
                        benchmark: formattedBenchmark,
                        trades: (response.trades || []).map((trade: Trade) => ({
                            id: trade.id || Math.random(),
                            timestamp: new Date(trade.timestamp).toLocaleString(),
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
                    });
                }
            } else {
                // Multi-asset backtest
                // Build strategy_configs based on mode
                let strategyConfigs: Record<string, any> = {};

                if (multiConfig.strategyMode === 'same') {
                    // Apply same strategy to all symbols
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
                    // Use different strategies per symbol (from multiConfig.strategies)
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
                    // Portfolio mode - use default strategy for all
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

                const request: any = {
                    symbols: multiConfig.symbols,
                    strategy_configs: strategyConfigs,
                    period: multiConfig.period,
                    interval: multiConfig.interval,
                    initial_capital: multiConfig.initialCapital
                };

                // Only add allocation fields if using custom allocation
                if (multiConfig.allocationMethod === 'custom') {
                    request.allocation_method = 'custom';
                    request.custom_allocations = multiConfig.allocations;
                }

                console.log('Running multi-asset backtest with:', request);

                const response = await backtest.runMulti(request);

                console.log('📥 Response received:', response);

                if (response) {
                    const result: BacktestResult = response.result;
                    setResults({
                        total_return: result.total_return,
                        total_return_pct: result.total_return_pct,
                        winning_trades: result.winning_trades,
                        losing_trades: result.losing_trades,
                        avg_profit: result.avg_profit,
                        win_rate: result.win_rate * 100, // API probably returns 0-1
                        sharpe_ratio: result.sharpe_ratio,
                        max_drawdown: result.max_drawdown * 100, // percent
                        total_trades: result.total_trades,
                        final_equity: result.final_equity,
                        avg_win: result.avg_win,
                        avg_loss: result.avg_loss,
                        profit_factor: result.profit_factor,
                        initial_capital: result.initial_capital,
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
                            timestamp: new Date(trade.timestamp).toLocaleString(),
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
                    });

                    console.log('✅ Results set!');
                }
            }
        } catch (error) {
            console.error("Backtest failed", error);
            // Optionally set error state or notification
        } finally {
            setIsRunning(false);
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <div>
                    <h2 className="text-3xl font-bold text-slate-100 tracking-tight">
                        Backtesting <span className="text-slate-400 font-normal">Laboratory</span>
                    </h2>
                    <p className="text-slate-400 text-sm mt-1 font-medium">Historical strategy validation and
                        performance analysis</p>
                </div>

                <div className="flex items-center space-x-4">
                    <div
                        className="flex items-center space-x-1 bg-slate-800/60 border border-slate-700/50 rounded-xl p-1">
                        <button
                            onClick={() => setBacktestMode('single')}
                            className={`px-5 py-2.5 rounded-lg text-sm font-semibold transition-all ${backtestMode === 'single'
                                ? 'bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white shadow-lg'
                                : 'text-slate-400 hover:text-slate-200'
                                }`}
                        >
                            <BarChart3 size={16} className="inline mr-2" strokeWidth={2} />
                            Single Asset
                        </button>
                        <button
                            onClick={() => setBacktestMode('multi')}
                            className={`px-5 py-2.5 rounded-lg text-sm font-semibold transition-all ${backtestMode === 'multi'
                                ? 'bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white shadow-lg'
                                : 'text-slate-400 hover:text-slate-200'
                                }`}
                        >
                            <TrendingUp size={16} className="inline mr-2" strokeWidth={2} />
                            Multi-Asset
                        </button>
                    </div>
                </div>
            </div>

            {backtestMode === 'single' ? (
                <SingleAssetBacktest
                    config={singleAssetConfig}
                    setConfig={setSingleAssetConfig}
                    strategies={strategiesList}
                    runBacktest={runBacktest}
                    isRunning={isRunning}
                    results={results}
                />
            ) : (
                <MultiAssetBacktest
                    config={multiConfig}
                    setConfig={setMultiConfig}
                    strategies={strategiesList}
                    runBacktest={runBacktest}
                    isRunning={isRunning}
                    results={results}
                    addSymbol={addSymbol}
                    removeSymbol={removeSymbol}
                />
            )}
        </div>
    );
};

export default BacktestPage;
