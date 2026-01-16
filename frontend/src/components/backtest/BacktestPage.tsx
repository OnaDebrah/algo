'use client'
import React, { useState } from 'react';
import { BarChart3, TrendingUp } from 'lucide-react';
import MultiAssetBacktest from "@/components/backtest/MultiAssetBacktest";
import SingleAssetBacktest from "@/components/backtest/SingleAssetBacktest";
import { strategies } from "@/components/strategies/Strategies";
import { BacktestResult, MultiConfig } from "@/types/backtest";

import { backtest, strategy as strategyApi } from "@/utils/api";
import { useEffect } from "react";

const BacktestPage = () => {
    const [backtestMode, setBacktestMode] = useState('single');
    const [strategiesList, setStrategiesList] = useState<any[]>(strategies); // Initialize with static then fetch

    useEffect(() => {
        const fetchStrategies = async () => {
            try {
                const res = await strategyApi.list();
                if (res.data) {
                    // Map API strategy to UI strategy format if needed
                    // For now assuming compatible or just using what we have
                    // We might need to map 'key' to 'id' if they differ
                    const mapped = res.data.map((s: any) => ({
                        ...s,
                        id: s.key, // UI uses id, API uses key
                        complexity: 'Intermediate', // Default if missing
                    }));
                    setStrategiesList(mapped);
                }
            } catch (e) {
                console.error("Failed to fetch strategies", e);
            }
        };
        fetchStrategies();
    }, []);

    const [config, setConfig] = useState({
        symbol: 'AAPL',
        period: '1y',
        interval: '1d',
        strategy: 'sma_crossover',
        initialCapital: 100000,
        maxPositionPct: 20,
        params: {}
    });

    const [multiConfig, setMultiConfig] = useState<MultiConfig>({
        symbols: [],
        symbolInput: '',
        period: '1y',
        interval: '1d',
        strategyMode: 'same',
        strategy: 'sma_crossover',
        strategies: {},
        allocationMethod: 'equal',
        allocations: {},
        initialCapital: 100000,
        maxPositionPct: 20
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

    const runBacktest = async () => {
        setIsRunning(true);
        setResults(null);

        try {
            if (backtestMode === 'single') {
                const payload = {
                    symbol: config.symbol,
                    strategy_key: config.strategy,
                    parameters: config.params || {},
                    period: config.period,
                    interval: config.interval,
                    initial_capital: config.initialCapital
                };

                const response = await backtest.runSingle(payload);

                if (response.data) {
                    const r = response.data.result;
                    // Map API response to UI BacktestResult
                    // Ensure these fields exist in API response or handle defaults
                    setResults({
                        type: 'single',
                        total_return: r.total_return_pct, // percent expected
                        win_rate: r.win_rate * 100, // API probably returns 0-1
                        sharpe_ratio: r.sharpe_ratio,
                        max_drawdown: r.max_drawdown * 100, // percent
                        total_trades: r.total_trades,
                        final_equity: r.final_equity,
                        equity_curve: response.data.equity_curve ? response.data.equity_curve.map((p: any) => ({
                            timestamp: new Date(p.timestamp).toLocaleDateString(), // Format timestamp
                            equity: p.equity
                        })) : [],

                        trades: (response.data.trades || []).map((t: any) => ({
                            id: t.id || Math.random(),
                            date: new Date(t.timestamp).toLocaleString(),
                            symbol: t.symbol,
                            type: t.order_type,
                            strategy: t.strategy,
                            quantity: t.quantity,
                            price: t.price,
                            total: t.price * t.quantity,
                            pnl: t.profit || 0,
                            status: t.profit !== null ? 'closed' : 'open'
                        }))
                    });
                }
            } else {
                const payload = {
                    symbols: multiConfig.symbols,
                    strategy_configs: multiConfig.strategyMode === 'same'
                        ? multiConfig.symbols.reduce((acc: any, sym) => ({ ...acc, [sym]: { strategy_key: multiConfig.strategy, parameters: {} } }), {})
                        : {}, // Todo: handle diverse strategies if UI supports it fully
                    allocation_method: multiConfig.allocationMethod,
                    custom_allocations: multiConfig.allocationMethod === 'custom' ? multiConfig.allocations : null,
                    period: multiConfig.period,
                    interval: multiConfig.interval,
                    initial_capital: multiConfig.initialCapital
                };

                const response = await backtest.runMulti(payload);
                if (response.data) {
                    const r = response.data.result;
                    setResults({
                        type: 'multi',
                        total_return: r.total_return_pct,
                        win_rate: r.win_rate * 100,
                        sharpe_ratio: r.sharpe_ratio,
                        max_drawdown: r.max_drawdown * 100,
                        total_trades: r.total_trades,
                        num_symbols: r.num_symbols,
                        avg_profit: r.avg_profit,
                        final_equity: r.final_equity,
                        symbol_stats: r.symbol_stats, // Check structure matching
                        equity_curve: response.data.equity_curve ? response.data.equity_curve.map((p: any) => ({
                            timestamp: new Date(p.timestamp).toLocaleDateString(),
                            equity: p.equity,
                            num_positions: 0 // API might not return this yet
                        })) : [],
                        trades: (response.data.trades || []).map((t: any) => ({
                            id: t.id || Math.random(),
                            date: new Date(t.timestamp).toLocaleString(),
                            symbol: t.symbol,
                            type: t.order_type,
                            strategy: t.strategy,
                            quantity: t.quantity,
                            price: t.price,
                            total: t.price * t.quantity,
                            pnl: t.profit || 0,
                            status: t.profit !== null ? 'closed' : 'open'
                        })),

                    });
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
                    <p className="text-slate-400 text-sm mt-1 font-medium">Historical strategy validation and performance analysis</p>
                </div>

                <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-1 bg-slate-800/60 border border-slate-700/50 rounded-xl p-1">
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
                    config={config}
                    setConfig={setConfig}
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