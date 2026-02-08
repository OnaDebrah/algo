// 'use client'
// import React, { useEffect, useState } from 'react';
// import { BarChart3, TrendingUp } from 'lucide-react';
// import MultiAssetBacktest from "@/components/backtest/MultiAssetBacktest";
// import SingleAssetBacktest from "@/components/backtest/SingleAssetBacktest";
// import { strategies } from "@/components/strategies/Strategies";
// import {
//     BacktestResult,
//     EquityCurvePoint,
//     MultiAssetConfig,
//     SingleAssetConfig,
//     Strategy,
//     StrategyInfo,
//     Trade
// } from "@/types/all_types";
//
// import { backtest, strategy as strategyApi } from "@/utils/api";
// import { formatDate } from "@/utils/formatters";
//
// const isPairsStrategy = (strategyKey: string): boolean => {
//     const pairsStrategies = ['kalman_filter', 'pairs_trading', 'cointegration'];
//     return pairsStrategies.includes(strategyKey);
// };
//
// const BacktestPage = () => {
//     const [backtestMode, setBacktestMode] = useState('single');
//     const [strategiesList, setStrategiesList] = useState<Strategy[]>(strategies);
//
//     useEffect(() => {
//         const fetchStrategies: () => Promise<void> = async () => {
//             try {
//                 const response = await strategyApi.list();
//                 if (response) {
//                     const mapped = response.map((s: StrategyInfo) => ({
//                         ...s,
//                         id: s.key,
//                         complexity: s.complexity as Strategy["complexity"],
//                         parameters: Array.isArray(s.parameters)
//                             ? s.parameters.reduce((acc: Record<string, any>, p: any) => {
//                                 acc[p.name] = p.default;
//                                 return acc;
//                             }, {})
//                             : s.parameters,
//                         time_horizon: s.time_horizon,
//                     }));
//                     setStrategiesList(mapped);
//                 }
//             } catch (e) {
//                 console.error("Failed to fetch strategies", e);
//             }
//         };
//         fetchStrategies();
//     }, []);
//
//     const [singleAssetConfig, setSingleAssetConfig] = useState<SingleAssetConfig>({
//         symbol: 'AAPL',
//         period: '1y',
//         interval: '1d',
//         strategy: 'sma_crossover',
//         initialCapital: 1000,
//         maxPositionPct: 20,
//         params: {}
//     });
//
//     const [multiConfig, setMultiConfig] = useState<MultiAssetConfig>({
//         symbols: [],
//         symbolInput: '',
//         period: '1y',
//         interval: '1d',
//         strategyMode: 'same',
//         params: {},
//         strategy: 'sma_crossover',
//         strategies: {},
//         allocationMethod: 'equal',
//         allocations: {},
//         initialCapital: 1000,
//         maxPositionPct: 20,
//         riskLevel: 'Intermediate'
//     });
//
//     const [results, setResults] = useState<BacktestResult | null>(null);
//     const [isRunning, setIsRunning] = useState(false);
//
//     const addSymbol = () => {
//         const symbol = multiConfig.symbolInput.trim().toUpperCase();
//         if (symbol && !multiConfig.symbols.includes(symbol)) {
//             setMultiConfig({
//                 ...multiConfig,
//                 symbols: [...multiConfig.symbols, symbol],
//                 symbolInput: '',
//                 allocations: {
//                     ...multiConfig.allocations,
//                     [symbol]: 100 / (multiConfig.symbols.length + 1)
//                 }
//             });
//         }
//     };
//
//     const removeSymbol = (symbolToRemove: string) => {
//         const newSymbols = multiConfig.symbols.filter(s => s !== symbolToRemove);
//         const newAllocations = { ...multiConfig.allocations };
//         delete newAllocations[symbolToRemove];
//         setMultiConfig({
//             ...multiConfig,
//             symbols: newSymbols,
//             allocations: newAllocations
//         });
//     };
//
//     const runBacktest: () => Promise<void> = async () => {
//         setIsRunning(true);
//         setResults(null);
//
//         try {
//             if (backtestMode === 'single') {
//                 const request = {
//                     symbol: singleAssetConfig.symbol,
//                     strategy_key: singleAssetConfig.strategy,
//                     parameters: singleAssetConfig.params || {},
//                     period: singleAssetConfig.period,
//                     interval: singleAssetConfig.interval,
//                     initial_capital: singleAssetConfig.initialCapital
//                 };
//
//                 const response = await backtest.runSingle(request);
//
//                 if (response) {
//                     const result: BacktestResult = response.result;
//
//                     const formattedBenchmark = response.benchmark ? {
//                         ...response.benchmark,
//                         equity_curve: response.benchmark.equity_curve.map((bp: EquityCurvePoint) => ({
//                             ...bp,
//                             timestamp: formatDate(bp.timestamp)
//                         }))
//                     } : undefined;
//
//                     setResults({
//                         total_return: result.total_return,
//                         total_return_pct: result.total_return_pct,
//                         winning_trades: result.winning_trades,
//                         losing_trades: result.losing_trades,
//                         avg_profit: result.avg_profit,
//                         win_rate: result.win_rate,
//                         sharpe_ratio: result.sharpe_ratio,
//                         max_drawdown: result.max_drawdown,
//                         total_trades: result.total_trades,
//                         final_equity: result.final_equity,
//                         avg_win: result.avg_win,
//                         avg_loss: result.avg_loss,
//                         profit_factor: result.profit_factor,
//                         initial_capital: result.initial_capital,
//                         symbol_stats: result.symbol_stats,
//                         equity_curve: response.equity_curve ? response.equity_curve.map((equityCurvePoint: EquityCurvePoint) => ({
//                             timestamp: formatDate(equityCurvePoint.timestamp),
//                             equity: equityCurvePoint.equity,
//                             cash: equityCurvePoint.cash,
//                             drawdown: equityCurvePoint.drawdown
//                         })) : [],
//                         benchmark: formattedBenchmark,
//                         trades: (response.trades || []).map((trade: Trade) => ({
//                             id: trade.id || Math.random(),
//                             executed_at: formatDate(trade.executed_at),
//                             symbol: trade.symbol,
//                             order_type: trade.order_type,
//                             strategy: trade.strategy,
//                             quantity: trade.quantity,
//                             price: trade.price,
//                             commission: trade.commission,
//                             total: trade.price * trade.quantity,
//                             profit: trade.profit || 0,
//                             profit_pct: trade.profit_pct,
//                             status: trade.profit !== null ? 'closed' : 'open'
//                         })),
//                         price_data: response.price_data,
//                     });
//                 }
//             } else {
//
//                 const isPairs = isPairsStrategy(multiConfig.strategy);
//
//                 let strategyConfigs: Record<string, any> = {};
//
//                 if (isPairs) {
//                     console.log('🔬 Pairs Trading Mode Detected');
//
//                     const asset1 = multiConfig.params.asset_1 || multiConfig.symbols[0];
//                     const asset2 = multiConfig.params.asset_2 || multiConfig.symbols[1];
//
//                     if (multiConfig.symbols.length !== 2 ||
//                         !multiConfig.symbols.includes(asset1) ||
//                         !multiConfig.symbols.includes(asset2)) {
//                         console.warn('Adjusting symbols to match pair:', asset1, asset2);
//                         setMultiConfig(prev => ({
//                             ...prev,
//                             symbols: [asset1, asset2]
//                         }));
//                     }
//
//                     const pairsConfig = {
//                         strategy_key: multiConfig.strategy,
//                         parameters: {
//                             ...multiConfig.params,
//                             asset_1: asset1,
//                             asset_2: asset2
//                         }
//                     };
//
//                     strategyConfigs = {
//                         [asset1]: pairsConfig,
//                         [asset2]: pairsConfig
//                     };
//
//                     console.log('📋 Pairs Strategy Config:', strategyConfigs);
//
//                 } else {
//                     if (multiConfig.strategyMode === 'same') {
//                         strategyConfigs = multiConfig.symbols.reduce((acc, sym: string) => ({
//                             ...acc,
//                             [sym]: {
//                                 strategy_key: multiConfig.strategy,
//                                 parameters: typeof multiConfig.params === 'object' && !Array.isArray(multiConfig.params)
//                                     ? multiConfig.params
//                                     : {}
//                             }
//                         }), {});
//                     } else if (multiConfig.strategyMode === 'different') {
//                         // Use different strategies per symbol
//                         strategyConfigs = multiConfig.symbols.reduce((acc, sym: string) => ({
//                             ...acc,
//                             [sym]: {
//                                 strategy_key: multiConfig.strategies[sym] || multiConfig.strategy,
//                                 parameters: typeof multiConfig.params === 'object' && !Array.isArray(multiConfig.params)
//                                     ? multiConfig.params
//                                     : {}
//                             }
//                         }), {});
//                     } else {
//                         // Portfolio mode
//                         strategyConfigs = multiConfig.symbols.reduce((acc, sym: string) => ({
//                             ...acc,
//                             [sym]: {
//                                 strategy_key: multiConfig.strategy,
//                                 parameters: typeof multiConfig.params === 'object' && !Array.isArray(multiConfig.params)
//                                     ? multiConfig.params
//                                     : {}
//                             }
//                         }), {});
//                     }
//                 }
//
//                 const request: any = {
//                     symbols: multiConfig.symbols,
//                     strategy_configs: strategyConfigs,
//                     period: multiConfig.period,
//                     interval: multiConfig.interval,
//                     initial_capital: multiConfig.initialCapital
//                 };
//
//                 // Only add allocation fields if using custom allocation
//                 if (multiConfig.allocationMethod === 'custom') {
//                     request.allocation_method = 'custom';
//                     request.custom_allocations = multiConfig.allocations;
//                 }
//
//                 console.log('🚀 Running multi-asset backtest with:', request);
//
//                 const response = await backtest.runMulti(request);
//
//                 console.log('📥 Response received:', response);
//
//                 if (response) {
//                     const result: BacktestResult = response.result;
//                     setResults({
//                         total_return: result.total_return,
//                         total_return_pct: result.total_return_pct,
//                         winning_trades: result.winning_trades,
//                         losing_trades: result.losing_trades,
//                         avg_profit: result.avg_profit,
//                         win_rate: result.win_rate,
//                         sharpe_ratio: result.sharpe_ratio,
//                         max_drawdown: result.max_drawdown,
//                         total_trades: result.total_trades,
//                         final_equity: result.final_equity,
//                         avg_win: result.avg_win,
//                         avg_loss: result.avg_loss,
//                         profit_factor: result.profit_factor,
//                         initial_capital: result.initial_capital,
//                         symbol_stats: result.symbol_stats || {},
//
//                         equity_curve: response.equity_curve ? response.equity_curve.map((equityCurvePoint: EquityCurvePoint) => ({
//                             timestamp: formatDate(equityCurvePoint.timestamp),
//                             equity: equityCurvePoint.equity,
//                             num_positions: 0,
//                             cash: equityCurvePoint.cash,
//                             drawdown: equityCurvePoint.drawdown
//                         })) : [],
//
//                         benchmark: response.benchmark ? {
//                             ...response.benchmark,
//                             equity_curve: response.benchmark.equity_curve.map((p: EquityCurvePoint) => ({
//                                 ...p,
//                                 timestamp: formatDate(p.timestamp)
//                             }))
//                         } : undefined,
//
//                         trades: (response.trades || []).map((trade: Trade) => ({
//                             id: trade.id || Math.random(),
//                             executed_at: new Date(trade.executed_at).toLocaleString(),
//                             symbol: trade.symbol,
//                             order_type: trade.order_type,
//                             strategy: trade.strategy,
//                             quantity: trade.quantity,
//                             price: trade.price,
//                             commission: trade.commission,
//                             total: trade.price * trade.quantity,
//                             profit: trade.profit || 0,
//                             profit_pct: trade.profit_pct,
//                             status: trade.profit !== null ? 'closed' : 'open'
//                         })),
//                     });
//
//                     console.log('✅ Results set!');
//                 }
//             }
//         } catch (error) {
//             console.error("Backtest failed", error);
//         } finally {
//             setIsRunning(false);
//         }
//     };
//
//     return (
//         <div className="space-y-6">
//             <div className="flex justify-between items-center">
//                 <div>
//                     <h2 className="text-3xl font-bold text-slate-100 tracking-tight">
//                         Backtesting <span className="text-slate-400 font-normal">Laboratory</span>
//                     </h2>
//                     <p className="text-slate-400 text-sm mt-1 font-medium">
//                         Historical strategy validation and performance analysis
//                     </p>
//                 </div>
//
//                 <div className="flex items-center space-x-4">
//                     <div className="flex items-center space-x-1 bg-slate-800/60 border border-slate-700/50 rounded-xl p-1">
//                         <button
//                             onClick={() => setBacktestMode('single')}
//                             className={`px-5 py-2.5 rounded-lg text-sm font-semibold transition-all ${
//                                 backtestMode === 'single'
//                                     ? 'bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white shadow-lg'
//                                     : 'text-slate-400 hover:text-slate-200'
//                             }`}
//                         >
//                             <BarChart3 size={16} className="inline mr-2" strokeWidth={2} />
//                             Single Asset
//                         </button>
//                         <button
//                             onClick={() => setBacktestMode('multi')}
//                             className={`px-5 py-2.5 rounded-lg text-sm font-semibold transition-all ${
//                                 backtestMode === 'multi'
//                                     ? 'bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white shadow-lg'
//                                     : 'text-slate-400 hover:text-slate-200'
//                             }`}
//                         >
//                             <TrendingUp size={16} className="inline mr-2" strokeWidth={2} />
//                             Multi-Asset
//                         </button>
//                     </div>
//                 </div>
//             </div>
//
//             {backtestMode === 'single' ? (
//                 <SingleAssetBacktest
//                     config={singleAssetConfig}
//                     setConfig={setSingleAssetConfig}
//                     strategies={strategiesList}
//                     runBacktest={runBacktest}
//                     isRunning={isRunning}
//                     results={results}
//                 />
//             ) : (
//                 <MultiAssetBacktest
//                     config={multiConfig}
//                     setConfig={setMultiConfig}
//                     strategies={strategiesList}
//                     runBacktest={runBacktest}
//                     isRunning={isRunning}
//                     results={results}
//                     addSymbol={addSymbol}
//                     removeSymbol={removeSymbol}
//                 />
//             )}
//         </div>
//     );
// };
//
// export default BacktestPage;

'use client'
import React, { useEffect, useState } from 'react';
import { BarChart3, TrendingUp, Rocket, CheckCircle, AlertCircle } from 'lucide-react';
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
    Trade,
    DeploymentConfig,
    BacktestResultToDeploy, UserSettings
} from "@/types/all_types";

import { backtest, strategy as strategyApi, live, settings as settingsApi } from "@/utils/api";
import { formatDate } from "@/utils/formatters";
import DeploymentModal from "@/components/strategies/DeploymentModel";

const isPairsStrategy = (strategyKey: string): boolean => {
    const pairsStrategies = ['kalman_filter', 'pairs_trading', 'cointegration'];
    return pairsStrategies.includes(strategyKey);
};

const BacktestPage = () => {
    const [backtestMode, setBacktestMode] = useState('single');
    const [strategiesList, setStrategiesList] = useState<Strategy[]>(strategies);

    // Deployment modal state
    const [showDeploymentModal, setShowDeploymentModal] = useState(false);
    const [deploymentBacktest, setDeploymentBacktest] = useState<BacktestResultToDeploy | null>(null);
    const [deploymentStatus, setDeploymentStatus] = useState<'idle' | 'deploying' | 'success' | 'error'>('idle');
    const [settings, setSettings] = useState<UserSettings | null>(null);
    const [dataSource, setDataSource] = useState<string | undefined>(undefined);
    const [slippage, setSlippage] = useState<number | undefined>(undefined);
    const [commission, setCommission] = useState<number | undefined>(undefined);
    const [initialCapital, setInitialCapital] = useState<number | undefined>(undefined);

    useEffect(() => {
        const loadSettings = async () => {
            const userSettings = await settingsApi.get();
            setSettings(userSettings);

            // Use settings as defaults
            setDataSource(userSettings.backtest.data_source);
            setSlippage(userSettings.backtest.slippage);
            setCommission(userSettings.backtest.commission);
            setInitialCapital(userSettings.backtest.initial_capital);
        };

        loadSettings();
    }, []);

    useEffect(() => {
        const fetchStrategies: () => Promise<void> = async () => {
            try {
                const response = await strategyApi.list();
                if (response) {
                    const mapped = response.map((s: StrategyInfo) => ({
                        ...s,
                        id: s.key,
                        complexity: s.complexity as Strategy["complexity"],
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
                        win_rate: result.win_rate,
                        sharpe_ratio: result.sharpe_ratio,
                        max_drawdown: result.max_drawdown,
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
                    });
                }
            } else {
                const isPairs = isPairsStrategy(multiConfig.strategy);

                let strategyConfigs: any = {};
                if (isPairs) {
                    if (multiConfig.symbols.length !== 2) {
                        alert('Pairs trading strategies require exactly 2 symbols');
                        setIsRunning(false);
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
                    const result: BacktestResult = response.result;
                    setResults({
                        total_return: result.total_return,
                        total_return_pct: result.total_return_pct,
                        winning_trades: result.winning_trades,
                        losing_trades: result.losing_trades,
                        avg_profit: result.avg_profit,
                        win_rate: result.win_rate,
                        sharpe_ratio: result.sharpe_ratio,
                        max_drawdown: result.max_drawdown,
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
                    });
                }
            }
        } catch (error) {
            console.error("Backtest failed", error);
        } finally {
            setIsRunning(false);
        }
    };

    // Handle "Go Live" button click
    const handleGoLive = () => {
        if (!results) return;

        const config = backtestMode === 'single' ? singleAssetConfig : multiConfig;
        const backtestToDeploy: BacktestResultToDeploy = {
            id: `bt_${Date.now()}`,
            strategy: backtestMode === 'single' ? config.strategy : multiConfig.strategy,
            symbols: backtestMode === 'single' ? [singleAssetConfig.symbol] : multiConfig.symbols,
            parameters: backtestMode === 'single' ? singleAssetConfig.params : multiConfig.params,
            total_return_pct: results.total_return_pct,
            sharpe_ratio: results.sharpe_ratio,
            max_drawdown: results.max_drawdown,
            total_trades: results.total_trades,
            win_rate: results.win_rate,
            initial_capital: results.initial_capital,
            period: config.period,
            interval: config.interval
        };

        setDeploymentBacktest(backtestToDeploy);
        setShowDeploymentModal(true);
    };

    // Handle actual deployment
    const handleDeploy = async (deployConfig: DeploymentConfig) => {
        setDeploymentStatus('deploying');

        try {
            await live.deploy(deployConfig);
            setDeploymentStatus('success');

            // Auto-close after 2 seconds on success
            setTimeout(() => {
                setShowDeploymentModal(false);
                setDeploymentStatus('idle');
            }, 2000);
        } catch (error) {
            console.error("Deployment failed:", error);
            setDeploymentStatus('error');
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <div>
                    <h2 className="text-3xl font-bold text-slate-100 tracking-tight">
                        Backtesting <span className="text-slate-400 font-normal">Laboratory</span>
                    </h2>
                    <p className="text-slate-400 text-sm mt-1 font-medium">
                        Historical strategy validation and performance analysis
                    </p>
                </div>

                <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-1 bg-slate-800/60 border border-slate-700/50 rounded-xl p-1">
                        <button
                            onClick={() => setBacktestMode('single')}
                            className={`px-5 py-2.5 rounded-lg text-sm font-semibold transition-all ${
                                backtestMode === 'single'
                                    ? 'bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white shadow-lg'
                                    : 'text-slate-400 hover:text-slate-200'
                            }`}
                        >
                            <BarChart3 size={16} className="inline mr-2" strokeWidth={2} />
                            Single Asset
                        </button>
                        <button
                            onClick={() => setBacktestMode('multi')}
                            className={`px-5 py-2.5 rounded-lg text-sm font-semibold transition-all ${
                                backtestMode === 'multi'
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

            {/* Go Live Button - Shows when results exist */}
            {results && (
                <div className="fixed bottom-8 right-8 z-50">
                    <button
                        onClick={handleGoLive}
                        className="group relative px-8 py-4 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 rounded-2xl text-white font-bold text-lg shadow-2xl shadow-emerald-500/50 transition-all transform hover:scale-105 flex items-center gap-3"
                    >
                        <Rocket size={24} className="group-hover:animate-bounce" />
                        Go Live with This Strategy
                        <div className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 rounded-full flex items-center justify-center">
                            <span className="text-xs font-black">!</span>
                        </div>
                    </button>
                </div>
            )}

            {/* Deployment Status Indicator */}
            {deploymentStatus === 'success' && (
                <div className="fixed top-8 right-8 z-50 bg-emerald-500 text-white px-6 py-4 rounded-xl shadow-lg flex items-center gap-3 animate-in slide-in-from-right">
                    <CheckCircle size={24} />
                    <div>
                        <div className="font-bold">Strategy Deployed Successfully!</div>
                        <div className="text-sm opacity-90">Check Live Execution page for details</div>
                    </div>
                </div>
            )}

            {deploymentStatus === 'error' && (
                <div className="fixed top-8 right-8 z-50 bg-red-500 text-white px-6 py-4 rounded-xl shadow-lg flex items-center gap-3 animate-in slide-in-from-right">
                    <AlertCircle size={24} />
                    <div>
                        <div className="font-bold">Deployment Failed</div>
                        <div className="text-sm opacity-90">Please try again or check settings</div>
                    </div>
                </div>
            )}

            {/* Deployment Modal */}
            {showDeploymentModal && deploymentBacktest && (
                <DeploymentModal
                    backtest={deploymentBacktest}
                    onClose={() => {
                        setShowDeploymentModal(false);
                        setDeploymentStatus('idle');
                    }}
                    onDeploy={handleDeploy}
                />
            )}
        </div>
    );
};

export default BacktestPage;
