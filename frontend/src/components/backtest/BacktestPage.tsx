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
import { useBacktestStore } from "@/store/useBacktestStore";

const isPairsStrategy = (strategyKey: string): boolean => {
    const pairsStrategies = ['kalman_filter', 'pairs_trading', 'cointegration'];
    return pairsStrategies.includes(strategyKey);
};

const BacktestPage = () => {
    const {
        backtestMode, setBacktestMode,
        singleConfig, setSingleConfig,
        multiConfig, setMultiConfig,
        results, isRunning, runBacktest
    } = useBacktestStore();

    const [strategiesList, setStrategiesList] = useState<Strategy[]>(strategies);

    // Deployment modal state
    const [showDeploymentModal, setShowDeploymentModal] = useState(false);
    const [deploymentBacktest, setDeploymentBacktest] = useState<BacktestResultToDeploy | null>(null);
    const [deploymentStatus, setDeploymentStatus] = useState<'idle' | 'deploying' | 'success' | 'error'>('idle');

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
                        parameterMetadata: Array.isArray(s.parameters) ? s.parameters : [],
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

    const addSymbol = () => {
        const symbol = multiConfig.symbolInput.trim().toUpperCase();
        if (symbol && !multiConfig.symbols.includes(symbol)) {
            setMultiConfig((prev) => ({
                ...prev,
                symbols: [...prev.symbols, symbol],
                symbolInput: '',
                allocations: {
                    ...prev.allocations,
                    [symbol]: 100 / (prev.symbols.length + 1)
                }
            }));
        }
    };

    const removeSymbol = (symbolToRemove: string) => {
        setMultiConfig((prev) => {
            const newSymbols = prev.symbols.filter(s => s !== symbolToRemove);
            const newAllocations = { ...prev.allocations };
            delete newAllocations[symbolToRemove];
            return {
                ...prev,
                symbols: newSymbols,
                allocations: newAllocations
            };
        });
    };

    // Handle "Go Live" button click
    const handleGoLive = () => {
        if (!results) return;

        const config = backtestMode === 'single' ? singleConfig : multiConfig;
        const backtestToDeploy: BacktestResultToDeploy = {
            id: `bt_${Date.now()}`,
            strategy: backtestMode === 'single' ? config.strategy : multiConfig.strategy,
            symbols: backtestMode === 'single' ? [singleConfig.symbol] : multiConfig.symbols,
            parameters: backtestMode === 'single' ? singleConfig.params : multiConfig.params,
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
                    config={singleConfig}
                    setConfig={setSingleConfig}
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
