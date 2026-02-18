'use client'
import React, {useEffect, useState} from 'react';
import {AlertCircle, BarChart3, CheckCircle, Rocket, TrendingUp, Zap} from 'lucide-react';
import MultiAssetBacktest from "@/components/backtest/MultiAssetBacktest";
import SingleAssetBacktest from "@/components/backtest/SingleAssetBacktest";
import WalkForwardAnalysis from "@/components/backtest/WalkForwardAnalysis";
import {strategies} from "@/components/strategies/Strategies";
import {BacktestResultToDeploy, DeploymentConfig, MultiAssetConfig, Strategy, StrategyInfo} from "@/types/all_types";

import {live, strategy as strategyApi} from "@/utils/api";
import DeploymentModal from "@/components/strategies/DeploymentModel";
import {useBacktestStore} from "@/store/useBacktestStore";

const BacktestPage = () => {
    const {
        backtestMode, setBacktestMode,
        singleConfig, setSingleConfig,
        multiConfig, setMultiConfig,
        singleResults, multiResults,
        results, isRunning, runBacktest
    } = useBacktestStore();

    const [strategiesList, setStrategiesList] = useState<Strategy[]>(strategies);

    const [showDeploymentModal, setShowDeploymentModal] = useState(false);
    const [deploymentBacktest, setDeploymentBacktest] = useState<BacktestResultToDeploy | null>(null);
    const [deploymentStatus, setDeploymentStatus] = useState<'idle' | 'deploying' | 'success' | 'error'>('idle');

    useEffect(() => {
        const fetchStrategies: () => Promise<void> = async () => {
            try {
                const response = await strategyApi.list(backtestMode);
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
                        backtest_mode: s.backtest_mode,
                    }));
                    setStrategiesList(mapped);
                }
            } catch (e) {
                console.error("Failed to fetch strategies", e);
                // Fallback: filter the hardcoded list client-side
                const filtered = strategies.filter(s =>
                    !s.backtest_mode || s.backtest_mode === backtestMode || s.backtest_mode === 'both'
                );
                setStrategiesList(filtered);
            }
        };
        fetchStrategies().then();
    }, [backtestMode]);

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
        setMultiConfig((prev: MultiAssetConfig) => {
            const newSymbols = prev.symbols.filter((s: string) => s !== symbolToRemove);
            const newAllocations = { ...prev.allocations };
            delete newAllocations[symbolToRemove];
            return {
                ...prev,
                symbols: newSymbols,
                allocations: newAllocations
            };
        });
    };

    const activeResults = backtestMode === 'single' ? singleResults : multiResults;

    const handleGoLive = () => {
        if (!activeResults) return;

        const config = backtestMode === 'single' ? singleConfig : multiConfig;
        const backtestToDeploy: BacktestResultToDeploy = {
            id: `bt_${Date.now()}`,
            strategy: backtestMode === 'single' ? config.strategy : multiConfig.strategy,
            symbols: backtestMode === 'single' ? [singleConfig.symbol] : multiConfig.symbols,
            parameters: backtestMode === 'single' ? singleConfig.params : multiConfig.params,
            total_return_pct: activeResults.total_return_pct,
            sharpe_ratio: activeResults.sharpe_ratio,
            max_drawdown: activeResults.max_drawdown,
            total_trades: activeResults.total_trades,
            win_rate: activeResults.win_rate,
            initial_capital: activeResults.initial_capital,
            period: config.period,
            interval: config.interval
        };

        setDeploymentBacktest(backtestToDeploy);
        setShowDeploymentModal(true);
    };

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
                        <button
                            onClick={() => setBacktestMode('walkforward' as any)}
                            className={`px-5 py-2.5 rounded-lg text-sm font-semibold transition-all ${backtestMode === ('walkforward' as any)
                                ? 'bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white shadow-lg'
                                : 'text-slate-400 hover:text-slate-200'
                                }`}
                        >
                            <Zap size={16} className="inline mr-2" strokeWidth={2} />
                            Walk-Forward
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
                    results={singleResults}
                />
            ) : backtestMode === 'multi' ? (
                <MultiAssetBacktest
                    config={multiConfig}
                    setConfig={setMultiConfig}
                    strategies={strategiesList}
                    runBacktest={runBacktest}
                    isRunning={isRunning}
                    results={multiResults}
                    addSymbol={addSymbol}
                    removeSymbol={removeSymbol}
                />
            ) : (
                <WalkForwardAnalysis
                    strategies={strategiesList}
                />
            )}

            {/* Go Live Button - Shows when results exist for active mode */}
            {activeResults && activeResults.sharpe_ratio >= 1 && activeResults.total_return >= 10.0 && (
                <div className="fixed bottom-8 right-8 z-50">
                    <button
                        onClick={handleGoLive}
                        className="group relative px-8 py-4 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 rounded-2xl text-white font-bold text-lg shadow-2xl shadow-emerald-500/50 transition-all transform hover:scale-105 flex items-center gap-3"
                    >
                        <Rocket size={24} className="group-hover:animate-bounce" />
                        Go Live with This Strategy
                        <div
                            className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 rounded-full flex items-center justify-center">
                            <span className="text-xs font-black">!</span>
                        </div>
                    </button>
                </div>
            )}

            {/* Deployment Status Indicator */}
            {deploymentStatus === 'success' && (
                <div
                    className="fixed top-8 right-8 z-50 bg-emerald-500 text-white px-6 py-4 rounded-xl shadow-lg flex items-center gap-3 animate-in slide-in-from-right">
                    <CheckCircle size={24} />
                    <div>
                        <div className="font-bold">Strategy Deployed Successfully!</div>
                        <div className="text-sm opacity-90">Check Live Execution page for details</div>
                    </div>
                </div>
            )}

            {deploymentStatus === 'error' && (
                <div
                    className="fixed top-8 right-8 z-50 bg-red-500 text-white px-6 py-4 rounded-xl shadow-lg flex items-center gap-3 animate-in slide-in-from-right">
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
