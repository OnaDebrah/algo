'use client'
import React, {useState, useEffect} from 'react';
import {Loader2} from "lucide-react";
import {MLForecast, OptionLeg, BacktestConfig, StrategyTemplate, ChainRequest, ChainResponse} from '@/types/all_types';
import Header from './Header';
import MLForecastPanel from './MLForecastPanel';
import ChainTab from './tabs/ChainTab';
import BuilderTab from './tabs/BuilderTab';
import CompareTab from './tabs/CompareTab';
import BacktestTab from './tabs/BacktestTab';
import VolatilityTab from './tabs/VolatilityTab';
import RiskTab from './tabs/RiskTab';
import {mockAPI} from "@/components/optionsdesk/contants/mockedApi";
import Tabs from "@/components/optionsdesk/tabs/Tabs";
import {options} from "@/utils/api";
import {AxiosResponse} from "axios";

const OptionsDesk = () => {
    const [selectedSymbol, setSelectedSymbol] = useState('SPY');
    const [selectedExpiry, setSelectedExpiry] = useState('');
    const [currentPrice, setCurrentPrice] = useState(450);
    const [optionsChain, setOptionsChain] = useState<ChainResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [activeTab, setActiveTab] = useState<'chain' | 'builder' | 'compare' | 'backtest' | 'volatility' | 'risk' | 'ml'>('chain');
    const [mlForecast, setMlForecast] = useState<MLForecast | null>(null);
    const [customLegs, setCustomLegs] = useState<OptionLeg[]>([]);
    const [selectedStrategies, setSelectedStrategies] = useState<StrategyTemplate[]>([]);
    const [strategyAnalysis, setStrategyAnalysis] = useState<any[]>([]);
    const [profitLossData, setProfitLossData] = useState<any[]>([]);
    const [greeksChartData, setGreeksChartData] = useState<any[]>([]);
    const [monteCarloDistribution, setMonteCarloDistribution] = useState<any>(null);
    const [riskMetrics, setRiskMetrics] = useState<any>(null);
    const [portfolioStats, setPortfolioStats] = useState<any>(null);
    const [strikeOptimizer, setStrikeOptimizer] = useState<any>(null);
    const [backtestResults, setBacktestResults] = useState<any>(null);
    const [equityData, setEquityData] = useState<any[]>([]);
    const [recentTrades, setRecentTrades] = useState<any[]>([]);

    const [newLeg, setNewLeg] = useState<Partial<OptionLeg>>({
        type: 'call',
        position: 'long',
        strike: 450,
        quantity: 1,
        expiration: '2024-03-15'
    });

    const [backtestConfig, setBacktestConfig] = useState<BacktestConfig>({
        symbol: "SPY",
        strategy_type: "",
        initial_capital: 10000,
        risk_free_rate: 0.04,
        start_date: "2024-01-01",
        end_date: "2024-12-31",
        entry_rules: {},
        exit_rules: {}
    });

    useEffect(() => {
        if (selectedSymbol) {
            fetchExpirations();
        }
    }, [selectedSymbol]);


    useEffect(() => {
        if (selectedExpiry && selectedSymbol) {
            fetchOptionsChain();
            fetchMLForecast();
        }
    }, [selectedExpiry]);

    const fetchExpirations = async () => {
        setIsLoading(true);
        try {
            const request: ChainRequest = {
                symbol: selectedSymbol,
                expiration: null
            }
            const response: AxiosResponse<ChainResponse> = await options.getChain(request);

            if (response && response.expiration_dates && response.expiration_dates.length > 0) {
                setOptionsChain(prevData => ({
                    ...prevData,
                    symbol: response.symbol,
                    current_price: response.current_price,
                    expiration_dates: response.expiration_dates,
                    calls: prevData?.calls || [],
                    puts: prevData?.puts || []
                } as ChainResponse));

                setSelectedExpiry(response.expiration_dates[0]);
            }
        } catch (err) {
            console.error("❌ Failed to fetch expirations:", err);
        } finally {
            setIsLoading(false);
        }
    };

    const fetchOptionsChain: () => Promise<void> = async () => {
        setIsLoading(true);
        try {
            const request: ChainRequest = {
                symbol: selectedSymbol,
                expiration: selectedExpiry || null
            };

            const response = await options.getChain(request);

            setOptionsChain(response);
            setCurrentPrice(response.current_price || 450);

            if (!selectedExpiry && response.expiration_dates && response.expiration_dates.length > 0) {
                setSelectedExpiry(response.expiration_dates[0]);
            }
        } catch (error) {
            console.error('Failed to fetch options chain:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const fetchMLForecast: () => Promise<void> = async () => {
        try {
            const mockForecast: MLForecast = {
                direction: Math.random() > 0.5 ? 'bullish' : 'bearish',
                confidence: 0.65 + Math.random() * 0.3,
                suggestedStrategies: ['Covered Call', 'Bull Put Spread', 'Iron Condor', 'Long Straddle'],
                priceTargets: {
                    low: currentPrice * (0.9 + Math.random() * 0.05),
                    median: currentPrice * (1.0 + Math.random() * 0.1),
                    high: currentPrice * (1.1 + Math.random() * 0.15)
                },
                timeline: {
                    short: '1-2 weeks',
                    medium: '1-3 months',
                    long: '3-6 months'
                }
            };
            setMlForecast(mockForecast);
        } catch (error) {
            console.error('Failed to fetch ML forecast:', error);
        }
    };

    const addCustomLeg: () => void = () => {
        if (!newLeg.strike || !newLeg.expiration) {
            alert('Please enter strike price and expiration date');
            return;
        }

        const leg: OptionLeg = {
            id: Date.now().toString(),
            type: newLeg.type || 'call',
            position: newLeg.position || 'long',
            strike: Number(newLeg.strike),
            quantity: Number(newLeg.quantity) || 1,
            expiration: newLeg.expiration,
            premium: Math.random() * 10 + 5,
            delta: (newLeg.type === 'call' ? 0.5 : -0.5) * (newLeg.position === 'long' ? 1 : -1),
            gamma: 0.02,
            theta: -0.05,
            vega: 0.15
        };

        setCustomLegs([...customLegs, leg]);

        setNewLeg({
            ...newLeg,
            strike: currentPrice,
            expiration: selectedExpiry || '2024-03-15'
        });
    };

    const removeLeg: (id: string) => void = (id: string) => {
        setCustomLegs(customLegs.filter(leg => leg.id !== id));
    };

    const addStrategyToCompare: (strategy: StrategyTemplate) => Promise<void> = async (strategy: StrategyTemplate) => {
        if (selectedStrategies.find(s => s.id === strategy.id)) return;

        const adjustedLegs = strategy.legs.map((leg: OptionLeg) => ({
            ...leg,
            strike: currentPrice + (leg.strike || 0),
            expiration: selectedExpiry || new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
        }));

        setIsLoading(true);
        try {
            const greeks = await mockAPI.calculateGreeks({});
            const analysis = await mockAPI.analyzeStrategy({});

            if (greeks && analysis) {
                const strategyAnalysisItem = {
                    id: strategy.id,
                    name: strategy.name,
                    analysis,
                    greeks
                };

                setSelectedStrategies([...selectedStrategies, strategy]);
                setStrategyAnalysis([...strategyAnalysis, strategyAnalysisItem]);

                if (analysis.payoff_diagram) {
                    const plData = analysis.payoff_diagram.map((point: any) => ({
                        price: point.price,
                        profit: point.payoff,
                        strategy: strategy.name
                    }));
                    setProfitLossData(prev => [...prev, ...plData]);
                }
            }
        } catch (error) {
            console.error('Failed to add strategy:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const removeStrategyFromCompare: (id: string) => void = (id: string) => {
        setSelectedStrategies(selectedStrategies.filter(s => s.id !== id));
        setStrategyAnalysis(strategyAnalysis.filter(s => s.id !== id));
    };

    const analyzeCustomStrategy: () => Promise<void> = async () => {
        if (customLegs.length === 0) return;

        setIsLoading(true);
        try {
            const greeks = await mockAPI.calculateGreeks({});
            const analysis = await mockAPI.analyzeStrategy({});
            const risk = await mockAPI.calculateRiskMetrics({});
            const monteCarlo = await mockAPI.runMonteCarlo({});
            const strikes = await mockAPI.optimizeStrike({});

            if (greeks && analysis) {
                setRiskMetrics(risk);
                setMonteCarloDistribution(monteCarlo);
                setStrikeOptimizer(strikes);

                const chartData = [{
                    name: 'Portfolio',
                    delta: greeks.delta,
                    gamma: greeks.gamma,
                    theta: greeks.theta,
                    vega: greeks.vega,
                    rho: greeks.rho
                }];
                setGreeksChartData(chartData);

                if (analysis.payoff_diagram) {
                    setProfitLossData(analysis.payoff_diagram.map((point: any) => ({
                        price: point.price,
                        profit: point.payoff,
                        strategy: 'Custom'
                    })));
                }
            }
        } catch (error) {
            console.error('Failed to analyze strategy:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const runStrategyBacktest: () => Promise<void> = async () => {
        if (!backtestConfig.start_date || !backtestConfig.end_date || !backtestConfig.strategy_type) {
            alert('Please fill all backtest parameters');
            return;
        }

        setIsLoading(true);
        setBacktestResults(null);

        try {
            const response: {
                data: {
                    result: { total_return: number; win_rate: number; profit_factor: number; sharpe_ratio: number };
                    equity_curve: { timestamp: string; equity: number; drawdown: number; cash: number }[];
                    trades: { symbol: string; strategy: string; profit: number; timestamp: string }[]
                }
            } = await mockAPI.runBacktest(backtestConfig);

            if (response && response.data) {
                const {result, equity_curve, trades} = response.data;

                setBacktestResults(result);

                const formattedCurve = equity_curve.map((point: any) => ({
                    date: new Date(point.timestamp).toLocaleDateString(),
                    equity: point.equity,
                    drawdown: point.drawdown || 0,
                    cash: point.cash
                }));
                setEquityData(formattedCurve);

                const formattedTrades = trades.map((t: any) => ({
                    symbol: t.symbol,
                    strategy: t.strategy,
                    profit: t.profit || 0,
                    time: new Date(t.timestamp).toLocaleDateString(),
                    status: (t.profit || 0) >= 0 ? 'win' : 'loss'
                }));
                setRecentTrades(formattedTrades);

                setActiveTab('backtest');
            }
        } catch (error) {
            console.error('Failed to run backtest:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const calculatePortfolioStats: () => Promise<void> = async () => {
        try {
            const stats = await mockAPI.calculatePortfolioStats({});
            setPortfolioStats(stats);
        } catch (error) {
            console.error('Failed to calculate portfolio stats:', error);
        }
    };

    useEffect(() => {
        calculatePortfolioStats();
    }, []);

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-6">
            <div className="max-w-[1800px] mx-auto space-y-6">
                <Header
                    selectedSymbol={selectedSymbol}
                    setSelectedSymbol={setSelectedSymbol}
                    fetchMLForecast={fetchMLForecast}
                    fetchOptionsChain={fetchOptionsChain}
                    isLoading={isLoading}
                />

                {mlForecast && (
                    <MLForecastPanel
                        mlForecast={mlForecast}
                        selectedSymbol={selectedSymbol}
                        selectedStrategies={selectedStrategies}
                        addStrategyToCompare={addStrategyToCompare}
                        isLoading={isLoading}
                    />
                )}

                <Tabs activeTab={activeTab} setActiveTab={setActiveTab}/>

                {activeTab === 'chain' && optionsChain && (
                    <ChainTab
                        selectedSymbol={selectedSymbol}
                        currentPrice={currentPrice}
                        selectedExpiry={selectedExpiry}
                        setSelectedExpiry={setSelectedExpiry}
                        optionsChain={optionsChain}
                    />
                )}

                {activeTab === 'builder' && (
                    <BuilderTab
                        customLegs={customLegs}
                        newLeg={newLeg}
                        setNewLeg={setNewLeg}
                        addCustomLeg={addCustomLeg}
                        removeLeg={removeLeg}
                        analyzeCustomStrategy={analyzeCustomStrategy}
                        profitLossData={profitLossData}
                        riskMetrics={riskMetrics}
                        strikeOptimizer={strikeOptimizer}
                        greeksChartData={greeksChartData}
                        currentPrice={currentPrice}
                        isLoading={isLoading}
                    />
                )}

                {activeTab === 'compare' && (
                    <CompareTab
                        selectedStrategies={selectedStrategies}
                        strategyAnalysis={strategyAnalysis}
                        addStrategyToCompare={addStrategyToCompare}
                        removeStrategyFromCompare={removeStrategyFromCompare}
                        profitLossData={profitLossData}
                        monteCarloDistribution={monteCarloDistribution}
                    />
                )}

                {activeTab === 'backtest' && (
                    <BacktestTab
                        backtestConfig={backtestConfig}
                        setBacktestConfig={setBacktestConfig}
                        runStrategyBacktest={runStrategyBacktest}
                        backtestResults={backtestResults}
                        equityData={equityData}
                        recentTrades={recentTrades}
                        isLoading={isLoading}
                    />
                )}

                {activeTab === 'volatility' && (
                    <VolatilityTab
                        optionsChain={optionsChain}
                        selectedExpiry={selectedExpiry}
                        currentPrice={currentPrice}
                    />
                )}

                {activeTab === 'risk' && (
                    <RiskTab
                        portfolioStats={portfolioStats}
                        riskMetrics={riskMetrics}
                    />
                )}

                {isLoading && (
                    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                        <div className="bg-slate-900 p-6 rounded-xl border border-slate-800 flex items-center gap-3">
                            <Loader2 className="animate-spin text-amber-400" size={24}/>
                            <span className="text-slate-300">Processing...</span>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default OptionsDesk;