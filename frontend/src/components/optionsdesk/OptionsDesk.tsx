'use client'
import React, {useEffect, useState} from 'react';
import {Loader2} from "lucide-react";
import {
    BacktestConfig,
    ChainRequest,
    ChainResponse,
    EquityCurvePoint,
    GreeksChartData,
    GreeksRequest,
    GreeksResponse,
    HistoricalDataPoint,
    MLForecast,
    MonteCarloRequest,
    MonteCarloResponse,
    OptionLeg,
    OptionsBacktestRequest,
    PayoffPoint,
    PortfolioStatsRequest,
    RecentTrades,
    RiskMetricsRequest,
    RiskMetricsResponse,
    StrategyAnalysis,
    StrategyAnalysisRequest,
    StrategyAnalysisResponse,
    StrategyComparisonRequest,
    StrategyComparisonResponse,
    StrategyTemplate,
    StrikeOptimizerRequest,
    StrikeOptimizerResponse,
    Trade
} from '@/types/all_types';
import Header from './Header';
import MLForecastPanel from './MLForecastPanel';
import ChainTab from './tabs/ChainTab';
import BuilderTab from './tabs/BuilderTab';
import CompareTab from './tabs/CompareTab';
import BacktestTab from './tabs/BacktestTab';
import VolatilityTab from './tabs/VolatilityTab';
import RiskTab from './tabs/RiskTab';
import Tabs from "@/components/optionsdesk/tabs/Tabs";
import {market, options, regime} from "@/utils/api";

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
    const [strategyAnalysis, setStrategyAnalysis] = useState<StrategyAnalysis[]>([]);
    const [comparisonData, setComparisonData] = useState<any[]>([]);
    const [profitLossData, setProfitLossData] = useState<any[]>([]);
    const [greeksChartData, setGreeksChartData] = useState<GreeksChartData[]>([]);
    const [monteCarloDistribution, setMonteCarloDistribution] = useState<any>(null);
    const [riskMetrics, setRiskMetrics] = useState<any>(null);
    const [portfolioStats, setPortfolioStats] = useState<any>(null);
    const [strikeOptimizer, setStrikeOptimizer] = useState<StrikeOptimizerResponse | null >(null);
    const [backtestResults, setBacktestResults] = useState<any>(null);
    const [equityData, setEquityData] = useState<any[]>([]);
    const [recentTrades, setRecentTrades] = useState<RecentTrades[]>([]);

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

    useEffect(() => {
        if (customLegs.length > 0) {
            calculatePortfolioStats();
        }
    }, [customLegs]);

    const fetchExpirations = async () => {
        setIsLoading(true);
        try {
            const request: ChainRequest = {
                symbol: selectedSymbol,
                expiration: null
            }
            const response: ChainResponse = await options.getChain(request);

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
            setCurrentPrice(response.current_price);

            if (!selectedExpiry && response.expiration_dates && response.expiration_dates.length > 0) {
                setSelectedExpiry(response.expiration_dates[0]);
            }
        } catch (error) {
            console.error('Failed to fetch options chain:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const fetchMLForecast = async () => {
        try {
            const regimeData = await regime.detect(selectedSymbol, {period: '1y'});

            let direction: 'bullish' | 'bearish' | 'neutral' = 'neutral';
            if (regimeData.current_regime.confidence >= 3 || regimeData.current_regime.metrics.volatility >= .5) {
                direction = 'bullish';
            } else if (regimeData.current_regime.confidence <= 3 || regimeData.current_regime.metrics.volatility <= .5) {
                direction = 'bearish';
            }

            const confidence = regimeData.current_regime.confidence || 0.8;

            const historicalData: HistoricalDataPoint[] = await market.getHistorical(selectedSymbol, {
                period: '1y',
                interval: '1d'
            });
            const recentPrices = historicalData.slice(-30).map(d => d.close);
            const avgPrice = recentPrices.reduce((a, b) => a + b, 0) / recentPrices.length;
            const volatility = Math.sqrt(recentPrices.reduce((sum, p) => sum + Math.pow(p - avgPrice, 2), 0) / recentPrices.length);
            //TODO add more metrics...
            const forecast: MLForecast = {
                direction,
                confidence,
                suggestedStrategies: direction === 'bullish'
                    ? ['Covered Call', 'Bull Put Spread']
                    : direction === 'bearish'
                        ? ['Long Put', 'Bear Call Spread']
                        : ['Iron Condor', 'Long Straddle'],
                priceTargets: {
                    low: currentPrice - volatility,
                    median: currentPrice,
                    high: currentPrice + volatility
                },
                timeline: {
                    short: '1-2 weeks',
                    medium: '1-3 months',
                    long: '3-6 months'
                }
            };
            setMlForecast(forecast);
        } catch (error) {
            console.error('Failed to fetch ML forecast:', error);
        }
    };

    const calculateGreeksForLegs = async (legs: OptionLeg[]): Promise<GreeksResponse | null> => {
        try {
            const request: GreeksRequest = {
                symbol: selectedSymbol,
                legs: legs.map(leg => ({
                    option_type: leg.type.toUpperCase(),
                    strike: Number(leg.strike),
                    expiration: leg.expiration,
                    quantity: leg.position === 'long' ? Number(leg.quantity) : -Number(leg.quantity),
                    premium: leg.premium ?? null
                })),
                volatility: 0.2
            };

            return await options.calculateGreeks(request);

        } catch (error) {
            console.error('Failed to calculate Greeks:', error);
            return null;
        }
    };

    const analyzeStrategy = async (strategyId: string, legs: OptionLeg[]): Promise<StrategyAnalysisResponse | null> => {
        try {
            const request: StrategyAnalysisRequest = {
                symbol: selectedSymbol,
                legs: legs.map(leg => ({
                    option_type: leg.type.toUpperCase(),
                    strike: Number(leg.strike),
                    expiration: leg.expiration,
                    quantity: leg.position === 'long' ? Number(leg.quantity) : -Number(leg.quantity),
                    premium: leg.premium ?? undefined
                })),
                volatility: 0.2
            };
            return await options.analyzeStrategy(request);
        } catch (error) {
            console.error('Failed to analyze strategy:', error);
            return null;
        }
    };

    const calculateRiskMetrics = async (portfolioValue: number, returns: number[] = []): Promise<RiskMetricsResponse | null> => {
        try {
            const request: RiskMetricsRequest = {
                portfolio_value: portfolioValue,
                returns: returns.length > 0 ? returns : [0.05, -0.02, 0.03, -0.01, 0.04],
                confidence_level: 0.95
            };
            return await options.calculateRiskMetrics(request);
        } catch (error) {
            console.error('Failed to calculate risk metrics:', error);
            return null;
        }
    };

    const runMonteCarlo = async (): Promise<MonteCarloResponse | null> => {
        try {
            const request: MonteCarloRequest = {
                current_price: currentPrice,
                volatility: 0.2,
                days: 30,
                num_simulations: 10000,
                drift: 0.08
            };
            return await options.runMonteCarlo(request);
        } catch (error) {
            console.error('Failed to run Monte Carlo:', error);
            return null;
        }
    };

    const compareStrategies = async (strategies: Record<string, any>[]): Promise<StrategyComparisonResponse | null> => {
        try {
            const request: StrategyComparisonRequest = {
                symbol: selectedSymbol,
                strategies: strategies.map(strat => ({
                    name: strat.name,
                    legs: strat.analysis?.legs || []
                }))
            };
            return await options.compareStrategies(request);
        } catch (error) {
            console.error('Failed to compare strategies:', error);
            return null;
        }
    };

    const optimizeStrikes = async (strategyType: string): Promise<StrikeOptimizerResponse | null> => {
        try {
            const request: StrikeOptimizerRequest = {
                symbol: selectedSymbol,
                current_price: currentPrice,
                volatility: 0.2,
                days_to_expiration: 30,
                strategy_type: strategyType,
                num_strikes: 10
            };

            return await options.optimizeStrike(request);
        } catch (error) {
            console.error('Failed to optimize strikes:', error);
            return null;
        }
    };

    const calculatePortfolioStats = async () => {
        try {
            if (customLegs.length === 0) {
                return;
            }

            const positions = customLegs.map(leg => ({
                pnl: (leg.premium || 0) * leg.quantity * (leg.position === 'long' ? -1 : 1),
                pnl_pct: ((leg.premium || 0) / leg.strike) * 100,
                days_held: 0
            }));

            const request: PortfolioStatsRequest = {
                positions
            };

            const stats = await options.calculatePortfolioStats(request);
            setPortfolioStats(stats);
        } catch (error) {
            console.error('Failed to calculate portfolio stats:', error);
        }
    };

    const addCustomLeg = () => {
        if (!newLeg.strike || !newLeg.expiration) {
            alert('Please enter strike price and expiration date');
            return;
        }

        const calculateLegGreeks = async () => {
            try {
                const greeksRequest: GreeksRequest = {
                    symbol: selectedSymbol,
                    legs: [{
                        option_type: (newLeg.type || 'call').toUpperCase() as 'CALL' | 'PUT',
                        strike: Number(newLeg.strike),
                        expiration: newLeg.expiration || '',
                        quantity: 1,
                        premium: null
                    }],
                    volatility: 0.2
                };

                return await options.calculateGreeks(greeksRequest);
            } catch (error) {
                console.error('Failed to calculate Greeks for leg:', error);
                return null;
            }
        };

        calculateLegGreeks().then(greeks => {
            const leg: OptionLeg = {
                id: Date.now().toString(),
                type: newLeg.type || 'call',
                position: newLeg.position || 'long',
                strike: Number(newLeg.strike),
                quantity: Number(newLeg.quantity) || 1,
                expiration: newLeg.expiration || '',
                premium: newLeg.premium,
                delta: greeks?.delta || 0,
                gamma: greeks?.gamma || 0,
                theta: greeks?.theta || 0,
                vega: greeks?.vega || 0
            };

            setCustomLegs([...customLegs, leg]);

            setNewLeg({
                ...newLeg,
                strike: currentPrice,
                expiration: selectedExpiry || new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
            });
        });
    };

    const removeLeg = (id: string) => {
        setCustomLegs(customLegs.filter(leg => leg.id !== id));
    };

    const addStrategyToCompare = async (strategy: StrategyTemplate) => {
        if (selectedStrategies.find(s => s.id === strategy.id)) return;

        const adjustedLegs = strategy.legs.map(leg => ({
            ...leg,
            strike: currentPrice + (leg.strike || 0),
            expiration: selectedExpiry || new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
        }));

        setIsLoading(true);
        try {
            const greeks = await calculateGreeksForLegs(adjustedLegs);
            const analysis = await analyzeStrategy(strategy.id, adjustedLegs);
            const risk = await calculateRiskMetrics(analysis?.initial_cost || 0);

            if (greeks && analysis) {
                const strategyAnalysisItem = {
                    id: strategy.id,
                    name: strategy.name,
                    analysis: analysis,
                    greeks: greeks,
                    riskMetrics: risk
                };

                setSelectedStrategies([...selectedStrategies, strategy]);
                setStrategyAnalysis([...strategyAnalysis, strategyAnalysisItem]);

                const comparison = await compareStrategies([...strategyAnalysis, strategyAnalysisItem]);
                if (comparison) {
                    setComparisonData(comparison.comparisons || []);
                }

                if (analysis.payoff_diagram) {
                    const plData = analysis.payoff_diagram.map((point: PayoffPoint) => ({
                        price: point.price,
                        profit: point.payoff,
                        strategy: strategy.name
                    }));
                    setProfitLossData(prev => [...prev, ...plData]);
                }
            }
        } catch (error) {
            console.error('Failed to add strategy:', error);
            alert('Failed to add strategy. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    const removeStrategyFromCompare: (id: string) => void = (id: string) => {
        setSelectedStrategies(selectedStrategies.filter((s: StrategyTemplate) => s.id !== id));
        setStrategyAnalysis(strategyAnalysis.filter(s => s.id !== id));
    };

    const analyzeCustomStrategy = async () => {
        if (customLegs.length === 0) return;

        setIsLoading(true);
        try {
            const greeks = await calculateGreeksForLegs(customLegs);
            const analysis = await analyzeStrategy('custom', customLegs);
            const risk = await calculateRiskMetrics(analysis?.initial_cost || 0);
            const monteCarlo = await runMonteCarlo();
            const strikes: StrikeOptimizerResponse | null = await optimizeStrikes('custom');

            if (greeks && analysis) {
                setRiskMetrics(risk);
                setMonteCarloDistribution(monteCarlo);
                setStrikeOptimizer(strikes);

                const chartData: GreeksChartData[] = [{
                    name: 'Portfolio',
                    delta: greeks.delta,
                    gamma: greeks.gamma,
                    theta: greeks.theta,
                    vega: greeks.vega,
                    rho: greeks.rho
                }];
                setGreeksChartData(chartData);

                if (analysis.payoff_diagram) {
                    setProfitLossData(analysis.payoff_diagram.map(point => ({
                        price: point.price,
                        profit: point.payoff,
                        strategy: 'Custom'
                    })));
                }
            }
        } catch (error) {
            console.error('Failed to analyze strategy:', error);
            alert('Failed to analyze strategy. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    const runStrategyBacktest = async () => {
        if (!backtestConfig.start_date || !backtestConfig.end_date || !backtestConfig.strategy_type) {
            alert('Please fill all backtest parameters');
            return;
        }

        setIsLoading(true);
        setBacktestResults(null);

        try {
            const request: OptionsBacktestRequest = {
                symbol: selectedSymbol,
                strategy_type: backtestConfig.strategy_type,
                initial_capital: backtestConfig.initial_capital,
                risk_free_rate: backtestConfig.risk_free_rate,
                start_date: backtestConfig.start_date,
                end_date: backtestConfig.end_date,
                entry_rules: backtestConfig.entry_rules,
                exit_rules: backtestConfig.exit_rules
            };

            const response = await options.runBacktest(request);

            if (response) {
                setBacktestResults(response);

                const formattedCurve = response.equity_curve.map((point: EquityCurvePoint) => ({
                    date: new Date(point.timestamp).toLocaleDateString(),
                    equity: point.equity,
                    drawdown: point.drawdown,
                    cash: point.equity
                }));
                setEquityData(formattedCurve);

                const formattedTrades: RecentTrades[] = response.trades.map((trade: Trade) => ({
                    symbol: selectedSymbol,
                    strategy: trade.strategy,
                    profit: trade.profit || 0,
                    time: new Date(trade.executed_at).toLocaleDateString(),
                    status: (trade.profit || 0) >= 0 ? 'win' : 'loss'
                }));
                setRecentTrades(formattedTrades);

                setActiveTab('backtest');
            }
        } catch (error) {
            console.error('Failed to run backtest:', error);
            alert('Failed to run backtest. Please try again.');
        } finally {
            setIsLoading(false);
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
