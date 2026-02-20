import {
    Activity,
    ArrowRight,
    BarChart3,
    BrainCircuit,
    Clock,
    Loader2,
    RefreshCw,
    Shield,
    Target,
    TrendingUp,
    Zap
} from "lucide-react";
import {
    Bar,
    BarChart,
    CartesianGrid,
    Cell,
    PieChart,
    Pie,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from "recharts";
import React, { useEffect, useState } from "react";
import { formatCurrency } from "@/utils/formatters";
import {
    AllocationResponse,
    CurrentRegimeResponse,
    FeaturesResponse,
    MLForecast,
    MonteCarloResponse,
    RegimeStrengthResponse,
    StrategyTemplate,
    TransitionResponse
} from "@/types/all_types";
import { STRATEGY_TEMPLATES } from "@/components/optionsdesk/contants/strategyTemplates";
import { regime, options } from "@/utils/api";

interface ForecastTabProps {
    selectedSymbol: string;
    currentPrice: number;
    mlForecast: MLForecast | null;
    selectedStrategies: StrategyTemplate[];
    addStrategyToCompare: (strategy: StrategyTemplate) => Promise<void>;
    isLoading: boolean;
}

const REGIME_COLORS: Record<string, string> = {
    'bull_quiet': '#10b981',
    'bull_volatile': '#34d399',
    'bear_quiet': '#ef4444',
    'bear_volatile': '#f87171',
    'neutral': '#6b7280',
    'crisis': '#dc2626',
    'recovery': '#f59e0b',
    'high_volatility': '#a855f7',
};

const REGIME_LABELS: Record<string, string> = {
    'bull_quiet': 'Bullish (Low Vol)',
    'bull_volatile': 'Bullish (High Vol)',
    'bear_quiet': 'Bearish (Low Vol)',
    'bear_volatile': 'Bearish (High Vol)',
    'neutral': 'Neutral',
    'crisis': 'Crisis',
    'recovery': 'Recovery',
    'high_volatility': 'High Volatility',
};

const ALLOCATION_COLORS = ['#10b981', '#3b82f6', '#a855f7', '#f59e0b', '#ef4444', '#6b7280'];

const ForecastTab: React.FC<ForecastTabProps> = ({
    selectedSymbol,
    currentPrice,
    mlForecast,
    selectedStrategies,
    addStrategyToCompare,
    isLoading: parentLoading,
}) => {
    const [loading, setLoading] = useState(false);
    const [regimeData, setRegimeData] = useState<CurrentRegimeResponse | null>(null);
    const [strengthData, setStrengthData] = useState<RegimeStrengthResponse | null>(null);
    const [transitionData, setTransitionData] = useState<TransitionResponse | null>(null);
    const [featuresData, setFeaturesData] = useState<FeaturesResponse | null>(null);
    const [allocationData, setAllocationData] = useState<AllocationResponse | null>(null);
    const [monteCarloData, setMonteCarloData] = useState<MonteCarloResponse | null>(null);

    const fetchAllData = async () => {
        setLoading(true);
        try {
            // Phase 1: fetch regime data first so we can derive MC inputs
            const [regimeRes, strengthRes, transitionRes, featuresRes, allocationRes] = await Promise.allSettled([
                regime.detect(selectedSymbol, { period: '2y' }),
                regime.getStrength(selectedSymbol, { period: '2y' }),
                regime.getTransitions(selectedSymbol, { period: '2y' }),
                regime.getFeatures(selectedSymbol, { period: '2y' }),
                regime.getAllocation(selectedSymbol, { period: '2y' }),
            ]);

            if (regimeRes.status === 'fulfilled') setRegimeData(regimeRes.value);
            if (strengthRes.status === 'fulfilled') setStrengthData(strengthRes.value);
            if (transitionRes.status === 'fulfilled') setTransitionData(transitionRes.value);
            if (featuresRes.status === 'fulfilled') setFeaturesData(featuresRes.value);
            if (allocationRes.status === 'fulfilled') setAllocationData(allocationRes.value);

            // Phase 2: derive Monte Carlo inputs from regime detection
            const detectedVol = regimeRes.status === 'fulfilled'
                ? regimeRes.value.current_regime?.metrics?.volatility
                : null;
            const detectedTrend = regimeRes.status === 'fulfilled'
                ? regimeRes.value.current_regime?.metrics?.trend_strength
                : null;

            const mcVol = (detectedVol && detectedVol > 0) ? detectedVol : 0.2;
            const mcDrift = detectedTrend != null ? detectedTrend * 0.1 : 0.05;

            try {
                const mcRes = await options.runMonteCarlo({
                    current_price: currentPrice,
                    volatility: mcVol,
                    days: 30,
                    num_simulations: 10000,
                    drift: mcDrift,
                });
                setMonteCarloData(mcRes);
            } catch (mcError) {
                console.error('Monte Carlo simulation failed:', mcError);
            }
        } catch (error) {
            console.error('Failed to fetch forecast data:', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (selectedSymbol) {
            fetchAllData();
        }
    }, [selectedSymbol]);

    const isAnyLoading = loading || parentLoading;

    const regimeName = regimeData?.current_regime?.name || 'unknown';
    const regimeLabel = REGIME_LABELS[regimeName] || regimeName;
    const regimeColor = REGIME_COLORS[regimeName] || '#6b7280';
    const confidence = regimeData?.current_regime?.confidence || 0;
    const healthScore = regimeData?.market_health_score || 0;

    // Build allocation pie data
    const allocationPieData = allocationData ? [
        { name: 'Trend Following', value: allocationData.allocation.trend_following },
        { name: 'Momentum', value: allocationData.allocation.momentum },
        { name: 'Volatility', value: allocationData.allocation.volatility_strategies },
        { name: 'Mean Reversion', value: allocationData.allocation.mean_reversion },
        { name: 'Stat Arb', value: allocationData.allocation.statistical_arbitrage },
        { name: 'Cash', value: allocationData.allocation.cash },
    ].filter(d => d.value > 0) : [];

    // Build feature importance bar data
    const featureBarData = featuresData?.top_features?.slice(0, 8).map(f => ({
        name: f.feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        importance: Math.round(f.importance * 100),
        value: f.current_value,
    })) || [];

    // Build transition data
    const transitionItems = transitionData?.likely_transitions?.slice(0, 5) || [];

    // Monte Carlo distribution buckets
    const mcDistribution = (() => {
        if (!monteCarloData?.simulated_prices?.length) return [];
        const prices = monteCarloData.simulated_prices;
        const min = Math.min(...prices);
        const max = Math.max(...prices);
        const bucketCount = 20;
        const bucketSize = (max - min) / bucketCount;
        const buckets: { price: string; count: number; midPrice: number }[] = [];
        for (let i = 0; i < bucketCount; i++) {
            const lo = min + i * bucketSize;
            const hi = lo + bucketSize;
            const count = prices.filter(p => p >= lo && p < hi).length;
            buckets.push({
                price: `$${((lo + hi) / 2).toFixed(0)}`,
                count,
                midPrice: (lo + hi) / 2,
            });
        }
        return buckets;
    })();

    // Recommended strategies based on regime
    const getRecommendedStrategies = (): { strategy: StrategyTemplate; reason: string }[] => {
        const results: { strategy: StrategyTemplate; reason: string }[] = [];
        const find = (id: string) => STRATEGY_TEMPLATES.find(s => s.id === id);

        const metrics = regimeData?.current_regime?.metrics;
        const vol = metrics?.volatility || 0;

        if (regimeName.includes('bull')) {
            const cc = find('covered_call');
            if (cc) results.push({ strategy: cc, reason: 'Generate income in bullish trend' });
            const vcs = find('vertical_call_spread');
            if (vcs) results.push({ strategy: vcs, reason: 'Defined-risk bullish exposure' });
            if (vol > 0.3) {
                const pp = find('protective_put');
                if (pp) results.push({ strategy: pp, reason: 'Hedge against volatile pullbacks' });
            }
        } else if (regimeName.includes('bear')) {
            const vps = find('vertical_put_spread');
            if (vps) results.push({ strategy: vps, reason: 'Defined-risk bearish exposure' });
            const pp = find('protective_put');
            if (pp) results.push({ strategy: pp, reason: 'Protect long positions' });
            const collar = find('collar');
            if (collar) results.push({ strategy: collar, reason: 'Cap downside while keeping upside' });
        } else if (regimeName === 'neutral' || regimeName === 'recovery') {
            const ic = find('iron_condor');
            if (ic) results.push({ strategy: ic, reason: 'Profit from range-bound market' });
            const bf = find('butterfly_spread');
            if (bf) results.push({ strategy: bf, reason: 'Target specific price at expiration' });
            const csp = find('cash_secured_put');
            if (csp) results.push({ strategy: csp, reason: 'Acquire stock at discount' });
        } else if (regimeName.includes('volatil') || regimeName === 'crisis') {
            const straddle = find('long_straddle');
            if (straddle) results.push({ strategy: straddle, reason: 'Profit from large moves in either direction' });
            const strangle = find('long_strangle');
            if (strangle) results.push({ strategy: strangle, reason: 'Lower-cost volatility play' });
            const pp = find('protective_put');
            if (pp) results.push({ strategy: pp, reason: 'Protect against continued downside' });
        }

        // Fallback
        if (results.length === 0) {
            const ic = find('iron_condor');
            if (ic) results.push({ strategy: ic, reason: 'Neutral strategy for uncertain markets' });
        }

        return results.slice(0, 4);
    };

    const recommendedStrategies = getRecommendedStrategies();

    return (
        <div className="space-y-6">
            {/* Top hero: Current regime + confidence + health */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
                {/* Regime Badge */}
                <div className="lg:col-span-2 bg-gradient-to-br from-purple-900/30 to-pink-900/20 border border-purple-500/30 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                            <BrainCircuit size={16} className="text-purple-400" />
                            Market Regime — {selectedSymbol}
                        </h3>
                        <button
                            onClick={fetchAllData}
                            disabled={isAnyLoading}
                            className="p-1.5 rounded-lg hover:bg-slate-700/50 text-slate-400 hover:text-slate-200 transition-colors disabled:opacity-50"
                        >
                            <RefreshCw size={14} className={isAnyLoading ? 'animate-spin' : ''} />
                        </button>
                    </div>

                    {isAnyLoading && !regimeData ? (
                        <div className="flex items-center justify-center py-8 text-slate-500 gap-2">
                            <Loader2 size={20} className="animate-spin" />
                            <span>Analyzing market conditions...</span>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            <div className="flex items-center gap-4">
                                <div
                                    className="text-3xl font-black"
                                    style={{ color: regimeColor }}
                                >
                                    {regimeName.includes('bull') ? '📈' : regimeName.includes('bear') ? '📉' : regimeName === 'crisis' ? '🚨' : '➖'}{' '}
                                    {regimeLabel}
                                </div>
                            </div>

                            <div className="grid grid-cols-3 gap-3">
                                <div className="p-3 bg-slate-800/50 rounded-lg">
                                    <div className="text-xs text-slate-500 mb-1">Confidence</div>
                                    <div className="text-xl font-bold text-purple-400">
                                        {(confidence * 100).toFixed(0)}%
                                    </div>
                                    <div className="mt-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-purple-500 rounded-full transition-all"
                                            style={{ width: `${confidence * 100}%` }}
                                        />
                                    </div>
                                </div>
                                <div className="p-3 bg-slate-800/50 rounded-lg">
                                    <div className="text-xs text-slate-500 mb-1">Market Health</div>
                                    <div className={`text-xl font-bold ${healthScore >= 0.7 ? 'text-emerald-400' : healthScore >= 0.4 ? 'text-amber-400' : 'text-red-400'}`}>
                                        {(healthScore * 100).toFixed(0)}%
                                    </div>
                                    <div className="mt-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full rounded-full transition-all ${healthScore >= 0.7 ? 'bg-emerald-500' : healthScore >= 0.4 ? 'bg-amber-500' : 'bg-red-500'}`}
                                            style={{ width: `${healthScore * 100}%` }}
                                        />
                                    </div>
                                </div>
                                <div className="p-3 bg-slate-800/50 rounded-lg">
                                    <div className="text-xs text-slate-500 mb-1">Current Price</div>
                                    <div className="text-xl font-bold text-slate-200">
                                        {formatCurrency(currentPrice)}
                                    </div>
                                    {mlForecast?.priceTargets && (
                                        <div className="text-xs text-slate-500 mt-1">
                                            Range: ${mlForecast.priceTargets.low.toFixed(0)}-${mlForecast.priceTargets.high.toFixed(0)}
                                        </div>
                                    )}
                                </div>
                            </div>

                            {regimeData?.current_regime?.metrics && (
                                <div className="grid grid-cols-4 gap-2">
                                    <div className="p-2 bg-slate-800/30 rounded-lg text-center">
                                        <div className="text-[10px] text-slate-500 uppercase">Volatility</div>
                                        <div className={`text-sm font-bold ${regimeData.current_regime.metrics.volatility > 0.3 ? 'text-red-400' : 'text-emerald-400'}`}>
                                            {(regimeData.current_regime.metrics.volatility * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                    <div className="p-2 bg-slate-800/30 rounded-lg text-center">
                                        <div className="text-[10px] text-slate-500 uppercase">Trend</div>
                                        <div className={`text-sm font-bold ${regimeData.current_regime.metrics.trend_strength > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                            {regimeData.current_regime.metrics.trend_strength.toFixed(2)}
                                        </div>
                                    </div>
                                    <div className="p-2 bg-slate-800/30 rounded-lg text-center">
                                        <div className="text-[10px] text-slate-500 uppercase">Liquidity</div>
                                        <div className="text-sm font-bold text-blue-400">
                                            {regimeData.current_regime.metrics.liquidity_score.toFixed(2)}
                                        </div>
                                    </div>
                                    <div className="p-2 bg-slate-800/30 rounded-lg text-center">
                                        <div className="text-[10px] text-slate-500 uppercase">Correlation</div>
                                        <div className="text-sm font-bold text-amber-400">
                                            {regimeData.current_regime.metrics.correlation_index.toFixed(2)}
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Regime Strength */}
                <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                    <h4 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
                        <Zap size={16} className="text-amber-400" />
                        Regime Strength
                    </h4>
                    {strengthData ? (
                        <div className="space-y-3">
                            <div className="relative w-24 h-24 mx-auto">
                                <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                                    <circle cx="50" cy="50" r="42" fill="none" stroke="#334155" strokeWidth="8" />
                                    <circle
                                        cx="50" cy="50" r="42" fill="none"
                                        stroke={strengthData.strength >= 0.7 ? '#10b981' : strengthData.strength >= 0.4 ? '#f59e0b' : '#ef4444'}
                                        strokeWidth="8"
                                        strokeDasharray={`${strengthData.strength * 264} 264`}
                                        strokeLinecap="round"
                                    />
                                </svg>
                                <div className="absolute inset-0 flex items-center justify-center">
                                    <span className="text-xl font-bold text-slate-200">
                                        {(strengthData.strength * 100).toFixed(0)}%
                                    </span>
                                </div>
                            </div>
                            <div className="text-center">
                                <div className="text-xs text-slate-500">
                                    {strengthData.confirming_signals}/{strengthData.total_signals} signals confirming
                                </div>
                                <div className="text-xs text-slate-400 mt-1">
                                    {strengthData.description}
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="flex items-center justify-center py-8 text-slate-600">
                            <Loader2 size={16} className="animate-spin" />
                        </div>
                    )}
                </div>

                {/* Transition Probabilities */}
                <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                    <h4 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
                        <Clock size={16} className="text-blue-400" />
                        Regime Outlook
                    </h4>
                    {transitionData ? (
                        <div className="space-y-3">
                            <div className="p-3 bg-slate-800/50 rounded-lg">
                                <div className="text-xs text-slate-500 mb-1">Expected Duration</div>
                                <div className="text-lg font-bold text-slate-200">
                                    {transitionData.expected_duration} days
                                </div>
                                <div className="text-xs text-slate-500">
                                    Median: {transitionData.median_duration} days
                                </div>
                            </div>
                            <div className="p-3 bg-slate-800/50 rounded-lg">
                                <div className="text-xs text-slate-500 mb-1">Prob. Regime Ends Next Week</div>
                                <div className={`text-lg font-bold ${transitionData.probability_end_next_week > 0.5 ? 'text-amber-400' : 'text-emerald-400'}`}>
                                    {(transitionData.probability_end_next_week * 100).toFixed(1)}%
                                </div>
                            </div>
                            {transitionItems.length > 0 && (
                                <div className="space-y-1.5">
                                    <div className="text-xs text-slate-500">Likely Transitions</div>
                                    {transitionItems.map((t, i) => (
                                        <div key={i} className="flex items-center gap-2 text-xs">
                                            <span className="text-slate-400">{REGIME_LABELS[t.from_regime] || t.from_regime}</span>
                                            <ArrowRight size={10} className="text-slate-600" />
                                            <span className="text-slate-300 font-medium">{REGIME_LABELS[t.to_regime] || t.to_regime}</span>
                                            <span className="ml-auto font-bold" style={{ color: REGIME_COLORS[t.to_regime] || '#6b7280' }}>
                                                {(t.probability * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="flex items-center justify-center py-8 text-slate-600">
                            <Loader2 size={16} className="animate-spin" />
                        </div>
                    )}
                </div>
            </div>

            {/* Middle row: Strategy Recommendations + Monte Carlo */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* AI Strategy Recommendations */}
                <div className="bg-gradient-to-br from-emerald-900/20 to-teal-900/10 border border-emerald-500/20 rounded-xl p-6">
                    <h4 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
                        <Target size={16} className="text-emerald-400" />
                        AI Strategy Recommendations
                    </h4>
                    <p className="text-xs text-slate-500 mb-4">
                        Based on current <span className="font-medium" style={{ color: regimeColor }}>{regimeLabel}</span> regime
                    </p>
                    <div className="space-y-3">
                        {recommendedStrategies.map(({ strategy, reason }, idx) => {
                            const alreadyAdded = selectedStrategies.some(s => s.id === strategy.id);
                            return (
                                <div key={idx} className="flex items-center gap-3 p-3 bg-slate-800/40 rounded-lg border border-slate-700/30 hover:border-emerald-500/30 transition-colors">
                                    <div className="flex-1 min-w-0">
                                        <div className="text-sm font-bold text-slate-200 flex items-center gap-2">
                                            {strategy.name}
                                            <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${strategy.risk === 'Defined' ? 'bg-emerald-500/20 text-emerald-400' :
                                                strategy.risk === 'Low' ? 'bg-blue-500/20 text-blue-400' :
                                                    strategy.risk === 'Unlimited' ? 'bg-red-500/20 text-red-400' :
                                                        'bg-amber-500/20 text-amber-400'
                                                }`}>
                                                {strategy.risk} Risk
                                            </span>
                                        </div>
                                        <div className="text-xs text-slate-500 mt-0.5">{reason}</div>
                                    </div>
                                    <button
                                        onClick={() => addStrategyToCompare(strategy)}
                                        disabled={alreadyAdded || selectedStrategies.length >= 4}
                                        className="px-3 py-1.5 bg-emerald-600/20 hover:bg-emerald-600/30 border border-emerald-500/30 rounded-lg text-xs font-bold text-emerald-400 transition-colors disabled:opacity-40 disabled:cursor-not-allowed whitespace-nowrap"
                                    >
                                        {alreadyAdded ? 'Added' : '+ Compare'}
                                    </button>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Monte Carlo Price Distribution */}
                <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                    <h4 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
                        <BarChart3 size={16} className="text-blue-400" />
                        30-Day Price Simulation (10K paths)
                    </h4>
                    {monteCarloData ? (
                        <>
                            <div className="h-48 mb-4">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={mcDistribution}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                        <XAxis dataKey="price" stroke="#64748b" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
                                        <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', fontSize: 12 }}
                                            labelStyle={{ color: '#94a3b8' }}
                                            formatter={(value) => [`${value ?? 0} paths`, 'Count']}
                                        />
                                        <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                                            {mcDistribution.map((entry, index) => (
                                                <Cell
                                                    key={index}
                                                    fill={entry.midPrice >= currentPrice ? '#10b981' : '#ef4444'}
                                                    fillOpacity={0.7}
                                                />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                            <div className="grid grid-cols-3 gap-2">
                                <div className="p-2 bg-slate-800/50 rounded-lg text-center">
                                    <div className="text-[10px] text-slate-500 uppercase">5th Percentile</div>
                                    <div className="text-sm font-bold text-red-400">
                                        {formatCurrency(monteCarloData.percentile_5)}
                                    </div>
                                </div>
                                <div className="p-2 bg-slate-800/50 rounded-lg text-center">
                                    <div className="text-[10px] text-slate-500 uppercase">Median</div>
                                    <div className="text-sm font-bold text-slate-200">
                                        {formatCurrency(monteCarloData.median_final_price)}
                                    </div>
                                </div>
                                <div className="p-2 bg-slate-800/50 rounded-lg text-center">
                                    <div className="text-[10px] text-slate-500 uppercase">95th Percentile</div>
                                    <div className="text-sm font-bold text-emerald-400">
                                        {formatCurrency(monteCarloData.percentile_95)}
                                    </div>
                                </div>
                            </div>
                            <div className="mt-3 p-2 bg-blue-900/20 border border-blue-500/20 rounded-lg text-center">
                                <span className="text-xs text-blue-400 font-medium">
                                    Probability of profit: {(monteCarloData.probability_above_current * 100).toFixed(1)}%
                                </span>
                            </div>
                        </>
                    ) : (
                        <div className="flex items-center justify-center py-16 text-slate-600">
                            <Loader2 size={16} className="animate-spin" />
                        </div>
                    )}
                </div>
            </div>

            {/* Bottom row: Feature Importance + Allocation */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Feature Importance */}
                <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                    <h4 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
                        <Activity size={16} className="text-purple-400" />
                        Key Market Drivers
                    </h4>
                    {featureBarData.length > 0 ? (
                        <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={featureBarData} layout="vertical" margin={{ left: 10 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                                    <XAxis type="number" stroke="#64748b" tick={{ fontSize: 10 }} domain={[0, 100]} />
                                    <YAxis type="category" dataKey="name" stroke="#64748b" tick={{ fontSize: 10 }} width={100} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', fontSize: 12 }}
                                        labelStyle={{ color: '#94a3b8' }}
                                        formatter={(value) => [`${value ?? 0}%`, 'Importance']}
                                    />
                                    <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                                        {featureBarData.map((_, index) => (
                                            <Cell key={index} fill={`hsl(${270 - index * 20}, 70%, 60%)`} fillOpacity={0.8} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    ) : (
                        <div className="flex items-center justify-center py-16 text-slate-600">
                            {loading ? <Loader2 size={16} className="animate-spin" /> : <span className="text-sm">No feature data available</span>}
                        </div>
                    )}
                </div>

                {/* Strategy Allocation */}
                <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                    <h4 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
                        <Shield size={16} className="text-amber-400" />
                        Recommended Allocation
                    </h4>
                    {allocationPieData.length > 0 ? (
                        <div className="flex items-center gap-6">
                            <div className="w-40 h-40 flex-shrink-0">
                                <ResponsiveContainer width="100%" height="100%">
                                    <PieChart>
                                        <Pie
                                            data={allocationPieData}
                                            cx="50%"
                                            cy="50%"
                                            innerRadius={35}
                                            outerRadius={65}
                                            paddingAngle={2}
                                            dataKey="value"
                                        >
                                            {allocationPieData.map((_, index) => (
                                                <Cell key={index} fill={ALLOCATION_COLORS[index % ALLOCATION_COLORS.length]} />
                                            ))}
                                        </Pie>
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', fontSize: 12 }}
                                            formatter={(value) => [`${(Number(value ?? 0) * 100).toFixed(0)}%`, 'Weight']}
                                        />
                                    </PieChart>
                                </ResponsiveContainer>
                            </div>
                            <div className="flex-1 space-y-2">
                                {allocationPieData.map((item, idx) => (
                                    <div key={idx} className="flex items-center gap-2">
                                        <div
                                            className="w-3 h-3 rounded-full flex-shrink-0"
                                            style={{ backgroundColor: ALLOCATION_COLORS[idx % ALLOCATION_COLORS.length] }}
                                        />
                                        <span className="text-xs text-slate-400 flex-1">{item.name}</span>
                                        <span className="text-xs font-bold text-slate-300">
                                            {(item.value * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ) : (
                        <div className="flex items-center justify-center py-16 text-slate-600">
                            {loading ? <Loader2 size={16} className="animate-spin" /> : <span className="text-sm">No allocation data available</span>}
                        </div>
                    )}
                </div>
            </div>

            {/* Historical Regimes Timeline */}
            {regimeData?.historical_regimes && regimeData.historical_regimes.length > 0 && (
                <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6">
                    <h4 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
                        <TrendingUp size={16} className="text-slate-400" />
                        Recent Regime History
                    </h4>
                    <div className="flex gap-1 h-8 rounded-lg overflow-hidden">
                        {regimeData.historical_regimes.slice(-30).map((r, idx) => (
                            <div
                                key={idx}
                                className="flex-1 transition-colors hover:opacity-80 cursor-default"
                                style={{ backgroundColor: REGIME_COLORS[r.name] || '#374151' }}
                                title={`${REGIME_LABELS[r.name] || r.name} (${(r.confidence * 100).toFixed(0)}% confidence)`}
                            />
                        ))}
                    </div>
                    <div className="flex justify-between mt-2">
                        <span className="text-[10px] text-slate-600">Oldest</span>
                        <div className="flex gap-3 flex-wrap justify-center">
                            {Object.entries(REGIME_LABELS).slice(0, 5).map(([key, label]) => (
                                <span key={key} className="flex items-center gap-1 text-[10px] text-slate-500">
                                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: REGIME_COLORS[key] }} />
                                    {label}
                                </span>
                            ))}
                        </div>
                        <span className="text-[10px] text-slate-600">Now</span>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ForecastTab;
