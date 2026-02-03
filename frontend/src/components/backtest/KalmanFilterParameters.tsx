/**
 * KalmanFilterParameters.tsx
 *
 * Add this component to your MultiAssetBacktest.tsx
 * It provides UI for Kalman Filter pairs trading parameters
 */

import React from 'react';
import { AlertCircle, TrendingUp, TrendingDown } from 'lucide-react';
import { MultiAssetConfig } from '@/types/all_types';

interface KalmanFilterParametersProps {
    config: MultiAssetConfig;
    setConfig: (config: MultiAssetConfig) => void;
}

const KalmanFilterParameters: React.FC<KalmanFilterParametersProps> = ({ config, setConfig }) => {
    // Suggested pairs
    const suggestedPairs = [
        { asset_1: 'AAPL', asset_2: 'MSFT', category: 'Tech Giants', correlation: '0.85' },
        { asset_1: 'GOOGL', asset_2: 'META', category: 'Ad-Driven', correlation: '0.78' },
        { asset_1: 'JPM', asset_2: 'BAC', category: 'Banks', correlation: '0.91' },
        { asset_1: 'KO', asset_2: 'PEP', category: 'Beverages', correlation: '0.82' },
        { asset_1: 'XOM', asset_2: 'CVX', category: 'Energy', correlation: '0.89' },
        { asset_1: 'NVDA', asset_2: 'AMD', category: 'Semiconductors', correlation: '0.76' },
    ];

    const handlePairSelect = (asset1: string, asset2: string) => {
        setConfig({
            ...config,
            symbols: [asset1, asset2],
            params: {
                ...config.params,
                asset_1: asset1,
                asset_2: asset2
            }
        });
    };

    const updateParam = (key: string, value: any) => {
        setConfig({
            ...config,
            params: {
                ...config.params,
                [key]: value
            }
        });
    };

    return (
        <div className="space-y-6">
            {/* Info Banner */}
            <div className="bg-gradient-to-r from-violet-500/10 to-fuchsia-500/10 border border-violet-500/30 rounded-xl p-4">
                <div className="flex items-start gap-3">
                    <AlertCircle className="text-violet-400 mt-0.5" size={20} />
                    <div className="flex-1">
                        <h4 className="text-sm font-bold text-violet-300 mb-1">Pairs Trading Strategy</h4>
                        <p className="text-xs text-slate-400">
                            Kalman Filter dynamically estimates the hedge ratio between two cointegrated assets.
                            It trades when the spread deviates from its mean, betting on mean reversion.
                        </p>
                    </div>
                </div>
            </div>

            {/* Pair Selection */}
            <div className="space-y-4">
                <h4 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                    <TrendingUp size={16} className="text-violet-400" />
                    Select Asset Pair
                </h4>

                {/* Manual Input */}
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <label className="block text-xs font-medium text-slate-400 mb-2">
                            Asset 1
                        </label>
                        <input
                            type="text"
                            placeholder="e.g., AAPL"
                            value={config.params.asset_1 || ''}
                            onChange={(e) => {
                                const val = e.target.value.toUpperCase();
                                updateParam('asset_1', val);
                                setConfig({
                                    ...config,
                                    symbols: [val, config.params.asset_2 || ''],
                                    params: { ...config.params, asset_1: val }
                                });
                            }}
                            className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                        />
                    </div>
                    <div>
                        <label className="block text-xs font-medium text-slate-400 mb-2">
                            Asset 2
                        </label>
                        <input
                            type="text"
                            placeholder="e.g., MSFT"
                            value={config.params.asset_2 || ''}
                            onChange={(e) => {
                                const val = e.target.value.toUpperCase();
                                updateParam('asset_2', val);
                                setConfig({
                                    ...config,
                                    symbols: [config.params.asset_1 || '', val],
                                    params: { ...config.params, asset_2: val }
                                });
                            }}
                            className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                        />
                    </div>
                </div>

                {/* Suggested Pairs */}
                <div>
                    <label className="block text-xs font-medium text-slate-400 mb-2">
                        Quick Select (Common Pairs)
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                        {suggestedPairs.map((pair, idx) => (
                            <button
                                key={idx}
                                onClick={() => handlePairSelect(pair.asset_1, pair.asset_2)}
                                className="px-3 py-2.5 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50 rounded-lg text-left transition-all group"
                            >
                                <div className="flex justify-between items-center mb-1">
                                    <span className="font-mono text-sm font-bold text-slate-200 group-hover:text-violet-400 transition-colors">
                                        {pair.asset_1} / {pair.asset_2}
                                    </span>
                                    <span className="text-xs text-emerald-400 font-medium">
                                        ρ {pair.correlation}
                                    </span>
                                </div>
                                <p className="text-xs text-slate-500">{pair.category}</p>
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Trading Thresholds */}
            <div className="space-y-4">
                <h4 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                    <TrendingDown size={16} className="text-amber-400" />
                    Trading Thresholds
                </h4>

                <div className="grid grid-cols-3 gap-4">
                    <div>
                        <label className="block text-xs font-medium text-slate-400 mb-2">
                            Entry Z-Score
                        </label>
                        <input
                            type="number"
                            value={config.params.entry_z ?? 2.0}
                            onChange={(e) => updateParam('entry_z', parseFloat(e.target.value))}
                            step="0.1"
                            min="1.0"
                            max="4.0"
                            className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200"
                        />
                        <p className="text-xs text-slate-600 mt-1">Higher = more conservative</p>
                    </div>

                    <div>
                        <label className="block text-xs font-medium text-slate-400 mb-2">
                            Exit Z-Score
                        </label>
                        <input
                            type="number"
                            value={config.params.exit_z ?? 0.5}
                            onChange={(e) => updateParam('exit_z', parseFloat(e.target.value))}
                            step="0.1"
                            min="0.0"
                            max="1.5"
                            className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200"
                        />
                        <p className="text-xs text-slate-600 mt-1">Lower = exit earlier</p>
                    </div>

                    <div>
                        <label className="block text-xs font-medium text-slate-400 mb-2">
                            Stop Loss Z
                        </label>
                        <input
                            type="number"
                            value={config.params.stop_loss_z ?? 3.0}
                            onChange={(e) => updateParam('stop_loss_z', parseFloat(e.target.value))}
                            step="0.1"
                            min="2.0"
                            max="5.0"
                            className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200"
                        />
                        <p className="text-xs text-slate-600 mt-1">Protect from divergence</p>
                    </div>
                </div>
            </div>

            {/* Advanced Kalman Settings (Collapsible) */}
            <details className="group">
                <summary className="cursor-pointer text-sm font-bold text-slate-200 flex items-center gap-2 hover:text-violet-400 transition-colors">
                    <span className="text-violet-400">⚙️</span>
                    Advanced Kalman Filter Settings
                    <span className="text-xs text-slate-500 ml-auto">(click to expand)</span>
                </summary>

                <div className="mt-4 space-y-4 p-4 bg-slate-800/30 rounded-lg border border-slate-700/50">
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-xs font-medium text-slate-400 mb-2">
                                Minimum Observations
                            </label>
                            <input
                                type="number"
                                value={config.params.min_obs ?? 20}
                                onChange={(e) => updateParam('min_obs', parseInt(e.target.value))}
                                min="10"
                                max="60"
                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200"
                            />
                            <p className="text-xs text-slate-600 mt-1">Warm-up period before trading</p>
                        </div>

                        <div>
                            <label className="block text-xs font-medium text-slate-400 mb-2">
                                Decay Factor
                            </label>
                            <input
                                type="number"
                                value={config.params.decay_factor ?? 0.99}
                                onChange={(e) => updateParam('decay_factor', parseFloat(e.target.value))}
                                step="0.01"
                                min="0.90"
                                max="1.0"
                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200"
                            />
                            <p className="text-xs text-slate-600 mt-1">Higher = longer memory</p>
                        </div>

                        <div>
                            <label className="block text-xs font-medium text-slate-400 mb-2">
                                Transitory Std
                            </label>
                            <input
                                type="number"
                                value={config.params.transitory_std ?? 0.01}
                                onChange={(e) => updateParam('transitory_std', parseFloat(e.target.value))}
                                step="0.001"
                                min="0.0001"
                                max="0.1"
                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200"
                            />
                            <p className="text-xs text-slate-600 mt-1">Lower = more stable hedge ratio</p>
                        </div>

                        <div>
                            <label className="block text-xs font-medium text-slate-400 mb-2">
                                Observation Std
                            </label>
                            <input
                                type="number"
                                value={config.params.observation_std ?? 0.1}
                                onChange={(e) => updateParam('observation_std', parseFloat(e.target.value))}
                                step="0.01"
                                min="0.01"
                                max="1.0"
                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200"
                            />
                            <p className="text-xs text-slate-600 mt-1">Higher = more noise filtering</p>
                        </div>
                    </div>
                </div>
            </details>

            {/* Parameter Summary */}
            {config.params.asset_1 && config.params.asset_2 && (
                <div className="p-4 bg-slate-800/40 rounded-lg border border-slate-700/50">
                    <div className="flex items-center justify-between mb-2">
                        <h5 className="text-xs font-bold text-slate-400">Configuration Summary</h5>
                        <span className="text-xs text-emerald-400">✓ Ready to backtest</span>
                    </div>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                        <div className="text-slate-500">Pair:</div>
                        <div className="text-slate-300 font-mono">{config.params.asset_1} / {config.params.asset_2}</div>

                        <div className="text-slate-500">Entry Threshold:</div>
                        <div className="text-slate-300">±{config.params.entry_z ?? 2.0}σ</div>

                        <div className="text-slate-500">Exit Threshold:</div>
                        <div className="text-slate-300">±{config.params.exit_z ?? 0.5}σ</div>

                        <div className="text-slate-500">Stop Loss:</div>
                        <div className="text-slate-300">±{config.params.stop_loss_z ?? 3.0}σ</div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default KalmanFilterParameters;
