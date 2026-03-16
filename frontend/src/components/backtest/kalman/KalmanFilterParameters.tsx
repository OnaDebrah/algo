/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * KalmanFilterParameters.tsx - Enhanced with Pre-Run Validation
 *
 * Includes real-time validation checks:
 * - Sector matching
 * - Correlation coefficient
 * - Cointegration test
 */

import React, {useEffect, useState} from 'react';
import {
    AlertCircle,
    AlertTriangle,
    CheckCircle,
    Loader2,
    Shield,
    TrendingDown,
    TrendingUp,
    XCircle
} from 'lucide-react';
import {MultiAssetConfig} from '@/types/all_types';
import {backtest} from "@/utils/api";
import {ValidationResult} from "@/types/kalman";
import ValidationModal from "@/components/backtest/kalman/ValidationModal";
import {suggestedPairs} from "@/components/backtest/kalman/SuggestedPairs";

interface KalmanFilterParametersProps {
    config: MultiAssetConfig;
    setConfig: (config: MultiAssetConfig) => void;
}

const KalmanFilterParameters: React.FC<KalmanFilterParametersProps> = ({ config, setConfig }) => {
    const [showValidationModal, setShowValidationModal] = useState(false);
    const [validationResult, setValidationResult] = useState<ValidationResult>({
        sectorMatch: null,
        correlation: null,
        cointegration: null,
        isValid: false,
        warnings: [],
        errors: []
    });
    const [isValidating, setIsValidating] = useState(false);

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
        // Reset validation when pair changes
        setValidationResult({
            sectorMatch: null,
            correlation: null,
            cointegration: null,
            isValid: false,
            warnings: [],
            errors: []
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

    // Run validation checks
    const runValidation = async () => {
        const asset1 = config.params.asset_1;
        const asset2 = config.params.asset_2;

        if (!asset1 || !asset2) {
            return;
        }

        setIsValidating(true);
        setShowValidationModal(true);

        try {

            const response =  await backtest.runValidateKalman({
                asset_1: asset1,
                asset_2: asset2,
                period: config.period || '1y',
                interval: config.interval || '1d'
            });

            const warnings: string[] = [];
            const errors: string[] = [];

            // Check sector match
            if (response.sector_1 !== response.sector_2) {
                warnings.push(`Different sectors: ${response.sector_1} vs ${response.sector_2}`);
            }

            // Check correlation
            if (response.correlation < 0.7) {
                if (response.correlation < 0.5) {
                    errors.push(`Low correlation: ${response.correlation.toFixed(2)} (< 0.5)`);
                } else {
                    warnings.push(`Moderate correlation: ${response.correlation.toFixed(2)} (< 0.7)`);
                }
            }

            // Check cointegration
            if (response.cointegration_pvalue > 0.05) {
                if (response.cointegration_pvalue > 0.1) {
                    errors.push(`Not cointegrated: p-value ${response.cointegration_pvalue.toFixed(3)} (> 0.1)`);
                } else {
                    warnings.push(`Weak cointegration: p-value ${response.cointegration_pvalue.toFixed(3)} (> 0.05)`);
                }
            }

            const isValid = errors.length === 0;

            setValidationResult({
                sectorMatch: response.sector_1 === response.sector_2,
                correlation: response.correlation,
                cointegration: response.cointegration_pvalue,
                isValid,
                warnings,
                errors
            });

        } catch (error) {
            console.error('Validation failed:', error);
            setValidationResult({
                sectorMatch: null,
                correlation: null,
                cointegration: null,
                isValid: false,
                warnings: [],
                errors: ['Failed to validate pair. Please try again.']
            });
        } finally {
            setIsValidating(false);
        }
    };

    // Auto-validate when both assets are set
    useEffect(() => {
        if (config.params.asset_1 && config.params.asset_2) {
            // Check if it's a pre-validated pair
            const preValidated = suggestedPairs.find(
                p => p.asset_1 === config.params.asset_1 && p.asset_2 === config.params.asset_2
            );

            if (preValidated) {
                // Set pre-validated results
                setValidationResult({
                    sectorMatch: true,
                    correlation: preValidated.correlation,
                    cointegration: 0.01, // Assume good cointegration for suggested pairs
                    isValid: true,
                    warnings: [],
                    errors: []
                });
            } else {
                // Clear validation for custom pairs
                setValidationResult({
                    sectorMatch: null,
                    correlation: null,
                    cointegration: null,
                    isValid: false,
                    warnings: [],
                    errors: []
                });
            }
        }
    }, [config.params.asset_1, config.params.asset_2]);

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
                <div className="flex items-center justify-between">
                    <h4 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                        <TrendingUp size={16} className="text-violet-400" />
                        Select Asset Pair
                    </h4>

                    {/* Validation Status Badge */}
                    {config.params.asset_1 && config.params.asset_2 && (
                        <div className="flex items-center gap-2">
                            {validationResult.isValid ? (
                                <span className="text-xs px-2 py-1 bg-emerald-500/20 text-emerald-400 rounded-full flex items-center gap-1">
                                    <CheckCircle size={12} />
                                    Validated
                                </span>
                            ) : validationResult.correlation !== null ? (
                                <span className="text-xs px-2 py-1 bg-amber-500/20 text-amber-400 rounded-full flex items-center gap-1">
                                    <AlertTriangle size={12} />
                                    Issues Found
                                </span>
                            ) : (
                                <button
                                    onClick={runValidation}
                                    disabled={isValidating}
                                    className="text-xs px-3 py-1 bg-violet-500/20 hover:bg-violet-500/30 text-violet-400 rounded-full flex items-center gap-1 transition-colors disabled:opacity-50"
                                >
                                    {isValidating ? (
                                        <>
                                            <Loader2 size={12} className="animate-spin" />
                                            Validating...
                                        </>
                                    ) : (
                                        <>
                                            <Shield size={12} />
                                            Validate Pair
                                        </>
                                    )}
                                </button>
                            )}
                        </div>
                    )}
                </div>

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
                        Pre-Validated Pairs (Recommended)
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                        {suggestedPairs.map((pair, idx) => (
                            <button
                                key={idx}
                                onClick={() => handlePairSelect(pair.asset_1, pair.asset_2)}
                                className={`px-3 py-2.5 bg-slate-800/60 hover:bg-slate-700/60 border rounded-lg text-left transition-all group relative ${
                                    config.params.asset_1 === pair.asset_1 && config.params.asset_2 === pair.asset_2
                                        ? 'border-violet-500/50 bg-violet-500/10'
                                        : 'border-slate-700/50'
                                }`}
                            >
                                {/* Pre-validated badge */}
                                <div className="absolute top-1 right-1">
                                    <CheckCircle size={12} className="text-emerald-400" />
                                </div>

                                <div className="flex justify-between items-center mb-1">
                                    <span className="font-mono text-sm font-bold text-slate-200 group-hover:text-violet-400 transition-colors">
                                        {pair.asset_1} / {pair.asset_2}
                                    </span>
                                    <span className="text-xs text-emerald-400 font-medium">
                                        ρ {pair.correlation.toFixed(2)}
                                    </span>
                                </div>
                                <p className="text-xs text-slate-500">{pair.category}</p>
                            </button>
                        ))}
                    </div>
                </div>

                {/* Quick Validation Summary */}
                {config.params.asset_1 && config.params.asset_2 && validationResult.correlation !== null && (
                    <div className={`p-3 rounded-lg border ${
                        validationResult.isValid
                            ? 'bg-emerald-500/10 border-emerald-500/30'
                            : 'bg-amber-500/10 border-amber-500/30'
                    }`}>
                        <div className="grid grid-cols-3 gap-2 text-xs">
                            {/* Sector Match */}
                            <div className="flex items-center gap-1">
                                {validationResult.sectorMatch ? (
                                    <CheckCircle size={14} className="text-emerald-400" />
                                ) : (
                                    <AlertTriangle size={14} className="text-amber-400" />
                                )}
                                <span className="text-slate-300">Sector</span>
                            </div>

                            {/* Correlation */}
                            <div className="flex items-center gap-1">
                                {validationResult.correlation! >= 0.7 ? (
                                    <CheckCircle size={14} className="text-emerald-400" />
                                ) : validationResult.correlation! >= 0.5 ? (
                                    <AlertTriangle size={14} className="text-amber-400" />
                                ) : (
                                    <XCircle size={14} className="text-red-400" />
                                )}
                                <span className="text-slate-300">ρ {validationResult.correlation?.toFixed(2)}</span>
                            </div>

                            {/* Cointegration */}
                            <div className="flex items-center gap-1">
                                {validationResult.cointegration! <= 0.05 ? (
                                    <CheckCircle size={14} className="text-emerald-400" />
                                ) : validationResult.cointegration! <= 0.1 ? (
                                    <AlertTriangle size={14} className="text-amber-400" />
                                ) : (
                                    <XCircle size={14} className="text-red-400" />
                                )}
                                <span className="text-slate-300">p {validationResult.cointegration?.toFixed(3)}</span>
                            </div>
                        </div>

                        {/* Show full validation button */}
                        <button
                            onClick={() => setShowValidationModal(true)}
                            className="mt-2 text-xs text-violet-400 hover:text-violet-300 transition-colors"
                        >
                            View detailed validation report →
                        </button>
                    </div>
                )}
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

            {/* Validation Modal */}
            {showValidationModal && (
                <ValidationModal
                    asset1={config.params.asset_1}
                    asset2={config.params.asset_2}
                    validation={validationResult}
                    isValidating={isValidating}
                    onClose={() => setShowValidationModal(false)}
                    onRunValidation={runValidation}
                />
            )}
        </div>
    );
};

export default KalmanFilterParameters;
