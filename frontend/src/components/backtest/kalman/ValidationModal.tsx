import React from 'react';
import {AlertTriangle, CheckCircle, Loader2, RefreshCw, Shield, X, XCircle} from 'lucide-react';
import {ValidationResult} from "@/types/kalman";
import ValidationCheck from "@/components/backtest/kalman/ValidationCheck";

interface ValidationModalProps {
    asset1: string;
    asset2: string;
    validation: ValidationResult;
    isValidating: boolean;
    onClose: () => void;
    onRunValidation: () => void;
}

const ValidationModal: React.FC<ValidationModalProps> = ({
                                                             asset1,
                                                             asset2,
                                                             validation,
                                                             isValidating,
                                                             onClose,
                                                             onRunValidation
                                                         }) => {
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <div className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-slate-700">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-violet-500/20 rounded-lg">
                            <Shield className="text-violet-400" size={24} />
                        </div>
                        <div>
                            <h3 className="text-lg font-bold text-slate-100">
                                Pairs Trading Validation
                            </h3>
                            <p className="text-sm text-slate-400">
                                {asset1} / {asset2}
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
                    >
                        <X className="text-slate-400" size={20} />
                    </button>
                </div>

                {/* Content */}
                <div className="p-6 space-y-6">
                    {/* Overall Status */}
                    {validation.correlation !== null && (
                        <div className={`p-4 rounded-xl border-2 ${
                            validation.isValid
                                ? 'bg-emerald-500/10 border-emerald-500/50'
                                : validation.warnings.length > 0 && validation.errors.length === 0
                                    ? 'bg-amber-500/10 border-amber-500/50'
                                    : 'bg-red-500/10 border-red-500/50'
                        }`}>
                            <div className="flex items-center gap-3">
                                {validation.isValid ? (
                                    <>
                                        <CheckCircle className="text-emerald-400" size={32} />
                                        <div>
                                            <h4 className="text-lg font-bold text-emerald-400">
                                                Pair Validated ✓
                                            </h4>
                                            <p className="text-sm text-slate-300">
                                                This pair meets all requirements for pairs trading
                                            </p>
                                        </div>
                                    </>
                                ) : validation.errors.length > 0 ? (
                                    <>
                                        <XCircle className="text-red-400" size={32} />
                                        <div>
                                            <h4 className="text-lg font-bold text-red-400">
                                                Validation Failed
                                            </h4>
                                            <p className="text-sm text-slate-300">
                                                This pair has critical issues
                                            </p>
                                        </div>
                                    </>
                                ) : (
                                    <>
                                        <AlertTriangle className="text-amber-400" size={32} />
                                        <div>
                                            <h4 className="text-lg font-bold text-amber-400">
                                                Warnings Detected
                                            </h4>
                                            <p className="text-sm text-slate-300">
                                                Pair can be traded but with caution
                                            </p>
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Validation Checks */}
                    <div className="space-y-4">
                        <h4 className="text-sm font-bold text-slate-200">Validation Checklist</h4>

                        {/* Sector Match */}
                        <ValidationCheck
                            title="Sector Matching"
                            description="Assets should be in the same sector for better cointegration"
                            status={validation.sectorMatch === true ? 'pass' : validation.sectorMatch === false ? 'warn' : 'pending'}
                            value={validation.sectorMatch ? "Same sector ✓" : "Different sectors"}
                        />

                        {/* Correlation */}
                        <ValidationCheck
                            title="Historical Correlation"
                            description="Correlation should be > 0.7 for reliable pairs trading"
                            status={
                                validation.correlation === null ? 'pending' :
                                    validation.correlation >= 0.7 ? 'pass' :
                                        validation.correlation >= 0.5 ? 'warn' : 'fail'
                            }
                            value={validation.correlation !== null ? `ρ = ${validation.correlation.toFixed(3)}` : 'Not tested'}
                            threshold="Target: > 0.70"
                        />

                        {/* Cointegration */}
                        <ValidationCheck
                            title="Cointegration Test"
                            description="Engle-Granger test p-value should be < 0.05"
                            status={
                                validation.cointegration === null ? 'pending' :
                                    validation.cointegration <= 0.05 ? 'pass' :
                                        validation.cointegration <= 0.1 ? 'warn' : 'fail'
                            }
                            value={validation.cointegration !== null ? `p = ${validation.cointegration.toFixed(4)}` : 'Not tested'}
                            threshold="Target: < 0.05"
                        />
                    </div>

                    {/* Errors */}
                    {validation.errors.length > 0 && (
                        <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
                            <h5 className="text-sm font-bold text-red-400 mb-2 flex items-center gap-2">
                                <XCircle size={16} />
                                Critical Issues
                            </h5>
                            <ul className="space-y-1 text-sm text-slate-300">
                                {validation.errors.map((error, idx) => (
                                    <li key={idx} className="flex items-start gap-2">
                                        <span className="text-red-400 mt-0.5">•</span>
                                        <span>{error}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* Warnings */}
                    {validation.warnings.length > 0 && (
                        <div className="p-4 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                            <h5 className="text-sm font-bold text-amber-400 mb-2 flex items-center gap-2">
                                <AlertTriangle size={16} />
                                Warnings
                            </h5>
                            <ul className="space-y-1 text-sm text-slate-300">
                                {validation.warnings.map((warning, idx) => (
                                    <li key={idx} className="flex items-start gap-2">
                                        <span className="text-amber-400 mt-0.5">•</span>
                                        <span>{warning}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* Recommendations */}
                    {!validation.isValid && validation.correlation !== null && (
                        <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                            <h5 className="text-sm font-bold text-blue-400 mb-2">
                                💡 Recommendations
                            </h5>
                            <ul className="space-y-1 text-sm text-slate-300">
                                {validation.correlation && validation.correlation < 0.7 && (
                                    <li>• Try pairs within the same industry subsector</li>
                                )}
                                {validation.cointegration && validation.cointegration > 0.05 && (
                                    <li>• Consider using a longer lookback period</li>
                                )}
                                {!validation.sectorMatch && (
                                    <li>• Select assets from the same sector for better results</li>
                                )}
                                <li>• Review the suggested pairs list for pre-validated options</li>
                            </ul>
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between p-6 border-t border-slate-700">
                    <button
                        onClick={onRunValidation}
                        disabled={isValidating}
                        className="px-4 py-2 bg-violet-500/20 hover:bg-violet-500/30 text-violet-400 rounded-lg font-medium transition-colors disabled:opacity-50 flex items-center gap-2"
                    >
                        {isValidating ? (
                            <>
                                <Loader2 size={16} className="animate-spin" />
                                Re-validating...
                            </>
                        ) : (
                            <>
                                <RefreshCw size={16} />
                                Re-validate
                            </>
                        )}
                    </button>
                    <button
                        onClick={onClose}
                        className="px-6 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-lg font-medium transition-colors"
                    >
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ValidationModal;
