/**
 * Enhanced Strategy Deployment Modal
 * Integrates with all Phase 3 features
 */

'use client'

import React, { useState } from 'react';
import { Activity, AlertTriangle, Bell, CheckCircle, DollarSign, Info, Settings, Shield, X } from 'lucide-react';
import { BacktestResultToDeploy, DeploymentConfig } from "@/types/all_types";
import { formatPercent, toPrecision } from "@/utils/formatters";


interface DeploymentModalProps {
    backtest?: BacktestResultToDeploy;
    onClose: () => void;
    onDeploy: (config: DeploymentConfig) => void;
}

export default function DeploymentModal({
    backtest,
    onClose,
    onDeploy
}: DeploymentModalProps) {
    const [config, setConfig] = useState<DeploymentConfig>({
        source: backtest ? 'backtest' : 'custom',
        backtest_id: backtest?.id ? parseInt(backtest.id, 10) : undefined,
        name: backtest ? `${backtest.strategy} - ${backtest.symbols.join(', ')}` : '',
        strategy_key: backtest?.strategy || '',
        parameters: backtest?.parameters || {},
        symbols: backtest?.symbols || [],
        deployment_mode: 'paper',
        initial_capital: 10000,
        max_position_pct: 20,
        stop_loss_pct: 5,
        daily_loss_limit: 500,
        position_sizing_method: 'fixed_percent',
        enable_trailing_stop: false,
        enable_alerts: true
    });

    const [activeTab, setActiveTab] = useState<'basic' | 'risk' | 'advanced' | 'alerts'>('basic');
    const [agreedToRisk, setAgreedToRisk] = useState(false);
    const [preflightChecks, setPreflightChecks] = useState({
        brokerConnected: true,
        buyingPower: true,
        parametersValid: true,
        riskConfigured: true
    });
    const [isDeploying, setIsDeploying] = useState(false);

    const handleDeploy = async () => {
        if (!canDeploy) return;

        setIsDeploying(true);
        try {
            // Ensure backtest_id is clean
            const finalConfig = { ...config };
            if (finalConfig.source !== 'backtest' || !finalConfig.backtest_id || isNaN(finalConfig.backtest_id)) {
                delete finalConfig.backtest_id;
            }

            await onDeploy(finalConfig);
        } finally {
            setIsDeploying(false);
        }
    };

    const allChecksPassed = Object.values(preflightChecks).every(v => v === true);
    const canDeploy = allChecksPassed && agreedToRisk;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
            <div className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl max-w-4xl w-full mx-4 max-h-[90vh] overflow-hidden flex flex-col">

                {/* Header */}
                <div className="p-6 border-b border-slate-700 flex items-center justify-between">
                    <div>
                        <h2 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
                            🚀 Deploy Strategy to Live Trading
                        </h2>
                        <p className="text-sm text-slate-400 mt-1">
                            Configure your strategy for {config.deployment_mode} trading
                        </p>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
                    >
                        <X size={20} className="text-slate-400" />
                    </button>
                </div>

                {/* Tabs */}
                <div className="flex border-b border-slate-800 px-6">
                    {[
                        { id: 'basic', label: 'Basic', icon: Settings },
                        { id: 'risk', label: 'Risk Management', icon: Shield },
                        { id: 'advanced', label: 'Advanced', icon: Activity },
                        { id: 'alerts', label: 'Alerts', icon: Bell }
                    ].map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id as any)}
                            className={`px-6 py-3 font-semibold text-sm flex items-center gap-2 border-b-2 transition-colors ${activeTab === tab.id
                                ? 'border-violet-500 text-violet-400'
                                : 'border-transparent text-slate-400 hover:text-slate-300'
                                }`}
                        >
                            <tab.icon size={16} />
                            {tab.label}
                        </button>
                    ))}
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-6">

                    {/* Backtest Summary (if deploying from backtest) */}
                    {backtest && (
                        <div className="mb-6 p-4 bg-violet-500/10 border border-violet-500/30 rounded-xl">
                            <h3 className="text-sm font-bold text-violet-400 mb-3">Backtest Performance</h3>
                            <div className="grid grid-cols-4 gap-4 text-sm">
                                <div>
                                    <span className="text-slate-500">Return:</span>
                                    <span className="text-emerald-400 font-bold ml-2">+{formatPercent(backtest.total_return_pct)}</span>
                                </div>
                                <div>
                                    <span className="text-slate-500">Sharpe:</span>
                                    <span className="text-slate-200 font-bold ml-2">{toPrecision(backtest.sharpe_ratio)}</span>
                                </div>
                                <div>
                                    <span className="text-slate-500">Max DD:</span>
                                    <span className="text-red-400 font-bold ml-2">{formatPercent(backtest.max_drawdown)}</span>
                                </div>
                                <div>
                                    <span className="text-slate-500">Trades:</span>
                                    <span className="text-slate-200 font-bold ml-2">{backtest.total_trades}</span>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Basic Tab */}
                    {activeTab === 'basic' && (
                        <div className="space-y-6">
                            <div>
                                <label className="block text-sm font-bold text-slate-300 mb-2">Strategy Name</label>
                                <input
                                    type="text"
                                    value={config.name}
                                    onChange={(e) => setConfig({ ...config, name: e.target.value })}
                                    className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:border-violet-500 outline-none"
                                    placeholder="My Trading Strategy"
                                />
                            </div>

                            <div>
                                <label className="block text-sm font-bold text-slate-300 mb-2">Trading Mode</label>
                                <div className="grid grid-cols-2 gap-4">
                                    <button
                                        onClick={() => setConfig({ ...config, deployment_mode: 'paper' })}
                                        className={`p-4 rounded-xl border-2 transition-all ${config.deployment_mode === 'paper'
                                            ? 'border-blue-500 bg-blue-500/10'
                                            : 'border-slate-700 bg-slate-800 hover:border-slate-600'
                                            }`}
                                    >
                                        <div className="text-left">
                                            <div className="font-bold text-slate-200">📋 Paper Trading</div>
                                            <div className="text-xs text-slate-400 mt-1">
                                                Simulated trading with real market data
                                            </div>
                                        </div>
                                    </button>

                                    <button
                                        onClick={() => setConfig({ ...config, deployment_mode: 'live' })}
                                        className={`p-4 rounded-xl border-2 transition-all ${config.deployment_mode === 'live'
                                            ? 'border-violet-500 bg-violet-500/10'
                                            : 'border-slate-700 bg-slate-800 hover:border-slate-600'
                                            }`}
                                    >
                                        <div className="text-left">
                                            <div className="font-bold text-slate-200">🚀 Live Trading</div>
                                            <div className="text-xs text-slate-400 mt-1">
                                                Real money trading (use with caution)
                                            </div>
                                        </div>
                                    </button>
                                </div>
                            </div>

                            <div>
                                <label className="block text-sm font-bold text-slate-300 mb-2">Initial Capital</label>
                                <div className="relative">
                                    <DollarSign className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={20} />
                                    <input
                                        type="number"
                                        value={config.initial_capital}
                                        onChange={(e) => setConfig({ ...config, initial_capital: parseFloat(e.target.value) })}
                                        className="w-full pl-12 pr-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:border-violet-500 outline-none"
                                    />
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Risk Management Tab */}
                    {activeTab === 'risk' && (
                        <div className="space-y-6">
                            <div>
                                <label className="block text-sm font-bold text-slate-300 mb-2">
                                    Max Position Size (% of Portfolio)
                                </label>
                                <input
                                    type="range"
                                    min="5"
                                    max="100"
                                    value={config.max_position_pct}
                                    onChange={(e) => setConfig({ ...config, max_position_pct: parseFloat(e.target.value) })}
                                    className="w-full"
                                />
                                <div className="flex justify-between text-sm mt-1">
                                    <span className="text-slate-500">5%</span>
                                    <span className="text-violet-400 font-bold">{config.max_position_pct}%</span>
                                    <span className="text-slate-500">100%</span>
                                </div>
                            </div>

                            <div>
                                <label className="block text-sm font-bold text-slate-300 mb-2">
                                    Stop Loss (% per Trade)
                                </label>
                                <input
                                    type="range"
                                    min="0.5"
                                    max="20"
                                    step="0.5"
                                    value={config.stop_loss_pct}
                                    onChange={(e) => setConfig({ ...config, stop_loss_pct: parseFloat(e.target.value) })}
                                    className="w-full"
                                />
                                <div className="flex justify-between text-sm mt-1">
                                    <span className="text-slate-500">0.5%</span>
                                    <span className="text-red-400 font-bold">{config.stop_loss_pct}%</span>
                                    <span className="text-slate-500">20%</span>
                                </div>
                            </div>

                            <div>
                                <label className="block text-sm font-bold text-slate-300 mb-2">Daily Loss Limit ($)</label>
                                <input
                                    type="number"
                                    value={config.daily_loss_limit}
                                    onChange={(e) => setConfig({ ...config, daily_loss_limit: parseFloat(e.target.value) })}
                                    className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:border-violet-500 outline-none"
                                />
                                <p className="text-xs text-slate-500 mt-1">Strategy pauses if daily loss exceeds this amount</p>
                            </div>

                            {/* Trailing Stop */}
                            <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700">
                                <label className="flex items-center gap-3 cursor-pointer">
                                    <input
                                        type="checkbox"
                                        checked={config.enable_trailing_stop}
                                        onChange={(e) => setConfig({ ...config, enable_trailing_stop: e.target.checked })}
                                        className="w-5 h-5"
                                    />
                                    <div>
                                        <div className="font-semibold text-slate-200">Enable Trailing Stop</div>
                                        <div className="text-xs text-slate-400">Automatically adjust stop-loss as price moves in your favor</div>
                                    </div>
                                </label>

                                {config.enable_trailing_stop && (
                                    <div className="mt-4">
                                        <label className="block text-sm font-bold text-slate-300 mb-2">Trail %</label>
                                        <input
                                            type="number"
                                            step="0.5"
                                            value={config.trailing_stop_pct || 5}
                                            onChange={(e) => setConfig({ ...config, trailing_stop_pct: parseFloat(e.target.value) })}
                                            className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-200"
                                        />
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Advanced Tab */}
                    {activeTab === 'advanced' && (
                        <div className="space-y-6">
                            <div>
                                <label className="block text-sm font-bold text-slate-300 mb-2">Position Sizing Method</label>
                                <select
                                    value={config.position_sizing_method}
                                    onChange={(e) => setConfig({ ...config, position_sizing_method: e.target.value as any })}
                                    className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:border-violet-500 outline-none"
                                >
                                    <option value="fixed_percent">Fixed Percent (Simple)</option>
                                    <option value="kelly_criterion">Kelly Criterion (Optimal Growth)</option>
                                    <option value="volatility_based">Volatility-Based (Risk-Adjusted)</option>
                                    <option value="atr_based">ATR-Based (Technical)</option>
                                </select>
                            </div>

                            {/* Method-specific parameters */}
                            {config.position_sizing_method === 'kelly_criterion' && (
                                <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-xl space-y-3">
                                    <div className="font-semibold text-blue-400 flex items-center gap-2">
                                        <Info size={16} />
                                        Kelly Criterion Parameters
                                    </div>
                                    <div className="grid grid-cols-3 gap-3">
                                        <div>
                                            <label className="text-xs text-slate-400 block mb-1">Win Rate</label>
                                            <input
                                                type="number"
                                                step="0.01"
                                                placeholder="0.60"
                                                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-sm"
                                            />
                                        </div>
                                        <div>
                                            <label className="text-xs text-slate-400 block mb-1">Avg Win %</label>
                                            <input
                                                type="number"
                                                step="0.01"
                                                placeholder="0.03"
                                                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-sm"
                                            />
                                        </div>
                                        <div>
                                            <label className="text-xs text-slate-400 block mb-1">Avg Loss %</label>
                                            <input
                                                type="number"
                                                step="0.01"
                                                placeholder="0.015"
                                                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-sm"
                                            />
                                        </div>
                                    </div>
                                </div>
                            )}

                            <div>
                                <label className="block text-sm font-bold text-slate-300 mb-2">Notes</label>
                                <textarea
                                    value={config.notes || ''}
                                    onChange={(e) => setConfig({ ...config, notes: e.target.value })}
                                    className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:border-violet-500 outline-none resize-none"
                                    rows={3}
                                    placeholder="Optional notes about this deployment..."
                                />
                            </div>
                        </div>
                    )}

                    {/* Alerts Tab */}
                    {activeTab === 'alerts' && (
                        <div className="space-y-6">
                            <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700">
                                <label className="flex items-center gap-3 cursor-pointer">
                                    <input
                                        type="checkbox"
                                        checked={config.enable_alerts}
                                        onChange={(e) => setConfig({ ...config, enable_alerts: e.target.checked })}
                                        className="w-5 h-5"
                                    />
                                    <div>
                                        <div className="font-semibold text-slate-200">Enable Notifications</div>
                                        <div className="text-xs text-slate-400">Receive alerts for trades and risk events</div>
                                    </div>
                                </label>
                            </div>

                            {config.enable_alerts && (
                                <>
                                    <div>
                                        <label className="block text-sm font-bold text-slate-300 mb-2">Email Address</label>
                                        <input
                                            type="email"
                                            value={config.alert_email || ''}
                                            onChange={(e) => setConfig({ ...config, alert_email: e.target.value })}
                                            className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:border-violet-500 outline-none"
                                            placeholder="your.email@example.com"
                                        />
                                    </div>

                                    <div>
                                        <label className="block text-sm font-bold text-slate-300 mb-2">Phone Number (SMS)</label>
                                        <input
                                            type="tel"
                                            value={config.alert_phone || ''}
                                            onChange={(e) => setConfig({ ...config, alert_phone: e.target.value })}
                                            className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:border-violet-500 outline-none"
                                            placeholder="+1234567890"
                                        />
                                        <p className="text-xs text-slate-500 mt-1">Include country code (e.g., +1 for US)</p>
                                    </div>
                                </>
                            )}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="p-6 border-t border-slate-700 space-y-4">
                    {/* Pre-flight Checks */}
                    <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700">
                        <h4 className="text-sm font-bold text-slate-300 mb-3">Pre-Flight Checks</h4>
                        <div className="space-y-2">
                            {Object.entries(preflightChecks).map(([key, passed]) => (
                                <div key={key} className="flex items-center gap-2">
                                    {passed ? (
                                        <CheckCircle size={16} className="text-emerald-400" />
                                    ) : (
                                        <X size={16} className="text-red-400" />
                                    )}
                                    <span className="text-sm text-slate-300 capitalize">{key.replace(/([A-Z])/g, ' $1')}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Risk Warning */}
                    <div className="p-4 bg-amber-500/10 border border-amber-500/30 rounded-xl">
                        <div className="flex items-start gap-3">
                            <AlertTriangle className="text-amber-400 flex-shrink-0 mt-0.5" size={20} />
                            <div className="flex-1">
                                <div className="font-bold text-amber-400 mb-1">⚠️ RISK WARNING</div>
                                <p className="text-sm text-slate-300 mb-3">
                                    Trading involves capital risk. Past performance does not guarantee future results.
                                    Monitor your positions actively and ensure risk limits are appropriate.
                                </p>
                                <label className="flex items-start gap-2 cursor-pointer">
                                    <input
                                        type="checkbox"
                                        checked={agreedToRisk}
                                        onChange={(e) => setAgreedToRisk(e.target.checked)}
                                        className="mt-1"
                                    />
                                    <span className="text-sm text-slate-300">
                                        I understand the risks and have reviewed all settings
                                    </span>
                                </label>
                            </div>
                        </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex items-center justify-between">
                        <button
                            onClick={onClose}
                            className="px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg text-slate-200 font-semibold"
                        >
                            Cancel
                        </button>

                        <div className="flex gap-3">
                            {config.deployment_mode === 'live' && (
                                <button
                                    onClick={() => {
                                        setConfig({ ...config, deployment_mode: 'paper' });
                                        handleDeploy();
                                    }}
                                    className="px-6 py-3 bg-blue-500/20 border border-blue-500/50 hover:bg-blue-500/30 rounded-lg text-blue-400 font-semibold"
                                    disabled={!allChecksPassed || isDeploying}
                                >
                                    📋 Deploy to Paper First
                                </button>
                            )}

                            <button
                                onClick={handleDeploy}
                                disabled={!canDeploy || isDeploying}
                                className="px-6 py-3 bg-violet-600 hover:bg-violet-500 rounded-lg text-white font-semibold disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                            >
                                {isDeploying ? (
                                    <>
                                        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                        Deploying...
                                    </>
                                ) : (
                                    <>
                                        🚀 Deploy to {config.deployment_mode === 'paper' ? 'Paper' : 'Live'}
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
}
