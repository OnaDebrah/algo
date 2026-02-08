'use client'

import React, { useEffect, useState } from "react";
import {
    Check,
    Copy,
    CreditCard,
    Database,
    Globe,
    Info,
    Key,
    Landmark,
    Link,
    Loader2,
    Percent,
    RefreshCw,
    Save,
    ShieldCheck,
    TrendingDown,
    Zap,
    AlertCircle,
    Eye,
    EyeOff,
    TestTube,
    Bell
} from "lucide-react";
import { settings } from "@/utils/api";
import { UserSettings } from "@/types/all_types";
import AlertManagement from "@/components/settings/AlertManagement";

const SettingsPage = () => {
    const [activeTab, setActiveTab] = useState('backtest');
    const [isLoading, setIsLoading] = useState(true);
    const [isSaving, setIsSaving] = useState(false);
    const [isTesting, setIsTesting] = useState(false);
    const [testResult, setTestResult] = useState<any>(null);

    // Settings state
    const [userSettings, setUserSettings] = useState<UserSettings | null>(null);

    // Backtest settings
    const [backtestDataSource, setBacktestDataSource] = useState('yahoo');
    const [slippage, setSlippage] = useState(0.001);
    const [commission, setCommission] = useState(0.002);
    const [initialCapital, setInitialCapital] = useState(10000);

    // Live trading settings
    const [liveDataSource, setLiveDataSource] = useState('alpaca');
    const [defaultBroker, setDefaultBroker] = useState('paper');
    const [autoConnect, setAutoConnect] = useState(false);

    // Broker credentials
    const [brokerApiKey, setBrokerApiKey] = useState('');
    const [brokerApiSecret, setBrokerApiSecret] = useState('');
    const [brokerBaseUrl, setBrokerBaseUrl] = useState('');
    const [showApiSecret, setShowApiSecret] = useState(false);
    const [brokerConfigured, setBrokerConfigured] = useState(false);

    // General settings
    const [theme, setTheme] = useState('dark');
    const [notifications, setNotifications] = useState(true);
    const [autoRefresh, setAutoRefresh] = useState(true);
    const [refreshInterval, setRefreshInterval] = useState(30);

    useEffect(() => {
        fetchSettings();
    }, []);

    const fetchSettings = async () => {
        setIsLoading(true);
        try {
            const response = await settings.get();
            if (response) {
                setUserSettings(response);

                // Backtest settings
                setBacktestDataSource(response.backtest?.data_source || "yahoo");
                setSlippage(response.backtest?.slippage || 0.001);
                setCommission(response.backtest?.commission || 0.002);
                setInitialCapital(response.backtest?.initial_capital || 10000);

                // Live trading settings
                if (response.live_trading) {
                    setLiveDataSource(response.live_trading.data_source || "alpaca");
                    setDefaultBroker(response.live_trading.default_broker || "paper");
                    setAutoConnect(response.live_trading.auto_connect || false);

                    if (response.live_trading.broker) {
                        setBrokerConfigured(response.live_trading.broker.is_configured || false);
                        setBrokerBaseUrl(response.live_trading.broker.base_url || '');
                    }
                }

                // General settings
                setTheme(response.general?.theme || 'dark');
                setNotifications(response.general?.notifications ?? true);
                setAutoRefresh(response.general?.auto_refresh ?? true);
                setRefreshInterval(response.general?.refresh_interval || 30);
            }
        } catch (error) {
            console.error("Failed to fetch settings:", error);
        } finally {
            setIsLoading(false);
        }
    };

    const handleSave = async () => {
        setIsSaving(true);
        try {
            const updatePayload: any = {};

            // Always include current tab's settings
            if (activeTab === 'backtest') {
                updatePayload.backtest = {
                    data_source: backtestDataSource,
                    slippage: slippage,
                    commission: commission,
                    initial_capital: initialCapital
                };
            } else if (activeTab === 'live') {
                updatePayload.live_trading = {
                    data_source: liveDataSource,
                    default_broker: defaultBroker,
                    auto_connect: autoConnect,
                };

                // Only include broker credentials if they were entered
                if (brokerApiKey || brokerApiSecret || brokerBaseUrl) {
                    updatePayload.live_trading.broker = {
                        broker_type: defaultBroker,
                        api_key: brokerApiKey || undefined,
                        api_secret: brokerApiSecret || undefined,
                        base_url: brokerBaseUrl || undefined
                    };
                }
            } else if (activeTab === 'general') {
                updatePayload.general = {
                    theme: theme,
                    notifications: notifications,
                    auto_refresh: autoRefresh,
                    refresh_interval: refreshInterval
                };
            }

            await settings.update(updatePayload);

            // Clear sensitive data from state after save
            if (activeTab === 'live' && (brokerApiKey || brokerApiSecret)) {
                setBrokerApiKey('');
                setBrokerApiSecret('');
                setBrokerConfigured(true);
            }

            setTestResult({
                type: 'success',
                message: 'Settings saved successfully!'
            });

            setTimeout(() => setTestResult(null), 3000);
        } catch (error) {
            console.error("Failed to save settings:", error);
            setTestResult({
                type: 'error',
                message: 'Failed to save settings. Please try again.'
            });
        } finally {
            setIsSaving(false);
        }
    };

    const handleTestConnection = async () => {
        setIsTesting(true);
        setTestResult(null);

        try {
            const response = await settings.testBrokerConnection();

            setTestResult({
                type: response.status === 'connected' ? 'success' : 'warning',
                message: response.message || 'Connection test completed',
                details: response
            });
        } catch (error: any) {
            setTestResult({
                type: 'error',
                message: error.response?.data?.detail || 'Connection test failed'
            });
        } finally {
            setIsTesting(false);
        }
    };

    const handleDeleteCredentials = async () => {
        if (!confirm('Are you sure you want to delete your broker credentials?')) {
            return;
        }

        try {
            await settings.deleteBrokerCredentials();
            setBrokerApiKey('');
            setBrokerApiSecret('');
            setBrokerBaseUrl('');
            setBrokerConfigured(false);

            setTestResult({
                type: 'success',
                message: 'Broker credentials deleted successfully'
            });

            setTimeout(() => setTestResult(null), 3000);
        } catch (error) {
            setTestResult({
                type: 'error',
                message: 'Failed to delete credentials'
            });
        }
    };

    const tabs = [
        { id: 'backtest', label: 'Backtest Settings', icon: TrendingDown },
        { id: 'live', label: 'Live Trading', icon: Zap },
        { id: 'alerts', label: 'Alerts', icon: Bell },
        { id: 'general', label: 'General', icon: Globe }
    ];

    const dataProviders = [
        {
            value: 'yahoo',
            label: 'Yahoo Finance',
            description: 'Free, reliable historical data',
            recommended: true
        },
        {
            value: 'alpaca',
            label: 'Alpaca Markets',
            description: 'Real-time market data'
        },
        {
            value: 'polygon',
            label: 'Polygon.io',
            description: 'Professional-grade data'
        },
        {
            value: 'iex',
            label: 'IEX Cloud',
            description: 'Exchange-sourced data'
        }
    ];

    const brokerOptions = [
        {
            value: 'paper',
            label: 'Paper Trading',
            description: 'Simulated trading (no real money)',
            recommended: true
        },
        {
            value: 'alpaca',
            label: 'Alpaca Markets',
            description: 'Commission-free stock & crypto trading'
        },
        {
            value: 'interactive_brokers',
            label: 'Interactive Brokers',
            description: 'Professional trading platform'
        },
        {
            value: 'td_ameritrade',
            label: 'TD Ameritrade',
            description: 'Full-service broker'
        }
    ];

    if (isLoading) {
        return (
            <div className="flex items-center justify-center h-96">
                <Loader2 className="animate-spin text-indigo-500" size={32} />
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h2 className="text-3xl font-bold text-slate-100 tracking-tight">
                        Settings <span className="text-slate-400 font-normal">& Preferences</span>
                    </h2>
                    <p className="text-slate-400 text-sm mt-1 font-medium">
                        Configure your trading environment and preferences
                    </p>
                </div>
                {activeTab !== 'alerts' && (
                    <button
                        onClick={handleSave}
                        disabled={isSaving}
                        className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white rounded-xl font-semibold text-sm transition-all shadow-xl shadow-violet-500/20 disabled:opacity-70"
                    >
                        {isSaving ? <Loader2 size={18} className="animate-spin" /> : <Save size={18} strokeWidth={2} />}
                        <span>{isSaving ? 'Saving...' : 'Save Changes'}</span>
                    </button>
                )}
            </div>

            {/* Status Message */}
            {testResult && (
                <div className={`p-4 rounded-xl border flex items-center gap-3 ${
                    testResult.type === 'success' ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' :
                        testResult.type === 'warning' ? 'bg-amber-500/10 border-amber-500/30 text-amber-400' :
                            'bg-red-500/10 border-red-500/30 text-red-400'
                }`}>
                    {testResult.type === 'success' ? <Check size={20} /> : <AlertCircle size={20} />}
                    <div className="flex-1">
                        <div className="font-semibold">{testResult.message}</div>
                        {testResult.details && (
                            <div className="text-sm opacity-80 mt-1">
                                Broker: {testResult.details.broker} | Status: {testResult.details.status}
                            </div>
                        )}
                    </div>
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Sidebar Tabs */}
                <div className="lg:col-span-1">
                    <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-2">
                        {tabs.map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-left transition-all ${
                                    activeTab === tab.id
                                        ? 'bg-violet-600 text-white shadow-lg'
                                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                                }`}
                            >
                                <tab.icon size={20} strokeWidth={2} />
                                <span className="font-semibold text-sm">{tab.label}</span>
                            </button>
                        ))}
                    </div>
                </div>

                {/* Content Area */}
                <div className="lg:col-span-3">
                    {activeTab === 'alerts' ? (
                        <AlertManagement />
                    ) : (
                        <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6 space-y-8">

                            {/* BACKTEST SETTINGS TAB */}
                            {activeTab === 'backtest' && (
                                <>
                                    <div>
                                        <h3 className="text-xl font-bold text-slate-100 mb-2 flex items-center gap-2">
                                            <Database size={24} className="text-violet-400" />
                                            Backtest Configuration
                                        </h3>
                                        <p className="text-slate-400 text-sm">
                                            Configure data sources and default parameters for backtesting
                                        </p>
                                    </div>

                                    {/* Data Source */}
                                    <div className="space-y-3">
                                        <label className="text-sm font-bold text-slate-300 flex items-center gap-2">
                                            <Database size={16} />
                                            Historical Data Provider
                                        </label>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                            {dataProviders.map((provider) => (
                                                <button
                                                    key={provider.value}
                                                    onClick={() => setBacktestDataSource(provider.value)}
                                                    className={`p-4 rounded-xl border-2 text-left transition-all ${
                                                        backtestDataSource === provider.value
                                                            ? 'border-violet-500 bg-violet-500/10'
                                                            : 'border-slate-700 bg-slate-800/30 hover:border-slate-600'
                                                    }`}
                                                >
                                                    <div className="flex items-start justify-between mb-2">
                                                        <div className="font-semibold text-slate-200">{provider.label}</div>
                                                        {provider.recommended && (
                                                            <span className="px-2 py-0.5 bg-emerald-500/20 text-emerald-400 rounded text-xs font-bold">
                                                                Recommended
                                                            </span>
                                                        )}
                                                        {backtestDataSource === provider.value && (
                                                            <Check className="text-violet-400" size={20} />
                                                        )}
                                                    </div>
                                                    <p className="text-xs text-slate-400">{provider.description}</p>
                                                </button>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Transaction Costs */}
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                        <div className="space-y-2">
                                            <label className="text-sm font-bold text-slate-300 flex items-center gap-2">
                                                <Percent size={16} />
                                                Slippage
                                            </label>
                                            <input
                                                type="number"
                                                step="0.0001"
                                                value={slippage}
                                                onChange={(e) => setSlippage(parseFloat(e.target.value))}
                                                className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-slate-200 focus:border-violet-500 outline-none"
                                            />
                                            <p className="text-xs text-slate-500">
                                                Default: 0.001 (0.1%)
                                            </p>
                                        </div>

                                        <div className="space-y-2">
                                            <label className="text-sm font-bold text-slate-300 flex items-center gap-2">
                                                <CreditCard size={16} />
                                                Commission
                                            </label>
                                            <input
                                                type="number"
                                                step="0.0001"
                                                value={commission}
                                                onChange={(e) => setCommission(parseFloat(e.target.value))}
                                                className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-slate-200 focus:border-violet-500 outline-none"
                                            />
                                            <p className="text-xs text-slate-500">
                                                Default: 0.002 (0.2%)
                                            </p>
                                        </div>
                                    </div>

                                    {/* Initial Capital */}
                                    <div className="space-y-2">
                                        <label className="text-sm font-bold text-slate-300 flex items-center gap-2">
                                            <Landmark size={16} />
                                            Default Initial Capital
                                        </label>
                                        <input
                                            type="number"
                                            step="1000"
                                            value={initialCapital}
                                            onChange={(e) => setInitialCapital(parseFloat(e.target.value))}
                                            className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-slate-200 focus:border-violet-500 outline-none"
                                        />
                                        <p className="text-xs text-slate-500">
                                            Default starting capital for new backtests
                                        </p>
                                    </div>
                                </>
                            )}

                            {/* LIVE TRADING TAB */}
                            {activeTab === 'live' && (
                                <>
                                    <div>
                                        <h3 className="text-xl font-bold text-slate-100 mb-2 flex items-center gap-2">
                                            <Zap size={24} className="text-emerald-400" />
                                            Live Trading Configuration
                                        </h3>
                                        <p className="text-slate-400 text-sm">
                                            Configure broker connections and real-time data sources
                                        </p>
                                    </div>

                                    {/* Warning Banner */}
                                    <div className="p-4 bg-amber-500/10 border border-amber-500/30 rounded-xl flex items-start gap-3">
                                        <ShieldCheck className="text-amber-400 shrink-0 mt-0.5" size={20} />
                                        <div className="text-sm">
                                            <div className="font-bold text-amber-400 mb-1">Security Notice</div>
                                            <p className="text-slate-300">
                                                Your broker credentials are encrypted and stored securely.
                                                We never share your credentials with third parties.
                                            </p>
                                        </div>
                                    </div>

                                    {/* Live Data Source */}
                                    <div className="space-y-3">
                                        <label className="text-sm font-bold text-slate-300 flex items-center gap-2">
                                            <Database size={16} />
                                            Real-Time Data Provider
                                        </label>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                            {dataProviders.slice(1).map((provider) => (
                                                <button
                                                    key={provider.value}
                                                    onClick={() => setLiveDataSource(provider.value)}
                                                    className={`p-4 rounded-xl border-2 text-left transition-all ${
                                                        liveDataSource === provider.value
                                                            ? 'border-emerald-500 bg-emerald-500/10'
                                                            : 'border-slate-700 bg-slate-800/30 hover:border-slate-600'
                                                    }`}
                                                >
                                                    <div className="flex items-start justify-between mb-2">
                                                        <div className="font-semibold text-slate-200">{provider.label}</div>
                                                        {liveDataSource === provider.value && (
                                                            <Check className="text-emerald-400" size={20} />
                                                        )}
                                                    </div>
                                                    <p className="text-xs text-slate-400">{provider.description}</p>
                                                </button>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Broker Selection */}
                                    <div className="space-y-3">
                                        <label className="text-sm font-bold text-slate-300 flex items-center gap-2">
                                            <Link size={16} />
                                            Default Broker
                                        </label>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                            {brokerOptions.map((broker) => (
                                                <button
                                                    key={broker.value}
                                                    onClick={() => setDefaultBroker(broker.value)}
                                                    className={`p-4 rounded-xl border-2 text-left transition-all ${
                                                        defaultBroker === broker.value
                                                            ? 'border-emerald-500 bg-emerald-500/10'
                                                            : 'border-slate-700 bg-slate-800/30 hover:border-slate-600'
                                                    }`}
                                                >
                                                    <div className="flex items-start justify-between mb-2">
                                                        <div className="font-semibold text-slate-200">{broker.label}</div>
                                                        {broker.recommended && (
                                                            <span className="px-2 py-0.5 bg-emerald-500/20 text-emerald-400 rounded text-xs font-bold">
                                                                Safe
                                                            </span>
                                                        )}
                                                        {defaultBroker === broker.value && (
                                                            <Check className="text-emerald-400" size={20} />
                                                        )}
                                                    </div>
                                                    <p className="text-xs text-slate-400">{broker.description}</p>
                                                </button>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Broker Credentials */}
                                    {defaultBroker !== 'paper' && (
                                        <div className="space-y-4 pt-4 border-t border-slate-700">
                                            <div className="flex items-center justify-between">
                                                <label className="text-sm font-bold text-slate-300 flex items-center gap-2">
                                                    <Key size={16} />
                                                    Broker Credentials
                                                </label>
                                                {brokerConfigured && (
                                                    <div className="flex items-center gap-2">
                                                        <span className="px-2 py-1 bg-emerald-500/20 text-emerald-400 rounded text-xs font-bold flex items-center gap-1">
                                                            <Check size={12} /> Configured
                                                        </span>
                                                        <button
                                                            onClick={handleDeleteCredentials}
                                                            className="text-xs text-red-400 hover:text-red-300 underline"
                                                        >
                                                            Delete
                                                        </button>
                                                    </div>
                                                )}
                                            </div>

                                            <div className="space-y-3">
                                                <div>
                                                    <label className="text-xs text-slate-400 mb-1 block">API Key</label>
                                                    <input
                                                        type="text"
                                                        value={brokerApiKey}
                                                        onChange={(e) => setBrokerApiKey(e.target.value)}
                                                        placeholder={brokerConfigured ? "••••••••••••••••" : "Enter your API key"}
                                                        className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-slate-200 focus:border-emerald-500 outline-none font-mono text-sm"
                                                    />
                                                </div>

                                                <div>
                                                    <label className="text-xs text-slate-400 mb-1 block">API Secret</label>
                                                    <div className="relative">
                                                        <input
                                                            type={showApiSecret ? "text" : "password"}
                                                            value={brokerApiSecret}
                                                            onChange={(e) => setBrokerApiSecret(e.target.value)}
                                                            placeholder={brokerConfigured ? "••••••••••••••••" : "Enter your API secret"}
                                                            className="w-full px-4 py-3 pr-12 bg-slate-800 border border-slate-700 rounded-xl text-slate-200 focus:border-emerald-500 outline-none font-mono text-sm"
                                                        />
                                                        <button
                                                            type="button"
                                                            onClick={() => setShowApiSecret(!showApiSecret)}
                                                            className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                                                        >
                                                            {showApiSecret ? <EyeOff size={18} /> : <Eye size={18} />}
                                                        </button>
                                                    </div>
                                                </div>

                                                <div>
                                                    <label className="text-xs text-slate-400 mb-1 block">
                                                        Base URL (Optional)
                                                    </label>
                                                    <input
                                                        type="text"
                                                        value={brokerBaseUrl}
                                                        onChange={(e) => setBrokerBaseUrl(e.target.value)}
                                                        placeholder="https://paper-api.alpaca.markets"
                                                        className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-slate-200 focus:border-emerald-500 outline-none font-mono text-sm"
                                                    />
                                                </div>
                                            </div>

                                            {/* Test Connection Button */}
                                            {brokerConfigured && (
                                                <button
                                                    onClick={handleTestConnection}
                                                    disabled={isTesting}
                                                    className="w-full py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-xl font-semibold flex items-center justify-center gap-2 transition-all disabled:opacity-50"
                                                >
                                                    {isTesting ? (
                                                        <Loader2 size={18} className="animate-spin" />
                                                    ) : (
                                                        <TestTube size={18} />
                                                    )}
                                                    {isTesting ? 'Testing...' : 'Test Connection'}
                                                </button>
                                            )}
                                        </div>
                                    )}

                                    {/* Auto-Connect Option */}
                                    <div className="space-y-2">
                                        <label className="flex items-center gap-3 cursor-pointer p-4 bg-slate-800/30 rounded-xl border border-slate-700 hover:border-slate-600 transition-all">
                                            <input
                                                type="checkbox"
                                                checked={autoConnect}
                                                onChange={(e) => setAutoConnect(e.target.checked)}
                                                className="w-5 h-5 rounded"
                                            />
                                            <div className="flex-1">
                                                <div className="text-sm font-semibold text-slate-200">
                                                    Auto-connect on startup
                                                </div>
                                                <div className="text-xs text-slate-400 mt-0.5">
                                                    Automatically connect to broker when you open the app
                                                </div>
                                            </div>
                                        </label>
                                    </div>
                                </>
                            )}

                            {/* GENERAL SETTINGS TAB */}
                            {activeTab === 'general' && (
                                <>
                                    <div>
                                        <h3 className="text-xl font-bold text-slate-100 mb-2 flex items-center gap-2">
                                            <Globe size={24} className="text-blue-400" />
                                            General Preferences
                                        </h3>
                                        <p className="text-slate-400 text-sm">
                                            Customize your app experience
                                        </p>
                                    </div>

                                    {/* Theme */}
                                    <div className="space-y-3">
                                        <label className="text-sm font-bold text-slate-300">Theme</label>
                                        <div className="grid grid-cols-2 gap-3">
                                            <button
                                                onClick={() => setTheme('dark')}
                                                className={`p-4 rounded-xl border-2 text-left transition-all ${
                                                    theme === 'dark'
                                                        ? 'border-blue-500 bg-blue-500/10'
                                                        : 'border-slate-700 bg-slate-800/30 hover:border-slate-600'
                                                }`}
                                            >
                                                <div className="flex items-center justify-between mb-2">
                                                    <div className="font-semibold text-slate-200">Dark Mode</div>
                                                    {theme === 'dark' && <Check className="text-blue-400" size={20} />}
                                                </div>
                                                <p className="text-xs text-slate-400">Easy on the eyes</p>
                                            </button>
                                            <button
                                                onClick={() => setTheme('light')}
                                                className={`p-4 rounded-xl border-2 text-left transition-all ${
                                                    theme === 'light'
                                                        ? 'border-blue-500 bg-blue-500/10'
                                                        : 'border-slate-700 bg-slate-800/30 hover:border-slate-600'
                                                }`}
                                            >
                                                <div className="flex items-center justify-between mb-2">
                                                    <div className="font-semibold text-slate-200">Light Mode</div>
                                                    {theme === 'light' && <Check className="text-blue-400" size={20} />}
                                                </div>
                                                <p className="text-xs text-slate-400">Classic look</p>
                                            </button>
                                        </div>
                                    </div>

                                    {/* Notifications */}
                                    <div className="space-y-2">
                                        <label className="flex items-center gap-3 cursor-pointer p-4 bg-slate-800/30 rounded-xl border border-slate-700 hover:border-slate-600 transition-all">
                                            <input
                                                type="checkbox"
                                                checked={notifications}
                                                onChange={(e) => setNotifications(e.target.checked)}
                                                className="w-5 h-5 rounded"
                                            />
                                            <div className="flex-1">
                                                <div className="text-sm font-semibold text-slate-200">
                                                    Enable Notifications
                                                </div>
                                                <div className="text-xs text-slate-400 mt-0.5">
                                                    Receive alerts for important events
                                                </div>
                                            </div>
                                        </label>
                                    </div>

                                    {/* Auto Refresh */}
                                    <div className="space-y-2">
                                        <label className="flex items-center gap-3 cursor-pointer p-4 bg-slate-800/30 rounded-xl border border-slate-700 hover:border-slate-600 transition-all">
                                            <input
                                                type="checkbox"
                                                checked={autoRefresh}
                                                onChange={(e) => setAutoRefresh(e.target.checked)}
                                                className="w-5 h-5 rounded"
                                            />
                                            <div className="flex-1">
                                                <div className="text-sm font-semibold text-slate-200">
                                                    Auto-refresh Data
                                                </div>
                                                <div className="text-xs text-slate-400 mt-0.5">
                                                    Automatically update market data
                                                </div>
                                            </div>
                                        </label>
                                    </div>

                                    {/* Refresh Interval */}
                                    {autoRefresh && (
                                        <div className="space-y-2">
                                            <label className="text-sm font-bold text-slate-300 flex items-center gap-2">
                                                <RefreshCw size={16} />
                                                Refresh Interval (seconds)
                                            </label>
                                            <input
                                                type="number"
                                                min="5"
                                                max="300"
                                                step="5"
                                                value={refreshInterval}
                                                onChange={(e) => setRefreshInterval(parseInt(e.target.value))}
                                                className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-slate-200 focus:border-blue-500 outline-none"
                                            />
                                            <p className="text-xs text-slate-500">
                                                How often to refresh data (5-300 seconds)
                                            </p>
                                        </div>
                                    )}
                                </>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default SettingsPage;
