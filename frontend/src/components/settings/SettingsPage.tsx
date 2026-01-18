import React, { useEffect, useState } from "react";
import {
    Bell, Check,
    Copy,
    CreditCard,
    Database,
    Globe, Info,
    Key,
    Landmark,
    Percent,
    RefreshCw,
    Save,
    ShieldCheck, TrendingDown,
    User,
    Loader2
} from "lucide-react";
import { settings } from "@/utils/api";
import { UserSettings, BacktestSettings, GeneralSettings } from "@/types/api";

const SettingsPage = () => {
    const [activeTab, setActiveTab] = useState('profile');
    const [isLoading, setIsLoading] = useState(true);
    const [isSaving, setIsSaving] = useState(false);

    // State for settings
    const [userSettings, setUserSettings] = useState<UserSettings | null>(null);
    const [dataSource, setDataSource] = useState('yahoo');
    const [slippage, setSlippage] = useState(0.001);
    const [commission, setCommission] = useState(0.002);

    useEffect(() => {
        fetchSettings();
    }, []);

    const fetchSettings = async () => {
        setIsLoading(true);
        try {
            const response = await settings.get();
            if (response.data) {
                setUserSettings(response.data);
                // Initialize local state from fetched settings
                setDataSource(response.data.backtest.data_source);
                setSlippage(response.data.backtest.slippage);
                setCommission(response.data.backtest.commission);
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
            await settings.update({
                backtest: {
                    data_source: dataSource,
                    slippage: slippage,
                    commission: commission
                }
            });
            // Optionally show success toast
        } catch (error) {
            console.error("Failed to save settings:", error);
        } finally {
            setIsSaving(false);
        }
    };

    const tabs = [
        { id: 'profile', label: 'Profile', icon: User },
        { id: 'data', label: 'Data Connections', icon: Database },
        { id: 'security', label: 'Security & API', icon: Key },
        { id: 'notifications', label: 'Notifications', icon: Bell },
        { id: 'billing', label: 'Billing', icon: CreditCard },
    ];

    const dataProviders = [
        { id: 'yahoo', name: 'Yahoo Finance', type: 'Public', status: 'Available', color: 'text-blue-400' },
        { id: 'alpaca', name: 'Alpaca Markets', type: 'Brokerage', status: 'Connected', color: 'text-yellow-500' },
        { id: 'ibkr', name: 'Interactive Brokers', type: 'Institutional', status: 'Setup Required', color: 'text-red-500' },
        { id: 'polygon', name: 'Polygon.io', type: 'Premium Feed', status: 'Available', color: 'text-purple-400' }
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
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h2 className="text-3xl font-bold text-slate-100 tracking-tight">
                        Settings <span className="text-slate-400 font-normal">& Preferences</span>
                    </h2>
                    <p className="text-slate-400 text-sm mt-1 font-medium">Manage your account and system configuration</p>
                </div>
                <button
                    onClick={handleSave}
                    disabled={isSaving}
                    className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white rounded-xl font-semibold text-sm transition-all shadow-xl shadow-violet-500/20 disabled:opacity-70"
                >
                    {isSaving ? <Loader2 size={18} className="animate-spin" /> : <Save size={18} strokeWidth={2} />}
                    <span>{isSaving ? 'Saving...' : 'Save Changes'}</span>
                </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Sidebar */}
                <div className="lg:col-span-1 space-y-1">
                    {tabs.map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`w-full flex items-center space-x-3 px-4 py-3 rounded-xl transition-all ${activeTab === tab.id
                                    ? 'bg-slate-800/70 text-slate-100 border border-slate-700/50'
                                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                                }`}
                        >
                            <tab.icon size={18} className={activeTab === tab.id ? 'text-violet-400' : ''} strokeWidth={2} />
                            <span className="text-sm font-semibold">{tab.label}</span>
                        </button>
                    ))}
                </div>

                {/* Main Panel */}
                <div className="lg:col-span-3">
                    <div className="bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-2xl overflow-hidden shadow-xl">
                        {activeTab === 'profile' && (
                            <div className="p-8 space-y-6">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div className="space-y-2">
                                        <label className="text-xs font-semibold text-slate-400 tracking-wide">Display Name</label>
                                        <input
                                            type="text"
                                            defaultValue="Institutional Trader"
                                            className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none transition-all text-sm text-slate-200"
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <label className="text-xs font-semibold text-slate-400 tracking-wide">Email Address</label>
                                        <input
                                            type="email"
                                            defaultValue="trader@institution.com"
                                            className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none transition-all text-sm text-slate-200"
                                        />
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <label className="text-xs font-semibold text-slate-400 tracking-wide">Bio</label>
                                    <textarea
                                        rows={4}
                                        defaultValue="Quantitative trader specializing in algorithmic strategies and portfolio optimization."
                                        className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none transition-all text-sm text-slate-200 resize-none"
                                    />
                                </div>
                            </div>
                        )}

                        {activeTab === 'security' && (
                            <div className="p-8 space-y-8">
                                <div className="p-4 rounded-xl bg-violet-500/10 border border-violet-500/30 flex items-start space-x-4">
                                    <ShieldCheck className="text-violet-400 shrink-0 mt-0.5" size={24} />
                                    <div>
                                        <p className="text-sm font-semibold text-slate-200 mb-1">API Key Security</p>
                                        <p className="text-xs text-slate-400 leading-relaxed">All API keys are encrypted with AES-256 and never stored in plain text. Enable IP restrictions for enhanced security.</p>
                                    </div>
                                </div>
                                <div className="space-y-4">
                                    <label className="text-xs font-semibold text-slate-400 tracking-wide flex items-center justify-between">
                                        API Secret Key
                                        <span className="text-violet-400 cursor-pointer hover:underline font-semibold">Regenerate</span>
                                    </label>
                                    <div className="flex space-x-2">
                                        <input
                                            type="password"
                                            value="••••••••••••••••••••••••••••"
                                            readOnly
                                            className="flex-1 px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-sm font-mono text-slate-400"
                                        />
                                        <button className="p-3 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50 rounded-xl transition-colors">
                                            <Copy size={18} strokeWidth={2} />
                                        </button>
                                        <button className="p-3 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50 rounded-xl transition-colors">
                                            <RefreshCw size={18} strokeWidth={2} />
                                        </button>
                                    </div>
                                </div>
                                <div className="pt-6 border-t border-slate-700/50 space-y-4">
                                    <label className="text-xs font-semibold text-slate-400 tracking-wide">Two-Factor Authentication</label>
                                    <button className="w-full flex items-center justify-between p-4 bg-slate-800/40 border border-slate-700/50 rounded-xl hover:bg-slate-800/60 transition-all">
                                        <div className="flex items-center space-x-3">
                                            <ShieldCheck className="text-emerald-400" size={20} strokeWidth={2} />
                                            <div className="text-left">
                                                <p className="text-sm font-semibold text-slate-200">Enabled</p>
                                                <p className="text-xs text-slate-500">Using authenticator app</p>
                                            </div>
                                        </div>
                                        <span className="text-xs font-semibold text-slate-400 hover:text-slate-300">Configure</span>
                                    </button>
                                </div>
                            </div>
                        )}

                        {activeTab === 'data' && (
                            <div className="p-8 space-y-8 animate-in fade-in slide-in-from-right-4 duration-500">
                                {/* Section 1: Data Source Selection (Existing logic) */}
                                <div className="space-y-4">
                                    <div className="flex justify-between items-center">
                                        <div>
                                            <h3 className="text-lg font-bold text-slate-100">Market Data Orchestration</h3>
                                            <p className="text-xs text-slate-400">Manage real-time and historical data providers.</p>
                                        </div>
                                        <div className="flex items-center space-x-2 px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
                                            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                                            <span className="text-[10px] font-bold text-emerald-400 uppercase">System Ready</span>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                        {dataProviders.map((provider) => (
                                            <div
                                                key={provider.id}
                                                onClick={() => setDataSource(provider.id)}
                                                className={`p-4 rounded-xl border-2 cursor-pointer transition-all relative overflow-hidden group ${dataSource === provider.id
                                                        ? 'border-violet-500 bg-violet-500/5'
                                                        : 'border-slate-700/50 bg-slate-800/30 hover:border-slate-600'
                                                    }`}
                                            >
                                                <div className="flex justify-between items-start mb-2 relative z-10">
                                                    <div className={`p-2 rounded-lg bg-slate-900 ${provider.color}`}>
                                                        <Globe size={20} />
                                                    </div>
                                                    {dataSource === provider.id && (
                                                        <Check size={16} className="text-violet-500" />
                                                    )}
                                                </div>
                                                <h4 className="text-sm font-bold text-slate-100 relative z-10">{provider.name}</h4>
                                                <p className="text-[10px] text-slate-500 font-bold uppercase tracking-wider relative z-10">{provider.type}</p>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* Section 2: Execution Engine Configuration (The New Part) */}
                                <div className="pt-8 border-t border-slate-700/50 space-y-6">
                                    <div>
                                        <h3 className="text-lg font-bold text-slate-100">Execution Modeling</h3>
                                        <p className="text-xs text-slate-400">Configure realistic constraints for backtesting and live execution.</p>
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                        {/* Slippage Modeling */}
                                        <div className="space-y-4 p-5 bg-slate-800/20 border border-slate-700/50 rounded-2xl">
                                            <div className="flex items-center justify-between">
                                                <div className="flex items-center space-x-2">
                                                    <TrendingDown size={18} className="text-orange-400" />
                                                    <label className="text-xs font-bold text-slate-300 uppercase tracking-widest">Slippage Model</label>
                                                </div>
                                                <Info size={14} className="text-slate-600 hover:text-slate-400 cursor-help" />
                                            </div>

                                            <div className="space-y-3">
                                                <div className="flex justify-between text-[10px] font-bold">
                                                    <span className="text-slate-500 uppercase">Estimated Impact</span>
                                                    <span className="text-orange-400">{(slippage * 100).toFixed(2)}% per trade</span>
                                                </div>
                                                <input
                                                    type="range"
                                                    min="0"
                                                    max="0.05"
                                                    step="0.0001"
                                                    value={slippage}
                                                    onChange={(e) => setSlippage(parseFloat(e.target.value))}
                                                    className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-violet-500"
                                                />
                                                <p className="text-[10px] text-slate-500 leading-relaxed italic">
                                                    Simulates the difference between the expected price and the price at which the trade is actually executed.
                                                </p>
                                            </div>
                                        </div>

                                        {/* Commission Structure */}
                                        <div className="space-y-4 p-5 bg-slate-800/20 border border-slate-700/50 rounded-2xl">
                                            <div className="flex items-center justify-between">
                                                <div className="flex items-center space-x-2">
                                                    <Landmark size={18} className="text-blue-400" />
                                                    <label className="text-xs font-bold text-slate-300 uppercase tracking-widest">Brokerage Fees</label>
                                                </div>
                                                <Percent size={14} className="text-slate-600" />
                                            </div>

                                            <div className="space-y-3">
                                                <label className="text-[10px] font-bold text-slate-500 uppercase">Flat Fee / Percentage</label>
                                                <div className="relative">
                                                    <span className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 font-mono text-xs">$</span>
                                                    <input
                                                        type="number"
                                                        value={commission}
                                                        onChange={(e) => setCommission(parseFloat(e.target.value))}
                                                        step="0.0001"
                                                        className="w-full pl-8 pr-4 py-2.5 bg-slate-900/50 border border-slate-700/50 rounded-xl outline-none focus:border-violet-500 text-sm text-slate-200 font-mono"
                                                    />
                                                </div>
                                                <p className="text-[10px] text-slate-500 leading-relaxed">
                                                    Applied to every filled order. Set to 0.00 for commission-free brokers (e.g., Alpaca, Robinhood).
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {activeTab === 'notifications' && (
                            <div className="p-8 space-y-4">
                                {[
                                    { label: 'Trade Executions', desc: 'Get notified when orders are filled', default: true },
                                    { label: 'Margin Alerts', desc: 'Critical alerts for account health', default: true },
                                    { label: 'Market Regime Changes', desc: 'Notifications when market conditions shift', default: false },
                                    { label: 'Strategy Performance', desc: 'Weekly performance summaries', default: true }
                                ].map((item, idx) => (
                                    <div key={idx} className="flex items-center justify-between p-4 bg-slate-800/40 border border-slate-700/50 rounded-xl hover:bg-slate-800/60 transition-all">
                                        <div>
                                            <p className="text-sm font-semibold text-slate-200">{item.label}</p>
                                            <p className="text-xs text-slate-500 mt-0.5">{item.desc}</p>
                                        </div>
                                        <input
                                            type="checkbox"
                                            defaultChecked={item.default}
                                            className="w-5 h-5 accent-violet-600 rounded cursor-pointer"
                                        />
                                    </div>
                                ))}
                            </div>
                        )}

                        {activeTab === 'billing' && (
                            <div className="p-8 space-y-6">
                                <div className="p-6 bg-gradient-to-br from-violet-500/10 to-fuchsia-500/10 border border-violet-500/30 rounded-xl">
                                    <div className="flex justify-between items-start mb-4">
                                        <div>
                                            <p className="text-xs font-semibold text-violet-400 tracking-wider mb-2">CURRENT PLAN</p>
                                            <h4 className="text-2xl font-bold text-slate-100">Institutional</h4>
                                            <p className="text-slate-400 text-sm mt-1">Full platform access with priority support</p>
                                        </div>
                                        <div className="text-right">
                                            <p className="text-3xl font-bold text-slate-100">$499</p>
                                            <p className="text-xs text-slate-500 font-medium">per month</p>
                                        </div>
                                    </div>
                                    <button className="w-full px-4 py-2.5 bg-white/10 hover:bg-white/20 border border-white/20 rounded-lg transition-all text-sm font-semibold text-slate-200">
                                        Manage Subscription
                                    </button>
                                </div>
                                <div className="space-y-3">
                                    <label className="text-xs font-semibold text-slate-400 tracking-wide">Payment Method</label>
                                    <div className="p-4 bg-slate-800/40 border border-slate-700/50 rounded-xl flex items-center justify-between">
                                        <div className="flex items-center space-x-3">
                                            <CreditCard className="text-slate-500" size={24} strokeWidth={2} />
                                            <div>
                                                <p className="text-sm font-semibold text-slate-200">•••• •••• •••• 4242</p>
                                                <p className="text-xs text-slate-500">Expires 12/25</p>
                                            </div>
                                        </div>
                                        <button className="text-xs font-semibold text-violet-400 hover:text-violet-300">Update</button>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default SettingsPage;