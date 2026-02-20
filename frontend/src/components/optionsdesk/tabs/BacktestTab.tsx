import {Calendar, ChevronDown, ChevronUp, DollarSign, LineChart as LineChartIcon, Loader2, Play, Settings, Sliders} from "lucide-react";
import {Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis} from "recharts";
import React, {useEffect, useMemo, useState} from "react";
import {STRATEGY_TEMPLATES} from "@/components/optionsdesk/contants/strategyTemplates";
import {BacktestConfig, BacktestResult, RecentTrades} from "@/types/all_types";

interface BackTestTabProps {
    backtestConfig: BacktestConfig,
    setBacktestConfig: React.Dispatch<React.SetStateAction<BacktestConfig>>,
    runStrategyBacktest: () => Promise<void>,
    backtestResults: BacktestResult,
    equityData: any,
    recentTrades: RecentTrades[],
    isLoading: boolean,
    expirationDates: string[],
    selectedSymbol: string,
}

/* ── Strategy-specific parameter definitions ──────────────────────── */

interface ParamField {
    key: string;
    label: string;
    type: "number" | "select";
    default: number | string;
    min?: number;
    max?: number;
    step?: number;
    suffix?: string;
    options?: { value: string; label: string }[];
    description?: string;
    rulesKey: "entry_rules" | "exit_rules";
}

const STRATEGY_PARAMS: Record<string, ParamField[]> = {
    covered_call: [
        {key: "otm_call_pct", label: "Call Strike (% OTM)", type: "number", default: 5, min: 1, max: 20, step: 1, suffix: "%", description: "How far OTM to sell the call", rulesKey: "entry_rules"},
        {key: "days_to_expiration", label: "Days to Expiration", type: "number", default: 30, min: 7, max: 90, step: 1, suffix: "days", rulesKey: "entry_rules"},
    ],
    cash_secured_put: [
        {key: "otm_put_pct", label: "Put Strike (% OTM)", type: "number", default: 5, min: 1, max: 20, step: 1, suffix: "%", description: "How far OTM to sell the put", rulesKey: "entry_rules"},
        {key: "days_to_expiration", label: "Days to Expiration", type: "number", default: 30, min: 7, max: 90, step: 1, suffix: "days", rulesKey: "entry_rules"},
    ],
    protective_put: [
        {key: "otm_put_pct", label: "Put Strike (% OTM)", type: "number", default: 5, min: 1, max: 15, step: 1, suffix: "%", description: "How far OTM for the protective put", rulesKey: "entry_rules"},
        {key: "days_to_expiration", label: "Days to Expiration", type: "number", default: 30, min: 7, max: 90, step: 1, suffix: "days", rulesKey: "entry_rules"},
    ],
    vertical_call_spread: [
        {key: "otm_call_pct", label: "Short Call (% OTM)", type: "number", default: 5, min: 1, max: 20, step: 1, suffix: "%", description: "Short call strike distance", rulesKey: "entry_rules"},
        {key: "days_to_expiration", label: "Days to Expiration", type: "number", default: 45, min: 14, max: 90, step: 1, suffix: "days", rulesKey: "entry_rules"},
    ],
    vertical_put_spread: [
        {key: "otm_put_pct", label: "Short Put (% OTM)", type: "number", default: 5, min: 1, max: 20, step: 1, suffix: "%", description: "Short put strike distance", rulesKey: "entry_rules"},
        {key: "days_to_expiration", label: "Days to Expiration", type: "number", default: 45, min: 14, max: 90, step: 1, suffix: "days", rulesKey: "entry_rules"},
    ],
    iron_condor: [
        {key: "otm_call_pct", label: "Call Wing (% OTM)", type: "number", default: 5, min: 2, max: 15, step: 1, suffix: "%", description: "Short call distance from price", rulesKey: "entry_rules"},
        {key: "otm_put_pct", label: "Put Wing (% OTM)", type: "number", default: 5, min: 2, max: 15, step: 1, suffix: "%", description: "Short put distance from price", rulesKey: "entry_rules"},
        {key: "days_to_expiration", label: "Days to Expiration", type: "number", default: 45, min: 14, max: 60, step: 1, suffix: "days", rulesKey: "entry_rules"},
    ],
    butterfly_spread: [
        {key: "otm_call_pct", label: "Wing Width (% OTM)", type: "number", default: 5, min: 2, max: 10, step: 1, suffix: "%", description: "Distance between strikes", rulesKey: "entry_rules"},
        {key: "days_to_expiration", label: "Days to Expiration", type: "number", default: 30, min: 14, max: 60, step: 1, suffix: "days", rulesKey: "entry_rules"},
    ],
    long_straddle: [
        {key: "days_to_expiration", label: "Days to Expiration", type: "number", default: 30, min: 7, max: 90, step: 1, suffix: "days", rulesKey: "entry_rules"},
    ],
    long_strangle: [
        {key: "otm_call_pct", label: "Call Strike (% OTM)", type: "number", default: 5, min: 2, max: 15, step: 1, suffix: "%", description: "OTM distance for call", rulesKey: "entry_rules"},
        {key: "otm_put_pct", label: "Put Strike (% OTM)", type: "number", default: 5, min: 2, max: 15, step: 1, suffix: "%", description: "OTM distance for put", rulesKey: "entry_rules"},
        {key: "days_to_expiration", label: "Days to Expiration", type: "number", default: 30, min: 7, max: 90, step: 1, suffix: "days", rulesKey: "entry_rules"},
    ],
    calendar_spread: [
        {key: "days_to_expiration", label: "Near-Term DTE", type: "number", default: 30, min: 14, max: 60, step: 1, suffix: "days", description: "Short leg expiration", rulesKey: "entry_rules"},
    ],
    diagonal_spread: [
        {key: "otm_call_pct", label: "Short Strike (% OTM)", type: "number", default: 5, min: 1, max: 15, step: 1, suffix: "%", description: "Short leg OTM distance", rulesKey: "entry_rules"},
        {key: "days_to_expiration", label: "Near-Term DTE", type: "number", default: 30, min: 14, max: 60, step: 1, suffix: "days", description: "Short leg expiration", rulesKey: "entry_rules"},
    ],
    collar: [
        {key: "otm_call_pct", label: "Call Strike (% OTM)", type: "number", default: 5, min: 1, max: 15, step: 1, suffix: "%", description: "Short call OTM distance", rulesKey: "entry_rules"},
        {key: "otm_put_pct", label: "Put Strike (% OTM)", type: "number", default: 5, min: 1, max: 15, step: 1, suffix: "%", description: "Long put OTM distance", rulesKey: "entry_rules"},
        {key: "days_to_expiration", label: "Days to Expiration", type: "number", default: 30, min: 7, max: 90, step: 1, suffix: "days", rulesKey: "entry_rules"},
    ],
    ratio_spread: [
        {key: "otm_call_pct", label: "Short Strike (% OTM)", type: "number", default: 5, min: 2, max: 15, step: 1, suffix: "%", description: "Short legs OTM distance", rulesKey: "entry_rules"},
        {key: "days_to_expiration", label: "Days to Expiration", type: "number", default: 30, min: 14, max: 60, step: 1, suffix: "days", rulesKey: "entry_rules"},
    ],
};

/* ── Shared entry/exit parameters (available for all strategies) ─── */

const ENTRY_SIGNAL_PARAMS: ParamField[] = [
    {
        key: "signal", label: "Entry Signal", type: "select", default: "regular",
        options: [
            {value: "regular", label: "Regular Interval"},
            {value: "rsi", label: "RSI Signal"},
            {value: "moving_average", label: "Moving Average Crossover"},
        ],
        rulesKey: "entry_rules",
    },
    {key: "entry_frequency", label: "Entry Frequency", type: "number", default: 30, min: 5, max: 120, step: 5, suffix: "days", description: "Enter every N trading days", rulesKey: "entry_rules"},
];

const EXIT_PARAMS: ParamField[] = [
    {key: "profit_target", label: "Profit Target", type: "number", default: 50, min: 10, max: 200, step: 5, suffix: "%", description: "Close at this % gain", rulesKey: "exit_rules"},
    {key: "loss_limit", label: "Loss Limit", type: "number", default: 50, min: 10, max: 100, step: 5, suffix: "%", description: "Close at this % loss", rulesKey: "exit_rules"},
    {key: "dte_exit", label: "DTE Exit", type: "number", default: 7, min: 1, max: 30, step: 1, suffix: "days", description: "Close when DTE reaches this", rulesKey: "exit_rules"},
];



/* ── Helper: convert UI values → backend-ready rules ─────────────── */
// Called on every param change so backtestConfig always has correct rules

function convertToBackendRules(
    rawEntry: Record<string, any>,
    rawExit: Record<string, any>,
): { entry_rules: Record<string, any>; exit_rules: Record<string, any> } {
    const entry: Record<string, any> = {};
    const exit: Record<string, any> = {};

    // Convert entry rules — UI stores % as integers (e.g. 5 = 5%)
    for (const [k, v] of Object.entries(rawEntry)) {
        if (k === "otm_call_pct") entry[k] = 1 + Number(v) / 100;       // 5 → 1.05
        else if (k === "otm_put_pct") entry[k] = 1 - Number(v) / 100;   // 5 → 0.95
        else if (k === "expiration") { if (v) entry[k] = String(v); }    // pass-through date string
        else if (k === "signal") entry[k] = String(v);                   // keep as string
        else entry[k] = typeof v === "string" ? v : Number(v);
    }

    // Convert exit rules — UI stores % as integers (e.g. 50 = 50%)
    for (const [k, v] of Object.entries(rawExit)) {
        if (k === "profit_target") exit[k] = Number(v) / 100;            // 50 → 0.5
        else if (k === "loss_limit") exit[k] = -(Number(v) / 100);       // 50 → -0.5
        else exit[k] = Number(v);
    }

    return {entry_rules: entry, exit_rules: exit};
}


/* ── Param input component ────────────────────────────────────────── */

function ParamInput({field, value, onChange}: {
    field: ParamField;
    value: number | string;
    onChange: (key: string, val: number | string, rulesKey: "entry_rules" | "exit_rules") => void;
}) {
    if (field.type === "select") {
        return (
            <div>
                <label className="block text-[10px] font-bold text-slate-500 uppercase mb-1.5">{field.label}</label>
                <select
                    value={value}
                    onChange={(e) => onChange(field.key, e.target.value, field.rulesKey)}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 outline-none focus:border-amber-500/60"
                >
                    {field.options?.map(o => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                    ))}
                </select>
            </div>
        );
    }

    return (
        <div>
            <label className="block text-[10px] font-bold text-slate-500 uppercase mb-1.5">{field.label}</label>
            <div className="relative">
                <input
                    type="number"
                    value={value}
                    min={field.min}
                    max={field.max}
                    step={field.step}
                    onChange={(e) => onChange(field.key, parseFloat(e.target.value) || 0, field.rulesKey)}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 outline-none focus:border-amber-500/60 pr-10"
                />
                {field.suffix && (
                    <span className="absolute right-2.5 top-1/2 -translate-y-1/2 text-[10px] text-slate-500 font-medium">{field.suffix}</span>
                )}
            </div>
            {field.description && (
                <p className="text-[9px] text-slate-600 mt-0.5">{field.description}</p>
            )}
        </div>
    );
}


/* ── Collapsible section ──────────────────────────────────────────── */

function CollapsibleSection({title, defaultOpen = true, children}: {
    title: string;
    defaultOpen?: boolean;
    children: React.ReactNode;
}) {
    const [open, setOpen] = useState(defaultOpen);
    return (
        <div className="border border-slate-800/60 rounded-lg overflow-hidden">
            <button
                onClick={() => setOpen(!open)}
                className="w-full flex items-center justify-between px-3 py-2 bg-slate-800/30 hover:bg-slate-800/50 transition-colors"
            >
                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">{title}</span>
                {open ? <ChevronUp size={12} className="text-slate-500"/> :
                    <ChevronDown size={12} className="text-slate-500"/>}
            </button>
            {open && <div className="p-3 space-y-3">{children}</div>}
        </div>
    );
}


/* ── Main Component ───────────────────────────────────────────────── */

const BackTestTab: React.FC<BackTestTabProps> = ({
                                                     backtestConfig,
                                                     setBacktestConfig,
                                                     runStrategyBacktest,
                                                     backtestResults,
                                                     equityData,
                                                     recentTrades,
                                                     isLoading,
                                                     expirationDates,
                                                     selectedSymbol,
                                                 }: BackTestTabProps) => {

    const selectedTemplate = useMemo(
        () => STRATEGY_TEMPLATES.find(s => s.id === backtestConfig.strategy_type),
        [backtestConfig.strategy_type]
    );

    const strategyParams = useMemo(
        () => STRATEGY_PARAMS[backtestConfig.strategy_type] || [],
        [backtestConfig.strategy_type]
    );

    // Local UI state stores human-readable values (e.g. 5 for 5%, 50 for 50%)
    // backtestConfig.entry_rules/exit_rules stores backend-ready values (e.g. 1.05, 0.5)
    const [uiEntry, setUiEntry] = useState<Record<string, any>>({});
    const [uiExit, setUiExit] = useState<Record<string, any>>({});

    // When strategy changes, seed UI defaults and sync backend-ready values
    useEffect(() => {
        if (!backtestConfig.strategy_type) return;
        const params = STRATEGY_PARAMS[backtestConfig.strategy_type] || [];
        const newUiEntry: Record<string, any> = {};
        const newUiExit: Record<string, any> = {};

        for (const p of params) {
            if (p.rulesKey === "entry_rules") newUiEntry[p.key] = p.default;
            else newUiExit[p.key] = p.default;
        }
        for (const p of ENTRY_SIGNAL_PARAMS) newUiEntry[p.key] = p.default;
        for (const p of EXIT_PARAMS) newUiExit[p.key] = p.default;

        setUiEntry(newUiEntry);
        setUiExit(newUiExit);

        // Convert to backend format and store
        const {entry_rules, exit_rules} = convertToBackendRules(newUiEntry, newUiExit);
        setBacktestConfig(prev => ({...prev, entry_rules, exit_rules}));
    }, [backtestConfig.strategy_type]);

    const handleParamChange = (key: string, val: number | string, rulesKey: "entry_rules" | "exit_rules") => {
        if (rulesKey === "entry_rules") {
            const updated = {...uiEntry, [key]: val};
            setUiEntry(updated);
            const {entry_rules} = convertToBackendRules(updated, uiExit);
            setBacktestConfig(prev => ({...prev, entry_rules}));
        } else {
            const updated = {...uiExit, [key]: val};
            setUiExit(updated);
            const {exit_rules} = convertToBackendRules(uiEntry, updated);
            setBacktestConfig(prev => ({...prev, exit_rules}));
        }
    };

    // getParamValue reads from UI state (human-readable values)
    const getUiValue = (field: ParamField): number | string => {
        const store = field.rulesKey === "entry_rules" ? uiEntry : uiExit;
        if (field.key in store) return store[field.key];
        return field.default;
    };

    return (
        <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* ── Left Panel: Parameters ──────────────── */}
                <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-5 h-fit space-y-5">
                    <h3 className="text-lg font-bold text-slate-200 flex items-center gap-2">
                        <Settings size={20} className="text-amber-400"/>
                        Parameters
                    </h3>

                    {/* Strategy selector */}
                    <div>
                        <label className="block text-xs font-bold text-slate-500 uppercase mb-2">Strategy</label>
                        <select
                            value={backtestConfig.strategy_type}
                            onChange={(e) => setBacktestConfig(prev => ({
                                ...prev,
                                strategy_type: e.target.value,
                                entry_rules: {},
                                exit_rules: {},
                            }))}
                            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 outline-none focus:border-amber-500"
                        >
                            <option value="">Select Strategy</option>
                            {STRATEGY_TEMPLATES.map(s => (
                                <option key={s.id} value={s.id}>{s.name}</option>
                            ))}
                        </select>
                    </div>

                    {/* Strategy info badge */}
                    {selectedTemplate && (
                        <div className="flex items-center gap-2 flex-wrap">
                            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                                selectedTemplate.risk === 'Low' ? 'bg-emerald-500/10 text-emerald-400' :
                                    selectedTemplate.risk === 'Moderate' || selectedTemplate.risk === 'Defined' ? 'bg-amber-500/10 text-amber-400' :
                                        selectedTemplate.risk === 'Limited' ? 'bg-blue-500/10 text-blue-400' :
                                            'bg-red-500/10 text-red-400'
                            }`}>
                                Risk: {selectedTemplate.risk}
                            </span>
                            <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-slate-700/50 text-slate-400">
                                {selectedTemplate.sentiment}
                            </span>
                            <span className="text-[10px] text-slate-500">
                                {selectedTemplate.legs.length} leg{selectedTemplate.legs.length > 1 ? 's' : ''}
                            </span>
                        </div>
                    )}

                    {/* Expiration selector — real chain expirations */}
                    {backtestConfig.strategy_type && expirationDates.length > 0 && (
                        <div>
                            <label className="block text-[10px] font-bold text-slate-500 uppercase mb-1.5 flex items-center gap-1">
                                <Calendar size={10}/>
                                Option Expiration ({selectedSymbol})
                            </label>
                            <select
                                value={uiEntry.expiration || ''}
                                onChange={(e) => handleParamChange("expiration", e.target.value, "entry_rules")}
                                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 outline-none focus:border-amber-500/60"
                            >
                                <option value="">Auto (DTE-based)</option>
                                {expirationDates.map(exp => {
                                    const d = new Date(exp);
                                    const dte = Math.round((d.getTime() - Date.now()) / (1000 * 60 * 60 * 24));
                                    return (
                                        <option key={exp} value={exp}>
                                            {d.toLocaleDateString('en-US', {month: 'short', day: 'numeric', year: 'numeric'})} ({dte}d)
                                        </option>
                                    );
                                })}
                            </select>
                            <p className="text-[9px] text-slate-600 mt-0.5">Leave as Auto to use DTE parameter, or pick a specific expiration</p>
                        </div>
                    )}

                    {/* Date range & capital */}
                    <div className="grid grid-cols-2 gap-3">
                        <div>
                            <label className="block text-[10px] font-bold text-slate-500 uppercase mb-1.5">Start</label>
                            <input
                                type="date"
                                value={backtestConfig.start_date}
                                onChange={(e) => setBacktestConfig(prev => ({...prev, start_date: e.target.value}))}
                                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-xs text-slate-200"
                            />
                        </div>
                        <div>
                            <label className="block text-[10px] font-bold text-slate-500 uppercase mb-1.5">End</label>
                            <input
                                type="date"
                                value={backtestConfig.end_date}
                                onChange={(e) => setBacktestConfig(prev => ({...prev, end_date: e.target.value}))}
                                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-xs text-slate-200"
                            />
                        </div>
                    </div>

                    <div>
                        <label className="block text-[10px] font-bold text-slate-500 uppercase mb-1.5">Initial Capital</label>
                        <div className="relative">
                            <DollarSign className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-500" size={13}/>
                            <input
                                type="number"
                                value={backtestConfig.initial_capital}
                                onChange={(e) => setBacktestConfig(prev => ({
                                    ...prev,
                                    initial_capital: parseFloat(e.target.value) || 0,
                                }))}
                                className="w-full pl-8 pr-4 py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-xs text-slate-200"
                            />
                        </div>
                    </div>

                    {/* ── Strategy-Specific Parameters ──── */}
                    {strategyParams.length > 0 && (
                        <CollapsibleSection title="Strategy Parameters">
                            {strategyParams.map(field => (
                                <ParamInput
                                    key={field.key}
                                    field={field}
                                    value={getUiValue(field)}
                                    onChange={handleParamChange}
                                />
                            ))}
                        </CollapsibleSection>
                    )}

                    {/* ── Entry Signal Configuration ───── */}
                    {backtestConfig.strategy_type && (
                        <CollapsibleSection title="Entry Signal" defaultOpen={false}>
                            {ENTRY_SIGNAL_PARAMS.map(field => (
                                <ParamInput
                                    key={field.key}
                                    field={field}
                                    value={getUiValue(field)}
                                    onChange={handleParamChange}
                                />
                            ))}
                        </CollapsibleSection>
                    )}

                    {/* ── Exit Rules Configuration ─────── */}
                    {backtestConfig.strategy_type && (
                        <CollapsibleSection title="Exit Rules" defaultOpen={false}>
                            {EXIT_PARAMS.map(field => (
                                <ParamInput
                                    key={field.key}
                                    field={field}
                                    value={getUiValue(field)}
                                    onChange={handleParamChange}
                                />
                            ))}
                        </CollapsibleSection>
                    )}

                    {/* Execute button */}
                    <button
                        onClick={runStrategyBacktest}
                        disabled={isLoading || !backtestConfig.strategy_type}
                        className="w-full py-3 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold rounded-xl flex items-center justify-center gap-2 shadow-lg shadow-emerald-900/20 disabled:opacity-50 transition-all"
                    >
                        {isLoading ? <Loader2 size={18} className="animate-spin"/> : <Play size={18}/>}
                        Execute Backtest
                    </button>
                </div>

                {/* ── Right Panel: Results ─────────────────── */}
                <div className="lg:col-span-2 space-y-6">
                    {backtestResults ? (
                        <>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                {[
                                    {
                                        label: 'Return',
                                        val: `${backtestResults.total_return.toFixed(2)}%`,
                                        color: backtestResults.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'
                                    },
                                    {
                                        label: 'Win Rate',
                                        val: `${backtestResults.win_rate.toFixed(1)}%`,
                                        color: 'text-amber-400'
                                    },
                                    {
                                        label: 'Profit Factor',
                                        val: backtestResults.profit_factor.toFixed(2),
                                        color: 'text-blue-400'
                                    },
                                    {
                                        label: 'Sharpe',
                                        val: backtestResults.sharpe_ratio.toFixed(2),
                                        color: 'text-purple-400'
                                    }
                                ].map((kpi, idx) => (
                                    <div key={idx}
                                         className="bg-slate-900/50 p-4 rounded-xl border border-slate-800/50">
                                        <div
                                            className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">{kpi.label}</div>
                                        <div className={`text-xl font-bold mt-1 ${kpi.color}`}>{kpi.val}</div>
                                    </div>
                                ))}
                            </div>

                            <div className="bg-slate-900/50 p-6 rounded-xl border border-slate-800/50">
                                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-6">Performance
                                    &
                                    Drawdown</h4>
                                <div className="space-y-2">
                                    <div className="h-[280px]">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <AreaChart data={equityData}
                                                       margin={{top: 0, right: 10, left: 0, bottom: 0}}>
                                                <defs>
                                                    <linearGradient id="colorEquityOpt" x1="0" y1="0" x2="0" y2="1">
                                                        <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.2}/>
                                                        <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                                                    </linearGradient>
                                                </defs>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false}/>
                                                <XAxis dataKey="date" hide/>
                                                <YAxis stroke="#475569" fontSize={10}
                                                       tickFormatter={(v: number) => `${(v / 1000).toFixed(0)}k`}/>
                                                <Tooltip
                                                    contentStyle={{
                                                        backgroundColor: '#0f172a',
                                                        border: '1px solid #1e293b'
                                                    }}/>
                                                <Area type="monotone" dataKey="equity" stroke="#f59e0b" strokeWidth={2}
                                                      fill="url(#colorEquityOpt)"/>
                                            </AreaChart>
                                        </ResponsiveContainer>
                                    </div>
                                    <div className="h-[100px] border-t border-slate-800 pt-2">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <AreaChart data={equityData}
                                                       margin={{top: 0, right: 10, left: 0, bottom: 0}}>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false}/>
                                                <XAxis dataKey="date" stroke="#475569" fontSize={9}/>
                                                <YAxis stroke="#475569" fontSize={9} domain={['auto', 0]}
                                                       tickFormatter={(v: number) => `${v}%`}/>
                                                <Tooltip
                                                    contentStyle={{
                                                        backgroundColor: '#0f172a',
                                                        border: '1px solid #334155'
                                                    }}/>
                                                <Area type="monotone" dataKey="drawdown" stroke="#ef4444" fill="#ef4444"
                                                      fillOpacity={0.1}/>
                                            </AreaChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>
                            </div>

                            <div className="bg-slate-900/50 rounded-xl border border-slate-800/50 overflow-hidden">
                                <div className="p-4 border-b border-slate-800 flex justify-between items-center">
                                    <h4 className="text-xs font-bold text-slate-400 uppercase">Recent Trades</h4>
                                    <span
                                        className="text-[10px] text-slate-500 font-medium">{recentTrades.length} Total Trades</span>
                                </div>
                                <div className="max-h-[300px] overflow-y-auto">
                                    <table className="w-full text-left text-[11px]">
                                        <thead className="sticky top-0 bg-slate-900 text-slate-500 font-bold uppercase">
                                        <tr>
                                            <th className="px-4 py-3">Date</th>
                                            <th className="px-4 py-3">Symbol</th>
                                            <th className="px-4 py-3">Strategy</th>
                                            <th className="px-4 py-3 text-right">Net P&L</th>
                                        </tr>
                                        </thead>
                                        <tbody className="divide-y divide-slate-800/50">
                                        {recentTrades.map((trade, i) => (
                                            <tr key={i} className="hover:bg-slate-800/30 transition-colors">
                                                <td className="px-4 py-3 text-slate-400">{trade.time}</td>
                                                <td className="px-4 py-3 font-bold text-slate-200">{trade.symbol}</td>
                                                <td className="px-4 py-3 text-slate-500">{trade.strategy}</td>
                                                <td className={`px-4 py-3 text-right font-bold ${trade.profit >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                    {trade.profit >= 0 ? '+' : ''}{trade.profit.toFixed(2)}
                                                </td>
                                            </tr>
                                        ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </>
                    ) : (
                        <div
                            className="h-full flex flex-col items-center justify-center text-slate-600 bg-slate-900/20 border-2 border-dashed border-slate-800/50 rounded-2xl py-20">
                            <Sliders size={48} className="mb-4 opacity-10"/>
                            <p className="text-sm font-medium">Select a strategy and configure parameters to begin</p>
                            <p className="text-xs text-slate-700 mt-1">Each strategy has unique parameters you can tune</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

export default BackTestTab;
