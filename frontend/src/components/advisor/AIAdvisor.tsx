'use client'
import React, {useState} from 'react';
import {
    AlertTriangle,
    ArrowRight,
    Award,
    BarChart4,
    BrainCircuit,
    CheckCircle2,
    ChevronRight,
    CircuitBoard,
    Clock,
    Coins,
    Crown,
    DollarSign,
    Download,
    Eye,
    Flame,
    Gem,
    Info,
    LineChart,
    Lock,
    Rocket,
    RotateCcw,
    Settings,
    Shield,
    ShieldCheck,
    Sparkles,
    Target as TargetIcon,
    TrendingUp,
    TrendingUp as TrendingUpIcon,
    Users,
    Wallet,
    Zap
} from "lucide-react";
import {
    Area,
    AreaChart,
    CartesianGrid,
    PolarAngleAxis,
    PolarGrid,
    PolarRadiusAxis,
    Radar,
    RadarChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from 'recharts';

import {advisor} from "@/utils/api";
import {Recommendation} from "@/types/all_types";

const AIAdvisor = () => {
    const [view, setView] = useState<'quiz' | 'results' | 'detail'>('quiz');
    const [isGenerating, setIsGenerating] = useState(false);
    const [selectedStrategy, setSelectedStrategy] = useState<number | null>(null);
    const [quizAnswers, setQuizAnswers] = useState({
        goal: 'Wealth Growth',
        risk: 'Moderate',
        experience: 'Intermediate',
        capital: 10000,
        timeHorizon: '1-3 years',
        markets: ['Stocks', 'Crypto']
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const [recommendations, setRecommendations] = useState<any[]>([]);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const [radarData, setRadarData] = useState<any[]>([]);

    // Icon Mapping Helper

    const getIcon = (iconName: string) => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const map: any = {Flame, CircuitBoard, Coins, Zap};
        return map[iconName] || Zap;
    };

    // Theme Icon Mapping Helper
    const getThemeIcon = (iconName: string, className: string) => {
        const Icon = getIcon(iconName);
        return <Icon className={className} size={20}/>;
    };

    const handleGenerate = async () => {
        setIsGenerating(true);
        try {
            const res = await advisor.generateGuide(quizAnswers);
            if (res.data) {
                // Map backend response to UI format
                const recs = res.data.recommendations.map((r: Recommendation) => ({
                    ...r,
                    icon: getIcon(r.icon), // Map string to component
                    theme: {
                        ...r.theme,
                        icon: getThemeIcon(r.theme.icon, r.theme.text) // Map theme icon
                    }
                }));
                setRecommendations(recs);
                setRadarData(res.data.radar_data);
                setView('results');
            }
        } catch (e) {
            console.error("Advisor error", e);
        } finally {
            setIsGenerating(false);
        }
    };

    const handleSelectStrategy = (id: number) => {
        setSelectedStrategy(id);
        setView('detail');
    };

    const handleBackToResults = () => {
        setView('results');
        setSelectedStrategy(null);
    };

    const RiskBadge = ({level, theme}: { level: string, theme: string }) => {
        const config = {
            high: {color: 'text-red-400', bg: 'bg-red-500/10', border: 'border-red-500/20'},
            medium: {color: 'text-teal-400', bg: 'bg-teal-500/10', border: 'border-teal-500/20'},
            low: {color: 'text-blue-400', bg: 'bg-blue-500/10', border: 'border-blue-500/20'}
        };

        const cfg = config[level as keyof typeof config] || config.medium;

        return (
            <span
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold uppercase ${cfg.bg} ${cfg.border} border ${cfg.color}`}>
                <Shield size={12}/>
                {level} Risk
            </span>
        );
    };

    type ButtonVariant = 'primary' | 'secondary' | 'outline';

    interface PurpleButtonProps {
        children: React.ReactNode;
        icon?: React.ReactElement;
        onClick?: () => void;
        disabled?: boolean;
        variant?: ButtonVariant;
        className?: string;
    }

    const PurpleButton = ({
                              children,
                              icon,
                              onClick,
                              variant = 'primary',
                              className = ''
                          }: PurpleButtonProps) => {

        const variants: Record<ButtonVariant, string> = {
            primary: 'bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 shadow-lg shadow-purple-600/30',
            secondary: 'bg-gradient-to-r from-purple-500/20 to-indigo-500/20 hover:from-purple-500/30 hover:to-indigo-500/30 border border-purple-500/30',
            outline: 'bg-transparent border-2 border-purple-500/50 hover:bg-purple-500/10'
        };

        return (
            <button
                onClick={onClick}
                // 3. TypeScript now knows variants[variant] is a valid string
                className={`py-3 px-6 rounded-xl font-semibold text-white transition-all duration-300 flex items-center justify-center gap-3 ${variants[variant]} ${className}`}
            >
                {icon && React.cloneElement(icon, 18)}
                {children}
            </button>
        );
    };

    // Quiz View with Colorful Design
    if (view === 'quiz') {
        return (
            <div className="space-y-6 animate-in fade-in duration-500 max-w-4xl mx-auto">
                <div
                    className="bg-gradient-to-br from-slate-900 via-slate-900/95 to-slate-900/90 border border-slate-800/50 rounded-2xl p-8 shadow-2xl shadow-slate-900/50 backdrop-blur-sm">
                    {/* Colorful Header */}
                    <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 mb-10">
                        <div className="flex items-center gap-4">
                            <div className="relative">
                                <div
                                    className="absolute inset-0 bg-gradient-to-r from-purple-600 to-pink-600 rounded-2xl blur-lg opacity-50"/>
                                <div className="relative p-3 bg-slate-900 rounded-2xl border border-purple-500/30">
                                    <BrainCircuit className="text-purple-400" size={32}/>
                                </div>
                            </div>
                            <div>
                                <h3 className="text-2xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent">
                                    AI Strategy Intelligence
                                </h3>
                                <p className="text-sm text-slate-400">Find your perfect trading strategy match</p>
                            </div>
                        </div>
                        <div
                            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-900/50 to-blue-900/50 rounded-xl border border-purple-700/30">
                            <div
                                className="w-2 h-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full animate-pulse"/>
                            <span className="text-xs font-medium text-slate-300">AI Processing Active</span>
                        </div>
                    </div>

                    {/* Colorful Progress Bar */}
                    <div className="mb-8">
                        <div className="flex justify-between text-xs font-medium text-slate-500 mb-2">
                            <span>Strategy Personalization</span>
                            <span>1/3</span>
                        </div>
                        <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                            <div
                                className="h-full w-1/3 bg-gradient-to-r from-purple-500 via-pink-500 to-orange-500 rounded-full transition-all duration-500"/>
                        </div>
                    </div>

                    {/* Quiz Questions */}
                    <div className="space-y-8">
                        {/* Investment Goals - Colorful Cards */}
                        <div className="space-y-4">
                            <div className="flex items-center gap-2">
                                <div
                                    className="w-6 h-6 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600 flex items-center justify-center">
                                    <span className="text-xs font-bold text-white">1</span>
                                </div>
                                <label className="text-sm font-semibold text-slate-200">Primary Investment Goal</label>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                                {[
                                    {
                                        icon: Rocket,
                                        label: "Wealth Growth",
                                        desc: "Maximum capital appreciation",
                                        color: "from-red-500 to-orange-500"
                                    },
                                    {
                                        icon: TrendingUpIcon,
                                        label: "Income",
                                        desc: "Regular monthly returns",
                                        color: "from-emerald-500 to-teal-500"
                                    },
                                    {
                                        icon: Shield,
                                        label: "Protection",
                                        desc: "Capital preservation",
                                        color: "from-blue-500 to-indigo-500"
                                    }
                                ].map((goal, idx) => (
                                    <button
                                        key={idx}
                                        className={`p-4 rounded-xl border-2 transition-all ${quizAnswers.goal === goal.label ? `border-opacity-100 bg-gradient-to-br ${goal.color}/20` : 'border-slate-800 bg-slate-900/50 hover:border-slate-700'}`}
                                        style={quizAnswers.goal === goal.label ? {
                                            borderImage: `linear-gradient(45deg, ${goal.color.replace('from-', '').replace(' to-', ', ')}) 1`
                                        } : {}}
                                        onClick={() => setQuizAnswers({...quizAnswers, goal: goal.label})}
                                    >
                                        <goal.icon
                                            className={`${quizAnswers.goal === goal.label ? 'text-white' : 'text-slate-500'}`}
                                            size={20}/>
                                        <p className={`text-sm font-semibold ${quizAnswers.goal === goal.label ? 'text-white' : 'text-slate-300'} mt-2`}>{goal.label}</p>
                                        <p className="text-xs text-slate-500 mt-1">{goal.desc}</p>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Risk Tolerance - Colorful Scale */}
                        <div className="space-y-4">
                            <div className="flex items-center gap-2">
                                <div
                                    className="w-6 h-6 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600 flex items-center justify-center">
                                    <span className="text-xs font-bold text-white">2</span>
                                </div>
                                <label className="text-sm font-semibold text-slate-200">Risk Tolerance</label>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-4 gap-2">
                                {[
                                    {level: "Conservative", color: "from-blue-500 to-cyan-500"},
                                    {level: "Moderate", color: "from-emerald-500 to-teal-500"},
                                    {level: "Aggressive", color: "from-orange-500 to-red-500"},
                                    {level: "Very Aggressive", color: "from-red-500 to-pink-500"}
                                ].map((risk, idx) => (
                                    <button
                                        key={idx}
                                        className={`py-3 px-4 rounded-lg text-sm font-medium transition-all ${quizAnswers.risk === risk.level ? `bg-gradient-to-r ${risk.color} text-white shadow-lg` : 'bg-slate-900/50 text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'}`}
                                        onClick={() => setQuizAnswers({...quizAnswers, risk: risk.level})}
                                    >
                                        {risk.level}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Experience Level - Colorful Icons */}
                        <div className="space-y-4">
                            <div className="flex items-center gap-2">
                                <div
                                    className="w-6 h-6 rounded-lg bg-gradient-to-r from-purple-600 to-pink-600 flex items-center justify-center">
                                    <span className="text-xs font-bold text-white">3</span>
                                </div>
                                <label className="text-sm font-semibold text-slate-200">Trading Experience</label>
                            </div>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                {['Beginner', 'Intermediate', 'Advanced', 'Expert'].map((exp, idx) => {
                                    const colors = [
                                        'from-blue-400 to-cyan-400',
                                        'from-emerald-400 to-teal-400',
                                        'from-orange-400 to-amber-400',
                                        'from-purple-400 to-pink-400'
                                    ];
                                    return (
                                        <button
                                            key={idx}
                                            className={`p-4 rounded-xl border transition-all ${quizAnswers.experience === exp ? `border-opacity-100 bg-gradient-to-br ${colors[idx]}/20` : 'border-slate-800 bg-slate-900/50 hover:border-slate-700'}`}
                                            onClick={() => setQuizAnswers({...quizAnswers, experience: exp})}
                                        >
                                            <div className="flex flex-col items-center gap-2">
                                                <div
                                                    className={`p-2 rounded-lg ${quizAnswers.experience === exp ? `bg-gradient-to-br ${colors[idx]}` : 'bg-slate-800'}`}>
                                                    <Crown size={16}
                                                           className={quizAnswers.experience === exp ? 'text-white' : 'text-slate-500'}/>
                                                </div>
                                                <span
                                                    className={`text-sm font-medium ${quizAnswers.experience === exp ? 'text-white' : 'text-slate-400'}`}>{exp}</span>
                                            </div>
                                        </button>
                                    );
                                })}
                            </div>
                        </div>

                        {/* Capital & Markets */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-4">
                                <label className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                                    <Wallet size={16} className="text-purple-400"/>
                                    Starting Capital
                                </label>
                                <div className="relative group">
                                    <div
                                        className="absolute left-4 top-1/2 -translate-y-1/2 text-purple-500 group-focus-within:text-purple-400 transition-colors">
                                        $
                                    </div>
                                    <input
                                        type="number"
                                        value={quizAnswers.capital}
                                        onChange={(e) => setQuizAnswers({
                                            ...quizAnswers,
                                            capital: parseInt(e.target.value)
                                        })}
                                        className="w-full bg-slate-950/50 border-2 border-slate-800 rounded-xl py-4 pl-10 pr-4 text-lg font-semibold text-slate-100 focus:border-purple-500 focus:ring-2 focus:ring-purple-500/30 outline-none transition-all"
                                    />
                                    <div className="absolute right-4 top-1/2 -translate-y-1/2 text-xs text-slate-500">
                                        USD
                                    </div>
                                </div>
                                <div className="flex gap-2">
                                    {[5000, 10000, 25000, 50000].map(amount => (
                                        <button
                                            key={amount}
                                            onClick={() => setQuizAnswers({...quizAnswers, capital: amount})}
                                            className="px-3 py-1.5 text-xs bg-slate-800 rounded-lg hover:bg-slate-700 transition-colors text-slate-300"
                                        >
                                            ${amount.toLocaleString()}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div className="space-y-4">
                                <label className="text-sm font-semibold text-slate-200">Preferred Markets</label>
                                <div className="flex flex-wrap gap-2">
                                    {[
                                        {name: 'Stocks', color: 'from-blue-500 to-cyan-500'},
                                        {name: 'Crypto', color: 'from-orange-500 to-amber-500'},
                                        {name: 'Forex', color: 'from-emerald-500 to-teal-500'},
                                        {name: 'Commodities', color: 'from-yellow-500 to-orange-500'},
                                        {name: 'Options', color: 'from-purple-500 to-pink-500'},
                                        {name: 'Futures', color: 'from-red-500 to-pink-500'}
                                    ].map((market, idx) => (
                                        <button
                                            key={market.name}
                                            onClick={() => {
                                                const markets = quizAnswers.markets.includes(market.name)
                                                    ? quizAnswers.markets.filter(m => m !== market.name)
                                                    : [...quizAnswers.markets, market.name];
                                                setQuizAnswers({...quizAnswers, markets});
                                            }}
                                            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${quizAnswers.markets.includes(market.name) ? `bg-gradient-to-r ${market.color} text-white shadow-lg` : 'bg-slate-900/50 text-slate-400 hover:bg-slate-800/50'}`}
                                        >
                                            {market.name}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Purple Generate Button */}
                    <PurpleButton
                        onClick={handleGenerate}
                        disabled={isGenerating}
                        icon={<Sparkles size={20}/>}
                        className={`w-full mt-10 py-4 ${isGenerating ? 'animate-pulse' : ''}`}
                    >
                        {isGenerating ? 'Analyzing Market Patterns...' : 'Generate AI Strategy'}
                    </PurpleButton>

                    <p className="text-center text-xs text-slate-500 mt-4 flex items-center justify-center gap-2">
                        <Lock size={12}/>
                        🔒 Your data is encrypted and never shared with third parties
                    </p>
                </div>
            </div>
        );
    }

    // Strategy Detail View
    if (view === 'detail' && selectedStrategy) {
        const strategy = recommendations.find(r => r.id === selectedStrategy);
        if (!strategy) {
            handleBackToResults();
            return null;
        }

        return (
            <div className="space-y-8 animate-in slide-in-from-bottom-4 duration-700 max-w-6xl mx-auto">
                {/* Back Navigation */}
                <button
                    onClick={handleBackToResults}
                    className="flex items-center gap-2 text-sm text-slate-400 hover:text-purple-400 transition-colors group"
                >
                    <ChevronRight className="rotate-180 group-hover:-translate-x-1 transition-transform" size={16}/>
                    Back to All Strategies
                </button>

                {/* Strategy Header */}
                <div className={`${strategy.theme.bg} border ${strategy.theme.border} rounded-2xl p-8`}>
                    <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
                        <div>
                            <div className="flex items-center gap-3 mb-3">
                                <RiskBadge level={strategy.risk_level} theme={strategy.theme}/>
                                <div
                                    className="flex items-center gap-1 px-3 py-1 bg-gradient-to-r from-purple-600/20 to-pink-600/20 rounded-lg border border-purple-500/30">
                                    <Gem size={12} className="text-purple-400"/>
                                    <span className="text-xs font-bold text-purple-400">AI Recommended</span>
                                </div>
                            </div>
                            <h2 className={`text-3xl font-bold ${strategy.theme.text}`}>{strategy.name}</h2>
                            <p className="text-lg text-slate-300 mb-4">{strategy.tagline}</p>
                            <p className="text-slate-400 max-w-3xl">{strategy.description}</p>
                        </div>
                        <div
                            className="text-center bg-slate-900/50 p-6 rounded-2xl border border-slate-800 min-w-[140px]">
                            <p className="text-5xl font-black bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                                {strategy.fit_score}%
                            </p>
                            <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mt-1">AI Fit
                                Score</p>
                        </div>
                    </div>

                    {/* Tags */}
                    <div className="flex flex-wrap gap-2 mt-6">
                        {strategy.tags.map((tag: string, idx: number) => (
                            <span key={idx}
                                  className="px-3 py-1.5 bg-slate-900/50 rounded-lg text-xs font-medium text-slate-300 border border-slate-800">
                                {tag}
                            </span>
                        ))}
                    </div>
                </div>

                {/* Colorful Stats Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[
                        {
                            icon: TrendingUp,
                            label: "Expected Return",
                            value: strategy.expected_return,
                            color: "text-emerald-400",
                            bg: "bg-emerald-500/10"
                        },
                        {
                            icon: Users,
                            label: "Similar Traders",
                            value: strategy.similar_traders,
                            color: "text-blue-400",
                            bg: "bg-blue-500/10"
                        },
                        {
                            icon: Clock,
                            label: "Time Commitment",
                            value: strategy.time_commitment,
                            color: "text-amber-400",
                            bg: "bg-amber-500/10"
                        },
                        {
                            icon: Award,
                            label: "Success Rate",
                            value: strategy.success_rate,
                            color: "text-purple-400",
                            bg: "bg-purple-500/10"
                        }
                    ].map((stat, idx) => (
                        <div key={idx} className={`${stat.bg} border border-slate-800 rounded-xl p-5`}>
                            <div className="flex items-center gap-3 mb-3">
                                <div className={`p-2 rounded-lg ${stat.bg}`}>
                                    <stat.icon className={stat.color} size={20}/>
                                </div>
                                <span className="text-xs font-bold text-slate-500 uppercase">{stat.label}</span>
                            </div>
                            <p className={`text-2xl font-bold ${stat.color}`}>{stat.value}</p>
                        </div>
                    ))}
                </div>

                {/* Main Content Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Left Column - Performance Chart */}
                    <div className="lg:col-span-2 space-y-6">
                        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
                            <h4 className="text-sm font-semibold text-slate-200 mb-6 flex items-center gap-2">
                                <LineChart size={18} className={strategy.theme.text}/>
                                Performance Analysis
                            </h4>
                            <div className="h-[300px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={strategy.performance_data}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                                        <XAxis dataKey="month" stroke="#64748b" fontSize={12}/>
                                        <YAxis stroke="#64748b" fontSize={12} tickFormatter={(value) => `${value}%`}/>
                                        <Tooltip
                                            contentStyle={{
                                                backgroundColor: '#0f172a',
                                                border: '1px solid #1e293b',
                                                borderRadius: '8px',
                                                color: '#e2e8f0'
                                            }}
                                            formatter={(value) => [`${value}%`, 'Return']}
                                        />
                                        <Area
                                            type="monotone"
                                            dataKey="return"
                                            stroke={strategy.theme.primary}
                                            fill={`url(#${strategy.id}Gradient)`}
                                            strokeWidth={2}
                                        />
                                        <defs>
                                            <linearGradient id={`${strategy.id}Gradient`} x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor={strategy.theme.primary} stopOpacity={0.8}/>
                                                <stop offset="95%" stopColor={strategy.theme.primary} stopOpacity={0}/>
                                            </linearGradient>
                                        </defs>
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Pros & Cons */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div
                                className="bg-gradient-to-br from-emerald-900/20 to-teal-900/20 border border-emerald-800/30 rounded-2xl p-6">
                                <h4 className="text-sm font-semibold text-emerald-400 mb-4 flex items-center gap-2">
                                    <CheckCircle2 size={18}/>
                                    Advantages
                                </h4>
                                <ul className="space-y-3">
                                    {strategy.pros.map((pro: string, idx: number) => (
                                        <li key={idx} className="flex items-start gap-3">
                                            <div
                                                className="w-6 h-6 rounded-full bg-emerald-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                                                <CheckCircle2 size={12} className="text-emerald-500"/>
                                            </div>
                                            <span className="text-sm text-slate-300">{pro}</span>
                                        </li>
                                    ))}
                                </ul>
                            </div>

                            <div
                                className="bg-gradient-to-br from-red-900/20 to-orange-900/20 border border-red-800/30 rounded-2xl p-6">
                                <h4 className="text-sm font-semibold text-red-400 mb-4 flex items-center gap-2">
                                    <AlertTriangle size={18}/>
                                    Considerations
                                </h4>
                                <ul className="space-y-3">
                                    {strategy.cons.map((con: string, idx: number) => (
                                        <li key={idx} className="flex items-start gap-3">
                                            <div
                                                className="w-6 h-6 rounded-full bg-red-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                                                <AlertTriangle size={12} className="text-red-500"/>
                                            </div>
                                            <span className="text-sm text-slate-300">{con}</span>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        </div>
                    </div>

                    {/* Right Column - Actions & Details */}
                    <div className="space-y-6">
                        {/* Best For */}
                        <div
                            className="bg-gradient-to-br from-blue-900/20 to-indigo-900/20 border border-blue-800/30 rounded-2xl p-6">
                            <h4 className="text-sm font-semibold text-blue-400 mb-4">Ideal For Traders Who</h4>
                            <ul className="space-y-3">
                                {strategy.best_for.map((item: string, idx: number) => (
                                    <li key={idx} className="flex items-center gap-3 text-sm text-slate-400">
                                        <div
                                            className="w-1.5 h-1.5 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500"/>
                                        {item}
                                    </li>
                                ))}
                            </ul>
                        </div>

                        {/* Why This Fits */}
                        <div
                            className="bg-gradient-to-br from-purple-900/20 to-pink-900/20 border border-purple-800/30 rounded-2xl p-6">
                            <h4 className="text-sm font-semibold text-purple-400 mb-4">AI Match Analysis</h4>
                            <ul className="space-y-3">
                                {strategy.why.map((reason: string, idx: number) => (
                                    <li key={idx} className="flex items-start gap-3">
                                        <div
                                            className="w-6 h-6 rounded-full bg-gradient-to-br from-purple-600/30 to-pink-600/30 flex items-center justify-center flex-shrink-0 mt-0.5">
                                            <span className="text-xs font-bold text-purple-400">{idx + 1}</span>
                                        </div>
                                        <span className="text-sm text-slate-300">{reason}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>

                        {/* Purple Action Buttons */}
                        <div className="space-y-3">
                            <PurpleButton icon={<Settings/>}>
                                Configure Strategy
                            </PurpleButton>
                            <PurpleButton icon={<Eye/>} variant="secondary">
                                View Live Simulation
                            </PurpleButton>
                            <PurpleButton icon={<Download/>} variant="outline">
                                Download Strategy PDF
                            </PurpleButton>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    // Results View with Colorful Design
    return (
        <div className="space-y-8 animate-in slide-in-from-bottom-4 duration-700 max-w-6xl mx-auto">
            {/* Results Header */}
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                <div>
                    <div className="flex items-center gap-3 mb-2">
                        <div
                            className="p-2 bg-gradient-to-br from-purple-600/30 to-pink-600/30 rounded-lg border border-purple-500/30">
                            <BrainCircuit className="text-purple-400" size={20}/>
                        </div>
                        <div>
                            <h3 className="text-2xl md:text-3xl font-bold text-slate-100">AI Strategy Intelligence</h3>
                            <p className="text-slate-400">
                                Personalized strategies based on your unique profile
                            </p>
                        </div>
                    </div>
                </div>
                <div className="flex items-center gap-3">
                    <PurpleButton
                        onClick={() => setView('quiz')}
                        icon={<RotateCcw size={16}/>}
                        variant="secondary"
                    >
                        Retake Quiz
                    </PurpleButton>
                    <PurpleButton icon={<Download size={16}/>}>
                        Export Report
                    </PurpleButton>
                </div>
            </div>

            {/* User Profile Summary */}
            <div className="bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                <h4 className="text-xs font-black text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                    <Users size={14} className="text-purple-400"/>
                    Your Trading Profile
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[
                        {
                            icon: ShieldCheck,
                            label: 'Risk Tolerance',
                            value: quizAnswers.risk,
                            color: 'text-blue-400',
                            bg: 'bg-blue-500/10'
                        },
                        {
                            icon: TargetIcon,
                            label: 'Investment Goal',
                            value: quizAnswers.goal,
                            color: 'text-emerald-400',
                            bg: 'bg-emerald-500/10'
                        },
                        {
                            icon: TrendingUpIcon,
                            label: 'Experience',
                            value: quizAnswers.experience,
                            color: 'text-amber-400',
                            bg: 'bg-amber-500/10'
                        },
                        {
                            icon: DollarSign,
                            label: 'Capital',
                            value: `$${quizAnswers.capital.toLocaleString()}`,
                            color: 'text-purple-400',
                            bg: 'bg-purple-500/10'
                        }
                    ].map((item, idx) => (
                        <div key={idx} className={`${item.bg} rounded-xl p-4 border border-slate-800/50`}>
                            <div className="flex items-center gap-3 mb-2">
                                <div className={`p-2 rounded-lg ${item.bg}`}>
                                    <item.icon className={item.color} size={16}/>
                                </div>
                                <span className="text-xs font-bold text-slate-500 uppercase">{item.label}</span>
                            </div>
                            <p className="text-lg font-semibold text-slate-100">{item.value}</p>
                        </div>
                    ))}
                </div>
            </div>

            {/* Strategy Comparison Radar */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 bg-slate-900/50 border border-slate-800/50 rounded-2xl p-6">
                    <div className="flex items-center justify-between mb-6">
                        <h4 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                            <BarChart4 size={18} className="text-purple-400"/>
                            Strategy Comparison Matrix
                        </h4>
                        <div className="flex items-center gap-4">
                            {[
                                {color: '#FF6B6B', label: 'Quantum Momentum'},
                                {color: '#4ECDC4', label: 'Neural Reversion'},
                                {color: '#FFD166', label: 'Alpha Harvest'}
                            ].map((strategy, idx) => (
                                <div key={idx} className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded-full" style={{backgroundColor: strategy.color}}/>
                                    <span className="text-xs text-slate-400">{strategy.label}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                    <div className="h-[350px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                                <PolarGrid stroke="#1e293b" strokeWidth={0.5}/>
                                <PolarAngleAxis
                                    dataKey="subject"
                                    tick={{fill: '#94a3b8', fontSize: 11, fontWeight: '600'}}
                                    axisLine={{stroke: '#334155'}}
                                />
                                <PolarRadiusAxis
                                    angle={30}
                                    domain={[0, 100]}
                                    tick={false}
                                    axisLine={false}
                                />
                                {[
                                    {dataKey: 'Quantum', stroke: '#FF6B6B', name: 'Quantum Momentum'},
                                    {dataKey: 'Neural', stroke: '#4ECDC4', name: 'Neural Reversion'},
                                    {dataKey: 'Alpha', stroke: '#FFD166', name: 'Alpha Harvest'}
                                ].map((strategy, idx) => (
                                    <Radar
                                        key={idx}
                                        name={strategy.name}
                                        dataKey={strategy.dataKey}
                                        stroke={strategy.stroke}
                                        fill={strategy.stroke}
                                        fillOpacity={0.1}
                                        strokeWidth={2}
                                    />
                                ))}
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#0f172a',
                                        border: '1px solid #1e293b',
                                        borderRadius: '8px',
                                        color: '#e2e8f0'
                                    }}
                                />
                            </RadarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* AI Insights Panel */}
                <div className="space-y-6">
                    <div
                        className="bg-gradient-to-br from-purple-900/30 to-pink-900/30 border border-purple-800/30 rounded-2xl p-6">
                        <h4 className="text-xs font-black text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                            <BrainCircuit size={14} className="text-purple-400"/>
                            AI Insights
                        </h4>
                        <div className="space-y-4">
                            <div
                                className="p-3 bg-gradient-to-r from-red-500/10 to-orange-500/10 rounded-xl border border-red-500/20">
                                <p className="text-xs text-red-400 font-semibold mb-1">🚀 Top Recommendation</p>
                                <p className="text-sm text-slate-300">
                                    Quantum Momentum aligns best with your growth focus and capital
                                </p>
                            </div>
                            <div
                                className="p-3 bg-gradient-to-r from-teal-500/10 to-cyan-500/10 rounded-xl border border-teal-500/20">
                                <p className="text-xs text-teal-400 font-semibold mb-1">⚖️ Diversification Tip</p>
                                <p className="text-sm text-slate-300">
                                    Allocate 40% to Neural Reversion for portfolio stability
                                </p>
                            </div>
                            <div
                                className="p-3 bg-gradient-to-r from-amber-500/10 to-yellow-500/10 rounded-xl border border-amber-500/20">
                                <p className="text-xs text-amber-400 font-semibold mb-1">💰 Income Opportunity</p>
                                <p className="text-sm text-slate-300">
                                    Alpha Harvest provides steady returns in sideways markets
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Colorful Strategy Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {recommendations.map((rec, i) => (
                    <div
                        key={i}
                        className={`${rec.theme.bg} border ${rec.theme.border} rounded-2xl overflow-hidden hover:scale-[1.02] transition-all duration-300 group cursor-pointer`}
                        onClick={() => handleSelectStrategy(rec.id)}
                    >
                        <div className="p-6 space-y-5">
                            {/* Header */}
                            <div className="flex justify-between items-start">
                                <div>
                                    <div className="flex items-center gap-2 mb-2">
                                        <RiskBadge level={rec.risk_level} theme={rec.theme}/>
                                        {i === 0 && (
                                            <div
                                                className="flex items-center gap-1 px-2 py-1 bg-gradient-to-r from-purple-600/20 to-pink-600/20 rounded-lg border border-purple-500/30">
                                                <Crown size={10} className="text-purple-400"/>
                                                <span className="text-[10px] font-bold text-purple-400">BEST FIT</span>
                                            </div>
                                        )}
                                    </div>
                                    <h5 className={`text-xl font-bold ${rec.theme.text} group-hover:opacity-90 transition-colors`}>
                                        {rec.name}
                                    </h5>
                                    <p className="text-sm text-slate-500 italic">`&{rec.tagline}`</p>
                                </div>
                                <div
                                    className="text-center bg-slate-900/50 p-3 rounded-xl border border-slate-800 min-w-[70px]">
                                    <p className="text-2xl font-black bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                                        {rec.fit_score}%
                                    </p>
                                    <p className="text-[9px] font-bold text-slate-500 uppercase tracking-tighter">AI
                                        Fit</p>
                                </div>
                            </div>

                            {/* Quick Stats */}
                            <div className="grid grid-cols-2 gap-2">
                                <div className="text-center p-3 bg-slate-900/50 rounded-lg">
                                    <p className="text-[9px] text-slate-500 font-bold uppercase mb-1">Risk</p>
                                    <p className={`text-xs font-black uppercase ${rec.theme.text}`}>{rec.risk_level}</p>
                                </div>
                                <div className="text-center p-3 bg-slate-900/50 rounded-lg">
                                    <p className="text-[9px] text-slate-500 font-bold uppercase mb-1">Return</p>
                                    <p className="text-xs font-black text-emerald-400">{rec.expected_return}</p>
                                </div>
                                <div className="text-center p-3 bg-slate-900/50 rounded-lg">
                                    <p className="text-[9px] text-slate-500 font-bold uppercase mb-1">Time</p>
                                    <p className="text-xs font-black text-slate-300">{rec.time_commitment}</p>
                                </div>
                                <div className="text-center p-3 bg-slate-900/50 rounded-lg">
                                    <p className="text-[9px] text-slate-500 font-bold uppercase mb-1">Success</p>
                                    <p className="text-xs font-black text-amber-400">{rec.success_rate}</p>
                                </div>
                            </div>

                            {/* Description */}
                            <p className="text-sm text-slate-400 line-clamp-2">
                                {rec.description}
                            </p>

                            {/* Why This Fits */}
                            <div className="space-y-2 pt-2">
                                <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Key
                                    Benefits</p>
                                <div className="space-y-1.5">
                                    {rec.why.slice(0, 2).map((reason: string, idx: number) => (
                                        <div key={idx} className="flex items-center gap-2 text-xs text-slate-300">
                                            <div
                                                className="w-1.5 h-1.5 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 flex-shrink-0"/>
                                            <span className="line-clamp-1">{reason}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Tags */}
                            <div className="flex flex-wrap gap-1.5 pt-2">
                                {rec.tags.map((tag: string, idx: number) => (
                                    <span key={idx}
                                          className="px-2 py-1 bg-slate-900/50 rounded text-[10px] font-medium text-slate-400 border border-slate-800">
                                        {tag}
                                    </span>
                                ))}
                            </div>
                        </div>

                        {/* Purple Action Button */}
                        <div className="px-6 pb-6">
                            <PurpleButton
                                icon={<ArrowRight/>}
                                variant="secondary"
                                className="w-full"
                            >
                                Explore Strategy
                            </PurpleButton>
                        </div>
                    </div>
                ))}
            </div>

            {/* Disclaimer */}
            <div
                className="bg-gradient-to-br from-slate-900/30 to-slate-800/30 border border-slate-800/50 rounded-2xl p-6">
                <div className="flex items-start gap-3">
                    <Info className="text-purple-500 flex-shrink-0 mt-0.5" size={16}/>
                    <div>
                        <p className="text-xs text-slate-400">
                            <strong className="text-slate-300">Disclaimer:</strong>
                            AI-generated strategies use machine learning for market analysis.
                            Past performance does not guarantee future results. Trading involves risk. Consult financial
                            advisors before investing. Strategies update based on market conditions.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AIAdvisor;