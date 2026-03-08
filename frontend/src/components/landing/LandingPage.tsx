'use client'

import React, { useEffect, useRef, useState } from 'react';
import {
    Activity, ArrowRight, BarChart3, Brain, ChevronRight,
    Code2, Cpu, DollarSign, GitBranch, Globe, Layers,
    LineChart, Lock, Play, Rocket, Shield, Sparkles,
    Target, TrendingUp, Zap
} from 'lucide-react';
import {
    ResponsiveContainer, AreaChart, Area, XAxis, YAxis,
    CartesianGrid, Tooltip
} from 'recharts';

interface LandingPageProps {
    onGetStarted: () => void;
    onSignIn: () => void;
}

// ─── Mock equity curve data ─────────────────────────────────
const equityData = Array.from({ length: 60 }, (_, i) => {
    const base = 10000;
    const stratGrowth = base + (i * 220) + Math.sin(i * 0.3) * 600 + Math.random() * 300;
    const benchGrowth = base + (i * 90) + Math.sin(i * 0.5) * 200 + Math.random() * 100;
    return {
        day: `Day ${i + 1}`,
        strategy: Math.round(stratGrowth),
        benchmark: Math.round(benchGrowth),
    };
});

// ─── Data ────────────────────────────────────────────────────
const METRICS = [
    { label: '20+', sub: 'Built-in Strategies', icon: Layers },
    { label: '6', sub: 'Asset Classes', icon: Globe },
    { label: 'Real-Time', sub: 'Live Execution', icon: Zap },
    { label: 'AI-Powered', sub: 'ML Studio', icon: Brain },
];

const FEATURES = [
    {
        title: 'Backtesting Lab',
        desc: 'Single & multi-asset backtesting with walk-forward analysis, Monte Carlo simulation, and Bayesian optimization.',
        icon: BarChart3,
        gradient: 'from-blue-500 to-cyan-500',
    },
    {
        title: 'AI Strategy Builder',
        desc: 'Describe your strategy in plain English and let AI generate production-ready Python code inheriting BaseStrategy.',
        icon: Sparkles,
        gradient: 'from-violet-500 to-purple-500',
    },
    {
        title: 'ML Strategy Studio',
        desc: 'Build and train ML models — Random Forest, Gradient Boosting, LSTM, and ensemble methods — with integrated feature engineering.',
        icon: Cpu,
        gradient: 'from-fuchsia-500 to-pink-500',
    },
    {
        title: 'Live Execution',
        desc: 'Deploy strategies to Alpaca and Interactive Brokers with real-time position tracking and automated risk controls.',
        icon: Zap,
        gradient: 'from-emerald-500 to-teal-500',
    },
    {
        title: 'Options Desk',
        desc: 'Options strategy builder with Greeks analytics, probability calculators, and volatility surface visualization.',
        icon: DollarSign,
        gradient: 'from-amber-500 to-orange-500',
    },
    {
        title: 'Portfolio Optimization',
        desc: 'Mean-variance, Black-Litterman, and risk parity optimization with efficient frontier visualization.',
        icon: Target,
        gradient: 'from-rose-500 to-red-500',
    },
];

const STEPS = [
    {
        step: '01',
        title: 'Build',
        desc: 'Design strategies visually, with AI, or in the code editor.',
        icon: Code2,
        gradient: 'from-violet-600 to-indigo-600',
    },
    {
        step: '02',
        title: 'Backtest',
        desc: 'Simulate on historical data with institutional-grade analytics.',
        icon: BarChart3,
        gradient: 'from-blue-600 to-cyan-600',
    },
    {
        step: '03',
        title: 'Optimize',
        desc: 'Walk-forward analysis and Bayesian parameter optimization.',
        icon: GitBranch,
        gradient: 'from-emerald-600 to-teal-600',
    },
    {
        step: '04',
        title: 'Deploy',
        desc: 'Go live with automated execution and real-time monitoring.',
        icon: Rocket,
        gradient: 'from-fuchsia-600 to-pink-600',
    },
];

// ─── Scroll-reveal hook ──────────────────────────────────────
function useReveal() {
    const ref = useRef<HTMLDivElement>(null);
    const [visible, setVisible] = useState(false);

    useEffect(() => {
        const el = ref.current;
        if (!el) return;
        const obs = new IntersectionObserver(
            ([entry]) => { if (entry.isIntersecting) setVisible(true); },
            { threshold: 0.15 }
        );
        obs.observe(el);
        return () => obs.disconnect();
    }, []);

    return { ref, visible };
}

function RevealSection({ children, className = '' }: { children: React.ReactNode; className?: string }) {
    const { ref, visible } = useReveal();
    return (
        <div
            ref={ref}
            className={`transition-all duration-700 ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'} ${className}`}
        >
            {children}
        </div>
    );
}

// ─── Component ───────────────────────────────────────────────
const LandingPage: React.FC<LandingPageProps> = ({ onGetStarted, onSignIn }) => {
    const [scrolled, setScrolled] = useState(false);

    useEffect(() => {
        const onScroll = () => setScrolled(window.scrollY > 40);
        window.addEventListener('scroll', onScroll, { passive: true });
        return () => window.removeEventListener('scroll', onScroll);
    }, []);

    return (
        <div className="min-h-screen bg-slate-950 text-slate-200 overflow-x-hidden">

            {/* ══════════════ NAVBAR ══════════════ */}
            <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled ? 'bg-slate-950/90 backdrop-blur-xl border-b border-slate-800/60 shadow-2xl' : 'bg-transparent'}`}>
                <div className="max-w-7xl mx-auto px-6 lg:px-8">
                    <div className="flex items-center justify-between h-16 lg:h-20">
                        {/* Logo */}
                        <div className="flex items-center space-x-3">
                            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center shadow-lg shadow-violet-500/30">
                                <Activity size={20} className="text-white" strokeWidth={2.5} />
                            </div>
                            <div>
                                <h1 className="text-lg font-bold text-slate-100 tracking-tight">ORACULUM</h1>
                                <div className="h-0.5 w-10 bg-gradient-to-r from-violet-500 to-fuchsia-500 rounded-full" />
                            </div>
                        </div>

                        {/* Links */}
                        <div className="hidden md:flex items-center space-x-8">
                            <a href="#features" className="text-sm text-slate-400 hover:text-slate-200 transition-colors font-medium">Features</a>
                            <a href="#how-it-works" className="text-sm text-slate-400 hover:text-slate-200 transition-colors font-medium">How It Works</a>
                            <a href="#preview" className="text-sm text-slate-400 hover:text-slate-200 transition-colors font-medium">Preview</a>
                        </div>

                        {/* CTAs */}
                        <div className="flex items-center space-x-3">
                            <button
                                onClick={onSignIn}
                                className="px-5 py-2.5 text-sm font-semibold text-slate-300 hover:text-white transition-colors"
                            >
                                Sign In
                            </button>
                            <button
                                onClick={onGetStarted}
                                className="px-5 py-2.5 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white text-sm font-semibold rounded-xl transition-all shadow-lg shadow-violet-500/20"
                            >
                                Get Started
                            </button>
                        </div>
                    </div>
                </div>
            </nav>

            {/* ══════════════ HERO ══════════════ */}
            <section className="relative pt-32 pb-20 lg:pt-44 lg:pb-32 overflow-hidden">
                {/* Background decorations */}
                <div className="absolute top-20 left-1/4 w-[600px] h-[600px] bg-violet-600/8 rounded-full blur-[150px] animate-float" />
                <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] bg-fuchsia-600/6 rounded-full blur-[120px] animate-float-delayed" />
                <div className="absolute inset-0 bg-[linear-gradient(rgba(139,92,246,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(139,92,246,0.03)_1px,transparent_1px)] bg-[size:60px_60px] [mask-image:radial-gradient(ellipse_at_center,black_30%,transparent_70%)]" />

                <div className="relative z-10 max-w-5xl mx-auto px-6 lg:px-8 text-center">
                    {/* Badge */}
                    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-violet-500/10 border border-violet-500/20 text-violet-400 text-xs font-semibold tracking-wider mb-8 animate-fade-in">
                        <Sparkles size={14} />
                        INSTITUTIONAL-GRADE QUANTITATIVE PLATFORM
                    </div>

                    <h1 className="text-4xl sm:text-5xl lg:text-7xl font-bold text-slate-100 leading-[1.1] tracking-tight mb-6 animate-fade-in">
                        Backtest, Optimize &{' '}
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-fuchsia-400">
                            Deploy
                        </span>
                        {' '}Trading Strategies
                    </h1>

                    <p className="text-lg lg:text-xl text-slate-400 max-w-2xl mx-auto leading-relaxed mb-10 animate-fade-in">
                        Build AI-powered trading strategies, simulate on historical data with institutional-grade analytics, and deploy to live markets — all in one platform.
                    </p>

                    {/* Dual CTA */}
                    <div className="flex flex-col sm:flex-row items-center justify-center gap-4 animate-fade-in">
                        <button
                            onClick={onGetStarted}
                            className="group px-8 py-4 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white rounded-xl font-bold text-sm uppercase tracking-widest transition-all shadow-xl shadow-violet-500/20 flex items-center gap-3"
                        >
                            Get Started Free
                            <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
                        </button>
                        <button
                            onClick={() => {
                                document.getElementById('preview')?.scrollIntoView({ behavior: 'smooth' });
                            }}
                            className="group px-8 py-4 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700/50 text-slate-300 rounded-xl font-bold text-sm uppercase tracking-widest transition-all flex items-center gap-3"
                        >
                            <Play size={16} className="text-violet-400" />
                            View Demo
                        </button>
                    </div>
                </div>
            </section>

            {/* ══════════════ METRICS BAR ══════════════ */}
            <RevealSection>
                <section className="max-w-6xl mx-auto px-6 lg:px-8 pb-20">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        {METRICS.map((m, i) => {
                            const Icon = m.icon;
                            return (
                                <div key={i} className="glass-card rounded-2xl p-6 text-center group cursor-default">
                                    <Icon size={24} className="mx-auto mb-3 text-violet-400 group-hover:scale-110 transition-transform" />
                                    <p className="text-2xl lg:text-3xl font-bold text-slate-100 mb-1">{m.label}</p>
                                    <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider">{m.sub}</p>
                                </div>
                            );
                        })}
                    </div>
                </section>
            </RevealSection>

            {/* ══════════════ FEATURES ══════════════ */}
            <section id="features" className="max-w-6xl mx-auto px-6 lg:px-8 py-20">
                <RevealSection>
                    <div className="text-center mb-16">
                        <p className="text-xs font-bold text-violet-400 uppercase tracking-[0.2em] mb-3">PLATFORM CAPABILITIES</p>
                        <h2 className="text-3xl lg:text-4xl font-bold text-slate-100 mb-4">
                            Everything You Need to{' '}
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-fuchsia-400">Trade Smarter</span>
                        </h2>
                        <p className="text-slate-400 max-w-xl mx-auto">
                            From strategy ideation to live deployment, Oraculum covers the full quantitative trading workflow.
                        </p>
                    </div>
                </RevealSection>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                    {FEATURES.map((f, i) => {
                        const Icon = f.icon;
                        return (
                            <RevealSection key={i}>
                                <div className="glass-card rounded-2xl p-7 h-full group cursor-default">
                                    <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${f.gradient} flex items-center justify-center mb-5 shadow-lg group-hover:scale-110 transition-transform duration-300`}>
                                        <Icon size={22} className="text-white" strokeWidth={2} />
                                    </div>
                                    <h3 className="text-lg font-bold text-slate-100 mb-2 group-hover:text-violet-300 transition-colors">{f.title}</h3>
                                    <p className="text-sm text-slate-400 leading-relaxed">{f.desc}</p>
                                </div>
                            </RevealSection>
                        );
                    })}
                </div>
            </section>

            {/* ══════════════ HOW IT WORKS ══════════════ */}
            <section id="how-it-works" className="py-20 relative">
                <div className="absolute inset-0 bg-gradient-to-b from-transparent via-violet-950/10 to-transparent" />
                <div className="relative z-10 max-w-6xl mx-auto px-6 lg:px-8">
                    <RevealSection>
                        <div className="text-center mb-16">
                            <p className="text-xs font-bold text-violet-400 uppercase tracking-[0.2em] mb-3">WORKFLOW</p>
                            <h2 className="text-3xl lg:text-4xl font-bold text-slate-100 mb-4">
                                From Idea to{' '}
                                <span className="text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-fuchsia-400">Live Market</span>
                                {' '}in 4 Steps
                            </h2>
                        </div>
                    </RevealSection>

                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                        {STEPS.map((s, i) => {
                            const Icon = s.icon;
                            return (
                                <RevealSection key={i}>
                                    <div className="relative">
                                        {/* Connector line */}
                                        {i < STEPS.length - 1 && (
                                            <div className="hidden lg:block absolute top-10 left-full w-full h-px bg-gradient-to-r from-slate-700 to-transparent z-0" />
                                        )}
                                        <div className="glass-card rounded-2xl p-7 relative z-10 group cursor-default">
                                            <div className="flex items-center gap-3 mb-4">
                                                <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${s.gradient} flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform`}>
                                                    <Icon size={18} className="text-white" />
                                                </div>
                                                <span className="text-xs font-bold text-slate-600 tracking-widest">STEP {s.step}</span>
                                            </div>
                                            <h3 className="text-xl font-bold text-slate-100 mb-2">{s.title}</h3>
                                            <p className="text-sm text-slate-400 leading-relaxed">{s.desc}</p>
                                        </div>
                                    </div>
                                </RevealSection>
                            );
                        })}
                    </div>
                </div>
            </section>

            {/* ══════════════ PRODUCT PREVIEW ══════════════ */}
            <section id="preview" className="max-w-6xl mx-auto px-6 lg:px-8 py-20">
                <RevealSection>
                    <div className="text-center mb-12">
                        <p className="text-xs font-bold text-violet-400 uppercase tracking-[0.2em] mb-3">PREVIEW</p>
                        <h2 className="text-3xl lg:text-4xl font-bold text-slate-100 mb-4">
                            See It in{' '}
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-fuchsia-400">Action</span>
                        </h2>
                    </div>
                </RevealSection>

                <RevealSection>
                    <div className="glass-panel rounded-2xl overflow-hidden">
                        {/* Mock title bar */}
                        <div className="bg-slate-900/80 px-6 py-3 flex items-center justify-between border-b border-slate-800/60">
                            <div className="flex items-center gap-4">
                                <div className="flex gap-1.5">
                                    <div className="w-3 h-3 rounded-full bg-red-500/70" />
                                    <div className="w-3 h-3 rounded-full bg-amber-500/70" />
                                    <div className="w-3 h-3 rounded-full bg-emerald-500/70" />
                                </div>
                                <span className="text-xs text-slate-500 font-mono">oraculum — Strategy Performance</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="px-2 py-1 bg-emerald-500/10 text-emerald-400 text-[10px] font-semibold rounded-md">LIVE</span>
                            </div>
                        </div>

                        {/* Metrics row */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-6 border-b border-slate-800/40">
                            {[
                                { label: 'Total Return', value: '+42.8%', color: 'text-emerald-400' },
                                { label: 'Sharpe Ratio', value: '2.15', color: 'text-violet-400' },
                                { label: 'Max Drawdown', value: '-12.4%', color: 'text-amber-400' },
                                { label: 'Win Rate', value: '68.3%', color: 'text-blue-400' },
                            ].map((m, i) => (
                                <div key={i} className="text-center">
                                    <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-wider mb-1">{m.label}</p>
                                    <p className={`text-2xl font-bold ${m.color}`}>{m.value}</p>
                                </div>
                            ))}
                        </div>

                        {/* Chart */}
                        <div className="p-6">
                            <div className="flex items-center justify-between mb-4">
                                <h4 className="text-sm font-semibold text-slate-300">Equity Growth Curve</h4>
                                <div className="flex items-center gap-4">
                                    <div className="flex items-center gap-1.5 text-xs text-slate-400">
                                        <div className="w-2 h-2 rounded-full bg-violet-500" />
                                        Strategy
                                    </div>
                                    <div className="flex items-center gap-1.5 text-xs text-slate-400">
                                        <div className="w-2 h-2 rounded-full bg-slate-600" />
                                        Buy & Hold
                                    </div>
                                </div>
                            </div>
                            <div className="h-[280px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={equityData}>
                                        <defs>
                                            <linearGradient id="landingGradient" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                                                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                                        <XAxis dataKey="day" stroke="#64748b" fontSize={10} axisLine={false} tickLine={false} interval={9} />
                                        <YAxis stroke="#64748b" fontSize={10} axisLine={false} tickLine={false} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px', color: '#e2e8f0' }}
                                            formatter={(value: number | undefined) => [`$${(value ?? 0).toLocaleString()}`, '']}
                                        />
                                        <Area type="monotone" dataKey="strategy" stroke="#8b5cf6" fill="url(#landingGradient)" strokeWidth={2} name="Strategy" />
                                        <Area type="monotone" dataKey="benchmark" stroke="#475569" fill="transparent" strokeWidth={2} strokeDasharray="5 5" name="Buy & Hold" />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>
                </RevealSection>
            </section>

            {/* ══════════════ FINAL CTA ══════════════ */}
            <section className="py-24 relative">
                <div className="absolute inset-0 bg-gradient-to-b from-transparent via-violet-950/15 to-transparent" />
                <RevealSection>
                    <div className="relative z-10 max-w-3xl mx-auto px-6 lg:px-8 text-center">
                        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center mx-auto mb-8 shadow-xl shadow-violet-500/30">
                            <Rocket size={28} className="text-white" />
                        </div>
                        <h2 className="text-3xl lg:text-4xl font-bold text-slate-100 mb-4">
                            Ready to Trade Smarter?
                        </h2>
                        <p className="text-slate-400 mb-10 leading-relaxed">
                            Join traders using institutional-grade tools to build, test, and deploy quantitative strategies.
                        </p>
                        <button
                            onClick={onGetStarted}
                            className="group px-10 py-4 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white rounded-xl font-bold text-sm uppercase tracking-widest transition-all shadow-xl shadow-violet-500/20 flex items-center gap-3 mx-auto"
                        >
                            Create Free Account
                            <ChevronRight size={18} className="group-hover:translate-x-1 transition-transform" />
                        </button>
                    </div>
                </RevealSection>
            </section>

            {/* ══════════════ FOOTER ══════════════ */}
            <footer className="border-t border-slate-800/60 py-12">
                <div className="max-w-6xl mx-auto px-6 lg:px-8">
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-10">
                        {/* Brand */}
                        <div className="md:col-span-1">
                            <div className="flex items-center space-x-3 mb-4">
                                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center">
                                    <Activity size={16} className="text-white" strokeWidth={2.5} />
                                </div>
                                <span className="text-sm font-bold text-slate-100 tracking-tight">ORACULUM</span>
                            </div>
                            <p className="text-xs text-slate-500 leading-relaxed">
                                Institutional-grade quantitative trading platform for modern traders.
                            </p>
                        </div>

                        {/* Links */}
                        <div>
                            <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-3">Platform</p>
                            <ul className="space-y-2">
                                {['Backtesting', 'ML Studio', 'Live Execution', 'Options Desk'].map(l => (
                                    <li key={l}><span className="text-xs text-slate-400 hover:text-slate-200 cursor-pointer transition-colors">{l}</span></li>
                                ))}
                            </ul>
                        </div>
                        <div>
                            <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-3">Resources</p>
                            <ul className="space-y-2">
                                {['Documentation', 'API Reference', 'Strategy Library', 'Marketplace'].map(l => (
                                    <li key={l}><span className="text-xs text-slate-400 hover:text-slate-200 cursor-pointer transition-colors">{l}</span></li>
                                ))}
                            </ul>
                        </div>
                        <div>
                            <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-3">Security</p>
                            <ul className="space-y-2">
                                <li className="flex items-center gap-2 text-xs text-slate-400">
                                    <Lock size={12} className="text-violet-400" /> AES-256 Encryption
                                </li>
                                <li className="flex items-center gap-2 text-xs text-slate-400">
                                    <Shield size={12} className="text-violet-400" /> SOC 2 Compliant
                                </li>
                            </ul>
                        </div>
                    </div>

                    <div className="border-t border-slate-800/60 pt-6 flex flex-col md:flex-row items-center justify-between text-xs text-slate-600">
                        <span>© {new Date().getFullYear()} Oraculum. All rights reserved.</span>
                        <div className="flex items-center gap-6 mt-3 md:mt-0">
                            <span className="hover:text-slate-400 cursor-pointer transition-colors">Privacy</span>
                            <span className="hover:text-slate-400 cursor-pointer transition-colors">Terms</span>
                            <span className="hover:text-slate-400 cursor-pointer transition-colors">Contact</span>
                        </div>
                    </div>
                </div>
            </footer>
        </div>
    );
};

export default LandingPage;
