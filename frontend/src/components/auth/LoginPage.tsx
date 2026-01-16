import { useState } from "react";
import { Activity, AlertCircle, ArrowRight, BarChart3, Cpu, RefreshCw, ShieldCheck, Target, Zap } from "lucide-react";
import { User } from "@/types/user";
import { auth } from "@/utils/api";

interface LoginPageProps {
    onLogin: (userData: User) => void;
    setCurrentPage: (page: string) => void;
}

const LoginPage = ({ onLogin, setCurrentPage }: LoginPageProps) => {
    const [isLogin, setIsLogin] = useState(true);
    const [formData, setFormData] = useState({ username: '', email: '', password: '' });
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const features = [
        {
            title: 'Performance Monitoring',
            icon: Activity,
            desc: 'Real-time portfolio tracking with millisecond-precision execution monitoring.',
            color: 'from-emerald-500 to-teal-500'
        },
        {
            title: 'AI-Powered Analysis',
            icon: Target,
            desc: 'Machine learning-driven regime detection and strategy recommendations.',
            color: 'from-violet-500 to-purple-500'
        },
        {
            title: 'Advanced Backtesting',
            icon: BarChart3,
            desc: 'Multi-asset portfolio simulation with comprehensive risk analytics.',
            color: 'from-blue-500 to-cyan-500'
        },
        {
            title: 'Strategy Development',
            icon: Cpu,
            desc: 'Build custom ML models with integrated optimization frameworks.',
            color: 'from-fuchsia-500 to-pink-500'
        }
    ];

    const handleSubmit = async () => {
        if (!formData.username || !formData.password) {
            setError('Please complete all required fields');
            return;
        }
        if (!isLogin && !formData.email) {
            setError('Email address is required');
            return;
        }

        setError('');
        setLoading(true);

        try {
            if (isLogin) {
                // Real Login
                const response = await auth.login({
                    email: formData.username,
                    password: formData.password
                });

                if (response.data && response.data.access_token) {
                    const { access_token, user: backendUser } = response.data;

                    // Save token to localStorage for the API client to pick up
                    localStorage.setItem('token', access_token);

                    const userForApp: User = {
                        name: backendUser.username,
                        email: backendUser.email,
                        tier: backendUser.tier || 'FREE',
                        token: access_token
                    };

                    onLogin(userForApp);
                    setCurrentPage('dashboard');
                }
            } else {
                // Real Registration
                await auth.register({
                    username: formData.username,
                    email: formData.email,
                    password: formData.password
                });

                setError('✓ Registration successful! Please sign in to continue.');
                setIsLogin(true);
            }
        } catch (err: any) {
            console.error("Auth error:", err);
            const msg = err.response?.data?.detail || "Authentication failed. Please check your credentials.";
            setError(msg);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen w-full flex">
            {/* Left Panel - Authentication */}
            <div className="w-full lg:w-[480px] flex flex-col justify-center p-8 md:p-12 border-r border-slate-800/80 bg-slate-900/95 backdrop-blur-xl z-10 shadow-2xl">
                <div className="mb-12">
                    <div className="flex items-center space-x-3 mb-8">
                        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center shadow-xl shadow-violet-500/30">
                            <Zap size={24} className="text-white" strokeWidth={2.5} />
                        </div>
                        <div>
                            <h1 className="text-2xl font-bold text-slate-100 tracking-tight">ORACULUM</h1>
                            <div className="h-0.5 w-12 bg-gradient-to-r from-violet-500 to-fuchsia-500 mt-1 rounded-full" />
                        </div>
                    </div>

                    <div className="space-y-2">
                        <h2 className="text-3xl font-bold text-slate-100">
                            {isLogin ? 'Welcome Back' : 'Get Started'}
                        </h2>
                        <p className="text-slate-400 text-base leading-relaxed">
                            {isLogin
                                ? 'Sign in to access your institutional trading terminal.'
                                : 'Create an account to unlock advanced quantitative tools.'}
                        </p>
                    </div>
                </div>

                <div className="space-y-5">
                    {!isLogin && (
                        <div className="space-y-4 p-4 rounded-xl bg-violet-500/5 border border-violet-500/20">
                            <div className="flex items-start space-x-3">
                                <AlertCircle size={20} className="text-violet-400 shrink-0 mt-0.5" />
                                <div>
                                    <p className="text-sm font-semibold text-slate-200">Account Personalization</p>
                                    <p className="text-xs text-slate-400 mt-1">Help us customize your experience with AI-driven insights.</p>
                                </div>
                            </div>
                            <div className="grid grid-cols-2 gap-3">
                                <div className="space-y-1.5">
                                    <label className="text-xs font-semibold text-slate-400">Investor Type</label>
                                    <select className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm text-slate-300 focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none transition-all">
                                        <option>Retail</option>
                                        <option>Professional</option>
                                        <option>Institutional</option>
                                        <option>Academic</option>
                                    </select>
                                </div>
                                <div className="space-y-1.5">
                                    <label className="text-xs font-semibold text-slate-400">Risk Profile</label>
                                    <select className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm text-slate-300 focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none transition-all">
                                        <option>Conservative</option>
                                        <option>Moderate</option>
                                        <option>Aggressive</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                    )}

                    <div className="space-y-4">
                        {!isLogin && (
                            <div className="space-y-1.5">
                                <label className="text-xs font-semibold text-slate-400 tracking-wide">Email Address</label>
                                <input
                                    type="email"
                                    placeholder="name@institution.com"
                                    value={formData.email}
                                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-4 py-3 text-slate-200 placeholder:text-slate-600 focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none transition-all"
                                />
                            </div>
                        )}
                        <div className="space-y-1.5">
                            <label className="text-xs font-semibold text-slate-400 tracking-wide">Username</label>
                            <input
                                type="text"
                                placeholder="Enter your username"
                                value={formData.username}
                                onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                                className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-4 py-3 text-slate-200 placeholder:text-slate-600 focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none transition-all"
                            />
                        </div>
                        <div className="space-y-1.5">
                            <label className="text-xs font-semibold text-slate-400 tracking-wide">Password</label>
                            <input
                                type="password"
                                placeholder="••••••••••"
                                value={formData.password}
                                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                                onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
                                className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-4 py-3 text-slate-200 placeholder:text-slate-600 focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none transition-all"
                            />
                        </div>
                    </div>

                    {error && (
                        <div className={`p-4 rounded-xl text-sm font-medium border ${error.startsWith('✓')
                            ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                            : 'bg-red-500/10 border-red-500/30 text-red-400'
                            }`}>
                            {error}
                        </div>
                    )}

                    <button
                        onClick={handleSubmit}
                        disabled={loading}
                        className="w-full py-4 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 disabled:from-slate-700 disabled:to-slate-700 text-white rounded-xl font-semibold transition-all shadow-xl shadow-violet-500/20 disabled:shadow-none flex items-center justify-center space-x-2"
                    >
                        {loading ? (
                            <RefreshCw size={20} className="animate-spin" />
                        ) : (
                            <span>{isLogin ? 'Sign In' : 'Create Account'}</span>
                        )}
                    </button>

                    <button
                        onClick={() => { setIsLogin(!isLogin); setError(''); }}
                        className="w-full text-center text-sm text-slate-400 hover:text-slate-300 transition-colors py-2 font-medium"
                    >
                        {isLogin ? "Don't have an account? Sign up" : "Already have an account? Sign in"}
                    </button>
                </div>

                <div className="mt-auto pt-8 border-t border-slate-800/80 flex items-center justify-between">
                    <div className="flex items-center space-x-2 text-xs text-slate-600 font-medium">
                        <ShieldCheck size={14} />
                        <span>AES-256 Encrypted</span>
                    </div>
                    <div className="text-xs text-slate-600 font-medium">v2.0.1</div>
                </div>
            </div>

            {/* Right Panel - Features */}
            <div className="hidden lg:flex flex-1 relative bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 items-center justify-center p-16 overflow-hidden">
                <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-violet-600/10 rounded-full blur-[150px]" />
                <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-fuchsia-600/10 rounded-full blur-[120px]" />

                <div className="relative z-10 max-w-2xl w-full">
                    <div className="mb-12">
                        <h3 className="text-4xl font-bold text-slate-100 leading-tight mb-4">
                            Institutional-Grade<br />
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-fuchsia-400">
                                Quantitative Platform
                            </span>
                        </h3>
                        <p className="text-slate-400 text-lg leading-relaxed">
                            Advanced analytics, AI-powered insights, and professional-grade backtesting tools for sophisticated traders.
                        </p>
                    </div>

                    <div className="grid grid-cols-1 gap-4">
                        {features.map((feature, i) => (
                            <div
                                key={i}
                                className="group flex items-start space-x-5 p-6 rounded-2xl bg-slate-800/40 backdrop-blur-sm border border-slate-700/50 hover:bg-slate-800/60 hover:border-slate-600/50 transition-all"
                            >
                                <div className={`p-3 rounded-xl bg-gradient-to-br ${feature.color} shadow-lg`}>
                                    <feature.icon size={24} className="text-white" strokeWidth={2} />
                                </div>
                                <div className="flex-1">
                                    <h4 className="text-lg font-semibold text-slate-100 mb-1 group-hover:text-violet-300 transition-colors">
                                        {feature.title}
                                    </h4>
                                    <p className="text-sm text-slate-400 leading-relaxed">
                                        {feature.desc}
                                    </p>
                                </div>
                                <ArrowRight
                                    className="self-center text-slate-700 opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all"
                                    size={20}
                                />
                            </div>
                        ))}
                    </div>

                    <div className="mt-12 pt-8 border-t border-slate-700/50 flex items-center justify-between">
                        <div className="flex items-center space-x-8">
                            <div>
                                <p className="text-xs font-semibold text-slate-500 tracking-wider mb-1">SYSTEM STATUS</p>
                                <div className="flex items-center text-sm font-semibold text-emerald-400">
                                    <span className="w-2 h-2 rounded-full bg-emerald-500 mr-2 animate-pulse" />
                                    All Systems Operational
                                </div>
                            </div>
                            <div className="h-10 w-px bg-slate-700/50" />
                            <div>
                                <p className="text-xs font-semibold text-slate-500 tracking-wider mb-1">ACTIVE USERS</p>
                                <p className="text-sm font-semibold text-slate-300">12,847</p>
                            </div>
                        </div>
                        <div className="text-right">
                            <p className="text-xs font-semibold text-slate-500 tracking-wider mb-1">AVG LATENCY</p>
                            <p className="text-sm font-semibold text-violet-400">18ms</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default LoginPage;