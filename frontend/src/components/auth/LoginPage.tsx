import { useState, KeyboardEvent, ChangeEvent } from "react";
import { Activity, AlertCircle, ArrowRight, BarChart3, Cpu, RefreshCw, ShieldCheck, Target, Zap } from "lucide-react";
import { User } from "@/types/all_types";
import { api } from "@/utils/api";

// Define investor types and risk profiles
type InvestorType = 'Retail' | 'Professional' | 'Institutional' | 'Academic';
type RiskProfile = 'Conservative' | 'Moderate' | 'Aggressive';

interface LoginPageProps {
    onLogin: (userData: User, token: string) => void;
    setCurrentPage: (page: string) => void;
}

interface LoginFormData {
    username: string;
    email: string;
    password: string;
    investorType?: InvestorType;
    riskProfile?: RiskProfile;
}

interface LoginResponse {
    user: User;
    access_token: string;
    refresh_token: string;
    token_type?: string;
}

const LoginPage = ({ onLogin, setCurrentPage }: LoginPageProps) => {
    const [isLogin, setIsLogin] = useState(true);
    const [formData, setFormData] = useState<LoginFormData>({
        username: '',
        email: '',
        password: '',
        investorType: 'Retail',
        riskProfile: 'Moderate'
    });

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

    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (): Promise<void> => {
        if (!validateForm()) {
            return;
        }

        setError('');
        setLoading(true);

        try {
            if (isLogin) {
                // Handle login
                // The backend expects email field, not username
                const loginData = {
                    email: formData.email || formData.username,
                    password: formData.password
                };

                console.log('Attempting login with:', loginData);

                // Direct call without .data since interceptor returns data
                const response = await api.auth.login(loginData) as unknown as LoginResponse;
                console.log('Login response:', response);

                // Extract user data and token - NO .data here!
                const userData = response.user;
                const token = response.access_token;

                // Update user for app
                const userForApp: User = {
                    id: userData.id,
                    username: userData.username,
                    email: userData.email,
                    tier: userData.tier || 'FREE',
                    is_active: userData.is_active,
                    created_at: userData.created_at,
                    last_login: userData.last_login,
                };

                console.log('Login successful, user:', userForApp);

                // Call the login handler with both user and token
                onLogin(userForApp, token);
                setCurrentPage('dashboard');
            } else {
                // Handle registration
                const registerData = {
                    username: formData.username,
                    email: formData.email,
                    password: formData.password
                };

                console.log('Attempting registration with:', registerData);

                // Direct call without .data
                const response = await api.auth.register(registerData) as unknown as LoginResponse;
                console.log('Registration response:', response);

                setError('✓ Registration successful! Please sign in to continue.');
                setIsLogin(true);

                // Clear form data
                setFormData({
                    username: '',
                    email: '',
                    password: '',
                    investorType: 'Retail',
                    riskProfile: 'Moderate'
                });
            }
        } catch (err: unknown) {
            console.error("Auth error details:", err);

            // Log full error for debugging
            if (err && typeof err === 'object') {
                console.log('Error object:', JSON.stringify(err, null, 2));
            }

            // Handle API errors
            if (typeof err === 'object' && err !== null && 'status' in err) {
                const apiError = err as { status: number; data?: any; message?: string };

                console.log('API Error status:', apiError.status);
                console.log('API Error data:', apiError.data);

                if (apiError.status === 401) {
                    setError("Invalid credentials. Please check your email and password.");
                } else if (apiError.status === 422) {
                    // Validation error
                    if (Array.isArray(apiError.data?.detail)) {
                        setError(apiError.data.detail[0]?.msg || "Validation failed");
                    } else if (typeof apiError.data?.detail === 'string') {
                        setError(apiError.data.detail);
                    } else {
                        setError("Invalid input data. Please check all fields.");
                    }
                } else if (apiError.status === 409) {
                    setError("User already exists with this email or username");
                } else {
                    setError(`Authentication failed (${apiError.status}). Please try again.`);
                }
            } else if (err instanceof Error) {
                setError(err.message || "Network error. Please check your connection.");
            } else {
                setError("An unexpected error occurred. Please try again.");
            }
        } finally {
            setLoading(false);
        }
    };

    const validateForm = (): boolean => {
        if (isLogin) {
            // For login, we need either email/username and password
            if (!formData.password.trim()) {
                setError('Password is required');
                return false;
            }

            // Check if we have either username or email
            if (!formData.username.trim() && !formData.email.trim()) {
                setError('Email or username is required');
                return false;
            }

            // If email is provided, validate format
            if (formData.email.trim()) {
                const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                if (!emailRegex.test(formData.email)) {
                    setError('Please enter a valid email address');
                    return false;
                }
            }
        } else {
            // For registration, we need all fields
            if (!formData.username.trim() || !formData.password.trim() || !formData.email.trim()) {
                setError('Please complete all required fields');
                return false;
            }

            // Validate email format
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(formData.email)) {
                setError('Please enter a valid email address');
                return false;
            }

            if (formData.password.length < 8) {
                setError('Password must be at least 8 characters long');
                return false;
            }
        }

        return true;
    };

    const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>): void => {
        if (e.key === 'Enter') {
            handleSubmit();
        }
    };

    const handleInputChange = (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>): void => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const toggleMode = (): void => {
        setIsLogin(!isLogin);
        setError('');
        // Clear form when toggling
        setFormData({
            username: '',
            email: '',
            password: '',
            investorType: 'Retail',
            riskProfile: 'Moderate'
        });
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
                                    <select
                                        name="investorType"
                                        value={formData.investorType}
                                        onChange={handleInputChange}
                                        className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm text-slate-300 focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none transition-all"
                                    >
                                        <option value="Retail">Retail</option>
                                        <option value="Professional">Professional</option>
                                        <option value="Institutional">Institutional</option>
                                        <option value="Academic">Academic</option>
                                    </select>
                                </div>
                                <div className="space-y-1.5">
                                    <label className="text-xs font-semibold text-slate-400">Risk Profile</label>
                                    <select
                                        name="riskProfile"
                                        value={formData.riskProfile}
                                        onChange={handleInputChange}
                                        className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2 text-sm text-slate-300 focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none transition-all"
                                    >
                                        <option value="Conservative">Conservative</option>
                                        <option value="Moderate">Moderate</option>
                                        <option value="Aggressive">Aggressive</option>
                                    </select>
                                </div>
                            </div>
                        </div>
                    )}

                    <div className="space-y-4">
                        <div className="space-y-1.5">
                            <label className="text-xs font-semibold text-slate-400 tracking-wide">
                                {isLogin ? 'Email or Username' : 'Username'}
                            </label>
                            <input
                                type="text"
                                name="username"
                                placeholder={isLogin ? "Enter your email or username" : "Enter your username"}
                                value={formData.username}
                                onChange={handleInputChange}
                                className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-4 py-3 text-slate-200 placeholder:text-slate-600 focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none transition-all"
                            />
                        </div>

                        {!isLogin && (
                            <div className="space-y-1.5">
                                <label className="text-xs font-semibold text-slate-400 tracking-wide">Email Address</label>
                                <input
                                    type="email"
                                    name="email"
                                    placeholder="name@institution.com"
                                    value={formData.email}
                                    onChange={handleInputChange}
                                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-4 py-3 text-slate-200 placeholder:text-slate-600 focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none transition-all"
                                />
                            </div>
                        )}

                        <div className="space-y-1.5">
                            <label className="text-xs font-semibold text-slate-400 tracking-wide">Password</label>
                            <input
                                type="password"
                                name="password"
                                placeholder="••••••••••"
                                value={formData.password}
                                onChange={handleInputChange}
                                onKeyPress={handleKeyPress}
                                className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-4 py-3 text-slate-200 placeholder:text-slate-600 focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none transition-all"
                            />
                            {!isLogin && (
                                <p className="text-xs text-slate-500 mt-1">Password must be at least 8 characters</p>
                            )}
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
                        className="w-full py-4 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 disabled:from-slate-700 disabled:to-slate-700 text-white rounded-xl font-semibold transition-all shadow-xl shadow-violet-500/20 disabled:shadow-none flex items-center justify-center space-x-2 disabled:opacity-70"
                    >
                        {loading ? (
                            <>
                                <RefreshCw size={20} className="animate-spin" />
                                <span>Processing...</span>
                            </>
                        ) : (
                            <span>{isLogin ? 'Sign In' : 'Create Account'}</span>
                        )}
                    </button>

                    <button
                        onClick={toggleMode}
                        className="w-full text-center text-sm text-slate-400 hover:text-slate-300 transition-colors py-2 font-medium hover:underline"
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
                                className="group flex items-start space-x-5 p-6 rounded-2xl bg-slate-800/40 backdrop-blur-sm border border-slate-700/50 hover:bg-slate-800/60 hover:border-slate-600/50 transition-all duration-300 cursor-pointer"
                                onClick={() => {
                                    if (!isLogin) {
                                        const featureTitles = features.map(f => f.title);
                                        setError(`The "${featureTitles[i]}" feature will be available after registration!`);
                                    }
                                }}
                            >
                                <div className={`p-3 rounded-xl bg-gradient-to-br ${feature.color} shadow-lg group-hover:scale-105 transition-transform duration-300`}>
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
                                    className="self-center text-slate-700 opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all duration-300"
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