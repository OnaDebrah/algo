import {useState, KeyboardEvent, ChangeEvent} from "react";
import {AlertCircle, ArrowRight, RefreshCw, ShieldCheck, Zap} from "lucide-react";
import {User} from "@/types/all_types";
import {api} from "@/utils/api";
import CountrySelect from "@/components/auth/CountrySelect";

type InvestorType = 'Retail' | 'Professional' | 'Institutional' | 'Academic';
type RiskProfile = 'Conservative' | 'Moderate' | 'Aggressive';

interface LoginPageProps {
    onLogin: (userData: User, token: string) => void;
    setCurrentPage: (page: string) => void;
    onBackToLanding?: () => void;
}

interface LoginFormData {
    username: string;
    email: string;
    password: string;
    country: string;
    investorType: InvestorType;
    riskProfile: RiskProfile;
}

interface LoginResponse {
    user: User;
    access_token: string;
    refresh_token: string;
    token_type?: string;
}

const INITIAL_FORM: LoginFormData = {
    username: '',
    email: '',
    password: '',
    country: '',
    investorType: 'Retail',
    riskProfile: 'Moderate',
};

const LoginPage = ({onLogin, setCurrentPage, onBackToLanding}: LoginPageProps) => {
    const [isLogin, setIsLogin] = useState(true);
    const [formData, setFormData] = useState<LoginFormData>({...INITIAL_FORM});
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (): Promise<void> => {
        if (!validateForm()) return;

        setError('');
        setLoading(true);

        try {
            if (isLogin) {
                const loginData = {
                    email: formData.email || formData.username,
                    password: formData.password,
                };

                const response = await api.auth.login(loginData) as unknown as LoginResponse;

                const userData = response.user;
                const userForApp: User = {
                    id: userData.id,
                    username: userData.username,
                    email: userData.email,
                    tier: userData.tier || 'FREE',
                    is_active: userData.is_active,
                    created_at: userData.created_at,
                    last_login: userData.last_login,
                    country: userData.country,
                    investor_type: userData.investor_type,
                    risk_profile: userData.risk_profile,
                };

                onLogin(userForApp, response.access_token);
                setCurrentPage('dashboard');
            } else {
                const registerData = {
                    username: formData.username,
                    email: formData.email,
                    password: formData.password,
                    country: formData.country || undefined,
                    investor_type: formData.investorType,
                    risk_profile: formData.riskProfile,
                };

                await api.auth.register(registerData) as unknown as LoginResponse;

                setError('✓ Registration successful! Please sign in to continue.');
                setIsLogin(true);
                setFormData({...INITIAL_FORM});
            }
        } catch (err: unknown) {
            if (typeof err === 'object' && err !== null && 'status' in err) {
                const apiError = err as {
                    status: number;
                    data?: { detail?: string | { msg: string }[] };
                    message?: string
                };
                if (apiError.status === 401) {
                    setError("Invalid credentials. Please check your email and password.");
                } else if (apiError.status === 422) {
                    if (Array.isArray(apiError.data?.detail)) {
                        setError(apiError.data.detail[0]?.msg || "Validation failed");
                    } else if (typeof apiError.data?.detail === 'string') {
                        setError(apiError.data.detail);
                    } else {
                        setError("Invalid input data. Please check all fields.");
                    }
                } else if (apiError.status === 409 || apiError.status === 400) {
                    setError(apiError.data?.detail as string || "User already exists with this email or username");
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
            if (!formData.password.trim()) {
                setError('Password is required');
                return false;
            }
            if (!formData.username.trim() && !formData.email.trim()) {
                setError('Email or username is required');
                return false;
            }
            if (formData.email.trim()) {
                const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                if (!emailRegex.test(formData.email)) {
                    setError('Please enter a valid email address');
                    return false;
                }
            }
        } else {
            if (!formData.username.trim() || !formData.password.trim() || !formData.email.trim()) {
                setError('Please complete all required fields');
                return false;
            }
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
        if (e.key === 'Enter') handleSubmit();
    };

    const handleInputChange = (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>): void => {
        const {name, value} = e.target;
        setFormData(prev => ({...prev, [name]: value}));
    };

    const toggleMode = (): void => {
        setIsLogin(!isLogin);
        setError('');
        setFormData({...INITIAL_FORM});
    };

    const inputClass = "w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-4 py-3 text-slate-200 placeholder:text-slate-600 focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none transition-all";
    const selectClass = "w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2.5 text-sm text-slate-300 focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none transition-all";
    const labelClass = "text-xs font-semibold text-slate-400 tracking-wide";

    return (
        <div className="min-h-screen w-full flex items-center justify-center bg-slate-950 relative overflow-hidden">
            {/* Background decorations */}
            <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-violet-600/10 rounded-full blur-[150px]"/>
            <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-fuchsia-600/10 rounded-full blur-[120px]"/>
            <div
                className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-indigo-600/5 rounded-full blur-[200px]"/>

            {/* Centered card */}
            <div className="relative z-10 w-full max-w-xl mx-4">
                <div className="glass-panel rounded-2xl p-8 md:p-10 shadow-2xl">
                    {/* Back to landing */}
                    {onBackToLanding && (
                        <button
                            onClick={onBackToLanding}
                            className="flex items-center gap-2 text-sm text-slate-400 hover:text-slate-200 transition-colors mb-6 group"
                        >
                            <ArrowRight size={16}
                                        className="rotate-180 group-hover:-translate-x-1 transition-transform"/>
                            Back to Home
                        </button>
                    )}

                    {/* Branding */}
                    <div className="mb-8">
                        <div className="flex items-center space-x-3 mb-6">
                            <div
                                className="w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center shadow-xl shadow-violet-500/30">
                                <Zap size={24} className="text-white" strokeWidth={2.5}/>
                            </div>
                            <div>
                            <span className="text-2xl font-bold tracking-tight flex items-center gap-2">
                              <span className="text-slate-100">ORACULUM</span>
                              <span
                                  className="text-xs font-semibold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
                                AI
                              </span>
                            </span>
                                <div
                                    className="h-0.5 w-12 bg-gradient-to-r from-violet-500 to-fuchsia-500 mt-1 rounded-full"/>
                            </div>
                        </div>
                        <h2 className="text-3xl font-bold text-slate-100">
                            {isLogin ? 'Welcome Back' : 'Create Account'}
                        </h2>
                        <p className="text-slate-400 text-base leading-relaxed mt-2">
                            {isLogin
                                ? 'Sign in to access your trading terminal.'
                                : 'Set up your profile to receive personalized strategy alerts and recommendations.'}
                        </p>
                    </div>

                    {/* Form */}
                    <div className="space-y-5">
                        {/* Account Personalization — registration only */}
                        {!isLogin && (
                            <div className="space-y-4 p-4 rounded-xl bg-violet-500/5 border border-violet-500/20">
                                <div className="flex items-start space-x-3">
                                    <AlertCircle size={20} className="text-violet-400 shrink-0 mt-0.5"/>
                                    <div>
                                        <p className="text-sm font-semibold text-slate-200">Profile & Preferences</p>
                                        <p className="text-xs text-slate-400 mt-1">Used for personalized strategy alerts
                                            and stock recommendations.</p>
                                    </div>
                                </div>
                                <div className="grid grid-cols-3 gap-3">
                                    <div className="space-y-1.5">
                                        <label className={labelClass}>Country</label>
                                        <CountrySelect
                                            value={formData.country}
                                            onChange={(value) => setFormData(prev => ({...prev, country: value}))}
                                        />
                                    </div>
                                    <div className="space-y-1.5">
                                        <label className={labelClass}>Investor Type</label>
                                        <select
                                            name="investorType"
                                            value={formData.investorType}
                                            onChange={handleInputChange}
                                            className={selectClass}
                                        >
                                            <option value="Retail">Retail</option>
                                            <option value="Professional">Professional</option>
                                            <option value="Institutional">Institutional</option>
                                            <option value="Academic">Academic</option>
                                        </select>
                                    </div>
                                    <div className="space-y-1.5">
                                        <label className={labelClass}>Risk Profile</label>
                                        <select
                                            name="riskProfile"
                                            value={formData.riskProfile}
                                            onChange={handleInputChange}
                                            className={selectClass}
                                        >
                                            <option value="Conservative">Conservative</option>
                                            <option value="Moderate">Moderate</option>
                                            <option value="Aggressive">Aggressive</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Core fields */}
                        <div className="space-y-4">
                            <div className="space-y-1.5">
                                <label className={labelClass}>
                                    {isLogin ? 'Email or Username' : 'Username'}
                                </label>
                                <input
                                    type="text"
                                    name="username"
                                    placeholder={isLogin ? "Enter your email or username" : "Enter your username"}
                                    value={formData.username}
                                    onChange={handleInputChange}
                                    className={inputClass}
                                />
                            </div>

                            {!isLogin && (
                                <div className="space-y-1.5">
                                    <label className={labelClass}>Email Address</label>
                                    <input
                                        type="email"
                                        name="email"
                                        placeholder="name@example.com"
                                        value={formData.email}
                                        onChange={handleInputChange}
                                        className={inputClass}
                                    />
                                </div>
                            )}

                            <div className="space-y-1.5">
                                <label className={labelClass}>Password</label>
                                <input
                                    type="password"
                                    name="password"
                                    placeholder="••••••••••"
                                    value={formData.password}
                                    onChange={handleInputChange}
                                    onKeyDown={handleKeyPress}
                                    className={inputClass}
                                />
                                {!isLogin && (
                                    <p className="text-xs text-slate-500 mt-1">Minimum 8 characters</p>
                                )}
                            </div>
                        </div>

                        {/* Error / success message */}
                        {error && (
                            <div className={`p-4 rounded-xl text-sm font-medium border ${
                                error.startsWith('✓')
                                    ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                                    : 'bg-red-500/10 border-red-500/30 text-red-400'
                            }`}>
                                {error}
                            </div>
                        )}

                        {/* Submit */}
                        <button
                            onClick={handleSubmit}
                            disabled={loading}
                            className="w-full py-4 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 disabled:from-slate-700 disabled:to-slate-700 text-white rounded-xl font-semibold transition-all shadow-xl shadow-violet-500/20 disabled:shadow-none flex items-center justify-center space-x-2 disabled:opacity-70"
                        >
                            {loading ? (
                                <>
                                    <RefreshCw size={20} className="animate-spin"/>
                                    <span>Processing...</span>
                                </>
                            ) : (
                                <span>{isLogin ? 'Sign In' : 'Create Account'}</span>
                            )}
                        </button>

                        {/* Toggle login/register */}
                        <button
                            onClick={toggleMode}
                            className="w-full text-center text-sm text-slate-400 hover:text-slate-300 transition-colors py-2 font-medium hover:underline"
                        >
                            {isLogin ? "Don't have an account? Sign up" : "Already have an account? Sign in"}
                        </button>
                    </div>

                    {/* Footer */}
                    <div className="mt-8 pt-6 border-t border-slate-800/80 flex items-center justify-between">
                        <div className="flex items-center space-x-2 text-xs text-slate-600 font-medium">
                            <ShieldCheck size={14}/>
                            <span>AES-256 Encrypted</span>
                        </div>
                        <div className="text-xs text-slate-600 font-medium">v2.0.1</div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default LoginPage;
