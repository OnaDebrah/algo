import {useState, KeyboardEvent} from "react";
import {ArrowRight, Eye, EyeOff, RefreshCw, ShieldCheck, Zap} from "lucide-react";
import {api} from "@/utils/api";

interface ResetPasswordPageProps {
    token: string;
    onBackToLogin: () => void;
}

const ResetPasswordPage = ({token, onBackToLogin}: ResetPasswordPageProps) => {
    const [newPassword, setNewPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState(false);

    const handleReset = async () => {
        if (!newPassword.trim()) {
            setError('Please enter a new password');
            return;
        }
        if (newPassword.length < 8) {
            setError('Password must be at least 8 characters long');
            return;
        }
        if (newPassword !== confirmPassword) {
            setError('Passwords do not match');
            return;
        }

        setLoading(true);
        setError('');
        try {
            await api.auth.resetPassword(token, newPassword);
            setSuccess(true);
        } catch (err: unknown) {
            if (typeof err === 'object' && err !== null && 'data' in err) {
                const apiErr = err as { data?: { detail?: string } };
                setError(apiErr.data?.detail || 'Failed to reset password. The link may have expired.');
            } else {
                setError('Failed to reset password. Please try again.');
            }
        } finally {
            setLoading(false);
        }
    };

    const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') handleReset();
    };

    const inputClass = "w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-4 py-3 text-slate-200 placeholder:text-slate-600 focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none transition-all";

    return (
        <div className="min-h-screen w-full flex items-center justify-center bg-slate-950 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-violet-600/10 rounded-full blur-[150px]"/>
            <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-fuchsia-600/10 rounded-full blur-[120px]"/>

            <div className="relative z-10 w-full max-w-xl mx-4">
                <div className="glass-panel rounded-2xl p-8 md:p-10 shadow-2xl">
                    {/* Branding */}
                    <div className="mb-8">
                        <div className="flex items-center space-x-3 mb-6">
                            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center shadow-xl shadow-violet-500/30">
                                <Zap size={24} className="text-white" strokeWidth={2.5}/>
                            </div>
                            <div>
                                <span className="text-2xl font-bold tracking-tight flex items-center gap-2">
                                    <span className="text-slate-100">ORACULUM</span>
                                    <span className="text-xs font-semibold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">AI</span>
                                </span>
                                <div className="h-0.5 w-12 bg-gradient-to-r from-violet-500 to-fuchsia-500 mt-1 rounded-full"/>
                            </div>
                        </div>
                        <h2 className="text-3xl font-bold text-slate-100">Reset Password</h2>
                        <p className="text-slate-400 text-base leading-relaxed mt-2">
                            Enter your new password below.
                        </p>
                    </div>

                    {success ? (
                        <div className="space-y-5">
                            <div className="p-4 rounded-xl text-sm font-medium border bg-emerald-500/10 border-emerald-500/30 text-emerald-400">
                                ✓ Your password has been reset successfully!
                            </div>
                            <button
                                onClick={onBackToLogin}
                                className="w-full py-4 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white rounded-xl font-semibold transition-all shadow-xl shadow-violet-500/20 flex items-center justify-center space-x-2"
                            >
                                <span>Sign In</span>
                                <ArrowRight size={18}/>
                            </button>
                        </div>
                    ) : (
                        <div className="space-y-5">
                            <div className="space-y-1.5">
                                <label className="text-xs font-semibold text-slate-400 tracking-wide">New Password</label>
                                <div className="relative">
                                    <input
                                        type={showPassword ? "text" : "password"}
                                        placeholder="••••••••••"
                                        value={newPassword}
                                        onChange={(e) => setNewPassword(e.target.value)}
                                        onKeyDown={handleKeyPress}
                                        className={`${inputClass} pr-12`}
                                    />
                                    <button
                                        type="button"
                                        onClick={() => setShowPassword(!showPassword)}
                                        className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors"
                                        tabIndex={-1}
                                    >
                                        {showPassword ? <EyeOff size={18}/> : <Eye size={18}/>}
                                    </button>
                                </div>
                                <p className="text-xs text-slate-500 mt-1">Minimum 8 characters</p>
                            </div>

                            <div className="space-y-1.5">
                                <label className="text-xs font-semibold text-slate-400 tracking-wide">Confirm Password</label>
                                <input
                                    type={showPassword ? "text" : "password"}
                                    placeholder="••••••••••"
                                    value={confirmPassword}
                                    onChange={(e) => setConfirmPassword(e.target.value)}
                                    onKeyDown={handleKeyPress}
                                    className={inputClass}
                                />
                            </div>

                            {error && (
                                <div className="p-4 rounded-xl text-sm font-medium border bg-red-500/10 border-red-500/30 text-red-400">
                                    {error}
                                </div>
                            )}

                            <button
                                onClick={handleReset}
                                disabled={loading}
                                className="w-full py-4 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 disabled:from-slate-700 disabled:to-slate-700 text-white rounded-xl font-semibold transition-all shadow-xl shadow-violet-500/20 disabled:shadow-none flex items-center justify-center space-x-2 disabled:opacity-70"
                            >
                                {loading ? (
                                    <>
                                        <RefreshCw size={20} className="animate-spin"/>
                                        <span>Resetting...</span>
                                    </>
                                ) : (
                                    <span>Reset Password</span>
                                )}
                            </button>

                            <button
                                onClick={onBackToLogin}
                                className="w-full text-center text-sm text-slate-400 hover:text-slate-300 transition-colors py-2 font-medium hover:underline"
                            >
                                Back to Sign In
                            </button>
                        </div>
                    )}

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

export default ResetPasswordPage;
