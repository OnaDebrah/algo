import {useState, useEffect} from "react";
import {AlertCircle, Check, CheckCircle, Copy, Eye, EyeOff, Loader2, Shield, X} from "lucide-react";
import {auth} from "@/utils/api";
import {TwoFactorSetupResponse} from "@/types/all_types";

interface TwoFactorSetupProps {
    onClose: () => void;
    onEnabled: () => void;
}

const TwoFactorSetup = ({onClose, onEnabled}: TwoFactorSetupProps) => {
    const [step, setStep] = useState<'qr' | 'backup'>('qr');
    const [setupData, setSetupData] = useState<TwoFactorSetupResponse | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [code, setCode] = useState('');
    const [verifying, setVerifying] = useState(false);
    const [showSecret, setShowSecret] = useState(false);
    const [backupCodes, setBackupCodes] = useState<string[]>([]);
    const [copied, setCopied] = useState(false);
    const [savedConfirmed, setSavedConfirmed] = useState(false);

    useEffect(() => {
        const fetchSetup = async () => {
            try {
                const response = await auth.setup2FA();
                setSetupData(response);
            } catch (err: any) {
                setError(err?.response?.data?.detail || 'Failed to initialize 2FA setup');
            } finally {
                setLoading(false);
            }
        };
        fetchSetup();
    }, []);

    const handleVerify = async () => {
        if (code.length !== 6) return;
        setVerifying(true);
        setError('');
        try {
            const response = await auth.verifySetup2FA(code);
            if (response?.backup_codes) {
                setBackupCodes(response.backup_codes);
                setStep('backup');
            }
        } catch (err: any) {
            setError(err?.response?.data?.detail || 'Invalid verification code');
            setCode('');
        } finally {
            setVerifying(false);
        }
    };

    useEffect(() => {
        if (code.length === 6 && step === 'qr') {
            handleVerify();
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [code]);

    const handleCopyAll = async () => {
        try {
            await navigator.clipboard.writeText(backupCodes.join('\n'));
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch {
            // Fallback for clipboard API not available
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/70 backdrop-blur-sm"
                onClick={onClose}
            />

            {/* Modal */}
            <div className="relative z-10 w-full max-w-lg mx-4 bg-slate-900 border border-slate-800/50 rounded-2xl shadow-2xl overflow-hidden">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-slate-800/50">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center shadow-lg shadow-violet-500/30">
                            <Shield size={20} className="text-white"/>
                        </div>
                        <div>
                            <h3 className="text-lg font-bold text-slate-100">
                                {step === 'qr' ? 'Set Up Two-Factor Authentication' : 'Backup Codes'}
                            </h3>
                            <p className="text-xs text-slate-400 mt-0.5">
                                {step === 'qr' ? 'Scan the QR code with your authenticator app' : 'Save these codes in a safe place'}
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="text-slate-500 hover:text-slate-300 transition-colors p-1"
                    >
                        <X size={20}/>
                    </button>
                </div>

                {/* Content */}
                <div className="p-6">
                    {loading ? (
                        <div className="flex items-center justify-center py-12">
                            <Loader2 size={32} className="animate-spin text-violet-400"/>
                        </div>
                    ) : step === 'qr' ? (
                        <div className="space-y-6">
                            {/* QR Code */}
                            {setupData && (
                                <div className="flex flex-col items-center">
                                    <div className="bg-white p-4 rounded-xl">
                                        <img
                                            src={`data:image/png;base64,${setupData.qr_image_base64}`}
                                            alt="2FA QR Code"
                                            className="w-48 h-48"
                                        />
                                    </div>

                                    {/* Show secret toggle */}
                                    <button
                                        onClick={() => setShowSecret(!showSecret)}
                                        className="flex items-center gap-2 text-sm text-violet-400 hover:text-violet-300 transition-colors mt-4"
                                    >
                                        {showSecret ? <EyeOff size={14}/> : <Eye size={14}/>}
                                        {showSecret ? 'Hide secret key' : "Can't scan? Show secret key"}
                                    </button>

                                    {showSecret && (
                                        <div className="mt-3 w-full p-3 bg-slate-800/50 border border-slate-700/50 rounded-lg">
                                            <p className="text-xs text-slate-400 mb-1">Manual entry key:</p>
                                            <p className="text-sm font-mono text-slate-200 break-all select-all">
                                                {setupData.secret}
                                            </p>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Verification code input */}
                            <div className="space-y-2">
                                <label className="text-sm font-semibold text-slate-300">
                                    Enter the 6-digit code from your app
                                </label>
                                <input
                                    type="text"
                                    maxLength={6}
                                    inputMode="numeric"
                                    value={code}
                                    onChange={(e) => {
                                        const val = e.target.value.replace(/[^0-9]/g, '');
                                        setCode(val);
                                        setError('');
                                    }}
                                    placeholder="000000"
                                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg px-4 py-3 text-slate-200 text-center text-xl font-mono tracking-[0.4em] placeholder:text-slate-700 focus:border-violet-500 focus:ring-1 focus:ring-violet-500 outline-none transition-all"
                                    disabled={verifying}
                                    autoFocus
                                />
                            </div>

                            {verifying && (
                                <div className="flex items-center justify-center gap-2 text-sm text-slate-400">
                                    <Loader2 size={16} className="animate-spin"/>
                                    <span>Verifying...</span>
                                </div>
                            )}

                            {error && (
                                <div className="p-3 rounded-xl text-sm font-medium border bg-red-500/10 border-red-500/30 text-red-400 flex items-center gap-2">
                                    <AlertCircle size={16}/>
                                    {error}
                                </div>
                            )}
                        </div>
                    ) : (
                        /* Backup Codes Step */
                        <div className="space-y-6">
                            <div className="p-4 bg-amber-500/10 border border-amber-500/30 rounded-xl flex items-start gap-3">
                                <AlertCircle className="text-amber-400 shrink-0 mt-0.5" size={18}/>
                                <div className="text-sm text-slate-300">
                                    <span className="font-bold text-amber-400">Important:</span> Save these backup codes somewhere safe.
                                    Each code can only be used once if you lose access to your authenticator app.
                                </div>
                            </div>

                            {/* Backup codes grid */}
                            <div className="grid grid-cols-2 gap-2">
                                {backupCodes.map((backupCode, index) => (
                                    <div
                                        key={index}
                                        className="px-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg font-mono text-sm text-slate-200 text-center"
                                    >
                                        {backupCode}
                                    </div>
                                ))}
                            </div>

                            {/* Copy all button */}
                            <button
                                onClick={handleCopyAll}
                                className="w-full flex items-center justify-center gap-2 py-2.5 bg-slate-800/50 border border-slate-700/50 rounded-xl text-sm text-slate-300 hover:text-slate-100 hover:border-slate-600 transition-all"
                            >
                                {copied ? (
                                    <>
                                        <CheckCircle size={16} className="text-emerald-400"/>
                                        <span className="text-emerald-400">Copied!</span>
                                    </>
                                ) : (
                                    <>
                                        <Copy size={16}/>
                                        <span>Copy All</span>
                                    </>
                                )}
                            </button>

                            {/* Saved confirmation */}
                            <label className="flex items-center gap-3 cursor-pointer p-3 bg-slate-800/30 rounded-xl border border-slate-700 hover:border-slate-600 transition-all">
                                <input
                                    type="checkbox"
                                    checked={savedConfirmed}
                                    onChange={(e) => setSavedConfirmed(e.target.checked)}
                                    className="w-5 h-5 rounded"
                                />
                                <span className="text-sm text-slate-300">
                                    I&apos;ve saved these codes in a safe place
                                </span>
                            </label>

                            {/* Done button */}
                            <button
                                onClick={onEnabled}
                                disabled={!savedConfirmed}
                                className="w-full py-3 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 disabled:from-slate-700 disabled:to-slate-700 text-white rounded-xl font-semibold transition-all shadow-xl shadow-violet-500/20 disabled:shadow-none disabled:opacity-70 flex items-center justify-center gap-2"
                            >
                                <Check size={18}/>
                                Done
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default TwoFactorSetup;
