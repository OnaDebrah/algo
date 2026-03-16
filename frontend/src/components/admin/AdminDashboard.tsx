/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useCallback, useEffect, useState } from 'react';
import {
    Activity,
    BarChart3,
    Check,
    ChevronDown,
    Cpu,
    RefreshCw,
    Search,
    ShieldCheck,
    TrendingUp,
    UserCheck,
    Users,
    X,
    Zap,
} from 'lucide-react';
import { api } from '@/utils/api';
import { AdminStats, AdminUserListItem } from '@/types/all_types';

/* ─── Tier badge ───────────────────────────────────────────────── */
const tierColor: Record<string, string> = {
    FREE: 'bg-slate-700 text-slate-300',
    BASIC: 'bg-blue-500/20 text-blue-400 border border-blue-500/30',
    PRO: 'bg-violet-500/20 text-violet-400 border border-violet-500/30',
    ENTERPRISE: 'bg-amber-500/20 text-amber-400 border border-amber-500/30',
};

const TierBadge = ({ tier }: { tier: string }) => (
    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${tierColor[tier] || tierColor.FREE}`}>
        {tier}
    </span>
);

/* ─── Stat Card ────────────────────────────────────────────────── */
const StatCard = ({
    label,
    value,
    icon: Icon,
    accent = 'text-violet-400',
}: {
    label: string;
    value: number | string;
    icon: any;
    accent?: string;
}) => (
    <div className="bg-slate-800/50 border border-slate-700/60 rounded-xl p-5 flex items-center gap-4">
        <div className={`w-11 h-11 rounded-xl flex items-center justify-center bg-slate-700/50 ${accent}`}>
            <Icon size={22} />
        </div>
        <div>
            <p className="text-2xl font-bold text-slate-100">{value}</p>
            <p className="text-xs text-slate-500">{label}</p>
        </div>
    </div>
);

/* ─── Main Component ───────────────────────────────────────────── */
const AdminDashboard: React.FC = () => {
    const [stats, setStats] = useState<AdminStats | null>(null);
    const [users, setUsers] = useState<AdminUserListItem[]>([]);
    const [usageLogs, setUsageLogs] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [search, setSearch] = useState('');
    const [tierFilter, setTierFilter] = useState('');
    const [activeTab, setActiveTab] = useState<'users' | 'submissions' | 'activity'>('users');
    const [submissions, setSubmissions] = useState<any[]>([]);
    const [rejectingId, setRejectingId] = useState<number | null>(null);
    const [rejectReason, setRejectReason] = useState('');

    const fetchAll = useCallback(async () => {
        setLoading(true);
        try {
            const [s, u, logs, subs] = await Promise.all([
                api.admin.getStats(),
                api.admin.getUsers({ search: search || undefined, tier: tierFilter || undefined, limit: 50 }),
                api.admin.getUsage({ limit: 50 }),
                api.admin.getSubmissions('pending_review'),
            ]);
            setStats(s);
            setUsers(u);
            setUsageLogs(logs);
            setSubmissions(subs as any[]);
        } catch (err) {
            console.error('Admin data fetch failed', err);
        } finally {
            setLoading(false);
        }
    }, [search, tierFilter]);

    useEffect(() => {
        fetchAll();
    }, [fetchAll]);

    const handleTierChange = async (userId: number, newTier: string) => {
        try {
            await api.admin.updateTier(userId, newTier);
            fetchAll();
        } catch (err) {
            console.error('Tier update failed', err);
        }
    };

    const handleStatusToggle = async (userId: number, currentActive: boolean) => {
        try {
            await api.admin.updateStatus(userId, !currentActive);
            fetchAll();
        } catch (err) {
            console.error('Status toggle failed', err);
        }
    };

    const handleApprove = async (strategyId: number) => {
        try {
            await api.admin.approveSubmission(strategyId);
            fetchAll();
        } catch (err) {
            console.error('Approval failed', err);
        }
    };

    const handleReject = async (strategyId: number) => {
        if (!rejectReason.trim()) return;
        try {
            await api.admin.rejectSubmission(strategyId, rejectReason);
            setRejectingId(null);
            setRejectReason('');
            fetchAll();
        } catch (err) {
            console.error('Rejection failed', err);
        }
    };

    if (loading && !stats) {
        return (
            <div className="flex items-center justify-center min-h-[60vh]">
                <div className="w-10 h-10 border-3 border-violet-500 border-t-transparent rounded-full animate-spin" />
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center">
                        <ShieldCheck size={22} className="text-white" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-slate-100">Admin Dashboard</h1>
                        <p className="text-sm text-slate-500">System overview & user management</p>
                    </div>
                </div>
                <button
                    onClick={fetchAll}
                    className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm text-slate-300 transition-colors"
                >
                    <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
                    Refresh
                </button>
            </div>

            {/* Stats Row */}
            {stats && (
                <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4">
                    <StatCard label="Total Users" value={stats.total_users} icon={Users} accent="text-blue-400" />
                    <StatCard label="Active Today" value={stats.active_today} icon={UserCheck} accent="text-emerald-400" />
                    <StatCard label="Total Backtests" value={stats.total_backtests} icon={BarChart3} accent="text-violet-400" />
                    <StatCard label="Today" value={stats.backtests_today} icon={Zap} accent="text-amber-400" />
                    <StatCard label="Live Strategies" value={stats.active_live_strategies} icon={TrendingUp} accent="text-cyan-400" />
                    <StatCard label="Models Trained" value={stats.models_trained} icon={Cpu} accent="text-fuchsia-400" />
                </div>
            )}

            {/* Tier distribution pills */}
            {stats && (
                <div className="flex items-center gap-3 flex-wrap">
                    <span className="text-xs text-slate-500 font-medium">Users by tier:</span>
                    {Object.entries(stats.users_by_tier).map(([tier, count]) => (
                        <span key={tier} className="flex items-center gap-1.5">
                            <TierBadge tier={tier} />
                            <span className="text-sm text-slate-400 font-medium">{count}</span>
                        </span>
                    ))}
                </div>
            )}

            {/* Tabs */}
            <div className="flex items-center gap-1 border-b border-slate-800">
                {([
                    { key: 'users' as const, label: 'User Management', badge: 0 },
                    { key: 'submissions' as const, label: 'Submissions', badge: submissions.length },
                    { key: 'activity' as const, label: 'Activity Feed', badge: 0 },
                ]).map(({ key, label, badge }) => (
                    <button
                        key={key}
                        onClick={() => setActiveTab(key as any)}
                        className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors flex items-center gap-2 ${
                            activeTab === key
                                ? 'border-violet-500 text-slate-100'
                                : 'border-transparent text-slate-500 hover:text-slate-300'
                        }`}
                    >
                        {label}
                        {badge ? (
                            <span className="bg-amber-500/20 text-amber-400 text-[10px] font-bold px-1.5 py-0.5 rounded-full">
                                {badge}
                            </span>
                        ) : null}
                    </button>
                ))}
            </div>

            {/* Users Tab */}
            {activeTab === 'users' && (
                <div className="space-y-4">
                    {/* Search + Filter */}
                    <div className="flex items-center gap-3">
                        <div className="relative flex-1 max-w-md">
                            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                            <input
                                value={search}
                                onChange={(e) => setSearch(e.target.value)}
                                placeholder="Search users..."
                                className="w-full pl-10 pr-4 py-2.5 bg-slate-800/50 border border-slate-700 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-violet-500/50"
                            />
                        </div>
                        <select
                            value={tierFilter}
                            onChange={(e) => setTierFilter(e.target.value)}
                            className="px-3 py-2.5 bg-slate-800/50 border border-slate-700 rounded-lg text-sm text-slate-300 focus:outline-none focus:border-violet-500/50"
                        >
                            <option value="">All Tiers</option>
                            <option value="FREE">Free</option>
                            <option value="BASIC">Basic</option>
                            <option value="PRO">Pro</option>
                            <option value="ENTERPRISE">Enterprise</option>
                        </select>
                    </div>

                    {/* Users Table */}
                    <div className="bg-slate-800/30 border border-slate-700/60 rounded-xl overflow-hidden">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-slate-700/60">
                                    <th className="text-left px-4 py-3 text-slate-500 font-medium">User</th>
                                    <th className="text-left px-4 py-3 text-slate-500 font-medium">Tier</th>
                                    <th className="text-left px-4 py-3 text-slate-500 font-medium">Status</th>
                                    <th className="text-left px-4 py-3 text-slate-500 font-medium">Backtests</th>
                                    <th className="text-left px-4 py-3 text-slate-500 font-medium">Joined</th>
                                    <th className="text-left px-4 py-3 text-slate-500 font-medium">Last Login</th>
                                    <th className="text-right px-4 py-3 text-slate-500 font-medium">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-700/40">
                                {users.map((u) => (
                                    <tr key={u.id} className="hover:bg-slate-800/30 transition-colors">
                                        <td className="px-4 py-3">
                                            <div>
                                                <p className="text-slate-200 font-medium">{u.username}</p>
                                                <p className="text-xs text-slate-500">{u.email}</p>
                                            </div>
                                        </td>
                                        <td className="px-4 py-3">
                                            <select
                                                value={u.tier}
                                                onChange={(e) => handleTierChange(u.id, e.target.value)}
                                                className="bg-transparent text-xs font-bold px-2 py-1 rounded border border-slate-600 text-slate-300 focus:outline-none focus:border-violet-500/50"
                                            >
                                                <option value="FREE">FREE</option>
                                                <option value="BASIC">BASIC</option>
                                                <option value="PRO">PRO</option>
                                                <option value="ENTERPRISE">ENTERPRISE</option>
                                            </select>
                                        </td>
                                        <td className="px-4 py-3">
                                            <button
                                                onClick={() => handleStatusToggle(u.id, u.is_active)}
                                                className={`text-xs font-medium px-2.5 py-1 rounded-full ${
                                                    u.is_active
                                                        ? 'bg-emerald-500/20 text-emerald-400'
                                                        : 'bg-red-500/20 text-red-400'
                                                }`}
                                            >
                                                {u.is_active ? 'Active' : 'Disabled'}
                                            </button>
                                        </td>
                                        <td className="px-4 py-3 text-slate-400">{u.backtest_count}</td>
                                        <td className="px-4 py-3 text-slate-500 text-xs">
                                            {u.created_at ? new Date(u.created_at).toLocaleDateString() : '-'}
                                        </td>
                                        <td className="px-4 py-3 text-slate-500 text-xs">
                                            {u.last_login ? new Date(u.last_login).toLocaleDateString() : 'Never'}
                                        </td>
                                        <td className="px-4 py-3 text-right">
                                            {u.is_superuser && (
                                                <span className="text-xs text-amber-400 font-bold">ADMIN</span>
                                            )}
                                        </td>
                                    </tr>
                                ))}
                                {users.length === 0 && (
                                    <tr>
                                        <td colSpan={7} className="px-4 py-8 text-center text-slate-500">
                                            No users found
                                        </td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* Submissions Tab */}
            {activeTab === 'submissions' && (
                <div className="bg-slate-800/30 border border-slate-700/60 rounded-xl overflow-hidden">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b border-slate-700/60">
                                <th className="text-left px-4 py-3 text-slate-500 font-medium">Strategy</th>
                                <th className="text-left px-4 py-3 text-slate-500 font-medium">Creator</th>
                                <th className="text-left px-4 py-3 text-slate-500 font-medium">Category</th>
                                <th className="text-right px-4 py-3 text-slate-500 font-medium">Sharpe</th>
                                <th className="text-right px-4 py-3 text-slate-500 font-medium">Return</th>
                                <th className="text-right px-4 py-3 text-slate-500 font-medium">Price</th>
                                <th className="text-left px-4 py-3 text-slate-500 font-medium">Submitted</th>
                                <th className="text-right px-4 py-3 text-slate-500 font-medium">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-700/40">
                            {submissions.map((s: any) => (
                                <tr key={s.id} className="hover:bg-slate-800/30 transition-colors">
                                    <td className="px-4 py-3">
                                        <p className="text-slate-200 font-medium">{s.name}</p>
                                        <p className="text-xs text-slate-500">{s.complexity}</p>
                                    </td>
                                    <td className="px-4 py-3 text-slate-400">{s.creator_name}</td>
                                    <td className="px-4 py-3">
                                        <span className="text-xs bg-slate-700 text-slate-300 px-2 py-0.5 rounded-full">
                                            {s.category}
                                        </span>
                                    </td>
                                    <td className="px-4 py-3 text-right text-blue-400 font-mono">
                                        {(s.sharpe_ratio || 0).toFixed(2)}
                                    </td>
                                    <td className="px-4 py-3 text-right text-emerald-400 font-mono">
                                        {(s.total_return || 0).toFixed(1)}%
                                    </td>
                                    <td className="px-4 py-3 text-right text-slate-300 font-mono">
                                        ${(s.price || 0).toFixed(0)}
                                    </td>
                                    <td className="px-4 py-3 text-slate-500 text-xs">
                                        {s.submitted_at ? new Date(s.submitted_at).toLocaleDateString() : '-'}
                                    </td>
                                    <td className="px-4 py-3 text-right">
                                        {rejectingId === s.id ? (
                                            <div className="flex items-center gap-2 justify-end">
                                                <input
                                                    value={rejectReason}
                                                    onChange={(e) => setRejectReason(e.target.value)}
                                                    placeholder="Reason..."
                                                    className="w-40 px-2 py-1 bg-slate-700 border border-slate-600 rounded text-xs text-slate-200 focus:outline-none"
                                                />
                                                <button
                                                    onClick={() => handleReject(s.id)}
                                                    disabled={!rejectReason.trim()}
                                                    className="px-2 py-1 bg-red-500/20 text-red-400 rounded text-xs font-medium hover:bg-red-500/30 disabled:opacity-40"
                                                >
                                                    Confirm
                                                </button>
                                                <button
                                                    onClick={() => { setRejectingId(null); setRejectReason(''); }}
                                                    className="text-slate-500 hover:text-slate-300"
                                                >
                                                    <X size={14} />
                                                </button>
                                            </div>
                                        ) : (
                                            <div className="flex items-center gap-2 justify-end">
                                                <button
                                                    onClick={() => handleApprove(s.id)}
                                                    className="flex items-center gap-1 px-2.5 py-1 bg-emerald-500/20 text-emerald-400 rounded text-xs font-medium hover:bg-emerald-500/30"
                                                >
                                                    <Check size={12} /> Approve
                                                </button>
                                                <button
                                                    onClick={() => setRejectingId(s.id)}
                                                    className="flex items-center gap-1 px-2.5 py-1 bg-red-500/20 text-red-400 rounded text-xs font-medium hover:bg-red-500/30"
                                                >
                                                    <X size={12} /> Reject
                                                </button>
                                            </div>
                                        )}
                                    </td>
                                </tr>
                            ))}
                            {submissions.length === 0 && (
                                <tr>
                                    <td colSpan={8} className="px-4 py-8 text-center text-slate-500">
                                        No pending submissions
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            )}

            {/* Activity Tab */}
            {activeTab === 'activity' && (
                <div className="bg-slate-800/30 border border-slate-700/60 rounded-xl overflow-hidden">
                    <div className="divide-y divide-slate-700/40">
                        {usageLogs.map((log: any) => (
                            <div key={log.id} className="px-4 py-3 flex items-center gap-4 hover:bg-slate-800/30 transition-colors">
                                <div className="w-8 h-8 rounded-full bg-slate-700/50 flex items-center justify-center">
                                    <Activity size={14} className="text-slate-400" />
                                </div>
                                <div className="flex-1 min-w-0">
                                    <p className="text-sm text-slate-200">
                                        <span className="font-medium">{log.username}</span>{' '}
                                        <span className="text-slate-400">{log.action}</span>
                                    </p>
                                    <p className="text-xs text-slate-500">{log.email}</p>
                                </div>
                                <span className="text-xs text-slate-500 shrink-0">
                                    {log.timestamp
                                        ? new Date(log.timestamp).toLocaleString()
                                        : '-'}
                                </span>
                            </div>
                        ))}
                        {usageLogs.length === 0 && (
                            <div className="px-4 py-8 text-center text-slate-500">No recent activity</div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default AdminDashboard;
