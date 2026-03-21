'use client';

import React, { useCallback, useEffect, useState } from 'react';
import { Copy, Crown, LogOut, Plus, Shield, Trash2, UserPlus, Users, X } from 'lucide-react';
import { teamsApi } from '@/utils/api';
import CommentThread from './CommentThread';

interface Team {
    id: number;
    name: string;
    description: string | null;
    invite_code: string | null;
    role: string;
    created_at: string | null;
}

interface Member {
    user_id: number;
    username: string;
    email: string;
    role: string;
    joined_at: string | null;
}

const TeamDashboard: React.FC = () => {
    const [teams, setTeams] = useState<Team[]>([]);
    const [selectedTeam, setSelectedTeam] = useState<Team | null>(null);
    const [members, setMembers] = useState<Member[]>([]);
    const [comments, setComments] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [showCreate, setShowCreate] = useState(false);
    const [showJoin, setShowJoin] = useState(false);
    const [createName, setCreateName] = useState('');
    const [createDesc, setCreateDesc] = useState('');
    const [joinTeamId, setJoinTeamId] = useState('');
    const [joinCode, setJoinCode] = useState('');
    const [copied, setCopied] = useState(false);
    const [tab, setTab] = useState<'members' | 'activity'>('members');

    const fetchTeams = useCallback(async () => {
        try {
            const data = await teamsApi.list();
            setTeams(data);
            if (data.length > 0 && !selectedTeam) {
                setSelectedTeam(data[0]);
            }
        } catch (err) {
            console.error('Failed to fetch teams', err);
        } finally {
            setLoading(false);
        }
    }, [selectedTeam]);

    const fetchTeamData = useCallback(async (team: Team) => {
        try {
            const [m, c] = await Promise.all([
                teamsApi.getMembers(team.id),
                teamsApi.getComments(team.id),
            ]);
            setMembers(m);
            setComments(c);
        } catch (err) {
            console.error('Failed to fetch team data', err);
        }
    }, []);

    useEffect(() => { fetchTeams(); }, [fetchTeams]);
    useEffect(() => { if (selectedTeam) fetchTeamData(selectedTeam); }, [selectedTeam, fetchTeamData]);

    const handleCreate = async () => {
        if (!createName.trim()) return;
        try {
            await teamsApi.create({ name: createName.trim(), description: createDesc.trim() || undefined });
            setShowCreate(false);
            setCreateName('');
            setCreateDesc('');
            setSelectedTeam(null);
            await fetchTeams();
        } catch (err) {
            console.error('Failed to create team', err);
        }
    };

    const handleJoin = async () => {
        if (!joinTeamId || !joinCode) return;
        try {
            await teamsApi.join(parseInt(joinTeamId), joinCode);
            setShowJoin(false);
            setJoinTeamId('');
            setJoinCode('');
            setSelectedTeam(null);
            await fetchTeams();
        } catch (err) {
            console.error('Failed to join team', err);
        }
    };

    const handleRemoveMember = async (userId: number) => {
        if (!selectedTeam) return;
        try {
            await teamsApi.removeMember(selectedTeam.id, userId);
            await fetchTeamData(selectedTeam);
        } catch (err) {
            console.error('Failed to remove member', err);
        }
    };

    const handleRoleChange = async (userId: number, newRole: string) => {
        if (!selectedTeam) return;
        try {
            await teamsApi.updateMemberRole(selectedTeam.id, userId, newRole);
            await fetchTeamData(selectedTeam);
        } catch (err) {
            console.error('Failed to update role', err);
        }
    };

    const handleAddComment = async (content: string, parentId?: number) => {
        if (!selectedTeam) return;
        await teamsApi.addComment(selectedTeam.id, {
            target_type: 'team',
            target_id: selectedTeam.id,
            content,
            parent_comment_id: parentId,
        });
        const c = await teamsApi.getComments(selectedTeam.id);
        setComments(c);
    };

    const copyInvite = () => {
        if (selectedTeam?.invite_code) {
            navigator.clipboard.writeText(selectedTeam.invite_code);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    };

    const roleIcon = (role: string) => {
        if (role === 'owner') return <Crown size={14} className="text-amber-400" />;
        if (role === 'admin') return <Shield size={14} className="text-violet-400" />;
        return null;
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="w-8 h-8 border-2 border-violet-500 border-t-transparent rounded-full animate-spin" />
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-foreground">Teams</h1>
                    <p className="text-sm text-muted-foreground mt-1">Collaborate with your team on strategies and backtests</p>
                </div>
                <div className="flex gap-2">
                    <button
                        onClick={() => setShowJoin(true)}
                        className="px-4 py-2 border border-border rounded-lg text-sm font-medium text-foreground hover:bg-accent transition-colors flex items-center gap-2"
                    >
                        <UserPlus size={16} /> Join Team
                    </button>
                    <button
                        onClick={() => setShowCreate(true)}
                        className="px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
                    >
                        <Plus size={16} /> New Team
                    </button>
                </div>
            </div>

            {/* Create Modal */}
            {showCreate && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                    <div className="bg-card border border-border rounded-xl p-6 w-full max-w-md space-y-4">
                        <div className="flex items-center justify-between">
                            <h2 className="text-lg font-semibold text-foreground">Create Team</h2>
                            <button onClick={() => setShowCreate(false)} className="text-muted-foreground hover:text-foreground"><X size={20} /></button>
                        </div>
                        <input
                            value={createName}
                            onChange={e => setCreateName(e.target.value)}
                            placeholder="Team name"
                            className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-violet-500"
                        />
                        <textarea
                            value={createDesc}
                            onChange={e => setCreateDesc(e.target.value)}
                            placeholder="Description (optional)"
                            rows={3}
                            className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-violet-500 resize-none"
                        />
                        <button
                            onClick={handleCreate}
                            disabled={!createName.trim()}
                            className="w-full py-2 bg-violet-600 hover:bg-violet-500 text-white rounded-lg text-sm font-medium disabled:opacity-50 transition-colors"
                        >
                            Create Team
                        </button>
                    </div>
                </div>
            )}

            {/* Join Modal */}
            {showJoin && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                    <div className="bg-card border border-border rounded-xl p-6 w-full max-w-md space-y-4">
                        <div className="flex items-center justify-between">
                            <h2 className="text-lg font-semibold text-foreground">Join Team</h2>
                            <button onClick={() => setShowJoin(false)} className="text-muted-foreground hover:text-foreground"><X size={20} /></button>
                        </div>
                        <input
                            value={joinTeamId}
                            onChange={e => setJoinTeamId(e.target.value)}
                            placeholder="Team ID"
                            type="number"
                            className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-violet-500"
                        />
                        <input
                            value={joinCode}
                            onChange={e => setJoinCode(e.target.value)}
                            placeholder="Invite code"
                            className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-violet-500"
                        />
                        <button
                            onClick={handleJoin}
                            disabled={!joinTeamId || !joinCode}
                            className="w-full py-2 bg-violet-600 hover:bg-violet-500 text-white rounded-lg text-sm font-medium disabled:opacity-50 transition-colors"
                        >
                            Join
                        </button>
                    </div>
                </div>
            )}

            {teams.length === 0 ? (
                <div className="text-center py-16 bg-card border border-border rounded-xl">
                    <Users size={48} className="mx-auto mb-4 text-muted-foreground opacity-50" />
                    <h3 className="text-lg font-medium text-foreground mb-2">No teams yet</h3>
                    <p className="text-sm text-muted-foreground mb-4">Create a team or join one with an invite code</p>
                </div>
            ) : (
                <div className="grid grid-cols-12 gap-6">
                    {/* Team List */}
                    <div className="col-span-3 space-y-2">
                        {teams.map(team => (
                            <button
                                key={team.id}
                                onClick={() => setSelectedTeam(team)}
                                className={`w-full text-left p-3 rounded-lg border transition-colors ${
                                    selectedTeam?.id === team.id
                                        ? 'bg-violet-600/10 border-violet-500/50 text-foreground'
                                        : 'bg-card border-border text-foreground/80 hover:bg-accent'
                                }`}
                            >
                                <div className="flex items-center gap-2">
                                    <Users size={16} className="text-violet-400 shrink-0" />
                                    <span className="font-medium text-sm truncate">{team.name}</span>
                                </div>
                                <span className="text-xs text-muted-foreground ml-6 capitalize">{team.role}</span>
                            </button>
                        ))}
                    </div>

                    {/* Team Detail */}
                    {selectedTeam && (
                        <div className="col-span-9 space-y-4">
                            {/* Team Header */}
                            <div className="bg-card border border-border rounded-xl p-5">
                                <div className="flex items-start justify-between">
                                    <div>
                                        <h2 className="text-xl font-semibold text-foreground">{selectedTeam.name}</h2>
                                        {selectedTeam.description && (
                                            <p className="text-sm text-muted-foreground mt-1">{selectedTeam.description}</p>
                                        )}
                                    </div>
                                    {selectedTeam.invite_code && (
                                        <button
                                            onClick={copyInvite}
                                            className="flex items-center gap-2 px-3 py-1.5 bg-accent border border-border rounded-lg text-xs text-foreground hover:bg-accent/80 transition-colors"
                                        >
                                            <Copy size={12} />
                                            {copied ? 'Copied!' : `Invite: ${selectedTeam.invite_code.slice(0, 8)}...`}
                                        </button>
                                    )}
                                </div>

                                {/* Tabs */}
                                <div className="flex gap-4 mt-4 border-t border-border pt-3">
                                    {(['members', 'activity'] as const).map(t => (
                                        <button
                                            key={t}
                                            onClick={() => setTab(t)}
                                            className={`text-sm font-medium capitalize pb-1 border-b-2 transition-colors ${
                                                tab === t ? 'border-violet-500 text-foreground' : 'border-transparent text-muted-foreground hover:text-foreground'
                                            }`}
                                        >
                                            {t}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Tab Content */}
                            {tab === 'members' && (
                                <div className="bg-card border border-border rounded-xl divide-y divide-border">
                                    {members.map(member => (
                                        <div key={member.user_id} className="flex items-center justify-between px-5 py-3">
                                            <div className="flex items-center gap-3">
                                                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center text-white text-xs font-bold">
                                                    {member.username.charAt(0).toUpperCase()}
                                                </div>
                                                <div>
                                                    <div className="flex items-center gap-2">
                                                        <span className="text-sm font-medium text-foreground">{member.username}</span>
                                                        {roleIcon(member.role)}
                                                    </div>
                                                    <span className="text-xs text-muted-foreground">{member.email}</span>
                                                </div>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                {(selectedTeam.role === 'owner' || selectedTeam.role === 'admin') && member.role !== 'owner' && (
                                                    <>
                                                        <select
                                                            value={member.role}
                                                            onChange={e => handleRoleChange(member.user_id, e.target.value)}
                                                            className="bg-background border border-border rounded px-2 py-1 text-xs text-foreground"
                                                        >
                                                            <option value="admin">Admin</option>
                                                            <option value="member">Member</option>
                                                            <option value="viewer">Viewer</option>
                                                        </select>
                                                        <button
                                                            onClick={() => handleRemoveMember(member.user_id)}
                                                            className="p-1.5 text-red-400 hover:bg-red-500/10 rounded transition-colors"
                                                            title="Remove member"
                                                        >
                                                            <Trash2 size={14} />
                                                        </button>
                                                    </>
                                                )}
                                                {member.role !== 'owner' && selectedTeam.role !== 'owner' && selectedTeam.role !== 'admin' && (
                                                    <span className="text-xs text-muted-foreground capitalize px-2 py-1 bg-accent rounded">{member.role}</span>
                                                )}
                                            </div>
                                        </div>
                                    ))}
                                    {/* Self-leave for non-owners */}
                                    {selectedTeam.role !== 'owner' && (
                                        <div className="px-5 py-3">
                                            <button
                                                onClick={async () => {
                                                    const me = members.find(m => m.role === selectedTeam.role);
                                                    if (me) {
                                                        await teamsApi.removeMember(selectedTeam.id, me.user_id);
                                                        setSelectedTeam(null);
                                                        await fetchTeams();
                                                    }
                                                }}
                                                className="flex items-center gap-2 text-sm text-red-400 hover:text-red-300 transition-colors"
                                            >
                                                <LogOut size={14} /> Leave Team
                                            </button>
                                        </div>
                                    )}
                                </div>
                            )}

                            {tab === 'activity' && (
                                <div className="bg-card border border-border rounded-xl p-5">
                                    <CommentThread comments={comments} onAddComment={handleAddComment} />
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default TeamDashboard;
