'use client'

import { useState, useEffect } from 'react';
import { Copy, Key, Loader2, Plus, RefreshCw, Trash2 } from 'lucide-react';

interface ApiKeyItem {
    id: number;
    key_prefix: string;
    name: string;
    permissions: string[];
    is_active: boolean;
    created_at: string;
    last_used_at: string | null;
    expires_at: string | null;
}

interface NewKeyResponse extends ApiKeyItem {
    full_key: string;
}

import { apiKeys } from '@/utils/api';

const ApiKeysTab = () => {
    const [keys, setKeys] = useState<ApiKeyItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [creating, setCreating] = useState(false);
    const [newKeyName, setNewKeyName] = useState('');
    const [showCreate, setShowCreate] = useState(false);
    const [newlyCreatedKey, setNewlyCreatedKey] = useState<string | null>(null);
    const [copied, setCopied] = useState(false);

    const fetchKeys = async () => {
        try {
            const data = await apiKeys.list();
            setKeys(data);
        } catch {
            console.error('Failed to fetch API keys');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { fetchKeys(); }, []);

    const handleCreate = async () => {
        if (!newKeyName.trim()) return;
        setCreating(true);
        try {
            const res = await apiKeys.create({ name: newKeyName.trim(), permissions: ['read', 'trade'] }) as NewKeyResponse;
            setNewlyCreatedKey(res.full_key);
            setNewKeyName('');
            setShowCreate(false);
            await fetchKeys();
        } catch {
            alert('Failed to create API key');
        } finally {
            setCreating(false);
        }
    };

    const handleRevoke = async (id: number) => {
        if (!confirm('Are you sure you want to revoke this API key? This cannot be undone.')) return;
        try {
            await apiKeys.revoke(id);
            await fetchKeys();
        } catch {
            alert('Failed to revoke key');
        }
    };

    const handleRotate = async (id: number) => {
        if (!confirm('This will revoke the current key and generate a new one. Continue?')) return;
        try {
            const res = await apiKeys.rotate(id) as NewKeyResponse;
            setNewlyCreatedKey(res.full_key);
            await fetchKeys();
        } catch {
            alert('Failed to rotate key');
        }
    };

    const copyKey = () => {
        if (newlyCreatedKey) {
            navigator.clipboard.writeText(newlyCreatedKey);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h3 className="text-lg font-bold text-slate-100">API Keys</h3>
                    <p className="text-sm text-slate-400 mt-1">
                        Manage API keys for programmatic access. Use the <code className="text-xs bg-slate-800 px-1.5 py-0.5 rounded">X-API-Key</code> header.
                    </p>
                </div>
                <button
                    onClick={() => setShowCreate(true)}
                    className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-semibold transition-all"
                >
                    <Plus size={16} /> New Key
                </button>
            </div>

            {/* Create form */}
            {showCreate && (
                <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4 space-y-3">
                    <input
                        type="text"
                        value={newKeyName}
                        onChange={(e) => setNewKeyName(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
                        placeholder="Key name (e.g. Production Bot)"
                        className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-sm text-slate-200 placeholder-slate-500 focus:border-indigo-500 outline-none"
                        autoFocus
                    />
                    <div className="flex gap-2">
                        <button
                            onClick={handleCreate}
                            disabled={creating || !newKeyName.trim()}
                            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-semibold transition-all disabled:opacity-50 flex items-center gap-2"
                        >
                            {creating ? <Loader2 size={14} className="animate-spin" /> : <Key size={14} />}
                            Generate
                        </button>
                        <button
                            onClick={() => { setShowCreate(false); setNewKeyName(''); }}
                            className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm transition-all"
                        >
                            Cancel
                        </button>
                    </div>
                </div>
            )}

            {/* Newly created key banner */}
            {newlyCreatedKey && (
                <div className="bg-emerald-900/30 border border-emerald-700/50 rounded-xl p-4 space-y-2">
                    <p className="text-sm font-semibold text-emerald-400">
                        Key created! Copy it now — it won&apos;t be shown again.
                    </p>
                    <div className="flex items-center gap-2">
                        <code className="flex-1 px-3 py-2 bg-slate-900 rounded-lg text-xs text-emerald-300 font-mono overflow-x-auto">
                            {newlyCreatedKey}
                        </code>
                        <button
                            onClick={copyKey}
                            className="p-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-all"
                            title="Copy"
                        >
                            <Copy size={16} className={copied ? 'text-emerald-400' : 'text-slate-400'} />
                        </button>
                    </div>
                    <button
                        onClick={() => setNewlyCreatedKey(null)}
                        className="text-xs text-slate-500 hover:text-slate-300 transition-all"
                    >
                        Dismiss
                    </button>
                </div>
            )}

            {/* Key list */}
            {loading ? (
                <div className="flex items-center justify-center py-12">
                    <Loader2 className="animate-spin text-slate-500" size={24} />
                </div>
            ) : keys.length === 0 ? (
                <div className="text-center py-12">
                    <Key className="mx-auto text-slate-600 mb-3" size={40} />
                    <p className="text-slate-400">No API keys yet</p>
                    <p className="text-xs text-slate-500 mt-1">Create a key to access the API programmatically</p>
                </div>
            ) : (
                <div className="space-y-2">
                    {keys.map((k) => (
                        <div
                            key={k.id}
                            className={`flex items-center justify-between p-4 rounded-xl border transition-all ${
                                k.is_active
                                    ? 'bg-slate-800/50 border-slate-700/50'
                                    : 'bg-slate-900/30 border-slate-800/30 opacity-60'
                            }`}
                        >
                            <div className="flex items-center gap-3 flex-1 min-w-0">
                                <Key size={16} className={k.is_active ? 'text-indigo-400' : 'text-slate-600'} />
                                <div className="min-w-0">
                                    <div className="flex items-center gap-2">
                                        <span className="text-sm font-semibold text-slate-200 truncate">{k.name}</span>
                                        <code className="text-xs text-slate-500 bg-slate-800 px-1.5 py-0.5 rounded">{k.key_prefix}...</code>
                                        {!k.is_active && (
                                            <span className="text-xs text-red-400 bg-red-900/30 px-2 py-0.5 rounded">Revoked</span>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-3 mt-1 text-xs text-slate-500">
                                        <span>Created {new Date(k.created_at).toLocaleDateString()}</span>
                                        {k.last_used_at && <span>Last used {new Date(k.last_used_at).toLocaleDateString()}</span>}
                                        {k.expires_at && <span>Expires {new Date(k.expires_at).toLocaleDateString()}</span>}
                                    </div>
                                </div>
                            </div>
                            {k.is_active && (
                                <div className="flex items-center gap-1">
                                    <button
                                        onClick={() => handleRotate(k.id)}
                                        className="p-2 hover:bg-slate-700 rounded-lg transition-all"
                                        title="Rotate key"
                                    >
                                        <RefreshCw size={14} className="text-slate-400" />
                                    </button>
                                    <button
                                        onClick={() => handleRevoke(k.id)}
                                        className="p-2 hover:bg-red-900/30 rounded-lg transition-all"
                                        title="Revoke key"
                                    >
                                        <Trash2 size={14} className="text-red-400" />
                                    </button>
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default ApiKeysTab;
