'use client'
import React, { useEffect, useState } from 'react';
import { X, Search, FileText, Calendar, ChevronRight, Loader2, FolderOpen } from 'lucide-react';
import { portfolio } from '@/utils/api';
import { Portfolio } from '@/types/all_types';

interface LoadConfigModalProps {
    mode: 'single' | 'multi';
    onClose: () => void;
    onSelect: (config: any) => void;
}

const LoadConfigModal: React.FC<LoadConfigModalProps> = ({ mode, onClose, onSelect }: LoadConfigModalProps) => {
    const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');

    useEffect(() => {
        const fetchPortfolios = async () => {
            try {
                setLoading(true);
                const data = await portfolio.list();

                // Filter portfolios that have a saved config in the description
                const filtered = data.filter(p => {
                    if (!p.description) return false;
                    try {
                        const config = JSON.parse(p.description);
                        if (mode === 'single') {
                            // Single backtests should have a symbol or type: 'single'
                            return config.type === 'single' || (config.symbol && !config.symbols);
                        } else {
                            // Multi backtests should have symbols or type: 'multi'
                            return config.type === 'multi' || config.symbols || config.allocations;
                        }
                    } catch (e) {
                        return false;
                    }
                });

                setPortfolios(filtered);
            } catch (error) {
                console.error('Failed to fetch portfolios:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchPortfolios();
    }, [mode]);

    const filteredList = portfolios.filter(p =>
        p.name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const handleSelect = (p: Portfolio) => {
        try {
            const config = JSON.parse(p.description || '{}');
            onSelect(config);
            onClose();
        } catch (e) {
            console.error('Failed to parse config:', e);
            alert('This configuration file is corrupted.');
        }
    };

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md">
            <div className="bg-slate-900 border border-slate-700/50 rounded-2xl w-full max-w-xl shadow-2xl overflow-hidden flex flex-col max-h-[80vh] animate-in zoom-in-95 duration-200">
                {/* Header */}
                <div className="p-6 border-b border-slate-800 flex items-center justify-between bg-slate-800/20">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-violet-500/20 rounded-lg border border-violet-500/30">
                            <FolderOpen size={20} className="text-violet-400" />
                        </div>
                        <div>
                            <h3 className="text-xl font-bold text-slate-100">Load Saved Configuration</h3>
                            <p className="text-sm text-slate-500">Select a previously saved {mode}-asset setup</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-slate-800 rounded-lg transition-colors text-slate-400 hover:text-slate-100"
                    >
                        <X size={20} />
                    </button>
                </div>

                {/* Search */}
                <div className="p-4 border-b border-slate-800 bg-slate-900">
                    <div className="relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
                        <input
                            type="text"
                            placeholder="Search saved configurations..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full pl-10 pr-4 py-2 bg-slate-800/50 border border-slate-700/50 rounded-xl text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50 transition-all"
                        />
                    </div>
                </div>

                {/* List Container */}
                <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                    {loading ? (
                        <div className="flex flex-col items-center justify-center py-20 gap-4">
                            <Loader2 className="text-violet-500 animate-spin" size={32} />
                            <p className="text-slate-400 font-medium">Fetching saved setups...</p>
                        </div>
                    ) : filteredList.length > 0 ? (
                        <div className="space-y-2">
                            {filteredList.map((p) => {
                                let configInfo = '';
                                try {
                                    const cfg = JSON.parse(p.description || '{}');
                                    if (mode === 'single') {
                                        configInfo = `${cfg.symbol} • ${cfg.strategy}`;
                                    } else {
                                        configInfo = `${cfg.symbols?.length || 0} Assets • ${cfg.strategy}`;
                                    }
                                } catch (e) { }

                                return (
                                    <button
                                        key={p.id}
                                        onClick={() => handleSelect(p)}
                                        className="w-full flex items-center justify-between p-4 bg-slate-800/30 hover:bg-slate-800/60 border border-slate-700/30 rounded-xl transition-all group group-hover:border-violet-500/30"
                                    >
                                        <div className="flex items-center gap-4 text-left">
                                            <div className="p-2 bg-slate-700/50 rounded-lg group-hover:bg-violet-500/20 transition-colors">
                                                <FileText size={18} className="text-slate-400 group-hover:text-violet-400" />
                                            </div>
                                            <div>
                                                <h4 className="font-bold text-slate-200 group-hover:text-white transition-colors">{p.name}</h4>
                                                <div className="flex items-center gap-2 mt-1">
                                                    <span className="text-xs text-slate-500 bg-slate-800 px-1.5 py-0.5 rounded border border-slate-700">{configInfo}</span>
                                                    <span className="text-[10px] text-slate-600 flex items-center gap-1 font-mono uppercase">
                                                        <Calendar size={10} />
                                                        {new Date(p.created_at).toLocaleDateString()}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                        <ChevronRight size={18} className="text-slate-600 group-hover:text-violet-400 transition-all group-hover:translate-x-1" />
                                    </button>
                                );
                            })}
                        </div>
                    ) : (
                        <div className="flex flex-col items-center justify-center py-20 text-center gap-4 opacity-50 px-8">
                            <div className="p-4 bg-slate-800/50 rounded-full">
                                <FileText size={40} className="text-slate-500" />
                            </div>
                            <div>
                                <h4 className="text-slate-300 font-bold">No Configurations Found</h4>
                                <p className="text-slate-500 text-sm mt-1">
                                    {searchQuery ? "We couldn't find any saved setups matching your search." : `You haven't saved any ${mode}-asset backtest configurations yet.`}
                                </p>
                            </div>
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="p-4 bg-slate-800/40 border-t border-slate-800 text-center">
                    <p className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">
                        Tip: Save your best setups to load them later
                    </p>
                </div>
            </div>
        </div>
    );
};

export default LoadConfigModal;
