'use client'
import React, {useEffect, useState} from 'react';
import {Activity, ArrowRight, BookOpen, Filter, Grid, List, Search, Target, Zap} from 'lucide-react';
import {strategy as strategyApi} from "@/utils/api";
import {StrategyInfo} from "@/types/all_types";

const StrategyLibrary = () => {
    const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
    const [loading, setLoading] = useState(true);
    const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedCategory, setSelectedCategory] = useState('All');

    useEffect(() => {
        const fetchStrategies = async () => {
            try {
                const response: StrategyInfo[] = await strategyApi.list();
                if (response) {
                    setStrategies(response);
                }
            } catch (error) {
                console.error("Failed to fetch strategies", error);
            } finally {
                setLoading(false);
            }
        };
        fetchStrategies();
    }, []);

    const categories = ['All', ...Array.from(new Set(strategies.map(s => s.category)))];

    const filteredStrategies = strategies.filter(s => {
        const matchesSearch = s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            s.description.toLowerCase().includes(searchQuery.toLowerCase());
        const matchesCategory = selectedCategory === 'All' || s.category === selectedCategory;
        return matchesSearch && matchesCategory;
    });

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-[400px]">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-dashed border-violet-500"></div>
            </div>
        );
    }

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            {/* Header */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                    <h1 className="text-3xl font-bold text-slate-100 flex items-center gap-3">
                        <BookOpen className="text-violet-400" size={32} />
                        Strategy Library
                    </h1>
                    <p className="text-slate-400 mt-2 max-w-2xl">
                        Explore our collection of algorithmic trading strategies. Each strategy is rigorously tested and documented.
                    </p>
                </div>

                <div className="flex items-center gap-2 bg-slate-900/50 p-1 rounded-xl border border-slate-800">
                    <div className="px-4 py-2 rounded-lg bg-slate-800 text-slate-300 text-sm font-bold">
                        {strategies.length} Strategies
                    </div>
                </div>
            </div>

            {/* Controls */}
            <div className="bg-slate-900/40 border border-slate-800/50 rounded-2xl p-4 flex flex-col md:flex-row gap-4 justify-between">
                <div className="relative flex-1 max-w-md">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
                    <input
                        type="text"
                        placeholder="Search strategies..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-xl py-2.5 pl-10 pr-4 text-slate-300 focus:border-violet-500 outline-none transition-all"
                    />
                </div>

                <div className="flex gap-3">
                    <div className="relative">
                        <Filter className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={16} />
                        <select
                            value={selectedCategory}
                            onChange={(e) => setSelectedCategory(e.target.value)}
                            className="bg-slate-950 border border-slate-800 rounded-xl py-2.5 pl-10 pr-8 text-slate-300 focus:border-violet-500 outline-none appearance-none cursor-pointer"
                        >
                            {categories.map(cat => (
                                <option key={cat} value={cat}>{cat}</option>
                            ))}
                        </select>
                    </div>

                    <div className="flex bg-slate-950 border border-slate-800 rounded-xl overflow-hidden p-1 gap-1">
                        <button
                            onClick={() => setViewMode('grid')}
                            className={`p-2 rounded-lg transition-all ${viewMode === 'grid' ? 'bg-slate-800 text-violet-400' : 'text-slate-500 hover:text-slate-300'}`}
                        >
                            <Grid size={18} />
                        </button>
                        <button
                            onClick={() => setViewMode('list')}
                            className={`p-2 rounded-lg transition-all ${viewMode === 'list' ? 'bg-slate-800 text-violet-400' : 'text-slate-500 hover:text-slate-300'}`}
                        >
                            <List size={18} />
                        </button>
                    </div>
                </div>
            </div>

            {/* Content */}
            {viewMode === 'grid' ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {filteredStrategies.map(strategy => (
                        <div key={strategy.key} className="group bg-gradient-to-br from-slate-900 via-slate-900 to-slate-800 border border-slate-800 rounded-2xl p-6 hover:border-violet-500/30 hover:shadow-2xl hover:shadow-violet-500/10 transition-all duration-300 flex flex-col h-full">
                            <div className="flex justify-between items-start mb-4">
                                <div className="p-3 bg-violet-500/10 rounded-xl border border-violet-500/20 group-hover:scale-110 transition-transform">
                                    <Activity className="text-violet-400" size={24} />
                                </div>
                                <span className="px-3 py-1 rounded-full text-xs font-bold bg-slate-800 border border-slate-700 text-slate-400 uppercase tracking-wider">
                                    {strategy.category}
                                </span>
                            </div>

                            <h3 className="text-xl font-bold text-slate-100 mb-2 group-hover:text-violet-400 transition-colors">
                                {strategy.name}
                            </h3>
                            <p className="text-slate-400 text-sm mb-6 flex-1 line-clamp-3">
                                {strategy.description}
                            </p>

                            <div className="space-y-4">
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="bg-slate-950/50 p-2.5 rounded-xl border border-slate-800">
                                        <p className="text-[10px] text-slate-500 font-bold uppercase mb-1">Complexity</p>
                                        <div className="flex items-center gap-1.5">
                                            <Zap size={12} className={
                                                strategy.complexity === 'Advanced' ? 'text-red-400' :
                                                    strategy.complexity === 'Intermediate' ? 'text-amber-400' : 'text-emerald-400'
                                            } />
                                            <span className="text-xs font-bold text-slate-300">{strategy.complexity || 'Intermediate'}</span>
                                        </div>
                                    </div>
                                    <div className="bg-slate-950/50 p-2.5 rounded-xl border border-slate-800">
                                        <p className="text-[10px] text-slate-500 font-bold uppercase mb-1">Horizon</p>
                                        <div className="flex items-center gap-1.5">
                                            <Target size={12} className="text-blue-400" />
                                            <span className="text-xs font-bold text-slate-300">{strategy.time_horizon || 'Medium'}</span>
                                        </div>
                                    </div>
                                </div>

                                {strategy.best_for && (
                                    <div className="flex flex-wrap gap-2">
                                        {strategy.best_for.slice(0, 3).map(tag => (
                                            <span key={tag} className="text-[10px] px-2 py-1 rounded bg-slate-800 text-slate-400 border border-slate-700">
                                                {tag}
                                            </span>
                                        ))}
                                    </div>
                                )}

                                <button className="w-full mt-2 py-3 bg-slate-800 hover:bg-violet-600 text-slate-300 hover:text-white rounded-xl text-sm font-bold transition-all flex items-center justify-center gap-2 group-hover:bg-violet-600 group-hover:text-white">
                                    View Details <ArrowRight size={16} />
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            ) : (
                <div className="space-y-4">
                    {filteredStrategies.map(strategy => (
                        <div key={strategy.key} className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 flex items-center justify-between hover:border-violet-500/30 transition-all">
                            <div className="flex items-center gap-6">
                                <div className="p-3 bg-slate-800 rounded-xl">
                                    <Activity className="text-violet-400" size={24} />
                                </div>
                                <div>
                                    <h3 className="text-lg font-bold text-slate-200">{strategy.name}</h3>
                                    <p className="text-sm text-slate-400 max-w-2xl truncate">{strategy.description}</p>
                                </div>
                            </div>
                            <div className="flex items-center gap-6">
                                <div className="flex items-center gap-4">
                                    <span className="text-xs font-bold bg-slate-800 px-3 py-1 rounded-lg text-slate-400 border border-slate-700">
                                        {strategy.category}
                                    </span>
                                    <div className="text-right">
                                        <p className="text-[10px] text-slate-500 uppercase font-bold">Complexity</p>
                                        <p className="text-xs font-bold text-slate-300">{strategy.complexity}</p>
                                    </div>
                                </div>
                                <button className="p-2 hover:bg-violet-500/20 rounded-lg text-slate-400 hover:text-violet-400 transition-colors">
                                    <ArrowRight size={20} />
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default StrategyLibrary;
