'use client'

import React, { useEffect, useState } from 'react';
import { Share2, TrendingUp, Trophy, Zap, MessageSquare, Clock, ArrowUpRight, Rocket, Award, ShieldCheck } from 'lucide-react';
import {api, social} from '@/utils/api';
import { ActivityResponse } from '@/types/social';

const SocialFeed = () => {
    const [activities, setActivities] = useState<ActivityResponse[]>([]);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        loadFeed();
        const interval = setInterval(loadFeed, 30000); // Refresh every 30s
        return () => clearInterval(interval);
    }, []);

    const loadFeed = async () => {
        try {
            const data = await social.getFeed(10);
            setActivities(data);
        } catch (error) {
            console.error("Failed to load social feed:", error);
        } finally {
            setIsLoading(false);
        }
    };

    const getActivityIcon = (type: string) => {
        switch (type) {
            case 'STRATEGY_PUBLISHED': return <Rocket className="text-violet-400" size={16} />;
            case 'BIG_WIN': return <TrendingUp className="text-emerald-400" size={16} />;
            case 'MILESTONE': return <Trophy className="text-amber-400" size={16} />;
            case 'VERIFIED': return <ShieldCheck className="text-blue-400" size={16} />;
            default: return <Zap className="text-slate-400" size={16} />;
        }
    };

    const formatTime = (dateStr: string) => {
        const date = new Date(dateStr);
        const now = new Date();
        const diffMs = now.getTime() - date.getTime();
        const diffMins = Math.floor(diffMs / 60000);

        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
        return date.toLocaleDateString();
    };

    return (
        <div className="bg-slate-900/50 border border-slate-800 rounded-3xl overflow-hidden flex flex-col h-full shadow-2xl backdrop-blur-md">
            {/* Header */}
            <div className="px-6 py-5 border-b border-slate-800 flex items-center justify-between bg-gradient-to-r from-slate-900 to-slate-800">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-violet-500/10 rounded-xl">
                        <Share2 className="text-violet-400" size={18} />
                    </div>
                    <div>
                        <h3 className="text-sm font-black text-slate-100 uppercase tracking-widest">Global Activity</h3>
                        <div className="flex items-center gap-1.5 mt-0.5">
                            <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
                            <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Live Feed</span>
                        </div>
                    </div>
                </div>
                <button className="p-2 hover:bg-slate-700/50 rounded-lg text-slate-500 transition-all">
                    <ArrowUpRight size={16} />
                </button>
            </div>

            {/* List */}
            <div className="flex-1 overflow-y-auto custom-scrollbar p-1">
                {isLoading ? (
                    <div className="flex flex-col items-center justify-center h-full py-12 space-y-4">
                        <div className="w-8 h-8 border-2 border-violet-500/20 border-t-violet-500 rounded-full animate-spin" />
                        <p className="text-[10px] text-slate-500 font-black uppercase tracking-widest">Syncing Feed...</p>
                    </div>
                ) : activities.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full py-12 px-6 text-center">
                        <MessageSquare className="text-slate-700 mb-3" size={32} />
                        <p className="text-xs text-slate-500 font-medium">No activity yet. Be the first to publish a strategy!</p>
                    </div>
                ) : (
                    <div className="space-y-1">
                        {activities.map((activity) => (
                            <div
                                key={activity.id}
                                className="group p-4 hover:bg-slate-800/40 transition-all cursor-pointer rounded-2xl mx-1"
                            >
                                <div className="flex items-start gap-4">
                                    <div className={`mt-1 p-2 rounded-xl bg-slate-950/50 border border-slate-800 group-hover:border-slate-700 transition-colors`}>
                                        {getActivityIcon(activity.activity_type)}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center justify-between gap-2 mb-1">
                                            <span className="text-xs font-black text-slate-200 truncate uppercase tracking-tight">
                                                @{activity.username}
                                            </span>
                                            <span className="text-[10px] font-bold text-slate-600 shrink-0 uppercase tracking-tighter">
                                                {formatTime(activity.created_at)}
                                            </span>
                                        </div>
                                        <p className="text-xs text-slate-400 leading-relaxed font-medium">
                                            {activity.content}
                                        </p>

                                        {activity.metadata_json && activity.metadata_json.profit_pct && (
                                            <div className="mt-2 flex items-center gap-2">
                                                <span className="text-[10px] font-black px-2 py-0.5 bg-emerald-500/10 text-emerald-400 rounded uppercase">
                                                    +{activity.metadata_json.profit_pct}% Profit
                                                </span>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Footer */}
            <div className="p-4 bg-slate-950/30 border-t border-slate-800 mt-auto">
                <button
                    onClick={loadFeed}
                    className="w-full py-3 bg-slate-800/50 hover:bg-slate-800 text-slate-400 hover:text-slate-200 text-[10px] font-black uppercase tracking-widest rounded-2xl transition-all flex items-center justify-center gap-2 border border-transparent hover:border-slate-700"
                >
                    <Clock size={12} />
                    View History
                </button>
            </div>
        </div>
    );
};

export default SocialFeed;
