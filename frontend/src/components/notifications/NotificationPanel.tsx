/* eslint-disable @typescript-eslint/no-explicit-any */
'use client'

import React from "react";
import {
    AlertCircle,
    Check,
    CheckCheck,
    DollarSign,
    ShoppingCart,
    TrendingUp,
    Trash2,
    X,
} from "lucide-react";
import {AppNotification} from "@/types/all_types";

interface NotificationPanelProps {
    notifications: AppNotification[];
    onMarkRead: (id: number) => void;
    onMarkAllRead: () => void;
    onDelete: (id: number) => void;
    onClose: () => void;
}

const typeIcons: Record<string, any> = {
    strategy: TrendingUp,
    marketplace: ShoppingCart,
    price_alert: DollarSign,
    system: AlertCircle,
};

const typeColors: Record<string, string> = {
    strategy: 'text-violet-400 bg-violet-500/20',
    marketplace: 'text-emerald-400 bg-emerald-500/20',
    price_alert: 'text-amber-400 bg-amber-500/20',
    system: 'text-blue-400 bg-blue-500/20',
};

function timeAgo(dateStr: string): string {
    const now = Date.now();
    const then = new Date(dateStr).getTime();
    const diffSec = Math.floor((now - then) / 1000);
    if (diffSec < 60) return 'just now';
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHr = Math.floor(diffMin / 60);
    if (diffHr < 24) return `${diffHr}h ago`;
    const diffDay = Math.floor(diffHr / 24);
    return `${diffDay}d ago`;
}

const NotificationPanel = ({notifications, onMarkRead, onMarkAllRead, onDelete, onClose}: NotificationPanelProps) => {
    const unreadCount = notifications.filter(n => !n.is_read).length;

    return (
        <div className="absolute right-0 top-full mt-2 w-96 bg-slate-900 border border-slate-800/50 rounded-xl shadow-2xl shadow-black/40 z-50 overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-slate-800/50">
                <div className="flex items-center gap-2">
                    <h3 className="text-sm font-bold text-slate-200">Notifications</h3>
                    {unreadCount > 0 && (
                        <span className="text-xs px-2 py-0.5 bg-violet-500/20 text-violet-400 rounded-full font-medium">
                            {unreadCount} new
                        </span>
                    )}
                </div>
                <div className="flex items-center gap-2">
                    {unreadCount > 0 && (
                        <button
                            onClick={onMarkAllRead}
                            className="text-xs text-slate-400 hover:text-slate-200 transition-colors flex items-center gap-1"
                        >
                            <CheckCheck size={14}/>
                            Mark all read
                        </button>
                    )}
                    <button onClick={onClose} className="text-slate-500 hover:text-slate-300 transition-colors">
                        <X size={16}/>
                    </button>
                </div>
            </div>

            {/* Notification list */}
            <div className="max-h-96 overflow-y-auto">
                {notifications.length === 0 ? (
                    <div className="p-8 text-center">
                        <AlertCircle size={32} className="text-slate-700 mx-auto mb-3"/>
                        <p className="text-sm text-slate-500">No notifications yet</p>
                    </div>
                ) : (
                    notifications.map((notif) => {
                        const Icon = typeIcons[notif.type] || AlertCircle;
                        const colorClass = typeColors[notif.type] || 'text-slate-400 bg-slate-500/20';
                        return (
                            <div
                                key={notif.id}
                                className={`flex items-start gap-3 p-4 border-b border-slate-800/30 hover:bg-slate-800/30 transition-colors cursor-pointer group ${
                                    !notif.is_read ? 'bg-slate-800/20' : ''
                                }`}
                                onClick={() => {
                                    if (!notif.is_read) onMarkRead(notif.id);
                                }}
                            >
                                <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${colorClass}`}>
                                    <Icon size={16}/>
                                </div>
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-start justify-between gap-2">
                                        <p className={`text-sm font-medium ${notif.is_read ? 'text-slate-400' : 'text-slate-200'}`}>
                                            {notif.title}
                                        </p>
                                        {!notif.is_read && (
                                            <div className="w-2 h-2 rounded-full bg-violet-400 shrink-0 mt-1.5"/>
                                        )}
                                    </div>
                                    <p className="text-xs text-slate-500 mt-0.5 line-clamp-2">{notif.message}</p>
                                    <p className="text-xs text-slate-600 mt-1">{timeAgo(notif.created_at)}</p>
                                </div>
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        onDelete(notif.id);
                                    }}
                                    className="text-slate-600 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100 shrink-0"
                                >
                                    <Trash2 size={14}/>
                                </button>
                            </div>
                        );
                    })
                )}
            </div>
        </div>
    );
};

export default NotificationPanel;
