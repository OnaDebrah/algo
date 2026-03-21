/* eslint-disable @typescript-eslint/no-explicit-any */
'use client'

import React, {useCallback, useEffect, useRef, useState} from "react";
import {Bell} from "lucide-react";
import {notifications as notificationsApi} from "@/utils/api";
import {AppNotification} from "@/types/all_types";
import NotificationPanel from "./NotificationPanel";

const NotificationBell = () => {
    const [unreadCount, setUnreadCount] = useState(0);
    const [notificationList, setNotificationList] = useState<AppNotification[]>([]);
    const [isOpen, setIsOpen] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);
    const wsRef = useRef<WebSocket | null>(null);

    const fetchUnreadCount = useCallback(async () => {
        try {
            const res = await notificationsApi.unreadCount();
            setUnreadCount(res?.count ?? 0);
        } catch {
            // silent fail
        }
    }, []);

    const fetchNotifications = useCallback(async () => {
        try {
            const res = await notificationsApi.list({limit: 20});
            if (res?.notifications) {
                setNotificationList(res.notifications);
                setUnreadCount(res.unread_count ?? 0);
            }
        } catch {
            // silent fail
        }
    }, []);

    // Poll unread count every 30s
    useEffect(() => {
        fetchUnreadCount();
        const interval = setInterval(fetchUnreadCount, 30000);
        return () => clearInterval(interval);
    }, [fetchUnreadCount]);

    // Fetch full list when panel opens
    useEffect(() => {
        if (isOpen) fetchNotifications();
    }, [isOpen, fetchNotifications]);

    // WebSocket for real-time notifications
    useEffect(() => {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        const wsHost = apiUrl.replace(/^https?:\/\//, '');
        const wsUrl = `${protocol}//${wsHost}/websocket/ws/notifications`;

        try {
            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'new_notification' && data.notification) {
                        setUnreadCount(prev => prev + 1);
                        setNotificationList(prev => [data.notification, ...prev].slice(0, 20));
                    }
                } catch {
                    // ignore parse errors
                }
            };

            ws.onclose = () => {
                // Reconnect after 5s
                setTimeout(() => {
                    if (wsRef.current === ws) {
                        wsRef.current = null;
                    }
                }, 5000);
            };
        } catch {
            // WS not available
        }

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
        };
    }, []);

    // Close panel when clicking outside
    useEffect(() => {
        const handleClickOutside = (e: MouseEvent) => {
            if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
                setIsOpen(false);
            }
        };
        if (isOpen) document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [isOpen]);

    const handleMarkRead = async (id: number) => {
        try {
            await notificationsApi.markRead(id);
            setNotificationList(prev => prev.map(n => n.id === id ? {...n, is_read: true} : n));
            setUnreadCount(prev => Math.max(0, prev - 1));
        } catch {
            // silent
        }
    };

    const handleMarkAllRead = async () => {
        try {
            await notificationsApi.markAllRead();
            setNotificationList(prev => prev.map(n => ({...n, is_read: true})));
            setUnreadCount(0);
        } catch {
            // silent
        }
    };

    const handleDelete = async (id: number) => {
        try {
            const wasUnread = notificationList.find(n => n.id === id && !n.is_read);
            await notificationsApi.delete(id);
            setNotificationList(prev => prev.filter(n => n.id !== id));
            if (wasUnread) setUnreadCount(prev => Math.max(0, prev - 1));
        } catch {
            // silent
        }
    };

    return (
        <div ref={containerRef} className="relative">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="relative text-slate-400 hover:text-slate-200 transition-colors p-1"
            >
                <Bell size={16}/>
                {unreadCount > 0 && (
                    <span className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 text-white text-[10px] font-bold rounded-full flex items-center justify-center leading-none">
                        {unreadCount > 9 ? '9+' : unreadCount}
                    </span>
                )}
            </button>

            {isOpen && (
                <NotificationPanel
                    notifications={notificationList}
                    onMarkRead={handleMarkRead}
                    onMarkAllRead={handleMarkAllRead}
                    onDelete={handleDelete}
                    onClose={() => setIsOpen(false)}
                />
            )}
        </div>
    );
};

export default NotificationBell;
