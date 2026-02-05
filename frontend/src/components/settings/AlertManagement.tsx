/**
 * Alert Management UI
 * Configure email and SMS notifications
 */

'use client'

import React, {useEffect, useState} from 'react';
import {AlertCircle, AlertTriangle, Bell, CheckCircle, Info, Mail, Smartphone} from 'lucide-react';
import {Alert, AlertPreferences} from "@/types/all_types";
import {alerts} from "@/utils/api";

export default function AlertManagement() {
    const [preferences, setPreferences] = useState<AlertPreferences>({
        email: '',
        phone: '',
        email_enabled: true,
        sms_enabled: false,
        min_level_email: 'info',
        min_level_sms: 'error'
    });

    const [recentAlerts, setRecentAlerts] = useState<Alert[]>([]);
    const [testingEmail, setTestingEmail] = useState(false);
    const [testingSMS, setTestingSMS] = useState(false);
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        loadPreferences();
        loadRecentAlerts();
    }, []);

    const loadPreferences = async () => {
        try {
            const response = await alerts.getPreferences();
            if (response) {
                setPreferences(response);
            }
        } catch (error) {
            console.error('Error loading preferences:', error);
        }
    };

    const loadRecentAlerts = async () => {
        try {
            const response = await alerts.getHistory();
            if (response) {
                setRecentAlerts(response)
            }
        } catch (error) {
            console.error('Error loading alerts:', error);
        }
    };

    const savePreferences = async () => {
        setSaving(true);
        try {
            const response = await alerts.updatePreferences(preferences);

            if (response) {
                alert('Preferences saved successfully');
            }
        } catch (error) {
            console.error('Error saving preferences:', error);
        } finally {
            setSaving(false);
        }
    };

    const sendTestAlert = async (channel: 'email' | 'sms') => {
        if (channel === 'email') setTestingEmail(true);
        else setTestingSMS(true);

        try {
            const response = await alerts.sendTest(channel);

            if (response) {
                alert(`Test ${channel} sent!`);
            }
        } catch (error) {
            console.error(`Error sending test ${channel}:`, error);
        } finally {
            if (channel === 'email') setTestingEmail(false);
            else setTestingSMS(false);
        }
    };

    const levelColors = {
        info: 'text-blue-400 bg-blue-500/10',
        warning: 'text-amber-400 bg-amber-500/10',
        error: 'text-red-400 bg-red-500/10',
        critical: 'text-purple-400 bg-purple-500/10'
    };

    const levelIcons = {
        info: Info,
        warning: AlertTriangle,
        error: AlertCircle,
        critical: AlertCircle
    };

    return (
        <div className="max-w-4xl mx-auto space-y-6">

            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
                        <Bell size={28} className="text-violet-500"/>
                        Alert Management
                    </h1>
                    <p className="text-slate-400 mt-1">Configure email and SMS notifications for your strategies</p>
                </div>
            </div>

            {/* Email Configuration */}
            <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-800">
                <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-3">
                        <Mail size={24} className="text-blue-500"/>
                        <div>
                            <h3 className="text-lg font-bold text-slate-100">Email Notifications</h3>
                            <p className="text-sm text-slate-400">Receive alerts via email</p>
                        </div>
                    </div>

                    <label className="relative inline-flex items-center cursor-pointer">
                        <input
                            type="checkbox"
                            checked={preferences.email_enabled}
                            onChange={(e) => setPreferences({...preferences, email_enabled: e.target.checked})}
                            className="sr-only peer"
                        />
                        <div
                            className="w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-600"></div>
                    </label>
                </div>

                {preferences.email_enabled && (
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-bold text-slate-300 mb-2">Email Address</label>
                            <input
                                type="email"
                                value={preferences.email}
                                onChange={(e) => setPreferences({...preferences, email: e.target.value})}
                                className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:border-violet-500 outline-none"
                                placeholder="your.email@example.com"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-bold text-slate-300 mb-2">Minimum Alert Level</label>
                            <select
                                value={preferences.min_level_email}
                                onChange={(e) => setPreferences({
                                    ...preferences,
                                    min_level_email: e.target.value as any
                                })}
                                className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:border-violet-500 outline-none"
                            >
                                <option value="info">Info (All alerts)</option>
                                <option value="warning">Warning and above</option>
                                <option value="error">Error and above</option>
                                <option value="critical">Critical only</option>
                            </select>
                            <p className="text-xs text-slate-500 mt-1">You'll receive email for this level and above</p>
                        </div>

                        <button
                            onClick={() => sendTestAlert('email')}
                            disabled={testingEmail || !preferences.email}
                            className="px-4 py-2 bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/50 rounded-lg text-blue-400 font-semibold disabled:opacity-50 flex items-center gap-2"
                        >
                            {testingEmail ? (
                                <>
                                    <div
                                        className="w-4 h-4 border-2 border-blue-400/30 border-t-blue-400 rounded-full animate-spin"/>
                                    Sending...
                                </>
                            ) : (
                                <>
                                    <Mail size={16}/>
                                    Send Test Email
                                </>
                            )}
                        </button>
                    </div>
                )}
            </div>

            {/* SMS Configuration */}
            <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-800">
                <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-3">
                        <Smartphone size={24} className="text-emerald-500"/>
                        <div>
                            <h3 className="text-lg font-bold text-slate-100">SMS Notifications</h3>
                            <p className="text-sm text-slate-400">Receive critical alerts via text message</p>
                        </div>
                    </div>

                    <label className="relative inline-flex items-center cursor-pointer">
                        <input
                            type="checkbox"
                            checked={preferences.sms_enabled}
                            onChange={(e) => setPreferences({...preferences, sms_enabled: e.target.checked})}
                            className="sr-only peer"
                        />
                        <div
                            className="w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-600"></div>
                    </label>
                </div>

                {preferences.sms_enabled && (
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-bold text-slate-300 mb-2">Phone Number</label>
                            <input
                                type="tel"
                                value={preferences.phone}
                                onChange={(e) => setPreferences({...preferences, phone: e.target.value})}
                                className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:border-violet-500 outline-none"
                                placeholder="+1234567890"
                            />
                            <p className="text-xs text-slate-500 mt-1">Include country code (e.g., +1 for US)</p>
                        </div>

                        <div>
                            <label className="block text-sm font-bold text-slate-300 mb-2">Minimum Alert Level</label>
                            <select
                                value={preferences.min_level_sms}
                                onChange={(e) => setPreferences({...preferences, min_level_sms: e.target.value as any})}
                                className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:border-violet-500 outline-none"
                            >
                                <option value="warning">Warning and above</option>
                                <option value="error">Error and above</option>
                                <option value="critical">Critical only</option>
                            </select>
                            <p className="text-xs text-slate-500 mt-1">Recommended: Error or Critical to avoid SMS
                                spam</p>
                        </div>

                        <button
                            onClick={() => sendTestAlert('sms')}
                            disabled={testingSMS || !preferences.phone}
                            className="px-4 py-2 bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/50 rounded-lg text-emerald-400 font-semibold disabled:opacity-50 flex items-center gap-2"
                        >
                            {testingSMS ? (
                                <>
                                    <div
                                        className="w-4 h-4 border-2 border-emerald-400/30 border-t-emerald-400 rounded-full animate-spin"/>
                                    Sending...
                                </>
                            ) : (
                                <>
                                    <Smartphone size={16}/>
                                    Send Test SMS
                                </>
                            )}
                        </button>
                    </div>
                )}
            </div>

            {/* Recent Alerts */}
            <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-800">
                <h3 className="text-lg font-bold text-slate-100 mb-6">Recent Alerts</h3>

                <div className="space-y-3">
                    {recentAlerts.length > 0 ? (
                        recentAlerts.map(alert => {
                            const Icon = levelIcons[alert.level];

                            return (
                                <div
                                    key={alert.id}
                                    className={`p-4 rounded-lg border ${levelColors[alert.level]} border-current`}
                                >
                                    <div className="flex items-start gap-3">
                                        <Icon size={20} className="flex-shrink-0 mt-0.5"/>
                                        <div className="flex-1">
                                            <div className="font-semibold text-sm">{alert.title}</div>
                                            <div className="text-xs opacity-80 mt-1">{alert.message}</div>
                                            <div className="flex items-center gap-3 mt-2 text-xs opacity-60">
                                                <span>{new Date(alert.created_at).toLocaleString()}</span>
                                                {alert.channels_sent.length > 0 && (
                                                    <span>• Sent via: {alert.channels_sent.join(', ')}</span>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            );
                        })
                    ) : (
                        <div className="text-center py-8 text-slate-500">
                            No alerts yet
                        </div>
                    )}
                </div>
            </div>

            {/* Save Button */}
            <div className="flex justify-end">
                <button
                    onClick={savePreferences}
                    disabled={saving}
                    className="px-6 py-3 bg-violet-600 hover:bg-violet-500 rounded-lg text-white font-semibold disabled:opacity-50 flex items-center gap-2"
                >
                    {saving ? (
                        <>
                            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"/>
                            Saving...
                        </>
                    ) : (
                        <>
                            <CheckCircle size={16}/>
                            Save Preferences
                        </>
                    )}
                </button>
            </div>

        </div>
    );
}
