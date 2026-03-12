'use client';

import {useEffect, useState} from 'react';
import {Command} from 'cmdk';
import {useNavigationStore} from '@/store/useNavigationStore';
import {Activity, BarChart3, Cpu, DollarSign, Search, Settings, Shield, Target, TrendingUp, Zap} from 'lucide-react';
import {toast} from 'sonner';

export const CommandPalette = () => {
    const [open, setOpen] = useState(false);
    const navigateTo = useNavigationStore(state => state.navigateTo);

    useEffect(() => {
        const down = (e: KeyboardEvent) => {
            if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
                e.preventDefault();
                setOpen((open) => !open);
            }
        };
        document.addEventListener('keydown', down);
        // Make body unscrollable when open
        if (open) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = '';
        }
        return () => {
            document.removeEventListener('keydown', down);
            document.body.style.overflow = '';
        };
    }, [open]);

    const runCommand = (command: () => void) => {
        setOpen(false);
        command();
    };

    if (!open) return null;

    return (
        <div className="fixed inset-0 z-[100] bg-slate-950/80 backdrop-blur-sm p-4 pt-[10vh] sm:pt-[20vh] animate-fade-in">
            <div className="mx-auto max-w-2xl transform divide-y divide-slate-800/80 overflow-hidden rounded-2xl bg-slate-900 shadow-2xl ring-1 ring-slate-800/80 transition-all">
                <Command
                    className="flex h-full w-full flex-col bg-transparent"
                    loop
                >
                    <div className="relative border-b border-slate-800/80">
                        <Search className="pointer-events-none absolute left-4 top-4 h-5 w-5 text-slate-500" />
                        <Command.Input
                            autoFocus
                            className="h-14 w-full border-0 bg-transparent pl-[3.25rem] pr-4 text-slate-200 placeholder:text-slate-500 outline-none text-base"
                            placeholder="Type a command or search modules..."
                        />
                        <div className="absolute right-4 top-4">
                            <span className="text-[10px] font-bold text-slate-500 bg-slate-800/50 px-2 py-1 rounded-md border border-slate-700/50">ESC</span>
                        </div>
                    </div>

                    <Command.List className="max-h-[60vh] overflow-y-auto p-2 scroll-py-2 custom-scrollbar">
                        <Command.Empty className="p-6 text-center text-sm text-slate-400">
                            No matching commands found.
                        </Command.Empty>

                        <Command.Group heading="Monitor" className="px-2 py-2 text-xs font-semibold text-slate-500 tracking-wider">
                            <Command.Item
                                onSelect={() => runCommand(() => navigateTo('dashboard'))}
                                className="flex cursor-pointer select-none items-center rounded-lg px-3 py-3 text-sm text-slate-300 hover:bg-slate-800/80 hover:text-white data-[selected=true]:bg-violet-500/20 data-[selected=true]:text-violet-300 transition-colors mt-1"
                            >
                                <Activity className="mr-3 h-4 w-4 text-violet-400" /> Performance Hub
                            </Command.Item>
                            <Command.Item
                                onSelect={() => runCommand(() => navigateTo('live'))}
                                className="flex cursor-pointer select-none items-center rounded-lg px-3 py-3 text-sm text-slate-300 hover:bg-slate-800/80 hover:text-white data-[selected=true]:bg-violet-500/20 data-[selected=true]:text-violet-300 transition-colors mt-1"
                            >
                                <Zap className="mr-3 h-4 w-4 text-emerald-400" /> Live Execution
                            </Command.Item>
                        </Command.Group>

                        <div className="h-px bg-slate-800/50 my-1 mx-2" />

                        <Command.Group heading="Analyze" className="px-2 py-2 text-xs font-semibold text-slate-500 tracking-wider">
                            <Command.Item
                                onSelect={() => runCommand(() => navigateTo('advisor'))}
                                className="flex cursor-pointer select-none items-center rounded-lg px-3 py-3 text-sm text-slate-300 hover:bg-slate-800/80 hover:text-white data-[selected=true]:bg-violet-500/20 data-[selected=true]:text-violet-300 transition-colors mt-1"
                            >
                                <Zap className="mr-3 h-4 w-4 text-amber-400" /> AI Strategy Advisor
                            </Command.Item>
                            <Command.Item
                                onSelect={() => runCommand(() => navigateTo('regime'))}
                                className="flex cursor-pointer select-none items-center rounded-lg px-3 py-3 text-sm text-slate-300 hover:bg-slate-800/80 hover:text-white data-[selected=true]:bg-violet-500/20 data-[selected=true]:text-violet-300 transition-colors mt-1"
                            >
                                <TrendingUp className="mr-3 h-4 w-4 text-orange-400" /> Market Regime
                            </Command.Item>
                            <Command.Item
                                onSelect={() => runCommand(() => navigateTo('crash-prediction'))}
                                className="flex cursor-pointer select-none items-center rounded-lg px-3 py-3 text-sm text-slate-300 hover:bg-slate-800/80 hover:text-white data-[selected=true]:bg-violet-500/20 data-[selected=true]:text-violet-300 transition-colors mt-1"
                            >
                                <Shield className="mr-3 h-4 w-4 text-red-400" /> Crash Prediction
                            </Command.Item>
                        </Command.Group>

                        <div className="h-px bg-slate-800/50 my-1 mx-2" />

                        <Command.Group heading="Research & Build" className="px-2 py-2 text-xs font-semibold text-slate-500 tracking-wider">
                            <Command.Item
                                onSelect={() => runCommand(() => navigateTo('backtest'))}
                                className="flex cursor-pointer select-none items-center rounded-lg px-3 py-3 text-sm text-slate-300 hover:bg-slate-800/80 hover:text-white data-[selected=true]:bg-violet-500/20 data-[selected=true]:text-violet-300 transition-colors mt-1"
                            >
                                <BarChart3 className="mr-3 h-4 w-4 text-blue-400" /> Backtesting Lab
                            </Command.Item>
                            <Command.Item
                                onSelect={() => runCommand(() => navigateTo('options'))}
                                className="flex cursor-pointer select-none items-center rounded-lg px-3 py-3 text-sm text-slate-300 hover:bg-slate-800/80 hover:text-white data-[selected=true]:bg-violet-500/20 data-[selected=true]:text-violet-300 transition-colors mt-1"
                            >
                                <DollarSign className="mr-3 h-4 w-4 text-green-400" /> Options Desk
                            </Command.Item>
                            <Command.Item
                                onSelect={() => runCommand(() => navigateTo('ml-studio'))}
                                className="flex cursor-pointer select-none items-center rounded-lg px-3 py-3 text-sm text-slate-300 hover:bg-slate-800/80 hover:text-white data-[selected=true]:bg-violet-500/20 data-[selected=true]:text-violet-300 transition-colors mt-1"
                            >
                                <Cpu className="mr-3 h-4 w-4 text-fuchsia-400" /> ML Strategy Studio
                            </Command.Item>
                            <Command.Item
                                onSelect={() => runCommand(() => navigateTo('strategy-builder'))}
                                className="flex cursor-pointer select-none items-center rounded-lg px-3 py-3 text-sm text-slate-300 hover:bg-slate-800/80 hover:text-white data-[selected=true]:bg-violet-500/20 data-[selected=true]:text-violet-300 transition-colors mt-1"
                            >
                                <Target className="mr-3 h-4 w-4 text-teal-400" /> Strategy Builder
                            </Command.Item>
                        </Command.Group>

                        <div className="h-px bg-slate-800/50 my-1 mx-2" />

                        <Command.Group heading="System Actions" className="px-2 py-2 text-xs font-semibold text-slate-500 tracking-wider">
                            <Command.Item
                                onSelect={() => runCommand(() => {
                                    toast.success('System caches have been cleared', { duration: 3000 });
                                })}
                                className="flex cursor-pointer select-none items-center rounded-lg px-3 py-3 text-sm text-slate-300 hover:bg-slate-800/80 hover:text-white data-[selected=true]:bg-violet-500/20 data-[selected=true]:text-violet-300 transition-colors mt-1"
                            >
                                <Activity className="mr-3 h-4 w-4 text-slate-400" /> Clear Caches
                            </Command.Item>
                            <Command.Item
                                onSelect={() => runCommand(() => {
                                    toast.loading('Running system diagnostics...', { duration: 3000 });
                                })}
                                className="flex cursor-pointer select-none items-center rounded-lg px-3 py-3 text-sm text-slate-300 hover:bg-slate-800/80 hover:text-white data-[selected=true]:bg-violet-500/20 data-[selected=true]:text-violet-300 transition-colors mt-1"
                            >
                                <Settings className="mr-3 h-4 w-4 text-slate-400" /> Run Diagnostics
                            </Command.Item>
                        </Command.Group>
                    </Command.List>
                </Command>
            </div>

            {/* Overlay click to close */}
            <div className="fixed inset-0 -z-10" onClick={() => setOpen(false)} />
        </div>
    );
};
