import {Bell, Globe, Key, Shield, TrendingDown, Zap} from "lucide-react";

export const tabs = [
    { id: 'backtest', label: 'Backtest Settings', icon: TrendingDown },
    { id: 'live', label: 'Live Trading', icon: Zap },
    { id: 'alerts', label: 'Alerts', icon: Bell },
    { id: 'security', label: 'Security', icon: Shield },
    { id: 'api-keys', label: 'API Keys', icon: Key },
    { id: 'general', label: 'General', icon: Globe }
];
