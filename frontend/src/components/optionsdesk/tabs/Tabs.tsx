import {BrainCircuit, GitCompare, Layers, Play, Settings, Shield, Zap} from "lucide-react";
import React from "react";

interface TabsProps {
    activeTab: "chain" | "builder" | "compare" | "backtest" | "volatility" | "risk" | "ml",
    setActiveTab: React.Dispatch<React.SetStateAction<"ml" | "chain" | "builder" | "compare" | "backtest" | "volatility" | "risk">>
}

const Tabs: React.FC<TabsProps> = ({
                                       activeTab,
                                       setActiveTab
                                   }: TabsProps) => {

    return (
        <div className="flex gap-2 p-1 bg-slate-900/50 rounded-xl border border-slate-800/50">
            {[
                {id: 'ml', label: 'AI Forecast', icon: BrainCircuit},
                {id: 'chain', label: 'Options Chain', icon: Layers},
                {id: 'builder', label: 'Custom Builder', icon: Settings},
                {id: 'compare', label: 'Compare', icon: GitCompare},
                {id: 'backtest', label: 'Backtest', icon: Play},
                {id: 'volatility', label: 'Volatility', icon: Zap},
                {id: 'risk', label: 'Risk Analysis', icon: Shield}
            ].map((tab) => (
                <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={`px-4 py-2.5 rounded-lg text-xs font-bold uppercase flex items-center justify-center gap-2 transition-all ${
                        activeTab === tab.id
                            ? 'bg-gradient-to-r from-amber-500 to-orange-500 text-white shadow-lg'
                            : 'text-slate-500 hover:text-slate-300'
                    }`}
                >
                    <tab.icon size={14}/>
                    {tab.label}
                </button>
            ))}
        </div>
    )
};

export default Tabs;