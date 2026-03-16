import {BarChart3, Code2, GitBranch, Rocket} from "lucide-react";

export const STEPS = [
    {
        step: '01',
        title: 'Build',
        desc: 'Design strategies visually, with AI, or in the code editor.',
        icon: Code2,
        gradient: 'from-violet-600 to-indigo-600',
    },
    {
        step: '02',
        title: 'Backtest',
        desc: 'Simulate on historical data with institutional-grade analytics.',
        icon: BarChart3,
        gradient: 'from-blue-600 to-cyan-600',
    },
    {
        step: '03',
        title: 'Optimize',
        desc: 'Walk-forward analysis and Bayesian parameter optimization.',
        icon: GitBranch,
        gradient: 'from-emerald-600 to-teal-600',
    },
    {
        step: '04',
        title: 'Deploy',
        desc: 'Go live with automated execution and real-time monitoring.',
        icon: Rocket,
        gradient: 'from-fuchsia-600 to-pink-600',
    },
];
