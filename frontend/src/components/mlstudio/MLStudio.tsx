/* eslint-disable @typescript-eslint/no-explicit-any */

'use client'
import React, {useEffect, useMemo, useState} from 'react';
import {
    Activity,
    BarChart3,
    BrainCircuit,
    CheckCircle,
    CheckCircle2,
    Clock,
    Cloud,
    Cpu,
    Cpu as CpuIcon,
    Filter,
    Gamepad2,
    GitBranch,
    Info,
    Layers,
    LineChart,
    Loader2,
    Maximize2,
    MessageSquare,
    Minus,
    PieChart,
    Play,
    Plus,
    Settings2,
    Shield,
    Target,
    Trash2,
    TreePine,
    TrendingUp,
    Upload,
    Zap
} from "lucide-react";
import {
    Area,
    Bar,
    BarChart,
    CartesianGrid,
    Cell,
    ComposedChart,
    Legend,
    Line,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from 'recharts';
import {mlstudio} from '@/utils/api';
import {MLModel, TrainingConfig, TrainingEpoch} from '@/types/all_types';
import {useBacktestStore} from '@/store/useBacktestStore';
import {useNavigationStore} from '@/store/useNavigationStore';
import ModelRegistry from "@/components/mlstudio/ModelRegistry";
import StrategyBuilder from "@/components/backtest/StrategyBuilder";

const MLStudio = () => {
    const navigateTo = useNavigationStore(state => state.navigateTo);
    const setVisualStrategy = useBacktestStore(state => state.setVisualStrategy);

    const [isTraining, setIsTraining] = useState(false);
    const [trainingProgress, setTrainingProgress] = useState(0);
    const [showResults, setShowResults] = useState(false);
    const [activeModelId, setActiveModelId] = useState<string | null>(null);
    const [models, setModels] = useState<MLModel[]>([]);
    const [selectedModel, setSelectedModel] = useState<MLModel | null>(null);
    const [expandedSection, setExpandedSection] = useState<'config' | 'features' | 'results'>('config');
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [isDeploying, setIsDeploying] = useState(false);
    const [isLoadingModels, setIsLoadingModels] = useState(true);
    const [loadError, setLoadError] = useState<string | null>(null);

    const [activeTab, setActiveTab] = useState<'train' | 'builder' | 'registry' | 'deployed'>('train');
    const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
    const [filterStatus, setFilterStatus] = useState<'all' | 'trained' | 'deployed'>('all');
    const [sortBy, setSortBy] = useState<'date' | 'accuracy' | 'training_time'>('date');
    const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
    const [showInactive, setShowInactive] = useState(false);

    // Training configuration
    const [config, setConfig] = useState<TrainingConfig>({
        symbol: 'AAPL',
        model_type: 'Random Forest',
        training_period: '1Y',
        test_size: 20,
        epochs: 50,
        batch_size: 32,
        learning_rate: 0.001,
        threshold: 0.6,
        use_feature_engineering: true,
        use_cross_validation: true,
        episodes: 200,
        gamma: 0.99,
    });

    // Load models on mount
    useEffect(() => {
        fetchModels();
    }, []);

    const fetchModels = async () => {
        setIsLoadingModels(true);
        setLoadError(null);
        try {
            const response = await mlstudio.getModels();
            if (response) {
                const mappedModels = response.map((m: MLModel) => ({
                    ...m,
                    testAccuracy: m.test_accuracy,
                    overfitScore: m.overfit_score,
                    trainingTime: m.training_time,
                    featureImportance: m.feature_importance,
                    isActive: m.is_active !== false,
                    lastUsed: m.last_used ? new Date(m.last_used) : null,
                    usageCount: m.usage_count || 0,
                    performanceHistory: m.performance_history || []
                }));
                setModels(mappedModels);
                if (mappedModels.length > 0 && !selectedModel) {
                    setSelectedModel(mappedModels[0]);
                    setActiveModelId(mappedModels[0].id);
                }
            }
        } catch (err: any) {
            console.error("Failed to fetch models:", err);
            setLoadError(err?.message || 'Failed to load models. Please try again.');
        } finally {
            setIsLoadingModels(false);
        }
    };

    // Training metrics over time — populated from backend after training
    const [trainingMetrics, setTrainingMetrics] = useState<TrainingEpoch[]>([]);

    // Sync training metrics when selected model changes (e.g. clicking a model in the registry)
    useEffect(() => {
        if (selectedModel?.training_history && selectedModel.training_history.length > 0) {
            setTrainingMetrics(selectedModel.training_history);
        } else if (selectedModel) {
            // Model exists but has no training history (older models) — clear the chart
            setTrainingMetrics([]);
        }
    }, [selectedModel?.id]);

    const filteredAndSortedModels = useMemo(() => {
        let filtered = models;

        if (filterStatus === 'trained') {
            filtered = filtered.filter(m => m.status === 'trained');
        } else if (filterStatus === 'deployed') {
            filtered = filtered.filter(m => m.status === 'deployed');
        }

        // Filter inactive models - check both camelCase (mapped) and snake_case (raw) fields
        if (!showInactive) {
            filtered = filtered.filter(m => (m as any).isActive !== false && m.is_active !== false);
        }

        filtered = [...filtered].sort((a, b) => {
            let comparison = 0;
            switch (sortBy) {
                case 'date':
                    comparison = new Date(a.created_at || 0).getTime() - new Date(b.created_at || 0).getTime();
                    break;
                case 'accuracy':
                    comparison = a.accuracy - b.accuracy;
                    break;
                case 'training_time':
                    comparison = a.training_time - b.training_time;
                    break;
            }
            return sortOrder === 'desc' ? -comparison : comparison;
        });

        return filtered;
    }, [models, filterStatus, sortBy, sortOrder, showInactive]);

    const deployedModels = useMemo(() =>
            models.filter(m => m.status === 'deployed'),
        [models]);

    const trainedModels = useMemo(() =>
            models.filter(m => m.status === 'trained'),
        [models]);

    const handleTrain = async () => {
        setIsTraining(true);
        setTrainingProgress(0);
        setTrainingMetrics([]);

        const interval = setInterval(() => {
            setTrainingProgress(prev => {
                if (prev >= 90) return prev;
                return prev + Math.random() * 5;
            });
        }, 300);

        try {
            const backendConfig: TrainingConfig = {
                symbol: config.symbol,
                model_type: config.model_type,
                training_period: config.training_period,
                test_size: config.test_size,
                epochs: config.epochs,
                batch_size: config.batch_size,
                learning_rate: config.learning_rate,
                threshold: config.threshold,
                use_feature_engineering: config.use_feature_engineering,
                use_cross_validation: config.use_cross_validation,
                // RL-specific fields (ignored by backend for non-RL models)
                episodes: config.episodes ?? 200,
                gamma: config.gamma ?? 0.99,
            };

            const response = await mlstudio.trainModel(backendConfig);
            setTrainingProgress(100);

            if (response) {
                const newModel = {
                    ...response,
                    testAccuracy: response.test_accuracy,
                    overfitScore: response.overfit_score,
                    trainingTime: response.training_time,
                    featureImportance: response.feature_importance,
                    createdAt: new Date(),
                    deployedAt: null,
                    isActive: true,
                    lastUsed: null,
                    usageCount: 0,
                    performanceHistory: []
                };

                // Set real training metrics from backend
                if (response.training_history && response.training_history.length > 0) {
                    setTrainingMetrics(response.training_history);
                }

                setModels(prev => [newModel, ...prev]);
                setSelectedModel(newModel);
                setActiveModelId(newModel.id);
                setShowResults(true);
                setExpandedSection('results');
                setActiveTab('registry');
            }
        } catch (err) {
            console.error("Training failed:", err);
            alert("Training failed. Check console for details.");
        } finally {
            clearInterval(interval);
            setIsTraining(false);
        }
    };

    const handleDeploy = async (modelId: string) => {
        setIsDeploying(true);
        try {
            await mlstudio.deployModel(modelId);
            setModels(prev => prev.map(model =>
                model.id === modelId
                    ? {
                        ...model,
                        status: 'deployed' as const,
                        deployedAt: new Date(),
                        isActive: true
                    }
                    : model
            ));

            // Update selected model if it's the one being deployed
            if (selectedModel?.id === modelId) {
                setSelectedModel(prev => prev ? {
                    ...prev,
                    status: 'deployed' as const,
                    deployedAt: new Date(),
                    isActive: true
                } : null);
            }

            alert('Model deployed successfully!');
        } catch (err) {
            console.error("Failed to deploy:", err);
            alert('Failed to deploy model');
        } finally {
            setIsDeploying(false);
        }
    };

    const handleUndeploy = async (modelId: string) => {
        try {
            await mlstudio.undeployModel(modelId);
            setModels(prev => prev.map(model =>
                model.id === modelId
                    ? {...model, status: 'trained' as const}
                    : model
            ));

            if (selectedModel?.id === modelId) {
                setSelectedModel(prev => prev ? {...prev, status: 'trained' as const} : null);
            }
        } catch (err) {
            console.error("Failed to undeploy:", err);
        }
    };

    const handleToggleActive = async (modelId: string, isActive: boolean) => {
        try {
            await mlstudio.updateModelStatus({modelId, isActive});
            setModels(prev => prev.map(model =>
                model.id === modelId
                    ? {...model, isActive}
                    : model
            ));

            if (selectedModel?.id === modelId) {
                setSelectedModel(prev => prev ? {...prev, isActive} : null);
            }
        } catch (err) {
            console.error("Failed to update model status:", err);
        }
    };

    const handleDelete = async (modelId: string) => {
        try {
            await mlstudio.deleteModel(modelId);
            setModels(prev => prev.filter(model => model.id !== modelId));
            if (selectedModel?.id === modelId) {
                const nextModel = models.find(m => m.id !== modelId);
                setSelectedModel(nextModel || null);
                setActiveModelId(nextModel?.id || null);
            }
        } catch (err) {
            console.error("Failed to delete model:", err);
        }
    };

    const handleRetrain = (model: MLModel) => {
        setConfig({
            ...config,
            symbol: model.symbol,
            model_type: model.type
        });
        setExpandedSection('config');
        setActiveTab('train');
        window.scrollTo({top: 0, behavior: 'smooth'});
    };

    const handleExportModel = async (modelId: string) => {
        try {
            const blob = await mlstudio.exportModel(modelId);
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `model-${modelId}.pkl`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } catch (err) {
            console.error("Failed to export model:", err);
        }
    };

    const renderModelIcon = (type: MLModel['type'], size = 16) => {
        switch (type) {
            case 'LSTM':
                return <BrainCircuit size={size}/>;
            case 'Random Forest':
            case 'random_forest':
                return <TreePine size={size}/>;
            case 'Gradient Boosting':
            case 'gradient_boosting':
                return <TrendingUp size={size}/>;
            case 'XGBoost':
                return <Zap size={size}/>;
            case 'SVM':
            case 'svm':
                return <Target size={size}/>;
            case 'Logistic Regression':
            case 'logistic_regression':
                return <Activity size={size}/>;
            case 'RL Portfolio':
            case 'rl_portfolio_allocator':
                return <Layers size={size}/>;
            case 'RL Regime':
            case 'rl_regime_allocator':
                return <GitBranch size={size}/>;
            case 'RL Risk-Aware':
            case 'rl_risk_sensitive':
                return <Shield size={size}/>;
            case 'RL Sentiment':
            case 'rl_sentiment_trader':
                return <MessageSquare size={size}/>;
            default:
                return <CpuIcon size={size}/>;
        }
    };

    // Check if current model type is an RL strategy
    const isRLModel = (type: string) =>
        ['RL Portfolio', 'RL Regime', 'RL Risk-Aware', 'RL Sentiment'].includes(type);

    const getStatusColor = (status: MLModel['status']) => {
        switch (status) {
            case 'deployed':
                return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
            case 'training':
                return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
            case 'trained':
                return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
            default:
                return 'bg-slate-500/20 text-slate-400 border-slate-500/30';
        }
    };

    // Feature engineering options
    const featureOptions = [
        {id: 'ta', label: 'Technical Indicators', description: 'RSI, MACD, Moving Averages, Bollinger Bands'},
        {id: 'vol', label: 'Volatility Metrics', description: 'ATR, Historical Volatility, Beta'},
        {id: 'sentiment', label: 'Market Sentiment', description: 'News analysis, Social media trends'},
        {id: 'volume', label: 'Volume Analysis', description: 'Volume profile, Order flow imbalance'},
        {id: 'fundamental', label: 'Fundamental Data', description: 'P/E ratios, Earnings, Revenue growth'},
    ];

    return (
        <div className="space-y-8 animate-in fade-in duration-700">
            {/* Header with Tabs */}
            <div>
                <div className="flex items-center justify-between mb-6">
                    <div>
                        <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-3">
                            <div className="p-2 bg-gradient-to-br from-violet-500/20 to-violet-500/20 rounded-xl">
                                <Cpu className="text-violet-400" size={24}/>
                            </div>
                            ML Strategy Studio
                        </h1>
                        <p className="text-slate-500 text-sm mt-1">Build, train, deploy and track machine learning
                            models</p>
                    </div>
                    <div className="flex gap-3">
                        <button
                            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 text-sm font-medium flex items-center gap-2 transition-colors">
                            <Upload size={16}/>
                            Import Dataset
                        </button>
                        <button
                            className="px-4 py-2 bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-violet-500 rounded-lg text-white text-sm font-medium flex items-center gap-2 transition-all shadow-lg shadow-violet-600/20">
                            <Cloud size={16}/>
                            Cloud Training
                        </button>
                    </div>
                </div>

                {/* Navigation Tabs */}
                <div className="flex border-b border-slate-800">
                    <button
                        onClick={() => setActiveTab('train')}
                        className={`px-6 py-3 text-sm font-medium border-b-2 transition-all ${activeTab === 'train' ? 'border-violet-500 text-violet-400' : 'border-transparent text-slate-500 hover:text-slate-300'}`}
                    >
                        <div className="flex items-center gap-2">
                            <Play size={16}/>
                            Train Models
                        </div>
                    </button>
                    <button
                        onClick={() => setActiveTab('builder')}
                        className={`px-6 py-3 text-sm font-medium border-b-2 transition-all ${activeTab === 'builder' ? 'border-violet-500 text-violet-400' : 'border-transparent text-slate-500 hover:text-slate-300'}`}
                    >
                        <div className="flex items-center gap-2">
                            <Plus size={16}/>
                            Strategy Builder
                            <span
                                className="bg-violet-500/20 text-violet-400 text-[10px] px-1.5 py-0.5 rounded uppercase font-black ml-1">New</span>
                        </div>
                    </button>
                    <button
                        onClick={() => setActiveTab('registry')}
                        className={`px-6 py-3 text-sm font-medium border-b-2 transition-all ${activeTab === 'registry' ? 'border-violet-500 text-violet-400' : 'border-transparent text-slate-500 hover:text-slate-300'}`}
                    >
                        <div className="flex items-center gap-2">
                            <BarChart3 size={16}/>
                            Model Registry
                            <span className="bg-slate-800 text-slate-300 text-xs px-2 py-0.5 rounded-full">
                                {models.length}
                            </span>
                        </div>
                    </button>
                    <button
                        onClick={() => setActiveTab('deployed')}
                        className={`px-6 py-3 text-sm font-medium border-b-2 transition-all ${activeTab === 'deployed' ? 'border-violet-500 text-violet-400' : 'border-transparent text-slate-500 hover:text-slate-300'}`}
                    >
                        <div className="flex items-center gap-2">
                            <CheckCircle size={16}/>
                            Deployed Models
                            <span className="bg-emerald-500/20 text-emerald-400 text-xs px-2 py-0.5 rounded-full">
                                {deployedModels.length}
                            </span>
                        </div>
                    </button>
                </div>
            </div>


            {/* 1. Main Training Area */}
            {activeTab === 'train' && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Configuration Panel */}
                    <div className="lg:col-span-2 space-y-6">
                        {/* Configuration Card */}
                        <div
                            className="bg-gradient-to-br from-slate-900/50 to-slate-950/50 border border-slate-800/50 rounded-2xl p-6">
                            <div className="flex items-center justify-between mb-6">
                                <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                                    <Settings2 className="text-violet-400" size={20}/>
                                    Training Configuration
                                </h3>
                                <button
                                    onClick={() => setShowAdvanced(!showAdvanced)}
                                    className="text-xs text-slate-500 hover:text-slate-300 flex items-center gap-1 transition-colors"
                                >
                                    {showAdvanced ? <Minus size={14}/> : <Plus size={14}/>}
                                    {showAdvanced ? 'Basic Mode' : 'Advanced Mode'}
                                </button>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                {/* Basic Configuration */}
                                <div className="space-y-4">
                                    <div className="space-y-2">
                                        <label className="text-xs font-medium text-slate-400 flex items-center gap-2">
                                            <Target size={12}/>
                                            Symbol for Training
                                        </label>
                                        <div className="relative">
                                            <input
                                                type="text"
                                                value={config.symbol}
                                                onChange={(e) => setConfig({
                                                    ...config,
                                                    symbol: e.target.value.toUpperCase()
                                                })}
                                                className="w-full bg-slate-950/70 border border-slate-800 rounded-xl py-3 px-4 text-sm text-slate-100 focus:border-violet-500 focus:ring-2 focus:ring-violet-500/30 outline-none transition-all"
                                            />
                                            <div className="absolute right-3 top-1/2 -translate-y-1/2 flex gap-1">
                                                {['AAPL', 'TSLA', 'MSFT', 'NVDA'].map((sym) => (
                                                    <button
                                                        key={sym}
                                                        onClick={() => setConfig({...config, symbol: sym})}
                                                        className="text-xs px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-slate-200 transition-colors"
                                                    >
                                                        {sym}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="space-y-2">
                                        <label className="text-xs font-medium text-slate-400 flex items-center gap-2">
                                            <BrainCircuit size={12}/>
                                            Model Type
                                        </label>

                                        {/* Machine Learning */}
                                        <div>
                                            <div className="text-[10px] uppercase tracking-wider text-slate-600 font-bold mb-1.5">Machine Learning</div>
                                            <div className="grid grid-cols-3 gap-1.5">
                                                {(['Random Forest', 'Gradient Boosting', 'XGBoost', 'SVM', 'Logistic Regression'] as const).map((type) => (
                                                    <button
                                                        key={type}
                                                        onClick={() => setConfig({...config, model_type: type})}
                                                        className={`p-2 rounded-lg border transition-all ${config.model_type === type
                                                            ? 'bg-violet-500/10 border-violet-500/50 text-violet-400'
                                                            : 'bg-slate-900/50 border-slate-800 text-slate-400 hover:border-slate-700'
                                                        }`}
                                                    >
                                                        <div className="flex items-center justify-center gap-1.5">
                                                            {renderModelIcon(type, 13)}
                                                            <span className="text-[11px] font-medium">{type}</span>
                                                        </div>
                                                    </button>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Deep Learning */}
                                        <div>
                                            <div className="text-[10px] uppercase tracking-wider text-slate-600 font-bold mb-1.5">Deep Learning</div>
                                            <div className="grid grid-cols-3 gap-1.5">
                                                {(['LSTM'] as const).map((type) => (
                                                    <button
                                                        key={type}
                                                        onClick={() => setConfig({...config, model_type: type})}
                                                        className={`p-2 rounded-lg border transition-all ${config.model_type === type
                                                            ? 'bg-cyan-500/10 border-cyan-500/50 text-cyan-400'
                                                            : 'bg-slate-900/50 border-slate-800 text-slate-400 hover:border-slate-700'
                                                        }`}
                                                    >
                                                        <div className="flex items-center justify-center gap-1.5">
                                                            {renderModelIcon(type, 13)}
                                                            <span className="text-[11px] font-medium">{type}</span>
                                                        </div>
                                                    </button>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Reinforcement Learning */}
                                        <div>
                                            <div className="text-[10px] uppercase tracking-wider text-slate-600 font-bold mb-1.5 flex items-center gap-1.5">
                                                Reinforcement Learning
                                                <span className="text-[9px] px-1.5 py-0.5 bg-amber-500/20 text-amber-400 rounded font-black">PPO</span>
                                            </div>
                                            <div className="grid grid-cols-2 gap-1.5">
                                                {(['RL Portfolio', 'RL Regime', 'RL Risk-Aware', 'RL Sentiment'] as const).map((type) => (
                                                    <button
                                                        key={type}
                                                        onClick={() => setConfig({...config, model_type: type})}
                                                        className={`p-2 rounded-lg border transition-all ${config.model_type === type
                                                            ? 'bg-amber-500/10 border-amber-500/50 text-amber-400'
                                                            : 'bg-slate-900/50 border-slate-800 text-slate-400 hover:border-slate-700'
                                                        }`}
                                                    >
                                                        <div className="flex items-center justify-center gap-1.5">
                                                            {renderModelIcon(type, 13)}
                                                            <span className="text-[11px] font-medium">{type}</span>
                                                        </div>
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="space-y-2">
                                        <label className="text-xs font-medium text-slate-400 flex items-center gap-2">
                                            <Clock size={12}/>
                                            Training Period
                                        </label>
                                        <div className="flex gap-2">
                                            {(['1M', '3M', '6M', '1Y', '5Y'] as const).map((period) => (
                                                <button
                                                    key={period}
                                                    onClick={() => setConfig({...config, training_period: period})}
                                                    className={`px-3 py-2 rounded-lg text-xs transition-all ${config.training_period === period
                                                        ? 'bg-violet-500 text-white'
                                                        : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                                                    }`}
                                                >
                                                    {period}
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                </div>

                                {/* Advanced Configuration */}
                                <div className="space-y-4">
                                    <div className="space-y-2">
                                        <label className="text-xs font-medium text-slate-400 flex items-center gap-2">
                                            <PieChart size={12}/>
                                            Test Set Size: {config.test_size}%
                                        </label>
                                        <input
                                            type="range"
                                            min="10"
                                            max="40"
                                            step="5"
                                            value={config.test_size}
                                            onChange={(e) => setConfig({
                                                ...config,
                                                test_size: parseInt(e.target.value)
                                            })}
                                            className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-violet-500"
                                        />
                                        <div className="flex justify-between text-[10px] text-slate-500">
                                            <span>10%</span>
                                            <span>25%</span>
                                            <span>40%</span>
                                        </div>
                                    </div>

                                    {config.model_type === 'LSTM' && (
                                        <div className="grid grid-cols-2 gap-4">
                                            <div className="space-y-2">
                                                <label className="text-xs font-medium text-slate-400">Epochs</label>
                                                <input
                                                    type="number"
                                                    min="1"
                                                    max="500"
                                                    value={config.epochs}
                                                    onChange={(e) => setConfig({
                                                        ...config,
                                                        epochs: parseInt(e.target.value)
                                                    })}
                                                    className="w-full bg-slate-950/70 border border-slate-800 rounded-xl py-2 px-3 text-sm text-slate-100 focus:border-violet-500 outline-none"
                                                />
                                            </div>
                                            <div className="space-y-2">
                                                <label className="text-xs font-medium text-slate-400">Batch Size</label>
                                                <select
                                                    value={config.batch_size}
                                                    onChange={(e) => setConfig({
                                                        ...config,
                                                        batch_size: parseInt(e.target.value)
                                                    })}
                                                    className="w-full bg-slate-950/70 border border-slate-800 rounded-xl py-2 px-3 text-sm text-slate-300 focus:border-violet-500 outline-none"
                                                >
                                                    <option value="16">16</option>
                                                    <option value="32">32</option>
                                                    <option value="64">64</option>
                                                    <option value="128">128</option>
                                                </select>
                                            </div>
                                        </div>
                                    )}

                                    {isRLModel(config.model_type) && (
                                        <div className="space-y-3 p-3 bg-amber-500/5 border border-amber-500/20 rounded-xl">
                                            <div className="flex items-center gap-2 mb-1">
                                                <Gamepad2 size={12} className="text-amber-400"/>
                                                <span className="text-[10px] font-bold text-amber-400 uppercase tracking-wider">RL Training</span>
                                            </div>
                                            <div className="grid grid-cols-2 gap-3">
                                                <div className="space-y-1.5">
                                                    <label className="text-xs font-medium text-slate-400">Episodes</label>
                                                    <input
                                                        type="number"
                                                        min="50"
                                                        max="2000"
                                                        step="50"
                                                        value={config.episodes ?? 200}
                                                        onChange={(e) => setConfig({
                                                            ...config,
                                                            episodes: parseInt(e.target.value)
                                                        })}
                                                        className="w-full bg-slate-950/70 border border-slate-800 rounded-lg py-2 px-3 text-sm text-slate-100 focus:border-amber-500 outline-none"
                                                    />
                                                    <div className="text-[9px] text-slate-600">PPO training iterations</div>
                                                </div>
                                                <div className="space-y-1.5">
                                                    <label className="text-xs font-medium text-slate-400">Discount (γ)</label>
                                                    <input
                                                        type="number"
                                                        min="0.9"
                                                        max="0.999"
                                                        step="0.01"
                                                        value={config.gamma ?? 0.99}
                                                        onChange={(e) => setConfig({
                                                            ...config,
                                                            gamma: parseFloat(e.target.value)
                                                        })}
                                                        className="w-full bg-slate-950/70 border border-slate-800 rounded-lg py-2 px-3 text-sm text-slate-100 focus:border-amber-500 outline-none"
                                                    />
                                                    <div className="text-[9px] text-slate-600">Future reward weight</div>
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {showAdvanced && (
                                        <div className="space-y-3 animate-in slide-in-from-top-2">
                                            <div className="space-y-2">
                                                <label className="text-xs font-medium text-slate-400">Learning
                                                    Rate</label>
                                                <input
                                                    type="range"
                                                    min="0.0001"
                                                    max="0.01"
                                                    step="0.0001"
                                                    value={config.learning_rate}
                                                    onChange={(e) => setConfig({
                                                        ...config,
                                                        learning_rate: parseFloat(e.target.value)
                                                    })}
                                                    className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-violet-500"
                                                />
                                                <div
                                                    className="text-xs text-slate-500 text-center">{config.learning_rate.toFixed(4)}</div>
                                            </div>

                                            <div className="flex items-center gap-3">
                                                <label className="flex items-center gap-2 text-xs text-slate-400">
                                                    <input
                                                        type="checkbox"
                                                        checked={config.use_feature_engineering}
                                                        onChange={(e) => setConfig({
                                                            ...config,
                                                            use_feature_engineering: e.target.checked
                                                        })}
                                                        className="rounded accent-violet-500"
                                                    />
                                                    Feature Engineering
                                                </label>
                                                <label className="flex items-center gap-2 text-xs text-slate-400">
                                                    <input
                                                        type="checkbox"
                                                        checked={config.use_cross_validation}
                                                        onChange={(e) => setConfig({
                                                            ...config,
                                                            use_cross_validation: e.target.checked
                                                        })}
                                                        className="rounded accent-violet-500"
                                                    />
                                                    5-Fold CV
                                                </label>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Feature Engineering Options */}
                            {config.use_feature_engineering && (
                                <div className="mt-6 p-4 bg-slate-950/50 border border-slate-800 rounded-xl">
                                    <div className="flex items-center gap-2 mb-3">
                                        <Filter className="text-violet-500" size={14}/>
                                        <span className="text-xs font-medium text-slate-400">Feature Engineering Pipeline</span>
                                    </div>
                                    <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                                        {featureOptions.map((feat) => (
                                            <div key={feat.id}
                                                 className="p-2 bg-slate-900/50 rounded-lg border border-slate-800">
                                                <div
                                                    className="text-[10px] font-medium text-violet-400">{feat.label}</div>
                                                <div className="text-[9px] text-slate-500 mt-1">{feat.description}</div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Training Button */}
                            <button
                                onClick={handleTrain}
                                disabled={isTraining || !config.symbol}
                                className={`w-full mt-6 py-4 rounded-xl font-bold transition-all flex items-center justify-center gap-3 ${isTraining
                                    ? 'bg-slate-800 text-slate-400'
                                    : 'bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-violet-500 text-white shadow-xl shadow-violet-600/20'
                                }`}
                            >
                                {isTraining ? (
                                    <>
                                        <div
                                            className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                                        <span>Training... {trainingProgress.toFixed(0)}%</span>
                                    </>
                                ) : (
                                    <>
                                        <Play size={18}/>
                                        <span>Launch Training Session</span>
                                    </>
                                )}
                            </button>

                            {/* Training Progress Bar */}
                            {isTraining && (
                                <div className="mt-4">
                                    <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-gradient-to-r from-violet-500 to-indigo-500 transition-all duration-300"
                                            style={{width: `${trainingProgress}%`}}
                                        />
                                    </div>
                                    <div className="flex justify-between text-xs text-slate-500 mt-2">
                                        <span>Data Loading</span>
                                        <span>Feature Engineering</span>
                                        <span>Model Training</span>
                                        <span>Validation</span>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Sidebar - Model Info & Metrics */}
                    <div className="space-y-6">
                        {/* Model Info Card */}
                        <div
                            className="bg-gradient-to-br from-slate-900/50 to-slate-950/50 border border-slate-800/50 rounded-2xl p-6">
                            <h4 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
                                <Info className="text-violet-400" size={16}/>
                                Model Information
                            </h4>
                            <div className="space-y-4">
                                {isRLModel(config.model_type) ? (
                                    <>
                                        <div className="p-3 bg-slate-900/30 rounded-lg">
                                            <p className="text-[10px] font-medium text-amber-400 uppercase mb-1">RL Objective</p>
                                            <p className="text-xs text-slate-300">
                                                {config.model_type === 'RL Portfolio' && 'Learns optimal portfolio weight allocation across multiple assets using PPO.'}
                                                {config.model_type === 'RL Regime' && 'Detects market regimes and allocates capital across sub-strategies dynamically.'}
                                                {config.model_type === 'RL Risk-Aware' && 'Maximizes risk-adjusted returns with explicit drawdown and CVaR penalties.'}
                                                {config.model_type === 'RL Sentiment' && 'Fuses technical indicators with sentiment signals for adaptive trading.'}
                                            </p>
                                        </div>
                                        <div className="p-3 bg-slate-900/30 rounded-lg">
                                            <p className="text-[10px] font-medium text-amber-400 uppercase mb-1">Training Method</p>
                                            <p className="text-xs text-slate-300">PPO actor-critic with {config.episodes ?? 200} episodes. Agent explores and exploits within historical data episodes.</p>
                                        </div>
                                        <div className="p-3 bg-slate-900/30 rounded-lg">
                                            <p className="text-[10px] font-medium text-blue-400 uppercase mb-1">Evaluation</p>
                                            <p className="text-xs text-slate-300">Evaluated on cumulative return, Sharpe ratio, max drawdown, and training stability.</p>
                                        </div>
                                    </>
                                ) : (
                                    <>
                                        <div className="p-3 bg-slate-900/30 rounded-lg">
                                            <p className="text-[10px] font-medium text-violet-400 uppercase mb-1">Prediction Target</p>
                                            <p className="text-xs text-slate-300">Price movement probability {config.threshold * 100}% over next 5 intervals</p>
                                        </div>
                                        <div className="p-3 bg-slate-900/30 rounded-lg">
                                            <p className="text-[10px] font-medium text-violet-400 uppercase mb-1">Data Requirements</p>
                                            <p className="text-xs text-slate-300">Minimum 500 data points. Multi-year datasets recommended for robust models.</p>
                                        </div>
                                        <div className="p-3 bg-slate-900/30 rounded-lg">
                                            <p className="text-[10px] font-medium text-blue-400 uppercase mb-1">Performance Metrics</p>
                                            <p className="text-xs text-slate-300">Models evaluated on accuracy, F1-score, precision, recall, and overfitting score.</p>
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>

                        {/* Quick Stats */}
                        {loadError ? (
                            <div className="bg-red-500/10 border border-red-500/30 p-4 rounded-2xl text-center">
                                <p className="text-red-400 text-sm mb-2">{loadError}</p>
                                <button onClick={fetchModels} className="text-xs text-violet-400 hover:text-violet-300">Retry</button>
                            </div>
                        ) : isLoadingModels ? (
                            <div className="bg-slate-900 p-6 rounded-2xl flex justify-center">
                                <Loader2 className="animate-spin text-slate-500"/>
                            </div>
                        ) : (
                            <div
                                className="bg-gradient-to-br from-slate-900/50 to-slate-950/50 border border-slate-800/50 rounded-2xl p-6">
                                <h4 className="text-sm font-bold text-slate-300 mb-4">Quick Stats</h4>
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="text-center p-3 bg-slate-900/30 rounded-lg">
                                        <div className="text-xl font-bold text-emerald-400">{models.length}</div>
                                        <div className="text-[10px] text-slate-500 mt-1">Models</div>
                                    </div>
                                    <div className="text-center p-3 bg-slate-900/30 rounded-lg">
                                        <div
                                            className="text-xl font-bold text-blue-400">{models.filter(m => m.status === 'deployed').length}</div>
                                        <div className="text-[10px] text-slate-500 mt-1">Deployed</div>
                                    </div>
                                    <div className="text-center p-3 bg-slate-900/30 rounded-lg">
                                        <div className="text-xl font-bold text-amber-400">
                                            {models.length > 0
                                                ? (models.reduce((acc, m) => acc + m.accuracy, 0) / models.length * 100).toFixed(1)
                                                : '0.0'
                                            }%
                                        </div>
                                        <div className="text-[10px] text-slate-500 mt-1">Avg Accuracy</div>
                                    </div>
                                    <div className="text-center p-3 bg-slate-900/30 rounded-lg">
                                        <div className="text-xl font-bold text-purple-400">
                                            {models.length > 0
                                                ? (models.reduce((acc, m) => acc + m.training_time, 0) / models.length / 60).toFixed(1)
                                                : '0.0'
                                            }
                                        </div>
                                        <div className="text-[10px] text-slate-500 mt-1">Avg Hours</div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {activeTab === 'builder' && (
                <div className="space-y-6">
                    <StrategyBuilder
                        models={models}
                        isLoading={isLoadingModels}
                        onSave={(config) => {
                            // Send enriched blocks to backtest store
                            setVisualStrategy(config.blocks);
                            navigateTo('backtest');
                        }}
                    />

                    <div
                        className="bg-slate-900/30 border border-slate-800 p-6 rounded-2xl flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <div className="w-12 h-12 bg-violet-500/10 rounded-xl flex items-center justify-center">
                                <Target className="text-violet-400" size={24}/>
                            </div>
                            <div>
                                <h4 className="text-sm font-bold text-white uppercase tracking-wider">Strategy Ready for
                                    Backtesting</h4>
                                <p className="text-xs text-slate-500">Your visual logic will be converted to
                                    the &quot;Visual Strategy Builder&quot; engine automatically.</p>
                            </div>
                        </div>
                        <button
                            onClick={() => navigateTo('backtest')}
                            className="px-6 py-3 bg-slate-800 hover:bg-slate-700 text-white rounded-xl text-sm font-bold transition-all border border-slate-700"
                        >
                            Go to Backtest →
                        </button>
                    </div>
                </div>
            )}

            {activeTab === 'registry' &&
                <ModelRegistry
                    setFilterStatus={setFilterStatus}
                    models={models}
                    filterStatus={filterStatus}
                    trainedModels={trainedModels}
                    deployedModels={deployedModels}
                    showInactive={showInactive}
                    setShowInactive={setShowInactive}
                    sortBy={sortBy}
                    setSortBy={setSortBy}
                    sortOrder={sortOrder}
                    setSortOrder={setSortOrder}
                    viewMode={viewMode}
                    setViewMode={setViewMode}
                    isLoadingModels={isLoadingModels}
                    filteredAndSortedModels={filteredAndSortedModels}
                    setActiveTab={setActiveTab}
                    setSelectedModel={setSelectedModel}
                    setActiveModelId={setActiveModelId}
                    setShowResults={setShowResults}
                    selectedModel={selectedModel}
                    renderModelIcon={renderModelIcon}
                    getStatusColor={getStatusColor}
                    isDeploying={isDeploying}
                    handleRetrain={handleRetrain}
                    handleDeploy={handleDeploy}
                    handleUndeploy={handleUndeploy}
                    handleDelete={handleDelete}
                    handleToggleActive={handleToggleActive}
                    handleExportModel={handleExportModel}
                />
            }

            {activeTab === 'deployed' && (
                <div className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        {deployedModels.map((model) => (
                                <div key={model.id}
                                     className="bg-gradient-to-br from-slate-900/50 to-slate-950/50 border border-emerald-500/20 rounded-2xl p-6">
                                    <div className="flex items-center justify-between mb-4">
                                        <div className="flex items-center gap-3">
                                            <div className="p-2 bg-emerald-500/20 rounded-lg">
                                                {renderModelIcon(model.type)}
                                            </div>
                                            <div>
                                                <div className="font-medium text-slate-100">{model.name}</div>
                                                <div className="text-xs text-slate-500">{model.symbol}</div>
                                            </div>
                                        </div>
                                        <div className="text-xs px-2 py-1 bg-emerald-500/20 text-emerald-400 rounded-full">
                                            Active
                                        </div>
                                    </div>

                                    <div className="space-y-3">
                                        <div className="flex justify-between text-sm">
                                            <span className="text-slate-500">Accuracy:</span>
                                            <span
                                                className="text-emerald-400 font-medium">{(model.accuracy * 100).toFixed(1)}%</span>
                                        </div>
                                        <div className="flex justify-between text-sm">
                                            <span className="text-slate-500">Live P&L:</span>
                                            <span className="text-emerald-400 font-medium">+2.4%</span>
                                        </div>
                                        <div className="flex justify-between text-sm">
                                            <span className="text-slate-500">Trades Today:</span>
                                            <span className="text-slate-300 font-medium">12</span>
                                        </div>
                                    </div>

                                    <div className="mt-4 pt-4 border-t border-slate-800/50">
                                        <button
                                            onClick={() => handleUndeploy(model.id)}
                                            className="w-full py-2 bg-amber-500/10 hover:bg-amber-500/20 rounded-lg text-amber-500 text-sm font-medium transition-colors border border-amber-500/20"
                                        >
                                            Stop Strategy
                                        </button>
                                    </div>
                                </div>
                            )
                        )}
                    </div>
                </div>
            )
            }

            {/* 2. Training Results - only shown on the Train tab */}
            {showResults && selectedModel && activeTab === 'train' && (
                <div className="space-y-6 animate-in slide-in-from-bottom-4">
                    <div className="flex items-center justify-between">
                        <h3 className="text-lg font-bold text-slate-100">Training Results</h3>
                        <div className="flex gap-2">
                            <button
                                onClick={() => selectedModel && handleDelete(selectedModel.id)}
                                className="px-4 py-2 bg-red-500/10 hover:bg-red-500/20 rounded-lg text-red-500 text-sm font-medium flex items-center gap-2 transition-colors border border-red-500/20"
                            >
                                <Trash2 size={16}/>
                                Delete Model
                            </button>
                            {selectedModel.status !== 'deployed' && (
                                <button
                                    onClick={() => handleDeploy(selectedModel.id)}
                                    disabled={isDeploying}
                                    className="px-4 py-2 bg-emerald-500/10 hover:bg-emerald-500/20 rounded-lg text-emerald-500 text-sm font-medium flex items-center gap-2 transition-colors border border-emerald-500/20"
                                >
                                    {isDeploying ? <Loader2 size={16} className="animate-spin"/> :
                                        <CheckCircle2 size={16}/>}
                                    Deploy Strategy
                                </button>
                            )}
                            <button
                                className="px-4 py-2 bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-violet-500 rounded-lg text-white text-sm font-medium flex items-center gap-2 transition-all shadow-lg shadow-violet-600/20">
                                <Maximize2 size={16}/>
                                Compare Models
                            </button>
                        </div>
                    </div>

                    {/* Performance Metrics */}
                    {(() => {
                        const isRL = selectedModel.type.startsWith('rl_');
                        return (
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                <div
                                    className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 p-5 rounded-2xl">
                                    <p className="text-xs font-medium text-slate-500 mb-2">
                                        {isRL ? 'Best Return / Sharpe' : 'Training Accuracy'}
                                    </p>
                                    <div className="flex items-end gap-2">
                                        <span className="text-2xl font-bold text-violet-400">
                                            {isRL
                                                ? selectedModel.accuracy.toFixed(3)
                                                : `${(selectedModel.accuracy * 100).toFixed(1)}%`
                                            }
                                        </span>
                                        {!isRL && <span className="text-xs text-slate-500 mb-1">±2.3%</span>}
                                    </div>
                                </div>
                                <div
                                    className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 p-5 rounded-2xl">
                                    <p className="text-xs font-medium text-slate-500 mb-2">
                                        {isRL ? 'Avg Tail Return' : 'Test Accuracy'}
                                    </p>
                                    <div className="flex items-end gap-2">
                                        <span className="text-2xl font-bold text-violet-400">
                                            {isRL
                                                ? selectedModel.test_accuracy.toFixed(4)
                                                : `${(selectedModel.test_accuracy * 100).toFixed(1)}%`
                                            }
                                        </span>
                                        {!isRL && <span className="text-xs text-slate-500 mb-1">±1.8%</span>}
                                    </div>
                                </div>
                                <div
                                    className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 p-5 rounded-2xl">
                                    <p className="text-xs font-medium text-slate-500 mb-2">
                                        {isRL ? 'Stability Score' : 'Overfit Score'}
                                    </p>
                                    <div className="flex items-center justify-between">
                                        <span
                                            className="text-2xl font-bold text-emerald-400">
                                            {isRL
                                                ? selectedModel.overfit_score.toFixed(4)
                                                : `${(selectedModel.overfit_score * 100).toFixed(1)}%`
                                            }
                                        </span>
                                        <CheckCircle2 className="text-emerald-500" size={20}/>
                                    </div>
                                </div>
                                <div
                                    className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 p-5 rounded-2xl">
                                    <p className="text-xs font-medium text-slate-500 mb-2">
                                        {isRL ? 'State Dimensions' : 'Features Used'}
                                    </p>
                                    <div className="flex items-end gap-2">
                                        <span className="text-2xl font-bold text-blue-400">{selectedModel.features}</span>
                                        <span className="text-xs text-slate-500 mb-1">
                                            {isRL ? 'dims' : 'indicators'}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        );
                    })()}

                    {/* Charts Section */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Feature Importance */}
                        <div
                            className="bg-gradient-to-br from-slate-900/50 to-slate-950/50 border border-slate-800/50 rounded-2xl p-6">
                            <div className="flex items-center justify-between mb-6">
                                <h4 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                                    <BarChart3 className="text-violet-500" size={18}/>
                                    {selectedModel.type.startsWith('rl_') ? 'Training Metrics' : 'Feature Importance'}
                                </h4>
                                <span
                                    className="text-xs text-slate-500">{selectedModel.feature_importance.length} {selectedModel.type.startsWith('rl_') ? 'metrics' : 'features'}</span>
                            </div>
                            <div className="h-[300px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={selectedModel.feature_importance} layout="vertical">
                                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false}/>
                                        <XAxis type="number" domain={[0, 0.35]} hide/>
                                        <YAxis
                                            dataKey="feature"
                                            type="category"
                                            tick={{fill: '#94a3b8', fontSize: 10}}
                                            width={80}
                                        />
                                        <Tooltip
                                            formatter={(value) => [`${(Number(value) * 100).toFixed(1)}%`, 'Importance']}
                                            contentStyle={{
                                                backgroundColor: '#0f172a',
                                                border: '1px solid #1e293b',
                                                borderRadius: '8px'
                                            }}
                                        />
                                        <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                                            {selectedModel.feature_importance.map((entry, index) => (
                                                <Cell
                                                    key={`cell-${index}`}
                                                    fill={`url(#gradient-${index})`}
                                                />
                                            ))}
                                        </Bar>
                                        <defs>
                                            {selectedModel.feature_importance.map((_, index) => (
                                                <linearGradient key={`gradient-${index}`} id={`gradient-${index}`}
                                                                x1="0"
                                                                y1="0" x2="1" y2="0">
                                                    <stop offset="0%" stopColor="#d946ef" stopOpacity={0.8}/>
                                                    <stop offset="100%" stopColor="#ec4899" stopOpacity={0.8}/>
                                                </linearGradient>
                                            ))}
                                        </defs>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Training Metrics Chart */}
                        <div
                            className="bg-gradient-to-br from-slate-900/50 to-slate-950/50 border border-slate-800/50 rounded-2xl p-6">
                            <div className="flex items-center justify-between mb-6">
                                <h4 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                                    <LineChart className="text-cyan-500" size={18}/>
                                    {selectedModel.type.startsWith('rl_') ? 'Episode Returns' : 'Training Progress'}
                                </h4>
                                {trainingMetrics.length > 1 && (
                                    <div className="flex gap-2">
                                        <span className="text-xs px-2 py-1 bg-red-500/20 rounded text-red-400">
                                            {selectedModel.type.startsWith('rl_') ? 'Neg Return' : 'Loss'}
                                        </span>
                                        <span className="text-xs px-2 py-1 bg-blue-500/20 rounded text-blue-400">
                                            {selectedModel.type.startsWith('rl_') ? 'Avg Return' : 'Accuracy'}
                                        </span>
                                    </div>
                                )}
                            </div>
                            <div className="h-[300px]">
                                {trainingMetrics.length > 1 ? (
                                    /* Multi-epoch chart: LSTM, GradientBoosting, RL */
                                    <ResponsiveContainer width="100%" height="100%">
                                        <ComposedChart data={trainingMetrics}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                                            <XAxis
                                                dataKey="epoch"
                                                stroke="#64748b"
                                                fontSize={10}
                                                label={selectedModel.type.startsWith('rl_') ? {value: 'Episode', position: 'insideBottom', offset: -2, fontSize: 10, fill: '#64748b'} : undefined}
                                            />
                                            <YAxis stroke="#64748b" fontSize={10}/>
                                            <Tooltip contentStyle={{
                                                backgroundColor: '#0f172a',
                                                border: '1px solid #1e293b'
                                            }}/>
                                            <Legend/>
                                            <Area
                                                type="monotone"
                                                dataKey="loss"
                                                fill="#ef4444"
                                                fillOpacity={0.1}
                                                stroke="#ef4444"
                                                strokeWidth={1}
                                                name={selectedModel.type.startsWith('rl_') ? 'Neg Return' : 'Training Loss'}
                                            />
                                            {!selectedModel.type.startsWith('rl_') && (
                                            <Area
                                                type="monotone"
                                                dataKey="val_loss"
                                                fill="#64748b"
                                                fillOpacity={0.1}
                                                stroke="#64748b"
                                                strokeDasharray="3 3"
                                                strokeWidth={1}
                                                name="Val Loss"
                                            />)}
                                            <Line
                                                type="monotone"
                                                dataKey="accuracy"
                                                stroke="#3b82f6"
                                                strokeWidth={2}
                                                dot={false}
                                                name={selectedModel.type.startsWith('rl_') ? 'Avg Return' : 'Accuracy'}
                                            />
                                        </ComposedChart>
                                    </ResponsiveContainer>
                                ) : trainingMetrics.length === 1 ? (
                                    /* Single-point summary: RF, SVM, Logistic */
                                    <div className="flex items-center justify-center h-full">
                                        <div className="grid grid-cols-2 gap-6 max-w-md w-full">
                                            <div
                                                className="bg-slate-800/50 rounded-xl p-5 text-center border border-slate-700/50">
                                                <p className="text-xs text-slate-500 mb-1 uppercase tracking-wider">Train
                                                    Accuracy</p>
                                                <p className="text-2xl font-bold text-blue-400">{(trainingMetrics[0].accuracy * 100).toFixed(1)}%</p>
                                            </div>
                                            <div
                                                className="bg-slate-800/50 rounded-xl p-5 text-center border border-slate-700/50">
                                                <p className="text-xs text-slate-500 mb-1 uppercase tracking-wider">Val
                                                    Accuracy</p>
                                                <p className="text-2xl font-bold text-emerald-400">{trainingMetrics[0].val_accuracy != null ? (trainingMetrics[0].val_accuracy * 100).toFixed(1) + '%' : 'N/A'}</p>
                                            </div>
                                            <div
                                                className="bg-slate-800/50 rounded-xl p-5 text-center border border-slate-700/50">
                                                <p className="text-xs text-slate-500 mb-1 uppercase tracking-wider">Train
                                                    Loss</p>
                                                <p className="text-2xl font-bold text-red-400">{trainingMetrics[0].loss.toFixed(4)}</p>
                                            </div>
                                            <div
                                                className="bg-slate-800/50 rounded-xl p-5 text-center border border-slate-700/50">
                                                <p className="text-xs text-slate-500 mb-1 uppercase tracking-wider">Val
                                                    Loss</p>
                                                <p className="text-2xl font-bold text-orange-400">{trainingMetrics[0].val_loss != null ? trainingMetrics[0].val_loss.toFixed(4) : 'N/A'}</p>
                                            </div>
                                        </div>
                                    </div>
                                ) : (
                                    /* Empty state */
                                    <div className="flex flex-col items-center justify-center h-full text-slate-500">
                                        <BrainCircuit size={48} className="mb-3 opacity-30"/>
                                        <p className="text-sm">Train a model to see metrics</p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )
            }
        </div>
    )
        ;
};

export default MLStudio;
