'use client'
import React, {useEffect, useMemo, useState} from 'react';
import {
    BarChart3,
    BrainCircuit,
    CheckCircle2,
    Clock,
    Cloud,
    Cpu,
    Cpu as CpuIcon,
    Filter,
    Info,
    LineChart,
    Loader2,
    Maximize2,
    Minus,
    PieChart,
    Play,
    Plus,
    Settings2,
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
import {MLModel, TrainingConfig} from '@/types/all_types';

const MLStudio = () => {
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
        use_cross_validation: true
    });

    // Load models on mount
    useEffect(() => {
        fetchModels();
    }, []);

    const fetchModels = async () => {
        setIsLoadingModels(true);
        try {
            const response = await mlstudio.getModels();
            if (response) {
                // Map backend snake_case to frontend camelCase if needed,
                // but our api.ts defines camelCase for fields that come as snake_case from Pydantic
                // The API needs to return matching keys or we need to map them here.
                // Assuming Pydantic v2 or aliasing works, or we map manually.
                // Let's assume manual mapping for safety given the previous schema.
                const mappedModels = response.map((m) => ({
                    ...m,
                    testAccuracy: m.test_accuracy,
                    overfitScore: m.overfit_score,
                    trainingTime: m.training_time,
                    featureImportance: m.feature_importance,
                }));
                setModels(mappedModels);
                if (mappedModels.length > 0 && !selectedModel) {
                    setSelectedModel(mappedModels[0]);
                    setActiveModelId(mappedModels[0].id);
                }
            }
        } catch (err) {
            console.error("Failed to fetch models:", err);
        } finally {
            setIsLoadingModels(false);
        }
    };

    // Training metrics over time (Mock for now, could also come from backend during training)
    const trainingMetrics = useMemo(() =>
        Array.from({ length: 50 }, (_, i) => ({
            epoch: i + 1,
            loss: Math.exp(-i * 0.1) + Math.random() * 0.1,
            accuracy: 0.5 + (i * 0.01) + Math.random() * 0.05,
            val_loss: Math.exp(-i * 0.08) + Math.random() * 0.12,
            val_accuracy: 0.5 + (i * 0.008) + Math.random() * 0.04
        }))
        , []);

    const handleTrain = async () => {
        setIsTraining(true);
        setTrainingProgress(0);

        // Simulate training progress
        const interval = setInterval(() => {
            setTrainingProgress(prev => {
                if (prev >= 90) return prev;
                return prev + Math.random() * 5;
            });
        }, 300);

        try {
            // Map frontend config to backend expected format (snake_case)
            const backendConfig = {
                symbol: config.symbol,
                model_type: config.model_type,
                training_period: config.training_period,
                test_size: config.test_size,
                epochs: config.epochs,
                batch_size: config.batch_size,
                learning_rate: config.learning_rate,
                threshold: config.threshold,
                use_feature_engineering: config.use_feature_engineering,
                use_cross_validation: config.use_cross_validation
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
                };

                setModels(prev => [newModel, ...prev]);
                setSelectedModel(newModel);
                setActiveModelId(newModel.id);
                setShowResults(true);
                setExpandedSection('results');
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
                    ? { ...model, status: 'deployed' as const }
                    : { ...model, status: model.status === 'deployed' ? 'trained' as const : model.status }
            ));
        } catch (err) {
            console.error("Failed to deploy:", err);
        } finally {
            setIsDeploying(false);
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
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    const renderModelIcon = (type: MLModel['type']) => {
        switch (type) {
            case 'LSTM': return <BrainCircuit size={16} />;
            case 'Random Forest': return <TreePine size={16} />;
            case 'Gradient Boosting': return <TrendingUp size={16} />;
            case 'XGBoost': return <Zap size={16} />;
            default: return <CpuIcon size={16} />;
        }
    };

    // Feature engineering options
    const featureOptions = [
        { id: 'ta', label: 'Technical Indicators', description: 'RSI, MACD, Moving Averages, Bollinger Bands' },
        { id: 'vol', label: 'Volatility Metrics', description: 'ATR, Historical Volatility, Beta' },
        { id: 'sentiment', label: 'Market Sentiment', description: 'News analysis, Social media trends' },
        { id: 'volume', label: 'Volume Analysis', description: 'Volume profile, Order flow imbalance' },
        { id: 'fundamental', label: 'Fundamental Data', description: 'P/E ratios, Earnings, Revenue growth' },
    ];

    return (
        <div className="space-y-8 animate-in fade-in duration-700">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-3">
                        <div className="p-2 bg-gradient-to-br from-fuchsia-500/20 to-pink-500/20 rounded-xl">
                            <Cpu className="text-fuchsia-400" size={24} />
                        </div>
                        ML Strategy Studio
                    </h1>
                    <p className="text-slate-500 text-sm mt-1">Build, train, and deploy machine learning models for algorithmic trading</p>
                </div>
                <div className="flex gap-3">
                    <button className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 text-sm font-medium flex items-center gap-2 transition-colors">
                        <Upload size={16} />
                        Import Dataset
                    </button>
                    <button className="px-4 py-2 bg-gradient-to-r from-fuchsia-600 to-pink-600 hover:from-fuchsia-500 hover:to-pink-500 rounded-lg text-white text-sm font-medium flex items-center gap-2 transition-all shadow-lg shadow-fuchsia-600/20">
                        <Cloud size={16} />
                        Cloud Training
                    </button>
                </div>
            </div>

            {/* 1. Main Training Area */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Configuration Panel */}
                <div className="lg:col-span-2 space-y-6">
                    {/* Configuration Card */}
                    <div className="bg-gradient-to-br from-slate-900/50 to-slate-950/50 border border-slate-800/50 rounded-2xl p-6">
                        <div className="flex items-center justify-between mb-6">
                            <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                                <Settings2 className="text-fuchsia-400" size={20} />
                                Training Configuration
                            </h3>
                            <button
                                onClick={() => setShowAdvanced(!showAdvanced)}
                                className="text-xs text-slate-500 hover:text-slate-300 flex items-center gap-1 transition-colors"
                            >
                                {showAdvanced ? <Minus size={14} /> : <Plus size={14} />}
                                {showAdvanced ? 'Basic Mode' : 'Advanced Mode'}
                            </button>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* Basic Configuration */}
                            <div className="space-y-4">
                                <div className="space-y-2">
                                    <label className="text-xs font-medium text-slate-400 flex items-center gap-2">
                                        <Target size={12} />
                                        Symbol for Training
                                    </label>
                                    <div className="relative">
                                        <input
                                            type="text"
                                            value={config.symbol}
                                            onChange={(e) => setConfig({ ...config, symbol: e.target.value.toUpperCase() })}
                                            className="w-full bg-slate-950/70 border border-slate-800 rounded-xl py-3 px-4 text-sm text-slate-100 focus:border-fuchsia-500 focus:ring-2 focus:ring-fuchsia-500/30 outline-none transition-all"
                                        />
                                        <div className="absolute right-3 top-1/2 -translate-y-1/2 flex gap-1">
                                            {['AAPL', 'TSLA', 'MSFT', 'NVDA'].map((sym) => (
                                                <button
                                                    key={sym}
                                                    onClick={() => setConfig({ ...config, symbol: sym })}
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
                                        <BrainCircuit size={12} />
                                        Model Type
                                    </label>
                                    <div className="grid grid-cols-2 gap-2">
                                        {(['Random Forest', 'Gradient Boosting', 'LSTM', 'XGBoost'] as const).map((type) => (
                                            <button
                                                key={type}
                                                onClick={() => setConfig({ ...config, model_type: type })}
                                                className={`p-3 rounded-xl border transition-all ${config.model_type === type
                                                    ? 'bg-fuchsia-500/10 border-fuchsia-500/50 text-fuchsia-400'
                                                    : 'bg-slate-900/50 border-slate-800 text-slate-400 hover:border-slate-700'
                                                    }`}
                                            >
                                                <div className="flex items-center justify-center gap-2">
                                                    {renderModelIcon(type)}
                                                    <span className="text-xs font-medium">{type}</span>
                                                </div>
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                <div className="space-y-2">
                                    <label className="text-xs font-medium text-slate-400 flex items-center gap-2">
                                        <Clock size={12} />
                                        Training Period
                                    </label>
                                    <div className="flex gap-2">
                                        {(['1M', '3M', '6M', '1Y', '5Y'] as const).map((period) => (
                                            <button
                                                key={period}
                                                onClick={() => setConfig({ ...config, training_period: period })}
                                                className={`px-3 py-2 rounded-lg text-xs transition-all ${config.training_period === period
                                                    ? 'bg-fuchsia-500 text-white'
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
                                        <PieChart size={12} />
                                        Test Set Size: {config.test_size}%
                                    </label>
                                    <input
                                        type="range"
                                        min="10"
                                        max="40"
                                        step="5"
                                        value={config.test_size}
                                        onChange={(e) => setConfig({ ...config, test_size: parseInt(e.target.value) })}
                                        className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-fuchsia-500"
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
                                                onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
                                                className="w-full bg-slate-950/70 border border-slate-800 rounded-xl py-2 px-3 text-sm text-slate-100 focus:border-fuchsia-500 outline-none"
                                            />
                                        </div>
                                        <div className="space-y-2">
                                            <label className="text-xs font-medium text-slate-400">Batch Size</label>
                                            <select
                                                value={config.batch_size}
                                                onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) })}
                                                className="w-full bg-slate-950/70 border border-slate-800 rounded-xl py-2 px-3 text-sm text-slate-300 focus:border-fuchsia-500 outline-none"
                                            >
                                                <option value="16">16</option>
                                                <option value="32">32</option>
                                                <option value="64">64</option>
                                                <option value="128">128</option>
                                            </select>
                                        </div>
                                    </div>
                                )}

                                {showAdvanced && (
                                    <div className="space-y-3 animate-in slide-in-from-top-2">
                                        <div className="space-y-2">
                                            <label className="text-xs font-medium text-slate-400">Learning Rate</label>
                                            <input
                                                type="range"
                                                min="0.0001"
                                                max="0.01"
                                                step="0.0001"
                                                value={config.learning_rate}
                                                onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) })}
                                                className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-fuchsia-500"
                                            />
                                            <div className="text-xs text-slate-500 text-center">{config.learning_rate.toFixed(4)}</div>
                                        </div>

                                        <div className="flex items-center gap-3">
                                            <label className="flex items-center gap-2 text-xs text-slate-400">
                                                <input
                                                    type="checkbox"
                                                    checked={config.use_feature_engineering}
                                                    onChange={(e) => setConfig({ ...config, use_feature_engineering: e.target.checked })}
                                                    className="rounded accent-fuchsia-500"
                                                />
                                                Feature Engineering
                                            </label>
                                            <label className="flex items-center gap-2 text-xs text-slate-400">
                                                <input
                                                    type="checkbox"
                                                    checked={config.use_cross_validation}
                                                    onChange={(e) => setConfig({ ...config, use_cross_validation: e.target.checked })}
                                                    className="rounded accent-fuchsia-500"
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
                                    <Filter className="text-fuchsia-500" size={14} />
                                    <span className="text-xs font-medium text-slate-400">Feature Engineering Pipeline</span>
                                </div>
                                <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                                    {featureOptions.map((feat) => (
                                        <div key={feat.id} className="p-2 bg-slate-900/50 rounded-lg border border-slate-800">
                                            <div className="text-[10px] font-medium text-fuchsia-400">{feat.label}</div>
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
                                    : 'bg-gradient-to-r from-fuchsia-600 to-pink-600 hover:from-fuchsia-500 hover:to-pink-500 text-white shadow-xl shadow-fuchsia-600/20'
                                }`}
                        >
                            {isTraining ? (
                                <>
                                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                                    <span>Training... {trainingProgress.toFixed(0)}%</span>
                                </>
                            ) : (
                                <>
                                    <Play size={18} />
                                    <span>Launch Training Session</span>
                                </>
                            )}
                        </button>

                        {/* Training Progress Bar */}
                        {isTraining && (
                            <div className="mt-4">
                                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-gradient-to-r from-fuchsia-500 to-pink-500 transition-all duration-300"
                                        style={{ width: `${trainingProgress}%` }}
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
                    <div className="bg-gradient-to-br from-slate-900/50 to-slate-950/50 border border-slate-800/50 rounded-2xl p-6">
                        <h4 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
                            <Info className="text-fuchsia-400" size={16} />
                            Model Information
                        </h4>
                        <div className="space-y-4">
                            <div className="p-3 bg-slate-900/30 rounded-lg">
                                <p className="text-[10px] font-medium text-fuchsia-400 uppercase mb-1">Prediction Target</p>
                                <p className="text-xs text-slate-300">Price movement probability {config.threshold * 100}% over next 5 intervals</p>
                            </div>
                            <div className="p-3 bg-slate-900/30 rounded-lg">
                                <p className="text-[10px] font-medium text-pink-400 uppercase mb-1">Data Requirements</p>
                                <p className="text-xs text-slate-300">Minimum 500 data points. Multi-year datasets recommended for robust models.</p>
                            </div>
                            <div className="p-3 bg-slate-900/30 rounded-lg">
                                <p className="text-[10px] font-medium text-blue-400 uppercase mb-1">Performance Metrics</p>
                                <p className="text-xs text-slate-300">Models evaluated on accuracy, F1-score, precision, recall, and overfitting score.</p>
                            </div>
                        </div>
                    </div>

                    {/* Quick Stats */}
                    {isLoadingModels ? (
                        <div className="bg-slate-900 p-6 rounded-2xl flex justify-center">
                            <Loader2 className="animate-spin text-slate-500" />
                        </div>
                    ) : (
                        <div className="bg-gradient-to-br from-slate-900/50 to-slate-950/50 border border-slate-800/50 rounded-2xl p-6">
                            <h4 className="text-sm font-bold text-slate-300 mb-4">Quick Stats</h4>
                            <div className="grid grid-cols-2 gap-3">
                                <div className="text-center p-3 bg-slate-900/30 rounded-lg">
                                    <div className="text-xl font-bold text-emerald-400">{models.length}</div>
                                    <div className="text-[10px] text-slate-500 mt-1">Models</div>
                                </div>
                                <div className="text-center p-3 bg-slate-900/30 rounded-lg">
                                    <div className="text-xl font-bold text-blue-400">{models.filter(m => m.status === 'deployed').length}</div>
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

            {/* 2. Training Results */}
            {showResults && selectedModel && (
                <div className="space-y-6 animate-in slide-in-from-bottom-4">
                    <div className="flex items-center justify-between">
                        <h3 className="text-lg font-bold text-slate-100">Training Results</h3>
                        <div className="flex gap-2">
                            <button
                                onClick={() => selectedModel && handleDelete(selectedModel.id)}
                                className="px-4 py-2 bg-red-500/10 hover:bg-red-500/20 rounded-lg text-red-500 text-sm font-medium flex items-center gap-2 transition-colors border border-red-500/20"
                            >
                                <Trash2 size={16} />
                                Delete Model
                            </button>
                            {selectedModel.status !== 'deployed' && (
                                <button
                                    onClick={() => handleDeploy(selectedModel.id)}
                                    disabled={isDeploying}
                                    className="px-4 py-2 bg-emerald-500/10 hover:bg-emerald-500/20 rounded-lg text-emerald-500 text-sm font-medium flex items-center gap-2 transition-colors border border-emerald-500/20"
                                >
                                    {isDeploying ? <Loader2 size={16} className="animate-spin" /> : <CheckCircle2 size={16} />}
                                    Deploy Strategy
                                </button>
                            )}
                            <button className="px-4 py-2 bg-gradient-to-r from-fuchsia-600 to-pink-600 hover:from-fuchsia-500 hover:to-pink-500 rounded-lg text-white text-sm font-medium flex items-center gap-2 transition-all shadow-lg shadow-fuchsia-600/20">
                                <Maximize2 size={16} />
                                Compare Models
                            </button>
                        </div>
                    </div>

                    {/* Performance Metrics */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 p-5 rounded-2xl">
                            <p className="text-xs font-medium text-slate-500 mb-2">Training Accuracy</p>
                            <div className="flex items-end gap-2">
                                <span className="text-2xl font-bold text-fuchsia-400">{(selectedModel.accuracy * 100).toFixed(1)}%</span>
                                <span className="text-xs text-slate-500 mb-1">±2.3%</span>
                            </div>
                        </div>
                        <div className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 p-5 rounded-2xl">
                            <p className="text-xs font-medium text-slate-500 mb-2">Test Accuracy</p>
                            <div className="flex items-end gap-2">
                                <span className="text-2xl font-bold text-pink-400">{(selectedModel.test_accuracy * 100).toFixed(1)}%</span>
                                <span className="text-xs text-slate-500 mb-1">±1.8%</span>
                            </div>
                        </div>
                        <div className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 p-5 rounded-2xl">
                            <p className="text-xs font-medium text-slate-500 mb-2">Overfit Score</p>
                            <div className="flex items-center justify-between">
                                <span className="text-2xl font-bold text-emerald-400">{(selectedModel.overfit_score * 100).toFixed(1)}%</span>
                                <CheckCircle2 className="text-emerald-500" size={20} />
                            </div>
                        </div>
                        <div className="bg-gradient-to-br from-slate-900/40 to-slate-950/40 border border-slate-800/50 p-5 rounded-2xl">
                            <p className="text-xs font-medium text-slate-500 mb-2">Features Used</p>
                            <div className="flex items-end gap-2">
                                <span className="text-2xl font-bold text-blue-400">{selectedModel.features}</span>
                                <span className="text-xs text-slate-500 mb-1">indicators</span>
                            </div>
                        </div>
                    </div>

                    {/* Charts Section */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Feature Importance */}
                        <div className="bg-gradient-to-br from-slate-900/50 to-slate-950/50 border border-slate-800/50 rounded-2xl p-6">
                            <div className="flex items-center justify-between mb-6">
                                <h4 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                                    <BarChart3 className="text-fuchsia-500" size={18} />
                                    Feature Importance
                                </h4>
                                <span className="text-xs text-slate-500">{selectedModel.feature_importance.length} features</span>
                            </div>
                            <div className="h-[300px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={selectedModel.feature_importance} layout="vertical">
                                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                                        <XAxis type="number" domain={[0, 0.35]} hide />
                                        <YAxis
                                            dataKey="feature"
                                            type="category"
                                            tick={{ fill: '#94a3b8', fontSize: 10 }}
                                            width={80}
                                        />
                                        <Tooltip
                                            formatter={(value) => [`${(Number(value) * 100).toFixed(1)}%`, 'Importance']}
                                            contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px' }}
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
                                                <linearGradient key={`gradient-${index}`} id={`gradient-${index}`} x1="0" y1="0" x2="1" y2="0">
                                                    <stop offset="0%" stopColor="#d946ef" stopOpacity={0.8} />
                                                    <stop offset="100%" stopColor="#ec4899" stopOpacity={0.8} />
                                                </linearGradient>
                                            ))}
                                        </defs>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Training Metrics Chart */}
                        <div className="bg-gradient-to-br from-slate-900/50 to-slate-950/50 border border-slate-800/50 rounded-2xl p-6">
                            <div className="flex items-center justify-between mb-6">
                                <h4 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                                    <LineChart className="text-cyan-500" size={18} />
                                    Training Progress
                                </h4>
                                <div className="flex gap-2">
                                    <button className="text-xs px-2 py-1 bg-slate-800 rounded text-slate-400">Loss</button>
                                    <button className="text-xs px-2 py-1 bg-blue-500/20 rounded text-blue-400">Accuracy</button>
                                </div>
                            </div>
                            <div className="h-[300px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <ComposedChart data={trainingMetrics}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                        <XAxis dataKey="epoch" stroke="#64748b" fontSize={10} />
                                        <YAxis stroke="#64748b" fontSize={10} />
                                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }} />
                                        <Legend />
                                        <Area
                                            type="monotone"
                                            dataKey="loss"
                                            fill="#ef4444"
                                            fillOpacity={0.1}
                                            stroke="#ef4444"
                                            strokeWidth={1}
                                            name="Training Loss"
                                        />
                                        <Area
                                            type="monotone"
                                            dataKey="val_loss"
                                            fill="#64748b"
                                            fillOpacity={0.1}
                                            stroke="#64748b"
                                            strokeDasharray="3 3"
                                            strokeWidth={1}
                                            name="Val Loss"
                                        />
                                        <Line
                                            type="monotone"
                                            dataKey="accuracy"
                                            stroke="#3b82f6"
                                            strokeWidth={2}
                                            dot={false}
                                            name="Accuracy"
                                        />
                                    </ComposedChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default MLStudio;
