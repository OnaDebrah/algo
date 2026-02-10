'use client'

import React, {useState} from 'react';
import {Activity, AlertCircle, DollarSign, Plus, Shield, Target, Trash2, TrendingUp, Upload, X} from 'lucide-react';
import {BacktestDataToPublish, PublishData} from "@/types/publish";

interface PublishModalProps {
    backtest: BacktestDataToPublish;
    onClose: () => void;
    onPublish: (data: PublishData) => void;
}

const PublishModal = ({ backtest, onClose, onPublish }: PublishModalProps) => {
    const [formData, setFormData] = useState<PublishData>({
        name: backtest.name || backtest.strategy_key,
        description: '',
        category: 'momentum',
        tags: [],
        complexity: 'intermediate',
        price: 0,
        pros: [''],
        cons: [''],
        risk_level: 'medium',
        recommended_capital: 10000
    });

    const [currentTag, setCurrentTag] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [errors, setErrors] = useState<Record<string, string>>({});

    const categories = [
        { value: 'momentum', label: 'Momentum' },
        { value: 'mean_reversion', label: 'Mean Reversion' },
        { value: 'arbitrage', label: 'Arbitrage' },
        { value: 'breakout', label: 'Breakout' },
        { value: 'trend_following', label: 'Trend Following' },
        { value: 'ml', label: 'Machine Learning' },
        { value: 'options', label: 'Options' },
        { value: 'other', label: 'Other' }
    ];

    const validateForm = () => {
        const newErrors: Record<string, string> = {};

        if (!formData.name.trim()) {
            newErrors.name = 'Strategy name is required';
        }

        if (!formData.description.trim() || formData.description.length < 50) {
            newErrors.description = 'Description must be at least 50 characters';
        }

        if (formData.pros.filter(p => p.trim()).length === 0) {
            newErrors.pros = 'Add at least one strength';
        }

        if (formData.cons.filter(c => c.trim()).length === 0) {
            newErrors.cons = 'Add at least one consideration';
        }

        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleSubmit = async () => {
        if (!validateForm()) {
            return;
        }

        setIsSubmitting(true);
        try {
            // Filter out empty pros/cons
            const cleanedData = {
                ...formData,
                pros: formData.pros.filter(p => p.trim()),
                cons: formData.cons.filter(c => c.trim())
            };

            onPublish(cleanedData);
        } finally {
            setIsSubmitting(false);
        }
    };

    const addTag = () => {
        if (currentTag.trim() && !formData.tags.includes(currentTag.trim())) {
            setFormData({
                ...formData,
                tags: [...formData.tags, currentTag.trim()]
            });
            setCurrentTag('');
        }
    };

    const removeTag = (tag: string) => {
        setFormData({
            ...formData,
            tags: formData.tags.filter(t => t !== tag)
        });
    };

    const addPro = () => {
        setFormData({
            ...formData,
            pros: [...formData.pros, '']
        });
    };

    const removePro = (index: number) => {
        if (formData.pros.length > 1) {
            setFormData({
                ...formData,
                pros: formData.pros.filter((_, i) => i !== index)
            });
        }
    };

    const updatePro = (index: number, value: string) => {
        const newPros = [...formData.pros];
        newPros[index] = value;
        setFormData({ ...formData, pros: newPros });
    };

    const addCon = () => {
        setFormData({
            ...formData,
            cons: [...formData.cons, '']
        });
    };

    const removeCon = (index: number) => {
        if (formData.cons.length > 1) {
            setFormData({
                ...formData,
                cons: formData.cons.filter((_, i) => i !== index)
            });
        }
    };

    const updateCon = (index: number, value: string) => {
        const newCons = [...formData.cons];
        newCons[index] = value;
        setFormData({ ...formData, cons: newCons });
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/90 backdrop-blur-md p-4 overflow-y-auto">
            <div className="bg-slate-900 border border-slate-700/50 w-full max-w-4xl rounded-3xl shadow-2xl relative my-8">
                {/* Close Button */}
                <button
                    onClick={onClose}
                    className="absolute top-6 right-6 p-2 bg-slate-800 hover:bg-slate-700 text-slate-400 rounded-xl transition-all z-10"
                >
                    <X size={24} />
                </button>

                {/* Header */}
                <div className="p-8 border-b border-slate-700/30">
                    <div className="flex items-center gap-4 mb-4">
                        <div className="p-3 bg-violet-500/10 rounded-2xl">
                            <Upload className="text-violet-400" size={32} />
                        </div>
                        <div>
                            <h2 className="text-3xl font-bold text-slate-100">Publish to Marketplace</h2>
                            <p className="text-slate-400 mt-1">
                                Share your strategy with the community
                            </p>
                        </div>
                    </div>

                    {/* Performance Preview */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-6">
                        <div className="bg-slate-800/40 border border-slate-700/30 p-3 rounded-xl">
                            <div className="flex items-center gap-2 mb-1">
                                <TrendingUp size={14} className="text-emerald-400" />
                                <span className="text-xs text-slate-500">Return</span>
                            </div>
                            <p className="text-lg font-bold text-emerald-400">
                                +{backtest.total_return_pct.toFixed(2)}%
                            </p>
                        </div>
                        <div className="bg-slate-800/40 border border-slate-700/30 p-3 rounded-xl">
                            <div className="flex items-center gap-2 mb-1">
                                <Activity size={14} className="text-blue-400" />
                                <span className="text-xs text-slate-500">Sharpe</span>
                            </div>
                            <p className="text-lg font-bold text-blue-400">
                                {backtest.sharpe_ratio.toFixed(2)}
                            </p>
                        </div>
                        <div className="bg-slate-800/40 border border-slate-700/30 p-3 rounded-xl">
                            <div className="flex items-center gap-2 mb-1">
                                <Shield size={14} className="text-red-400" />
                                <span className="text-xs text-slate-500">Drawdown</span>
                            </div>
                            <p className="text-lg font-bold text-red-400">
                                {backtest.max_drawdown.toFixed(2)}%
                            </p>
                        </div>
                        <div className="bg-slate-800/40 border border-slate-700/30 p-3 rounded-xl">
                            <div className="flex items-center gap-2 mb-1">
                                <Target size={14} className="text-violet-400" />
                                <span className="text-xs text-slate-500">Win Rate</span>
                            </div>
                            <p className="text-lg font-bold text-violet-400">
                                {backtest.win_rate.toFixed(1)}%
                            </p>
                        </div>
                    </div>
                </div>

                {/* Form */}
                <div className="p-8 space-y-6 max-h-[60vh] overflow-y-auto">
                    {/* Strategy Name */}
                    <div>
                        <label className="block text-sm font-bold text-slate-300 mb-2">
                            Strategy Name *
                        </label>
                        <input
                            type="text"
                            value={formData.name}
                            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                            className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-slate-200 focus:border-violet-500 outline-none"
                            placeholder="e.g., Momentum Breakout Strategy"
                        />
                        {errors.name && (
                            <p className="text-red-400 text-xs mt-1 flex items-center gap-1">
                                <AlertCircle size={12} /> {errors.name}
                            </p>
                        )}
                    </div>

                    {/* Description */}
                    <div>
                        <label className="block text-sm font-bold text-slate-300 mb-2">
                            Description * (min 50 characters)
                        </label>
                        <textarea
                            value={formData.description}
                            onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                            rows={4}
                            className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-slate-200 focus:border-violet-500 outline-none resize-none"
                            placeholder="Describe your strategy... What makes it unique? What markets does it work best in? What are the key indicators?"
                        />
                        <div className="flex justify-between mt-1">
                            {errors.description && (
                                <p className="text-red-400 text-xs flex items-center gap-1">
                                    <AlertCircle size={12} /> {errors.description}
                                </p>
                            )}
                            <p className="text-xs text-slate-500 ml-auto">
                                {formData.description.length} / 50 characters
                            </p>
                        </div>
                    </div>

                    {/* Category & Complexity */}
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm font-bold text-slate-300 mb-2">
                                Category
                            </label>
                            <select
                                value={formData.category}
                                onChange={(e) => setFormData({ ...formData, category: e.target.value })}
                                className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-slate-200 focus:border-violet-500 outline-none"
                            >
                                {categories.map(cat => (
                                    <option key={cat.value} value={cat.value}>
                                        {cat.label}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <div>
                            <label className="block text-sm font-bold text-slate-300 mb-2">
                                Complexity
                            </label>
                            <select
                                value={formData.complexity}
                                onChange={(e) => setFormData({ ...formData, complexity: e.target.value as any })}
                                className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-slate-200 focus:border-violet-500 outline-none"
                            >
                                <option value="beginner">Beginner</option>
                                <option value="intermediate">Intermediate</option>
                                <option value="advanced">Advanced</option>
                            </select>
                        </div>
                    </div>

                    {/* Tags */}
                    <div>
                        <label className="block text-sm font-bold text-slate-300 mb-2">
                            Tags
                        </label>
                        <div className="flex gap-2 mb-2">
                            <input
                                type="text"
                                value={currentTag}
                                onChange={(e) => setCurrentTag(e.target.value)}
                                onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), addTag())}
                                className="flex-1 px-4 py-2 bg-slate-800 border border-slate-700 rounded-xl text-slate-200 focus:border-violet-500 outline-none"
                                placeholder="Add tags (e.g., stocks, crypto, swing-trading)"
                            />
                            <button
                                onClick={addTag}
                                className="px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white rounded-xl font-semibold transition-all"
                            >
                                <Plus size={20} />
                            </button>
                        </div>
                        <div className="flex flex-wrap gap-2">
                            {formData.tags.map(tag => (
                                <span
                                    key={tag}
                                    className="px-3 py-1 bg-violet-500/20 text-violet-400 rounded-full text-sm flex items-center gap-2"
                                >
                                    {tag}
                                    <button
                                        onClick={() => removeTag(tag)}
                                        className="hover:text-violet-300"
                                    >
                                        <X size={14} />
                                    </button>
                                </span>
                            ))}
                        </div>
                    </div>

                    {/* Price & Risk Level */}
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm font-bold text-slate-300 mb-2">
                                Price (USD)
                            </label>
                            <div className="relative">
                                <DollarSign className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
                                <input
                                    type="number"
                                    min="0"
                                    step="1"
                                    value={formData.price}
                                    onChange={(e) => setFormData({ ...formData, price: parseFloat(e.target.value) || 0 })}
                                    className="w-full pl-10 pr-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-slate-200 focus:border-violet-500 outline-none"
                                    placeholder="0 for free"
                                />
                            </div>
                            <p className="text-xs text-slate-500 mt-1">
                                Set to $0 to make it free
                            </p>
                        </div>

                        <div>
                            <label className="block text-sm font-bold text-slate-300 mb-2">
                                Risk Level
                            </label>
                            <select
                                value={formData.risk_level}
                                onChange={(e) => setFormData({ ...formData, risk_level: e.target.value as any })}
                                className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-slate-200 focus:border-violet-500 outline-none"
                            >
                                <option value="low">Low Risk</option>
                                <option value="medium">Medium Risk</option>
                                <option value="high">High Risk</option>
                            </select>
                        </div>
                    </div>

                    {/* Recommended Capital */}
                    <div>
                        <label className="block text-sm font-bold text-slate-300 mb-2">
                            Recommended Capital (USD)
                        </label>
                        <input
                            type="number"
                            min="1000"
                            step="1000"
                            value={formData.recommended_capital}
                            onChange={(e) => setFormData({ ...formData, recommended_capital: parseFloat(e.target.value) || 10000 })}
                            className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-slate-200 focus:border-violet-500 outline-none"
                        />
                        <p className="text-xs text-slate-500 mt-1">
                            Minimum capital recommended for this strategy
                        </p>
                    </div>

                    {/* Strengths */}
                    <div>
                        <label className="block text-sm font-bold text-slate-300 mb-2">
                            Strengths *
                        </label>
                        <div className="space-y-2">
                            {formData.pros.map((pro, index) => (
                                <div key={index} className="flex gap-2">
                                    <input
                                        type="text"
                                        value={pro}
                                        onChange={(e) => updatePro(index, e.target.value)}
                                        className="flex-1 px-4 py-2 bg-slate-800 border border-slate-700 rounded-xl text-slate-200 focus:border-emerald-500 outline-none"
                                        placeholder="e.g., High win rate in trending markets"
                                    />
                                    {formData.pros.length > 1 && (
                                        <button
                                            onClick={() => removePro(index)}
                                            className="px-3 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-xl transition-all"
                                        >
                                            <Trash2 size={18} />
                                        </button>
                                    )}
                                </div>
                            ))}
                            <button
                                onClick={addPro}
                                className="flex items-center gap-2 px-4 py-2 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 rounded-xl font-semibold text-sm transition-all"
                            >
                                <Plus size={16} /> Add Strength
                            </button>
                        </div>
                        {errors.pros && (
                            <p className="text-red-400 text-xs mt-1 flex items-center gap-1">
                                <AlertCircle size={12} /> {errors.pros}
                            </p>
                        )}
                    </div>

                    {/* Considerations */}
                    <div>
                        <label className="block text-sm font-bold text-slate-300 mb-2">
                            Considerations *
                        </label>
                        <div className="space-y-2">
                            {formData.cons.map((con, index) => (
                                <div key={index} className="flex gap-2">
                                    <input
                                        type="text"
                                        value={con}
                                        onChange={(e) => updateCon(index, e.target.value)}
                                        className="flex-1 px-4 py-2 bg-slate-800 border border-slate-700 rounded-xl text-slate-200 focus:border-amber-500 outline-none"
                                        placeholder="e.g., Requires monitoring during market hours"
                                    />
                                    {formData.cons.length > 1 && (
                                        <button
                                            onClick={() => removeCon(index)}
                                            className="px-3 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-xl transition-all"
                                        >
                                            <Trash2 size={18} />
                                        </button>
                                    )}
                                </div>
                            ))}
                            <button
                                onClick={addCon}
                                className="flex items-center gap-2 px-4 py-2 bg-amber-500/20 hover:bg-amber-500/30 text-amber-400 rounded-xl font-semibold text-sm transition-all"
                            >
                                <Plus size={16} /> Add Consideration
                            </button>
                        </div>
                        {errors.cons && (
                            <p className="text-red-400 text-xs mt-1 flex items-center gap-1">
                                <AlertCircle size={12} /> {errors.cons}
                            </p>
                        )}
                    </div>
                </div>

                {/* Footer */}
                <div className="p-8 border-t border-slate-700/30 flex items-center justify-between">
                    <p className="text-sm text-slate-400">
                        * Required fields
                    </p>
                    <div className="flex gap-3">
                        <button
                            onClick={onClose}
                            className="px-6 py-3 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl font-semibold transition-all"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleSubmit}
                            disabled={isSubmitting}
                            className="px-6 py-3 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white rounded-xl font-semibold transition-all shadow-xl shadow-violet-500/20 disabled:opacity-50 flex items-center gap-2"
                        >
                            {isSubmitting ? (
                                <>
                                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                    Publishing...
                                </>
                            ) : (
                                <>
                                    <Upload size={18} />
                                    Publish to Marketplace
                                </>
                            )}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PublishModal;
