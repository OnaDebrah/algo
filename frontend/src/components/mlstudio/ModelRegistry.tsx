import {BarChart} from "recharts";
import {
    Activity,
    Calendar,
    CheckCircle,
    CheckCircle2,
    Download,
    Eye,
    EyeOff,
    Loader2,
    RefreshCw,
    Shield,
    Trash2
} from "lucide-react";
import {MLModel} from "@/types/all_types";
import {JSX} from "react";


interface ModelRegistryProps {
    setFilterStatus: (status: 'all' | 'trained' | 'deployed') => void;
    models: MLModel[];
    filterStatus: 'all' | 'trained' | 'deployed';
    trainedModels: MLModel[];
    deployedModels: MLModel[];
    showInactive: boolean;
    setShowInactive: (show: boolean) => void;
    sortBy: 'date' | 'accuracy' | 'training_time';
    setSortBy: (sort: 'date' | 'accuracy' | 'training_time') => void;
    sortOrder: 'asc' | 'desc';
    setSortOrder: (order: 'asc' | 'desc') => void;
    viewMode: 'grid' | 'list';
    setViewMode: (mode: 'grid' | 'list') => void;
    isLoadingModels: boolean;
    filteredAndSortedModels: MLModel[];
    setActiveTab: (tab: 'train' | 'registry' | 'deployed') => void;
    setSelectedModel: (model: MLModel) => void;
    setActiveModelId: (id: string) => void;
    setShowResults: (show: boolean) => void;
    selectedModel: MLModel | null;
    renderModelIcon: (type: MLModel['type']) => JSX.Element;
    getStatusColor: (status: MLModel['status']) => string;
    isDeploying: boolean;
    handleRetrain: (model: MLModel) => void;
    handleDeploy: (modelId: string) => void;
    handleUndeploy: (modelId: string) => void;
    handleDelete: (modelId: string) => void;
    handleToggleActive: (modelId: string, isActive: boolean) => void;
    handleExportModel: (modelId: string) => void;
}

const ModelRegistry: React.FC<ModelRegistryProps> = ({
                                                         setFilterStatus,
                                                         models,
                                                         filterStatus,
                                                         trainedModels,
                                                         deployedModels,
                                                         showInactive,
                                                         setShowInactive,
                                                         sortBy,
                                                         setSortBy,
                                                         sortOrder,
                                                         setSortOrder,
                                                         viewMode,
                                                         setViewMode,
                                                         isLoadingModels,
                                                         filteredAndSortedModels,
                                                         setActiveTab,
                                                         setSelectedModel,
                                                         setActiveModelId,
                                                         setShowResults,
                                                         selectedModel,
                                                         renderModelIcon,
                                                         getStatusColor,
                                                         isDeploying,
                                                         handleRetrain,
                                                         handleDeploy,
                                                         handleUndeploy,
                                                         handleDelete,
                                                         handleToggleActive,
                                                         handleExportModel,
                                                     }: ModelRegistryProps) => {

    const ListIcon = (props: any) => (
        <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="8" y1="6" x2="21" y2="6"/>
            <line x1="8" y1="12" x2="21" y2="12"/>
            <line x1="8" y1="18" x2="21" y2="18"/>
            <line x1="3" y1="6" x2="3.01" y2="6"/>
            <line x1="3" y1="12" x2="3.01" y2="12"/>
            <line x1="3" y1="18" x2="3.01" y2="18"/>
        </svg>
    );

    return (
        <div className="space-y-6">
            {/* Filters and Controls */}
            <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2 bg-slate-900/50 rounded-lg p-1">
                        <button
                            onClick={() => setFilterStatus('all')}
                            className={`px-3 py-1 rounded-md text-sm transition-colors ${filterStatus === 'all' ? 'bg-slate-800 text-slate-100' : 'text-slate-500 hover:text-slate-300'}`}
                        >
                            All ({models.length})
                        </button>
                        <button
                            onClick={() => setFilterStatus('trained')}
                            className={`px-3 py-1 rounded-md text-sm transition-colors ${filterStatus === 'trained' ? 'bg-blue-500/20 text-blue-400' : 'text-slate-500 hover:text-slate-300'}`}
                        >
                            Trained ({trainedModels.length})
                        </button>
                        <button
                            onClick={() => setFilterStatus('deployed')}
                            className={`px-3 py-1 rounded-md text-sm transition-colors ${filterStatus === 'deployed' ? 'bg-emerald-500/20 text-emerald-400' : 'text-slate-500 hover:text-slate-300'}`}
                        >
                            Deployed ({deployedModels.length})
                        </button>
                    </div>

                    <label className="flex items-center gap-2 text-sm text-slate-500">
                        <input
                            type="checkbox"
                            checked={showInactive}
                            onChange={(e) => setShowInactive(e.target.checked)}
                            className="rounded accent-fuchsia-500"
                        />
                        Show Inactive
                    </label>
                </div>

                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2">
                        <span className="text-sm text-slate-500">Sort by:</span>
                        <select
                            value={sortBy}
                            onChange={(e) => setSortBy(e.target.value as any)}
                            className="bg-slate-900 border border-slate-800 rounded-lg px-3 py-1 text-sm text-slate-300 focus:border-fuchsia-500 outline-none"
                        >
                            <option value="date">Date Created</option>
                            <option value="accuracy">Accuracy</option>
                            <option value="training_time">Training Time</option>
                        </select>
                        <button
                            onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                            className="p-1 hover:bg-slate-800 rounded-lg transition-colors"
                        >
                            {sortOrder === 'asc' ? '↑' : '↓'}
                        </button>
                    </div>

                    <div className="flex items-center gap-1 bg-slate-900/50 rounded-lg p-1">
                        <button
                            onClick={() => setViewMode('grid')}
                            className={`p-2 rounded-md ${viewMode === 'grid' ? 'bg-slate-800' : 'hover:bg-slate-800'}`}
                        >
                            <BarChart className="text-slate-400"/>
                        </button>
                        <button
                            onClick={() => setViewMode('list')}
                            className={`p-2 rounded-md ${viewMode === 'list' ? 'bg-slate-800' : 'hover:bg-slate-800'}`}
                        >
                            <ListIcon size={16} className="text-slate-400"/>
                        </button>
                    </div>
                </div>
            </div>

            {/* Models Grid/List */}
            {isLoadingModels ? (
                <div className="flex justify-center py-12">
                    <Loader2 className="animate-spin text-slate-500" size={32}/>
                </div>
            ) : filteredAndSortedModels.length === 0 ? (
                <div className="text-center py-12">
                    <div className="text-slate-500 mb-3">No models found</div>
                    <button
                        onClick={() => setActiveTab('train')}
                        className="px-4 py-2 bg-gradient-to-r from-fuchsia-600 to-violet-600 rounded-lg text-white text-sm font-medium hover:from-fuchsia-500 hover:to-violet-500 transition-all"
                    >
                        Train Your First Model
                    </button>
                </div>
            ) : (
                <div
                    className={viewMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6' : 'space-y-4'}>
                    {filteredAndSortedModels.map((model) => (
                        <div
                            key={model.id}
                            className={`bg-gradient-to-br from-slate-900/50 to-slate-950/50 border ${selectedModel?.id === model.id ? 'border-fuchsia-500/50' : 'border-slate-800/50'} rounded-2xl p-5 hover:border-slate-700/50 transition-all cursor-pointer ${!model.is_active ? 'opacity-60' : ''}`}
                            onClick={() => {
                                setSelectedModel(model);
                                setActiveModelId(model.id);
                                setShowResults(true);
                            }}
                        >
                            {/* Model Header */}
                            <div className="flex items-start justify-between mb-4">
                                <div className="flex items-center gap-3">
                                    <div
                                        className={`p-2 rounded-lg ${model.status === 'deployed' ? 'bg-emerald-500/20' : 'bg-blue-500/20'}`}>
                                        {renderModelIcon(model.type)}
                                    </div>
                                    <div>
                                        <div className="font-medium text-slate-100">{model.name}</div>
                                        <div className="text-xs text-slate-500">{model.symbol} • {model.type}</div>
                                    </div>
                                </div>

                                <div
                                    className={`text-xs px-2 py-1 rounded-full border ${getStatusColor(model.status)}`}>
                                    {model.status}
                                </div>
                            </div>

                            {/* Model Stats */}
                            <div className="grid grid-cols-3 gap-3 mb-4">
                                <div className="text-center">
                                    <div
                                        className="text-lg font-bold text-fuchsia-400">{(model.accuracy * 100).toFixed(1)}%
                                    </div>
                                    <div className="text-xs text-slate-500">Accuracy</div>
                                </div>
                                <div className="text-center">
                                    <div className="text-lg font-bold text-violet-400">{model.features}</div>
                                    <div className="text-xs text-slate-500">Features</div>
                                </div>
                                <div className="text-center">
                                    <div
                                        className="text-lg font-bold text-cyan-400">{(model.training_time / 60).toFixed(1)}h
                                    </div>
                                    <div className="text-xs text-slate-500">Training</div>
                                </div>
                            </div>

                            {/* Model Metadata */}
                            <div className="text-xs text-slate-500 space-y-1 mb-4">
                                <div className="flex items-center gap-2">
                                    <Calendar size={12}/>
                                    Created: {model.created_at}
                                </div>
                                {model.deployed_at && (
                                    <div className="flex items-center gap-2">
                                        <CheckCircle size={12}/>
                                        Deployed: {model.deployed_at}
                                    </div>
                                )}
                                {model.last_used && (
                                    <div className="flex items-center gap-2">
                                        <Activity size={12}/>
                                        Last used: {model.last_used}
                                    </div>
                                )}
                            </div>

                            {/* Action Buttons */}
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        handleRetrain(model);
                                    }}
                                    className="flex-1 px-3 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 text-xs font-medium flex items-center justify-center gap-1 transition-colors"
                                >
                                    <RefreshCw size={12}/>
                                    Retrain
                                </button>

                                {model.status === 'trained' ? (
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            handleDeploy(model.id);
                                        }}
                                        disabled={isDeploying}
                                        className="flex-1 px-3 py-2 bg-emerald-500/10 hover:bg-emerald-500/20 rounded-lg text-emerald-500 text-xs font-medium flex items-center justify-center gap-1 transition-colors border border-emerald-500/20"
                                    >
                                        {isDeploying ? <Loader2 size={12} className="animate-spin"/> :
                                            <CheckCircle2 size={12}/>}
                                        Deploy
                                    </button>
                                ) : (
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            handleUndeploy(model.id);
                                        }}
                                        className="flex-1 px-3 py-2 bg-amber-500/10 hover:bg-amber-500/20 rounded-lg text-amber-500 text-xs font-medium flex items-center justify-center gap-1 transition-colors border border-amber-500/20"
                                    >
                                        <Shield size={12}/>
                                        Undeploy
                                    </button>
                                )}

                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        handleToggleActive(model.id, !model.is_active);
                                    }}
                                    className="p-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-slate-200 transition-colors"
                                    title={model.is_active ? 'Deactivate' : 'Activate'}
                                >
                                    {model.is_active ? <EyeOff size={14}/> : <Eye size={14}/>}
                                </button>

                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        handleExportModel(model.id);
                                    }}
                                    className="p-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-slate-200 transition-colors"
                                    title="Export"
                                >
                                    <Download size={14}/>
                                </button>

                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        handleDelete(model.id);
                                    }}
                                    className="p-2 bg-red-500/10 hover:bg-red-500/20 rounded-lg text-red-500 hover:text-red-400 transition-colors"
                                    title="Delete"
                                >
                                    <Trash2 size={14}/>
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
};

export default ModelRegistry;
