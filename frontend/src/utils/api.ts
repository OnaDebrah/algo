// api/index.ts
import axios from 'axios';
import {
    // Auth
    LoginResponse,
    User,
    UserCreate,
    UserLogin,

    // Backtest
    SingleBacktestRequest,
    SingleBacktestResponse,
    MultiAssetBacktestRequest,
    MultiAssetBacktestResponse,
    OptionsBacktestRequest,
    OptionsBacktestResponse,
    BacktestHistoryItem,

    // Portfolio
    Portfolio,
    PortfolioCreate,
    PortfolioUpdate,
    PortfolioMetrics,
    Position,
    PortfolioTrade,

    // Market Data
    QuoteData,
    HistoricalDataPoint,
    OptionsChain,
    SearchResult,

    // Strategy
    StrategyInfo,
    StrategyConfig,

    // Analyst & Advisor
    AnalystReport,
    GuideRequest,
    GuideResponse,

    // Alerts
    EmailAlertRequest,
    SMSAlertRequest,
    AlertTestResponse,

    // Live Trading
    LiveStatus,
    ExecutionOrder,

    // Marketplace
    StrategyListing,
    StrategyListingDetailed,
    StrategyPublishRequest,
    StrategyReviewSchema,

    // ML Studio
    MLModel,
    TrainingConfig,
    DeployedMLModel,

    // Options
    ChainRequest,
    ChainResponse,
    GreeksRequest,
    GreeksResponse,
    StrategyAnalysisRequest,
    StrategyAnalysisResponse,
    StrategyComparisonRequest,
    StrategyComparisonResponse,
    ProbabilityRequest,
    ProbabilityResponse,
    StrikeOptimizerRequest,
    StrikeOptimizerResponse,
    PortfolioStatsRequest,
    PortfolioStatsResponse,
    RiskMetricsRequest,
    RiskMetricsResponse,
    MonteCarloRequest,
    MonteCarloResponse,

    // Market Regime
    CurrentRegimeResponse,
    AllocationResponse,
    RegimeStrengthResponse,
    TransitionResponse,
    FeaturesResponse,

    // Settings
    UserSettings,
    SettingsUpdate,

    // HTTP Error
    HTTPValidationError, RegimeData, RegimeHistoryResponse, PairsValidationRequest, PairsValidationResponse,
    LiveOrderPlacement, LiveOrderUpdate, MarketplaceFilterParams, AlertPreferences, Alert, DeploymentConfig,
    BrokerConnectionResponse, MLModelStatusRequest, ModelPrediction, BrokerSettings
} from '@/types/all_types';
import {
    BaseOptimizationRequest, BlackLittermanRequest, BlackLittermanResponse,
    CompareStrategiesResponse, EfficientFrontierRequest, EfficientFrontierResponse,
    OptimiseBacktestResponse, OptimizationResponse,
    PortfolioBacktestRequest, SharpeOptimizationRequest, TargetReturnRequest
} from "@/types/optimise";
import {
    AccountResponse,
    ConnectRequest,
    ControlResponse, LiveStrategy,
    StrategyDetailsResponse,
    StrategyUpdateRequest,
    UpdateResponse
} from "@/types/live";
import { ActivityResponse } from "@/types/social";
import {AnalystReportParams} from "@/types/analyst";

// Use environment variable or default to localhost
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const IS_BROWSER = typeof window !== 'undefined';

const client = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
    withCredentials: true,
});

// Request interceptor to add auth token
client.interceptors.request.use(
    (config) => {
        if (IS_BROWSER) {
            const token = localStorage.getItem('access_token');
            if (token) {
                config.headers.Authorization = `Bearer ${token}`;
            }
        }
        return config;
    },
    (error) => Promise.reject(error)
);

// Response interceptor to handle errors
client.interceptors.response.use(
    (response) => response.data,
    (error) => {
        if (error.response?.status === 401) {
            // Handle unauthorized access
            if (IS_BROWSER) {
                localStorage.removeItem('access_token');
                localStorage.removeItem('refresh_token');
                localStorage.removeItem('user');
                // Redirect to login if not already there
                if (!window.location.pathname.includes('/login') && !window.location.pathname.includes('/register')) {
                    window.location.href = '/login';
                }
            }
        }

        if (error.response?.data) {
            return Promise.reject({
                status: error.response.status,
                data: error.response.data,
                message: error.response.data.detail || error.message
            });
        }

        return Promise.reject(error);
    }
);

// ==================== AUTHENTICATION ====================
export const auth = {
    register: (data: UserCreate) => client.post<LoginResponse>('/auth/register', data),
    login: (data: UserLogin) => client.post<LoginResponse>('/auth/login', data),
    logout: () => client.post('/auth/logout'),
    getMe: () => client.get<User>('/auth/me'),

    // Token refresh (if implemented)
    refreshToken: (refreshToken: string) =>
        client.post<{ access_token: string }>('/auth/refresh', { refresh_token: refreshToken }),
};

// ==================== BACKTEST ====================
export const backtest = {
    // Single asset backtest
    runSingle: async (request: SingleBacktestRequest): Promise<SingleBacktestResponse> => {
        return client.post<SingleBacktestResponse>('/backtest/single', request);
    },
    runMulti: async (request: MultiAssetBacktestRequest): Promise<MultiAssetBacktestResponse> => {
        return client.post<MultiAssetBacktestResponse>('/backtest/multi', request);
    },

    runValidateKalman: async (request: PairsValidationRequest): Promise<PairsValidationResponse> => {
        return client.post<PairsValidationResponse>('/backtest/validated', request);
    },

    getSuggestion: async (request: PairsValidationRequest): Promise<PairsValidationResponse> => {
        return client.post<PairsValidationResponse>('/backtest/suggested', request);
    },

    runOptions: (request: OptionsBacktestRequest) =>
        client.post<OptionsBacktestResponse>('/backtest/options', request),

    getHistory: (params?: {
        limit?: number;
        offset?: number;
        backtest_type?: 'single' | 'multi' | 'options';
        status?: 'pending' | 'running' | 'completed' | 'failed';
        symbol?: string;
    }) => client.get<BacktestHistoryItem[]>('/backtest/history', { params }),

    getHistoryCount: (params?: { backtest_type?: string; status?: string }) =>
        client.get<{ count: number }>('/backtest/history/count', { params }),

    getDetails: (backtest_id: number) =>
        client.get<BacktestHistoryItem>(`/backtest/history/${backtest_id}`),

    deleteBacktest: (backtest_id: number) =>
        client.delete(`/backtest/history/${backtest_id}`),

    updateBacktestName: (backtest_id: number, name: string) =>
        client.put(`/backtest/history/${backtest_id}/name`, null, { params: { name } }),

    getStatsSummary: () =>
        client.get<Record<string, unknown>>('/backtest/history/stats/summary'),

    /**
     * Run Bayesian optimization on strategy parameters
     */
    bayesian: (request: any) =>
        client.post<any>('/optimise/bayesian', request),
};

// ==================== PORTFOLIO ====================
export const portfolio = {
    // Portfolio CRUD
    list: () => client.get<Portfolio[]>('/portfolio/'),

    create: (data: PortfolioCreate) =>
        client.post<Portfolio>('/portfolio/', data),

    get: (portfolio_id: number) =>
        client.get<Portfolio>(`/portfolio/${portfolio_id}`),

    update: (portfolio_id: number, data: PortfolioUpdate) =>
        client.put<Portfolio>(`/portfolio/${portfolio_id}`, data),

    delete: (portfolio_id: number) =>
        client.delete(`/portfolio/${portfolio_id}`),

    getMetrics: (portfolio_id: number) =>
        client.get<PortfolioMetrics>(`/portfolio/${portfolio_id}/metrics`),

    getPositions: (portfolio_id: number) =>
        client.get<Position[]>(`/portfolio/${portfolio_id}/positions`),

    getTrades: (portfolio_id: number, params?: { limit?: number; offset?: number }) =>
        client.get<PortfolioTrade[]>(`/portfolio/${portfolio_id}/trades`, { params }),
};

// ==================== MARKET DATA ====================
export interface HistoricalDataParams {
    period?: string;
    interval?: string;
    start?: string;
    end?: string;
    use_cache?: boolean;
}

export interface NewsParams {
    limit?: number;
}

export interface SearchParams {
    q: string;
    limit?: number;
}

export const market = {
    getQuote: (symbol: string, use_cache: boolean = true) =>
        client.get<QuoteData>(`/market/quote/${symbol}`, { params: { use_cache } }),

    getQuotes: (symbols: string[], use_cache: boolean = true) =>
        client.post<QuoteData[]>('/market/quotes', symbols, { params: { use_cache } }),

    getHistorical: (symbol: string, params?: HistoricalDataParams) =>
        client.get<HistoricalDataPoint[]>(`/market/historical/${symbol}`, { params }),

    getOptionChain: (symbol: string, expiration?: string) =>
        client.get<OptionsChain>(`/market/options/${symbol}`, { params: { expiration } }),

    getFundamentals: (symbol: string) =>
        client.get<Record<string, any>>(`/market/fundamentals/${symbol}`),

    getNews: (symbol: string, params?: NewsParams) =>
        client.get<Record<string, any>[]>(`/market/news/${symbol}`, { params }),

    search: (query: string, limit: number = 10) =>
        client.get<SearchResult[]>(`/market/search`, { params: { q: query, limit } }),

    validateSymbol: (symbol: string) =>
        client.get<{ valid: boolean; symbol: string; message?: string }>(`/market/validate/${symbol}`),

    getMarketStatus: () =>
        client.get<{ is_open: boolean; next_open?: string; next_close?: string }>(`/market/status`),

    getCacheStats: () =>
        client.get<Record<string, any>>(`/market/cache/stats`),

    clearCache: (data_type?: string) =>
        client.post<{ success: boolean; message: string }>(`/market/cache/clear`, null, { params: { data_type } }),

    cleanupCache: () =>
        client.post<{ success: boolean; cleaned: number }>(`/market/cache/cleanup`),
};

// ==================== STRATEGY ====================
export const strategy = {
    list: async (mode?: 'single' | 'multi'): Promise<StrategyInfo[]> => {
        const params = mode ? { mode } : {};
        return client.get<StrategyInfo[]>('/strategy/list', { params })
    },

    get: (strategy_key: string) =>
        client.get<StrategyInfo>(`/strategy/${strategy_key}`),
};

// ==================== ANALYTICS ====================
export interface AnalyticsPeriod {
    period: string;
}

export const analytics = {
    getPerformance: async (portfolio_id: number, params: AnalyticsPeriod) => {
        return client.get<Record<string, any>>(`/analytics/performance/${portfolio_id}`, { params })
    },

    getReturnsAnalysis: (portfolio_id: number) =>
        client.get<Record<string, any>>(`/analytics/returns/${portfolio_id}`),

    getRiskMetrics: (portfolio_id: number) =>
        client.get<Record<string, any>>(`/analytics/risk/${portfolio_id}`),

    getDrawdownAnalysis: (portfolio_id: number) =>
        client.get<Record<string, any>>(`/analytics/drawdown/${portfolio_id}`),
};

// ==================== MARKET REGIME ====================
export interface RegimeParams {
    period?: string;
}

export const regime = {
    detect: async (symbol: string, params?: RegimeParams): Promise<CurrentRegimeResponse> => {
        return client.get<CurrentRegimeResponse>(`/regime/detect/${symbol}`, { params })
    },

    getHistory: (symbol: string, params?: RegimeParams) =>
        client.get<RegimeHistoryResponse>(`/regime/history/${symbol}`, { params }),

    getReport: (symbol: string, params?: RegimeParams) =>
        client.get<Record<string, any>>(`/regime/report/${symbol}`, { params }),

    detectBatch: (symbols: string[], params?: RegimeParams) =>
        client.post<CurrentRegimeResponse[]>(`/regime/batch`, symbols, { params }),

    trainModel: (symbol: string, params?: RegimeParams) =>
        client.post<{ success: boolean; model_id: string }>(`/regime/train/${symbol}`, null, { params }),

    getWarning: (symbol: string, params?: RegimeParams) =>
        client.get<Record<string, any>>(`/regime/warning/${symbol}`, { params }),

    getAllocation: (symbol: string, params?: RegimeParams) =>
        client.get<AllocationResponse>(`/regime/allocation/${symbol}`, { params }),

    getStrength: (symbol: string, params?: RegimeParams) =>
        client.get<RegimeStrengthResponse>(`/regime/strength/${symbol}`, { params }),

    getTransitions: (symbol: string, params?: RegimeParams) =>
        client.get<TransitionResponse>(`/regime/transitions/${symbol}`, { params }),

    getFeatures: (symbol: string, params?: RegimeParams) =>
        client.get<FeaturesResponse>(`/regime/features/${symbol}`, { params }),

    clearDetectorCache: (symbol: string) =>
        client.delete<{ success: boolean; message: string }>(`/regime/cache/${symbol}`),

    clearAllCache: () =>
        client.delete<{ success: boolean; cleared: number }>(`/regime/cache`),
};

// ==================== ANALYST ====================

export const analyst = {
    getReport: (ticker: string, params?: AnalystReportParams) =>
        client.get<AnalystReport>(`/analyst/report/${ticker}`, { params }),
};

// ==================== ADVISOR ====================
export const advisor = {
    generateGuide: (data: GuideRequest) =>
        client.post<GuideResponse>('/advisor/guide', data),
};

// ==================== ALERTS ====================
export const alerts = {
    sendEmail: (data: EmailAlertRequest) =>
        client.post<{ success: boolean; message_id?: string }>('/alerts/email', data),

    sendSMS: (data: SMSAlertRequest) =>
        client.post<{ success: boolean; message_id?: string }>('/alerts/sms', data),

    test: () =>
        client.get<AlertTestResponse>('/alerts/test'),

    getPreferences: () =>
        client.get<AlertPreferences>('/alerts/preferences'),

    updatePreferences: (preferences: AlertPreferences) =>
        client.put<{ success: boolean }>('/alerts/preferences', preferences),

    getHistory: (limit: number = 10) =>
        client.get<Alert[]>(`/alerts/history`, { params: { limit } }),

    sendTest: (channel: 'email' | 'sms') =>
        client.post<{ success: boolean }>(`/alerts/test`, { channel }),
};

// ==================== LIVE TRADING ====================

export const live = {
    getStatus: () =>
        client.get<LiveStatus>('/live/status'),

    connect: (request: ConnectRequest) =>
        client.post<LiveStatus>('/live/connect', request),

    disconnect: () =>
        client.post<LiveStatus>('/live/disconnect'),

    startEngine: () =>
        client.post<{ success: boolean; message: string }>('/live/engine/start'),

    stopEngine: () =>
        client.post<{ success: boolean; message: string }>('/live/engine/stop'),

    getOrders: () =>
        client.get<ExecutionOrder[]>('/live/orders'),

    placeOrder: (order: LiveOrderPlacement) =>
        client.post<ExecutionOrder>('/live/order', order),

    cancelOrder: (id: string) =>
        client.delete<{ success: boolean; message: string }>(`/live/order/${id}`),

    modifyOrder: (id: string, updates: LiveOrderUpdate) =>
        client.put<ExecutionOrder>(`/live/order/${id}`, updates),

    list: async (params?: { status?: string; mode?: string }): Promise<LiveStrategy[]> => {
        return client.get<LiveStrategy[]>('/live/strategy', { params })
    },

    getDetails: (id: number) =>
        client.get<StrategyDetailsResponse>(`/live/strategy/${id}`),

    controlStrategy: (id: number, action: 'start' | 'pause' | 'stop' | 'restart') =>
        client.post<ControlResponse>(`/live/strategy/${id}/control`, { action }),

    update: (id: number, data: Partial<StrategyUpdateRequest>) =>
        client.patch<UpdateResponse>(`/live/strategy/${id}`, data),

    getVersions: (id: number) =>
        client.get<any[]>(`/live/strategy/${id}/versions`),

    rollback: (id: number, versionId: number) =>
        client.post<{ status: string; message: string; current_version: number }>(`/live/strategy/${id}/rollback/${versionId}`),

    delete: (id: number) =>
        client.delete<{ strategy_id: number; message: string }>(`/live/strategy/${id}`),

    deploy: (config: DeploymentConfig) =>
        client.post<{ strategy_id: number; message: string }>(`/live/strategy/deploy`, config),

    autoConnect: () =>
        client.post<{ status: string; broker?: string; message: string }>('/live/auto-connect'),

    getAccount: () =>
        client.get<AccountResponse>('/live/account'),
};

// ==================== SOCIAL ====================
export const social = {
    getFeed: (limit: number = 50) =>
        client.get<ActivityResponse[]>('/social/activity', { params: { limit } }),

    logActivity: (content: string, type: string = "MANUAL") =>
        client.post<{ status: string }>('/social/activity/log', null, { params: { content, activity_type: type } }),
};

// ==================== MARKETPLACE ====================
export const marketplace = {
    browse: (params?: MarketplaceFilterParams) =>
        client.get<StrategyListing[]>('/marketplace/', { params }),

    getDetails: (strategy_id: number) =>
        client.get<StrategyListingDetailed>(`/marketplace/${strategy_id}`),

    favorite: (strategy_id: number) =>
        client.post<{ success: boolean; message: string }>(`/marketplace/${strategy_id}/favorite`),

    unfavorite: (strategy_id: number) =>
        client.delete<{ success: boolean; message: string }>(`/marketplace/${strategy_id}/favorite`),

    recordDownload: (strategy_id: number) =>
        client.post<{ success: boolean; downloads: number }>(`/marketplace/${strategy_id}/download`),

    publish: (request: StrategyPublishRequest) =>
        client.post<{ success: boolean; strategy_id: number; message: string }>(`/marketplace/publish`, request),

    getReviews: (strategy_id: number, limit: number = 20) =>
        client.get<StrategyReviewSchema[]>(`/marketplace/${strategy_id}/reviews`, { params: { limit } }),

    createReview: (strategy_id: number, data: { rating: number; review_text: string; performance_achieved?: Record<string, any> }) =>
        client.post<{ id: number; status: string }>(`/marketplace/${strategy_id}/review`, data),
};

// ==================== ML STUDIO ====================

export const mlstudio = {
    getModels: () =>
        client.get<MLModel[]>('/mlstudio/models'),

    trainModel: (config: TrainingConfig) =>
        client.post<MLModel>('/mlstudio/train', config),

    deployModel: (model_id: string) =>
        client.post<{ success: boolean; deployed: boolean; endpoint?: string }>(`/mlstudio/deploy/${model_id}`),

    undeployModel: (model_id: string) =>
        client.post<{ success: boolean; deployed: boolean; endpoint?: string }>(`/mlstudio/undeploy/${model_id}`),

    updateModelStatus: (request: MLModelStatusRequest) =>
            client.post<{ success: boolean; deployed: boolean; endpoint?: string }>(`/mlstudio/update-status/`, request),

    exportModel: (model_id: string) =>
        client.post<Blob | MediaSource>(`/mlstudio/export/${model_id}`),

    deleteModel: (model_id: string) =>
        client.delete<{ success: boolean; message: string }>(`/mlstudio/models/${model_id}`),

    predictWithModel: (model_id: string) =>
        client.get<ModelPrediction>(`/mlstudio/models/${model_id}/predict`),

    getDeployedModels: () =>
        client.get<DeployedMLModel[]>('/mlstudio/models/deployed'),
};

// ==================== OPTIONS ====================
export const options = {
    getChain: async (request: ChainRequest): Promise<ChainResponse> => {
        return client.post<ChainResponse>('/options/chain', request)
    },

    runBacktest: async (request: OptionsBacktestRequest): Promise<OptionsBacktestResponse> => {
        return client.post<OptionsBacktestResponse>('/options/backtest', request);
    },

    analyzeStrategy: async (request: StrategyAnalysisRequest): Promise<StrategyAnalysisResponse> => {
        return client.post<StrategyAnalysisResponse>('/options/analyze', request)
    },

    calculateGreeks: async (request: GreeksRequest): Promise<GreeksResponse> => {
        return client.post<GreeksResponse>('/options/greeks', request)
    },

    compareStrategies: async (request: StrategyComparisonRequest): Promise<StrategyComparisonResponse> => {
        return client.post<StrategyComparisonResponse>('/options/compare', request)
    },

    calculateProbability: async (request: ProbabilityRequest): Promise<ProbabilityResponse> => {
        return client.post<ProbabilityResponse>('/options/analytics/probability', request)
    },

    optimizeStrike: async (request: StrikeOptimizerRequest): Promise<StrikeOptimizerResponse> => {
        return client.post<StrikeOptimizerResponse>('/options/analytics/optimize-strike', request)
    },

    calculateRiskMetrics: async (request: RiskMetricsRequest): Promise<RiskMetricsResponse> => {
        return client.post<RiskMetricsResponse>('/options/analytics/risk-metrics', request)
    },
    calculatePortfolioStats: async (request: PortfolioStatsRequest): Promise<PortfolioStatsResponse> => {
        return client.post<PortfolioStatsResponse>('/options/analytics/portfolio-stats', request)
    },

    runMonteCarlo: async (request: MonteCarloRequest): Promise<MonteCarloResponse> => {
        return client.post<MonteCarloResponse>('/options/analytics/monte-carlo', request)
    }
};

// ==================== PORTFOLIO OPTIMIZATION ====================
export const optimization = {
    /**
     * Optimize portfolio for maximum Sharpe ratio
     */
    sharpe: (request: SharpeOptimizationRequest) =>
        client.post<OptimizationResponse>('/optimise/sharpe', request),

    /**
     * Optimize portfolio for minimum volatility
     */
    minVolatility: (request: BaseOptimizationRequest) =>
        client.post<OptimizationResponse>('/optimise/min-volatility', request),

    /**
     * Optimize portfolio for target return
     */
    targetReturn: (request: TargetReturnRequest) =>
        client.post<OptimizationResponse>('/optimise/target-return', request),

    /**
     * Create equal-weighted portfolio
     */
    equalWeight: (request: BaseOptimizationRequest) =>
        client.post<OptimizationResponse>('/optimise/equal-weight', request),

    /**
     * Create risk parity portfolio
     */
    riskParity: (request: BaseOptimizationRequest) =>
        client.post<OptimizationResponse>('/optimise/risk-parity', request),

    /**
     * Black-Litterman optimization with investor views
     */
    blackLitterman: (request: BlackLittermanRequest) =>
        client.post<BlackLittermanResponse>('/optimise/black-litterman', request),

    /**
     * Generate efficient frontier
     */
    efficientFrontier: (request: EfficientFrontierRequest) =>
        client.post<EfficientFrontierResponse>('/optimise/efficient-frontier', request),

    /**
     * Compare all optimization strategies
     */
    compareStrategies: (request: BaseOptimizationRequest) =>
        client.post<CompareStrategiesResponse>('/optimise/compare-strategies', request),
};

// ==================== PORTFOLIO BACKTEST ====================
export const portfolioBacktest = {
    /**
     * Run portfolio backtest
     */
    run: (request: PortfolioBacktestRequest) =>
        client.post<OptimiseBacktestResponse>('/optimise/backtest', request),
};

// ==================== HELPER UTILITIES ====================
export const portfolioHelpers = {
    /**
     * Validate portfolio symbols
     */
    validateSymbols(symbols: string[]): boolean {
        if (!Array.isArray(symbols) || symbols.length < 2) {
            throw new Error('Need at least 2 symbols');
        }
        return true;
    },

    /**
     * Validate portfolio weights
     */
    validateWeights(weights: Record<string, number>): boolean {
        const total = Object.values(weights).reduce((sum, w) => sum + w, 0);
        if (Math.abs(total - 1.0) > 0.01) {
            throw new Error('Weights must sum to 1.0');
        }
        return true;
    },

    /**
     * Sort weights by value (descending)
     */
    sortWeights(weights: Record<string, number>): [string, number][] {
        return Object.entries(weights).sort(([, a], [, b]) => b - a);
    },

    /**
     * Calculate portfolio concentration (Herfindahl index)
     */
    calculateConcentration(weights: Record<string, number>): number {
        return Object.values(weights).reduce((sum, w) => sum + w * w, 0);
    },

    /**
     * Get top N holdings
     */
    getTopHoldings(weights: Record<string, number>, n = 5): Record<string, number> {
        const sorted = this.sortWeights(weights);
        return Object.fromEntries(sorted.slice(0, n));
    },
};

// ==================== SETTINGS ====================
export const settings = {
    get: () =>
        client.get<UserSettings>('/settings/'),

    update: (request: SettingsUpdate) =>
        client.put<UserSettings>('/settings/', request),

    reset: () =>
        client.post<UserSettings>('/settings/reset'),

    testBrokerConnection: () =>
        client.post<BrokerConnectionResponse>('/settings/broker/test-connection'),

    deleteBrokerCredentials: () =>
        client.delete<{ message: string; timestamp: string }>('/settings/broker/credentials'),

};

// ==================== HEALTH & UTILITY ====================
export const health = {
    check: () =>
        client.get<{ status: string; timestamp: string }>('/health'),

    root: () =>
        client.get<{ message: string; version: string }>('/'),
};

// ==================== EXPORT ALL ====================
export const api = {
    auth,
    backtest,
    portfolio,
    market,
    strategy,
    analytics,
    regime,
    analyst,
    advisor,
    alerts,
    live,
    social,
    marketplace,
    mlstudio,
    options,
    settings,
    health,

    // Direct axios instance for custom requests
    client: client,

    // Helper methods
    setAuthToken: (token: string) => {
        if (IS_BROWSER) {
            localStorage.setItem('access_token', token);
        }
        client.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    },

    clearAuthToken: () => {
        if (IS_BROWSER) {
            localStorage.removeItem('access_token');
            localStorage.removeItem('refresh_token');
            localStorage.removeItem('user');
        }
        delete client.defaults.headers.common['Authorization'];
    },

    // File upload helper (if needed)
    uploadFile: (url: string, file: File, data?: Record<string, unknown>) => {
        const formData = new FormData();
        formData.append('file', file);
        if (data) {
            Object.entries(data).forEach(([key, value]) => {
                formData.append(key, String(value));
            });
        }
        return client.post<{ success: boolean; file_id: string; url?: string }>(url, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
    },
};

// ==================== TYPES FOR QUERY HOOKS ====================
export type ApiEndpoint =
    | 'auth/register'
    | 'auth/login'
    | 'auth/logout'
    | 'auth/me'
    | 'backtest/single'
    | 'backtest/multi'
    | 'backtest/options'
    | 'backtest/history'
    | 'backtest/history/count'
    | 'backtest/history/{backtest_id}'
    | 'portfolio/'
    | 'portfolio/{portfolio_id}'
    | 'portfolio/{portfolio_id}/metrics'
    | 'portfolio/{portfolio_id}/positions'
    | 'portfolio/{portfolio_id}/trades'
    | 'market/quote/{symbol}'
    | 'market/quotes'
    | 'market/historical/{symbol}'
    | 'market/options/{symbol}'
    | 'market/fundamentals/{symbol}'
    | 'market/news/{symbol}'
    | 'market/search'
    | 'market/validate/{symbol}'
    | 'market/status'
    | 'strategy/list'
    | 'strategy/{strategy_key}'
    | 'analytics/performance/{portfolio_id}'
    | 'analytics/returns/{portfolio_id}'
    | 'analytics/risk/{portfolio_id}'
    | 'analytics/drawdown/{portfolio_id}'
    | 'regime/detect/{symbol}'
    | 'regime/history/{symbol}'
    | 'regime/report/{symbol}'
    | 'regime/batch'
    | 'regime/strength/{symbol}'
    | 'regime/transitions/{symbol}'
    | 'regime/features/{symbol}'
    | 'analyst/report/{ticker}'
    | 'advisor/guide'
    | 'alerts/email'
    | 'alerts/sms'
    | 'alerts/test'
    | 'live/status'
    | 'live/connect'
    | 'live/disconnect'
    | 'live/engine/start'
    | 'live/engine/stop'
    | 'live/orders'
    | 'marketplace/'
    | 'marketplace/{strategy_id}'
    | 'marketplace/{strategy_id}/favorite'
    | 'marketplace/{strategy_id}/download'
    | 'marketplace/publish'
    | 'mlstudio/models'
    | 'mlstudio/train'
    | 'mlstudio/deploy/{model_id}'
    | 'mlstudio/models/{model_id}'
    | 'mlstudio/models/{model_id}/predict'
    | 'options/chain'
    | 'options/backtest'
    | 'options/analyze'
    | 'options/greeks'
    | 'options/compare'
    | 'options/analytics/probability'
    | 'options/analytics/optimize-strike'
    | 'options/analytics/risk-metrics'
    | 'options/analytics/portfolio-stats'
    | 'options/analytics/monte-carlo'
    | 'settings/'
    | 'settings/reset'
    | 'health'
    | '/';

// ==================== ERROR TYPES ====================
export interface ApiError {
    status: number;
    data?: HTTPValidationError | Record<string, unknown>;
    message: string;
}

export interface ApiResponse<T> {
    data: T;
    status: number;
    message?: string;
}

export default api;
