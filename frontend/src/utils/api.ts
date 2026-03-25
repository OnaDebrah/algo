/* eslint-disable @typescript-eslint/no-explicit-any */

import axios from 'axios';
import {
    Alert,
    AlertPreferences,
    AlertTestResponse,
    AllocationResponse,
    AnalystReport,
    BacktestHistoryItem,
    BrokerConnectionResponse,
    ChainRequest,
    ChainResponse,
    CrashAlertConfig,
    CrashDashboardData,
    CrashPredictionHistoryItem,
    CrashPredictionResult,
    CurrentRegimeResponse,
    DeployedMLModel,
    DeploymentConfig,
    EmailAlertRequest,
    ExecutionOrder,
    FeaturesResponse,
    GreeksRequest,
    GreeksResponse,
    GuideRequest,
    GuideResponse,
    HedgeRecommendation,
    HistoricalAccuracyData,
    HistoricalDataPoint,
    HTTPValidationError,
    LiveOrderPlacement,
    LiveOrderUpdate,
    LiveStatus,
    LoginResponse,
    MarketplaceFilterParams,
    MarketStressResult,
    MLModel,
    MLModelStatusRequest,
    ModelPrediction,
    MonteCarloRequest,
    MonteCarloResponse,
    MultiAssetBacktestRequest,
    MultiAssetBacktestResponse,
    OptionsBacktestRequest,
    OptionsBacktestResponse,
    OptionsChain,
    PairsValidationRequest,
    PairsValidationResponse,
    Portfolio,
    PortfolioCreate,
    PortfolioMetrics,
    PortfolioStatsRequest,
    PortfolioStatsResponse,
    PortfolioTrade,
    PortfolioUpdate,
    Position,
    ProbabilityRequest,
    ProbabilityResponse,
    QuoteData,
    RegimeHistoryResponse,
    RegimeStrengthResponse,
    RiskMetricsRequest,
    RiskMetricsResponse,
    SearchResult,
    SectorScanResult,
    SectorSummary,
    SettingsUpdate,
    SingleBacktestRequest,
    SingleBacktestResponse,
    SMSAlertRequest,
    StockRanking,
    StrategyAnalysisRequest,
    StrategyAnalysisResponse,
    StrategyComparisonRequest,
    StrategyComparisonResponse,
    StrategyInfo,
    StrategyListing,
    StrategyListingDetailed,
    StrategyPublishRequest,
    StrategyRecommendation,
    StrategyReviewSchema,
    StrikeOptimizerRequest,
    StrikeOptimizerResponse,
    TrainingConfig,
    TransitionResponse,
    User,
    UserCreate,
    UserLogin,
    UserSettings,
    WFARequest,
    WFAResponse,
    AdminUserListItem,
    AdminUserDetail,
    AdminStats,
    PricingData,
    QuotaStatus,
    OptimizePreviewResponse
} from '@/types/all_types';
import {
    BaseOptimizationRequest,
    BlackLittermanRequest,
    BlackLittermanResponse,
    CompareStrategiesResponse,
    EfficientFrontierRequest,
    EfficientFrontierResponse,
    OptimiseBacktestResponse,
    OptimizationResponse,
    PortfolioBacktestRequest,
    SharpeOptimizationRequest,
    TargetReturnRequest
} from "@/types/optimise";
import {
    AccountResponse,
    ConnectRequest,
    ControlResponse,
    LiveStrategy,
    StrategyDetailsResponse,
    StrategyUpdateRequest,
    UpdateResponse
} from "@/types/live";
import {ActivityResponse} from "@/types/social";
import {AnalystReportParams} from "@/types/analyst";

// Use environment variable or default to localhost
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const client = axios.create({
    baseURL: API_URL,
    timeout: 30000,
    headers: {
        'Content-Type': 'application/json',
    },
    withCredentials: true,
});

// Request interceptor — auth is handled via httpOnly cookies (sent automatically
// with withCredentials: true). No need to read localStorage for tokens.
client.interceptors.request.use(
    (config) => config,
    (error) => Promise.reject(error)
);

// Response interceptor to handle errors
client.interceptors.response.use(
    (response) => response.data,
    (error) => {
        // 401 handling: Don't redirect — AppShell renders <LoginPage /> when
        // user is null. Redirecting to '/login' breaks the app because no
        // such Next.js route exists; login is handled by conditional rendering.
        // The error propagates to the caller (restoreSession, autoConnect, etc.)
        // which already handles it gracefully.

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

    refreshToken: (refreshToken: string) =>
        client.post<{ access_token: string }>('/auth/refresh', { refresh_token: refreshToken }),

    forgotPassword: (email: string) =>
        client.post<{ message: string }>('/auth/forgot-password', { email }),
    resetPassword: (token: string, newPassword: string) =>
        client.post<{ message: string }>('/auth/reset-password', { token, new_password: newPassword }),

    setup2FA: () => client.post('/auth/2fa/setup') as Promise<any>,
    verifySetup2FA: (code: string) => client.post('/auth/2fa/verify-setup', { code }) as Promise<any>,
    verify2FA: (pending_2fa_token: string, code: string) => client.post('/auth/2fa/verify', { pending_2fa_token, code }) as Promise<any>,
    disable2FA: (code: string) => client.post('/auth/2fa/disable', { code }) as Promise<any>,
};

// ==================== BACKTEST ====================

/** Response from POST endpoints after Celery dispatch */
export interface BacktestSubmission {
    backtest_id: number;
    task_id: string;
    status: string;
    message: string;
}

/**
 * Poll /backtest/history/{id} until status is 'completed' or 'failed'.
 * Returns the full BacktestHistoryItem once done.
 */
async function pollForResult(
    backtestId: number,
    intervalMs: number = 2000,
    maxAttempts: number = 300  // 10 minutes at 2s
): Promise<BacktestHistoryItem> {
    console.log(`[pollForResult] Starting poll for backtest_id=${backtestId}`);
    for (let i = 0; i < maxAttempts; i++) {
        const details = await client.get<BacktestHistoryItem>(
            `/backtest/history/${backtestId}`
        );

        console.log(`[pollForResult] Poll #${i + 1}: status=${details.status}`);

        if (details.status === 'completed' || details.status === 'failed') {
            if (details.status === 'failed') {
                throw new Error(details.error_message || 'Backtest failed');
            }
            console.log(`[pollForResult] Completed! extended_results keys:`, details.extended_results ? Object.keys(details.extended_results) : 'null');
            return details;
        }

        // Wait before next poll
        await new Promise(resolve => setTimeout(resolve, intervalMs));
    }
    throw new Error('Backtest timed out');
}

export const backtest = {
    /**
     * Submit a single-asset backtest (returns immediately, polls for result).
     * Returns the same shape as before so callers are unaffected.
     */
    runSingle: async (request: SingleBacktestRequest): Promise<SingleBacktestResponse> => {
        const submission = await client.post<BacktestSubmission>('/backtest/single', request);
        const completed = await pollForResult(submission.backtest_id);

        // Reconstruct the nested response shape the store expects
        // from the flat BacktestHistoryItem returned by polling
        const item = completed as any;
        const ext = item.extended_results || {};
        return {
            result: {
                // Core metrics from top-level columns
                total_return: ext.total_return ?? item.total_return_pct ?? 0,
                total_return_pct: item.total_return_pct ?? 0,
                win_rate: item.win_rate ?? 0,
                sharpe_ratio: item.sharpe_ratio ?? 0,
                max_drawdown: item.max_drawdown ?? 0,
                total_trades: item.total_trades ?? 0,
                final_equity: item.final_equity ?? 0,
                initial_capital: ext.initial_capital ?? item.initial_capital ?? 0,
                // Trade statistics from extended_results
                winning_trades: ext.winning_trades ?? 0,
                losing_trades: ext.losing_trades ?? 0,
                avg_profit: ext.avg_profit ?? 0,
                avg_win: ext.avg_win ?? 0,
                avg_loss: ext.avg_loss ?? 0,
                profit_factor: ext.profit_factor ?? 0,
                symbol_stats: ext.symbol_stats ?? {},
                // Factor attribution from extended_results
                alpha: ext.alpha ?? 0,
                beta: ext.beta ?? 0,
                r_squared: ext.r_squared ?? 0,
                // Advanced metrics from extended_results
                sortino_ratio: ext.sortino_ratio ?? 0,
                calmar_ratio: ext.calmar_ratio ?? 0,
                var_95: ext.var_95 ?? 0,
                cvar_95: ext.cvar_95 ?? 0,
                volatility: ext.volatility ?? 0,
                expectancy: ext.expectancy ?? 0,
                total_commission: ext.total_commission ?? 0,
                // Monthly returns matrix from extended_results
                monthly_returns_matrix: ext.monthly_returns_matrix ?? undefined,
            },
            equity_curve: item.equity_curve ?? [],
            trades: item.trades ?? [],
            // Price data and benchmark from extended_results
            price_data: ext.price_data ?? null,
            benchmark: ext.benchmark ?? null,
        } as SingleBacktestResponse;
    },

    /**
     * Submit a multi-asset backtest (returns immediately, polls for result).
     */
    runMulti: async (request: MultiAssetBacktestRequest): Promise<MultiAssetBacktestResponse> => {
        const submission = await client.post<BacktestSubmission>('/backtest/multi', request);
        const completed = await pollForResult(submission.backtest_id);

        // Reconstruct the nested response shape from flat BacktestHistoryItem
        const item = completed as any;
        const ext = item.extended_results || {};
        return {
            result: {
                // Core metrics from top-level columns
                total_return: ext.total_return ?? item.total_return_pct ?? 0,
                total_return_pct: item.total_return_pct ?? 0,
                win_rate: item.win_rate ?? 0,
                sharpe_ratio: item.sharpe_ratio ?? 0,
                max_drawdown: item.max_drawdown ?? 0,
                total_trades: item.total_trades ?? 0,
                final_equity: item.final_equity ?? 0,
                initial_capital: ext.initial_capital ?? item.initial_capital ?? 0,
                // Trade statistics from extended_results
                winning_trades: ext.winning_trades ?? 0,
                losing_trades: ext.losing_trades ?? 0,
                avg_profit: ext.avg_profit ?? 0,
                avg_win: ext.avg_win ?? 0,
                avg_loss: ext.avg_loss ?? 0,
                profit_factor: ext.profit_factor ?? 0,
                symbol_stats: ext.symbol_stats ?? {},
                // Factor attribution from extended_results
                alpha: ext.alpha ?? 0,
                beta: ext.beta ?? 0,
                r_squared: ext.r_squared ?? 0,
                // Advanced metrics from extended_results
                sortino_ratio: ext.sortino_ratio ?? 0,
                calmar_ratio: ext.calmar_ratio ?? 0,
                var_95: ext.var_95 ?? 0,
                cvar_95: ext.cvar_95 ?? 0,
                volatility: ext.volatility ?? 0,
                expectancy: ext.expectancy ?? 0,
                total_commission: ext.total_commission ?? 0,
                // Multi-asset specific
                num_symbols: ext.num_symbols ?? item.symbols?.length ?? 0,
            },
            equity_curve: item.equity_curve ?? [],
            trades: item.trades ?? [],
            // Price data and benchmark from extended_results
            price_data: ext.price_data ?? null,
            benchmark: ext.benchmark ?? null,
        } as MultiAssetBacktestResponse;
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
     * Run Bayesian optimization on strategy parameters.
     * Dispatches a Celery task and polls for the result (same pattern as WFA).
     */
    bayesian: async (request: any) => {
        const submission = await client.post<BacktestSubmission>('/optimise/bayesian', request);
        console.log('[bayesian] Submission response:', JSON.stringify(submission));
        const completed = await pollForResult(submission.backtest_id);
        console.log('[bayesian] Poll completed. Status:', completed.status, 'Has extended_results.bayesian:', !!completed.extended_results?.bayesian);
        const result = completed.extended_results?.bayesian || completed;
        console.log('[bayesian] Returning result with keys:', Object.keys(result));
        return result;
    },

    /**
     * Submit Walk-Forward Analysis (returns immediately, polls for result).
     */
    walkForward: async (request: WFARequest): Promise<WFAResponse> => {
        const submission = await client.post<BacktestSubmission>('/backtest/walk-forward', request);
        const completed = await pollForResult(submission.backtest_id);

        const wfaData = completed.extended_results?.wfa_results;
        if (wfaData) {
            return wfaData as unknown as WFAResponse;
        }
        // Fallback: check trades for backward compatibility with older data
        if (completed.trades && !Array.isArray(completed.trades)) {
            return completed.trades as unknown as WFAResponse;
        }
        return completed as unknown as WFAResponse;
    },

    /**
     * Check Celery task status directly (optional, for advanced UIs).
     */
    getTaskStatus: (taskId: string) =>
        client.get<{ task_id: string; status: string; result?: any; error?: string }>(
            `/backtest/task/${taskId}`
        ),
};

// ==================== PORTFOLIO ====================
export const portfolio = {

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

    getHistorical: async (symbol: string, params?: HistoricalDataParams): Promise<HistoricalDataPoint[]> => {
        const res = await client.get(`/market/historical/${symbol}`, { params });
        // Backend returns { symbol, period, interval, data: [...] }; interceptor unwraps AxiosResponse
        return Array.isArray(res) ? res : (res as any)?.data || [];
    },

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

export interface StrategyGenerateRequest {
    prompt: string;
    style: string;
    timeframe?: string;
}

export interface StrategyGenerateResponse {
    code: string;
    explanation: string;
    example_usage: string;
    provider: string;
}

export interface StrategyValidateResponse {
    is_valid: boolean;
    errors: string[];
    warnings: string[];
}

export interface CustomStrategy {
    id: number;
    name: string;
    description: string | null;
    code: string;
    strategy_type: string;
    parameters: Record<string, unknown> | null;
    is_validated: boolean;
    ai_generated: boolean;
    ai_explanation: string | null;
    created_at: string;
    updated_at: string;
}

export interface CustomStrategyCreate {
    name: string;
    description?: string;
    code: string;
    strategy_type?: string;
    parameters?: Record<string, unknown>;
    ai_generated?: boolean;
    ai_explanation?: string;
}

export interface CustomBacktestRequest {
    code: string;
    symbol: string;
    period?: string;
    interval?: string;
    initial_capital?: number;
    commission_rate?: number;
    slippage_rate?: number;
    strategy_name?: string;
    custom_strategy_id?: number;
}

export const strategy = {
    list: async (mode?: 'single' | 'multi'): Promise<StrategyInfo[]> => {
        const params = mode ? { mode } : {};
        return client.get<StrategyInfo[]>('/strategy/list', { params })
    },

    get: (strategy_key: string) =>
        client.get<StrategyInfo>(`/strategy/${strategy_key}`),

    generate: (request: StrategyGenerateRequest) =>
        client.post<StrategyGenerateResponse>('/strategy/generate', request),

    validate: (request: { code: string }) =>
        client.post<StrategyValidateResponse>('/strategy/validate', request),

    // Custom strategy backtest (polls for result like backtest.runSingle)
    backtestCustom: async (request: CustomBacktestRequest): Promise<SingleBacktestResponse> => {
        const submission = await client.post<BacktestSubmission>('/strategy/backtest', request);
        const completed = await pollForResult(submission.backtest_id);

        // Reconstruct the nested response shape (same mapping as backtest.runSingle)
        const item = completed as Record<string, any>;
        const ext = item.extended_results || {};
        return {
            result: {
                total_return: ext.total_return ?? item.total_return_pct ?? 0,
                total_return_pct: item.total_return_pct ?? 0,
                win_rate: item.win_rate ?? 0,
                sharpe_ratio: item.sharpe_ratio ?? 0,
                max_drawdown: item.max_drawdown ?? 0,
                total_trades: item.total_trades ?? 0,
                final_equity: item.final_equity ?? 0,
                initial_capital: ext.initial_capital ?? item.initial_capital ?? 0,
                winning_trades: ext.winning_trades ?? 0,
                losing_trades: ext.losing_trades ?? 0,
                avg_profit: ext.avg_profit ?? 0,
                avg_win: ext.avg_win ?? 0,
                avg_loss: ext.avg_loss ?? 0,
                profit_factor: ext.profit_factor ?? 0,
                symbol_stats: ext.symbol_stats ?? {},
                alpha: ext.alpha ?? 0,
                beta: ext.beta ?? 0,
                r_squared: ext.r_squared ?? 0,
                sortino_ratio: ext.sortino_ratio ?? 0,
                calmar_ratio: ext.calmar_ratio ?? 0,
                var_95: ext.var_95 ?? 0,
                cvar_95: ext.cvar_95 ?? 0,
                volatility: ext.volatility ?? 0,
                expectancy: ext.expectancy ?? 0,
                total_commission: ext.total_commission ?? 0,
                monthly_returns_matrix: ext.monthly_returns_matrix ?? undefined,
            },
            equity_curve: item.equity_curve ?? [],
            trades: item.trades ?? [],
            price_data: ext.price_data ?? null,
            benchmark: ext.benchmark ?? null,
        } as SingleBacktestResponse;
    },

    // CRUD for saved custom strategies
    listCustom: () =>
        client.get<CustomStrategy[]>('/strategy/custom'),

    getCustom: (id: number) =>
        client.get<CustomStrategy>(`/strategy/custom/${id}`),

    createCustom: (data: CustomStrategyCreate) =>
        client.post<CustomStrategy>('/strategy/custom', data),

    updateCustom: (id: number, data: Partial<CustomStrategyCreate>) =>
        client.put<CustomStrategy>(`/strategy/custom/${id}`, data),

    deleteCustom: (id: number) =>
        client.delete(`/strategy/custom/${id}`),
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

    correlation: (symbols: string[], period: string = '1Y') =>
        client.post<{ symbols: string[]; matrix: number[][]; data_points: number; errors: string[] }>(
            '/analytics/correlation', { symbols, period }
        ),
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

    leaderboard: (params?: { metric?: string; category?: string; limit?: number }) =>
        client.get('/marketplace/leaderboard', { params }) as Promise<any>,

    cloneStrategy: (strategy_id: number) =>
        client.post<{ id: number; name: string; parent_strategy_id: number; message: string }>(
            `/marketplace/strategies/${strategy_id}/clone`
        ),

    // Discussion / Comments
    getComments: (strategy_id: number, limit: number = 50) =>
        client.get(`/marketplace/${strategy_id}/comments`, { params: { limit } }),

    createComment: (strategy_id: number, data: { content: string; parent_comment_id?: number }) =>
        client.post(`/marketplace/${strategy_id}/comments`, data),

    updateComment: (comment_id: number, data: { content: string }) =>
        client.put(`/marketplace/comments/${comment_id}`, data),

    deleteComment: (comment_id: number) =>
        client.delete(`/marketplace/comments/${comment_id}`),
};

// ==================== ALTERNATIVE DATA (SEC EDGAR) ====================

export const alternativeData = {
    getFilings: (ticker: string, params?: { form_type?: string; limit?: number }) =>
        client.get(`/alternative-data/filings/${ticker}`, { params }),

    getInsiderTransactions: (ticker: string, limit: number = 20) =>
        client.get(`/alternative-data/insider-transactions/${ticker}`, { params: { limit } }),

    getCompanyFacts: (ticker: string) =>
        client.get(`/alternative-data/company-facts/${ticker}`),
};

// ==================== CRYPTO ====================

export const cryptoApi = {
    marketOverview: () => client.get('/crypto/market-overview'),
    topCoins: (limit: number = 50) => client.get('/crypto/top', { params: { limit } }),
    trending: () => client.get('/crypto/trending'),
    quote: (symbol: string) => client.get(`/crypto/quote/${symbol}`),
    coinDetail: (coinId: string) => client.get(`/crypto/coin/${coinId}`),
    historical: (symbol: string, params?: { period?: string; interval?: string }) =>
        client.get(`/crypto/historical/${symbol}`, { params }),
    correlation: (coins?: string, period?: string) =>
        client.get('/crypto/analytics/correlation', { params: { coins, period } }),
    momentum: () => client.get('/crypto/analytics/momentum'),
    sectors: () => client.get('/crypto/analytics/sectors'),
    volatility: (coins?: string, period?: string) =>
        client.get('/crypto/analytics/volatility', { params: { coins, period } }),
    predictions: () => client.get('/crypto/analytics/predictions'),
    fearGreed: () => client.get('/crypto/analytics/fear-greed'),
    technicalSignals: (coins?: string) =>
        client.get('/crypto/analytics/technical-signals', { params: { coins } }),
    btcDominance: () => client.get('/crypto/analytics/btc-dominance'),
    portfolioOptimize: (holdings: { coin_id: string; amount_usd: number }[]) =>
        client.post('/crypto/analytics/portfolio-optimize', { holdings }),
};

// ==================== WEBHOOKS ====================

export const webhooks = {
    testAuth: (apiKey: string) =>
        client.get('/webhooks/test', { headers: { 'X-API-Key': apiKey } }),

    getHistory: (apiKey: string, limit: number = 20) =>
        client.get('/webhooks/history', { headers: { 'X-API-Key': apiKey }, params: { limit } }),
};

// ==================== AUDIT / TRADE JOURNAL ====================

export const auditApi = {
    getEvents: (params?: { event_type?: string; category?: string; search?: string; page?: number; page_size?: number }) =>
        client.get('/audit/events', { params }),
    getJournal: (params?: { search?: string; page?: number; page_size?: number }) =>
        client.get('/audit/journal', { params }),
    updateNotes: (eventId: number, data: { notes?: string; tags?: string[] }) =>
        client.patch(`/audit/events/${eventId}/notes`, data),
    exportCsv: async (eventType?: string) => {
        const res = await client.get('/audit/events/export/csv', { params: { event_type: eventType }, responseType: 'blob' });
        const url = window.URL.createObjectURL(res as unknown as Blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'audit_trail.csv';
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    },
};

// ==================== API KEYS ====================

export const apiKeys = {
    list: () => client.get<any[]>('/api-keys/'),
    create: (data: { name: string; permissions?: string[]; expires_in_days?: number }) =>
        client.post('/api-keys/', data),
    revoke: (keyId: number) => client.delete(`/api-keys/${keyId}`),
    rotate: (keyId: number) => client.post(`/api-keys/${keyId}/rotate`),
};

// ==================== RESEARCH ====================
export const research = {
    execute: (code: string, timeout?: number) => client.post('/research/execute', { code, timeout }),
    getTemplates: () => client.get('/research/templates'),
    getHistory: () => client.get('/research/history'),
    suggest: (prompt: string, context?: string) => client.post('/research/suggest', { prompt, context }),
};

// ==================== TEAMS ====================

export const teamsApi = {
    list: () => client.get<any[]>('/teams/'),
    create: (data: { name: string; description?: string }) => client.post('/teams/', data),
    join: (teamId: number, inviteCode: string) =>
        client.post(`/teams/${teamId}/join?invite_code=${encodeURIComponent(inviteCode)}`),
    getMembers: (teamId: number) => client.get<any[]>(`/teams/${teamId}/members`),
    updateMemberRole: (teamId: number, userId: number, role: string) =>
        client.patch(`/teams/${teamId}/members/${userId}`, { role }),
    removeMember: (teamId: number, userId: number) =>
        client.delete(`/teams/${teamId}/members/${userId}`),
    addComment: (teamId: number, data: { target_type: string; target_id: number; content: string; parent_comment_id?: number }) =>
        client.post(`/teams/${teamId}/comments`, data),
    getComments: (teamId: number, targetType?: string, targetId?: number) => {
        const params = new URLSearchParams();
        if (targetType) params.set('target_type', targetType);
        if (targetId) params.set('target_id', String(targetId));
        return client.get<any[]>(`/teams/${teamId}/comments?${params.toString()}`);
    },
};

// ==================== EXPORT ====================

const triggerDownload = (blob: Blob, filename: string) => {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
};

export const exportApi = {
    backtestCsv: async (runId: number) => {
        const res = await client.get(`/export/backtest/${runId}/csv`, { responseType: 'blob' });
        triggerDownload(res as unknown as Blob, `backtest_${runId}.csv`);
    },
    backtestPdf: async (runId: number) => {
        const res = await client.get(`/export/backtest/${runId}/pdf`, { responseType: 'blob' });
        triggerDownload(res as unknown as Blob, `backtest_${runId}.pdf`);
    },
    paperTradesCsv: async (portfolioId: number) => {
        const res = await client.get(`/export/paper/${portfolioId}/csv`, { responseType: 'blob' });
        triggerDownload(res as unknown as Blob, `paper_trades_${portfolioId}.csv`);
    },
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
    },

    scanArbitrage: async (request: {
        symbol: string;
        arb_types?: string[];
        volatility_model?: string;
        entry_threshold?: number;
        min_liquidity?: number;
    }): Promise<{
        symbol: string;
        spot_price: number;
        opportunities: Array<{
            type: string;
            asset: string;
            strike: number | null;
            direction: number;
            confidence: number;
            mispricing: number | null;
            entry_price: number | null;
            size: number | null;
            details: Record<string, any> | null;
        }>;
        vol_surface_summary: Record<string, any> | null;
        greek_exposure: Record<string, any> | null;
        scanned_at: string;
    }> => {
        return client.post('/options/arbitrage/scan', request) as any;
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

// ==================== ADMIN ====================
export const admin = {
    getUsers: (params?: { search?: string; tier?: string; is_active?: boolean; limit?: number; offset?: number }) =>
        client.get<AdminUserListItem[]>('/admin/users', { params }) as Promise<AdminUserListItem[]>,

    getUser: (userId: number) =>
        client.get<AdminUserDetail>(`/admin/users/${userId}`) as Promise<AdminUserDetail>,

    updateTier: (userId: number, tier: string) =>
        client.put(`/admin/users/${userId}/tier`, { tier }),

    updateStatus: (userId: number, is_active: boolean) =>
        client.put(`/admin/users/${userId}/status`, { is_active }),

    getStats: () =>
        client.get<AdminStats>('/admin/stats') as Promise<AdminStats>,

    getUsage: (params?: { action?: string; days?: number; limit?: number; offset?: number }) =>
        client.get('/admin/usage', { params }),

    getSubmissions: (status?: string) =>
        client.get('/admin/submissions', { params: { status: status || 'pending_review' } }),

    approveSubmission: (strategyId: number) =>
        client.put(`/admin/submissions/${strategyId}/approve`),

    rejectSubmission: (strategyId: number, rejection_reason: string) =>
        client.put(`/admin/submissions/${strategyId}/reject`, { rejection_reason }),
};

// ==================== PRICING & QUOTA ====================
export const pricing = {
    getTiers: () =>
        client.get<PricingData>('/pricing/tiers') as Promise<PricingData>,

    getQuota: () =>
        client.get<QuotaStatus>('/pricing/quota') as Promise<QuotaStatus>,
};

// ==================== DEPLOY OPTIMIZATION ====================
export const deployOptimize = {
    preview: (symbols: string[], lookbackDays: number = 252, initialCapital: number = 100000) =>
        client.post<OptimizePreviewResponse>('/deploy/optimize-preview', {
            symbols,
            lookback_days: lookbackDays,
            initial_capital: initialCapital,
        }) as Promise<OptimizePreviewResponse>,

    apply: (symbols: string[], method: string, lookbackDays: number = 252) =>
        client.post('/deploy/optimize-apply', {
            symbols,
            method,
            lookback_days: lookbackDays,
        }),
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

// ==================== SECTOR SCANNER ====================
export const sector = {
    list: () =>
        client.get<SectorSummary[]>('/sector/list'),

    scan: (period: string = '6mo') =>
        client.post<SectorScanResult>('/sector/scan', null, { params: { period } }),

    stocks: (sectorName: string, topN: number = 10) =>
        client.post<StockRanking[]>(`/sector/stocks/${encodeURIComponent(sectorName)}`, null, { params: { top_n: topN } }),

    recommend: (symbol: string) =>
        client.post<StrategyRecommendation[]>(`/sector/recommend/${symbol}`),
};

// ==================== CRASH PREDICTION ====================
export const crashPrediction = {
    predict: (symbol: string) =>
        client.get<CrashPredictionResult>(`/crash/predict/${symbol}`),

    getStress: (symbols?: string[]) =>
        client.get<MarketStressResult>('/crash/stress', {
            params: symbols ? { symbols: symbols.join(',') } : {},
        }),

    getHedgeRecommendation: (portfolioValue: number, portfolioBeta: number = 1.0, primaryIndex: string = 'SPY') =>
        client.get<HedgeRecommendation>('/crash/hedge-recommendation', {
            params: {
                portfolio_value: portfolioValue,
                portfolio_beta: portfolioBeta,
                primary_index: primaryIndex,
            },
        }),

    getHistory: (params?: { symbol?: string; limit?: number; offset?: number }) =>
        client.get<{ total: number; offset: number; limit: number; predictions: CrashPredictionHistoryItem[] }>(
            '/crash/history',
            { params },
        ),

    getDashboard: (symbol: string, portfolioValue: number = 100000) =>
        client.get<CrashDashboardData>(`/crash/dashboard/${symbol}`, {
            params: { portfolio_value: portfolioValue },
        }),

    configureAlerts: (config: CrashAlertConfig) =>
        client.post<{ success: boolean; message: string; config: CrashAlertConfig }>(
            '/crash/alert/configure',
            null,
            {
                params: {
                    crash_threshold: config.crash_threshold,
                    stress_threshold: config.stress_threshold,
                    email_enabled: config.email_enabled,
                    sms_enabled: config.sms_enabled,
                },
            },
        ),

    getHistoricalAccuracy: (symbol: string, stride: number = 20, threshold: number = 0.33) =>
        client.get<HistoricalAccuracyData>(`/crash/accuracy/${symbol}`, {
            params: { stride, threshold },
            timeout: 600000, // 10 min timeout for first computation
        }),
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
    sector,
    crashPrediction,
    analyst,
    advisor,
    alerts,
    live,
    social,
    marketplace,
    mlstudio,
    options,
    admin,
    pricing,
    deployOptimize,
    settings,
    health,
    research,

    // Direct axios instance for custom requests
    client: client,

    // Helper methods
    // Auth tokens are now handled via httpOnly cookies — these are kept
    // as no-ops for backward compatibility with any callers.
    setAuthToken: (_token: string) => {
        // No-op: auth is managed via httpOnly cookies set by the server
    },

    clearAuthToken: () => {
        // No-op: cookies are cleared by calling POST /auth/logout
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

// ==================== PAYMENTS ====================
export const payments = {
    createCheckoutSession: (strategyId: number) =>
        client.post<{ checkout_url: string }>('/payments/create-checkout-session', { strategy_id: strategyId }) as Promise<{ checkout_url: string }>,

    getPurchases: () =>
        client.get<number[]>('/payments/purchases') as Promise<number[]>,

    checkPurchase: (strategyId: number) =>
        client.get<{ purchased: boolean }>(`/payments/check/${strategyId}`) as Promise<{ purchased: boolean }>,
};

// ==================== NOTIFICATIONS ====================
export const notifications = {
    list: (params?: { limit?: number; offset?: number; unread_only?: boolean }) =>
        client.get('/notifications', { params }) as Promise<any>,
    unreadCount: () =>
        client.get('/notifications/unread-count') as Promise<{ count: number }>,
    markRead: (id: number) =>
        client.post(`/notifications/${id}/read`) as Promise<any>,
    markAllRead: () =>
        client.post('/notifications/read-all') as Promise<any>,
    delete: (id: number) =>
        client.delete(`/notifications/${id}`) as Promise<any>,
};

export const priceAlerts = {
    list: () =>
        client.get('/price-alerts') as Promise<any>,
    create: (data: { symbol: string; condition: string; target_price: number }) =>
        client.post('/price-alerts', data) as Promise<any>,
    delete: (id: number) =>
        client.delete(`/price-alerts/${id}`) as Promise<any>,
};

// ==================== WATCHLIST ====================
export const watchlists = {
    list: () =>
        client.get('/watchlists') as Promise<any>,
    create: (name: string) =>
        client.post('/watchlists', { name }) as Promise<any>,
    remove: (id: number) =>
        client.delete(`/watchlists/${id}`) as Promise<any>,
    addSymbol: (watchlistId: number, symbol: string, notes?: string) =>
        client.post(`/watchlists/${watchlistId}/symbols`, { symbol, notes }) as Promise<any>,
    removeSymbol: (watchlistId: number, symbol: string) =>
        client.delete(`/watchlists/${watchlistId}/symbols/${symbol}`) as Promise<any>,
    getQuotes: (watchlistId: number) =>
        client.get(`/watchlists/${watchlistId}/quotes`) as Promise<any>,
    screen: (filters: { min_price?: number; max_price?: number; min_change_pct?: number; max_change_pct?: number; min_volume?: number }) =>
        client.post('/screener', filters) as Promise<any>,
};

// ==================== PAPER TRADING ====================
export const paper = {
    listPortfolios: () =>
        client.get('/paper/portfolios') as Promise<any>,
    createPortfolio: (name: string, initial_cash: number = 100000, strategyKey?: string, strategySymbol?: string, strategyParams?: Record<string, unknown>, tradeQuantity?: number, dataInterval?: string) =>
        client.post('/paper/portfolios', { name, initial_cash, strategy_key: strategyKey, strategy_symbol: strategySymbol, strategy_params: strategyParams, trade_quantity: tradeQuantity ?? 100, data_interval: dataInterval ?? '1d' }) as Promise<any>,
    getPortfolio: (id: number) =>
        client.get(`/paper/portfolios/${id}`) as Promise<any>,
    placeTrade: (portfolioId: number, symbol: string, side: string, quantity: number) =>
        client.post(`/paper/portfolios/${portfolioId}/trade`, { symbol, side, quantity }) as Promise<any>,
    getTrades: (portfolioId: number, limit: number = 100) =>
        client.get(`/paper/portfolios/${portfolioId}/trades`, { params: { limit } }) as Promise<any>,
    getPerformance: (portfolioId: number) =>
        client.get(`/paper/portfolios/${portfolioId}/performance`) as Promise<any>,
    getEquity: (portfolioId: number) =>
        client.get(`/paper/portfolios/${portfolioId}/equity`) as Promise<any>,
    deactivate: (id: number) =>
        client.delete(`/paper/portfolios/${id}`) as Promise<any>,
    // Strategy integration
    listStrategies: () =>
        client.get('/paper/strategies') as Promise<any>,
    attachStrategy: (portfolioId: number, strategyKey: string, strategySymbol: string, strategyParams?: Record<string, unknown>, tradeQuantity?: number, dataInterval?: string) =>
        client.post(`/paper/portfolios/${portfolioId}/attach-strategy`, { strategy_key: strategyKey, strategy_symbol: strategySymbol, strategy_params: strategyParams, trade_quantity: tradeQuantity ?? 100, data_interval: dataInterval ?? '1d' }) as Promise<any>,
    detachStrategy: (portfolioId: number) =>
        client.post(`/paper/portfolios/${portfolioId}/detach-strategy`) as Promise<any>,
    runSignal: (portfolioId: number, autoExecute: boolean = true) =>
        client.post(`/paper/portfolios/${portfolioId}/run-signal`, null, { params: { auto_execute: autoExecute } }) as Promise<any>,
    refreshEquity: (portfolioId: number) =>
        client.post(`/paper/portfolios/${portfolioId}/refresh-equity`) as Promise<any>,
    marketStatus: () =>
        client.get('/paper/market-status') as Promise<any>,
};

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
