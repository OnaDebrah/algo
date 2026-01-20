// api/index.ts
import axios from 'axios';
import {
    // Auth
    LoginResponse,
    User,
    UserCreate,
    UserLogin,

    // Backtest
    BacktestRequest,
    BacktestResponse,
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
    ConnectRequest,
    ExecutionOrder,

    // Marketplace
    StrategyListing,
    StrategyListingDetailed,
    StrategyPublishRequest,

    // ML Studio
    MLModel,
    TrainingConfig,

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
    HTTPValidationError, RegimeData
} from '@/types/all_types';

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

    // Format error response
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
  runSingle: (data: BacktestRequest) =>
    client.post<BacktestResponse>('/backtest/single', data),

  // Multi-asset backtest
  runMulti: (data: MultiAssetBacktestRequest) =>
    client.post<MultiAssetBacktestResponse>('/backtest/multi', data),

  // Options backtest
  runOptions: (data: OptionsBacktestRequest) =>
    client.post<OptionsBacktestResponse>('/backtest/options', data),

  getHistory: (params?: {
    limit?: number;
    offset?: number;
    backtest_type?: string;
    status?: string;
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
    client.get<Record<string, unknown>>(`/market/fundamentals/${symbol}`),

  getNews: (symbol: string, params?: NewsParams) =>
    client.get<Record<string, unknown>[]>(`/market/news/${symbol}`, { params }),

  search: (query: string, limit: number = 10) =>
    client.get<SearchResult[]>(`/market/search`, { params: { q: query, limit } }),

  validateSymbol: (symbol: string) =>
    client.get<{ valid: boolean; symbol: string; message?: string }>(`/market/validate/${symbol}`),

  getMarketStatus: () =>
    client.get<{ is_open: boolean; next_open?: string; next_close?: string }>(`/market/status`),

  getCacheStats: () =>
    client.get<Record<string, unknown>>(`/market/cache/stats`),

  clearCache: (data_type?: string) =>
    client.post<{ success: boolean; message: string }>(`/market/cache/clear`, null, { params: { data_type } }),

  cleanupCache: () =>
    client.post<{ success: boolean; cleaned: number }>(`/market/cache/cleanup`),
};

// ==================== STRATEGY ====================
export const strategy = {
  list: () => client.get<StrategyInfo[]>('/strategy/list'),

  get: (strategy_key: string) =>
    client.get<StrategyInfo>(`/strategy/${strategy_key}`),
};

// ==================== ANALYTICS ====================
export interface AnalyticsPeriod {
  period: string;
}

export const analytics = {
  getPerformance: (portfolio_id: number, params: AnalyticsPeriod) =>
    client.get<Record<string, any>>(`/analytics/performance/${portfolio_id}`, { params }),

  getReturnsAnalysis: (portfolio_id: number) =>
    client.get<Record<string, unknown>>(`/analytics/returns/${portfolio_id}`),

  getRiskMetrics: (portfolio_id: number) =>
    client.get<Record<string, unknown>>(`/analytics/risk/${portfolio_id}`),

  getDrawdownAnalysis: (portfolio_id: number) =>
    client.get<Record<string, unknown>>(`/analytics/drawdown/${portfolio_id}`),
};

// ==================== MARKET REGIME ====================
export interface RegimeParams {
  period?: string;
}

export const regime = {
  detect: (symbol: string, params?: RegimeParams) =>
    client.get<CurrentRegimeResponse>(`/regime/detect/${symbol}`, { params }),

  getHistory: (symbol: string, params?: RegimeParams) =>
    client.get<RegimeData[]>(`/regime/history/${symbol}`, { params }),

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
export interface AnalystReportParams {
  depth?: string;
}

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
};

// ==================== LIVE TRADING ====================
export interface LiveOrderPlacement {
  symbol: string;
  side: 'BUY' | 'SELL';
  qty: number;
  type: 'MARKET' | 'LIMIT' | 'STOP';
  price?: number;
  stop_price?: number;
}

export interface LiveOrderUpdate {
  price?: number;
  qty?: number;
  stop_price?: number;
}

export const live = {
  getStatus: () =>
    client.get<LiveStatus>('/live/status'),

  connect: (data: ConnectRequest) =>
    client.post<LiveStatus>('/live/connect', data),

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
};

// ==================== MARKETPLACE ====================
export interface MarketplaceFilterParams {
  category?: string;
  complexity?: string;
  min_sharpe?: number;
  min_return?: number;
  max_drawdown?: number;
  search?: string;
  sort_by?: string;
  limit?: number;
}

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

  publish: (data: StrategyPublishRequest) =>
    client.post<{ success: boolean; strategy_id: number; message: string }>(`/marketplace/publish`, data),
};

// ==================== ML STUDIO ====================
export interface ModelPrediction {
  prediction: number;
  confidence: number;
  timestamp: string;
}

export const mlstudio = {
  getModels: () =>
    client.get<MLModel[]>('/mlstudio/models'),

  trainModel: (config: TrainingConfig) =>
    client.post<MLModel>('/mlstudio/train', config),

  deployModel: (model_id: string) =>
    client.post<{ success: boolean; deployed: boolean; endpoint?: string }>(`/mlstudio/deploy/${model_id}`),

  deleteModel: (model_id: string) =>
    client.delete<{ success: boolean; message: string }>(`/mlstudio/models/${model_id}`),

  predictWithModel: (model_id: string) =>
    client.get<ModelPrediction>(`/mlstudio/models/${model_id}/predict`),
};

// ==================== OPTIONS ====================
// export const options = {
//   getChain: (data: ChainRequest) =>
//     client.post<ChainResponse>('/options/chain', data),
//
//   runBacktest: (data: OptionsBacktestRequest) =>
//     client.post<OptionsBacktestResponse>('/options/backtest', data),
//
//   analyzeStrategy: (data: StrategyAnalysisRequest) =>
//     client.post<StrategyAnalysisResponse>('/options/analyze', data),
//
//   calculateGreeks: (data: GreeksRequest) =>
//     client.post<GreeksResponse>('/options/greeks', data),
//
//   compareStrategies: (data: StrategyComparisonRequest) =>
//     client.post<StrategyComparisonResponse>('/options/compare', data),
//
//   calculateProbability: (data: ProbabilityRequest) =>
//     client.post<ProbabilityResponse>('/options/analytics/probability', data),
//
//   optimizeStrike: (data: StrikeOptimizerRequest) =>
//     client.post<StrikeOptimizerResponse>('/options/analytics/optimize-strike', data),
//
//   calculateRiskMetrics: (data: RiskMetricsRequest) =>
//     client.post<RiskMetricsResponse>('/options/analytics/risk-metrics', data),
//
//   calculatePortfolioStats: (data: PortfolioStatsRequest) =>
//     client.post<PortfolioStatsResponse>('/options/analytics/portfolio-stats', data),
//
//   runMonteCarlo: (data: MonteCarloRequest) =>
//     client.post<MonteCarloResponse>('/options/analytics/monte-carlo', data),
// };

export const options = {
  // Chain endpoint
  getChain: async (request: { symbol: string; expiration?: string }) => {
    return client.post('/options/chain', request);
  },

  // Backtest endpoint
  runBacktest: async (request: any) => {
    return client.post('/options/backtest', request);
  },

  // Strategy analysis
  analyzeStrategy: async (request: any) => {
    return client.post('/options/analyze', request);
  },

  // Greeks calculation
  calculateGreeks: async (request: any) => {
    return client.post('/options/greeks', request);
  },

  // Strategy comparison
  compareStrategies: async (request: any) => {
    return client.post('/options/compare', request);
  },

  // Risk metrics
  calculateRiskMetrics: async (request: any) => {
    return client.post('/options/analytics/risk-metrics', request);
  },

  // Monte Carlo simulation
  runMonteCarlo: async (request: any) => {
    return client.post('/options/analytics/monte-carlo', request);
  },

  // Strike optimization
  optimizeStrike: async (request: any) => {
    return client.post('/options/analytics/optimize-strike', request);
  },

  // Portfolio statistics
  calculatePortfolioStats: async (request: any) => {
    return client.post('/options/analytics/portfolio-stats', request);
  },

  // Probability calculation
  calculateProbability: async (request: any) => {
    return client.post('/options/analytics/probability', request);
  }
};

// ==================== SETTINGS ====================
export const settings = {
  get: () =>
    client.get<UserSettings>('/settings/'),

  update: (data: SettingsUpdate) =>
    client.put<UserSettings>('/settings/', data),

  reset: () =>
    client.post<UserSettings>('/settings/reset'),
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