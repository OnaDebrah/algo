import axios from 'axios';
import {
    LoginResponse,
    User,
    BacktestRequest,
    BacktestResponse,
    Portfolio,
    PortfolioCreate,
    Quote,
    StrategyInfo,
    HistoricalData,
    AnalystReport,
    LiveStatus,
    ConnectRequest,
    ExecutionOrder,
    StrategyListing,
    MLModel,
    TrainingConfig,
    OptionsBacktestRequest,
    ChainRequest,
    OptionsBacktestResult,
    ChainResponse,
    CurrentRegimeResponse,
    UserSettings,
    SettingsUpdate
} from '@/types/api.types';

// Use environment variable or default to localhost
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor to add auth token
api.interceptors.request.use(
    (config) => {
        if (typeof window !== 'undefined') {
            const token = localStorage.getItem('token');
            if (token) {
                config.headers.Authorization = `Bearer ${token}`;
            }
        }
        return config;
    },
    (error) => Promise.reject(error)
);

// Response interceptor to handle errors
api.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response?.status === 401) {
            // Handle unauthorized access (e.g., redirect to login)
            if (typeof window !== 'undefined') {
                localStorage.removeItem('token');
                // Optional: window.location.href = '/login';
            }
        }
        return Promise.reject(error);
    }
);

export const auth = {
    register: (data: any) => api.post<LoginResponse>('/auth/register', data),
    login: (data: any) => api.post<LoginResponse>('/auth/login', data),
    logout: () => api.post('/auth/logout'),
    getMe: () => api.get<User>('/auth/me'),
};

export const backtest = {
    runSingle: (data: BacktestRequest) => api.post<BacktestResponse>('/backtest/single', data),
    runMulti: (data: any) => api.post('/backtest/multi', data),
    runOptions: (data: any) => api.post('/backtest/options', data),
    getHistory: (params: any) => api.get('/backtest/history', { params }),
    getDetails: (id: number) => api.get(`/backtest/history/${id}`),
};

export const portfolio = {
    list: () => api.get<Portfolio[]>('/portfolio/'),
    create: (data: PortfolioCreate) => api.post<Portfolio>('/portfolio/', data),
    get: (id: number) => api.get<Portfolio>(`/portfolio/${id}`),
    update: (id: number, data: any) => api.put<Portfolio>(`/portfolio/${id}`, data),
    delete: (id: number) => api.delete(`/portfolio/${id}`),
    getMetrics: (id: number) => api.get(`/portfolio/${id}/metrics`),
    getPositions: (id: number) => api.get(`/portfolio/${id}/positions`),
    getTrades: (id: number) => api.get(`/portfolio/${id}/trades`),
};

export const market = {
    getQuote: (symbol: string) => api.get<Quote>(`/market/quote/${symbol}`),
    getQuotes: (symbols: string[]) => api.post<Quote[]>('/market/quotes', symbols),
    getHistorical: (symbol: string, params: any) => api.get<HistoricalData>(`/market/historical/${symbol}`, { params }),
    search: (query: string) => api.get(`/market/search`, { params: { q: query } }),
};

export const strategy = {
    list: () => api.get<StrategyInfo[]>('/strategy/list'),
    get: (key: string) => api.get<StrategyInfo>(`/strategy/${key}`),
};

export const advisor = {
    guide: (data: any) => api.post('/advisor/guide', data),
};

export const analyst = {
    getReport: (ticker: string) => api.get<AnalystReport>(`/analyst/report/${ticker}`),
};

export const marketplace = {
    getAll: (params?: any) => api.get<StrategyListing[]>('/marketplace/', { params }),
    getDetails: (id: number) => api.get<any>(`/marketplace/${id}`),
    favorite: (id: number) => api.post(`/marketplace/${id}/favorite`),
    unfavorite: (id: number) => api.delete(`/marketplace/${id}/favorite`),
    recordDownload: (id: number) => api.post(`/marketplace/${id}/download`),
};

export const mlstudio = {
    list: () => api.get<MLModel[]>('/mlstudio/models'),
    train: (config: object) => api.post<MLModel>('/mlstudio/train', config),
    deploy: (id: string) => api.post(`/mlstudio/deploy/${id}`),
    delete: (id: string) => api.delete(`/mlstudio/models/${id}`),
};

export const options = {
    // Existing endpoints
    getChain: (data: ChainRequest) => api.post<ChainResponse>('/options/chain', data),
    backtest: (data: OptionsBacktestRequest) => api.post<OptionsBacktestResult>('/options/backtest', data),

    // Strategy analysis endpoints
    analyze: (data: any) => api.post('/options/analyze', data),
    getGreeks: (data: any) => api.post('/options/greeks', data),
    compare: (data: any) => api.post('/options/compare', data),

    // Analytics endpoints
    calculateProbability: (data: any) => api.post('/options/analytics/probability', data),
    optimizeStrike: (data: any) => api.post('/options/analytics/optimize-strike', data),
    getRiskMetrics: (data: any) => api.post('/options/analytics/risk-metrics', data),
    getPortfolioStats: (data: any) => api.post('/options/analytics/portfolio-stats', data),
    runMonteCarlo: (data: any) => api.post('/options/analytics/monte-carlo', data),
};

export const regime = {
    detect: (symbol: string) => api.get<CurrentRegimeResponse>(`/regime/detect/${symbol}`),
};

export const settings = {
    get: () => api.get<UserSettings>('/settings/'),
    update: (data: SettingsUpdate) => api.put<UserSettings>('/settings/', data),
    reset: () => api.post<UserSettings>('/settings/reset'),
};

export const live = {
    getStatus: () => api.get<LiveStatus>('/live/status'),
    connect: (data: ConnectRequest) => api.post<LiveStatus>('/live/connect', data),
    disconnect: () => api.post<LiveStatus>('/live/disconnect'),
    getOrders: () => api.get<ExecutionOrder[]>('/live/orders'),
    placeOrder: (order: Partial<ExecutionOrder>) => api.post<ExecutionOrder>('/live/order', order),
    cancelOrder: (id: string) => api.delete(`/live/order/${id}`),
};

export const analytics = {
    getPerformance: (portfolioId: number, period: string) => api.get(`/analytics/performance/${portfolioId}`, { params: { period } }),
};

export default api;
