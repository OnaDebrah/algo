export interface User {
    username: string;
    email: string;
    id: number;
    tier: string;
    is_active: boolean;
    created_at: string;
    last_login?: string | null;
}

export interface LoginResponse {
    user: User;
    access_token: string;
    refresh_token: string;
    token_type: string;
}

export interface BacktestRequest {
    symbol: string;
    strategy_key: string;
    parameters: Record<string, any>;
    period?: string;
    interval?: string;
    initial_capital?: number;
    commission_rate?: number;
    slippage_rate?: number;
}

export interface BacktestResult {
    total_return: number;
    total_return_pct: number;
    win_rate: number;
    sharpe_ratio: number;
    max_drawdown: number;
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    avg_profit: number;
    avg_win: number;
    avg_loss: number;
    profit_factor: number;
    final_equity: number;
    initial_capital: number;
}

export interface EquityCurvePoint {
    timestamp: string;
    equity: number;
    cash: number;
    drawdown?: number | null;
}

export interface BacktestTrade {
    id?: number | null;
    symbol: string;
    order_type: string;
    quantity: number;
    price: number;
    commission: number;
    timestamp: string;
    strategy: string;
    profit?: number | null;
    profit_pct?: number | null;
}

export interface BacktestResponse {
    result: BacktestResult;
    equity_curve: EquityCurvePoint[];
    trades: BacktestTrade[];
    price_data?: any[] | null;
}

export interface Portfolio {
    id: number;
    user_id: number;
    name: string;
    description?: string | null;
    initial_capital: number;
    current_capital: number;
    is_active: boolean;
    created_at: string;
    updated_at?: string | null;
    positions?: Position[] | null;
    recent_trades?: PortfolioTrade[] | null;
}

export interface PortfolioCreate {
    name: string;
    description?: string | null;
    initial_capital: number;
    current_capital?: number; // Optional as it might be set to initial
}

export interface Position {
    id: number;
    portfolio_id: number;
    symbol: string;
    quantity: number;
    avg_entry_price: number;
    current_price?: number | null;
    unrealized_pnl?: number | null;
    unrealized_pnl_pct?: number | null;
    market_value: number;
    created_at: string;
    updated_at?: string | null;
}

export interface PortfolioTrade {
    id: number | null;
    portfolio_id: number | null;
    symbol: string;
    order_type: string;
    quantity: number;
    price: number;
    commission: number;
    total_value: number;
    strategy: string | null;
    notes: string | null;
    executed_at: string | null;
    profit: number | null;
    profit_pct: number | null;
}

export interface Quote {
    symbol: string;
    price: number;
    change: number;
    changePct: number;
    volume: number;
    marketCap?: number | null;
    high?: number | null;
    low?: number | null;
    open?: number | null;
    previousClose?: number | null;
    timestamp: string;
}

export interface HistoricalData {
    symbol: string;
    data: any[];
}

export interface StrategyInfo {
    key: string;
    name: string;
    description: string;
    category: string;
    complexity?: string;
    time_horizon?: string;
    best_for?: string[];
    parameters: StrategyParameter[];
}

export interface StrategyParameter {
    name: string;
    type: string;
    default: any;
    min?: number | null;
    max?: number | null;
    description: string;
    options?: string[] | null;
}

// --- Analyst Types ---

export interface ValuationMetric {
    subject: string;
    score: number;
    benchmark: number;
    description: string;
}

export interface MACDData {
    value: number;
    signal: number;
    histogram: number;
}

export interface TechnicalData {
    rsi: number;
    rsi_signal: string;
    ma_20: number;
    ma_50: number;
    ma_200: number;
    support_levels: number[];
    resistance_levels: number[];
    trend_strength: number;
    macd: MACDData;
    volume_trend: string;
}

export interface FundamentalData {
    pe_ratio: number;
    pb_ratio: number;
    peg_ratio: number;
    debt_to_equity: number;
    roe: number;
    revenue_growth: number;
    eps_growth: number;
    profit_margin: number;
    dividend_yield: number;
}

export interface SentimentData {
    institutional: number;
    retail: number;
    analyst: number;
    news: number;
    social: number;
    options: number;
}

export interface RisksData {
    regulatory: string[];
    competitive: string[];
    market: string[];
    financial: string[];
    operational: string[];
}

export interface AnalystReport {
    company_name: string;
    ticker: string;
    recommendation: 'Strong Buy' | 'Buy' | 'Hold' | 'Sell' | 'Strong Sell';
    recommendation_confidence: number;
    current_price: number;
    target_price: number;
    upside: number;
    risk_rating: 'Low' | 'Medium' | 'High' | 'Very High';
    investment_thesis: string;
    sector: string;
    industry: string;
    market_cap: string;
    last_updated: string;
    valuation: ValuationMetric[];
    technical: TechnicalData;
    fundamental: FundamentalData;
    sentiment: SentimentData;
    risks: RisksData;
}

// Live Execution Types
export enum BrokerType {
    PAPER = "Paper Trading",
    ALPACA = "Alpaca Markets",
    IBKR = "Interactive Brokers"
}

export enum EngineStatus {
    IDLE = "idle",
    RUNNING = "running",
    PAUSED = "paused"
}

export interface ExecutionOrder {
    id: string;
    symbol: string;
    side: 'BUY' | 'SELL';
    qty: number;
    type: 'MARKET' | 'LIMIT' | 'STOP';
    status: 'PENDING' | 'FILLED' | 'CANCELLED' | 'REJECTED';
    price?: number;
    time: string;
}

export interface LiveStatus {
    is_connected: boolean;
    engine_status: EngineStatus;
    active_broker: BrokerType;
}

export interface ConnectRequest {
    broker: BrokerType;
    api_key?: string;
    api_secret?: string;
}
export interface StrategyListing {
    id: string;
    name: string;
    creator: string;
    description: string;
    rating: number;
    reviews: number;
    price: number;
    category: string;
    complexity: string;
    time_horizon: string;
    monthly_return: number;
    drawdown: number;
    sharpe_ratio: number;
    total_downloads: number;
    tags: string[];
    best_for: string[];
    pros: string[];
    cons: string[];
    is_favorite: boolean;
    is_verified: boolean;
    publish_date: string;
}

export interface MLModel {
    id: string;
    name: string;
    type: string;
    status: string;
    created_at: string;
    accuracy?: number;
    metrics?: Record<string, number>;
    features?: string[];
    target?: string;
}

export interface TrainingConfig {
    name: string;
    type: string;
    target: string;
    features: string[];
    parameters: Record<string, any>;
    test_split: number;
}

// Options Types
export interface OptionContract {
    strike: number;
    type: 'call' | 'put';
    expiration: string;
    premium: number;
    bid: number;
    ask: number;
    volume: number;
    openInterest: number;
    impliedVolatility: number;
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
}

export interface ChainRequest {
    symbol: string;
    expiration?: string;
}

export interface ChainResponse {
    symbol: string;
    underlying_price: number;
    expiration_dates: string[];
    calls: OptionContract[];
    puts: OptionContract[];
}

export interface OptionsBacktestRequest {
    symbol: string;
    strategy_type: string;
    initial_capital: number;
    risk_free_rate: number;
    start_date: string;
    end_date: string;
    entry_rules: Record<string, any>;
    exit_rules: Record<string, any>;
}

export interface OptionsBacktestResult {
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    win_rate: number;
    total_return: number;
    max_drawdown: number;
    sharpe_ratio: number;
    profit_factor: number;
    total_profit: number;
    total_loss: number;
    equity_curve: { date: string; equity: number }[];
    trades: { date: string; type: string; price: number; pnl: number; strategy: string }[];
}

// Regime Types
export interface RegimeMetrics {
    volatility: number;
    trend_strength: number;
    liquidity_score: number;
    correlation_index: number;
}

export interface RegimeData {
    id: string;
    name: string;
    description: string;
    start_date: string;
    end_date?: string | null;
    confidence: number;
    metrics: RegimeMetrics;
}

export interface CurrentRegimeResponse {
    symbol: string;
    current_regime: RegimeData;
    historical_regimes: RegimeData[];
    market_health_score: number;
}

// Settings Types
export interface BacktestSettings {
    data_source: string;
    slippage: number;
    commission: number;
    initial_capital: number;
}

export interface GeneralSettings {
    theme: string;
    notifications: boolean;
    auto_refresh: boolean;
    refresh_interval: number;
}

export interface UserSettings {
    user_id?: number;
    backtest: BacktestSettings;
    general: GeneralSettings;
}

export interface SettingsUpdate {
    backtest?: Partial<BacktestSettings>;
    general?: Partial<GeneralSettings>;
}
