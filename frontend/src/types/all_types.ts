// ==================== BASE TYPES ====================
export type ScalarParam = string | number | boolean;

export interface ValidationError {
    loc: Array<string | number>;
    msg: string;
    type: string;
}

export interface HTTPValidationError {
    detail: ValidationError[];
}

// ==================== AUTHENTICATION ====================

export interface User {
    username: string;
    email: string;
    id: number;
    tier: string;
    is_active: boolean;
    created_at: string;
    last_login: string | null;
}

export interface UserCreate {
    username: string;
    email: string;
    password: string;
}

export interface UserLogin {
    email: string;
    password: string;
}

export interface LoginResponse {
    user: User;
    access_token: string;
    refresh_token: string;
    token_type?: string;
}

// ==================== STRATEGY ====================

export interface StrategyParameter {
    name: string;
    type: string;
    default: any;
    min: number | null;
    max: number | null;
    description: string;
    options: string[] | null;
}

export interface StrategyInfo {
    key: string;
    name: string;
    description: string;
    category: string;
    complexity: string | null;
    time_horizon: string | null;
    best_for: string[];
    historical_return: number;
    total_trades: number;
    parameters: StrategyParameter[];
}

export interface StrategyConfig {
    strategy_type: string;
    parameters: Record<string, any>;
}

// ==================== BACKTEST ====================
export interface SingleBacktestRequest {
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
    num_symbols?: number;
    equity_curve?: EquityCurvePoint[];
    trades?: Trade[];
    price_data?: Record<string, any>[] | null;
    symbol_stats?: Record<string, SymbolStats>
}

export interface EquityCurvePoint {
    timestamp: string;
    equity: number;
    cash: number;
    drawdown: number | null;
}

export interface Trade {
    id: number | null;
    symbol: string;
    order_type: string;
    quantity: number;
    price: number;
    commission: number;
    timestamp: string;
    strategy: string;
    profit: number | null;
    profit_pct: number | null;
}

export interface SingleBacktestResponse {
    result: BacktestResult;
    equity_curve: EquityCurvePoint[];
    trades: Trade[];
    price_data: Record<string, any>[] | null;
}

export interface SingleAssetConfig {
    symbol: string;
    period: string;
    interval: string;
    strategy: string;
    initialCapital: number;
    maxPositionPct: number;
    params: Record<string, any>
    riskLevel?: string;
    commission?: number;
}

// ==================== MULTI-ASSET BACKTEST ====================
export interface MultiAssetConfig {
    symbols: string[];
    symbolInput: string;
    period: string;
    interval: string;
    strategyMode: 'same' | 'different' | 'portfolio';
    params: Record<string, any>;
    strategy: string;
    strategies: Record<string, string>;
    allocationMethod: string;
    allocations: Record<string, number>;
    initialCapital: number;
    maxPositionPct: number;
    riskLevel: string;
}

export interface Strategy {
    id: string;
    name: string;
    category: string;
    parameters: Record<string, any>;
    description: string;
    complexity: 'Beginner' | 'Intermediate' | 'Advanced' | 'Expert' | 'Institutional';
    time_horizon: string | null;
    best_for?: string[];
    rating?: number;
    monthly_return?: number;
    drawdown?: number;
    sharpe_ratio?: number;
}

export interface SymbolStats {
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    total_return: number;
    win_rate: number;
    avg_profit: number;
}

export interface MultiAssetBacktestResult {
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
    symbol_stats: Record<string, SymbolStats>;
    num_symbols: number;
}

export interface MultiAssetBacktestRequest {
    symbols: string[];
    strategy_configs?: Record<string, StrategyConfig>;
    allocation_method?: string;
    custom_allocations: Record<string, number> | null;
    period?: string;
    interval?: string;
    initial_capital?: number;
    commission_rate?: number;
    slippage_rate?: number;
}

export interface MultiAssetBacktestResponse {
    result: MultiAssetBacktestResult;
    equity_curve: EquityCurvePoint[];
    trades: Trade[];
}

// ==================== OPTIONS BACKTEST ====================

export interface OptionsBacktestRequest {
    symbol: string;
    strategy_type: string;
    entry_rules: Record<string, any>;
    exit_rules: Record<string, any>;
    start_date?: string,
    end_date?: string
    period?: string;
    risk_free_rate?: number;
    interval?: string;
    initial_capital?: number;
    volatility?: number;
    commission?: number;
}

export interface OptionsBacktestResult {
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
    avg_days_held: number;
    avg_pnl_pct: number;
}

export interface OptionsBacktestResponse {
    result: OptionsBacktestResult;
    equity_curve: EquityCurvePoint[];
    trades: Trade[];
}

// ==================== BACKTEST HISTORY ====================

export interface BacktestHistoryItem {
    id: number;
    name: string | null;
    backtest_type: string;
    symbols: string[];
    strategy_config: Record<string, any>;
    period: string;
    interval: string;
    initial_capital: number;
    total_return_pct: number | null;
    sharpe_ratio: number | null;
    max_drawdown: number | null;
    win_rate: number | null;
    total_trades: number | null;
    final_equity: number | null;
    status: string;
    error_message: string | null;
    created_at: string | null;
    completed_at: string | null;
}

// ==================== PORTFOLIO ====================

export interface PortfolioCreate {
    name: string;
    description: string | null;
    initial_capital: number;
}

export interface PortfolioUpdate {
    name?: string | null;
    description?: string | null;
    is_active?: boolean | null;
}

export interface Position {
    id: number;
    portfolio_id: number;
    symbol: string;
    quantity: number;
    avg_entry_price: number;
    current_price: number | null;
    unrealized_pnl: number | null;
    unrealized_pnl_pct: number | null;
    market_value: number;
    created_at: string;
    updated_at: string | null;
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

export interface PortfolioMetrics {
    nav: number;
    prev_nav: number;
    exposure: number;
    unrealized_pnl: number;
    cash: number;
    total_value: number;
    daily_return: number;
    daily_return_pct: number;
}

export interface Portfolio {
    id: number;
    user_id: number;
    name: string;
    description: string | null;
    initial_capital: number;
    current_capital: number;
    is_active: boolean;
    created_at: string;
    updated_at: string | null;
    positions: Position[] | null;
    recent_trades: PortfolioTrade[] | null;
}

// ==================== MARKET DATA ====================

export interface QuoteData {
    symbol: string;
    price: number;
    change: number;
    changePercent: number;
    volume: number;
    marketCap: number | null;
    // Add other fields as needed
}

export interface HistoricalDataPoint {
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

export interface OptionsChain {
    symbol: string;
    underlying_price: number;
    expiration_dates: string[];
    calls: Record<string, any>[];
    puts: Record<string, any>[];
}

export interface SearchResult {
    symbol: string;
    name: string;
    type: string;
    exchange: string;
}

// ==================== MARKET REGIME ====================

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
    end_date: string | null;
    confidence: number;
    metrics: RegimeMetrics;
}

export interface CurrentRegimeResponse {
    symbol: string;
    current_regime: RegimeData;
    historical_regimes: RegimeData[];
    market_health_score: number;
}

export interface StrategyAllocation {
    trend_following: number;
    momentum: number;
    volatility_strategies: number;
    mean_reversion: number;
    statistical_arbitrage: number;
    cash: number;
}

export interface AllocationResponse {
    symbol: string;
    current_regime: string;
    confidence: number;
    allocation: StrategyAllocation;
    timestamp: string;
}

export interface RegimeStrengthResponse {
    symbol: string;
    current_regime: string;
    strength: number;
    confirming_signals: number;
    total_signals: number;
    description: string;
    timestamp: string;
}

export interface TransitionProbability {
    from_regime: string;
    to_regime: string;
    probability: number;
}

export interface TransitionResponse {
    symbol: string;
    current_regime: string;
    expected_duration: number;
    median_duration: number;
    probability_end_next_week: number;
    likely_transitions: TransitionProbability[];
    timestamp: string;
}

export interface FeatureImportance {
    feature: string;
    importance: number;
    current_value: number;
}

export interface FeaturesResponse {
    symbol: string;
    current_regime: string;
    top_features: FeatureImportance[];
    timestamp: string;
}

// ==================== ANALYST & ADVISOR ====================

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
    recommendation: string;
    recommendation_confidence: number;
    current_price: number;
    target_price: number;
    upside: number;
    risk_rating: string;
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

export interface Theme {
    primary: string;
    secondary: string;
    bg: string;
    border: string;
    text: string;
    icon: string;
}

export interface Recommendation {
    id: number;
    name: string;
    tagline: string;
    description: string;
    fit_score: number;
    risk_level: string;
    theme: Theme;
    expected_return: string;
    similar_traders: string;
    time_commitment: string;
    success_rate: string;
    min_capital: number;
    why: string[];
    pros: string[];
    cons: string[];
    best_for: string[];
    performance_data: Record<string, any>[];
    allocation_data: Record<string, any>[];
    tags: string[];
    icon: string;
}

export interface GuideRequest {
    goal: string;
    risk: string;
    experience: string;
    capital: number;
    timeHorizon: string;
    markets: string[];
}

export interface GuideResponse {
    recommendations: Recommendation[];
    radar_data: Record<string, any>[];
}

// ==================== ALERTS ====================

export interface EmailAlertRequest {
    subject: string;
    message: string;
    to_email: string | null;
}

export interface SMSAlertRequest {
    message: string;
    to_number: string | null;
}

export interface AlertTestResponse {
    success: boolean;
    message: string;
    email_status: string;
    sms_status: string;
}

// ==================== LIVE TRADING ====================

export enum BrokerType {
    PAPER_TRADING = "Paper Trading",
    ALPACA = "Alpaca Markets",
    INTERACTIVE_BROKERS = "Interactive Brokers"
}

export enum EngineStatus {
    IDLE = "idle",
    RUNNING = "running",
    PAUSED = "paused"
}

export enum OrderSide {
    BUY = "BUY",
    SELL = "SELL"
}

export enum OrderType {
    MARKET = "MARKET",
    LIMIT = "LIMIT",
    STOP = "STOP"
}

export enum OrderStatus {
    PENDING = "PENDING",
    FILLED = "FILLED",
    CANCELLED = "CANCELLED",
    REJECTED = "REJECTED"
}

export interface ConnectRequest {
    broker: BrokerType;
    api_key: string | null;
    api_secret: string | null;
}

export interface ExecutionOrder {
    id: string;
    symbol: string;
    side: OrderSide;
    qty: number;
    type: OrderType;
    status: OrderStatus;
    price: number | null;
    time: string;
}

export interface LiveStatus {
    is_connected: boolean;
    engine_status: EngineStatus;
    active_broker: BrokerType;
}

// ==================== MARKETPLACE ====================

export interface BacktestResultsSchema {
    total_return: number;
    annualized_return: number;
    sharpe_ratio: number;
    sortino_ratio: number;
    max_drawdown: number;
    max_drawdown_duration: number;
    calmar_ratio: number;
    num_trades: number;
    win_rate: number;
    profit_factor: number;
    avg_win: number;
    avg_loss: number;
    avg_trade_duration: number;
    volatility: number;
    var_95: number;
    cvar_95: number;
    equity_curve: Record<string, any>[];
    trades: Record<string, any>[];
    daily_returns: Record<string, any>[];
    start_date: string | null;
    end_date: string | null;
    initial_capital: number;
    symbols: string[];
}

export interface StrategyReviewSchema {
    id: number | null;
    strategy_id: number;
    user_id: number;
    username: string;
    rating: number;
    review_text: string;
    performance_achieved: Record<string, any> | null;
    created_at: string;
}

export interface StrategyListing {
    id: string | number;
    name: string;
    creator: string;
    description: string;
    rating: number;
    reviews: number;
    price: number;
    category: string;
    complexity: string;
    time_horizon?: string;
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

export interface StrategyListingDetailed extends StrategyListing {
    backtest_results: BacktestResultsSchema | null;
    reviews_list: StrategyReviewSchema[];
}

export interface StrategyPublishRequest {
    name: string;
    description: string;
    category: string;
    complexity: string;
    price: number;
    is_public?: boolean;
    tags?: string[];
    backtest_id?: number | null;
}

// ==================== ML STUDIO ====================

export interface MLFeatureImportance {
    feature: string;
    importance: number;
}

export interface MLModel {
    id: string;
    name: string;
    type: string;
    symbol: string;
    accuracy: number;
    test_accuracy: number;
    overfit_score: number;
    features: number;
    training_time: number;
    created: string;
    status: string;
    feature_importance: MLFeatureImportance[];
    hyperparams: Record<string, any>;
}

export interface TrainingConfig {
    symbol: string;
    model_type: string;
    training_period: string;
    test_size: number;
    epochs: number;
    batch_size: number;
    learning_rate: number;
    threshold: number;
    use_feature_engineering: boolean;
    use_cross_validation: boolean;
}

// ==================== OPTIONS ANALYTICS ====================

export interface OptionLegRequest {
    option_type: string;
    strike: number;
    expiration: string;
    quantity: number;
    premium: number | null;
}

export interface ChainRequest {
    symbol: string;
    expiration: string | null;
    includeGreeks?: boolean,
    includeIV?: boolean
}

export interface ChainResponse {
    symbol: string;
    current_price: number;
    expiration_dates: string[];
    calls: Record<string, any>[];
    puts: Record<string, any>[];
}

export interface GreeksRequest {
    symbol: string;
    legs: OptionLegRequest[];
    volatility: number | null;
}

export interface GreeksResponse {
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
    rho: number;
}

export interface PayoffPoint {
    price: number;
    payoff: number;
}

export interface StrategyAnalysisRequest {
    symbol: string;
    legs: OptionLegRequest[];
    volatility: number | null;
}

export interface StrategyAnalysisResponse {
    symbol: string;
    current_price: number;
    initial_cost: number;
    greeks: GreeksResponse;
    breakeven_points: number[];
    max_profit: number;
    max_profit_condition: string;
    max_loss: number;
    max_loss_condition: string;
    probability_of_profit: number;
    payoff_diagram: PayoffPoint[];
}

export interface StrategyComparisonRequest {
    symbol: string;
    strategies: Record<string, any>[];
}

export interface StrategyComparisonResponse {
    symbol: string;
    current_price: number;
    comparisons: Record<string, any>[];
}

export interface ProbabilityRequest {
    current_price: number;
    strike: number;
    days_to_expiration: number;
    volatility: number;
    risk_free_rate?: number;
    option_type: string;
}

export interface ProbabilityResponse {
    probability_itm: number;
    probability_otm: number;
    probability_touch: number;
    expected_return_long: number;
    expected_return_short: number;
}

export interface StrikeAnalysis {
    strike: number;
    moneyness: number;
    premium_estimate: number;
    prob_itm: number;
    prob_otm: number;
    expected_return: number;
}

export interface StrikeOptimizerRequest {
    symbol: string;
    current_price: number;
    volatility: number;
    days_to_expiration: number;
    strategy_type: string;
    num_strikes?: number;
}

export interface StrikeOptimizerResponse {
    symbol: string;
    strategy_type: string;
    current_price: number;
    strikes: StrikeAnalysis[];
}

export interface PortfolioPosition {
    pnl: number;
    pnl_pct: number;
    days_held: number;
}

export interface PortfolioStatsRequest {
    positions: PortfolioPosition[];
}

export interface PortfolioStatsResponse {
    total_pnl: number;
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    win_rate: number;
    avg_win: number;
    avg_loss: number;
    largest_win: number;
    largest_loss: number;
    profit_factor: number;
    avg_days_held: number;
    avg_return_pct: number;
    std_return_pct: number;
    kelly_fraction: number;
    expectancy: number;
}

export interface RiskMetricsRequest {
    portfolio_value: number;
    returns: number[];
    confidence_level?: number;
}

export interface RiskMetricsResponse {
    var_95: number;
    cvar_95: number;
    kelly_fraction: number | null;
    recommendation: string;
}

export interface MonteCarloRequest {
    current_price: number;
    volatility: number;
    days: number;
    num_simulations?: number;
    drift?: number;
}

export interface MonteCarloResponse {
    mean_final_price: number;
    median_final_price: number;
    std_final_price: number;
    percentile_5: number;
    percentile_95: number;
    probability_above_current: number;
    simulated_prices: number[];
}

export interface OptionLeg {
    id: string;
    type: 'call' | 'put' | 'stock'
    position: 'long' | 'short';
    strike: number;
    quantity: number;
    expiration: string;
    premium?: number;
    iv?: number;
    delta?: number;
    gamma?: number;
    theta?: number;
    vega?: number;
}

export interface MLForecast {
    direction: 'bullish' | 'bearish' | 'neutral';
    confidence: number;
    suggestedStrategies: string[];
    priceTargets?: {
        low: number;
        median: number;
        high: number;
    };
    timeline?: {
        short: string;
        medium: string;
        long: string;
    };
}

export interface StrategyTemplate {
    id: string;
    name: string;
    description: string;
    risk: string;
    sentiment: string;
    icon: any;
    legs: OptionLeg[];
    typicalSetup?: {
        strikes: number[];
        expirations: string[];
    };
}

export interface BacktestConfig {
    symbol: string;
    strategy_type: string;
    initial_capital: number;
    risk_free_rate: number;
    start_date: string;
    end_date: string;
    entry_rules: Record<string, any>;
    exit_rules: Record<string, any>;
}

// ==================== SETTINGS ====================

export interface BacktestSettings {
    data_source?: string;
    slippage?: number;
    commission?: number;
    initial_capital?: number;
}

export interface GeneralSettings {
    theme?: string;
    notifications?: boolean;
    auto_refresh?: boolean;
    refresh_interval?: number;
}

export interface UserSettings {
    user_id: number | null;
    backtest: BacktestSettings;
    general: GeneralSettings;
}

export interface SettingsUpdate {
    backtest?: BacktestSettings | null;
    general?: GeneralSettings | null;
}

// ==================== API RESPONSE TYPES ====================

export interface ApiResponse<T> {
    data: T;
    status: number;
    message?: string;
}

export interface PaginatedResponse<T> {
    items: T[];
    total: number;
    page: number;
    size: number;
    pages: number;
}

// ==================== FRONTEND SPECIFIC TYPES ====================

export interface ChartDataPoint {
    x: string | number;
    y: number;

    [key: string]: any;
}

export interface TimeSeriesData {
    timestamp: string;
    value: number;
}

export interface DashboardMetrics {
    totalPortfolios: number;
    activePositions: number;
    totalPnl: number;
    todayPnl: number;
    winRate: number;
    sharpeRatio: number;
}

export interface NavigationItem {
    id: string;
    label: string;
    icon: string;
    path: string;
    children?: NavigationItem[];
}

export interface BreadcrumbItem {
    label: string;
    path: string;
}

// ==================== FILTER TYPES ====================

export interface BacktestFilter {
    backtest_type?: string;
    status?: string;
    symbol?: string;
    startDate?: string;
    endDate?: string;
}

export interface MarketplaceFilter {
    category?: string;
    complexity?: string;
    min_sharpe?: number;
    min_return?: number;
    max_drawdown?: number;
    search?: string;
    sort_by?: string;
}

// ==================== FORM TYPES ====================

export interface LoginFormData {
    email: string;
    password: string;
    rememberMe?: boolean;
}

export interface RegisterFormData {
    username: string;
    email: string;
    password: string;
    confirmPassword: string;
}

export interface BacktestFormData {
    symbol: string;
    strategyKey: string;
    parameters: Record<string, any>;
    period: string;
    interval: string;
    initialCapital: number;
    commissionRate: number;
    slippageRate: number;
}

export interface PortfolioFormData {
    name: string;
    description: string;
    initialCapital: number;
}

// ==================== STATE TYPES ====================

export interface AuthState {
    user: User | null;
    token: string | null;
    refreshToken: string | null;
    isLoading: boolean;
    error: string | null;
}

export interface BacktestState {
    currentBacktest: SingleBacktestResponse | null;
    history: BacktestHistoryItem[];
    isLoading: boolean;
    error: string | null;
}

export interface PortfolioState {
    portfolios: Portfolio[];
    currentPortfolio: Portfolio | null;
    isLoading: boolean;
    error: string | null;
}

export interface MarketDataState {
    quotes: Record<string, QuoteData>;
    historicalData: Record<string, HistoricalDataPoint[]>;
    isLoading: boolean;
    error: string | null;
}

export interface UIState {
    theme: 'light' | 'dark';
    sidebarOpen: boolean;
    notifications: any[];
    isLoading: boolean;
}

// ==================== EVENT TYPES ====================

export interface BacktestCompletedEvent {
    backtestId: number;
    result: BacktestResult;
    timestamp: string;
}

export interface TradeExecutedEvent {
    portfolioId: number;
    symbol: string;
    quantity: number;
    price: number;
    side: OrderSide;
    timestamp: string;
}

export interface AlertEvent {
    type: 'info' | 'warning' | 'error' | 'success';
    title: string;
    message: string;
    timestamp: string;
}

// ==================== UTILITY TYPES ====================

export type Optional<T, K extends keyof T> = Pick<Partial<T>, K> & Omit<T, K>;
export type Nullable<T> = { [K in keyof T]: T[K] | null };
export type DeepPartial<T> = {
    [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

// ==================== CONSTANTS ====================

export const BACKTEST_PERIODS = [
    '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
] as const;

export const BACKTEST_INTERVALS = [
    '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
] as const;

export const STRATEGY_CATEGORIES = [
    'Trend Following',
    'Mean Reversion',
    'Momentum',
    'Volatility',
    'Arbitrage',
    'Options',
    'Machine Learning'
] as const;

export const COMPLEXITY_LEVELS = [
    'Beginner',
    'Intermediate',
    'Advanced',
    'Expert'
] as const;

export const TIME_HORIZONS = [
    'Intraday',
    'Short-term',
    'Medium-term',
    'Long-term'
] as const;

// ==================== API ENDPOINT TYPES ====================

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