export enum BrokerType {
    PAPER = 'paper',
    ALPACA_PAPER = 'alpaca_paper',
    ALPACA_LIVE = 'alpaca_live',
    IB_PAPER = 'ib_paper',
    IB_LIVE = 'ib_live'
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

export interface LiveStrategy {
    id: number;
    name: string;
    strategy_key: string;
    symbols: string[];
    status: 'RUNNING' | 'PAUSED' | 'STOPPED' | 'ERROR';
    deployment_mode: 'paper' | 'live';
    current_equity: number;
    initial_capital: number;
    total_return: number;
    total_return_pct: number;
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    sharpe_ratio: number | null;
    max_drawdown: number;
    daily_pnl: number;
    deployed_at: string;
    last_trade_at: string | null;
    broker: string;
}

export interface LiveEquityPoint {
    timestamp: string;
    equity: number;
    cash: number;
    daily_pnl: number;
    total_pnl: number;
    drawdown_pct: number;
}

export interface LiveTrade {
    id: number;
    symbol: string;
    side: 'BUY' | 'SELL';
    quantity: number;
    entry_price: number | null;
    exit_price: number | null;
    status: 'open' | 'closed';
    profit: number | null;
    profit_pct: number | null;
    opened_at: string;
    closed_at: string | null;
}

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

export interface StrategyPerformance {
    id: number;
    name: string;
    status: string;
    deployment_mode: string;
    current_equity: number;
    initial_capital: number;
    total_return: number;
    total_return_pct: number;
    daily_pnl: number;
    sharpe_ratio: number | null;
    max_drawdown: number;
    total_trades: number;
    win_rate: number;
}

export interface PortfolioMetrics {
    total_equity: number;
    total_invested: number;
    total_pnl: number;
    total_pnl_pct: number;
    active_strategies: number;
    total_strategies: number;
    best_performer: StrategyPerformance | null;
    worst_performer: StrategyPerformance | null;
}

/**
 * Full details including history and backtest comparison
 */
export interface StrategyDetailsResponse {
    strategy: LiveStrategy;
    equity_curve: LiveEquityPoint[];
    current_equity: number;
    initial_capital: number;
    trades: LiveTrade[];
    backtest_comparison: {
        backtest_return_pct: number;
        live_return_pct: number;
        backtest_sharpe: number;
        live_sharpe: number;
        backtest_max_drawdown: number;
        live_max_drawdown: number;
    } | null;
}

/**
 * Request body for updating strategy parameters
 */
export interface StrategyUpdateRequest {
    parameters?: Record<string, any>;
    max_position_pct?: number;
    stop_loss_pct?: number;
    daily_loss_limit?: number;
    notes?: string;
}

/**
 * Response after a successful update
 */
export interface UpdateResponse {
    strategy_id: number;
    version: number;
    updated_at: string;
    message: string;
}

/**
 * Response after a control action (start/stop/pause)
 */
export interface ControlResponse {
    strategy_id: number;
    status: 'running' | 'paused' | 'stopped';
    action: string;
    message: string;
    timestamp: string;
}

export interface AccountResponse {
    cash: number;
    equity: number;
    buying_power: number;
    margin_used: number;
    unrealized_pnl: number;
}
