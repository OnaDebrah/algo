export interface BaseOptimizationRequest {
  symbols: string[];
  lookback_days?: number;
}

export interface SharpeOptimizationRequest extends BaseOptimizationRequest {
  risk_free_rate?: number;
}

export interface TargetReturnRequest extends BaseOptimizationRequest {
  target_return: number;
}

export interface BlackLittermanRequest extends BaseOptimizationRequest {
  views: Record<string, number>;
  confidence?: number;
}

export interface EfficientFrontierRequest extends BaseOptimizationRequest {
  num_portfolios?: number;
}

export interface PortfolioBacktestRequest {
  symbols: string[];
  weights: Record<string, number>;
  start_capital?: number;
  period?: string;
}

// Response Interfaces
export interface OptimizationResponse {
  weights: Record<string, number>;
  expected_return: number;
  volatility: number;
  sharpe_ratio: number;
  method: string;
}

export interface BlackLittermanResponse extends OptimizationResponse {
  views: Record<string, number>;
}

export interface FrontierPortfolio {
  'return': number;
  volatility: number;
  sharpe: number;
  weights: Record<string, number>;
}

export interface EfficientFrontierResponse {
  num_portfolios: number;
  portfolios: FrontierPortfolio[];
}

export interface StrategyComparison {
  max_sharpe: OptimizationResponse;
  min_volatility: OptimizationResponse;
  equal_weight: OptimizationResponse;
  risk_parity: OptimizationResponse;
}

export interface CompareStrategiesResponse {
  symbols: string[];
  lookback_days: number;
  strategies: StrategyComparison;
}

export interface OptimiseBacktestResponse {
  total_return: number;
  volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
  final_value: number;
}

export interface HealthCheckResponse {
  status: string;
  service: string;
  version: string;
}