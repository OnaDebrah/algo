import {Trade} from "@/types/portfolio";
import {StrategyConfig} from "@/types/strategy";

export interface BacktestRequest {
  symbol: string
  strategyKey: string
  parameters: Record<string, any>
  period: string
  interval: string
  initialCapital: number
  commissionRate: number
  slippageRate: number
}

export interface BacktestResult {
  totalReturn: number
  totalReturnPct: number
  winRate: number
  sharpeRatio: number
  maxDrawdown: number
  totalTrades: number
  winningTrades: number
  losingTrades: number
  avgProfit: number
  avgWin: number
  avgLoss: number
  profitFactor: number
  finalEquity: number
  initialCapital: number
}

export interface EquityCurvePoint {
  timestamp: string
  equity: number
  cash: number
  drawdown?: number
}

export interface BacktestResponse {
  result: BacktestResult
  equityCurve: EquityCurvePoint[]
  trades: Trade[]
  priceData: PriceData[]
}

export interface PriceData {
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

// Multi-asset backtest
export interface MultiAssetBacktestRequest {
  symbols: string[]
  strategyConfigs: Record<string, StrategyConfig>
  allocationMethod: 'equal' | 'custom' | 'risk_parity'
  customAllocations?: Record<string, number>
  period: string
  interval: string
  initialCapital: number
  commissionRate: number
  slippageRate: number
}

export interface MultiAssetBacktestResult extends BacktestResult {
  symbolStats: Record<string, SymbolStats>
  numSymbols: number
}

export interface SymbolStats {
  totalProfit: number
  numTrades: number
  winRate: number
  avgProfit: number
  strategy: string
}

// Options backtest
export interface OptionsBacktestRequest {
  symbol: string
  strategyType: string
  entryRules: Record<string, any>
  exitRules: Record<string, any>
  period: string
  interval: string
  initialCapital: number
  volatility: number
  commission: number
}

export interface OptionsBacktestResult extends BacktestResult {
  avgDaysHeld: number
  avgPnlPct: number
}
