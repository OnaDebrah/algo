// import {EquityCurvePoint} from "@/types/api";
//
// export interface BacktestConfig {
//     symbols: string[];
//     symbolInput: string;
//     period: '1mo' | '1y' | '5y';
//     interval: '1h' | '1d';
//     strategyMode: 'same' | 'different';
//     strategy: string;
//     initialCapital: number;
//     allocationMethod: 'equal' | 'custom' | 'risk-parity';
// }
//
// export interface MultiConfig {
//     symbols: string[];
//     symbolInput: string;
//     period: string;
//     interval: string;
//     strategyMode: 'same' | 'different';
//     strategy?: string;
//     strategies: Record<string, string>;
//     allocationMethod: string;
//     allocations: Record<string, number>;
//     initialCapital: number;
//     maxPositionPct: number;
// }
//
// export interface Strategy {
//     id: string;
//     name: string;
//     category: string;
//     params: Record<string, string | number | boolean>;
//     description: string;
//     complexity: 'Beginner' | 'Intermediate' | 'Advanced' | 'Expert' | 'Institutional';
//     time_horizon?: string;
//     best_for?: string[];
//     rating?: number;
//     monthly_return?: number;
//     drawdown?: number;
//     sharpe_ratio?: number;
// }
//
// export interface Trade {
//     id: number | string;
//     date?: string;
//     entry_time?: string;
//     exit_time?: string;
//     symbol: string;
//     type?: string;
//     side?: 'LONG' | 'SHORT';
//     strategy?: string;
//     quantity: number;
//     price?: number;
//     entry_price?: number;
//     exit_price?: number;
//     total?: number;
//     pnl: number;
//     status: 'open' | 'closed';
// }
//
// export interface BacktestResult {
//     type: 'single' | 'multi';
//     total_return: number;
//     win_rate: number;
//     sharpe_ratio: number;
//     max_drawdown: number;
//     total_trades: number;
//     final_equity: number;
//     equity_curve: Array<{
//         timestamp: string;
//         equity: number;
//         num_positions?: number;
//     }>;
//     trades?: Trade[];
//     avg_win?: number;
//     avg_loss?: number;
//     num_symbols?: number;
//     avg_profit?: number;
//     symbol_stats?: Record<string, {
//         strategy: string;
//         total_profit: number;
//         num_trades: number;
//         win_rate: number;
//         avg_profit: number;
//     }>;
// }
//
// export interface BacktestHistoryItem {
//   id: number;
//   name?: string | null;
//   backtest_type: string; // 'single', 'multi', 'options'
//   symbols: string[];
//   strategy_config: Record<string, any>; // JSON config
//   period: string;
//   interval: string;
//   initial_capital: number;
//
//   // Results (optional - only for completed)
//   total_return_pct?: number | null;
//   sharpe_ratio?: number | null;
//   max_drawdown?: number | null;
//   win_rate?: number | null;
//   total_trades?: number | null;
//   final_equity?: number | null;
//
//   // Metadata
//   status: string; // 'pending', 'running', 'completed', 'failed'
//   error_message?: string | null;
//   created_at?: string | null; // ISO date string
//   completed_at?: string | null; // ISO date string
// }
//
// // OptionsBacktestResponse (from your backend schema)
// export interface OptionsBacktestResponse {
//   result: OptionsBacktestResult;
//   equity_curve: EquityCurvePoint[];
//   trades: Trade[];
// }
//
// export interface OptionsBacktestResult extends BacktestResult {
//   avg_days_held: number;
//   avg_pnl_pct: number;
// }
