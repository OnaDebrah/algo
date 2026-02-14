export interface PublishData {
    name: string;
    description: string;
    category: string;
    tags: string[];
    complexity: 'beginner' | 'intermediate' | 'advanced';
    price: number;
    pros: string[];
    cons: string[];
    risk_level: 'low' | 'medium' | 'high';
    recommended_capital: number;
}

export interface BacktestDataToPublish {
    id: number;
    name: string;
    strategy_key: string;
    symbols: string[];
    total_return_pct: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    total_trades: number;
    period: string;
    parameters?: Record<string, any> | undefined;
}
