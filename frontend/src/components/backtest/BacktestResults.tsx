'use client'

import {useBacktestStore} from '@/lib/store/backtestStore'
import {Card, CardContent, CardHeader, CardTitle} from '@/components/ui/card'
import {Tabs, TabsContent, TabsList, TabsTrigger} from '@/components/ui/tabs'
import {formatCurrency, formatPercent, formatNumber} from '@/lib/utils/formatters'
import {TrendingUp, TrendingDown, Target, Activity} from 'lucide-react'
import {EquityCurveChart} from '@/components/charts/EquityCurveChart'
import {TradeLogTable} from '@/components/backtest/TradeLogTable'

export function BacktestResults() {
    const {result, equityCurve, trades} = useBacktestStore()

    if (!result) {
        return (
            <Card>
                <CardContent className="flex h-[600px] items-center justify-center">
                    <div className="text-center text-muted-foreground">
                        <Activity className="mx-auto mb-4 h-12 w-12 opacity-50"/>
                        <p className="text-lg font-medium">No backtest results</p>
                        <p className="text-sm">Configure and run a backtest to see results</p>
                    </div>
                </CardContent>
            </Card>
        )
    }

    return (
        <div className="space-y-4">
            {/* Key Metrics */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Total Return</CardTitle>
                        <TrendingUp className="h-4 w-4 text-muted-foreground"/>
                    </CardHeader>
                    <CardContent>
                        <div className={`text-2xl font-bold ${result.totalReturn >= 0 ? 'text-profit' : 'text-loss'}`}>
                            {formatPercent(result.totalReturn)}
                        </div>
                        <p className="text-xs text-muted-foreground">
                            {formatCurrency(result.finalEquity - result.initialCapital)}
                        </p>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
                        <Target className="h-4 w-4 text-muted-foreground"/>
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{formatPercent(result.winRate)}</div>
                        <p className="text-xs text-muted-foreground">
                            {result.winningTrades}/{result.totalTrades} trades
                        </p>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Sharpe Ratio</CardTitle>
                        <Activity className="h-4 w-4 text-muted-foreground"/>
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{formatNumber(result.sharpeRatio)}</div>
                        <p className="text-xs text-muted-foreground">Risk-adjusted return</p>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Max Drawdown</CardTitle>
                        <TrendingDown className="h-4 w-4 text-muted-foreground"/>
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold text-loss">
                            {formatPercent(result.maxDrawdown)}
                        </div>
                        <p className="text-xs text-muted-foreground">Peak to trough</p>
                    </CardContent>
                </Card>
            </div>

            {/* Tabs */}
            <Tabs defaultValue="equity" className="space-y-4">
                <TabsList>
                    <TabsTrigger value="equity">Equity Curve</TabsTrigger>
                    <TabsTrigger value="trades">Trade Log</TabsTrigger>
                    <TabsTrigger value="metrics">Detailed Metrics</TabsTrigger>
                </TabsList>

                <TabsContent value="equity">
                    <Card>
                        <CardHeader>
                            <CardTitle>Equity Curve</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="h-[400px]">
                                <EquityCurveChart data={equityCurve}/>
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="trades">
                    <Card>
                        <CardHeader>
                            <CardTitle>Trade Log ({trades.length} trades)</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <TradeLogTable trades={trades}/>
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="metrics">
                    <Card>
                        <CardHeader>
                            <CardTitle>Detailed Performance Metrics</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="grid gap-4 md:grid-cols-2">
                                <div className="space-y-3">
                                    <h4 className="font-semibold">Returns</h4>
                                    <div className="space-y-2 text-sm">
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Total Return:</span>
                                            <span className="font-medium">{formatPercent(result.totalReturn)}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Avg Profit:</span>
                                            <span className="font-medium">{formatCurrency(result.avgProfit)}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Avg Win:</span>
                                            <span
                                                className="font-medium text-profit">{formatCurrency(result.avgWin)}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Avg Loss:</span>
                                            <span
                                                className="font-medium text-loss">{formatCurrency(result.avgLoss)}</span>
                                        </div>
                                    </div>
                                </div>

                                <div className="space-y-3">
                                    <h4 className="font-semibold">Risk Metrics</h4>
                                    <div className="space-y-2 text-sm">
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Sharpe Ratio:</span>
                                            <span className="font-medium">{formatNumber(result.sharpeRatio)}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Max Drawdown:</span>
                                            <span
                                                className="font-medium text-loss">{formatPercent(result.maxDrawdown)}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Profit Factor:</span>
                                            <span className="font-medium">{formatNumber(result.profitFactor)}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Win Rate:</span>
                                            <span className="font-medium">{formatPercent(result.winRate)}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>
        </div>
    )
}
