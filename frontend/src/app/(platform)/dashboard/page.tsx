'use client'

import { useEffect } from 'react'
import { usePortfolio } from '@/lib/hooks/usePortfolio'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { formatCurrency, formatPercent } from '@/lib/utils/formatters'
import { TrendingUp, TrendingDown, DollarSign, Activity, Target, AlertCircle } from 'lucide-react'
import { EquityCurveChart } from '@/components/charts/EquityCurveChart'
import { PositionsTable } from '@/components/portfolio/PositionsTable'
import { RecentTradesTable } from '@/components/portfolio/RecentTradesTable'

export default function DashboardPage() {
  const { portfolio, metrics, positions, trades, isLoading } = usePortfolio('default')

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Portfolio Dashboard</h1>
        <p className="text-muted-foreground">
          Monitor your trading performance and positions
        </p>
      </div>

      {/* Key Metrics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Value</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metrics ? formatCurrency(metrics.totalValue) : '--'}
            </div>
            <p className="text-xs text-muted-foreground">
              {metrics && formatPercent(((metrics.totalValue - metrics.prevNav) / metrics.prevNav) * 100)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Unrealized P&L</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${metrics && metrics.unrealizedPnl >= 0 ? 'text-profit' : 'text-loss'}`}>
              {metrics ? formatCurrency(metrics.unrealizedPnl) : '--'}
            </div>
            <p className="text-xs text-muted-foreground">
              From {positions?.length || 0} positions
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Exposure</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metrics ? formatCurrency(metrics.exposure) : '--'}
            </div>
            <p className="text-xs text-muted-foreground">
              {metrics && formatPercent((metrics.exposure / metrics.totalValue) * 100)} of portfolio
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Available Cash</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metrics ? formatCurrency(metrics.cash) : '--'}
            </div>
            <p className="text-xs text-muted-foreground">
              {metrics && formatPercent((metrics.cash / metrics.totalValue) * 100)} of portfolio
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="positions">Positions</TabsTrigger>
          <TabsTrigger value="trades">Trade History</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Equity Curve</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[400px]">
                <EquityCurveChart data={[]} />
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Top Performing Positions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {positions?.slice(0, 5).map((position) => (
                    <div key={position.id} className="flex items-center justify-between">
                      <div>
                        <div className="font-medium">{position.symbol}</div>
                        <div className="text-sm text-muted-foreground">
                          {position.quantity} shares @ {formatCurrency(position.entryPrice)}
                        </div>
                      </div>
                      <div className={`text-right ${position.unrealizedPnl >= 0 ? 'text-profit' : 'text-loss'}`}>
                        <div className="font-medium">{formatCurrency(position.unrealizedPnl)}</div>
                        <div className="text-sm">{formatPercent(position.unrealizedPnlPct)}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Recent Trades</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {trades?.slice(0, 5).map((trade) => (
                    <div key={trade.id} className="flex items-center justify-between">
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{trade.symbol}</span>
                          <span className={`text-xs px-2 py-0.5 rounded ${trade.orderType === 'BUY' ? 'bg-profit/20 text-profit' : 'bg-loss/20 text-loss'}`}>
                            {trade.orderType}
                          </span>
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {trade.quantity} @ {formatCurrency(trade.price)}
                        </div>
                      </div>
                      {trade.profit !== undefined && (
                        <div className={`text-right ${trade.profit >= 0 ? 'text-profit' : 'text-loss'}`}>
                          <div className="font-medium">{formatCurrency(trade.profit)}</div>
                          <div className="text-sm">{formatPercent(trade.profitPct || 0)}</div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="positions">
          <Card>
            <CardHeader>
              <CardTitle>Open Positions</CardTitle>
            </CardHeader>
            <CardContent>
              <PositionsTable positions={positions || []} />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="trades">
          <Card>
            <CardHeader>
              <CardTitle>Trade History</CardTitle>
            </CardHeader>
            <CardContent>
              <RecentTradesTable trades={trades || []} />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
