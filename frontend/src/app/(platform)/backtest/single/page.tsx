'use client'

import {useState} from 'react'
import {useQuery} from '@tanstack/react-query'
import {useBacktest} from '@/lib/hooks/useBacktest'
import {strategiesApi} from '@/lib/api/strategies'
import {Card, CardContent, CardHeader, CardTitle} from '@/components/ui/card'
import {Button} from '@/components/ui/button'
import {Input} from '@/components/ui/input'
import {Label} from '@/components/ui/label'
import {Select, SelectContent, SelectItem, SelectTrigger, SelectValue} from '@/components/ui/select'
import {BacktestResults} from '@/components/backtest/BacktestResults'
import {Play, Settings} from 'lucide-react'
import type {BacktestRequest} from '@/types'

export default function SingleAssetBacktestPage() {
  const { runBacktest, isRunning } = useBacktest()
  const { data: catalog } = useQuery({
    queryKey: ['strategies', 'catalog'],
    queryFn: () => strategiesApi.getCatalog(),
  })

  const [config, setConfig] = useState<Partial<BacktestRequest>>({
    symbol: 'AAPL',
    period: '1y',
    interval: '1d',
    initialCapital: 100000,
    commissionRate: 0.001,
    slippageRate: 0.0005,
  })

  const [selectedStrategy, setSelectedStrategy] = useState<string>('')

  const handleRun = () => {
    if (!selectedStrategy || !config.symbol) return

    runBacktest({
      ...config,
      strategyKey: selectedStrategy,
      parameters: catalog?.[selectedStrategy]?.parameters || {},
    } as BacktestRequest)
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Single Asset Backtest</h1>
        <p className="text-muted-foreground">
          Test trading strategies on individual stocks
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Configuration Panel */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Symbol</Label>
              <Input
                placeholder="AAPL"
                value={config.symbol}
                onChange={(e) => setConfig({ ...config, symbol: e.target.value.toUpperCase() })}
              />
            </div>

            <div className="space-y-2">
              <Label>Strategy</Label>
              <Select value={selectedStrategy} onValueChange={setSelectedStrategy}>
                <SelectTrigger>
                  <SelectValue placeholder="Select strategy" />
                </SelectTrigger>
                <SelectContent>
                  {catalog && Object.entries(catalog).map(([key, strategy]) => (
                    <SelectItem key={key} value={key}>
                      {strategy.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Period</Label>
                <Select
                  value={config.period}
                  onValueChange={(value) => setConfig({ ...config, period: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1mo">1 Month</SelectItem>
                    <SelectItem value="3mo">3 Months</SelectItem>
                    <SelectItem value="6mo">6 Months</SelectItem>
                    <SelectItem value="1y">1 Year</SelectItem>
                    <SelectItem value="2y">2 Years</SelectItem>
                    <SelectItem value="5y">5 Years</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Interval</Label>
                <Select
                  value={config.interval}
                  onValueChange={(value) => setConfig({ ...config, interval: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1h">1 Hour</SelectItem>
                    <SelectItem value="1d">1 Day</SelectItem>
                    <SelectItem value="1wk">1 Week</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Initial Capital</Label>
              <Input
                type="number"
                value={config.initialCapital}
                onChange={(e) =>
                  setConfig({ ...config, initialCapital: Number(e.target.value) })
                }
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Commission (%)</Label>
                <Input
                  type="number"
                  step="0.001"
                  value={(config.commissionRate || 0) * 100}
                  onChange={(e) =>
                    setConfig({ ...config, commissionRate: Number(e.target.value) / 100 })
                  }
                />
              </div>

              <div className="space-y-2">
                <Label>Slippage (%)</Label>
                <Input
                  type="number"
                  step="0.001"
                  value={(config.slippageRate || 0) * 100}
                  onChange={(e) =>
                    setConfig({ ...config, slippageRate: Number(e.target.value) / 100 })
                  }
                />
              </div>
            </div>

            <Button
              onClick={handleRun}
              disabled={isRunning || !selectedStrategy}
              className="w-full"
            >
              {isRunning ? (
                <>
                  <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Run Backtest
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Results Panel */}
        <div className="lg:col-span-2">
          <BacktestResults />
        </div>
      </div>
    </div>
  )
}
