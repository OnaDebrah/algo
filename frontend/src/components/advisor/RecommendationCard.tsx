import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Target, TrendingUp, AlertCircle } from 'lucide-react'
import type { StrategyRecommendation } from '@/types'

export function RecommendationCard({ recommendation }: { recommendation: StrategyRecommendation }) {
  const riskColor = {
    low: 'bg-green-500/20 text-green-500',
    medium: 'bg-yellow-500/20 text-yellow-500',
    high: 'bg-red-500/20 text-red-500',
  }[recommendation.riskLevel] || 'bg-gray-500/20'

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <CardTitle className="text-2xl">{recommendation.name}</CardTitle>
            <div className="flex items-center gap-2">
              <Badge className={riskColor}>{recommendation.riskLevel.toUpperCase()} RISK</Badge>
              <Badge variant="outline">
                <Target className="mr-1 h-3 w-3" />
                {recommendation.fitScore}% Match
              </Badge>
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="space-y-4">
          <div>
            <h4 className="font-semibold mb-2">Why This Strategy Fits You</h4>
            <ul className="space-y-2">
              {recommendation.whyRecommended.map((reason, i) => (
                <li key={i} className="flex items-start gap-2 text-sm">
                  <TrendingUp className="h-4 w-4 text-primary mt-0.5" />
                  <span>{reason}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="rounded-lg bg-primary/5 p-4">
            <div className="flex items-start gap-2">
              <AlertCircle className="h-5 w-5 text-primary mt-0.5" />
              <div>
                <div className="font-semibold text-sm mb-1">Performance Insight</div>
                <p className="text-sm text-muted-foreground">{recommendation.personalizedInsight}</p>
              </div>
            </div>
          </div>

          {recommendation.riskAdjustment && (
            <div className="rounded-lg border border-warning/50 bg-warning/5 p-4">
              <div className="text-sm">
                <span className="font-semibold">Risk Adjustment:</span> {recommendation.riskAdjustment}
              </div>
            </div>
          )}
        </div>

        <Button className="w-full">Start Backtesting This Strategy</Button>
      </CardContent>
    </Card>
  )
}
