import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Star, Download, Heart } from 'lucide-react'
import { formatNumber } from '@/lib/utils/formatters'
import type { MarketplaceStrategy } from '@/types'
import Link from 'next/link'

export function StrategyCard({ strategy }: { strategy: MarketplaceStrategy }) {
  return (
    <Card className="flex flex-col">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <CardTitle className="text-xl">{strategy.name}</CardTitle>
            <p className="text-sm text-muted-foreground">by {strategy.creatorName}</p>
          </div>
          {strategy.price === 0 ? (
            <Badge variant="secondary">FREE</Badge>
          ) : (
            <Badge>${strategy.price}</Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="flex-1 space-y-4">
        <p className="text-sm text-muted-foreground line-clamp-3">{strategy.description}</p>

        <div className="flex flex-wrap gap-2">
          <Badge variant="outline">{strategy.category}</Badge>
          <Badge variant="outline">{strategy.complexity}</Badge>
        </div>

        {strategy.performanceMetrics && (
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-muted-foreground">Return:</span>
              <span className="ml-1 font-medium">{formatNumber(strategy.performanceMetrics.totalReturn)}%</span>
            </div>
            <div>
              <span className="text-muted-foreground">Sharpe:</span>
              <span className="ml-1 font-medium">{formatNumber(strategy.performanceMetrics.sharpeRatio)}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Win Rate:</span>
              <span className="ml-1 font-medium">{formatNumber(strategy.performanceMetrics.winRate)}%</span>
            </div>
            <div>
              <span className="text-muted-foreground">Trades:</span>
              <span className="ml-1 font-medium">{strategy.performanceMetrics.numTrades}</span>
            </div>
          </div>
        )}

        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <div className="flex items-center gap-1">
            <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
            <span>{strategy.rating.toFixed(1)}</span>
            <span>({strategy.numRatings})</span>
          </div>
          <div className="flex items-center gap-1">
            <Download className="h-4 w-4" />
            <span>{strategy.downloads}</span>
          </div>
        </div>
      </CardContent>

      <CardFooter className="flex gap-2">
        <Button asChild className="flex-1">
          <Link href={`/marketplace/${strategy.id}`}>View Details</Link>
        </Button>
        <Button variant="outline" size="icon">
          <Heart className="h-4 w-4" />
        </Button>
      </CardFooter>
    </Card>
  )
}
