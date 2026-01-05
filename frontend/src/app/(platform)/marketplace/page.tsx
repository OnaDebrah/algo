'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { marketplaceApi } from '@/lib/api/marketplace'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { StrategyCard } from '@/components/marketplace/StrategyCard'
import { Search, TrendingUp, Store } from 'lucide-react'

export default function MarketplacePage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [category, setCategory] = useState<string>()
  const [sortBy, setSortBy] = useState('rating')

  const { data: strategies, isLoading } = useQuery({
    queryKey: ['marketplace', 'browse', { searchQuery, category, sortBy }],
    queryFn: () => marketplaceApi.browse({ searchQuery: searchQuery || undefined, category, sortBy }),
  })

  const { data: stats } = useQuery({
    queryKey: ['marketplace', 'stats'],
    queryFn: () => marketplaceApi.getStats(),
  })

  return (
    <div className="space-y-6">
      {/* Header with Stats */}
      <div className="rounded-lg bg-gradient-to-r from-primary/20 to-primary/5 p-8">
        <div className="mb-6">
          <h1 className="text-4xl font-bold">Strategy Marketplace</h1>
          <p className="mt-2 text-lg text-muted-foreground">
            Discover, share, and clone trading strategies from the community
          </p>
        </div>

        <div className="grid gap-4 md:grid-cols-4">
          <Card>
            <CardContent className="pt-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-primary">{stats?.totalStrategies || 0}</div>
                <div className="text-sm text-muted-foreground">Total Strategies</div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-primary">{stats?.totalDownloads || 0}</div>
                <div className="text-sm text-muted-foreground">Total Downloads</div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-primary">{stats?.averageRating.toFixed(1) || 0}⭐</div>
                <div className="text-sm text-muted-foreground">Avg Rating</div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-primary">{stats?.totalCreators || 0}</div>
                <div className="text-sm text-muted-foreground">Active Creators</div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col gap-4 md:flex-row">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Search strategies..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
            <Select value={category} onValueChange={setCategory}>
              <SelectTrigger className="w-full md:w-[200px]">
                <SelectValue placeholder="Category" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Categories</SelectItem>
                <SelectItem value="trend_following">Trend Following</SelectItem>
                <SelectItem value="mean_reversion">Mean Reversion</SelectItem>
                <SelectItem value="momentum">Momentum</SelectItem>
                <SelectItem value="volatility">Volatility</SelectItem>
              </SelectContent>
            </Select>
            <Select value={sortBy} onValueChange={setSortBy}>
              <SelectTrigger className="w-full md:w-[200px]">
                <SelectValue placeholder="Sort by" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="rating">Rating</SelectItem>
                <SelectItem value="downloads">Downloads</SelectItem>
                <SelectItem value="created_at">Newest</SelectItem>
                <SelectItem value="updated_at">Recently Updated</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Strategy Grid */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {isLoading ? (
          Array(6).fill(0).map((_, i) => (
            <Card key={i} className="animate-pulse">
              <CardHeader>
                <div className="h-6 w-3/4 rounded bg-muted" />
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="h-4 w-full rounded bg-muted" />
                  <div className="h-4 w-2/3 rounded bg-muted" />
                </div>
              </CardContent>
            </Card>
          ))
        ) : (
          strategies?.map((strategy) => (
            <StrategyCard key={strategy.id} strategy={strategy} />
          ))
        )}
      </div>
    </div>
  )
}
