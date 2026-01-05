export interface MarketplaceStrategy {
  id: string
  name: string
  description: string
  creatorId: string
  creatorName: string
  strategyType: string
  category: string
  complexity: string
  parameters: Record<string, any>
  performanceMetrics: PerformanceMetrics
  price: number
  isPublic: boolean
  isVerified: boolean
  version: string
  tags: string[]
  downloads: number
  rating: number
  numRatings: number
  numReviews: number
  createdAt: string
  updatedAt: string
}

export interface PerformanceMetrics {
  totalReturn: number
  sharpeRatio: number
  maxDrawdown: number
  winRate: number
  numTrades: number
}

export interface StrategyReview {
  id: string
  strategyId: string
  userId: string
  username: string
  rating: number
  reviewText: string
  performanceAchieved?: PerformanceMetrics
  createdAt: string
}

export interface PublishStrategyRequest {
  name: string
  description: string
  strategyType: string
  complexity: string
  parameters: Record<string, any>
  performanceMetrics?: PerformanceMetrics
  price: number
  isPublic: boolean
  tags: string[]
}
