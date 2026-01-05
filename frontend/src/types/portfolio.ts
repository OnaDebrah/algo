export interface Portfolio {
  id: string
  name: string
  initialCapital: number
  currentCapital: number
  totalReturn: number
  totalReturnPct: number
  createdAt: string
  updatedAt: string
}

export interface Position {
  id: string
  symbol: string
  side: 'long' | 'short'
  quantity: number
  entryPrice: number
  currentPrice: number
  unrealizedPnl: number
  unrealizedPnlPct: number
  entryTime: string
}

export interface Trade {
  id: string
  symbol: string
  orderType: 'BUY' | 'SELL'
  quantity: number
  price: number
  commission: number
  timestamp: string
  strategy: string
  profit?: number
  profitPct?: number
}

export interface PortfolioMetrics {
  nav: number
  prevNav: number
  exposure: number
  unrealizedPnl: number
  cash: number
  totalValue: number
  dailyReturn: number
  dailyReturnPct: number
}
