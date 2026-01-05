export interface UserProfile {
  goals: string[]
  riskTolerance: string
  timeHorizon: string
  experience: string
  timeCommitment: string
  capital: number
  marketPreference: string
}

export interface StrategyRecommendation {
  strategyKey: string
  name: string
  fitScore: number
  whyRecommended: string[]
  personalizedInsight: string
  riskAdjustment?: string
  expectedReturn: [number, number]
  riskLevel: string
  similarTradersUsage: string
}
