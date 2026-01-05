export interface Strategy {
  key: string
  name: string
  description: string
  category: string
  complexity: 'Beginner' | 'Intermediate' | 'Advanced' | 'Expert'
  timeHorizon: string
  riskLevel: 'low' | 'medium' | 'high'
  parameters: Record<string, StrategyParameter>
  pros: string[]
  cons: string[]
}

export interface StrategyParameter {
  default: any
  range?: [number, number] | any[]
  description: string
}

export interface StrategyConfig {
  strategyKey: string
  parameters: Record<string, any>
}
