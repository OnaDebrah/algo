export interface RegimeDetectionResult {
  regime: string
  confidence: number
  scores: Record<string, number>
  method: string
  strategyAllocation: Record<string, number>
  regimeStrength: number
  durationPrediction?: DurationPrediction
  changeWarning?: ChangeWarning
  statisticalRegime?: string
  mlRegime?: string
}

export interface DurationPrediction {
  currentRegime: string
  expectedDuration: number
  medianDuration: number
  stdDuration: number
  probabilityEndNextWeek: number
  sampleSize: number
}

export interface ChangeWarning {
  warning: boolean
  confidenceTrend: number
  disagreementRate: number
  recommendation: string
}
