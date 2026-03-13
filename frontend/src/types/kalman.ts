export interface ValidationResult {
    sectorMatch: boolean | null;
    correlation: number | null;
    cointegration: number | null;
    isValid: boolean;
    warnings: string[];
    errors: string[];
}
