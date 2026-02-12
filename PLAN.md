# Plan: Separate Strategy Catalog by Backtest Mode

## Problem
The strategy catalog is a flat list with no concept of which backtest mode each strategy supports. Multi-asset strategies (e.g., Kalman Filter, StatArb) crash if selected in single-asset mode, and vice versa. The frontend uses fragile hardcoded ID exclusion lists to prevent this, but the backend has zero validation.

## Solution
Add a `backtest_mode` field (`"single"`, `"multi"`, or `"both"`) to every strategy in the catalog. Filter server-side via an API query param. Add backend validation to reject wrong-mode strategies. Remove hardcoded frontend exclusion lists.

## Changes (7 files)

### Step 1: Backend — Add `backtest_mode` to catalog dataclass
**File:** `backend/app/strategies/strategy_catalog.py`
- Add `backtest_mode: Literal["single", "multi", "both"] = "single"` to the `StrategyInfo` dataclass
- Add `backtest_mode=` to all 31 strategy entries in `_build_catalog()`:
  - `"both"`: sma_crossover, macd, adaptive_trend, kama, multi_kama, donchian, donchian_atr, filtered_donchian, rsi, parabolic_sar, bb_mean_reversion (work in both single and multi-independent modes)
  - `"single"`: volatility_breakout, volatility_targeting, dynamic_scaling, variance_risk_premium, ml_random_forest, ml_gradient_boosting, ml_svm, ml_logistic, ml_lstm, mc_ml_sentiment, covered_call, iron_condor, butterfly_spread, straddle
  - `"multi"`: ts_momentum, cs_momentum, pairs_trading, kalman_filter, kalman_filter_hft, sector_neutral, risk_parity_stat_arb
- Add `get_by_mode(mode)` method to `StrategyCatalog` class

### Step 2: Backend — Add `backtest_mode` to Pydantic schema
**File:** `backend/app/schemas/strategy.py`
- Add `backtest_mode: str = "single"` to the `StrategyInfo` Pydantic model (line 30)

### Step 3: Backend — Filter API by mode
**File:** `backend/app/api/routes/strategy.py`
- Add `Query` to FastAPI imports
- Add optional `mode` query param to `GET /strategy/list`: `mode: Optional[str] = Query(None)`
- Filter catalog using `catalog.get_by_mode(mode)` when mode is provided
- Include `backtest_mode` in both `list_strategies` and `get_strategy` response construction

### Step 4: Backend — Validate mode in BacktestService
**File:** `backend/app/services/backtest_service.py`
- In `run_single_backtest()` (line ~146): reject if `strategy_info.backtest_mode == "multi"`
- In `run_multi_asset_backtest()` (line ~247): reject if any strategy config has `backtest_mode == "single"`

### Step 5: Frontend — Add `backtest_mode` to TypeScript types
**File:** `frontend/src/types/all_types.ts`
- Add `backtest_mode?: 'single' | 'multi' | 'both'` to `StrategyInfo` interface (line 66)
- Add `backtest_mode?: 'single' | 'multi' | 'both'` to `Strategy` interface (line 207)

### Step 6: Frontend — Pass mode to API call
**File:** `frontend/src/utils/api.ts`
- Update `strategy.list()` to accept optional `mode` param and pass as query string

### Step 7: Frontend — Remove hardcoded filters, use server-side filtering
**File:** `frontend/src/components/backtest/BacktestPage.tsx`
- Remove the `singleBacktestStrategies` and `multiBacktestStrategies` hardcoded filter blocks
- Update `useEffect` to pass `backtestMode` to `strategyApi.list(backtestMode)` and add `backtestMode` to dependency array
- Pass `strategiesList` directly to both `SingleAssetBacktest` and `MultiAssetBacktest`

**File:** `frontend/src/components/strategies/Strategies.tsx`
- Add `backtest_mode` to each entry in the hardcoded fallback list (used before API responds)
