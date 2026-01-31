import { useState, useCallback } from 'react';
import { backtest } from '@/utils/api';
import { MultiAssetBacktestRequest, MultiAssetBacktestResponse } from '@/types/all_types';

export const useMultiAssetBacktest = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<MultiAssetBacktestResponse | null>(null);

  const runMultiBacktest = useCallback(async (request: MultiAssetBacktestRequest) => {
    setLoading(true);
    setError(null);

    try {
      const response = await backtest.runMulti(request);
      setData(response);
      return response;
    } catch (err: any) {
      const errorMessage = err.response?.detail || err.message || 'Multi-asset backtest failed';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return {
    loading,
    error,
    data,
    runMultiBacktest,
    reset,
  };
};
