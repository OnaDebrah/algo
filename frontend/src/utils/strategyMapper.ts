import { StrategyInfo } from '@/types/all_types';

export const mapFrontendStrategyToBackend = (frontendStrategies: any[]): {
    key: string;
    name: string;
    description: string;
    category: string;
    parameters: {
        name: string;
        type: string;
        default: unknown;
        description: string;
        min: number | undefined;
        max: number | undefined;
        options: undefined
    }[]
}[] => {

  return frontendStrategies.map(strat => ({
    key: strat.id,
    name: strat.name,
    description: strat.description,
    category: strat.category,
    parameters: Object.entries(strat.params || {}).map(([name, defaultValue]) => ({
      name,
      type: typeof defaultValue === 'number' ? 'number' : 'string',
      default: defaultValue,
      description: '',
      min: typeof defaultValue === 'number' ? 0 : undefined,
      max: typeof defaultValue === 'number' ? 100 : undefined,
      options: undefined,
    })),
  }));
};
