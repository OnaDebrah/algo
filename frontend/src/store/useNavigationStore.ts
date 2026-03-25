import { create } from 'zustand';

type PageKey =
    | 'dashboard'
    | 'backtest'
    | 'ml-studio'
    | 'live'
    | 'advisor'
    | 'regime'
    | 'crash-prediction'
    | 'analyst'
    | 'options'
    | 'portfolio'
    | 'sector-scanner'
    | 'strategy-builder'
    | 'marketplace'
    | 'admin'
    | 'pricing'
    | 'settings'
    | 'paper-trading'
    | 'trade-journal'
    | 'economic-calendar'
    | 'teams'
    | 'getting-started'
    | 'crypto'
    | 'research';

interface NavigationState {
    /** Set by child components to request a page change */
    pendingPage: PageKey | null;
    /** Optional payload for cross-page data passing (e.g., backtest → paper trade) */
    pendingPayload: Record<string, unknown> | null;
    navigateTo: (page: PageKey, payload?: Record<string, unknown>) => void;
    clearPending: () => void;
}

export const useNavigationStore = create<NavigationState>((set) => ({
    pendingPage: null,
    pendingPayload: null,
    navigateTo: (page, payload) => set({ pendingPage: page, pendingPayload: payload ?? null }),
    clearPending: () => set({ pendingPage: null, pendingPayload: null }),
}));
