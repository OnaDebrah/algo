import { create } from 'zustand';

type PageKey =
    | 'dashboard'
    | 'backtest'
    | 'ml-studio'
    | 'live'
    | 'advisor'
    | 'regime'
    | 'analyst'
    | 'options'
    | 'portfolio'
    | 'strategy-builder'
    | 'marketplace'
    | 'settings';

interface NavigationState {
    /** Set by child components to request a page change */
    pendingPage: PageKey | null;
    navigateTo: (page: PageKey) => void;
    clearPending: () => void;
}

export const useNavigationStore = create<NavigationState>((set) => ({
    pendingPage: null,
    navigateTo: (page) => set({ pendingPage: page }),
    clearPending: () => set({ pendingPage: null }),
}));
