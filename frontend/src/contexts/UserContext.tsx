'use client'

import { createContext, useContext, ReactNode } from 'react';
import { User } from '@/types/all_types';

const UserContext = createContext<User | null>(null);

export const UserProvider = ({ user, children }: { user: User | null; children: ReactNode }) => (
    <UserContext.Provider value={user}>{children}</UserContext.Provider>
);

export const useUser = () => useContext(UserContext);
