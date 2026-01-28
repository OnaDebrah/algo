'use client';
import React from 'react';
import StrategyLibrary from '@/components/strategies/StrategyLibrary';

export default function LibraryPage() {
    return (
        <div className="p-8 pb-20 gap-16 sm:p-12 font-[family-name:var(--font-geist-sans)] max-w-7xl mx-auto">
            <StrategyLibrary />
        </div>
    );
}
