import { useCallback, useRef, useState } from 'react';

export interface Drawing {
    id: string;
    type: 'trendline' | 'horizontal' | 'fibonacci' | 'rectangle' | 'text';
    points: { time: number; price: number }[];
    color: string;
    lineWidth: number;
    label?: string;
}

const storageKey = (symbol: string, timeframe: string) =>
    `chart_drawings_${symbol}_${timeframe}`;

export function useChartDrawings(symbol: string, timeframe: string) {
    const [drawings, setDrawings] = useState<Drawing[]>(() => {
        try {
            const raw = localStorage.getItem(storageKey(symbol, timeframe));
            return raw ? JSON.parse(raw) : [];
        } catch {
            return [];
        }
    });

    const drawingsRef = useRef(drawings);
    drawingsRef.current = drawings;

    const persist = useCallback((updated: Drawing[]) => {
        setDrawings(updated);
        try {
            localStorage.setItem(storageKey(symbol, timeframe), JSON.stringify(updated));
        } catch {
            // storage full — silently ignore
        }
    }, [symbol, timeframe]);

    const addDrawing = useCallback((drawing: Omit<Drawing, 'id'>) => {
        const newDrawing: Drawing = { ...drawing, id: crypto.randomUUID() };
        persist([...drawingsRef.current, newDrawing]);
        return newDrawing;
    }, [persist]);

    const removeDrawing = useCallback((id: string) => {
        persist(drawingsRef.current.filter(d => d.id !== id));
    }, [persist]);

    const clearAll = useCallback(() => {
        persist([]);
    }, [persist]);

    const undo = useCallback(() => {
        if (drawingsRef.current.length === 0) return;
        persist(drawingsRef.current.slice(0, -1));
    }, [persist]);

    const loadForTimeframe = useCallback((newSymbol: string, newTimeframe: string) => {
        try {
            const raw = localStorage.getItem(storageKey(newSymbol, newTimeframe));
            const loaded = raw ? JSON.parse(raw) : [];
            setDrawings(loaded);
            drawingsRef.current = loaded;
        } catch {
            setDrawings([]);
        }
    }, []);

    return { drawings, addDrawing, removeDrawing, clearAll, undo, loadForTimeframe };
}
