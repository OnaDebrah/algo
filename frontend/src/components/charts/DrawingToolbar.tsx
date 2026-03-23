'use client';

import React from 'react';
import { Minus, MousePointer, MoveHorizontal, Redo2, Trash2, TrendingUp, Type } from 'lucide-react';

export type DrawingTool = 'select' | 'trendline' | 'horizontal' | 'fibonacci' | 'rectangle' | 'text';

interface DrawingToolbarProps {
    activeTool: DrawingTool;
    onToolChange: (tool: DrawingTool) => void;
    activeColor: string;
    onColorChange: (color: string) => void;
    onUndo: () => void;
    onClearAll: () => void;
    drawingCount: number;
}

const COLORS = ['#6366f1', '#10b981', '#ef4444', '#f59e0b', '#3b82f6', '#ec4899', '#ffffff'];

const tools: { id: DrawingTool; icon: React.ReactNode; label: string }[] = [
    { id: 'select', icon: <MousePointer size={14} />, label: 'Select' },
    { id: 'trendline', icon: <TrendingUp size={14} />, label: 'Trendline' },
    { id: 'horizontal', icon: <Minus size={14} />, label: 'Horizontal' },
    { id: 'fibonacci', icon: <MoveHorizontal size={14} />, label: 'Fibonacci' },
    { id: 'rectangle', icon: <span className="w-3 h-2.5 border border-current inline-block" />, label: 'Rectangle' },
    { id: 'text', icon: <Type size={14} />, label: 'Text' },
];

const DrawingToolbar: React.FC<DrawingToolbarProps> = ({
    activeTool,
    onToolChange,
    activeColor,
    onColorChange,
    onUndo,
    onClearAll,
    drawingCount,
}) => {
    return (
        <div className="flex items-center gap-1 bg-card border border-border rounded-lg p-1">
            {/* Tools */}
            {tools.map(tool => (
                <button
                    key={tool.id}
                    onClick={() => onToolChange(tool.id)}
                    title={tool.label}
                    className={`p-1.5 rounded transition-colors ${
                        activeTool === tool.id
                            ? 'bg-violet-600 text-white'
                            : 'text-muted-foreground hover:text-foreground hover:bg-accent'
                    }`}
                >
                    {tool.icon}
                </button>
            ))}

            {/* Divider */}
            <div className="w-px h-5 bg-border mx-1" />

            {/* Colors */}
            {COLORS.map(color => (
                <button
                    key={color}
                    onClick={() => onColorChange(color)}
                    className={`w-4 h-4 rounded-full border-2 transition-transform ${
                        activeColor === color ? 'border-foreground scale-125' : 'border-transparent'
                    }`}
                    style={{ backgroundColor: color }}
                />
            ))}

            {/* Divider */}
            <div className="w-px h-5 bg-border mx-1" />

            {/* Actions */}
            <button
                onClick={onUndo}
                disabled={drawingCount === 0}
                title="Undo"
                className="p-1.5 rounded text-muted-foreground hover:text-foreground hover:bg-accent disabled:opacity-30 transition-colors"
            >
                <Redo2 size={14} />
            </button>
            <button
                onClick={onClearAll}
                disabled={drawingCount === 0}
                title="Clear all"
                className="p-1.5 rounded text-muted-foreground hover:text-red-400 hover:bg-red-500/10 disabled:opacity-30 transition-colors"
            >
                <Trash2 size={14} />
            </button>
        </div>
    );
};

export default DrawingToolbar;
