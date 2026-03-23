/* eslint-disable @typescript-eslint/no-explicit-any */
'use client'

import React, {useCallback, useEffect, useRef, useState} from "react";
import {ArrowLeft, ArrowRight, X} from "lucide-react";
import tourSteps, {TourStep} from "./tourSteps";

interface TourOverlayProps {
    onComplete: () => void;
    onNavigate?: (page: string) => void;
}

const STORAGE_KEY = 'oraculum_tour_completed';

export function isTourCompleted(): boolean {
    if (typeof window === 'undefined') return true;
    return localStorage.getItem(STORAGE_KEY) === 'true';
}

export function resetTour(): void {
    localStorage.removeItem(STORAGE_KEY);
}

const TourOverlay: React.FC<TourOverlayProps> = ({onComplete, onNavigate}) => {
    const [stepIndex, setStepIndex] = useState(0);
    const [targetRect, setTargetRect] = useState<DOMRect | null>(null);
    const overlayRef = useRef<HTMLDivElement>(null);

    const step = tourSteps[stepIndex];
    const totalSteps = tourSteps.length;

    const findTarget = useCallback(() => {
        const el = document.querySelector(step.target);
        if (el) {
            setTargetRect(el.getBoundingClientRect());
        } else {
            setTargetRect(null);
        }
    }, [step.target]);

    // Navigate to the correct page if step requires it
    useEffect(() => {
        if (step.page && onNavigate) {
            onNavigate(step.page);
        }
        // Small delay to let the page render before measuring
        const timeout = setTimeout(findTarget, 300);
        return () => clearTimeout(timeout);
    }, [stepIndex, step.page, onNavigate, findTarget]);

    // Re-measure on resize
    useEffect(() => {
        window.addEventListener('resize', findTarget);
        return () => window.removeEventListener('resize', findTarget);
    }, [findTarget]);

    const handleNext = () => {
        if (stepIndex < totalSteps - 1) {
            setStepIndex(stepIndex + 1);
        } else {
            handleComplete();
        }
    };

    const handlePrev = () => {
        if (stepIndex > 0) {
            setStepIndex(stepIndex - 1);
        }
    };

    const handleComplete = () => {
        localStorage.setItem(STORAGE_KEY, 'true');
        onComplete();
    };

    // Calculate tooltip position
    const getTooltipStyle = (): React.CSSProperties => {
        if (!targetRect) {
            return {top: '50%', left: '50%', transform: 'translate(-50%, -50%)'};
        }

        const padding = 16;
        const tooltipWidth = 340;

        switch (step.placement) {
            case 'right':
                return {
                    top: targetRect.top + targetRect.height / 2,
                    left: targetRect.right + padding,
                    transform: 'translateY(-50%)',
                };
            case 'left':
                return {
                    top: targetRect.top + targetRect.height / 2,
                    left: targetRect.left - padding - tooltipWidth,
                    transform: 'translateY(-50%)',
                };
            case 'bottom':
                return {
                    top: targetRect.bottom + padding,
                    left: targetRect.left + targetRect.width / 2,
                    transform: 'translateX(-50%)',
                };
            case 'top':
                return {
                    top: targetRect.top - padding,
                    left: targetRect.left + targetRect.width / 2,
                    transform: 'translate(-50%, -100%)',
                };
            default:
                return {
                    top: targetRect.bottom + padding,
                    left: targetRect.left + targetRect.width / 2,
                    transform: 'translateX(-50%)',
                };
        }
    };

    return (
        <div ref={overlayRef} className="fixed inset-0 z-[9999]">
            {/* Backdrop with spotlight cutout using CSS clip-path */}
            <svg className="absolute inset-0 w-full h-full" style={{pointerEvents: 'none'}}>
                <defs>
                    <mask id="spotlight-mask">
                        <rect width="100%" height="100%" fill="white"/>
                        {targetRect && (
                            <rect
                                x={targetRect.left - 8}
                                y={targetRect.top - 8}
                                width={targetRect.width + 16}
                                height={targetRect.height + 16}
                                rx={12}
                                fill="black"
                            />
                        )}
                    </mask>
                </defs>
                <rect
                    width="100%"
                    height="100%"
                    fill="rgba(0,0,0,0.7)"
                    mask="url(#spotlight-mask)"
                    style={{pointerEvents: 'all'}}
                    onClick={handleNext}
                />
            </svg>

            {/* Spotlight ring */}
            {targetRect && (
                <div
                    className="absolute border-2 border-violet-400/60 rounded-xl pointer-events-none"
                    style={{
                        left: targetRect.left - 8,
                        top: targetRect.top - 8,
                        width: targetRect.width + 16,
                        height: targetRect.height + 16,
                        boxShadow: '0 0 0 4px rgba(139, 92, 246, 0.15)',
                    }}
                />
            )}

            {/* Tooltip */}
            <div
                className="absolute w-[340px] bg-slate-900 border border-slate-700/50 rounded-2xl shadow-2xl shadow-black/50 overflow-hidden animate-in fade-in slide-in-from-bottom-2 duration-300"
                style={getTooltipStyle()}
            >
                {/* Progress bar */}
                <div className="h-1 bg-slate-800">
                    <div
                        className="h-full bg-gradient-to-r from-violet-500 to-fuchsia-500 transition-all duration-300"
                        style={{width: `${((stepIndex + 1) / totalSteps) * 100}%`}}
                    />
                </div>

                <div className="p-5">
                    <div className="flex items-start justify-between mb-3">
                        <div>
                            <p className="text-xs font-semibold text-violet-400 mb-1">
                                Step {stepIndex + 1} of {totalSteps}
                            </p>
                            <h3 className="text-base font-bold text-slate-100">{step.title}</h3>
                        </div>
                        <button
                            onClick={handleComplete}
                            className="text-slate-500 hover:text-slate-300 transition-colors -mt-1"
                        >
                            <X size={16}/>
                        </button>
                    </div>

                    <p className="text-sm text-slate-400 leading-relaxed">{step.content}</p>

                    <div className="flex items-center justify-between mt-5">
                        <button
                            onClick={handleComplete}
                            className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
                        >
                            Skip Tour
                        </button>

                        <div className="flex items-center gap-2">
                            {stepIndex > 0 && (
                                <button
                                    onClick={handlePrev}
                                    className="flex items-center gap-1 px-3 py-1.5 text-sm text-slate-400 hover:text-slate-200 transition-colors"
                                >
                                    <ArrowLeft size={14}/> Back
                                </button>
                            )}
                            <button
                                onClick={handleNext}
                                className="flex items-center gap-1 px-4 py-1.5 bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white text-sm font-medium rounded-lg transition-all shadow-lg shadow-violet-500/20"
                            >
                                {stepIndex === totalSteps - 1 ? 'Finish' : 'Next'} <ArrowRight size={14}/>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default TourOverlay;
