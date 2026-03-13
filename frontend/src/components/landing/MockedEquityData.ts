// ─── Mock equity curve data ─────────────────────────────────
export const equityData = Array.from({length: 60}, (_, i) => {
    const base = 10000;
    const stratGrowth = base + (i * 220) + Math.sin(i * 0.3) * 600 + Math.random() * 300;
    const benchGrowth = base + (i * 90) + Math.sin(i * 0.5) * 200 + Math.random() * 100;
    return {
        day: `Day ${i + 1}`,
        strategy: Math.round(stratGrowth),
        benchmark: Math.round(benchGrowth),
    };
});
