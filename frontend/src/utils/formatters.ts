
export const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
};

export const formatPercent = (value: number) => {
    if (value === undefined || value === null) return 0;
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
};

export const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
};

export const toPrecision = (value: number | undefined | null, precision: number = 4): number => {
    if (value === undefined || value === null) return 0;

    const factor = Math.pow(10, precision);
    return Math.round(value * factor) / factor;
};

