/**
 * Utility functions for the trading dashboard
 */

// Format number with commas and specified decimals
function formatNumber(number, decimals = 2) {
    if (number === null || number === undefined || isNaN(number)) {
        return '0.00';
    }
    return parseFloat(number).toFixed(decimals).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Format price with appropriate precision based on value
function formatPrice(price, symbol = null) {
    if (price === null || price === undefined || isNaN(price)) {
        return '0.00';
    }
    
    // Determine precision based on price value or symbol
    let precision = 2;
    
    if (symbol) {
        if (symbol.includes('BTC')) {
            precision = 2;
        } else if (symbol.includes('ETH')) {
            precision = 2;
        } else if (symbol.includes('BNB')) {
            precision = 2;
        } else if (price < 0.1) {
            precision = 6;
        } else if (price < 1) {
            precision = 4;
        } else if (price < 10) {
            precision = 3;
        }
    } else {
        if (price < 0.0001) {
            precision = 8;
        } else if (price < 0.01) {
            precision = 6;
        } else if (price < 1) {
            precision = 4;
        } else if (price < 10) {
            precision = 3;
        }
    }
    
    return parseFloat(price).toFixed(precision);
}

// Format percentage
function formatPercent(percent, includeSign = true) {
    if (percent === null || percent === undefined || isNaN(percent)) {
        return '0.00%';
    }
    
    const sign = includeSign && percent > 0 ? '+' : '';
    return `${sign}${parseFloat(percent).toFixed(2)}%`;
}

// Format timestamp to date time
function formatDateTime(timestamp) {
    if (!timestamp) return '-';
    const date = new Date(timestamp);
    return date.toLocaleString();
}

// Format timestamp to time only
function formatTime(timestamp) {
    if (!timestamp) return '-';
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
}

// Truncate string with ellipsis
function truncateString(str, maxLength) {
    if (!str) return '';
    if (str.length <= maxLength) return str;
    return str.substring(0, maxLength) + '...';
}

// Get color class based on sign
function getColorClass(value) {
    if (value === 0) return '';
    return value > 0 ? 'price-up' : 'price-down';
}

// Format currency value
function formatCurrency(value, currency = 'USD') {
    if (value === null || value === undefined || isNaN(value)) {
        return '$0.00';
    }
    
    const formatter = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: currency === 'USD' ? 'USD' : currency,
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
    
    return formatter.format(value);
}

// Calculate percentage change
function calculatePercentChange(current, previous) {
    if (!previous) return 0;
    return ((current - previous) / previous) * 100;
}

// Format asset name
function formatAssetName(symbol) {
    if (!symbol) return '';
    if (symbol.endsWith('USDT')) {
        return symbol.replace('USDT', '') + '/USDT';
    }
    return symbol;
}

// Get indicator signal class
function getSignalClass(signal) {
    if (signal === 'BUY') return 'signal-buy';
    if (signal === 'SELL') return 'signal-sell';
    return 'signal-neutral';
}

// Debounce function to limit function calls
function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

// API request helper
async function fetchAPI(endpoint) {
    try {
        const response = await fetch(endpoint);
        if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${endpoint}:`, error);
        return null;
    }
}

// POST API request helper
async function postAPI(endpoint, data) {
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error(`Error posting to ${endpoint}:`, error);
        return null;
    }
}
