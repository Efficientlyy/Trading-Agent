/**
 * Chart initialization and management for the trading dashboard
 */

// Main chart instance
let priceChart = null;
let candlestickSeries = null;
let volumeSeries = null;
let sma20Series = null;
let sma50Series = null;
let bbUpperSeries = null;
let bbLowerSeries = null;
let bbMiddleSeries = null;

// Initialize the main price chart
function initializePriceChart() {
    const container = document.getElementById('price-chart');
    
    // Create chart
    priceChart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: container.clientHeight,
        layout: {
            backgroundColor: '#121826',
            textColor: '#d1d4dc',
        },
        grid: {
            vertLines: {
                color: 'rgba(42, 53, 72, 0.5)',
            },
            horzLines: {
                color: 'rgba(42, 53, 72, 0.5)',
            },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
        },
        rightPriceScale: {
            borderColor: 'rgba(42, 53, 72, 0.5)',
        },
        timeScale: {
            borderColor: 'rgba(42, 53, 72, 0.5)',
            timeVisible: true,
        },
    });
    
    // Create candlestick series
    candlestickSeries = priceChart.addCandlestickSeries({
        upColor: '#00c076',
        downColor: '#f6465d',
        borderDownColor: '#f6465d',
        borderUpColor: '#00c076',
        wickDownColor: '#f6465d',
        wickUpColor: '#00c076',
    });
    
    // Add volume series
    volumeSeries = priceChart.addHistogramSeries({
        color: '#26a69a',
        priceFormat: {
            type: 'volume',
        },
        priceScaleId: '',
        scaleMargins: {
            top: 0.8,
            bottom: 0,
        },
    });
    
    // Add SMA-20 series
    sma20Series = priceChart.addLineSeries({
        color: '#2196F3',
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
    });
    
    // Add SMA-50 series
    sma50Series = priceChart.addLineSeries({
        color: '#FF9800',
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
    });
    
    // Add Bollinger Bands
    bbUpperSeries = priceChart.addLineSeries({
        color: 'rgba(255, 255, 255, 0.5)',
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
    });
    
    bbMiddleSeries = priceChart.addLineSeries({
        color: 'rgba(255, 255, 255, 0.5)',
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
    });
    
    bbLowerSeries = priceChart.addLineSeries({
        color: 'rgba(255, 255, 255, 0.5)',
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
    });
    
    // Handle resize
    window.addEventListener('resize', () => {
        if (priceChart) {
            priceChart.resize(
                container.clientWidth,
                container.clientHeight
            );
        }
    });
}

// Update the price chart with new data
function updatePriceChart(symbol, interval) {
    fetchAPI(`/api/klines/${symbol}/${interval}`)
        .then(data => {
            if (!data || data.length === 0) return;
            
            // Process candlestick data
            const candlesticks = data.map(item => ({
                time: item.open_time / 1000,
                open: parseFloat(item.open),
                high: parseFloat(item.high),
                low: parseFloat(item.low),
                close: parseFloat(item.close)
            }));
            
            // Process volume data
            const volumes = data.map(item => ({
                time: item.open_time / 1000,
                value: parseFloat(item.volume),
                color: parseFloat(item.close) >= parseFloat(item.open) 
                    ? 'rgba(0, 192, 118, 0.5)' 
                    : 'rgba(246, 70, 93, 0.5)'
            }));
            
            // Update the chart
            candlestickSeries.setData(candlesticks);
            volumeSeries.setData(volumes);
            
            // Update technical indicators
            updateChartIndicators(symbol, interval);
        });
}

// Update technical indicators on the chart
function updateChartIndicators(symbol, interval) {
    fetchAPI(`/api/indicators/${symbol}/${interval}`)
        .then(data => {
            if (!data) return;
            
            // Get timestamps from candlestick data
            fetchAPI(`/api/klines/${symbol}/${interval}`)
                .then(klines => {
                    if (!klines || klines.length === 0) return;
                    
                    // Prepare SMA data
                    const sma20Data = klines.map((item, index) => {
                        if (index < 20) return null;
                        const prices = klines.slice(index - 20, index).map(k => parseFloat(k.close));
                        const sma = prices.reduce((sum, price) => sum + price, 0) / 20;
                        return {
                            time: item.open_time / 1000,
                            value: sma
                        };
                    }).filter(item => item !== null);
                    
                    const sma50Data = klines.map((item, index) => {
                        if (index < 50) return null;
                        const prices = klines.slice(index - 50, index).map(k => parseFloat(k.close));
                        const sma = prices.reduce((sum, price) => sum + price, 0) / 50;
                        return {
                            time: item.open_time / 1000,
                            value: sma
                        };
                    }).filter(item => item !== null);
                    
                    // Prepare Bollinger Bands data
                    const bbData = klines.map((item, index) => {
                        if (index < 20) return null;
                        
                        const prices = klines.slice(index - 20, index).map(k => parseFloat(k.close));
                        const sma = prices.reduce((sum, price) => sum + price, 0) / 20;
                        
                        // Calculate standard deviation
                        const variance = prices.reduce((sum, price) => sum + Math.pow(price - sma, 2), 0) / 20;
                        const stdDev = Math.sqrt(variance);
                        
                        return {
                            time: item.open_time / 1000,
                            middle: sma,
                            upper: sma + (2 * stdDev),
                            lower: sma - (2 * stdDev)
                        };
                    }).filter(item => item !== null);
                    
                    // Update indicator series
                    sma20Series.setData(sma20Data);
                    sma50Series.setData(sma50Data);
                    
                    const bbUpperData = bbData.map(item => ({
                        time: item.time,
                        value: item.upper
                    }));
                    
                    const bbMiddleData = bbData.map(item => ({
                        time: item.time,
                        value: item.middle
                    }));
                    
                    const bbLowerData = bbData.map(item => ({
                        time: item.time,
                        value: item.lower
                    }));
                    
                    bbUpperSeries.setData(bbUpperData);
                    bbMiddleSeries.setData(bbMiddleData);
                    bbLowerSeries.setData(bbLowerData);
                });
            
            // Update indicator display
            updateIndicatorDisplay(data);
        });
}

// Update technical indicator display
function updateIndicatorDisplay(data) {
    if (!data) return;
    
    // RSI
    const rsiValue = document.getElementById('rsi-value');
    const rsiSignal = document.getElementById('rsi-signal');
    
    if (data.rsi_14) {
        const rsi = parseFloat(data.rsi_14).toFixed(2);
        rsiValue.textContent = rsi;
        
        let rsiSignalText = 'NEUTRAL';
        let rsiSignalClass = 'signal-neutral';
        
        if (rsi > 70) {
            rsiSignalText = 'SELL';
            rsiSignalClass = 'signal-sell';
        } else if (rsi < 30) {
            rsiSignalText = 'BUY';
            rsiSignalClass = 'signal-buy';
        }
        
        rsiSignal.textContent = rsiSignalText;
        rsiSignal.className = rsiSignalClass;
    }
    
    // MACD
    const macdValue = document.getElementById('macd-value');
    const macdSignal = document.getElementById('macd-signal');
    
    if (data.macd && data.macd_signal) {
        const macd = parseFloat(data.macd).toFixed(4);
        const macdSignalValue = parseFloat(data.macd_signal).toFixed(4);
        macdValue.textContent = `${macd} / ${macdSignalValue}`;
        
        let macdSignalText = 'NEUTRAL';
        let macdSignalClass = 'signal-neutral';
        
        if (parseFloat(data.macd) > parseFloat(data.macd_signal)) {
            macdSignalText = 'BUY';
            macdSignalClass = 'signal-buy';
        } else if (parseFloat(data.macd) < parseFloat(data.macd_signal)) {
            macdSignalText = 'SELL';
            macdSignalClass = 'signal-sell';
        }
        
        macdSignal.textContent = macdSignalText;
        macdSignal.className = macdSignalClass;
    }
    
    // MA Crossover
    const maValue = document.getElementById('ma-value');
    const maSignal = document.getElementById('ma-signal');
    
    if (data.sma_20 && data.sma_50) {
        const sma20 = parseFloat(data.sma_20).toFixed(2);
        const sma50 = parseFloat(data.sma_50).toFixed(2);
        maValue.textContent = `SMA20: ${sma20} / SMA50: ${sma50}`;
        
        let maSignalText = 'NEUTRAL';
        let maSignalClass = 'signal-neutral';
        
        if (parseFloat(data.sma_20) > parseFloat(data.sma_50)) {
            maSignalText = 'BUY';
            maSignalClass = 'signal-buy';
        } else if (parseFloat(data.sma_20) < parseFloat(data.sma_50)) {
            maSignalText = 'SELL';
            maSignalClass = 'signal-sell';
        }
        
        maSignal.textContent = maSignalText;
        maSignal.className = maSignalClass;
    }
    
    // Bollinger Bands
    const bbValue = document.getElementById('bb-value');
    const bbSignal = document.getElementById('bb-signal');
    
    if (data.close && data.bb_upper && data.bb_lower) {
        const close = parseFloat(data.close).toFixed(2);
        const upper = parseFloat(data.bb_upper).toFixed(2);
        const lower = parseFloat(data.bb_lower).toFixed(2);
        bbValue.textContent = `${lower} < ${close} < ${upper}`;
        
        let bbSignalText = 'NEUTRAL';
        let bbSignalClass = 'signal-neutral';
        
        if (parseFloat(data.close) > parseFloat(data.bb_upper)) {
            bbSignalText = 'SELL';
            bbSignalClass = 'signal-sell';
        } else if (parseFloat(data.close) < parseFloat(data.bb_lower)) {
            bbSignalText = 'BUY';
            bbSignalClass = 'signal-buy';
        }
        
        bbSignal.textContent = bbSignalText;
        bbSignal.className = bbSignalClass;
    }
}

// Initialize portfolio chart
function initializePortfolioChart() {
    const container = document.getElementById('portfolio-chart');
    if (!container) return;
    
    const portfolioChart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: container.clientHeight,
        layout: {
            backgroundColor: '#121826',
            textColor: '#d1d4dc',
        },
        grid: {
            vertLines: {
                color: 'rgba(42, 53, 72, 0.5)',
            },
            horzLines: {
                color: 'rgba(42, 53, 72, 0.5)',
            },
        },
        rightPriceScale: {
            borderColor: 'rgba(42, 53, 72, 0.5)',
        },
        timeScale: {
            borderColor: 'rgba(42, 53, 72, 0.5)',
            timeVisible: true,
        },
    });
    
    // Add portfolio value series
    const portfolioSeries = portfolioChart.addAreaSeries({
        topColor: 'rgba(33, 150, 243, 0.56)',
        bottomColor: 'rgba(33, 150, 243, 0.04)',
        lineColor: 'rgba(33, 150, 243, 1)',
        lineWidth: 2,
    });
    
    // Generate sample data for now
    const currentTime = Math.floor(Date.now() / 1000);
    const sampleData = Array.from({ length: 30 }, (_, i) => {
        const time = currentTime - (29 - i) * 24 * 60 * 60;
        const value = 10000 + Math.random() * 2000 * Math.sin(i / 4);
        return { time, value };
    });
    
    portfolioSeries.setData(sampleData);
    
    // Handle resize
    window.addEventListener('resize', () => {
        portfolioChart.resize(
            container.clientWidth,
            container.clientHeight
        );
    });
}

// Initialize performance chart
function initializePerformanceChart() {
    const container = document.getElementById('performance-chart');
    if (!container) return;
    
    const performanceChart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: container.clientHeight,
        layout: {
            backgroundColor: '#121826',
            textColor: '#d1d4dc',
        },
        grid: {
            vertLines: {
                color: 'rgba(42, 53, 72, 0.5)',
            },
            horzLines: {
                color: 'rgba(42, 53, 72, 0.5)',
            },
        },
        rightPriceScale: {
            borderColor: 'rgba(42, 53, 72, 0.5)',
        },
        timeScale: {
            borderColor: 'rgba(42, 53, 72, 0.5)',
            timeVisible: true,
        },
    });
    
    // Add cumulative return series
    const returnSeries = performanceChart.addLineSeries({
        color: '#26a69a',
        lineWidth: 2,
    });
    
    // Generate sample data for now
    const currentTime = Math.floor(Date.now() / 1000);
    const sampleData = Array.from({ length: 30 }, (_, i) => {
        const time = currentTime - (29 - i) * 24 * 60 * 60;
        let value = 0;
        for (let j = 0; j <= i; j++) {
            value += Math.random() * 0.5 - 0.1;
        }
        return { time, value: value * 100 };
    });
    
    returnSeries.setData(sampleData);
    
    // Handle resize
    window.addEventListener('resize', () => {
        performanceChart.resize(
            container.clientWidth,
            container.clientHeight
        );
    });
}

// Initialize activity chart
function initializeActivityChart() {
    const container = document.getElementById('activity-chart');
    if (!container) return;
    
    const activityChart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: container.clientHeight,
        layout: {
            backgroundColor: '#121826',
            textColor: '#d1d4dc',
        },
        grid: {
            vertLines: {
                color: 'rgba(42, 53, 72, 0.5)',
            },
            horzLines: {
                color: 'rgba(42, 53, 72, 0.5)',
            },
        },
        rightPriceScale: {
            borderColor: 'rgba(42, 53, 72, 0.5)',
        },
        timeScale: {
            borderColor: 'rgba(42, 53, 72, 0.5)',
            timeVisible: true,
        },
    });
    
    // Add buy series
    const buySeries = activityChart.addHistogramSeries({
        color: '#26a69a',
        priceFormat: {
            type: 'volume',
        },
        priceScaleId: 'right',
    });
    
    // Add sell series
    const sellSeries = activityChart.addHistogramSeries({
        color: '#ef5350',
        priceFormat: {
            type: 'volume',
        },
        priceScaleId: 'right',
    });
    
    // Generate sample data for now
    const currentTime = Math.floor(Date.now() / 1000);
    const buyData = Array.from({ length: 30 }, (_, i) => {
        const time = currentTime - (29 - i) * 24 * 60 * 60;
        return { time, value: Math.random() * 10 };
    });
    
    const sellData = Array.from({ length: 30 }, (_, i) => {
        const time = currentTime - (29 - i) * 24 * 60 * 60;
        return { time, value: -Math.random() * 10 };
    });
    
    buySeries.setData(buyData);
    sellSeries.setData(sellData);
    
    // Handle resize
    window.addEventListener('resize', () => {
        activityChart.resize(
            container.clientWidth,
            container.clientHeight
        );
    });
}

// Initialize asset performance chart
function initializeAssetPerformanceChart() {
    const container = document.getElementById('asset-performance-chart');
    if (!container) return;
    
    const ctx = document.createElement('canvas');
    ctx.width = container.clientWidth;
    ctx.height = container.clientHeight;
    container.appendChild(ctx);
    
    const assetChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOGE'],
            datasets: [{
                label: 'Return (%)',
                data: [15.2, -4.5, 8.7, -2.3, 20.1, -12.8],
                backgroundColor: [
                    '#26a69a',
                    '#ef5350',
                    '#26a69a',
                    '#ef5350',
                    '#26a69a',
                    '#ef5350'
                ],
                borderColor: 'rgba(0,0,0,0)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(42, 53, 72, 0.5)'
                    },
                    ticks: {
                        color: '#d1d4dc'
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#d1d4dc'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}
