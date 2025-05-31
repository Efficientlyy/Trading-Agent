// Enhanced Chart Implementation for MEXC Trading Dashboard
console.log("Loading Enhanced Chart Implementation");

document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM Content Loaded - Initializing enhanced chart");
    
    // Chart state variables
    let chart;
    let candleSeries;
    let volumeSeries;
    let currentInterval = '1m';
    let chartInitialized = false;
    
    // Initialize the chart with proper sizing and configuration
    function initEnhancedChart() {
        console.log("Initializing enhanced price chart");
        const chartContainer = document.getElementById('price-chart');
        
        if (!chartContainer) {
            console.error("Chart container not found");
            return;
        }
        
        // Check if LightweightCharts is available
        if (typeof LightweightCharts === 'undefined') {
            console.error("LightweightCharts library not loaded");
            chartContainer.innerHTML = '<div style="color: white; padding: 20px;">Chart library not loaded. Please refresh the page.</div>';
            return;
        }
        
        // Force container to have explicit dimensions
        chartContainer.style.width = '100%';
        chartContainer.style.height = '400px';
        chartContainer.style.display = 'block';
        
        // Create chart with professional dark theme
        chart = LightweightCharts.createChart(chartContainer, {
            width: chartContainer.clientWidth,
            height: chartContainer.clientHeight,
            layout: {
                background: { type: 'solid', color: '#121826' },
                textColor: '#a0aec0',
                fontSize: 12,
                fontFamily: 'Inter, sans-serif',
            },
            grid: {
                vertLines: { color: 'rgba(45, 55, 72, 0.5)', style: 1 },
                horzLines: { color: 'rgba(45, 55, 72, 0.5)', style: 1 },
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
                vertLine: {
                    color: '#3182ce',
                    width: 1,
                    style: LightweightCharts.LineStyle.Dashed,
                    labelBackgroundColor: '#232f42',
                },
                horzLine: {
                    color: '#3182ce',
                    width: 1,
                    style: LightweightCharts.LineStyle.Dashed,
                    labelBackgroundColor: '#232f42',
                },
            },
            timeScale: {
                borderColor: '#2d3748',
                timeVisible: true,
                secondsVisible: false,
                tickMarkFormatter: (time, tickMarkType, locale) => {
                    const date = new Date(time * 1000);
                    const hours = date.getHours().toString().padStart(2, '0');
                    const minutes = date.getMinutes().toString().padStart(2, '0');
                    return `${hours}:${minutes}`;
                },
            },
            rightPriceScale: {
                borderColor: '#2d3748',
                scaleMargins: {
                    top: 0.1,
                    bottom: 0.2,
                },
            },
            handleScroll: {
                vertTouchDrag: false,
            },
        });

        // Add candlestick series
        candleSeries = chart.addCandlestickSeries({
            upColor: '#48bb78',
            downColor: '#e53e3e',
            borderUpColor: '#48bb78',
            borderDownColor: '#e53e3e',
            wickUpColor: '#48bb78',
            wickDownColor: '#e53e3e',
            priceFormat: {
                type: 'price',
                precision: 2,
                minMove: 0.01,
            },
        });
        
        // Add volume series
        volumeSeries = chart.addHistogramSeries({
            color: '#4299e1',
            priceFormat: {
                type: 'volume',
            },
            priceScaleId: '',
            scaleMargins: {
                top: 0.8,
                bottom: 0,
            },
        });
        
        // Add EMA indicator
        const ema20Series = chart.addLineSeries({
            color: '#ecc94b',
            lineWidth: 2,
            priceLineVisible: false,
            lastValueVisible: false,
            title: 'EMA 20',
        });
        
        // Add Bollinger Bands
        const upperBandSeries = chart.addLineSeries({
            color: 'rgba(76, 175, 80, 0.5)',
            lineWidth: 1,
            priceLineVisible: false,
            lastValueVisible: false,
            title: 'Upper Band',
        });
        
        const lowerBandSeries = chart.addLineSeries({
            color: 'rgba(76, 175, 80, 0.5)',
            lineWidth: 1,
            priceLineVisible: false,
            lastValueVisible: false,
            title: 'Lower Band',
        });
        
        // Resize chart on window resize
        window.addEventListener('resize', () => {
            if (chart) {
                chart.applyOptions({
                    width: chartContainer.clientWidth,
                    height: chartContainer.clientHeight,
                });
            }
        });
        
        // Set chart as initialized
        chartInitialized = true;
        
        // Load initial data
        fetchAndRenderKlines();
        
        // Return series for external use
        return {
            candleSeries,
            volumeSeries,
            ema20Series,
            upperBandSeries,
            lowerBandSeries
        };
    }
    
    // Calculate EMA
    function calculateEMA(data, period) {
        const k = 2 / (period + 1);
        let emaData = [];
        let ema = data[0].close;
        
        for (let i = 0; i < data.length; i++) {
            ema = data[i].close * k + ema * (1 - k);
            emaData.push({
                time: data[i].time,
                value: ema
            });
        }
        
        return emaData;
    }
    
    // Calculate Bollinger Bands
    function calculateBollingerBands(data, period, multiplier) {
        let sumSquaredDiff = 0;
        let sum = 0;
        let upperBand = [];
        let lowerBand = [];
        
        // Calculate SMA first
        for (let i = 0; i < period; i++) {
            sum += data[i].close;
        }
        
        let sma = sum / period;
        
        // Calculate standard deviation and bands
        for (let i = 0; i < data.length; i++) {
            if (i >= period) {
                sum = sum - data[i - period].close + data[i].close;
                sma = sum / period;
            }
            
            // Calculate sum of squared differences from mean
            sumSquaredDiff = 0;
            for (let j = Math.max(0, i - period + 1); j <= i; j++) {
                sumSquaredDiff += Math.pow(data[j].close - sma, 2);
            }
            
            // Calculate standard deviation
            const stdDev = Math.sqrt(sumSquaredDiff / period);
            
            // Calculate upper and lower bands
            upperBand.push({
                time: data[i].time,
                value: sma + (multiplier * stdDev)
            });
            
            lowerBand.push({
                time: data[i].time,
                value: sma - (multiplier * stdDev)
            });
        }
        
        return { upperBand, lowerBand };
    }
    
    // Fetch and render klines data
    async function fetchAndRenderKlines() {
        console.log("Fetching klines data for interval:", currentInterval);
        try {
            const response = await fetch(`/api/klines?interval=${currentInterval}`);
            const data = await response.json();
            console.log("Klines data received:", data);
            
            if (!data || data.length === 0) {
                console.error("No klines data received");
                return;
            }
            
            if (!chartInitialized) {
                console.error("Chart not initialized");
                return;
            }
            
            // Format data for candlestick series
            const formattedData = data.map(kline => ({
                time: kline.time / 1000,
                open: kline.open,
                high: kline.high,
                low: kline.low,
                close: kline.close,
                volume: kline.volume
            }));
            
            // Set candlestick data
            candleSeries.setData(formattedData);
            
            // Set volume data
            const volumeData = formattedData.map(d => ({
                time: d.time,
                value: d.volume,
                color: d.open <= d.close ? 'rgba(72, 187, 120, 0.5)' : 'rgba(229, 62, 62, 0.5)'
            }));
            volumeSeries.setData(volumeData);
            
            // Calculate and set EMA
            const ema20Data = calculateEMA(formattedData, 20);
            chart.getSeries().filter(s => s.options().title === 'EMA 20')[0].setData(ema20Data);
            
            // Calculate and set Bollinger Bands
            const { upperBand, lowerBand } = calculateBollingerBands(formattedData, 20, 2);
            chart.getSeries().filter(s => s.options().title === 'Upper Band')[0].setData(upperBand);
            chart.getSeries().filter(s => s.options().title === 'Lower Band')[0].setData(lowerBand);
            
            // Update current price if available
            if (formattedData.length > 0) {
                const lastCandle = formattedData[formattedData.length - 1];
                const priceElement = document.getElementById('current-price');
                if (priceElement) {
                    priceElement.textContent = lastCandle.close.toFixed(2);
                }
                
                // Update buy/sell price inputs
                const buyPriceInput = document.getElementById('buy-price');
                const sellPriceInput = document.getElementById('sell-price');
                if (buyPriceInput) buyPriceInput.value = lastCandle.close.toFixed(2);
                if (sellPriceInput) sellPriceInput.value = lastCandle.close.toFixed(2);
            }
            
            // Fit content to view
            chart.timeScale().fitContent();
            
        } catch (error) {
            console.error('Error fetching klines:', error);
        }
    }
    
    // Set up time frame buttons
    function setupTimeFrameButtons() {
        const timeFrameButtons = document.querySelectorAll('.time-frame');
        if (timeFrameButtons.length > 0) {
            timeFrameButtons.forEach(button => {
                button.addEventListener('click', () => {
                    const activeButton = document.querySelector('.time-frame.active');
                    if (activeButton) {
                        activeButton.classList.remove('active');
                    }
                    button.classList.add('active');
                    currentInterval = button.dataset.interval;
                    fetchAndRenderKlines();
                });
            });
        }
    }
    
    // Initialize chart and set up event listeners
    function init() {
        // Initialize chart with a slight delay to ensure container is ready
        setTimeout(() => {
            initEnhancedChart();
            setupTimeFrameButtons();
            
            // Set up polling for chart updates
            setInterval(fetchAndRenderKlines, 30000);
        }, 500);
    }
    
    // Start initialization
    init();
    
    // Expose functions for external use
    window.enhancedChart = {
        refreshChart: fetchAndRenderKlines,
        setInterval: (interval) => {
            currentInterval = interval;
            fetchAndRenderKlines();
        }
    };
});
