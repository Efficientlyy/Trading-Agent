// Fixed Chart Implementation for MEXC Trading Dashboard
console.log("Loading Fixed Chart Implementation");

document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM Content Loaded - Initializing fixed chart");
    
    // Initialize the chart with proper configuration
    function initFixedChart() {
        console.log("Initializing fixed price chart");
        const chartContainer = document.getElementById('price-chart');
        
        if (!chartContainer) {
            console.error("Chart container not found");
            return;
        }
        
        // Clear loading indicator
        chartContainer.innerHTML = '';
        
        // Force container to have explicit dimensions
        chartContainer.style.width = '100%';
        chartContainer.style.height = '400px';
        chartContainer.style.display = 'block';
        
        // Check if LightweightCharts is available
        if (typeof LightweightCharts === 'undefined') {
            console.error("LightweightCharts library not loaded");
            chartContainer.innerHTML = '<div style="color: white; padding: 20px;">Chart library not loaded. Please refresh the page.</div>';
            return;
        }
        
        try {
            // Create chart with professional dark theme
            const chart = LightweightCharts.createChart(chartContainer, {
                width: chartContainer.clientWidth,
                height: chartContainer.clientHeight,
                layout: {
                    background: { type: 'solid', color: '#121826' },
                    textColor: '#a0aec0',
                },
                grid: {
                    vertLines: { color: 'rgba(45, 55, 72, 0.5)' },
                    horzLines: { color: 'rgba(45, 55, 72, 0.5)' },
                },
                timeScale: {
                    borderColor: '#2d3748',
                    timeVisible: true,
                },
                rightPriceScale: {
                    borderColor: '#2d3748',
                },
            });
            
            // Create candlestick series manually
            const candleSeries = chart.addCandlestickSeries({
                upColor: '#48bb78',
                downColor: '#e53e3e',
                borderUpColor: '#48bb78',
                borderDownColor: '#e53e3e',
                wickUpColor: '#48bb78',
                wickDownColor: '#e53e3e',
            });
            
            // Fetch klines data
            fetchKlinesData(candleSeries, chart);
            
            // Resize chart on window resize
            window.addEventListener('resize', () => {
                if (chart) {
                    chart.applyOptions({
                        width: chartContainer.clientWidth,
                        height: chartContainer.clientHeight,
                    });
                }
            });
            
            console.log("Chart initialization successful");
            
        } catch (error) {
            console.error("Error initializing chart:", error);
            
            // Fallback to simple chart if advanced features fail
            createSimpleChart(chartContainer);
        }
    }
    
    // Fetch klines data from API
    function fetchKlinesData(candleSeries, chart) {
        console.log("Fetching klines data");
        
        fetch('/api/klines?interval=1m')
            .then(response => response.json())
            .then(data => {
                console.log("Klines data received:", data.length, "candles");
                
                if (!data || data.length === 0) {
                    console.error("No klines data received");
                    return;
                }
                
                // Format data for candlestick series
                const formattedData = data.map(kline => ({
                    time: kline.time / 1000,
                    open: kline.open,
                    high: kline.high,
                    low: kline.low,
                    close: kline.close
                }));
                
                console.log("Formatted data sample:", formattedData.slice(0, 3));
                
                // Set candlestick data
                candleSeries.setData(formattedData);
                
                // Fit content to view
                chart.timeScale().fitContent();
                
                console.log("Chart data set successfully");
                
                // Update current price if available
                if (formattedData.length > 0) {
                    const lastCandle = formattedData[formattedData.length - 1];
                    const priceElement = document.getElementById('current-price');
                    if (priceElement) {
                        priceElement.textContent = lastCandle.close.toFixed(2);
                    }
                }
            })
            .catch(error => {
                console.error('Error fetching klines:', error);
                
                // Create sample data if API fails
                createSampleData(candleSeries, chart);
            });
    }
    
    // Create sample data if API fails
    function createSampleData(candleSeries, chart) {
        console.log("Creating sample chart data");
        
        const currentTime = Math.floor(Date.now() / 1000);
        const sampleData = [];
        
        // Generate 50 sample candles
        for (let i = 0; i < 50; i++) {
            const basePrice = 104000 + Math.random() * 1000;
            const time = currentTime - (50 - i) * 60;
            
            sampleData.push({
                time: time,
                open: basePrice,
                high: basePrice + Math.random() * 200,
                low: basePrice - Math.random() * 200,
                close: basePrice + (Math.random() * 400 - 200),
            });
        }
        
        // Set sample data
        candleSeries.setData(sampleData);
        
        // Fit content to view
        chart.timeScale().fitContent();
        
        console.log("Sample chart data set");
    }
    
    // Create simple chart as fallback
    function createSimpleChart(container) {
        console.log("Creating simple chart fallback");
        
        // Clear container
        container.innerHTML = '';
        
        // Create canvas element
        const canvas = document.createElement('canvas');
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        container.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        
        // Draw simple line chart
        fetch('/api/klines?interval=1m')
            .then(response => response.json())
            .then(data => {
                if (!data || data.length === 0) {
                    drawNoDataMessage(ctx, canvas.width, canvas.height);
                    return;
                }
                
                // Extract close prices
                const prices = data.map(kline => kline.close);
                
                // Draw chart
                drawSimpleLineChart(ctx, prices, canvas.width, canvas.height);
            })
            .catch(error => {
                console.error('Error fetching data for simple chart:', error);
                drawNoDataMessage(ctx, canvas.width, canvas.height);
            });
    }
    
    // Draw simple line chart
    function drawSimpleLineChart(ctx, prices, width, height) {
        const padding = 40;
        const chartWidth = width - padding * 2;
        const chartHeight = height - padding * 2;
        
        // Find min and max prices
        const minPrice = Math.min(...prices);
        const maxPrice = Math.max(...prices);
        const priceRange = maxPrice - minPrice;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw background
        ctx.fillStyle = '#121826';
        ctx.fillRect(0, 0, width, height);
        
        // Draw grid
        ctx.strokeStyle = 'rgba(45, 55, 72, 0.5)';
        ctx.lineWidth = 1;
        
        // Horizontal grid lines
        for (let i = 0; i <= 4; i++) {
            const y = padding + (chartHeight / 4) * i;
            ctx.beginPath();
            ctx.moveTo(padding, y);
            ctx.lineTo(width - padding, y);
            ctx.stroke();
            
            // Price labels
            const price = maxPrice - (priceRange / 4) * i;
            ctx.fillStyle = '#a0aec0';
            ctx.font = '12px Inter, sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(price.toFixed(2), padding - 10, y + 4);
        }
        
        // Draw price line
        ctx.strokeStyle = '#3182ce';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        prices.forEach((price, index) => {
            const x = padding + (chartWidth / (prices.length - 1)) * index;
            const y = padding + chartHeight - ((price - minPrice) / priceRange) * chartHeight;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Draw title
        ctx.fillStyle = '#e6e9f0';
        ctx.font = 'bold 14px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('BTC/USDC Price Chart', width / 2, 20);
    }
    
    // Draw no data message
    function drawNoDataMessage(ctx, width, height) {
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw background
        ctx.fillStyle = '#121826';
        ctx.fillRect(0, 0, width, height);
        
        // Draw message
        ctx.fillStyle = '#a0aec0';
        ctx.font = '14px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('No chart data available', width / 2, height / 2);
    }
    
    // Initialize chart with a delay to ensure container is ready
    setTimeout(initFixedChart, 500);
});
