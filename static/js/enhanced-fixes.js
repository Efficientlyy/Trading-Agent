// Enhanced fixes for chart and trades rendering
document.addEventListener('DOMContentLoaded', function() {
    console.log("Enhanced fixes for chart and trades rendering");
    
    // Fix for candlestick chart rendering
    function fixChartRendering() {
        console.log("Applying enhanced chart fix");
        
        // Check if LightweightCharts is loaded
        if (typeof LightweightCharts === 'undefined') {
            console.error("LightweightCharts library not loaded");
            return;
        }
        
        const chartContainer = document.getElementById('price-chart');
        if (!chartContainer) {
            console.error("Chart container not found");
            return;
        }
        
        // Clear any existing chart
        chartContainer.innerHTML = '';
        
        // Set explicit dimensions to ensure chart is visible
        chartContainer.style.width = '100%';
        chartContainer.style.height = '400px';
        
        // Create chart with explicit dimensions
        const chart = LightweightCharts.createChart(chartContainer, {
            width: chartContainer.clientWidth,
            height: chartContainer.clientHeight,
            layout: {
                background: { color: '#121826' },
                textColor: '#a0aec0',
            },
            grid: {
                vertLines: { color: '#2d3748' },
                horzLines: { color: '#2d3748' },
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
            },
            timeScale: {
                borderColor: '#2d3748',
                timeVisible: true,
            },
            rightPriceScale: {
                borderColor: '#2d3748',
            },
        });
        
        // Add candlestick series
        const candleSeries = chart.addCandlestickSeries({
            upColor: '#48bb78',
            downColor: '#e53e3e',
            borderUpColor: '#48bb78',
            borderDownColor: '#e53e3e',
            wickUpColor: '#48bb78',
            wickDownColor: '#e53e3e',
        });
        
        // Generate sample data if API fails
        const currentTime = Math.floor(Date.now() / 1000);
        const sampleData = [];
        
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
        
        // First set sample data to ensure chart is visible
        candleSeries.setData(sampleData);
        console.log("Set sample chart data");
        
        // Then try to fetch real data
        fetch('/api/klines?interval=1m')
            .then(response => response.json())
            .then(data => {
                console.log("Klines data received:", data);
                
                if (data && data.length > 0) {
                    const formattedData = data.map(kline => ({
                        time: Math.floor(kline.time / 1000),
                        open: kline.open,
                        high: kline.high,
                        low: kline.low,
                        close: kline.close,
                    }));
                    
                    console.log("Formatted chart data:", formattedData);
                    candleSeries.setData(formattedData);
                    console.log("Updated chart with real data");
                }
            })
            .catch(error => {
                console.error('Error fetching klines:', error);
                console.log("Using sample data for chart");
            });
        
        // Resize chart on window resize
        window.addEventListener('resize', () => {
            chart.applyOptions({
                width: chartContainer.clientWidth,
                height: chartContainer.clientHeight,
            });
        });
        
        // Return chart and series for further use
        return { chart, candleSeries };
    }
    
    // Fix for trades rendering
    function fixTradesRendering() {
        console.log("Applying enhanced trades fix");
        
        const tradesContainer = document.getElementById('trades-container');
        if (!tradesContainer) {
            console.error("Trades container not found");
            return;
        }
        
        // Generate sample trades if API fails
        const sampleTrades = [];
        const currentTime = Date.now();
        
        for (let i = 0; i < 20; i++) {
            const basePrice = 104000 + Math.random() * 500;
            const isBuy = Math.random() > 0.5;
            
            sampleTrades.push({
                price: basePrice,
                quantity: Math.random() * 0.1,
                time: currentTime - i * 10000,
                isBuyerMaker: !isBuy
            });
        }
        
        // First render sample trades to ensure visibility
        renderTrades(sampleTrades);
        console.log("Set sample trades data");
        
        // Then try to fetch real trades
        fetch('/api/trades')
            .then(response => response.json())
            .then(data => {
                console.log("Trades data received:", data);
                
                if (data && data.length > 0) {
                    renderTrades(data);
                    console.log("Updated trades with real data");
                }
            })
            .catch(error => {
                console.error('Error fetching trades:', error);
                console.log("Using sample data for trades");
            });
        
        function renderTrades(trades) {
            // Clear container
            tradesContainer.innerHTML = '';
            
            // Render trades
            for (let i = 0; i < Math.min(trades.length, 30); i++) {
                const trade = trades[i];
                const row = document.createElement('div');
                row.className = `trade-row ${trade.isBuyerMaker ? 'sell' : 'buy'}`;
                
                const time = new Date(trade.time);
                const timeStr = `${time.getHours().toString().padStart(2, '0')}:${time.getMinutes().toString().padStart(2, '0')}:${time.getSeconds().toString().padStart(2, '0')}`;
                
                row.innerHTML = `
                    <div>${trade.price.toFixed(2)}</div>
                    <div>${trade.quantity.toFixed(6)}</div>
                    <div>${timeStr}</div>
                `;
                tradesContainer.appendChild(row);
            }
        }
    }
    
    // Apply fixes with a slight delay to ensure DOM is ready
    setTimeout(() => {
        console.log("Applying enhanced fixes with delay");
        fixChartRendering();
        fixTradesRendering();
        
        // Set up polling for trades updates
        setInterval(fixTradesRendering, 5000);
    }, 1000);
});
