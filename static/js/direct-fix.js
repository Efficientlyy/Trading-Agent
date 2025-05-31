// Direct DOM manipulation fix for chart and trades
document.addEventListener('DOMContentLoaded', function() {
    console.log("Direct DOM manipulation fix for chart and trades");
    
    // Wait for page to fully load
    setTimeout(() => {
        // Fix for chart container
        const chartContainer = document.getElementById('price-chart');
        if (chartContainer) {
            console.log("Fixing chart container");
            
            // Force visible dimensions
            chartContainer.style.width = '100%';
            chartContainer.style.height = '400px';
            chartContainer.style.display = 'block';
            chartContainer.style.backgroundColor = '#1a2332';
            
            // Create chart with explicit dimensions
            if (typeof LightweightCharts !== 'undefined') {
                console.log("Creating chart with LightweightCharts");
                
                // Create chart
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
                
                // Generate sample data
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
                
                // Set sample data
                candleSeries.setData(sampleData);
                console.log("Set sample chart data");
            } else {
                console.error("LightweightCharts not loaded");
                chartContainer.innerHTML = '<div style="color: white; padding: 20px;">Chart library not loaded. Please refresh the page.</div>';
            }
        } else {
            console.error("Chart container not found");
        }
        
        // Fix for trades container
        const tradesContainer = document.getElementById('trades-container');
        if (tradesContainer) {
            console.log("Fixing trades container");
            
            // Force visible dimensions
            tradesContainer.style.display = 'block';
            tradesContainer.style.maxHeight = '300px';
            tradesContainer.style.overflow = 'auto';
            
            // Generate sample trades
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
            
            // Clear container
            tradesContainer.innerHTML = '';
            
            // Render sample trades
            for (let i = 0; i < sampleTrades.length; i++) {
                const trade = sampleTrades[i];
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
            
            console.log("Set sample trades data");
        } else {
            console.error("Trades container not found");
        }
    }, 1000);
});
