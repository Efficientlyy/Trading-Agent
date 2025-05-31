// Fix for candlestick chart rendering
function fixChartRendering() {
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
      vertLine: {
        color: '#3182ce',
        width: 1,
        style: LightweightCharts.LineStyle.Dashed,
      },
      horzLine: {
        color: '#3182ce',
        width: 1,
        style: LightweightCharts.LineStyle.Dashed,
      },
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
  
  // Fetch klines data
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
      }
    })
    .catch(error => {
      console.error('Error fetching klines:', error);
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
  // Fetch trades data
  fetch('/api/trades')
    .then(response => response.json())
    .then(data => {
      console.log("Trades data received:", data);
      
      if (data && data.length > 0) {
        const tradesContainer = document.getElementById('trades-container');
        
        if (!tradesContainer) {
          console.error("Trades container not found");
          return;
        }
        
        // Clear container
        tradesContainer.innerHTML = '';
        
        // Render trades
        for (let i = 0; i < Math.min(data.length, 30); i++) {
          const trade = data[i];
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
    })
    .catch(error => {
      console.error('Error fetching trades:', error);
    });
}

// Call both fixes when the page is loaded
document.addEventListener('DOMContentLoaded', function() {
  console.log("Applying dashboard fixes");
  
  // Fix chart rendering
  const chartComponents = fixChartRendering();
  
  // Fix trades rendering
  fixTradesRendering();
  
  // Set up polling for updates
  setInterval(fixTradesRendering, 5000);
  
  // Set up time frame buttons
  const timeFrameButtons = document.querySelectorAll('.time-frame');
  if (timeFrameButtons.length > 0 && chartComponents) {
    timeFrameButtons.forEach(button => {
      button.addEventListener('click', () => {
        const activeButton = document.querySelector('.time-frame.active');
        if (activeButton) {
          activeButton.classList.remove('active');
        }
        button.classList.add('active');
        
        const interval = button.dataset.interval;
        
        // Fetch klines for the selected interval
        fetch(`/api/klines?interval=${interval}`)
          .then(response => response.json())
          .then(data => {
            if (data && data.length > 0 && chartComponents.candleSeries) {
              const formattedData = data.map(kline => ({
                time: Math.floor(kline.time / 1000),
                open: kline.open,
                high: kline.high,
                low: kline.low,
                close: kline.close,
              }));
              
              chartComponents.candleSeries.setData(formattedData);
            }
          })
          .catch(error => {
            console.error('Error fetching klines:', error);
          });
      });
    });
  }
});
