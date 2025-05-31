// MEXC Trading Dashboard - Frontend JavaScript
console.log("Loading MEXC Trading Dashboard JS");

document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM Content Loaded - Initializing dashboard");
    
    // Initialize variables
    let currentPrice = 0;
    let orderBook = { bids: [], asks: [] };
    let trades = [];
    let account = { balances: { BTC: 0, USDC: 0 } };
    let currentInterval = '1m';
    let chart;
    let candleSeries;

    // Initialize the chart
    function initChart() {
        console.log("Initializing price chart");
        const chartContainer = document.getElementById('price-chart');
        
        // Check if LightweightCharts is available
        if (typeof LightweightCharts === 'undefined') {
            console.error("LightweightCharts library not loaded");
            return;
        }
        
        chart = LightweightCharts.createChart(chartContainer, {
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

        candleSeries = chart.addCandlestickSeries({
            upColor: '#48bb78',
            downColor: '#e53e3e',
            borderUpColor: '#48bb78',
            borderDownColor: '#e53e3e',
            wickUpColor: '#48bb78',
            wickDownColor: '#e53e3e',
        });

        // Resize chart on window resize
        window.addEventListener('resize', () => {
            chart.applyOptions({
                width: chartContainer.clientWidth,
                height: chartContainer.clientHeight,
            });
        });

        // Load initial data
        fetchKlines();
    }

    // Fetch ticker data
    async function fetchTicker() {
        console.log("Fetching ticker data");
        try {
            const response = await fetch('/api/ticker');
            const data = await response.json();
            console.log("Ticker data received:", data);
            
            if (data && data.price) {
                updatePrice(data.price);
            }
        } catch (error) {
            console.error('Error fetching ticker:', error);
        }
    }

    // Update ticker price
    function updatePrice(price, change = 0) {
        console.log("Updating price display to:", price);
        currentPrice = price;
        
        const priceElement = document.getElementById('current-price');
        if (priceElement) {
            priceElement.textContent = price.toFixed(2);
        } else {
            console.error("Price element not found");
        }
        
        // Update buy/sell price inputs if they exist
        const buyPriceInput = document.getElementById('buy-price');
        const sellPriceInput = document.getElementById('sell-price');
        
        if (buyPriceInput) buyPriceInput.value = price.toFixed(2);
        if (sellPriceInput) sellPriceInput.value = price.toFixed(2);
        
        // Update price change if provided
        const changeElement = document.getElementById('price-change');
        if (changeElement) {
            changeElement.textContent = change > 0 ? `+${change.toFixed(2)}%` : `${change.toFixed(2)}%`;
            changeElement.className = 'ticker-change ' + (change >= 0 ? 'positive' : 'negative');
        }
    }

    // Fetch klines data
    async function fetchKlines() {
        console.log("Fetching klines data");
        try {
            const response = await fetch(`/api/klines?interval=${currentInterval}`);
            const data = await response.json();
            console.log("Klines data received:", data);
            
            if (data && data.length > 0 && candleSeries) {
                const formattedData = data.map(kline => ({
                    time: kline.time / 1000,
                    open: kline.open,
                    high: kline.high,
                    low: kline.low,
                    close: kline.close,
                }));

                candleSeries.setData(formattedData);
                
                // Update current price if available
                if (formattedData.length > 0) {
                    updatePrice(formattedData[formattedData.length - 1].close);
                }
            }
        } catch (error) {
            console.error('Error fetching klines:', error);
        }
    }

    // Fetch and update order book
    async function fetchOrderBook() {
        console.log("Fetching order book data");
        try {
            const response = await fetch('/api/orderbook');
            const data = await response.json();
            console.log("Order book data received:", data);
            
            if (data && data.bids && data.asks) {
                orderBook = data;
                renderOrderBook();
            }
        } catch (error) {
            console.error('Error fetching order book:', error);
        }
    }

    // Render order book
    function renderOrderBook() {
        console.log("Rendering order book");
        const asksContainer = document.getElementById('asks-container');
        const bidsContainer = document.getElementById('bids-container');
        
        if (!asksContainer || !bidsContainer) {
            console.error("Order book containers not found");
            return;
        }
        
        // Clear containers
        asksContainer.innerHTML = '';
        bidsContainer.innerHTML = '';
        
        // Calculate max volume for depth visualization
        const maxBidVolume = Math.max(...orderBook.bids.map(bid => bid[1]), 0.1);
        const maxAskVolume = Math.max(...orderBook.asks.map(ask => ask[1]), 0.1);
        
        // Render asks (in reverse order - highest to lowest)
        const sortedAsks = [...orderBook.asks].sort((a, b) => b[0] - a[0]);
        for (let i = 0; i < Math.min(sortedAsks.length, 15); i++) {
            const [price, amount] = sortedAsks[i];
            const total = price * amount;
            const depthPercentage = (amount / maxAskVolume) * 100;
            
            const row = document.createElement('div');
            row.className = 'order-book-row ask';
            row.innerHTML = `
                <span>${price.toFixed(2)}</span>
                <span>${amount.toFixed(6)}</span>
                <span>${total.toFixed(2)}</span>
                <div class="depth-bar ask" style="width: ${depthPercentage}%"></div>
            `;
            asksContainer.appendChild(row);
        }
        
        // Render bids
        for (let i = 0; i < Math.min(orderBook.bids.length, 15); i++) {
            const [price, amount] = orderBook.bids[i];
            const total = price * amount;
            const depthPercentage = (amount / maxBidVolume) * 100;
            
            const row = document.createElement('div');
            row.className = 'order-book-row bid';
            row.innerHTML = `
                <span>${price.toFixed(2)}</span>
                <span>${amount.toFixed(6)}</span>
                <span>${total.toFixed(2)}</span>
                <div class="depth-bar bid" style="width: ${depthPercentage}%"></div>
            `;
            bidsContainer.appendChild(row);
        }
        
        // Update spread
        const spreadElement = document.getElementById('spread');
        if (spreadElement && orderBook.asks.length > 0 && orderBook.bids.length > 0) {
            const lowestAsk = Math.min(...orderBook.asks.map(ask => ask[0]));
            const highestBid = Math.max(...orderBook.bids.map(bid => bid[0]));
            const spread = lowestAsk - highestBid;
            const spreadPercentage = (spread / lowestAsk) * 100;
            
            spreadElement.textContent = `Spread: ${spread.toFixed(2)} (${spreadPercentage.toFixed(2)}%)`;
        }
    }

    // Fetch and update trades
    async function fetchTrades() {
        console.log("Fetching trades data");
        try {
            const response = await fetch('/api/trades');
            const data = await response.json();
            console.log("Trades data received:", data);
            
            if (data && data.length > 0) {
                trades = data;
                renderTrades();
            }
        } catch (error) {
            console.error('Error fetching trades:', error);
        }
    }

    // Render trades
    function renderTrades() {
        console.log("Rendering trades");
        const tradesContainer = document.getElementById('trades-container');
        
        if (!tradesContainer) {
            console.error("Trades container not found");
            return;
        }
        
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

    // Fetch account information
    async function fetchAccount() {
        console.log("Fetching account data");
        try {
            const response = await fetch('/api/account');
            const data = await response.json();
            console.log("Account data received:", data);
            
            if (data && data.balances) {
                account = data;
                renderAccount();
            }
        } catch (error) {
            console.error('Error fetching account:', error);
        }
    }

    // Render account information
    function renderAccount() {
        console.log("Rendering account balances");
        const btcBalanceElement = document.getElementById('btc-balance');
        const usdcBalanceElement = document.getElementById('usdc-balance');
        
        if (btcBalanceElement) {
            btcBalanceElement.textContent = account.balances.BTC.toFixed(8);
        } else {
            console.error("BTC balance element not found");
        }
        
        if (usdcBalanceElement) {
            usdcBalanceElement.textContent = account.balances.USDC.toFixed(2);
        } else {
            console.error("USDC balance element not found");
        }
    }

    // Place order
    async function placeOrder(side, amount, price) {
        console.log(`Placing ${side} order: ${amount} BTC at ${price} USDC`);
        try {
            const response = await fetch('/api/order', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    symbol: 'BTCUSDC',
                    side: side,
                    type: 'LIMIT',
                    quantity: amount,
                    price: price,
                }),
            });
            
            const result = await response.json();
            console.log("Order result:", result);
            
            if (result.success) {
                alert(`${side} order placed successfully!`);
                fetchAccount();
            } else {
                alert(`Error: ${result.message}`);
            }
        } catch (error) {
            console.error('Error placing order:', error);
            alert('Failed to place order. Please try again.');
        }
    }

    // Update buy total
    function updateBuyTotal() {
        const buyAmountElement = document.getElementById('buy-amount');
        const buyPriceElement = document.getElementById('buy-price');
        const buyTotalElement = document.getElementById('buy-total');
        
        if (buyAmountElement && buyPriceElement && buyTotalElement) {
            const amount = parseFloat(buyAmountElement.value) || 0;
            const price = parseFloat(buyPriceElement.value) || 0;
            const total = amount * price;
            buyTotalElement.textContent = total.toFixed(2);
        }
    }

    // Update sell total
    function updateSellTotal() {
        const sellAmountElement = document.getElementById('sell-amount');
        const sellPriceElement = document.getElementById('sell-price');
        const sellTotalElement = document.getElementById('sell-total');
        
        if (sellAmountElement && sellPriceElement && sellTotalElement) {
            const amount = parseFloat(sellAmountElement.value) || 0;
            const price = parseFloat(sellPriceElement.value) || 0;
            const total = amount * price;
            sellTotalElement.textContent = total.toFixed(2);
        }
    }

    // Initialize the application
    function init() {
        console.log("Initializing dashboard application");
        
        // Immediately fetch and display data
        fetchTicker();
        fetchOrderBook();
        fetchTrades();
        fetchAccount();
        
        // Initialize chart if container exists
        const chartContainer = document.getElementById('price-chart');
        if (chartContainer && typeof LightweightCharts !== 'undefined') {
            // Delay chart initialization slightly to ensure container is ready
            setTimeout(() => {
                initChart();
            }, 100);
        } else {
            console.error("Chart container not found or LightweightCharts not loaded");
        }
        
        // Set up polling for updates
        setInterval(fetchTicker, 3000);
        setInterval(fetchOrderBook, 5000);
        setInterval(fetchTrades, 5000);
        setInterval(fetchAccount, 10000);
        
        // Set up event listeners for time frame buttons
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
                    fetchKlines();
                });
            });
        }
        
        // Set up event listeners for buy/sell forms
        const buyAmountInput = document.getElementById('buy-amount');
        const buyPriceInput = document.getElementById('buy-price');
        const sellAmountInput = document.getElementById('sell-amount');
        const sellPriceInput = document.getElementById('sell-price');
        const buyButton = document.getElementById('buy-btn');
        const sellButton = document.getElementById('sell-btn');
        
        if (buyAmountInput && buyPriceInput) {
            buyAmountInput.addEventListener('input', updateBuyTotal);
            buyPriceInput.addEventListener('input', updateBuyTotal);
        }
        
        if (sellAmountInput && sellPriceInput) {
            sellAmountInput.addEventListener('input', updateSellTotal);
            sellPriceInput.addEventListener('input', updateSellTotal);
        }
        
        if (buyButton) {
            buyButton.addEventListener('click', () => {
                const amount = parseFloat(buyAmountInput.value);
                const price = parseFloat(buyPriceInput.value);
                
                if (isNaN(amount) || amount <= 0) {
                    alert('Please enter a valid amount');
                    return;
                }
                
                if (isNaN(price) || price <= 0) {
                    alert('Please enter a valid price');
                    return;
                }
                
                placeOrder('BUY', amount, price);
            });
        }
        
        if (sellButton) {
            sellButton.addEventListener('click', () => {
                const amount = parseFloat(sellAmountInput.value);
                const price = parseFloat(sellPriceInput.value);
                
                if (isNaN(amount) || amount <= 0) {
                    alert('Please enter a valid amount');
                    return;
                }
                
                if (isNaN(price) || price <= 0) {
                    alert('Please enter a valid price');
                    return;
                }
                
                placeOrder('SELL', amount, price);
            });
        }
        
        console.log("Dashboard initialization complete");
    }

    // Start the application
    init();
    
    // Expose functions to global scope for debugging
    window.mexcDashboard = {
        fetchTicker,
        fetchOrderBook,
        fetchTrades,
        fetchAccount,
        fetchKlines,
        updatePrice,
        renderOrderBook,
        renderTrades,
        renderAccount
    };
});
