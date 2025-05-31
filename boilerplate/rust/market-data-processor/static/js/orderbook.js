/**
 * Orderbook rendering and management
 */

// Store current orderbook data
let currentOrderbook = {
    asks: [],
    bids: []
};

// Maximum number of levels to display
const MAX_LEVELS = 15;

// Update orderbook display
function updateOrderbook(symbol) {
    fetchAPI(`/api/orderbook/${symbol}`)
        .then(data => {
            if (!data) return;
            
            currentOrderbook = data;
            renderOrderbook();
        });
}

// Render orderbook data
function renderOrderbook() {
    const asksContainer = document.getElementById('orderbook-asks');
    const bidsContainer = document.getElementById('orderbook-bids');
    const obPriceElement = document.getElementById('ob-current-price');
    
    if (!asksContainer || !bidsContainer || !obPriceElement) return;
    
    // Clear containers
    asksContainer.innerHTML = '';
    bidsContainer.innerHTML = '';
    
    // Get current ticker price
    const symbol = document.getElementById('trading-pair-select').value;
    fetchAPI(`/api/ticker/${symbol}`)
        .then(ticker => {
            if (!ticker) return;
            
            // Update current price
            obPriceElement.textContent = formatPrice(ticker.price, symbol);
            obPriceElement.className = '';
        });
    
    // Calculate max total for depth visualization
    let maxAsksTotal = 0;
    let maxBidsTotal = 0;
    
    // Sort asks (lowest to highest)
    const sortedAsks = [...currentOrderbook.asks].sort((a, b) => a[0] - b[0]);
    
    // Sort bids (highest to lowest)
    const sortedBids = [...currentOrderbook.bids].sort((a, b) => b[0] - a[0]);
    
    // Calculate cumulative totals for asks
    let asksCumulative = [];
    let runningTotal = 0;
    
    for (let i = 0; i < sortedAsks.length; i++) {
        runningTotal += parseFloat(sortedAsks[i][1]);
        asksCumulative.push(runningTotal);
        if (runningTotal > maxAsksTotal) maxAsksTotal = runningTotal;
    }
    
    // Calculate cumulative totals for bids
    let bidsCumulative = [];
    runningTotal = 0;
    
    for (let i = 0; i < sortedBids.length; i++) {
        runningTotal += parseFloat(sortedBids[i][1]);
        bidsCumulative.push(runningTotal);
        if (runningTotal > maxBidsTotal) maxBidsTotal = runningTotal;
    }
    
    // Render asks (in reverse order - highest to lowest)
    for (let i = Math.min(sortedAsks.length, MAX_LEVELS) - 1; i >= 0; i--) {
        const price = parseFloat(sortedAsks[i][0]);
        const amount = parseFloat(sortedAsks[i][1]);
        const total = asksCumulative[i];
        const depthPercent = (total / maxAsksTotal) * 100;
        
        const row = document.createElement('div');
        row.className = 'orderbook-row ask';
        
        // Create depth bar
        const depthBar = document.createElement('div');
        depthBar.className = 'depth-bar';
        depthBar.style.width = `${depthPercent}%`;
        row.appendChild(depthBar);
        
        // Create price, amount, total columns
        const priceCol = document.createElement('div');
        priceCol.className = 'ask-price';
        priceCol.textContent = formatPrice(price);
        
        const amountCol = document.createElement('div');
        amountCol.textContent = formatNumber(amount, 5);
        
        const totalCol = document.createElement('div');
        totalCol.textContent = formatNumber(total, 5);
        
        row.appendChild(priceCol);
        row.appendChild(amountCol);
        row.appendChild(totalCol);
        
        asksContainer.appendChild(row);
    }
    
    // Render bids (highest to lowest)
    for (let i = 0; i < Math.min(sortedBids.length, MAX_LEVELS); i++) {
        const price = parseFloat(sortedBids[i][0]);
        const amount = parseFloat(sortedBids[i][1]);
        const total = bidsCumulative[i];
        const depthPercent = (total / maxBidsTotal) * 100;
        
        const row = document.createElement('div');
        row.className = 'orderbook-row bid';
        
        // Create depth bar
        const depthBar = document.createElement('div');
        depthBar.className = 'depth-bar';
        depthBar.style.width = `${depthPercent}%`;
        row.appendChild(depthBar);
        
        // Create price, amount, total columns
        const priceCol = document.createElement('div');
        priceCol.className = 'bid-price';
        priceCol.textContent = formatPrice(price);
        
        const amountCol = document.createElement('div');
        amountCol.textContent = formatNumber(amount, 5);
        
        const totalCol = document.createElement('div');
        totalCol.textContent = formatNumber(total, 5);
        
        row.appendChild(priceCol);
        row.appendChild(amountCol);
        row.appendChild(totalCol);
        
        bidsContainer.appendChild(row);
    }
}

// Update recent trades display
function updateRecentTrades(symbol) {
    fetchAPI(`/api/trades/${symbol}`)
        .then(data => {
            if (!data || !data.length) return;
            
            const tradesBody = document.getElementById('recent-trades-body');
            if (!tradesBody) return;
            
            // Clear container
            tradesBody.innerHTML = '';
            
            // Process recent trades (newest first)
            const recentTrades = [...data].sort((a, b) => b.time - a.time).slice(0, 15);
            
            for (const trade of recentTrades) {
                const row = document.createElement('tr');
                row.className = 'trade-row';
                
                // Price cell
                const priceCell = document.createElement('td');
                priceCell.className = trade.buyer_maker ? 'trade-sell' : 'trade-buy';
                priceCell.textContent = formatPrice(trade.price);
                row.appendChild(priceCell);
                
                // Amount cell
                const amountCell = document.createElement('td');
                amountCell.textContent = formatNumber(trade.quantity, 5);
                row.appendChild(amountCell);
                
                // Time cell
                const timeCell = document.createElement('td');
                timeCell.textContent = formatTime(trade.time);
                row.appendChild(timeCell);
                
                tradesBody.appendChild(row);
            }
        });
}
