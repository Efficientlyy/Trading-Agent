// Enhanced Trades Display Implementation for MEXC Trading Dashboard
console.log("Loading Enhanced Trades Display Implementation v2");

document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM Content Loaded - Initializing enhanced trades display v2");
    
    // Trades state variables
    let tradesData = [];
    let tradesInitialized = false;
    let lastTradeTimestamp = 0;
    
    // Initialize the trades display with proper styling and animation
    function initEnhancedTrades() {
        console.log("Initializing enhanced trades display v2");
        const tradesContainer = document.getElementById('trades-container');
        
        if (!tradesContainer) {
            console.error("Trades container not found");
            return;
        }
        
        // Force container to have explicit dimensions and visibility
        const tradesContent = tradesContainer.closest('.trades-content');
        if (tradesContent) {
            tradesContent.style.maxHeight = '300px';
            tradesContent.style.height = '100%';
            tradesContent.style.overflowY = 'auto';
            tradesContent.style.display = 'block';
        }
        
        const tradesCard = tradesContainer.closest('.trades');
        if (tradesCard) {
            tradesCard.style.display = 'block';
            tradesCard.style.visibility = 'visible';
            tradesCard.style.height = 'auto';
            tradesCard.style.minHeight = '300px';
        }
        
        tradesContainer.style.maxHeight = '250px';
        tradesContainer.style.overflowY = 'auto';
        tradesContainer.style.display = 'block';
        
        // Add custom styling for smooth animations
        const style = document.createElement('style');
        style.textContent = `
            .trade-row {
                transition: background-color 0.5s ease;
                position: relative;
                padding: 4px 0;
                border-bottom: 1px solid rgba(45, 55, 72, 0.3);
            }
            
            .trade-row.new {
                animation: fadeIn 0.5s ease;
            }
            
            .trade-row.buy::before {
                content: "";
                position: absolute;
                left: -10px;
                top: 50%;
                transform: translateY(-50%);
                width: 4px;
                height: 80%;
                background-color: var(--buy-color);
                border-radius: 2px;
            }
            
            .trade-row.sell::before {
                content: "";
                position: absolute;
                left: -10px;
                top: 50%;
                transform: translateY(-50%);
                width: 4px;
                height: 80%;
                background-color: var(--sell-color);
                border-radius: 2px;
            }
            
            @keyframes fadeIn {
                from { background-color: rgba(72, 187, 120, 0.2); }
                to { background-color: transparent; }
            }
            
            @keyframes fadeInSell {
                from { background-color: rgba(229, 62, 62, 0.2); }
                to { background-color: transparent; }
            }
            
            .trade-row.buy.new {
                animation: fadeIn 0.5s ease;
            }
            
            .trade-row.sell.new {
                animation: fadeInSell 0.5s ease;
            }
            
            .trade-size-indicator {
                position: absolute;
                right: 0;
                top: 0;
                bottom: 0;
                opacity: 0.1;
                z-index: 0;
            }
            
            .trade-size-indicator.buy {
                background-color: var(--buy-color);
            }
            
            .trade-size-indicator.sell {
                background-color: var(--sell-color);
            }
            
            .trade-row span {
                position: relative;
                z-index: 1;
            }
            
            .trade-price {
                font-weight: 600;
            }
            
            .trade-quantity {
                opacity: 0.9;
            }
            
            .trade-time {
                font-size: 0.8em;
                opacity: 0.8;
            }
            
            /* Make trades section more visible */
            .trades {
                display: block !important;
                visibility: visible !important;
                height: auto !important;
                min-height: 300px !important;
                overflow: visible !important;
            }
            
            .trades-content {
                height: auto !important;
                min-height: 250px !important;
                max-height: 300px !important;
                overflow-y: auto !important;
                display: block !important;
            }
            
            #trades-container {
                display: block !important;
                min-height: 100px !important;
                max-height: 250px !important;
                overflow-y: auto !important;
            }
            
            /* Add placeholder message if no trades */
            #trades-container:empty::after {
                content: "Waiting for trades...";
                display: block;
                text-align: center;
                padding: 20px;
                color: var(--text-secondary);
                font-style: italic;
            }
        `;
        document.head.appendChild(style);
        
        // Set trades as initialized
        tradesInitialized = true;
        
        // Load initial data
        fetchAndRenderTrades();
    }
    
    // Format trade time
    function formatTradeTime(timestamp) {
        const date = new Date(timestamp);
        const hours = date.getHours().toString().padStart(2, '0');
        const minutes = date.getMinutes().toString().padStart(2, '0');
        const seconds = date.getSeconds().toString().padStart(2, '0');
        return `${hours}:${minutes}:${seconds}`;
    }
    
    // Calculate relative size for visual indicator
    function calculateRelativeSize(quantity, maxQuantity) {
        return Math.min(Math.max((quantity / maxQuantity) * 100, 5), 100);
    }
    
    // Fetch and render trades data
    async function fetchAndRenderTrades() {
        console.log("Fetching trades data v2");
        try {
            const response = await fetch('/api/trades');
            const data = await response.json();
            console.log("Trades data received v2:", data.length, "trades");
            
            if (!data || data.length === 0) {
                console.error("No trades data received");
                return;
            }
            
            if (!tradesInitialized) {
                console.error("Trades display not initialized");
                return;
            }
            
            // Update trades data
            const newTrades = data.filter(trade => trade.time > lastTradeTimestamp);
            if (newTrades.length > 0) {
                lastTradeTimestamp = Math.max(...data.map(trade => trade.time));
            }
            
            tradesData = data;
            renderTrades(newTrades.length > 0);
            
        } catch (error) {
            console.error('Error fetching trades:', error);
            // Create sample trades if API fails
            createSampleTrades();
        }
    }
    
    // Create sample trades if API fails
    function createSampleTrades() {
        console.log("Creating sample trades data");
        
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
        
        tradesData = sampleTrades;
        renderTrades(true);
    }
    
    // Render trades with animation for new trades
    function renderTrades(hasNewTrades) {
        console.log("Rendering trades v2, new trades:", hasNewTrades);
        const tradesContainer = document.getElementById('trades-container');
        
        if (!tradesContainer) {
            console.error("Trades container not found");
            return;
        }
        
        // Find maximum quantity for size scaling
        const maxQuantity = Math.max(...tradesData.map(trade => trade.quantity), 0.01);
        
        // Clear container if no new trades or too many trades
        if (!hasNewTrades || tradesData.length > 30) {
            tradesContainer.innerHTML = '';
        }
        
        // Render trades (limited to most recent 30)
        const tradesToRender = hasNewTrades ? tradesData : tradesData.slice(0, 30);
        
        for (let i = 0; i < tradesToRender.length; i++) {
            const trade = tradesToRender[i];
            const isNew = hasNewTrades && i < tradesToRender.length - tradesContainer.childElementCount;
            
            // Skip if this trade is already rendered (unless we're doing a full refresh)
            if (!isNew && hasNewTrades) continue;
            
            const row = document.createElement('div');
            row.className = `trade-row ${trade.isBuyerMaker ? 'sell' : 'buy'} ${isNew ? 'new' : ''}`;
            
            const timeStr = formatTradeTime(trade.time);
            const relativeSize = calculateRelativeSize(trade.quantity, maxQuantity);
            
            row.innerHTML = `
                <div class="trade-price">${trade.price.toFixed(2)}</div>
                <div class="trade-quantity">${trade.quantity.toFixed(6)}</div>
                <div class="trade-time">${timeStr}</div>
                <div class="trade-size-indicator ${trade.isBuyerMaker ? 'sell' : 'buy'}" 
                     style="width: ${relativeSize}%"></div>
            `;
            
            // Add new trades at the top
            if (isNew) {
                tradesContainer.insertBefore(row, tradesContainer.firstChild);
            } else {
                tradesContainer.appendChild(row);
            }
            
            // Remove excess trades
            if (tradesContainer.childElementCount > 30) {
                tradesContainer.removeChild(tradesContainer.lastChild);
            }
        }
        
        // Remove 'new' class after animation completes
        setTimeout(() => {
            const newTrades = tradesContainer.querySelectorAll('.trade-row.new');
            newTrades.forEach(trade => {
                trade.classList.remove('new');
            });
        }, 500);
    }
    
    // Initialize trades display and set up polling
    function init() {
        // Initialize trades with a slight delay to ensure container is ready
        setTimeout(() => {
            initEnhancedTrades();
            
            // Set up polling for trades updates
            setInterval(fetchAndRenderTrades, 5000);
        }, 500);
    }
    
    // Start initialization
    init();
    
    // Expose functions for external use
    window.enhancedTrades = {
        refreshTrades: fetchAndRenderTrades
    };
});
