/**
 * Trading functionality for the dashboard
 */

// Store current ticker data
let currentTicker = null;

// Update ticker display
function updateTicker(symbol) {
    fetchAPI(`/api/ticker/${symbol}`)
        .then(data => {
            if (!data) return;
            
            // Store current ticker data
            currentTicker = data;
            
            // Update price display
            const priceElement = document.getElementById('current-price');
            const changeElement = document.getElementById('price-change');
            const chartSymbolElement = document.getElementById('chart-symbol');
            
            if (priceElement && changeElement && chartSymbolElement) {
                // Update symbol
                chartSymbolElement.textContent = formatAssetName(symbol);
                
                // Update price
                priceElement.textContent = formatPrice(data.price, symbol);
                
                // Calculate price change percentage
                const openPrice = parseFloat(data.open);
                const currentPrice = parseFloat(data.price);
                const changePercent = ((currentPrice - openPrice) / openPrice) * 100;
                
                // Update change display
                changeElement.textContent = formatPercent(changePercent);
                
                // Set color class
                if (changePercent > 0) {
                    changeElement.className = 'ms-1 price-up';
                } else if (changePercent < 0) {
                    changeElement.className = 'ms-1 price-down';
                } else {
                    changeElement.className = 'ms-1';
                }
                
                // Update title
                document.title = `${formatPrice(data.price, symbol)} ${formatAssetName(symbol)} | MEXC Trading`;
                
                // Update limit order price input
                const limitPriceInput = document.getElementById('limit-price');
                if (limitPriceInput) {
                    // Only update if not focused
                    if (document.activeElement !== limitPriceInput) {
                        limitPriceInput.value = formatPrice(data.price, symbol);
                    }
                }
            }
        });
}

// Update balances display
function updateBalances() {
    fetchAPI('/api/balances')
        .then(data => {
            if (!data) return;
            
            // Update balance displays
            const usdtBalanceElement = document.getElementById('usdt-balance');
            const btcBalanceElement = document.getElementById('btc-balance');
            const portfolioValueElement = document.getElementById('portfolio-value');
            const dailyPnlElement = document.getElementById('daily-pnl');
            
            if (usdtBalanceElement && data.USDT !== undefined) {
                usdtBalanceElement.textContent = formatNumber(data.USDT);
            }
            
            if (btcBalanceElement && data.BTC !== undefined) {
                btcBalanceElement.textContent = formatNumber(data.BTC, 8);
            }
            
            // Calculate portfolio value
            let portfolioValue = 0;
            
            // Add USDT value directly
            if (data.USDT !== undefined) {
                portfolioValue += data.USDT;
            }
            
            // Calculate BTC value in USDT
            if (data.BTC !== undefined) {
                // Get BTC price
                fetchAPI('/api/ticker/BTCUSDT')
                    .then(ticker => {
                        if (ticker) {
                            const btcValue = data.BTC * ticker.price;
                            portfolioValue += btcValue;
                            
                            // Update portfolio value display
                            if (portfolioValueElement) {
                                portfolioValueElement.textContent = formatCurrency(portfolioValue);
                            }
                            
                            // Simulate daily PnL for now
                            if (dailyPnlElement) {
                                // Random PnL between -5% and +5%
                                const pnlPercent = (Math.random() * 10) - 5;
                                const pnlValue = portfolioValue * (pnlPercent / 100);
                                
                                dailyPnlElement.textContent = `${formatCurrency(pnlValue)} (${formatPercent(pnlPercent)})`;
                                
                                if (pnlPercent > 0) {
                                    dailyPnlElement.className = 'metric-value positive';
                                } else if (pnlPercent < 0) {
                                    dailyPnlElement.className = 'metric-value negative';
                                } else {
                                    dailyPnlElement.className = 'metric-value';
                                }
                            }
                        }
                    });
            } else {
                // Just use USDT value if no BTC
                if (portfolioValueElement) {
                    portfolioValueElement.textContent = formatCurrency(portfolioValue);
                }
            }
        });
}

// Update open positions display
function updatePositions() {
    fetchAPI('/api/positions')
        .then(positions => {
            if (!positions) return;
            
            const positionsContainer = document.getElementById('open-positions');
            if (!positionsContainer) return;
            
            // Clear container
            positionsContainer.innerHTML = '';
            
            // If no positions, show message
            if (Object.keys(positions).length === 0) {
                const row = document.createElement('tr');
                const cell = document.createElement('td');
                cell.colSpan = 6;
                cell.textContent = 'No open positions';
                cell.className = 'text-center text-muted';
                row.appendChild(cell);
                positionsContainer.appendChild(row);
                return;
            }
            
            // Process each position
            for (const [asset, position] of Object.entries(positions)) {
                // Skip positions with zero quantity
                if (position.quantity <= 0) continue;
                
                // Create symbol
                const symbol = asset + 'USDT';
                
                // Get current price for the asset
                fetchAPI(`/api/ticker/${symbol}`)
                    .then(ticker => {
                        if (!ticker) return;
                        
                        const currentPrice = parseFloat(ticker.price);
                        const entryPrice = parseFloat(position.avg_price);
                        const quantity = parseFloat(position.quantity);
                        
                        // Calculate P&L
                        const positionValue = quantity * currentPrice;
                        const costBasis = quantity * entryPrice;
                        const pnl = positionValue - costBasis;
                        const pnlPercent = (pnl / costBasis) * 100;
                        
                        // Create row
                        const row = document.createElement('tr');
                        
                        // Symbol cell
                        const symbolCell = document.createElement('td');
                        symbolCell.textContent = formatAssetName(symbol);
                        row.appendChild(symbolCell);
                        
                        // Size cell
                        const sizeCell = document.createElement('td');
                        sizeCell.textContent = formatNumber(quantity, 6);
                        row.appendChild(sizeCell);
                        
                        // Entry price cell
                        const entryCell = document.createElement('td');
                        entryCell.textContent = formatPrice(entryPrice);
                        row.appendChild(entryCell);
                        
                        // Current price cell
                        const priceCell = document.createElement('td');
                        priceCell.textContent = formatPrice(currentPrice);
                        row.appendChild(priceCell);
                        
                        // P&L cell
                        const pnlCell = document.createElement('td');
                        pnlCell.textContent = `${formatCurrency(pnl)} (${formatPercent(pnlPercent)})`;
                        pnlCell.className = pnl >= 0 ? 'text-success' : 'text-danger';
                        row.appendChild(pnlCell);
                        
                        // Action cell
                        const actionCell = document.createElement('td');
                        const closeButton = document.createElement('button');
                        closeButton.className = 'btn btn-sm btn-danger';
                        closeButton.textContent = 'Close';
                        closeButton.addEventListener('click', () => placeMarketOrder(symbol, 'SELL', quantity));
                        actionCell.appendChild(closeButton);
                        row.appendChild(actionCell);
                        
                        positionsContainer.appendChild(row);
                    });
            }
        });
}

// Update portfolio assets display
function updatePortfolioAssets() {
    fetchAPI('/api/balances')
        .then(balances => {
            if (!balances) return;
            
            const assetsContainer = document.getElementById('portfolio-assets');
            if (!assetsContainer) return;
            
            // Clear container
            assetsContainer.innerHTML = '';
            
            // Get all tickers for value calculation
            fetchAPI('/api/tickers')
                .then(tickers => {
                    if (!tickers) return;
                    
                    let totalValue = 0;
                    const assets = [];
                    
                    // Process USDT first
                    if (balances.USDT) {
                        const usdtValue = parseFloat(balances.USDT);
                        totalValue += usdtValue;
                        
                        assets.push({
                            asset: 'USDT',
                            balance: usdtValue,
                            value: usdtValue,
                            allocation: 0  // Will calculate after we know total
                        });
                    }
                    
                    // Process other assets
                    for (const [asset, balance] of Object.entries(balances)) {
                        // Skip USDT (already processed) and zero balances
                        if (asset === 'USDT' || balance <= 0) continue;
                        
                        const symbol = asset + 'USDT';
                        
                        // Get ticker for this asset
                        if (tickers[symbol]) {
                            const assetValue = balance * tickers[symbol].price;
                            totalValue += assetValue;
                            
                            assets.push({
                                asset,
                                balance,
                                value: assetValue,
                                allocation: 0  // Will calculate after we know total
                            });
                        }
                    }
                    
                    // Calculate allocations
                    for (const asset of assets) {
                        asset.allocation = (asset.value / totalValue) * 100;
                    }
                    
                    // Sort by value (descending)
                    assets.sort((a, b) => b.value - a.value);
                    
                    // Render assets
                    for (const asset of assets) {
                        const row = document.createElement('tr');
                        
                        // Asset cell
                        const assetCell = document.createElement('td');
                        assetCell.textContent = asset.asset;
                        row.appendChild(assetCell);
                        
                        // Balance cell
                        const balanceCell = document.createElement('td');
                        balanceCell.textContent = formatNumber(asset.balance, asset.asset === 'USDT' ? 2 : 8);
                        row.appendChild(balanceCell);
                        
                        // Value cell
                        const valueCell = document.createElement('td');
                        valueCell.textContent = formatCurrency(asset.value);
                        row.appendChild(valueCell);
                        
                        // Allocation cell
                        const allocationCell = document.createElement('td');
                        allocationCell.textContent = formatPercent(asset.allocation);
                        row.appendChild(allocationCell);
                        
                        assetsContainer.appendChild(row);
                    }
                    
                    // Initialize portfolio chart
                    updatePortfolioChart(assets);
                });
        });
}

// Update trade history display
function updateTradeHistory() {
    fetchAPI('/api/order_history')
        .then(data => {
            if (!data) return;
            
            const historyContainer = document.getElementById('trade-history');
            if (!historyContainer) return;
            
            // Clear container
            historyContainer.innerHTML = '';
            
            // If no trades, show message
            if (data.length === 0) {
                const row = document.createElement('tr');
                const cell = document.createElement('td');
                cell.colSpan = 5;
                cell.textContent = 'No trade history';
                cell.className = 'text-center text-muted';
                row.appendChild(cell);
                historyContainer.appendChild(row);
                return;
            }
            
            // Sort by time (newest first)
            const sortedTrades = [...data].sort((a, b) => b.time - a.time);
            
            // Process each trade
            for (const trade of sortedTrades) {
                const row = document.createElement('tr');
                
                // Symbol cell
                const symbolCell = document.createElement('td');
                symbolCell.textContent = formatAssetName(trade.symbol);
                row.appendChild(symbolCell);
                
                // Side cell
                const sideCell = document.createElement('td');
                sideCell.textContent = trade.side;
                sideCell.className = trade.side === 'BUY' ? 'text-success' : 'text-danger';
                row.appendChild(sideCell);
                
                // Price cell
                const priceCell = document.createElement('td');
                priceCell.textContent = formatPrice(trade.price);
                row.appendChild(priceCell);
                
                // Quantity cell
                const quantityCell = document.createElement('td');
                quantityCell.textContent = formatNumber(trade.quantity, 6);
                row.appendChild(quantityCell);
                
                // Time cell
                const timeCell = document.createElement('td');
                timeCell.textContent = formatDateTime(trade.time);
                row.appendChild(timeCell);
                
                historyContainer.appendChild(row);
            }
        });
}

// Update portfolio chart
function updatePortfolioChart(assets) {
    // This is a placeholder for a real implementation
    // Would use the assets array to create a pie chart
    console.log('Portfolio assets for chart:', assets);
}

// Place a market order
function placeMarketOrder(symbol, side, quantity) {
    if (!symbol || !side || !quantity) {
        alert('Please fill in all fields');
        return;
    }
    
    const order = {
        symbol,
        side,
        type: 'MARKET',
        quantity: parseFloat(quantity)
    };
    
    postAPI('/api/place_order', order)
        .then(response => {
            if (response && response.error) {
                alert(`Order Error: ${response.error}`);
            } else if (response) {
                // Update displays
                updateBalances();
                updatePositions();
                updateTradeHistory();
                updatePortfolioAssets();
                
                // Show success message
                alert(`${side} order executed successfully`);
            }
        });
}

// Place a limit order
function placeLimitOrder(symbol, side, price, quantity) {
    if (!symbol || !side || !price || !quantity) {
        alert('Please fill in all fields');
        return;
    }
    
    const order = {
        symbol,
        side,
        type: 'LIMIT',
        price: parseFloat(price),
        quantity: parseFloat(quantity)
    };
    
    postAPI('/api/place_order', order)
        .then(response => {
            if (response && response.error) {
                alert(`Order Error: ${response.error}`);
            } else if (response) {
                // Update displays
                updateBalances();
                updatePositions();
                updateTradeHistory();
                updatePortfolioAssets();
                
                // Show success message
                alert(`${side} limit order placed successfully`);
            }
        });
}

// Setup trading interface event listeners
function setupTradingInterface() {
    // Market buy button
    const marketBuyBtn = document.getElementById('market-buy-btn');
    if (marketBuyBtn) {
        marketBuyBtn.addEventListener('click', () => {
            const symbol = document.getElementById('trading-pair-select').value;
            const quantity = document.getElementById('market-quantity').value;
            placeMarketOrder(symbol, 'BUY', quantity);
        });
    }
    
    // Market sell button
    const marketSellBtn = document.getElementById('market-sell-btn');
    if (marketSellBtn) {
        marketSellBtn.addEventListener('click', () => {
            const symbol = document.getElementById('trading-pair-select').value;
            const quantity = document.getElementById('market-quantity').value;
            placeMarketOrder(symbol, 'SELL', quantity);
        });
    }
    
    // Limit buy button
    const limitBuyBtn = document.getElementById('limit-buy-btn');
    if (limitBuyBtn) {
        limitBuyBtn.addEventListener('click', () => {
            const symbol = document.getElementById('trading-pair-select').value;
            const price = document.getElementById('limit-price').value;
            const quantity = document.getElementById('limit-quantity').value;
            placeLimitOrder(symbol, 'BUY', price, quantity);
        });
    }
    
    // Limit sell button
    const limitSellBtn = document.getElementById('limit-sell-btn');
    if (limitSellBtn) {
        limitSellBtn.addEventListener('click', () => {
            const symbol = document.getElementById('trading-pair-select').value;
            const price = document.getElementById('limit-price').value;
            const quantity = document.getElementById('limit-quantity').value;
            placeLimitOrder(symbol, 'SELL', price, quantity);
        });
    }
}
