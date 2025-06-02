# MEXC Trading System Implementation Instructions

## Current Status Assessment

I've reviewed the current implementation and identified several critical gaps:

1. The dashboard is showing BTC/USDC instead of the required BTC/USDC trading pair
2. The implementation is only a metrics simulation without real trading functionality
3. Core components are missing:
   - No price charts (only a placeholder)
   - No order book visualization
   - No trade history
   - No paper trading interface

## Required Implementation Steps

Please follow these precise steps to implement the full BTC/USDC trading system:

### 1. Pull Latest Repository Changes

```bash
cd Trading-Agent
git pull origin master
```

This will bring in all the latest changes, including:
- Complete MEXC API integration for BTC/USDC
- Real-time WebSocket implementation
- Full dashboard components
- Paper trading functionality
- Updated documentation

### 2. Review the ONBOARDING.md File

The ONBOARDING.md file contains comprehensive documentation on:
- System architecture
- Component relationships
- Development workflow
- Testing procedures

### 3. Set Up Environment

Create a `.env` file in the root directory:

```bash
cat > .env << EOL
MEXC_API_KEY=your_api_key_here
MEXC_API_SECRET=your_api_secret_here
EOL
```

Note: The system will work with public data even without API keys.

### 4. Build and Run the Full System

```bash
docker-compose up -d
```

This will start all components:
- Market Data Processor with real MEXC API integration
- Dashboard with full trading functionality
- Prometheus and Grafana for monitoring

### 5. Implement BTC/USDC Trading Pair

Ensure the system is configured for BTC/USDC:

1. Update the default trading pair in `boilerplate/rust/market-data-processor/src/utils/config.rs`:
   ```rust
   pub const DEFAULT_SYMBOL: &str = "BTCUSDC";
   ```

2. Modify the frontend configuration in `boilerplate/rust/market-data-processor/dashboard/src/services/apiService.js`:
   ```javascript
   const defaultPair = 'BTCUSDC';
   ```

### 6. Implement Full Dashboard Components

Ensure these components are properly implemented:

1. **Price Chart**: `boilerplate/rust/market-data-processor/dashboard/src/components/PriceChart.js`
   - Real-time candlestick chart
   - Technical indicators
   - Time interval selection

2. **Order Book**: `boilerplate/rust/market-data-processor/dashboard/src/components/OrderBook.js`
   - Bid/ask visualization
   - Market depth display
   - Real-time updates

3. **Trade History**: `boilerplate/rust/market-data-processor/dashboard/src/components/TradeHistory.js`
   - Recent trades table
   - Real-time updates
   - Color-coded buy/sell indicators

4. **Paper Trading**: `boilerplate/rust/market-data-processor/dashboard/src/components/PaperTrading.js`
   - Order placement interface
   - Account balance display
   - Position management

### 7. Connect to Real-Time Data

Implement WebSocket connections for real-time data:

1. WebSocket service: `boilerplate/rust/market-data-processor/dashboard/src/services/websocketService.js`
2. Market data hooks: `boilerplate/rust/market-data-processor/dashboard/src/hooks/useMarketData.js`
3. Connection status component: `boilerplate/rust/market-data-processor/dashboard/src/components/ConnectionStatus.js`

### 8. Verify Implementation

After implementation, verify:

1. Dashboard shows real-time BTC/USDC price chart
2. Order book displays market depth for BTC/USDC
3. Trade history shows recent BTC/USDC trades
4. Paper trading interface allows simulated trading
5. All components update in real-time

## Expected Result

The final implementation should match the design in the repository, with:

1. A professional dark-themed UI
2. Real-time BTC/USDC data from MEXC
3. Interactive trading components
4. Paper trading functionality
5. System monitoring integration

## Reference Files

- System architecture: `/architecture/`
- Implementation details: `/docs/`
- Test results: `/test_results/`

Please ensure all components are implemented as specified, with a focus on BTC/USDC trading pair and real-time functionality.
