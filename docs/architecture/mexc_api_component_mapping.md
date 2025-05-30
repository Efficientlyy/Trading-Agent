# MEXC API to System Component Mapping

This document provides a detailed mapping between the modular trading system architecture components and the specific MEXC API endpoints and features they will utilize.

## 1. Data Acquisition Layer

### MEXC Market Data API Client

| System Component | MEXC API Endpoint | Description |
|------------------|------------------|-------------|
| Market Data Fetcher | `GET /api/v3/exchangeInfo` | Retrieves trading rules and symbol information |
| Order Book Fetcher | `GET /api/v3/depth` | Retrieves order book data with configurable depth |
| Trade History Fetcher | `GET /api/v3/trades` | Retrieves recent trades for a symbol |
| Historical Trade Fetcher | `GET /api/v3/historicalTrades` | Retrieves historical trades for analysis |
| Candlestick Data Fetcher | `GET /api/v3/klines` | Retrieves OHLCV data for technical analysis |
| Ticker Fetcher | `GET /api/v3/ticker/24hr` | Retrieves 24-hour price statistics |
| Price Ticker Fetcher | `GET /api/v3/ticker/price` | Retrieves latest price information |
| Order Book Ticker Fetcher | `GET /api/v3/ticker/bookTicker` | Retrieves best bid/ask prices and quantities |

### MEXC WebSocket Client

| System Component | MEXC WebSocket Stream | Description |
|------------------|----------------------|-------------|
| Trade Stream Handler | `spot@public.deals.v3.api@{symbol}` | Processes real-time trade data |
| Kline Stream Handler | `spot@public.kline.v3.api@{symbol}@{interval}` | Processes real-time candlestick updates |
| Depth Stream Handler | `spot@public.increase.depth.v3.api@{symbol}` | Processes order book updates |
| Book Ticker Stream Handler | `spot@public.bookTicker.v3.api@{symbol}` | Processes best bid/ask updates |
| Mini Ticker Stream Handler | `spot@public.miniTicker.v3.api@{symbol}` | Processes simplified ticker updates |

## 2. Data Processing Layer

| System Component | MEXC API Data Source | Processing Function |
|------------------|---------------------|---------------------|
| Market Data Processor | Exchange Info, Ticker Data | Normalizes market metadata and statistics |
| Order Book Processor | Depth Data, Book Ticker | Constructs and maintains local order book |
| Trade Processor | Trade Data | Aggregates and analyzes trade patterns |
| Candlestick Processor | Kline Data | Prepares time-series data for technical analysis |
| Historical Data Manager | Historical Trades, Klines | Stores and indexes historical data |

## 3. Signal Generation Layer

| System Component | MEXC API Data Dependency | Signal Output |
|------------------|--------------------------|---------------|
| Technical Analysis Agent | Kline Data | Technical indicators (RSI, MACD, etc.) |
| Pattern Recognition Agent | Kline Data, Trade Data | Chart patterns and formations |
| Order Book Analysis Agent | Depth Data | Order imbalances, support/resistance levels |
| Volume Analysis Agent | Trade Data, Kline Data | Volume profiles and anomalies |

## 4. Decision Making Layer

| System Component | Input Data | MEXC API Integration |
|------------------|-----------|---------------------|
| Signal Aggregator | All signal outputs | N/A (Internal processing) |
| LLM Decision Engine | Aggregated signals, market context | N/A (Uses external LLM API) |
| Risk Management Module | Account data, position data | Uses account information from MEXC API |
| Decision Output Formatter | LLM decisions | Formats for execution layer |

## 5. Execution Layer

| System Component | MEXC API Endpoint | Function |
|------------------|------------------|----------|
| Trading Executor | `POST /api/v3/order` | Places new orders based on decisions |
| Order Test Module | `POST /api/v3/order/test` | Tests order parameters before execution |
| Batch Order Handler | `POST /api/v3/batchOrders` | Places multiple orders efficiently |
| Order Manager | `GET /api/v3/order` | Queries order status |
| | `DELETE /api/v3/order` | Cancels specific orders |
| | `DELETE /api/v3/openOrders` | Cancels all open orders for a symbol |
| | `GET /api/v3/openOrders` | Retrieves all open orders |
| | `GET /api/v3/allOrders` | Retrieves historical orders |
| Position Manager | `GET /api/v3/account` | Retrieves account balances and positions |
| Trade History Manager | `GET /api/v3/myTrades` | Retrieves account trade history |

## 6. Visualization Layer

| System Component | MEXC API Data Source | Visualization Purpose |
|------------------|---------------------|----------------------|
| Market Data Visualizer | Kline Data, Ticker Data | Price charts and market overview |
| Order Book Visualizer | Depth Data | Order book heatmaps and depth charts |
| Trade Visualizer | Trade Data | Trade execution points on charts |
| Signal Visualizer | Signal outputs + Kline Data | Overlay signals on price charts |
| Position Visualizer | Account Data | Current positions and allocations |
| Performance Dashboard | Trade History, Account Data | P&L metrics and performance analytics |

## WebSocket Integration Details

### Connection Management

```javascript
// Example WebSocket connection management with MEXC
const ws = new WebSocket('ws://wbs-api.mexc.com/ws');

// Connection opened
ws.on('open', () => {
  console.log('Connected to MEXC WebSocket');
  
  // Subscribe to multiple streams
  const subscriptions = [
    'spot@public.deals.v3.api@BTCUSDT',
    'spot@public.kline.v3.api@BTCUSDT@15m',
    'spot@public.increase.depth.v3.api@BTCUSDT'
  ];
  
  subscriptions.forEach(channel => {
    ws.send(JSON.stringify({
      method: 'SUBSCRIPTION',
      params: [channel]
    }));
  });
});

// Implement ping/pong for connection maintenance
setInterval(() => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ method: 'PING' }));
  }
}, 30000);

// Reconnection logic
ws.on('close', () => {
  console.log('WebSocket connection closed');
  setTimeout(() => {
    // Reconnect logic
  }, 5000);
});
```

## Authentication Integration

### API Key Management

The system will securely store and manage MEXC API keys, using them for authenticated requests:

```javascript
// Example authentication integration
function signRequest(endpoint, params, apiKey, apiSecret) {
  const timestamp = Date.now();
  const queryString = Object.entries(params)
    .map(([key, value]) => `${key}=${value}`)
    .join('&');
  
  const signature = createHmacSignature(
    queryString + `&timestamp=${timestamp}`, 
    apiSecret
  );
  
  return {
    url: `https://api.mexc.com${endpoint}?${queryString}&timestamp=${timestamp}&signature=${signature}`,
    headers: {
      'X-MEXC-APIKEY': apiKey
    }
  };
}
```

## Rate Limit Management

The system will implement rate limit tracking and backoff strategies to comply with MEXC API limits:

```javascript
// Example rate limit management
class RateLimitManager {
  constructor() {
    this.weightLimits = {
      minute: { max: 1200, current: 0, resetTime: Date.now() + 60000 }
    };
    this.requestCounts = {
      minute: { max: 60, current: 0, resetTime: Date.now() + 60000 }
    };
  }
  
  async executeRequest(endpoint, weight = 1) {
    // Check if we're approaching limits
    if (this.weightLimits.minute.current + weight >= this.weightLimits.minute.max) {
      const waitTime = this.weightLimits.minute.resetTime - Date.now();
      if (waitTime > 0) {
        await new Promise(resolve => setTimeout(resolve, waitTime));
        this.resetLimits();
      }
    }
    
    // Update counters
    this.weightLimits.minute.current += weight;
    this.requestCounts.minute.current += 1;
    
    // Execute the actual request
    // ...
    
    return result;
  }
  
  resetLimits() {
    const now = Date.now();
    if (now >= this.weightLimits.minute.resetTime) {
      this.weightLimits.minute.current = 0;
      this.weightLimits.minute.resetTime = now + 60000;
      this.requestCounts.minute.current = 0;
      this.requestCounts.minute.resetTime = now + 60000;
    }
  }
}
```

## Multi-Pair Trading Implementation

The system will handle multiple trading pairs by creating separate instances of key components:

```javascript
// Example multi-pair management
class MultiPairManager {
  constructor(symbols) {
    this.symbols = symbols;
    this.dataProcessors = {};
    this.signalGenerators = {};
    this.orderManagers = {};
    
    // Initialize components for each symbol
    symbols.forEach(symbol => {
      this.dataProcessors[symbol] = new DataProcessor(symbol);
      this.signalGenerators[symbol] = new SignalGenerator(symbol);
      this.orderManagers[symbol] = new OrderManager(symbol);
    });
  }
  
  // Subscribe to all websocket streams
  subscribeToAllStreams(ws) {
    this.symbols.forEach(symbol => {
      const streams = [
        `spot@public.deals.v3.api@${symbol}`,
        `spot@public.kline.v3.api@${symbol}@15m`,
        `spot@public.increase.depth.v3.api@${symbol}`
      ];
      
      streams.forEach(stream => {
        ws.send(JSON.stringify({
          method: 'SUBSCRIPTION',
          params: [stream]
        }));
      });
    });
  }
  
  // Process incoming websocket messages
  processMessage(message) {
    // Extract symbol from channel
    const channelParts = message.c.split('@');
    const symbolPart = channelParts[channelParts.length - 1];
    const symbol = symbolPart.split('_')[0];
    
    if (this.dataProcessors[symbol]) {
      this.dataProcessors[symbol].processUpdate(message);
      this.signalGenerators[symbol].checkForSignals();
    }
  }
}
```

## LLM Integration with MEXC Data

The LLM decision engine will receive formatted market data and signals from MEXC:

```javascript
// Example LLM integration with MEXC data
async function prepareLLMInput(symbol, technicalSignals, orderBookData, recentTrades) {
  // Format market data for LLM consumption
  const marketContext = {
    symbol,
    currentPrice: orderBookData.asks[0][0],
    priceChange24h: ticker24hr.priceChangePercent,
    volume24h: ticker24hr.volume,
    technicalIndicators: {
      rsi: technicalSignals.rsi,
      macd: technicalSignals.macd,
      bollingerBands: technicalSignals.bollingerBands
    },
    orderBookImbalance: calculateImbalance(orderBookData),
    recentTradesSummary: summarizeTrades(recentTrades)
  };
  
  // Create prompt for LLM
  const prompt = `
    You are a cryptocurrency trading assistant analyzing ${symbol}.
    
    Current market data:
    - Price: ${marketContext.currentPrice} USD
    - 24h Change: ${marketContext.priceChange24h}%
    - 24h Volume: ${marketContext.volume24h} USD
    
    Technical indicators:
    - RSI (14): ${marketContext.technicalIndicators.rsi}
    - MACD: ${marketContext.technicalIndicators.macd.line} (Signal: ${marketContext.technicalIndicators.macd.signal})
    - Bollinger Bands: Current price is ${marketContext.technicalIndicators.bollingerBands.position} the bands
    
    Order book shows a ${marketContext.orderBookImbalance > 0 ? 'buying' : 'selling'} imbalance of ${Math.abs(marketContext.orderBookImbalance)}%.
    
    Recent trades summary: ${marketContext.recentTradesSummary}
    
    Based on this information, should I BUY, SELL, or HOLD ${symbol}? Provide your reasoning and a confidence score from 0-100%.
  `;
  
  return prompt;
}
```

## Conclusion

This mapping provides a comprehensive guide for integrating each component of the modular trading system with the appropriate MEXC API endpoints and features. By following this mapping, developers can ensure that all parts of the system are properly connected to the exchange's functionality, enabling seamless data flow from market data acquisition through signal generation, decision making, and trade execution.

The architecture is designed to be flexible and extensible, allowing for the addition of new components and features as the system evolves. The use of standardized interfaces between components ensures that individual modules can be developed, tested, and deployed independently, facilitating a modular development approach.
