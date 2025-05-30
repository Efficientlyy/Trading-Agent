import axios from 'axios';

// Configure axios
const api = axios.create({
  baseURL: '/api',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Error handling interceptor
api.interceptors.response.use(
  response => response.data,
  error => {
    console.error('API Error:', error.response || error);
    return Promise.reject(error);
  }
);

// Account Data API
export const fetchAccountData = async () => {
  // For testing, return mock data
  if (process.env.NODE_ENV === 'development') {
    return getMockAccountData();
  }
  
  return api.get('/account');
};

// Market Data API
export const fetchMarketData = async () => {
  // For testing, return mock data
  if (process.env.NODE_ENV === 'development') {
    return getMockMarketData();
  }
  
  return api.get('/market/data');
};

// Recent Trades API
export const fetchRecentTrades = async () => {
  // For testing, return mock data
  if (process.env.NODE_ENV === 'development') {
    return getMockRecentTrades();
  }
  
  return api.get('/trades/recent');
};

// Order History API
export const fetchOrderHistory = async (params) => {
  // For testing, return mock data
  if (process.env.NODE_ENV === 'development') {
    return getMockOrderHistory();
  }
  
  return api.get('/orders/history', { params });
};

// Place Order API
export const placeOrder = async (orderData) => {
  // For testing, just log and return mock response
  if (process.env.NODE_ENV === 'development') {
    console.log('Placing order:', orderData);
    return { success: true, orderId: `mock-${Date.now()}` };
  }
  
  return api.post('/orders', orderData);
};

// Cancel Order API
export const cancelOrder = async (orderId) => {
  // For testing, just log and return mock response
  if (process.env.NODE_ENV === 'development') {
    console.log('Cancelling order:', orderId);
    return { success: true };
  }
  
  return api.delete(`/orders/${orderId}`);
};

// Fetch Paper Trading Settings
export const fetchPaperTradingSettings = async () => {
  // For testing, return mock data
  if (process.env.NODE_ENV === 'development') {
    return getMockPaperTradingSettings();
  }
  
  return api.get('/settings/paper-trading');
};

// Update Paper Trading Settings
export const updatePaperTradingSettings = async (settings) => {
  // For testing, just log and return mock response
  if (process.env.NODE_ENV === 'development') {
    console.log('Updating paper trading settings:', settings);
    return { success: true };
  }
  
  return api.put('/settings/paper-trading', settings);
};

// Reset Paper Trading Account
export const resetPaperTradingAccount = async () => {
  // For testing, just log and return mock response
  if (process.env.NODE_ENV === 'development') {
    console.log('Resetting paper trading account');
    return { success: true };
  }
  
  return api.post('/account/reset');
};

// Mock data for development
const getMockAccountData = () => {
  return {
    balances: [
      { asset: 'USDT', free: 5432.78, locked: 1000.00, usdValue: 6432.78 },
      { asset: 'BTC', free: 0.25, locked: 0.05, usdValue: 8750.00 },
      { asset: 'ETH', free: 2.5, locked: 0, usdValue: 5000.00 }
    ],
    totalValue: 20182.78,
    pnl: 182.78,
    pnlPercentage: 0.91,
    performance: {
      profitLoss: 0.91,
      winRate: 62.5,
      maxDrawdown: 5.3,
      sharpeRatio: 1.2,
      totalTrades: 48,
      successfulTrades: 30,
      averageProfitPerTrade: 0.4,
      averageTradeTime: 38
    },
    activeOrders: [
      {
        id: 'ord-001',
        symbol: 'BTCUSDT',
        side: 'BUY',
        type: 'LIMIT',
        price: 35000.00,
        quantity: 0.05,
        timestamp: Date.now() - 3600000
      },
      {
        id: 'ord-002',
        symbol: 'ETHUSDT',
        side: 'SELL',
        type: 'STOP_LIMIT',
        price: 1800.00,
        quantity: 0.5,
        timestamp: Date.now() - 1800000
      }
    ]
  };
};

const getMockMarketData = () => {
  // Generate some price data points
  const now = Date.now();
  const hour = 3600000;
  const prices = [];
  
  // BTC price data
  let btcPrice = 35000;
  for (let i = 24; i >= 0; i--) {
    btcPrice = btcPrice * (1 + (Math.random() * 0.02 - 0.01));
    prices.push({
      symbol: 'BTCUSDT',
      timestamp: now - (i * hour / 24),
      price: btcPrice,
      volume: Math.random() * 10 + 5
    });
  }
  
  // ETH price data
  let ethPrice = 2000;
  for (let i = 24; i >= 0; i--) {
    ethPrice = ethPrice * (1 + (Math.random() * 0.03 - 0.015));
    prices.push({
      symbol: 'ETHUSDT',
      timestamp: now - (i * hour / 24),
      price: ethPrice,
      volume: Math.random() * 20 + 10
    });
  }
  
  return {
    prices,
    lastUpdated: now,
    availableSymbols: ['BTCUSDT', 'ETHUSDT']
  };
};

const getMockRecentTrades = () => {
  const now = Date.now();
  const minute = 60000;
  
  return [
    {
      id: 'trade-001',
      symbol: 'BTCUSDT',
      side: 'BUY',
      price: 34950.25,
      quantity: 0.05,
      timestamp: now - (10 * minute)
    },
    {
      id: 'trade-002',
      symbol: 'ETHUSDT',
      side: 'SELL',
      price: 1950.75,
      quantity: 1.2,
      timestamp: now - (25 * minute)
    },
    {
      id: 'trade-003',
      symbol: 'BTCUSDT',
      side: 'BUY',
      price: 34800.50,
      quantity: 0.08,
      timestamp: now - (60 * minute)
    },
    {
      id: 'trade-004',
      symbol: 'ETHUSDT',
      side: 'BUY',
      price: 1925.30,
      quantity: 0.5,
      timestamp: now - (120 * minute)
    },
    {
      id: 'trade-005',
      symbol: 'BTCUSDT',
      side: 'SELL',
      price: 35100.00,
      quantity: 0.03,
      timestamp: now - (180 * minute)
    }
  ];
};

const getMockOrderHistory = () => {
  const now = Date.now();
  const hour = 3600000;
  
  return [
    {
      id: 'ord-101',
      symbol: 'BTCUSDT',
      side: 'BUY',
      type: 'LIMIT',
      price: 34500.00,
      quantity: 0.1,
      status: 'FILLED',
      timestamp: now - (2 * hour),
      fillPrice: 34500.00,
      fillQuantity: 0.1,
      fee: 3.45
    },
    {
      id: 'ord-102',
      symbol: 'ETHUSDT',
      side: 'SELL',
      type: 'MARKET',
      price: null,
      quantity: 1.0,
      status: 'FILLED',
      timestamp: now - (5 * hour),
      fillPrice: 1920.25,
      fillQuantity: 1.0,
      fee: 1.92
    },
    {
      id: 'ord-103',
      symbol: 'BTCUSDT',
      side: 'BUY',
      type: 'STOP_LIMIT',
      price: 33500.00,
      quantity: 0.15,
      status: 'CANCELED',
      timestamp: now - (12 * hour),
      fillPrice: null,
      fillQuantity: 0,
      fee: 0
    },
    {
      id: 'ord-104',
      symbol: 'ETHUSDT',
      side: 'BUY',
      type: 'LIMIT',
      price: 1850.00,
      quantity: 2.0,
      status: 'FILLED',
      timestamp: now - (24 * hour),
      fillPrice: 1850.00,
      fillQuantity: 2.0,
      fee: 3.70
    },
    {
      id: 'ord-105',
      symbol: 'BTCUSDT',
      side: 'SELL',
      type: 'MARKET',
      price: null,
      quantity: 0.2,
      status: 'FILLED',
      timestamp: now - (36 * hour),
      fillPrice: 34200.50,
      fillQuantity: 0.2,
      fee: 6.84
    }
  ];
};

const getMockPaperTradingSettings = () => {
  return {
    initialBalances: {
      USDT: 10000,
      BTC: 0.5,
      ETH: 5
    },
    tradingPairs: ['BTCUSDT', 'ETHUSDT'],
    maxPositionSize: 1.0, // 100% of available balance
    defaultOrderSize: 0.1, // 10% of available balance
    maxDrawdownPercent: 10,
    slippageModel: 'REALISTIC',
    latencyModel: 'NORMAL',
    tradingFees: 0.001 // 0.1%
  };
};
