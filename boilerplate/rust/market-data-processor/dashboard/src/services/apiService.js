// API service for interacting with the backend
import axios from 'axios';

const API_BASE_URL = '/api/v1';

// WebSocket connection
let ws = null;
let tickerSubscribers = [];
let orderBookSubscribers = [];
let tradesSubscribers = [];

// Create WebSocket connection
const createWebSocketConnection = () => {
  if (ws) return;

  const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log('WebSocket connection established');
    // Subscribe to BTC/USDC data
    ws.send(JSON.stringify({ type: 'subscribe', channel: 'ticker', symbol: 'BTCUSDC' }));
    ws.send(JSON.stringify({ type: 'subscribe', channel: 'orderbook', symbol: 'BTCUSDC' }));
    ws.send(JSON.stringify({ type: 'subscribe', channel: 'trades', symbol: 'BTCUSDC' }));
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      
      if (data.type === 'ticker') {
        tickerSubscribers.forEach(callback => callback(data.data));
      } else if (data.type === 'orderbook') {
        orderBookSubscribers.forEach(callback => callback(data.data));
      } else if (data.type === 'trades') {
        tradesSubscribers.forEach(callback => callback(data.data));
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  };

  ws.onclose = () => {
    console.log('WebSocket connection closed');
    ws = null;
    // Attempt to reconnect after a delay
    setTimeout(createWebSocketConnection, 5000);
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    ws.close();
  };
};

// Subscribe to ticker updates
export const subscribeToTickerUpdates = (symbol, callback) => {
  if (!ws) {
    createWebSocketConnection();
  }
  
  tickerSubscribers.push(callback);
  
  // Return unsubscribe function
  return () => {
    tickerSubscribers = tickerSubscribers.filter(cb => cb !== callback);
  };
};

// Subscribe to order book updates
export const subscribeToOrderBookUpdates = (symbol, callback) => {
  if (!ws) {
    createWebSocketConnection();
  }
  
  orderBookSubscribers.push(callback);
  
  // Return unsubscribe function
  return () => {
    orderBookSubscribers = orderBookSubscribers.filter(cb => cb !== callback);
  };
};

// Subscribe to trades updates
export const subscribeToTradesUpdates = (symbol, callback) => {
  if (!ws) {
    createWebSocketConnection();
  }
  
  tradesSubscribers.push(callback);
  
  // Return unsubscribe function
  return () => {
    tradesSubscribers = tradesSubscribers.filter(cb => cb !== callback);
  };
};

// Fetch current ticker data
export const fetchTicker = async (symbol = 'BTCUSDC') => {
  try {
    const response = await axios.get(`${API_BASE_URL}/ticker?symbol=${symbol}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching ticker data:', error);
    throw error;
  }
};

// Fetch order book data
export const fetchOrderBook = async (symbol = 'BTCUSDC') => {
  try {
    const response = await axios.get(`${API_BASE_URL}/orderbook?symbol=${symbol}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching order book data:', error);
    throw error;
  }
};

// Fetch recent trades
export const fetchTrades = async (symbol = 'BTCUSDC') => {
  try {
    const response = await axios.get(`${API_BASE_URL}/trades?symbol=${symbol}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching trades data:', error);
    throw error;
  }
};

// Fetch account information (paper trading)
export const fetchAccount = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/account`);
    return response.data;
  } catch (error) {
    console.error('Error fetching account data:', error);
    throw error;
  }
};

// Place a paper trade order
export const placePaperOrder = async (orderData) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/order`, orderData);
    return response.data;
  } catch (error) {
    console.error('Error placing order:', error);
    throw error;
  }
};

// Cancel a paper trade order
export const cancelPaperOrder = async (orderId) => {
  try {
    const response = await axios.delete(`${API_BASE_URL}/order/${orderId}`);
    return response.data;
  } catch (error) {
    console.error('Error canceling order:', error);
    throw error;
  }
};

// Fetch historical candlestick data
export const fetchHistoricalData = async (symbol = 'BTCUSDC', interval = '1h') => {
  try {
    const response = await axios.get(`${API_BASE_URL}/klines?symbol=${symbol}&interval=${interval}`);
    
    // Transform data for lightweight-charts format
    return response.data.map(candle => ({
      time: candle.time / 1000, // Convert to seconds for lightweight-charts
      open: parseFloat(candle.open),
      high: parseFloat(candle.high),
      low: parseFloat(candle.low),
      close: parseFloat(candle.close),
    }));
  } catch (error) {
    console.error('Error fetching historical data:', error);
    
    // For development/testing, return mock data if API fails
    return generateMockHistoricalData();
  }
};

// Generate mock historical data for development/testing
const generateMockHistoricalData = () => {
  const data = [];
  const now = Math.floor(Date.now() / 1000) * 1000;
  let price = 35000 + Math.random() * 1000;
  
  for (let i = 0; i < 100; i++) {
    const time = now - (99 - i) * 3600 * 1000; // hourly candles
    const open = price;
    const close = open * (0.995 + Math.random() * 0.01); // +/- 0.5%
    price = close;
    const high = Math.max(open, close) * (1 + Math.random() * 0.005);
    const low = Math.min(open, close) * (1 - Math.random() * 0.005);
    
    data.push({
      time: time / 1000, // Convert to seconds for lightweight-charts
      open,
      high,
      low,
      close,
    });
  }
  
  return data;
};

// Initialize WebSocket connection
createWebSocketConnection();

export default {
  fetchTicker,
  fetchOrderBook,
  fetchTrades,
  fetchAccount,
  fetchHistoricalData,
  placePaperOrder,
  cancelPaperOrder,
  subscribeToTickerUpdates,
  subscribeToOrderBookUpdates,
  subscribeToTradesUpdates,
};
