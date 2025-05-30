// Custom hook for fetching and managing market data
import { useState, useEffect } from 'react';
import { fetchTicker, fetchOrderBook, fetchTrades } from '../services/apiService';
import { useWebSocket } from '../services/websocketService';

// Hook for market data with both REST and WebSocket sources
export const useMarketData = (symbol = 'BTCUSDC') => {
  // State for market data
  const [ticker, setTicker] = useState(null);
  const [orderBook, setOrderBook] = useState(null);
  const [trades, setTrades] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Get WebSocket data
  const wsData = useWebSocket();
  
  // Initial data load via REST API
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Fetch initial data in parallel
        const [tickerData, orderBookData, tradesData] = await Promise.all([
          fetchTicker(symbol),
          fetchOrderBook(symbol),
          fetchTrades(symbol)
        ]);
        
        // Update state with fetched data
        setTicker(tickerData);
        setOrderBook(orderBookData);
        setTrades(tradesData);
        
        setLoading(false);
      } catch (err) {
        console.error('Error loading market data:', err);
        setError('Failed to load market data. Please try again later.');
        setLoading(false);
      }
    };
    
    loadInitialData();
    
    // Refresh data every 30 seconds as fallback if WebSocket fails
    const intervalId = setInterval(loadInitialData, 30000);
    
    return () => {
      clearInterval(intervalId);
    };
  }, [symbol]);
  
  // Update state with WebSocket data when available
  useEffect(() => {
    if (wsData.ticker) {
      setTicker(wsData.ticker);
    }
    
    if (wsData.orderBook) {
      setOrderBook(wsData.orderBook);
    }
    
    if (wsData.trades && wsData.trades.length > 0) {
      setTrades(wsData.trades);
    }
  }, [wsData.ticker, wsData.orderBook, wsData.trades]);
  
  return {
    ticker,
    orderBook,
    trades,
    loading,
    error,
    connected: wsData.connected
  };
};

export default useMarketData;
