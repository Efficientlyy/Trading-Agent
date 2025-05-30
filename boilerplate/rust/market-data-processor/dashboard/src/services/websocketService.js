// WebSocket service for real-time data
import { useEffect, useState } from 'react';

// WebSocket connection URL
const WS_URL = process.env.NODE_ENV === 'production' 
  ? `ws://${window.location.host}/ws` 
  : 'ws://localhost:8080/ws';

// Create WebSocket connection
const createWebSocket = () => {
  const ws = new WebSocket(WS_URL);
  
  ws.onopen = () => {
    console.log('WebSocket connection established');
    // Subscribe to BTC/USDC data
    ws.send(JSON.stringify({
      type: 'subscribe',
      channel: 'BTCUSDC'
    }));
  };
  
  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
  };
  
  ws.onclose = (event) => {
    console.log('WebSocket connection closed:', event.code, event.reason);
    // Attempt to reconnect after 5 seconds
    setTimeout(() => {
      console.log('Attempting to reconnect WebSocket...');
      createWebSocket();
    }, 5000);
  };
  
  return ws;
};

// Hook for using WebSocket data
export const useWebSocket = () => {
  const [ticker, setTicker] = useState(null);
  const [orderBook, setOrderBook] = useState(null);
  const [trades, setTrades] = useState([]);
  const [connected, setConnected] = useState(false);
  
  useEffect(() => {
    const ws = createWebSocket();
    
    ws.onopen = () => {
      console.log('WebSocket connection established');
      setConnected(true);
      // Subscribe to BTC/USDC data
      ws.send(JSON.stringify({
        type: 'subscribe',
        channel: 'BTCUSDC'
      }));
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
          case 'ticker':
            setTicker(data.data);
            break;
          case 'orderbook':
            setOrderBook(data.data);
            break;
          case 'trades':
            setTrades(prevTrades => {
              // Add new trade to the beginning of the array
              const newTrades = [data.data, ...prevTrades];
              // Limit to 50 trades
              return newTrades.slice(0, 50);
            });
            break;
          default:
            console.log('Unknown message type:', data.type);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnected(false);
    };
    
    ws.onclose = (event) => {
      console.log('WebSocket connection closed:', event.code, event.reason);
      setConnected(false);
      // Attempt to reconnect after 5 seconds
      setTimeout(() => {
        console.log('Attempting to reconnect WebSocket...');
      }, 5000);
    };
    
    // Clean up on unmount
    return () => {
      ws.close();
    };
  }, []);
  
  return { ticker, orderBook, trades, connected };
};

export default { useWebSocket };
