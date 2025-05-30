// OrderBook component for displaying BTC/USDC order book
import React, { useEffect, useState } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow,
  CircularProgress
} from '@mui/material';
import { fetchOrderBook, subscribeToOrderBookUpdates } from '../services/apiService';

const OrderBook = ({ symbol = 'BTCUSDC', depth = 10 }) => {
  const [orderBook, setOrderBook] = useState({ bids: [], asks: [] });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Load initial order book data
    const loadOrderBook = async () => {
      try {
        setIsLoading(true);
        const data = await fetchOrderBook(symbol);
        
        if (data) {
          // Sort bids (descending) and asks (ascending)
          const bids = [...data.bids].sort((a, b) => b[0] - a[0]).slice(0, depth);
          const asks = [...data.asks].sort((a, b) => a[0] - b[0]).slice(0, depth);
          
          setOrderBook({ bids, asks });
        }
        
        setIsLoading(false);
      } catch (err) {
        console.error('Failed to load order book data:', err);
        setError('Failed to load order book data. Please try again later.');
        setIsLoading(false);
      }
    };

    loadOrderBook();

    // Subscribe to real-time order book updates
    const unsubscribe = subscribeToOrderBookUpdates(symbol, (data) => {
      if (data) {
        // Sort bids (descending) and asks (ascending)
        const bids = [...data.bids].sort((a, b) => b[0] - a[0]).slice(0, depth);
        const asks = [...data.asks].sort((a, b) => a[0] - b[0]).slice(0, depth);
        
        setOrderBook({ bids, asks });
      }
    });

    // Cleanup
    return () => {
      unsubscribe();
    };
  }, [symbol, depth]);

  // Calculate total quantities for depth visualization
  const maxBidQuantity = Math.max(...orderBook.bids.map(bid => bid[1]), 0);
  const maxAskQuantity = Math.max(...orderBook.asks.map(ask => ask[1]), 0);
  const maxQuantity = Math.max(maxBidQuantity, maxAskQuantity);

  return (
    <Paper elevation={3} sx={{ p: 2, borderRadius: 2, backgroundColor: '#1E1E1E' }}>
      <Typography variant="h6" color="text.primary" sx={{ mb: 2 }}>
        Order Book
      </Typography>
      
      {isLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
          <CircularProgress />
        </Box>
      ) : error ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
          <Typography color="error">{error}</Typography>
        </Box>
      ) : (
        <Box sx={{ display: 'flex', flexDirection: 'column', height: 300 }}>
          {/* Asks (Sell orders) - displayed in reverse order (lowest ask first) */}
          <TableContainer sx={{ flex: 1, maxHeight: 150 }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell>Price (USDC)</TableCell>
                  <TableCell align="right">Amount (BTC)</TableCell>
                  <TableCell align="right">Total</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {orderBook.asks.map((ask, index) => {
                  const price = parseFloat(ask[0]);
                  const amount = parseFloat(ask[1]);
                  const total = price * amount;
                  const percentOfMax = (amount / maxQuantity) * 100;
                  
                  return (
                    <TableRow 
                      key={`ask-${index}`}
                      sx={{ 
                        position: 'relative',
                        '&::after': {
                          content: '""',
                          position: 'absolute',
                          top: 0,
                          right: 0,
                          bottom: 0,
                          width: `${percentOfMax}%`,
                          backgroundColor: 'rgba(235, 77, 75, 0.2)',
                          zIndex: 0
                        }
                      }}
                    >
                      <TableCell sx={{ color: '#eb4d4b', position: 'relative', zIndex: 1 }}>
                        {price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </TableCell>
                      <TableCell align="right" sx={{ position: 'relative', zIndex: 1 }}>
                        {amount.toLocaleString(undefined, { minimumFractionDigits: 6, maximumFractionDigits: 6 })}
                      </TableCell>
                      <TableCell align="right" sx={{ position: 'relative', zIndex: 1 }}>
                        {total.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
          
          {/* Spread */}
          <Box sx={{ py: 1, textAlign: 'center', borderTop: '1px solid rgba(255, 255, 255, 0.1)', borderBottom: '1px solid rgba(255, 255, 255, 0.1)' }}>
            {orderBook.bids.length > 0 && orderBook.asks.length > 0 && (
              <Typography variant="body2" color="text.secondary">
                Spread: {(parseFloat(orderBook.asks[0][0]) - parseFloat(orderBook.bids[0][0])).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ({((parseFloat(orderBook.asks[0][0]) / parseFloat(orderBook.bids[0][0]) - 1) * 100).toFixed(2)}%)
              </Typography>
            )}
          </Box>
          
          {/* Bids (Buy orders) */}
          <TableContainer sx={{ flex: 1, maxHeight: 150 }}>
            <Table size="small">
              <TableBody>
                {orderBook.bids.map((bid, index) => {
                  const price = parseFloat(bid[0]);
                  const amount = parseFloat(bid[1]);
                  const total = price * amount;
                  const percentOfMax = (amount / maxQuantity) * 100;
                  
                  return (
                    <TableRow 
                      key={`bid-${index}`}
                      sx={{ 
                        position: 'relative',
                        '&::after': {
                          content: '""',
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          bottom: 0,
                          width: `${percentOfMax}%`,
                          backgroundColor: 'rgba(46, 213, 115, 0.2)',
                          zIndex: 0
                        }
                      }}
                    >
                      <TableCell sx={{ color: '#2ed573', position: 'relative', zIndex: 1 }}>
                        {price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </TableCell>
                      <TableCell align="right" sx={{ position: 'relative', zIndex: 1 }}>
                        {amount.toLocaleString(undefined, { minimumFractionDigits: 6, maximumFractionDigits: 6 })}
                      </TableCell>
                      <TableCell align="right" sx={{ position: 'relative', zIndex: 1 }}>
                        {total.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      )}
    </Paper>
  );
};

export default OrderBook;
