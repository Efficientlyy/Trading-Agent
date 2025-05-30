// TradeHistory component for displaying recent BTC/USDC trades
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
import { fetchTrades, subscribeToTradesUpdates } from '../services/apiService';

const TradeHistory = ({ symbol = 'BTCUSDC', limit = 20 }) => {
  const [trades, setTrades] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Load initial trades data
    const loadTrades = async () => {
      try {
        setIsLoading(true);
        const data = await fetchTrades(symbol);
        
        if (data && Array.isArray(data)) {
          setTrades(data.slice(0, limit));
        }
        
        setIsLoading(false);
      } catch (err) {
        console.error('Failed to load trades data:', err);
        setError('Failed to load trades data. Please try again later.');
        setIsLoading(false);
      }
    };

    loadTrades();

    // Subscribe to real-time trades updates
    const unsubscribe = subscribeToTradesUpdates(symbol, (newTrade) => {
      if (newTrade) {
        setTrades(prevTrades => {
          // Add new trade to the beginning and maintain limit
          const updatedTrades = [newTrade, ...prevTrades].slice(0, limit);
          return updatedTrades;
        });
      }
    });

    // Cleanup
    return () => {
      unsubscribe();
    };
  }, [symbol, limit]);

  // Format timestamp to readable time
  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  };

  return (
    <Paper elevation={3} sx={{ p: 2, borderRadius: 2, backgroundColor: '#1E1E1E' }}>
      <Typography variant="h6" color="text.primary" sx={{ mb: 2 }}>
        Recent Trades
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
        <TableContainer sx={{ maxHeight: 300 }}>
          <Table size="small" stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell>Price (USDC)</TableCell>
                <TableCell align="right">Amount (BTC)</TableCell>
                <TableCell align="right">Time</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {trades.map((trade) => (
                <TableRow key={trade.id}>
                  <TableCell sx={{ 
                    color: trade.isBuyerMaker ? '#eb4d4b' : '#2ed573',
                  }}>
                    {parseFloat(trade.price).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </TableCell>
                  <TableCell align="right">
                    {parseFloat(trade.quantity).toLocaleString(undefined, { minimumFractionDigits: 6, maximumFractionDigits: 6 })}
                  </TableCell>
                  <TableCell align="right">
                    {formatTime(trade.timestamp)}
                  </TableCell>
                </TableRow>
              ))}
              {trades.length === 0 && (
                <TableRow>
                  <TableCell colSpan={3} align="center">
                    No trades available
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Paper>
  );
};

export default TradeHistory;
