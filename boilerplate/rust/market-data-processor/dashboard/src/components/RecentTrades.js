import React from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Divider, 
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip
} from '@mui/material';
import {
  ArrowUpward as ArrowUpwardIcon,
  ArrowDownward as ArrowDownwardIcon
} from '@mui/icons-material';

// Format timestamp to readable format
const formatTimestamp = (timestamp) => {
  const date = new Date(timestamp);
  return date.toLocaleString();
};

// Format price with appropriate precision
const formatPrice = (price, symbol) => {
  // Use more decimal places for higher-value cryptocurrencies
  const precision = symbol.includes('BTC') ? 8 : 
                   symbol.includes('ETH') ? 6 : 2;
  return price.toFixed(precision);
};

function RecentTrades({ trades = [] }) {
  if (!trades || trades.length === 0) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Recent Trades
          </Typography>
          <Divider sx={{ mb: 2 }} />
          <Typography variant="body2" color="text.secondary">
            No recent trades available
          </Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Recent Trades
        </Typography>
        <Divider sx={{ mb: 2 }} />
        
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Time</TableCell>
                <TableCell>Symbol</TableCell>
                <TableCell>Side</TableCell>
                <TableCell align="right">Price</TableCell>
                <TableCell align="right">Amount</TableCell>
                <TableCell align="right">Value</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {trades.map((trade) => (
                <TableRow key={trade.id} hover>
                  <TableCell>
                    {formatTimestamp(trade.timestamp)}
                  </TableCell>
                  <TableCell>
                    {trade.symbol}
                  </TableCell>
                  <TableCell>
                    <Chip
                      size="small"
                      icon={trade.side === 'BUY' ? <ArrowUpwardIcon /> : <ArrowDownwardIcon />}
                      label={trade.side}
                      color={trade.side === 'BUY' ? 'success' : 'error'}
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell align="right">
                    {formatPrice(trade.price, trade.symbol)}
                  </TableCell>
                  <TableCell align="right">
                    {trade.quantity.toFixed(6)}
                  </TableCell>
                  <TableCell align="right">
                    ${(trade.price * trade.quantity).toFixed(2)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        
        {trades.length > 0 && (
          <Box sx={{ mt: 2, textAlign: 'right' }}>
            <Typography variant="body2" color="text.secondary">
              Showing {trades.length} most recent trades
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}

export default RecentTrades;
