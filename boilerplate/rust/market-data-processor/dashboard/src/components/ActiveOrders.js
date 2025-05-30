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
  Chip,
  Button,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  ArrowUpward as ArrowUpwardIcon,
  ArrowDownward as ArrowDownwardIcon,
  Cancel as CancelIcon
} from '@mui/icons-material';
import { cancelOrder } from '../services/apiService';

// Format timestamp to readable format
const formatTimestamp = (timestamp) => {
  const date = new Date(timestamp);
  return date.toLocaleString();
};

// Get order type color
const getOrderTypeColor = (type) => {
  switch (type.toUpperCase()) {
    case 'LIMIT':
      return 'primary';
    case 'MARKET':
      return 'warning';
    case 'STOP_LOSS':
      return 'error';
    case 'STOP_LIMIT':
      return 'info';
    default:
      return 'default';
  }
};

function ActiveOrders({ orders = [] }) {
  const handleCancelOrder = async (orderId) => {
    try {
      await cancelOrder(orderId);
      // Order will be removed from the list on the next data refresh
    } catch (error) {
      console.error('Failed to cancel order:', error);
      // You might want to show a snackbar or notification here
    }
  };

  if (!orders || orders.length === 0) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Active Orders
          </Typography>
          <Divider sx={{ mb: 2 }} />
          <Typography variant="body2" color="text.secondary">
            No active orders at the moment
          </Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Active Orders
        </Typography>
        <Divider sx={{ mb: 2 }} />
        
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Time</TableCell>
                <TableCell>Symbol</TableCell>
                <TableCell>Type</TableCell>
                <TableCell>Side</TableCell>
                <TableCell align="right">Price</TableCell>
                <TableCell align="right">Amount</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {orders.map((order) => (
                <TableRow key={order.id} hover>
                  <TableCell>
                    {formatTimestamp(order.timestamp)}
                  </TableCell>
                  <TableCell>
                    {order.symbol}
                  </TableCell>
                  <TableCell>
                    <Chip
                      size="small"
                      label={order.type}
                      color={getOrderTypeColor(order.type)}
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell>
                    <Chip
                      size="small"
                      icon={order.side === 'BUY' ? <ArrowUpwardIcon /> : <ArrowDownwardIcon />}
                      label={order.side}
                      color={order.side === 'BUY' ? 'success' : 'error'}
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell align="right">
                    {order.price ? order.price.toFixed(2) : 'Market'}
                  </TableCell>
                  <TableCell align="right">
                    {order.quantity.toFixed(6)}
                  </TableCell>
                  <TableCell align="right">
                    <Tooltip title="Cancel Order">
                      <IconButton 
                        size="small" 
                        color="error"
                        onClick={() => handleCancelOrder(order.id)}
                      >
                        <CancelIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        
        {orders.length > 0 && (
          <Box sx={{ mt: 2, textAlign: 'right' }}>
            <Button 
              variant="outlined" 
              color="error" 
              size="small"
              onClick={() => orders.forEach(order => handleCancelOrder(order.id))}
            >
              Cancel All Orders
            </Button>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}

export default ActiveOrders;
