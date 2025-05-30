import React, { useState } from 'react';
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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  TextField,
  Button,
  CircularProgress,
  Pagination
} from '@mui/material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { useQuery } from 'react-query';
import { fetchOrderHistory } from '../services/apiService';

// Format timestamp to readable format
const formatTimestamp = (timestamp) => {
  const date = new Date(timestamp);
  return date.toLocaleString();
};

// Get status color based on order status
const getStatusColor = (status) => {
  switch (status.toUpperCase()) {
    case 'FILLED':
      return 'success';
    case 'PARTIALLY_FILLED':
      return 'info';
    case 'CANCELED':
      return 'warning';
    case 'REJECTED':
      return 'error';
    case 'NEW':
      return 'primary';
    default:
      return 'default';
  }
};

function OrderHistory() {
  // Filter state
  const [symbol, setSymbol] = useState('');
  const [status, setStatus] = useState('');
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);
  const [page, setPage] = useState(1);
  const [limit] = useState(10); // Items per page
  
  // Query for order history
  const { 
    data: orderHistoryResponse, 
    isLoading, 
    error,
    refetch
  } = useQuery(['orderHistory', { symbol, status, page, limit }], 
    () => fetchOrderHistory({ 
      symbol, 
      status, 
      startDate: startDate ? startDate.getTime() : null,
      endDate: endDate ? endDate.getTime() : null,
      page, 
      limit 
    }),
    {
      keepPreviousData: true
    }
  );
  
  // Extract data from response
  const orders = orderHistoryResponse?.orders || [];
  const totalOrders = orderHistoryResponse?.total || 0;
  const totalPages = Math.ceil(totalOrders / limit);
  
  // Handle filter changes
  const handleFilterChange = () => {
    setPage(1); // Reset to first page when filters change
    refetch();
  };
  
  // Handle pagination change
  const handlePageChange = (event, value) => {
    setPage(value);
  };
  
  // Reset filters
  const handleResetFilters = () => {
    setSymbol('');
    setStatus('');
    setStartDate(null);
    setEndDate(null);
    setPage(1);
  };
  
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Order History
        </Typography>
        <Divider sx={{ mb: 3 }} />
        
        {/* Filters */}
        <Box sx={{ mb: 3 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} sm={6} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Symbol</InputLabel>
                <Select
                  value={symbol}
                  label="Symbol"
                  onChange={(e) => setSymbol(e.target.value)}
                >
                  <MenuItem value="">All</MenuItem>
                  <MenuItem value="BTCUSDT">BTCUSDT</MenuItem>
                  <MenuItem value="ETHUSDT">ETHUSDT</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={6} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Status</InputLabel>
                <Select
                  value={status}
                  label="Status"
                  onChange={(e) => setStatus(e.target.value)}
                >
                  <MenuItem value="">All</MenuItem>
                  <MenuItem value="FILLED">Filled</MenuItem>
                  <MenuItem value="PARTIALLY_FILLED">Partially Filled</MenuItem>
                  <MenuItem value="CANCELED">Canceled</MenuItem>
                  <MenuItem value="REJECTED">Rejected</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <LocalizationProvider dateAdapter={AdapterDateFns}>
              <Grid item xs={12} sm={6} md={2}>
                <DatePicker
                  label="Start Date"
                  value={startDate}
                  onChange={(newValue) => setStartDate(newValue)}
                  renderInput={(params) => <TextField size="small" {...params} fullWidth />}
                />
              </Grid>
              
              <Grid item xs={12} sm={6} md={2}>
                <DatePicker
                  label="End Date"
                  value={endDate}
                  onChange={(newValue) => setEndDate(newValue)}
                  renderInput={(params) => <TextField size="small" {...params} fullWidth />}
                />
              </Grid>
            </LocalizationProvider>
            
            <Grid item xs={12} sm={6} md={2}>
              <Button 
                variant="contained" 
                fullWidth
                onClick={handleFilterChange}
              >
                Apply Filters
              </Button>
            </Grid>
            
            <Grid item xs={12} sm={6} md={2}>
              <Button 
                variant="outlined" 
                fullWidth
                onClick={handleResetFilters}
              >
                Reset
              </Button>
            </Grid>
          </Grid>
        </Box>
        
        {/* Order Table */}
        {isLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Box sx={{ textAlign: 'center', p: 3 }}>
            <Typography color="error">
              Error loading order history. Please try again.
            </Typography>
            <Button 
              variant="contained" 
              sx={{ mt: 2 }}
              onClick={() => refetch()}
            >
              Retry
            </Button>
          </Box>
        ) : orders.length === 0 ? (
          <Box sx={{ textAlign: 'center', p: 3 }}>
            <Typography color="text.secondary">
              No orders found matching your filters.
            </Typography>
          </Box>
        ) : (
          <>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Date</TableCell>
                    <TableCell>Symbol</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Side</TableCell>
                    <TableCell align="right">Price</TableCell>
                    <TableCell align="right">Amount</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell align="right">Total</TableCell>
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
                        {order.type}
                      </TableCell>
                      <TableCell>
                        <Chip
                          size="small"
                          label={order.side}
                          color={order.side === 'BUY' ? 'success' : 'error'}
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell align="right">
                        {order.fillPrice ? order.fillPrice.toFixed(2) : order.price ? order.price.toFixed(2) : 'Market'}
                      </TableCell>
                      <TableCell align="right">
                        {order.fillQuantity ? order.fillQuantity.toFixed(6) : order.quantity.toFixed(6)}
                      </TableCell>
                      <TableCell>
                        <Chip
                          size="small"
                          label={order.status}
                          color={getStatusColor(order.status)}
                        />
                      </TableCell>
                      <TableCell align="right">
                        {order.fillPrice && order.fillQuantity 
                          ? `$${(order.fillPrice * order.fillQuantity).toFixed(2)}`
                          : '-'}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
            
            {/* Pagination */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Showing {orders.length} of {totalOrders} orders
              </Typography>
              
              <Pagination 
                count={totalPages} 
                page={page} 
                onChange={handlePageChange} 
                color="primary"
              />
            </Box>
          </>
        )}
      </CardContent>
    </Card>
  );
}

export default OrderHistory;
