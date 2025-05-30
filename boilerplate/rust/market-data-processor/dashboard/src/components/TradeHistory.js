import React, { useState } from 'react';
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow, 
  Paper,
  TablePagination,
  Chip,
  Typography,
  Box
} from '@mui/material';

// Sample trade history data
const tradeHistoryData = [
  {
    id: 'ord123',
    symbol: 'BTCUSDT',
    side: 'BUY',
    type: 'MARKET',
    quantity: 0.05,
    price: 50100.25,
    value: 2505.01,
    fee: 2.51,
    status: 'FILLED',
    timestamp: '2025-05-30T04:52:15Z'
  },
  {
    id: 'ord122',
    symbol: 'ETHUSDT',
    side: 'BUY',
    type: 'LIMIT',
    quantity: 0.75,
    price: 3050.50,
    value: 2287.88,
    fee: 2.29,
    status: 'FILLED',
    timestamp: '2025-05-30T04:45:30Z'
  },
  {
    id: 'ord121',
    symbol: 'BTCUSDT',
    side: 'SELL',
    type: 'MARKET',
    quantity: 0.03,
    price: 50050.75,
    value: 1501.52,
    fee: 1.50,
    status: 'FILLED',
    timestamp: '2025-05-30T04:30:45Z'
  },
  {
    id: 'ord120',
    symbol: 'ADAUSDT',
    side: 'BUY',
    type: 'LIMIT',
    quantity: 500,
    price: 0.58,
    value: 290.00,
    fee: 0.29,
    status: 'FILLED',
    timestamp: '2025-05-30T04:15:20Z'
  },
  {
    id: 'ord119',
    symbol: 'BTCUSDT',
    side: 'SELL',
    type: 'LIMIT',
    quantity: 0.02,
    price: 50200.00,
    value: 1004.00,
    fee: 1.00,
    status: 'FILLED',
    timestamp: '2025-05-30T03:55:10Z'
  },
  {
    id: 'ord118',
    symbol: 'ETHUSDT',
    side: 'SELL',
    type: 'MARKET',
    quantity: 0.5,
    price: 3045.25,
    value: 1522.63,
    fee: 1.52,
    status: 'FILLED',
    timestamp: '2025-05-30T03:40:05Z'
  },
  {
    id: 'ord117',
    symbol: 'XRPUSDT',
    side: 'BUY',
    type: 'LIMIT',
    quantity: 1000,
    price: 0.45,
    value: 450.00,
    fee: 0.45,
    status: 'FILLED',
    timestamp: '2025-05-30T03:25:15Z'
  },
  {
    id: 'ord116',
    symbol: 'BTCUSDT',
    side: 'BUY',
    type: 'MARKET',
    quantity: 0.04,
    price: 49950.00,
    value: 1998.00,
    fee: 2.00,
    status: 'FILLED',
    timestamp: '2025-05-30T03:10:30Z'
  },
  {
    id: 'ord115',
    symbol: 'ADAUSDT',
    side: 'SELL',
    type: 'LIMIT',
    quantity: 300,
    price: 0.59,
    value: 177.00,
    fee: 0.18,
    status: 'FILLED',
    timestamp: '2025-05-30T02:55:25Z'
  },
  {
    id: 'ord114',
    symbol: 'ETHUSDT',
    side: 'BUY',
    type: 'LIMIT',
    quantity: 0.25,
    price: 3030.00,
    value: 757.50,
    fee: 0.76,
    status: 'FILLED',
    timestamp: '2025-05-30T02:40:50Z'
  }
];

function TradeHistory() {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(5);

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };
  
  // Format timestamp to a more readable format
  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };
  
  return (
    <Box>
      <TableContainer component={Paper} sx={{ mb: 2 }}>
        <Table sx={{ minWidth: 650 }} size="small" aria-label="trade history table">
          <TableHead>
            <TableRow>
              <TableCell>Symbol</TableCell>
              <TableCell>Side</TableCell>
              <TableCell>Type</TableCell>
              <TableCell align="right">Quantity</TableCell>
              <TableCell align="right">Price</TableCell>
              <TableCell align="right">Value</TableCell>
              <TableCell align="right">Fee</TableCell>
              <TableCell align="right">Status</TableCell>
              <TableCell align="right">Time</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {tradeHistoryData
              .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
              .map((trade) => (
                <TableRow
                  key={trade.id}
                  sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                >
                  <TableCell component="th" scope="row">
                    {trade.symbol}
                  </TableCell>
                  <TableCell>
                    <Chip 
                      label={trade.side} 
                      color={trade.side === 'BUY' ? 'success' : 'error'} 
                      size="small" 
                    />
                  </TableCell>
                  <TableCell>{trade.type}</TableCell>
                  <TableCell align="right">{trade.quantity}</TableCell>
                  <TableCell align="right">${trade.price.toFixed(2)}</TableCell>
                  <TableCell align="right">${trade.value.toFixed(2)}</TableCell>
                  <TableCell align="right">${trade.fee.toFixed(2)}</TableCell>
                  <TableCell align="right">
                    <Chip 
                      label={trade.status} 
                      color={trade.status === 'FILLED' ? 'success' : 'warning'} 
                      size="small" 
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell align="right">{formatTime(trade.timestamp)}</TableCell>
                </TableRow>
              ))}
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        rowsPerPageOptions={[5, 10, 25]}
        component="div"
        count={tradeHistoryData.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={handleChangePage}
        onRowsPerPageChange={handleChangeRowsPerPage}
      />
    </Box>
  );
}

export default TradeHistory;
