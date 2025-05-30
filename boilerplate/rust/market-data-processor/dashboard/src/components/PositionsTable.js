import React, { useState } from 'react';
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow, 
  Paper,
  Button,
  Chip,
  IconButton,
  Typography,
  Box
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

// Sample positions data
const positionsData = [
  {
    id: 'pos1',
    symbol: 'BTCUSDT',
    side: 'LONG',
    entryPrice: 50245.75,
    quantity: 0.12,
    currentPrice: 51210.50,
    pnl: 115.77,
    pnlPercent: 1.91,
    openTime: '2025-05-29T14:23:45Z'
  },
  {
    id: 'pos2',
    symbol: 'ETHUSDT',
    side: 'LONG',
    entryPrice: 3120.30,
    quantity: 1.5,
    currentPrice: 3157.25,
    pnl: 55.43,
    pnlPercent: 1.18,
    openTime: '2025-05-29T18:12:30Z'
  },
  {
    id: 'pos3',
    symbol: 'ADAUSDT',
    side: 'SHORT',
    entryPrice: 0.58,
    quantity: 1000,
    currentPrice: 0.56,
    pnl: 20.00,
    pnlPercent: 3.45,
    openTime: '2025-05-30T02:45:15Z'
  }
];

function PositionsTable() {
  const [positions, setPositions] = useState(positionsData);
  
  const handleClosePosition = (positionId) => {
    // In a real app, this would send a request to close the position
    console.log(`Closing position: ${positionId}`);
    
    // Remove the position from the table
    setPositions(positions.filter(position => position.id !== positionId));
  };
  
  // Format timestamp to a more readable format
  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };
  
  // Calculate the total P&L across all positions
  const totalPnl = positions.reduce((sum, position) => sum + position.pnl, 0);
  
  return (
    <Box>
      {positions.length > 0 ? (
        <>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="subtitle1">
              Total Positions: {positions.length}
            </Typography>
            <Typography 
              variant="subtitle1" 
              color={totalPnl >= 0 ? 'success.main' : 'error.main'}
            >
              Total P&L: ${totalPnl.toFixed(2)} ({totalPnl >= 0 ? '+' : ''}{(totalPnl / positions.reduce((sum, position) => sum + (position.entryPrice * position.quantity), 0) * 100).toFixed(2)}%)
            </Typography>
          </Box>
          
          <TableContainer component={Paper} sx={{ mb: 2 }}>
            <Table sx={{ minWidth: 650 }} size="small" aria-label="positions table">
              <TableHead>
                <TableRow>
                  <TableCell>Symbol</TableCell>
                  <TableCell>Side</TableCell>
                  <TableCell align="right">Size</TableCell>
                  <TableCell align="right">Entry Price</TableCell>
                  <TableCell align="right">Current Price</TableCell>
                  <TableCell align="right">P&L</TableCell>
                  <TableCell align="right">Open Time</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {positions.map((position) => (
                  <TableRow
                    key={position.id}
                    sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                  >
                    <TableCell component="th" scope="row">
                      {position.symbol}
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={position.side} 
                        color={position.side === 'LONG' ? 'success' : 'error'} 
                        size="small" 
                      />
                    </TableCell>
                    <TableCell align="right">{position.quantity}</TableCell>
                    <TableCell align="right">${position.entryPrice.toFixed(2)}</TableCell>
                    <TableCell align="right">${position.currentPrice.toFixed(2)}</TableCell>
                    <TableCell 
                      align="right"
                      sx={{ 
                        color: position.pnl >= 0 ? 'success.main' : 'error.main',
                        fontWeight: 'bold'
                      }}
                    >
                      ${position.pnl.toFixed(2)} ({position.pnl >= 0 ? '+' : ''}{position.pnlPercent.toFixed(2)}%)
                    </TableCell>
                    <TableCell align="right">{formatTime(position.openTime)}</TableCell>
                    <TableCell align="right">
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => handleClosePosition(position.id)}
                        title="Close Position"
                      >
                        <CloseIcon fontSize="small" />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </>
      ) : (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Typography variant="body1" color="text.secondary" gutterBottom>
            No open positions
          </Typography>
          <Button variant="outlined" href="#/trading">
            Place an Order
          </Button>
        </Box>
      )}
    </Box>
  );
}

export default PositionsTable;
