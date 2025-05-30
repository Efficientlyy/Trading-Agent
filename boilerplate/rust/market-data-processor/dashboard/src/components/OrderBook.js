import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow,
  Divider
} from '@mui/material';
import { styled } from '@mui/material/styles';

// Styled components for the order book
const OrderBookContainer = styled(Box)(({ theme }) => ({
  height: 400,
  overflow: 'hidden',
  display: 'flex',
  flexDirection: 'column',
}));

const OrderBookRow = styled(TableRow)(({ type, theme }) => ({
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
  },
}));

const PriceCell = styled(TableCell)(({ type, theme }) => ({
  color: type === 'ask' ? theme.palette.error.main : theme.palette.success.main,
  fontWeight: 500,
}));

const QuantityCell = styled(TableCell)(({ theme }) => ({
  textAlign: 'right',
}));

const TotalCell = styled(TableCell)(({ theme }) => ({
  textAlign: 'right',
}));

const DepthVisualizerContainer = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: 0,
  bottom: 0,
  right: 0,
  zIndex: 0,
  transition: 'width 0.3s ease-in-out',
}));

const DepthVisualizer = ({ type, depth, maxDepth }) => {
  const width = `${(depth / maxDepth) * 100}%`;
  
  return (
    <DepthVisualizerContainer
      sx={{
        width,
        backgroundColor: type === 'ask' ? 'rgba(244, 67, 54, 0.1)' : 'rgba(76, 175, 80, 0.1)',
      }}
    />
  );
};

// Mock data for the order book
const generateMockOrderBook = (basePrice, symbol) => {
  const asks = [];
  const bids = [];
  
  // Generate 15 asks (sell orders) above the base price
  for (let i = 0; i < 15; i++) {
    const price = basePrice * (1 + (i + 1) * 0.001);
    const quantity = Math.random() * 2 + 0.1;
    asks.push({
      price,
      quantity,
      total: price * quantity,
    });
  }
  
  // Sort asks from lowest to highest
  asks.sort((a, b) => a.price - b.price);
  
  // Generate 15 bids (buy orders) below the base price
  for (let i = 0; i < 15; i++) {
    const price = basePrice * (1 - (i + 1) * 0.001);
    const quantity = Math.random() * 2 + 0.1;
    bids.push({
      price,
      quantity,
      total: price * quantity,
    });
  }
  
  // Sort bids from highest to lowest
  bids.sort((a, b) => b.price - a.price);
  
  return { asks, bids };
};

function OrderBook({ symbol }) {
  const [orderBook, setOrderBook] = useState(null);
  
  // Calculate max depth for visualizer
  const maxDepth = orderBook ? Math.max(
    ...orderBook.asks.map(ask => ask.total),
    ...orderBook.bids.map(bid => bid.total)
  ) : 0;
  
  useEffect(() => {
    // In a real application, this would fetch data from an API
    // and potentially set up a websocket connection for real-time updates
    const basePrice = symbol === 'BTCUSDT' ? 50000 : 3000;
    setOrderBook(generateMockOrderBook(basePrice, symbol));
    
    // Simulate occasional updates
    const interval = setInterval(() => {
      setOrderBook(generateMockOrderBook(basePrice + (Math.random() * 100 - 50), symbol));
    }, 5000);
    
    return () => clearInterval(interval);
  }, [symbol]);
  
  if (!orderBook) {
    return <Box>Loading order book...</Box>;
  }
  
  return (
    <OrderBookContainer>
      <TableContainer sx={{ maxHeight: 200, overflow: 'auto' }}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell>Price (USDT)</TableCell>
              <TableCell align="right">Quantity</TableCell>
              <TableCell align="right">Total</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {orderBook.asks.map((ask, index) => (
              <OrderBookRow key={`ask-${index}`} type="ask">
                <PriceCell type="ask">
                  {ask.price.toFixed(2)}
                  <DepthVisualizer type="ask" depth={ask.total} maxDepth={maxDepth} />
                </PriceCell>
                <QuantityCell>{ask.quantity.toFixed(4)}</QuantityCell>
                <TotalCell>{ask.total.toFixed(2)}</TotalCell>
              </OrderBookRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      
      <Box sx={{ p: 1, display: 'flex', justifyContent: 'center', bgcolor: 'background.default' }}>
        <Typography variant="h6" color={orderBook.bids[0] && orderBook.bids[0].price > orderBook.asks[0].price ? 'success.main' : 'text.primary'}>
          {orderBook.bids[0] ? orderBook.bids[0].price.toFixed(2) : '0.00'}
        </Typography>
      </Box>
      
      <TableContainer sx={{ maxHeight: 200, overflow: 'auto' }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Price (USDT)</TableCell>
              <TableCell align="right">Quantity</TableCell>
              <TableCell align="right">Total</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {orderBook.bids.map((bid, index) => (
              <OrderBookRow key={`bid-${index}`} type="bid">
                <PriceCell type="bid">
                  {bid.price.toFixed(2)}
                  <DepthVisualizer type="bid" depth={bid.total} maxDepth={maxDepth} />
                </PriceCell>
                <QuantityCell>{bid.quantity.toFixed(4)}</QuantityCell>
                <TotalCell>{bid.total.toFixed(2)}</TotalCell>
              </OrderBookRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </OrderBookContainer>
  );
}

export default OrderBook;
