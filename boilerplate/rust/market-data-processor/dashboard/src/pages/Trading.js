import React, { useState } from 'react';
import { 
  Typography, 
  Grid, 
  Paper, 
  Box, 
  Button, 
  TextField, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem,
  ToggleButtonGroup,
  ToggleButton,
  Divider,
  Alert
} from '@mui/material';
import { styled } from '@mui/material/styles';
import TradeHistory from '../components/TradeHistory';
import PositionsTable from '../components/PositionsTable';

const Item = styled(Paper)(({ theme }) => ({
  backgroundColor: theme.palette.background.paper,
  padding: theme.spacing(3),
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[3],
  height: '100%',
}));

function Trading() {
  const [orderType, setOrderType] = useState('MARKET');
  const [side, setSide] = useState('BUY');
  const [symbol, setSymbol] = useState('BTCUSDT');
  const [quantity, setQuantity] = useState('');
  const [price, setPrice] = useState('');
  const [orderSuccess, setOrderSuccess] = useState(false);
  
  const handleOrderTypeChange = (event) => {
    setOrderType(event.target.value);
  };
  
  const handleSymbolChange = (event) => {
    setSymbol(event.target.value);
  };
  
  const handleSideChange = (event, newSide) => {
    if (newSide !== null) {
      setSide(newSide);
    }
  };
  
  const handleSubmitOrder = () => {
    // Simulate order submission
    console.log({
      symbol,
      side,
      type: orderType,
      quantity,
      price: orderType === 'LIMIT' ? price : undefined
    });
    
    // Show success message
    setOrderSuccess(true);
    
    // Reset form
    setQuantity('');
    setPrice('');
    
    // Hide success message after 3 seconds
    setTimeout(() => {
      setOrderSuccess(false);
    }, 3000);
  };
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Trading
      </Typography>
      <Typography variant="subtitle1" color="text.secondary" paragraph>
        Place orders, manage positions, and view your trading history
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6} lg={4}>
          <Item>
            <Typography variant="h6" gutterBottom>Place Order</Typography>
            
            {orderSuccess && (
              <Alert severity="success" sx={{ mb: 2 }}>
                Order placed successfully!
              </Alert>
            )}
            
            <Box sx={{ mb: 2 }}>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel id="symbol-select-label">Symbol</InputLabel>
                <Select
                  labelId="symbol-select-label"
                  id="symbol-select"
                  value={symbol}
                  label="Symbol"
                  onChange={handleSymbolChange}
                >
                  <MenuItem value="BTCUSDT">BTC/USDT</MenuItem>
                  <MenuItem value="ETHUSDT">ETH/USDT</MenuItem>
                  <MenuItem value="XRPUSDT">XRP/USDT</MenuItem>
                  <MenuItem value="ADAUSDT">ADA/USDT</MenuItem>
                </Select>
              </FormControl>
              
              <Box sx={{ mb: 2 }}>
                <ToggleButtonGroup
                  color={side === 'BUY' ? 'success' : 'error'}
                  value={side}
                  exclusive
                  onChange={handleSideChange}
                  fullWidth
                >
                  <ToggleButton value="BUY">Buy</ToggleButton>
                  <ToggleButton value="SELL">Sell</ToggleButton>
                </ToggleButtonGroup>
              </Box>
              
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel id="order-type-select-label">Order Type</InputLabel>
                <Select
                  labelId="order-type-select-label"
                  id="order-type-select"
                  value={orderType}
                  label="Order Type"
                  onChange={handleOrderTypeChange}
                >
                  <MenuItem value="MARKET">Market</MenuItem>
                  <MenuItem value="LIMIT">Limit</MenuItem>
                </Select>
              </FormControl>
              
              <TextField
                label="Quantity"
                variant="outlined"
                fullWidth
                value={quantity}
                onChange={(e) => setQuantity(e.target.value)}
                sx={{ mb: 2 }}
                type="number"
                InputProps={{ inputProps: { min: 0, step: 0.001 } }}
              />
              
              {orderType === 'LIMIT' && (
                <TextField
                  label="Price"
                  variant="outlined"
                  fullWidth
                  value={price}
                  onChange={(e) => setPrice(e.target.value)}
                  sx={{ mb: 2 }}
                  type="number"
                  InputProps={{ inputProps: { min: 0, step: 0.01 } }}
                />
              )}
              
              <Button 
                variant="contained" 
                fullWidth
                color={side === 'BUY' ? 'success' : 'error'}
                onClick={handleSubmitOrder}
                disabled={!quantity || (orderType === 'LIMIT' && !price)}
              >
                {side === 'BUY' ? 'Buy' : 'Sell'} {symbol.replace('USDT', '')}
              </Button>
            </Box>
          </Item>
        </Grid>
        
        <Grid item xs={12} md={6} lg={8}>
          <Item>
            <Typography variant="h6" gutterBottom>Open Positions</Typography>
            <PositionsTable />
          </Item>
        </Grid>
        
        <Grid item xs={12}>
          <Item>
            <Typography variant="h6" gutterBottom>Recent Trades</Typography>
            <TradeHistory />
          </Item>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Trading;
