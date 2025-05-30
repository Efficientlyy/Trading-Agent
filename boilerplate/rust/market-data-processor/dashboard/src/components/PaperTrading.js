// PaperTrading component for BTC/USDC paper trading
import React, { useEffect, useState } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  TextField, 
  Button, 
  Grid,
  CircularProgress,
  Tabs,
  Tab,
  Divider,
  Alert
} from '@mui/material';
import { fetchAccount, fetchTicker, placePaperOrder } from '../services/apiService';

const PaperTrading = ({ symbol = 'BTCUSDC' }) => {
  const [account, setAccount] = useState({ usdc: 0, btc: 0 });
  const [ticker, setTicker] = useState({ price: 0 });
  const [orderType, setOrderType] = useState('market');
  const [side, setSide] = useState('buy');
  const [quantity, setQuantity] = useState('');
  const [price, setPrice] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);

  useEffect(() => {
    // Load account and ticker data
    const loadData = async () => {
      try {
        setIsLoading(true);
        
        // Fetch account data
        const accountData = await fetchAccount();
        if (accountData) {
          setAccount({
            usdc: accountData.balances.USDC || 0,
            btc: accountData.balances.BTC || 0
          });
        }
        
        // Fetch ticker data
        const tickerData = await fetchTicker(symbol);
        if (tickerData) {
          setTicker({ price: tickerData.price });
          setPrice(tickerData.price.toString());
        }
        
        setIsLoading(false);
      } catch (err) {
        console.error('Failed to load data:', err);
        setError('Failed to load account or market data. Please try again later.');
        setIsLoading(false);
      }
    };

    loadData();
    
    // Refresh data every 10 seconds
    const intervalId = setInterval(loadData, 10000);
    
    // Cleanup
    return () => {
      clearInterval(intervalId);
    };
  }, [symbol]);

  // Handle order type change
  const handleOrderTypeChange = (event, newValue) => {
    setOrderType(newValue);
  };

  // Handle order submission
  const handleSubmit = async () => {
    try {
      setError(null);
      setSuccessMessage(null);
      setIsSubmitting(true);
      
      // Validate inputs
      if (!quantity || parseFloat(quantity) <= 0) {
        setError('Please enter a valid quantity');
        setIsSubmitting(false);
        return;
      }
      
      if (orderType === 'limit' && (!price || parseFloat(price) <= 0)) {
        setError('Please enter a valid price');
        setIsSubmitting(false);
        return;
      }
      
      // Create order object
      const orderData = {
        symbol,
        side,
        type: orderType,
        quantity: parseFloat(quantity),
        price: orderType === 'limit' ? parseFloat(price) : undefined
      };
      
      // Submit order
      const result = await placePaperOrder(orderData);
      
      // Update account data
      const accountData = await fetchAccount();
      if (accountData) {
        setAccount({
          usdc: accountData.balances.USDC || 0,
          btc: accountData.balances.BTC || 0
        });
      }
      
      // Show success message
      setSuccessMessage(`${side.toUpperCase()} order placed successfully`);
      
      // Reset form
      setQuantity('');
      
      setIsSubmitting(false);
    } catch (err) {
      console.error('Error placing order:', err);
      setError('Failed to place order. Please try again.');
      setIsSubmitting(false);
    }
  };

  // Calculate max quantity based on available balance
  const calculateMaxQuantity = () => {
    if (side === 'buy') {
      return account.usdc / (orderType === 'market' ? ticker.price * 1.005 : parseFloat(price) || ticker.price);
    } else {
      return account.btc;
    }
  };

  // Handle max button click
  const handleMaxClick = () => {
    const maxQty = calculateMaxQuantity();
    setQuantity(maxQty.toFixed(6));
  };

  return (
    <Paper elevation={3} sx={{ p: 2, borderRadius: 2, backgroundColor: '#1E1E1E' }}>
      <Typography variant="h6" color="text.primary" sx={{ mb: 2 }}>
        Paper Trading
      </Typography>
      
      {isLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Box>
          {/* Account Balance */}
          <Paper variant="outlined" sx={{ p: 2, mb: 2, backgroundColor: 'rgba(0, 0, 0, 0.2)' }}>
            <Typography variant="subtitle2" color="text.secondary">
              Paper Trading Account
            </Typography>
            <Grid container spacing={2} sx={{ mt: 1 }}>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  USDC Balance
                </Typography>
                <Typography variant="h6" color="text.primary">
                  ${account.usdc.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="text.secondary">
                  BTC Balance
                </Typography>
                <Typography variant="h6" color="text.primary">
                  {account.btc.toLocaleString(undefined, { minimumFractionDigits: 8, maximumFractionDigits: 8 })}
                </Typography>
              </Grid>
            </Grid>
          </Paper>
          
          {/* Order Form */}
          <Box sx={{ mb: 2 }}>
            {/* Buy/Sell Tabs */}
            <Box sx={{ display: 'flex', mb: 2 }}>
              <Button 
                variant={side === 'buy' ? 'contained' : 'outlined'} 
                color="success"
                onClick={() => setSide('buy')}
                sx={{ flex: 1, mr: 1 }}
              >
                Buy
              </Button>
              <Button 
                variant={side === 'sell' ? 'contained' : 'outlined'} 
                color="error"
                onClick={() => setSide('sell')}
                sx={{ flex: 1 }}
              >
                Sell
              </Button>
            </Box>
            
            {/* Order Type Tabs */}
            <Tabs 
              value={orderType} 
              onChange={handleOrderTypeChange} 
              sx={{ mb: 2 }}
              variant="fullWidth"
            >
              <Tab label="Market" value="market" />
              <Tab label="Limit" value="limit" />
            </Tabs>
            
            {/* Order Form */}
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <TextField
                    label="Quantity (BTC)"
                    type="number"
                    value={quantity}
                    onChange={(e) => setQuantity(e.target.value)}
                    fullWidth
                    variant="outlined"
                    size="small"
                    InputProps={{
                      inputProps: { min: 0, step: 0.000001 }
                    }}
                  />
                  <Button 
                    variant="text" 
                    onClick={handleMaxClick}
                    sx={{ ml: 1 }}
                  >
                    MAX
                  </Button>
                </Box>
              </Grid>
              
              {orderType === 'limit' && (
                <Grid item xs={12}>
                  <TextField
                    label="Price (USDC)"
                    type="number"
                    value={price}
                    onChange={(e) => setPrice(e.target.value)}
                    fullWidth
                    variant="outlined"
                    size="small"
                    InputProps={{
                      inputProps: { min: 0, step: 0.01 }
                    }}
                  />
                </Grid>
              )}
              
              <Grid item xs={12}>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  {side === 'buy' ? 'Total Cost' : 'Total Proceeds'}: ${((parseFloat(quantity) || 0) * (orderType === 'market' ? ticker.price : parseFloat(price) || 0)).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </Typography>
                
                <Button
                  variant="contained"
                  color={side === 'buy' ? 'success' : 'error'}
                  fullWidth
                  onClick={handleSubmit}
                  disabled={isSubmitting}
                >
                  {isSubmitting ? <CircularProgress size={24} /> : `${side === 'buy' ? 'Buy' : 'Sell'} ${orderType === 'market' ? 'Market' : 'Limit'}`}
                </Button>
              </Grid>
            </Grid>
          </Box>
          
          {/* Error and Success Messages */}
          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
          
          {successMessage && (
            <Alert severity="success" sx={{ mt: 2 }}>
              {successMessage}
            </Alert>
          )}
        </Box>
      )}
    </Paper>
  );
};

export default PaperTrading;
