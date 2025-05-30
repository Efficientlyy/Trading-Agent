import React, { useState } from 'react';
import { Typography, Grid, Paper, Box, FormControl, InputLabel, Select, MenuItem } from '@mui/material';
import { styled } from '@mui/material/styles';
import PriceChart from '../components/PriceChart';
import OrderBook from '../components/OrderBook';

const Item = styled(Paper)(({ theme }) => ({
  backgroundColor: theme.palette.background.paper,
  padding: theme.spacing(3),
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[3],
  height: '100%',
}));

function MarketData() {
  const [symbol, setSymbol] = useState('BTCUSDT');
  const [timeframe, setTimeframe] = useState('1h');
  
  const handleSymbolChange = (event) => {
    setSymbol(event.target.value);
  };
  
  const handleTimeframeChange = (event) => {
    setTimeframe(event.target.value);
  };
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Market Data
      </Typography>
      <Typography variant="subtitle1" color="text.secondary" paragraph>
        Real-time price charts, order book visualization, and market indicators
      </Typography>
      
      <Box sx={{ mb: 3, display: 'flex', gap: 2 }}>
        <FormControl sx={{ minWidth: 120 }}>
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
        
        <FormControl sx={{ minWidth: 120 }}>
          <InputLabel id="timeframe-select-label">Timeframe</InputLabel>
          <Select
            labelId="timeframe-select-label"
            id="timeframe-select"
            value={timeframe}
            label="Timeframe"
            onChange={handleTimeframeChange}
          >
            <MenuItem value="1m">1 minute</MenuItem>
            <MenuItem value="5m">5 minutes</MenuItem>
            <MenuItem value="15m">15 minutes</MenuItem>
            <MenuItem value="1h">1 hour</MenuItem>
            <MenuItem value="4h">4 hours</MenuItem>
            <MenuItem value="1d">1 day</MenuItem>
          </Select>
        </FormControl>
      </Box>
      
      <Grid container spacing={3}>
        <Grid item xs={12} lg={8}>
          <Item>
            <Typography variant="h6" gutterBottom>Price Chart - {symbol}</Typography>
            <PriceChart symbol={symbol} timeframe={timeframe} />
          </Item>
        </Grid>
        
        <Grid item xs={12} lg={4}>
          <Item>
            <Typography variant="h6" gutterBottom>Order Book</Typography>
            <OrderBook symbol={symbol} />
          </Item>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Item>
            <Typography variant="h6" gutterBottom>Market Indicators</Typography>
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" color="text.secondary">
                24h Volume
              </Typography>
              <Typography variant="h5">
                $1.25B
              </Typography>
            </Box>
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" color="text.secondary">
                24h Change
              </Typography>
              <Typography variant="h5" color="success.main">
                +2.5%
              </Typography>
            </Box>
            <Box>
              <Typography variant="subtitle2" color="text.secondary">
                Volatility (30d)
              </Typography>
              <Typography variant="h5">
                3.8%
              </Typography>
            </Box>
          </Item>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Item>
            <Typography variant="h6" gutterBottom>Recent Trades</Typography>
            <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
              {[...Array(10)].map((_, index) => (
                <Box 
                  key={index} 
                  sx={{ 
                    p: 1, 
                    borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                    color: index % 2 === 0 ? 'success.main' : 'error.main'
                  }}
                >
                  <Grid container>
                    <Grid item xs={4}>
                      <Typography variant="body2">
                        {index % 2 === 0 ? 'Buy' : 'Sell'}
                      </Typography>
                    </Grid>
                    <Grid item xs={4}>
                      <Typography variant="body2">
                        ${(Math.random() * 1000 + 50000).toFixed(2)}
                      </Typography>
                    </Grid>
                    <Grid item xs={4}>
                      <Typography variant="body2">
                        {(Math.random() * 0.1).toFixed(4)} BTC
                      </Typography>
                    </Grid>
                  </Grid>
                </Box>
              ))}
            </Box>
          </Item>
        </Grid>
      </Grid>
    </Box>
  );
}

export default MarketData;
