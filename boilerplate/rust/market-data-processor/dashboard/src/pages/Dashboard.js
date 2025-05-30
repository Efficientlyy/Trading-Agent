// Main Dashboard page for BTC/USDC trading
import React from 'react';
import { Box, Grid, Container, Typography } from '@mui/material';
import PriceChart from '../components/PriceChart';
import OrderBook from '../components/OrderBook';
import TradeHistory from '../components/TradeHistory';
import PaperTrading from '../components/PaperTrading';

const Dashboard = () => {
  const symbol = 'BTCUSDC';

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" sx={{ mb: 4 }}>
        MEXC Trading Dashboard
      </Typography>
      
      <Grid container spacing={3}>
        {/* Price Chart */}
        <Grid item xs={12} lg={8}>
          <PriceChart symbol={symbol} />
        </Grid>
        
        {/* Paper Trading */}
        <Grid item xs={12} lg={4}>
          <PaperTrading symbol={symbol} />
        </Grid>
        
        {/* Order Book */}
        <Grid item xs={12} md={6}>
          <OrderBook symbol={symbol} />
        </Grid>
        
        {/* Trade History */}
        <Grid item xs={12} md={6}>
          <TradeHistory symbol={symbol} />
        </Grid>
      </Grid>
    </Container>
  );
};

export default Dashboard;
