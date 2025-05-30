import React from 'react';
import { 
  Grid, 
  Card, 
  CardContent, 
  Typography, 
  Divider,
  Box,
  Button,
  CircularProgress
} from '@mui/material';
import { useQuery } from 'react-query';
import PortfolioSummary from '../components/PortfolioSummary';
import PerformanceMetrics from '../components/PerformanceMetrics';
import RecentTrades from '../components/RecentTrades';
import PriceChart from '../components/PriceChart';
import ActiveOrders from '../components/ActiveOrders';
import { fetchAccountData, fetchMarketData, fetchRecentTrades } from '../services/apiService';

function Dashboard() {
  const { 
    data: accountData, 
    isLoading: isLoadingAccount, 
    error: accountError 
  } = useQuery('accountData', fetchAccountData, {
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  const { 
    data: marketData, 
    isLoading: isLoadingMarket,
    error: marketError 
  } = useQuery('marketData', fetchMarketData, {
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  const { 
    data: recentTrades, 
    isLoading: isLoadingTrades,
    error: tradesError 
  } = useQuery('recentTrades', fetchRecentTrades, {
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  const isLoading = isLoadingAccount || isLoadingMarket || isLoadingTrades;
  const hasError = accountError || marketError || tradesError;

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 10 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (hasError) {
    return (
      <Box sx={{ textAlign: 'center', mt: 10 }}>
        <Typography variant="h5" color="error" gutterBottom>
          Error loading dashboard data
        </Typography>
        <Button variant="contained" onClick={() => window.location.reload()}>
          Retry
        </Button>
      </Box>
    );
  }

  return (
    <Grid container spacing={3}>
      {/* Portfolio Overview */}
      <Grid item xs={12} md={8}>
        <PortfolioSummary data={accountData} />
      </Grid>
      
      {/* Performance Metrics */}
      <Grid item xs={12} md={4}>
        <PerformanceMetrics data={accountData?.performance} />
      </Grid>
      
      {/* Price Chart */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Market Data
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <PriceChart data={marketData} />
          </CardContent>
        </Card>
      </Grid>
      
      {/* Active Orders */}
      <Grid item xs={12} md={6}>
        <ActiveOrders orders={accountData?.activeOrders || []} />
      </Grid>
      
      {/* Recent Trades */}
      <Grid item xs={12} md={6}>
        <RecentTrades trades={recentTrades || []} />
      </Grid>
    </Grid>
  );
}

export default Dashboard;
