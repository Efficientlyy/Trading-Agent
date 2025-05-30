import React from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Grid, 
  Divider, 
  Box,
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  LinearProgress
} from '@mui/material';
import { 
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  AccountBalance as AccountBalanceIcon
} from '@mui/icons-material';

const formatCurrency = (value) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(value);
};

const formatCrypto = (value, symbol) => {
  return `${value.toFixed(8)} ${symbol}`;
};

function PortfolioSummary({ data }) {
  // If no data is provided, return empty state
  if (!data) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Portfolio Summary
          </Typography>
          <Divider sx={{ mb: 2 }} />
          <Typography variant="body2" color="text.secondary">
            No portfolio data available
          </Typography>
        </CardContent>
      </Card>
    );
  }

  // Extract relevant data
  const { balances, totalValue, pnl, pnlPercentage } = data;
  const isProfitable = pnl >= 0;

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Portfolio Summary
        </Typography>
        <Divider sx={{ mb: 2 }} />
        
        {/* Total Value */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Total Portfolio Value
          </Typography>
          <Typography variant="h4">
            {formatCurrency(totalValue)}
          </Typography>
          
          {/* P&L Information */}
          <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
            {isProfitable ? (
              <TrendingUpIcon color="success" sx={{ mr: 1 }} />
            ) : (
              <TrendingDownIcon color="error" sx={{ mr: 1 }} />
            )}
            <Typography 
              variant="body1" 
              color={isProfitable ? 'success.main' : 'error.main'}
            >
              {formatCurrency(pnl)} ({pnlPercentage.toFixed(2)}%)
            </Typography>
          </Box>
        </Box>
        
        {/* Asset List */}
        <Typography variant="subtitle2" gutterBottom>
          Asset Allocation
        </Typography>
        <List sx={{ width: '100%' }}>
          {balances.map((balance) => (
            <ListItem key={balance.asset} sx={{ px: 0 }}>
              <ListItemAvatar>
                <Avatar alt={balance.asset}>
                  {balance.asset.substring(0, 2)}
                </Avatar>
              </ListItemAvatar>
              <ListItemText
                primary={
                  <Typography variant="body1">
                    {balance.asset}
                  </Typography>
                }
                secondary={
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">
                      {formatCrypto(balance.free, balance.asset)}
                    </Typography>
                    <Typography variant="body2" color="text.primary">
                      {formatCurrency(balance.usdValue)}
                    </Typography>
                  </Box>
                }
              />
            </ListItem>
          ))}
        </List>
        
        {/* Allocation Chart (Linear Progress bars) */}
        <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
          Portfolio Distribution
        </Typography>
        <Grid container spacing={1}>
          {balances.map((balance) => {
            // Calculate percentage of total portfolio
            const percentage = (balance.usdValue / totalValue) * 100;
            
            return (
              <Grid item xs={12} key={`${balance.asset}-progress`}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Typography variant="body2" sx={{ minWidth: 60 }}>
                    {balance.asset}
                  </Typography>
                  <Box sx={{ width: '100%', mr: 1 }}>
                    <LinearProgress 
                      variant="determinate" 
                      value={percentage} 
                      color={balance.asset === 'USDT' ? 'info' : 'primary'}
                      sx={{ height: 8, borderRadius: 5 }}
                    />
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {percentage.toFixed(1)}%
                  </Typography>
                </Box>
              </Grid>
            );
          })}
        </Grid>
      </CardContent>
    </Card>
  );
}

export default PortfolioSummary;
