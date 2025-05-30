import React from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Divider, 
  Box,
  Grid,
  Tooltip
} from '@mui/material';
import { 
  ShowChart as ShowChartIcon,
  Timeline as TimelineIcon,
  TrendingDown as TrendingDownIcon,
  BarChart as BarChartIcon
} from '@mui/icons-material';

function MetricItem({ icon, title, value, suffix, color, tooltip }) {
  return (
    <Tooltip title={tooltip} arrow placement="top">
      <Grid item xs={6}>
        <Box sx={{ display: 'flex', flexDirection: 'column', p: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            {icon}
            <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
              {title}
            </Typography>
          </Box>
          <Typography 
            variant="h6" 
            color={color || 'text.primary'}
            sx={{ fontWeight: 'medium' }}
          >
            {value}{suffix}
          </Typography>
        </Box>
      </Grid>
    </Tooltip>
  );
}

function PerformanceMetrics({ data }) {
  // If no data is provided, return empty state
  if (!data) {
    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Performance Metrics
          </Typography>
          <Divider sx={{ mb: 2 }} />
          <Typography variant="body2" color="text.secondary">
            No performance data available
          </Typography>
        </CardContent>
      </Card>
    );
  }

  // Extract metrics data
  const { 
    profitLoss, 
    winRate, 
    maxDrawdown, 
    sharpeRatio,
    totalTrades,
    successfulTrades,
    averageProfitPerTrade,
    averageTradeTime
  } = data;

  const isProfitable = profitLoss >= 0;

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Performance Metrics
        </Typography>
        <Divider sx={{ mb: 2 }} />
        
        <Grid container spacing={2}>
          {/* Profit/Loss */}
          <MetricItem 
            icon={<ShowChartIcon color={isProfitable ? "success" : "error"} />}
            title="Profit/Loss"
            value={isProfitable ? `+${profitLoss.toFixed(2)}` : profitLoss.toFixed(2)}
            suffix="%"
            color={isProfitable ? "success.main" : "error.main"}
            tooltip="Overall profit or loss percentage since inception"
          />
          
          {/* Win Rate */}
          <MetricItem 
            icon={<TimelineIcon color="primary" />}
            title="Win Rate"
            value={winRate.toFixed(1)}
            suffix="%"
            tooltip="Percentage of trades that resulted in profit"
          />
          
          {/* Maximum Drawdown */}
          <MetricItem 
            icon={<TrendingDownIcon color="warning" />}
            title="Max Drawdown"
            value={maxDrawdown.toFixed(2)}
            suffix="%"
            color="warning.main"
            tooltip="Largest percentage drop from peak to trough"
          />
          
          {/* Sharpe Ratio */}
          <MetricItem 
            icon={<BarChartIcon color="info" />}
            title="Sharpe Ratio"
            value={sharpeRatio.toFixed(2)}
            suffix=""
            tooltip="Risk-adjusted return measure (higher is better)"
          />
        </Grid>
        
        <Divider sx={{ my: 2 }} />
        
        {/* Trade Statistics */}
        <Typography variant="subtitle2" gutterBottom>
          Trade Statistics
        </Typography>
        
        <Grid container spacing={1} sx={{ mt: 1 }}>
          <Grid item xs={6}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="text.secondary">
                Total Trades:
              </Typography>
              <Typography variant="body2">
                {totalTrades}
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={6}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="text.secondary">
                Successful:
              </Typography>
              <Typography variant="body2">
                {successfulTrades}
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={6}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="text.secondary">
                Avg Profit:
              </Typography>
              <Typography variant="body2">
                {averageProfitPerTrade > 0 ? '+' : ''}{averageProfitPerTrade.toFixed(2)}%
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={6}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="text.secondary">
                Avg Duration:
              </Typography>
              <Typography variant="body2">
                {averageTradeTime}m
              </Typography>
            </Box>
          </Grid>
        </Grid>
        
      </CardContent>
    </Card>
  );
}

export default PerformanceMetrics;
