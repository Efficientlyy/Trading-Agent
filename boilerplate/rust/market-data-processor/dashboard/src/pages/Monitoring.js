import React, { useState, useEffect } from 'react';
import { 
  Typography, 
  Grid, 
  Paper, 
  Box, 
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  Chip,
  LinearProgress,
  Button,
  IconButton
} from '@mui/material';
import { styled } from '@mui/material/styles';
import RefreshIcon from '@mui/icons-material/Refresh';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import WarningIcon from '@mui/icons-material/Warning';

const Item = styled(Paper)(({ theme }) => ({
  backgroundColor: theme.palette.background.paper,
  padding: theme.spacing(3),
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[3],
  height: '100%',
}));

// Sample system status data
const mockSystemStatus = {
  overall_status: "OK",
  timestamp: new Date().toISOString(),
  components: {
    market_data: {
      status: "OK",
      last_updated: new Date().toISOString(),
      details: null,
      metrics: {
        connection_latency_ms: 45,
        websocket_reconnects: 0,
        messages_processed: 15234
      }
    },
    paper_trading: {
      status: "OK",
      last_updated: new Date().toISOString(),
      details: null,
      metrics: {
        orders_processed: 127,
        average_fill_time_ms: 125,
        open_positions: 3
      }
    },
    order_execution: {
      status: "DEGRADED",
      last_updated: new Date().toISOString(),
      details: "Increased latency observed",
      metrics: {
        average_execution_time_ms: 350,
        success_rate: 98.5,
        queue_size: 2
      }
    }
  },
  trading_stats: {
    total_trades: 127,
    successful_trades: 124,
    failed_trades: 3,
    total_volume: 5.23,
    current_balance: {
      USDT: 9875.34,
      BTC: 1.05
    },
    profit_loss: 234.56,
    profit_loss_percent: 2.34,
    drawdown: 102.45,
    drawdown_percent: 1.02
  }
};

// Sample log data
const mockLogs = [
  { timestamp: '2025-05-30T04:58:23.123Z', level: 'INFO', message: 'Market data connection established', component: 'market_data' },
  { timestamp: '2025-05-30T04:58:24.456Z', level: 'INFO', message: 'Paper trading service started', component: 'paper_trading' },
  { timestamp: '2025-05-30T04:59:12.789Z', level: 'WARN', message: 'Increased latency in order execution', component: 'order_execution' },
  { timestamp: '2025-05-30T05:01:45.123Z', level: 'INFO', message: 'Order 12345 executed successfully', component: 'paper_trading' },
  { timestamp: '2025-05-30T05:02:22.456Z', level: 'ERROR', message: 'Failed to update market data for ADAUSDT', component: 'market_data' },
  { timestamp: '2025-05-30T05:03:33.789Z', level: 'INFO', message: 'System status check passed', component: 'system' },
  { timestamp: '2025-05-30T05:05:10.123Z', level: 'INFO', message: 'Account balance updated', component: 'paper_trading' },
  { timestamp: '2025-05-30T05:06:45.456Z', level: 'WARN', message: 'API rate limit approaching threshold', component: 'market_data' },
  { timestamp: '2025-05-30T05:07:22.789Z', level: 'INFO', message: 'Order 12346 placed', component: 'paper_trading' },
  { timestamp: '2025-05-30T05:08:33.123Z', level: 'INFO', message: 'Performance metrics updated', component: 'system' },
];

function Monitoring() {
  const [tabValue, setTabValue] = useState(0);
  const [systemStatus, setSystemStatus] = useState(mockSystemStatus);
  const [logs, setLogs] = useState(mockLogs);
  const [loading, setLoading] = useState(false);
  
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };
  
  const handleRefresh = () => {
    setLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      setSystemStatus({
        ...mockSystemStatus,
        timestamp: new Date().toISOString()
      });
      setLoading(false);
    }, 1000);
  };
  
  const getStatusIcon = (status) => {
    switch (status) {
      case 'OK':
        return <CheckCircleIcon color="success" />;
      case 'DEGRADED':
        return <WarningIcon color="warning" />;
      case 'ERROR':
        return <ErrorIcon color="error" />;
      default:
        return null;
    }
  };
  
  const getLogLevelChip = (level) => {
    switch (level) {
      case 'INFO':
        return <Chip size="small" label="INFO" color="info" />;
      case 'WARN':
        return <Chip size="small" label="WARN" color="warning" />;
      case 'ERROR':
        return <Chip size="small" label="ERROR" color="error" />;
      default:
        return <Chip size="small" label={level} />;
    }
  };
  
  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  };
  
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <div>
          <Typography variant="h4" gutterBottom>
            System Monitoring
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Monitor system health, component status, and logs
          </Typography>
        </div>
        <Button 
          variant="outlined" 
          startIcon={<RefreshIcon />}
          onClick={handleRefresh}
          disabled={loading}
        >
          Refresh
        </Button>
      </Box>
      
      {loading && <LinearProgress sx={{ mb: 3 }} />}
      
      <Tabs value={tabValue} onChange={handleTabChange} sx={{ mb: 3 }}>
        <Tab label="System Status" />
        <Tab label="Components" />
        <Tab label="Logs" />
      </Tabs>
      
      {/* System Status Tab */}
      {tabValue === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6} lg={4}>
            <Item>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">Overall Status</Typography>
                <Box sx={{ ml: 'auto' }}>
                  {getStatusIcon(systemStatus.overall_status)}
                </Box>
              </Box>
              
              <Typography variant="subtitle2" color="text.secondary">
                Last Updated
              </Typography>
              <Typography variant="body1" gutterBottom>
                {new Date(systemStatus.timestamp).toLocaleString()}
              </Typography>
              
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  Trading Stats
                </Typography>
                <Box sx={{ mt: 1 }}>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Total Trades
                      </Typography>
                      <Typography variant="h6">
                        {systemStatus.trading_stats.total_trades}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Success Rate
                      </Typography>
                      <Typography variant="h6" color="success.main">
                        {((systemStatus.trading_stats.successful_trades / systemStatus.trading_stats.total_trades) * 100).toFixed(1)}%
                      </Typography>
                    </Grid>
                  </Grid>
                </Box>
              </Box>
            </Item>
          </Grid>
          
          <Grid item xs={12} md={6} lg={4}>
            <Item>
              <Typography variant="h6" gutterBottom>Account Status</Typography>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  Current Balance
                </Typography>
                {Object.entries(systemStatus.trading_stats.current_balance).map(([currency, amount]) => (
                  <Box key={currency} sx={{ mt: 1 }}>
                    <Typography variant="body1">
                      {currency}: {amount}
                    </Typography>
                  </Box>
                ))}
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  Profit/Loss
                </Typography>
                <Typography 
                  variant="h6" 
                  color={systemStatus.trading_stats.profit_loss > 0 ? 'success.main' : 'error.main'}
                >
                  {systemStatus.trading_stats.profit_loss > 0 ? '+' : ''}{systemStatus.trading_stats.profit_loss} USDT 
                  ({systemStatus.trading_stats.profit_loss_percent > 0 ? '+' : ''}{systemStatus.trading_stats.profit_loss_percent}%)
                </Typography>
              </Box>
              
              <Box>
                <Typography variant="subtitle2" color="text.secondary">
                  Max Drawdown
                </Typography>
                <Typography variant="h6" color="error.main">
                  -{systemStatus.trading_stats.drawdown} USDT (-{systemStatus.trading_stats.drawdown_percent}%)
                </Typography>
              </Box>
            </Item>
          </Grid>
          
          <Grid item xs={12} md={12} lg={4}>
            <Item>
              <Typography variant="h6" gutterBottom>Component Status</Typography>
              
              <List>
                {Object.entries(systemStatus.components).map(([name, component]) => (
                  <ListItem 
                    key={name}
                    secondaryAction={getStatusIcon(component.status)}
                  >
                    <ListItemText
                      primary={name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      secondary={component.details || `Last updated: ${formatTimestamp(component.last_updated)}`}
                    />
                  </ListItem>
                ))}
              </List>
            </Item>
          </Grid>
        </Grid>
      )}
      
      {/* Components Tab */}
      {tabValue === 1 && (
        <Grid container spacing={3}>
          {Object.entries(systemStatus.components).map(([name, component]) => (
            <Grid item xs={12} md={6} lg={4} key={name}>
              <Item>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">
                    {name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </Typography>
                  <Box sx={{ ml: 'auto' }}>
                    {getStatusIcon(component.status)}
                  </Box>
                </Box>
                
                {component.details && (
                  <Typography variant="body2" color="text.secondary" paragraph>
                    {component.details}
                  </Typography>
                )}
                
                <Typography variant="subtitle2" color="text.secondary">
                  Metrics
                </Typography>
                
                <Box sx={{ mt: 1 }}>
                  {Object.entries(component.metrics).map(([metricName, value]) => (
                    <Box key={metricName} sx={{ mb: 1 }}>
                      <Typography variant="body2" color="text.secondary">
                        {metricName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </Typography>
                      <Typography variant="body1">
                        {value}
                      </Typography>
                    </Box>
                  ))}
                </Box>
                
                <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
                  Last updated: {new Date(component.last_updated).toLocaleString()}
                </Typography>
              </Item>
            </Grid>
          ))}
        </Grid>
      )}
      
      {/* Logs Tab */}
      {tabValue === 2 && (
        <Item>
          <Typography variant="h6" gutterBottom>System Logs</Typography>
          
          <List sx={{ width: '100%' }}>
            {logs.map((log, index) => (
              <ListItem key={index} divider={index < logs.length - 1}>
                <Grid container spacing={2} alignItems="center">
                  <Grid item xs={12} sm={2}>
                    <Typography variant="caption" color="text.secondary">
                      {formatTimestamp(log.timestamp)}
                    </Typography>
                  </Grid>
                  <Grid item xs={6} sm={1}>
                    {getLogLevelChip(log.level)}
                  </Grid>
                  <Grid item xs={6} sm={2}>
                    <Chip 
                      size="small" 
                      label={log.component} 
                      variant="outlined"
                    />
                  </Grid>
                  <Grid item xs={12} sm={7}>
                    <Typography variant="body2">
                      {log.message}
                    </Typography>
                  </Grid>
                </Grid>
              </ListItem>
            ))}
          </List>
          
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
            <Button variant="outlined">Load More Logs</Button>
          </Box>
        </Item>
      )}
    </Box>
  );
}

export default Monitoring;
