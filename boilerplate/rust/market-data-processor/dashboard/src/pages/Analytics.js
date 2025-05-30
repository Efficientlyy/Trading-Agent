import React from 'react';
import { Typography, Grid, Paper, Box } from '@mui/material';
import { styled } from '@mui/material/styles';
import PerformanceChart from '../components/PerformanceChart';
import PnLAnalysis from '../components/PnLAnalysis';

const Item = styled(Paper)(({ theme }) => ({
  backgroundColor: theme.palette.background.paper,
  padding: theme.spacing(3),
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[3],
  height: '100%',
}));

function Analytics() {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Analytics
      </Typography>
      <Typography variant="subtitle1" color="text.secondary" paragraph>
        Track your trading performance and analyze your strategy effectiveness
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Item>
            <Typography variant="h6" gutterBottom>Performance Over Time</Typography>
            <PerformanceChart />
          </Item>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Item>
            <Typography variant="h6" gutterBottom>Key Metrics</Typography>
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" color="text.secondary">
                Win Rate
              </Typography>
              <Typography variant="h4" color="success.main">
                68.5%
              </Typography>
            </Box>
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" color="text.secondary">
                Average Return
              </Typography>
              <Typography variant="h4" color="success.main">
                2.3%
              </Typography>
            </Box>
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" color="text.secondary">
                Sharpe Ratio
              </Typography>
              <Typography variant="h4">
                1.75
              </Typography>
            </Box>
            <Box>
              <Typography variant="subtitle2" color="text.secondary">
                Max Drawdown
              </Typography>
              <Typography variant="h4" color="error.main">
                -7.2%
              </Typography>
            </Box>
          </Item>
        </Grid>
        
        <Grid item xs={12}>
          <Item>
            <Typography variant="h6" gutterBottom>Profit and Loss Analysis</Typography>
            <PnLAnalysis />
          </Item>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Analytics;
