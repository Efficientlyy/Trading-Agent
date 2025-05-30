import React from 'react';
import { Box, Grid, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Sample P&L data by trading pair
const pnlData = [
  { pair: 'BTC/USDT', profit: 850, loss: -320, trades: 42 },
  { pair: 'ETH/USDT', profit: 620, loss: -180, trades: 35 },
  { pair: 'XRP/USDT', profit: 120, loss: -75, trades: 28 },
  { pair: 'ADA/USDT', profit: 95, loss: -60, trades: 22 },
  { pair: 'DOT/USDT', profit: 210, loss: -90, trades: 18 },
];

function PnLAnalysis() {
  return (
    <Grid container spacing={3}>
      <Grid item xs={12} lg={8}>
        <Box sx={{ width: '100%', height: 400 }}>
          <ResponsiveContainer>
            <BarChart
              data={pnlData}
              margin={{
                top: 20,
                right: 30,
                left: 20,
                bottom: 5,
              }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="pair" />
              <YAxis />
              <Tooltip 
                formatter={(value) => [`$${Math.abs(value)}`, value >= 0 ? 'Profit' : 'Loss']}
              />
              <Legend />
              <Bar dataKey="profit" name="Profit" fill="#4caf50" />
              <Bar dataKey="loss" name="Loss" fill="#f44336" />
            </BarChart>
          </ResponsiveContainer>
        </Box>
      </Grid>
      
      <Grid item xs={12} lg={4}>
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-around' }}>
          <Box>
            <Typography variant="subtitle2" color="text.secondary">
              Most Profitable Pair
            </Typography>
            <Typography variant="h5" color="success.main">
              BTC/USDT
            </Typography>
            <Typography variant="body2">
              $850 profit from 42 trades
            </Typography>
          </Box>
          
          <Box>
            <Typography variant="subtitle2" color="text.secondary">
              Total Profit
            </Typography>
            <Typography variant="h5" color="success.main">
              $1,895
            </Typography>
          </Box>
          
          <Box>
            <Typography variant="subtitle2" color="text.secondary">
              Total Loss
            </Typography>
            <Typography variant="h5" color="error.main">
              -$725
            </Typography>
          </Box>
          
          <Box>
            <Typography variant="subtitle2" color="text.secondary">
              Net P&L
            </Typography>
            <Typography variant="h5" color="success.main">
              $1,170
            </Typography>
          </Box>
          
          <Box>
            <Typography variant="subtitle2" color="text.secondary">
              Average P&L Per Trade
            </Typography>
            <Typography variant="h5">
              $8.12
            </Typography>
          </Box>
        </Box>
      </Grid>
    </Grid>
  );
}

export default PnLAnalysis;
