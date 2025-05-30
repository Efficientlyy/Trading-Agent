import React, { useState } from 'react';
import { 
  Box, 
  ToggleButtonGroup, 
  ToggleButton, 
  Typography, 
  FormControl, 
  Select, 
  MenuItem,
  Grid
} from '@mui/material';
import { 
  Line, 
  LineChart, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  Legend, 
  ResponsiveContainer,
  Area,
  ComposedChart
} from 'recharts';

// Custom tooltip component for chart
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <Box
        sx={{
          backgroundColor: 'background.paper',
          border: '1px solid',
          borderColor: 'divider',
          p: 1,
          borderRadius: 1,
          boxShadow: 1
        }}
      >
        <Typography variant="body2" color="text.primary">
          {new Date(label).toLocaleTimeString()}
        </Typography>
        {payload.map((entry) => (
          <Typography
            key={entry.name}
            variant="body2"
            color={entry.color}
          >
            {`${entry.name}: ${entry.value.toFixed(2)}`}
          </Typography>
        ))}
      </Box>
    );
  }
  return null;
};

function PriceChart({ data }) {
  const [timeframe, setTimeframe] = useState('1h');
  const [symbol, setSymbol] = useState('BTCUSDT');
  
  if (!data || !data.prices || data.prices.length === 0) {
    return (
      <Box sx={{ height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Typography variant="body1" color="text.secondary">
          No market data available
        </Typography>
      </Box>
    );
  }
  
  // Filter data based on selected timeframe and symbol
  const filteredData = data.prices
    .filter(item => item.symbol === symbol)
    .map(item => ({
      timestamp: new Date(item.timestamp).getTime(),
      price: item.price,
      volume: item.volume || 0
    }));
  
  // Calculate min and max for price axis padding
  const prices = filteredData.map(item => item.price);
  const minPrice = Math.min(...prices) * 0.9995;
  const maxPrice = Math.max(...prices) * 1.0005;
  
  return (
    <Box sx={{ width: '100%' }}>
      <Grid container spacing={2} alignItems="center" sx={{ mb: 2 }}>
        <Grid item>
          <FormControl size="small">
            <Select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              variant="outlined"
              sx={{ minWidth: 120 }}
            >
              {data.availableSymbols.map((sym) => (
                <MenuItem key={sym} value={sym}>{sym}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        <Grid item>
          <ToggleButtonGroup
            value={timeframe}
            exclusive
            onChange={(e, newValue) => {
              if (newValue !== null) {
                setTimeframe(newValue);
              }
            }}
            size="small"
          >
            <ToggleButton value="15m">15m</ToggleButton>
            <ToggleButton value="1h">1h</ToggleButton>
            <ToggleButton value="4h">4h</ToggleButton>
            <ToggleButton value="1d">1d</ToggleButton>
          </ToggleButtonGroup>
        </Grid>
        <Grid item xs />
        <Grid item>
          <Typography variant="body2" color="text.secondary">
            Last Updated: {new Date(data.lastUpdated).toLocaleTimeString()}
          </Typography>
        </Grid>
      </Grid>
      
      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart data={filteredData}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
          <XAxis 
            dataKey="timestamp" 
            tickFormatter={(timestamp) => new Date(timestamp).toLocaleTimeString()} 
            domain={['dataMin', 'dataMax']}
            type="number"
            scale="time"
            stroke="#aaa"
          />
          <YAxis 
            dataKey="price" 
            domain={[minPrice, maxPrice]} 
            tickFormatter={(price) => price.toFixed(2)}
            stroke="#aaa" 
            yAxisId="price"
          />
          <YAxis 
            dataKey="volume" 
            orientation="right" 
            stroke="#aaa" 
            yAxisId="volume"
          />
          <RechartsTooltip content={<CustomTooltip />} />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="price" 
            stroke="#2196f3" 
            dot={false} 
            name="Price" 
            yAxisId="price"
            strokeWidth={2}
          />
          <Area
            type="monotone"
            dataKey="volume"
            fill="rgba(102, 187, 106, 0.2)"
            stroke="#66bb6a"
            name="Volume"
            yAxisId="volume"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </Box>
  );
}

export default PriceChart;
