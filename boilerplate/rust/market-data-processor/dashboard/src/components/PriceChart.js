// Frontend React component for BTC/USDC chart
import React, { useEffect, useRef, useState } from 'react';
import { createChart, CrosshairMode } from 'lightweight-charts';
import { Box, Typography, CircularProgress, Paper } from '@mui/material';
import { fetchHistoricalData, subscribeToTickerUpdates } from '../services/apiService';

const PriceChart = ({ symbol = 'BTCUSDC', timeframe = '1h' }) => {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const candleSeries = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastPrice, setLastPrice] = useState(null);

  useEffect(() => {
    // Create chart instance
    if (chartContainerRef.current) {
      // Initialize chart
      chartRef.current = createChart(chartContainerRef.current, {
        width: chartContainerRef.current.clientWidth,
        height: 400,
        layout: {
          backgroundColor: '#131722',
          textColor: '#d1d4dc',
        },
        grid: {
          vertLines: {
            color: 'rgba(42, 46, 57, 0.5)',
          },
          horzLines: {
            color: 'rgba(42, 46, 57, 0.5)',
          },
        },
        crosshair: {
          mode: CrosshairMode.Normal,
        },
        priceScale: {
          borderColor: 'rgba(197, 203, 206, 0.8)',
        },
        timeScale: {
          borderColor: 'rgba(197, 203, 206, 0.8)',
          timeVisible: true,
        },
      });

      // Add candlestick series
      candleSeries.current = chartRef.current.addCandlestickSeries({
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
      });

      // Handle window resize
      const handleResize = () => {
        if (chartRef.current) {
          chartRef.current.applyOptions({
            width: chartContainerRef.current.clientWidth,
          });
        }
      };

      window.addEventListener('resize', handleResize);

      // Load historical data
      const loadData = async () => {
        try {
          setIsLoading(true);
          const data = await fetchHistoricalData(symbol, timeframe);
          
          if (data && data.length > 0) {
            candleSeries.current.setData(data);
            setLastPrice(data[data.length - 1].close);
          }
          
          setIsLoading(false);
        } catch (err) {
          console.error('Failed to load chart data:', err);
          setError('Failed to load chart data. Please try again later.');
          setIsLoading(false);
        }
      };

      loadData();

      // Subscribe to real-time updates
      const unsubscribe = subscribeToTickerUpdates(symbol, (tickerData) => {
        if (candleSeries.current && tickerData) {
          // Update last candle or add new one based on timestamp
          const lastPrice = parseFloat(tickerData.price);
          setLastPrice(lastPrice);
          
          // Update the chart with new price data
          const currentTime = Math.floor(Date.now() / 1000) * 1000;
          candleSeries.current.update({
            time: currentTime / 1000,
            open: lastPrice,
            high: lastPrice,
            low: lastPrice,
            close: lastPrice,
          });
        }
      });

      // Cleanup
      return () => {
        window.removeEventListener('resize', handleResize);
        if (chartRef.current) {
          chartRef.current.remove();
        }
        unsubscribe();
      };
    }
  }, [symbol, timeframe]);

  return (
    <Paper elevation={3} sx={{ p: 2, borderRadius: 2, backgroundColor: '#1E1E1E' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" color="text.primary">
          {symbol}
        </Typography>
        {lastPrice && (
          <Typography variant="h6" color="text.primary">
            ${parseFloat(lastPrice).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </Typography>
        )}
      </Box>
      
      {isLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
          <CircularProgress />
        </Box>
      ) : error ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
          <Typography color="error">{error}</Typography>
        </Box>
      ) : (
        <div ref={chartContainerRef} style={{ height: 400 }} />
      )}
    </Paper>
  );
};

export default PriceChart;
