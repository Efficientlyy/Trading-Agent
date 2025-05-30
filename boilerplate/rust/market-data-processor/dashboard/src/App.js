import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// Page components
import Dashboard from './pages/Dashboard';
import OrderHistory from './pages/OrderHistory';
import PaperTradingSettings from './pages/PaperTradingSettings';
import Trading from './pages/Trading';
import Analytics from './pages/Analytics';
import MarketData from './pages/MarketData';
import Monitoring from './pages/Monitoring';

// Layout component
import Layout from './components/Layout';

// Create a dark theme instance
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#f50057',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
    success: {
      main: '#4caf50',
    },
    error: {
      main: '#f44336',
    },
    warning: {
      main: '#ff9800',
    },
    info: {
      main: '#2196f3',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 500,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 500,
    },
    h3: {
      fontSize: '1.8rem',
      fontWeight: 500,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 500,
    },
    h5: {
      fontSize: '1.3rem',
      fontWeight: 500,
    },
    h6: {
      fontSize: '1.1rem',
      fontWeight: 500,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          padding: 16,
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Layout>
        <Routes>
          {/* Dashboard/Overview */}
          <Route path="/" element={<Dashboard />} />
          
          {/* Trading Section */}
          <Route path="/trading" element={<Trading />} />
          
          {/* Analytics Section */}
          <Route path="/analytics" element={<Analytics />} />
          
          {/* Market Data Section */}
          <Route path="/market-data" element={<MarketData />} />
          
          {/* History Section */}
          <Route path="/order-history" element={<OrderHistory />} />
          <Route path="/trade-history" element={<OrderHistory />} /> {/* Reusing OrderHistory for now */}
          
          {/* Settings Section */}
          <Route path="/settings" element={<PaperTradingSettings />} />
          
          {/* Monitoring Section */}
          <Route path="/monitoring" element={<Monitoring />} />
          
          {/* Fallback route for any undefined paths */}
          <Route path="*" element={<Dashboard />} />
        </Routes>
      </Layout>
    </ThemeProvider>
  );
}

export default App;
