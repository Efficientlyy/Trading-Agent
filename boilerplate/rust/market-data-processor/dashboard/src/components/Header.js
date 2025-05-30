// Header component for navigation
import React from 'react';
import { AppBar, Toolbar, Typography, Box, Button } from '@mui/material';
import { Link } from 'react-router-dom';
import ConnectionStatus from './ConnectionStatus';

const Header = () => {
  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          MEXC Trading System
        </Typography>
        <Box>
          <Button color="inherit" component={Link} to="/">
            Trading Dashboard
          </Button>
        </Box>
        <ConnectionStatus />
        <Typography variant="body2" sx={{ ml: 2 }}>
          BTC/USDC
        </Typography>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
