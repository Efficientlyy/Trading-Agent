// Connection status component for displaying WebSocket connection status
import React from 'react';
import { Box, Typography, Chip } from '@mui/material';
import { useWebSocket } from '../services/websocketService';

const ConnectionStatus = () => {
  const { connected } = useWebSocket();
  
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', ml: 2 }}>
      <Chip
        size="small"
        label={connected ? "Connected" : "Disconnected"}
        color={connected ? "success" : "error"}
        variant="outlined"
        sx={{ mr: 1 }}
      />
      <Typography variant="body2" color="text.secondary">
        {connected ? "Live Data" : "Reconnecting..."}
      </Typography>
    </Box>
  );
};

export default ConnectionStatus;
