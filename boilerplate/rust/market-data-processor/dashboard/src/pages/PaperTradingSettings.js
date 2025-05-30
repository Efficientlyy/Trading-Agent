import React, { useState } from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Divider, 
  Box,
  Grid,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  CircularProgress,
  Chip,
  Stack
} from '@mui/material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { fetchPaperTradingSettings, updatePaperTradingSettings, resetPaperTradingAccount } from '../services/apiService';

function PaperTradingSettings() {
  const queryClient = useQueryClient();
  
  // State for reset confirmation dialog
  const [resetDialogOpen, setResetDialogOpen] = useState(false);
  
  // Fetch current settings
  const { 
    data: settings, 
    isLoading, 
    error,
    refetch
  } = useQuery('paperTradingSettings', fetchPaperTradingSettings);
  
  // Local state for form values
  const [formValues, setFormValues] = useState({
    initialBalances: {
      USDT: 10000,
      BTC: 0.5,
      ETH: 5
    },
    tradingPairs: ['BTCUSDT', 'ETHUSDT'],
    maxPositionSize: 1.0,
    defaultOrderSize: 0.1,
    maxDrawdownPercent: 10,
    slippageModel: 'REALISTIC',
    latencyModel: 'NORMAL',
    tradingFees: 0.001
  });
  
  // Update form values when settings are loaded
  React.useEffect(() => {
    if (settings) {
      setFormValues(settings);
    }
  }, [settings]);
  
  // Update settings mutation
  const updateMutation = useMutation(updatePaperTradingSettings, {
    onSuccess: () => {
      queryClient.invalidateQueries('paperTradingSettings');
      // You might want to show a success notification here
    }
  });
  
  // Reset account mutation
  const resetMutation = useMutation(resetPaperTradingAccount, {
    onSuccess: () => {
      queryClient.invalidateQueries('paperTradingSettings');
      queryClient.invalidateQueries('accountData');
      setResetDialogOpen(false);
      // You might want to show a success notification here
    }
  });
  
  // Handle form changes
  const handleChange = (path, value) => {
    const newFormValues = { ...formValues };
    
    if (path.includes('.')) {
      const [parent, child] = path.split('.');
      newFormValues[parent][child] = value;
    } else {
      newFormValues[path] = value;
    }
    
    setFormValues(newFormValues);
  };
  
  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    updateMutation.mutate(formValues);
  };
  
  // Handle account reset
  const handleReset = () => {
    resetMutation.mutate();
  };
  
  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 10 }}>
        <CircularProgress />
      </Box>
    );
  }
  
  if (error) {
    return (
      <Box sx={{ textAlign: 'center', mt: 10 }}>
        <Typography variant="h5" color="error" gutterBottom>
          Error loading settings
        </Typography>
        <Button variant="contained" onClick={() => refetch()}>
          Retry
        </Button>
      </Box>
    );
  }

  return (
    <>
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Paper Trading Settings
          </Typography>
          <Divider sx={{ mb: 3 }} />
          
          {updateMutation.isError && (
            <Alert severity="error" sx={{ mb: 3 }}>
              Failed to update settings. Please try again.
            </Alert>
          )}
          
          {updateMutation.isSuccess && (
            <Alert severity="success" sx={{ mb: 3 }}>
              Settings updated successfully.
            </Alert>
          )}
          
          <form onSubmit={handleSubmit}>
            <Grid container spacing={3}>
              {/* Initial Balances */}
              <Grid item xs={12}>
                <Typography variant="subtitle1" gutterBottom>
                  Initial Balances
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={4}>
                    <TextField
                      label="USDT"
                      type="number"
                      fullWidth
                      value={formValues.initialBalances.USDT}
                      onChange={(e) => handleChange('initialBalances.USDT', parseFloat(e.target.value))}
                      InputProps={{
                        inputProps: { min: 0, step: 100 }
                      }}
                    />
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <TextField
                      label="BTC"
                      type="number"
                      fullWidth
                      value={formValues.initialBalances.BTC}
                      onChange={(e) => handleChange('initialBalances.BTC', parseFloat(e.target.value))}
                      InputProps={{
                        inputProps: { min: 0, step: 0.01 }
                      }}
                    />
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <TextField
                      label="ETH"
                      type="number"
                      fullWidth
                      value={formValues.initialBalances.ETH}
                      onChange={(e) => handleChange('initialBalances.ETH', parseFloat(e.target.value))}
                      InputProps={{
                        inputProps: { min: 0, step: 0.1 }
                      }}
                    />
                  </Grid>
                </Grid>
              </Grid>
              
              {/* Trading Pairs */}
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Trading Pairs</InputLabel>
                  <Select
                    multiple
                    value={formValues.tradingPairs}
                    onChange={(e) => handleChange('tradingPairs', e.target.value)}
                    renderValue={(selected) => (
                      <Stack direction="row" spacing={0.5} flexWrap="wrap">
                        {selected.map((value) => (
                          <Chip key={value} label={value} size="small" />
                        ))}
                      </Stack>
                    )}
                  >
                    <MenuItem value="BTCUSDT">BTCUSDT</MenuItem>
                    <MenuItem value="ETHUSDT">ETHUSDT</MenuItem>
                    <MenuItem value="DOGEUSDT">DOGEUSDT</MenuItem>
                    <MenuItem value="ADAUSDT">ADAUSDT</MenuItem>
                    <MenuItem value="SOLUSDT">SOLUSDT</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              {/* Trading Fees */}
              <Grid item xs={12} sm={6}>
                <TextField
                  label="Trading Fees (%)"
                  type="number"
                  fullWidth
                  value={formValues.tradingFees * 100}
                  onChange={(e) => handleChange('tradingFees', parseFloat(e.target.value) / 100)}
                  InputProps={{
                    inputProps: { min: 0, max: 1, step: 0.01 }
                  }}
                  helperText="Trading fee percentage (e.g., 0.1%)"
                />
              </Grid>
              
              {/* Position Size Slider */}
              <Grid item xs={12} sm={6}>
                <Typography id="max-position-size-slider" gutterBottom>
                  Max Position Size: {(formValues.maxPositionSize * 100).toFixed(0)}%
                </Typography>
                <Slider
                  value={formValues.maxPositionSize * 100}
                  onChange={(e, newValue) => handleChange('maxPositionSize', newValue / 100)}
                  aria-labelledby="max-position-size-slider"
                  valueLabelDisplay="auto"
                  step={5}
                  marks
                  min={5}
                  max={100}
                />
                <Typography variant="caption" color="text.secondary">
                  Maximum percentage of available balance for a single position
                </Typography>
              </Grid>
              
              {/* Order Size Slider */}
              <Grid item xs={12} sm={6}>
                <Typography id="default-order-size-slider" gutterBottom>
                  Default Order Size: {(formValues.defaultOrderSize * 100).toFixed(0)}%
                </Typography>
                <Slider
                  value={formValues.defaultOrderSize * 100}
                  onChange={(e, newValue) => handleChange('defaultOrderSize', newValue / 100)}
                  aria-labelledby="default-order-size-slider"
                  valueLabelDisplay="auto"
                  step={5}
                  marks
                  min={5}
                  max={50}
                />
                <Typography variant="caption" color="text.secondary">
                  Default order size as percentage of available balance
                </Typography>
              </Grid>
              
              {/* Max Drawdown */}
              <Grid item xs={12} sm={6}>
                <Typography id="max-drawdown-slider" gutterBottom>
                  Max Drawdown: {formValues.maxDrawdownPercent}%
                </Typography>
                <Slider
                  value={formValues.maxDrawdownPercent}
                  onChange={(e, newValue) => handleChange('maxDrawdownPercent', newValue)}
                  aria-labelledby="max-drawdown-slider"
                  valueLabelDisplay="auto"
                  step={1}
                  marks
                  min={5}
                  max={30}
                />
                <Typography variant="caption" color="text.secondary">
                  Maximum allowed drawdown before trading is paused
                </Typography>
              </Grid>
              
              {/* Slippage Model */}
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Slippage Model</InputLabel>
                  <Select
                    value={formValues.slippageModel}
                    onChange={(e) => handleChange('slippageModel', e.target.value)}
                  >
                    <MenuItem value="NONE">None (Perfect Execution)</MenuItem>
                    <MenuItem value="MINIMAL">Minimal (0.01-0.05%)</MenuItem>
                    <MenuItem value="REALISTIC">Realistic (0.05-0.2%)</MenuItem>
                    <MenuItem value="HIGH">High (0.2-1%)</MenuItem>
                  </Select>
                  <Typography variant="caption" color="text.secondary">
                    Simulated price slippage for order execution
                  </Typography>
                </FormControl>
              </Grid>
              
              {/* Latency Model */}
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Latency Model</InputLabel>
                  <Select
                    value={formValues.latencyModel}
                    onChange={(e) => handleChange('latencyModel', e.target.value)}
                  >
                    <MenuItem value="NONE">None (Instant Execution)</MenuItem>
                    <MenuItem value="LOW">Low (50-200ms)</MenuItem>
                    <MenuItem value="NORMAL">Normal (200-500ms)</MenuItem>
                    <MenuItem value="HIGH">High (500-1000ms)</MenuItem>
                  </Select>
                  <Typography variant="caption" color="text.secondary">
                    Simulated network latency for order execution
                  </Typography>
                </FormControl>
              </Grid>
              
              {/* Action Buttons */}
              <Grid item xs={12}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
                  <Button 
                    variant="outlined" 
                    color="error"
                    onClick={() => setResetDialogOpen(true)}
                  >
                    Reset Account
                  </Button>
                  
                  <Button 
                    type="submit" 
                    variant="contained"
                    disabled={updateMutation.isLoading}
                  >
                    {updateMutation.isLoading ? <CircularProgress size={24} /> : 'Save Settings'}
                  </Button>
                </Box>
              </Grid>
            </Grid>
          </form>
        </CardContent>
      </Card>
      
      {/* Reset Confirmation Dialog */}
      <Dialog
        open={resetDialogOpen}
        onClose={() => setResetDialogOpen(false)}
      >
        <DialogTitle>
          Reset Paper Trading Account?
        </DialogTitle>
        <DialogContent>
          <DialogContentText>
            This will reset your paper trading account to the initial state defined in your settings.
            All open orders will be canceled, all positions will be closed, and your balance will be reset.
            This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResetDialogOpen(false)}>
            Cancel
          </Button>
          <Button 
            onClick={handleReset} 
            color="error" 
            variant="contained"
            disabled={resetMutation.isLoading}
          >
            {resetMutation.isLoading ? <CircularProgress size={24} /> : 'Reset Account'}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

export default PaperTradingSettings;
