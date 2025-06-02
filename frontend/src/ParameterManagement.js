// Parameter Management Frontend
// This file implements the React-based frontend for the Trading-Agent parameter management system

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Container, Box, Typography, Tabs, Tab, Paper, Grid, Slider, Switch,
  FormControl, InputLabel, Select, MenuItem, TextField, Button,
  Chip, Divider, Alert, CircularProgress, Accordion, AccordionSummary,
  AccordionDetails, Tooltip, Dialog, DialogTitle, DialogContent,
  DialogActions, Snackbar
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import InfoIcon from '@mui/icons-material/Info';
import WarningIcon from '@mui/icons-material/Warning';
import SaveIcon from '@mui/icons-material/Save';
import RestoreIcon from '@mui/icons-material/Restore';
import SettingsIcon from '@mui/icons-material/Settings';
import SecurityIcon from '@mui/icons-material/Security';
import TimelineIcon from '@mui/icons-material/Timeline';
import SpeedIcon from '@mui/icons-material/Speed';
import MonitorIcon from '@mui/icons-material/Monitor';
import StorageIcon from '@mui/icons-material/Storage';
import TuneIcon from '@mui/icons-material/Tune';

// API base URL
const API_BASE_URL = 'http://localhost:5001/api';

// Parameter categories
const CATEGORIES = {
  BASIC: 'basic',
  ADVANCED: 'advanced',
  EXPERT: 'expert'
};

// Module icons
const MODULE_ICONS = {
  'market_data': <StorageIcon />,
  'pattern_recognition': <TimelineIcon />,
  'signal_generation': <TuneIcon />,
  'decision_making': <TuneIcon />,
  'order_execution': <TuneIcon />,
  'risk_management': <SecurityIcon />,
  'visualization': <MonitorIcon />,
  'monitoring_dashboard': <MonitorIcon />,
  'performance_optimization': <SpeedIcon />,
  'system_settings': <SettingsIcon />
};

// Main App Component
function ParameterManagementApp() {
  // State
  const [loading, setLoading] = useState(true);
  const [parameters, setParameters] = useState({});
  const [metadata, setMetadata] = useState({});
  const [presets, setPresets] = useState({});
  const [currentModule, setCurrentModule] = useState('market_data');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showExpert, setShowExpert] = useState(false);
  const [validationErrors, setValidationErrors] = useState({});
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });
  const [confirmDialog, setConfirmDialog] = useState({ open: false, title: '', message: '', onConfirm: null });
  const [savePresetDialog, setSavePresetDialog] = useState(false);
  const [customPresetName, setCustomPresetName] = useState('');
  const [customPresetDescription, setCustomPresetDescription] = useState('');

  // Fetch data on component mount
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        // Fetch all data in parallel
        const [parametersRes, metadataRes, presetsRes] = await Promise.all([
          axios.get(`${API_BASE_URL}/parameters`),
          axios.get(`${API_BASE_URL}/parameters/metadata`),
          axios.get(`${API_BASE_URL}/parameters/presets`)
        ]);

        setParameters(parametersRes.data);
        setMetadata(metadataRes.data);
        setPresets(presetsRes.data);
      } catch (error) {
        console.error('Error fetching data:', error);
        setSnackbar({
          open: true,
          message: 'Failed to load parameters. Please try again.',
          severity: 'error'
        });
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Handle module change
  const handleModuleChange = (event, newModule) => {
    setCurrentModule(newModule);
  };

  // Handle parameter change
  const handleParameterChange = (paramName, value) => {
    setParameters(prevParams => ({
      ...prevParams,
      [currentModule]: {
        ...prevParams[currentModule],
        [paramName]: value
      }
    }));
  };

  // Save parameters for current module
  const saveParameters = async () => {
    try {
      setLoading(true);
      
      // Validate parameters first
      const validationRes = await axios.post(`${API_BASE_URL}/parameters/validate`, {
        module: currentModule,
        parameters: parameters[currentModule]
      });
      
      if (!validationRes.data.valid) {
        setValidationErrors(validationRes.data.errors);
        setSnackbar({
          open: true,
          message: 'Validation failed. Please check the errors and try again.',
          severity: 'error'
        });
        return;
      }
      
      // Save parameters
      await axios.put(`${API_BASE_URL}/parameters/${currentModule}`, parameters[currentModule]);
      
      setSnackbar({
        open: true,
        message: `${currentModule} parameters saved successfully!`,
        severity: 'success'
      });
      
      // Clear validation errors
      setValidationErrors({});
    } catch (error) {
      console.error('Error saving parameters:', error);
      setSnackbar({
        open: true,
        message: 'Failed to save parameters. Please try again.',
        severity: 'error'
      });
    } finally {
      setLoading(false);
    }
  };

  // Reset parameters for current module
  const resetModuleParameters = () => {
    setConfirmDialog({
      open: true,
      title: 'Reset Parameters',
      message: `Are you sure you want to reset all ${currentModule} parameters to default values?`,
      onConfirm: async () => {
        try {
          setLoading(true);
          
          // Get default parameters
          const defaultRes = await axios.get(`${API_BASE_URL}/parameters/reset`);
          
          // Update state with default parameters for current module
          setParameters(prevParams => ({
            ...prevParams,
            [currentModule]: defaultRes.data[currentModule]
          }));
          
          setSnackbar({
            open: true,
            message: `${currentModule} parameters reset to defaults!`,
            severity: 'success'
          });
        } catch (error) {
          console.error('Error resetting parameters:', error);
          setSnackbar({
            open: true,
            message: 'Failed to reset parameters. Please try again.',
            severity: 'error'
          });
        } finally {
          setLoading(false);
        }
      }
    });
  };

  // Apply preset
  const applyPreset = (presetName) => {
    setConfirmDialog({
      open: true,
      title: 'Apply Preset',
      message: `Are you sure you want to apply the "${presetName}" preset? This will change multiple parameters across different modules.`,
      onConfirm: async () => {
        try {
          setLoading(true);
          
          // Apply preset
          const res = await axios.post(`${API_BASE_URL}/parameters/presets/${presetName}`);
          
          // Update state with new parameters
          setParameters(res.data.parameters);
          
          setSnackbar({
            open: true,
            message: `Preset "${presetName}" applied successfully!`,
            severity: 'success'
          });
        } catch (error) {
          console.error('Error applying preset:', error);
          setSnackbar({
            open: true,
            message: 'Failed to apply preset. Please try again.',
            severity: 'error'
          });
        } finally {
          setLoading(false);
        }
      }
    });
  };

  // Save custom preset
  const saveCustomPreset = async () => {
    if (!customPresetName) {
      setSnackbar({
        open: true,
        message: 'Please enter a name for your preset.',
        severity: 'warning'
      });
      return;
    }
    
    try {
      setLoading(true);
      
      // Save custom preset
      await axios.post(`${API_BASE_URL}/parameters/presets/custom`, {
        name: customPresetName,
        description: customPresetDescription
      });
      
      // Refresh presets
      const presetsRes = await axios.get(`${API_BASE_URL}/parameters/presets`);
      setPresets(presetsRes.data);
      
      setSnackbar({
        open: true,
        message: `Custom preset "${customPresetName}" saved successfully!`,
        severity: 'success'
      });
      
      // Close dialog
      setSavePresetDialog(false);
      setCustomPresetName('');
      setCustomPresetDescription('');
    } catch (error) {
      console.error('Error saving custom preset:', error);
      setSnackbar({
        open: true,
        message: 'Failed to save custom preset. Please try again.',
        severity: 'error'
      });
    } finally {
      setLoading(false);
    }
  };

  // Render parameter control based on type
  const renderParameterControl = (paramName, paramValue, metadata) => {
    if (!metadata) {
      return <Typography color="error">Metadata not found for {paramName}</Typography>;
    }

    const hasError = validationErrors[paramName] && validationErrors[paramName].length > 0;
    
    switch (metadata.type) {
      case 'numeric':
        return (
          <Box>
            <Slider
              value={paramValue}
              min={metadata.min}
              max={metadata.max}
              step={(metadata.max - metadata.min) / 100}
              onChange={(e, newValue) => handleParameterChange(paramName, newValue)}
              valueLabelDisplay="auto"
              marks={[
                { value: metadata.min, label: metadata.min.toString() },
                { value: metadata.max, label: metadata.max.toString() }
              ]}
              color={hasError ? "error" : "primary"}
            />
            <Box display="flex" justifyContent="space-between">
              <Typography variant="caption">Default: {metadata.default}</Typography>
              <TextField
                variant="outlined"
                size="small"
                value={paramValue}
                onChange={(e) => {
                  const newValue = parseFloat(e.target.value);
                  if (!isNaN(newValue)) {
                    handleParameterChange(paramName, newValue);
                  }
                }}
                error={hasError}
                helperText={hasError ? validationErrors[paramName][0] : ''}
                sx={{ width: '100px' }}
              />
            </Box>
          </Box>
        );
        
      case 'boolean':
        return (
          <Switch
            checked={paramValue}
            onChange={(e) => handleParameterChange(paramName, e.target.checked)}
            color={hasError ? "error" : "primary"}
          />
        );
        
      case 'option':
        return (
          <FormControl fullWidth error={hasError}>
            <Select
              value={paramValue}
              onChange={(e) => handleParameterChange(paramName, e.target.value)}
            >
              {metadata.options.map(option => (
                <MenuItem key={option} value={option}>{option}</MenuItem>
              ))}
            </Select>
            {hasError && <Typography color="error" variant="caption">{validationErrors[paramName][0]}</Typography>}
          </FormControl>
        );
        
      case 'array':
        return (
          <Box>
            <FormControl fullWidth error={hasError}>
              <Select
                multiple
                value={paramValue}
                onChange={(e) => handleParameterChange(paramName, e.target.value)}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={value} />
                    ))}
                  </Box>
                )}
              >
                {metadata.options.map(option => (
                  <MenuItem key={option} value={option}>{option}</MenuItem>
                ))}
              </Select>
              {hasError && <Typography color="error" variant="caption">{validationErrors[paramName][0]}</Typography>}
            </FormControl>
          </Box>
        );
        
      default:
        return (
          <TextField
            fullWidth
            value={paramValue}
            onChange={(e) => handleParameterChange(paramName, e.target.value)}
            error={hasError}
            helperText={hasError ? validationErrors[paramName][0] : ''}
          />
        );
    }
  };

  // Render parameter card
  const renderParameterCard = (paramName, paramValue) => {
    const fullParamName = `${currentModule}.${paramName}`;
    const paramMetadata = metadata[fullParamName] || { 
      type: 'string', 
      category: CATEGORIES.BASIC,
      description: 'No description available'
    };
    
    // Skip if parameter should not be shown based on category
    if (paramMetadata.category === CATEGORIES.ADVANCED && !showAdvanced) return null;
    if (paramMetadata.category === CATEGORIES.EXPERT && !showExpert) return null;
    
    // Determine card color based on category
    const cardColor = {
      [CATEGORIES.BASIC]: '#e8f5e9',
      [CATEGORIES.ADVANCED]: '#e3f2fd',
      [CATEGORIES.EXPERT]: '#fff3e0'
    }[paramMetadata.category];
    
    return (
      <Grid item xs={12} md={6} key={paramName}>
        <Paper 
          elevation={2} 
          sx={{ 
            p: 2, 
            borderTop: `4px solid ${cardColor}`,
            height: '100%'
          }}
        >
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
            <Typography variant="subtitle1" fontWeight="bold">
              {paramName.replace(/_/g, ' ').replace(/([A-Z])/g, ' $1').trim()}
            </Typography>
            <Tooltip title={paramMetadata.description}>
              <InfoIcon fontSize="small" color="action" />
            </Tooltip>
          </Box>
          
          <Typography variant="body2" color="textSecondary" mb={2}>
            {paramMetadata.description}
          </Typography>
          
          {renderParameterControl(paramName, paramValue, paramMetadata)}
          
          {paramMetadata.category === CATEGORIES.EXPERT && (
            <Box mt={2}>
              <Alert severity="warning" icon={<WarningIcon />} variant="outlined">
                Expert setting - use with caution
              </Alert>
            </Box>
          )}
        </Paper>
      </Grid>
    );
  };

  // Render preset cards
  const renderPresetCards = () => {
    return (
      <Box mt={4}>
        <Typography variant="h6" gutterBottom>Parameter Presets</Typography>
        <Grid container spacing={2}>
          {Object.entries(presets).map(([presetName, presetData]) => (
            <Grid item xs={12} sm={6} md={4} key={presetName}>
              <Paper elevation={3} sx={{ p: 2, height: '100%' }}>
                <Typography variant="subtitle1" fontWeight="bold">
                  {presetName.charAt(0).toUpperCase() + presetName.slice(1).replace(/_/g, ' ')}
                </Typography>
                <Typography variant="body2" color="textSecondary" mb={2}>
                  {presetData.description || 'No description available'}
                </Typography>
                <Button 
                  variant="outlined" 
                  color="primary"
                  onClick={() => applyPreset(presetName)}
                  fullWidth
                >
                  Load Preset
                </Button>
              </Paper>
            </Grid>
          ))}
          
          {/* Add custom preset card */}
          <Grid item xs={12} sm={6} md={4}>
            <Paper 
              elevation={3} 
              sx={{ 
                p: 2, 
                height: '100%', 
                display: 'flex', 
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                backgroundColor: '#f5f5f5'
              }}
            >
              <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                Save Custom Preset
              </Typography>
              <Typography variant="body2" color="textSecondary" mb={2} textAlign="center">
                Save current configuration as a custom preset
              </Typography>
              <Button 
                variant="contained" 
                color="primary"
                onClick={() => setSavePresetDialog(true)}
                startIcon={<SaveIcon />}
              >
                Save Current Settings
              </Button>
            </Paper>
          </Grid>
        </Grid>
      </Box>
    );
  };

  // Loading indicator
  if (loading && Object.keys(parameters).length === 0) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100vh">
        <CircularProgress />
        <Typography variant="h6" ml={2}>Loading parameters...</Typography>
      </Box>
    );
  }

  return (
    <Container maxWidth="xl">
      <Box py={4}>
        <Typography variant="h4" gutterBottom>Trading Agent Configuration</Typography>
        <Typography variant="subtitle1" color="textSecondary" gutterBottom>
          Configure parameters for all system modules
        </Typography>
        
        <Paper sx={{ mt: 4 }}>
          <Tabs
            value={currentModule}
            onChange={handleModuleChange}
            variant="scrollable"
            scrollButtons="auto"
            sx={{ borderBottom: 1, borderColor: 'divider' }}
          >
            {Object.keys(parameters).map(module => (
              <Tab 
                key={module} 
                value={module} 
                label={module.replace(/_/g, ' ')} 
                icon={MODULE_ICONS[module]} 
                iconPosition="start"
              />
            ))}
          </Tabs>
          
          <Box p={3}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
              <Typography variant="h5">
                {currentModule.replace(/_/g, ' ')}
              </Typography>
              
              <Box>
                <Button
                  variant="outlined"
                  color="primary"
                  onClick={saveParameters}
                  startIcon={<SaveIcon />}
                  sx={{ mr: 1 }}
                  disabled={loading}
                >
                  Save Changes
                </Button>
                <Button
                  variant="outlined"
                  color="secondary"
                  onClick={resetModuleParameters}
                  startIcon={<RestoreIcon />}
                  disabled={loading}
                >
                  Reset to Defaults
                </Button>
              </Box>
            </Box>
            
            <Box mb={3}>
              <FormControl component="fieldset">
                <Typography variant="subtitle2" gutterBottom>Show Parameters:</Typography>
                <Box display="flex" alignItems="center">
                  <Typography variant="body2" color="primary" sx={{ mr: 1 }}>Basic</Typography>
                  <Switch
                    checked={showAdvanced}
                    onChange={(e) => setShowAdvanced(e.target.checked)}
                    color="primary"
                  />
                  <Typography variant="body2" color="primary" sx={{ mr: 2 }}>Advanced</Typography>
                  <Switch
                    checked={showExpert}
                    onChange={(e) => setShowExpert(e.target.checked)}
                    color="warning"
                  />
                  <Typography variant="body2" color="warning.main">Expert</Typography>
                </Box>
              </FormControl>
            </Box>
            
            {loading && (
              <Box display="flex" justifyContent="center" my={4}>
                <CircularProgress size={24} sx={{ mr: 1 }} />
                <Typography>Loading...</Typography>
              </Box>
            )}
            
            <Grid container spacing={3}>
              {parameters[currentModule] && Object.entries(parameters[currentModule]).map(([paramName, paramValue]) => 
                renderParameterCard(paramName, paramValue)
              )}
            </Grid>
          </Box>
        </Paper>
        
        {/* Presets Section */}
        {renderPresetCards()}
      </Box>
      
      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        message={snackbar.message}
      />
      
      {/* Confirmation Dialog */}
      <Dialog
        open={confirmDialog.open}
        onClose={() => setConfirmDialog({ ...confirmDialog, open: false })}
      >
        <DialogTitle>{confirmDialog.title}</DialogTitle>
        <DialogContent>
          <Typography>{confirmDialog.message}</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmDialog({ ...confirmDialog, open: false })}>
            Cancel
          </Button>
          <Button 
            onClick={() => {
              confirmDialog.onConfirm();
              setConfirmDialog({ ...confirmDialog, open: false });
            }} 
            color="primary" 
            variant="contained"
          >
            Confirm
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Save Preset Dialog */}
      <Dialog
        open={savePresetDialog}
        onClose={() => setSavePresetDialog(false)}
      >
        <DialogTitle>Save Custom Preset</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Preset Name"
            fullWidth
            variant="outlined"
            value={customPresetName}
            onChange={(e) => setCustomPresetName(e.target.value)}
          />
          <TextField
            margin="dense"
            label="Description"
            fullWidth
            variant="outlined"
            multiline
            rows={3}
            value={customPresetDescription}
            onChange={(e) => setCustomPresetDescription(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSavePresetDialog(false)}>
            Cancel
          </Button>
          <Button 
            onClick={saveCustomPreset} 
            color="primary" 
            variant="contained"
            disabled={!customPresetName}
          >
            Save Preset
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}

export default ParameterManagementApp;
