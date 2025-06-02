#!/usr/bin/env python
"""
Enhanced Pattern Recognition Model with Residual Blocks

This module provides an enhanced deep learning model for pattern recognition
in financial time series data, with improved architecture using residual blocks.
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_dl_model_fixed.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_dl_model_fixed")

class ResidualBlock(nn.Module):
    """Residual block for deep learning model"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        """Initialize residual block
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for convolutions
            dilation: Dilation factor for convolutions
        """
        super(ResidualBlock, self).__init__()
        
        # Calculate padding to maintain temporal dimension
        padding = (kernel_size - 1) * dilation // 2
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection (1x1 conv if dimensions don't match)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        """Forward pass
        
        Args:
            x: Input tensor [batch_size, in_channels, sequence_length]
            
        Returns:
            Output tensor [batch_size, out_channels, sequence_length]
        """
        # Main path
        residual = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        residual = self.skip(residual)
        
        # Add residual and apply activation
        out = F.relu(out + residual)
        
        return out

class EnhancedPatternRecognitionModel(nn.Module):
    """Enhanced pattern recognition model with residual blocks"""
    
    def __init__(self, 
                 input_dim: int = 9,
                 hidden_dim: int = 64,
                 output_dim: int = 3,
                 sequence_length: int = 60,
                 forecast_horizon: int = 10,
                 model_type: str = "hybrid",
                 device: str = None):
        """Initialize enhanced pattern recognition model
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
            output_dim: Number of output classes/patterns
            sequence_length: Length of input sequence
            forecast_horizon: Number of future steps to predict
            model_type: Model architecture type (cnn, lstm, hybrid)
            device: Device to use (cpu, cuda, or None for auto-detection)
        """
        super(EnhancedPatternRecognitionModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.model_type = model_type
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Create model based on type
        if model_type == "cnn":
            self.model = self._create_cnn_model()
        elif model_type == "lstm":
            self.model = self._create_lstm_model()
        elif model_type == "hybrid":
            self.model = self._create_hybrid_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        logger.info(f"Initialized EnhancedPatternRecognitionModel with {model_type} architecture on {self.device}")
    
    def _create_cnn_model(self):
        """Create CNN model with residual blocks
        
        Returns:
            nn.Module: CNN model
        """
        class CNNModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, sequence_length, forecast_horizon):
                super(CNNModel, self).__init__()
                
                # Input layer
                self.input_layer = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
                
                # Residual blocks with increasing dilation
                self.res_blocks = nn.ModuleList([
                    ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=1),
                    ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
                    ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4)
                ])
                
                # Output layer
                self.output_layer = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
                
                # Forecast horizon
                self.forecast_horizon = forecast_horizon
            
            def forward(self, x):
                # Input shape: [batch_size, sequence_length, input_dim]
                # Reshape for 1D convolution: [batch_size, input_dim, sequence_length]
                x = x.permute(0, 2, 1)
                
                # Input layer
                x = F.relu(self.input_layer(x))
                
                # Residual blocks
                for block in self.res_blocks:
                    x = block(x)
                
                # Output layer
                x = self.output_layer(x)
                
                # Get last forecast_horizon steps
                x = x[:, :, -self.forecast_horizon:]
                
                # Reshape to [batch_size, forecast_horizon, output_dim]
                x = x.permute(0, 2, 1)
                
                return x
        
        return CNNModel(self.input_dim, self.hidden_dim, self.output_dim, self.sequence_length, self.forecast_horizon)
    
    def _create_lstm_model(self):
        """Create LSTM model
        
        Returns:
            nn.Module: LSTM model
        """
        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, sequence_length, forecast_horizon):
                super(LSTMModel, self).__init__()
                
                # LSTM layers
                self.lstm1 = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=2,
                    batch_first=True,
                    dropout=0.2,
                    bidirectional=True
                )
                
                # Output layer
                self.output_layer = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
                
                # Forecast horizon
                self.forecast_horizon = forecast_horizon
            
            def forward(self, x):
                # Input shape: [batch_size, sequence_length, input_dim]
                
                # LSTM layer
                lstm_out, _ = self.lstm1(x)
                
                # Get last forecast_horizon steps
                lstm_out = lstm_out[:, -self.forecast_horizon:, :]
                
                # Output layer
                output = self.output_layer(lstm_out)
                
                return output
        
        return LSTMModel(self.input_dim, self.hidden_dim, self.output_dim, self.sequence_length, self.forecast_horizon)
    
    def _create_hybrid_model(self):
        """Create hybrid CNN-LSTM model
        
        Returns:
            nn.Module: Hybrid model
        """
        class HybridModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, sequence_length, forecast_horizon):
                super(HybridModel, self).__init__()
                
                # CNN feature extraction
                self.input_layer = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
                
                # Residual blocks
                self.res_blocks = nn.ModuleList([
                    ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=1),
                    ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2)
                ])
                
                # LSTM for temporal modeling
                self.lstm = nn.LSTM(
                    input_size=hidden_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True
                )
                
                # Output layer
                self.output_layer = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
                
                # Forecast horizon
                self.forecast_horizon = forecast_horizon
            
            def forward(self, x):
                # Input shape: [batch_size, sequence_length, input_dim]
                batch_size = x.size(0)
                
                # Reshape for 1D convolution: [batch_size, input_dim, sequence_length]
                x = x.permute(0, 2, 1)
                
                # CNN feature extraction
                x = F.relu(self.input_layer(x))
                
                # Residual blocks
                for block in self.res_blocks:
                    x = block(x)
                
                # Reshape for LSTM: [batch_size, sequence_length, hidden_dim]
                x = x.permute(0, 2, 1)
                
                # LSTM layer
                lstm_out, _ = self.lstm(x)
                
                # Get last forecast_horizon steps
                lstm_out = lstm_out[:, -self.forecast_horizon:, :]
                
                # Output layer
                output = self.output_layer(lstm_out)
                
                return output
        
        return HybridModel(self.input_dim, self.hidden_dim, self.output_dim, self.sequence_length, self.forecast_horizon)
    
    def forward(self, x):
        """Forward pass
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_dim]
            
        Returns:
            Output tensor [batch_size, forecast_horizon, output_dim]
        """
        return self.model(x)
    
    def save_model(self, path):
        """Save model to file
        
        Args:
            path: Path to save model
            
        Returns:
            bool: Success
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'sequence_length': self.sequence_length,
                'forecast_horizon': self.forecast_horizon,
                'model_type': self.model_type
            }, path)
            
            logger.info(f"Model saved to {path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, path):
        """Load model from file
        
        Args:
            path: Path to load model from
            
        Returns:
            bool: Success
        """
        try:
            # Check if file exists
            if not os.path.exists(path):
                logger.error(f"Model file not found: {path}")
                return False
            
            # Load model
            checkpoint = torch.load(path, map_location=self.device)
            
            # Update model parameters
            self.input_dim = checkpoint.get('input_dim', self.input_dim)
            self.hidden_dim = checkpoint.get('hidden_dim', self.hidden_dim)
            self.output_dim = checkpoint.get('output_dim', self.output_dim)
            self.sequence_length = checkpoint.get('sequence_length', self.sequence_length)
            self.forecast_horizon = checkpoint.get('forecast_horizon', self.forecast_horizon)
            self.model_type = checkpoint.get('model_type', self.model_type)
            
            # Recreate model with loaded parameters
            if self.model_type == "cnn":
                self.model = self._create_cnn_model()
            elif self.model_type == "lstm":
                self.model = self._create_lstm_model()
            elif self.model_type == "hybrid":
                self.model = self._create_hybrid_model()
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            logger.info(f"Model loaded from {path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, x):
        """Make prediction
        
        Args:
            x: Input tensor or numpy array [batch_size, sequence_length, input_dim]
            
        Returns:
            numpy array: Predictions [batch_size, forecast_horizon, output_dim]
        """
        try:
            # Convert to tensor if numpy array
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            
            # Move to device
            x = x.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Make prediction
            with torch.no_grad():
                predictions = self.model(x)
            
            # Convert to numpy array
            predictions = predictions.cpu().numpy()
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None
    
    # Add mock prediction method for testing
    def predict_mock(self, x):
        """Mock prediction method for testing
        
        Args:
            x: Input tensor or numpy array [batch_size, sequence_length, input_dim]
            
        Returns:
            numpy array: Mock predictions [batch_size, forecast_horizon, output_dim]
        """
        try:
            # Get batch size and forecast horizon
            if isinstance(x, np.ndarray):
                batch_size = x.shape[0]
            else:
                batch_size = x.size(0)
            
            # Create mock predictions with high confidence for all pattern types
            mock_predictions = np.ones((batch_size, self.forecast_horizon, self.output_dim)) * 0.92
            
            logger.info(f"Generated mock predictions with shape {mock_predictions.shape}")
            return mock_predictions
        
        except Exception as e:
            logger.error(f"Error making mock prediction: {str(e)}")
            return None

# For backward compatibility
TemporalConvNet = EnhancedPatternRecognitionModel
