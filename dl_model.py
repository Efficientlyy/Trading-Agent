#!/usr/bin/env python
"""
Deep Learning Model for Pattern Recognition in Financial Markets

This module provides a deep learning model for pattern recognition in financial markets,
including temporal convolutional networks and transformer-based architectures.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dl_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dl_model")

class TemporalBlock(nn.Module):
    """Temporal convolutional block with residual connection"""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """Initialize temporal block
        
        Args:
            n_inputs: Number of input channels
            n_outputs: Number of output channels
            kernel_size: Size of the convolutional kernel
            stride: Stride of the convolution
            dilation: Dilation of the convolution
            padding: Padding size
            dropout: Dropout probability
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.bn2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        """Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, n_inputs, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, n_outputs, sequence_length)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for pattern recognition"""
    
    def __init__(self, input_dim=9, hidden_dim=64, output_dim=3, num_levels=3, 
                 kernel_size=3, dropout=0.2, sequence_length=60, forecast_horizon=10):
        """Initialize temporal convolutional network
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
            output_dim: Number of output classes
            num_levels: Number of temporal blocks
            kernel_size: Size of the convolutional kernel
            dropout: Dropout probability
            sequence_length: Length of input sequence
            forecast_horizon: Number of future steps to predict
        """
        super(TemporalConvNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        
        # Calculate padding based on kernel size
        padding = (kernel_size - 1) // 2
        
        # Create temporal blocks with increasing dilation
        layers = []
        num_channels = [hidden_dim] * num_levels
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                       stride=1, dilation=dilation,
                                       padding=padding, dropout=dropout))
        
        self.temporal_blocks = nn.Sequential(*layers)
        
        # Output layer for each forecast step
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for _ in range(forecast_horizon)
        ])
        
        # Initialize weights
        self.init_weights()
        
        logger.info(f"Initialized TemporalConvNet with input_dim={input_dim}, "
                   f"hidden_dim={hidden_dim}, output_dim={output_dim}")
    
    def init_weights(self):
        """Initialize weights"""
        for layer in self.output_layers:
            layer.weight.data.normal_(0, 0.01)
            layer.bias.data.fill_(0)
    
    def forward(self, x):
        """Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, forecast_horizon, output_dim)
        """
        # Transpose to (batch_size, input_dim, sequence_length)
        x = x.transpose(1, 2)
        
        # Apply temporal blocks
        features = self.temporal_blocks(x)
        
        # Global average pooling
        features = F.adaptive_avg_pool1d(features, 1).squeeze(-1)
        
        # Apply output layers for each forecast step
        outputs = []
        for layer in self.output_layers:
            outputs.append(layer(features))
        
        # Stack outputs along forecast dimension
        return torch.stack(outputs, dim=1)
    
    def predict(self, x):
        """Make prediction
        
        Args:
            x: Input tensor or numpy array of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            numpy array of shape (batch_size, forecast_horizon, output_dim)
        """
        # Convert to tensor if numpy array
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Set model to evaluation mode
        self.eval()
        
        # Make prediction
        with torch.no_grad():
            output = self.forward(x)
        
        # Convert to numpy array
        return output.cpu().numpy()
    
    def predict_mock(self, x):
        """Make mock prediction for testing
        
        Args:
            x: Input tensor or numpy array of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            numpy array of shape (batch_size, forecast_horizon, output_dim)
        """
        # Convert to tensor if numpy array
        if isinstance(x, np.ndarray):
            batch_size = x.shape[0]
        else:
            batch_size = x.size(0)
        
        # Generate mock predictions with some values above threshold
        mock_predictions = np.random.uniform(0.5, 0.9, size=(batch_size, self.forecast_horizon, self.output_dim))
        
        # Ensure at least some predictions are above common thresholds (0.7, 0.8)
        # This guarantees pattern signals will be generated in tests
        for i in range(batch_size):
            for j in range(self.forecast_horizon):
                # Randomly select one output dimension to have high confidence
                high_conf_idx = np.random.randint(0, self.output_dim)
                mock_predictions[i, j, high_conf_idx] = np.random.uniform(0.75, 0.95)
        
        logger.info(f"Generated mock predictions with shape {mock_predictions.shape}")
        return mock_predictions
    
    def save_model(self, path):
        """Save model to file
        
        Args:
            path: Path to save model
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            torch.save({
                'model_state_dict': self.state_dict(),
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'num_levels': self.num_levels,
                'kernel_size': self.kernel_size,
                'dropout': self.dropout,
                'sequence_length': self.sequence_length,
                'forecast_horizon': self.forecast_horizon
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
            bool: True if successful, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(path):
                logger.error(f"Model file not found: {path}")
                return False
            
            # Load model
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            
            # Update model parameters
            self.input_dim = checkpoint.get('input_dim', self.input_dim)
            self.hidden_dim = checkpoint.get('hidden_dim', self.hidden_dim)
            self.output_dim = checkpoint.get('output_dim', self.output_dim)
            self.num_levels = checkpoint.get('num_levels', self.num_levels)
            self.kernel_size = checkpoint.get('kernel_size', self.kernel_size)
            self.dropout = checkpoint.get('dropout', self.dropout)
            self.sequence_length = checkpoint.get('sequence_length', self.sequence_length)
            self.forecast_horizon = checkpoint.get('forecast_horizon', self.forecast_horizon)
            
            # Load state dict
            self.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"Model loaded from {path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

class ResidualBlock(nn.Module):
    """Residual block for transformer-based model"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """Initialize residual block
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super(ResidualBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x, attn=None):
        """Forward pass
        
        Args:
            x: Input tensor
            attn: Attention function (optional)
            
        Returns:
            Output tensor
        """
        # Apply attention if provided
        if attn is not None:
            x = x + self.dropout(attn(self.norm1(x)))
        
        # Apply feed-forward
        x = x + self.dropout(self.ff(self.norm2(x)))
        
        return x

class EnhancedPatternRecognitionModel(nn.Module):
    """Enhanced pattern recognition model with transformer architecture"""
    
    def __init__(self, input_dim=9, hidden_dim=64, output_dim=3, num_layers=3,
                 num_heads=4, dropout=0.1, sequence_length=60, forecast_horizon=10,
                 model_type="hybrid", device=None):
        """Initialize enhanced pattern recognition model
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
            output_dim: Number of output classes
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            sequence_length: Length of input sequence
            forecast_horizon: Number of future steps to predict
            model_type: Model type (transformer, tcn, or hybrid)
            device: Device to use (cpu, cuda, or None for auto-detection)
        """
        super(EnhancedPatternRecognitionModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.model_type = model_type
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.register_buffer('pos_encoding', self._get_positional_encoding(sequence_length, hidden_dim))
        
        # Transformer layers
        if model_type in ["transformer", "hybrid"]:
            self.transformer_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, 
                                          dim_feedforward=hidden_dim*4, dropout=dropout,
                                          batch_first=True)
                for _ in range(num_layers)
            ])
        
        # TCN layers
        if model_type in ["tcn", "hybrid"]:
            self.tcn = TemporalConvNet(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,  # Output same dimension as hidden
                num_levels=num_layers,
                kernel_size=3,
                dropout=dropout,
                sequence_length=sequence_length,
                forecast_horizon=1  # Only need features, not forecasts
            )
        
        # Output layers for each forecast step
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for _ in range(forecast_horizon)
        ])
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"Initialized EnhancedPatternRecognitionModel with input_dim={input_dim}, "
                   f"hidden_dim={hidden_dim}, output_dim={output_dim}, model_type={model_type}")
    
    def _get_positional_encoding(self, seq_len, d_model):
        """Get positional encoding
        
        Args:
            seq_len: Sequence length
            d_model: Model dimension
            
        Returns:
            Positional encoding tensor
        """
        pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        
        return pe
    
    def forward(self, x):
        """Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, forecast_horizon, output_dim)
        """
        batch_size = x.size(0)
        
        if self.model_type == "transformer":
            # Apply embedding
            x = self.embedding(x)
            
            # Add positional encoding
            x = x + self.pos_encoding.unsqueeze(0)
            
            # Apply transformer layers
            for layer in self.transformer_layers:
                x = layer(x)
            
            # Use last token for prediction
            features = x[:, -1, :]
        
        elif self.model_type == "tcn":
            # Apply TCN
            features = self.tcn(x).squeeze(1)
        
        else:  # hybrid
            # Apply embedding for transformer
            x_transformer = self.embedding(x)
            
            # Add positional encoding
            x_transformer = x_transformer + self.pos_encoding.unsqueeze(0)
            
            # Apply transformer layers
            for layer in self.transformer_layers:
                x_transformer = layer(x_transformer)
            
            # Use last token for transformer features
            transformer_features = x_transformer[:, -1, :]
            
            # Apply TCN
            tcn_features = self.tcn(x).squeeze(1)
            
            # Combine features
            features = transformer_features + tcn_features
        
        # Apply output layers for each forecast step
        outputs = []
        for layer in self.output_layers:
            outputs.append(layer(features))
        
        # Stack outputs along forecast dimension
        return torch.stack(outputs, dim=1)
    
    def predict(self, x):
        """Make prediction
        
        Args:
            x: Input tensor or numpy array of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            numpy array of shape (batch_size, forecast_horizon, output_dim)
        """
        # Convert to tensor if numpy array
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        else:
            x = x.to(self.device)
        
        # Set model to evaluation mode
        self.eval()
        
        # Make prediction
        with torch.no_grad():
            output = self.forward(x)
        
        # Convert to numpy array
        return output.cpu().numpy()
    
    def predict_mock(self, x):
        """Make mock prediction for testing
        
        Args:
            x: Input tensor or numpy array of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            numpy array of shape (batch_size, forecast_horizon, output_dim)
        """
        # Convert to tensor if numpy array
        if isinstance(x, np.ndarray):
            batch_size = x.shape[0]
        else:
            batch_size = x.size(0)
        
        # Generate mock predictions with some values above threshold
        mock_predictions = np.random.uniform(0.5, 0.9, size=(batch_size, self.forecast_horizon, self.output_dim))
        
        # Ensure at least some predictions are above common thresholds (0.7, 0.8)
        # This guarantees pattern signals will be generated in tests
        for i in range(batch_size):
            for j in range(self.forecast_horizon):
                # Randomly select one output dimension to have high confidence
                high_conf_idx = np.random.randint(0, self.output_dim)
                mock_predictions[i, j, high_conf_idx] = np.random.uniform(0.75, 0.95)
        
        logger.info(f"Generated mock predictions with shape {mock_predictions.shape}")
        return mock_predictions
    
    def save_model(self, path):
        """Save model to file
        
        Args:
            path: Path to save model
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            torch.save({
                'model_state_dict': self.state_dict(),
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'dropout': self.dropout,
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
            bool: True if successful, False otherwise
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
            self.num_layers = checkpoint.get('num_layers', self.num_layers)
            self.num_heads = checkpoint.get('num_heads', self.num_heads)
            self.dropout = checkpoint.get('dropout', self.dropout)
            self.sequence_length = checkpoint.get('sequence_length', self.sequence_length)
            self.forecast_horizon = checkpoint.get('forecast_horizon', self.forecast_horizon)
            self.model_type = checkpoint.get('model_type', self.model_type)
            
            # Load state dict
            self.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"Model loaded from {path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
