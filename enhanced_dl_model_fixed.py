#!/usr/bin/env python
"""
Enhanced Deep Learning Model for Pattern Recognition with Attention Mechanism

This module provides an enhanced deep learning model for pattern recognition
in financial market data, with attention mechanisms and residual connections.
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
        logging.FileHandler("enhanced_dl_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_dl_model")

class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions"""
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3, 
                 dilation: int = 1, 
                 dropout: float = 0.2):
        """Initialize residual block
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for convolutions
            dilation: Dilation factor for convolutions
            dropout: Dropout rate
        """
        super(ResidualBlock, self).__init__()
        
        # Calculate padding to maintain sequence length
        padding = (kernel_size - 1) * dilation // 2
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        
        # Batch normalization
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, sequence_length)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, sequence_length)
        """
        # Save input for residual connection
        residual = self.residual(x)
        
        # First convolutional layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Second convolutional layer
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add residual connection
        out = out + residual
        
        # Apply activation
        out = F.relu(out)
        
        return out

class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network with residual connections"""
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 kernel_size: int = 3, 
                 dropout: float = 0.2, 
                 num_layers: int = 4):
        """Initialize TCN
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
            num_layers: Number of residual blocks
        """
        super(TemporalConvNet, self).__init__()
        
        # Create list of residual blocks with increasing dilation
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels=input_dim if i == 0 else hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=2**i,
                dropout=dropout
            )
            for i in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, hidden_dim)
        """
        # Transpose to (batch_size, input_dim, sequence_length) for 1D convolution
        x = x.transpose(1, 2)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Transpose back to (batch_size, sequence_length, hidden_dim)
        x = x.transpose(1, 2)
        
        return x

class SelfAttention(nn.Module):
    """Self-attention mechanism"""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        """Initialize self-attention
        
        Args:
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super(SelfAttention, self).__init__()
        
        # Query, key, value projections
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Scaling factor
        self.scale = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, hidden_dim)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights
        """
        # Project input to query, key, value
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights

class CrossAttention(nn.Module):
    """Cross-attention mechanism"""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        """Initialize cross-attention
        
        Args:
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super(CrossAttention, self).__init__()
        
        # Query, key, value projections
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output = nn.Linear(hidden_dim, hidden_dim)
        
        # Scaling factor
        self.scale = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                query: torch.Tensor, 
                key_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass
        
        Args:
            query: Query tensor of shape (batch_size, query_length, hidden_dim)
            key_value: Key-value tensor of shape (batch_size, kv_length, hidden_dim)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights
        """
        # Project inputs
        q = self.query(query)
        k = self.key(key_value)
        v = self.value(key_value)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        
        # Apply output projection
        output = self.output(context)
        
        return output, attn_weights

class TransformerBlock(nn.Module):
    """Transformer block with self-attention and cross-attention"""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        """Initialize transformer block
        
        Args:
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super(TransformerBlock, self).__init__()
        
        # Self-attention
        self.self_attention = SelfAttention(hidden_dim, dropout)
        
        # Cross-attention
        self.cross_attention = CrossAttention(hidden_dim, dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                x: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, hidden_dim)
            context: Context tensor for cross-attention (optional)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, hidden_dim)
        """
        # Self-attention
        attn_output, _ = self.self_attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention (if context is provided)
        if context is not None:
            cross_output, _ = self.cross_attention(x, context)
            x = self.norm2(x + self.dropout(cross_output))
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class HybridModel(nn.Module):
    """Hybrid model combining TCN, LSTM, and Transformer"""
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 sequence_length: int, 
                 forecast_horizon: int, 
                 dropout: float = 0.2):
        """Initialize hybrid model
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
            output_dim: Number of output features
            sequence_length: Length of input sequence
            forecast_horizon: Number of steps to forecast
            dropout: Dropout rate
        """
        super(HybridModel, self).__init__()
        
        # Save parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        
        # Feature embedding
        self.feature_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Temporal Convolutional Network
        self.tcn = TemporalConvNet(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            kernel_size=3,
            dropout=dropout,
            num_layers=4
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Transformer
        self.transformer = nn.ModuleList([
            TransformerBlock(hidden_dim * 2, dropout)
            for _ in range(2)
        ])
        
        # Cross-attention for sequence-to-sequence
        self.cross_attention = CrossAttention(hidden_dim * 2, dropout)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim * 2, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, forecast_horizon, output_dim)
        """
        batch_size = x.size(0)
        
        # Handle dynamic input dimensions
        if x.size(2) != self.input_dim:
            # If input has fewer dimensions than expected, pad with zeros
            if x.size(2) < self.input_dim:
                padding = torch.zeros(batch_size, x.size(1), self.input_dim - x.size(2), device=x.device)
                x = torch.cat([x, padding], dim=2)
            # If input has more dimensions than expected, select the first input_dim dimensions
            else:
                x = x[:, :, :self.input_dim]
        
        # Feature embedding
        x = self.feature_embedding(x)
        
        # Apply TCN
        tcn_output = self.tcn(x)
        
        # Apply LSTM
        lstm_output, _ = self.lstm(tcn_output)
        
        # Apply Transformer
        transformer_output = lstm_output
        for transformer_layer in self.transformer:
            transformer_output = transformer_layer(transformer_output)
        
        # Generate decoder input (use the last hidden state repeated)
        decoder_input = transformer_output[:, -1:, :].repeat(1, self.forecast_horizon, 1)
        
        # Apply cross-attention for sequence-to-sequence
        forecast, _ = self.cross_attention(decoder_input, transformer_output)
        
        # Apply output projection
        output = self.output_projection(forecast)
        
        return output

class EnhancedPatternRecognitionModel:
    """Enhanced model for pattern recognition in financial market data"""
    
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
            output_dim: Number of output features
            sequence_length: Length of input sequence
            forecast_horizon: Number of steps to forecast
            model_type: Type of model (hybrid, tcn, lstm, transformer)
            device: Device to use (cpu, cuda, or None for auto-detection)
        """
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
        
        logger.info(f"Using device: {self.device}")
        
        # Create model
        if model_type == "hybrid":
            self.model = HybridModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                sequence_length=sequence_length,
                forecast_horizon=forecast_horizon
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        logger.info(f"Initialized EnhancedPatternRecognitionModel with input_dim={input_dim}, "
                   f"hidden_dim={hidden_dim}, output_dim={output_dim}")
    
    def save_model(self, path: str) -> bool:
        """Save model to file
        
        Args:
            path: Path to save model
            
        Returns:
            bool: Success
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model state
            state_dict = {
                "model_state": self.model.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "output_dim": self.output_dim,
                    "sequence_length": self.sequence_length,
                    "forecast_horizon": self.forecast_horizon,
                    "model_type": self.model_type
                }
            }
            
            torch.save(state_dict, path)
            logger.info(f"Model saved to {path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, path: str) -> bool:
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
            
            # Load model state
            state_dict = torch.load(path, map_location=self.device)
            
            # Check if state_dict has the expected format
            if "model_state" not in state_dict or "config" not in state_dict:
                logger.error(f"Invalid model file format: {path}")
                return False
            
            # Update configuration
            config = state_dict["config"]
            self.input_dim = config.get("input_dim", self.input_dim)
            self.hidden_dim = config.get("hidden_dim", self.hidden_dim)
            self.output_dim = config.get("output_dim", self.output_dim)
            self.sequence_length = config.get("sequence_length", self.sequence_length)
            self.forecast_horizon = config.get("forecast_horizon", self.forecast_horizon)
            self.model_type = config.get("model_type", self.model_type)
            
            # Create new model with updated configuration
            if self.model_type == "hybrid":
                self.model = HybridModel(
                    input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=self.output_dim,
                    sequence_length=self.sequence_length,
                    forecast_horizon=self.forecast_horizon
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Load model state
            self.model.load_state_dict(state_dict["model_state"])
            
            logger.info(f"Model loaded from {path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make prediction
        
        Args:
            x: Input data of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            np.ndarray: Predictions of shape (batch_size, forecast_horizon, output_dim)
        """
        try:
            # Convert to tensor
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            
            # Handle dynamic input dimensions
            if x_tensor.size(2) != self.input_dim:
                # If input has fewer dimensions than expected, pad with zeros
                if x_tensor.size(2) < self.input_dim:
                    padding = torch.zeros(x_tensor.size(0), x_tensor.size(1), self.input_dim - x_tensor.size(2), device=self.device)
                    x_tensor = torch.cat([x_tensor, padding], dim=2)
                # If input has more dimensions than expected, select the first input_dim dimensions
                else:
                    x_tensor = x_tensor[:, :, :self.input_dim]
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(x_tensor)
            
            # Convert to numpy
            y_pred = y_pred.cpu().numpy()
            
            return y_pred
        
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return np.zeros((x.shape[0], self.forecast_horizon, self.output_dim))

# Example usage
if __name__ == "__main__":
    # Create model
    model = EnhancedPatternRecognitionModel(
        input_dim=9,
        hidden_dim=64,
        output_dim=3,
        sequence_length=60,
        forecast_horizon=10,
        model_type="hybrid",
        device="cpu"
    )
    
    # Create sample data
    x = np.random.randn(16, 60, 9)
    
    # Make prediction
    y_pred = model.predict(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_pred.shape}")
    
    # Save model
    model.save_model("models/pattern_recognition_model.pt")
    
    # Load model
    model.load_model("models/pattern_recognition_model.pt")
    
    # Make prediction again
    y_pred2 = model.predict(x)
    
    # Check if predictions are the same
    print(f"Predictions match: {np.allclose(y_pred, y_pred2)}")
