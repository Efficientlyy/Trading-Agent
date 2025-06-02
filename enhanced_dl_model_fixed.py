#!/usr/bin/env python
"""
Enhanced Deep Learning Model for Trading-Agent System

This module provides enhanced deep learning models for the Trading-Agent system,
with improved pattern recognition capabilities.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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

class HybridLSTMAttention(nn.Module):
    """Hybrid LSTM with attention mechanism for pattern recognition"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, sequence_length: int, forecast_horizon: int):
        """Initialize hybrid LSTM with attention
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            sequence_length: Sequence length
            forecast_horizon: Forecast horizon
        """
        super(HybridLSTMAttention, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # Bidirectional
            num_heads=4,
            dropout=0.1
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, output_dim * forecast_horizon)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, forecast_horizon, output_dim)
        """
        # LSTM layer
        lstm_out, _ = self.lstm(x)  # (batch_size, sequence_length, hidden_dim * 2)
        
        # Attention layer
        attn_input = lstm_out.permute(1, 0, 2)  # (sequence_length, batch_size, hidden_dim * 2)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, sequence_length, hidden_dim * 2)
        
        # Get last sequence element
        last_output = attn_output[:, -1, :]  # (batch_size, hidden_dim * 2)
        
        # Output layers
        fc1_out = self.relu(self.fc1(last_output))  # (batch_size, hidden_dim)
        fc1_out = self.dropout(fc1_out)
        fc2_out = self.fc2(fc1_out)  # (batch_size, output_dim * forecast_horizon)
        
        # Reshape to (batch_size, forecast_horizon, output_dim)
        output = fc2_out.view(-1, self.forecast_horizon, self.output_dim)
        
        # Apply sigmoid to get confidence scores
        output = self.sigmoid(output)
        
        return output

class EnhancedPatternRecognitionModel:
    """Enhanced pattern recognition model for deep learning"""
    
    def __init__(self, 
                 input_dim: int = 16,  # Updated from 9 to 16 to match feature adapter output
                 hidden_dim: int = 64, 
                 output_dim: int = 3, 
                 sequence_length: int = 40,  # Reduced from 60 to 40 for real data compatibility
                 forecast_horizon: int = 10,
                 model_type: str = "hybrid",
                 device: str = "cpu"):
        """Initialize enhanced pattern recognition model
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            sequence_length: Sequence length
            forecast_horizon: Forecast horizon
            model_type: Model type (hybrid, lstm, transformer)
            device: Device to use for inference
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.model_type = model_type
        self.device = device
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        logger.info(f"Initialized EnhancedPatternRecognitionModel with input_dim={input_dim}, model_type={model_type}, device={device}")
    
    def _create_model(self) -> nn.Module:
        """Create model based on model type
        
        Returns:
            nn.Module: Model
        """
        if self.model_type == "hybrid":
            return HybridLSTMAttention(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                sequence_length=self.sequence_length,
                forecast_horizon=self.forecast_horizon
            )
        else:
            # Default to hybrid model
            logger.warning(f"Unknown model type: {self.model_type}, using hybrid model")
            return HybridLSTMAttention(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                sequence_length=self.sequence_length,
                forecast_horizon=self.forecast_horizon
            )
    
    def load_model(self, model_path: str) -> bool:
        """Load model from file
        
        Args:
            model_path: Path to model file
            
        Returns:
            bool: Success
        """
        try:
            # Load model weights
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def save_model(self, model_path: str) -> bool:
        """Save model to file
        
        Args:
            model_path: Path to model file
            
        Returns:
            bool: Success
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model weights
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"Saved model to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001) -> Dict:
        """Train model
        
        Args:
            X: Input features
            y: Target labels
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            dict: Training history
        """
        try:
            # Convert to tensors
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
            
            # Create dataset
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Define loss function and optimizer
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
            # Training history
            history = {
                "loss": [],
                "val_loss": []
            }
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                epoch_loss = 0.0
                
                for batch_X, batch_y in dataloader:
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Update statistics
                    epoch_loss += loss.item() * batch_X.size(0)
                
                # Calculate average loss
                epoch_loss /= len(dataset)
                
                # Update history
                history["loss"].append(epoch_loss)
                
                # Log progress
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
            
            logger.info(f"Training completed, final loss: {history['loss'][-1]:.4f}")
            return history
        
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {"loss": [], "val_loss": []}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predictions
        """
        try:
            # Check input shape and log it
            logger.info(f"Input shape for prediction: {X.shape}")
            
            # Handle input shape mismatch
            if X.shape[2] != self.input_dim:
                logger.warning(f"Input dimension mismatch: expected {self.input_dim}, got {X.shape[2]}")
                
                # If input has more features than expected, select first input_dim features
                if X.shape[2] > self.input_dim:
                    logger.info(f"Truncating input from {X.shape[2]} to {self.input_dim} features")
                    X = X[:, :, :self.input_dim]
                # If input has fewer features than expected, pad with zeros
                else:
                    logger.info(f"Padding input from {X.shape[2]} to {self.input_dim} features")
                    padding = np.zeros((X.shape[0], X.shape[1], self.input_dim - X.shape[2]))
                    X = np.concatenate([X, padding], axis=2)
            
            # Convert to tensor
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
            
            # Convert to numpy
            predictions = outputs.cpu().numpy()
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return np.array([])
    
    def predict_mock(self, X: np.ndarray) -> np.ndarray:
        """Make mock predictions for testing
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Mock predictions
        """
        try:
            # Create mock predictions with high confidence
            batch_size = X.shape[0]
            predictions = np.ones((batch_size, self.forecast_horizon, self.output_dim)) * 0.92
            
            # Add some randomness
            predictions += np.random.normal(0, 0.05, predictions.shape)
            
            # Ensure values are between 0 and 1
            predictions = np.clip(predictions, 0, 1)
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error making mock predictions: {str(e)}")
            return np.array([])
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model
        
        Args:
            X: Input features
            y: Target labels
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            # Handle input shape mismatch
            if X.shape[2] != self.input_dim:
                logger.warning(f"Input dimension mismatch: expected {self.input_dim}, got {X.shape[2]}")
                
                # If input has more features than expected, select first input_dim features
                if X.shape[2] > self.input_dim:
                    logger.info(f"Truncating input from {X.shape[2]} to {self.input_dim} features")
                    X = X[:, :, :self.input_dim]
                # If input has fewer features than expected, pad with zeros
                else:
                    logger.info(f"Padding input from {X.shape[2]} to {self.input_dim} features")
                    padding = np.zeros((X.shape[0], X.shape[1], self.input_dim - X.shape[2]))
                    X = np.concatenate([X, padding], axis=2)
            
            # Convert to tensors
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
            
            # Define loss function
            criterion = nn.BCELoss()
            
            # Evaluate model
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor)
                loss = criterion(outputs, y_tensor).item()
                
                # Convert to numpy
                predictions = outputs.cpu().numpy()
                targets = y_tensor.cpu().numpy()
                
                # Calculate metrics
                accuracy = np.mean((predictions > 0.5) == (targets > 0.5))
                precision = np.sum((predictions > 0.5) * (targets > 0.5)) / (np.sum(predictions > 0.5) + 1e-10)
                recall = np.sum((predictions > 0.5) * (targets > 0.5)) / (np.sum(targets > 0.5) + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            # Return metrics
            metrics = {
                "loss": loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            
            logger.info(f"Evaluation metrics: {metrics}")
            return metrics
        
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
