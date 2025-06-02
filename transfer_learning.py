#!/usr/bin/env python
"""
Transfer Learning Setup for Enhanced Deep Learning Pattern Recognition.

This module provides functionality to implement transfer learning for the
deep learning pattern recognition system, allowing models to be pre-trained
on general market data and fine-tuned on specific assets.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Union, Optional, Callable
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('transfer_learning')

class TransferLearningManager:
    """Class for managing transfer learning for deep learning models."""
    
    def __init__(self, 
                base_model: nn.Module,
                model_dir: str = 'models',
                device: str = None):
        """Initialize the transfer learning manager.
        
        Args:
            base_model: Base model for transfer learning
            model_dir: Directory to save models
            device: Device to use for training ('cpu', 'cuda', or None for auto-detection)
        """
        self.base_model = base_model
        self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.base_model = self.base_model.to(self.device)
        
        logger.info(f"Initialized TransferLearningManager with device={self.device}")
    
    def pretrain(self, 
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                epochs: int = 10,
                learning_rate: float = 0.001,
                weight_decay: float = 1e-5,
                patience: int = 5,
                model_name: str = 'pretrained_model',
                loss_fn: Callable = None,
                optimizer: optim.Optimizer = None,
                scheduler: optim.lr_scheduler._LRScheduler = None) -> Dict:
        """Pretrain the base model on general market data.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            patience: Patience for early stopping
            model_name: Name for the pretrained model
            loss_fn: Loss function (defaults to MSELoss)
            optimizer: Optimizer (defaults to Adam)
            scheduler: Learning rate scheduler
            
        Returns:
            Dictionary with training history
        """
        # Set model to training mode
        self.base_model.train()
        
        # Define loss function if not provided
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        
        # Define optimizer if not provided
        if optimizer is None:
            optimizer = optim.Adam(
                self.base_model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        # Define scheduler if not provided
        if scheduler is None:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=patience // 2,
                verbose=True
            )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_epoch = 0
        no_improvement = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            self.base_model.train()
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.base_model(inputs)
                loss = loss_fn(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item()
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}")
            
            # Calculate average training loss
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss = 0.0
            self.base_model.eval()
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    # Move data to device
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass
                    outputs = self.base_model(inputs)
                    loss = loss_fn(outputs, targets)
                    
                    # Update statistics
                    val_loss += loss.item()
            
            # Calculate average validation loss
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
            
            # Update learning rate
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)
            
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Log epoch results
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improvement = 0
                
                # Save best model
                self.save_model(model_name)
                logger.info(f"Saved best model at epoch {epoch+1}")
            else:
                no_improvement += 1
                logger.info(f"No improvement for {no_improvement} epochs")
            
            # Early stopping
            if no_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"Pretraining completed. Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
        return history
    
    def fine_tune(self,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                freeze_layers: List[str] = None,
                epochs: int = 5,
                learning_rate: float = 0.0001,
                weight_decay: float = 1e-5,
                patience: int = 3,
                pretrained_model: str = 'pretrained_model',
                fine_tuned_model: str = 'fine_tuned_model',
                loss_fn: Callable = None,
                optimizer: optim.Optimizer = None,
                scheduler: optim.lr_scheduler._LRScheduler = None) -> Dict:
        """Fine-tune the pretrained model on specific asset data.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            freeze_layers: List of layer names to freeze during fine-tuning
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            patience: Patience for early stopping
            pretrained_model: Name of the pretrained model to load
            fine_tuned_model: Name for the fine-tuned model
            loss_fn: Loss function (defaults to MSELoss)
            optimizer: Optimizer (defaults to Adam)
            scheduler: Learning rate scheduler
            
        Returns:
            Dictionary with training history
        """
        # Load pretrained model
        self.load_model(pretrained_model)
        
        # Freeze specified layers
        if freeze_layers is not None:
            self._freeze_layers(freeze_layers)
        
        # Set model to training mode
        self.base_model.train()
        
        # Define loss function if not provided
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        
        # Define optimizer if not provided
        if optimizer is None:
            # Only optimize non-frozen parameters
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.base_model.parameters()),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        # Define scheduler if not provided
        if scheduler is None:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=patience // 2,
                verbose=True
            )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_epoch = 0
        no_improvement = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            self.base_model.train()
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.base_model(inputs)
                loss = loss_fn(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item()
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}")
            
            # Calculate average training loss
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss = 0.0
            self.base_model.eval()
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    # Move data to device
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass
                    outputs = self.base_model(inputs)
                    loss = loss_fn(outputs, targets)
                    
                    # Update statistics
                    val_loss += loss.item()
            
            # Calculate average validation loss
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
            
            # Update learning rate
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)
            
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Log epoch results
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improvement = 0
                
                # Save best model
                self.save_model(fine_tuned_model)
                logger.info(f"Saved best model at epoch {epoch+1}")
            else:
                no_improvement += 1
                logger.info(f"No improvement for {no_improvement} epochs")
            
            # Early stopping
            if no_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"Fine-tuning completed. Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
        return history
    
    def save_model(self, model_name: str) -> str:
        """Save the model to disk.
        
        Args:
            model_name: Name for the saved model
            
        Returns:
            Path to the saved model
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.pt")
        torch.save(self.base_model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, model_name: str) -> nn.Module:
        """Load a model from disk.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.pt")
        self.base_model.load_state_dict(torch.load(model_path, map_location=self.device))
        logger.info(f"Model loaded from {model_path}")
        return self.base_model
    
    def _freeze_layers(self, layer_names: List[str]) -> None:
        """Freeze specified layers in the model.
        
        Args:
            layer_names: List of layer names to freeze
        """
        for name, param in self.base_model.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = False
                    logger.info(f"Frozen layer: {name}")


class TransferDatasetCreator:
    """Class for creating datasets for transfer learning."""
    
    def __init__(self, 
                sequence_length: int = 60,
                forecast_horizon: int = 10,
                train_ratio: float = 0.7,
                val_ratio: float = 0.15,
                test_ratio: float = 0.15,
                batch_size: int = 32,
                shuffle: bool = True,
                num_workers: int = 4):
        """Initialize the transfer dataset creator.
        
        Args:
            sequence_length: Length of input sequences
            forecast_horizon: Number of steps to forecast
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle the data
            num_workers: Number of workers for DataLoader
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        logger.info(f"Initialized TransferDatasetCreator with sequence_length={sequence_length}, forecast_horizon={forecast_horizon}")
    
    def create_datasets(self, data: pd.DataFrame, target_column: str) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Create datasets for transfer learning.
        
        Args:
            data: Input data as pandas DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create sequences
        X, y = self._create_sequences(data, target_column)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(self.val_ratio + self.test_ratio), random_state=42
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.test_ratio / (self.val_ratio + self.test_ratio), random_state=42
        )
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )
        
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test), torch.FloatTensor(y_test)
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        logger.info(f"Created datasets with {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test samples")
        return train_loader, val_loader, test_loader
    
    def _create_sequences(self, data: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series forecasting.
        
        Args:
            data: Input data as pandas DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (X, y) as numpy arrays
        """
        X = []
        y = []
        
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            # Input sequence
            X.append(data.iloc[i:i+self.sequence_length].values)
            
            # Target sequence
            if target_column == 'all':
                y.append(data.iloc[i+self.sequence_length:i+self.sequence_length+self.forecast_horizon].values)
            else:
                y.append(data.iloc[i+self.sequence_length:i+self.sequence_length+self.forecast_horizon][target_column].values)
        
        return np.array(X), np.array(y)


class ModelAdapter:
    """Class for adapting models for transfer learning."""
    
    def __init__(self, base_model: nn.Module):
        """Initialize the model adapter.
        
        Args:
            base_model: Base model to adapt
        """
        self.base_model = base_model
        logger.info(f"Initialized ModelAdapter")
    
    def adapt_for_regression(self, output_dim: int) -> nn.Module:
        """Adapt the model for regression tasks.
        
        Args:
            output_dim: Dimension of the output
            
        Returns:
            Adapted model
        """
        # Get the output layer
        output_layer = list(self.base_model.children())[-1]
        
        # Replace the output layer
        if isinstance(output_layer, nn.Linear):
            in_features = output_layer.in_features
            new_output_layer = nn.Linear(in_features, output_dim)
            
            # Create a new model with the replaced output layer
            new_model = nn.Sequential(
                *list(self.base_model.children())[:-1],
                new_output_layer
            )
            
            logger.info(f"Adapted model for regression with output_dim={output_dim}")
            return new_model
        else:
            raise ValueError("Last layer is not a Linear layer")
    
    def adapt_for_classification(self, num_classes: int) -> nn.Module:
        """Adapt the model for classification tasks.
        
        Args:
            num_classes: Number of classes
            
        Returns:
            Adapted model
        """
        # Get the output layer
        output_layer = list(self.base_model.children())[-1]
        
        # Replace the output layer
        if isinstance(output_layer, nn.Linear):
            in_features = output_layer.in_features
            new_output_layer = nn.Linear(in_features, num_classes)
            
            # Create a new model with the replaced output layer
            new_model = nn.Sequential(
                *list(self.base_model.children())[:-1],
                new_output_layer,
                nn.Softmax(dim=1)
            )
            
            logger.info(f"Adapted model for classification with num_classes={num_classes}")
            return new_model
        else:
            raise ValueError("Last layer is not a Linear layer")
    
    def add_attention_layer(self) -> nn.Module:
        """Add an attention layer to the model.
        
        Returns:
            Model with attention layer
        """
        # Get the output layer
        output_layer = list(self.base_model.children())[-1]
        
        # Add attention layer before the output layer
        if isinstance(output_layer, nn.Linear):
            in_features = output_layer.in_features
            attention_layer = nn.MultiheadAttention(
                embed_dim=in_features,
                num_heads=8,
                batch_first=True
            )
            
            # Create a new model with the attention layer
            new_model = nn.Sequential(
                *list(self.base_model.children())[:-1],
                attention_layer,
                output_layer
            )
            
            logger.info(f"Added attention layer to model")
            return new_model
        else:
            raise ValueError("Last layer is not a Linear layer")


if __name__ == "__main__":
    import argparse
    from enhanced_dl_model_fixed import EnhancedPatternRecognitionModel
    
    parser = argparse.ArgumentParser(description='Transfer Learning Setup')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory for models')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory for data')
    parser.add_argument('--input_dim', type=int, default=9, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=3, help='Output dimension')
    parser.add_argument('--sequence_length', type=int, default=60, help='Sequence length')
    parser.add_argument('--forecast_horizon', type=int, default=10, help='Forecast horizon')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    # Create model
    model = EnhancedPatternRecognitionModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim
    )
    
    # Create transfer learning manager
    manager = TransferLearningManager(
        base_model=model,
        model_dir=args.model_dir
    )
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    
    # General market data for pretraining
    general_data = pd.DataFrame(
        np.random.randn(1000, args.input_dim),
        columns=[f'feature_{i}' for i in range(args.input_dim)]
    )
    general_data['target'] = general_data.sum(axis=1) + np.random.randn(1000) * 0.1
    
    # Specific asset data for fine-tuning
    specific_data = pd.DataFrame(
        np.random.randn(500, args.input_dim),
        columns=[f'feature_{i}' for i in range(args.input_dim)]
    )
    specific_data['target'] = specific_data.sum(axis=1) * 1.5 + np.random.randn(500) * 0.1
    
    # Create datasets
    dataset_creator = TransferDatasetCreator(
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size
    )
    
    general_train_loader, general_val_loader, general_test_loader = dataset_creator.create_datasets(
        general_data, 'target'
    )
    
    specific_train_loader, specific_val_loader, specific_test_loader = dataset_creator.create_datasets(
        specific_data, 'target'
    )
    
    # Pretrain on general market data
    pretrain_history = manager.pretrain(
        train_loader=general_train_loader,
        val_loader=general_val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_name='pretrained_model'
    )
    
    # Fine-tune on specific asset data
    fine_tune_history = manager.fine_tune(
        train_loader=specific_train_loader,
        val_loader=specific_val_loader,
        freeze_layers=['encoder'],
        epochs=args.epochs // 2,
        learning_rate=args.learning_rate / 10,
        pretrained_model='pretrained_model',
        fine_tuned_model='fine_tuned_model'
    )
    
    # Save history
    pd.DataFrame(pretrain_history).to_csv(f"{args.model_dir}/pretrain_history.csv", index=False)
    pd.DataFrame(fine_tune_history).to_csv(f"{args.model_dir}/fine_tune_history.csv", index=False)
    
    logger.info("Transfer learning setup completed")
