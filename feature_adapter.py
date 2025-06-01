#!/usr/bin/env python
"""
Feature Adapter for Deep Learning Pattern Recognition

This module provides a feature adapter to ensure compatibility between
the data pipeline and the model by managing feature dimensionality.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_adapter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("feature_adapter")

class FeatureAdapter:
    """Adapter to ensure feature compatibility between data pipeline and model"""
    
    def __init__(self, 
                 input_dim: int = 9,
                 feature_selection: List[str] = None,
                 config_path: str = None):
        """Initialize feature adapter
        
        Args:
            input_dim: Expected input dimension for the model
            feature_selection: List of feature names to select (if None, will select first input_dim features)
            config_path: Path to configuration file
        """
        self.input_dim = input_dim
        self.feature_selection = feature_selection
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Update feature selection from config if not provided
        if self.feature_selection is None and "feature_selection" in self.config:
            self.feature_selection = self.config["feature_selection"]
        
        logger.info(f"Initialized FeatureAdapter with input_dim={input_dim}, feature_selection={feature_selection}")
    
    def _load_config(self) -> Dict:
        """Load configuration from file or use defaults
        
        Returns:
            dict: Configuration dictionary
        """
        default_config = {
            "input_dim": self.input_dim,
            "feature_selection": [
                "close", "volume", "rsi", "macd", "bb_percent_b", 
                "volatility", "momentum", "order_imbalance", "spread"
            ],
            "feature_importance": {
                "close": 1.0,
                "volume": 0.9,
                "rsi": 0.8,
                "macd": 0.8,
                "bb_percent_b": 0.7,
                "volatility": 0.7,
                "momentum": 0.6,
                "order_imbalance": 0.6,
                "spread": 0.5
            }
        }
        
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    for key, value in loaded_config.items():
                        if key in default_config:
                            if isinstance(value, dict) and isinstance(default_config[key], dict):
                                default_config[key].update(value)
                            else:
                                default_config[key] = value
                        else:
                            default_config[key] = value
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
                logger.info("Using default configuration")
        
        # Update instance variables from config
        self.input_dim = default_config["input_dim"]
        
        return default_config
    
    def adapt_features(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Adapt features to match expected input dimension
        
        Args:
            X: Input features of shape (batch_size, sequence_length, n_features)
            feature_names: List of feature names
            
        Returns:
            tuple: (adapted_X, selected_feature_names)
        """
        # Check if adaptation is needed
        if X.shape[2] == self.input_dim:
            logger.info("No adaptation needed, feature dimensions already match")
            return X, feature_names
        
        # Check if we have feature selection
        if self.feature_selection:
            # Select features by name
            selected_indices = []
            selected_names = []
            
            for name in self.feature_selection:
                if name in feature_names:
                    idx = feature_names.index(name)
                    selected_indices.append(idx)
                    selected_names.append(name)
                else:
                    logger.warning(f"Feature '{name}' not found in input features")
            
            # Check if we have enough features
            if len(selected_indices) < self.input_dim:
                logger.warning(f"Only {len(selected_indices)} features selected, but {self.input_dim} required")
                # Add more features if needed
                for i, name in enumerate(feature_names):
                    if i not in selected_indices and len(selected_indices) < self.input_dim:
                        selected_indices.append(i)
                        selected_names.append(name)
            
            # Select features
            adapted_X = X[:, :, selected_indices[:self.input_dim]]
            selected_feature_names = selected_names[:self.input_dim]
            
            logger.info(f"Adapted features from {X.shape[2]} to {adapted_X.shape[2]} dimensions")
            logger.info(f"Selected features: {selected_feature_names}")
            
            return adapted_X, selected_feature_names
        else:
            # Select first input_dim features
            adapted_X = X[:, :, :self.input_dim]
            selected_feature_names = feature_names[:self.input_dim]
            
            logger.info(f"Adapted features from {X.shape[2]} to {adapted_X.shape[2]} dimensions")
            logger.info(f"Selected features: {selected_feature_names}")
            
            return adapted_X, selected_feature_names
    
    def adapt_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adapt DataFrame to include only selected features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Adapted DataFrame
        """
        # Check if we have feature selection
        if self.feature_selection:
            # Check which features exist in the DataFrame
            available_features = [f for f in self.feature_selection if f in df.columns]
            
            # Check if we have enough features
            if len(available_features) < self.input_dim:
                logger.warning(f"Only {len(available_features)} features available, but {self.input_dim} required")
                # Add more features if needed
                for col in df.columns:
                    if col not in available_features and len(available_features) < self.input_dim:
                        available_features.append(col)
            
            # Select features
            selected_features = available_features[:self.input_dim]
            
            # Keep timestamp column if it exists
            if "timestamp" in df.columns and "timestamp" not in selected_features:
                selected_features = ["timestamp"] + selected_features
            
            # Select columns
            adapted_df = df[selected_features]
            
            logger.info(f"Adapted DataFrame from {len(df.columns)} to {len(adapted_df.columns)} columns")
            logger.info(f"Selected columns: {adapted_df.columns.tolist()}")
            
            return adapted_df
        else:
            # Select first input_dim features plus timestamp if it exists
            selected_features = df.columns[:self.input_dim].tolist()
            
            # Keep timestamp column if it exists
            if "timestamp" in df.columns and "timestamp" not in selected_features:
                selected_features = ["timestamp"] + selected_features[:self.input_dim-1]
            
            # Select columns
            adapted_df = df[selected_features]
            
            logger.info(f"Adapted DataFrame from {len(df.columns)} to {len(adapted_df.columns)} columns")
            logger.info(f"Selected columns: {adapted_df.columns.tolist()}")
            
            return adapted_df

# Example usage
if __name__ == "__main__":
    # Create adapter
    adapter = FeatureAdapter(input_dim=9)
    
    # Create sample data
    X = np.random.randn(10, 30, 27)
    feature_names = [f"feature_{i}" for i in range(27)]
    
    # Adapt features
    adapted_X, selected_features = adapter.adapt_features(X, feature_names)
    
    print(f"Original shape: {X.shape}")
    print(f"Adapted shape: {adapted_X.shape}")
    print(f"Selected features: {selected_features}")
