#!/usr/bin/env python
"""
Enhanced Feature Adapter for Deep Learning Models

This module provides enhanced feature adaptation for deep learning models,
with improved feature selection and transformation capabilities.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.feature_selection import mutual_info_regression, f_regression
from collections import OrderedDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_feature_adapter_fixed.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_feature_adapter_fixed")

class EnhancedFeatureAdapter:
    """Enhanced feature adapter for deep learning models"""
    
    def __init__(self, 
                 input_dim: int = 9,
                 importance_method: str = "mutual_info",
                 config_path: str = None,
                 cache_enabled: bool = True,
                 cache_size: int = 100):
        """Initialize enhanced feature adapter
        
        Args:
            input_dim: Number of input features to select
            importance_method: Method for feature importance calculation
            config_path: Path to configuration file
            cache_enabled: Whether to enable caching
            cache_size: Maximum number of items in cache
        """
        self.input_dim = input_dim
        self.importance_method = importance_method
        self.config_path = config_path
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size
        
        # Initialize cache
        self.feature_cache = OrderedDict()
        
        # Load configuration
        self.config = self._load_config()
        
        # Update parameters from config
        self.input_dim = self.config.get("input_dim", self.input_dim)
        self.importance_method = self.config.get("importance_method", self.importance_method)
        self.cache_enabled = self.config.get("cache_enabled", self.cache_enabled)
        self.cache_size = self.config.get("cache_size", self.cache_size)
        
        # Feature importance scores
        self.feature_importance = {}
        
        # Market regime detection thresholds
        self.volatility_threshold = self.config.get("volatility_threshold", 0.02)
        self.momentum_threshold = self.config.get("momentum_threshold", 0.01)
        
        logger.info(f"Initialized EnhancedFeatureAdapter with input_dim={input_dim}, "
                   f"importance_method={importance_method}")
    
    def _load_config(self) -> Dict:
        """Load configuration from file or use defaults
        
        Returns:
            dict: Configuration dictionary
        """
        default_config = {
            "input_dim": self.input_dim,
            "importance_method": self.importance_method,
            "cache_enabled": self.cache_enabled,
            "cache_size": self.cache_size,
            "volatility_threshold": 0.02,
            "momentum_threshold": 0.01,
            "feature_groups": {
                "price": ["close", "open", "high", "low"],
                "volume": ["volume"],
                "technical": ["rsi", "macd", "bollinger_upper", "bollinger_lower", "atr"],
                "derived": ["momentum", "volatility", "trend"]
            }
        }
        
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    for key, value in loaded_config.items():
                        default_config[key] = value
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
                logger.info("Using default configuration")
        
        return default_config
    
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """Detect market regime from data
        
        Args:
            df: Market data DataFrame
            
        Returns:
            str: Market regime (high_volatility, trending, ranging)
        """
        try:
            # Check if DataFrame is empty
            if df is None or df.empty:
                logger.warning("Empty DataFrame provided for market regime detection")
                return "ranging"  # Default to ranging
            
            # Calculate volatility (standard deviation of returns)
            if 'close' in df.columns:
                close_prices = df['close'].values
                returns = np.diff(close_prices) / close_prices[:-1]
                volatility = np.std(returns)
            else:
                # Try to find any price column
                price_cols = [col for col in df.columns if col in ['close', 'price', 'mid']]
                if price_cols:
                    prices = df[price_cols[0]].values
                    returns = np.diff(prices) / prices[:-1]
                    volatility = np.std(returns)
                else:
                    logger.warning("No price column found for volatility calculation")
                    volatility = 0.0
            
            # Calculate momentum (rate of change)
            if len(close_prices) >= 10:
                momentum = (close_prices[-1] - close_prices[-10]) / close_prices[-10]
            else:
                momentum = 0.0
            
            # Determine market regime
            if volatility > self.volatility_threshold:
                regime = "high_volatility"
            elif abs(momentum) > self.momentum_threshold:
                regime = "trending"
            else:
                regime = "ranging"
            
            logger.info(f"Detected market regime: {regime} (volatility={volatility:.4f}, momentum={momentum:.4f})")
            return regime
        
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return "ranging"  # Default to ranging
    
    def adapt_features(self, X: np.ndarray, feature_names: List[str], market_regime: str = None) -> Tuple[np.ndarray, List[str]]:
        """Adapt features for deep learning model
        
        Args:
            X: Input features [batch_size, sequence_length, feature_dim]
            feature_names: List of feature names
            market_regime: Market regime (high_volatility, trending, ranging)
            
        Returns:
            tuple: (Adapted features, Selected feature names)
        """
        try:
            # Check if X is empty
            if X is None or X.shape[0] == 0 or X.shape[2] == 0:
                logger.warning("Empty input provided for feature adaptation")
                return X, feature_names
            
            # Check cache first if enabled
            if self.cache_enabled:
                # Create cache key from X shape, feature names, and market regime
                cache_key = (X.shape, tuple(feature_names), market_regime)
                
                if cache_key in self.feature_cache:
                    logger.info("Using cached feature adaptation")
                    return self.feature_cache[cache_key]
            
            # Detect market regime if not provided
            if market_regime is None:
                # Create a simple DataFrame from the last timestep
                df = pd.DataFrame({
                    name: X[0, -1, i] for i, name in enumerate(feature_names)
                }, index=[0])
                market_regime = self.detect_market_regime(df)
            
            # Calculate feature importance
            importance_scores = self._calculate_feature_importance(X, feature_names)
            
            # Select features based on market regime and importance
            selected_indices = self._select_features(importance_scores, feature_names, market_regime)
            
            # Extract selected features
            X_adapted = X[:, :, selected_indices]
            selected_features = [feature_names[i] for i in selected_indices]
            
            # Store in cache if enabled
            if self.cache_enabled:
                # Add to cache
                self.feature_cache[cache_key] = (X_adapted, selected_features)
                
                # Limit cache size
                while len(self.feature_cache) > self.cache_size:
                    self.feature_cache.popitem(last=False)
            
            logger.info(f"Adapted features from {X.shape[2]} to {X_adapted.shape[2]} dimensions")
            return X_adapted, selected_features
        
        except Exception as e:
            logger.error(f"Error adapting features: {str(e)}")
            return X, feature_names
    
    def _calculate_feature_importance(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance
        
        Args:
            X: Input features [batch_size, sequence_length, feature_dim]
            feature_names: List of feature names
            
        Returns:
            dict: Feature importance scores
        """
        try:
            # Reshape X for feature importance calculation
            # Use the last timestep for each sequence
            X_last = X[:, -1, :]  # [batch_size, feature_dim]
            
            # Create target variable (use next timestep of first feature as proxy)
            if X.shape[1] > 1:
                y = X[:, -1, 0]  # [batch_size]
            else:
                # If only one timestep, use random target
                y = np.random.randn(X.shape[0])
            
            # Calculate feature importance
            if self.importance_method == "mutual_info":
                importance = mutual_info_regression(X_last, y)
            elif self.importance_method == "f_regression":
                importance, _ = f_regression(X_last, y)
            else:
                # Default to equal importance
                importance = np.ones(X.shape[2])
            
            # Create dictionary of feature importance
            importance_dict = {
                feature_names[i]: float(importance[i])
                for i in range(len(feature_names))
            }
            
            # Store feature importance
            self.feature_importance = importance_dict
            
            return importance_dict
        
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            # Return equal importance
            return {name: 1.0 for name in feature_names}
    
    def _select_features(self, importance_scores: Dict[str, float], feature_names: List[str], market_regime: str) -> List[int]:
        """Select features based on importance and market regime
        
        Args:
            importance_scores: Feature importance scores
            feature_names: List of feature names
            market_regime: Market regime
            
        Returns:
            list: Indices of selected features
        """
        try:
            # Get feature groups from config
            feature_groups = self.config.get("feature_groups", {})
            
            # Determine feature allocation based on market regime
            if market_regime == "high_volatility":
                # In high volatility, prioritize technical indicators and derived metrics
                group_weights = {
                    "price": 0.2,
                    "volume": 0.2,
                    "technical": 0.4,
                    "derived": 0.2
                }
            elif market_regime == "trending":
                # In trending markets, prioritize momentum and price
                group_weights = {
                    "price": 0.3,
                    "volume": 0.1,
                    "technical": 0.3,
                    "derived": 0.3
                }
            else:  # ranging
                # In ranging markets, prioritize technical indicators
                group_weights = {
                    "price": 0.2,
                    "volume": 0.1,
                    "technical": 0.5,
                    "derived": 0.2
                }
            
            # Calculate number of features to select from each group
            total_features = min(self.input_dim, len(feature_names))
            group_allocation = {}
            
            for group, weight in group_weights.items():
                group_allocation[group] = max(1, int(total_features * weight))
            
            # Adjust allocation to match total_features
            while sum(group_allocation.values()) > total_features:
                # Find group with highest allocation and reduce by 1
                max_group = max(group_allocation.items(), key=lambda x: x[1])[0]
                group_allocation[max_group] -= 1
            
            while sum(group_allocation.values()) < total_features:
                # Find group with lowest allocation and increase by 1
                min_group = min(group_allocation.items(), key=lambda x: x[1])[0]
                group_allocation[min_group] += 1
            
            # Select features from each group based on importance
            selected_indices = []
            
            for group, count in group_allocation.items():
                # Get features in this group
                group_features = feature_groups.get(group, [])
                
                # Find indices of features in this group
                group_indices = [
                    i for i, name in enumerate(feature_names)
                    if any(gf in name for gf in group_features)
                ]
                
                # If not enough features in this group, continue
                if len(group_indices) == 0:
                    continue
                
                # Sort by importance
                group_importance = {
                    i: importance_scores.get(feature_names[i], 0.0)
                    for i in group_indices
                }
                
                sorted_indices = sorted(
                    group_indices,
                    key=lambda i: group_importance[i],
                    reverse=True
                )
                
                # Select top features
                selected_indices.extend(sorted_indices[:count])
            
            # If we still don't have enough features, add more based on importance
            if len(selected_indices) < total_features:
                # Get indices not already selected
                remaining_indices = [
                    i for i in range(len(feature_names))
                    if i not in selected_indices
                ]
                
                # Sort by importance
                remaining_importance = {
                    i: importance_scores.get(feature_names[i], 0.0)
                    for i in remaining_indices
                }
                
                sorted_indices = sorted(
                    remaining_indices,
                    key=lambda i: remaining_importance[i],
                    reverse=True
                )
                
                # Add more features
                selected_indices.extend(
                    sorted_indices[:total_features - len(selected_indices)]
                )
            
            # Ensure we don't exceed the input_dim
            selected_indices = selected_indices[:total_features]
            
            logger.info(f"Selected {len(selected_indices)} features for market regime {market_regime}")
            return selected_indices
        
        except Exception as e:
            logger.error(f"Error selecting features: {str(e)}")
            # Return first input_dim features
            return list(range(min(self.input_dim, len(feature_names))))
    
    # Add transform method for compatibility with test scripts
    def transform(self, X: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        """Transform features for deep learning model (compatibility method)
        
        Args:
            X: Input features [batch_size, sequence_length, feature_dim]
            feature_names: List of feature names (optional)
            
        Returns:
            np.ndarray: Transformed features
        """
        try:
            # If feature names not provided, create generic names
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[2])]
            
            # Adapt features
            X_adapted, _ = self.adapt_features(X, feature_names)
            
            logger.info(f"Transformed features from {X.shape} to {X_adapted.shape}")
            return X_adapted
        
        except Exception as e:
            logger.error(f"Error transforming features: {str(e)}")
            return X
