#!/usr/bin/env python
"""
Enhanced Feature Adapter for Deep Learning Pattern Recognition

This module provides an enhanced feature adapter for deep learning pattern recognition
in financial market data, with dynamic feature importance and market regime detection.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import OrderedDict
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_feature_adapter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_feature_adapter")

class EnhancedFeatureAdapter:
    """Enhanced feature adapter with dynamic feature importance and market regime detection"""
    
    def __init__(self, 
                 input_dim: int = 9, 
                 importance_method: str = "mutual_info", 
                 config_path: str = None,
                 cache_enabled: bool = True,
                 cache_size: int = 100):
        """Initialize enhanced feature adapter
        
        Args:
            input_dim: Number of input features required by the model
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
        
        # Initialize feature importance
        self.feature_importance = {}
        
        # Initialize market regime detector
        self.market_regimes = self.config.get("market_regimes", {
            "trending": {
                "features": ["momentum", "macd", "adx", "price_velocity"],
                "weight": 1.5
            },
            "ranging": {
                "features": ["bb_percent_b", "rsi", "stochastic", "mean_reversion"],
                "weight": 1.2
            },
            "volatile": {
                "features": ["volatility", "atr", "price_range", "volume_delta"],
                "weight": 1.3
            },
            "normal": {
                "features": ["price", "volume", "vwap", "open_close_ratio"],
                "weight": 1.0
            }
        })
        
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
            "default_features": [
                "price", "volume", "rsi", "macd", "bb_percent_b", 
                "volatility", "momentum", "order_imbalance", "spread"
            ],
            "feature_groups": {
                "price_based": ["open", "high", "low", "close", "price", "vwap"],
                "volume_based": ["volume", "volume_delta", "volume_ma", "volume_std"],
                "momentum": ["rsi", "macd", "macd_signal", "macd_hist", "momentum"],
                "volatility": ["volatility", "atr", "bb_width", "price_range"],
                "mean_reversion": ["bb_percent_b", "stochastic", "mean_reversion"],
                "trend": ["adx", "price_velocity", "trend_strength"],
                "liquidity": ["spread", "order_imbalance", "market_depth"],
                "time": ["hour_of_day", "day_of_week", "day_of_month", "month_of_year", "is_weekend"]
            },
            "market_regimes": {
                "trending": {
                    "features": ["momentum", "macd", "adx", "price_velocity"],
                    "weight": 1.5
                },
                "ranging": {
                    "features": ["bb_percent_b", "rsi", "stochastic", "mean_reversion"],
                    "weight": 1.2
                },
                "volatile": {
                    "features": ["volatility", "atr", "price_range", "volume_delta"],
                    "weight": 1.3
                },
                "normal": {
                    "features": ["price", "volume", "vwap", "open_close_ratio"],
                    "weight": 1.0
                }
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
            str: Market regime (trending, ranging, volatile, normal)
        """
        try:
            # Check if DataFrame is empty
            if df.empty:
                logger.warning("Empty DataFrame provided for market regime detection")
                return "normal"
            
            # Extract price data if available
            if "close" in df.columns:
                prices = df["close"].values
            elif "price" in df.columns:
                prices = df["price"].values
            else:
                # Use first numeric column as price
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        prices = df[col].values
                        break
                else:
                    logger.warning("No numeric columns found for market regime detection")
                    return "normal"
            
            # Calculate returns
            if len(prices) > 1:
                returns = np.diff(prices) / prices[:-1]
            else:
                logger.warning("Insufficient data for market regime detection")
                return "normal"
            
            # Calculate metrics
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Calculate momentum (price change over last N periods)
            if len(prices) > 20:
                momentum = (prices[-1] / prices[-20]) - 1
            else:
                momentum = 0
            
            # Calculate mean reversion tendency
            if len(prices) > 20:
                ma20 = np.mean(prices[-20:])
                mean_reversion = (ma20 - prices[-1]) / ma20
            else:
                mean_reversion = 0
            
            # Determine regime
            if volatility > 0.03:  # High volatility
                regime = "volatile"
            elif abs(momentum) > 0.05:  # Strong momentum
                regime = "trending"
            elif abs(mean_reversion) > 0.02:  # Mean reversion
                regime = "ranging"
            else:
                regime = "normal"
            
            logger.info(f"Detected market regime: {regime} (volatility={volatility:.4f}, "
                       f"momentum={momentum:.4f}, mean_reversion={mean_reversion:.4f})")
            
            return regime
        
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return "normal"
    
    def get_feature_importance(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance
        
        Args:
            X: Input data of shape (batch_size, sequence_length, num_features)
            feature_names: List of feature names
            
        Returns:
            dict: Feature importance scores
        """
        try:
            # Check if we have enough data
            if X.shape[0] == 0 or X.shape[2] == 0:
                logger.warning("Insufficient data for feature importance calculation")
                return {name: 1.0 for name in feature_names}
            
            # Calculate feature importance based on method
            if self.importance_method == "mutual_info":
                # Calculate variance of each feature
                variances = np.var(X, axis=(0, 1))
                
                # Calculate correlation matrix
                X_reshaped = X.reshape(-1, X.shape[2])
                corr_matrix = np.corrcoef(X_reshaped.T)
                
                # Calculate mutual information (approximated by correlation)
                importance = {}
                for i, name in enumerate(feature_names):
                    if i < len(variances):
                        # Combine variance and correlation
                        var_score = variances[i] / np.max(variances) if np.max(variances) > 0 else 0
                        corr_score = np.mean(np.abs(corr_matrix[i])) if i < len(corr_matrix) else 0
                        
                        # Final score
                        importance[name] = 0.7 * var_score + 0.3 * corr_score
                    else:
                        importance[name] = 0.0
            
            elif self.importance_method == "variance":
                # Calculate variance of each feature
                variances = np.var(X, axis=(0, 1))
                
                # Normalize
                if np.max(variances) > 0:
                    variances = variances / np.max(variances)
                
                # Create importance dictionary
                importance = {}
                for i, name in enumerate(feature_names):
                    if i < len(variances):
                        importance[name] = variances[i]
                    else:
                        importance[name] = 0.0
            
            else:
                # Default: equal importance
                importance = {name: 1.0 for name in feature_names}
            
            # Ensure all features have importance scores
            for name in feature_names:
                if name not in importance:
                    importance[name] = 0.0
            
            return importance
        
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {name: 1.0 for name in feature_names}
    
    def adapt_features(self, 
                       X: np.ndarray, 
                       feature_names: List[str], 
                       market_regime: str = None) -> Tuple[np.ndarray, List[str]]:
        """Adapt features based on importance and market regime
        
        Args:
            X: Input data of shape (batch_size, sequence_length, num_features)
            feature_names: List of feature names
            market_regime: Market regime (trending, ranging, volatile, normal)
            
        Returns:
            Tuple[np.ndarray, List[str]]: Adapted features and selected feature names
        """
        try:
            # Check if X is empty
            if X.shape[0] == 0 or X.shape[2] == 0:
                logger.warning("Empty input data for feature adaptation")
                # Return empty array with correct dimensions
                return np.zeros((X.shape[0], X.shape[1], self.input_dim)), []
            
            # Check cache first if enabled
            if self.cache_enabled:
                # Create cache key from feature names and market regime
                cache_key = (tuple(feature_names), market_regime)
                
                if cache_key in self.feature_cache:
                    logger.info("Using cached feature adaptation")
                    indices, selected_features = self.feature_cache[cache_key]
                    
                    # Move to front of cache (most recently used)
                    self.feature_cache.move_to_end(cache_key)
                    
                    # Check if indices are valid
                    if max(indices, default=0) < X.shape[2]:
                        # Select features using cached indices
                        X_adapted = X[:, :, indices]
                        return X_adapted, selected_features
            
            # Detect market regime if not provided
            if market_regime is None:
                # Create DataFrame from X for regime detection
                # Use last timestep for simplicity
                df = pd.DataFrame({
                    name: X[0, -1, i] for i, name in enumerate(feature_names) if i < X.shape[2]
                })
                
                market_regime = self.detect_market_regime(df)
            
            # Calculate feature importance
            importance = self.get_feature_importance(X, feature_names)
            
            # Adjust importance based on market regime
            if market_regime in self.market_regimes:
                regime_features = self.market_regimes[market_regime]["features"]
                regime_weight = self.market_regimes[market_regime]["weight"]
                
                for feature in regime_features:
                    if feature in importance:
                        importance[feature] *= regime_weight
            
            # Sort features by importance
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            # Select top features
            num_features = min(self.input_dim, len(sorted_features))
            selected_features = [f[0] for f in sorted_features[:num_features]]
            
            # Get indices of selected features
            indices = []
            for feature in selected_features:
                if feature in feature_names:
                    idx = feature_names.index(feature)
                    if idx < X.shape[2]:
                        indices.append(idx)
            
            # Handle case where we don't have enough features
            if len(indices) < self.input_dim:
                logger.warning(f"Only {len(indices)} features selected, but {self.input_dim} required")
                
                # Add remaining features in order
                for i in range(min(X.shape[2], len(feature_names))):
                    if i not in indices and len(indices) < self.input_dim:
                        indices.append(i)
                        if i < len(feature_names):
                            selected_features.append(feature_names[i])
                
                # If still not enough, pad with zeros
                if len(indices) < self.input_dim:
                    # Create adapted array with zeros
                    X_adapted = np.zeros((X.shape[0], X.shape[1], self.input_dim))
                    
                    # Fill with available features
                    for i, idx in enumerate(indices):
                        if idx < X.shape[2]:
                            X_adapted[:, :, i] = X[:, :, idx]
                    
                    # Cache result if enabled
                    if self.cache_enabled:
                        # Manage cache size
                        if len(self.feature_cache) >= self.cache_size:
                            # Remove oldest item (first item in OrderedDict)
                            self.feature_cache.popitem(last=False)
                        
                        # Add to cache
                        self.feature_cache[cache_key] = (indices, selected_features)
                    
                    logger.info(f"Selected features: {selected_features}")
                    return X_adapted, selected_features
            
            # Select features
            X_adapted = X[:, :, indices]
            
            # Handle case where we have fewer features than required
            if X_adapted.shape[2] < self.input_dim:
                # Pad with zeros
                padding = np.zeros((X.shape[0], X.shape[1], self.input_dim - X_adapted.shape[2]))
                X_adapted = np.concatenate([X_adapted, padding], axis=2)
            
            # Handle case where we have more features than required
            elif X_adapted.shape[2] > self.input_dim:
                X_adapted = X_adapted[:, :, :self.input_dim]
                selected_features = selected_features[:self.input_dim]
            
            # Cache result if enabled
            if self.cache_enabled:
                # Manage cache size
                if len(self.feature_cache) >= self.cache_size:
                    # Remove oldest item (first item in OrderedDict)
                    self.feature_cache.popitem(last=False)
                
                # Add to cache
                self.feature_cache[cache_key] = (indices, selected_features)
            
            logger.info(f"Selected features: {selected_features}")
            return X_adapted, selected_features
        
        except Exception as e:
            logger.error(f"Error adapting features: {str(e)}")
            
            # Return empty array with correct dimensions
            X_adapted = np.zeros((X.shape[0], X.shape[1], self.input_dim))
            return X_adapted, []

# Example usage
if __name__ == "__main__":
    # Create adapter
    adapter = EnhancedFeatureAdapter(
        input_dim=9,
        importance_method="mutual_info",
        cache_enabled=True
    )
    
    # Create sample data
    batch_size = 16
    sequence_length = 60
    num_features = 27
    
    X = np.random.randn(batch_size, sequence_length, num_features)
    feature_names = [f"feature_{i}" for i in range(num_features)]
    
    # Adapt features
    X_adapted, selected_features = adapter.adapt_features(X, feature_names)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {X_adapted.shape}")
    print(f"Selected features: {selected_features}")
    
    # Test caching
    start_time = time.time()
    X_adapted2, selected_features2 = adapter.adapt_features(X, feature_names)
    cache_time = time.time() - start_time
    
    print(f"Cache time: {cache_time:.6f}s")
    print(f"Features match: {selected_features == selected_features2}")
