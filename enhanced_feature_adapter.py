#!/usr/bin/env python
"""
Enhanced Feature Adapter with Dynamic Feature Importance

This module provides an enhanced feature adapter with dynamic feature importance
scoring to ensure optimal feature selection for the deep learning model.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler

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
    """Enhanced adapter with dynamic feature importance for optimal feature selection"""
    
    def __init__(self, 
                 input_dim: int = 9,
                 feature_selection: List[str] = None,
                 importance_method: str = "mutual_info",
                 config_path: str = None,
                 cache_enabled: bool = True,
                 cache_size: int = 100):
        """Initialize enhanced feature adapter
        
        Args:
            input_dim: Expected input dimension for the model
            feature_selection: List of feature names to select (if None, will use dynamic selection)
            importance_method: Method for feature importance calculation ('mutual_info', 'correlation', 'gradient')
            config_path: Path to configuration file
            cache_enabled: Whether to enable feature caching
            cache_size: Maximum number of items in cache
        """
        self.input_dim = input_dim
        self.feature_selection = feature_selection
        self.importance_method = importance_method
        self.config_path = config_path
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size
        
        # Initialize cache
        self.feature_cache = {}
        self.importance_cache = {}
        
        # Load configuration
        self.config = self._load_config()
        
        # Update feature selection from config if not provided
        if self.feature_selection is None and "feature_selection" in self.config:
            self.feature_selection = self.config["feature_selection"]
        
        # Initialize feature importance
        self.feature_importance = self.config.get("feature_importance", {})
        
        logger.info(f"Initialized EnhancedFeatureAdapter with input_dim={input_dim}, "
                   f"importance_method={importance_method}, cache_enabled={cache_enabled}")
    
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
            },
            "importance_method": self.importance_method,
            "cache_enabled": self.cache_enabled,
            "cache_size": self.cache_size,
            "market_regime_weights": {
                "trending": {
                    "momentum": 1.5,
                    "macd": 1.3,
                    "rsi": 1.2
                },
                "ranging": {
                    "bb_percent_b": 1.5,
                    "volatility": 1.3,
                    "order_imbalance": 1.2
                },
                "volatile": {
                    "volatility": 1.8,
                    "atr": 1.5,
                    "spread": 1.3
                }
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
        self.importance_method = default_config["importance_method"]
        self.cache_enabled = default_config["cache_enabled"]
        self.cache_size = default_config["cache_size"]
        
        return default_config
    
    def _calculate_feature_importance(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance using specified method
        
        Args:
            X: Input features of shape (batch_size, sequence_length, n_features)
            feature_names: List of feature names
            
        Returns:
            dict: Feature importance scores
        """
        # Check cache first if enabled
        if self.cache_enabled:
            cache_key = hash(X.tobytes())
            if cache_key in self.importance_cache:
                logger.info("Using cached feature importance")
                return self.importance_cache[cache_key]
        
        # Reshape X for importance calculation
        # Take the last time step for each sample
        X_last = X[:, -1, :]
        
        # Calculate target (simple future return prediction for importance calculation)
        y = np.mean(X[:, 1:, 0] - X[:, :-1, 0], axis=1)  # Mean of price changes
        
        importance_scores = {}
        
        if self.importance_method == "mutual_info":
            # Calculate mutual information
            try:
                mi_scores = mutual_info_regression(X_last, y)
                # Normalize scores
                if np.sum(mi_scores) > 0:
                    mi_scores = mi_scores / np.sum(mi_scores)
                
                # Create dictionary
                for i, name in enumerate(feature_names):
                    importance_scores[name] = float(mi_scores[i])
            except Exception as e:
                logger.error(f"Error calculating mutual information: {str(e)}")
                # Fall back to default importance
                importance_scores = {name: self.feature_importance.get(name, 0.5) for name in feature_names}
        
        elif self.importance_method == "correlation":
            # Calculate correlation with target
            try:
                corr_scores = np.zeros(X_last.shape[1])
                for i in range(X_last.shape[1]):
                    corr = np.corrcoef(X_last[:, i], y)[0, 1]
                    corr_scores[i] = abs(corr)  # Use absolute correlation
                
                # Normalize scores
                if np.sum(corr_scores) > 0:
                    corr_scores = corr_scores / np.sum(corr_scores)
                
                # Create dictionary
                for i, name in enumerate(feature_names):
                    importance_scores[name] = float(corr_scores[i])
            except Exception as e:
                logger.error(f"Error calculating correlation: {str(e)}")
                # Fall back to default importance
                importance_scores = {name: self.feature_importance.get(name, 0.5) for name in feature_names}
        
        else:
            # Use default importance
            importance_scores = {name: self.feature_importance.get(name, 0.5) for name in feature_names}
        
        # Cache results if enabled
        if self.cache_enabled:
            # Manage cache size
            if len(self.importance_cache) >= self.cache_size:
                # Remove oldest item
                oldest_key = next(iter(self.importance_cache))
                del self.importance_cache[oldest_key]
            
            # Add to cache
            self.importance_cache[cache_key] = importance_scores
        
        return importance_scores
    
    def _adjust_importance_for_market_regime(self, importance_scores: Dict[str, float], 
                                            market_regime: str = "normal") -> Dict[str, float]:
        """Adjust feature importance based on market regime
        
        Args:
            importance_scores: Feature importance scores
            market_regime: Current market regime ('trending', 'ranging', 'volatile', 'normal')
            
        Returns:
            dict: Adjusted feature importance scores
        """
        # Get regime weights
        regime_weights = self.config.get("market_regime_weights", {}).get(market_regime, {})
        
        if not regime_weights:
            return importance_scores
        
        # Adjust scores
        adjusted_scores = importance_scores.copy()
        
        for feature, weight in regime_weights.items():
            if feature in adjusted_scores:
                adjusted_scores[feature] *= weight
        
        # Normalize scores
        total = sum(adjusted_scores.values())
        if total > 0:
            for feature in adjusted_scores:
                adjusted_scores[feature] /= total
        
        return adjusted_scores
    
    def adapt_features(self, X: np.ndarray, feature_names: List[str], 
                      market_regime: str = "normal") -> Tuple[np.ndarray, List[str]]:
        """Adapt features to match expected input dimension with dynamic importance
        
        Args:
            X: Input features of shape (batch_size, sequence_length, n_features)
            feature_names: List of feature names
            market_regime: Current market regime ('trending', 'ranging', 'volatile', 'normal')
            
        Returns:
            tuple: (adapted_X, selected_feature_names)
        """
        # Check cache first if enabled
        if self.cache_enabled:
            cache_key = (hash(X.tobytes()), market_regime)
            if cache_key in self.feature_cache:
                logger.info("Using cached feature adaptation")
                return self.feature_cache[cache_key]
        
        # Check if adaptation is needed
        if X.shape[2] == self.input_dim:
            logger.info("No adaptation needed, feature dimensions already match")
            return X, feature_names
        
        # Calculate feature importance
        importance_scores = self._calculate_feature_importance(X, feature_names)
        
        # Adjust for market regime
        adjusted_scores = self._adjust_importance_for_market_regime(importance_scores, market_regime)
        
        # Sort features by importance
        sorted_features = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top features
        selected_names = [f[0] for f in sorted_features[:self.input_dim]]
        selected_indices = [feature_names.index(name) for name in selected_names if name in feature_names]
        
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
        
        # Cache results if enabled
        if self.cache_enabled:
            # Manage cache size
            if len(self.feature_cache) >= self.cache_size:
                # Remove oldest item
                oldest_key = next(iter(self.feature_cache))
                del self.feature_cache[oldest_key]
            
            # Add to cache
            self.feature_cache[cache_key] = (adapted_X, selected_feature_names)
        
        return adapted_X, selected_feature_names
    
    def adapt_dataframe(self, df: pd.DataFrame, market_regime: str = "normal") -> pd.DataFrame:
        """Adapt DataFrame to include only selected features with dynamic importance
        
        Args:
            df: Input DataFrame
            market_regime: Current market regime ('trending', 'ranging', 'volatile', 'normal')
            
        Returns:
            DataFrame: Adapted DataFrame
        """
        # Calculate feature importance if we have enough data
        if len(df) > 10:
            # Convert DataFrame to numpy array for importance calculation
            X = df.select_dtypes(include=[np.number]).values
            X = X.reshape(1, len(df), -1)  # Add batch dimension
            feature_names = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Calculate importance
            importance_scores = self._calculate_feature_importance(X, feature_names)
            
            # Adjust for market regime
            adjusted_scores = self._adjust_importance_for_market_regime(importance_scores, market_regime)
            
            # Sort features by importance
            sorted_features = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select top features
            selected_names = [f[0] for f in sorted_features[:self.input_dim]]
            
            # Check which features exist in the DataFrame
            available_features = [f for f in selected_names if f in df.columns]
            
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
                selected_features = ["timestamp"] + selected_features[:self.input_dim-1]
            
            # Select columns
            adapted_df = df[selected_features]
            
            logger.info(f"Adapted DataFrame from {len(df.columns)} to {len(adapted_df.columns)} columns")
            logger.info(f"Selected columns: {adapted_df.columns.tolist()}")
            
            return adapted_df
        else:
            # Not enough data for importance calculation, use default selection
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
    
    def get_feature_importance(self, X: np.ndarray, feature_names: List[str], 
                              market_regime: str = "normal") -> Dict[str, float]:
        """Get feature importance scores
        
        Args:
            X: Input features of shape (batch_size, sequence_length, n_features)
            feature_names: List of feature names
            market_regime: Current market regime ('trending', 'ranging', 'volatile', 'normal')
            
        Returns:
            dict: Feature importance scores
        """
        # Calculate importance
        importance_scores = self._calculate_feature_importance(X, feature_names)
        
        # Adjust for market regime
        adjusted_scores = self._adjust_importance_for_market_regime(importance_scores, market_regime)
        
        return adjusted_scores
    
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """Detect market regime from data
        
        Args:
            df: Input DataFrame with market data
            
        Returns:
            str: Detected market regime ('trending', 'ranging', 'volatile', 'normal')
        """
        try:
            # Check if we have enough data
            if len(df) < 20:
                return "normal"
            
            # Check if we have required columns
            required_cols = ["close", "high", "low"]
            if not all(col in df.columns for col in required_cols):
                return "normal"
            
            # Calculate volatility (using ATR-like measure)
            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift(1))
            low_close = np.abs(df["low"] - df["close"].shift(1))
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            
            # Calculate trend strength (using ADX-like measure)
            close = df["close"]
            price_change = close.diff(1)
            up_move = price_change.copy()
            down_move = price_change.copy()
            up_move[up_move < 0] = 0
            down_move[down_move > 0] = 0
            down_move = abs(down_move)
            
            avg_up = up_move.rolling(14).mean().iloc[-1]
            avg_down = down_move.rolling(14).mean().iloc[-1]
            
            if avg_up + avg_down == 0:
                trend_strength = 0
            else:
                trend_strength = 100 * abs(avg_up - avg_down) / (avg_up + avg_down)
            
            # Calculate range-bound indicator (using BB width)
            std = close.rolling(20).std().iloc[-1]
            mean = close.rolling(20).mean().iloc[-1]
            bb_width = 2 * std / mean
            
            # Determine regime
            volatility_threshold = 0.015  # 1.5% ATR/price
            trend_threshold = 25  # ADX-like threshold
            range_threshold = 0.05  # 5% BB width/price
            
            price = close.iloc[-1]
            normalized_atr = atr / price
            normalized_bb_width = bb_width
            
            if normalized_atr > volatility_threshold:
                return "volatile"
            elif trend_strength > trend_threshold:
                return "trending"
            elif normalized_bb_width < range_threshold:
                return "ranging"
            else:
                return "normal"
        
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return "normal"

# Example usage
if __name__ == "__main__":
    # Create adapter
    adapter = EnhancedFeatureAdapter(input_dim=9, importance_method="mutual_info")
    
    # Create sample data
    X = np.random.randn(10, 30, 27)
    feature_names = [f"feature_{i}" for i in range(27)]
    
    # Adapt features
    adapted_X, selected_features = adapter.adapt_features(X, feature_names)
    
    print(f"Original shape: {X.shape}")
    print(f"Adapted shape: {adapted_X.shape}")
    print(f"Selected features: {selected_features}")
    
    # Get feature importance
    importance = adapter.get_feature_importance(X, feature_names)
    
    print("Feature importance:")
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {score:.4f}")
