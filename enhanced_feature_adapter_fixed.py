#!/usr/bin/env python
"""
Enhanced Feature Adapter for Trading-Agent System

This module provides enhanced feature adaptation for the Trading-Agent system,
with improved feature selection and transformation capabilities.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union

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
    
    def __init__(self, input_dim: int = 9, importance_method: str = "mutual_info"):
        """Initialize enhanced feature adapter
        
        Args:
            input_dim: Input dimension
            importance_method: Feature importance method
        """
        self.input_dim = input_dim
        self.importance_method = importance_method
        
        # Initialize feature importance
        self.feature_importance = {}
        
        # Initialize market regime detection
        self.volatility_window = 20
        self.momentum_window = 10
        
        logger.info(f"Initialized EnhancedFeatureAdapter with input_dim={input_dim}, importance_method={importance_method}")
    
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """Detect market regime from market data
        
        Args:
            df: Market data DataFrame
            
        Returns:
            str: Market regime (trending, ranging, volatile)
        """
        try:
            # Check if DataFrame is empty
            if df is None or df.empty or len(df) < max(self.volatility_window, self.momentum_window):
                logger.warning(f"Insufficient data for market regime detection, defaulting to 'ranging'")
                return "ranging"
            
            # Ensure 'close' column exists
            if 'close' not in df.columns:
                logger.warning("No 'close' column in DataFrame, defaulting to 'ranging'")
                return "ranging"
            
            # Calculate volatility (standard deviation of returns)
            returns = df['close'].pct_change().dropna()
            volatility = returns.rolling(window=min(self.volatility_window, len(returns))).std().iloc[-1]
            
            # Calculate momentum (rate of change)
            momentum = (df['close'].iloc[-1] / df['close'].iloc[-min(self.momentum_window, len(df))]) - 1
            
            # Log metrics
            logger.info(f"Detected market regime: volatility={volatility:.4f}, momentum={momentum:.4f}")
            
            # Determine market regime
            if volatility > 0.02:  # High volatility
                return "volatile"
            elif abs(momentum) > 0.01:  # Strong momentum
                return "trending"
            else:
                return "ranging"
        
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return "ranging"  # Default to ranging
    
    def adapt_features(self, X: np.ndarray, feature_names: List[str], market_regime: str) -> Tuple[np.ndarray, List[str]]:
        """Adapt features based on market regime
        
        Args:
            X: Input features
            feature_names: Feature names
            market_regime: Market regime
            
        Returns:
            tuple: Adapted features and selected feature names
        """
        try:
            # Check if input is valid
            if X is None or X.size == 0:
                logger.warning("Empty input for feature adaptation")
                return X, feature_names
            
            # Log input shape
            logger.info(f"Input shape for feature adaptation: {X.shape}")
            
            # Select features based on market regime
            if market_regime == "trending":
                # For trending markets, prioritize momentum indicators
                selected_indices = self._select_trending_features(X, feature_names)
            elif market_regime == "volatile":
                # For volatile markets, prioritize volatility indicators
                selected_indices = self._select_volatile_features(X, feature_names)
            else:  # ranging
                # For ranging markets, prioritize oscillators
                selected_indices = self._select_ranging_features(X, feature_names)
            
            # Ensure we have at least one feature
            if not selected_indices:
                logger.warning(f"No features selected for {market_regime} regime, using all features")
                selected_indices = list(range(X.shape[2]))
            
            # Select features
            X_selected = X[:, :, selected_indices]
            selected_names = [feature_names[i] for i in selected_indices]
            
            # Log selected features
            logger.info(f"Selected {len(selected_indices)} features for {market_regime} regime: {selected_names}")
            
            # Apply transformations based on market regime
            X_transformed = self._transform_features(X_selected, market_regime)
            
            # Log transformation result
            logger.info(f"Transformed features shape: {X_transformed.shape}")
            
            return X_transformed, selected_names
        
        except Exception as e:
            logger.error(f"Error adapting features: {str(e)}")
            return X, feature_names
    
    def _select_trending_features(self, X: np.ndarray, feature_names: List[str]) -> List[int]:
        """Select features for trending markets
        
        Args:
            X: Input features
            feature_names: Feature names
            
        Returns:
            list: Selected feature indices
        """
        # Define trending feature keywords
        trending_keywords = ['trend', 'momentum', 'macd', 'adx', 'rsi', 'ema', 'close']
        
        # Select features containing trending keywords
        selected_indices = []
        for i, name in enumerate(feature_names):
            if any(keyword in name.lower() for keyword in trending_keywords):
                selected_indices.append(i)
        
        # If no features selected, use all features
        if not selected_indices:
            selected_indices = list(range(X.shape[2]))
        
        return selected_indices
    
    def _select_volatile_features(self, X: np.ndarray, feature_names: List[str]) -> List[int]:
        """Select features for volatile markets
        
        Args:
            X: Input features
            feature_names: Feature names
            
        Returns:
            list: Selected feature indices
        """
        # Define volatile feature keywords
        volatile_keywords = ['volatility', 'atr', 'bollinger', 'std', 'high', 'low', 'volume']
        
        # Select features containing volatile keywords
        selected_indices = []
        for i, name in enumerate(feature_names):
            if any(keyword in name.lower() for keyword in volatile_keywords):
                selected_indices.append(i)
        
        # If no features selected, use all features
        if not selected_indices:
            selected_indices = list(range(X.shape[2]))
        
        return selected_indices
    
    def _select_ranging_features(self, X: np.ndarray, feature_names: List[str]) -> List[int]:
        """Select features for ranging markets
        
        Args:
            X: Input features
            feature_names: Feature names
            
        Returns:
            list: Selected feature indices
        """
        # Define ranging feature keywords
        ranging_keywords = ['rsi', 'stoch', 'cci', 'williams', 'bollinger', 'open', 'close']
        
        # Select features containing ranging keywords
        selected_indices = []
        for i, name in enumerate(feature_names):
            if any(keyword in name.lower() for keyword in ranging_keywords):
                selected_indices.append(i)
        
        # If no features selected, use all features
        if not selected_indices:
            selected_indices = list(range(X.shape[2]))
        
        return selected_indices
    
    def _transform_features(self, X: np.ndarray, market_regime: str) -> np.ndarray:
        """Transform features based on market regime
        
        Args:
            X: Input features
            market_regime: Market regime
            
        Returns:
            np.ndarray: Transformed features
        """
        # Check if input is valid
        if X is None or X.size == 0:
            return X
        
        # Apply transformations based on market regime
        if market_regime == "trending":
            # For trending markets, emphasize directional changes
            return self._transform_trending(X)
        elif market_regime == "volatile":
            # For volatile markets, emphasize range and volatility
            return self._transform_volatile(X)
        else:  # ranging
            # For ranging markets, emphasize mean reversion
            return self._transform_ranging(X)
    
    def _transform_trending(self, X: np.ndarray) -> np.ndarray:
        """Transform features for trending markets
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Transformed features
        """
        try:
            # Create a copy to avoid modifying the original
            X_transformed = X.copy()
            
            # Calculate momentum features (difference between current and previous)
            if X.shape[1] > 1:  # Need at least 2 time steps
                momentum = X_transformed[:, 1:, :] - X_transformed[:, :-1, :]
                
                # Pad to maintain shape
                padding = np.zeros((X.shape[0], 1, X.shape[2]))
                momentum_padded = np.concatenate([padding, momentum], axis=1)
                
                # Add momentum as new features
                X_transformed = np.concatenate([X_transformed, momentum_padded], axis=2)
            
            return X_transformed
        
        except Exception as e:
            logger.error(f"Error transforming trending features: {str(e)}")
            return X
    
    def _transform_volatile(self, X: np.ndarray) -> np.ndarray:
        """Transform features for volatile markets
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Transformed features
        """
        try:
            # Create a copy to avoid modifying the original
            X_transformed = X.copy()
            
            # Calculate volatility features (absolute difference)
            if X.shape[1] > 1:  # Need at least 2 time steps
                volatility = np.abs(X_transformed[:, 1:, :] - X_transformed[:, :-1, :])
                
                # Pad to maintain shape
                padding = np.zeros((X.shape[0], 1, X.shape[2]))
                volatility_padded = np.concatenate([padding, volatility], axis=1)
                
                # Add volatility as new features
                X_transformed = np.concatenate([X_transformed, volatility_padded], axis=2)
            
            return X_transformed
        
        except Exception as e:
            logger.error(f"Error transforming volatile features: {str(e)}")
            return X
    
    def _transform_ranging(self, X: np.ndarray) -> np.ndarray:
        """Transform features for ranging markets
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Transformed features
        """
        try:
            # Create a copy to avoid modifying the original
            X_transformed = X.copy()
            
            # Calculate mean reversion features (difference from moving average)
            if X.shape[1] > 5:  # Need at least 5 time steps for meaningful average
                # Calculate moving average (last 5 steps)
                ma = np.mean(X_transformed[:, -5:, :], axis=1, keepdims=True)
                
                # Calculate deviation from moving average
                deviation = X_transformed - np.repeat(ma, X_transformed.shape[1], axis=1)
                
                # Add deviation as new features
                X_transformed = np.concatenate([X_transformed, deviation], axis=2)
            
            return X_transformed
        
        except Exception as e:
            logger.error(f"Error transforming ranging features: {str(e)}")
            return X
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features (compatibility method)
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Transformed features
        """
        # Default transformation for compatibility
        try:
            # Detect market regime from the data
            # This is a simplified version since we don't have the full DataFrame
            
            # Calculate volatility (standard deviation)
            if X.shape[1] > 1:
                volatility = np.std(X[:, -self.volatility_window:, 0], axis=1).mean()
            else:
                volatility = 0.0
            
            # Calculate momentum (difference between last and first)
            if X.shape[1] > 1:
                momentum = (X[:, -1, 0] - X[:, 0, 0]).mean()
            else:
                momentum = 0.0
            
            # Determine market regime
            if volatility > 0.02:
                market_regime = "volatile"
            elif abs(momentum) > 0.01:
                market_regime = "trending"
            else:
                market_regime = "ranging"
            
            # Log detected regime
            logger.info(f"Detected market regime: {market_regime} (volatility={volatility:.4f}, momentum={momentum:.4f})")
            
            # Apply transformations based on market regime
            return self._transform_features(X, market_regime)
        
        except Exception as e:
            logger.error(f"Error in transform method: {str(e)}")
            return X
