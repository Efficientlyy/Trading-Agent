#!/usr/bin/env python
"""
Data Pipeline for Deep Learning Pattern Recognition

This module provides data preprocessing and feature engineering for the
deep learning pattern recognition models.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dl_data_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dl_data_pipeline")

class MarketDataPreprocessor:
    """Preprocessor for market data to be used in deep learning models"""
    
    def __init__(self, 
                 config_path=None,
                 sequence_length=60,
                 forecast_horizon=10,
                 normalization="minmax",
                 add_technical_indicators=True,
                 add_temporal_features=True):
        """Initialize preprocessor
        
        Args:
            config_path: Path to configuration file
            sequence_length: Length of input sequences
            forecast_horizon: Length of forecast horizon
            normalization: Normalization method (minmax, standard, none)
            add_technical_indicators: Whether to add technical indicators
            add_temporal_features: Whether to add temporal features
        """
        self.config_path = config_path
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.normalization = normalization
        self.add_technical_indicators = add_technical_indicators
        self.add_temporal_features = add_temporal_features
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize scalers
        self.scalers = {}
        
        logger.info(f"Initialized MarketDataPreprocessor with sequence_length={sequence_length}, forecast_horizon={forecast_horizon}")
    
    def _load_config(self) -> Dict:
        """Load configuration from file or use defaults
        
        Returns:
            dict: Configuration dictionary
        """
        default_config = {
            "preprocessing": {
                "sequence_length": self.sequence_length,
                "forecast_horizon": self.forecast_horizon,
                "normalization": self.normalization,
                "add_technical_indicators": self.add_technical_indicators,
                "add_temporal_features": self.add_temporal_features
            },
            "technical_indicators": {
                "rsi": {"window": 14},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "bollinger_bands": {"window": 20, "num_std": 2},
                "atr": {"window": 14},
                "vwap": {"window": 14}
            },
            "temporal_features": {
                "hour_of_day": True,
                "day_of_week": True,
                "day_of_month": True,
                "month_of_year": True,
                "is_weekend": True
            },
            "normalization": {
                "method": self.normalization,
                "feature_ranges": {
                    "price": (-1, 1),
                    "volume": (0, 1),
                    "indicators": (-1, 1)
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
        preprocessing_config = default_config["preprocessing"]
        self.sequence_length = preprocessing_config["sequence_length"]
        self.forecast_horizon = preprocessing_config["forecast_horizon"]
        self.normalization = preprocessing_config["normalization"]
        self.add_technical_indicators = preprocessing_config["add_technical_indicators"]
        self.add_temporal_features = preprocessing_config["add_temporal_features"]
        
        return default_config
    
    def preprocess_data(self, data, is_training=True):
        """Preprocess market data for deep learning
        
        Args:
            data: Market data (DataFrame or dict)
            is_training: Whether this is training data
            
        Returns:
            tuple: (X, y, feature_names)
        """
        # Convert to DataFrame if dict
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Add derived features
        if self.add_technical_indicators:
            df = self._add_derived_features(df)
        
        # Add temporal features
        if self.add_temporal_features:
            df = self._add_temporal_features(df)
        
        # Normalize data
        df = self._normalize_data(df, is_training)
        
        # Create sequences
        X, y, timestamps, feature_names = self._create_sequences(df)
        
        return X, y, timestamps, feature_names
    
    def _add_derived_features(self, df):
        """Add derived features to DataFrame
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame: DataFrame with derived features
        """
        # Check if required columns exist
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame. Skipping derived features.")
                return df
        
        # Get technical indicator config
        ti_config = self.config["technical_indicators"]
        
        # Add RSI
        if "rsi" in ti_config:
            window = ti_config["rsi"]["window"]
            df["rsi"] = self._calculate_rsi(df["close"], window)
        
        # Add MACD
        if "macd" in ti_config:
            fast = ti_config["macd"]["fast"]
            slow = ti_config["macd"]["slow"]
            signal = ti_config["macd"]["signal"]
            df["macd"], df["macd_signal"], df["macd_hist"] = self._calculate_macd(df["close"], fast, slow, signal)
        
        # Add Bollinger Bands
        if "bollinger_bands" in ti_config:
            window = ti_config["bollinger_bands"]["window"]
            num_std = ti_config["bollinger_bands"]["num_std"]
            df["bb_upper"], df["bb_middle"], df["bb_lower"] = self._calculate_bollinger_bands(df["close"], window, num_std)
            df["bb_percent_b"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        # Add ATR
        if "atr" in ti_config:
            window = ti_config["atr"]["window"]
            df["atr"] = self._calculate_atr(df["high"], df["low"], df["close"], window)
        
        # Add VWAP
        if "vwap" in ti_config:
            window = ti_config["vwap"]["window"]
            df["vwap"] = self._calculate_vwap(df["high"], df["low"], df["close"], df["volume"], window)
        
        # Add price momentum
        df["momentum"] = df["close"].pct_change(5)
        
        # Add volatility
        df["volatility"] = df["close"].rolling(window=20).std() / df["close"].rolling(window=20).mean()
        
        # Add price range
        df["price_range"] = (df["high"] - df["low"]) / df["close"]
        
        # Add volume momentum
        df["volume_momentum"] = df["volume"].pct_change(5)
        
        # Add price-volume correlation
        df["price_volume_corr"] = df["close"].rolling(window=20).corr(df["volume"])
        
        # Fill NaN values
        df = df.fillna(method="bfill").fillna(method="ffill").fillna(0)
        
        return df
    
    def _add_temporal_features(self, df):
        """Add temporal features to DataFrame
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame: DataFrame with temporal features
        """
        # Check if timestamp column exists
        if "timestamp" not in df.columns:
            logger.warning("Timestamp column not found in DataFrame. Skipping temporal features.")
            return df
        
        # Get temporal features config
        temp_config = self.config["temporal_features"]
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Add hour of day
        if temp_config.get("hour_of_day", True):
            df["hour_of_day"] = df["timestamp"].dt.hour / 24.0
        
        # Add day of week
        if temp_config.get("day_of_week", True):
            df["day_of_week"] = df["timestamp"].dt.dayofweek / 6.0
        
        # Add day of month
        if temp_config.get("day_of_month", True):
            df["day_of_month"] = (df["timestamp"].dt.day - 1) / 30.0
        
        # Add month of year
        if temp_config.get("month_of_year", True):
            df["month_of_year"] = (df["timestamp"].dt.month - 1) / 11.0
        
        # Add is weekend
        if temp_config.get("is_weekend", True):
            df["is_weekend"] = df["timestamp"].dt.dayofweek.isin([5, 6]).astype(float)
        
        return df
    
    def _normalize_data(self, df, is_training=True):
        """Normalize data
        
        Args:
            df: DataFrame
            is_training: Whether this is training data
            
        Returns:
            DataFrame: Normalized DataFrame
        """
        # Get normalization config
        norm_config = self.config["normalization"]
        method = norm_config["method"]
        
        # Skip normalization if method is none
        if method.lower() == "none":
            return df
        
        # Create a copy of the DataFrame
        normalized_df = df.copy()
        
        # Skip timestamp column
        if "timestamp" in normalized_df.columns:
            timestamp_col = normalized_df["timestamp"].copy()
            normalized_df = normalized_df.drop(columns=["timestamp"])
        else:
            timestamp_col = None
        
        # Normalize each column
        for col in normalized_df.columns:
            # Skip columns with all zeros or NaNs
            if normalized_df[col].isna().all() or (normalized_df[col] == 0).all():
                continue
            
            # Create scaler if training
            if is_training:
                if method.lower() == "minmax":
                    scaler = MinMaxScaler()
                elif method.lower() == "standard":
                    scaler = StandardScaler()
                else:
                    raise ValueError(f"Unknown normalization method: {method}")
                
                # Fit scaler
                scaler.fit(normalized_df[[col]])
                
                # Save scaler
                self.scalers[col] = scaler
            
            # Transform data
            if col in self.scalers:
                normalized_df[col] = self.scalers[col].transform(normalized_df[[col]])
        
        # Add timestamp back
        if timestamp_col is not None:
            normalized_df["timestamp"] = timestamp_col
        
        return normalized_df
    
    def _create_sequences(self, df):
        """Create sequences for deep learning
        
        Args:
            df: DataFrame
            
        Returns:
            tuple: (X, y, timestamps, feature_names)
        """
        # Check if we have enough data
        if len(df) < self.sequence_length + self.forecast_horizon:
            logger.warning(f"Not enough data to create sequences. Need at least {self.sequence_length + self.forecast_horizon} rows.")
            return np.array([]), np.array([]), np.array([]), []
        
        # Extract timestamp column if exists
        if "timestamp" in df.columns:
            timestamps = df["timestamp"].values
            df = df.drop(columns=["timestamp"])
        else:
            timestamps = np.arange(len(df))
        
        # Get feature names
        feature_names = df.columns.tolist()
        
        # Convert to numpy array
        data = df.values
        
        # Create sequences
        X = []
        y = []
        ts = []
        
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            # Input sequence
            X.append(data[i:i+self.sequence_length])
            
            # Target sequence
            y.append(data[i+self.sequence_length:i+self.sequence_length+self.forecast_horizon])
            
            # Timestamp
            ts.append(timestamps[i+self.sequence_length])
        
        return np.array(X), np.array(y), np.array(ts), feature_names
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index
        
        Args:
            prices: Price series
            window: Window size
            
        Returns:
            Series: RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate Moving Average Convergence Divergence
        
        Args:
            prices: Price series
            fast: Fast EMA window
            slow: Slow EMA window
            signal: Signal EMA window
            
        Returns:
            tuple: (MACD, Signal, Histogram)
        """
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD
        macd = ema_fast - ema_slow
        
        # Calculate Signal
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        # Calculate Histogram
        histogram = macd - signal_line
        
        return macd, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands
        
        Args:
            prices: Price series
            window: Window size
            num_std: Number of standard deviations
            
        Returns:
            tuple: (Upper Band, Middle Band, Lower Band)
        """
        # Calculate middle band (SMA)
        middle_band = prices.rolling(window=window).mean()
        
        # Calculate standard deviation
        std = prices.rolling(window=window).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return upper_band, middle_band, lower_band
    
    def _calculate_atr(self, high, low, close, window=14):
        """Calculate Average True Range
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Window size
            
        Returns:
            Series: ATR values
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        # Combine True Ranges
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    def _calculate_vwap(self, high, low, close, volume, window=14):
        """Calculate Volume Weighted Average Price
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume
            window: Window size
            
        Returns:
            Series: VWAP values
        """
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate VWAP
        vwap = (typical_price * volume).rolling(window=window).sum() / volume.rolling(window=window).sum()
        
        return vwap
    
    def save_scalers(self, path):
        """Save scalers to file
        
        Args:
            path: Path to save scalers
            
        Returns:
            bool: Success
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save scalers
            with open(path, 'wb') as f:
                pickle.dump(self.scalers, f)
            
            logger.info(f"Saved scalers to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving scalers: {str(e)}")
            return False
    
    def load_scalers(self, path):
        """Load scalers from file
        
        Args:
            path: Path to load scalers
            
        Returns:
            bool: Success
        """
        try:
            # Check if file exists
            if not os.path.exists(path):
                logger.warning(f"Scalers file not found at {path}")
                return False
            
            # Load scalers
            with open(path, 'rb') as f:
                self.scalers = pickle.load(f)
            
            logger.info(f"Loaded scalers from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading scalers: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Create preprocessor
    preprocessor = MarketDataPreprocessor()
    
    # Create sample data
    data = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1min"),
        "open": np.random.normal(100, 5, 100),
        "high": np.random.normal(105, 5, 100),
        "low": np.random.normal(95, 5, 100),
        "close": np.random.normal(100, 5, 100),
        "volume": np.random.normal(1000, 200, 100)
    })
    
    # Preprocess data
    X, y, timestamps, feature_names = preprocessor.preprocess_data(data)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Feature names: {feature_names}")
