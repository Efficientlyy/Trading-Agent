#!/usr/bin/env python
"""
Outlier Detection and Handling for Enhanced Deep Learning Pattern Recognition.

This module provides functionality to detect and handle outliers in market data
to improve the robustness of the deep learning pattern recognition system.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Union, Optional
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('outlier_detection')

class OutlierDetector:
    """Class for detecting outliers in market data."""
    
    def __init__(self, 
                method: str = 'isolation_forest',
                contamination: float = 0.05,
                window_size: int = 100,
                n_neighbors: int = 20,
                z_threshold: float = 3.0):
        """Initialize the outlier detector.
        
        Args:
            method: Outlier detection method ('isolation_forest', 'lof', 'z_score', 'mad')
            contamination: Expected proportion of outliers (for isolation_forest and lof)
            window_size: Window size for rolling statistics
            n_neighbors: Number of neighbors (for lof)
            z_threshold: Z-score threshold for outlier detection
        """
        self.method = method
        self.contamination = contamination
        self.window_size = window_size
        self.n_neighbors = n_neighbors
        self.z_threshold = z_threshold
        
        # Initialize detector based on method
        if method == 'isolation_forest':
            self.detector = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
        elif method == 'lof':
            self.detector = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=contamination,
                n_jobs=-1
            )
        elif method in ['z_score', 'mad']:
            self.detector = None
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
        
        logger.info(f"Initialized OutlierDetector with method={method}")
    
    def detect_outliers(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Detect outliers in the data.
        
        Args:
            data: Input data as numpy array or pandas DataFrame
            
        Returns:
            Boolean mask where True indicates an outlier
        """
        # Convert DataFrame to numpy array if needed
        if isinstance(data, pd.DataFrame):
            data_values = data.values
        else:
            data_values = data
        
        # Handle missing values
        if np.isnan(data_values).any():
            data_values = np.nan_to_num(data_values, nan=0.0)
        
        # Detect outliers based on method
        if self.method == 'isolation_forest':
            self.detector.fit(data_values)
            # Convert to boolean mask (True for outliers)
            return self.detector.predict(data_values) == -1
        
        elif self.method == 'lof':
            # LOF returns -1 for outliers, 1 for inliers
            outlier_labels = self.detector.fit_predict(data_values)
            return outlier_labels == -1
        
        elif self.method == 'z_score':
            # Calculate z-scores
            z_scores = np.abs(stats.zscore(data_values, axis=0, nan_policy='omit'))
            # Mark as outlier if any feature exceeds threshold
            return np.any(z_scores > self.z_threshold, axis=1)
        
        elif self.method == 'mad':
            # Calculate median absolute deviation
            median = np.median(data_values, axis=0)
            mad = np.median(np.abs(data_values - median), axis=0)
            # Avoid division by zero
            mad = np.where(mad == 0, 1e-8, mad)
            # Calculate modified z-scores
            modified_z_scores = 0.6745 * np.abs(data_values - median) / mad
            # Mark as outlier if any feature exceeds threshold
            return np.any(modified_z_scores > self.z_threshold, axis=1)
    
    def detect_outliers_rolling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers using rolling windows.
        
        Args:
            data: Input data as pandas DataFrame with DatetimeIndex
            
        Returns:
            DataFrame with boolean column 'is_outlier'
        """
        result = pd.DataFrame(index=data.index)
        result['is_outlier'] = False
        
        # Process data in rolling windows
        for i in range(self.window_size, len(data) + 1):
            window_data = data.iloc[i - self.window_size:i]
            if i < len(data):
                current_point = data.iloc[i:i+1]
                
                # Detect outliers in window
                window_with_point = pd.concat([window_data, current_point])
                outliers = self.detect_outliers(window_with_point)
                
                # Check if current point is an outlier
                if outliers[-1]:
                    result.iloc[i, 0] = True
        
        return result


class OutlierHandler:
    """Class for handling outliers in market data."""
    
    def __init__(self, 
                method: str = 'winsorize',
                detector_method: str = 'isolation_forest',
                contamination: float = 0.05,
                window_size: int = 100,
                z_threshold: float = 3.0,
                winsorize_limits: Tuple[float, float] = (0.05, 0.05)):
        """Initialize the outlier handler.
        
        Args:
            method: Outlier handling method ('winsorize', 'clip', 'median', 'mean', 'interpolate')
            detector_method: Outlier detection method
            contamination: Expected proportion of outliers
            window_size: Window size for rolling statistics
            z_threshold: Z-score threshold for outlier detection
            winsorize_limits: Limits for winsorization (lower, upper)
        """
        self.method = method
        self.window_size = window_size
        self.winsorize_limits = winsorize_limits
        
        # Initialize detector
        self.detector = OutlierDetector(
            method=detector_method,
            contamination=contamination,
            window_size=window_size,
            z_threshold=z_threshold
        )
        
        logger.info(f"Initialized OutlierHandler with method={method}")
    
    def handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers in the data.
        
        Args:
            data: Input data as pandas DataFrame
            
        Returns:
            DataFrame with outliers handled
        """
        # Make a copy to avoid modifying the original data
        result = data.copy()
        
        # Detect outliers
        outliers = self.detector.detect_outliers(data)
        
        # Handle outliers based on method
        if self.method == 'winsorize':
            for col in result.columns:
                result[col] = self._winsorize(result[col].values)
        
        elif self.method == 'clip':
            for col in result.columns:
                col_data = result[col].values
                q_low, q_high = np.percentile(col_data, [5, 95])
                result[col] = np.clip(col_data, q_low, q_high)
        
        elif self.method == 'median':
            for col in result.columns:
                col_median = result[col].median()
                result.loc[outliers, col] = col_median
        
        elif self.method == 'mean':
            for col in result.columns:
                col_mean = result[col].mean()
                result.loc[outliers, col] = col_mean
        
        elif self.method == 'interpolate':
            # Mark outliers as NaN
            for col in result.columns:
                result.loc[outliers, col] = np.nan
            # Interpolate missing values
            result = result.interpolate(method='linear')
            # Fill remaining NaNs (at edges)
            result = result.fillna(method='ffill').fillna(method='bfill')
        
        else:
            raise ValueError(f"Unsupported outlier handling method: {self.method}")
        
        # Count outliers
        num_outliers = np.sum(outliers)
        logger.info(f"Detected and handled {num_outliers} outliers ({num_outliers/len(data)*100:.2f}%)")
        
        return result
    
    def handle_outliers_rolling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers using rolling windows.
        
        Args:
            data: Input data as pandas DataFrame with DatetimeIndex
            
        Returns:
            DataFrame with outliers handled
        """
        # Make a copy to avoid modifying the original data
        result = data.copy()
        
        # Process data in rolling windows
        for i in range(self.window_size, len(data)):
            window_data = data.iloc[i - self.window_size:i]
            current_point = data.iloc[i:i+1]
            
            # Detect if current point is an outlier
            window_with_point = pd.concat([window_data, current_point])
            outliers = self.detector.detect_outliers(window_with_point)
            
            # Handle outlier if detected
            if outliers[-1]:
                if self.method == 'winsorize':
                    for col in result.columns:
                        limits = self._calculate_limits(window_data[col].values)
                        result.iloc[i, result.columns.get_loc(col)] = np.clip(
                            current_point[col].values[0], 
                            limits[0], 
                            limits[1]
                        )
                
                elif self.method == 'clip':
                    for col in result.columns:
                        q_low, q_high = np.percentile(window_data[col].values, [5, 95])
                        result.iloc[i, result.columns.get_loc(col)] = np.clip(
                            current_point[col].values[0], 
                            q_low, 
                            q_high
                        )
                
                elif self.method == 'median':
                    for col in result.columns:
                        result.iloc[i, result.columns.get_loc(col)] = window_data[col].median()
                
                elif self.method == 'mean':
                    for col in result.columns:
                        result.iloc[i, result.columns.get_loc(col)] = window_data[col].mean()
                
                elif self.method == 'interpolate':
                    # Use previous and next points for interpolation
                    if i > 0 and i < len(data) - 1:
                        for col in result.columns:
                            prev_val = result.iloc[i-1, result.columns.get_loc(col)]
                            next_val = data.iloc[i+1, data.columns.get_loc(col)]
                            result.iloc[i, result.columns.get_loc(col)] = (prev_val + next_val) / 2
        
        return result
    
    def _winsorize(self, data: np.ndarray) -> np.ndarray:
        """Winsorize data to limit extreme values.
        
        Args:
            data: Input data as numpy array
            
        Returns:
            Winsorized data
        """
        lower_limit, upper_limit = self.winsorize_limits
        quantiles = np.percentile(data, [lower_limit * 100, (1 - upper_limit) * 100])
        return np.clip(data, quantiles[0], quantiles[1])
    
    def _calculate_limits(self, data: np.ndarray) -> Tuple[float, float]:
        """Calculate limits for outlier handling.
        
        Args:
            data: Input data as numpy array
            
        Returns:
            Tuple of (lower_limit, upper_limit)
        """
        lower_limit, upper_limit = self.winsorize_limits
        quantiles = np.percentile(data, [lower_limit * 100, (1 - upper_limit) * 100])
        return (quantiles[0], quantiles[1])


class RobustPreprocessor:
    """Class for robust preprocessing of market data."""
    
    def __init__(self, 
                outlier_method: str = 'isolation_forest',
                handling_method: str = 'winsorize',
                contamination: float = 0.05,
                window_size: int = 100,
                z_threshold: float = 3.0,
                winsorize_limits: Tuple[float, float] = (0.05, 0.05),
                scaling_method: str = 'robust'):
        """Initialize the robust preprocessor.
        
        Args:
            outlier_method: Outlier detection method
            handling_method: Outlier handling method
            contamination: Expected proportion of outliers
            window_size: Window size for rolling statistics
            z_threshold: Z-score threshold for outlier detection
            winsorize_limits: Limits for winsorization
            scaling_method: Scaling method ('robust', 'standard', 'minmax')
        """
        self.outlier_handler = OutlierHandler(
            method=handling_method,
            detector_method=outlier_method,
            contamination=contamination,
            window_size=window_size,
            z_threshold=z_threshold,
            winsorize_limits=winsorize_limits
        )
        
        self.scaling_method = scaling_method
        self.scalers = {}
        
        logger.info(f"Initialized RobustPreprocessor with scaling_method={scaling_method}")
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit the preprocessor and transform the data.
        
        Args:
            data: Input data as pandas DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Handle outliers
        cleaned_data = self.outlier_handler.handle_outliers(data)
        
        # Scale features
        scaled_data = self._scale_features(cleaned_data, fit=True)
        
        return scaled_data
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using the fitted preprocessor.
        
        Args:
            data: Input data as pandas DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Handle outliers
        cleaned_data = self.outlier_handler.handle_outliers(data)
        
        # Scale features
        scaled_data = self._scale_features(cleaned_data, fit=False)
        
        return scaled_data
    
    def _scale_features(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scale features using the specified method.
        
        Args:
            data: Input data as pandas DataFrame
            fit: Whether to fit the scalers
            
        Returns:
            Scaled DataFrame
        """
        result = pd.DataFrame(index=data.index)
        
        for col in data.columns:
            if fit or col not in self.scalers:
                # Calculate scaling parameters
                if self.scaling_method == 'robust':
                    median = np.median(data[col])
                    iqr = np.percentile(data[col], 75) - np.percentile(data[col], 25)
                    # Avoid division by zero
                    iqr = max(iqr, 1e-8)
                    self.scalers[col] = {'center': median, 'scale': iqr}
                
                elif self.scaling_method == 'standard':
                    mean = np.mean(data[col])
                    std = np.std(data[col])
                    # Avoid division by zero
                    std = max(std, 1e-8)
                    self.scalers[col] = {'center': mean, 'scale': std}
                
                elif self.scaling_method == 'minmax':
                    min_val = np.min(data[col])
                    max_val = np.max(data[col])
                    # Avoid division by zero
                    range_val = max(max_val - min_val, 1e-8)
                    self.scalers[col] = {'center': min_val, 'scale': range_val}
            
            # Apply scaling
            if self.scaling_method == 'minmax':
                result[col] = (data[col] - self.scalers[col]['center']) / self.scalers[col]['scale']
            else:
                result[col] = (data[col] - self.scalers[col]['center']) / self.scalers[col]['scale']
        
        return result
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled data back to original scale.
        
        Args:
            data: Scaled data as pandas DataFrame
            
        Returns:
            Data in original scale
        """
        result = pd.DataFrame(index=data.index)
        
        for col in data.columns:
            if col in self.scalers:
                if self.scaling_method == 'minmax':
                    result[col] = data[col] * self.scalers[col]['scale'] + self.scalers[col]['center']
                else:
                    result[col] = data[col] * self.scalers[col]['scale'] + self.scalers[col]['center']
            else:
                result[col] = data[col]
        
        return result


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description='Outlier Detection and Handling')
    parser.add_argument('--data_file', type=str, default=None, help='Path to CSV data file')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory for output files')
    parser.add_argument('--method', type=str, default='isolation_forest', 
                        choices=['isolation_forest', 'lof', 'z_score', 'mad'],
                        help='Outlier detection method')
    parser.add_argument('--handling', type=str, default='winsorize',
                        choices=['winsorize', 'clip', 'median', 'mean', 'interpolate'],
                        help='Outlier handling method')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate synthetic data if no file provided
    if args.data_file is None:
        # Generate time index
        index = pd.date_range(start='2025-01-01', periods=1000, freq='1min')
        
        # Generate price data with outliers
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.normal(0, 1, 1000)) * 0.1
        
        # Add outliers
        outlier_indices = np.random.choice(1000, 50, replace=False)
        prices[outlier_indices] += np.random.normal(0, 10, 50)
        
        # Create DataFrame
        data = pd.DataFrame({'price': prices}, index=index)
        
        # Add volume
        data['volume'] = np.random.lognormal(5, 1, 1000)
        data.loc[outlier_indices, 'volume'] *= 5
        
        # Add returns
        data['returns'] = data['price'].pct_change().fillna(0)
        
        logger.info("Generated synthetic data")
    else:
        # Load data from file
        data = pd.read_csv(args.data_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded data from {args.data_file}")
    
    # Create outlier handler
    handler = OutlierHandler(
        method=args.handling,
        detector_method=args.method
    )
    
    # Handle outliers
    cleaned_data = handler.handle_outliers(data)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    for i, col in enumerate(data.columns):
        plt.subplot(len(data.columns), 1, i+1)
        plt.plot(data.index, data[col], 'b-', alpha=0.5, label='Original')
        plt.plot(data.index, cleaned_data[col], 'r-', label='Cleaned')
        plt.title(f'{col} - Original vs Cleaned')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/outlier_handling_{args.method}_{args.handling}.png")
    
    # Save cleaned data
    cleaned_data.to_csv(f"{args.output_dir}/cleaned_data_{args.method}_{args.handling}.csv")
    
    logger.info(f"Results saved to {args.output_dir}")
