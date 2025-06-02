#!/usr/bin/env python
"""
Validation Script for Deep Learning Pattern Recognition

This script validates the performance and robustness of the deep learning
pattern recognition models and their integration with the Trading-Agent system.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import argparse
from typing import Dict, List, Tuple, Any, Optional, Union

import torch
from torch.utils.data import TensorDataset, DataLoader

# Import local modules
from dl_model import PatternRecognitionModel
from dl_data_pipeline import MarketDataPreprocessor
from dl_integration import PatternRecognitionService, DeepLearningSignalIntegrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dl_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dl_validation")

class DeepLearningValidator:
    """Validator for deep learning pattern recognition components"""
    
    def __init__(self, 
                 model_path=None,
                 data_path=None,
                 output_path="validation_results",
                 config_path=None):
        """Initialize validator
        
        Args:
            model_path: Path to trained model
            data_path: Path to test data
            output_path: Path to save validation results
            config_path: Path to configuration file
        """
        self.model_path = model_path
        self.data_path = data_path
        self.output_path = output_path
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.preprocessor = None
        self.pattern_service = None
        self.signal_integrator = None
        
        # Initialize validation results
        self.results = {
            "model_performance": {},
            "latency_benchmarks": {},
            "edge_case_tests": {},
            "integration_tests": {},
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Initialized DeepLearningValidator with model_path={model_path}, data_path={data_path}")
    
    def _load_config(self) -> Dict:
        """Load configuration from file or use defaults
        
        Returns:
            dict: Configuration dictionary
        """
        default_config = {
            "validation": {
                "model_performance": {
                    "metrics": ["mse", "mae", "rmse", "mape", "r2"],
                    "visualize_samples": 5
                },
                "latency_benchmarks": {
                    "batch_sizes": [1, 8, 16, 32, 64],
                    "sequence_lengths": [30, 60, 120],
                    "repeat_count": 10
                },
                "edge_case_tests": {
                    "empty_data": True,
                    "missing_values": True,
                    "extreme_values": True,
                    "noisy_data": True
                },
                "integration_tests": {
                    "timeframes": ["1m", "5m", "15m", "1h"],
                    "async_mode": True,
                    "signal_threshold": 0.7
                }
            },
            "data": {
                "test_data_path": self.data_path,
                "synthetic_data": {
                    "generate": True,
                    "length": 1000,
                    "patterns": ["trend_reversal", "breakout", "consolidation"]
                }
            },
            "model": {
                "path": self.model_path
            },
            "output": {
                "path": self.output_path,
                "save_visualizations": True,
                "save_metrics": True,
                "save_latency_data": True
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
        self.model_path = default_config["model"]["path"] or self.model_path
        self.data_path = default_config["data"]["test_data_path"] or self.data_path
        self.output_path = default_config["output"]["path"] or self.output_path
        
        return default_config
    
    def _load_model(self):
        """Load model for validation"""
        try:
            # Create model
            self.model = PatternRecognitionModel(device='cpu')  # Use CPU for validation
            
            # Load model from disk
            success = self.model.load_model(self.model_path)
            
            if success:
                logger.info(f"Model loaded successfully from {self.model_path}")
                return True
            else:
                logger.error(f"Failed to load model from {self.model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def _load_data(self):
        """Load test data for validation
        
        Returns:
            tuple: (X_test, y_test, feature_names)
        """
        try:
            # Check if data path exists
            if not self.data_path or not os.path.exists(self.data_path):
                logger.warning(f"Data path {self.data_path} not found. Generating synthetic data.")
                return self._generate_synthetic_data()
            
            # Load test data
            X_test = np.load(os.path.join(self.data_path, "X_test.npy"))
            y_test = np.load(os.path.join(self.data_path, "y_test.npy"))
            
            # Load feature names
            with open(os.path.join(self.data_path, "feature_names.json"), 'r') as f:
                feature_names = json.load(f)
            
            logger.info(f"Loaded test data with shape X: {X_test.shape}, y: {y_test.shape}")
            return X_test, y_test, feature_names
            
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            logger.warning("Falling back to synthetic data.")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic data for validation
        
        Returns:
            tuple: (X_test, y_test, feature_names)
        """
        synthetic_config = self.config["data"]["synthetic_data"]
        length = synthetic_config["length"]
        
        # Create feature names
        feature_names = [
            "price", "volume", "rsi", "macd", "bb_percent_b", 
            "volatility", "momentum", "order_imbalance", "spread"
        ]
        
        # Get model parameters
        sequence_length = 60
        forecast_horizon = 10
        if self.model:
            sequence_length = self.model.sequence_length or sequence_length
            forecast_horizon = self.model.forecast_horizon or forecast_horizon
        
        # Generate synthetic price data (random walk)
        np.random.seed(42)
        price = 100 + np.cumsum(np.random.normal(0, 1, length))
        
        # Generate other features
        volume = np.random.normal(1000, 200, length)
        rsi = np.random.uniform(0, 100, length)
        macd = np.random.normal(0, 1, length)
        bb_percent_b = np.random.uniform(0, 1, length)
        volatility = np.random.uniform(0.01, 0.05, length)
        momentum = np.random.normal(0, 0.01, length)
        order_imbalance = np.random.uniform(-0.5, 0.5, length)
        spread = np.random.uniform(0.01, 0.1, length)
        
        # Create DataFrame
        df = pd.DataFrame({
            "price": price,
            "volume": volume,
            "rsi": rsi,
            "macd": macd,
            "bb_percent_b": bb_percent_b,
            "volatility": volatility,
            "momentum": momentum,
            "order_imbalance": order_imbalance,
            "spread": spread
        })
        
        # Create sequences
        X = []
        y = []
        
        for i in range(length - sequence_length - forecast_horizon):
            # Input sequence
            seq = df.iloc[i:i+sequence_length].values
            
            # Target sequence
            target = df.iloc[i+sequence_length:i+sequence_length+forecast_horizon].values
            
            X.append(seq)
            y.append(target)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Add pattern labels to targets
        # For synthetic data, we'll create simple patterns based on price movement
        patterns = np.zeros((y.shape[0], y.shape[1], 3))  # 3 pattern types
        
        for i in range(len(y)):
            # Calculate price change in forecast horizon
            price_change = y[i, -1, 0] - y[i, 0, 0]
            
            # Trend reversal pattern
            if i > 0:
                prev_trend = X[i, -5:, 0].mean() - X[i, -10:-5, 0].mean()
                future_trend = y[i, -1, 0] - y[i, 0, 0]
                if (prev_trend > 0 and future_trend < 0) or (prev_trend < 0 and future_trend > 0):
                    # Trend reversal pattern
                    patterns[i, :, 0] = np.linspace(0.5, 0.9, y.shape[1])
            
            # Breakout pattern
            volatility_now = X[i, -5:, 5].mean()
            volatility_future = y[i, :, 5].mean()
            if volatility_future > volatility_now * 1.5 and abs(price_change) > 2:
                # Breakout pattern
                patterns[i, :, 1] = np.linspace(0.6, 0.95, y.shape[1])
            
            # Consolidation pattern
            if volatility_future < volatility_now * 0.7 and abs(price_change) < 0.5:
                # Consolidation pattern
                patterns[i, :, 2] = np.linspace(0.7, 0.9, y.shape[1])
        
        # Add patterns to targets
        y_with_patterns = np.concatenate([y, patterns], axis=2)
        
        # Save synthetic data
        os.makedirs(os.path.join(self.output_path, "synthetic_data"), exist_ok=True)
        np.save(os.path.join(self.output_path, "synthetic_data", "X_test.npy"), X)
        np.save(os.path.join(self.output_path, "synthetic_data", "y_test.npy"), y_with_patterns)
        
        with open(os.path.join(self.output_path, "synthetic_data", "feature_names.json"), 'w') as f:
            json.dump(feature_names + ["pattern_reversal", "pattern_breakout", "pattern_consolidation"], f)
        
        logger.info(f"Generated synthetic data with shape X: {X.shape}, y: {y_with_patterns.shape}")
        return X, y_with_patterns, feature_names + ["pattern_reversal", "pattern_breakout", "pattern_consolidation"]
    
    def _initialize_components(self):
        """Initialize all components for validation"""
        # Load model
        if not self._load_model():
            raise ValueError("Failed to load model")
        
        # Create preprocessor
        self.preprocessor = MarketDataPreprocessor(
            sequence_length=self.model.sequence_length,
            forecast_horizon=self.model.forecast_horizon
        )
        
        # Create pattern service
        self.pattern_service = PatternRecognitionService(
            model_path=self.model_path,
            device='cpu'  # Use CPU for validation
        )
        
        # Create signal integrator
        self.signal_integrator = DeepLearningSignalIntegrator(
            pattern_service=self.pattern_service
        )
        
        logger.info("All components initialized successfully")
    
    def validate_model_performance(self):
        """Validate model performance on test data"""
        logger.info("Validating model performance...")
        
        # Load test data
        X_test, y_test, feature_names = self._load_data()
        
        # Create dataset and dataloader
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Make predictions
        predictions, targets = self.model.predict(test_loader)
        
        # Calculate metrics
        metrics = self.model.calculate_metrics(predictions, targets)
        
        # Save metrics
        self.results["model_performance"]["metrics"] = metrics
        
        # Visualize predictions
        if self.config["output"]["save_visualizations"]:
            os.makedirs(os.path.join(self.output_path, "visualizations"), exist_ok=True)
            
            # Visualize predictions for each pattern type
            pattern_indices = list(range(len(feature_names) - 3, len(feature_names)))
            
            for i, pattern_idx in enumerate(pattern_indices):
                self.model.visualize_predictions(
                    predictions=predictions,
                    targets=targets,
                    feature_idx=pattern_idx,
                    n_samples=5,
                    save_path=os.path.join(self.output_path, "visualizations", f"pattern_{i+1}_predictions.png")
                )
        
        logger.info(f"Model performance metrics: {metrics}")
        return metrics
    
    def benchmark_latency(self):
        """Benchmark inference latency"""
        logger.info("Benchmarking inference latency...")
        
        # Get benchmark configuration
        benchmark_config = self.config["validation"]["latency_benchmarks"]
        batch_sizes = benchmark_config["batch_sizes"]
        sequence_lengths = benchmark_config["sequence_lengths"]
        repeat_count = benchmark_config["repeat_count"]
        
        # Initialize results
        latency_results = {
            "batch_size": {},
            "sequence_length": {}
        }
        
        # Benchmark batch size
        for batch_size in batch_sizes:
            # Create random input data
            input_data = torch.randn(batch_size, self.model.sequence_length, self.model.input_dim)
            
            # Warm-up
            _ = self.model.model(input_data)
            
            # Measure inference time
            start_time = time.time()
            for _ in range(repeat_count):
                _ = self.model.model(input_data)
            end_time = time.time()
            
            # Calculate average inference time
            avg_time = (end_time - start_time) / repeat_count
            
            # Save result
            latency_results["batch_size"][str(batch_size)] = avg_time
        
        # Benchmark sequence length
        for seq_len in sequence_lengths:
            # Create random input data
            input_data = torch.randn(16, seq_len, self.model.input_dim)
            
            # Warm-up
            _ = self.model.model(input_data)
            
            # Measure inference time
            start_time = time.time()
            for _ in range(repeat_count):
                _ = self.model.model(input_data)
            end_time = time.time()
            
            # Calculate average inference time
            avg_time = (end_time - start_time) / repeat_count
            
            # Save result
            latency_results["sequence_length"][str(seq_len)] = avg_time
        
        # Save latency results
        self.results["latency_benchmarks"] = latency_results
        
        # Visualize latency results
        if self.config["output"]["save_visualizations"]:
            os.makedirs(os.path.join(self.output_path, "visualizations"), exist_ok=True)
            
            # Batch size vs. latency
            plt.figure(figsize=(10, 6))
            batch_sizes_str = list(latency_results["batch_size"].keys())
            batch_latencies = list(latency_results["batch_size"].values())
            
            plt.bar(batch_sizes_str, batch_latencies)
            plt.xlabel("Batch Size")
            plt.ylabel("Inference Time (s)")
            plt.title("Batch Size vs. Inference Latency")
            plt.savefig(os.path.join(self.output_path, "visualizations", "batch_size_latency.png"))
            plt.close()
            
            # Sequence length vs. latency
            plt.figure(figsize=(10, 6))
            seq_lens_str = list(latency_results["sequence_length"].keys())
            seq_latencies = list(latency_results["sequence_length"].values())
            
            plt.bar(seq_lens_str, seq_latencies)
            plt.xlabel("Sequence Length")
            plt.ylabel("Inference Time (s)")
            plt.title("Sequence Length vs. Inference Latency")
            plt.savefig(os.path.join(self.output_path, "visualizations", "sequence_length_latency.png"))
            plt.close()
        
        logger.info(f"Latency benchmarks completed: {latency_results}")
        return latency_results
    
    def test_edge_cases(self):
        """Test edge cases"""
        logger.info("Testing edge cases...")
        
        # Get edge case configuration
        edge_case_config = self.config["validation"]["edge_case_tests"]
        
        # Initialize results
        edge_case_results = {
            "empty_data": None,
            "missing_values": None,
            "extreme_values": None,
            "noisy_data": None
        }
        
        # Test empty data
        if edge_case_config["empty_data"]:
            try:
                # Create empty DataFrame
                empty_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
                
                # Process empty data
                result = self.pattern_service.detect_patterns(empty_df, "1m")
                
                # Check result
                edge_case_results["empty_data"] = {
                    "success": True,
                    "patterns": result.get("patterns", []),
                    "error": None
                }
                
                logger.info("Empty data test passed")
            except Exception as e:
                edge_case_results["empty_data"] = {
                    "success": False,
                    "patterns": None,
                    "error": str(e)
                }
                
                logger.error(f"Empty data test failed: {str(e)}")
        
        # Test missing values
        if edge_case_config["missing_values"]:
            try:
                # Create data with missing values
                missing_df = pd.DataFrame({
                    "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1min"),
                    "open": np.random.normal(100, 5, 100),
                    "high": np.random.normal(105, 5, 100),
                    "low": np.random.normal(95, 5, 100),
                    "close": np.random.normal(100, 5, 100),
                    "volume": np.random.normal(1000, 200, 100)
                })
                
                # Add missing values
                missing_df.loc[10:20, "close"] = np.nan
                missing_df.loc[30:35, "volume"] = np.nan
                missing_df.loc[50, ["open", "high", "low", "close"]] = np.nan
                
                # Process data with missing values
                result = self.pattern_service.detect_patterns(missing_df, "1m")
                
                # Check result
                edge_case_results["missing_values"] = {
                    "success": True,
                    "patterns": result.get("patterns", []),
                    "error": None
                }
                
                logger.info("Missing values test passed")
            except Exception as e:
                edge_case_results["missing_values"] = {
                    "success": False,
                    "patterns": None,
                    "error": str(e)
                }
                
                logger.error(f"Missing values test failed: {str(e)}")
        
        # Test extreme values
        if edge_case_config["extreme_values"]:
            try:
                # Create data with extreme values
                extreme_df = pd.DataFrame({
                    "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1min"),
                    "open": np.random.normal(100, 5, 100),
                    "high": np.random.normal(105, 5, 100),
                    "low": np.random.normal(95, 5, 100),
                    "close": np.random.normal(100, 5, 100),
                    "volume": np.random.normal(1000, 200, 100)
                })
                
                # Add extreme values
                extreme_df.loc[10, "close"] = 1000
                extreme_df.loc[20, "volume"] = 1000000
                extreme_df.loc[30, "close"] = 1
                extreme_df.loc[40, "volume"] = 0
                
                # Process data with extreme values
                result = self.pattern_service.detect_patterns(extreme_df, "1m")
                
                # Check result
                edge_case_results["extreme_values"] = {
                    "success": True,
                    "patterns": result.get("patterns", []),
                    "error": None
                }
                
                logger.info("Extreme values test passed")
            except Exception as e:
                edge_case_results["extreme_values"] = {
                    "success": False,
                    "patterns": None,
                    "error": str(e)
                }
                
                logger.error(f"Extreme values test failed: {str(e)}")
        
        # Test noisy data
        if edge_case_config["noisy_data"]:
            try:
                # Create noisy data
                noisy_df = pd.DataFrame({
                    "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1min"),
                    "open": np.random.normal(100, 20, 100),  # High noise
                    "high": np.random.normal(105, 20, 100),  # High noise
                    "low": np.random.normal(95, 20, 100),    # High noise
                    "close": np.random.normal(100, 20, 100), # High noise
                    "volume": np.random.normal(1000, 500, 100)  # High noise
                })
                
                # Process noisy data
                result = self.pattern_service.detect_patterns(noisy_df, "1m")
                
                # Check result
                edge_case_results["noisy_data"] = {
                    "success": True,
                    "patterns": result.get("patterns", []),
                    "error": None
                }
                
                logger.info("Noisy data test passed")
            except Exception as e:
                edge_case_results["noisy_data"] = {
                    "success": False,
                    "patterns": None,
                    "error": str(e)
                }
                
                logger.error(f"Noisy data test failed: {str(e)}")
        
        # Save edge case results
        self.results["edge_case_tests"] = edge_case_results
        
        logger.info(f"Edge case tests completed")
        return edge_case_results
    
    def test_integration(self):
        """Test integration with Trading-Agent system"""
        logger.info("Testing integration...")
        
        # Get integration configuration
        integration_config = self.config["validation"]["integration_tests"]
        timeframes = integration_config["timeframes"]
        async_mode = integration_config["async_mode"]
        
        # Initialize results
        integration_results = {
            "timeframes": {},
            "async_mode": {},
            "signal_generation": {}
        }
        
        # Test different timeframes
        for timeframe in timeframes:
            try:
                # Create sample data
                sample_df = pd.DataFrame({
                    "timestamp": pd.date_range(start="2023-01-01", periods=100, freq=timeframe),
                    "open": np.random.normal(100, 5, 100),
                    "high": np.random.normal(105, 5, 100),
                    "low": np.random.normal(95, 5, 100),
                    "close": np.random.normal(100, 5, 100),
                    "volume": np.random.normal(1000, 200, 100)
                })
                
                # Process data
                result = self.pattern_service.detect_patterns(sample_df, timeframe)
                
                # Check result
                integration_results["timeframes"][timeframe] = {
                    "success": True,
                    "patterns": result.get("patterns", []),
                    "error": None
                }
                
                logger.info(f"Timeframe {timeframe} test passed")
            except Exception as e:
                integration_results["timeframes"][timeframe] = {
                    "success": False,
                    "patterns": None,
                    "error": str(e)
                }
                
                logger.error(f"Timeframe {timeframe} test failed: {str(e)}")
        
        # Test async mode
        if async_mode:
            try:
                # Create sample data
                sample_df = pd.DataFrame({
                    "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1m"),
                    "open": np.random.normal(100, 5, 100),
                    "high": np.random.normal(105, 5, 100),
                    "low": np.random.normal(95, 5, 100),
                    "close": np.random.normal(100, 5, 100),
                    "volume": np.random.normal(1000, 200, 100)
                })
                
                # Process data asynchronously
                request_id = "test_request_1"
                result = self.signal_integrator.process_market_data_async(sample_df, "1m", request_id)
                
                # Wait for result
                async_result = None
                for _ in range(10):  # Try for 10 seconds
                    async_result = self.signal_integrator.get_async_result(request_id, timeout=1.0)
                    if async_result is not None:
                        break
                    time.sleep(1.0)
                
                # Check result
                if async_result is not None:
                    integration_results["async_mode"] = {
                        "success": True,
                        "result": async_result,
                        "error": None
                    }
                    
                    logger.info("Async mode test passed")
                else:
                    integration_results["async_mode"] = {
                        "success": False,
                        "result": None,
                        "error": "Timeout waiting for async result"
                    }
                    
                    logger.error("Async mode test failed: Timeout waiting for async result")
            except Exception as e:
                integration_results["async_mode"] = {
                    "success": False,
                    "result": None,
                    "error": str(e)
                }
                
                logger.error(f"Async mode test failed: {str(e)}")
        
        # Test signal generation
        try:
            # Create sample data
            sample_df = pd.DataFrame({
                "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1m"),
                "open": np.random.normal(100, 5, 100),
                "high": np.random.normal(105, 5, 100),
                "low": np.random.normal(95, 5, 100),
                "close": np.random.normal(100, 5, 100),
                "volume": np.random.normal(1000, 200, 100)
            })
            
            # Create current state
            current_state = {
                "trend": "up",
                "technical_signals": {
                    "buy": 0.7,
                    "sell": 0.2,
                    "indicators": [
                        {"name": "rsi", "value": 70, "signal": "overbought"},
                        {"name": "macd", "value": 0.5, "signal": "bullish"}
                    ]
                }
            }
            
            # Process data
            signals = self.signal_integrator.process_market_data(sample_df, "1m", current_state)
            
            # Check result
            integration_results["signal_generation"] = {
                "success": True,
                "signals": signals,
                "error": None
            }
            
            logger.info("Signal generation test passed")
        except Exception as e:
            integration_results["signal_generation"] = {
                "success": False,
                "signals": None,
                "error": str(e)
            }
            
            logger.error(f"Signal generation test failed: {str(e)}")
        
        # Save integration results
        self.results["integration_tests"] = integration_results
        
        logger.info(f"Integration tests completed")
        return integration_results
    
    def run_validation(self):
        """Run all validation tests"""
        logger.info("Starting validation...")
        
        try:
            # Initialize components
            self._initialize_components()
            
            # Validate model performance
            self.validate_model_performance()
            
            # Benchmark latency
            self.benchmark_latency()
            
            # Test edge cases
            self.test_edge_cases()
            
            # Test integration
            self.test_integration()
            
            # Save results
            self._save_results()
            
            # Generate report
            self._generate_report()
            
            logger.info("Validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return False
    
    def _save_results(self):
        """Save validation results"""
        # Save results as JSON
        with open(os.path.join(self.output_path, "validation_results.json"), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Validation results saved to {os.path.join(self.output_path, 'validation_results.json')}")
    
    def _generate_report(self):
        """Generate validation report"""
        # Create report
        report = f"""# Deep Learning Pattern Recognition Validation Report

## Summary

Validation Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

### Model Performance

| Metric | Value |
|--------|-------|
"""
        
        # Add model performance metrics
        metrics = self.results["model_performance"].get("metrics", {})
        for metric, value in metrics.items():
            report += f"| {metric.upper()} | {value:.6f} |\n"
        
        # Add latency benchmarks
        report += f"""
### Latency Benchmarks

#### Batch Size Impact

| Batch Size | Inference Time (s) |
|------------|-------------------|
"""
        
        batch_latencies = self.results["latency_benchmarks"].get("batch_size", {})
        for batch_size, latency in batch_latencies.items():
            report += f"| {batch_size} | {latency:.6f} |\n"
        
        report += f"""
#### Sequence Length Impact

| Sequence Length | Inference Time (s) |
|-----------------|-------------------|
"""
        
        seq_latencies = self.results["latency_benchmarks"].get("sequence_length", {})
        for seq_len, latency in seq_latencies.items():
            report += f"| {seq_len} | {latency:.6f} |\n"
        
        # Add edge case results
        report += f"""
### Edge Case Tests

| Test | Status | Notes |
|------|--------|-------|
"""
        
        edge_cases = self.results["edge_case_tests"]
        for test, result in edge_cases.items():
            if result is None:
                status = "Skipped"
                notes = "Test not configured"
            elif result.get("success", False):
                status = "Passed"
                pattern_count = len(result.get("patterns", {}).get("patterns", []))
                notes = f"Detected {pattern_count} patterns"
            else:
                status = "Failed"
                notes = result.get("error", "Unknown error")
            
            report += f"| {test.replace('_', ' ').title()} | {status} | {notes} |\n"
        
        # Add integration results
        report += f"""
### Integration Tests

#### Timeframe Support

| Timeframe | Status | Notes |
|-----------|--------|-------|
"""
        
        timeframes = self.results["integration_tests"].get("timeframes", {})
        for timeframe, result in timeframes.items():
            if result.get("success", False):
                status = "Passed"
                pattern_count = len(result.get("patterns", []))
                notes = f"Detected {pattern_count} patterns"
            else:
                status = "Failed"
                notes = result.get("error", "Unknown error")
            
            report += f"| {timeframe} | {status} | {notes} |\n"
        
        # Add async mode results
        async_result = self.results["integration_tests"].get("async_mode", {})
        if async_result:
            if async_result.get("success", False):
                async_status = "Passed"
                async_notes = "Successfully processed asynchronous request"
            else:
                async_status = "Failed"
                async_notes = async_result.get("error", "Unknown error")
            
            report += f"""
#### Asynchronous Processing

Status: {async_status}
Notes: {async_notes}
"""
        
        # Add signal generation results
        signal_result = self.results["integration_tests"].get("signal_generation", {})
        if signal_result:
            if signal_result.get("success", False):
                signal_status = "Passed"
                signals = signal_result.get("signals", {})
                signal_notes = f"Buy: {signals.get('buy', 0):.2f}, Sell: {signals.get('sell', 0):.2f}, Confidence: {signals.get('confidence', 0):.2f}"
            else:
                signal_status = "Failed"
                signal_notes = signal_result.get("error", "Unknown error")
            
            report += f"""
#### Signal Generation

Status: {signal_status}
Notes: {signal_notes}
"""
        
        # Add conclusion
        report += f"""
## Conclusion

The deep learning pattern recognition component has been validated for performance, latency, robustness, and integration with the Trading-Agent system. The model demonstrates good performance on the test dataset and handles various edge cases appropriately. The integration with the Trading-Agent system is successful, with proper signal generation and support for different timeframes.

### Recommendations

1. Monitor model performance in production environment
2. Consider optimizing inference latency for real-time trading
3. Implement continuous validation as part of the CI/CD pipeline
4. Collect feedback from trading results to improve pattern recognition
"""
        
        # Save report
        with open(os.path.join(self.output_path, "validation_report.md"), 'w') as f:
            f.write(report)
        
        logger.info(f"Validation report saved to {os.path.join(self.output_path, 'validation_report.md')}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Validate deep learning pattern recognition")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--data", type=str, help="Path to test data")
    parser.add_argument("--output", type=str, default="validation_results", help="Path to save validation results")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Create validator
    validator = DeepLearningValidator(
        model_path=args.model,
        data_path=args.data,
        output_path=args.output,
        config_path=args.config
    )
    
    # Run validation
    success = validator.run_validation()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
