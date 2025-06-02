#!/usr/bin/env python
"""
Enhanced Deep Learning Integration for Trading-Agent System

This module provides enhanced deep learning integration for the Trading-Agent system,
with improved pattern recognition and signal generation capabilities.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import OrderedDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_dl_integration_fixed.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_dl_integration_fixed")

# Import local modules
try:
    from enhanced_dl_model_fixed import EnhancedPatternRecognitionModel
    from enhanced_feature_adapter_fixed import EnhancedFeatureAdapter
    from dl_data_pipeline import MarketDataPreprocessor
except ImportError:
    logger.error("Failed to import required modules")
    raise

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle non-serializable types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.datetime64, pd.Timestamp)):
            return str(obj)
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)  # Convert any other objects to strings

class EnhancedPatternRecognitionService:
    """Enhanced pattern recognition service for deep learning models"""
    
    def __init__(self, 
                 model_path: str = None,
                 device: str = "cpu",
                 async_mode: bool = False,
                 config_path: str = None,
                 cache_enabled: bool = True,
                 cache_size: int = 100):
        """Initialize enhanced pattern recognition service
        
        Args:
            model_path: Path to model file
            device: Device to use for inference
            async_mode: Whether to use async mode
            config_path: Path to configuration file
            cache_enabled: Whether to enable caching
            cache_size: Maximum number of items in cache
        """
        self.model_path = model_path
        self.device = device
        self.async_mode = async_mode
        self.config_path = config_path
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size
        
        # Initialize cache
        self.prediction_cache = OrderedDict()
        
        # Load configuration
        self.config = self._load_config()
        
        # Update parameters from config
        self.device = self.config.get("device", self.device)
        self.async_mode = self.config.get("async_mode", self.async_mode)
        self.cache_enabled = self.config.get("cache_enabled", self.cache_enabled)
        self.cache_size = self.config.get("cache_size", self.cache_size)
        
        # Initialize model
        self.model = self._load_model()
        
        # Initialize preprocessor
        self.preprocessor = MarketDataPreprocessor(
            sequence_length=self.config.get("sequence_length", 60),
            forecast_horizon=self.config.get("forecast_horizon", 10)
        )
        
        # Initialize feature adapter
        self.feature_adapter = EnhancedFeatureAdapter(
            input_dim=self.config.get("input_dim", 9),
            importance_method=self.config.get("importance_method", "mutual_info")
        )
        
        # Load pattern registry
        self.pattern_registry = self._load_pattern_registry()
        
        # Set batch size
        self.batch_size = self.config.get("batch_size", 32)
        
        logger.info(f"Initialized EnhancedPatternRecognitionService with model_path={model_path}, device={device}, async_mode={async_mode}")
    
    def _load_config(self) -> Dict:
        """Load configuration from file or use defaults
        
        Returns:
            dict: Configuration dictionary
        """
        default_config = {
            "device": self.device,
            "async_mode": self.async_mode,
            "cache_enabled": self.cache_enabled,
            "cache_size": self.cache_size,
            "sequence_length": 60,
            "forecast_horizon": 10,
            "input_dim": 9,
            "importance_method": "mutual_info",
            "batch_size": 32,
            "confidence_threshold": 0.6
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
    
    def _load_model(self) -> EnhancedPatternRecognitionModel:
        """Load model from file or initialize new model
        
        Returns:
            EnhancedPatternRecognitionModel: Pattern recognition model
        """
        try:
            # Initialize model
            model = EnhancedPatternRecognitionModel(
                input_dim=self.config.get("input_dim", 9),
                hidden_dim=self.config.get("hidden_dim", 64),
                output_dim=self.config.get("output_dim", 3),
                sequence_length=self.config.get("sequence_length", 60),
                forecast_horizon=self.config.get("forecast_horizon", 10),
                model_type=self.config.get("model_type", "hybrid"),
                device=self.device
            )
            
            # Load model weights if path is provided
            if self.model_path and os.path.exists(self.model_path):
                try:
                    model.load_model(self.model_path)
                    logger.info(f"Loaded model from {self.model_path}")
                except Exception as e:
                    logger.error(f"Error loading model: {str(e)}")
                    logger.warning(f"Failed to load model from {self.model_path}, using initialized model")
            
            return model
        
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def _load_pattern_registry(self) -> Dict:
        """Load pattern registry from file or use defaults
        
        Returns:
            dict: Pattern registry
        """
        default_registry = {
            "patterns": [
                {
                    "id": "trend_reversal",
                    "name": "Trend Reversal",
                    "description": "Identifies potential trend reversal points",
                    "output_index": 0,
                    "confidence_threshold": 0.6,
                    "timeframes": ["1m", "5m", "15m", "1h"]
                },
                {
                    "id": "breakout",
                    "name": "Breakout",
                    "description": "Identifies potential breakout patterns",
                    "output_index": 1,
                    "confidence_threshold": 0.6,
                    "timeframes": ["5m", "15m", "1h"]
                },
                {
                    "id": "consolidation",
                    "name": "Consolidation",
                    "description": "Identifies market consolidation patterns",
                    "output_index": 2,
                    "confidence_threshold": 0.6,
                    "timeframes": ["15m", "1h", "4h"]
                }
            ]
        }
        
        # Try to load from file
        registry_path = "patterns/pattern_registry.json"
        
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    loaded_registry = json.load(f)
                logger.info(f"Loaded pattern registry from {registry_path}")
                return loaded_registry
            except Exception as e:
                logger.error(f"Error loading pattern registry: {str(e)}")
                logger.info("Using default pattern registry")
        
        return default_registry
    
    def detect_patterns(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Detect patterns in market data
        
        Args:
            df: Market data DataFrame
            timeframe: Timeframe of the data
            
        Returns:
            dict: Detected patterns
        """
        # Check if DataFrame is empty
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided for pattern detection")
            return {
                "patterns": [],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f")
            }
        
        # Check cache first if enabled
        if self.cache_enabled:
            try:
                cache_key = (hash(df.to_json()), timeframe)
                
                if cache_key in self.prediction_cache:
                    logger.info("Using cached pattern detection result")
                    return self.prediction_cache[cache_key]
            except Exception as e:
                logger.error(f"Error checking cache: {str(e)}")
                # Continue without caching
        
        # Detect patterns
        if self.async_mode:
            return self._detect_patterns_async(df, timeframe)
        else:
            return self._detect_patterns_sync(df, timeframe)
    
    def _detect_patterns_sync(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Detect patterns synchronously
        
        Args:
            df: Market data DataFrame
            timeframe: Timeframe of the data
            
        Returns:
            dict: Detected patterns
        """
        # Check if DataFrame is empty
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided for pattern detection")
            return {
                "patterns": [],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "error": "Empty DataFrame"
            }
        
        # Check cache first if enabled
        if self.cache_enabled:
            try:
                cache_key = (hash(df.to_json()), timeframe)
                
                if cache_key in self.prediction_cache:
                    logger.info("Using cached pattern detection result")
                    return self.prediction_cache[cache_key]
            except Exception as e:
                logger.error(f"Error checking cache: {str(e)}")
                # Continue without caching
        
        try:
            # Detect market regime
            market_regime = self.feature_adapter.detect_market_regime(df)
            
            # Preprocess data
            try:
                X, _, timestamps, feature_names = self.preprocessor.preprocess_data(df)
                
                # Check if X is empty
                if X is None or X.shape[0] == 0 or X.shape[2] == 0:
                    logger.warning("Empty preprocessed data")
                    return {
                        "patterns": [],
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                        "error": "Empty preprocessed data"
                    }
                
                # Adapt features
                X_adapted, selected_features = self.feature_adapter.adapt_features(X, feature_names, market_regime)
                logger.info(f"Adapted features from {X.shape[2]} to {X_adapted.shape[2]} dimensions")
                
                # Convert to tensor
                X_tensor = torch.tensor(X_adapted, dtype=torch.float32)
                
                # Make prediction
                self.model.model.eval()
                with torch.no_grad():
                    # Process in batches if needed
                    if len(X_tensor) > self.batch_size:
                        all_outputs = []
                        for i in range(0, len(X_tensor), self.batch_size):
                            batch = X_tensor[i:i+self.batch_size].to(self.model.device)
                            outputs = self.model.model(batch)
                            all_outputs.append(outputs.cpu().numpy())
                        outputs = np.concatenate(all_outputs, axis=0)
                    else:
                        outputs = self.model.model(X_tensor.to(self.model.device)).cpu().numpy()
                
                # Process outputs
                detected_patterns = []
                
                # Get confidence threshold from config
                confidence_threshold = self.config.get("confidence_threshold", 0.6)
                
                # Debug: Log outputs shape and values
                logger.info(f"Model outputs shape: {outputs.shape}")
                logger.info(f"Model outputs sample: {outputs[0, 0, :] if outputs.shape[0] > 0 and outputs.shape[1] > 0 else 'Empty'}")
                logger.info(f"Global confidence threshold: {confidence_threshold}")
                
                # Debug: Log pattern registry
                logger.info(f"Pattern registry: {json.dumps(self.pattern_registry, cls=CustomJSONEncoder, indent=2)}")
                
                # Check each pattern in registry
                for pattern in self.pattern_registry.get("patterns", []):
                    pattern_id = pattern.get("id")
                    pattern_name = pattern.get("name")
                    output_index = pattern.get("output_index", 0)
                    pattern_threshold = pattern.get("confidence_threshold", confidence_threshold)
                    pattern_timeframes = pattern.get("timeframes", [])
                    
                    # Debug: Log pattern details
                    logger.info(f"Processing pattern: {pattern_id}, output_index: {output_index}, threshold: {pattern_threshold}")
                    
                    # Skip if timeframe not supported
                    if pattern_timeframes and timeframe not in pattern_timeframes:
                        logger.info(f"Skipping pattern {pattern_id} for timeframe {timeframe} (supported: {pattern_timeframes})")
                        continue
                    
                    # Get confidence scores for this pattern
                    if output_index < outputs.shape[2]:
                        # Get confidence for each sample and forecast step
                        for i in range(len(outputs)):
                            for j in range(outputs.shape[1]):
                                confidence = float(outputs[i, j, output_index])
                                
                                # Debug: Log confidence value
                                logger.info(f"Pattern {pattern_id}, sample {i}, step {j}: confidence = {confidence}, threshold = {pattern_threshold}")
                                
                                # Check if confidence exceeds threshold
                                if confidence >= pattern_threshold:
                                    # Debug: Log pattern detection
                                    logger.info(f"Detected pattern {pattern_id} with confidence {confidence} >= {pattern_threshold}")
                                    
                                    # Add to detected patterns
                                    detected_patterns.append({
                                        "id": pattern_id,
                                        "name": pattern_name,
                                        "confidence": confidence,
                                        "timestamp": timestamps[i] if i < len(timestamps) else None,
                                        "forecast_step": j + 1,
                                        "timeframe": timeframe
                                    })
                    else:
                        logger.warning(f"Output index {output_index} for pattern {pattern_id} is out of bounds (outputs shape: {outputs.shape})")
                
                # Create result
                result = {
                    "patterns": detected_patterns,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f")
                }
                
                # Debug: Log result using custom JSON encoder
                logger.info(f"Detected {len(detected_patterns)} patterns: {json.dumps(detected_patterns, cls=CustomJSONEncoder, indent=2)}")
                
                # Cache result if enabled
                if self.cache_enabled:
                    try:
                        cache_key = (hash(df.to_json()), timeframe)
                        self.prediction_cache[cache_key] = result
                        
                        # Trim cache if needed
                        if len(self.prediction_cache) > self.cache_size:
                            # Remove oldest item
                            oldest_key = next(iter(self.prediction_cache))
                            del self.prediction_cache[oldest_key]
                    except Exception as e:
                        logger.error(f"Error caching prediction: {str(e)}")
                
                return result
            
            except Exception as e:
                logger.error(f"Error in data preprocessing: {str(e)}")
                return {
                    "patterns": [],
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                    "error": f"Error in data preprocessing: {str(e)}"
                }
        
        except Exception as e:
            logger.error(f"Error in pattern detection: {str(e)}")
            return {
                "patterns": [],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "error": f"Error in pattern detection: {str(e)}"
            }
    
    def _detect_patterns_async(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Detect patterns asynchronously
        
        Args:
            df: Market data DataFrame
            timeframe: Timeframe of the data
            
        Returns:
            dict: Detected patterns
        """
        # This is a placeholder for async implementation
        # In a real implementation, this would use asyncio or similar
        logger.warning("Async pattern detection not implemented, falling back to sync")
        return self._detect_patterns_sync(df, timeframe)


class EnhancedDeepLearningSignalIntegrator:
    """Enhanced deep learning signal integrator for trading signals"""
    
    def __init__(self, 
                 pattern_service: EnhancedPatternRecognitionService = None,
                 config_path: str = None):
        """Initialize enhanced deep learning signal integrator
        
        Args:
            pattern_service: Pattern recognition service
            config_path: Path to configuration file
        """
        self.pattern_service = pattern_service
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize pattern service if not provided
        if self.pattern_service is None:
            self.pattern_service = EnhancedPatternRecognitionService(
                model_path=self.config.get("model_path", "models/pattern_recognition_model.pt"),
                device=self.config.get("device", "cpu"),
                async_mode=self.config.get("async_mode", False),
                config_path=self.config.get("pattern_config_path", None)
            )
        
        logger.info(f"Initialized EnhancedDeepLearningSignalIntegrator")
    
    def _load_config(self) -> Dict:
        """Load configuration from file or use defaults
        
        Returns:
            dict: Configuration dictionary
        """
        default_config = {
            "model_path": "models/pattern_recognition_model.pt",
            "device": "cpu",
            "async_mode": False,
            "pattern_config_path": None,
            "signal_weights": {
                "trend_reversal": 0.8,
                "breakout": 0.7,
                "consolidation": 0.5
            },
            "timeframes": ["1m", "5m", "15m", "1h", "4h"]
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
    
    def integrate_signals(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Integrate deep learning signals with traditional signals
        
        Args:
            df: Market data DataFrame
            timeframe: Timeframe of the data
            
        Returns:
            dict: Integrated signals
        """
        # Check if DataFrame is empty
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided for signal integration")
            return {
                "signals": [],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "error": "Empty DataFrame"
            }
        
        try:
            # Detect patterns
            pattern_signals = self.pattern_service.detect_patterns(df, timeframe)
            
            # Extract patterns
            patterns = pattern_signals.get("patterns", [])
            
            # Create integrated signals
            integrated_signals = []
            
            # Get signal weights from config
            signal_weights = self.config.get("signal_weights", {})
            
            # Process each pattern
            for pattern in patterns:
                pattern_id = pattern.get("id")
                confidence = pattern.get("confidence", 0.0)
                timestamp = pattern.get("timestamp")
                
                # Get weight for this pattern
                weight = signal_weights.get(pattern_id, 0.5)
                
                # Calculate signal strength
                signal_strength = confidence * weight
                
                # Determine signal type
                if pattern_id == "trend_reversal":
                    signal_type = "reversal"
                elif pattern_id == "breakout":
                    signal_type = "breakout"
                elif pattern_id == "consolidation":
                    signal_type = "consolidation"
                else:
                    signal_type = "unknown"
                
                # Create signal
                signal = {
                    "timestamp": timestamp,
                    "type": signal_type,
                    "strength": signal_strength,
                    "confidence": confidence,
                    "source": "deep_learning",
                    "pattern": pattern_id,
                    "timeframe": timeframe
                }
                
                integrated_signals.append(signal)
            
            # Create result
            result = {
                "signals": integrated_signals,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "timeframe": timeframe
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error in signal integration: {str(e)}")
            return {
                "signals": [],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "error": f"Error in signal integration: {str(e)}"
            }
