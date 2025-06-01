#!/usr/bin/env python
"""
Integration of Feature Adapter with Deep Learning Pattern Recognition

This module updates the dl_integration.py file to incorporate the feature adapter
to resolve the dimensionality mismatch between the data pipeline and model.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import threading
import queue
import time

import torch
from torch.utils.data import TensorDataset, DataLoader

# Import local modules
from dl_model import PatternRecognitionModel
from dl_data_pipeline import MarketDataPreprocessor
from feature_adapter import FeatureAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dl_integration_fixed.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dl_integration_fixed")

class PatternRecognitionService:
    """Service for deep learning pattern recognition integration with Trading-Agent"""
    
    def __init__(self, 
                 model_path=None,
                 config_path=None,
                 preprocessor_config_path=None,
                 device=None,
                 batch_size=32,
                 async_mode=True,
                 confidence_threshold=0.7):
        """Initialize pattern recognition service
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
            preprocessor_config_path: Path to preprocessor configuration file
            device: Device to use for inference ('cpu', 'cuda', or None for auto-detection)
            batch_size: Batch size for inference
            async_mode: Whether to use asynchronous inference
            confidence_threshold: Threshold for pattern confidence
        """
        self.model_path = model_path
        self.config_path = config_path
        self.preprocessor_config_path = preprocessor_config_path
        self.batch_size = batch_size
        self.async_mode = async_mode
        self.confidence_threshold = confidence_threshold
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize model
        self.model = None
        if model_path and os.path.exists(model_path):
            self._load_model()
        
        # Initialize preprocessor
        self.preprocessor = self._create_preprocessor()
        
        # Initialize feature adapter
        self.feature_adapter = FeatureAdapter(input_dim=9)  # Model expects 9 features
        
        # Initialize async inference components
        self.inference_queue = queue.Queue() if async_mode else None
        self.result_queue = queue.Queue() if async_mode else None
        self.inference_thread = None
        self.running = False
        
        # Initialize pattern registry
        self.pattern_registry = self._initialize_pattern_registry()
        
        # Start async inference thread if enabled
        if async_mode:
            self._start_async_inference()
        
        logger.info(f"Initialized PatternRecognitionService with model_path={model_path}, device={self.device}, async_mode={async_mode}")
    
    def _load_config(self) -> Dict:
        """Load configuration from file or use defaults
        
        Returns:
            dict: Configuration dictionary
        """
        default_config = {
            "model": {
                "type": "tcn",
                "confidence_mapping": "sigmoid"  # How to map model outputs to confidence scores
            },
            "inference": {
                "batch_size": self.batch_size,
                "async_mode": self.async_mode,
                "confidence_threshold": self.confidence_threshold,
                "max_queue_size": 100,
                "inference_timeout": 5.0  # seconds
            },
            "patterns": {
                "registry_path": "patterns/pattern_registry.json",
                "default_patterns": [
                    {
                        "id": "trend_reversal",
                        "name": "Trend Reversal",
                        "description": "Identifies potential trend reversal points",
                        "output_index": 0,
                        "confidence_threshold": 0.75,
                        "timeframes": ["1m", "5m", "15m", "1h"]
                    },
                    {
                        "id": "breakout",
                        "name": "Breakout",
                        "description": "Identifies potential breakout patterns",
                        "output_index": 1,
                        "confidence_threshold": 0.8,
                        "timeframes": ["5m", "15m", "1h"]
                    },
                    {
                        "id": "consolidation",
                        "name": "Consolidation",
                        "description": "Identifies market consolidation patterns",
                        "output_index": 2,
                        "confidence_threshold": 0.7,
                        "timeframes": ["15m", "1h", "4h"]
                    }
                ]
            },
            "integration": {
                "signal_weight": 0.6,  # Weight of pattern signals in overall decision
                "signal_decay": 0.9,   # Decay factor for signal strength over time
                "min_signal_strength": 0.3  # Minimum signal strength to consider
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
        inference_config = default_config["inference"]
        self.batch_size = inference_config["batch_size"]
        self.async_mode = inference_config["async_mode"]
        self.confidence_threshold = inference_config["confidence_threshold"]
        
        return default_config
    
    def _load_model(self):
        """Load model from disk"""
        try:
            # Create empty model
            self.model = PatternRecognitionModel(device=self.device)
            
            # Load model from disk
            success = self.model.load_model(self.model_path)
            
            if success:
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                logger.error(f"Failed to load model from {self.model_path}")
                self.model = None
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
    
    def _create_preprocessor(self):
        """Create data preprocessor
        
        Returns:
            MarketDataPreprocessor: Preprocessor instance
        """
        # Get model parameters if model is loaded
        sequence_length = None
        forecast_horizon = None
        
        if self.model is not None:
            sequence_length = self.model.sequence_length
            forecast_horizon = self.model.forecast_horizon
        
        # Create preprocessor
        preprocessor = MarketDataPreprocessor(
            config_path=self.preprocessor_config_path,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon
        )
        
        return preprocessor
    
    def _initialize_pattern_registry(self):
        """Initialize pattern registry
        
        Returns:
            dict: Pattern registry
        """
        # Get registry path from config
        registry_path = self.config["patterns"]["registry_path"]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        
        # Check if registry file exists
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    pattern_registry = json.load(f)
                logger.info(f"Loaded pattern registry from {registry_path}")
                return pattern_registry
            except Exception as e:
                logger.error(f"Error loading pattern registry: {str(e)}")
        
        # Use default patterns from config
        default_patterns = self.config["patterns"]["default_patterns"]
        
        # Create registry
        pattern_registry = {
            "patterns": default_patterns,
            "last_updated": datetime.now().isoformat()
        }
        
        # Save registry
        try:
            with open(registry_path, 'w') as f:
                json.dump(pattern_registry, f, indent=2)
            logger.info(f"Created pattern registry at {registry_path}")
        except Exception as e:
            logger.error(f"Error saving pattern registry: {str(e)}")
        
        return pattern_registry
    
    def _start_async_inference(self):
        """Start asynchronous inference thread"""
        if self.inference_thread is not None and self.inference_thread.is_alive():
            logger.warning("Async inference thread already running")
            return
        
        self.running = True
        self.inference_thread = threading.Thread(target=self._async_inference_worker)
        self.inference_thread.daemon = True
        self.inference_thread.start()
        
        logger.info("Started async inference thread")
    
    def _stop_async_inference(self):
        """Stop asynchronous inference thread"""
        self.running = False
        
        if self.inference_thread is not None:
            self.inference_thread.join(timeout=5.0)
            if self.inference_thread.is_alive():
                logger.warning("Async inference thread did not terminate gracefully")
            else:
                logger.info("Stopped async inference thread")
        
        self.inference_thread = None
    
    def _async_inference_worker(self):
        """Worker function for asynchronous inference thread"""
        logger.info("Async inference worker started")
        
        while self.running:
            try:
                # Get item from queue with timeout
                try:
                    item = self.inference_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process item
                request_id, market_data, timeframe = item
                
                # Perform inference
                try:
                    patterns = self._process_market_data(market_data, timeframe)
                    
                    # Put result in result queue
                    self.result_queue.put((request_id, patterns, None))
                except Exception as e:
                    logger.error(f"Error in async inference: {str(e)}")
                    self.result_queue.put((request_id, None, str(e)))
                
                # Mark task as done
                self.inference_queue.task_done()
            
            except Exception as e:
                logger.error(f"Error in async inference worker: {str(e)}")
        
        logger.info("Async inference worker stopped")
    
    def _process_market_data(self, market_data, timeframe):
        """Process market data and detect patterns
        
        Args:
            market_data: Market data (DataFrame or dict)
            timeframe: Timeframe of the data
            
        Returns:
            dict: Detected patterns
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Convert market data to DataFrame if it's a dict
        if isinstance(market_data, dict):
            market_data = pd.DataFrame(market_data)
        
        # Preprocess data
        try:
            # Extract features
            features_df = self.preprocessor._add_derived_features(market_data.copy())
            features_df = self.preprocessor._add_temporal_features(features_df)
            
            # Normalize data
            normalized_df = self.preprocessor._normalize_data(features_df)
            
            # Create sequence
            X, _, _, feature_names = self.preprocessor._create_sequences(normalized_df)
            
            # Check if we have enough data
            if len(X) == 0:
                logger.warning("Not enough data to create sequences")
                return {"patterns": [], "timestamp": datetime.now().isoformat()}
            
            # Apply feature adapter to match model input dimensions
            X_adapted, selected_features = self.feature_adapter.adapt_features(X, feature_names)
            logger.info(f"Adapted features from {X.shape[2]} to {X_adapted.shape[2]} dimensions")
            
            # Convert to tensor
            X_tensor = torch.tensor(X_adapted, dtype=torch.float32)
            
            # Create dataloader
            dataloader = DataLoader(
                TensorDataset(X_tensor),
                batch_size=self.batch_size,
                shuffle=False
            )
            
            # Perform inference
            self.model.model.eval()
            predictions = []
            
            with torch.no_grad():
                for batch in dataloader:
                    # Move data to device
                    batch_data = batch[0].to(self.device)
                    
                    # Forward pass
                    output = self.model.model(batch_data)
                    
                    # Move predictions to CPU
                    predictions.append(output.cpu().numpy())
            
            # Concatenate predictions
            predictions = np.concatenate(predictions, axis=0)
            
            # Map predictions to confidence scores
            confidence_mapping = self.config["model"]["confidence_mapping"]
            if confidence_mapping == "sigmoid":
                confidence_scores = 1 / (1 + np.exp(-predictions))
            elif confidence_mapping == "softmax":
                confidence_scores = np.exp(predictions) / np.sum(np.exp(predictions), axis=2, keepdims=True)
            elif confidence_mapping == "tanh":
                confidence_scores = (np.tanh(predictions) + 1) / 2
            else:
                # Default to raw values
                confidence_scores = predictions
            
            # Detect patterns
            patterns = self._detect_patterns(confidence_scores, timeframe)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")
            raise
    
    def _detect_patterns(self, confidence_scores, timeframe):
        """Detect patterns from model predictions
        
        Args:
            confidence_scores: Confidence scores from model
            timeframe: Timeframe of the data
            
        Returns:
            dict: Detected patterns
        """
        # Get patterns from registry
        patterns = self.pattern_registry["patterns"]
        
        # Filter patterns by timeframe
        timeframe_patterns = [p for p in patterns if timeframe in p["timeframes"]]
        
        # Initialize results
        detected_patterns = []
        
        # Get latest confidence scores (last sequence)
        latest_scores = confidence_scores[-1]
        
        # Check each pattern
        for pattern in timeframe_patterns:
            pattern_id = pattern["id"]
            output_index = pattern["output_index"]
            threshold = pattern.get("confidence_threshold", self.confidence_threshold)
            
            # Get confidence score for this pattern
            if output_index < latest_scores.shape[1]:
                # Get confidence across forecast horizon
                pattern_confidence = latest_scores[:, output_index]
                
                # Use maximum confidence
                max_confidence = float(np.max(pattern_confidence))
                
                # Check if confidence exceeds threshold
                if max_confidence >= threshold:
                    # Get time step of maximum confidence
                    max_step = int(np.argmax(pattern_confidence))
                    
                    # Add to detected patterns
                    detected_patterns.append({
                        "id": pattern_id,
                        "name": pattern["name"],
                        "confidence": max_confidence,
                        "timeframe": timeframe,
                        "forecast_step": max_step,
                        "timestamp": datetime.now().isoformat()
                    })
        
        # Create result
        result = {
            "patterns": detected_patterns,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def detect_patterns(self, market_data, timeframe, async_request=None):
        """Detect patterns in market data
        
        Args:
            market_data: Market data (DataFrame or dict)
            timeframe: Timeframe of the data
            async_request: Request ID for asynchronous inference
            
        Returns:
            dict: Detected patterns (or request ID if async)
        """
        # Check if model is loaded
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Check if using async mode
        if self.async_mode and async_request is not None:
            # Put request in queue
            self.inference_queue.put((async_request, market_data, timeframe))
            return {"request_id": async_request}
        
        # Synchronous inference
        return self._process_market_data(market_data, timeframe)
    
    def get_async_result(self, request_id, timeout=None):
        """Get result of asynchronous inference
        
        Args:
            request_id: Request ID
            timeout: Timeout in seconds
            
        Returns:
            dict: Detected patterns or None if not ready
        """
        if not self.async_mode:
            raise ValueError("Not in async mode")
        
        # Check if result is available
        try:
            # Get result from queue
            result_request_id, patterns, error = self.result_queue.get(timeout=timeout)
            
            # Check if this is the requested result
            if result_request_id != request_id:
                # Put back in queue and return None
                self.result_queue.put((result_request_id, patterns, error))
                return None
            
            # Mark task as done
            self.result_queue.task_done()
            
            # Check for error
            if error is not None:
                return {"error": error, "request_id": request_id}
            
            # Return patterns
            return patterns
            
        except queue.Empty:
            # Result not ready
            return None
    
    def generate_trading_signals(self, patterns, current_state=None):
        """Generate trading signals from detected patterns
        
        Args:
            patterns: Detected patterns
            current_state: Current market state
            
        Returns:
            dict: Trading signals
        """
        # Get integration config
        integration_config = self.config["integration"]
        signal_weight = integration_config["signal_weight"]
        min_signal_strength = integration_config["min_signal_strength"]
        
        # Initialize signals
        signals = {
            "buy": 0.0,
            "sell": 0.0,
            "patterns": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Check if we have patterns
        if not patterns or "patterns" not in patterns or not patterns["patterns"]:
            return signals
        
        # Process each pattern
        for pattern in patterns["patterns"]:
            pattern_id = pattern["id"]
            confidence = pattern["confidence"]
            
            # Determine signal direction based on pattern ID
            # This is a simplified example - in a real system, this would be more sophisticated
            if pattern_id == "trend_reversal":
                # Trend reversal could be buy or sell depending on current trend
                if current_state and "trend" in current_state:
                    if current_state["trend"] == "up":
                        signals["sell"] += confidence * signal_weight
                    else:
                        signals["buy"] += confidence * signal_weight
                else:
                    # Without trend information, assume equal probability
                    signals["buy"] += confidence * signal_weight * 0.5
                    signals["sell"] += confidence * signal_weight * 0.5
            
            elif pattern_id == "breakout":
                # Breakout is typically a buy signal
                signals["buy"] += confidence * signal_weight
            
            elif pattern_id == "consolidation":
                # Consolidation is neutral, slightly favoring continuation
                if current_state and "trend" in current_state:
                    if current_state["trend"] == "up":
                        signals["buy"] += confidence * signal_weight * 0.7
                    else:
                        signals["sell"] += confidence * signal_weight * 0.7
            
            # Add pattern to signal details
            signals["patterns"].append({
                "id": pattern_id,
                "name": pattern["name"],
                "confidence": confidence,
                "timeframe": pattern["timeframe"]
            })
        
        # Normalize signals
        max_signal = max(signals["buy"], signals["sell"])
        if max_signal > 0:
            signals["buy"] /= max_signal
            signals["sell"] /= max_signal
        
        # Apply minimum signal strength
        if signals["buy"] < min_signal_strength:
            signals["buy"] = 0.0
        
        if signals["sell"] < min_signal_strength:
            signals["sell"] = 0.0
        
        return signals
    
    def update_pattern_registry(self, new_patterns=None):
        """Update pattern registry
        
        Args:
            new_patterns: New patterns to add or update
            
        Returns:
            dict: Updated pattern registry
        """
        if new_patterns is None:
            return self.pattern_registry
        
        # Get existing patterns
        existing_patterns = self.pattern_registry["patterns"]
        
        # Update existing patterns or add new ones
        for new_pattern in new_patterns:
            pattern_id = new_pattern["id"]
            
            # Check if pattern already exists
            existing_idx = None
            for i, pattern in enumerate(existing_patterns):
                if pattern["id"] == pattern_id:
                    existing_idx = i
                    break
            
            # Update or add
            if existing_idx is not None:
                existing_patterns[existing_idx] = new_pattern
            else:
                existing_patterns.append(new_pattern)
        
        # Update registry
        self.pattern_registry["patterns"] = existing_patterns
        self.pattern_registry["last_updated"] = datetime.now().isoformat()
        
        # Save registry
        registry_path = self.config["patterns"]["registry_path"]
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        
        try:
            with open(registry_path, 'w') as f:
                json.dump(self.pattern_registry, f, indent=2)
            logger.info(f"Updated pattern registry at {registry_path}")
        except Exception as e:
            logger.error(f"Error saving pattern registry: {str(e)}")
        
        return self.pattern_registry
    
    def get_pattern_registry(self):
        """Get pattern registry
        
        Returns:
            dict: Pattern registry
        """
        return self.pattern_registry
    
    def shutdown(self):
        """Shutdown service"""
        if self.async_mode:
            self._stop_async_inference()
        
        logger.info("Service shutdown complete")

class DeepLearningSignalIntegrator:
    """Integrates deep learning pattern signals with Trading-Agent system"""
    
    def __init__(self, 
                 pattern_service=None,
                 config_path=None,
                 signal_decay_factor=0.9,
                 signal_history_length=20):
        """Initialize signal integrator
        
        Args:
            pattern_service: PatternRecognitionService instance
            config_path: Path to configuration file
            signal_decay_factor: Decay factor for historical signals
            signal_history_length: Maximum length of signal history
        """
        self.pattern_service = pattern_service
        self.config_path = config_path
        self.signal_decay_factor = signal_decay_factor
        self.signal_history_length = signal_history_length
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize signal history
        self.signal_history = []
        
        # Initialize pattern service if not provided
        if self.pattern_service is None:
            self._initialize_pattern_service()
        
        logger.info(f"Initialized DeepLearningSignalIntegrator with signal_decay_factor={signal_decay_factor}")
    
    def _load_config(self) -> Dict:
        """Load configuration from file or use defaults
        
        Returns:
            dict: Configuration dictionary
        """
        default_config = {
            "integration": {
                "signal_decay_factor": self.signal_decay_factor,
                "signal_history_length": self.signal_history_length,
                "signal_weights": {
                    "pattern": 0.6,
                    "technical": 0.3,
                    "fundamental": 0.1
                },
                "timeframe_weights": {
                    "1m": 0.1,
                    "5m": 0.2,
                    "15m": 0.3,
                    "1h": 0.4
                }
            },
            "pattern_service": {
                "model_path": "models/pattern_recognition_model.pt",
                "config_path": "config/pattern_service_config.json",
                "async_mode": True
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
        integration_config = default_config["integration"]
        self.signal_decay_factor = integration_config["signal_decay_factor"]
        self.signal_history_length = integration_config["signal_history_length"]
        
        return default_config
    
    def _initialize_pattern_service(self):
        """Initialize pattern recognition service"""
        pattern_service_config = self.config["pattern_service"]
        
        try:
            self.pattern_service = PatternRecognitionService(
                model_path=pattern_service_config["model_path"],
                config_path=pattern_service_config["config_path"],
                async_mode=pattern_service_config["async_mode"]
            )
            logger.info("Initialized pattern recognition service")
        except Exception as e:
            logger.error(f"Error initializing pattern recognition service: {str(e)}")
            self.pattern_service = None
    
    def process_market_data(self, market_data, timeframe, current_state=None):
        """Process market data and generate integrated signals
        
        Args:
            market_data: Market data (DataFrame or dict)
            timeframe: Timeframe of the data
            current_state: Current market state
            
        Returns:
            dict: Integrated signals
        """
        if self.pattern_service is None:
            raise ValueError("Pattern recognition service not initialized")
        
        # Detect patterns
        patterns = self.pattern_service.detect_patterns(market_data, timeframe)
        
        # Generate trading signals from patterns
        pattern_signals = self.pattern_service.generate_trading_signals(patterns, current_state)
        
        # Integrate signals
        integrated_signals = self._integrate_signals(pattern_signals, timeframe, current_state)
        
        # Update signal history
        self._update_signal_history(integrated_signals)
        
        return integrated_signals
    
    def process_market_data_async(self, market_data, timeframe, request_id, current_state=None):
        """Process market data asynchronously
        
        Args:
            market_data: Market data (DataFrame or dict)
            timeframe: Timeframe of the data
            request_id: Request ID for asynchronous processing
            current_state: Current market state
            
        Returns:
            dict: Request information
        """
        if self.pattern_service is None:
            raise ValueError("Pattern recognition service not initialized")
        
        # Detect patterns asynchronously
        result = self.pattern_service.detect_patterns(market_data, timeframe, async_request=request_id)
        
        return {
            "request_id": request_id,
            "status": "processing",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_async_result(self, request_id, timeout=None, current_state=None):
        """Get result of asynchronous processing
        
        Args:
            request_id: Request ID
            timeout: Timeout in seconds
            current_state: Current market state
            
        Returns:
            dict: Integrated signals or None if not ready
        """
        if self.pattern_service is None:
            raise ValueError("Pattern recognition service not initialized")
        
        # Get pattern detection result
        patterns = self.pattern_service.get_async_result(request_id, timeout)
        
        # Check if result is ready
        if patterns is None:
            return None
        
        # Check for error
        if "error" in patterns:
            return {
                "error": patterns["error"],
                "request_id": request_id,
                "status": "error",
                "timestamp": datetime.now().isoformat()
            }
        
        # Generate trading signals from patterns
        pattern_signals = self.pattern_service.generate_trading_signals(patterns, current_state)
        
        # Get timeframe from patterns
        timeframe = None
        if patterns and "patterns" in patterns and patterns["patterns"]:
            timeframe = patterns["patterns"][0].get("timeframe", None)
        
        # Integrate signals
        integrated_signals = self._integrate_signals(pattern_signals, timeframe, current_state)
        
        # Update signal history
        self._update_signal_history(integrated_signals)
        
        # Add request information
        integrated_signals["request_id"] = request_id
        integrated_signals["status"] = "completed"
        
        return integrated_signals
    
    def _integrate_signals(self, pattern_signals, timeframe, current_state=None):
        """Integrate pattern signals with other signals
        
        Args:
            pattern_signals: Pattern-based signals
            timeframe: Timeframe of the data
            current_state: Current market state
            
        Returns:
            dict: Integrated signals
        """
        # Get integration config
        integration_config = self.config["integration"]
        signal_weights = integration_config["signal_weights"]
        timeframe_weights = integration_config["timeframe_weights"]
        
        # Get timeframe weight
        timeframe_weight = timeframe_weights.get(timeframe, 0.25)  # Default to 0.25 if not specified
        
        # Initialize integrated signals
        integrated_signals = {
            "buy": 0.0,
            "sell": 0.0,
            "hold": 0.0,
            "confidence": 0.0,
            "sources": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Add pattern signals
        pattern_weight = signal_weights["pattern"]
        integrated_signals["buy"] += pattern_signals["buy"] * pattern_weight * timeframe_weight
        integrated_signals["sell"] += pattern_signals["sell"] * pattern_weight * timeframe_weight
        
        # Add source information
        integrated_signals["sources"].append({
            "type": "pattern",
            "weight": pattern_weight * timeframe_weight,
            "buy": pattern_signals["buy"],
            "sell": pattern_signals["sell"],
            "patterns": pattern_signals.get("patterns", [])
        })
        
        # Add technical signals if available in current state
        if current_state and "technical_signals" in current_state:
            technical_signals = current_state["technical_signals"]
            technical_weight = signal_weights["technical"]
            
            integrated_signals["buy"] += technical_signals.get("buy", 0.0) * technical_weight * timeframe_weight
            integrated_signals["sell"] += technical_signals.get("sell", 0.0) * technical_weight * timeframe_weight
            
            integrated_signals["sources"].append({
                "type": "technical",
                "weight": technical_weight * timeframe_weight,
                "buy": technical_signals.get("buy", 0.0),
                "sell": technical_signals.get("sell", 0.0),
                "indicators": technical_signals.get("indicators", [])
            })
        
        # Add fundamental signals if available in current state
        if current_state and "fundamental_signals" in current_state:
            fundamental_signals = current_state["fundamental_signals"]
            fundamental_weight = signal_weights["fundamental"]
            
            integrated_signals["buy"] += fundamental_signals.get("buy", 0.0) * fundamental_weight * timeframe_weight
            integrated_signals["sell"] += fundamental_signals.get("sell", 0.0) * fundamental_weight * timeframe_weight
            
            integrated_signals["sources"].append({
                "type": "fundamental",
                "weight": fundamental_weight * timeframe_weight,
                "buy": fundamental_signals.get("buy", 0.0),
                "sell": fundamental_signals.get("sell", 0.0),
                "factors": fundamental_signals.get("factors", [])
            })
        
        # Calculate hold signal
        integrated_signals["hold"] = 1.0 - (integrated_signals["buy"] + integrated_signals["sell"])
        if integrated_signals["hold"] < 0.0:
            # Normalize if buy + sell > 1.0
            total = integrated_signals["buy"] + integrated_signals["sell"]
            integrated_signals["buy"] /= total
            integrated_signals["sell"] /= total
            integrated_signals["hold"] = 0.0
        
        # Calculate confidence
        integrated_signals["confidence"] = max(integrated_signals["buy"], integrated_signals["sell"])
        
        # Add historical context
        integrated_signals["historical_context"] = self._get_historical_context()
        
        return integrated_signals
    
    def _update_signal_history(self, signals):
        """Update signal history
        
        Args:
            signals: New signals to add to history
        """
        # Add new signals to history
        self.signal_history.append({
            "buy": signals["buy"],
            "sell": signals["sell"],
            "hold": signals["hold"],
            "confidence": signals["confidence"],
            "timestamp": signals["timestamp"]
        })
        
        # Trim history if too long
        if len(self.signal_history) > self.signal_history_length:
            self.signal_history = self.signal_history[-self.signal_history_length:]
    
    def _get_historical_context(self):
        """Get historical context from signal history
        
        Returns:
            dict: Historical context
        """
        if not self.signal_history:
            return {
                "trend": "neutral",
                "strength": 0.0,
                "consistency": 0.0
            }
        
        # Calculate trend
        buy_signals = [s["buy"] for s in self.signal_history]
        sell_signals = [s["sell"] for s in self.signal_history]
        
        # Apply decay factor to give more weight to recent signals
        weights = [self.signal_decay_factor ** i for i in range(len(buy_signals))]
        weights.reverse()  # Reverse to give higher weight to recent signals
        
        # Normalize weights
        weights_sum = sum(weights)
        if weights_sum > 0:
            weights = [w / weights_sum for w in weights]
        
        # Calculate weighted averages
        avg_buy = sum(b * w for b, w in zip(buy_signals, weights))
        avg_sell = sum(s * w for s, w in zip(sell_signals, weights))
        
        # Determine trend
        trend = "neutral"
        if avg_buy > avg_sell:
            trend = "bullish"
        elif avg_sell > avg_buy:
            trend = "bearish"
        
        # Calculate strength
        strength = abs(avg_buy - avg_sell)
        
        # Calculate consistency
        if trend == "bullish":
            consistency = sum(1 for b, s in zip(buy_signals, sell_signals) if b > s) / len(buy_signals)
        elif trend == "bearish":
            consistency = sum(1 for b, s in zip(buy_signals, sell_signals) if s > b) / len(buy_signals)
        else:
            consistency = sum(1 for b, s in zip(buy_signals, sell_signals) if abs(b - s) < 0.1) / len(buy_signals)
        
        return {
            "trend": trend,
            "strength": float(strength),
            "consistency": float(consistency),
            "avg_buy": float(avg_buy),
            "avg_sell": float(avg_sell)
        }
    
    def get_signal_history(self):
        """Get signal history
        
        Returns:
            list: Signal history
        """
        return self.signal_history
    
    def shutdown(self):
        """Shutdown integrator"""
        if self.pattern_service is not None:
            self.pattern_service.shutdown()
        
        logger.info("Signal integrator shutdown complete")

# Example usage
if __name__ == "__main__":
    # Create pattern recognition service
    pattern_service = PatternRecognitionService(
        model_path="models/pattern_recognition_model.pt",
        config_path="config/pattern_service_config.json"
    )
    
    # Create signal integrator
    signal_integrator = DeepLearningSignalIntegrator(
        pattern_service=pattern_service,
        config_path="config/signal_integrator_config.json"
    )
    
    # Example market data (replace with actual data)
    market_data = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1min"),
        "open": np.random.normal(100, 5, 100),
        "high": np.random.normal(105, 5, 100),
        "low": np.random.normal(95, 5, 100),
        "close": np.random.normal(100, 5, 100),
        "volume": np.random.normal(1000, 200, 100)
    })
    
    # Process market data
    signals = signal_integrator.process_market_data(market_data, "1m")
    
    print(f"Integrated signals: {signals}")
    
    # Shutdown
    signal_integrator.shutdown()
