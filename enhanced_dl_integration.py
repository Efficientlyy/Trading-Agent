#!/usr/bin/env python
"""
Enhanced Deep Learning Integration for Pattern Recognition

This module provides enhanced integration between the deep learning pattern recognition
model and the Trading-Agent system, with improved performance and robustness.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
from threading import Thread, Event
from queue import Queue, Empty
import time

# Import local modules
from enhanced_dl_model import EnhancedPatternRecognitionModel
from enhanced_feature_adapter import EnhancedFeatureAdapter
from dl_data_pipeline import MarketDataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_dl_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_dl_integration")

class EnhancedPatternRecognitionService:
    """Enhanced service for pattern recognition with improved performance and robustness"""
    
    def __init__(self, 
                 model_path: str,
                 device: str = None,
                 async_mode: bool = True,
                 batch_size: int = 16,
                 config_path: str = None,
                 cache_enabled: bool = True,
                 cache_size: int = 100):
        """Initialize enhanced pattern recognition service
        
        Args:
            model_path: Path to trained model
            device: Device to use (cpu, cuda, or None for auto-detection)
            async_mode: Whether to use asynchronous inference
            batch_size: Batch size for inference
            config_path: Path to configuration file
            cache_enabled: Whether to enable prediction caching
            cache_size: Maximum number of items in cache
        """
        self.model_path = model_path
        self.device = device
        self.async_mode = async_mode
        self.batch_size = batch_size
        self.config_path = config_path
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size
        
        # Initialize cache
        self.prediction_cache = {}
        
        # Load configuration
        self.config = self._load_config()
        
        # Update parameters from config
        self.async_mode = self.config.get("async_mode", self.async_mode)
        self.batch_size = self.config.get("batch_size", self.batch_size)
        self.cache_enabled = self.config.get("cache_enabled", self.cache_enabled)
        self.cache_size = self.config.get("cache_size", self.cache_size)
        
        # Create model
        self.model = EnhancedPatternRecognitionModel(
            input_dim=self.config.get("input_dim", 9),
            hidden_dim=self.config.get("hidden_dim", 64),
            output_dim=self.config.get("output_dim", 3),
            sequence_length=self.config.get("sequence_length", 60),
            forecast_horizon=self.config.get("forecast_horizon", 10),
            model_type=self.config.get("model_type", "hybrid"),
            device=self.device
        )
        
        # Load model
        success = self.model.load_model(self.model_path)
        if success:
            logger.info(f"Model loaded successfully from {self.model_path}")
        else:
            logger.error(f"Failed to load model from {self.model_path}")
        
        # Create data preprocessor
        self.preprocessor = MarketDataPreprocessor(
            sequence_length=self.config.get("sequence_length", 60),
            forecast_horizon=self.config.get("forecast_horizon", 10)
        )
        
        # Create feature adapter
        self.feature_adapter = EnhancedFeatureAdapter(
            input_dim=self.config.get("input_dim", 9),
            importance_method=self.config.get("importance_method", "mutual_info"),
            config_path=self.config.get("feature_adapter_config", None),
            cache_enabled=self.cache_enabled,
            cache_size=self.cache_size
        )
        
        # Load pattern registry
        self.pattern_registry = self._load_pattern_registry()
        
        # Initialize async inference if enabled
        if self.async_mode:
            self.inference_queue = Queue()
            self.result_queue = Queue()
            self.stop_event = Event()
            self.inference_thread = Thread(target=self._async_inference_worker)
            self.inference_thread.daemon = True
            self.inference_thread.start()
            logger.info("Async inference worker started")
            logger.info("Started async inference thread")
        
        logger.info(f"Initialized EnhancedPatternRecognitionService with model_path={model_path}, "
                   f"device={device}, async_mode={async_mode}")
    
    def _load_config(self) -> Dict:
        """Load configuration from file or use defaults
        
        Returns:
            dict: Configuration dictionary
        """
        default_config = {
            "input_dim": 9,
            "hidden_dim": 64,
            "output_dim": 3,
            "sequence_length": 60,
            "forecast_horizon": 10,
            "model_type": "hybrid",
            "async_mode": self.async_mode,
            "batch_size": self.batch_size,
            "cache_enabled": self.cache_enabled,
            "cache_size": self.cache_size,
            "confidence_threshold": 0.7,
            "importance_method": "mutual_info",
            "feature_adapter_config": None,
            "pattern_registry_path": "patterns/pattern_registry.json"
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
    
    def _load_pattern_registry(self) -> Dict:
        """Load pattern registry from file or use defaults
        
        Returns:
            dict: Pattern registry
        """
        registry_path = self.config.get("pattern_registry_path", "patterns/pattern_registry.json")
        default_registry = {
            "patterns": [
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
            ],
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    loaded_registry = json.load(f)
                    logger.info(f"Loaded pattern registry from {registry_path}")
                    return loaded_registry
            except Exception as e:
                logger.error(f"Error loading pattern registry: {str(e)}")
                logger.info("Using default pattern registry")
                
                # Save default registry
                try:
                    with open(registry_path, 'w') as f:
                        json.dump(default_registry, f, indent=2)
                except Exception as e:
                    logger.error(f"Error saving default pattern registry: {str(e)}")
        else:
            logger.info(f"Pattern registry not found at {registry_path}, using default")
            
            # Save default registry
            try:
                with open(registry_path, 'w') as f:
                    json.dump(default_registry, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving default pattern registry: {str(e)}")
        
        return default_registry
    
    def _async_inference_worker(self):
        """Worker thread for asynchronous inference"""
        while not self.stop_event.is_set():
            try:
                # Get item from queue with timeout
                item = self.inference_queue.get(timeout=0.1)
                
                # Process item
                request_id, df, timeframe = item
                
                try:
                    # Detect patterns
                    patterns = self._detect_patterns_sync(df, timeframe)
                    
                    # Put result in result queue
                    self.result_queue.put((request_id, patterns, None))
                except Exception as e:
                    # Put error in result queue
                    self.result_queue.put((request_id, None, str(e)))
                
                # Mark task as done
                self.inference_queue.task_done()
            
            except Empty:
                # Queue is empty, sleep briefly
                time.sleep(0.01)
            
            except Exception as e:
                logger.error(f"Error in async inference worker: {str(e)}")
                time.sleep(0.1)
    
    def _detect_patterns_sync(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Detect patterns synchronously
        
        Args:
            df: Market data DataFrame
            timeframe: Timeframe of the data
            
        Returns:
            dict: Detected patterns
        """
        # Check cache first if enabled
        if self.cache_enabled:
            # Create cache key from DataFrame and timeframe
            cache_key = (hash(df.to_json()), timeframe)
            
            if cache_key in self.prediction_cache:
                logger.info("Using cached prediction")
                return self.prediction_cache[cache_key]
        
        # Detect market regime
        market_regime = self.feature_adapter.detect_market_regime(df)
        
        # Preprocess data
        try:
            X, _, timestamps, feature_names = self.preprocessor.preprocess_data(df)
            
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
            confidence_threshold = self.config.get("confidence_threshold", 0.7)
            
            # Check each pattern in registry
            for pattern in self.pattern_registry.get("patterns", []):
                pattern_id = pattern.get("id")
                pattern_name = pattern.get("name")
                output_index = pattern.get("output_index", 0)
                pattern_threshold = pattern.get("confidence_threshold", confidence_threshold)
                pattern_timeframes = pattern.get("timeframes", [])
                
                # Skip if timeframe not supported
                if pattern_timeframes and timeframe not in pattern_timeframes:
                    continue
                
                # Get confidence scores for this pattern
                if output_index < outputs.shape[2]:
                    # Get confidence for each sample and forecast step
                    for i in range(len(outputs)):
                        for j in range(outputs.shape[1]):
                            confidence = float(outputs[i, j, output_index])
                            
                            # Check if confidence exceeds threshold
                            if confidence >= pattern_threshold:
                                # Add to detected patterns
                                detected_patterns.append({
                                    "id": pattern_id,
                                    "name": pattern_name,
                                    "confidence": confidence,
                                    "timestamp": timestamps[i] if i < len(timestamps) else None,
                                    "forecast_step": j + 1,
                                    "timeframe": timeframe
                                })
            
            # Create result
            result = {
                "patterns": detected_patterns,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f")
            }
            
            # Cache result if enabled
            if self.cache_enabled:
                # Manage cache size
                if len(self.prediction_cache) >= self.cache_size:
                    # Remove oldest item
                    oldest_key = next(iter(self.prediction_cache))
                    del self.prediction_cache[oldest_key]
                
                # Add to cache
                self.prediction_cache[cache_key] = result
            
            return result
        
        except Exception as e:
            logger.error(f"Error detecting patterns: {str(e)}")
            return {"patterns": [], "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f")}
    
    def detect_patterns(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Detect patterns in market data
        
        Args:
            df: Market data DataFrame
            timeframe: Timeframe of the data
            
        Returns:
            dict: Detected patterns
        """
        if self.async_mode:
            # Generate request ID
            request_id = hash(time.time())
            
            # Put request in queue
            self.inference_queue.put((request_id, df, timeframe))
            
            # Wait for result
            while True:
                try:
                    # Check result queue
                    result_id, result, error = self.result_queue.get(timeout=1.0)
                    
                    # Check if this is our result
                    if result_id == request_id:
                        if error:
                            logger.error(f"Error in async inference: {error}")
                            return {"patterns": [], "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f")}
                        else:
                            return result
                    else:
                        # Put back in queue
                        self.result_queue.put((result_id, result, error))
                
                except Empty:
                    # No result yet, check if queue is still being processed
                    if not self.inference_thread.is_alive():
                        logger.error("Async inference thread died")
                        # Restart thread
                        self.inference_thread = Thread(target=self._async_inference_worker)
                        self.inference_thread.daemon = True
                        self.inference_thread.start()
                        logger.info("Restarted async inference thread")
                    
                    # Check if request is still in queue
                    if self.inference_queue.empty():
                        logger.error("Request lost, retrying synchronously")
                        return self._detect_patterns_sync(df, timeframe)
        else:
            # Synchronous mode
            return self._detect_patterns_sync(df, timeframe)
    
    def shutdown(self):
        """Shutdown service"""
        if self.async_mode:
            # Stop async inference worker
            self.stop_event.set()
            self.inference_thread.join(timeout=1.0)
            logger.info("Async inference worker stopped")

class EnhancedDeepLearningSignalIntegrator:
    """Enhanced integrator for deep learning signals with improved robustness"""
    
    def __init__(self, 
                 pattern_service: EnhancedPatternRecognitionService,
                 signal_decay_factor: float = 0.9,
                 config_path: str = None):
        """Initialize enhanced deep learning signal integrator
        
        Args:
            pattern_service: Pattern recognition service
            signal_decay_factor: Factor for signal decay over time
            config_path: Path to configuration file
        """
        self.pattern_service = pattern_service
        self.signal_decay_factor = signal_decay_factor
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Update parameters from config
        self.signal_decay_factor = self.config.get("signal_decay_factor", self.signal_decay_factor)
        
        # Initialize signal history
        self.signal_history = []
        self.max_history_length = self.config.get("max_history_length", 100)
        
        logger.info(f"Initialized EnhancedDeepLearningSignalIntegrator with signal_decay_factor={signal_decay_factor}")
    
    def _load_config(self) -> Dict:
        """Load configuration from file or use defaults
        
        Returns:
            dict: Configuration dictionary
        """
        default_config = {
            "signal_decay_factor": self.signal_decay_factor,
            "max_history_length": 100,
            "pattern_weight": 0.6,
            "technical_weight": 0.3,
            "fundamental_weight": 0.1,
            "timeframe_weights": {
                "1m": 0.2,
                "5m": 0.3,
                "15m": 0.3,
                "1h": 0.2
            },
            "pattern_confidence_scaling": True,
            "circuit_breaker": {
                "enabled": True,
                "max_consecutive_signals": 5,
                "cooldown_period": 60  # seconds
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
        
        return default_config
    
    def _calculate_pattern_signal(self, patterns: List[Dict], timeframe: str) -> Dict:
        """Calculate signal from patterns
        
        Args:
            patterns: List of detected patterns
            timeframe: Timeframe of the data
            
        Returns:
            dict: Pattern signal
        """
        # Default signal
        signal = {
            "buy": 0.0,
            "sell": 0.0,
            "patterns": []
        }
        
        # Check if we have patterns
        if not patterns:
            return signal
        
        # Get timeframe weight
        timeframe_weight = self.config.get("timeframe_weights", {}).get(timeframe, 1.0)
        
        # Process each pattern
        for pattern in patterns:
            pattern_id = pattern.get("id")
            confidence = pattern.get("confidence", 0.0)
            
            # Add to patterns list
            signal["patterns"].append({
                "id": pattern_id,
                "name": pattern.get("name"),
                "confidence": confidence,
                "timestamp": pattern.get("timestamp"),
                "forecast_step": pattern.get("forecast_step"),
                "timeframe": timeframe
            })
            
            # Update buy/sell signal based on pattern type
            if pattern_id == "trend_reversal":
                # Trend reversal could be bullish or bearish
                # For simplicity, assume equal probability
                signal["buy"] += confidence * 0.5 * timeframe_weight
                signal["sell"] += confidence * 0.5 * timeframe_weight
            
            elif pattern_id == "breakout":
                # Breakout is generally bullish
                signal["buy"] += confidence * timeframe_weight
            
            elif pattern_id == "consolidation":
                # Consolidation is neutral
                pass
            
            # Add more pattern types as needed
        
        # Normalize signals
        total = signal["buy"] + signal["sell"]
        if total > 0:
            signal["buy"] /= total
            signal["sell"] /= total
        
        return signal
    
    def _apply_circuit_breaker(self, signal: Dict) -> Dict:
        """Apply circuit breaker to prevent excessive signals
        
        Args:
            signal: Signal dictionary
            
        Returns:
            dict: Modified signal
        """
        # Check if circuit breaker is enabled
        if not self.config.get("circuit_breaker", {}).get("enabled", False):
            return signal
        
        # Get circuit breaker parameters
        max_consecutive = self.config.get("circuit_breaker", {}).get("max_consecutive_signals", 5)
        cooldown_period = self.config.get("circuit_breaker", {}).get("cooldown_period", 60)
        
        # Check history for consecutive similar signals
        consecutive_count = 0
        last_signal_time = None
        
        for hist in reversed(self.signal_history):
            # Check if signal is similar
            if (hist["buy"] > 0.5 and signal["buy"] > 0.5) or (hist["sell"] > 0.5 and signal["sell"] > 0.5):
                consecutive_count += 1
                if last_signal_time is None:
                    last_signal_time = hist.get("timestamp")
            else:
                break
        
        # Check if we need to apply circuit breaker
        if consecutive_count >= max_consecutive:
            logger.warning(f"Circuit breaker triggered after {consecutive_count} consecutive similar signals")
            
            # Check cooldown period
            if last_signal_time:
                try:
                    last_time = time.strptime(last_signal_time, "%Y-%m-%dT%H:%M:%S.%f")
                    elapsed = time.time() - time.mktime(last_time)
                    
                    if elapsed < cooldown_period:
                        logger.info(f"In cooldown period, reducing signal strength")
                        # Reduce signal strength
                        signal["buy"] *= 0.5
                        signal["sell"] *= 0.5
                        signal["circuit_breaker_applied"] = True
                except Exception as e:
                    logger.error(f"Error checking cooldown period: {str(e)}")
        
        return signal
    
    def _update_signal_history(self, signal: Dict):
        """Update signal history
        
        Args:
            signal: Signal dictionary
        """
        # Add to history
        self.signal_history.append(signal)
        
        # Trim history if needed
        if len(self.signal_history) > self.max_history_length:
            self.signal_history = self.signal_history[-self.max_history_length:]
    
    def _calculate_historical_context(self) -> Dict:
        """Calculate historical context from signal history
        
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
        
        avg_buy = sum(buy_signals) / len(buy_signals)
        avg_sell = sum(sell_signals) / len(sell_signals)
        
        if avg_buy > avg_sell + 0.1:
            trend = "bullish"
        elif avg_sell > avg_buy + 0.1:
            trend = "bearish"
        else:
            trend = "neutral"
        
        # Calculate strength
        strength = abs(avg_buy - avg_sell)
        
        # Calculate consistency
        if trend == "bullish":
            consistency = sum(1 for s in buy_signals if s > 0.5) / len(buy_signals)
        elif trend == "bearish":
            consistency = sum(1 for s in sell_signals if s > 0.5) / len(sell_signals)
        else:
            consistency = 0.0
        
        return {
            "trend": trend,
            "strength": strength,
            "consistency": consistency
        }
    
    def process_market_data(self, df: pd.DataFrame, timeframe: str, current_state: Dict) -> Dict:
        """Process market data and generate signals
        
        Args:
            df: Market data DataFrame
            timeframe: Timeframe of the data
            current_state: Current market state
            
        Returns:
            dict: Integrated signals
        """
        try:
            # Detect patterns
            pattern_result = self.pattern_service.detect_patterns(df, timeframe)
            patterns = pattern_result.get("patterns", [])
            
            # Calculate pattern signal
            pattern_signal = self._calculate_pattern_signal(patterns, timeframe)
            
            # Get weights from config
            pattern_weight = self.config.get("pattern_weight", 0.6)
            technical_weight = self.config.get("technical_weight", 0.3)
            fundamental_weight = self.config.get("fundamental_weight", 0.1)
            
            # Get technical signals from current state
            technical_signal = {
                "buy": current_state.get("technical_signals", {}).get("buy", 0.0),
                "sell": current_state.get("technical_signals", {}).get("sell", 0.0),
                "indicators": current_state.get("technical_signals", {}).get("indicators", [])
            }
            
            # Get fundamental signals from current state (if available)
            fundamental_signal = {
                "buy": current_state.get("fundamental_signals", {}).get("buy", 0.0),
                "sell": current_state.get("fundamental_signals", {}).get("sell", 0.0)
            }
            
            # Calculate integrated signal
            integrated_buy = (
                pattern_signal["buy"] * pattern_weight +
                technical_signal["buy"] * technical_weight +
                fundamental_signal["buy"] * fundamental_weight
            )
            
            integrated_sell = (
                pattern_signal["sell"] * pattern_weight +
                technical_signal["sell"] * technical_weight +
                fundamental_signal["sell"] * fundamental_weight
            )
            
            # Normalize
            total = integrated_buy + integrated_sell
            if total > 0:
                integrated_buy /= total
                integrated_sell /= total
            
            # Calculate hold signal
            integrated_hold = 1.0 - (integrated_buy + integrated_sell)
            
            # Create signal
            signal = {
                "buy": integrated_buy,
                "sell": integrated_sell,
                "hold": integrated_hold,
                "confidence": max(integrated_buy, integrated_sell),
                "sources": [
                    {
                        "type": "pattern",
                        "weight": pattern_weight,
                        "buy": pattern_signal["buy"],
                        "sell": pattern_signal["sell"],
                        "patterns": pattern_signal["patterns"]
                    },
                    {
                        "type": "technical",
                        "weight": technical_weight,
                        "buy": technical_signal["buy"],
                        "sell": technical_signal["sell"],
                        "indicators": technical_signal["indicators"]
                    }
                ],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f")
            }
            
            # Add fundamental source if available
            if fundamental_weight > 0:
                signal["sources"].append({
                    "type": "fundamental",
                    "weight": fundamental_weight,
                    "buy": fundamental_signal["buy"],
                    "sell": fundamental_signal["sell"]
                })
            
            # Apply circuit breaker
            signal = self._apply_circuit_breaker(signal)
            
            # Update signal history
            self._update_signal_history(signal)
            
            # Add historical context
            signal["historical_context"] = self._calculate_historical_context()
            
            return signal
        
        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")
            return {
                "buy": 0.0,
                "sell": 0.0,
                "hold": 1.0,
                "confidence": 0.0,
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f")
            }
    
    def shutdown(self):
        """Shutdown integrator"""
        # Shutdown pattern service
        if self.pattern_service:
            self.pattern_service.shutdown()

# Example usage
if __name__ == "__main__":
    # Create pattern service
    pattern_service = EnhancedPatternRecognitionService(
        model_path="models/pattern_recognition_model.pt",
        device="cpu",
        async_mode=True
    )
    
    # Create signal integrator
    signal_integrator = EnhancedDeepLearningSignalIntegrator(
        pattern_service=pattern_service
    )
    
    # Create sample data
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="1min"),
        "open": np.random.normal(100, 1, 100),
        "high": np.random.normal(101, 1, 100),
        "low": np.random.normal(99, 1, 100),
        "close": np.random.normal(100, 1, 100),
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
    signal = signal_integrator.process_market_data(df, "1m", current_state)
    
    print(f"Integrated signal: {signal}")
    
    # Shutdown
    signal_integrator.shutdown()
