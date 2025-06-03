#!/usr/bin/env python
"""
Unified Data Pipeline for LLM Strategic Overseer

This module provides a centralized data access layer for consistent data format
and access across all components of the LLM Strategic Overseer system.
"""

import os
import sys
import json
import time
import logging
import threading
import asyncio
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("unified_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("unified_pipeline")

class DataCache:
    """Cache for market data with TTL support"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        """Initialize data cache
        
        Args:
            max_size: Maximum number of items in cache
            ttl: Time to live in seconds
        """
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.Lock()
        
        logger.info(f"Initialized DataCache with max_size={max_size}, ttl={ttl}s")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached item or None if not found or expired
        """
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check if item is expired
            if time.time() - self.timestamps[key] > self.ttl:
                self.cache.pop(key)
                self.timestamps.pop(key)
                return None
            
            return self.cache[key]
    
    def set(self, key: str, value: Any):
        """Set item in cache
        
        Args:
            key: Cache key
            value: Item to cache
        """
        with self.lock:
            # If cache is full, remove oldest item
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = min(self.timestamps, key=self.timestamps.get)
                self.cache.pop(oldest_key)
                self.timestamps.pop(oldest_key)
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            
            logger.info("Cache cleared")

class UnifiedDataPipeline:
    """
    Unified Data Pipeline for centralized data access.
    
    This class provides a centralized data access layer with standardized
    data formats, caching, and data integrity validation.
    """
    
    def __init__(self, event_bus=None):
        """
        Initialize Unified Data Pipeline.
        
        Args:
            event_bus: Event Bus instance
        """
        self.event_bus = event_bus
        
        # Initialize caches
        self.market_data_cache = DataCache(max_size=100, ttl=60)  # 1 minute TTL for market data
        self.klines_cache = DataCache(max_size=100, ttl=300)      # 5 minutes TTL for klines
        self.pattern_cache = DataCache(max_size=50, ttl=3600)     # 1 hour TTL for patterns
        self.decision_cache = DataCache(max_size=50, ttl=3600)    # 1 hour TTL for decisions
        
        # Initialize data stores
        self.market_data = {}
        self.klines_data = {}
        self.pattern_data = {}
        self.decision_data = {}
        
        # Supported assets and timeframes
        self.supported_assets = ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
        self.supported_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        # Register event handlers if event bus is provided
        if self.event_bus:
            self._register_event_handlers()
        
        logger.info("Unified Data Pipeline initialized")
    
    def set_event_bus(self, event_bus):
        """
        Set Event Bus instance.
        
        Args:
            event_bus: Event Bus instance
        """
        self.event_bus = event_bus
        self._register_event_handlers()
        
        logger.info("Event Bus set")
    
    def _register_event_handlers(self):
        """Register event handlers with Event Bus."""
        self.event_bus.subscribe("market.data", self._handle_market_data)
        self.event_bus.subscribe("market.klines", self._handle_klines_data)
        self.event_bus.subscribe("visualization.pattern_detected", self._handle_pattern_data)
        self.event_bus.subscribe("llm.strategic_decision", self._handle_decision_data)
        
        logger.info("Event handlers registered")
    
    async def _handle_market_data(self, topic: str, data: Dict[str, Any]):
        """
        Handle market data event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        symbol = data.get("symbol")
        if not symbol:
            logger.warning("Market data missing symbol")
            return
        
        # Store market data
        self.update_market_data(symbol, data)
    
    async def _handle_klines_data(self, topic: str, data: Dict[str, Any]):
        """
        Handle klines data event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        symbol = data.get("symbol")
        timeframe = data.get("timeframe")
        if not symbol or not timeframe:
            logger.warning("Klines data missing symbol or timeframe")
            return
        
        # Store klines data
        self.update_klines_data(symbol, timeframe, data.get("klines", []))
    
    async def _handle_pattern_data(self, topic: str, data: Dict[str, Any]):
        """
        Handle pattern data event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        pattern_id = data.get("id")
        if not pattern_id:
            logger.warning("Pattern data missing ID")
            return
        
        # Store pattern data
        self.update_pattern_data(pattern_id, data)
    
    async def _handle_decision_data(self, topic: str, data: Dict[str, Any]):
        """
        Handle decision data event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        decision_id = data.get("id")
        if not decision_id:
            logger.warning("Decision data missing ID")
            return
        
        # Store decision data
        self.update_decision_data(decision_id, data)
    
    def update_market_data(self, symbol: str, data: Dict[str, Any]) -> bool:
        """
        Update market data.
        
        Args:
            symbol: Trading pair symbol
            data: Market data
            
        Returns:
            True if updated successfully, False otherwise
        """
        # Validate symbol
        if symbol not in self.supported_assets:
            logger.warning(f"Unsupported symbol: {symbol}")
            return False
        
        # Validate data
        if not self._validate_market_data(data):
            logger.warning(f"Invalid market data for {symbol}")
            return False
        
        # Add timestamp if not present
        if "timestamp" not in data:
            data["timestamp"] = datetime.now().isoformat()
        
        # Standardize data format
        std_data = self._standardize_market_data(data)
        
        # Update market data store
        self.market_data[symbol] = std_data
        
        # Update cache
        self.market_data_cache.set(symbol, std_data)
        
        # Publish event if event bus is available
        if self.event_bus:
            asyncio.create_task(self.event_bus.publish("pipeline.market_data_updated", {
                "symbol": symbol,
                "data": std_data
            }))
        
        logger.info(f"Updated market data for {symbol}")
        return True
    
    def update_klines_data(self, symbol: str, timeframe: str, klines: List[Dict[str, Any]]) -> bool:
        """
        Update klines data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            klines: Klines data
            
        Returns:
            True if updated successfully, False otherwise
        """
        # Validate symbol and timeframe
        if symbol not in self.supported_assets:
            logger.warning(f"Unsupported symbol: {symbol}")
            return False
        
        if timeframe not in self.supported_timeframes:
            logger.warning(f"Unsupported timeframe: {timeframe}")
            return False
        
        # Validate klines
        if not self._validate_klines_data(klines):
            logger.warning(f"Invalid klines data for {symbol} {timeframe}")
            return False
        
        # Standardize klines
        std_klines = self._standardize_klines_data(klines)
        
        # Initialize symbol data if not exists
        if symbol not in self.klines_data:
            self.klines_data[symbol] = {}
        
        # Update klines data store
        self.klines_data[symbol][timeframe] = std_klines
        
        # Update cache
        cache_key = f"{symbol}_{timeframe}"
        self.klines_cache.set(cache_key, std_klines)
        
        # Publish event if event bus is available
        if self.event_bus:
            asyncio.create_task(self.event_bus.publish("pipeline.klines_updated", {
                "symbol": symbol,
                "timeframe": timeframe,
                "klines": std_klines
            }))
        
        logger.info(f"Updated klines data for {symbol} {timeframe}")
        return True
    
    def update_pattern_data(self, pattern_id: str, data: Dict[str, Any]) -> bool:
        """
        Update pattern data.
        
        Args:
            pattern_id: Pattern ID
            data: Pattern data
            
        Returns:
            True if updated successfully, False otherwise
        """
        # Validate data
        if not self._validate_pattern_data(data):
            logger.warning(f"Invalid pattern data for {pattern_id}")
            return False
        
        # Add timestamp if not present
        if "timestamp" not in data:
            data["timestamp"] = datetime.now().isoformat()
        
        # Standardize data format
        std_data = self._standardize_pattern_data(data)
        
        # Update pattern data store
        self.pattern_data[pattern_id] = std_data
        
        # Update cache
        self.pattern_cache.set(pattern_id, std_data)
        
        # Publish event if event bus is available
        if self.event_bus:
            asyncio.create_task(self.event_bus.publish("pipeline.pattern_updated", {
                "pattern_id": pattern_id,
                "data": std_data
            }))
        
        logger.info(f"Updated pattern data for {pattern_id}")
        return True
    
    def update_decision_data(self, decision_id: str, data: Dict[str, Any]) -> bool:
        """
        Update decision data.
        
        Args:
            decision_id: Decision ID
            data: Decision data
            
        Returns:
            True if updated successfully, False otherwise
        """
        # Validate data
        if not self._validate_decision_data(data):
            logger.warning(f"Invalid decision data for {decision_id}")
            return False
        
        # Add timestamp if not present
        if "timestamp" not in data:
            data["timestamp"] = datetime.now().isoformat()
        
        # Standardize data format
        std_data = self._standardize_decision_data(data)
        
        # Update decision data store
        self.decision_data[decision_id] = std_data
        
        # Update cache
        self.decision_cache.set(decision_id, std_data)
        
        # Publish event if event bus is available
        if self.event_bus:
            asyncio.create_task(self.event_bus.publish("pipeline.decision_updated", {
                "decision_id": decision_id,
                "data": std_data
            }))
        
        logger.info(f"Updated decision data for {decision_id}")
        return True
    
    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get market data for symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Market data or None if not found
        """
        # Check cache first
        cached_data = self.market_data_cache.get(symbol)
        if cached_data:
            return cached_data
        
        # Check data store
        if symbol in self.market_data:
            return self.market_data[symbol]
        
        return None
    
    def get_klines_data(self, symbol: str, timeframe: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get klines data for symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            Klines data or None if not found
        """
        # Check cache first
        cache_key = f"{symbol}_{timeframe}"
        cached_data = self.klines_cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # Check data store
        if symbol in self.klines_data and timeframe in self.klines_data[symbol]:
            return self.klines_data[symbol][timeframe]
        
        return None
    
    def get_pattern_data(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Get pattern data by ID.
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Pattern data or None if not found
        """
        # Check cache first
        cached_data = self.pattern_cache.get(pattern_id)
        if cached_data:
            return cached_data
        
        # Check data store
        if pattern_id in self.pattern_data:
            return self.pattern_data[pattern_id]
        
        return None
    
    def get_decision_data(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """
        Get decision data by ID.
        
        Args:
            decision_id: Decision ID
            
        Returns:
            Decision data or None if not found
        """
        # Check cache first
        cached_data = self.decision_cache.get(decision_id)
        if cached_data:
            return cached_data
        
        # Check data store
        if decision_id in self.decision_data:
            return self.decision_data[decision_id]
        
        return None
    
    def get_all_market_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all market data.
        
        Returns:
            Dictionary of market data by symbol
        """
        return self.market_data.copy()
    
    def get_all_klines_data(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Get all klines data.
        
        Returns:
            Dictionary of klines data by symbol and timeframe
        """
        return self.klines_data.copy()
    
    def get_all_pattern_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all pattern data.
        
        Returns:
            Dictionary of pattern data by ID
        """
        return self.pattern_data.copy()
    
    def get_all_decision_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all decision data.
        
        Returns:
            Dictionary of decision data by ID
        """
        return self.decision_data.copy()
    
    def get_patterns_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get patterns for symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            List of patterns for symbol
        """
        patterns = []
        for pattern_id, pattern in self.pattern_data.items():
            if pattern.get("symbol") == symbol:
                patterns.append(pattern)
        
        # Sort by timestamp (newest first)
        patterns.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return patterns
    
    def get_decisions_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get decisions for symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            List of decisions for symbol
        """
        decisions = []
        for decision_id, decision in self.decision_data.items():
            if decision.get("symbol") == symbol:
                decisions.append(decision)
        
        # Sort by timestamp (newest first)
        decisions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return decisions
    
    def get_klines_dataframe(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Get klines data as pandas DataFrame.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            DataFrame with OHLCV data or None if not found
        """
        klines = self.get_klines_data(symbol, timeframe)
        if not klines:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(klines)
        
        # Set timestamp as index
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        
        # Convert columns to numeric
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
        
        return df
    
    def _validate_market_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate market data.
        
        Args:
            data: Market data
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["symbol"]
        for field in required_fields:
            if field not in data:
                logger.warning(f"Market data missing required field: {field}")
                return False
        
        return True
    
    def _validate_klines_data(self, klines: List[Dict[str, Any]]) -> bool:
        """
        Validate klines data.
        
        Args:
            klines: Klines data
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(klines, list):
            logger.warning("Klines data must be a list")
            return False
        
        required_fields = ["timestamp", "open", "high", "low", "close"]
        for kline in klines:
            for field in required_fields:
                if field not in kline:
                    logger.warning(f"Kline missing required field: {field}")
                    return False
        
        return True
    
    def _validate_pattern_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate pattern data.
        
        Args:
            data: Pattern data
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["pattern_type", "symbol"]
        for field in required_fields:
            if field not in data:
                logger.warning(f"Pattern data missing required field: {field}")
                return False
        
        return True
    
    def _validate_decision_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate decision data.
        
        Args:
            data: Decision data
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["decision_type"]
        for field in required_fields:
            if field not in data:
                logger.warning(f"Decision data missing required field: {field}")
                return False
        
        return True
    
    def _standardize_market_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize market data format.
        
        Args:
            data: Market data
            
        Returns:
            Standardized market data
        """
        # Create a copy to avoid modifying the original
        std_data = data.copy()
        
        # Ensure all required fields are present
        if "timestamp" not in std_data:
            std_data["timestamp"] = datetime.now().isoformat()
        
        # Convert numeric fields
        numeric_fields = ["price", "volume", "bid", "ask", "high", "low"]
        for field in numeric_fields:
            if field in std_data and not isinstance(std_data[field], (int, float)):
                try:
                    std_data[field] = float(std_data[field])
                except (ValueError, TypeError):
                    std_data[field] = 0.0
        
        return std_data
    
    def _standardize_klines_data(self, klines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Standardize klines data format.
        
        Args:
            klines: Klines data
            
        Returns:
            Standardized klines data
        """
        std_klines = []
        
        for kline in klines:
            # Create a copy to avoid modifying the original
            std_kline = kline.copy()
            
            # Ensure timestamp is in ISO format
            if "timestamp" in std_kline and not isinstance(std_kline["timestamp"], str):
                if isinstance(std_kline["timestamp"], (int, float)):
                    # Convert from milliseconds to ISO format
                    std_kline["timestamp"] = datetime.fromtimestamp(std_kline["timestamp"] / 1000).isoformat()
                else:
                    std_kline["timestamp"] = datetime.now().isoformat()
            
            # Convert numeric fields
            numeric_fields = ["open", "high", "low", "close", "volume"]
            for field in numeric_fields:
                if field in std_kline and not isinstance(std_kline[field], (int, float)):
                    try:
                        std_kline[field] = float(std_kline[field])
                    except (ValueError, TypeError):
                        std_kline[field] = 0.0
            
            std_klines.append(std_kline)
        
        return std_klines
    
    def _standardize_pattern_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize pattern data format.
        
        Args:
            data: Pattern data
            
        Returns:
            Standardized pattern data
        """
        # Create a copy to avoid modifying the original
        std_data = data.copy()
        
        # Ensure all required fields are present
        if "timestamp" not in std_data:
            std_data["timestamp"] = datetime.now().isoformat()
        
        if "confidence" not in std_data:
            std_data["confidence"] = 0.5
        
        # Convert numeric fields
        numeric_fields = ["confidence", "price_target", "stop_loss"]
        for field in numeric_fields:
            if field in std_data and not isinstance(std_data[field], (int, float)):
                try:
                    std_data[field] = float(std_data[field])
                except (ValueError, TypeError):
                    std_data[field] = 0.0
        
        return std_data
    
    def _standardize_decision_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize decision data format.
        
        Args:
            data: Decision data
            
        Returns:
            Standardized decision data
        """
        # Create a copy to avoid modifying the original
        std_data = data.copy()
        
        # Ensure all required fields are present
        if "timestamp" not in std_data:
            std_data["timestamp"] = datetime.now().isoformat()
        
        if "confidence" not in std_data:
            std_data["confidence"] = 0.5
        
        # Convert numeric fields
        numeric_fields = ["confidence", "price_target", "stop_loss"]
        for field in numeric_fields:
            if field in std_data and not isinstance(std_data[field], (int, float)):
                try:
                    std_data[field] = float(std_data[field])
                except (ValueError, TypeError):
                    std_data[field] = 0.0
        
        return std_data


# For testing
async def test():
    """Test function."""
    # Create unified data pipeline
    pipeline = UnifiedDataPipeline()
    
    # Test market data
    pipeline.update_market_data("BTC/USDC", {
        "symbol": "BTC/USDC",
        "price": 50000.0,
        "volume": 100.0,
        "bid": 49990.0,
        "ask": 50010.0
    })
    
    # Test klines data
    pipeline.update_klines_data("BTC/USDC", "1h", [
        {
            "timestamp": datetime.now().isoformat(),
            "open": 50000.0,
            "high": 51000.0,
            "low": 49000.0,
            "close": 50500.0,
            "volume": 100.0
        },
        {
            "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
            "open": 49500.0,
            "high": 50500.0,
            "low": 49000.0,
            "close": 50000.0,
            "volume": 90.0
        }
    ])
    
    # Test pattern data
    pipeline.update_pattern_data("pattern1", {
        "pattern_type": "double_bottom",
        "symbol": "BTC/USDC",
        "timeframe": "1h",
        "confidence": 0.8,
        "direction": "bullish"
    })
    
    # Test decision data
    pipeline.update_decision_data("decision1", {
        "decision_type": "entry",
        "symbol": "BTC/USDC",
        "direction": "bullish",
        "confidence": 0.85,
        "summary": "Enter long position based on bullish pattern"
    })
    
    # Test retrieval
    market_data = pipeline.get_market_data("BTC/USDC")
    print(f"Market data: {market_data}")
    
    klines_data = pipeline.get_klines_data("BTC/USDC", "1h")
    print(f"Klines data: {klines_data}")
    
    pattern_data = pipeline.get_pattern_data("pattern1")
    print(f"Pattern data: {pattern_data}")
    
    decision_data = pipeline.get_decision_data("decision1")
    print(f"Decision data: {decision_data}")
    
    # Test DataFrame conversion
    df = pipeline.get_klines_dataframe("BTC/USDC", "1h")
    print(f"DataFrame:\n{df}")


if __name__ == "__main__":
    asyncio.run(test())
