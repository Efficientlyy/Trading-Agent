#!/usr/bin/env python
"""
Pattern Recognition and Indicator Analysis Module for LLM Strategic Overseer.

This module provides pattern recognition and technical indicator analysis
for trading assets, feeding signals to the LLM Strategic Overseer.
"""

import os
import sys
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import traceback
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'pattern_recognition.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import event bus
from ..core.event_bus import EventBus

class PatternRecognition:
    """
    Pattern Recognition and Indicator Analysis Module.
    
    This class provides pattern recognition and technical indicator analysis
    for trading assets, feeding signals to the LLM Strategic Overseer.
    """
    
    def __init__(self, config, event_bus: Optional[EventBus] = None):
        """
        Initialize Pattern Recognition.
        
        Args:
            config: Configuration object
            event_bus: Event bus instance (optional, will create new if None)
        """
        self.config = config
        
        # Initialize event bus if not provided
        self.event_bus = event_bus if event_bus else EventBus()
        
        # Initialize subscriptions
        self.subscriptions = {}
        
        # Initialize data storage
        self.market_data = {}
        self.indicators = {}
        self.patterns = {}
        
        # Initialize pattern settings
        self.pattern_settings = {
            "enabled": self.config.get("analysis.patterns_enabled", True),
            "min_data_points": self.config.get("analysis.min_data_points", 30),
            "detection_interval": self.config.get("analysis.detection_interval", 5),  # Run detection every N data points
            "patterns": self.config.get("analysis.patterns", [
                "double_top", "double_bottom", "head_and_shoulders", 
                "inverse_head_and_shoulders", "triangle", "wedge"
            ])
        }
        
        # Initialize indicator settings
        self.indicator_settings = {
            "enabled": self.config.get("analysis.indicators_enabled", True),
            "calculation_interval": self.config.get("analysis.calculation_interval", 1),  # Run calculation every N data points
            "indicators": self.config.get("analysis.indicators", [
                "sma", "ema", "rsi", "macd", "bollinger"
            ]),
            "sma_window": self.config.get("analysis.sma_window", 20),
            "ema_window": self.config.get("analysis.ema_window", 20),
            "rsi_window": self.config.get("analysis.rsi_window", 14),
            "macd_fast": self.config.get("analysis.macd_fast", 12),
            "macd_slow": self.config.get("analysis.macd_slow", 26),
            "macd_signal": self.config.get("analysis.macd_signal", 9),
            "bollinger_window": self.config.get("analysis.bollinger_window", 20),
            "bollinger_std": self.config.get("analysis.bollinger_std", 2.0)
        }
        
        # Initialize supported symbols
        self.symbols = self.config.get("trading.symbols", ["BTC/USDC", "ETH/USDC", "SOL/USDC"])
        
        # Initialize data counters
        self.data_counters = {symbol: 0 for symbol in self.symbols}
        
        # Subscribe to relevant events
        self._subscribe_to_events()
        
        logger.info("Pattern Recognition initialized")
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        # Subscribe to market data events
        self.subscriptions["market_data"] = self.event_bus.subscribe(
            "trading.market_data", self._handle_market_data
        )
        
        logger.info("Subscribed to events")
    
    async def _handle_market_data(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Handle market data event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        try:
            if not data.get("success", False):
                return
            
            symbol = data.get("symbol")
            if not symbol or symbol not in self.symbols:
                return
            
            # Initialize symbol data if not exists
            if symbol not in self.market_data:
                self.market_data[symbol] = {
                    "timestamp": [],
                    "price": [],
                    "volume": []
                }
            
            # Add data point
            timestamp = datetime.fromisoformat(data.get("timestamp"))
            price = data.get("price")
            volume = data.get("volume_24h", 0)
            
            self.market_data[symbol]["timestamp"].append(timestamp)
            self.market_data[symbol]["price"].append(price)
            self.market_data[symbol]["volume"].append(volume)
            
            # Increment data counter
            self.data_counters[symbol] += 1
            
            # Calculate indicators if enabled and interval reached
            if (self.indicator_settings["enabled"] and 
                self.data_counters[symbol] % self.indicator_settings["calculation_interval"] == 0):
                await self.calculate_indicators(symbol)
            
            # Detect patterns if enabled, enough data points, and interval reached
            if (self.pattern_settings["enabled"] and 
                len(self.market_data[symbol]["price"]) >= self.pattern_settings["min_data_points"] and
                self.data_counters[symbol] % self.pattern_settings["detection_interval"] == 0):
                await self.detect_patterns(symbol)
            
            logger.debug(f"Updated market data for {symbol}")
        except Exception as e:
            logger.error(f"Error handling market data: {e}")
            logger.error(traceback.format_exc())
    
    async def calculate_indicators(self, symbol: str) -> None:
        """
        Calculate technical indicators for symbol.
        
        Args:
            symbol: Symbol to calculate indicators for
        """
        try:
            if symbol not in self.market_data or not self.market_data[symbol]["timestamp"]:
                logger.warning(f"No market data for {symbol}")
                return
            
            # Get price data
            prices = self.market_data[symbol]["price"]
            
            # Initialize symbol indicators if not exists
            if symbol not in self.indicators:
                self.indicators[symbol] = {}
            
            # Calculate SMA
            if "sma" in self.indicator_settings["indicators"]:
                window = self.indicator_settings["sma_window"]
                if len(prices) >= window:
                    sma = self._calculate_sma(prices, window)
                    self.indicators[symbol]["sma"] = sma
                    
                    # Publish indicator event
                    await self.event_bus.publish(
                        "analysis.indicator",
                        {
                            "symbol": symbol,
                            "indicator_type": "sma",
                            "values": sma,
                            "window": window,
                            "timestamp": datetime.now().isoformat()
                        },
                        "normal"
                    )
            
            # Calculate EMA
            if "ema" in self.indicator_settings["indicators"]:
                window = self.indicator_settings["ema_window"]
                if len(prices) >= window:
                    ema = self._calculate_ema(prices, window)
                    self.indicators[symbol]["ema"] = ema
                    
                    # Publish indicator event
                    await self.event_bus.publish(
                        "analysis.indicator",
                        {
                            "symbol": symbol,
                            "indicator_type": "ema",
                            "values": ema,
                            "window": window,
                            "timestamp": datetime.now().isoformat()
                        },
                        "normal"
                    )
            
            # Calculate Bollinger Bands
            if "bollinger" in self.indicator_settings["indicators"]:
                window = self.indicator_settings["bollinger_window"]
                num_std = self.indicator_settings["bollinger_std"]
                if len(prices) >= window:
                    upper, lower = self._calculate_bollinger_bands(prices, window, num_std)
                    self.indicators[symbol]["bollinger_upper"] = upper
                    self.indicators[symbol]["bollinger_lower"] = lower
                    
                    # Publish indicator event
                    await self.event_bus.publish(
                        "analysis.indicator",
                        {
                            "symbol": symbol,
                            "indicator_type": "bollinger_upper",
                            "values": upper,
                            "window": window,
                            "std": num_std,
                            "timestamp": datetime.now().isoformat()
                        },
                        "normal"
                    )
                    
                    await self.event_bus.publish(
                        "analysis.indicator",
                        {
                            "symbol": symbol,
                            "indicator_type": "bollinger_lower",
                            "values": lower,
                            "window": window,
                            "std": num_std,
                            "timestamp": datetime.now().isoformat()
                        },
                        "normal"
                    )
            
            # Calculate RSI
            if "rsi" in self.indicator_settings["indicators"]:
                window = self.indicator_settings["rsi_window"]
                if len(prices) >= window + 1:
                    rsi = self._calculate_rsi(prices, window)
                    self.indicators[symbol]["rsi"] = rsi
                    
                    # Publish indicator event
                    await self.event_bus.publish(
                        "analysis.indicator",
                        {
                            "symbol": symbol,
                            "indicator_type": "rsi",
                            "values": rsi,
                            "window": window,
                            "timestamp": datetime.now().isoformat()
                        },
                        "normal"
                    )
                    
                    # Check for RSI signals
                    await self._check_rsi_signals(symbol, rsi)
            
            # Calculate MACD
            if "macd" in self.indicator_settings["indicators"]:
                fast = self.indicator_settings["macd_fast"]
                slow = self.indicator_settings["macd_slow"]
                signal_window = self.indicator_settings["macd_signal"]
                
                if len(prices) >= slow + signal_window:
                    macd, signal, histogram = self._calculate_macd(prices, fast, slow, signal_window)
                    self.indicators[symbol]["macd"] = macd
                    self.indicators[symbol]["macd_signal"] = signal
                    self.indicators[symbol]["macd_histogram"] = histogram
                    
                    # Publish indicator events
                    await self.event_bus.publish(
                        "analysis.indicator",
                        {
                            "symbol": symbol,
                            "indicator_type": "macd",
                            "values": macd,
                            "fast": fast,
                            "slow": slow,
                            "timestamp": datetime.now().isoformat()
                        },
                        "normal"
                    )
                    
                    await self.event_bus.publish(
                        "analysis.indicator",
                        {
                            "symbol": symbol,
                            "indicator_type": "macd_signal",
                            "values": signal,
                            "window": signal_window,
                            "timestamp": datetime.now().isoformat()
                        },
                        "normal"
                    )
                    
                    await self.event_bus.publish(
                        "analysis.indicator",
                        {
                            "symbol": symbol,
                            "indicator_type": "macd_histogram",
                            "values": histogram,
                            "timestamp": datetime.now().isoformat()
                        },
                        "normal"
                    )
                    
                    # Check for MACD signals
                    await self._check_macd_signals(symbol, macd, signal, histogram)
            
            logger.debug(f"Calculated indicators for {symbol}")
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            logger.error(traceback.format_exc())
    
    async def detect_patterns(self, symbol: str) -> None:
        """
        Detect patterns for symbol.
        
        Args:
            symbol: Symbol to detect patterns for
        """
        try:
            if symbol not in self.market_data or not self.market_data[symbol]["timestamp"]:
                logger.warning(f"No market data for {symbol}")
                return
            
            # Get price data
            prices = self.market_data[symbol]["price"]
            timestamps = self.market_data[symbol]["timestamp"]
            
            # Initialize symbol patterns if not exists
            if symbol not in self.patterns:
                self.patterns[symbol] = []
            
            # Detect Double Top
            if "double_top" in self.pattern_settings["patterns"]:
                double_top = self._detect_double_top(prices, timestamps)
                if double_top:
                    # Add to patterns
                    self.patterns[symbol].append(double_top)
                    
                    # Publish pattern event
                    await self.event_bus.publish(
                        "analysis.pattern",
                        {
                            "symbol": symbol,
                            "pattern_type": "double_top",
                            "pattern_data": double_top["data"],
                            "timestamp": double_top["timestamp"].isoformat(),
                            "confidence": double_top["confidence"]
                        },
                        "high"
                    )
                    
                    logger.info(f"Detected Double Top pattern for {symbol}")
            
            # Detect Double Bottom
            if "double_bottom" in self.pattern_settings["patterns"]:
                double_bottom = self._detect_double_bottom(prices, timestamps)
                if double_bottom:
                    # Add to patterns
                    self.patterns[symbol].append(double_bottom)
                    
                    # Publish pattern event
                    await self.event_bus.publish(
                        "analysis.pattern",
                        {
                            "symbol": symbol,
                            "pattern_type": "double_bottom",
                            "pattern_data": double_bottom["data"],
                            "timestamp": double_bottom["timestamp"].isoformat(),
                            "confidence": double_bottom["confidence"]
                        },
                        "high"
                    )
                    
                    logger.info(f"Detected Double Bottom pattern for {symbol}")
            
            # Detect Head and Shoulders
            if "head_and_shoulders" in self.pattern_settings["patterns"]:
                head_and_shoulders = self._detect_head_and_shoulders(prices, timestamps)
                if head_and_shoulders:
                    # Add to patterns
                    self.patterns[symbol].append(head_and_shoulders)
                    
                    # Publish pattern event
                    await self.event_bus.publish(
                        "analysis.pattern",
                        {
                            "symbol": symbol,
                            "pattern_type": "head_and_shoulders",
                            "pattern_data": head_and_shoulders["data"],
                            "timestamp": head_and_shoulders["timestamp"].isoformat(),
                            "confidence": head_and_shoulders["confidence"]
                        },
                        "high"
                    )
                    
                    logger.info(f"Detected Head and Shoulders pattern for {symbol}")
            
            # Detect Inverse Head and Shoulders
            if "inverse_head_and_shoulders" in self.pattern_settings["patterns"]:
                inverse_head_and_shoulders = self._detect_inverse_head_and_shoulders(prices, timestamps)
                if inverse_head_and_shoulders:
                    # Add to patterns
                    self.patterns[symbol].append(inverse_head_and_shoulders)
                    
                    # Publish pattern event
                    await self.event_bus.publish(
                        "analysis.pattern",
                        {
                            "symbol": symbol,
                            "pattern_type": "inverse_head_and_shoulders",
                            "pattern_data": inverse_head_and_shoulders["data"],
                            "timestamp": inverse_head_and_shoulders["timestamp"].isoformat(),
                            "confidence": inverse_head_and_shoulders["confidence"]
                        },
                        "high"
                    )
                    
                    logger.info(f"Detected Inverse Head and Shoulders pattern for {symbol}")
            
            logger.debug(f"Detected patterns for {symbol}")
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            logger.error(traceback.format_exc())
    
    async def _check_rsi_signals(self, symbol: str, rsi: List[float]) -> None:
        """
        Check for RSI signals.
        
        Args:
            symbol: Symbol to check signals for
            rsi: RSI values
        """
        try:
            # Filter out None values
            valid_rsi = [x for x in rsi if x is not None]
            
            if not valid_rsi:
                return
            
            latest_rsi = valid_rsi[-1]
            
            # Check for overbought (RSI > 70)
            if latest_rsi > 70:
                await self.event_bus.publish(
                    "analysis.signal",
                    {
                        "symbol": symbol,
                        "signal_type": "rsi_overbought",
                        "value": latest_rsi,
                        "threshold": 70,
                        "timestamp": datetime.now().isoformat()
                    },
                    "high"
                )
                
                logger.info(f"RSI Overbought signal for {symbol}: {latest_rsi}")
            
            # Check for oversold (RSI < 30)
            elif latest_rsi < 30:
                await self.event_bus.publish(
                    "analysis.signal",
                    {
                        "symbol": symbol,
                        "signal_type": "rsi_oversold",
                        "value": latest_rsi,
                        "threshold": 30,
                        "timestamp": datetime.now().isoformat()
                    },
                    "high"
                )
                
                logger.info(f"RSI Oversold signal for {symbol}: {latest_rsi}")
        except Exception as e:
            logger.error(f"Error checking RSI signals: {e}")
    
    async def _check_macd_signals(self, symbol: str, macd: List[float], signal: List[float], histogram: List[float]) -> None:
        """
        Check for MACD signals.
        
        Args:
            symbol: Symbol to check signals for
            macd: MACD line values
            signal: Signal line values
            histogram: Histogram values
        """
        try:
            # Filter out None values
            valid_macd = [x for x in macd if x is not None]
            valid_signal = [x for x in signal if x is not None]
            valid_histogram = [x for x in histogram if x is not None]
            
            if not valid_macd or not valid_signal or not valid_histogram:
                return
            
            # Check for MACD crossover (MACD crosses above Signal)
            if len(valid_macd) >= 2 and len(valid_signal) >= 2:
                if valid_macd[-2] < valid_signal[-2] and valid_macd[-1] > valid_signal[-1]:
                    await self.event_bus.publish(
                        "analysis.signal",
                        {
                            "symbol": symbol,
                            "signal_type": "macd_bullish_crossover",
                            "macd_value": valid_macd[-1],
                            "signal_value": valid_signal[-1],
                            "timestamp": datetime.now().isoformat()
                        },
                        "high"
                    )
                    
                    logger.info(f"MACD Bullish Crossover signal for {symbol}")
                
                # Check for MACD crossunder (MACD crosses below Signal)
                elif valid_macd[-2] > valid_signal[-2] and valid_macd[-1] < valid_signal[-1]:
                    await self.event_bus.publish(
                        "analysis.signal",
                        {
                            "symbol": symbol,
                            "signal_type": "macd_bearish_crossover",
                            "macd_value": valid_macd[-1],
                            "signal_value": valid_signal[-1],
                            "timestamp": datetime.now().isoformat()
                        },
                        "high"
                    )
                    
                    logger.info(f"MACD Bearish Crossover signal for {symbol}")
        except Exception as e:
            logger.error(f"Error checking MACD signals: {e}")
    
    def _calculate_sma(self, prices: List[float], window: int) -> List[float]:
        """
        Calculate Simple Moving Average.
        
        Args:
            prices: List of prices
            window: Window size
            
        Returns:
            List of SMA values
        """
        sma = []
        for i in range(len(prices)):
            if i < window - 1:
                sma.append(None)
            else:
                sma.append(sum(prices[i - window + 1:i + 1]) / window)
        return sma
    
    def _calculate_ema(self, prices: List[float], window: int) -> List[float]:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: List of prices
            window: Window size
            
        Returns:
            List of EMA values
        """
        ema = []
        multiplier = 2 / (window + 1)
        
        # Start with SMA
        sma = sum(prices[:window]) / window
        ema.append(sma)
        
        # Calculate EMA
        for i in range(1, len(prices) - window + 1):
            ema_value = (prices[i + window - 1] - ema[-1]) * multiplier + ema[-1]
            ema.append(ema_value)
        
        # Pad with None for alignment
        return [None] * (window - 1) + ema
    
    def _calculate_bollinger_bands(self, prices: List[float], window: int, num_std: float = 2.0) -> tuple:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: List of prices
            window: Window size
            num_std: Number of standard deviations
            
        Returns:
            Tuple of (upper band, lower band)
        """
        upper_band = []
        lower_band = []
        
        for i in range(len(prices)):
            if i < window - 1:
                upper_band.append(None)
                lower_band.append(None)
            else:
                window_slice = prices[i - window + 1:i + 1]
                sma = sum(window_slice) / window
                std = np.std(window_slice)
                
                upper_band.append(sma + num_std * std)
                lower_band.append(sma - num_std * std)
        
        return upper_band, lower_band
    
    def _calculate_rsi(self, prices: List[float], window: int) -> List[float]:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: List of prices
            window: Window size
            
        Returns:
            List of RSI values
        """
        rsi = []
        gains = []
        losses = []
        
        # Calculate price changes
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change >= 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        # Pad with None for alignment
        rsi.append(None)
        
        # Calculate RSI
        for i in range(len(gains)):
            if i < window - 1:
                rsi.append(None)
            else:
                avg_gain = sum(gains[i - window + 1:i + 1]) / window
                avg_loss = sum(losses[i - window + 1:i + 1]) / window
                
                if avg_loss == 0:
                    rsi.append(100)
                else:
                    rs = avg_gain / avg_loss
                    rsi.append(100 - (100 / (1 + rs)))
        
        return rsi
    
    def _calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """
        Calculate MACD.
        
        Args:
            prices: List of prices
            fast: Fast EMA window
            slow: Slow EMA window
            signal: Signal EMA window
            
        Returns:
            Tuple of (MACD line, signal line, histogram)
        """
        # Calculate fast EMA
        ema_fast = []
        multiplier_fast = 2 / (fast + 1)
        
        # Start with SMA
        sma_fast = sum(prices[:fast]) / fast
        ema_fast.append(sma_fast)
        
        # Calculate fast EMA
        for i in range(1, len(prices) - fast + 1):
            ema_value = (prices[i + fast - 1] - ema_fast[-1]) * multiplier_fast + ema_fast[-1]
            ema_fast.append(ema_value)
        
        # Pad with None for alignment
        ema_fast = [None] * (fast - 1) + ema_fast
        
        # Calculate slow EMA
        ema_slow = []
        multiplier_slow = 2 / (slow + 1)
        
        # Start with SMA
        sma_slow = sum(prices[:slow]) / slow
        ema_slow.append(sma_slow)
        
        # Calculate slow EMA
        for i in range(1, len(prices) - slow + 1):
            ema_value = (prices[i + slow - 1] - ema_slow[-1]) * multiplier_slow + ema_slow[-1]
            ema_slow.append(ema_value)
        
        # Pad with None for alignment
        ema_slow = [None] * (slow - 1) + ema_slow
        
        # Calculate MACD line
        macd_line = []
        for i in range(len(prices)):
            if i < slow - 1:
                macd_line.append(None)
            else:
                macd_line.append(ema_fast[i] - ema_slow[i])
        
        # Calculate signal line
        signal_line = []
        macd_values = [x for x in macd_line if x is not None]
        
        # Start with SMA of MACD
        sma_signal = sum(macd_values[:signal]) / signal
        signal_line.append(sma_signal)
        
        # Calculate signal EMA
        multiplier_signal = 2 / (signal + 1)
        for i in range(1, len(macd_values) - signal + 1):
            signal_value = (macd_values[i + signal - 1] - signal_line[-1]) * multiplier_signal + signal_line[-1]
            signal_line.append(signal_value)
        
        # Pad with None for alignment
        signal_line = [None] * (slow + signal - 2) + signal_line
        
        # Calculate histogram
        histogram = []
        for i in range(len(prices)):
            if i < slow + signal - 2:
                histogram.append(None)
            else:
                histogram.append(macd_line[i] - signal_line[i])
        
        return macd_line, signal_line, histogram
    
    def _detect_double_top(self, prices: List[float], timestamps: List[datetime]) -> Optional[Dict[str, Any]]:
        """
        Detect Double Top pattern.
        
        Args:
            prices: List of prices
            timestamps: List of timestamps
            
        Returns:
            Pattern data if detected, None otherwise
        """
        try:
            # Need at least 20 data points
            if len(prices) < 20:
                return None
            
            # Look at the last 20 data points
            window = min(20, len(prices))
            price_window = prices[-window:]
            
            # Find peaks
            peaks = []
            for i in range(1, len(price_window) - 1):
                if price_window[i] > price_window[i - 1] and price_window[i] > price_window[i + 1]:
                    peaks.append((i, price_window[i]))
            
            # Need at least 2 peaks
            if len(peaks) < 2:
                return None
            
            # Check for double top
            for i in range(len(peaks) - 1):
                for j in range(i + 1, len(peaks)):
                    # Peaks should be similar in height (within 1%)
                    height_diff = abs(peaks[i][1] - peaks[j][1]) / peaks[i][1]
                    if height_diff < 0.01:
                        # Peaks should be separated by at least 3 data points
                        if abs(peaks[i][0] - peaks[j][0]) >= 3:
                            # Calculate confidence based on height difference and separation
                            confidence = 0.9 - height_diff * 10
                            
                            # Get actual indices in the full price list
                            idx1 = len(prices) - window + peaks[i][0]
                            idx2 = len(prices) - window + peaks[j][0]
                            
                            return {
                                "type": "double_top",
                                "data": {
                                    "peak1_index": idx1,
                                    "peak1_price": prices[idx1],
                                    "peak2_index": idx2,
                                    "peak2_price": prices[idx2]
                                },
                                "timestamp": timestamps[-1],
                                "confidence": confidence
                            }
            
            return None
        except Exception as e:
            logger.error(f"Error detecting double top: {e}")
            return None
    
    def _detect_double_bottom(self, prices: List[float], timestamps: List[datetime]) -> Optional[Dict[str, Any]]:
        """
        Detect Double Bottom pattern.
        
        Args:
            prices: List of prices
            timestamps: List of timestamps
            
        Returns:
            Pattern data if detected, None otherwise
        """
        try:
            # Need at least 20 data points
            if len(prices) < 20:
                return None
            
            # Look at the last 20 data points
            window = min(20, len(prices))
            price_window = prices[-window:]
            
            # Find troughs
            troughs = []
            for i in range(1, len(price_window) - 1):
                if price_window[i] < price_window[i - 1] and price_window[i] < price_window[i + 1]:
                    troughs.append((i, price_window[i]))
            
            # Need at least 2 troughs
            if len(troughs) < 2:
                return None
            
            # Check for double bottom
            for i in range(len(troughs) - 1):
                for j in range(i + 1, len(troughs)):
                    # Troughs should be similar in height (within 1%)
                    height_diff = abs(troughs[i][1] - troughs[j][1]) / troughs[i][1]
                    if height_diff < 0.01:
                        # Troughs should be separated by at least 3 data points
                        if abs(troughs[i][0] - troughs[j][0]) >= 3:
                            # Calculate confidence based on height difference and separation
                            confidence = 0.9 - height_diff * 10
                            
                            # Get actual indices in the full price list
                            idx1 = len(prices) - window + troughs[i][0]
                            idx2 = len(prices) - window + troughs[j][0]
                            
                            return {
                                "type": "double_bottom",
                                "data": {
                                    "trough1_index": idx1,
                                    "trough1_price": prices[idx1],
                                    "trough2_index": idx2,
                                    "trough2_price": prices[idx2]
                                },
                                "timestamp": timestamps[-1],
                                "confidence": confidence
                            }
            
            return None
        except Exception as e:
            logger.error(f"Error detecting double bottom: {e}")
            return None
    
    def _detect_head_and_shoulders(self, prices: List[float], timestamps: List[datetime]) -> Optional[Dict[str, Any]]:
        """
        Detect Head and Shoulders pattern.
        
        Args:
            prices: List of prices
            timestamps: List of timestamps
            
        Returns:
            Pattern data if detected, None otherwise
        """
        try:
            # Need at least 30 data points
            if len(prices) < 30:
                return None
            
            # Look at the last 30 data points
            window = min(30, len(prices))
            price_window = prices[-window:]
            
            # Find peaks
            peaks = []
            for i in range(1, len(price_window) - 1):
                if price_window[i] > price_window[i - 1] and price_window[i] > price_window[i + 1]:
                    peaks.append((i, price_window[i]))
            
            # Need at least 3 peaks
            if len(peaks) < 3:
                return None
            
            # Check for head and shoulders
            for i in range(len(peaks) - 2):
                # Left shoulder, head, right shoulder
                left = peaks[i]
                head = peaks[i + 1]
                right = peaks[i + 2]
                
                # Head should be higher than shoulders
                if head[1] > left[1] and head[1] > right[1]:
                    # Shoulders should be similar in height (within 10%)
                    shoulder_diff = abs(left[1] - right[1]) / left[1]
                    if shoulder_diff < 0.1:
                        # Calculate confidence based on pattern quality
                        head_height = head[1] - min(left[1], right[1])
                        pattern_quality = head_height / head[1]
                        confidence = 0.7 + pattern_quality * 0.2 - shoulder_diff
                        
                        # Get actual indices in the full price list
                        left_idx = len(prices) - window + left[0]
                        head_idx = len(prices) - window + head[0]
                        right_idx = len(prices) - window + right[0]
                        
                        return {
                            "type": "head_and_shoulders",
                            "data": {
                                "left_shoulder_index": left_idx,
                                "left_shoulder_price": prices[left_idx],
                                "head_index": head_idx,
                                "head_price": prices[head_idx],
                                "right_shoulder_index": right_idx,
                                "right_shoulder_price": prices[right_idx]
                            },
                            "timestamp": timestamps[-1],
                            "confidence": confidence
                        }
            
            return None
        except Exception as e:
            logger.error(f"Error detecting head and shoulders: {e}")
            return None
    
    def _detect_inverse_head_and_shoulders(self, prices: List[float], timestamps: List[datetime]) -> Optional[Dict[str, Any]]:
        """
        Detect Inverse Head and Shoulders pattern.
        
        Args:
            prices: List of prices
            timestamps: List of timestamps
            
        Returns:
            Pattern data if detected, None otherwise
        """
        try:
            # Need at least 30 data points
            if len(prices) < 30:
                return None
            
            # Look at the last 30 data points
            window = min(30, len(prices))
            price_window = prices[-window:]
            
            # Find troughs
            troughs = []
            for i in range(1, len(price_window) - 1):
                if price_window[i] < price_window[i - 1] and price_window[i] < price_window[i + 1]:
                    troughs.append((i, price_window[i]))
            
            # Need at least 3 troughs
            if len(troughs) < 3:
                return None
            
            # Check for inverse head and shoulders
            for i in range(len(troughs) - 2):
                # Left shoulder, head, right shoulder
                left = troughs[i]
                head = troughs[i + 1]
                right = troughs[i + 2]
                
                # Head should be lower than shoulders
                if head[1] < left[1] and head[1] < right[1]:
                    # Shoulders should be similar in height (within 10%)
                    shoulder_diff = abs(left[1] - right[1]) / left[1]
                    if shoulder_diff < 0.1:
                        # Calculate confidence based on pattern quality
                        head_depth = min(left[1], right[1]) - head[1]
                        pattern_quality = head_depth / head[1]
                        confidence = 0.7 + pattern_quality * 0.2 - shoulder_diff
                        
                        # Get actual indices in the full price list
                        left_idx = len(prices) - window + left[0]
                        head_idx = len(prices) - window + head[0]
                        right_idx = len(prices) - window + right[0]
                        
                        return {
                            "type": "inverse_head_and_shoulders",
                            "data": {
                                "left_shoulder_index": left_idx,
                                "left_shoulder_price": prices[left_idx],
                                "head_index": head_idx,
                                "head_price": prices[head_idx],
                                "right_shoulder_index": right_idx,
                                "right_shoulder_price": prices[right_idx]
                            },
                            "timestamp": timestamps[-1],
                            "confidence": confidence
                        }
            
            return None
        except Exception as e:
            logger.error(f"Error detecting inverse head and shoulders: {e}")
            return None


# For testing
async def test():
    """Test function."""
    from ..config.config import Config
    
    # Create configuration
    config = Config()
    
    # Create event bus
    event_bus = EventBus()
    
    # Create pattern recognition
    pattern_recognition = PatternRecognition(config, event_bus)
    
    # Generate mock market data
    symbol = "BTC/USDC"
    timestamps = []
    prices = []
    volumes = []
    
    # Generate 100 data points
    base_price = 50000
    for i in range(100):
        timestamp = datetime.now().replace(minute=0, second=0, microsecond=0) - pd.Timedelta(hours=100 - i)
        price = base_price + (i * 100) + (np.sin(i / 10) * 1000)
        volume = 100 + (np.random.random() * 50)
        
        timestamps.append(timestamp)
        prices.append(price)
        volumes.append(volume)
    
    # Add to market data
    pattern_recognition.market_data[symbol] = {
        "timestamp": timestamps,
        "price": prices,
        "volume": volumes
    }
    
    # Calculate indicators
    await pattern_recognition.calculate_indicators(symbol)
    
    # Detect patterns
    await pattern_recognition.detect_patterns(symbol)
    
    # Print indicators
    print(f"Indicators for {symbol}:")
    for indicator_type, values in pattern_recognition.indicators[symbol].items():
        print(f"  {indicator_type}: {values[-5:]} (last 5 values)")
    
    # Print patterns
    print(f"Patterns for {symbol}:")
    for pattern in pattern_recognition.patterns[symbol]:
        print(f"  {pattern['type']}: {pattern['data']}")


if __name__ == "__main__":
    asyncio.run(test())
