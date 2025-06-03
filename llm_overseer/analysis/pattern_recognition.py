#!/usr/bin/env python
"""
Pattern Recognition and Indicator Analysis Module

This module analyzes chart data to identify technical patterns and indicator signals,
publishing findings to the event bus for strategic decision-making and visualization.
"""

import os
import sys
import json
import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logging
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pattern_recognition.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pattern_recognition")

class PatternRecognition:
    """
    Analyzes chart data for technical patterns and indicator signals.
    
    Subscribes to kline updates, performs analysis, and publishes findings
    to the event bus.
    """
    
    def __init__(self, event_bus=None, data_pipeline=None):
        """
        Initialize Pattern Recognition module.
        
        Args:
            event_bus: Event Bus instance
            data_pipeline: Unified Data Pipeline instance
        """
        self.event_bus = event_bus
        self.data_pipeline = data_pipeline
        
        # Supported assets and timeframes (can be configured)
        self.supported_assets = ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
        self.supported_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        # Analysis parameters
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.sma_short_period = 20
        self.sma_long_period = 50
        
        # Register event handlers if event bus is provided
        if self.event_bus:
            self._register_event_handlers()
        
        logger.info("Pattern Recognition module initialized")
    
    def set_event_bus(self, event_bus):
        """
        Set Event Bus instance.
        
        Args:
            event_bus: Event Bus instance
        """
        self.event_bus = event_bus
        self._register_event_handlers()
        logger.info("Event Bus set for Pattern Recognition")
    
    def set_data_pipeline(self, data_pipeline):
        """
        Set Unified Data Pipeline instance.
        
        Args:
            data_pipeline: Unified Data Pipeline instance
        """
        self.data_pipeline = data_pipeline
        logger.info("Unified Data Pipeline set for Pattern Recognition")
    
    def _register_event_handlers(self):
        """
        Register event handlers with Event Bus.
        
        Subscribes to kline updates to trigger analysis.
        """
        if self.event_bus:
            self.event_bus.subscribe("pipeline.klines_updated", self._handle_klines_update)
            logger.info("Pattern Recognition event handlers registered")
        else:
            logger.warning("Event Bus not set, cannot register handlers")
    
    async def _handle_klines_update(self, topic: str, data: Dict[str, Any]):
        """
        Handle klines update event from the pipeline.
        
        Triggers analysis for the updated symbol and timeframe.
        
        Args:
            topic: Event topic
            data: Event data containing symbol, timeframe, and klines
        """
        logger.debug(f"Handling klines update for topic: {topic}")
        symbol = data.get("symbol")
        timeframe = data.get("timeframe")
        klines_data = data.get("klines")
        
        if not symbol or not timeframe or not klines_data:
            logger.warning("Klines update missing required fields for analysis")
            return
        
        # Check if symbol and timeframe are supported
        if symbol not in self.supported_assets or timeframe not in self.supported_timeframes:
            logger.debug(f"Unsupported symbol or timeframe for analysis: {symbol} {timeframe}")
            return
        
        # Convert klines to DataFrame
        try:
            logger.debug(f"Converting klines data to DataFrame for {symbol} {timeframe}")
            df = pd.DataFrame(klines_data)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            else:
                logger.warning(f"Timestamp column missing in klines data for {symbol} {timeframe}")
                return
                
            # Convert relevant columns to numeric
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
                else:
                     logger.warning(f"Column {col} missing in klines data for {symbol} {timeframe}")
                     # Allow analysis to proceed if only some columns are missing, 
                     # but log the warning.
            
            if df.empty or "close" not in df.columns:
                 logger.warning(f"Kline data is empty or missing 'close' column for {symbol} {timeframe}")
                 return

            logger.debug(f"Successfully converted klines data to DataFrame with shape {df.shape}")
            logger.debug(f"DataFrame columns: {df.columns.tolist()}")
            logger.debug(f"DataFrame head: {df.head(2)}")
            logger.debug(f"DataFrame tail: {df.tail(2)}")

        except Exception as e:
            logger.error(f"Error processing klines data for {symbol} {timeframe}: {e}")
            return
            
        # Perform analysis
        await self.analyze_data(symbol, timeframe, df)
    
    async def analyze_data(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """
        Perform pattern recognition and indicator analysis on the DataFrame.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe of the data
            df: Pandas DataFrame containing kline data with timestamp index
        """
        if df.empty:
            logger.warning(f"Cannot analyze empty DataFrame for {symbol} {timeframe}")
            return
            
        logger.info(f"Analyzing data for {symbol} {timeframe} ({len(df)} rows)")
        
        # --- Indicator Analysis --- 
        # Force signal generation for testing
        await self._generate_test_signals(symbol, timeframe, df)
        
        # Regular analysis
        await self._analyze_rsi(symbol, timeframe, df)
        await self._analyze_sma_crossover(symbol, timeframe, df)
        await self._analyze_macd(symbol, timeframe, df)
        
        # --- Pattern Recognition --- 
        await self._detect_double_bottom(symbol, timeframe, df)
        await self._detect_double_top(symbol, timeframe, df)
        await self._detect_head_and_shoulders(symbol, timeframe, df)
        
        logger.info(f"Analysis complete for {symbol} {timeframe}")

    async def _generate_test_signals(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Generate test signals for testing purposes."""
        logger.debug("Generating test signals for testing purposes")
        
        current_price = df["close"].iloc[-1]
        timestamp = df.index[-1].isoformat()
        
        # Generate a test RSI signal
        signal_data = {
            "indicator": "RSI_TEST",
            "value": 25.0,  # Oversold value
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": "oversold",
            "direction": "bullish_potential",
            "confidence": 0.8,
            "timestamp": timestamp,
            "price": float(current_price)
        }
        
        logger.debug(f"Publishing test signal: {signal_data}")
        await self._publish_signal("indicator.signal", signal_data)
        
        # Generate a test pattern
        pattern_data = {
            "pattern_type": "double_bottom_test",
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": "bullish",
            "confidence": 0.85,
            "price": float(current_price),
            "timestamp": timestamp,
            "details": {
                "first_bottom": float(current_price - 500),
                "second_bottom": float(current_price - 480),
                "middle_peak": float(current_price - 200)
            }
        }
        
        logger.debug(f"Publishing test pattern: {pattern_data}")
        await self._publish_pattern("visualization.pattern_detected", pattern_data)

    async def _analyze_rsi(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Analyze RSI indicator."""
        logger.debug(f"Analyzing RSI for {symbol} {timeframe}")
        if len(df) < 15: # Need enough data for RSI(14)
            logger.debug(f"Not enough data for RSI analysis: {len(df)} rows")
            return
            
        try:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            
            rs = avg_gain / avg_loss.replace(0, np.nan) # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(50) # Fill NaNs with neutral value
            
            last_rsi = rsi.iloc[-1]
            prev_rsi = rsi.iloc[-2] if len(rsi) > 1 else None
            
            logger.debug(f"RSI calculation complete. Last RSI: {last_rsi}, Previous RSI: {prev_rsi}")
            
            current_price = df["close"].iloc[-1]
            timestamp = df.index[-1].isoformat()

            signal_data = {
                "indicator": "RSI",
                "value": float(last_rsi),
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": timestamp,
                "price": float(current_price)
            }

            # Oversold condition
            if last_rsi < self.rsi_oversold:
                signal_data["signal"] = "oversold"
                signal_data["direction"] = "bullish_potential"
                signal_data["confidence"] = float(1 - (last_rsi / self.rsi_oversold))
                logger.debug(f"RSI oversold condition detected: {last_rsi} < {self.rsi_oversold}")
                await self._publish_signal("indicator.signal", signal_data)
            
            # Overbought condition
            elif last_rsi > self.rsi_overbought:
                signal_data["signal"] = "overbought"
                signal_data["direction"] = "bearish_potential"
                signal_data["confidence"] = float((last_rsi - self.rsi_overbought) / (100 - self.rsi_overbought))
                logger.debug(f"RSI overbought condition detected: {last_rsi} > {self.rsi_overbought}")
                await self._publish_signal("indicator.signal", signal_data)
                
            # Crossover signals (optional)
            if prev_rsi is not None:
                if prev_rsi <= self.rsi_oversold and last_rsi > self.rsi_oversold:
                    signal_data["signal"] = "oversold_exit"
                    signal_data["direction"] = "bullish"
                    signal_data["confidence"] = 0.7 # Example confidence
                    logger.debug(f"RSI oversold exit detected: {prev_rsi} <= {self.rsi_oversold} and {last_rsi} > {self.rsi_oversold}")
                    await self._publish_signal("indicator.signal", signal_data)
                elif prev_rsi >= self.rsi_overbought and last_rsi < self.rsi_overbought:
                    signal_data["signal"] = "overbought_exit"
                    signal_data["direction"] = "bearish"
                    signal_data["confidence"] = 0.7 # Example confidence
                    logger.debug(f"RSI overbought exit detected: {prev_rsi} >= {self.rsi_overbought} and {last_rsi} < {self.rsi_overbought}")
                    await self._publish_signal("indicator.signal", signal_data)

        except Exception as e:
            logger.error(f"Error analyzing RSI for {symbol} {timeframe}: {e}")

    async def _analyze_sma_crossover(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Analyze Simple Moving Average (SMA) crossovers."""
        logger.debug(f"Analyzing SMA crossover for {symbol} {timeframe}")
        if len(df) < self.sma_long_period:
            logger.debug(f"Not enough data for SMA analysis: {len(df)} rows < {self.sma_long_period}")
            return
            
        try:
            sma_short = df["close"].rolling(window=self.sma_short_period).mean()
            sma_long = df["close"].rolling(window=self.sma_long_period).mean()
            
            # Check for crossover at the last two points
            if len(sma_short) < 2 or len(sma_long) < 2:
                logger.debug("Not enough SMA data points for crossover analysis")
                return
                
            last_short = sma_short.iloc[-1]
            last_long = sma_long.iloc[-1]
            prev_short = sma_short.iloc[-2]
            prev_long = sma_long.iloc[-2]
            
            logger.debug(f"SMA calculation complete. Last short: {last_short}, Last long: {last_long}, Prev short: {prev_short}, Prev long: {prev_long}")
            
            current_price = df["close"].iloc[-1]
            timestamp = df.index[-1].isoformat()

            signal_data = {
                "indicator": "SMA_Crossover",
                "short_period": self.sma_short_period,
                "long_period": self.sma_long_period,
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": timestamp,
                "price": float(current_price)
            }

            # Golden Cross (Short SMA crosses above Long SMA)
            if prev_short <= prev_long and last_short > last_long:
                signal_data["signal"] = "golden_cross"
                signal_data["direction"] = "bullish"
                signal_data["confidence"] = 0.75 # Example confidence
                logger.debug(f"Golden cross detected: {prev_short} <= {prev_long} and {last_short} > {last_long}")
                await self._publish_signal("indicator.signal", signal_data)
                
            # Death Cross (Short SMA crosses below Long SMA)
            elif prev_short >= prev_long and last_short < last_long:
                signal_data["signal"] = "death_cross"
                signal_data["direction"] = "bearish"
                signal_data["confidence"] = 0.75 # Example confidence
                logger.debug(f"Death cross detected: {prev_short} >= {prev_long} and {last_short} < {last_long}")
                await self._publish_signal("indicator.signal", signal_data)

        except Exception as e:
            logger.error(f"Error analyzing SMA crossover for {symbol} {timeframe}: {e}")
            
    async def _analyze_macd(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Analyze MACD indicator."""
        logger.debug(f"Analyzing MACD for {symbol} {timeframe}")
        if len(df) < 26: # Need enough data for MACD (12, 26, 9)
            logger.debug(f"Not enough data for MACD analysis: {len(df)} rows < 26")
            return

        try:
            ema_12 = df["close"].ewm(span=12, adjust=False).mean()
            ema_26 = df["close"].ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            signal_line = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal_line

            # Check for crossover at the last two points
            if len(macd) < 2 or len(signal_line) < 2:
                logger.debug("Not enough MACD data points for crossover analysis")
                return

            last_macd = macd.iloc[-1]
            last_signal = signal_line.iloc[-1]
            prev_macd = macd.iloc[-2]
            prev_signal = signal_line.iloc[-2]
            
            logger.debug(f"MACD calculation complete. Last MACD: {last_macd}, Last signal: {last_signal}, Prev MACD: {prev_macd}, Prev signal: {prev_signal}")
            
            current_price = df["close"].iloc[-1]
            timestamp = df.index[-1].isoformat()

            signal_data = {
                "indicator": "MACD",
                "macd_value": float(last_macd),
                "signal_value": float(last_signal),
                "histogram_value": float(histogram.iloc[-1]),
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": timestamp,
                "price": float(current_price)
            }

            # Bullish Crossover (MACD crosses above Signal Line)
            if prev_macd <= prev_signal and last_macd > last_signal:
                signal_data["signal"] = "bullish_crossover"
                signal_data["direction"] = "bullish"
                signal_data["confidence"] = 0.7 # Example confidence
                logger.debug(f"MACD bullish crossover detected: {prev_macd} <= {prev_signal} and {last_macd} > {last_signal}")
                await self._publish_signal("indicator.signal", signal_data)

            # Bearish Crossover (MACD crosses below Signal Line)
            elif prev_macd >= prev_signal and last_macd < last_signal:
                signal_data["signal"] = "bearish_crossover"
                signal_data["direction"] = "bearish"
                signal_data["confidence"] = 0.7 # Example confidence
                logger.debug(f"MACD bearish crossover detected: {prev_macd} >= {prev_signal} and {last_macd} < {last_signal}")
                await self._publish_signal("indicator.signal", signal_data)

        except Exception as e:
            logger.error(f"Error analyzing MACD for {symbol} {timeframe}: {e}")

    async def _detect_double_bottom(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Detect Double Bottom pattern."""
        logger.debug(f"Detecting double bottom for {symbol} {timeframe}")
        if len(df) < 30:  # Need enough data for pattern detection
            logger.debug(f"Not enough data for double bottom detection: {len(df)} rows < 30")
            return
            
        try:
            # Simple double bottom detection (can be improved with more sophisticated algorithms)
            # Look for two lows with similar prices separated by a higher high
            window = min(30, len(df))
            recent_data = df.iloc[-window:]
            
            # Find local minima
            is_min = (recent_data['low'] < recent_data['low'].shift(1)) & (recent_data['low'] < recent_data['low'].shift(-1))
            minima_idx = is_min[is_min].index
            
            logger.debug(f"Found {len(minima_idx)} local minima in recent data")
            
            if len(minima_idx) >= 2:
                # Get the two most recent minima
                last_min_idx = minima_idx[-1]
                prev_min_idx = minima_idx[-2]
                
                last_min_price = recent_data.loc[last_min_idx, 'low']
                prev_min_price = recent_data.loc[prev_min_idx, 'low']
                
                logger.debug(f"Last minimum: {last_min_price} at {last_min_idx}, Previous minimum: {prev_min_price} at {prev_min_idx}")
                
                # Check if the two minima have similar prices (within 2%)
                price_diff_pct = abs(last_min_price - prev_min_price) / prev_min_price
                
                if price_diff_pct < 0.02:
                    # Check if there's a higher high between the two minima
                    between_data = df.loc[prev_min_idx:last_min_idx]
                    max_between = between_data['high'].max()
                    
                    logger.debug(f"Price difference: {price_diff_pct:.2%}, Max between: {max_between}")
                    
                    if max_between > last_min_price * 1.03 and max_between > prev_min_price * 1.03:
                        # Double bottom detected
                        current_price = df["close"].iloc[-1]
                        timestamp = df.index[-1].isoformat()
                        
                        # Calculate confidence based on pattern clarity
                        confidence = 0.7 * (1 - price_diff_pct * 10)  # Higher confidence for more similar lows
                        
                        pattern_data = {
                            "pattern_type": "double_bottom",
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "direction": "bullish",
                            "confidence": float(min(0.95, max(0.5, confidence))),  # Limit confidence between 0.5 and 0.95
                            "price": float(current_price),
                            "timestamp": timestamp,
                            "details": {
                                "first_bottom": float(prev_min_price),
                                "second_bottom": float(last_min_price),
                                "middle_peak": float(max_between)
                            }
                        }
                        
                        logger.debug(f"Double bottom pattern detected with confidence {confidence:.2f}")
                        await self._publish_pattern("visualization.pattern_detected", pattern_data)
        
        except Exception as e:
            logger.error(f"Error detecting double bottom for {symbol} {timeframe}: {e}")

    async def _detect_double_top(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Detect Double Top pattern."""
        logger.debug(f"Detecting double top for {symbol} {timeframe}")
        if len(df) < 30:  # Need enough data for pattern detection
            logger.debug(f"Not enough data for double top detection: {len(df)} rows < 30")
            return
            
        try:
            # Simple double top detection (can be improved with more sophisticated algorithms)
            # Look for two highs with similar prices separated by a lower low
            window = min(30, len(df))
            recent_data = df.iloc[-window:]
            
            # Find local maxima
            is_max = (recent_data['high'] > recent_data['high'].shift(1)) & (recent_data['high'] > recent_data['high'].shift(-1))
            maxima_idx = is_max[is_max].index
            
            logger.debug(f"Found {len(maxima_idx)} local maxima in recent data")
            
            if len(maxima_idx) >= 2:
                # Get the two most recent maxima
                last_max_idx = maxima_idx[-1]
                prev_max_idx = maxima_idx[-2]
                
                last_max_price = recent_data.loc[last_max_idx, 'high']
                prev_max_price = recent_data.loc[prev_max_idx, 'high']
                
                logger.debug(f"Last maximum: {last_max_price} at {last_max_idx}, Previous maximum: {prev_max_price} at {prev_max_idx}")
                
                # Check if the two maxima have similar prices (within 2%)
                price_diff_pct = abs(last_max_price - prev_max_price) / prev_max_price
                
                if price_diff_pct < 0.02:
                    # Check if there's a lower low between the two maxima
                    between_data = df.loc[prev_max_idx:last_max_idx]
                    min_between = between_data['low'].min()
                    
                    logger.debug(f"Price difference: {price_diff_pct:.2%}, Min between: {min_between}")
                    
                    if min_between < last_max_price * 0.97 and min_between < prev_max_price * 0.97:
                        # Double top detected
                        current_price = df["close"].iloc[-1]
                        timestamp = df.index[-1].isoformat()
                        
                        # Calculate confidence based on pattern clarity
                        confidence = 0.7 * (1 - price_diff_pct * 10)  # Higher confidence for more similar highs
                        
                        pattern_data = {
                            "pattern_type": "double_top",
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "direction": "bearish",
                            "confidence": float(min(0.95, max(0.5, confidence))),  # Limit confidence between 0.5 and 0.95
                            "price": float(current_price),
                            "timestamp": timestamp,
                            "details": {
                                "first_top": float(prev_max_price),
                                "second_top": float(last_max_price),
                                "middle_trough": float(min_between)
                            }
                        }
                        
                        logger.debug(f"Double top pattern detected with confidence {confidence:.2f}")
                        await self._publish_pattern("visualization.pattern_detected", pattern_data)
        
        except Exception as e:
            logger.error(f"Error detecting double top for {symbol} {timeframe}: {e}")

    async def _detect_head_and_shoulders(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Detect Head and Shoulders pattern."""
        logger.debug(f"Detecting head and shoulders for {symbol} {timeframe}")
        if len(df) < 40:  # Need enough data for pattern detection
            logger.debug(f"Not enough data for head and shoulders detection: {len(df)} rows < 40")
            return
            
        try:
            # Simple head and shoulders detection
            # Look for three peaks with the middle one higher than the other two
            window = min(40, len(df))
            recent_data = df.iloc[-window:]
            
            # Find local maxima
            is_max = (recent_data['high'] > recent_data['high'].shift(1)) & (recent_data['high'] > recent_data['high'].shift(-1))
            maxima_idx = is_max[is_max].index
            
            logger.debug(f"Found {len(maxima_idx)} local maxima in recent data")
            
            if len(maxima_idx) >= 3:
                # Get the three most recent maxima
                right_shoulder_idx = maxima_idx[-1]
                head_idx = maxima_idx[-2]
                left_shoulder_idx = maxima_idx[-3]
                
                right_shoulder_price = recent_data.loc[right_shoulder_idx, 'high']
                head_price = recent_data.loc[head_idx, 'high']
                left_shoulder_price = recent_data.loc[left_shoulder_idx, 'high']
                
                logger.debug(f"Right shoulder: {right_shoulder_price} at {right_shoulder_idx}, Head: {head_price} at {head_idx}, Left shoulder: {left_shoulder_price} at {left_shoulder_idx}")
                
                # Check if the head is higher than both shoulders
                if head_price > right_shoulder_price * 1.02 and head_price > left_shoulder_price * 1.02:
                    # Check if shoulders are at similar levels (within 3%)
                    shoulder_diff_pct = abs(right_shoulder_price - left_shoulder_price) / left_shoulder_price
                    
                    logger.debug(f"Shoulder difference: {shoulder_diff_pct:.2%}")
                    
                    if shoulder_diff_pct < 0.03:
                        # Find the neckline (connecting the lows between the peaks)
                        left_trough_idx = recent_data.loc[left_shoulder_idx:head_idx]['low'].idxmin()
                        right_trough_idx = recent_data.loc[head_idx:right_shoulder_idx]['low'].idxmin()
                        
                        left_trough_price = recent_data.loc[left_trough_idx, 'low']
                        right_trough_price = recent_data.loc[right_trough_idx, 'low']
                        
                        logger.debug(f"Left trough: {left_trough_price} at {left_trough_idx}, Right trough: {right_trough_price} at {right_trough_idx}")
                        
                        # Check if the pattern is complete (price breaks below neckline)
                        current_price = df["close"].iloc[-1]
                        timestamp = df.index[-1].isoformat()
                        
                        # Calculate neckline at current position
                        days_since_left_trough = (df.index[-1] - left_trough_idx).days
                        total_days = (right_trough_idx - left_trough_idx).days
                        if total_days > 0:  # Avoid division by zero
                            slope = (right_trough_price - left_trough_price) / total_days
                            neckline_price = left_trough_price + slope * days_since_left_trough
                            
                            logger.debug(f"Neckline price at current position: {neckline_price}")
                            
                            # Calculate confidence based on pattern clarity
                            confidence = 0.7 * (1 - shoulder_diff_pct * 10)  # Higher confidence for more similar shoulders
                            
                            # Determine if pattern is complete (price below neckline)
                            pattern_complete = current_price < neckline_price
                            
                            pattern_data = {
                                "pattern_type": "head_and_shoulders",
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "direction": "bearish",
                                "confidence": float(min(0.95, max(0.5, confidence))),  # Limit confidence between 0.5 and 0.95
                                "price": float(current_price),
                                "timestamp": timestamp,
                                "details": {
                                    "left_shoulder": float(left_shoulder_price),
                                    "head": float(head_price),
                                    "right_shoulder": float(right_shoulder_price),
                                    "neckline": float(neckline_price),
                                    "pattern_complete": pattern_complete
                                }
                            }
                            
                            logger.debug(f"Head and shoulders pattern detected with confidence {confidence:.2f}")
                            await self._publish_pattern("visualization.pattern_detected", pattern_data)
        
        except Exception as e:
            logger.error(f"Error detecting head and shoulders for {symbol} {timeframe}: {e}")

    async def _publish_signal(self, topic: str, data: Dict[str, Any]):
        """
        Publish detected signal to the event bus.
        
        Args:
            topic: Event topic (e.g., "indicator.signal")
            data: Data payload for the event
        """
        if self.event_bus:
            try:
                logger.debug(f"Publishing signal to topic {topic}: {data}")
                await self.event_bus.publish(topic, data, priority="normal")
                logger.info(f"Published signal to topic {topic}: {data.get('signal', 'N/A')} for {data.get('symbol')} {data.get('timeframe')}")
            except Exception as e:
                logger.error(f"Error publishing signal to event bus: {e}")
        else:
            logger.warning("Event bus not available, cannot publish signal")

    async def _publish_pattern(self, topic: str, data: Dict[str, Any]):
        """
        Publish detected pattern to the event bus.
        
        Args:
            topic: Event topic (e.g., "visualization.pattern_detected")
            data: Data payload for the event
        """
        if self.event_bus:
            try:
                logger.debug(f"Publishing pattern to topic {topic}: {data}")
                await self.event_bus.publish(topic, data, priority="normal")
                logger.info(f"Published pattern to topic {topic}: {data.get('pattern_type', 'N/A')} for {data.get('symbol')} {data.get('timeframe')}")
            except Exception as e:
                logger.error(f"Error publishing pattern to event bus: {e}")
        else:
            logger.warning("Event bus not available, cannot publish pattern")


# For testing
async def test():
    """Test function for PatternRecognition."""
    from datetime import timedelta
    from core.event_bus import EventBus # Assuming event_bus is in core
    
    # Create components
    event_bus = EventBus()
    pattern_recognition = PatternRecognition(event_bus=event_bus)
    
    # Start event processing
    event_bus.start_processing()
    
    # Create sample kline data
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(100)]
    close_prices = 50000 + np.cumsum(np.random.randn(100) * 100)
    sample_df = pd.DataFrame({
        "open": close_prices - 10,
        "high": close_prices + 50,
        "low": close_prices - 50,
        "close": close_prices,
        "volume": np.random.rand(100) * 1000
    }, index=timestamps)
    
    # Simulate klines update event
    klines_list = sample_df.reset_index().to_dict(orient="records")
    # Convert timestamp back to string for event payload
    for kline in klines_list:
        kline["timestamp"] = kline["timestamp"].isoformat()
        
    await event_bus.publish("pipeline.klines_updated", {
        "symbol": "BTC/USDC",
        "timeframe": "1h",
        "klines": klines_list
    })
    
    # Wait for analysis to complete
    await asyncio.sleep(1)
    
    # Stop event processing
    event_bus.stop_processing()

if __name__ == "__main__":
    # Create analysis directory if it doesn't exist
    if not os.path.exists("analysis"):
        os.makedirs("analysis")
    # Run test
    asyncio.run(test())
