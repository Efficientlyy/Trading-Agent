#!/usr/bin/env python
"""
Real-Time Chart Visualization Module for Trading-Agent

This module provides real-time chart visualization capabilities for multiple assets
(BTC, ETH, SOL) with support for different timeframes and technical indicators.
"""

import os
import sys
import json
import time
import logging
import asyncio
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chart_visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("chart_visualization")

class ChartVisualization:
    """
    Real-Time Chart Visualization for multiple assets.
    
    This class provides real-time chart visualization capabilities for multiple assets
    with support for different timeframes, technical indicators, and strategic markers.
    """
    
    def __init__(self, event_bus=None, data_pipeline=None):
        """
        Initialize Chart Visualization.
        
        Args:
            event_bus: Event Bus instance
            data_pipeline: Unified Data Pipeline instance
        """
        self.event_bus = event_bus
        self.data_pipeline = data_pipeline
        
        # Supported assets and timeframes
        self.supported_assets = ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
        self.supported_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        # Chart data
        self.chart_data = {}
        for symbol in self.supported_assets:
            self.chart_data[symbol] = {}
            for timeframe in self.supported_timeframes:
                self.chart_data[symbol][timeframe] = {
                    "klines": pd.DataFrame(),
                    "indicators": {},
                    "markers": []
                }
        
        # Active charts
        self.active_charts = {}
        
        # Chart update callbacks
        self.update_callbacks = {}
        
        # Register event handlers if event bus is provided
        if self.event_bus:
            self._register_event_handlers()
        
        logger.info("Chart Visualization initialized")
    
    def set_event_bus(self, event_bus):
        """
        Set Event Bus instance.
        
        Args:
            event_bus: Event Bus instance
        """
        self.event_bus = event_bus
        self._register_event_handlers()
        
        logger.info("Event Bus set")
    
    def set_data_pipeline(self, data_pipeline):
        """
        Set Unified Data Pipeline instance.
        
        Args:
            data_pipeline: Unified Data Pipeline instance
        """
        self.data_pipeline = data_pipeline
        
        # Initialize chart data from pipeline
        self._initialize_chart_data()
        
        logger.info("Unified Data Pipeline set")
    
    def _register_event_handlers(self):
        """Register event handlers with Event Bus."""
        self.event_bus.subscribe("pipeline.klines_updated", self._handle_klines_update)
        self.event_bus.subscribe("pipeline.market_data_updated", self._handle_market_data_update)
        self.event_bus.subscribe("visualization.strategic_decision", self._handle_strategic_decision)
        self.event_bus.subscribe("visualization.pattern_detected", self._handle_pattern_detected)
        self.event_bus.subscribe("visualization.risk_alert", self._handle_risk_alert)
        
        logger.info("Event handlers registered")
    
    def _initialize_chart_data(self):
        """Initialize chart data from data pipeline."""
        if not self.data_pipeline:
            logger.warning("Data pipeline not set, cannot initialize chart data")
            return
        
        for symbol in self.supported_assets:
            for timeframe in self.supported_timeframes:
                # Get klines data
                df = self.data_pipeline.get_klines_dataframe(symbol, timeframe)
                if df is not None and not df.empty:
                    self.chart_data[symbol][timeframe]["klines"] = df
                    
                    # Calculate indicators
                    self._calculate_indicators(symbol, timeframe)
        
        logger.info("Chart data initialized from pipeline")
    
    def _calculate_indicators(self, symbol: str, timeframe: str):
        """
        Calculate technical indicators for chart.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
        """
        df = self.chart_data[symbol][timeframe]["klines"]
        if df.empty:
            return
        
        # Calculate SMA
        self.chart_data[symbol][timeframe]["indicators"]["sma_20"] = df["close"].rolling(window=20).mean()
        self.chart_data[symbol][timeframe]["indicators"]["sma_50"] = df["close"].rolling(window=50).mean()
        self.chart_data[symbol][timeframe]["indicators"]["sma_200"] = df["close"].rolling(window=200).mean()
        
        # Calculate EMA
        self.chart_data[symbol][timeframe]["indicators"]["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        self.chart_data[symbol][timeframe]["indicators"]["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
        
        # Calculate MACD
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        self.chart_data[symbol][timeframe]["indicators"]["macd"] = macd
        self.chart_data[symbol][timeframe]["indicators"]["macd_signal"] = signal
        self.chart_data[symbol][timeframe]["indicators"]["macd_histogram"] = histogram
        
        # Calculate RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        self.chart_data[symbol][timeframe]["indicators"]["rsi"] = rsi
        
        # Calculate Bollinger Bands
        sma_20 = df["close"].rolling(window=20).mean()
        std_20 = df["close"].rolling(window=20).std()
        upper_band = sma_20 + (std_20 * 2)
        lower_band = sma_20 - (std_20 * 2)
        
        self.chart_data[symbol][timeframe]["indicators"]["bb_upper"] = upper_band
        self.chart_data[symbol][timeframe]["indicators"]["bb_middle"] = sma_20
        self.chart_data[symbol][timeframe]["indicators"]["bb_lower"] = lower_band
        
        logger.info(f"Calculated indicators for {symbol} {timeframe}")
    
    async def _handle_klines_update(self, topic: str, data: Dict[str, Any]):
        """
        Handle klines update event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        symbol = data.get("symbol")
        timeframe = data.get("timeframe")
        klines = data.get("klines")
        
        if not symbol or not timeframe or not klines:
            logger.warning("Klines update missing required fields")
            return
        
        # Check if symbol and timeframe are supported
        if symbol not in self.supported_assets or timeframe not in self.supported_timeframes:
            logger.warning(f"Unsupported symbol or timeframe: {symbol} {timeframe}")
            return
        
        # Convert klines to DataFrame
        df = pd.DataFrame(klines)
        
        # Set timestamp as index
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        
        # Convert columns to numeric
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
        
        # Update chart data
        self.chart_data[symbol][timeframe]["klines"] = df
        
        # Calculate indicators
        self._calculate_indicators(symbol, timeframe)
        
        # Notify active charts
        await self._notify_chart_update(symbol, timeframe)
    
    async def _handle_market_data_update(self, topic: str, data: Dict[str, Any]):
        """
        Handle market data update event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        symbol = data.get("symbol")
        
        if not symbol:
            logger.warning("Market data update missing symbol")
            return
        
        # Check if symbol is supported
        if symbol not in self.supported_assets:
            logger.warning(f"Unsupported symbol: {symbol}")
            return
        
        # Update real-time price for all timeframes
        for timeframe in self.supported_timeframes:
            # Notify active charts
            await self._notify_chart_update(symbol, timeframe, real_time_price=data.get("price"))
    
    async def _handle_strategic_decision(self, topic: str, data: Dict[str, Any]):
        """
        Handle strategic decision event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        symbol = data.get("symbol")
        
        if not symbol:
            logger.warning("Strategic decision missing symbol")
            return
        
        # Check if symbol is supported
        if symbol not in self.supported_assets:
            logger.warning(f"Unsupported symbol: {symbol}")
            return
        
        # Create marker
        marker = {
            "type": "decision",
            "decision_type": data.get("decision_type", "unknown"),
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "price": data.get("price"),
            "direction": data.get("direction", "neutral"),
            "confidence": data.get("confidence", 0.5),
            "color": self._get_decision_color(data.get("direction", "neutral")),
            "icon": self._get_decision_icon(data.get("decision_type", "unknown")),
            "text": data.get("summary", "Strategic decision")
        }
        
        # Add marker to all timeframes
        for timeframe in self.supported_timeframes:
            self.chart_data[symbol][timeframe]["markers"].append(marker)
            
            # Limit markers to 100
            if len(self.chart_data[symbol][timeframe]["markers"]) > 100:
                self.chart_data[symbol][timeframe]["markers"] = self.chart_data[symbol][timeframe]["markers"][-100:]
            
            # Notify active charts
            await self._notify_chart_update(symbol, timeframe)
    
    async def _handle_pattern_detected(self, topic: str, data: Dict[str, Any]):
        """
        Handle pattern detected event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        symbol = data.get("symbol")
        timeframe = data.get("timeframe")
        
        if not symbol or not timeframe:
            logger.warning("Pattern detected missing symbol or timeframe")
            return
        
        # Check if symbol and timeframe are supported
        if symbol not in self.supported_assets or timeframe not in self.supported_timeframes:
            logger.warning(f"Unsupported symbol or timeframe: {symbol} {timeframe}")
            return
        
        # Create marker
        marker = {
            "type": "pattern",
            "pattern_type": data.get("pattern_type", "unknown"),
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "price": data.get("price"),
            "direction": data.get("direction", "neutral"),
            "confidence": data.get("confidence", 0.5),
            "color": self._get_pattern_color(data.get("pattern_type", "unknown"), data.get("direction", "neutral")),
            "icon": self._get_pattern_icon(data.get("pattern_type", "unknown")),
            "text": f"{data.get('pattern_type', 'Unknown pattern')} ({data.get('confidence', 0.5):.2f})"
        }
        
        # Add marker to specific timeframe
        self.chart_data[symbol][timeframe]["markers"].append(marker)
        
        # Limit markers to 100
        if len(self.chart_data[symbol][timeframe]["markers"]) > 100:
            self.chart_data[symbol][timeframe]["markers"] = self.chart_data[symbol][timeframe]["markers"][-100:]
        
        # Notify active charts
        await self._notify_chart_update(symbol, timeframe)
    
    async def _handle_risk_alert(self, topic: str, data: Dict[str, Any]):
        """
        Handle risk alert event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        symbol = data.get("symbol")
        
        if not symbol:
            logger.warning("Risk alert missing symbol")
            return
        
        # Check if symbol is supported
        if symbol not in self.supported_assets:
            logger.warning(f"Unsupported symbol: {symbol}")
            return
        
        # Create marker
        marker = {
            "type": "alert",
            "alert_type": data.get("alert_type", "unknown"),
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "price": data.get("price"),
            "severity": data.get("severity", "medium"),
            "color": self._get_alert_color(data.get("severity", "medium")),
            "icon": self._get_alert_icon(data.get("alert_type", "unknown")),
            "text": data.get("message", "Risk alert")
        }
        
        # Add marker to all timeframes
        for timeframe in self.supported_timeframes:
            self.chart_data[symbol][timeframe]["markers"].append(marker)
            
            # Limit markers to 100
            if len(self.chart_data[symbol][timeframe]["markers"]) > 100:
                self.chart_data[symbol][timeframe]["markers"] = self.chart_data[symbol][timeframe]["markers"][-100:]
            
            # Notify active charts
            await self._notify_chart_update(symbol, timeframe)
    
    async def _notify_chart_update(self, symbol: str, timeframe: str, real_time_price: Optional[float] = None):
        """
        Notify chart update to subscribers.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            real_time_price: Real-time price (optional)
        """
        chart_key = f"{symbol}_{timeframe}"
        
        if chart_key in self.update_callbacks:
            for callback in self.update_callbacks[chart_key]:
                if callback:
                    try:
                        # Create update data
                        update_data = {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Add real-time price if available
                        if real_time_price is not None:
                            update_data["real_time_price"] = real_time_price
                        
                        # Call callback
                        if asyncio.iscoroutinefunction(callback):
                            await callback(update_data)
                        else:
                            callback(update_data)
                    except Exception as e:
                        logger.error(f"Error in chart update callback: {e}")
    
    def activate_chart(self, symbol: str, timeframe: str) -> bool:
        """
        Activate chart for symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            True if activated successfully, False otherwise
        """
        # Check if symbol and timeframe are supported
        if symbol not in self.supported_assets:
            logger.warning(f"Unsupported symbol: {symbol}")
            return False
        
        if timeframe not in self.supported_timeframes:
            logger.warning(f"Unsupported timeframe: {timeframe}")
            return False
        
        # Activate chart
        chart_key = f"{symbol}_{timeframe}"
        self.active_charts[chart_key] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "activated_at": datetime.now().isoformat()
        }
        
        logger.info(f"Activated chart for {symbol} {timeframe}")
        return True
    
    def deactivate_chart(self, symbol: str, timeframe: str) -> bool:
        """
        Deactivate chart for symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            True if deactivated successfully, False otherwise
        """
        chart_key = f"{symbol}_{timeframe}"
        
        if chart_key in self.active_charts:
            self.active_charts.pop(chart_key)
            logger.info(f"Deactivated chart for {symbol} {timeframe}")
            return True
        
        logger.warning(f"Chart not active: {symbol} {timeframe}")
        return False
    
    def subscribe_to_chart_updates(self, symbol: str, timeframe: str, callback: Callable[[Dict[str, Any]], None]) -> int:
        """
        Subscribe to chart updates.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            callback: Callback function to handle chart updates
            
        Returns:
            Subscription ID
        """
        # Check if symbol and timeframe are supported
        if symbol not in self.supported_assets:
            logger.warning(f"Unsupported symbol: {symbol}")
            return -1
        
        if timeframe not in self.supported_timeframes:
            logger.warning(f"Unsupported timeframe: {timeframe}")
            return -1
        
        # Register callback
        chart_key = f"{symbol}_{timeframe}"
        
        if chart_key not in self.update_callbacks:
            self.update_callbacks[chart_key] = []
        
        subscription_id = len(self.update_callbacks[chart_key])
        self.update_callbacks[chart_key].append(callback)
        
        logger.info(f"New subscriber to chart updates for {symbol} {timeframe}: {subscription_id}")
        return subscription_id
    
    def unsubscribe_from_chart_updates(self, symbol: str, timeframe: str, subscription_id: int) -> bool:
        """
        Unsubscribe from chart updates.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            subscription_id: Subscription ID
            
        Returns:
            True if unsubscribed successfully, False otherwise
        """
        chart_key = f"{symbol}_{timeframe}"
        
        if chart_key not in self.update_callbacks:
            logger.warning(f"No subscribers for chart: {symbol} {timeframe}")
            return False
        
        if subscription_id >= len(self.update_callbacks[chart_key]):
            logger.warning(f"Invalid subscription ID: {subscription_id}")
            return False
        
        self.update_callbacks[chart_key][subscription_id] = None
        logger.info(f"Unsubscribed from chart updates for {symbol} {timeframe}: {subscription_id}")
        return True
    
    def get_chart_data(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Get chart data for symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            Chart data
        """
        # Check if symbol and timeframe are supported
        if symbol not in self.supported_assets:
            logger.warning(f"Unsupported symbol: {symbol}")
            return {}
        
        if timeframe not in self.supported_timeframes:
            logger.warning(f"Unsupported timeframe: {timeframe}")
            return {}
        
        # Get chart data
        klines = self.chart_data[symbol][timeframe]["klines"]
        indicators = self.chart_data[symbol][timeframe]["indicators"]
        markers = self.chart_data[symbol][timeframe]["markers"]
        
        # Convert DataFrame to dict
        klines_dict = {}
        if not klines.empty:
            klines_dict = klines.reset_index().to_dict(orient="records")
        
        # Convert indicators to dict
        indicators_dict = {}
        for indicator_name, indicator_data in indicators.items():
            if isinstance(indicator_data, pd.Series):
                indicators_dict[indicator_name] = indicator_data.dropna().to_dict()
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "klines": klines_dict,
            "indicators": indicators_dict,
            "markers": markers
        }
    
    def get_supported_assets(self) -> List[str]:
        """
        Get supported assets.
        
        Returns:
            List of supported assets
        """
        return self.supported_assets
    
    def get_supported_timeframes(self) -> List[str]:
        """
        Get supported timeframes.
        
        Returns:
            List of supported timeframes
        """
        return self.supported_timeframes
    
    def get_active_charts(self) -> Dict[str, Dict[str, Any]]:
        """
        Get active charts.
        
        Returns:
            Dictionary of active charts
        """
        return self.active_charts
    
    def _get_decision_color(self, direction: str) -> str:
        """
        Get color for decision based on direction.
        
        Args:
            direction: Decision direction
            
        Returns:
            Color code
        """
        color_map = {
            "bullish": "#4CAF50",  # Green
            "bearish": "#F44336",  # Red
            "neutral": "#FFC107",  # Amber
            "unknown": "#9E9E9E"   # Gray
        }
        
        return color_map.get(direction.lower(), "#9E9E9E")
    
    def _get_decision_icon(self, decision_type: str) -> str:
        """
        Get icon for decision based on type.
        
        Args:
            decision_type: Decision type
            
        Returns:
            Icon name
        """
        icon_map = {
            "entry": "login",
            "exit": "logout",
            "risk_adjustment": "shield",
            "position_sizing": "resize",
            "market_trend": "trending_up",
            "pattern_confirmation": "check_circle",
            "emergency": "warning"
        }
        
        return icon_map.get(decision_type.lower(), "info")
    
    def _get_pattern_color(self, pattern_type: str, direction: str) -> str:
        """
        Get color for pattern based on type and direction.
        
        Args:
            pattern_type: Pattern type
            direction: Pattern direction
            
        Returns:
            Color code
        """
        if direction.lower() == "bullish":
            return "#4CAF50"  # Green
        elif direction.lower() == "bearish":
            return "#F44336"  # Red
        else:
            return "#9E9E9E"  # Gray
    
    def _get_pattern_icon(self, pattern_type: str) -> str:
        """
        Get icon for pattern based on type.
        
        Args:
            pattern_type: Pattern type
            
        Returns:
            Icon name
        """
        icon_map = {
            "double_top": "signal_cellular_4_bar",
            "double_bottom": "signal_cellular_4_bar",
            "head_and_shoulders": "timeline",
            "inverse_head_and_shoulders": "timeline",
            "triangle": "change_history",
            "wedge": "call_split",
            "flag": "flag",
            "pennant": "flag",
            "rectangle": "crop_square",
            "cup_and_handle": "coffee",
            "rounding_bottom": "panorama_fish_eye",
            "rounding_top": "panorama_fish_eye"
        }
        
        return icon_map.get(pattern_type.lower(), "auto_graph")
    
    def _get_alert_color(self, severity: str) -> str:
        """
        Get color for alert based on severity.
        
        Args:
            severity: Alert severity
            
        Returns:
            Color code
        """
        color_map = {
            "low": "#2196F3",     # Blue
            "medium": "#FFC107",  # Amber
            "high": "#F44336",    # Red
            "critical": "#B71C1C" # Dark Red
        }
        
        return color_map.get(severity.lower(), "#9E9E9E")
    
    def _get_alert_icon(self, alert_type: str) -> str:
        """
        Get icon for alert based on type.
        
        Args:
            alert_type: Alert type
            
        Returns:
            Icon name
        """
        icon_map = {
            "price_alert": "monetization_on",
            "volatility_alert": "flash_on",
            "liquidity_alert": "water",
            "trend_reversal": "swap_vert",
            "stop_loss": "pan_tool",
            "take_profit": "thumb_up",
            "risk_limit": "report_problem"
        }
        
        return icon_map.get(alert_type.lower(), "notifications")


# For testing
async def test():
    """Test function."""
    from core.event_bus import EventBus
    from data.unified_pipeline import UnifiedDataPipeline
    
    # Create components
    event_bus = EventBus()
    data_pipeline = UnifiedDataPipeline()
    chart_visualization = ChartVisualization()
    
    # Connect components
    data_pipeline.set_event_bus(event_bus)
    chart_visualization.set_event_bus(event_bus)
    chart_visualization.set_data_pipeline(data_pipeline)
    
    # Start event processing
    event_bus.start_processing()
    
    # Activate chart
    chart_visualization.activate_chart("BTC/USDC", "1h")
    
    # Subscribe to chart updates
    def print_chart_update(update):
        print(f"Chart update: {update}")
    
    chart_visualization.subscribe_to_chart_updates("BTC/USDC", "1h", print_chart_update)
    
    # Publish klines update
    klines = [
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
    ]
    
    await event_bus.publish("pipeline.klines_updated", {
        "symbol": "BTC/USDC",
        "timeframe": "1h",
        "klines": klines
    })
    
    # Publish strategic decision
    await event_bus.publish("visualization.strategic_decision", {
        "decision_type": "entry",
        "symbol": "BTC/USDC",
        "direction": "bullish",
        "confidence": 0.85,
        "summary": "Enter long position based on bullish pattern",
        "timestamp": datetime.now().isoformat()
    })
    
    # Wait for events to be processed
    await asyncio.sleep(1)
    
    # Get chart data
    chart_data = chart_visualization.get_chart_data("BTC/USDC", "1h")
    print(f"Chart data: {chart_data}")
    
    # Stop event processing
    event_bus.stop_processing()


if __name__ == "__main__":
    asyncio.run(test())
