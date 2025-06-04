#!/usr/bin/env python
"""
Real-time Chart Visualization for LLM Strategic Overseer.

This module provides real-time chart visualization for traded assets,
with support for technical indicators and pattern recognition.
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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from io import BytesIO
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'chart_visualization.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import event bus
from ..core.event_bus import EventBus

class ChartVisualization:
    """
    Real-time Chart Visualization for LLM Strategic Overseer.
    
    This class provides real-time chart visualization for traded assets,
    with support for technical indicators and pattern recognition.
    """
    
    def __init__(self, config, event_bus: Optional[EventBus] = None):
        """
        Initialize Chart Visualization.
        
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
        
        # Initialize chart settings
        self.chart_settings = {
            "timeframe": self.config.get("visualization.timeframe", "1h"),
            "max_points": self.config.get("visualization.max_points", 100),
            "indicators": self.config.get("visualization.indicators", ["sma", "rsi", "macd"]),
            "theme": self.config.get("visualization.theme", "dark"),
            "show_patterns": self.config.get("visualization.show_patterns", True),
            "show_signals": self.config.get("visualization.show_signals", True)
        }
        
        # Initialize supported symbols
        self.symbols = self.config.get("trading.symbols", ["BTC/USDC", "ETH/USDC", "SOL/USDC"])
        
        # Initialize chart output directory
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output",
            "charts"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Subscribe to relevant events
        self._subscribe_to_events()
        
        logger.info("Chart Visualization initialized")
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        # Subscribe to market data events
        self.subscriptions["market_data"] = self.event_bus.subscribe(
            "trading.market_data", self._handle_market_data
        )
        
        # Subscribe to indicator events
        self.subscriptions["indicator"] = self.event_bus.subscribe(
            "analysis.indicator", self._handle_indicator
        )
        
        # Subscribe to pattern events
        self.subscriptions["pattern"] = self.event_bus.subscribe(
            "analysis.pattern", self._handle_pattern
        )
        
        # Subscribe to strategy decision events
        self.subscriptions["strategy_decision"] = self.event_bus.subscribe(
            "llm.strategy_decision", self._handle_strategy_decision
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
            
            # Limit data points
            if len(self.market_data[symbol]["timestamp"]) > self.chart_settings["max_points"]:
                self.market_data[symbol]["timestamp"] = self.market_data[symbol]["timestamp"][-self.chart_settings["max_points"]:]
                self.market_data[symbol]["price"] = self.market_data[symbol]["price"][-self.chart_settings["max_points"]:]
                self.market_data[symbol]["volume"] = self.market_data[symbol]["volume"][-self.chart_settings["max_points"]:]
            
            # Update chart
            await self.update_chart(symbol)
            
            logger.debug(f"Updated market data for {symbol}")
        except Exception as e:
            logger.error(f"Error handling market data: {e}")
            logger.error(traceback.format_exc())
    
    async def _handle_indicator(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Handle indicator event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        try:
            symbol = data.get("symbol")
            indicator_type = data.get("indicator_type")
            indicator_values = data.get("values", [])
            
            if not symbol or not indicator_type or symbol not in self.symbols:
                return
            
            # Initialize symbol indicators if not exists
            if symbol not in self.indicators:
                self.indicators[symbol] = {}
            
            # Add indicator data
            self.indicators[symbol][indicator_type] = indicator_values
            
            # Update chart
            await self.update_chart(symbol)
            
            logger.debug(f"Updated {indicator_type} indicator for {symbol}")
        except Exception as e:
            logger.error(f"Error handling indicator: {e}")
            logger.error(traceback.format_exc())
    
    async def _handle_pattern(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Handle pattern event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        try:
            symbol = data.get("symbol")
            pattern_type = data.get("pattern_type")
            pattern_data = data.get("pattern_data", {})
            
            if not symbol or not pattern_type or symbol not in self.symbols:
                return
            
            # Initialize symbol patterns if not exists
            if symbol not in self.patterns:
                self.patterns[symbol] = []
            
            # Add pattern data with timestamp
            pattern = {
                "type": pattern_type,
                "data": pattern_data,
                "timestamp": datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
            }
            
            self.patterns[symbol].append(pattern)
            
            # Limit patterns (keep last 20)
            if len(self.patterns[symbol]) > 20:
                self.patterns[symbol] = self.patterns[symbol][-20:]
            
            # Update chart
            await self.update_chart(symbol)
            
            logger.debug(f"Added {pattern_type} pattern for {symbol}")
        except Exception as e:
            logger.error(f"Error handling pattern: {e}")
            logger.error(traceback.format_exc())
    
    async def _handle_strategy_decision(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Handle strategy decision event.
        
        Args:
            topic: Event topic
            data: Event data
        """
        try:
            if "trade" not in data:
                return
            
            trade_data = data.get("trade", {})
            symbol = trade_data.get("symbol")
            
            if not symbol or symbol not in self.symbols:
                return
            
            # Update chart with strategy decision
            await self.update_chart(symbol, strategy_decision=data)
            
            logger.debug(f"Updated chart with strategy decision for {symbol}")
        except Exception as e:
            logger.error(f"Error handling strategy decision: {e}")
            logger.error(traceback.format_exc())
    
    async def update_chart(self, symbol: str, strategy_decision: Optional[Dict[str, Any]] = None) -> str:
        """
        Update chart for symbol.
        
        Args:
            symbol: Symbol to update chart for
            strategy_decision: Optional strategy decision to highlight
            
        Returns:
            Path to generated chart image
        """
        try:
            if symbol not in self.market_data or not self.market_data[symbol]["timestamp"]:
                logger.warning(f"No market data for {symbol}")
                return ""
            
            # Create figure
            fig = self._create_chart(symbol, strategy_decision)
            
            # Save chart
            output_path = os.path.join(self.output_dir, f"{symbol.replace('/', '_')}_chart.png")
            fig.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            
            # Publish chart updated event
            await self.event_bus.publish(
                "visualization.chart_updated",
                {
                    "symbol": symbol,
                    "chart_path": output_path,
                    "timestamp": datetime.now().isoformat()
                },
                "normal"
            )
            
            logger.debug(f"Updated chart for {symbol}")
            return output_path
        except Exception as e:
            logger.error(f"Error updating chart: {e}")
            logger.error(traceback.format_exc())
            return ""
    
    def _create_chart(self, symbol: str, strategy_decision: Optional[Dict[str, Any]] = None) -> Figure:
        """
        Create chart for symbol.
        
        Args:
            symbol: Symbol to create chart for
            strategy_decision: Optional strategy decision to highlight
            
        Returns:
            Matplotlib figure
        """
        # Set theme
        if self.chart_settings["theme"] == "dark":
            plt.style.use("dark_background")
        else:
            plt.style.use("default")
        
        # Create figure and subplots
        fig, (ax_price, ax_volume) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
        
        # Get data
        timestamps = self.market_data[symbol]["timestamp"]
        prices = self.market_data[symbol]["price"]
        volumes = self.market_data[symbol]["volume"]
        
        # Plot price
        ax_price.plot(timestamps, prices, label="Price", color="white", linewidth=1.5)
        ax_price.set_title(f"{symbol} - {self.chart_settings['timeframe']} Chart")
        ax_price.set_ylabel("Price")
        ax_price.grid(True, alpha=0.3)
        
        # Plot volume
        ax_volume.bar(timestamps, volumes, label="Volume", color="blue", alpha=0.7)
        ax_volume.set_xlabel("Time")
        ax_volume.set_ylabel("Volume")
        ax_volume.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax_price, ax_volume]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot indicators
        if symbol in self.indicators:
            self._plot_indicators(ax_price, symbol, timestamps)
        
        # Plot patterns
        if self.chart_settings["show_patterns"] and symbol in self.patterns:
            self._plot_patterns(ax_price, symbol, timestamps, prices)
        
        # Plot strategy decision
        if self.chart_settings["show_signals"] and strategy_decision:
            self._plot_strategy_decision(ax_price, strategy_decision, timestamps, prices)
        
        # Add legend
        ax_price.legend(loc="upper left")
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    def _plot_indicators(self, ax, symbol: str, timestamps) -> None:
        """
        Plot indicators on chart.
        
        Args:
            ax: Matplotlib axis
            symbol: Symbol to plot indicators for
            timestamps: Timestamps for x-axis
        """
        indicators = self.indicators[symbol]
        
        # Plot SMA
        if "sma" in indicators and "sma" in self.chart_settings["indicators"]:
            sma_values = indicators["sma"]
            if len(sma_values) == len(timestamps):
                ax.plot(timestamps, sma_values, label="SMA", color="yellow", linewidth=1, alpha=0.8)
        
        # Plot EMA
        if "ema" in indicators and "ema" in self.chart_settings["indicators"]:
            ema_values = indicators["ema"]
            if len(ema_values) == len(timestamps):
                ax.plot(timestamps, ema_values, label="EMA", color="orange", linewidth=1, alpha=0.8)
        
        # Plot Bollinger Bands
        if "bollinger_upper" in indicators and "bollinger_lower" in indicators and "bollinger" in self.chart_settings["indicators"]:
            upper_values = indicators["bollinger_upper"]
            lower_values = indicators["bollinger_lower"]
            if len(upper_values) == len(timestamps) and len(lower_values) == len(timestamps):
                ax.plot(timestamps, upper_values, label="Bollinger Upper", color="green", linewidth=1, alpha=0.5)
                ax.plot(timestamps, lower_values, label="Bollinger Lower", color="green", linewidth=1, alpha=0.5)
                ax.fill_between(timestamps, upper_values, lower_values, color="green", alpha=0.1)
    
    def _plot_patterns(self, ax, symbol: str, timestamps, prices) -> None:
        """
        Plot patterns on chart.
        
        Args:
            ax: Matplotlib axis
            symbol: Symbol to plot patterns for
            timestamps: Timestamps for x-axis
            prices: Prices for y-axis
        """
        patterns = self.patterns[symbol]
        
        for pattern in patterns:
            pattern_type = pattern["type"]
            pattern_timestamp = pattern["timestamp"]
            
            # Find closest timestamp index
            closest_idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - pattern_timestamp))
            
            if closest_idx < len(prices):
                pattern_price = prices[closest_idx]
                
                # Plot pattern marker
                if pattern_type == "double_bottom":
                    ax.scatter(pattern_timestamp, pattern_price, marker="^", color="lime", s=100, label=f"Double Bottom")
                elif pattern_type == "double_top":
                    ax.scatter(pattern_timestamp, pattern_price, marker="v", color="red", s=100, label=f"Double Top")
                elif pattern_type == "head_and_shoulders":
                    ax.scatter(pattern_timestamp, pattern_price, marker="X", color="orange", s=100, label=f"Head & Shoulders")
                elif pattern_type == "inverse_head_and_shoulders":
                    ax.scatter(pattern_timestamp, pattern_price, marker="X", color="cyan", s=100, label=f"Inv. Head & Shoulders")
                else:
                    ax.scatter(pattern_timestamp, pattern_price, marker="*", color="yellow", s=100, label=f"Pattern: {pattern_type}")
    
    def _plot_strategy_decision(self, ax, strategy_decision: Dict[str, Any], timestamps, prices) -> None:
        """
        Plot strategy decision on chart.
        
        Args:
            ax: Matplotlib axis
            strategy_decision: Strategy decision data
            timestamps: Timestamps for x-axis
            prices: Prices for y-axis
        """
        if "trade" not in strategy_decision:
            return
        
        trade_data = strategy_decision["trade"]
        side = trade_data.get("side")
        price = trade_data.get("price")
        
        # Use latest timestamp
        latest_timestamp = timestamps[-1] if timestamps else datetime.now()
        
        # Plot strategy decision marker
        if side == "buy":
            ax.scatter(latest_timestamp, price, marker="^", color="lime", s=150, label="Buy Signal")
            ax.axhline(y=price, color="lime", linestyle="--", alpha=0.5)
        elif side == "sell":
            ax.scatter(latest_timestamp, price, marker="v", color="red", s=150, label="Sell Signal")
            ax.axhline(y=price, color="red", linestyle="--", alpha=0.5)
    
    def get_chart_as_base64(self, symbol: str) -> str:
        """
        Get chart as base64 encoded string.
        
        Args:
            symbol: Symbol to get chart for
            
        Returns:
            Base64 encoded chart image
        """
        try:
            if symbol not in self.market_data or not self.market_data[symbol]["timestamp"]:
                logger.warning(f"No market data for {symbol}")
                return ""
            
            # Create figure
            fig = self._create_chart(symbol)
            
            # Convert to base64
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            
            # Encode as base64
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            
            return img_base64
        except Exception as e:
            logger.error(f"Error getting chart as base64: {e}")
            logger.error(traceback.format_exc())
            return ""
    
    def get_latest_chart_path(self, symbol: str) -> str:
        """
        Get path to latest chart for symbol.
        
        Args:
            symbol: Symbol to get chart for
            
        Returns:
            Path to chart image
        """
        output_path = os.path.join(self.output_dir, f"{symbol.replace('/', '_')}_chart.png")
        if os.path.exists(output_path):
            return output_path
        return ""
    
    def calculate_indicators(self, symbol: str) -> None:
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
            if "sma" in self.chart_settings["indicators"]:
                window = 20
                if len(prices) >= window:
                    sma = self._calculate_sma(prices, window)
                    self.indicators[symbol]["sma"] = sma
            
            # Calculate EMA
            if "ema" in self.chart_settings["indicators"]:
                window = 20
                if len(prices) >= window:
                    ema = self._calculate_ema(prices, window)
                    self.indicators[symbol]["ema"] = ema
            
            # Calculate Bollinger Bands
            if "bollinger" in self.chart_settings["indicators"]:
                window = 20
                if len(prices) >= window:
                    upper, lower = self._calculate_bollinger_bands(prices, window)
                    self.indicators[symbol]["bollinger_upper"] = upper
                    self.indicators[symbol]["bollinger_lower"] = lower
            
            # Calculate RSI
            if "rsi" in self.chart_settings["indicators"]:
                window = 14
                if len(prices) >= window + 1:
                    rsi = self._calculate_rsi(prices, window)
                    self.indicators[symbol]["rsi"] = rsi
            
            # Calculate MACD
            if "macd" in self.chart_settings["indicators"]:
                if len(prices) >= 26:
                    macd, signal, histogram = self._calculate_macd(prices)
                    self.indicators[symbol]["macd"] = macd
                    self.indicators[symbol]["macd_signal"] = signal
                    self.indicators[symbol]["macd_histogram"] = histogram
            
            logger.debug(f"Calculated indicators for {symbol}")
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            logger.error(traceback.format_exc())
    
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


# For testing
async def test():
    """Test function."""
    from ..config.config import Config
    
    # Create configuration
    config = Config()
    
    # Create event bus
    event_bus = EventBus()
    
    # Create chart visualization
    visualization = ChartVisualization(config, event_bus)
    
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
    visualization.market_data[symbol] = {
        "timestamp": timestamps,
        "price": prices,
        "volume": volumes
    }
    
    # Calculate indicators
    visualization.calculate_indicators(symbol)
    
    # Add pattern
    pattern = {
        "type": "double_bottom",
        "data": {},
        "timestamp": timestamps[70]
    }
    visualization.patterns[symbol] = [pattern]
    
    # Update chart
    chart_path = await visualization.update_chart(symbol)
    print(f"Chart saved to: {chart_path}")


if __name__ == "__main__":
    asyncio.run(test())
