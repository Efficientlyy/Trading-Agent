#!/usr/bin/env python
"""
Advanced Chart Component for Trading-Agent System

This module provides sophisticated chart visualization capabilities for the Trading-Agent system,
supporting BTC, ETH, and SOL trading pairs with advanced technical indicators and pattern recognition.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("chart_component")

class TechnicalIndicator:
    """Base class for technical indicators"""
    
    def __init__(self, name: str, color: str = None):
        """Initialize technical indicator
        
        Args:
            name: Indicator name
            color: Indicator color
        """
        self.name = name
        self.color = color or "#1E88E5"  # Default to blue
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicator values
        
        Args:
            df: Price data with OHLCV columns
            
        Returns:
            DataFrame with indicator values
        """
        raise NotImplementedError("Subclasses must implement calculate()")
    
    def plot(self, fig: go.Figure, row: int, col: int, df: pd.DataFrame) -> go.Figure:
        """Plot indicator on figure
        
        Args:
            fig: Plotly figure
            row: Row index
            col: Column index
            df: DataFrame with indicator values
            
        Returns:
            Updated figure
        """
        raise NotImplementedError("Subclasses must implement plot()")

class RSI(TechnicalIndicator):
    """Relative Strength Index indicator"""
    
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        """Initialize RSI indicator
        
        Args:
            period: RSI period
            overbought: Overbought threshold
            oversold: Oversold threshold
        """
        super().__init__(f"RSI({period})")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI values
        
        Args:
            df: Price data with OHLCV columns
            
        Returns:
            DataFrame with RSI values
        """
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        df[self.name] = rsi
        return df
    
    def plot(self, fig: go.Figure, row: int, col: int, df: pd.DataFrame) -> go.Figure:
        """Plot RSI on figure
        
        Args:
            fig: Plotly figure
            row: Row index
            col: Column index
            df: DataFrame with RSI values
            
        Returns:
            Updated figure
        """
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[self.name],
                name=self.name,
                line=dict(color=self.color, width=1.5)
            ),
            row=row,
            col=col
        )
        
        # Add overbought/oversold lines
        fig.add_shape(
            type="line",
            x0=df.index[0],
            y0=self.overbought,
            x1=df.index[-1],
            y1=self.overbought,
            line=dict(color="red", width=1, dash="dash"),
            row=row,
            col=col
        )
        
        fig.add_shape(
            type="line",
            x0=df.index[0],
            y0=self.oversold,
            x1=df.index[-1],
            y1=self.oversold,
            line=dict(color="green", width=1, dash="dash"),
            row=row,
            col=col
        )
        
        # Set y-axis range
        fig.update_yaxes(range=[0, 100], row=row, col=col)
        
        return fig

class MACD(TechnicalIndicator):
    """Moving Average Convergence Divergence indicator"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """Initialize MACD indicator
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal EMA period
        """
        super().__init__(f"MACD({fast_period},{slow_period},{signal_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD values
        
        Args:
            df: Price data with OHLCV columns
            
        Returns:
            DataFrame with MACD values
        """
        # Calculate EMAs
        fast_ema = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = df['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate MACD line and signal line
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Add to dataframe
        df[f"{self.name}_line"] = macd_line
        df[f"{self.name}_signal"] = signal_line
        df[f"{self.name}_hist"] = histogram
        
        return df
    
    def plot(self, fig: go.Figure, row: int, col: int, df: pd.DataFrame) -> go.Figure:
        """Plot MACD on figure
        
        Args:
            fig: Plotly figure
            row: Row index
            col: Column index
            df: DataFrame with MACD values
            
        Returns:
            Updated figure
        """
        # Plot MACD line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[f"{self.name}_line"],
                name="MACD Line",
                line=dict(color="#1E88E5", width=1.5)
            ),
            row=row,
            col=col
        )
        
        # Plot signal line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[f"{self.name}_signal"],
                name="Signal Line",
                line=dict(color="#FFC107", width=1.5)
            ),
            row=row,
            col=col
        )
        
        # Plot histogram
        colors = ["#4CAF50" if val >= 0 else "#F44336" for val in df[f"{self.name}_hist"]]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df[f"{self.name}_hist"],
                name="Histogram",
                marker=dict(color=colors),
                opacity=0.7
            ),
            row=row,
            col=col
        )
        
        # Add zero line
        fig.add_shape(
            type="line",
            x0=df.index[0],
            y0=0,
            x1=df.index[-1],
            y1=0,
            line=dict(color="gray", width=1, dash="dash"),
            row=row,
            col=col
        )
        
        return fig

class BollingerBands(TechnicalIndicator):
    """Bollinger Bands indicator"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """Initialize Bollinger Bands indicator
        
        Args:
            period: SMA period
            std_dev: Standard deviation multiplier
        """
        super().__init__(f"BB({period},{std_dev})")
        self.period = period
        self.std_dev = std_dev
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands values
        
        Args:
            df: Price data with OHLCV columns
            
        Returns:
            DataFrame with Bollinger Bands values
        """
        # Calculate middle band (SMA)
        df[f"{self.name}_middle"] = df['close'].rolling(window=self.period).mean()
        
        # Calculate standard deviation
        rolling_std = df['close'].rolling(window=self.period).std()
        
        # Calculate upper and lower bands
        df[f"{self.name}_upper"] = df[f"{self.name}_middle"] + (rolling_std * self.std_dev)
        df[f"{self.name}_lower"] = df[f"{self.name}_middle"] - (rolling_std * self.std_dev)
        
        # Calculate bandwidth and %B
        df[f"{self.name}_bandwidth"] = (df[f"{self.name}_upper"] - df[f"{self.name}_lower"]) / df[f"{self.name}_middle"]
        df[f"{self.name}_percent_b"] = (df['close'] - df[f"{self.name}_lower"]) / (df[f"{self.name}_upper"] - df[f"{self.name}_lower"])
        
        return df
    
    def plot(self, fig: go.Figure, row: int, col: int, df: pd.DataFrame) -> go.Figure:
        """Plot Bollinger Bands on figure
        
        Args:
            fig: Plotly figure
            row: Row index
            col: Column index
            df: DataFrame with Bollinger Bands values
            
        Returns:
            Updated figure
        """
        # Plot upper band
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[f"{self.name}_upper"],
                name="Upper Band",
                line=dict(color="#F44336", width=1, dash="dash"),
                opacity=0.7
            ),
            row=row,
            col=col
        )
        
        # Plot middle band
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[f"{self.name}_middle"],
                name="Middle Band",
                line=dict(color="#1E88E5", width=1.5)
            ),
            row=row,
            col=col
        )
        
        # Plot lower band
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[f"{self.name}_lower"],
                name="Lower Band",
                line=dict(color="#4CAF50", width=1, dash="dash"),
                opacity=0.7,
                fill='tonexty',
                fillcolor='rgba(30, 136, 229, 0.1)'
            ),
            row=row,
            col=col
        )
        
        return fig

class VolumeProfile(TechnicalIndicator):
    """Volume Profile indicator"""
    
    def __init__(self, num_bins: int = 20, width_percentage: float = 0.2):
        """Initialize Volume Profile indicator
        
        Args:
            num_bins: Number of price bins
            width_percentage: Width of volume bars as percentage of chart
        """
        super().__init__("Volume Profile")
        self.num_bins = num_bins
        self.width_percentage = width_percentage
        
    def calculate(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float]:
        """Calculate Volume Profile values
        
        Args:
            df: Price data with OHLCV columns
            
        Returns:
            Tuple of (bins, volumes, point_of_control)
        """
        # Get price range
        price_min = df['low'].min()
        price_max = df['high'].max()
        
        # Create price bins
        bins = np.linspace(price_min, price_max, self.num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate volume per bin
        volumes = np.zeros(self.num_bins)
        
        for i, row in df.iterrows():
            # Find bins that overlap with this candle
            candle_min_bin = np.searchsorted(bins, row['low']) - 1
            candle_max_bin = np.searchsorted(bins, row['high']) - 1
            
            # Distribute volume across bins
            if candle_min_bin == candle_max_bin:
                volumes[candle_min_bin] += row['volume']
            else:
                # Simple distribution based on price range overlap
                for bin_idx in range(candle_min_bin, candle_max_bin + 1):
                    if bin_idx < 0 or bin_idx >= self.num_bins:
                        continue
                    
                    bin_low = bins[bin_idx]
                    bin_high = bins[bin_idx + 1]
                    
                    overlap_low = max(bin_low, row['low'])
                    overlap_high = min(bin_high, row['high'])
                    
                    if overlap_high > overlap_low:
                        overlap_ratio = (overlap_high - overlap_low) / (row['high'] - row['low'])
                        volumes[bin_idx] += row['volume'] * overlap_ratio
        
        # Find point of control (price level with highest volume)
        poc_idx = np.argmax(volumes)
        point_of_control = bin_centers[poc_idx]
        
        return bin_centers, volumes, point_of_control
    
    def plot(self, fig: go.Figure, row: int, col: int, df: pd.DataFrame) -> go.Figure:
        """Plot Volume Profile on figure
        
        Args:
            fig: Plotly figure
            row: Row index
            col: Column index
            df: DataFrame with OHLCV data
            
        Returns:
            Updated figure
        """
        # Calculate volume profile
        bins, volumes, poc = self.calculate(df)
        
        # Normalize volumes
        max_volume = max(volumes)
        norm_volumes = volumes / max_volume
        
        # Calculate bar width based on chart width
        x_range = fig.layout.xaxis.range
        if x_range is None:
            x_range = [0, len(df)]
        
        chart_width = x_range[1] - x_range[0]
        bar_width = chart_width * self.width_percentage
        
        # Get the last timestamp for positioning
        last_timestamp = df.index[-1]
        
        # Plot horizontal volume bars
        for i, (price, volume) in enumerate(zip(bins, norm_volumes)):
            fig.add_shape(
                type="rect",
                x0=last_timestamp,
                y0=price - (bins[1] - bins[0]) / 2,
                x1=last_timestamp + bar_width * volume,
                y1=price + (bins[1] - bins[0]) / 2,
                fillcolor="rgba(30, 136, 229, 0.3)",
                line=dict(color="rgba(30, 136, 229, 0.5)", width=1),
                opacity=0.7,
                row=row,
                col=col
            )
        
        # Add point of control line
        fig.add_shape(
            type="line",
            x0=df.index[0],
            y0=poc,
            x1=last_timestamp + bar_width,
            y1=poc,
            line=dict(color="#F44336", width=2, dash="dash"),
            row=row,
            col=col
        )
        
        return fig

class PatternRecognition:
    """Pattern recognition visualization"""
    
    def __init__(self):
        """Initialize pattern recognition"""
        self.patterns = {
            "head_and_shoulders": {
                "color": "#F44336",
                "marker": "triangle-down",
                "name": "Head and Shoulders"
            },
            "inverse_head_and_shoulders": {
                "color": "#4CAF50",
                "marker": "triangle-up",
                "name": "Inverse Head and Shoulders"
            },
            "double_top": {
                "color": "#F44336",
                "marker": "circle",
                "name": "Double Top"
            },
            "double_bottom": {
                "color": "#4CAF50",
                "marker": "circle",
                "name": "Double Bottom"
            },
            "triple_top": {
                "color": "#F44336",
                "marker": "star",
                "name": "Triple Top"
            },
            "triple_bottom": {
                "color": "#4CAF50",
                "marker": "star",
                "name": "Triple Bottom"
            },
            "ascending_triangle": {
                "color": "#4CAF50",
                "marker": "diamond",
                "name": "Ascending Triangle"
            },
            "descending_triangle": {
                "color": "#F44336",
                "marker": "diamond",
                "name": "Descending Triangle"
            },
            "symmetrical_triangle": {
                "color": "#FFC107",
                "marker": "diamond",
                "name": "Symmetrical Triangle"
            },
            "flag": {
                "color": "#9C27B0",
                "marker": "square",
                "name": "Flag"
            },
            "pennant": {
                "color": "#9C27B0",
                "marker": "pentagon",
                "name": "Pennant"
            }
        }
    
    def plot_patterns(self, fig: go.Figure, row: int, col: int, patterns: List[Dict]) -> go.Figure:
        """Plot detected patterns on figure
        
        Args:
            fig: Plotly figure
            row: Row index
            col: Column index
            patterns: List of detected patterns with timestamp, pattern_type, and confidence
            
        Returns:
            Updated figure
        """
        if not patterns:
            return fig
        
        # Group patterns by type
        pattern_groups = {}
        
        for pattern in patterns:
            pattern_type = pattern.get("pattern_type")
            if pattern_type not in self.patterns:
                continue
                
            if pattern_type not in pattern_groups:
                pattern_groups[pattern_type] = []
                
            pattern_groups[pattern_type].append(pattern)
        
        # Plot each pattern group
        for pattern_type, pattern_list in pattern_groups.items():
            pattern_info = self.patterns[pattern_type]
            
            timestamps = [p["timestamp"] for p in pattern_list]
            prices = [p["price"] for p in pattern_list]
            confidences = [p["confidence"] for p in pattern_list]
            
            # Scale marker size based on confidence
            marker_sizes = [max(8, min(20, c * 20)) for c in confidences]
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=prices,
                    mode="markers",
                    name=pattern_info["name"],
                    marker=dict(
                        symbol=pattern_info["marker"],
                        color=pattern_info["color"],
                        size=marker_sizes,
                        line=dict(color="white", width=1)
                    ),
                    hovertemplate=(
                        f"{pattern_info['name']}<br>" +
                        "Price: %{y:$.2f}<br>" +
                        "Confidence: %{text:.0%}<br>" +
                        "Time: %{x}<extra></extra>"
                    ),
                    text=confidences,
                    showlegend=True
                ),
                row=row,
                col=col
            )
        
        return fig

class TradingSignals:
    """Trading signals visualization"""
    
    def __init__(self):
        """Initialize trading signals visualization"""
        pass
    
    def plot_signals(self, fig: go.Figure, row: int, col: int, signals: List[Dict]) -> go.Figure:
        """Plot trading signals on figure
        
        Args:
            fig: Plotly figure
            row: Row index
            col: Column index
            signals: List of trading signals with timestamp, signal_type, price, and confidence
            
        Returns:
            Updated figure
        """
        if not signals:
            return fig
        
        # Separate buy and sell signals
        buy_signals = [s for s in signals if s.get("signal_type", "").lower() == "buy"]
        sell_signals = [s for s in signals if s.get("signal_type", "").lower() == "sell"]
        
        # Plot buy signals
        if buy_signals:
            timestamps = [s["timestamp"] for s in buy_signals]
            prices = [s["price"] for s in buy_signals]
            confidences = [s.get("confidence", 0.5) for s in buy_signals]
            
            # Scale marker size based on confidence
            marker_sizes = [max(10, min(25, c * 25)) for c in confidences]
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=prices,
                    mode="markers",
                    name="Buy Signal",
                    marker=dict(
                        symbol="triangle-up",
                        color="#4CAF50",
                        size=marker_sizes,
                        line=dict(color="white", width=1)
                    ),
                    hovertemplate=(
                        "Buy Signal<br>" +
                        "Price: %{y:$.2f}<br>" +
                        "Confidence: %{text:.0%}<br>" +
                        "Time: %{x}<extra></extra>"
                    ),
                    text=confidences,
                    showlegend=True
                ),
                row=row,
                col=col
            )
        
        # Plot sell signals
        if sell_signals:
            timestamps = [s["timestamp"] for s in sell_signals]
            prices = [s["price"] for s in sell_signals]
            confidences = [s.get("confidence", 0.5) for s in sell_signals]
            
            # Scale marker size based on confidence
            marker_sizes = [max(10, min(25, c * 25)) for c in confidences]
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=prices,
                    mode="markers",
                    name="Sell Signal",
                    marker=dict(
                        symbol="triangle-down",
                        color="#F44336",
                        size=marker_sizes,
                        line=dict(color="white", width=1)
                    ),
                    hovertemplate=(
                        "Sell Signal<br>" +
                        "Price: %{y:$.2f}<br>" +
                        "Confidence: %{text:.0%}<br>" +
                        "Time: %{x}<extra></extra>"
                    ),
                    text=confidences,
                    showlegend=True
                ),
                row=row,
                col=col
            )
        
        return fig

class PredictionVisualization:
    """Price prediction visualization"""
    
    def __init__(self):
        """Initialize price prediction visualization"""
        pass
    
    def plot_predictions(self, fig: go.Figure, row: int, col: int, df: pd.DataFrame, 
                        predictions: Dict[str, List[float]], timestamps: List) -> go.Figure:
        """Plot price predictions on figure
        
        Args:
            fig: Plotly figure
            row: Row index
            col: Column index
            df: DataFrame with OHLCV data
            predictions: Dictionary with prediction types and values
            timestamps: List of future timestamps for predictions
            
        Returns:
            Updated figure
        """
        if not predictions or not timestamps:
            return fig
        
        # Plot each prediction type
        for pred_type, pred_values in predictions.items():
            if len(pred_values) != len(timestamps):
                logger.warning(f"Prediction values and timestamps length mismatch for {pred_type}")
                continue
            
            # Set color and style based on prediction type
            if pred_type.lower() == "baseline":
                color = "#9E9E9E"
                dash = "dash"
                width = 1
            elif pred_type.lower() == "ml":
                color = "#2196F3"
                dash = "solid"
                width = 2
            elif pred_type.lower() == "ensemble":
                color = "#9C27B0"
                dash = "solid"
                width = 2
            else:
                color = "#FFC107"
                dash = "dot"
                width = 1
            
            # Connect last actual price to first prediction
            if len(df) > 0:
                last_actual_timestamp = df.index[-1]
                last_actual_price = df['close'].iloc[-1]
                
                # Create connected line
                x_values = [last_actual_timestamp] + timestamps
                y_values = [last_actual_price] + pred_values
                
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode="lines",
                        name=f"{pred_type} Prediction",
                        line=dict(color=color, width=width, dash=dash),
                        hovertemplate=(
                            f"{pred_type} Prediction<br>" +
                            "Price: %{y:$.2f}<br>" +
                            "Time: %{x}<extra></extra>"
                        )
                    ),
                    row=row,
                    col=col
                )
            
        return fig

class ChartComponent:
    """Advanced chart component for Trading-Agent system"""
    
    def __init__(self, dark_mode: bool = True):
        """Initialize chart component
        
        Args:
            dark_mode: Whether to use dark mode theme
        """
        self.dark_mode = dark_mode
        self.indicators = {}
        self.pattern_recognition = PatternRecognition()
        self.trading_signals = TradingSignals()
        self.prediction_viz = PredictionVisualization()
        
        # Register default indicators
        self.register_indicator("rsi", RSI())
        self.register_indicator("macd", MACD())
        self.register_indicator("bollinger", BollingerBands())
        self.register_indicator("volume_profile", VolumeProfile())
        
        logger.info("Initialized ChartComponent")
    
    def register_indicator(self, id: str, indicator: TechnicalIndicator):
        """Register technical indicator
        
        Args:
            id: Indicator ID
            indicator: Technical indicator instance
        """
        self.indicators[id] = indicator
        logger.info(f"Registered indicator: {id} ({indicator.name})")
    
    def create_chart(self, df: pd.DataFrame, title: str = None, 
                    active_indicators: List[str] = None,
                    patterns: List[Dict] = None,
                    signals: List[Dict] = None,
                    predictions: Dict[str, List[float]] = None,
                    prediction_timestamps: List = None,
                    width: int = 1200, 
                    height: int = 800) -> go.Figure:
        """Create interactive chart
        
        Args:
            df: DataFrame with OHLCV data
            title: Chart title
            active_indicators: List of active indicator IDs
            patterns: List of detected patterns
            signals: List of trading signals
            predictions: Dictionary with prediction types and values
            prediction_timestamps: List of future timestamps for predictions
            width: Chart width
            height: Chart height
            
        Returns:
            Plotly figure
        """
        if df is None or len(df) == 0:
            logger.error("Empty dataframe provided")
            return go.Figure()
        
        # Set default active indicators if not provided
        if active_indicators is None:
            active_indicators = ["rsi", "macd", "bollinger"]
        
        # Calculate number of rows based on active indicators
        num_rows = 2  # Price and volume
        
        if "rsi" in active_indicators and "rsi" in self.indicators:
            num_rows += 1
        
        if "macd" in active_indicators and "macd" in self.indicators:
            num_rows += 1
        
        # Create subplots
        fig = make_subplots(
            rows=num_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5] + [0.5 / (num_rows - 1)] * (num_rows - 1),
            subplot_titles=["Price", "Volume"] + 
                          (["RSI"] if "rsi" in active_indicators and "rsi" in self.indicators else []) +
                          (["MACD"] if "macd" in active_indicators and "macd" in self.indicators else [])
        )
        
        # Calculate indicators
        df_with_indicators = df.copy()
        
        for indicator_id in active_indicators:
            if indicator_id in self.indicators:
                df_with_indicators = self.indicators[indicator_id].calculate(df_with_indicators)
        
        # Plot candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price",
                increasing_line_color="#26A69A",
                decreasing_line_color="#EF5350"
            ),
            row=1,
            col=1
        )
        
        # Plot Bollinger Bands if active
        if "bollinger" in active_indicators and "bollinger" in self.indicators:
            fig = self.indicators["bollinger"].plot(fig, 1, 1, df_with_indicators)
        
        # Plot Volume Profile if active
        if "volume_profile" in active_indicators and "volume_profile" in self.indicators:
            fig = self.indicators["volume_profile"].plot(fig, 1, 1, df)
        
        # Plot patterns if provided
        if patterns:
            fig = self.pattern_recognition.plot_patterns(fig, 1, 1, patterns)
        
        # Plot trading signals if provided
        if signals:
            fig = self.trading_signals.plot_signals(fig, 1, 1, signals)
        
        # Plot predictions if provided
        if predictions and prediction_timestamps:
            fig = self.prediction_viz.plot_predictions(fig, 1, 1, df, predictions, prediction_timestamps)
        
        # Plot volume
        colors = ["#26A69A" if row['close'] >= row['open'] else "#EF5350" for _, row in df.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name="Volume",
                marker=dict(color=colors),
                opacity=0.7
            ),
            row=2,
            col=1
        )
        
        # Plot RSI if active
        current_row = 3
        if "rsi" in active_indicators and "rsi" in self.indicators:
            fig = self.indicators["rsi"].plot(fig, current_row, 1, df_with_indicators)
            current_row += 1
        
        # Plot MACD if active
        if "macd" in active_indicators and "macd" in self.indicators:
            fig = self.indicators["macd"].plot(fig, current_row, 1, df_with_indicators)
        
        # Set chart title
        if title:
            fig.update_layout(title=title)
        
        # Set theme
        if self.dark_mode:
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#1E1E1E",
                plot_bgcolor="#1E1E1E"
            )
        
        # Update layout
        fig.update_layout(
            width=width,
            height=height,
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_rangeslider_visible=False
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        if "rsi" in active_indicators and "rsi" in self.indicators:
            fig.update_yaxes(title_text="RSI", row=3, col=1)
        
        if "macd" in active_indicators and "macd" in self.indicators:
            fig.update_yaxes(title_text="MACD", row=current_row, col=1)
        
        return fig
    
    def save_chart(self, fig: go.Figure, filename: str):
        """Save chart to file
        
        Args:
            fig: Plotly figure
            filename: Output filename
        """
        fig.write_html(filename)
        logger.info(f"Chart saved to {filename}")
    
    def get_chart_html(self, fig: go.Figure) -> str:
        """Get chart HTML
        
        Args:
            fig: Plotly figure
            
        Returns:
            Chart HTML
        """
        return fig.to_html(include_plotlyjs='cdn', full_html=False)

if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Get sample data
    btc_data = yf.download("BTC-USD", period="1mo", interval="1d")
    
    # Create chart component
    chart = ChartComponent(dark_mode=True)
    
    # Create chart
    fig = chart.create_chart(
        btc_data,
        title="BTC/USD Daily Chart",
        active_indicators=["rsi", "macd", "bollinger", "volume_profile"]
    )
    
    # Save chart
    chart.save_chart(fig, "btc_chart.html")
