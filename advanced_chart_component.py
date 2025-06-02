#!/usr/bin/env python
"""
Advanced Chart Component for Trading-Agent System

This module provides an advanced chart component with technical indicators,
pattern visualization, and multi-timeframe support for BTC, ETH, and SOL.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("advanced_chart_component")

class TechnicalIndicator:
    """Base class for technical indicators"""
    
    def __init__(self, name, params=None):
        """Initialize technical indicator
        
        Args:
            name: Indicator name
            params: Indicator parameters
        """
        self.name = name
        self.params = params or {}
        
    def calculate(self, data):
        """Calculate indicator values
        
        Args:
            data: Price data
            
        Returns:
            dict: Indicator values
        """
        raise NotImplementedError("Subclasses must implement calculate()")
    
    def get_config(self):
        """Get indicator configuration for chart
        
        Returns:
            dict: Indicator configuration
        """
        raise NotImplementedError("Subclasses must implement get_config()")

class RSIIndicator(TechnicalIndicator):
    """Relative Strength Index indicator"""
    
    def __init__(self, params=None):
        """Initialize RSI indicator
        
        Args:
            params: Indicator parameters
        """
        default_params = {
            "period": 14,
            "overbought": 70,
            "oversold": 30
        }
        
        # Merge default params with provided params
        merged_params = {**default_params, **(params or {})}
        
        super().__init__("RSI", merged_params)
    
    def calculate(self, data):
        """Calculate RSI values
        
        Args:
            data: Price data with 'close' column
            
        Returns:
            dict: RSI values
        """
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list) and len(data) > 0 and 'close' in data[0]:
                # Convert list of dicts to DataFrame
                data = pd.DataFrame(data)
            else:
                raise ValueError("Data must be a DataFrame or list of dicts with 'close' column")
        
        period = self.params["period"]
        
        # Calculate price changes
        delta = data['close'].diff()
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Handle NaN values
        rsi = rsi.fillna(50)
        
        return {
            "name": self.name,
            "values": rsi.tolist(),
            "overbought": self.params["overbought"],
            "oversold": self.params["oversold"]
        }
    
    def get_config(self):
        """Get RSI configuration for chart
        
        Returns:
            dict: RSI configuration
        """
        return {
            "name": self.name,
            "type": "line",
            "position": "separate",
            "height": 120,
            "precision": 2,
            "overbought": self.params["overbought"],
            "oversold": self.params["oversold"],
            "colors": {
                "line": "#8A2BE2",
                "overbought": "rgba(255, 0, 0, 0.3)",
                "oversold": "rgba(0, 255, 0, 0.3)"
            }
        }

class MACDIndicator(TechnicalIndicator):
    """Moving Average Convergence Divergence indicator"""
    
    def __init__(self, params=None):
        """Initialize MACD indicator
        
        Args:
            params: Indicator parameters
        """
        default_params = {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9
        }
        
        # Merge default params with provided params
        merged_params = {**default_params, **(params or {})}
        
        super().__init__("MACD", merged_params)
    
    def calculate(self, data):
        """Calculate MACD values
        
        Args:
            data: Price data with 'close' column
            
        Returns:
            dict: MACD values
        """
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list) and len(data) > 0 and 'close' in data[0]:
                # Convert list of dicts to DataFrame
                data = pd.DataFrame(data)
            else:
                raise ValueError("Data must be a DataFrame or list of dicts with 'close' column")
        
        fast_period = self.params["fast_period"]
        slow_period = self.params["slow_period"]
        signal_period = self.params["signal_period"]
        
        # Calculate EMAs
        fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line and signal line
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            "name": self.name,
            "macd": macd_line.tolist(),
            "signal": signal_line.tolist(),
            "histogram": histogram.tolist()
        }
    
    def get_config(self):
        """Get MACD configuration for chart
        
        Returns:
            dict: MACD configuration
        """
        return {
            "name": self.name,
            "type": "macd",
            "position": "separate",
            "height": 150,
            "precision": 2,
            "colors": {
                "macd": "#2196F3",
                "signal": "#FF9800",
                "histogram": {
                    "positive": "rgba(0, 255, 0, 0.7)",
                    "negative": "rgba(255, 0, 0, 0.7)"
                }
            }
        }

class BollingerBandsIndicator(TechnicalIndicator):
    """Bollinger Bands indicator"""
    
    def __init__(self, params=None):
        """Initialize Bollinger Bands indicator
        
        Args:
            params: Indicator parameters
        """
        default_params = {
            "period": 20,
            "std_dev": 2
        }
        
        # Merge default params with provided params
        merged_params = {**default_params, **(params or {})}
        
        super().__init__("BollingerBands", merged_params)
    
    def calculate(self, data):
        """Calculate Bollinger Bands values
        
        Args:
            data: Price data with 'close' column
            
        Returns:
            dict: Bollinger Bands values
        """
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list) and len(data) > 0 and 'close' in data[0]:
                # Convert list of dicts to DataFrame
                data = pd.DataFrame(data)
            else:
                raise ValueError("Data must be a DataFrame or list of dicts with 'close' column")
        
        period = self.params["period"]
        std_dev = self.params["std_dev"]
        
        # Calculate middle band (SMA)
        middle_band = data['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        std = data['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        # Handle NaN values
        middle_band = middle_band.fillna(data['close'])
        upper_band = upper_band.fillna(data['close'] * 1.05)
        lower_band = lower_band.fillna(data['close'] * 0.95)
        
        return {
            "name": self.name,
            "middle": middle_band.tolist(),
            "upper": upper_band.tolist(),
            "lower": lower_band.tolist()
        }
    
    def get_config(self):
        """Get Bollinger Bands configuration for chart
        
        Returns:
            dict: Bollinger Bands configuration
        """
        return {
            "name": self.name,
            "type": "bands",
            "position": "overlay",
            "precision": 2,
            "colors": {
                "middle": "#FF9800",
                "upper": "rgba(76, 175, 80, 0.7)",
                "lower": "rgba(76, 175, 80, 0.7)",
                "fill": "rgba(76, 175, 80, 0.1)"
            }
        }

class VolumeIndicator(TechnicalIndicator):
    """Volume indicator"""
    
    def __init__(self, params=None):
        """Initialize Volume indicator
        
        Args:
            params: Indicator parameters
        """
        default_params = {
            "ma_period": 20
        }
        
        # Merge default params with provided params
        merged_params = {**default_params, **(params or {})}
        
        super().__init__("Volume", merged_params)
    
    def calculate(self, data):
        """Calculate Volume values
        
        Args:
            data: Price data with 'volume' column
            
        Returns:
            dict: Volume values
        """
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list) and len(data) > 0 and 'volume' in data[0]:
                # Convert list of dicts to DataFrame
                data = pd.DataFrame(data)
            else:
                raise ValueError("Data must be a DataFrame or list of dicts with 'volume' column")
        
        ma_period = self.params["ma_period"]
        
        # Calculate volume MA
        volume_ma = data['volume'].rolling(window=ma_period).mean()
        
        # Handle NaN values
        volume_ma = volume_ma.fillna(data['volume'])
        
        return {
            "name": self.name,
            "values": data['volume'].tolist(),
            "ma": volume_ma.tolist()
        }
    
    def get_config(self):
        """Get Volume configuration for chart
        
        Returns:
            dict: Volume configuration
        """
        return {
            "name": self.name,
            "type": "histogram",
            "position": "separate",
            "height": 100,
            "precision": 0,
            "colors": {
                "up": "rgba(76, 175, 80, 0.7)",
                "down": "rgba(255, 0, 0, 0.7)",
                "ma": "#FF9800"
            }
        }

class PatternRecognition:
    """Pattern recognition visualization"""
    
    def __init__(self):
        """Initialize pattern recognition"""
        self.patterns = []
    
    def add_pattern(self, pattern):
        """Add pattern
        
        Args:
            pattern: Pattern data
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        try:
            self.patterns.append(pattern)
            return True
        except Exception as e:
            logger.error(f"Error adding pattern: {str(e)}")
            return False
    
    def clear_patterns(self):
        """Clear patterns
        
        Returns:
            bool: True if cleared successfully, False otherwise
        """
        try:
            self.patterns = []
            return True
        except Exception as e:
            logger.error(f"Error clearing patterns: {str(e)}")
            return False
    
    def get_patterns(self):
        """Get patterns
        
        Returns:
            list: Patterns
        """
        return self.patterns
    
    def get_visualization_data(self):
        """Get visualization data for patterns
        
        Returns:
            list: Visualization data
        """
        visualization_data = []
        
        for pattern in self.patterns:
            # Extract pattern data
            pattern_type = pattern.get("type", "unknown")
            start_time = pattern.get("start_time")
            end_time = pattern.get("end_time")
            confidence = pattern.get("confidence", 0.5)
            
            # Skip if missing required data
            if not start_time or not end_time:
                continue
            
            # Create visualization data
            viz_data = {
                "type": pattern_type,
                "start_time": start_time,
                "end_time": end_time,
                "confidence": confidence,
                "color": self._get_pattern_color(pattern_type),
                "label": self._get_pattern_label(pattern_type)
            }
            
            visualization_data.append(viz_data)
        
        return visualization_data
    
    def _get_pattern_color(self, pattern_type):
        """Get color for pattern type
        
        Args:
            pattern_type: Pattern type
            
        Returns:
            str: Color
        """
        # Define colors for different pattern types
        pattern_colors = {
            "double_top": "#FF5252",
            "double_bottom": "#4CAF50",
            "head_and_shoulders": "#FF9800",
            "inverse_head_and_shoulders": "#2196F3",
            "triangle": "#9C27B0",
            "wedge": "#3F51B5",
            "flag": "#00BCD4",
            "pennant": "#009688",
            "rectangle": "#8BC34A",
            "cup_and_handle": "#CDDC39",
            "rounding_bottom": "#FFC107",
            "rounding_top": "#FF5722"
        }
        
        return pattern_colors.get(pattern_type.lower(), "#757575")
    
    def _get_pattern_label(self, pattern_type):
        """Get label for pattern type
        
        Args:
            pattern_type: Pattern type
            
        Returns:
            str: Label
        """
        # Convert snake_case to Title Case with spaces
        label = pattern_type.replace("_", " ").title()
        return label

class SignalVisualization:
    """Trading signal visualization"""
    
    def __init__(self):
        """Initialize signal visualization"""
        self.signals = []
    
    def add_signal(self, signal):
        """Add signal
        
        Args:
            signal: Signal data
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        try:
            self.signals.append(signal)
            return True
        except Exception as e:
            logger.error(f"Error adding signal: {str(e)}")
            return False
    
    def clear_signals(self):
        """Clear signals
        
        Returns:
            bool: True if cleared successfully, False otherwise
        """
        try:
            self.signals = []
            return True
        except Exception as e:
            logger.error(f"Error clearing signals: {str(e)}")
            return False
    
    def get_signals(self):
        """Get signals
        
        Returns:
            list: Signals
        """
        return self.signals
    
    def get_visualization_data(self):
        """Get visualization data for signals
        
        Returns:
            list: Visualization data
        """
        visualization_data = []
        
        for signal in self.signals:
            # Extract signal data
            signal_type = signal.get("type", "unknown")
            timestamp = signal.get("timestamp")
            direction = signal.get("direction", "neutral")
            confidence = signal.get("confidence", 0.5)
            
            # Skip if missing required data
            if not timestamp:
                continue
            
            # Create visualization data
            viz_data = {
                "type": signal_type,
                "timestamp": timestamp,
                "direction": direction,
                "confidence": confidence,
                "color": self._get_signal_color(direction),
                "shape": self._get_signal_shape(signal_type),
                "label": self._get_signal_label(signal_type, direction)
            }
            
            visualization_data.append(viz_data)
        
        return visualization_data
    
    def _get_signal_color(self, direction):
        """Get color for signal direction
        
        Args:
            direction: Signal direction
            
        Returns:
            str: Color
        """
        # Define colors for different directions
        direction_colors = {
            "buy": "#4CAF50",
            "sell": "#FF5252",
            "neutral": "#FFC107"
        }
        
        return direction_colors.get(direction.lower(), "#757575")
    
    def _get_signal_shape(self, signal_type):
        """Get shape for signal type
        
        Args:
            signal_type: Signal type
            
        Returns:
            str: Shape
        """
        # Define shapes for different signal types
        signal_shapes = {
            "pattern": "triangle",
            "order_book": "circle",
            "technical": "square",
            "sentiment": "diamond"
        }
        
        return signal_shapes.get(signal_type.lower(), "circle")
    
    def _get_signal_label(self, signal_type, direction):
        """Get label for signal
        
        Args:
            signal_type: Signal type
            direction: Signal direction
            
        Returns:
            str: Label
        """
        # Convert snake_case to Title Case with spaces
        type_label = signal_type.replace("_", " ").title()
        direction_label = direction.title()
        
        return f"{type_label} ({direction_label})"

class PredictionVisualization:
    """Price prediction visualization"""
    
    def __init__(self):
        """Initialize prediction visualization"""
        self.predictions = []
    
    def add_prediction(self, prediction):
        """Add prediction
        
        Args:
            prediction: Prediction data
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        try:
            self.predictions.append(prediction)
            return True
        except Exception as e:
            logger.error(f"Error adding prediction: {str(e)}")
            return False
    
    def clear_predictions(self):
        """Clear predictions
        
        Returns:
            bool: True if cleared successfully, False otherwise
        """
        try:
            self.predictions = []
            return True
        except Exception as e:
            logger.error(f"Error clearing predictions: {str(e)}")
            return False
    
    def get_predictions(self):
        """Get predictions
        
        Returns:
            list: Predictions
        """
        return self.predictions
    
    def get_visualization_data(self):
        """Get visualization data for predictions
        
        Returns:
            list: Visualization data
        """
        visualization_data = []
        
        for prediction in self.predictions:
            # Extract prediction data
            start_time = prediction.get("start_time")
            end_time = prediction.get("end_time")
            values = prediction.get("values", [])
            confidence = prediction.get("confidence", 0.5)
            direction = prediction.get("direction", "neutral")
            
            # Skip if missing required data
            if not start_time or not end_time or not values:
                continue
            
            # Create visualization data
            viz_data = {
                "start_time": start_time,
                "end_time": end_time,
                "values": values,
                "confidence": confidence,
                "direction": direction,
                "color": self._get_prediction_color(direction, confidence),
                "label": self._get_prediction_label(direction, confidence)
            }
            
            visualization_data.append(viz_data)
        
        return visualization_data
    
    def _get_prediction_color(self, direction, confidence):
        """Get color for prediction
        
        Args:
            direction: Prediction direction
            confidence: Prediction confidence
            
        Returns:
            str: Color
        """
        # Define base colors for different directions
        direction_colors = {
            "up": "#4CAF50",
            "down": "#FF5252",
            "neutral": "#FFC107"
        }
        
        # Get base color
        base_color = direction_colors.get(direction.lower(), "#757575")
        
        # Adjust opacity based on confidence
        opacity = max(0.3, min(0.9, confidence))
        
        # Convert hex to rgba
        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)
        
        return f"rgba({r}, {g}, {b}, {opacity})"
    
    def _get_prediction_label(self, direction, confidence):
        """Get label for prediction
        
        Args:
            direction: Prediction direction
            confidence: Prediction confidence
            
        Returns:
            str: Label
        """
        direction_label = direction.title()
        confidence_pct = int(confidence * 100)
        
        return f"{direction_label} ({confidence_pct}%)"

class AdvancedChartComponent:
    """Advanced chart component with technical indicators and pattern visualization"""
    
    def __init__(self, data_service=None):
        """Initialize advanced chart component
        
        Args:
            data_service: Data service
        """
        self.data_service = data_service
        self.indicators = {}
        self.pattern_recognition = PatternRecognition()
        self.signal_visualization = SignalVisualization()
        self.prediction_visualization = PredictionVisualization()
        
        # Available indicators
        self.available_indicators = {
            "RSI": RSIIndicator,
            "MACD": MACDIndicator,
            "BollingerBands": BollingerBandsIndicator,
            "Volume": VolumeIndicator
        }
        
        logger.info("Initialized AdvancedChartComponent")
    
    def add_indicator(self, name, params=None):
        """Add technical indicator
        
        Args:
            name: Indicator name
            params: Indicator parameters
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        try:
            if name in self.available_indicators:
                indicator_class = self.available_indicators[name]
                self.indicators[name] = indicator_class(params)
                logger.info(f"Added indicator: {name}")
                return True
            else:
                logger.warning(f"Indicator not available: {name}")
                return False
        except Exception as e:
            logger.error(f"Error adding indicator: {str(e)}")
            return False
    
    def remove_indicator(self, name):
        """Remove technical indicator
        
        Args:
            name: Indicator name
            
        Returns:
            bool: True if removed successfully, False otherwise
        """
        try:
            if name in self.indicators:
                del self.indicators[name]
                logger.info(f"Removed indicator: {name}")
                return True
            else:
                logger.warning(f"Indicator not found: {name}")
                return False
        except Exception as e:
            logger.error(f"Error removing indicator: {str(e)}")
            return False
    
    def get_indicators(self):
        """Get technical indicators
        
        Returns:
            dict: Technical indicators
        """
        return self.indicators
    
    def get_available_indicators(self):
        """Get available indicators
        
        Returns:
            list: Available indicators
        """
        return list(self.available_indicators.keys())
    
    def calculate_indicators(self, data):
        """Calculate indicator values
        
        Args:
            data: Price data
            
        Returns:
            dict: Indicator values
        """
        indicator_values = {}
        
        for name, indicator in self.indicators.items():
            try:
                indicator_values[name] = indicator.calculate(data)
            except Exception as e:
                logger.error(f"Error calculating indicator {name}: {str(e)}")
                indicator_values[name] = {"name": name, "error": str(e)}
        
        return indicator_values
    
    def get_indicator_configs(self):
        """Get indicator configurations
        
        Returns:
            list: Indicator configurations
        """
        configs = []
        
        for name, indicator in self.indicators.items():
            try:
                configs.append(indicator.get_config())
            except Exception as e:
                logger.error(f"Error getting config for indicator {name}: {str(e)}")
        
        return configs
    
    def add_pattern(self, pattern):
        """Add pattern
        
        Args:
            pattern: Pattern data
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        return self.pattern_recognition.add_pattern(pattern)
    
    def clear_patterns(self):
        """Clear patterns
        
        Returns:
            bool: True if cleared successfully, False otherwise
        """
        return self.pattern_recognition.clear_patterns()
    
    def get_patterns(self):
        """Get patterns
        
        Returns:
            list: Patterns
        """
        return self.pattern_recognition.get_patterns()
    
    def get_pattern_visualization_data(self):
        """Get visualization data for patterns
        
        Returns:
            list: Visualization data
        """
        return self.pattern_recognition.get_visualization_data()
    
    def add_signal(self, signal):
        """Add signal
        
        Args:
            signal: Signal data
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        return self.signal_visualization.add_signal(signal)
    
    def clear_signals(self):
        """Clear signals
        
        Returns:
            bool: True if cleared successfully, False otherwise
        """
        return self.signal_visualization.clear_signals()
    
    def get_signals(self):
        """Get signals
        
        Returns:
            list: Signals
        """
        return self.signal_visualization.get_signals()
    
    def get_signal_visualization_data(self):
        """Get visualization data for signals
        
        Returns:
            list: Visualization data
        """
        return self.signal_visualization.get_visualization_data()
    
    def add_prediction(self, prediction):
        """Add prediction
        
        Args:
            prediction: Prediction data
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        return self.prediction_visualization.add_prediction(prediction)
    
    def clear_predictions(self):
        """Clear predictions
        
        Returns:
            bool: True if cleared successfully, False otherwise
        """
        return self.prediction_visualization.clear_predictions()
    
    def get_predictions(self):
        """Get predictions
        
        Returns:
            list: Predictions
        """
        return self.prediction_visualization.get_predictions()
    
    def get_prediction_visualization_data(self):
        """Get visualization data for predictions
        
        Returns:
            list: Visualization data
        """
        return self.prediction_visualization.get_visualization_data()
    
    def get_chart_data(self, asset=None, interval="1m", limit=100):
        """Get chart data
        
        Args:
            asset: Asset to get data for (default: current asset)
            interval: Kline interval
            limit: Number of klines
            
        Returns:
            dict: Chart data
        """
        if not self.data_service:
            logger.warning("Data service not available")
            return None
        
        try:
            # Get klines
            klines = self.data_service.get_klines(asset, interval, limit)
            
            # Convert to DataFrame for indicator calculation
            df = pd.DataFrame(klines)
            
            # Calculate indicators
            indicator_values = self.calculate_indicators(df)
            
            # Get patterns
            patterns = self.get_pattern_visualization_data()
            
            # Get signals
            signals = self.get_signal_visualization_data()
            
            # Get predictions
            predictions = self.get_prediction_visualization_data()
            
            # Create chart data
            chart_data = {
                "klines": klines,
                "indicators": indicator_values,
                "patterns": patterns,
                "signals": signals,
                "predictions": predictions
            }
            
            return chart_data
        
        except Exception as e:
            logger.error(f"Error getting chart data: {str(e)}")
            return None
    
    def get_chart_config(self):
        """Get chart configuration
        
        Returns:
            dict: Chart configuration
        """
        # Get indicator configs
        indicator_configs = self.get_indicator_configs()
        
        # Create chart config
        chart_config = {
            "chart": {
                "type": "candlestick",
                "height": 500,
                "width": "100%",
                "background": "#121826",
                "textColor": "#e6e9f0",
                "grid": {
                    "vertLines": {"color": "#2d3748"},
                    "horzLines": {"color": "#2d3748"}
                },
                "crosshair": {
                    "mode": "normal",
                    "vertLine": {"color": "#4299e1", "width": 1, "style": "dashed"},
                    "horzLine": {"color": "#4299e1", "width": 1, "style": "dashed"}
                },
                "timeScale": {
                    "borderColor": "#2d3748",
                    "timeVisible": True,
                    "secondsVisible": False
                },
                "rightPriceScale": {
                    "borderColor": "#2d3748"
                }
            },
            "series": {
                "candlestick": {
                    "upColor": "#48bb78",
                    "downColor": "#e53e3e",
                    "borderUpColor": "#48bb78",
                    "borderDownColor": "#e53e3e",
                    "wickUpColor": "#48bb78",
                    "wickDownColor": "#e53e3e"
                }
            },
            "indicators": indicator_configs,
            "patterns": {
                "enabled": True,
                "opacity": 0.3,
                "lineWidth": 2
            },
            "signals": {
                "enabled": True,
                "size": 12,
                "opacity": 0.8
            },
            "predictions": {
                "enabled": True,
                "opacity": 0.3,
                "lineWidth": 2
            }
        }
        
        return chart_config
    
    def get_chart_html(self, container_id="chart-container"):
        """Get HTML for chart
        
        Args:
            container_id: Container ID
            
        Returns:
            str: HTML
        """
        # Get chart config
        chart_config = self.get_chart_config()
        
        # Convert to JSON
        chart_config_json = json.dumps(chart_config)
        
        # Create HTML
        html = f"""
        <div id="{container_id}" style="width: 100%; height: 500px;"></div>
        <script>
            // Initialize chart
            function initChart() {{
                const chartContainer = document.getElementById('{container_id}');
                const chartConfig = {chart_config_json};
                
                // Create chart
                const chart = LightweightCharts.createChart(chartContainer, chartConfig.chart);
                
                // Add candlestick series
                const candleSeries = chart.addCandlestickSeries(chartConfig.series.candlestick);
                
                // Add indicators
                const indicatorSeries = {{}};
                for (const config of chartConfig.indicators) {{
                    if (config.position === 'overlay') {{
                        // Add overlay indicator
                        if (config.type === 'bands') {{
                            // Add Bollinger Bands
                            indicatorSeries[config.name] = {{
                                middle: chart.addLineSeries({{
                                    color: config.colors.middle,
                                    lineWidth: 1,
                                    priceLineVisible: false
                                }}),
                                upper: chart.addLineSeries({{
                                    color: config.colors.upper,
                                    lineWidth: 1,
                                    priceLineVisible: false
                                }}),
                                lower: chart.addLineSeries({{
                                    color: config.colors.lower,
                                    lineWidth: 1,
                                    priceLineVisible: false
                                }})
                            }};
                        }}
                    }} else {{
                        // Add separate indicator
                        const indicatorPane = chart.addPane({{
                            height: config.height
                        }});
                        
                        if (config.type === 'line') {{
                            // Add line indicator (e.g., RSI)
                            indicatorSeries[config.name] = {{
                                line: indicatorPane.addLineSeries({{
                                    color: config.colors.line,
                                    lineWidth: 2,
                                    priceLineVisible: false
                                }})
                            }};
                            
                            // Add overbought/oversold lines
                            if (config.overbought && config.oversold) {{
                                indicatorSeries[config.name].overbought = indicatorPane.addLineSeries({{
                                    color: config.colors.overbought,
                                    lineWidth: 1,
                                    priceLineVisible: false
                                }});
                                
                                indicatorSeries[config.name].oversold = indicatorPane.addLineSeries({{
                                    color: config.colors.oversold,
                                    lineWidth: 1,
                                    priceLineVisible: false
                                }});
                            }}
                        }} else if (config.type === 'macd') {{
                            // Add MACD indicator
                            indicatorSeries[config.name] = {{
                                macd: indicatorPane.addLineSeries({{
                                    color: config.colors.macd,
                                    lineWidth: 2,
                                    priceLineVisible: false
                                }}),
                                signal: indicatorPane.addLineSeries({{
                                    color: config.colors.signal,
                                    lineWidth: 1,
                                    priceLineVisible: false
                                }}),
                                histogram: indicatorPane.addHistogramSeries({{
                                    color: config.colors.histogram.positive,
                                    priceFormat: {{
                                        type: 'price',
                                        precision: config.precision,
                                        minMove: 0.01
                                    }}
                                }})
                            }};
                        }} else if (config.type === 'histogram') {{
                            // Add histogram indicator (e.g., Volume)
                            indicatorSeries[config.name] = {{
                                histogram: indicatorPane.addHistogramSeries({{
                                    color: config.colors.up,
                                    priceFormat: {{
                                        type: 'volume',
                                        precision: config.precision
                                    }}
                                }}),
                                ma: indicatorPane.addLineSeries({{
                                    color: config.colors.ma,
                                    lineWidth: 2,
                                    priceLineVisible: false
                                }})
                            }};
                        }}
                    }}
                }}
                
                // Fetch data
                fetch('/api/chart-data')
                    .then(response => response.json())
                    .then(data => {{
                        if (data && data.klines) {{
                            // Set candlestick data
                            const candleData = data.klines.map(kline => ({{
                                time: Math.floor(kline.time / 1000),
                                open: kline.open,
                                high: kline.high,
                                low: kline.low,
                                close: kline.close
                            }}));
                            
                            candleSeries.setData(candleData);
                            
                            // Set indicator data
                            for (const [name, values] of Object.entries(data.indicators)) {{
                                if (name === 'BollingerBands' && indicatorSeries[name]) {{
                                    // Set Bollinger Bands data
                                    const middleData = values.middle.map((value, i) => ({{
                                        time: Math.floor(data.klines[i].time / 1000),
                                        value: value
                                    }}));
                                    
                                    const upperData = values.upper.map((value, i) => ({{
                                        time: Math.floor(data.klines[i].time / 1000),
                                        value: value
                                    }}));
                                    
                                    const lowerData = values.lower.map((value, i) => ({{
                                        time: Math.floor(data.klines[i].time / 1000),
                                        value: value
                                    }}));
                                    
                                    indicatorSeries[name].middle.setData(middleData);
                                    indicatorSeries[name].upper.setData(upperData);
                                    indicatorSeries[name].lower.setData(lowerData);
                                }} else if (name === 'RSI' && indicatorSeries[name]) {{
                                    // Set RSI data
                                    const lineData = values.values.map((value, i) => ({{
                                        time: Math.floor(data.klines[i].time / 1000),
                                        value: value
                                    }}));
                                    
                                    indicatorSeries[name].line.setData(lineData);
                                    
                                    // Set overbought/oversold lines
                                    const overboughtData = data.klines.map(kline => ({{
                                        time: Math.floor(kline.time / 1000),
                                        value: values.overbought
                                    }}));
                                    
                                    const oversoldData = data.klines.map(kline => ({{
                                        time: Math.floor(kline.time / 1000),
                                        value: values.oversold
                                    }}));
                                    
                                    indicatorSeries[name].overbought.setData(overboughtData);
                                    indicatorSeries[name].oversold.setData(oversoldData);
                                }} else if (name === 'MACD' && indicatorSeries[name]) {{
                                    // Set MACD data
                                    const macdData = values.macd.map((value, i) => ({{
                                        time: Math.floor(data.klines[i].time / 1000),
                                        value: value
                                    }}));
                                    
                                    const signalData = values.signal.map((value, i) => ({{
                                        time: Math.floor(data.klines[i].time / 1000),
                                        value: value
                                    }}));
                                    
                                    const histogramData = values.histogram.map((value, i) => ({{
                                        time: Math.floor(data.klines[i].time / 1000),
                                        value: value,
                                        color: value >= 0 ? chartConfig.indicators.find(i => i.name === 'MACD').colors.histogram.positive : chartConfig.indicators.find(i => i.name === 'MACD').colors.histogram.negative
                                    }}));
                                    
                                    indicatorSeries[name].macd.setData(macdData);
                                    indicatorSeries[name].signal.setData(signalData);
                                    indicatorSeries[name].histogram.setData(histogramData);
                                }} else if (name === 'Volume' && indicatorSeries[name]) {{
                                    // Set Volume data
                                    const histogramData = values.values.map((value, i) => ({{
                                        time: Math.floor(data.klines[i].time / 1000),
                                        value: value,
                                        color: data.klines[i].close >= data.klines[i].open ? chartConfig.indicators.find(i => i.name === 'Volume').colors.up : chartConfig.indicators.find(i => i.name === 'Volume').colors.down
                                    }}));
                                    
                                    const maData = values.ma.map((value, i) => ({{
                                        time: Math.floor(data.klines[i].time / 1000),
                                        value: value
                                    }}));
                                    
                                    indicatorSeries[name].histogram.setData(histogramData);
                                    indicatorSeries[name].ma.setData(maData);
                                }}
                            }}
                            
                            // Add patterns
                            if (chartConfig.patterns.enabled && data.patterns) {{
                                for (const pattern of data.patterns) {{
                                    // Add pattern visualization
                                    // Implementation depends on pattern type
                                }}
                            }}
                            
                            // Add signals
                            if (chartConfig.signals.enabled && data.signals) {{
                                for (const signal of data.signals) {{
                                    // Add signal marker
                                    // Implementation depends on signal type
                                }}
                            }}
                            
                            // Add predictions
                            if (chartConfig.predictions.enabled && data.predictions) {{
                                for (const prediction of data.predictions) {{
                                    // Add prediction visualization
                                    // Implementation depends on prediction type
                                }}
                            }}
                        }}
                    }})
                    .catch(error => {{
                        console.error('Error fetching chart data:', error);
                    }});
                
                // Resize chart on window resize
                window.addEventListener('resize', () => {{
                    chart.applyOptions({{
                        width: chartContainer.clientWidth,
                        height: chartContainer.clientHeight
                    }});
                }});
                
                return {{
                    chart: chart,
                    candleSeries: candleSeries,
                    indicatorSeries: indicatorSeries
                }};
            }}
            
            // Initialize chart when DOM is loaded
            document.addEventListener('DOMContentLoaded', initChart);
        </script>
        """
        
        return html

# Example usage
if __name__ == "__main__":
    # Import data service
    from multi_asset_data_service import MultiAssetDataService
    
    # Initialize data service
    data_service = MultiAssetDataService()
    
    # Initialize chart component
    chart_component = AdvancedChartComponent(data_service)
    
    # Add indicators
    chart_component.add_indicator("RSI")
    chart_component.add_indicator("MACD")
    chart_component.add_indicator("BollingerBands")
    chart_component.add_indicator("Volume")
    
    # Get chart HTML
    chart_html = chart_component.get_chart_html()
    
    # Save to file
    with open("advanced_chart.html", "w") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Advanced Chart</title>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
            <style>
                :root {{
                    --bg-primary: #121826;
                    --bg-secondary: #1a2332;
                    --bg-tertiary: #232f42;
                    --text-primary: #e6e9f0;
                    --text-secondary: #a0aec0;
                    --accent-primary: #3182ce;
                    --accent-secondary: #4299e1;
                    --success: #48bb78;
                    --danger: #e53e3e;
                    --warning: #ecc94b;
                    --border-color: #2d3748;
                }}
                
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'Inter', sans-serif;
                    background-color: var(--bg-primary);
                    color: var(--text-primary);
                    line-height: 1.5;
                    padding: 20px;
                }}
                
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                
                header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 1rem 0;
                    border-bottom: 1px solid var(--border-color);
                    margin-bottom: 1rem;
                }}
                
                .logo {{
                    font-size: 1.5rem;
                    font-weight: 700;
                    color: var(--text-primary);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <div class="logo">Advanced Chart</div>
                </header>
                
                {chart_html}
            </div>
        </body>
        </html>
        """)
    
    print("Advanced chart HTML saved to advanced_chart.html")
