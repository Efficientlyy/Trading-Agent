#!/usr/bin/env python
"""
Technical Indicators Module

This module provides implementations of various technical indicators used for
market analysis and signal generation in the Trading-Agent system.

Indicators include:
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- Volume Weighted Average Price (VWAP)
- Exponential Moving Average (EMA)
- Simple Moving Average (SMA)
- Average True Range (ATR)

Each indicator is implemented with configurable parameters and robust error handling.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Union, Tuple, Optional
from error_handling_utils import handle_api_error, log_exception, safe_get

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("indicators")

class TechnicalIndicators:
    """
    Class containing implementations of various technical indicators.
    All methods are static to allow for easy use without instantiation.
    """
    
    @staticmethod
    @handle_api_error
    def calculate_sma(prices: np.ndarray, period: int = 20) -> Optional[float]:
        """
        Calculate Simple Moving Average
        
        Args:
            prices: Array of price data
            period: Period for SMA calculation
            
        Returns:
            SMA value or None if calculation fails
        """
        try:
            if len(prices) < period:
                logger.warning(f"Not enough data for SMA calculation. Need {period}, got {len(prices)}")
                return None
                
            return np.mean(prices[-period:])
        except Exception as e:
            log_exception(e, "calculate_sma")
            return None
    
    @staticmethod
    @handle_api_error
    def calculate_ema(prices: np.ndarray, period: int = 20, smoothing: float = 2.0) -> Optional[float]:
        """
        Calculate Exponential Moving Average
        
        Args:
            prices: Array of price data
            period: Period for EMA calculation
            smoothing: Smoothing factor (typically 2 for standard EMA)
            
        Returns:
            EMA value or None if calculation fails
        """
        try:
            if len(prices) < period:
                logger.warning(f"Not enough data for EMA calculation. Need {period}, got {len(prices)}")
                return None
                
            # Calculate multiplier
            multiplier = smoothing / (period + 1)
            
            # Calculate initial SMA
            sma = np.mean(prices[:period])
            
            # Calculate EMA
            ema = sma
            for price in prices[period:]:
                ema = (price - ema) * multiplier + ema
                
            return ema
        except Exception as e:
            log_exception(e, "calculate_ema")
            return None
    
    @staticmethod
    @handle_api_error
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> Optional[float]:
        """
        Calculate Relative Strength Index
        
        Args:
            prices: Array of price data
            period: Period for RSI calculation
            
        Returns:
            RSI value (0-100) or None if calculation fails
        """
        try:
            if len(prices) <= period:
                logger.warning(f"Not enough data for RSI calculation. Need >{period}, got {len(prices)}")
                return None
                
            # Calculate price changes
            deltas = np.diff(prices)
            
            # Separate gains and losses
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Calculate average gains and losses
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            # Calculate subsequent average gains and losses
            for i in range(period, len(deltas)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            # Calculate RS and RSI
            if avg_loss == 0:
                return 100.0
                
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            log_exception(e, "calculate_rsi")
            return None
    
    @staticmethod
    @handle_api_error
    def calculate_macd(prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Optional[Dict[str, float]]:
        """
        Calculate Moving Average Convergence Divergence
        
        Args:
            prices: Array of price data
            fast_period: Period for fast EMA
            slow_period: Period for slow EMA
            signal_period: Period for signal line
            
        Returns:
            Dictionary with MACD line, signal line, and histogram values, or None if calculation fails
        """
        try:
            if len(prices) < max(fast_period, slow_period, signal_period):
                logger.warning(f"Not enough data for MACD calculation. Need >{max(fast_period, slow_period, signal_period)}, got {len(prices)}")
                return None
                
            # Calculate fast and slow EMAs
            ema_fast = np.zeros_like(prices)
            ema_slow = np.zeros_like(prices)
            
            # Initialize with SMA
            ema_fast[:fast_period] = np.nan
            ema_slow[:slow_period] = np.nan
            
            ema_fast[fast_period-1] = np.mean(prices[:fast_period])
            ema_slow[slow_period-1] = np.mean(prices[:slow_period])
            
            # Calculate subsequent EMAs
            alpha_fast = 2 / (fast_period + 1)
            alpha_slow = 2 / (slow_period + 1)
            
            for i in range(fast_period, len(prices)):
                ema_fast[i] = prices[i] * alpha_fast + ema_fast[i-1] * (1 - alpha_fast)
                
            for i in range(slow_period, len(prices)):
                ema_slow[i] = prices[i] * alpha_slow + ema_slow[i-1] * (1 - alpha_slow)
            
            # Calculate MACD line
            macd_line = ema_fast[slow_period-1:] - ema_slow[slow_period-1:]
            
            # Calculate signal line
            signal_line = np.zeros_like(macd_line)
            signal_line[:signal_period] = np.nan
            signal_line[signal_period-1] = np.mean(macd_line[:signal_period])
            
            alpha_signal = 2 / (signal_period + 1)
            for i in range(signal_period, len(macd_line)):
                signal_line[i] = macd_line[i] * alpha_signal + signal_line[i-1] * (1 - alpha_signal)
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return {
                "macd_line": macd_line[-1],
                "signal_line": signal_line[-1],
                "histogram": histogram[-1]
            }
        except Exception as e:
            log_exception(e, "calculate_macd")
            return None
    
    @staticmethod
    @handle_api_error
    def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Optional[Dict[str, float]]:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Array of price data
            period: Period for moving average
            std_dev: Number of standard deviations for bands
            
        Returns:
            Dictionary with upper band, middle band, and lower band values, or None if calculation fails
        """
        try:
            if len(prices) < period:
                logger.warning(f"Not enough data for Bollinger Bands calculation. Need {period}, got {len(prices)}")
                return None
                
            # Calculate middle band (SMA)
            middle_band = np.mean(prices[-period:])
            
            # Calculate standard deviation
            sigma = np.std(prices[-period:])
            
            # Calculate upper and lower bands
            upper_band = middle_band + (std_dev * sigma)
            lower_band = middle_band - (std_dev * sigma)
            
            # Calculate bandwidth and %B
            bandwidth = (upper_band - lower_band) / middle_band
            percent_b = (prices[-1] - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
            
            return {
                "upper_band": upper_band,
                "middle_band": middle_band,
                "lower_band": lower_band,
                "bandwidth": bandwidth,
                "percent_b": percent_b
            }
        except Exception as e:
            log_exception(e, "calculate_bollinger_bands")
            return None
    
    @staticmethod
    @handle_api_error
    def calculate_vwap(prices: np.ndarray, volumes: np.ndarray, reset_daily: bool = True, timestamp: Optional[np.ndarray] = None) -> Optional[float]:
        """
        Calculate Volume Weighted Average Price
        
        Args:
            prices: Array of price data
            volumes: Array of volume data
            reset_daily: Whether to reset VWAP calculation daily
            timestamp: Array of timestamps (required if reset_daily is True)
            
        Returns:
            VWAP value or None if calculation fails
        """
        try:
            if len(prices) != len(volumes):
                logger.warning(f"Price and volume arrays must be the same length. Prices: {len(prices)}, Volumes: {len(volumes)}")
                return None
                
            if len(prices) == 0:
                logger.warning("Empty price and volume arrays")
                return None
                
            if reset_daily and timestamp is None:
                logger.warning("Timestamp array required for daily VWAP reset")
                return None
                
            # If reset_daily is True, filter data for current day
            if reset_daily and timestamp is not None:
                # Convert timestamps to days
                days = np.array([ts // (24 * 60 * 60 * 1000) for ts in timestamp])
                current_day = days[-1]
                
                # Filter data for current day
                day_mask = days == current_day
                day_prices = prices[day_mask]
                day_volumes = volumes[day_mask]
                
                # Calculate VWAP for current day
                if len(day_prices) == 0:
                    logger.warning("No data for current day")
                    return None
                    
                return np.sum(day_prices * day_volumes) / np.sum(day_volumes)
            else:
                # Calculate VWAP for all data
                return np.sum(prices * volumes) / np.sum(volumes)
        except Exception as e:
            log_exception(e, "calculate_vwap")
            return None
    
    @staticmethod
    @handle_api_error
    def calculate_atr(high_prices: np.ndarray, low_prices: np.ndarray, close_prices: np.ndarray, period: int = 14) -> Optional[float]:
        """
        Calculate Average True Range
        
        Args:
            high_prices: Array of high prices
            low_prices: Array of low prices
            close_prices: Array of close prices
            period: Period for ATR calculation
            
        Returns:
            ATR value or None if calculation fails
        """
        try:
            if len(high_prices) != len(low_prices) or len(high_prices) != len(close_prices):
                logger.warning("High, low, and close price arrays must be the same length")
                return None
                
            if len(high_prices) <= period:
                logger.warning(f"Not enough data for ATR calculation. Need >{period}, got {len(high_prices)}")
                return None
                
            # Calculate true ranges
            tr = np.zeros(len(high_prices))
            
            # First true range is high - low
            tr[0] = high_prices[0] - low_prices[0]
            
            # Calculate subsequent true ranges
            for i in range(1, len(high_prices)):
                tr[i] = max(
                    high_prices[i] - low_prices[i],
                    abs(high_prices[i] - close_prices[i-1]),
                    abs(low_prices[i] - close_prices[i-1])
                )
            
            # Calculate initial ATR as simple average
            atr = np.mean(tr[:period])
            
            # Calculate subsequent ATRs using smoothing
            for i in range(period, len(tr)):
                atr = ((period - 1) * atr + tr[i]) / period
            
            return atr
        except Exception as e:
            log_exception(e, "calculate_atr")
            return None
    
    @staticmethod
    @handle_api_error
    def detect_divergence(prices: np.ndarray, indicator_values: np.ndarray, lookback: int = 10) -> Optional[Dict[str, Union[bool, str, float]]]:
        """
        Detect divergence between price and indicator
        
        Args:
            prices: Array of price data
            indicator_values: Array of indicator values (e.g., RSI)
            lookback: Number of periods to look back for divergence
            
        Returns:
            Dictionary with divergence information or None if calculation fails
        """
        try:
            if len(prices) != len(indicator_values):
                logger.warning("Price and indicator arrays must be the same length")
                return None
                
            if len(prices) < lookback:
                logger.warning(f"Not enough data for divergence detection. Need {lookback}, got {len(prices)}")
                return None
                
            # Get relevant data
            recent_prices = prices[-lookback:]
            recent_indicators = indicator_values[-lookback:]
            
            # Find local extrema
            price_highs = []
            price_lows = []
            indicator_highs = []
            indicator_lows = []
            
            for i in range(1, lookback-1):
                # Price highs and lows
                if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
                    price_highs.append((i, recent_prices[i]))
                if recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1]:
                    price_lows.append((i, recent_prices[i]))
                    
                # Indicator highs and lows
                if recent_indicators[i] > recent_indicators[i-1] and recent_indicators[i] > recent_indicators[i+1]:
                    indicator_highs.append((i, recent_indicators[i]))
                if recent_indicators[i] < recent_indicators[i-1] and recent_indicators[i] < recent_indicators[i+1]:
                    indicator_lows.append((i, recent_indicators[i]))
            
            # Check for divergence
            bullish_divergence = False
            bearish_divergence = False
            
            # Need at least 2 lows/highs to detect divergence
            if len(price_lows) >= 2 and len(indicator_lows) >= 2:
                # Check for bullish divergence (price making lower lows, indicator making higher lows)
                if price_lows[-1][1] < price_lows[-2][1] and indicator_lows[-1][1] > indicator_lows[-2][1]:
                    bullish_divergence = True
            
            if len(price_highs) >= 2 and len(indicator_highs) >= 2:
                # Check for bearish divergence (price making higher highs, indicator making lower highs)
                if price_highs[-1][1] > price_highs[-2][1] and indicator_highs[-1][1] < indicator_highs[-2][1]:
                    bearish_divergence = True
            
            # Determine divergence type and strength
            divergence_type = None
            divergence_strength = 0.0
            
            if bullish_divergence:
                divergence_type = "bullish"
                # Calculate strength based on the difference in slopes
                price_slope = (price_lows[-1][1] - price_lows[-2][1]) / (price_lows[-1][0] - price_lows[-2][0])
                indicator_slope = (indicator_lows[-1][1] - indicator_lows[-2][1]) / (indicator_lows[-1][0] - indicator_lows[-2][0])
                divergence_strength = abs(price_slope - indicator_slope)
            elif bearish_divergence:
                divergence_type = "bearish"
                # Calculate strength based on the difference in slopes
                price_slope = (price_highs[-1][1] - price_highs[-2][1]) / (price_highs[-1][0] - price_highs[-2][0])
                indicator_slope = (indicator_highs[-1][1] - indicator_highs[-2][1]) / (indicator_highs[-1][0] - indicator_highs[-2][0])
                divergence_strength = abs(price_slope - indicator_slope)
            
            return {
                "divergence_detected": bullish_divergence or bearish_divergence,
                "divergence_type": divergence_type,
                "divergence_strength": divergence_strength
            }
        except Exception as e:
            log_exception(e, "detect_divergence")
            return None
    
    @staticmethod
    @handle_api_error
    def calculate_all_indicators(market_data: Dict[str, np.ndarray]) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Calculate all indicators for given market data
        
        Args:
            market_data: Dictionary containing price and volume data
                Required keys: 'close', 'high', 'low', 'volume', 'timestamp'
            
        Returns:
            Dictionary with all indicator values
        """
        try:
            # Extract data
            close_prices = market_data.get('close')
            high_prices = market_data.get('high')
            low_prices = market_data.get('low')
            volumes = market_data.get('volume')
            timestamps = market_data.get('timestamp')
            
            # Validate data
            if close_prices is None or len(close_prices) == 0:
                logger.warning("No close price data provided")
                return {}
            
            # Initialize results
            results = {}
            
            # Calculate indicators
            results['sma_20'] = TechnicalIndicators.calculate_sma(close_prices, period=20)
            results['sma_50'] = TechnicalIndicators.calculate_sma(close_prices, period=50)
            results['sma_200'] = TechnicalIndicators.calculate_sma(close_prices, period=200)
            
            results['ema_12'] = TechnicalIndicators.calculate_ema(close_prices, period=12)
            results['ema_26'] = TechnicalIndicators.calculate_ema(close_prices, period=26)
            
            results['rsi_14'] = TechnicalIndicators.calculate_rsi(close_prices, period=14)
            
            results['macd'] = TechnicalIndicators.calculate_macd(close_prices)
            
            results['bollinger_bands'] = TechnicalIndicators.calculate_bollinger_bands(close_prices)
            
            if volumes is not None and len(volumes) == len(close_prices):
                results['vwap'] = TechnicalIndicators.calculate_vwap(
                    close_prices, volumes, reset_daily=True, timestamp=timestamps
                )
            
            if high_prices is not None and low_prices is not None:
                if len(high_prices) == len(close_prices) and len(low_prices) == len(close_prices):
                    results['atr_14'] = TechnicalIndicators.calculate_atr(
                        high_prices, low_prices, close_prices, period=14
                    )
            
            # Check for divergence
            if results['rsi_14'] is not None:
                # Create array of RSI values for divergence detection
                rsi_values = np.zeros_like(close_prices)
                rsi_values[-1] = results['rsi_14']
                
                # This is a simplified approach; in practice, you would maintain a history of RSI values
                results['rsi_divergence'] = TechnicalIndicators.detect_divergence(
                    close_prices[-20:], rsi_values[-20:], lookback=10
                )
            
            return results
        except Exception as e:
            log_exception(e, "calculate_all_indicators")
            return {}


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n = 200
    close_prices = np.cumsum(np.random.normal(0, 1, n)) + 100
    high_prices = close_prices + np.random.uniform(0, 2, n)
    low_prices = close_prices - np.random.uniform(0, 2, n)
    volumes = np.random.uniform(1000, 5000, n)
    timestamps = np.array([1622505600000 + i * 60000 for i in range(n)])  # 1-minute intervals
    
    # Calculate indicators
    rsi = TechnicalIndicators.calculate_rsi(close_prices)
    macd = TechnicalIndicators.calculate_macd(close_prices)
    bb = TechnicalIndicators.calculate_bollinger_bands(close_prices)
    vwap = TechnicalIndicators.calculate_vwap(close_prices, volumes, reset_daily=True, timestamp=timestamps)
    atr = TechnicalIndicators.calculate_atr(high_prices, low_prices, close_prices)
    
    # Print results
    print(f"RSI: {rsi:.2f}")
    print(f"MACD Line: {macd['macd_line']:.4f}, Signal: {macd['signal_line']:.4f}, Histogram: {macd['histogram']:.4f}")
    print(f"Bollinger Bands - Upper: {bb['upper_band']:.2f}, Middle: {bb['middle_band']:.2f}, Lower: {bb['lower_band']:.2f}")
    print(f"VWAP: {vwap:.2f}")
    print(f"ATR: {atr:.4f}")
    
    # Calculate all indicators
    market_data = {
        'close': close_prices,
        'high': high_prices,
        'low': low_prices,
        'volume': volumes,
        'timestamp': timestamps
    }
    
    all_indicators = TechnicalIndicators.calculate_all_indicators(market_data)
    print("\nAll Indicators:")
    for key, value in all_indicators.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")
