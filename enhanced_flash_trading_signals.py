#!/usr/bin/env python
"""
Enhanced Flash Trading Signals with Technical Indicators

This module extends the original flash_trading_signals.py with advanced technical indicators,
multi-timeframe analysis, and dynamic thresholding for improved signal generation.
"""

import time
import logging
import json
import os
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from threading import Thread, Event, RLock
from collections import deque

# Import mock client for testing
from mock_exchange_client import MockExchangeClient

# Try to import real client, but don't fail if not available
try:
    from optimized_mexc_client import OptimizedMexcClient
    REAL_CLIENT_AVAILABLE = True
except ImportError:
    REAL_CLIENT_AVAILABLE = False
    logging.warning("OptimizedMexcClient not available, using MockExchangeClient for all operations")

try:
    from trading_session_manager import TradingSessionManager
    from indicators import TechnicalIndicators
    from error_handling_utils import safe_get, safe_get_nested, validate_api_response, handle_api_error, log_exception, parse_float_safely
except ImportError:
    # Create minimal versions for testing
    class TradingSessionManager:
        def get_current_session(self):
            return "EUROPE"
    
    class TechnicalIndicators:
        @staticmethod
        def calculate_all_indicators(market_data):
            return {
                "rsi": 50.0,
                "macd": 0.0,
                "macd_signal": 0.0,
                "macd_hist": 0.0,
                "bollinger_upper": market_data['close'][-1] * 1.02,
                "bollinger_middle": market_data['close'][-1],
                "bollinger_lower": market_data['close'][-1] * 0.98,
                "atr": market_data['close'][-1] * 0.01
            }
    
    def handle_api_error(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"API error in {func.__name__}: {str(e)}")
                return None
        return wrapper
    
    def safe_get(data, key, default=None):
        return data.get(key, default) if data else default
    
    def safe_get_nested(data, keys, default=None):
        for key in keys:
            if not data or key not in data:
                return default
            data = data[key]
        return data
    
    def validate_api_response(response):
        return True
    
    def log_exception(e, context=""):
        logging.error(f"Exception in {context}: {str(e)}")
    
    def parse_float_safely(value, default=0.0):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("enhanced_flash_trading_signals")

class EnhancedMarketState:
    """Enhanced market state with technical indicators and multi-timeframe support"""
    
    def __init__(self, symbol):
        """Initialize enhanced market state for a symbol"""
        self.symbol = symbol
        self.timestamp = int(time.time() * 1000)
        
        # Order book state
        self.bids = []
        self.asks = []
        self.bid_price = None
        self.ask_price = None
        self.mid_price = None
        self.spread = None
        self.spread_bps = None
        self.order_imbalance = 0.0
        
        # Price history (multi-timeframe)
        self.price_history = {
            '1m': deque(maxlen=300),    # 5 hours of 1-minute data
            '5m': deque(maxlen=288),    # 24 hours of 5-minute data
            '15m': deque(maxlen=192),   # 48 hours of 15-minute data
            '1h': deque(maxlen=168)     # 7 days of 1-hour data
        }
        self.volume_history = {
            '1m': deque(maxlen=300),
            '5m': deque(maxlen=288),
            '15m': deque(maxlen=192),
            '1h': deque(maxlen=168)
        }
        self.timestamp_history = {
            '1m': deque(maxlen=300),
            '5m': deque(maxlen=288),
            '15m': deque(maxlen=192),
            '1h': deque(maxlen=168)
        }
        
        # High/Low price history for ATR calculation
        self.high_price_history = {
            '1m': deque(maxlen=300),
            '5m': deque(maxlen=288),
            '15m': deque(maxlen=192),
            '1h': deque(maxlen=168)
        }
        self.low_price_history = {
            '1m': deque(maxlen=300),
            '5m': deque(maxlen=288),
            '15m': deque(maxlen=192),
            '1h': deque(maxlen=168)
        }
        
        # Technical indicators
        self.indicators = {
            '1m': {},
            '5m': {},
            '15m': {},
            '1h': {}
        }
        
        # Derived metrics
        self.momentum = 0.0
        self.volatility = 0.0
        self.trend = 0.0
        
        # Trading metrics
        self.last_trade_price = None
        self.last_trade_side = None
        self.last_trade_time = None
        
        # Liquidity metrics
        self.bid_liquidity = 0.0
        self.ask_liquidity = 0.0
        self.slippage_estimate = 0.0
        
        # Timeframe management
        self.last_candle_close = {
            '1m': 0,
            '5m': 0,
            '15m': 0,
            '1h': 0
        }
    
    def update_order_book(self, bids, asks):
        """Update order book state and calculate liquidity metrics"""
        if not bids or not asks:
            return False
        
        self.bids = bids
        self.asks = asks
        self.timestamp = int(time.time() * 1000)
        
        # Update prices
        self.bid_price = float(bids[0][0])
        self.ask_price = float(asks[0][0])
        self.mid_price = (self.bid_price + self.ask_price) / 2
        self.spread = self.ask_price - self.bid_price
        self.spread_bps = (self.spread / self.mid_price) * 10000  # Basis points
        
        # Calculate order book imbalance
        bid_volume = sum(float(bid[1]) for bid in bids[:5])
        ask_volume = sum(float(ask[1]) for ask in asks[:5])
        total_volume = bid_volume + ask_volume
        
        if total_volume > 0:
            self.order_imbalance = (bid_volume - ask_volume) / total_volume
        
        # Calculate liquidity metrics
        self._calculate_liquidity_metrics(bids, asks)
        
        # Update price history for all timeframes
        self._update_price_history()
        
        # Calculate technical indicators for all timeframes
        self._calculate_technical_indicators()
        
        # Calculate derived metrics if we have enough history
        if len(self.price_history['1m']) >= 10:
            self._calculate_derived_metrics()
        
        return True
    
    def _calculate_liquidity_metrics(self, bids, asks):
        """Calculate liquidity and slippage metrics from order book"""
        try:
            # Calculate total liquidity within 1% of mid price
            price_range = self.mid_price * 0.01  # 1% range
            
            # Bid liquidity (sum of volumes for bids within 1% of mid price)
            self.bid_liquidity = sum(
                float(bid[1]) for bid in bids 
                if float(bid[0]) >= self.mid_price - price_range
            )
            
            # Ask liquidity (sum of volumes for asks within 1% of mid price)
            self.ask_liquidity = sum(
                float(ask[1]) for ask in asks 
                if float(ask[0]) <= self.mid_price + price_range
            )
            
            # Estimate slippage for a standard order size (0.1 BTC/ETH)
            standard_order_size = 0.1
            
            # Simulate market buy
            buy_slippage = self._estimate_slippage(asks, standard_order_size, "buy")
            
            # Simulate market sell
            sell_slippage = self._estimate_slippage(bids, standard_order_size, "sell")
            
            # Average slippage
            self.slippage_estimate = (buy_slippage + sell_slippage) / 2
            
        except Exception as e:
            logger.error(f"Error calculating liquidity metrics: {str(e)}")
    
    def _estimate_slippage(self, orders, size, side):
        """Estimate slippage for a given order size and side"""
        try:
            remaining_size = size
            executed_value = 0.0
            
            for order in orders:
                price = float(order[0])
                volume = float(order[1])
                
                if remaining_size <= 0:
                    break
                
                executed_volume = min(remaining_size, volume)
                executed_value += executed_volume * price
                remaining_size -= executed_volume
            
            # If we couldn't fill the entire order
            if remaining_size > 0:
                return 0.05  # Assume 5% slippage for unfillable orders
            
            # Calculate average execution price
            avg_price = executed_value / size
            
            # Calculate slippage in basis points
            if side == "buy":
                slippage_bps = (avg_price - self.ask_price) / self.ask_price * 10000
            else:
                slippage_bps = (self.bid_price - avg_price) / self.bid_price * 10000
            
            return max(0, slippage_bps)
            
        except Exception as e:
            logger.error(f"Error estimating slippage: {str(e)}")
            return 0.0
    
    def _update_price_history(self):
        """Update price history for all timeframes"""
        current_time = self.timestamp
        
        # Always update 1m data
        self._update_timeframe_data('1m', current_time)
        
        # Update other timeframes if their candle has closed
        if self._is_candle_closed('5m', current_time):
            self._update_timeframe_data('5m', current_time)
            
        if self._is_candle_closed('15m', current_time):
            self._update_timeframe_data('15m', current_time)
            
        if self._is_candle_closed('1h', current_time):
            self._update_timeframe_data('1h', current_time)
    
    def _is_candle_closed(self, timeframe, current_time):
        """Check if a candle has closed for the given timeframe"""
        minutes_map = {'1m': 1, '5m': 5, '15m': 15, '1h': 60}
        minutes = minutes_map.get(timeframe, 1)
        
        # Convert to minutes since epoch
        current_minutes = current_time // (60 * 1000)
        last_update_minutes = self.last_candle_close[timeframe] // (60 * 1000)
        
        # Check if we've crossed a candle boundary
        return (current_minutes // minutes) > (last_update_minutes // minutes)
    
    def _update_timeframe_data(self, timeframe, current_time):
        """Update price history for a specific timeframe"""
        self.price_history[timeframe].append(self.mid_price)
        self.timestamp_history[timeframe].append(current_time)
        
        # Use bid/ask as proxy for high/low in absence of actual candle data
        self.high_price_history[timeframe].append(self.ask_price)
        self.low_price_history[timeframe].append(self.bid_price)
        
        # Update last candle close time
        self.last_candle_close[timeframe] = current_time
    
    def _calculate_technical_indicators(self):
        """Calculate technical indicators for all timeframes"""
        for timeframe in self.price_history.keys():
            if len(self.price_history[timeframe]) < 20:
                continue  # Not enough data
                
            # Convert deques to numpy arrays
            prices = np.array(list(self.price_history[timeframe]))
            highs = np.array(list(self.high_price_history[timeframe]))
            lows = np.array(list(self.low_price_history[timeframe]))
            
            # Create market data dictionary
            market_data = {
                'close': prices,
                'high': highs,
                'low': lows,
                'timestamp': np.array(list(self.timestamp_history[timeframe]))
            }
            
            # Calculate all indicators
            self.indicators[timeframe] = TechnicalIndicators.calculate_all_indicators(market_data)
    
    def _calculate_derived_metrics(self):
        """Calculate derived metrics from price history"""
        # Use 1-minute timeframe for these calculations
        prices = np.array(list(self.price_history['1m']))
        
        # Calculate momentum (rate of change)
        if len(prices) >= 10:
            self.momentum = (prices[-1] - prices[-10]) / prices[-10]
        
        # Calculate volatility (standard deviation of returns)
        if len(prices) >= 6:  # Need at least 6 prices for 5 returns
            try:
                # Get price differences (n-1 elements)
                price_diffs = np.diff(prices[-5:])
                
                # Get denominator prices (must be same length as price_diffs)
                denominator_prices = prices[-6:-1]
                
                # Ensure both arrays have the same shape
                min_length = min(len(price_diffs), len(denominator_prices))
                price_diffs = price_diffs[:min_length]
                denominator_prices = denominator_prices[:min_length]
                
                # Calculate returns with validated shapes
                returns = price_diffs / denominator_prices
                
                # Calculate volatility
                if len(returns) > 0:
                    self.volatility = np.std(returns) * np.sqrt(len(returns))
                else:
                    self.volatility = 0.0
            except Exception as e:
                logger.error(f"Error calculating volatility: {str(e)}")
                self.volatility = 0.0
        
        # Calculate trend (simple moving average direction)
        if len(prices) >= 20:
            sma_short = np.mean(prices[-5:])
            sma_long = np.mean(prices[-20:])
            self.trend = sma_short - sma_long

class EnhancedFlashTradingSignals:
    """Enhanced signal generation with technical indicators and multi-timeframe analysis"""
    
    def __init__(self, client_instance=None, api_key=None, api_secret=None, env_path=None):
        """Initialize enhanced flash trading signals
        
        Args:
            client_instance: Existing client instance to use (preferred)
            api_key: API key for exchange (used only if client_instance is None)
            api_secret: API secret for exchange (used only if client_instance is None)
            env_path: Path to .env file (used only if client_instance is None)
        """
        # Use existing client instance if provided, otherwise create new one
        if client_instance is not None:
            self.api_client = client_instance
            logger.info("Using provided client instance for EnhancedSignalGenerator")
        else:
            # Try to use real client if available, otherwise use mock
            if REAL_CLIENT_AVAILABLE:
                try:
                    # Initialize API client with direct credentials
                    self.api_client = OptimizedMexcClient(api_key, api_secret, env_path)
                    logger.info("Created new OptimizedMexcClient instance for EnhancedSignalGenerator")
                except Exception as e:
                    logger.warning(f"Failed to create OptimizedMexcClient: {str(e)}")
                    logger.info("Falling back to MockExchangeClient")
                    self.api_client = MockExchangeClient()
            else:
                # Use mock client
                logger.info("Using MockExchangeClient for EnhancedSignalGenerator")
                self.api_client = MockExchangeClient()
        
        # Initialize session manager
        self.session_manager = TradingSessionManager()
        
        # Signal history
        self.signals = []
        self.max_signals = 1000
        
        # Market state cache
        self.market_states = {}
        
        # Thread safety for client access
        self.client_lock = RLock()
        
        # Thread safety for market state updates
        self.market_state_lock = RLock()
        
        # Dynamic thresholds
        self.dynamic_thresholds = {}
        
        # Configuration dictionary for compatibility with flash_trading.py
        self.config = {
            "imbalance_threshold": 0.2,
            "momentum_threshold": 0.005,
            "volatility_threshold": 0.002,
            "min_signal_strength": 0.1,
            "position_size": 0.1
        }
        
        # Thread management for compatibility with flash_trading.py
        self.running = False
        self.symbols = []
        self.update_thread = None
        self.stop_event = Event()
        
        # Initialize dynamic thresholds
        self._initialize_dynamic_thresholds()
    
    def _initialize_dynamic_thresholds(self):
        """Initialize dynamic thresholds for all signal types"""
        # Base thresholds for each signal type
        base_thresholds = {
            "order_imbalance": 0.08,
            "momentum": 0.02,
            "volatility": 0.03,
            "rsi": 70.0,  # RSI overbought threshold
            "bollinger": 0.8,  # % distance from middle band
            "macd": 0.0002  # MACD histogram threshold
        }
        
        # Initialize dynamic thresholds for each session
        for session in ["ASIA", "EUROPE", "US"]:
            self.dynamic_thresholds[session] = {
                signal_type: {
                    "base": value,
                    "current": value,
                    "min": value * 0.5,
                    "max": value * 2.0,
                    "adjustment_factor": 1.0
                }
                for signal_type, value in base_thresholds.items()
            }
    
    @handle_api_error
    def start(self, symbols):
        """Start signal generation for specified symbols
        
        Args:
            symbols: List of trading pair symbols (e.g., ['BTCUSDC', 'ETHUSDC'])
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("Signal generator already running")
            return False
            
        if not symbols or not isinstance(symbols, list):
            logger.error(f"Invalid symbols list: {symbols}")
            return False
        
        self.symbols = symbols
        self.running = True
        self.stop_event.clear()
        
        # Initialize market states
        for symbol in symbols:
            if symbol not in self.market_states:
                self.market_states[symbol] = EnhancedMarketState(symbol)
        
        # Start update thread
        self.update_thread = Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        logger.info(f"Started signal generation for symbols: {symbols}")
        return True
    
    def stop(self):
        """Stop signal generation
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.running:
            logger.warning("Signal generator not running")
            return False
        
        self.running = False
        self.stop_event.set()
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)
        
        logger.info("Stopped signal generation")
        return True
    
    def _update_loop(self):
        """Update loop for market data and signal generation"""
        while self.running and not self.stop_event.is_set():
            try:
                # Update market data for all symbols
                for symbol in self.symbols:
                    self._update_market_data(symbol)
                    
                    # Generate signals
                    self._generate_signals(symbol)
                
                # Sleep to avoid excessive API calls
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in update loop: {str(e)}")
                time.sleep(5.0)  # Sleep longer on error
    
    def _update_market_data(self, symbol):
        """Update market data for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDC')
        """
        try:
            with self.client_lock:
                # Get order book
                order_book = self.api_client.get_order_book(symbol)
                
                if not order_book:
                    logger.warning(f"Empty order book for {symbol}")
                    return
                
                # Extract bids and asks
                bids = safe_get(order_book, 'bids', [])
                asks = safe_get(order_book, 'asks', [])
                
                # Update market state
                with self.market_state_lock:
                    if symbol not in self.market_states:
                        self.market_states[symbol] = EnhancedMarketState(symbol)
                    
                    self.market_states[symbol].update_order_book(bids, asks)
                
        except Exception as e:
            logger.error(f"Error updating market data for {symbol}: {str(e)}")
    
    def _generate_signals(self, symbol):
        """Generate signals for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDC')
        """
        try:
            with self.market_state_lock:
                if symbol not in self.market_states:
                    return
                
                market_state = self.market_states[symbol]
                
                # Skip if not enough data
                if len(market_state.price_history['1m']) < 20:
                    return
                
                # Get current trading session
                current_session = self.session_manager.get_current_session()
                
                # Get thresholds for current session
                thresholds = self.dynamic_thresholds.get(current_session, {})
                
                # Check for order imbalance signal
                self._check_order_imbalance_signal(symbol, market_state, thresholds)
                
                # Check for momentum signal
                self._check_momentum_signal(symbol, market_state, thresholds)
                
                # Check for volatility signal
                self._check_volatility_signal(symbol, market_state, thresholds)
                
                # Check for technical indicator signals
                self._check_technical_signals(symbol, market_state, thresholds)
                
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {str(e)}")
    
    def _check_order_imbalance_signal(self, symbol, market_state, thresholds):
        """Check for order imbalance signal
        
        Args:
            symbol: Trading pair symbol
            market_state: Market state object
            thresholds: Dynamic thresholds for current session
        """
        try:
            # Get threshold
            threshold = thresholds.get("order_imbalance", {}).get("current", 0.08)
            
            # Check for significant imbalance
            if abs(market_state.order_imbalance) > threshold:
                signal_type = "buy" if market_state.order_imbalance > 0 else "sell"
                signal_strength = abs(market_state.order_imbalance)
                
                # Create signal
                signal = {
                    "timestamp": market_state.timestamp,
                    "symbol": symbol,
                    "type": signal_type,
                    "source": "order_imbalance",
                    "strength": signal_strength,
                    "price": market_state.mid_price,
                    "threshold": threshold,
                    "value": market_state.order_imbalance
                }
                
                # Add signal
                self._add_signal(signal)
                
        except Exception as e:
            logger.error(f"Error checking order imbalance signal: {str(e)}")
    
    def _check_momentum_signal(self, symbol, market_state, thresholds):
        """Check for momentum signal
        
        Args:
            symbol: Trading pair symbol
            market_state: Market state object
            thresholds: Dynamic thresholds for current session
        """
        try:
            # Get threshold
            threshold = thresholds.get("momentum", {}).get("current", 0.02)
            
            # Check for significant momentum
            if abs(market_state.momentum) > threshold:
                signal_type = "buy" if market_state.momentum > 0 else "sell"
                signal_strength = abs(market_state.momentum) / threshold
                
                # Create signal
                signal = {
                    "timestamp": market_state.timestamp,
                    "symbol": symbol,
                    "type": signal_type,
                    "source": "momentum",
                    "strength": signal_strength,
                    "price": market_state.mid_price,
                    "threshold": threshold,
                    "value": market_state.momentum
                }
                
                # Add signal
                self._add_signal(signal)
                
        except Exception as e:
            logger.error(f"Error checking momentum signal: {str(e)}")
    
    def _check_volatility_signal(self, symbol, market_state, thresholds):
        """Check for volatility signal
        
        Args:
            symbol: Trading pair symbol
            market_state: Market state object
            thresholds: Dynamic thresholds for current session
        """
        try:
            # Get threshold
            threshold = thresholds.get("volatility", {}).get("current", 0.03)
            
            # Check for significant volatility
            if market_state.volatility > threshold:
                # Volatility signals are neutral (can be used to adjust position size)
                signal_type = "neutral"
                signal_strength = market_state.volatility / threshold
                
                # Create signal
                signal = {
                    "timestamp": market_state.timestamp,
                    "symbol": symbol,
                    "type": signal_type,
                    "source": "volatility",
                    "strength": signal_strength,
                    "price": market_state.mid_price,
                    "threshold": threshold,
                    "value": market_state.volatility
                }
                
                # Add signal
                self._add_signal(signal)
                
        except Exception as e:
            logger.error(f"Error checking volatility signal: {str(e)}")
    
    def _check_technical_signals(self, symbol, market_state, thresholds):
        """Check for technical indicator signals
        
        Args:
            symbol: Trading pair symbol
            market_state: Market state object
            thresholds: Dynamic thresholds for current session
        """
        try:
            # Check RSI signals
            self._check_rsi_signals(symbol, market_state, thresholds)
            
            # Check Bollinger Band signals
            self._check_bollinger_signals(symbol, market_state, thresholds)
            
            # Check MACD signals
            self._check_macd_signals(symbol, market_state, thresholds)
            
        except Exception as e:
            logger.error(f"Error checking technical signals: {str(e)}")
    
    def _check_rsi_signals(self, symbol, market_state, thresholds):
        """Check for RSI signals
        
        Args:
            symbol: Trading pair symbol
            market_state: Market state object
            thresholds: Dynamic thresholds for current session
        """
        try:
            # Get threshold
            threshold = thresholds.get("rsi", {}).get("current", 70.0)
            
            # Check 5m timeframe
            if '5m' in market_state.indicators:
                rsi = market_state.indicators['5m'].get('rsi')
                
                if rsi is not None:
                    # Overbought
                    if rsi > threshold:
                        signal_type = "sell"
                        signal_strength = (rsi - threshold) / (100 - threshold)
                        
                        # Create signal
                        signal = {
                            "timestamp": market_state.timestamp,
                            "symbol": symbol,
                            "type": signal_type,
                            "source": "rsi_overbought",
                            "strength": signal_strength,
                            "price": market_state.mid_price,
                            "threshold": threshold,
                            "value": rsi,
                            "timeframe": "5m"
                        }
                        
                        # Add signal
                        self._add_signal(signal)
                    
                    # Oversold
                    elif rsi < (100 - threshold):
                        signal_type = "buy"
                        signal_strength = ((100 - threshold) - rsi) / (100 - threshold)
                        
                        # Create signal
                        signal = {
                            "timestamp": market_state.timestamp,
                            "symbol": symbol,
                            "type": signal_type,
                            "source": "rsi_oversold",
                            "strength": signal_strength,
                            "price": market_state.mid_price,
                            "threshold": 100 - threshold,
                            "value": rsi,
                            "timeframe": "5m"
                        }
                        
                        # Add signal
                        self._add_signal(signal)
                
        except Exception as e:
            logger.error(f"Error checking RSI signals: {str(e)}")
    
    def _check_bollinger_signals(self, symbol, market_state, thresholds):
        """Check for Bollinger Band signals
        
        Args:
            symbol: Trading pair symbol
            market_state: Market state object
            thresholds: Dynamic thresholds for current session
        """
        try:
            # Get threshold
            threshold = thresholds.get("bollinger", {}).get("current", 0.8)
            
            # Check 5m timeframe
            if '5m' in market_state.indicators:
                upper = market_state.indicators['5m'].get('bollinger_upper')
                middle = market_state.indicators['5m'].get('bollinger_middle')
                lower = market_state.indicators['5m'].get('bollinger_lower')
                
                if upper is not None and middle is not None and lower is not None:
                    price = market_state.mid_price
                    
                    # Calculate distance from middle band as percentage of band width
                    band_width = upper - lower
                    
                    if band_width > 0:
                        # Upper band touch/break
                        if price >= (middle + threshold * band_width / 2):
                            signal_type = "sell"
                            signal_strength = min(1.0, (price - middle) / (upper - middle))
                            
                            # Create signal
                            signal = {
                                "timestamp": market_state.timestamp,
                                "symbol": symbol,
                                "type": signal_type,
                                "source": "bollinger_upper",
                                "strength": signal_strength,
                                "price": price,
                                "threshold": threshold,
                                "value": (price - middle) / (band_width / 2),
                                "timeframe": "5m"
                            }
                            
                            # Add signal
                            self._add_signal(signal)
                        
                        # Lower band touch/break
                        elif price <= (middle - threshold * band_width / 2):
                            signal_type = "buy"
                            signal_strength = min(1.0, (middle - price) / (middle - lower))
                            
                            # Create signal
                            signal = {
                                "timestamp": market_state.timestamp,
                                "symbol": symbol,
                                "type": signal_type,
                                "source": "bollinger_lower",
                                "strength": signal_strength,
                                "price": price,
                                "threshold": threshold,
                                "value": (middle - price) / (band_width / 2),
                                "timeframe": "5m"
                            }
                            
                            # Add signal
                            self._add_signal(signal)
                
        except Exception as e:
            logger.error(f"Error checking Bollinger Band signals: {str(e)}")
    
    def _check_macd_signals(self, symbol, market_state, thresholds):
        """Check for MACD signals
        
        Args:
            symbol: Trading pair symbol
            market_state: Market state object
            thresholds: Dynamic thresholds for current session
        """
        try:
            # Get threshold
            threshold = thresholds.get("macd", {}).get("current", 0.0002)
            
            # Check 5m timeframe
            if '5m' in market_state.indicators:
                macd = market_state.indicators['5m'].get('macd')
                signal = market_state.indicators['5m'].get('macd_signal')
                hist = market_state.indicators['5m'].get('macd_hist')
                
                if macd is not None and signal is not None and hist is not None:
                    # MACD crosses above signal line
                    if hist > threshold and hist > 0:
                        signal_type = "buy"
                        signal_strength = min(1.0, hist / threshold)
                        
                        # Create signal
                        macd_signal = {
                            "timestamp": market_state.timestamp,
                            "symbol": symbol,
                            "type": signal_type,
                            "source": "macd_cross_above",
                            "strength": signal_strength,
                            "price": market_state.mid_price,
                            "threshold": threshold,
                            "value": hist,
                            "timeframe": "5m"
                        }
                        
                        # Add signal
                        self._add_signal(macd_signal)
                    
                    # MACD crosses below signal line
                    elif hist < -threshold and hist < 0:
                        signal_type = "sell"
                        signal_strength = min(1.0, -hist / threshold)
                        
                        # Create signal
                        macd_signal = {
                            "timestamp": market_state.timestamp,
                            "symbol": symbol,
                            "type": signal_type,
                            "source": "macd_cross_below",
                            "strength": signal_strength,
                            "price": market_state.mid_price,
                            "threshold": threshold,
                            "value": hist,
                            "timeframe": "5m"
                        }
                        
                        # Add signal
                        self._add_signal(macd_signal)
                
        except Exception as e:
            logger.error(f"Error checking MACD signals: {str(e)}")
    
    def _add_signal(self, signal):
        """Add a signal to the signal history
        
        Args:
            signal: Signal dictionary
        """
        # Add signal to history
        self.signals.append(signal)
        
        # Trim signal history if needed
        if len(self.signals) > self.max_signals:
            self.signals = self.signals[-self.max_signals:]
        
        # Log signal
        logger.info(f"Generated signal: {signal['source']} {signal['type']} for {signal['symbol']} (strength: {signal['strength']:.2f})")
    
    def generate_signals(self, symbol, timeframe="5m", limit=100, use_mock_data=False):
        """Generate signals for a symbol using historical data
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1m', '5m', '15m', '1h')
            limit: Number of candles to fetch
            use_mock_data: Whether to use mock data instead of real API
            
        Returns:
            list: Generated signals
        """
        try:
            # Clear previous signals
            self.signals = []
            
            # Get historical data
            with self.client_lock:
                # Use mock data if requested or if real client is not available
                if use_mock_data or not REAL_CLIENT_AVAILABLE:
                    logger.info(f"Using mock data for {symbol} {timeframe}")
                    klines = self.api_client.get_klines(symbol, timeframe, limit)
                else:
                    logger.info(f"Fetching real data for {symbol} {timeframe}")
                    klines = self.api_client.get_klines(symbol, timeframe, limit)
            
            if not klines:
                logger.warning(f"No klines data for {symbol} {timeframe}")
                return []
            
            # Process klines data
            timestamps = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []
            
            for kline in klines:
                timestamps.append(int(kline[0]))
                opens.append(float(kline[1]))
                highs.append(float(kline[2]))
                lows.append(float(kline[3]))
                closes.append(float(kline[4]))
                volumes.append(float(kline[5]))
            
            # Create market data dictionary
            market_data = {
                'timestamp': np.array(timestamps),
                'open': np.array(opens),
                'high': np.array(highs),
                'low': np.array(lows),
                'close': np.array(closes),
                'volume': np.array(volumes)
            }
            
            # Calculate technical indicators
            indicators = TechnicalIndicators.calculate_all_indicators(market_data)
            
            # Generate signals
            self._generate_signals_from_indicators(symbol, market_data, indicators, timeframe)
            
            return self.signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return []
    
    def _generate_signals_from_indicators(self, symbol, market_data, indicators, timeframe):
        """Generate signals from technical indicators
        
        Args:
            symbol: Trading pair symbol
            market_data: Market data dictionary
            indicators: Technical indicators dictionary
            timeframe: Timeframe
        """
        try:
            # Get current trading session
            current_session = self.session_manager.get_current_session()
            
            # Get thresholds for current session
            thresholds = self.dynamic_thresholds.get(current_session, {})
            
            # Get data
            timestamps = market_data['timestamp']
            closes = market_data['close']
            
            # Check RSI signals
            if 'rsi' in indicators:
                rsi_values = indicators['rsi']
                rsi_threshold = thresholds.get("rsi", {}).get("current", 70.0)
                
                for i in range(1, len(rsi_values)):
                    # Overbought
                    if rsi_values[i] > rsi_threshold:
                        signal_type = "sell"
                        signal_strength = (rsi_values[i] - rsi_threshold) / (100 - rsi_threshold)
                        
                        # Create signal
                        signal = {
                            "timestamp": int(timestamps[i]),
                            "symbol": symbol,
                            "type": signal_type,
                            "source": "rsi_overbought",
                            "strength": float(signal_strength),
                            "price": float(closes[i]),
                            "threshold": float(rsi_threshold),
                            "value": float(rsi_values[i]),
                            "timeframe": timeframe
                        }
                        
                        # Add signal
                        self._add_signal(signal)
                    
                    # Oversold
                    elif rsi_values[i] < (100 - rsi_threshold):
                        signal_type = "buy"
                        signal_strength = ((100 - rsi_threshold) - rsi_values[i]) / (100 - rsi_threshold)
                        
                        # Create signal
                        signal = {
                            "timestamp": int(timestamps[i]),
                            "symbol": symbol,
                            "type": signal_type,
                            "source": "rsi_oversold",
                            "strength": float(signal_strength),
                            "price": float(closes[i]),
                            "threshold": float(100 - rsi_threshold),
                            "value": float(rsi_values[i]),
                            "timeframe": timeframe
                        }
                        
                        # Add signal
                        self._add_signal(signal)
            
            # Check Bollinger Band signals
            if all(k in indicators for k in ['bollinger_upper', 'bollinger_middle', 'bollinger_lower']):
                upper = indicators['bollinger_upper']
                middle = indicators['bollinger_middle']
                lower = indicators['bollinger_lower']
                
                bollinger_threshold = thresholds.get("bollinger", {}).get("current", 0.8)
                
                for i in range(1, len(upper)):
                    # Calculate band width
                    band_width = upper[i] - lower[i]
                    
                    if band_width > 0:
                        # Upper band touch/break
                        if closes[i] >= (middle[i] + bollinger_threshold * band_width / 2):
                            signal_type = "sell"
                            signal_strength = min(1.0, (closes[i] - middle[i]) / (upper[i] - middle[i]))
                            
                            # Create signal
                            signal = {
                                "timestamp": int(timestamps[i]),
                                "symbol": symbol,
                                "type": signal_type,
                                "source": "bollinger_upper",
                                "strength": float(signal_strength),
                                "price": float(closes[i]),
                                "threshold": float(bollinger_threshold),
                                "value": float((closes[i] - middle[i]) / (band_width / 2)),
                                "timeframe": timeframe
                            }
                            
                            # Add signal
                            self._add_signal(signal)
                        
                        # Lower band touch/break
                        elif closes[i] <= (middle[i] - bollinger_threshold * band_width / 2):
                            signal_type = "buy"
                            signal_strength = min(1.0, (middle[i] - closes[i]) / (middle[i] - lower[i]))
                            
                            # Create signal
                            signal = {
                                "timestamp": int(timestamps[i]),
                                "symbol": symbol,
                                "type": signal_type,
                                "source": "bollinger_lower",
                                "strength": float(signal_strength),
                                "price": float(closes[i]),
                                "threshold": float(bollinger_threshold),
                                "value": float((middle[i] - closes[i]) / (band_width / 2)),
                                "timeframe": timeframe
                            }
                            
                            # Add signal
                            self._add_signal(signal)
            
            # Check MACD signals
            if all(k in indicators for k in ['macd', 'macd_signal', 'macd_hist']):
                macd = indicators['macd']
                signal_line = indicators['macd_signal']
                hist = indicators['macd_hist']
                
                macd_threshold = thresholds.get("macd", {}).get("current", 0.0002)
                
                for i in range(1, len(macd)):
                    # MACD crosses above signal line
                    if hist[i] > macd_threshold and hist[i] > 0 and hist[i-1] <= 0:
                        signal_type = "buy"
                        signal_strength = min(1.0, hist[i] / macd_threshold)
                        
                        # Create signal
                        signal = {
                            "timestamp": int(timestamps[i]),
                            "symbol": symbol,
                            "type": signal_type,
                            "source": "macd_cross_above",
                            "strength": float(signal_strength),
                            "price": float(closes[i]),
                            "threshold": float(macd_threshold),
                            "value": float(hist[i]),
                            "timeframe": timeframe
                        }
                        
                        # Add signal
                        self._add_signal(signal)
                    
                    # MACD crosses below signal line
                    elif hist[i] < -macd_threshold and hist[i] < 0 and hist[i-1] >= 0:
                        signal_type = "sell"
                        signal_strength = min(1.0, -hist[i] / macd_threshold)
                        
                        # Create signal
                        signal = {
                            "timestamp": int(timestamps[i]),
                            "symbol": symbol,
                            "type": signal_type,
                            "source": "macd_cross_below",
                            "strength": float(signal_strength),
                            "price": float(closes[i]),
                            "threshold": float(macd_threshold),
                            "value": float(hist[i]),
                            "timeframe": timeframe
                        }
                        
                        # Add signal
                        self._add_signal(signal)
            
        except Exception as e:
            logger.error(f"Error generating signals from indicators: {str(e)}")
