"""
Flash Trading Signals with Session Awareness

This module provides signal generation and decision making for flash trading,
with dynamic adaptation based on the current global trading session.
"""

import time
import logging
import json
import os
import uuid
import numpy as np
from datetime import datetime, timezone
from threading import Thread, Event, RLock
from collections import deque
from optimized_mexc_client import OptimizedMexcClient
from trading_session_manager import TradingSessionManager
from error_handling_utils import safe_get, safe_get_nested, validate_api_response, handle_api_error, log_exception, parse_float_safely

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("flash_trading_signals")

class MarketState:
    """Represents the current state of a market for signal generation"""
    
    def __init__(self, symbol):
        """Initialize market state for a symbol"""
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
        
        # Price history
        self.price_history = deque(maxlen=100)
        self.volume_history = deque(maxlen=100)
        self.timestamp_history = deque(maxlen=100)
        
        # Derived metrics
        self.momentum = 0.0
        self.volatility = 0.0
        self.trend = 0.0
        
        # Trading metrics
        self.last_trade_price = None
        self.last_trade_side = None
        self.last_trade_time = None
    
    def update_order_book(self, bids, asks):
        """Update order book state"""
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
        
        # Update price history
        self.price_history.append(self.mid_price)
        self.timestamp_history.append(self.timestamp)
        
        # Calculate derived metrics if we have enough history
        if len(self.price_history) >= 10:
            self._calculate_derived_metrics()
        
        return True
    
    def _calculate_derived_metrics(self):
        """Calculate derived metrics from price history"""
        # Convert to numpy array for calculations
        prices = np.array(list(self.price_history))
        
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

class FlashTradingSignals:
    """Signal generation and decision making for flash trading"""
    
    def __init__(self, client_instance=None, api_key=None, api_secret=None, env_path=None):
        """Initialize flash trading signals
        
        Args:
            client_instance: Existing OptimizedMexcClient instance to use (preferred)
            api_key: API key for MEXC (used only if client_instance is None)
            api_secret: API secret for MEXC (used only if client_instance is None)
            env_path: Path to .env file (used only if client_instance is None)
        """
        # Use existing client instance if provided, otherwise create new one
        if client_instance is not None and isinstance(client_instance, OptimizedMexcClient):
            self.api_client = client_instance
            logger.info("Using provided client instance for SignalGenerator")
        else:
            # Initialize API client with direct credentials
            self.api_client = OptimizedMexcClient(api_key, api_secret, env_path)
            logger.info("Created new client instance for SignalGenerator")
        
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
            
        # Store symbols
        self.symbols = symbols
        
        # Reset stop event
        self.stop_event = Event()
        
        # Start update thread
        self.update_thread = Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Set running state
        self.running = True
        
        logger.info(f"Signal generator started for symbols: {symbols}")
        return True
        
    @handle_api_error
    def stop(self):
        """Stop signal generation
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.running:
            logger.warning("Signal generator not running")
            return False
            
        # Signal thread to stop
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)
            
        # Set running state
        self.running = False
        
        logger.info("Signal generator stopped")
        return True
        
    @handle_api_error
    def _update_loop(self):
        """Background thread for updating market states"""
        try:
            logger.info("Market state update loop started")
            
            while not self.stop_event.is_set():
                try:
                    # Update market state for each symbol
                    for symbol in self.symbols:
                        if self.stop_event.is_set():
                            break
                            
                        try:
                            self._update_market_state(symbol)
                        except Exception as e:
                            log_exception(e, f"_update_market_state for {symbol}")
                            
                    # Sleep for a short interval
                    self.stop_event.wait(0.5)
                    
                except Exception as e:
                    log_exception(e, "_update_loop iteration")
                    # Sleep before retrying
                    self.stop_event.wait(1.0)
                    
        except Exception as e:
            log_exception(e, "_update_loop")
        finally:
            logger.info("Market state update loop stopped")
    @handle_api_error
    def _update_market_state(self, symbol):
        """Update market state for a symbol with thread safety and robust validation
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDC')
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        # DIAGNOSTIC: Log before API call
        logger.debug(f"Updating market state for {symbol}")
        
        try:
            # Thread-safe client access
            with self.client_lock:
                # Get order book with diagnostics
                logger.debug(f"Requesting order book for {symbol}")
                order_book = self.api_client.get_order_book(symbol, limit=20)
                logger.debug(f"Order book response type: {type(order_book)}")
            
            # Validate order book structure with diagnostics
            if order_book is None:
                logger.error(f"CRITICAL: Null order book response for {symbol}")
                return False
                
            if not isinstance(order_book, dict):
                logger.error(f"Invalid order book response type for {symbol}: {type(order_book)}")
                return False
                
            if 'bids' not in order_book or 'asks' not in order_book:
                logger.error(f"Missing bids or asks in order book for {symbol}")
                return False
            
            # Safe access with validation
            bids = safe_get(order_book, "bids", [])
            asks = safe_get(order_book, "asks", [])
            
            if not bids or not asks:
                logger.warning(f"Empty bids or asks in order book for {symbol}")
                return False
            
            # Validate bid/ask structure with diagnostics
            try:
                # Validate at least one valid bid and ask
                if len(bids) == 0 or len(asks) == 0:
                    logger.warning(f"No bids or asks available for {symbol}")
                    return False
                
                # Validate bid/ask format
                if not isinstance(bids[0], list) or len(bids[0]) < 2:
                    logger.error(f"Invalid bid format for {symbol}: {bids[0]}")
                    return False
                
                if not isinstance(asks[0], list) or len(asks[0]) < 2:
                    logger.error(f"Invalid ask format for {symbol}: {asks[0]}")
                    return False
                
                # Thread-safe market state update
                with self.market_state_lock:
                    # Create market state if it doesn't exist
                    if symbol not in self.market_states:
                        self.market_states[symbol] = MarketState(symbol)
                    
                    # Update market state
                    return self.market_states[symbol].update_order_book(bids, asks)
            except Exception as e:
                log_exception(e, f"_update_market_state validation for {symbol}")
                return False
                
        except Exception as e:
            log_exception(e, f"_update_market_state for {symbol}")
            return False
    
    @handle_api_error
    def generate_signals(self, symbol):
        """Generate trading signals for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDC')
            
        Returns:
            list: List of trading signals
        """
        # DIAGNOSTIC: Log before generating signals
        logger.debug(f"Generating signals for {symbol}")
        
        try:
            # Update market state
            if not self._update_market_state(symbol):
                logger.warning(f"Failed to update market state for {symbol}")
                return []
            
            # Get current trading session
            current_session = self.session_manager.get_current_session_name()
            
            # Thread-safe market state access
            with self.market_state_lock:
                if symbol not in self.market_states:
                    logger.warning(f"No market state available for {symbol}")
                    return []
                
                market_state = self.market_states[symbol]
            
            # Get session-specific parameters
            session_params = self.session_manager.get_session_parameters(current_session)
            
            # Extract thresholds from session parameters with safe defaults
            imbalance_threshold = safe_get(session_params, "imbalance_threshold", 0.2)
            momentum_threshold = safe_get(session_params, "momentum_threshold", 0.005)
            volatility_threshold = safe_get(session_params, "volatility_threshold", 0.002)
            
            signals = []
            
            # Order imbalance signal
            if abs(market_state.order_imbalance) > imbalance_threshold:
                signal_type = "BUY" if market_state.order_imbalance > 0 else "SELL"
                signals.append({
                    "type": signal_type,
                    "source": "order_imbalance",
                    "strength": abs(market_state.order_imbalance),
                    "timestamp": int(time.time() * 1000),
                    "price": market_state.mid_price,
                    "symbol": symbol,
                    "session": current_session
                })
            
            # Momentum signal
            normalized_momentum = market_state.momentum / market_state.mid_price
            if abs(normalized_momentum) > momentum_threshold:
                signal_type = "BUY" if normalized_momentum > 0 else "SELL"
                signals.append({
                    "type": signal_type,
                    "source": "momentum",
                    "strength": abs(normalized_momentum),
                    "timestamp": int(time.time() * 1000),
                    "price": market_state.mid_price,
                    "symbol": symbol,
                    "session": current_session
                })
            
            # Volatility breakout signal
            if market_state.volatility > 0:
                normalized_volatility = market_state.volatility / market_state.mid_price
                if normalized_volatility > volatility_threshold:
                    # Determine direction based on recent trend
                    signal_type = "BUY" if market_state.trend > 0 else "SELL"
                    signals.append({
                        "type": signal_type,
                        "source": "volatility_breakout",
                        "strength": normalized_volatility,
                        "timestamp": int(time.time() * 1000),
                        "price": market_state.mid_price,
                        "symbol": symbol,
                        "session": current_session
                    })
            
            return signals
            
        except Exception as e:
            log_exception(e, f"generate_signals for {symbol}")
            return []
    
    @handle_api_error
    def get_recent_signals(self, count=10, symbol=None):
        """Get recent signals, optionally filtered by symbol"""
        try:
            # Thread-safe signals access
            signals_copy = self.signals.copy()
            
            # Filter by symbol if provided
            if symbol:
                signals_copy = [s for s in signals_copy if s.get("symbol") == symbol]
            
            # Return most recent signals
            return signals_copy[-count:]
        except Exception as e:
            log_exception(e, "get_recent_signals")
            return []
    
    @handle_api_error
    def add_signal(self, signal):
        """Add a new signal to history"""
        try:
            # Thread-safe signals update
            self.signals.append(signal)
            
            # Trim signal history if needed
            if len(self.signals) > self.max_signals:
                self.signals = self.signals[-self.max_signals:]
        except Exception as e:
            log_exception(e, "add_signal")
    
    @handle_api_error
    def make_trading_decision(self, symbol, signals):
        """Make a trading decision based on signals
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDC')
            signals: List of trading signals
            
        Returns:
            dict: Trading decision or None if no decision
        """
        # DIAGNOSTIC: Log before making decision
        logger.debug(f"Making trading decision for {symbol} with {len(signals)} signals")
        
        try:
            # Validate inputs with diagnostics
            if not symbol or not isinstance(symbol, str):
                logger.error(f"Invalid symbol: {symbol}")
                return None
                
            if not signals or not isinstance(signals, list):
                logger.debug(f"No valid signals for {symbol}")
                return None
            
            # Filter signals for this symbol and session
            current_session = self.session_manager.get_current_session_name()
            valid_signals = [
                s for s in signals 
                if s.get("symbol") == symbol and s.get("session") == current_session
            ]
            
            if not valid_signals:
                logger.debug(f"No valid signals for {symbol} in session {current_session}")
                return None
            
            # Get session-specific parameters
            session_params = self.session_manager.get_session_parameters(current_session)
            
            # Extract decision parameters with safe defaults
            min_signal_strength = safe_get(session_params, "min_signal_strength", 0.1)
            position_size = safe_get(session_params, "position_size", 0.1)
            
            # Calculate aggregate signal
            buy_strength = sum(s.get("strength", 0) for s in valid_signals if s.get("type") == "BUY")
            sell_strength = sum(s.get("strength", 0) for s in valid_signals if s.get("type") == "SELL")
            
            # Determine decision
            if buy_strength > sell_strength and buy_strength >= min_signal_strength:
                # Thread-safe market state access
                with self.market_state_lock:
                    if symbol not in self.market_states:
                        logger.warning(f"No market state available for {symbol}")
                        return None
                    
                    price = self.market_states[symbol].ask_price
                
                return {
                    "symbol": symbol,
                    "side": "BUY",
                    "order_type": "MARKET",
                    "size": position_size,
                    "price": price,
                    "time_in_force": "GTC",
                    "timestamp": int(time.time() * 1000),
                    "session": current_session,
                    "signal_strength": buy_strength
                }
            elif sell_strength > buy_strength and sell_strength >= min_signal_strength:
                # Thread-safe market state access
                with self.market_state_lock:
                    if symbol not in self.market_states:
                        logger.warning(f"No market state available for {symbol}")
                        return None
                    
                    price = self.market_states[symbol].bid_price
                
                return {
                    "symbol": symbol,
                    "side": "SELL",
                    "order_type": "MARKET",
                    "size": position_size,
                    "price": price,
                    "time_in_force": "GTC",
                    "timestamp": int(time.time() * 1000),
                    "session": current_session,
                    "signal_strength": sell_strength
                }
            
            return None
            
        except Exception as e:
            log_exception(e, f"make_trading_decision for {symbol}")
            return None
    
    def get_account(self):
        """Compatibility method for extended_testing.py - calls get_account_info()"""
        return self.get_account_info()
    
    def get_account_info(self):
        """Get account information (placeholder for paper trading)"""
        return {
            "balances": {
                "USDC": 10000.0,
                "BTC": 0.5,
                "ETH": 5.0
            }
        }
