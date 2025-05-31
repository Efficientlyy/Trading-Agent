#!/usr/bin/env python
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
        self.momentum = prices[-1] - prices[0]
        
        # Calculate volatility (standard deviation)
        self.volatility = np.std(prices)
        
        # Calculate trend (positive = uptrend, negative = downtrend)
        if len(prices) >= 20:
            short_ma = np.mean(prices[-10:])
            long_ma = np.mean(prices[-20:])
            self.trend = short_ma - long_ma
        else:
            self.trend = self.momentum
    
    def update_trade(self, price, side, timestamp=None):
        """Update last trade information"""
        self.last_trade_price = price
        self.last_trade_side = side
        self.last_trade_time = timestamp or int(time.time() * 1000)


class SignalGenerator:
    """Generates trading signals based on market state with session awareness"""
    
    def __init__(self, client=None, env_path=None, config=None):
        """Initialize signal generator with API client and configuration"""
        self.client = client or OptimizedMexcClient(env_path=env_path)
        self.config = {
            "imbalance_threshold": 0.2,
            "volatility_threshold": 0.1,
            "momentum_threshold": 0.05,
            "min_spread_bps": 1.0,
            "max_spread_bps": 50.0,
            "order_book_depth": 10,
            "update_interval_ms": 100,
            "signal_interval_ms": 50,
            "use_cached_data": True,
            "cache_max_age_ms": 200
        }
        
        if config:
            self.config.update(config)
        
        # Market states by symbol
        self.market_states = {}
        
        # Recent signals
        self.recent_signals = deque(maxlen=1000)
        
        # Running state
        self.running = False
        self.stop_event = Event()
        self.update_thread = None
        self.signal_thread = None
        
        # Lock for thread safety
        self.lock = RLock()
        
        # Session manager for session-aware parameters
        self.session_manager = TradingSessionManager()
        
        # Statistics
        self.stats = {
            "updates_processed": 0,
            "signals_generated": 0,
            "decisions_made": 0,
            "orders_placed": 0,
            "errors": 0
        }
    
    def start(self, symbols):
        """Start signal generation for specified symbols"""
        if self.running:
            logger.warning("Signal generator already running")
            return False
        
        with self.lock:
            # Initialize market states
            for symbol in symbols:
                self.market_states[symbol] = MarketState(symbol)
            
            # Reset stop event
            self.stop_event.clear()
            
            # Start update thread
            self.update_thread = Thread(target=self._update_loop, args=(symbols,))
            self.update_thread.daemon = True
            self.update_thread.start()
            
            # Start signal thread
            self.signal_thread = Thread(target=self._signal_loop, args=(symbols,))
            self.signal_thread.daemon = True
            self.signal_thread.start()
            
            # Set running state
            self.running = True
            
            logger.info(f"Signal generator started for symbols: {symbols}")
            return True
    
    def stop(self):
        """Stop signal generation"""
        if not self.running:
            logger.warning("Signal generator not running")
            return False
        
        with self.lock:
            # Set stop event
            self.stop_event.set()
            
            # Wait for threads to stop
            if self.update_thread:
                self.update_thread.join(timeout=2.0)
            
            if self.signal_thread:
                self.signal_thread.join(timeout=2.0)
            
            # Reset running state
            self.running = False
            
            # Save state
            self._save_state()
            
            logger.info("Signal generator stopped")
            return True
    
    def _update_loop(self, symbols):
        """Background thread for updating market states"""
        while not self.stop_event.is_set():
            try:
                # Update market states for all symbols
                for symbol in symbols:
                    self._update_market_state(symbol)
                
                # Update statistics
                self.stats["updates_processed"] += 1
                
                # Sleep for update interval
                self.stop_event.wait(self.config["update_interval_ms"] / 1000)
                
            except Exception as e:
                logger.error(f"Error in update loop: {str(e)}")
                self.stats["errors"] += 1
                self.stop_event.wait(1.0)  # Sleep longer on error
    
    def _signal_loop(self, symbols):
        """Background thread for generating signals"""
        while not self.stop_event.is_set():
            try:
                # Generate signals for all symbols
                for symbol in symbols:
                    signals = self.generate_signals(symbol)
                    
                    # Add signals to recent signals
                    with self.lock:
                        for signal in signals:
                            self.recent_signals.append(signal)
                            self.stats["signals_generated"] += 1
                
                # Sleep for signal interval
                self.stop_event.wait(self.config["signal_interval_ms"] / 1000)
                
            except Exception as e:
                logger.error(f"Error in signal loop: {str(e)}")
                self.stats["errors"] += 1
                self.stop_event.wait(1.0)  # Sleep longer on error
    
    def _update_market_state(self, symbol):
        """Update market state for a symbol"""
        try:
            # Get order book
            order_book = self.client.get_order_book(
                symbol, 
                limit=self.config["order_book_depth"],
                use_cache=self.config["use_cached_data"],
                max_age_ms=self.config["cache_max_age_ms"]
            )
            
            if order_book and 'bids' in order_book and 'asks' in order_book:
                with self.lock:
                    if symbol not in self.market_states:
                        self.market_states[symbol] = MarketState(symbol)
                    
                    self.market_states[symbol].update_order_book(
                        order_book["bids"],
                        order_book["asks"]
                    )
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating market state for {symbol}: {str(e)}")
            return False
    
    def generate_signals(self, symbol):
        """Generate trading signals based on market state and current session"""
        try:
            with self.lock:
                if symbol not in self.market_states:
                    logger.warning(f"No market state available for {symbol}")
                    return []
                
                market_state = self.market_states[symbol]
            
            # Skip if we don't have enough data
            if not market_state.bids or not market_state.asks or len(market_state.price_history) < 10:
                return []
            
            # Get current trading session
            current_session = self.session_manager.get_current_session_name()
            
            # Get session-specific parameters
            session_params = self.session_manager.get_session_parameter
            
            # Use session-specific thresholds or fall back to defaults
            imbalance_threshold = session_params("imbalance_threshold", self.config["imbalance_threshold"])
            volatility_threshold = session_params("volatility_threshold", self.config["volatility_threshold"])
            momentum_threshold = session_params("momentum_threshold", self.config["momentum_threshold"])
            min_spread_bps = session_params("min_spread_bps", self.config["min_spread_bps"])
            max_spread_bps = session_params("max_spread_bps", self.config["max_spread_bps"])
            
            signals = []
            
            # Check if spread is within acceptable range
            if market_state.spread_bps < min_spread_bps or market_state.spread_bps > max_spread_bps:
                return []
            
            # Order book imbalance signal
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
            logger.error(f"Error generating signals for {symbol}: {str(e)}")
            return []
    
    def get_recent_signals(self, count=10, symbol=None):
        """Get recent signals, optionally filtered by symbol"""
        with self.lock:
            if symbol:
                filtered = [s for s in self.recent_signals if s["symbol"] == symbol]
                return list(filtered)[-count:]
            else:
                return list(self.recent_signals)[-count:]
    
    def make_trading_decision(self, symbol, signals=None):
        """Make trading decision based on signals and current session"""
        try:
            # Get signals if not provided
            if signals is None:
                signals = self.get_recent_signals(10, symbol)
            
            if not signals:
                return None
            
            # Get current trading session
            current_session = self.session_manager.get_current_session_name()
            logger.info(f"Making decision for session: {current_session}")
            
            # Get session-specific parameters directly from session_parameters
            session_params = self.session_manager.get_all_session_parameters()
            
            # Use session-specific parameters with explicit fallbacks
            position_size_factor = session_params.get("position_size_factor", 1.0)
            take_profit_bps = session_params.get("take_profit_bps", 20.0)
            stop_loss_bps = session_params.get("stop_loss_bps", 10.0)
            time_in_force = session_params.get("time_in_force", "IOC")
            
            # Log the session-specific parameters being used
            logger.info(f"Using session parameters: position_size_factor={position_size_factor}, "
                       f"take_profit_bps={take_profit_bps}, stop_loss_bps={stop_loss_bps}")
            
            # Group signals by type
            buy_signals = [s for s in signals if s["type"] == "BUY"]
            sell_signals = [s for s in signals if s["type"] == "SELL"]
            
            # Calculate aggregate strength
            buy_strength = sum(s["strength"] for s in buy_signals)
            sell_strength = sum(s["strength"] for s in sell_signals)
            
            # Get market state
            with self.lock:
                if symbol not in self.market_states:
                    return None
                market_state = self.market_states[symbol]
            
            # Make decision
            decision = None
            
            # Adjust thresholds based on session
            buy_threshold = 0.5
            sell_threshold = 0.5
            strength_ratio = 1.5
            
            if current_session == "ASIA":
                # More conservative in Asian session due to higher volatility
                buy_threshold = 0.6
                sell_threshold = 0.6
                strength_ratio = 1.8
            elif current_session == "US":
                # More aggressive in US session due to higher liquidity
                buy_threshold = 0.4
                sell_threshold = 0.4
                strength_ratio = 1.3
            
            # BUY decision
            if buy_strength > buy_threshold and buy_strength > sell_strength * strength_ratio:
                # Calculate base size
                base_size = 0.1 * min(buy_strength, 10.0)
                
                # Apply session-specific position size factor
                size = base_size * position_size_factor
                
                # Round to appropriate precision
                size = round(size, 6)
                
                decision = {
                    "action": "BUY",
                    "symbol": symbol,
                    "size": size,
                    "price": market_state.bid_price,  # Place at bid for limit orders
                    "order_type": "LIMIT",
                    "time_in_force": time_in_force,
                    "timestamp": int(time.time() * 1000),
                    "signal_count": len(buy_signals),
                    "signal_strength": buy_strength,
                    "session": current_session,
                    "position_size_factor": position_size_factor  # Include for debugging
                }
            
            # SELL decision
            elif sell_strength > sell_threshold and sell_strength > buy_strength * strength_ratio:
                # Calculate base size
                base_size = 0.1 * min(sell_strength, 10.0)
                
                # Apply session-specific position size factor
                size = base_size * position_size_factor
                
                # Round to appropriate precision
                size = round(size, 6)
                
                decision = {
                    "action": "SELL",
                    "symbol": symbol,
                    "size": size,
                    "price": market_state.ask_price,  # Place at ask for limit orders
                    "order_type": "LIMIT",
                    "time_in_force": time_in_force,
                    "timestamp": int(time.time() * 1000),
                    "signal_count": len(sell_signals),
                    "signal_strength": sell_strength,
                    "session": current_session,
                    "position_size_factor": position_size_factor  # Include for debugging
                }
            
            # Update statistics
            if decision:
                self.stats["decisions_made"] += 1
                logger.info(f"Decision made: {decision['action']} {decision['size']} {symbol} with factor {position_size_factor}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making trading decision for {symbol}: {str(e)}")
            return None
    
    def place_order(self, decision):
        """Place an order based on trading decision"""
        try:
            if not decision:
                return None
            
            # Extract decision details
            symbol = decision["symbol"]
            side = decision["action"]
            order_type = decision["order_type"]
            quantity = decision["size"]
            price = decision["price"]
            time_in_force = decision["time_in_force"]
            
            # Place order
            response = self.client.create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=price,
                timeInForce=time_in_force
            )
            
            if response and "orderId" in response:
                # Update statistics
                self.stats["orders_placed"] += 1
                
                # Log order
                logger.info(f"Order placed: {side} {quantity} {symbol} @ {price}")
                
                return response
            else:
                logger.warning(f"Order placement failed: {response}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to place order: {str(e)}")
            return None
    
    def _save_state(self, filename="signal_generator_state.json"):
        """Save current state to file"""
        try:
            state = {
                "timestamp": int(time.time() * 1000),
                "stats": self.stats,
                "recent_signals": list(self.recent_signals)[-100:],  # Save last 100 signals
                "config": self.config
            }
            
            with open(filename, "w") as f:
                json.dump(state, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Flash Trading Signals')
    parser.add_argument('--env', default=".env-secure/.env", help='Path to .env file')
    parser.add_argument('--test', action='store_true', help='Run test mode')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    parser.add_argument('--symbols', default="BTCUSDC,ETHUSDC", help='Comma-separated list of symbols')
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = args.symbols.split(",")
    
    # Create signal generator
    signal_generator = SignalGenerator(env_path=args.env)
    
    # Print current session
    current_session = signal_generator.session_manager.get_current_session_name()
    print(f"Current trading session: {current_session}")
    
    # Print session parameters
    session_params = signal_generator.session_manager.get_all_session_parameters()
    print("Session parameters:")
    for param, value in session_params.items():
        print(f"  {param}: {value}")
    
    # Run test if requested
    if args.test:
        print(f"Running signal generator test for {args.duration} seconds...")
        
        # Start signal generator
        signal_generator.start(symbols)
        
        try:
            # Run for specified duration
            start_time = time.time()
            end_time = start_time + args.duration
            
            while time.time() < end_time:
                # Sleep for a bit
                time.sleep(1)
                
                # Print status every 5 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 5 == 0:
                    # Get recent signals
                    signals = signal_generator.get_recent_signals(10)
                    
                    # Print status
                    print(f"\nElapsed: {elapsed:.1f}s, Signals: {len(signals)}")
                    
                    # Make trading decisions
                    for symbol in symbols:
                        symbol_signals = [s for s in signals if s["symbol"] == symbol]
                        if symbol_signals:
                            decision = signal_generator.make_trading_decision(symbol, symbol_signals)
                            if decision:
                                print(f"Decision for {symbol}: {decision['action']} {decision['size']} @ {decision['price']}")
                                
                                # Place order in test mode
                                response = signal_generator.place_order(decision)
                                if response:
                                    print(f"Order placed: {response.get('orderId')}")
        
        finally:
            # Stop signal generator
            signal_generator.stop()
            
            # Print statistics
            print("\nSignal Generator Statistics:")
            for stat, value in signal_generator.stats.items():
                print(f"  {stat}: {value}")
