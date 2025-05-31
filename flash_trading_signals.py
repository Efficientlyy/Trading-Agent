#!/usr/bin/env python
"""
Flash Trading Signal Generator and Decision Engine

This module provides optimized signal generation and decision making
capabilities for flash trading, with a focus on ultra-low latency
and high-frequency trading strategies.
"""

import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
import threading
import json
import os
from optimized_mexc_client import OptimizedMexcClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("flash_trading_signals")

class MarketState:
    """Market state container with efficient updates and calculations"""
    
    def __init__(self, symbol, max_history=100):
        """Initialize market state for a symbol"""
        self.symbol = symbol
        self.max_history = max_history
        
        # Order book state
        self.bids = []
        self.asks = []
        self.bid_depth = 0
        self.ask_depth = 0
        self.spread = 0
        self.mid_price = 0
        self.last_order_book_update = 0
        
        # Price history
        self.price_history = deque(maxlen=max_history)
        self.volume_history = deque(maxlen=max_history)
        self.timestamp_history = deque(maxlen=max_history)
        
        # Derived metrics
        self.volatility = 0
        self.trend = 0
        self.momentum = 0
        self.order_imbalance = 0
        
        # Lock for thread safety
        self.lock = threading.RLock()
    
    def update_order_book(self, bids, asks, timestamp=None):
        """Update order book state with new data"""
        with self.lock:
            self.bids = bids
            self.asks = asks
            self.last_order_book_update = timestamp or time.time() * 1000
            
            if bids and asks:
                self.bid_depth = sum(float(qty) for _, qty in bids)
                self.ask_depth = sum(float(qty) for _, qty in asks)
                
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                
                self.spread = best_ask - best_bid
                self.mid_price = (best_bid + best_ask) / 2
                
                # Calculate order imbalance
                self.order_imbalance = (self.bid_depth - self.ask_depth) / (self.bid_depth + self.ask_depth)
    
    def update_price(self, price, volume=0, timestamp=None):
        """Update price history with new data point"""
        with self.lock:
            current_time = timestamp or time.time() * 1000
            
            self.price_history.append(float(price))
            self.volume_history.append(float(volume))
            self.timestamp_history.append(current_time)
            
            # Update derived metrics if we have enough history
            if len(self.price_history) >= 2:
                # Simple volatility (standard deviation of recent prices)
                if len(self.price_history) >= 10:
                    self.volatility = np.std(list(self.price_history)[-10:])
                
                # Trend (positive = up, negative = down)
                price_array = np.array(list(self.price_history))
                if len(price_array) >= 20:
                    self.trend = np.mean(price_array[-10:]) - np.mean(price_array[-20:-10])
                
                # Momentum (rate of change)
                self.momentum = self.price_history[-1] - self.price_history[-2]
    
    def to_dict(self):
        """Convert market state to dictionary for serialization"""
        with self.lock:
            return {
                "symbol": self.symbol,
                "timestamp": int(time.time() * 1000),
                "bid_price": float(self.bids[0][0]) if self.bids else None,
                "ask_price": float(self.asks[0][0]) if self.asks else None,
                "mid_price": self.mid_price,
                "spread": self.spread,
                "bid_depth": self.bid_depth,
                "ask_depth": self.ask_depth,
                "order_imbalance": self.order_imbalance,
                "volatility": self.volatility,
                "trend": self.trend,
                "momentum": self.momentum,
                "last_price": self.price_history[-1] if self.price_history else None,
                "last_volume": self.volume_history[-1] if self.volume_history else None
            }

class SignalGenerator:
    """Signal generator for flash trading strategies"""
    
    def __init__(self, client=None, env_path=None):
        """Initialize signal generator with API client"""
        if client:
            self.client = client
        else:
            self.client = OptimizedMexcClient(env_path=env_path)
        
        # Market state by symbol
        self.market_states = {}
        
        # Signal thresholds and parameters
        self.config = {
            "imbalance_threshold": 0.2,      # Order book imbalance threshold
            "volatility_threshold": 0.1,      # Price volatility threshold
            "momentum_threshold": 0.05,       # Price momentum threshold
            "min_spread_bps": 1.0,            # Minimum spread in basis points
            "max_spread_bps": 50.0,           # Maximum spread in basis points
            "min_order_size": 0.001,          # Minimum order size in BTC
            "max_position": 0.1,              # Maximum position size in BTC
            "take_profit_bps": 20.0,          # Take profit in basis points
            "stop_loss_bps": 10.0,            # Stop loss in basis points
            "order_book_depth": 10,           # Order book depth to monitor
            "update_interval_ms": 100,        # Market state update interval
            "signal_interval_ms": 50,         # Signal generation interval
            "use_cached_data": True,          # Use cached market data
            "cache_max_age_ms": 200           # Maximum age of cached data
        }
        
        # Active positions and orders
        self.positions = {}
        self.orders = {}
        
        # Signal history
        self.signals = []
        self.max_signals = 1000
        
        # Monitoring and statistics
        self.stats = {
            "signals_generated": 0,
            "orders_placed": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "profitable_trades": 0,
            "losing_trades": 0,
            "total_profit_loss": 0.0,
            "start_time": time.time(),
            "last_update": time.time()
        }
        
        # Background threads
        self.running = False
        self.update_thread = None
        self.signal_thread = None
    
    def get_or_create_market_state(self, symbol):
        """Get existing market state or create new one"""
        if symbol not in self.market_states:
            self.market_states[symbol] = MarketState(symbol)
        return self.market_states[symbol]
    
    def update_market_state(self, symbol):
        """Update market state with latest data"""
        try:
            # Get order book
            order_book = self.client.get_order_book(
                symbol, 
                limit=self.config["order_book_depth"],
                use_cache=self.config["use_cached_data"],
                max_age_ms=self.config["cache_max_age_ms"]
            )
            
            if order_book and 'bids' in order_book and 'asks' in order_book:
                market_state = self.get_or_create_market_state(symbol)
                market_state.update_order_book(order_book['bids'], order_book['asks'])
                
                # Get latest price
                if market_state.bids and market_state.asks:
                    # Use mid price from order book
                    latest_price = market_state.mid_price
                    market_state.update_price(latest_price)
                
                return True
            else:
                logger.warning(f"Invalid order book data for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating market state for {symbol}: {str(e)}")
            return False
    
    def generate_signals(self, symbol):
        """Generate trading signals based on market state"""
        try:
            if symbol not in self.market_states:
                logger.warning(f"No market state available for {symbol}")
                return []
            
            market_state = self.market_states[symbol]
            
            # Skip if we don't have enough data
            if not market_state.bids or not market_state.asks or len(market_state.price_history) < 10:
                return []
            
            signals = []
            
            # Order book imbalance signal
            if abs(market_state.order_imbalance) > self.config["imbalance_threshold"]:
                signal_type = "BUY" if market_state.order_imbalance > 0 else "SELL"
                signals.append({
                    "type": signal_type,
                    "source": "order_imbalance",
                    "strength": abs(market_state.order_imbalance),
                    "timestamp": int(time.time() * 1000),
                    "price": market_state.mid_price,
                    "symbol": symbol
                })
            
            # Momentum signal
            normalized_momentum = market_state.momentum / market_state.mid_price
            if abs(normalized_momentum) > self.config["momentum_threshold"]:
                signal_type = "BUY" if normalized_momentum > 0 else "SELL"
                signals.append({
                    "type": signal_type,
                    "source": "momentum",
                    "strength": abs(normalized_momentum),
                    "timestamp": int(time.time() * 1000),
                    "price": market_state.mid_price,
                    "symbol": symbol
                })
            
            # Volatility breakout signal
            if market_state.volatility > 0:
                normalized_volatility = market_state.volatility / market_state.mid_price
                if normalized_volatility > self.config["volatility_threshold"]:
                    # Determine direction based on recent trend
                    signal_type = "BUY" if market_state.trend > 0 else "SELL"
                    signals.append({
                        "type": signal_type,
                        "source": "volatility_breakout",
                        "strength": normalized_volatility,
                        "timestamp": int(time.time() * 1000),
                        "price": market_state.mid_price,
                        "symbol": symbol
                    })
            
            # Store signals
            if signals:
                self.signals.extend(signals)
                if len(self.signals) > self.max_signals:
                    self.signals = self.signals[-self.max_signals:]
                
                self.stats["signals_generated"] += len(signals)
                self.stats["last_update"] = time.time()
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {str(e)}")
            return []
    
    def make_trading_decision(self, symbol, signals):
        """Make trading decision based on signals and current positions"""
        if not signals:
            return None
        
        # Get current position
        current_position = self.positions.get(symbol, 0)
        
        # Aggregate signals by type
        buy_signals = [s for s in signals if s["type"] == "BUY"]
        sell_signals = [s for s in signals if s["type"] == "SELL"]
        
        # Calculate aggregate signal strength
        buy_strength = sum(s["strength"] for s in buy_signals)
        sell_strength = sum(s["strength"] for s in sell_signals)
        
        # No decision if signals are balanced or too weak
        if abs(buy_strength - sell_strength) < 0.1:
            return None
        
        # Determine decision direction
        if buy_strength > sell_strength:
            direction = "BUY"
            strength = buy_strength - sell_strength
        else:
            direction = "SELL"
            strength = sell_strength - buy_strength
        
        # Check position limits
        if direction == "BUY" and current_position >= self.config["max_position"]:
            return None
        
        if direction == "SELL" and current_position <= -self.config["max_position"]:
            return None
        
        # Calculate order size based on signal strength and position limits
        base_size = self.config["min_order_size"]
        max_additional = self.config["max_position"] - abs(current_position)
        size_factor = min(1.0, strength * 2)  # Scale by signal strength
        
        order_size = base_size + (max_additional * size_factor)
        order_size = max(self.config["min_order_size"], min(order_size, max_additional))
        
        # Round to appropriate precision
        order_size = round(order_size, 6)
        
        # Get market state for price information
        market_state = self.market_states.get(symbol)
        if not market_state:
            return None
        
        # Determine price based on direction
        if direction == "BUY":
            # Buy at best ask plus a small buffer
            price = float(market_state.asks[0][0]) * 1.001
        else:
            # Sell at best bid minus a small buffer
            price = float(market_state.bids[0][0]) * 0.999
        
        # Round price to appropriate precision
        price = round(price, 2)
        
        # Create decision
        decision = {
            "action": direction,
            "symbol": symbol,
            "size": order_size,
            "price": price,
            "order_type": "LIMIT",
            "time_in_force": "IOC",  # Immediate-or-Cancel for flash trading
            "timestamp": int(time.time() * 1000),
            "signal_count": len(signals),
            "signal_strength": strength
        }
        
        return decision
    
    def execute_decision(self, decision):
        """Execute trading decision by placing order"""
        if not decision:
            return None
        
        try:
            # Prepare order parameters
            order_params = {
                "symbol": decision["symbol"],
                "side": decision["action"],
                "type": decision["order_type"],  # Fixed: use 'type' instead of 'order_type'
                "timeInForce": decision["time_in_force"],
                "quantity": str(decision["size"]),
                "price": str(decision["price"]),
                "newClientOrderId": f"flash_{int(time.time() * 1000)}"
            }
            
            # Place order
            order_result = self.client.place_order(
                symbol=decision["symbol"],
                side=decision["action"],
                type=decision["order_type"],  # Fixed: use 'type' parameter
                timeInForce=decision["time_in_force"],
                quantity=str(decision["size"]),
                price=str(decision["price"]),
                newClientOrderId=f"flash_{int(time.time() * 1000)}"
            )
            
            if order_result:
                # Update statistics
                self.stats["orders_placed"] += 1
                self.stats["last_update"] = time.time()
                
                # Track order
                order_id = order_result.get("orderId")
                if order_id:
                    self.orders[order_id] = {
                        "id": order_id,
                        "symbol": decision["symbol"],
                        "side": decision["action"],
                        "quantity": decision["size"],
                        "price": decision["price"],
                        "status": "NEW",
                        "timestamp": int(time.time() * 1000),
                        "decision": decision
                    }
                
                logger.info(f"Order placed: {decision['action']} {decision['size']} {decision['symbol']} @ {decision['price']}")
                return order_result
            else:
                logger.warning(f"Failed to place order: {decision}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing decision: {str(e)}")
            return None
    
    def update_market_data_loop(self, symbols, interval_ms=100):
        """Background loop to update market data"""
        while self.running:
            try:
                for symbol in symbols:
                    self.update_market_state(symbol)
                
                # Sleep for interval
                time.sleep(interval_ms / 1000)
                
            except Exception as e:
                logger.error(f"Error in market data update loop: {str(e)}")
                time.sleep(1)  # Longer sleep on error
    
    def signal_generation_loop(self, symbols, interval_ms=50):
        """Background loop to generate signals and make decisions"""
        while self.running:
            try:
                for symbol in symbols:
                    # Generate signals
                    signals = self.generate_signals(symbol)
                    
                    if signals:
                        # Make trading decision
                        decision = self.make_trading_decision(symbol, signals)
                        
                        if decision:
                            # Execute decision
                            self.execute_decision(decision)
                
                # Sleep for interval
                time.sleep(interval_ms / 1000)
                
            except Exception as e:
                logger.error(f"Error in signal generation loop: {str(e)}")
                time.sleep(1)  # Longer sleep on error
    
    def start(self, symbols):
        """Start background processing for specified symbols"""
        if self.running:
            logger.warning("Signal generator already running")
            return False
        
        self.running = True
        
        # Start market data update thread
        self.update_thread = threading.Thread(
            target=self.update_market_data_loop,
            args=(symbols, self.config["update_interval_ms"]),
            daemon=True
        )
        self.update_thread.start()
        
        # Start signal generation thread
        self.signal_thread = threading.Thread(
            target=self.signal_generation_loop,
            args=(symbols, self.config["signal_interval_ms"]),
            daemon=True
        )
        self.signal_thread.start()
        
        logger.info(f"Signal generator started for symbols: {symbols}")
        return True
    
    def stop(self):
        """Stop background processing"""
        self.running = False
        
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
            self.update_thread = None
        
        if self.signal_thread:
            self.signal_thread.join(timeout=2.0)
            self.signal_thread = None
        
        logger.info("Signal generator stopped")
    
    def get_market_state(self, symbol):
        """Get current market state for a symbol"""
        if symbol in self.market_states:
            return self.market_states[symbol].to_dict()
        return None
    
    def get_recent_signals(self, limit=10):
        """Get recent trading signals"""
        return self.signals[-limit:] if self.signals else []
    
    def get_statistics(self):
        """Get performance statistics"""
        stats = self.stats.copy()
        stats["uptime_seconds"] = time.time() - stats["start_time"]
        stats["signals_per_second"] = stats["signals_generated"] / max(1, stats["uptime_seconds"])
        stats["orders_per_second"] = stats["orders_placed"] / max(1, stats["uptime_seconds"])
        return stats
    
    def save_state(self, filename="signal_generator_state.json"):
        """Save current state to file"""
        state = {
            "timestamp": int(time.time() * 1000),
            "config": self.config,
            "stats": self.get_statistics(),
            "signals": self.get_recent_signals(100),
            "orders": list(self.orders.values())
        }
        
        with open(filename, "w") as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"State saved to {filename}")
    
    def load_config(self, filename):
        """Load configuration from file"""
        try:
            with open(filename, "r") as f:
                config = json.load(f)
            
            self.config.update(config)
            logger.info(f"Configuration loaded from {filename}")
            return True
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Flash Trading Signal Generator')
    parser.add_argument('--env', default=".env-secure/.env", help='Path to .env file')
    parser.add_argument('--symbol', default="BTCUSDT", help='Symbol to trade')
    parser.add_argument('--test', action='store_true', help='Run in test mode (no orders)')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    
    args = parser.parse_args()
    
    # Create client and signal generator
    client = OptimizedMexcClient(env_path=args.env)
    signal_gen = SignalGenerator(client=client)
    
    if args.test:
        print(f"Running in test mode for {args.duration} seconds...")
        
        # Override config for testing
        signal_gen.config["use_cached_data"] = True
        signal_gen.config["update_interval_ms"] = 500
        signal_gen.config["signal_interval_ms"] = 1000
        
        # Start signal generator
        signal_gen.start([args.symbol])
        
        # Run for specified duration
        start_time = time.time()
        try:
            while time.time() - start_time < args.duration:
                # Print market state every 5 seconds
                if int(time.time() - start_time) % 5 == 0:
                    market_state = signal_gen.get_market_state(args.symbol)
                    if market_state:
                        print(f"\nMarket State ({args.symbol}):")
                        print(f"  Bid/Ask: {market_state['bid_price']}/{market_state['ask_price']}")
                        print(f"  Spread: {market_state['spread']:.6f} ({market_state['spread']/market_state['mid_price']*10000:.2f} bps)")
                        print(f"  Imbalance: {market_state['order_imbalance']:.4f}")
                        print(f"  Volatility: {market_state['volatility']:.6f}")
                        print(f"  Trend: {market_state['trend']:.6f}")
                    
                    # Print recent signals
                    signals = signal_gen.get_recent_signals(5)
                    if signals:
                        print("\nRecent Signals:")
                        for signal in reversed(signals):
                            print(f"  {signal['type']} ({signal['source']}) - Strength: {signal['strength']:.4f}")
                    
                    # Print statistics
                    stats = signal_gen.get_statistics()
                    print(f"\nStatistics:")
                    print(f"  Signals: {stats['signals_generated']} ({stats['signals_per_second']:.2f}/s)")
                    print(f"  Orders: {stats['orders_placed']}")
                    print(f"  Uptime: {stats['uptime_seconds']:.1f}s")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        
        # Stop signal generator
        signal_gen.stop()
        
        # Print final statistics
        stats = signal_gen.get_statistics()
        print(f"\nFinal Statistics:")
        print(f"  Signals Generated: {stats['signals_generated']}")
        print(f"  Orders Placed: {stats['orders_placed']}")
        print(f"  Signals/Second: {stats['signals_per_second']:.2f}")
        print(f"  Orders/Second: {stats['orders_per_second']:.2f}")
        print(f"  Uptime: {stats['uptime_seconds']:.1f}s")
        
        # Save state
        signal_gen.save_state()
    
    else:
        print("This script should be run with --test flag for safety")
    
    # Clean up
    client.close()
