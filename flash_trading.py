#!/usr/bin/env python
"""
Flash Trading Integration Module

This module integrates the optimized MEXC client, paper trading system,
and signal generation engine for flash trading with zero-fee pairs.
"""

import time
import logging
import argparse
import json
import os
from datetime import datetime
from optimized_mexc_client import OptimizedMexcClient
from paper_trading import PaperTradingSystem
from flash_trading_config import FlashTradingConfig
from flash_trading_signals import SignalGenerator, MarketState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("flash_trading")

class FlashTradingSystem:
    """Flash Trading System integrating all components"""
    
    def __init__(self, env_path=None, config_path=None):
        """Initialize the flash trading system"""
        # Load configuration
        self.config = FlashTradingConfig(config_path)
        
        # Create API client
        self.client = OptimizedMexcClient(env_path=env_path)
        
        # Create paper trading system
        self.paper_trading = PaperTradingSystem(self.client, self.config)
        
        # Create signal generator
        self.signal_generator = SignalGenerator(self.client, env_path)
        
        # Configure signal generator with our settings
        signal_config = self.config.get_signal_config()
        self.signal_generator.config.update(signal_config)
        
        # Running state
        self.running = False
        self.start_time = None
        self.stats = {
            "signals_generated": 0,
            "orders_placed": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "profitable_trades": 0,
            "losing_trades": 0,
            "total_profit_loss": 0.0
        }
    
    def start(self):
        """Start the flash trading system"""
        if self.running:
            logger.warning("Flash trading system already running")
            return False
        
        # Get enabled trading pairs
        trading_pairs = self.config.get_enabled_trading_pairs()
        if not trading_pairs:
            logger.error("No enabled trading pairs found")
            return False
        
        # Extract symbols
        symbols = [pair["symbol"] for pair in trading_pairs]
        
        # Start signal generator
        self.signal_generator.start(symbols)
        
        # Set running state
        self.running = True
        self.start_time = time.time()
        
        logger.info(f"Flash trading system started for symbols: {symbols}")
        return True
    
    def stop(self):
        """Stop the flash trading system"""
        if not self.running:
            logger.warning("Flash trading system not running")
            return False
        
        # Stop signal generator
        self.signal_generator.stop()
        
        # Set running state
        self.running = False
        
        logger.info("Flash trading system stopped")
        return True
    
    def process_signals_and_execute(self):
        """Process signals and execute trades using paper trading"""
        if not self.running:
            logger.warning("Flash trading system not running")
            return False
        
        # Get enabled trading pairs with robust error handling
        try:
            trading_pairs = self.config.get_enabled_trading_pairs()
            if not trading_pairs:
                logger.warning("No enabled trading pairs found in configuration")
                return False
                
            if not isinstance(trading_pairs, list):
                logger.error(f"Invalid trading pairs format: {type(trading_pairs)}, expected list")
                return False
        except Exception as e:
            logger.error(f"Error getting enabled trading pairs: {str(e)}")
            return False
        
        # Process each trading pair with validation
        for pair_config in trading_pairs:
            try:
                # Validate pair configuration
                if not isinstance(pair_config, dict):
                    logger.error(f"Invalid pair configuration format: {type(pair_config)}, expected dict")
                    continue
                    
                symbol = pair_config.get("symbol")
                if not symbol or not isinstance(symbol, str):
                    logger.error(f"Invalid symbol in pair configuration: {symbol}")
                    continue
                
                # Get recent signals with validation
                signals = self.signal_generator.get_recent_signals(10)
                if signals is None:
                    logger.warning(f"Failed to get recent signals for {symbol}")
                    continue
                    
                if not isinstance(signals, list):
                    logger.error(f"Invalid signals format: {type(signals)}, expected list")
                    continue
                
                # Filter signals for current symbol
                signals = [s for s in signals if isinstance(s, dict) and s.get("symbol") == symbol]
                
                if not signals:
                    logger.debug(f"No signals found for {symbol}")
                    continue
                
                # Make trading decision with validation
                try:
                    decision = self.signal_generator.make_trading_decision(symbol, signals)
                    
                    if decision:
                        # Execute with paper trading
                        self._execute_paper_trading_decision(decision)
                except Exception as e:
                    logger.error(f"Error making trading decision for {symbol}: {str(e)}")
                    continue
                
            except Exception as e:
                logger.error(f"Error processing trading pair {pair_config}: {str(e)}")
                continue
        
        # Process open paper trading orders with error handling
        try:
            self.paper_trading.process_open_orders()
        except Exception as e:
            logger.error(f"Error processing open orders: {str(e)}")
        
        return True
    
    def _execute_paper_trading_decision(self, decision):
        """Execute trading decision using paper trading"""
        try:
            # Validate decision object
            if not isinstance(decision, dict):
                logger.error(f"Invalid decision object: {decision}")
                return None
                
            # Extract and validate required fields with robust error handling
            try:
                symbol = decision["symbol"]
                if not symbol or not isinstance(symbol, str):
                    logger.error(f"Invalid symbol in decision: {symbol}")
                    return None
                    
                # Support both "side" and legacy "action" fields with validation
                side = decision.get("side", decision.get("action"))
                if not side or side not in ["BUY", "SELL"]:
                    logger.error(f"Invalid side in decision: {side}")
                    return None
                    
                order_type = decision.get("order_type")
                if not order_type or order_type not in ["LIMIT", "MARKET"]:
                    logger.error(f"Invalid order_type in decision: {order_type}")
                    return None
                    
                quantity = decision.get("size")
                if not quantity or not isinstance(quantity, (int, float, str)) or float(quantity) <= 0:
                    logger.error(f"Invalid quantity in decision: {quantity}")
                    return None
                    
                price = decision.get("price")
                if order_type == "LIMIT" and (not price or not isinstance(price, (int, float, str)) or float(price) <= 0):
                    logger.error(f"Invalid price in decision: {price}")
                    return None
                    
                time_in_force = decision.get("time_in_force", "GTC")
                if order_type == "LIMIT" and time_in_force not in ["GTC", "IOC", "FOK"]:
                    logger.error(f"Invalid time_in_force in decision: {time_in_force}")
                    return None
                    
            except KeyError as e:
                logger.error(f"Missing required field in decision: {e}")
                return None
            
            # Place paper trading order
            order = self.paper_trading.place_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                time_in_force=time_in_force
            )
            
            if order:
                # Update statistics
                self.stats["orders_placed"] += 1
                
                # Log order
                logger.info(f"Paper order placed: {side} {quantity} {symbol} @ {price}")
                
                return order
            else:
                logger.warning(f"Failed to place paper order: {decision}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing paper trading decision: {str(e)}")
            return None
    
    def run_for_duration(self, duration_seconds, update_interval=1.0):
        """Run the flash trading system for a specified duration"""
        # Start the system
        if not self.start():
            return False
        
        try:
            # Run for specified duration
            end_time = time.time() + duration_seconds
            
            while time.time() < end_time and self.running:
                # Process signals and execute trades
                self.process_signals_and_execute()
                
                # Sleep for update interval
                time.sleep(update_interval)
                
                # Print status every 5 seconds
                elapsed = time.time() - self.start_time
                if int(elapsed) % 5 == 0:
                    self._print_status()
        
        except KeyboardInterrupt:
            logger.info("Flash trading interrupted by user")
        
        finally:
            # Stop the system
            self.stop()
            
            # Print final status
            self._print_final_status()
        
        return True
    
    def _print_status(self):
        """Print current status"""
        # Get account information
        account = self.paper_trading.get_account()
        
        # Get open orders
        open_orders = self.paper_trading.get_open_orders()
        
        # Print status
        print("\n--- Flash Trading Status ---")
        print("Balances:")
        for balance in account["balances"]:
            if balance["free"] > 0:
                print(f"  {balance['asset']}: {balance['free']}")
        
        print("\nOpen Orders:")
        for order in open_orders:
            print(f"  {order['symbol']} {order['side']} {order['quantity']} @ {order['price']}")
        
        print("\nStatistics:")
        print(f"  Signals: {self.signal_generator.stats['signals_generated']}")
        print(f"  Orders: {self.stats['orders_placed']}")
        print(f"  Uptime: {time.time() - self.start_time:.1f}s")
    
    def _print_final_status(self):
        """Print final status"""
        # Get account information
        account = self.paper_trading.get_account()
        
        # Print status
        print("\n=== Flash Trading Final Status ===")
        print("Balances:")
        for balance in account["balances"]:
            if balance["free"] > 0:
                print(f"  {balance['asset']}: {balance['free']}")
        
        print("\nStatistics:")
        print(f"  Signals Generated: {self.signal_generator.stats['signals_generated']}")
        print(f"  Orders Placed: {self.stats['orders_placed']}")
        print(f"  Uptime: {time.time() - self.start_time:.1f}s")
    
    def save_state(self, filename="flash_trading_state.json"):
        """Save current state to file"""
        state = {
            "timestamp": int(time.time() * 1000),
            "stats": self.stats,
            "signals": self.signal_generator.get_recent_signals(100),
            "uptime": time.time() - self.start_time if self.start_time else 0
        }
        
        with open(filename, "w") as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"State saved to {filename}")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flash Trading System')
    parser.add_argument('--env', default=".env-secure/.env", help='Path to .env file')
    parser.add_argument('--config', default="flash_trading_config.json", help='Path to config file')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    parser.add_argument('--reset', action='store_true', help='Reset paper trading system')
    
    args = parser.parse_args()
    
    # Create flash trading system
    flash_trading = FlashTradingSystem(args.env, args.config)
    
    # Reset paper trading if requested
    if args.reset:
        flash_trading.paper_trading.reset()
        print("Paper trading system reset to initial state")
    
    # Run for specified duration
    print(f"Running flash trading system for {args.duration} seconds...")
    flash_trading.run_for_duration(args.duration)
    
    # Save state
    flash_trading.save_state()
