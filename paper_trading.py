#!/usr/bin/env python
"""
Paper Trading System for Flash Trading

This module provides a paper trading implementation that simulates trading
with real market data but without using real funds. It's designed for
testing flash trading strategies with zero-fee pairs (BTCUSDC, ETHUSDC).
"""

import time
import json
import logging
import os
import uuid
from datetime import datetime
from threading import RLock
from optimized_mexc_client import OptimizedMexcClient
from flash_trading_config import FlashTradingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("paper_trading")

class PaperTradingSystem:
    """Paper trading system for flash trading strategies"""
    
    def __init__(self, client=None, config=None):
        """Initialize paper trading system with API client and configuration"""
        self.client = client or OptimizedMexcClient()
        self.config = config or FlashTradingConfig()
        
        # Load paper trading configuration
        self.paper_config = self.config.config["paper_trading"]
        
        # Initialize balances
        self.balances = self.paper_config["initial_balance"].copy()
        
        # Initialize orders and trades
        self.open_orders = {}
        self.order_history = []
        self.trade_history = []
        
        # Market data cache
        self.market_data = {}
        
        # Lock for thread safety
        self.lock = RLock()
        
        # Load state if enabled
        if self.paper_config["persist_state"]:
            self._load_state()
    
    def _load_state(self):
        """Load paper trading state from file"""
        state_file = self.paper_config["state_file"]
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                # Restore state
                if "balances" in state:
                    self.balances = state["balances"]
                if "open_orders" in state:
                    self.open_orders = state["open_orders"]
                if "order_history" in state:
                    self.order_history = state["order_history"]
                if "trade_history" in state:
                    self.trade_history = state["trade_history"]
                
                logger.info(f"Paper trading state loaded from {state_file}")
            except Exception as e:
                logger.error(f"Error loading paper trading state: {str(e)}")
    
    def _save_state(self):
        """Save paper trading state to file"""
        if not self.paper_config["persist_state"]:
            return
        
        state_file = self.paper_config["state_file"]
        try:
            state = {
                "timestamp": int(time.time() * 1000),
                "balances": self.balances,
                "open_orders": self.open_orders,
                "order_history": self.order_history,
                "trade_history": self.trade_history
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Paper trading state saved to {state_file}")
        except Exception as e:
            logger.error(f"Error saving paper trading state: {str(e)}")
    
    def get_balance(self, asset):
        """Get paper trading balance for an asset"""
        with self.lock:
            return self.balances.get(asset, 0.0)
    
    def get_all_balances(self):
        """Get all paper trading balances"""
        with self.lock:
            return self.balances.copy()
    
    def _update_market_data(self, symbol):
        """Update market data for a symbol"""
        try:
            # Validate symbol
            if not symbol or not isinstance(symbol, str):
                logger.error(f"Invalid symbol for market data update: {symbol}")
                return False
                
            # Get order book with robust error handling
            order_book = self.client.get_order_book(symbol, limit=10)
            
            # Validate order book data
            if not order_book:
                logger.warning(f"Empty order book response for {symbol}")
                return False
                
            if not isinstance(order_book, dict):
                logger.error(f"Invalid order book response type for {symbol}: {type(order_book)}")
                return False
                
            if 'bids' not in order_book or 'asks' not in order_book:
                logger.error(f"Missing bids or asks in order book for {symbol}")
                return False
                
            if not order_book["bids"] or not order_book["asks"]:
                logger.warning(f"Empty bids or asks in order book for {symbol}")
                return False
                
            # Store validated market data
            self.market_data[symbol] = {
                "timestamp": int(time.time() * 1000),
                "bids": order_book["bids"],
                "asks": order_book["asks"]
            }
            return True
        except Exception as e:
            logger.error(f"Error updating market data for {symbol}: {str(e)}")
            return False
    
    def _get_current_price(self, symbol, side):
        """Get current price for a symbol and side"""
        # Update market data if needed
        if symbol not in self.market_data or time.time() * 1000 - self.market_data[symbol]["timestamp"] > 5000:
            if not self._update_market_data(symbol):
                return None
        
        # Get price from order book
        if side == "BUY":
            # Buy at ask price
            return float(self.market_data[symbol]["asks"][0][0])
        else:
            # Sell at bid price
            return float(self.market_data[symbol]["bids"][0][0])
    
    def _apply_slippage(self, price, side):
        """Apply simulated slippage to price"""
        if not self.paper_config["simulate_slippage"]:
            return price
        
        slippage_factor = self.paper_config["slippage_bps"] / 10000.0
        if side == "BUY":
            # Higher price for buys
            return price * (1 + slippage_factor)
        else:
            # Lower price for sells
            return price * (1 - slippage_factor)
    
    def place_order(self, symbol, side, order_type, quantity, price=None, time_in_force="GTC"):
        """Place a paper trading order"""
        with self.lock:
            # Validate symbol
            pair_config = self.config.get_trading_pair_config(symbol)
            if not pair_config:
                logger.warning(f"Invalid symbol: {symbol}")
                return None
            
            # Extract assets
            base_asset = pair_config["base_asset"]
            quote_asset = pair_config["quote_asset"]
            
            # Validate quantity
            quantity = float(quantity)
            if quantity < pair_config["min_order_size"]:
                logger.warning(f"Order quantity {quantity} below minimum {pair_config['min_order_size']}")
                return None
            
            # Get current price if not provided
            if price is None:
                if order_type == "MARKET":
                    price = self._get_current_price(symbol, side)
                    if price is None:
                        logger.warning(f"Could not determine market price for {symbol}")
                        return None
                else:
                    logger.warning(f"Price must be provided for {order_type} orders")
                    return None
            else:
                price = float(price)
            
            # Apply slippage for market orders
            if order_type == "MARKET":
                price = self._apply_slippage(price, side)
            
            # Check balance
            if side == "BUY":
                # Check quote asset balance
                required_balance = price * quantity
                if self.balances.get(quote_asset, 0) < required_balance:
                    logger.warning(f"Insufficient {quote_asset} balance for order")
                    return None
            else:
                # Check base asset balance
                if self.balances.get(base_asset, 0) < quantity:
                    logger.warning(f"Insufficient {base_asset} balance for order")
                    return None
            
            # Create order
            order_id = str(uuid.uuid4())
            timestamp = int(time.time() * 1000)
            
            order = {
                "orderId": order_id,
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "timeInForce": time_in_force,
                "quantity": quantity,
                "price": price,
                "status": "NEW",
                "timestamp": timestamp,
                "executedQty": 0.0,
                "cummulativeQuoteQty": 0.0,
                "fills": []
            }
            
            # Add to open orders
            self.open_orders[order_id] = order
            
            # Add to order history
            self.order_history.append(order.copy())
            
            # Process order immediately for market orders or IOC limit orders
            if order_type == "MARKET" or time_in_force == "IOC":
                self._process_order(order_id)
            
            # Save state
            self._save_state()
            
            # Log order
            if self.paper_config["log_trades"]:
                logger.info(f"Paper order placed: {side} {quantity} {symbol} @ {price}")
            
            return order
    
    def _process_order(self, order_id):
        """Process an open order"""
        if order_id not in self.open_orders:
            return False
        
        order = self.open_orders[order_id]
        symbol = order["symbol"]
        side = order["side"]
        order_type = order["type"]
        quantity = order["quantity"]
        price = order["price"]
        
        # Get current market price
        market_price = self._get_current_price(symbol, side)
        if market_price is None:
            return False
        
        # Check if order can be filled
        can_fill = False
        fill_price = price
        
        if order_type == "MARKET":
            can_fill = True
            fill_price = market_price
        elif side == "BUY" and market_price <= price:
            can_fill = True
            fill_price = price
        elif side == "SELL" and market_price >= price:
            can_fill = True
            fill_price = price
        
        if not can_fill:
            # Cancel order if it's IOC
            if order["timeInForce"] == "IOC":
                self._cancel_order(order_id)
            return False
        
        # Determine fill quantity
        fill_quantity = quantity
        if self.paper_config["simulate_partial_fills"]:
            import random
            if random.random() < self.paper_config["partial_fill_probability"]:
                fill_quantity = quantity * random.uniform(0.1, 0.9)
                fill_quantity = round(fill_quantity, 8)
        
        # Extract assets
        pair_config = self.config.get_trading_pair_config(symbol)
        base_asset = pair_config["base_asset"]
        quote_asset = pair_config["quote_asset"]
        
        # Calculate fill amount
        fill_amount = fill_price * fill_quantity
        
        # Update balances
        if side == "BUY":
            # Deduct quote asset, add base asset
            self.balances[quote_asset] = self.balances.get(quote_asset, 0) - fill_amount
            self.balances[base_asset] = self.balances.get(base_asset, 0) + fill_quantity
        else:
            # Add quote asset, deduct base asset
            self.balances[quote_asset] = self.balances.get(quote_asset, 0) + fill_amount
            self.balances[base_asset] = self.balances.get(base_asset, 0) - fill_quantity
        
        # Create fill
        timestamp = int(time.time() * 1000)
        fill = {
            "price": fill_price,
            "quantity": fill_quantity,
            "commission": 0.0,  # Zero fee for BTCUSDC and ETHUSDC
            "commissionAsset": quote_asset,
            "tradeId": str(uuid.uuid4())
        }
        
        # Update order
        order["executedQty"] = fill_quantity
        order["cummulativeQuoteQty"] = fill_amount
        order["fills"].append(fill)
        
        # Update status
        if fill_quantity >= quantity:
            order["status"] = "FILLED"
            del self.open_orders[order_id]
        else:
            order["status"] = "PARTIALLY_FILLED"
            order["quantity"] = quantity - fill_quantity
        
        # Add to trade history
        trade = {
            "symbol": symbol,
            "orderId": order_id,
            "side": side,
            "price": fill_price,
            "quantity": fill_quantity,
            "quoteQty": fill_amount,
            "commission": 0.0,
            "commissionAsset": quote_asset,
            "timestamp": timestamp
        }
        self.trade_history.append(trade)
        
        # Log trade
        if self.paper_config["log_trades"]:
            logger.info(f"Paper trade executed: {side} {fill_quantity} {symbol} @ {fill_price}")
        
        return True
    
    def cancel_order(self, symbol, order_id):
        """Cancel a paper trading order"""
        with self.lock:
            return self._cancel_order(order_id)
    
    def _cancel_order(self, order_id):
        """Internal method to cancel an order"""
        if order_id not in self.open_orders:
            return False
        
        # Update order status
        order = self.open_orders[order_id]
        order["status"] = "CANCELED"
        
        # Remove from open orders
        del self.open_orders[order_id]
        
        # Update order history
        for hist_order in self.order_history:
            if hist_order["orderId"] == order_id:
                hist_order["status"] = "CANCELED"
                break
        
        # Save state
        self._save_state()
        
        # Log cancellation
        if self.paper_config["log_trades"]:
            logger.info(f"Paper order canceled: {order_id}")
        
        return True
    
    def get_open_orders(self, symbol=None):
        """Get open paper trading orders"""
        with self.lock:
            if symbol:
                return [order for order in self.open_orders.values() if order["symbol"] == symbol]
            else:
                return list(self.open_orders.values())
    
    def get_order(self, order_id):
        """Get paper trading order by ID"""
        with self.lock:
            # Check open orders
            if order_id in self.open_orders:
                return self.open_orders[order_id].copy()
            
            # Check order history
            for order in self.order_history:
                if order["orderId"] == order_id:
                    return order.copy()
            
            return None
    
    def get_account(self):
        """Get paper trading account information"""
        with self.lock:
            return {
                "balances": [{"asset": asset, "free": amount, "locked": 0.0} 
                            for asset, amount in self.balances.items()],
                "canTrade": True,
                "canDeposit": False,
                "canWithdraw": False,
                "updateTime": int(time.time() * 1000),
                "accountType": "PAPER_TRADING"
            }
    
    def process_open_orders(self):
        """Process all open orders"""
        with self.lock:
            order_ids = list(self.open_orders.keys())
            for order_id in order_ids:
                self._process_order(order_id)
            
            # Save state if any orders were processed
            if order_ids:
                self._save_state()
    
    def reset(self):
        """Reset paper trading system to initial state"""
        with self.lock:
            # Reset balances
            self.balances = self.paper_config["initial_balance"].copy()
            
            # Clear orders and trades
            self.open_orders = {}
            self.order_history = []
            self.trade_history = []
            
            # Save state
            self._save_state()
            
            logger.info("Paper trading system reset to initial state")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Paper Trading System')
    parser.add_argument('--env', default=".env-secure/.env", help='Path to .env file')
    parser.add_argument('--config', default="flash_trading_config.json", help='Path to config file')
    parser.add_argument('--reset', action='store_true', help='Reset paper trading system')
    parser.add_argument('--test', action='store_true', help='Run test trades')
    
    args = parser.parse_args()
    
    # Create client and config
    client = OptimizedMexcClient(env_path=args.env)
    config = FlashTradingConfig(args.config)
    
    # Create paper trading system
    paper_trading = PaperTradingSystem(client, config)
    
    # Reset if requested
    if args.reset:
        paper_trading.reset()
        print("Paper trading system reset to initial state")
    
    # Print account information
    account = paper_trading.get_account()
    print("Paper Trading Account:")
    for balance in account["balances"]:
        if balance["free"] > 0:
            print(f"  {balance['asset']}: {balance['free']}")
    
    # Run test trades if requested
    if args.test:
        # Get enabled trading pairs
        trading_pairs = config.get_enabled_trading_pairs()
        if not trading_pairs:
            print("No enabled trading pairs found")
            exit(1)
        
        # Use first enabled pair
        symbol = trading_pairs[0]["symbol"]
        base_asset = trading_pairs[0]["base_asset"]
        quote_asset = trading_pairs[0]["quote_asset"]
        
        print(f"\nRunning test trades for {symbol}...")
        
        # Get current price
        order_book = client.get_order_book(symbol)
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            print(f"Could not get order book for {symbol}")
            exit(1)
        
        bid_price = float(order_book["bids"][0][0])
        ask_price = float(order_book["asks"][0][0])
        mid_price = (bid_price + ask_price) / 2
        
        print(f"Current prices - Bid: {bid_price}, Ask: {ask_price}, Mid: {mid_price}")
        
        # Place a market buy order
        buy_quantity = trading_pairs[0]["min_order_size"]
        print(f"\nPlacing market buy order: {buy_quantity} {symbol}")
        buy_order = paper_trading.place_order(
            symbol=symbol,
            side="BUY",
            order_type="MARKET",
            quantity=buy_quantity
        )
        
        if buy_order:
            print(f"Order placed: {buy_order['orderId']}")
            print(f"Status: {buy_order['status']}")
            print(f"Executed quantity: {buy_order['executedQty']}")
            print(f"Price: {buy_order['price']}")
        else:
            print("Failed to place buy order")
        
        # Wait a moment
        time.sleep(1)
        
        # Place a limit sell order
        if buy_order and buy_order['status'] == 'FILLED':
            sell_quantity = buy_order['executedQty']
            sell_price = buy_order['price'] * 1.01  # 1% higher
            
            print(f"\nPlacing limit sell order: {sell_quantity} {symbol} @ {sell_price}")
            sell_order = paper_trading.place_order(
                symbol=symbol,
                side="SELL",
                order_type="LIMIT",
                quantity=sell_quantity,
                price=sell_price,
                time_in_force="GTC"
            )
            
            if sell_order:
                print(f"Order placed: {sell_order['orderId']}")
                print(f"Status: {sell_order['status']}")
            else:
                print("Failed to place sell order")
        
        # Print updated account information
        account = paper_trading.get_account()
        print("\nUpdated Paper Trading Account:")
        for balance in account["balances"]:
            if balance["free"] > 0:
                print(f"  {balance['asset']}: {balance['free']}")
        
        # Print open orders
        open_orders = paper_trading.get_open_orders()
        if open_orders:
            print("\nOpen Orders:")
            for order in open_orders:
                print(f"  {order['orderId']}: {order['side']} {order['quantity']} {order['symbol']} @ {order['price']}")
    
    # Clean up
    client.close()
