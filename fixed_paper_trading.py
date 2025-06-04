#!/usr/bin/env python
"""
Enhanced Paper Trading System with Order Creation

This module extends the paper trading system with proper order creation,
management, and notification integration.
"""

import os
import sys
import json
import time
import logging
import threading
import uuid
from queue import Queue
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from enhanced_logging_fixed import EnhancedLogger
from optimized_mexc_client import OptimizedMexcClient

# Initialize enhanced logger
logger = EnhancedLogger("paper_trading_fixed")

class FixedPaperTradingSystem:
    """Enhanced paper trading system with proper order creation and management"""
    
    def __init__(self, client=None, config=None):
        """Initialize paper trading system
        
        Args:
            client: Exchange client (optional)
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.logger = logger
        self.client = client or OptimizedMexcClient()
        
        # Initialize state
        self.balance = self.config.get('initial_balance', {
            'USDC': 10000.0,
            'BTC': 0.1,
            'ETH': 1.0,
            'SOL': 10.0
        })
        
        self.orders = {}
        self.trades = []
        self.positions = {}
        
        # Order book cache
        self.order_books = {}
        
        # Last prices
        self.last_prices = {}
        
        # Running flag
        self.running = False
        
        # Order processing queue
        self.order_queue = Queue()
        
        # Notification callback
        self.notification_callback = None
        
        # Initialize
        self.initialize()
        
        self.logger.system.info("Fixed paper trading system initialized")
    
    def initialize(self):
        """Initialize paper trading system"""
        # Initialize positions
        for symbol in ['BTCUSDC', 'ETHUSDC', 'SOLUSDC']:
            base_asset = symbol.replace('USDC', '')
            self.positions[symbol] = {
                'symbol': symbol,
                'base_asset': base_asset,
                'quote_asset': 'USDC',
                'base_quantity': self.balance.get(base_asset, 0.0),
                'quote_quantity': 0.0,
                'entry_price': 0.0,
                'current_price': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'timestamp': int(time.time() * 1000)
            }
        
        # Initialize last prices
        self.update_market_data()
    
    def set_notification_callback(self, callback):
        """Set notification callback
        
        Args:
            callback: Notification callback function
        """
        self.notification_callback = callback
        self.logger.system.info("Notification callback set")
    
    def notify(self, notification_type, data):
        """Send notification
        
        Args:
            notification_type: Type of notification
            data: Notification data
        """
        if self.notification_callback:
            self.notification_callback(notification_type, data)
    
    def start(self):
        """Start paper trading system"""
        self.logger.system.info("Starting paper trading system")
        
        try:
            # Start order processing thread
            self.running = True
            self.order_thread = threading.Thread(target=self.process_orders)
            self.order_thread.daemon = True
            self.order_thread.start()
            
            self.logger.system.info("Paper trading system started")
        except Exception as e:
            self.logger.log_error("Error starting paper trading system", component="paper_trading")
            raise
    
    def stop(self):
        """Stop paper trading system"""
        self.logger.system.info("Stopping paper trading system")
        
        try:
            # Stop order processing
            self.running = False
            
            # Wait for thread to terminate
            if hasattr(self, 'order_thread') and self.order_thread.is_alive():
                self.order_thread.join(timeout=5.0)
            
            self.logger.system.info("Paper trading system stopped")
        except Exception as e:
            self.logger.log_error("Error stopping paper trading system", component="paper_trading")
            raise
    
    def process_orders(self):
        """Process orders from the queue"""
        self.logger.system.info("Order processing thread started")
        last_heartbeat = time.time()
        
        while self.running:
            try:
                # Send heartbeat log periodically
                current_time = time.time()
                if current_time - last_heartbeat > 60:
                    self.logger.system.debug("Order processing thread heartbeat")
                    last_heartbeat = current_time
                
                # Process orders from queue
                if not self.order_queue.empty():
                    order_action = self.order_queue.get(timeout=0.1)
                    self.execute_order_action(order_action)
                else:
                    time.sleep(0.01)
            except Exception as e:
                self.logger.log_error("Error in order processing thread", component="paper_trading")
                time.sleep(1)  # Prevent tight loop on persistent errors
        
        self.logger.system.info("Order processing thread stopped")
    
    def execute_order_action(self, order_action):
        """Execute an order action
        
        Args:
            order_action: Order action dictionary
        """
        try:
            # Extract action data
            action_type = order_action.get('type', 'unknown')
            order_id = order_action.get('order_id')
            
            if action_type == 'create':
                # Create order
                order = order_action.get('order', {})
                self.create_order_internal(order)
            elif action_type == 'cancel':
                # Cancel order
                reason = order_action.get('reason', 'User requested')
                self.cancel_order_internal(order_id, reason)
            elif action_type == 'fill':
                # Fill order
                price = order_action.get('price')
                self.fill_order_internal(order_id, price)
            else:
                self.logger.system.warning(f"Unknown order action type: {action_type}")
        except Exception as e:
            self.logger.log_error(f"Error executing order action: {str(e)}", component="paper_trading")
    
    def create_order(self, symbol, side, order_type, quantity, price=None):
        """Create a new order
        
        Args:
            symbol: Trading pair symbol
            side: Order side (BUY or SELL)
            order_type: Order type (LIMIT or MARKET)
            quantity: Order quantity
            price: Order price (required for LIMIT orders)
            
        Returns:
            str: Order ID
        """
        try:
            # Generate order ID
            order_id = f"ORD-{uuid.uuid4()}"
            
            # Create order
            order = {
                'orderId': order_id,
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': float(quantity),
                'price': float(price) if price is not None else None,
                'status': 'NEW',
                'timestamp': int(time.time() * 1000)
            }
            
            # Add to order queue
            self.order_queue.put({
                'type': 'create',
                'order': order
            })
            
            self.logger.system.info(f"Order {order_id} queued for creation")
            
            return order_id
        except Exception as e:
            self.logger.log_error(f"Error creating order: {str(e)}", component="paper_trading")
            return None
    
    def create_order_internal(self, order):
        """Create order internally
        
        Args:
            order: Order dictionary
        """
        try:
            # Extract order data
            order_id = order.get('orderId')
            symbol = order.get('symbol')
            side = order.get('side')
            order_type = order.get('type')
            quantity = order.get('quantity')
            price = order.get('price')
            
            # Validate order
            if not self.validate_order(symbol, side, order_type, quantity, price):
                self.logger.system.warning(f"Order {order_id} validation failed")
                return
            
            # Add order to orders dictionary
            self.orders[order_id] = order
            
            # Log order creation
            self.logger.system.info(f"Order {order_id} created: {side} {quantity} {symbol} at {price}")
            
            # Send notification
            self.notify('order_created', order)
            
            # Check if order can be filled immediately
            if order_type == 'MARKET':
                # Fill market order immediately
                current_price = self.get_current_price(symbol)
                self.fill_order_internal(order_id, current_price)
        except Exception as e:
            self.logger.log_error(f"Error creating order internally: {str(e)}", component="paper_trading")
    
    def validate_order(self, symbol, side, order_type, quantity, price):
        """Validate order parameters
        
        Args:
            symbol: Trading pair symbol
            side: Order side (BUY or SELL)
            order_type: Order type (LIMIT or MARKET)
            quantity: Order quantity
            price: Order price
            
        Returns:
            bool: Whether the order is valid
        """
        try:
            # Check symbol
            if symbol not in ['BTCUSDC', 'ETHUSDC', 'SOLUSDC']:
                self.logger.system.warning(f"Invalid symbol: {symbol}")
                return False
            
            # Check side
            if side not in ['BUY', 'SELL']:
                self.logger.system.warning(f"Invalid side: {side}")
                return False
            
            # Check order type
            if order_type not in ['LIMIT', 'MARKET']:
                self.logger.system.warning(f"Invalid order type: {order_type}")
                return False
            
            # Check quantity
            if quantity <= 0:
                self.logger.system.warning(f"Invalid quantity: {quantity}")
                return False
            
            # Check price for LIMIT orders
            if order_type == 'LIMIT' and (price is None or price <= 0):
                self.logger.system.warning(f"Invalid price for LIMIT order: {price}")
                return False
            
            # Check balance
            base_asset = symbol.replace('USDC', '')
            if side == 'SELL':
                # Check if enough base asset
                if self.balance.get(base_asset, 0.0) < quantity:
                    self.logger.system.warning(f"Insufficient {base_asset} balance: {self.balance.get(base_asset, 0.0)} < {quantity}")
                    return False
            else:  # BUY
                # Check if enough quote asset
                cost = quantity * (price or self.get_current_price(symbol))
                if self.balance.get('USDC', 0.0) < cost:
                    self.logger.system.warning(f"Insufficient USDC balance: {self.balance.get('USDC', 0.0)} < {cost}")
                    return False
            
            return True
        except Exception as e:
            self.logger.log_error(f"Error validating order: {str(e)}", component="paper_trading")
            return False
    
    def cancel_order(self, order_id, reason="User requested"):
        """Cancel an order
        
        Args:
            order_id: Order ID
            reason: Cancellation reason
            
        Returns:
            bool: Whether the cancellation was successful
        """
        try:
            # Check if order exists
            if order_id not in self.orders:
                self.logger.system.warning(f"Order {order_id} not found")
                return False
            
            # Add to order queue
            self.order_queue.put({
                'type': 'cancel',
                'order_id': order_id,
                'reason': reason
            })
            
            self.logger.system.info(f"Order {order_id} queued for cancellation")
            
            return True
        except Exception as e:
            self.logger.log_error(f"Error cancelling order: {str(e)}", component="paper_trading")
            return False
    
    def cancel_order_internal(self, order_id, reason="User requested"):
        """Cancel order internally
        
        Args:
            order_id: Order ID
            reason: Cancellation reason
        """
        try:
            # Check if order exists
            if order_id not in self.orders:
                self.logger.system.warning(f"Order {order_id} not found")
                return
            
            # Get order
            order = self.orders[order_id]
            
            # Update order status
            order['status'] = 'CANCELED'
            order['cancelTime'] = int(time.time() * 1000)
            order['cancelReason'] = reason
            
            # Log order cancellation
            self.logger.system.info(f"Order {order_id} cancelled: {reason}")
            
            # Send notification
            self.notify('order_cancelled', order)
        except Exception as e:
            self.logger.log_error(f"Error cancelling order internally: {str(e)}", component="paper_trading")
    
    def fill_order(self, order_id, price=None):
        """Fill an order
        
        Args:
            order_id: Order ID
            price: Fill price (optional)
            
        Returns:
            bool: Whether the fill was successful
        """
        try:
            # Check if order exists
            if order_id not in self.orders:
                self.logger.system.warning(f"Order {order_id} not found")
                return False
            
            # Add to order queue
            self.order_queue.put({
                'type': 'fill',
                'order_id': order_id,
                'price': price
            })
            
            self.logger.system.info(f"Order {order_id} queued for filling")
            
            return True
        except Exception as e:
            self.logger.log_error(f"Error filling order: {str(e)}", component="paper_trading")
            return False
    
    def fill_order_internal(self, order_id, price=None):
        """Fill order internally
        
        Args:
            order_id: Order ID
            price: Fill price (optional)
        """
        try:
            # Check if order exists
            if order_id not in self.orders:
                self.logger.system.warning(f"Order {order_id} not found")
                return
            
            # Get order
            order = self.orders[order_id]
            
            # Check if order is already filled or cancelled
            if order['status'] in ['FILLED', 'CANCELED']:
                self.logger.system.warning(f"Order {order_id} already {order['status']}")
                return
            
            # Get fill price
            fill_price = price or order.get('price') or self.get_current_price(order['symbol'])
            
            # Update order
            order['status'] = 'FILLED'
            order['price'] = fill_price
            order['fillTime'] = int(time.time() * 1000)
            
            # Create trade
            trade = {
                'tradeId': f"TRD-{uuid.uuid4()}",
                'orderId': order_id,
                'symbol': order['symbol'],
                'side': order['side'],
                'quantity': order['quantity'],
                'price': fill_price,
                'timestamp': int(time.time() * 1000)
            }
            
            # Add trade to trades list
            self.trades.append(trade)
            
            # Update balance and position
            self.update_balance_and_position(order, fill_price)
            
            # Log order fill
            self.logger.system.info(f"Order {order_id} filled: {order['quantity']} {order['symbol']} at {fill_price}")
            
            # Send notification
            self.notify('order_filled', order)
        except Exception as e:
            self.logger.log_error(f"Error filling order internally: {str(e)}", component="paper_trading")
    
    def update_balance_and_position(self, order, price):
        """Update balance and position after order fill
        
        Args:
            order: Order dictionary
            price: Fill price
        """
        try:
            # Extract order data
            symbol = order['symbol']
            side = order['side']
            quantity = order['quantity']
            
            # Get base and quote assets
            base_asset = symbol.replace('USDC', '')
            quote_asset = 'USDC'
            
            # Update balance
            if side == 'BUY':
                # Deduct quote asset
                cost = quantity * price
                self.balance[quote_asset] -= cost
                # Add base asset
                self.balance[base_asset] = self.balance.get(base_asset, 0.0) + quantity
            else:  # SELL
                # Deduct base asset
                self.balance[base_asset] -= quantity
                # Add quote asset
                proceeds = quantity * price
                self.balance[quote_asset] = self.balance.get(quote_asset, 0.0) + proceeds
            
            # Update position
            position = self.positions.get(symbol, {
                'symbol': symbol,
                'base_asset': base_asset,
                'quote_asset': quote_asset,
                'base_quantity': 0.0,
                'quote_quantity': 0.0,
                'entry_price': 0.0,
                'current_price': price,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'timestamp': int(time.time() * 1000)
            })
            
            # Update position quantities
            position['base_quantity'] = self.balance.get(base_asset, 0.0)
            position['current_price'] = price
            position['timestamp'] = int(time.time() * 1000)
            
            # Calculate PnL
            if side == 'BUY':
                # Update entry price (weighted average)
                if position['base_quantity'] > 0:
                    position['entry_price'] = (
                        (position['entry_price'] * (position['base_quantity'] - quantity) + price * quantity)
                        / position['base_quantity']
                    )
            else:  # SELL
                # Calculate realized PnL
                if position['entry_price'] > 0:
                    realized_pnl = (price - position['entry_price']) * quantity
                    position['realized_pnl'] += realized_pnl
            
            # Calculate unrealized PnL
            if position['entry_price'] > 0 and position['base_quantity'] > 0:
                position['unrealized_pnl'] = (price - position['entry_price']) * position['base_quantity']
            else:
                position['unrealized_pnl'] = 0.0
            
            # Update position
            self.positions[symbol] = position
            
            self.logger.system.info(f"Balance and position updated for {symbol}")
        except Exception as e:
            self.logger.log_error(f"Error updating balance and position: {str(e)}", component="paper_trading")
    
    def get_current_price(self, symbol):
        """Get current price for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            float: Current price
        """
        try:
            # Check if price is cached
            if symbol in self.last_prices:
                return self.last_prices[symbol]
            
            # Get price from exchange
            ticker = self.client.get_ticker(symbol)
            price = float(ticker.get('last', 0.0))
            
            # Cache price
            self.last_prices[symbol] = price
            
            return price
        except Exception as e:
            self.logger.log_error(f"Error getting current price: {str(e)}", component="paper_trading")
            return 0.0
    
    def update_market_data(self):
        """Update market data"""
        try:
            # Update prices
            for symbol in ['BTCUSDC', 'ETHUSDC', 'SOLUSDC']:
                self.get_current_price(symbol)
            
            # Update order books
            for symbol in ['BTCUSDC', 'ETHUSDC', 'SOLUSDC']:
                self.update_order_book(symbol)
            
            self.logger.system.debug("Market data updated")
        except Exception as e:
            self.logger.log_error(f"Error updating market data: {str(e)}", component="paper_trading")
    
    def update_order_book(self, symbol):
        """Update order book for a symbol
        
        Args:
            symbol: Trading pair symbol
        """
        try:
            # Get order book from exchange
            order_book = self.client.get_order_book(symbol)
            
            # Cache order book
            self.order_books[symbol] = order_book
            
            self.logger.system.debug(f"Order book updated for {symbol}")
        except Exception as e:
            self.logger.log_error(f"Error updating order book: {str(e)}", component="paper_trading")
    
    def get_balance(self, asset=None):
        """Get balance
        
        Args:
            asset: Asset symbol (optional)
            
        Returns:
            dict or float: Balance dictionary or asset balance
        """
        if asset:
            return self.balance.get(asset, 0.0)
        else:
            return self.balance
    
    def get_position(self, symbol=None):
        """Get position
        
        Args:
            symbol: Trading pair symbol (optional)
            
        Returns:
            dict or list: Position dictionary or list of positions
        """
        if symbol:
            return self.positions.get(symbol, {})
        else:
            return list(self.positions.values())
    
    def get_orders(self, symbol=None, status=None):
        """Get orders
        
        Args:
            symbol: Trading pair symbol (optional)
            status: Order status (optional)
            
        Returns:
            list: List of orders
        """
        orders = list(self.orders.values())
        
        # Filter by symbol
        if symbol:
            orders = [order for order in orders if order['symbol'] == symbol]
        
        # Filter by status
        if status:
            orders = [order for order in orders if order['status'] == status]
        
        return orders
    
    def get_trades(self, symbol=None):
        """Get trades
        
        Args:
            symbol: Trading pair symbol (optional)
            
        Returns:
            list: List of trades
        """
        trades = self.trades
        
        # Filter by symbol
        if symbol:
            trades = [trade for trade in trades if trade['symbol'] == symbol]
        
        return trades
    
    def get_order_book(self, symbol):
        """Get order book
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            dict: Order book
        """
        # Check if order book is cached
        if symbol in self.order_books:
            return self.order_books[symbol]
        
        # Update order book
        self.update_order_book(symbol)
        
        return self.order_books.get(symbol, {})
    
    def get_ticker(self, symbol):
        """Get ticker
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            dict: Ticker
        """
        price = self.get_current_price(symbol)
        
        return {
            'symbol': symbol,
            'last': price,
            'bid': price * 0.999,
            'ask': price * 1.001,
            'volume': 0.0,
            'timestamp': int(time.time() * 1000)
        }


# Example usage
if __name__ == "__main__":
    # Create paper trading system
    paper_trading = FixedPaperTradingSystem()
    
    # Start paper trading system
    paper_trading.start()
    
    # Create test order
    order_id = paper_trading.create_order('BTCUSDC', 'BUY', 'LIMIT', 0.001, 105000.0)
    
    # Wait for order processing
    time.sleep(1)
    
    # Fill order
    paper_trading.fill_order(order_id)
    
    # Wait for order processing
    time.sleep(1)
    
    # Get balance
    balance = paper_trading.get_balance()
    print(f"Balance: {balance}")
    
    # Get position
    position = paper_trading.get_position('BTCUSDC')
    print(f"Position: {position}")
    
    # Run for a while
    try:
        print("Running Fixed Paper Trading System for 10 seconds...")
        time.sleep(10)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Stop paper trading system
        paper_trading.stop()
        print("Fixed Paper Trading System stopped")
