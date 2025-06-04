#!/usr/bin/env python
"""
Enhanced Dashboard Visualization Integration

This module connects the dashboard visualization components to live trading data,
ensuring real-time updates and accurate representation of market state and trading activity.
"""

import os
import sys
import json
import time
import logging
import threading
from queue import Queue
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from enhanced_logging_fixed import EnhancedLogger
from fixed_paper_trading import FixedPaperTradingSystem
from optimized_mexc_client import OptimizedMexcClient
from visualization.data_service_adapter import DataService

# Initialize enhanced logger
logger = EnhancedLogger("dashboard_visualization")

class EnhancedDashboardIntegration:
    """Enhanced dashboard integration with live trading data"""
    
    def __init__(self, config=None):
        """Initialize dashboard integration
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.logger = logger
        
        # Initialize components
        self.client = OptimizedMexcClient()
        self.paper_trading = FixedPaperTradingSystem(self.client, self.config)
        self.data_service = DataService()
        
        # Data update interval (ms)
        self.update_interval = self.config.get('dashboard_update_interval_ms', 1000)
        
        # Running flag
        self.running = False
        
        # Data update queue
        self.update_queue = Queue()
        
        # Initialize data stores
        self.market_data = {}
        self.trading_data = {}
        self.signal_data = {}
        self.decision_data = {}
        
        # Initialize
        self.initialize()
        
        self.logger.system.info("Enhanced dashboard integration initialized")
    
    def initialize(self):
        """Initialize dashboard integration"""
        # Initialize market data
        for symbol in ['BTCUSDC', 'ETHUSDC', 'SOLUSDC']:
            self.market_data[symbol] = {
                'symbol': symbol,
                'last_price': 0.0,
                'bid_price': 0.0,
                'ask_price': 0.0,
                'volume_24h': 0.0,
                'price_change_24h': 0.0,
                'price_change_pct_24h': 0.0,
                'high_24h': 0.0,
                'low_24h': 0.0,
                'timestamp': int(time.time() * 1000),
                'price_history': [],
                'volume_history': []
            }
        
        # Initialize trading data
        self.trading_data = {
            'balance': {},
            'positions': {},
            'orders': {},
            'trades': [],
            'pnl_history': []
        }
        
        # Initialize signal data
        self.signal_data = {
            'recent_signals': [],
            'signal_history': {}
        }
        
        # Initialize decision data
        self.decision_data = {
            'recent_decisions': [],
            'decision_history': {}
        }
    
    def start(self):
        """Start dashboard integration"""
        self.logger.system.info("Starting dashboard integration")
        
        try:
            # Start paper trading system
            self.paper_trading.start()
            
            # Set up notification callback
            self.paper_trading.set_notification_callback(self.handle_trading_notification)
            
            # Start data update thread
            self.running = True
            self.update_thread = threading.Thread(target=self.update_data)
            self.update_thread.daemon = True
            self.update_thread.start()
            
            self.logger.system.info("Dashboard integration started")
        except Exception as e:
            self.logger.log_error("Error starting dashboard integration", component="dashboard")
            raise
    
    def stop(self):
        """Stop dashboard integration"""
        self.logger.system.info("Stopping dashboard integration")
        
        try:
            # Stop data update thread
            self.running = False
            
            # Wait for thread to terminate
            if hasattr(self, 'update_thread') and self.update_thread.is_alive():
                self.update_thread.join(timeout=5.0)
            
            # Stop paper trading system
            self.paper_trading.stop()
            
            self.logger.system.info("Dashboard integration stopped")
        except Exception as e:
            self.logger.log_error("Error stopping dashboard integration", component="dashboard")
            raise
    
    def update_data(self):
        """Update data for dashboard visualization"""
        self.logger.system.info("Data update thread started")
        last_update = time.time() * 1000
        
        while self.running:
            try:
                # Check if it's time to update
                current_time = time.time() * 1000
                if current_time - last_update >= self.update_interval:
                    # Update market data
                    self.update_market_data()
                    
                    # Update trading data
                    self.update_trading_data()
                    
                    # Process any pending updates
                    self.process_update_queue()
                    
                    # Update data service
                    self.update_data_service()
                    
                    # Update last update time
                    last_update = current_time
                
                # Sleep for a short time
                time.sleep(0.01)
            except Exception as e:
                self.logger.log_error(f"Error in data update thread: {str(e)}", component="dashboard")
                time.sleep(1)  # Prevent tight loop on persistent errors
        
        self.logger.system.info("Data update thread stopped")
    
    def update_market_data(self):
        """Update market data"""
        try:
            # Update market data for each symbol
            for symbol in self.market_data:
                # Get ticker
                ticker = self.client.get_ticker(symbol)
                
                # Update market data
                self.market_data[symbol]['last_price'] = float(ticker.get('last', 0.0))
                self.market_data[symbol]['bid_price'] = float(ticker.get('bid', 0.0))
                self.market_data[symbol]['ask_price'] = float(ticker.get('ask', 0.0))
                self.market_data[symbol]['volume_24h'] = float(ticker.get('volume', 0.0))
                self.market_data[symbol]['timestamp'] = int(time.time() * 1000)
                
                # Update price history
                price = self.market_data[symbol]['last_price']
                timestamp = self.market_data[symbol]['timestamp']
                
                # Add to price history (limit to 1000 points)
                self.market_data[symbol]['price_history'].append({
                    'price': price,
                    'timestamp': timestamp
                })
                if len(self.market_data[symbol]['price_history']) > 1000:
                    self.market_data[symbol]['price_history'].pop(0)
                
                # Calculate price change
                if len(self.market_data[symbol]['price_history']) > 1:
                    first_price = self.market_data[symbol]['price_history'][0]['price']
                    self.market_data[symbol]['price_change_24h'] = price - first_price
                    if first_price > 0:
                        self.market_data[symbol]['price_change_pct_24h'] = (price - first_price) / first_price * 100
                
                # Calculate high and low
                prices = [p['price'] for p in self.market_data[symbol]['price_history']]
                if prices:
                    self.market_data[symbol]['high_24h'] = max(prices)
                    self.market_data[symbol]['low_24h'] = min(prices)
            
            self.logger.system.debug("Market data updated")
        except Exception as e:
            self.logger.log_error(f"Error updating market data: {str(e)}", component="dashboard")
    
    def update_trading_data(self):
        """Update trading data"""
        try:
            # Update balance
            self.trading_data['balance'] = self.paper_trading.get_balance()
            
            # Update positions
            positions = self.paper_trading.get_position()
            self.trading_data['positions'] = {p['symbol']: p for p in positions}
            
            # Update orders
            orders = self.paper_trading.get_orders()
            self.trading_data['orders'] = {o['orderId']: o for o in orders}
            
            # Update trades
            self.trading_data['trades'] = self.paper_trading.get_trades()
            
            # Calculate PnL history
            total_pnl = 0
            for symbol, position in self.trading_data['positions'].items():
                total_pnl += position.get('unrealized_pnl', 0) + position.get('realized_pnl', 0)
            
            # Add to PnL history (limit to 1000 points)
            self.trading_data['pnl_history'].append({
                'pnl': total_pnl,
                'timestamp': int(time.time() * 1000)
            })
            if len(self.trading_data['pnl_history']) > 1000:
                self.trading_data['pnl_history'].pop(0)
            
            self.logger.system.debug("Trading data updated")
        except Exception as e:
            self.logger.log_error(f"Error updating trading data: {str(e)}", component="dashboard")
    
    def process_update_queue(self):
        """Process updates from the queue"""
        try:
            # Process all pending updates
            while not self.update_queue.empty():
                update = self.update_queue.get_nowait()
                
                # Process update based on type
                update_type = update.get('type')
                data = update.get('data')
                
                if update_type == 'signal':
                    self.process_signal_update(data)
                elif update_type == 'decision':
                    self.process_decision_update(data)
                elif update_type == 'order':
                    self.process_order_update(data)
                elif update_type == 'trade':
                    self.process_trade_update(data)
                else:
                    self.logger.system.warning(f"Unknown update type: {update_type}")
        except Exception as e:
            self.logger.log_error(f"Error processing update queue: {str(e)}", component="dashboard")
    
    def process_signal_update(self, signal):
        """Process signal update
        
        Args:
            signal: Signal data
        """
        try:
            # Add to recent signals (limit to 10)
            self.signal_data['recent_signals'].append(signal)
            if len(self.signal_data['recent_signals']) > 10:
                self.signal_data['recent_signals'].pop(0)
            
            # Add to signal history
            symbol = signal.get('symbol')
            if symbol not in self.signal_data['signal_history']:
                self.signal_data['signal_history'][symbol] = []
            
            # Add to symbol history (limit to 100 per symbol)
            self.signal_data['signal_history'][symbol].append(signal)
            if len(self.signal_data['signal_history'][symbol]) > 100:
                self.signal_data['signal_history'][symbol].pop(0)
            
            self.logger.system.debug(f"Signal update processed: {signal.get('id')}")
        except Exception as e:
            self.logger.log_error(f"Error processing signal update: {str(e)}", component="dashboard")
    
    def process_decision_update(self, decision):
        """Process decision update
        
        Args:
            decision: Decision data
        """
        try:
            # Add to recent decisions (limit to 10)
            self.decision_data['recent_decisions'].append(decision)
            if len(self.decision_data['recent_decisions']) > 10:
                self.decision_data['recent_decisions'].pop(0)
            
            # Add to decision history
            symbol = decision.get('symbol')
            if symbol not in self.decision_data['decision_history']:
                self.decision_data['decision_history'][symbol] = []
            
            # Add to symbol history (limit to 100 per symbol)
            self.decision_data['decision_history'][symbol].append(decision)
            if len(self.decision_data['decision_history'][symbol]) > 100:
                self.decision_data['decision_history'][symbol].pop(0)
            
            self.logger.system.debug(f"Decision update processed: {decision.get('id')}")
        except Exception as e:
            self.logger.log_error(f"Error processing decision update: {str(e)}", component="dashboard")
    
    def process_order_update(self, order):
        """Process order update
        
        Args:
            order: Order data
        """
        try:
            # Update order in trading data
            order_id = order.get('orderId')
            if order_id:
                self.trading_data['orders'][order_id] = order
            
            self.logger.system.debug(f"Order update processed: {order_id}")
        except Exception as e:
            self.logger.log_error(f"Error processing order update: {str(e)}", component="dashboard")
    
    def process_trade_update(self, trade):
        """Process trade update
        
        Args:
            trade: Trade data
        """
        try:
            # Add trade to trading data
            self.trading_data['trades'].append(trade)
            
            # Limit to 1000 trades
            if len(self.trading_data['trades']) > 1000:
                self.trading_data['trades'].pop(0)
            
            self.logger.system.debug(f"Trade update processed: {trade.get('tradeId')}")
        except Exception as e:
            self.logger.log_error(f"Error processing trade update: {str(e)}", component="dashboard")
    
    def update_data_service(self):
        """Update data service with latest data"""
        try:
            # Update market data
            self.data_service.update_market_data(self.market_data)
            
            # Update trading data
            self.data_service.update_trading_data(self.trading_data)
            
            # Update signal data
            self.data_service.update_signal_data(self.signal_data)
            
            # Update decision data
            self.data_service.update_decision_data(self.decision_data)
            
            self.logger.system.debug("Data service updated")
        except Exception as e:
            self.logger.log_error(f"Error updating data service: {str(e)}", component="dashboard")
    
    def handle_trading_notification(self, notification_type, data):
        """Handle trading notification
        
        Args:
            notification_type: Type of notification
            data: Notification data
        """
        try:
            # Add to update queue based on notification type
            if notification_type == 'signal':
                self.update_queue.put({
                    'type': 'signal',
                    'data': data
                })
            elif notification_type == 'decision':
                self.update_queue.put({
                    'type': 'decision',
                    'data': data
                })
            elif notification_type in ['order_created', 'order_filled', 'order_cancelled']:
                self.update_queue.put({
                    'type': 'order',
                    'data': data
                })
            elif notification_type == 'trade':
                self.update_queue.put({
                    'type': 'trade',
                    'data': data
                })
            
            self.logger.system.debug(f"Trading notification handled: {notification_type}")
        except Exception as e:
            self.logger.log_error(f"Error handling trading notification: {str(e)}", component="dashboard")
    
    def get_dashboard_data(self):
        """Get dashboard data
        
        Returns:
            dict: Dashboard data
        """
        return {
            'market_data': self.market_data,
            'trading_data': self.trading_data,
            'signal_data': self.signal_data,
            'decision_data': self.decision_data
        }


# Example usage
if __name__ == "__main__":
    # Create dashboard integration
    dashboard = EnhancedDashboardIntegration()
    
    # Start dashboard integration
    dashboard.start()
    
    # Run for a while
    try:
        print("Running Enhanced Dashboard Integration for 30 seconds...")
        
        # Create test order
        order_id = dashboard.paper_trading.create_order('BTCUSDC', 'BUY', 'LIMIT', 0.001, 105000.0)
        print(f"Created test order: {order_id}")
        
        # Wait for a bit
        time.sleep(5)
        
        # Fill order
        dashboard.paper_trading.fill_order(order_id)
        print(f"Filled test order: {order_id}")
        
        # Wait for a bit
        time.sleep(5)
        
        # Get dashboard data
        data = dashboard.get_dashboard_data()
        print(f"Market data for BTCUSDC: {data['market_data']['BTCUSDC']['last_price']}")
        print(f"Balance: {data['trading_data']['balance']}")
        print(f"Positions: {len(data['trading_data']['positions'])}")
        print(f"Orders: {len(data['trading_data']['orders'])}")
        print(f"Trades: {len(data['trading_data']['trades'])}")
        
        # Wait for the rest of the time
        time.sleep(20)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Stop dashboard integration
        dashboard.stop()
        print("Enhanced Dashboard Integration stopped")
