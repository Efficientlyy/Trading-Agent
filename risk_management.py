#!/usr/bin/env python
"""
Risk Management Controls for Trading-Agent System

This module provides comprehensive risk management controls for the Trading-Agent system,
including position sizing, stop-loss/take-profit mechanisms, circuit breakers,
and risk exposure limits to ensure capital preservation and trading safety.
"""

import os
import json
import time
import logging
import threading
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum, auto
from dataclasses import dataclass
from collections import deque

# Import error handling and logging utilities
try:
    from error_handling_and_logging import LoggerFactory, log_execution_time, PerformanceMonitor
except ImportError:
    # Fallback if error_handling_and_logging is not available
    from logging import getLogger as LoggerFactory
    
    def log_execution_time(logger=None, level='DEBUG'):
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    class PerformanceMonitor:
        def __init__(self, logger=None):
            pass
        
        def start(self, operation_name):
            pass
        
        def end(self, operation_name, log_level='DEBUG'):
            return 0

# Configure logging
logger = LoggerFactory.get_logger(
    'risk_management',
    log_level='INFO',
    log_file='risk_management.log'
)

class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    EXTREME = auto()

class TradingStatus(Enum):
    """Trading status enumeration"""
    NORMAL = auto()
    CAUTION = auto()
    RESTRICTED = auto()
    HALTED = auto()

@dataclass
class RiskParameters:
    """Risk parameters for trading"""
    # Position sizing
    max_position_size: float = 0.1  # Maximum position size as fraction of portfolio
    max_position_value: float = 1000.0  # Maximum position value in quote currency
    
    # Stop-loss and take-profit
    stop_loss_pct: float = 0.02  # Stop-loss percentage
    take_profit_pct: float = 0.05  # Take-profit percentage
    trailing_stop_pct: float = 0.01  # Trailing stop percentage
    
    # Circuit breakers
    price_change_threshold: float = 0.05  # Price change threshold for circuit breaker
    volume_spike_threshold: float = 3.0  # Volume spike threshold as multiple of average
    
    # Risk exposure limits
    max_daily_loss_pct: float = 0.05  # Maximum daily loss as fraction of portfolio
    max_open_positions: int = 3  # Maximum number of open positions
    max_exposure_per_asset: float = 0.2  # Maximum exposure per asset as fraction of portfolio
    max_total_exposure: float = 0.5  # Maximum total exposure as fraction of portfolio
    
    # Volatility controls
    max_volatility: float = 0.03  # Maximum acceptable volatility
    volatility_lookback: int = 20  # Lookback period for volatility calculation
    
    # Time-based controls
    trading_hours_start: str = "00:00"  # Trading hours start (24h format)
    trading_hours_end: str = "23:59"  # Trading hours end (24h format)
    max_trades_per_day: int = 10  # Maximum trades per day
    min_time_between_trades: int = 300  # Minimum time between trades in seconds
    
    # Market condition controls
    market_trend_threshold: float = 0.02  # Market trend threshold
    correlation_threshold: float = 0.7  # Correlation threshold
    
    # Risk level thresholds
    medium_risk_threshold: float = 0.3  # Medium risk threshold as fraction of max risk
    high_risk_threshold: float = 0.6  # High risk threshold as fraction of max risk
    extreme_risk_threshold: float = 0.9  # Extreme risk threshold as fraction of max risk

class RiskManager:
    """Risk manager for trading system"""
    
    def __init__(self, risk_parameters=None, portfolio_value=10000.0):
        """Initialize risk manager
        
        Args:
            risk_parameters: Risk parameters
            portfolio_value: Initial portfolio value
        """
        self.risk_parameters = risk_parameters or RiskParameters()
        self.portfolio_value = portfolio_value
        self.open_positions = {}
        self.trade_history = []
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_trade_time = 0
        self.price_history = {}
        self.volume_history = {}
        self.volatility_history = {}
        self.market_trend = {}
        self.correlations = {}
        self.risk_level = RiskLevel.LOW
        self.trading_status = TradingStatus.NORMAL
        self.lock = threading.RLock()
        self.performance_monitor = PerformanceMonitor(logger=logger)
        
        # Initialize trading day
        self._reset_daily_metrics()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"Initialized RiskManager with portfolio value {portfolio_value}")
    
    def _monitoring_loop(self):
        """Monitoring loop for risk management"""
        while True:
            try:
                # Check if trading day has changed
                current_date = datetime.datetime.now().date()
                if hasattr(self, 'trading_date') and self.trading_date != current_date:
                    self._reset_daily_metrics()
                
                # Update risk metrics
                self._update_risk_metrics()
                
                # Check circuit breakers
                self._check_circuit_breakers()
                
                # Sleep for a while
                time.sleep(10)
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(30)  # Sleep longer on error
    
    def _reset_daily_metrics(self):
        """Reset daily metrics"""
        with self.lock:
            self.trading_date = datetime.datetime.now().date()
            self.daily_pnl = 0.0
            self.daily_trades = 0
            
            logger.info(f"Reset daily metrics for trading date {self.trading_date}")
    
    def _update_risk_metrics(self):
        """Update risk metrics"""
        with self.lock:
            # Calculate total exposure
            total_exposure = sum(position['value'] for position in self.open_positions.values())
            exposure_ratio = total_exposure / self.portfolio_value
            
            # Calculate risk level
            if exposure_ratio >= self.risk_parameters.extreme_risk_threshold:
                new_risk_level = RiskLevel.EXTREME
            elif exposure_ratio >= self.risk_parameters.high_risk_threshold:
                new_risk_level = RiskLevel.HIGH
            elif exposure_ratio >= self.risk_parameters.medium_risk_threshold:
                new_risk_level = RiskLevel.MEDIUM
            else:
                new_risk_level = RiskLevel.LOW
            
            # Update risk level if changed
            if new_risk_level != self.risk_level:
                logger.info(f"Risk level changed from {self.risk_level} to {new_risk_level}")
                self.risk_level = new_risk_level
            
            # Update trading status based on risk level and other factors
            self._update_trading_status()
    
    def _update_trading_status(self):
        """Update trading status"""
        with self.lock:
            # Check daily loss limit
            daily_loss_pct = self.daily_pnl / self.portfolio_value
            
            if daily_loss_pct <= -self.risk_parameters.max_daily_loss_pct:
                new_status = TradingStatus.HALTED
            elif self.risk_level == RiskLevel.EXTREME:
                new_status = TradingStatus.RESTRICTED
            elif self.risk_level == RiskLevel.HIGH:
                new_status = TradingStatus.CAUTION
            else:
                new_status = TradingStatus.NORMAL
            
            # Update trading status if changed
            if new_status != self.trading_status:
                logger.info(f"Trading status changed from {self.trading_status} to {new_status}")
                self.trading_status = new_status
    
    def _check_circuit_breakers(self):
        """Check circuit breakers"""
        with self.lock:
            triggered_breakers = []
            
            # Check price change circuit breakers
            for symbol, prices in self.price_history.items():
                if len(prices) < 2:
                    continue
                
                # Calculate price change
                price_change = (prices[-1] - prices[-2]) / prices[-2]
                
                if abs(price_change) >= self.risk_parameters.price_change_threshold:
                    triggered_breakers.append({
                        'type': 'price_change',
                        'symbol': symbol,
                        'change': price_change,
                        'threshold': self.risk_parameters.price_change_threshold
                    })
            
            # Check volume spike circuit breakers
            for symbol, volumes in self.volume_history.items():
                if len(volumes) < 10:  # Need enough history
                    continue
                
                # Calculate average volume
                avg_volume = sum(volumes[:-1]) / len(volumes[:-1])
                
                if volumes[-1] >= avg_volume * self.risk_parameters.volume_spike_threshold:
                    triggered_breakers.append({
                        'type': 'volume_spike',
                        'symbol': symbol,
                        'ratio': volumes[-1] / avg_volume,
                        'threshold': self.risk_parameters.volume_spike_threshold
                    })
            
            # Handle triggered circuit breakers
            if triggered_breakers:
                logger.warning(f"Circuit breakers triggered: {triggered_breakers}")
                
                # Update trading status
                self.trading_status = TradingStatus.RESTRICTED
    
    def can_place_order(self, symbol, side, quantity, price):
        """Check if order can be placed
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            price: Order price
            
        Returns:
            tuple: (can_place, reason)
        """
        self.performance_monitor.start("can_place_order")
        
        with self.lock:
            # Check trading status
            if self.trading_status == TradingStatus.HALTED:
                self.performance_monitor.end("can_place_order")
                return False, "Trading is halted"
            
            # Check trading hours
            current_time = datetime.datetime.now().time()
            start_time = datetime.datetime.strptime(self.risk_parameters.trading_hours_start, "%H:%M").time()
            end_time = datetime.datetime.strptime(self.risk_parameters.trading_hours_end, "%H:%M").time()
            
            if not (start_time <= current_time <= end_time):
                self.performance_monitor.end("can_place_order")
                return False, "Outside trading hours"
            
            # Check max trades per day
            if self.daily_trades >= self.risk_parameters.max_trades_per_day:
                self.performance_monitor.end("can_place_order")
                return False, "Maximum daily trades reached"
            
            # Check time between trades
            current_timestamp = time.time()
            if current_timestamp - self.last_trade_time < self.risk_parameters.min_time_between_trades:
                self.performance_monitor.end("can_place_order")
                return False, "Minimum time between trades not elapsed"
            
            # Check position sizing for buys
            if side.lower() == 'buy':
                # Calculate position value
                position_value = quantity * price
                
                # Check max position value
                if position_value > self.risk_parameters.max_position_value:
                    self.performance_monitor.end("can_place_order")
                    return False, "Position value exceeds maximum"
                
                # Check max position size
                position_size_ratio = position_value / self.portfolio_value
                if position_size_ratio > self.risk_parameters.max_position_size:
                    self.performance_monitor.end("can_place_order")
                    return False, "Position size exceeds maximum"
                
                # Check max exposure per asset
                current_exposure = self.open_positions.get(symbol, {}).get('value', 0)
                new_exposure = current_exposure + position_value
                exposure_ratio = new_exposure / self.portfolio_value
                
                if exposure_ratio > self.risk_parameters.max_exposure_per_asset:
                    self.performance_monitor.end("can_place_order")
                    return False, "Asset exposure exceeds maximum"
                
                # Check max total exposure
                total_exposure = sum(position['value'] for position in self.open_positions.values())
                new_total_exposure = total_exposure + position_value
                total_exposure_ratio = new_total_exposure / self.portfolio_value
                
                if total_exposure_ratio > self.risk_parameters.max_total_exposure:
                    self.performance_monitor.end("can_place_order")
                    return False, "Total exposure exceeds maximum"
                
                # Check max open positions
                if len(self.open_positions) >= self.risk_parameters.max_open_positions and symbol not in self.open_positions:
                    self.performance_monitor.end("can_place_order")
                    return False, "Maximum open positions reached"
            
            # Additional checks for restricted trading
            if self.trading_status == TradingStatus.RESTRICTED:
                # Only allow closing positions
                if side.lower() == 'buy' and symbol not in self.open_positions:
                    self.performance_monitor.end("can_place_order")
                    return False, "Trading restricted to closing positions only"
            
            # Additional checks for caution trading
            if self.trading_status == TradingStatus.CAUTION:
                # Reduce position sizes
                if side.lower() == 'buy':
                    # Apply 50% reduction to position size
                    reduced_max_position_size = self.risk_parameters.max_position_size * 0.5
                    position_size_ratio = position_value / self.portfolio_value
                    
                    if position_size_ratio > reduced_max_position_size:
                        self.performance_monitor.end("can_place_order")
                        return False, "Position size exceeds reduced maximum under caution"
            
            self.performance_monitor.end("can_place_order")
            return True, "Order allowed"
    
    def calculate_position_size(self, symbol, price, risk_per_trade=None):
        """Calculate appropriate position size
        
        Args:
            symbol: Trading symbol
            price: Current price
            risk_per_trade: Risk per trade (optional)
            
        Returns:
            float: Recommended position size
        """
        self.performance_monitor.start("calculate_position_size")
        
        with self.lock:
            # Default risk per trade
            if risk_per_trade is None:
                # Adjust risk based on risk level
                if self.risk_level == RiskLevel.LOW:
                    risk_per_trade = 0.01  # 1% of portfolio
                elif self.risk_level == RiskLevel.MEDIUM:
                    risk_per_trade = 0.0075  # 0.75% of portfolio
                elif self.risk_level == RiskLevel.HIGH:
                    risk_per_trade = 0.005  # 0.5% of portfolio
                else:  # EXTREME
                    risk_per_trade = 0.0025  # 0.25% of portfolio
            
            # Calculate risk amount
            risk_amount = self.portfolio_value * risk_per_trade
            
            # Get volatility for symbol
            volatility = self.volatility_history.get(symbol, [0.02])[-1]
            
            # Calculate stop-loss distance
            stop_loss_distance = price * max(volatility, self.risk_parameters.stop_loss_pct)
            
            # Calculate position size
            if stop_loss_distance > 0:
                position_size = risk_amount / stop_loss_distance
            else:
                position_size = 0
            
            # Apply position size limits
            max_position_value = min(
                self.portfolio_value * self.risk_parameters.max_position_size,
                self.risk_parameters.max_position_value
            )
            
            position_value = position_size * price
            
            if position_value > max_position_value:
                position_size = max_position_value / price
            
            # Adjust for trading status
            if self.trading_status == TradingStatus.CAUTION:
                position_size *= 0.5  # 50% reduction
            elif self.trading_status == TradingStatus.RESTRICTED:
                position_size *= 0.25  # 75% reduction
            elif self.trading_status == TradingStatus.HALTED:
                position_size = 0  # No trading
            
            self.performance_monitor.end("calculate_position_size")
            return position_size
    
    def calculate_stop_loss(self, symbol, entry_price, side, custom_pct=None):
        """Calculate stop-loss price
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            side: Order side (buy/sell)
            custom_pct: Custom stop-loss percentage (optional)
            
        Returns:
            float: Stop-loss price
        """
        with self.lock:
            # Get stop-loss percentage
            stop_loss_pct = custom_pct or self.risk_parameters.stop_loss_pct
            
            # Get volatility for symbol
            volatility = self.volatility_history.get(symbol, [0.02])[-1]
            
            # Use max of fixed percentage and volatility
            adjusted_stop_loss_pct = max(stop_loss_pct, volatility)
            
            # Calculate stop-loss price
            if side.lower() == 'buy':
                stop_loss_price = entry_price * (1 - adjusted_stop_loss_pct)
            else:
                stop_loss_price = entry_price * (1 + adjusted_stop_loss_pct)
            
            return stop_loss_price
    
    def calculate_take_profit(self, symbol, entry_price, side, custom_pct=None):
        """Calculate take-profit price
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            side: Order side (buy/sell)
            custom_pct: Custom take-profit percentage (optional)
            
        Returns:
            float: Take-profit price
        """
        with self.lock:
            # Get take-profit percentage
            take_profit_pct = custom_pct or self.risk_parameters.take_profit_pct
            
            # Calculate take-profit price
            if side.lower() == 'buy':
                take_profit_price = entry_price * (1 + take_profit_pct)
            else:
                take_profit_price = entry_price * (1 - take_profit_pct)
            
            return take_profit_price
    
    def update_trailing_stop(self, symbol, current_price, side):
        """Update trailing stop price
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            side: Order side (buy/sell)
            
        Returns:
            float: Updated trailing stop price
        """
        with self.lock:
            # Get position
            position = self.open_positions.get(symbol)
            
            if not position:
                return None
            
            # Get trailing stop percentage
            trailing_stop_pct = self.risk_parameters.trailing_stop_pct
            
            # Calculate new trailing stop
            if side.lower() == 'buy':
                # For long positions, trailing stop moves up
                new_stop = current_price * (1 - trailing_stop_pct)
                
                # Only update if new stop is higher
                if 'trailing_stop' not in position or new_stop > position['trailing_stop']:
                    position['trailing_stop'] = new_stop
            else:
                # For short positions, trailing stop moves down
                new_stop = current_price * (1 + trailing_stop_pct)
                
                # Only update if new stop is lower
                if 'trailing_stop' not in position or new_stop < position['trailing_stop']:
                    position['trailing_stop'] = new_stop
            
            return position['trailing_stop']
    
    def update_price_data(self, symbol, price, volume=None, timestamp=None):
        """Update price and volume data
        
        Args:
            symbol: Trading symbol
            price: Current price
            volume: Trading volume (optional)
            timestamp: Data timestamp (optional)
            
        Returns:
            None
        """
        with self.lock:
            # Initialize price history if needed
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=100)
            
            # Add price to history
            self.price_history[symbol].append(price)
            
            # Update volume history if provided
            if volume is not None:
                if symbol not in self.volume_history:
                    self.volume_history[symbol] = deque(maxlen=100)
                
                self.volume_history[symbol].append(volume)
            
            # Update volatility
            self._update_volatility(symbol)
            
            # Update market trend
            self._update_market_trend(symbol)
            
            # Check stop-loss and take-profit for open positions
            if symbol in self.open_positions:
                self._check_exit_conditions(symbol, price)
    
    def _update_volatility(self, symbol):
        """Update volatility for symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            None
        """
        prices = self.price_history.get(symbol, [])
        
        if len(prices) >= self.risk_parameters.volatility_lookback:
            # Calculate returns
            returns = np.array([
                (prices[i] - prices[i-1]) / prices[i-1]
                for i in range(1, len(prices))
            ])
            
            # Calculate volatility (standard deviation of returns)
            volatility = np.std(returns)
            
            # Initialize volatility history if needed
            if symbol not in self.volatility_history:
                self.volatility_history[symbol] = deque(maxlen=100)
            
            # Add volatility to history
            self.volatility_history[symbol].append(volatility)
    
    def _update_market_trend(self, symbol):
        """Update market trend for symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            None
        """
        prices = self.price_history.get(symbol, [])
        
        if len(prices) >= 20:
            # Calculate short-term and long-term moving averages
            short_ma = sum(prices[-5:]) / 5
            long_ma = sum(prices[-20:]) / 20
            
            # Calculate trend
            trend = (short_ma - long_ma) / long_ma
            
            # Update market trend
            self.market_trend[symbol] = trend
    
    def _check_exit_conditions(self, symbol, price):
        """Check exit conditions for position
        
        Args:
            symbol: Trading symbol
            price: Current price
            
        Returns:
            None
        """
        position = self.open_positions.get(symbol)
        
        if not position:
            return
        
        # Get position details
        entry_price = position['entry_price']
        side = position['side']
        stop_loss = position.get('stop_loss')
        take_profit = position.get('take_profit')
        trailing_stop = position.get('trailing_stop')
        
        # Check stop-loss
        if stop_loss is not None:
            if (side.lower() == 'buy' and price <= stop_loss) or (side.lower() == 'sell' and price >= stop_loss):
                logger.info(f"Stop-loss triggered for {symbol} at {price}")
                self._close_position(symbol, price, 'stop_loss')
                return
        
        # Check take-profit
        if take_profit is not None:
            if (side.lower() == 'buy' and price >= take_profit) or (side.lower() == 'sell' and price <= take_profit):
                logger.info(f"Take-profit triggered for {symbol} at {price}")
                self._close_position(symbol, price, 'take_profit')
                return
        
        # Check trailing stop
        if trailing_stop is not None:
            if (side.lower() == 'buy' and price <= trailing_stop) or (side.lower() == 'sell' and price >= trailing_stop):
                logger.info(f"Trailing stop triggered for {symbol} at {price}")
                self._close_position(symbol, price, 'trailing_stop')
                return
    
    def _close_position(self, symbol, price, reason):
        """Close position
        
        Args:
            symbol: Trading symbol
            price: Current price
            reason: Close reason
            
        Returns:
            None
        """
        with self.lock:
            position = self.open_positions.get(symbol)
            
            if not position:
                return
            
            # Calculate PnL
            entry_price = position['entry_price']
            quantity = position['quantity']
            side = position['side']
            
            if side.lower() == 'buy':
                pnl = (price - entry_price) * quantity
            else:
                pnl = (entry_price - price) * quantity
            
            # Update daily PnL
            self.daily_pnl += pnl
            
            # Add to trade history
            trade_record = {
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'exit_price': price,
                'quantity': quantity,
                'pnl': pnl,
                'entry_time': position['entry_time'],
                'exit_time': time.time(),
                'reason': reason
            }
            
            self.trade_history.append(trade_record)
            
            # Remove from open positions
            del self.open_positions[symbol]
            
            logger.info(f"Closed position for {symbol} at {price}, PnL: {pnl}, reason: {reason}")
    
    def open_position(self, symbol, side, quantity, price):
        """Open new position
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            price: Order price
            
        Returns:
            dict: Position details
        """
        with self.lock:
            # Check if can place order
            can_place, reason = self.can_place_order(symbol, side, quantity, price)
            
            if not can_place:
                logger.warning(f"Cannot open position for {symbol}: {reason}")
                return None
            
            # Calculate position value
            position_value = quantity * price
            
            # Create position
            position = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'entry_price': price,
                'value': position_value,
                'entry_time': time.time()
            }
            
            # Calculate stop-loss and take-profit
            position['stop_loss'] = self.calculate_stop_loss(symbol, price, side)
            position['take_profit'] = self.calculate_take_profit(symbol, price, side)
            position['trailing_stop'] = None  # Will be updated later
            
            # Add to open positions
            self.open_positions[symbol] = position
            
            # Update metrics
            self.daily_trades += 1
            self.last_trade_time = time.time()
            
            logger.info(f"Opened position for {symbol} at {price}, quantity: {quantity}, side: {side}")
            
            return position
    
    def update_position(self, symbol, quantity=None, stop_loss=None, take_profit=None):
        """Update existing position
        
        Args:
            symbol: Trading symbol
            quantity: New quantity (optional)
            stop_loss: New stop-loss price (optional)
            take_profit: New take-profit price (optional)
            
        Returns:
            dict: Updated position details
        """
        with self.lock:
            position = self.open_positions.get(symbol)
            
            if not position:
                logger.warning(f"Position not found for {symbol}")
                return None
            
            # Update quantity if provided
            if quantity is not None:
                old_quantity = position['quantity']
                position['quantity'] = quantity
                position['value'] = quantity * position['entry_price']
                
                logger.info(f"Updated quantity for {symbol} from {old_quantity} to {quantity}")
            
            # Update stop-loss if provided
            if stop_loss is not None:
                old_stop_loss = position.get('stop_loss')
                position['stop_loss'] = stop_loss
                
                logger.info(f"Updated stop-loss for {symbol} from {old_stop_loss} to {stop_loss}")
            
            # Update take-profit if provided
            if take_profit is not None:
                old_take_profit = position.get('take_profit')
                position['take_profit'] = take_profit
                
                logger.info(f"Updated take-profit for {symbol} from {old_take_profit} to {take_profit}")
            
            return position
    
    def get_position(self, symbol):
        """Get position details
        
        Args:
            symbol: Trading symbol
            
        Returns:
            dict: Position details
        """
        with self.lock:
            return self.open_positions.get(symbol)
    
    def get_all_positions(self):
        """Get all open positions
        
        Returns:
            dict: All open positions
        """
        with self.lock:
            return self.open_positions.copy()
    
    def get_trade_history(self, limit=None):
        """Get trade history
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            list: Trade history
        """
        with self.lock:
            if limit is None:
                return self.trade_history.copy()
            else:
                return self.trade_history[-limit:].copy()
    
    def get_risk_metrics(self):
        """Get current risk metrics
        
        Returns:
            dict: Risk metrics
        """
        with self.lock:
            # Calculate total exposure
            total_exposure = sum(position['value'] for position in self.open_positions.values())
            exposure_ratio = total_exposure / self.portfolio_value
            
            # Calculate asset exposures
            asset_exposures = {
                symbol: position['value'] / self.portfolio_value
                for symbol, position in self.open_positions.items()
            }
            
            # Calculate volatilities
            volatilities = {
                symbol: values[-1] if values else None
                for symbol, values in self.volatility_history.items()
            }
            
            # Calculate market trends
            market_trends = self.market_trend.copy()
            
            return {
                'portfolio_value': self.portfolio_value,
                'daily_pnl': self.daily_pnl,
                'daily_pnl_pct': self.daily_pnl / self.portfolio_value,
                'daily_trades': self.daily_trades,
                'max_trades_per_day': self.risk_parameters.max_trades_per_day,
                'open_positions_count': len(self.open_positions),
                'max_open_positions': self.risk_parameters.max_open_positions,
                'total_exposure': total_exposure,
                'total_exposure_ratio': exposure_ratio,
                'max_total_exposure': self.risk_parameters.max_total_exposure,
                'asset_exposures': asset_exposures,
                'max_exposure_per_asset': self.risk_parameters.max_exposure_per_asset,
                'volatilities': volatilities,
                'market_trends': market_trends,
                'risk_level': self.risk_level.name,
                'trading_status': self.trading_status.name
            }
    
    def update_portfolio_value(self, new_value):
        """Update portfolio value
        
        Args:
            new_value: New portfolio value
            
        Returns:
            None
        """
        with self.lock:
            old_value = self.portfolio_value
            self.portfolio_value = new_value
            
            logger.info(f"Updated portfolio value from {old_value} to {new_value}")
    
    def update_risk_parameters(self, new_parameters):
        """Update risk parameters
        
        Args:
            new_parameters: New risk parameters
            
        Returns:
            None
        """
        with self.lock:
            # Update parameters
            for key, value in new_parameters.items():
                if hasattr(self.risk_parameters, key):
                    setattr(self.risk_parameters, key, value)
            
            logger.info(f"Updated risk parameters: {new_parameters}")
    
    def get_risk_parameters(self):
        """Get current risk parameters
        
        Returns:
            RiskParameters: Risk parameters
        """
        with self.lock:
            return self.risk_parameters

class RiskDashboard:
    """Dashboard for risk management visualization"""
    
    def __init__(self, risk_manager):
        """Initialize risk dashboard
        
        Args:
            risk_manager: Risk manager instance
        """
        self.risk_manager = risk_manager
        self.logger = logger
        
        self.logger.info("Initialized RiskDashboard")
    
    def get_dashboard_data(self):
        """Get dashboard data
        
        Returns:
            dict: Dashboard data
        """
        # Get risk metrics
        risk_metrics = self.risk_manager.get_risk_metrics()
        
        # Get open positions
        positions = self.risk_manager.get_all_positions()
        
        # Get trade history
        trade_history = self.risk_manager.get_trade_history(limit=10)
        
        # Format positions for display
        formatted_positions = []
        for symbol, position in positions.items():
            entry_price = position['entry_price']
            current_price = self.risk_manager.price_history.get(symbol, [entry_price])[-1]
            
            if position['side'].lower() == 'buy':
                pnl = (current_price - entry_price) * position['quantity']
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl = (entry_price - current_price) * position['quantity']
                pnl_pct = (entry_price - current_price) / entry_price
            
            formatted_positions.append({
                'symbol': symbol,
                'side': position['side'],
                'quantity': position['quantity'],
                'entry_price': entry_price,
                'current_price': current_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'stop_loss': position.get('stop_loss'),
                'take_profit': position.get('take_profit'),
                'trailing_stop': position.get('trailing_stop')
            })
        
        # Format trade history for display
        formatted_history = []
        for trade in trade_history:
            pnl_pct = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
            if trade['side'].lower() == 'sell':
                pnl_pct = -pnl_pct
            
            formatted_history.append({
                'symbol': trade['symbol'],
                'side': trade['side'],
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'quantity': trade['quantity'],
                'pnl': trade['pnl'],
                'pnl_pct': pnl_pct,
                'entry_time': datetime.datetime.fromtimestamp(trade['entry_time']).strftime('%Y-%m-%d %H:%M:%S'),
                'exit_time': datetime.datetime.fromtimestamp(trade['exit_time']).strftime('%Y-%m-%d %H:%M:%S'),
                'reason': trade['reason']
            })
        
        # Create risk indicators
        risk_indicators = {
            'portfolio_value': risk_metrics['portfolio_value'],
            'daily_pnl': risk_metrics['daily_pnl'],
            'daily_pnl_pct': risk_metrics['daily_pnl_pct'],
            'daily_trades': f"{risk_metrics['daily_trades']} / {risk_metrics['max_trades_per_day']}",
            'open_positions': f"{risk_metrics['open_positions_count']} / {risk_metrics['max_open_positions']}",
            'total_exposure': f"{risk_metrics['total_exposure_ratio'] * 100:.1f}% / {risk_metrics['max_total_exposure'] * 100:.1f}%",
            'risk_level': risk_metrics['risk_level'],
            'trading_status': risk_metrics['trading_status']
        }
        
        # Create exposure chart data
        exposure_data = []
        for symbol, exposure in risk_metrics['asset_exposures'].items():
            exposure_data.append({
                'symbol': symbol,
                'exposure': exposure,
                'max_exposure': risk_metrics['max_exposure_per_asset']
            })
        
        # Create volatility chart data
        volatility_data = []
        for symbol, volatility in risk_metrics['volatilities'].items():
            if volatility is not None:
                volatility_data.append({
                    'symbol': symbol,
                    'volatility': volatility
                })
        
        # Create market trend chart data
        trend_data = []
        for symbol, trend in risk_metrics['market_trends'].items():
            trend_data.append({
                'symbol': symbol,
                'trend': trend
            })
        
        return {
            'risk_indicators': risk_indicators,
            'positions': formatted_positions,
            'trade_history': formatted_history,
            'exposure_data': exposure_data,
            'volatility_data': volatility_data,
            'trend_data': trend_data
        }
    
    def get_risk_alerts(self):
        """Get risk alerts
        
        Returns:
            list: Risk alerts
        """
        alerts = []
        risk_metrics = self.risk_manager.get_risk_metrics()
        
        # Check daily PnL
        if risk_metrics['daily_pnl_pct'] <= -0.03:
            alerts.append({
                'level': 'high',
                'message': f"Daily loss exceeds 3%: {risk_metrics['daily_pnl_pct'] * 100:.1f}%"
            })
        elif risk_metrics['daily_pnl_pct'] <= -0.01:
            alerts.append({
                'level': 'medium',
                'message': f"Daily loss exceeds 1%: {risk_metrics['daily_pnl_pct'] * 100:.1f}%"
            })
        
        # Check exposure
        if risk_metrics['total_exposure_ratio'] >= 0.8 * risk_metrics['max_total_exposure']:
            alerts.append({
                'level': 'high',
                'message': f"Total exposure near maximum: {risk_metrics['total_exposure_ratio'] * 100:.1f}% / {risk_metrics['max_total_exposure'] * 100:.1f}%"
            })
        
        # Check trading status
        if risk_metrics['trading_status'] == 'HALTED':
            alerts.append({
                'level': 'critical',
                'message': "Trading is halted due to risk limits"
            })
        elif risk_metrics['trading_status'] == 'RESTRICTED':
            alerts.append({
                'level': 'high',
                'message': "Trading is restricted to closing positions only"
            })
        elif risk_metrics['trading_status'] == 'CAUTION':
            alerts.append({
                'level': 'medium',
                'message': "Trading under caution with reduced position sizes"
            })
        
        # Check positions near stop-loss
        positions = self.risk_manager.get_all_positions()
        for symbol, position in positions.items():
            current_price = self.risk_manager.price_history.get(symbol, [position['entry_price']])[-1]
            stop_loss = position.get('stop_loss')
            
            if stop_loss is not None:
                if position['side'].lower() == 'buy':
                    distance_to_stop = (current_price - stop_loss) / current_price
                    if distance_to_stop <= 0.01:
                        alerts.append({
                            'level': 'high',
                            'message': f"{symbol} price near stop-loss: {distance_to_stop * 100:.1f}% away"
                        })
                else:
                    distance_to_stop = (stop_loss - current_price) / current_price
                    if distance_to_stop <= 0.01:
                        alerts.append({
                            'level': 'high',
                            'message': f"{symbol} price near stop-loss: {distance_to_stop * 100:.1f}% away"
                        })
        
        return alerts
    
    def get_risk_recommendations(self):
        """Get risk management recommendations
        
        Returns:
            list: Risk recommendations
        """
        recommendations = []
        risk_metrics = self.risk_manager.get_risk_metrics()
        
        # Check risk level
        if risk_metrics['risk_level'] == 'HIGH' or risk_metrics['risk_level'] == 'EXTREME':
            recommendations.append({
                'type': 'reduce_exposure',
                'message': f"Consider reducing overall exposure due to {risk_metrics['risk_level']} risk level"
            })
        
        # Check daily PnL
        if risk_metrics['daily_pnl_pct'] <= -0.02:
            recommendations.append({
                'type': 'reduce_trading',
                'message': f"Consider reducing trading activity due to daily loss of {risk_metrics['daily_pnl_pct'] * 100:.1f}%"
            })
        
        # Check volatility
        high_volatility_symbols = []
        for symbol, volatility in risk_metrics['volatilities'].items():
            if volatility is not None and volatility > self.risk_manager.risk_parameters.max_volatility:
                high_volatility_symbols.append(symbol)
        
        if high_volatility_symbols:
            recommendations.append({
                'type': 'high_volatility',
                'message': f"High volatility detected for {', '.join(high_volatility_symbols)}, consider adjusting position sizes"
            })
        
        # Check market trends
        negative_trend_symbols = []
        for symbol, trend in risk_metrics['market_trends'].items():
            if trend < -self.risk_manager.risk_parameters.market_trend_threshold:
                negative_trend_symbols.append(symbol)
        
        if negative_trend_symbols:
            recommendations.append({
                'type': 'negative_trend',
                'message': f"Negative market trend detected for {', '.join(negative_trend_symbols)}, consider defensive positioning"
            })
        
        # Check position concentration
        if risk_metrics['open_positions_count'] > 0:
            max_exposure = max(risk_metrics['asset_exposures'].values())
            max_symbol = max(risk_metrics['asset_exposures'].items(), key=lambda x: x[1])[0]
            
            if max_exposure > 0.5 * risk_metrics['total_exposure_ratio']:
                recommendations.append({
                    'type': 'concentration',
                    'message': f"High concentration in {max_symbol} ({max_exposure * 100:.1f}% of portfolio), consider diversifying"
                })
        
        return recommendations

# Example usage
if __name__ == "__main__":
    # Create risk manager
    risk_manager = RiskManager(portfolio_value=10000.0)
    
    # Create risk dashboard
    risk_dashboard = RiskDashboard(risk_manager)
    
    # Update price data
    risk_manager.update_price_data("BTC/USDC", 35000.0, 10.0)
    risk_manager.update_price_data("ETH/USDC", 2000.0, 100.0)
    
    # Open positions
    risk_manager.open_position("BTC/USDC", "buy", 0.1, 35000.0)
    risk_manager.open_position("ETH/USDC", "buy", 1.0, 2000.0)
    
    # Update price data (price increase)
    risk_manager.update_price_data("BTC/USDC", 36000.0, 15.0)
    risk_manager.update_price_data("ETH/USDC", 2100.0, 120.0)
    
    # Update trailing stops
    risk_manager.update_trailing_stop("BTC/USDC", 36000.0, "buy")
    risk_manager.update_trailing_stop("ETH/USDC", 2100.0, "buy")
    
    # Get dashboard data
    dashboard_data = risk_dashboard.get_dashboard_data()
    
    # Print dashboard data
    print("Risk Indicators:")
    for key, value in dashboard_data['risk_indicators'].items():
        print(f"  {key}: {value}")
    
    print("\nOpen Positions:")
    for position in dashboard_data['positions']:
        print(f"  {position['symbol']}: {position['quantity']} @ {position['entry_price']}, PnL: {position['pnl']:.2f} ({position['pnl_pct'] * 100:.2f}%)")
    
    # Get risk alerts
    alerts = risk_dashboard.get_risk_alerts()
    
    print("\nRisk Alerts:")
    for alert in alerts:
        print(f"  [{alert['level']}] {alert['message']}")
    
    # Get risk recommendations
    recommendations = risk_dashboard.get_risk_recommendations()
    
    print("\nRisk Recommendations:")
    for recommendation in recommendations:
        print(f"  [{recommendation['type']}] {recommendation['message']}")
    
    # Update price data (price decrease to trigger stop-loss)
    risk_manager.update_price_data("BTC/USDC", 34000.0, 20.0)
    
    # Get updated positions
    positions = risk_manager.get_all_positions()
    
    print("\nUpdated Positions:")
    for symbol, position in positions.items():
        print(f"  {symbol}: {position['quantity']} @ {position['entry_price']}")
    
    # Get trade history
    trade_history = risk_manager.get_trade_history()
    
    print("\nTrade History:")
    for trade in trade_history:
        print(f"  {trade['symbol']}: {trade['quantity']} @ {trade['entry_price']} -> {trade['exit_price']}, PnL: {trade['pnl']:.2f}, Reason: {trade['reason']}")
