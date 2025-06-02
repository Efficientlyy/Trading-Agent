#!/usr/bin/env python
"""
Risk Management Controller for Trading-Agent System

This module provides comprehensive risk management capabilities for the Trading-Agent system,
including position sizing, exposure limits, circuit breakers, and risk metrics.
"""

import os
import sys
import json
import time
import logging
import threading
import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import error handling
from error_handling.error_manager import handle_error, ErrorCategory, ErrorSeverity, safe_execute

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("risk_management.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("risk_controller")

class RiskLevel(Enum):
    """Risk level enum"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    EXTREME = 4

class TradingStatus(Enum):
    """Trading status enum"""
    ACTIVE = "active"
    RESTRICTED = "restricted"
    SUSPENDED = "suspended"
    EMERGENCY_STOP = "emergency_stop"

class RiskParameters:
    """Risk parameters for Trading-Agent system"""
    
    def __init__(self, 
                 max_position_size_usd: float = 1000.0,
                 max_position_size_pct: float = 0.05,
                 max_total_exposure_usd: float = 5000.0,
                 max_total_exposure_pct: float = 0.25,
                 max_single_asset_exposure_pct: float = 0.10,
                 max_daily_loss_pct: float = 0.03,
                 max_drawdown_pct: float = 0.10,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.05,
                 trailing_stop_activation_pct: float = 0.01,
                 trailing_stop_distance_pct: float = 0.01,
                 price_circuit_breaker_pct: float = 0.05,
                 volume_circuit_breaker_factor: float = 3.0,
                 risk_level: RiskLevel = RiskLevel.MEDIUM):
        """Initialize risk parameters
        
        Args:
            max_position_size_usd: Maximum position size in USD
            max_position_size_pct: Maximum position size as percentage of portfolio
            max_total_exposure_usd: Maximum total exposure in USD
            max_total_exposure_pct: Maximum total exposure as percentage of portfolio
            max_single_asset_exposure_pct: Maximum single asset exposure as percentage of portfolio
            max_daily_loss_pct: Maximum daily loss as percentage of portfolio
            max_drawdown_pct: Maximum drawdown as percentage of portfolio
            stop_loss_pct: Default stop loss as percentage of entry price
            take_profit_pct: Default take profit as percentage of entry price
            trailing_stop_activation_pct: Trailing stop activation threshold as percentage of entry price
            trailing_stop_distance_pct: Trailing stop distance as percentage of current price
            price_circuit_breaker_pct: Price circuit breaker threshold as percentage change
            volume_circuit_breaker_factor: Volume circuit breaker threshold as multiple of average volume
            risk_level: Risk level
        """
        self.max_position_size_usd = max_position_size_usd
        self.max_position_size_pct = max_position_size_pct
        self.max_total_exposure_usd = max_total_exposure_usd
        self.max_total_exposure_pct = max_total_exposure_pct
        self.max_single_asset_exposure_pct = max_single_asset_exposure_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_activation_pct = trailing_stop_activation_pct
        self.trailing_stop_distance_pct = trailing_stop_distance_pct
        self.price_circuit_breaker_pct = price_circuit_breaker_pct
        self.volume_circuit_breaker_factor = volume_circuit_breaker_factor
        self.risk_level = risk_level
        
        logger.info(f"Initialized RiskParameters with risk_level={risk_level.name}")
    
    def adjust_for_risk_level(self):
        """Adjust parameters based on risk level"""
        if self.risk_level == RiskLevel.LOW:
            self.max_position_size_usd *= 0.5
            self.max_position_size_pct *= 0.5
            self.max_total_exposure_usd *= 0.5
            self.max_total_exposure_pct *= 0.5
            self.max_single_asset_exposure_pct *= 0.5
            self.stop_loss_pct *= 0.75
            self.take_profit_pct *= 1.25
            self.price_circuit_breaker_pct *= 0.75
        
        elif self.risk_level == RiskLevel.HIGH:
            self.max_position_size_usd *= 1.5
            self.max_position_size_pct *= 1.5
            self.max_total_exposure_usd *= 1.5
            self.max_total_exposure_pct *= 1.5
            self.max_single_asset_exposure_pct *= 1.5
            self.stop_loss_pct *= 1.25
            self.take_profit_pct *= 0.75
            self.price_circuit_breaker_pct *= 1.25
        
        elif self.risk_level == RiskLevel.EXTREME:
            self.max_position_size_usd *= 2.0
            self.max_position_size_pct *= 2.0
            self.max_total_exposure_usd *= 2.0
            self.max_total_exposure_pct *= 2.0
            self.max_single_asset_exposure_pct *= 2.0
            self.stop_loss_pct *= 1.5
            self.take_profit_pct *= 0.5
            self.price_circuit_breaker_pct *= 1.5
        
        logger.info(f"Adjusted parameters for risk_level={self.risk_level.name}")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary
        
        Returns:
            Dictionary representation
        """
        return {
            "max_position_size_usd": self.max_position_size_usd,
            "max_position_size_pct": self.max_position_size_pct,
            "max_total_exposure_usd": self.max_total_exposure_usd,
            "max_total_exposure_pct": self.max_total_exposure_pct,
            "max_single_asset_exposure_pct": self.max_single_asset_exposure_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "trailing_stop_activation_pct": self.trailing_stop_activation_pct,
            "trailing_stop_distance_pct": self.trailing_stop_distance_pct,
            "price_circuit_breaker_pct": self.price_circuit_breaker_pct,
            "volume_circuit_breaker_factor": self.volume_circuit_breaker_factor,
            "risk_level": self.risk_level.name
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RiskParameters':
        """Create from dictionary
        
        Args:
            data: Dictionary representation
            
        Returns:
            RiskParameters instance
        """
        risk_level = RiskLevel[data.pop("risk_level")]
        return cls(risk_level=risk_level, **data)

class Position:
    """Trading position"""
    
    def __init__(self, 
                 symbol: str,
                 quantity: float,
                 entry_price: float,
                 entry_time: datetime,
                 stop_loss: Optional[float] = None,
                 take_profit: Optional[float] = None,
                 trailing_stop: Optional[float] = None,
                 position_id: Optional[str] = None):
        """Initialize position
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            entry_price: Entry price
            entry_time: Entry time
            stop_loss: Stop loss price
            take_profit: Take profit price
            trailing_stop: Trailing stop price
            position_id: Position ID
        """
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop
        self.position_id = position_id or f"pos_{int(time.time())}_{symbol.replace('/', '_')}"
        self.last_price = entry_price
        self.highest_price = entry_price
        self.lowest_price = entry_price
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_pct = 0.0
        
        logger.info(f"Initialized Position {self.position_id}: {symbol} {quantity} @ {entry_price}")
    
    def update_price(self, current_price: float):
        """Update position with current price
        
        Args:
            current_price: Current price
        """
        self.last_price = current_price
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)
        
        # Calculate unrealized P&L
        self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        self.unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
    
    def update_trailing_stop(self, risk_params: RiskParameters):
        """Update trailing stop based on current price
        
        Args:
            risk_params: Risk parameters
        """
        if self.trailing_stop is None:
            # Initialize trailing stop if price has moved in favorable direction
            price_change_pct = (self.last_price - self.entry_price) / self.entry_price
            
            if price_change_pct >= risk_params.trailing_stop_activation_pct:
                self.trailing_stop = self.last_price * (1 - risk_params.trailing_stop_distance_pct)
                logger.info(f"Position {self.position_id}: Trailing stop activated at {self.trailing_stop}")
        
        else:
            # Update trailing stop if price has moved higher
            new_trailing_stop = self.last_price * (1 - risk_params.trailing_stop_distance_pct)
            
            if new_trailing_stop > self.trailing_stop:
                self.trailing_stop = new_trailing_stop
                logger.info(f"Position {self.position_id}: Trailing stop updated to {self.trailing_stop}")
    
    def should_close(self) -> Tuple[bool, str]:
        """Check if position should be closed
        
        Returns:
            Tuple of (should_close, reason)
        """
        if self.stop_loss is not None and self.last_price <= self.stop_loss:
            return True, "stop_loss"
        
        if self.take_profit is not None and self.last_price >= self.take_profit:
            return True, "take_profit"
        
        if self.trailing_stop is not None and self.last_price <= self.trailing_stop:
            return True, "trailing_stop"
        
        return False, ""
    
    def get_value(self) -> float:
        """Get position value
        
        Returns:
            Position value
        """
        return self.quantity * self.last_price
    
    def get_exposure(self) -> float:
        """Get position exposure
        
        Returns:
            Position exposure
        """
        return abs(self.quantity * self.last_price)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary
        
        Returns:
            Dictionary representation
        """
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_stop": self.trailing_stop,
            "last_price": self.last_price,
            "highest_price": self.highest_price,
            "lowest_price": self.lowest_price,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "value": self.get_value(),
            "exposure": self.get_exposure()
        }

class RiskController:
    """Risk controller for Trading-Agent system"""
    
    def __init__(self, 
                 portfolio_value: float = 10000.0,
                 risk_parameters: Optional[RiskParameters] = None,
                 data_window_size: int = 100):
        """Initialize risk controller
        
        Args:
            portfolio_value: Initial portfolio value
            risk_parameters: Risk parameters
            data_window_size: Size of data window for metrics
        """
        self.portfolio_value = portfolio_value
        self.risk_parameters = risk_parameters or RiskParameters()
        self.data_window_size = data_window_size
        
        # Apply risk level adjustments
        self.risk_parameters.adjust_for_risk_level()
        
        # Initialize positions and metrics
        self.positions = {}
        self.closed_positions = []
        self.daily_pnl = 0.0
        self.peak_portfolio_value = portfolio_value
        self.trading_status = TradingStatus.ACTIVE
        self.status_reason = ""
        
        # Initialize price and volume data
        self.price_data = {}
        self.volume_data = {}
        
        # Initialize circuit breaker state
        self.circuit_breakers = {}
        
        # Initialize lock for thread safety
        self.lock = threading.Lock()
        
        logger.info(f"Initialized RiskController with portfolio_value={portfolio_value}, risk_level={self.risk_parameters.risk_level.name}")
    
    def calculate_position_size(self, symbol: str, price: float, confidence: float = 0.5) -> Tuple[float, float, Dict]:
        """Calculate position size based on risk parameters
        
        Args:
            symbol: Trading symbol
            price: Current price
            confidence: Signal confidence (0.0-1.0)
            
        Returns:
            Tuple of (quantity, value, details)
        """
        with self.lock:
            # Calculate maximum position size based on USD limit
            max_size_usd = self.risk_parameters.max_position_size_usd
            
            # Calculate maximum position size based on portfolio percentage
            max_size_pct = self.portfolio_value * self.risk_parameters.max_position_size_pct
            
            # Use the smaller of the two limits
            max_position_value = min(max_size_usd, max_size_pct)
            
            # Adjust based on confidence
            adjusted_position_value = max_position_value * confidence
            
            # Calculate quantity
            quantity = adjusted_position_value / price
            
            # Round quantity to appropriate precision
            if symbol.startswith("BTC"):
                quantity = round(quantity, 6)
            elif symbol.startswith("ETH"):
                quantity = round(quantity, 5)
            else:
                quantity = round(quantity, 4)
            
            # Recalculate value
            value = quantity * price
            
            details = {
                "max_size_usd": max_size_usd,
                "max_size_pct": max_size_pct,
                "max_position_value": max_position_value,
                "confidence": confidence,
                "adjusted_position_value": adjusted_position_value,
                "quantity": quantity,
                "value": value
            }
            
            logger.info(f"Calculated position size for {symbol}: {quantity} ({value:.2f} USD)")
            
            return quantity, value, details
    
    def check_position_limits(self, symbol: str, quantity: float, price: float) -> Tuple[bool, str]:
        """Check if position is within limits
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Current price
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        with self.lock:
            # Calculate position value
            position_value = quantity * price
            
            # Check maximum position size
            if position_value > self.risk_parameters.max_position_size_usd:
                return False, f"Position value ({position_value:.2f} USD) exceeds maximum ({self.risk_parameters.max_position_size_usd:.2f} USD)"
            
            if position_value > self.portfolio_value * self.risk_parameters.max_position_size_pct:
                return False, f"Position value ({position_value:.2f} USD) exceeds maximum percentage ({self.risk_parameters.max_position_size_pct:.2%} of portfolio)"
            
            # Calculate current exposure
            current_exposure = sum(pos.get_exposure() for pos in self.positions.values())
            
            # Check total exposure
            if current_exposure + position_value > self.risk_parameters.max_total_exposure_usd:
                return False, f"Total exposure ({current_exposure + position_value:.2f} USD) would exceed maximum ({self.risk_parameters.max_total_exposure_usd:.2f} USD)"
            
            if current_exposure + position_value > self.portfolio_value * self.risk_parameters.max_total_exposure_pct:
                return False, f"Total exposure ({current_exposure + position_value:.2f} USD) would exceed maximum percentage ({self.risk_parameters.max_total_exposure_pct:.2%} of portfolio)"
            
            # Calculate current exposure for this asset
            asset = symbol.split('/')[0]
            asset_exposure = sum(pos.get_exposure() for sym, pos in self.positions.items() if sym.startswith(asset))
            
            # Check single asset exposure
            if asset_exposure + position_value > self.portfolio_value * self.risk_parameters.max_single_asset_exposure_pct:
                return False, f"{asset} exposure ({asset_exposure + position_value:.2f} USD) would exceed maximum percentage ({self.risk_parameters.max_single_asset_exposure_pct:.2%} of portfolio)"
            
            # Check daily loss limit
            if self.daily_pnl < -self.portfolio_value * self.risk_parameters.max_daily_loss_pct:
                return False, f"Daily loss limit ({self.risk_parameters.max_daily_loss_pct:.2%} of portfolio) has been reached"
            
            # Check drawdown limit
            current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
            if current_drawdown > self.risk_parameters.max_drawdown_pct:
                return False, f"Maximum drawdown ({self.risk_parameters.max_drawdown_pct:.2%}) has been reached"
            
            # Check trading status
            if self.trading_status != TradingStatus.ACTIVE:
                return False, f"Trading is currently {self.trading_status.value}: {self.status_reason}"
            
            # Check circuit breakers
            if symbol in self.circuit_breakers and self.circuit_breakers[symbol]["active"]:
                return False, f"Circuit breaker for {symbol} is active: {self.circuit_breakers[symbol]['reason']}"
            
            return True, ""
    
    def open_position(self, symbol: str, quantity: float, price: float) -> Tuple[Optional[Position], str]:
        """Open new position
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Entry price
            
        Returns:
            Tuple of (position, message)
        """
        with self.lock:
            # Check position limits
            is_allowed, reason = self.check_position_limits(symbol, quantity, price)
            
            if not is_allowed:
                logger.warning(f"Cannot open position for {symbol}: {reason}")
                return None, reason
            
            # Calculate stop loss and take profit
            stop_loss = price * (1 - self.risk_parameters.stop_loss_pct)
            take_profit = price * (1 + self.risk_parameters.take_profit_pct)
            
            # Create position
            position = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Add to positions
            self.positions[position.position_id] = position
            
            logger.info(f"Opened position {position.position_id}: {symbol} {quantity} @ {price}")
            
            return position, "Position opened successfully"
    
    def close_position(self, position_id: str, price: float, reason: str = "manual") -> Tuple[bool, str]:
        """Close position
        
        Args:
            position_id: Position ID
            price: Exit price
            reason: Close reason
            
        Returns:
            Tuple of (success, message)
        """
        with self.lock:
            if position_id not in self.positions:
                return False, f"Position {position_id} not found"
            
            position = self.positions[position_id]
            
            # Calculate realized P&L
            realized_pnl = (price - position.entry_price) * position.quantity
            realized_pnl_pct = (price - position.entry_price) / position.entry_price
            
            # Update portfolio value
            self.portfolio_value += realized_pnl
            
            # Update daily P&L
            self.daily_pnl += realized_pnl
            
            # Update peak portfolio value
            self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)
            
            # Add to closed positions
            closed_position = position.to_dict()
            closed_position.update({
                "exit_price": price,
                "exit_time": datetime.now().isoformat(),
                "realized_pnl": realized_pnl,
                "realized_pnl_pct": realized_pnl_pct,
                "close_reason": reason
            })
            
            self.closed_positions.append(closed_position)
            
            # Remove from positions
            del self.positions[position_id]
            
            logger.info(f"Closed position {position_id}: {position.symbol} {position.quantity} @ {price} ({reason})")
            
            return True, f"Position closed successfully: {realized_pnl:.2f} USD ({realized_pnl_pct:.2%})"
    
    def update_positions(self, price_updates: Dict[str, float]):
        """Update positions with current prices
        
        Args:
            price_updates: Dictionary of symbol -> price
        """
        with self.lock:
            positions_to_close = []
            
            for position_id, position in self.positions.items():
                if position.symbol in price_updates:
                    price = price_updates[position.symbol]
                    
                    # Update position
                    position.update_price(price)
                    
                    # Update trailing stop
                    position.update_trailing_stop(self.risk_parameters)
                    
                    # Check if position should be closed
                    should_close, reason = position.should_close()
                    
                    if should_close:
                        positions_to_close.append((position_id, price, reason))
            
            # Close positions
            for position_id, price, reason in positions_to_close:
                self.close_position(position_id, price, reason)
    
    def update_price_data(self, symbol: str, price: float, volume: float = None):
        """Update price and volume data
        
        Args:
            symbol: Trading symbol
            price: Current price
            volume: Current volume
        """
        with self.lock:
            # Initialize data structures if needed
            if symbol not in self.price_data:
                self.price_data[symbol] = []
            
            if symbol not in self.volume_data:
                self.volume_data[symbol] = []
            
            # Add price data
            self.price_data[symbol].append(price)
            
            # Limit data size
            if len(self.price_data[symbol]) > self.data_window_size:
                self.price_data[symbol] = self.price_data[symbol][-self.data_window_size:]
            
            # Add volume data if provided
            if volume is not None:
                self.volume_data[symbol].append(volume)
                
                # Limit data size
                if len(self.volume_data[symbol]) > self.data_window_size:
                    self.volume_data[symbol] = self.volume_data[symbol][-self.data_window_size:]
            
            # Check circuit breakers
            self.check_circuit_breakers(symbol)
    
    def check_circuit_breakers(self, symbol: str):
        """Check circuit breakers for symbol
        
        Args:
            symbol: Trading symbol
        """
        with self.lock:
            if symbol not in self.price_data or len(self.price_data[symbol]) < 2:
                return
            
            # Initialize circuit breaker state if needed
            if symbol not in self.circuit_breakers:
                self.circuit_breakers[symbol] = {
                    "active": False,
                    "reason": "",
                    "triggered_at": None,
                    "reset_at": None
                }
            
            # Get current and previous price
            current_price = self.price_data[symbol][-1]
            previous_price = self.price_data[symbol][-2]
            
            # Calculate price change
            price_change = (current_price - previous_price) / previous_price
            
            # Check price circuit breaker
            if abs(price_change) > self.risk_parameters.price_circuit_breaker_pct:
                self.circuit_breakers[symbol] = {
                    "active": True,
                    "reason": f"Price change of {price_change:.2%} exceeds threshold of {self.risk_parameters.price_circuit_breaker_pct:.2%}",
                    "triggered_at": datetime.now(),
                    "reset_at": datetime.now() + timedelta(minutes=5)
                }
                
                logger.warning(f"Circuit breaker triggered for {symbol}: {self.circuit_breakers[symbol]['reason']}")
                
                # Update trading status if needed
                if self.trading_status == TradingStatus.ACTIVE:
                    self.trading_status = TradingStatus.RESTRICTED
                    self.status_reason = f"Circuit breaker triggered for {symbol}"
                
                return
            
            # Check volume circuit breaker
            if symbol in self.volume_data and len(self.volume_data[symbol]) > 10:
                current_volume = self.volume_data[symbol][-1]
                avg_volume = sum(self.volume_data[symbol][-10:-1]) / 9  # Average of previous 9 volumes
                
                if current_volume > avg_volume * self.risk_parameters.volume_circuit_breaker_factor:
                    self.circuit_breakers[symbol] = {
                        "active": True,
                        "reason": f"Volume of {current_volume:.2f} exceeds threshold of {avg_volume * self.risk_parameters.volume_circuit_breaker_factor:.2f}",
                        "triggered_at": datetime.now(),
                        "reset_at": datetime.now() + timedelta(minutes=5)
                    }
                    
                    logger.warning(f"Circuit breaker triggered for {symbol}: {self.circuit_breakers[symbol]['reason']}")
                    
                    # Update trading status if needed
                    if self.trading_status == TradingStatus.ACTIVE:
                        self.trading_status = TradingStatus.RESTRICTED
                        self.status_reason = f"Circuit breaker triggered for {symbol}"
                    
                    return
            
            # Reset circuit breaker if it was active and reset time has passed
            if self.circuit_breakers[symbol]["active"] and self.circuit_breakers[symbol]["reset_at"] is not None:
                if datetime.now() > self.circuit_breakers[symbol]["reset_at"]:
                    self.circuit_breakers[symbol] = {
                        "active": False,
                        "reason": "",
                        "triggered_at": None,
                        "reset_at": None
                    }
                    
                    logger.info(f"Circuit breaker reset for {symbol}")
                    
                    # Update trading status if needed
                    if self.trading_status == TradingStatus.RESTRICTED and self.status_reason.startswith(f"Circuit breaker triggered for {symbol}"):
                        self.trading_status = TradingStatus.ACTIVE
                        self.status_reason = ""
    
    def reset_daily_metrics(self):
        """Reset daily metrics"""
        with self.lock:
            self.daily_pnl = 0.0
            logger.info("Daily metrics reset")
    
    def emergency_stop(self, reason: str):
        """Trigger emergency stop
        
        Args:
            reason: Emergency stop reason
        """
        with self.lock:
            self.trading_status = TradingStatus.EMERGENCY_STOP
            self.status_reason = reason
            
            logger.critical(f"EMERGENCY STOP: {reason}")
    
    def get_risk_metrics(self) -> Dict:
        """Get risk metrics
        
        Returns:
            Dictionary with risk metrics
        """
        with self.lock:
            # Calculate current exposure
            total_exposure = sum(pos.get_exposure() for pos in self.positions.values())
            exposure_pct = total_exposure / self.portfolio_value if self.portfolio_value > 0 else 0
            
            # Calculate asset exposures
            asset_exposures = {}
            for position in self.positions.values():
                asset = position.symbol.split('/')[0]
                if asset not in asset_exposures:
                    asset_exposures[asset] = 0
                asset_exposures[asset] += position.get_exposure()
            
            # Calculate asset exposure percentages
            asset_exposure_pcts = {asset: exposure / self.portfolio_value for asset, exposure in asset_exposures.items()}
            
            # Calculate drawdown
            drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0
            
            return {
                "portfolio_value": self.portfolio_value,
                "peak_portfolio_value": self.peak_portfolio_value,
                "total_exposure": total_exposure,
                "exposure_pct": exposure_pct,
                "asset_exposures": asset_exposures,
                "asset_exposure_pcts": asset_exposure_pcts,
                "daily_pnl": self.daily_pnl,
                "daily_pnl_pct": self.daily_pnl / self.portfolio_value if self.portfolio_value > 0 else 0,
                "drawdown": drawdown,
                "trading_status": self.trading_status.value,
                "status_reason": self.status_reason,
                "open_positions_count": len(self.positions),
                "closed_positions_count": len(self.closed_positions),
                "circuit_breakers": {symbol: cb for symbol, cb in self.circuit_breakers.items() if cb["active"]}
            }
    
    def get_position_summary(self) -> Dict:
        """Get position summary
        
        Returns:
            Dictionary with position summary
        """
        with self.lock:
            open_positions = [position.to_dict() for position in self.positions.values()]
            
            # Get recent closed positions (last 10)
            recent_closed = self.closed_positions[-10:] if self.closed_positions else []
            
            return {
                "open_positions": open_positions,
                "recent_closed_positions": recent_closed
            }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary
        
        Returns:
            Dictionary representation
        """
        with self.lock:
            return {
                "portfolio_value": self.portfolio_value,
                "risk_parameters": self.risk_parameters.to_dict(),
                "positions": [position.to_dict() for position in self.positions.values()],
                "closed_positions": self.closed_positions,
                "daily_pnl": self.daily_pnl,
                "peak_portfolio_value": self.peak_portfolio_value,
                "trading_status": self.trading_status.value,
                "status_reason": self.status_reason,
                "circuit_breakers": self.circuit_breakers
            }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RiskController':
        """Create from dictionary
        
        Args:
            data: Dictionary representation
            
        Returns:
            RiskController instance
        """
        risk_params = RiskParameters.from_dict(data["risk_parameters"])
        
        controller = cls(
            portfolio_value=data["portfolio_value"],
            risk_parameters=risk_params
        )
        
        controller.closed_positions = data["closed_positions"]
        controller.daily_pnl = data["daily_pnl"]
        controller.peak_portfolio_value = data["peak_portfolio_value"]
        controller.trading_status = TradingStatus(data["trading_status"])
        controller.status_reason = data["status_reason"]
        controller.circuit_breakers = data["circuit_breakers"]
        
        # Recreate positions
        for pos_data in data["positions"]:
            position = Position(
                symbol=pos_data["symbol"],
                quantity=pos_data["quantity"],
                entry_price=pos_data["entry_price"],
                entry_time=datetime.fromisoformat(pos_data["entry_time"]),
                stop_loss=pos_data["stop_loss"],
                take_profit=pos_data["take_profit"],
                trailing_stop=pos_data["trailing_stop"],
                position_id=pos_data["position_id"]
            )
            
            position.last_price = pos_data["last_price"]
            position.highest_price = pos_data["highest_price"]
            position.lowest_price = pos_data["lowest_price"]
            position.unrealized_pnl = pos_data["unrealized_pnl"]
            position.unrealized_pnl_pct = pos_data["unrealized_pnl_pct"]
            
            controller.positions[position.position_id] = position
        
        return controller

# Global risk controller instance
risk_controller = RiskController()

def calculate_position_size(symbol: str, price: float, confidence: float = 0.5) -> Tuple[float, float, Dict]:
    """Global function for calculating position size
    
    Args:
        symbol: Trading symbol
        price: Current price
        confidence: Signal confidence (0.0-1.0)
        
    Returns:
        Tuple of (quantity, value, details)
    """
    return risk_controller.calculate_position_size(symbol, price, confidence)

def check_position_limits(symbol: str, quantity: float, price: float) -> Tuple[bool, str]:
    """Global function for checking position limits
    
    Args:
        symbol: Trading symbol
        quantity: Position quantity
        price: Current price
        
    Returns:
        Tuple of (is_allowed, reason)
    """
    return risk_controller.check_position_limits(symbol, quantity, price)

def open_position(symbol: str, quantity: float, price: float) -> Tuple[Optional[Position], str]:
    """Global function for opening position
    
    Args:
        symbol: Trading symbol
        quantity: Position quantity
        price: Entry price
        
    Returns:
        Tuple of (position, message)
    """
    return risk_controller.open_position(symbol, quantity, price)

def close_position(position_id: str, price: float, reason: str = "manual") -> Tuple[bool, str]:
    """Global function for closing position
    
    Args:
        position_id: Position ID
        price: Exit price
        reason: Close reason
        
    Returns:
        Tuple of (success, message)
    """
    return risk_controller.close_position(position_id, price, reason)

def update_positions(price_updates: Dict[str, float]):
    """Global function for updating positions
    
    Args:
        price_updates: Dictionary of symbol -> price
    """
    risk_controller.update_positions(price_updates)

def update_price_data(symbol: str, price: float, volume: float = None):
    """Global function for updating price data
    
    Args:
        symbol: Trading symbol
        price: Current price
        volume: Current volume
    """
    risk_controller.update_price_data(symbol, price, volume)

def get_risk_metrics() -> Dict:
    """Global function for getting risk metrics
    
    Returns:
        Dictionary with risk metrics
    """
    return risk_controller.get_risk_metrics()

def get_position_summary() -> Dict:
    """Global function for getting position summary
    
    Returns:
        Dictionary with position summary
    """
    return risk_controller.get_position_summary()

if __name__ == "__main__":
    # Example usage
    # Initialize risk controller with custom parameters
    custom_params = RiskParameters(
        max_position_size_usd=500.0,
        risk_level=RiskLevel.LOW
    )
    
    controller = RiskController(
        portfolio_value=10000.0,
        risk_parameters=custom_params
    )
    
    # Calculate position size
    quantity, value, details = controller.calculate_position_size("BTC/USDC", 35000.0, 0.8)
    print(f"Position size: {quantity} BTC (${value:.2f})")
    
    # Check position limits
    is_allowed, reason = controller.check_position_limits("BTC/USDC", quantity, 35000.0)
    print(f"Position allowed: {is_allowed}, Reason: {reason}")
    
    # Open position
    position, message = controller.open_position("BTC/USDC", quantity, 35000.0)
    print(f"Open position: {message}")
    
    # Update price
    controller.update_price_data("BTC/USDC", 36000.0, 10.5)
    controller.update_positions({"BTC/USDC": 36000.0})
    
    # Get risk metrics
    metrics = controller.get_risk_metrics()
    print(f"Risk metrics: {metrics}")
    
    # Get position summary
    summary = controller.get_position_summary()
    print(f"Position summary: {summary}")
    
    # Close position
    if position:
        success, message = controller.close_position(position.position_id, 36000.0, "manual")
        print(f"Close position: {message}")
