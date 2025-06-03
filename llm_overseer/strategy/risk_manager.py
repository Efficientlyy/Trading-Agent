#!/usr/bin/env python
"""
Risk manager module for LLM Strategic Overseer.

This module implements risk management logic for controlling position sizes,
setting stop losses, and managing overall portfolio risk.
"""

import os
import sys
import logging
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Risk manager for controlling trading risk.
    
    Implements risk management logic for controlling position sizes,
    setting stop losses, and managing overall portfolio risk.
    """
    
    def __init__(self, config):
        """
        Initialize risk manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Load risk management configuration
        self.max_position_size = self.config.get("strategy.risk.max_position_size", 0.1)  # 10% of capital
        self.max_total_exposure = self.config.get("strategy.risk.max_total_exposure", 0.5)  # 50% of capital
        self.max_drawdown = self.config.get("strategy.risk.max_drawdown", 0.1)  # 10% drawdown
        self.default_stop_loss = self.config.get("strategy.risk.default_stop_loss", 0.02)  # 2% stop loss
        self.risk_reward_ratio = self.config.get("strategy.risk.risk_reward_ratio", 2.0)  # 1:2 risk-reward
        
        # Initialize tracking variables
        self.current_exposure = 0.0
        self.peak_capital = 0.0
        self.current_capital = 0.0
        self.current_drawdown = 0.0
        self.positions = {}
        self.risk_alerts = []
        
        # Load historical data if available
        self.data_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "risk_management.json"
        )
        self._load_history()
        
        logger.info(f"Risk manager initialized with max position size {self.max_position_size*100:.0f}%")
    
    def _load_history(self) -> None:
        """Load risk management history from file."""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.current_exposure = data.get("current_exposure", 0.0)
                    self.peak_capital = data.get("peak_capital", 0.0)
                    self.current_capital = data.get("current_capital", 0.0)
                    self.current_drawdown = data.get("current_drawdown", 0.0)
                    self.positions = data.get("positions", {})
                    self.risk_alerts = data.get("risk_alerts", [])
                    
                    logger.info(f"Loaded risk management history with {len(self.positions)} positions")
            except Exception as e:
                logger.error(f"Error loading risk management history: {e}")
    
    def _save_history(self) -> None:
        """Save risk management history to file."""
        try:
            data = {
                "current_exposure": self.current_exposure,
                "peak_capital": self.peak_capital,
                "current_capital": self.current_capital,
                "current_drawdown": self.current_drawdown,
                "positions": self.positions,
                "risk_alerts": self.risk_alerts[-100:]  # Keep last 100 alerts
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved risk management history with {len(self.positions)} positions")
        except Exception as e:
            logger.error(f"Error saving risk management history: {e}")
    
    def update_capital(self, capital: float) -> None:
        """
        Update current capital.
        
        Args:
            capital: Current capital
        """
        old_capital = self.current_capital
        self.current_capital = capital
        
        # Update peak capital if current capital is higher
        if capital > self.peak_capital:
            self.peak_capital = capital
        
        # Update drawdown
        if self.peak_capital > 0:
            self.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        # Check for drawdown alert
        if self.current_drawdown > self.max_drawdown:
            self._add_risk_alert(
                "HIGH",
                f"Maximum drawdown exceeded: {self.current_drawdown*100:.2f}% > {self.max_drawdown*100:.2f}%",
                {
                    "drawdown": self.current_drawdown,
                    "max_allowed": self.max_drawdown,
                    "peak_capital": self.peak_capital,
                    "current_capital": self.current_capital
                }
            )
        
        logger.info(f"Updated capital: ${capital:.2f} (change: ${capital - old_capital:.2f})")
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss_price: float) -> Dict[str, Any]:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            entry_price: Entry price
            stop_loss_price: Stop loss price
            
        Returns:
            Position size calculation result
        """
        # Calculate risk per trade
        risk_per_trade = self.current_capital * self.max_position_size * self.default_stop_loss
        
        # Calculate price risk
        price_risk = abs(entry_price - stop_loss_price) / entry_price
        
        # Calculate position size based on risk
        if price_risk > 0:
            position_size_by_risk = risk_per_trade / (price_risk * entry_price)
        else:
            position_size_by_risk = 0
        
        # Calculate position size based on max position size
        position_size_by_max = self.current_capital * self.max_position_size / entry_price
        
        # Calculate position size based on max exposure
        remaining_exposure = self.max_total_exposure - self.current_exposure
        position_size_by_exposure = remaining_exposure * self.current_capital / entry_price
        
        # Take the minimum of all calculations
        position_size = min(position_size_by_risk, position_size_by_max, position_size_by_exposure)
        
        # Ensure position size is positive
        position_size = max(0, position_size)
        
        # Calculate position value
        position_value = position_size * entry_price
        
        # Calculate percentage of capital
        position_pct = position_value / self.current_capital if self.current_capital > 0 else 0
        
        # Prepare result
        result = {
            "symbol": symbol,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            "position_size": position_size,
            "position_value": position_value,
            "position_pct": position_pct,
            "risk_amount": position_value * price_risk,
            "risk_pct": position_pct * price_risk,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Calculated position size for {symbol}: {position_size:.8f} (${position_value:.2f}, {position_pct*100:.2f}% of capital)")
        
        return result
    
    def calculate_stop_loss(self, symbol: str, entry_price: float, direction: str, 
                          volatility: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate optimal stop loss price based on volatility and risk parameters.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            entry_price: Entry price
            direction: Trade direction ("buy" or "sell")
            volatility: Price volatility (optional)
            
        Returns:
            Stop loss calculation result
        """
        # Use default stop loss if volatility is not provided
        if volatility is None:
            volatility = self.default_stop_loss
        
        # Calculate ATR-based stop loss (minimum 1.5x default)
        atr_stop = max(volatility, self.default_stop_loss * 1.5)
        
        # Calculate stop loss price based on direction
        if direction.lower() == "buy":
            stop_loss_price = entry_price * (1 - atr_stop)
        else:
            stop_loss_price = entry_price * (1 + atr_stop)
        
        # Calculate take profit based on risk-reward ratio
        if direction.lower() == "buy":
            take_profit_price = entry_price * (1 + atr_stop * self.risk_reward_ratio)
        else:
            take_profit_price = entry_price * (1 - atr_stop * self.risk_reward_ratio)
        
        # Prepare result
        result = {
            "symbol": symbol,
            "entry_price": entry_price,
            "direction": direction,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "stop_loss_pct": abs(stop_loss_price - entry_price) / entry_price,
            "take_profit_pct": abs(take_profit_price - entry_price) / entry_price,
            "risk_reward_ratio": self.risk_reward_ratio,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Calculated stop loss for {direction.upper()} {symbol}: {stop_loss_price:.2f} ({result['stop_loss_pct']*100:.2f}%)")
        
        return result
    
    def add_position(self, symbol: str, direction: str, entry_price: float, 
                   position_size: float, stop_loss_price: float, 
                   take_profit_price: float) -> Dict[str, Any]:
        """
        Add a new position to risk management.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            direction: Trade direction ("buy" or "sell")
            entry_price: Entry price
            position_size: Position size in base currency
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            
        Returns:
            Position addition result
        """
        # Calculate position value
        position_value = position_size * entry_price
        
        # Calculate position percentage of capital
        position_pct = position_value / self.current_capital if self.current_capital > 0 else 0
        
        # Check if position exceeds max position size
        if position_pct > self.max_position_size:
            self._add_risk_alert(
                "HIGH",
                f"Position size exceeds maximum: {position_pct*100:.2f}% > {self.max_position_size*100:.2f}%",
                {
                    "symbol": symbol,
                    "position_pct": position_pct,
                    "max_allowed": self.max_position_size
                }
            )
        
        # Check if total exposure would exceed max
        new_exposure = self.current_exposure + position_pct
        if new_exposure > self.max_total_exposure:
            self._add_risk_alert(
                "HIGH",
                f"Total exposure exceeds maximum: {new_exposure*100:.2f}% > {self.max_total_exposure*100:.2f}%",
                {
                    "current_exposure": self.current_exposure,
                    "new_exposure": new_exposure,
                    "max_allowed": self.max_total_exposure
                }
            )
        
        # Create position object
        position_id = f"{symbol}_{datetime.now().timestamp()}"
        position = {
            "id": position_id,
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "position_size": position_size,
            "position_value": position_value,
            "position_pct": position_pct,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "risk_amount": position_value * abs(entry_price - stop_loss_price) / entry_price,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add position to tracking
        self.positions[position_id] = position
        
        # Update current exposure
        self.current_exposure += position_pct
        
        # Save history
        self._save_history()
        
        logger.info(f"Added {direction.upper()} position for {symbol}: {position_size:.8f} at {entry_price:.2f}")
        
        return {
            "success": True,
            "position_id": position_id,
            "position": position,
            "current_exposure": self.current_exposure,
            "timestamp": datetime.now().isoformat()
        }
    
    def close_position(self, position_id: str, exit_price: float) -> Dict[str, Any]:
        """
        Close an existing position.
        
        Args:
            position_id: Position ID
            exit_price: Exit price
            
        Returns:
            Position closure result
        """
        # Check if position exists
        if position_id not in self.positions:
            logger.warning(f"Position {position_id} not found")
            return {
                "success": False,
                "error": f"Position {position_id} not found",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get position
        position = self.positions[position_id]
        
        # Calculate profit/loss
        if position["direction"].lower() == "buy":
            pnl = (exit_price - position["entry_price"]) * position["position_size"]
            pnl_pct = (exit_price - position["entry_price"]) / position["entry_price"]
        else:
            pnl = (position["entry_price"] - exit_price) * position["position_size"]
            pnl_pct = (position["entry_price"] - exit_price) / position["entry_price"]
        
        # Update position with exit details
        position["exit_price"] = exit_price
        position["exit_timestamp"] = datetime.now().isoformat()
        position["pnl"] = pnl
        position["pnl_pct"] = pnl_pct
        position["status"] = "closed"
        
        # Update current exposure
        self.current_exposure -= position["position_pct"]
        self.current_exposure = max(0, self.current_exposure)  # Ensure non-negative
        
        # Save history
        self._save_history()
        
        logger.info(f"Closed position {position_id} at {exit_price:.2f} with P&L: ${pnl:.2f} ({pnl_pct*100:.2f}%)")
        
        return {
            "success": True,
            "position_id": position_id,
            "position": position,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "current_exposure": self.current_exposure,
            "timestamp": datetime.now().isoformat()
        }
    
    def update_position_stop_loss(self, position_id: str, new_stop_loss: float) -> Dict[str, Any]:
        """
        Update stop loss for an existing position.
        
        Args:
            position_id: Position ID
            new_stop_loss: New stop loss price
            
        Returns:
            Stop loss update result
        """
        # Check if position exists
        if position_id not in self.positions:
            logger.warning(f"Position {position_id} not found")
            return {
                "success": False,
                "error": f"Position {position_id} not found",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get position
        position = self.positions[position_id]
        
        # Check if position is closed
        if position.get("status") == "closed":
            logger.warning(f"Cannot update stop loss for closed position {position_id}")
            return {
                "success": False,
                "error": f"Cannot update stop loss for closed position {position_id}",
                "timestamp": datetime.now().isoformat()
            }
        
        # Update stop loss
        old_stop_loss = position["stop_loss_price"]
        position["stop_loss_price"] = new_stop_loss
        
        # Update risk amount
        position["risk_amount"] = position["position_value"] * abs(position["entry_price"] - new_stop_loss) / position["entry_price"]
        
        # Save history
        self._save_history()
        
        logger.info(f"Updated stop loss for position {position_id}: {old_stop_loss:.2f} -> {new_stop_loss:.2f}")
        
        return {
            "success": True,
            "position_id": position_id,
            "old_stop_loss": old_stop_loss,
            "new_stop_loss": new_stop_loss,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_position(self, position_id: str) -> Dict[str, Any]:
        """
        Get position details.
        
        Args:
            position_id: Position ID
            
        Returns:
            Position details
        """
        if position_id not in self.positions:
            logger.warning(f"Position {position_id} not found")
            return {
                "success": False,
                "error": f"Position {position_id} not found",
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "success": True,
            "position": self.positions[position_id],
            "timestamp": datetime.now().isoformat()
        }
    
    def get_active_positions(self) -> Dict[str, Any]:
        """
        Get all active positions.
        
        Returns:
            Active positions
        """
        active_positions = {
            pos_id: pos for pos_id, pos in self.positions.items()
            if pos.get("status") != "closed"
        }
        
        return {
            "success": True,
            "positions": active_positions,
            "count": len(active_positions),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics.
        
        Returns:
            Risk metrics
        """
        # Calculate risk metrics
        active_positions = {
            pos_id: pos for pos_id, pos in self.positions.items()
            if pos.get("status") != "closed"
        }
        
        total_position_value = sum(pos["position_value"] for pos in active_positions.values())
        total_risk_amount = sum(pos["risk_amount"] for pos in active_positions.values())
        
        # Calculate risk metrics
        risk_metrics = {
            "current_exposure": self.current_exposure,
            "max_exposure": self.max_total_exposure,
            "exposure_utilization": self.current_exposure / self.max_total_exposure if self.max_total_exposure > 0 else 0,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "drawdown_utilization": self.current_drawdown / self.max_drawdown if self.max_drawdown > 0 else 0,
            "total_position_value": total_position_value,
            "total_risk_amount": total_risk_amount,
            "portfolio_at_risk": total_risk_amount / self.current_capital if self.current_capital > 0 else 0,
            "active_positions_count": len(active_positions),
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "metrics": risk_metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_risk_alerts(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get recent risk alerts.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            Risk alerts
        """
        return {
            "success": True,
            "alerts": self.risk_alerts[-limit:],
            "count": len(self.risk_alerts),
            "timestamp": datetime.now().isoformat()
        }
    
    def _add_risk_alert(self, level: str, message: str, data: Dict[str, Any]) -> None:
        """
        Add a risk alert.
        
        Args:
            level: Alert level ("LOW", "MEDIUM", "HIGH")
            message: Alert message
            data: Alert data
        """
        alert = {
            "level": level,
            "message": message,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        self.risk_alerts.append(alert)
        
        logger.warning(f"Risk alert ({level}): {message}")
    
    def set_max_position_size(self, max_position_size: float) -> Dict[str, Any]:
        """
        Set maximum position size.
        
        Args:
            max_position_size: Maximum position size as percentage of capital (0.0 to 1.0)
            
        Returns:
            Update result
        """
        if max_position_size < 0.0 or max_position_size > 1.0:
            logger.warning(f"Invalid max position size: {max_position_size}, must be between 0.0 and 1.0")
            return {
                "success": False,
                "error": f"Invalid max position size: {max_position_size}, must be between 0.0 and 1.0",
                "timestamp": datetime.now().isoformat()
            }
        
        old_value = self.max_position_size
        self.max_position_size = max_position_size
        
        # Save history
        self._save_history()
        
        logger.info(f"Set max position size: {old_value*100:.0f}% -> {max_position_size*100:.0f}%")
        
        return {
            "success": True,
            "parameter": "max_position_size",
            "old_value": old_value,
            "new_value": max_position_size,
            "timestamp": datetime.now().isoformat()
        }
    
    def set_max_total_exposure(self, max_total_exposure: float) -> Dict[str, Any]:
        """
        Set maximum total exposure.
        
        Args:
            max_total_exposure: Maximum total exposure as percentage of capital (0.0 to 1.0)
            
        Returns:
            Update result
        """
        if max_total_exposure < 0.0 or max_total_exposure > 1.0:
            logger.warning(f"Invalid max total exposure: {max_total_exposure}, must be between 0.0 and 1.0")
            return {
                "success": False,
                "error": f"Invalid max total exposure: {max_total_exposure}, must be between 0.0 and 1.0",
                "timestamp": datetime.now().isoformat()
            }
        
        old_value = self.max_total_exposure
        self.max_total_exposure = max_total_exposure
        
        # Save history
        self._save_history()
        
        logger.info(f"Set max total exposure: {old_value*100:.0f}% -> {max_total_exposure*100:.0f}%")
        
        return {
            "success": True,
            "parameter": "max_total_exposure",
            "old_value": old_value,
            "new_value": max_total_exposure,
            "timestamp": datetime.now().isoformat()
        }
    
    def set_risk_reward_ratio(self, risk_reward_ratio: float) -> Dict[str, Any]:
        """
        Set risk-reward ratio.
        
        Args:
            risk_reward_ratio: Risk-reward ratio (e.g., 2.0 for 1:2)
            
        Returns:
            Update result
        """
        if risk_reward_ratio < 0.5 or risk_reward_ratio > 5.0:
            logger.warning(f"Invalid risk-reward ratio: {risk_reward_ratio}, must be between 0.5 and 5.0")
            return {
                "success": False,
                "error": f"Invalid risk-reward ratio: {risk_reward_ratio}, must be between 0.5 and 5.0",
                "timestamp": datetime.now().isoformat()
            }
        
        old_value = self.risk_reward_ratio
        self.risk_reward_ratio = risk_reward_ratio
        
        # Save history
        self._save_history()
        
        logger.info(f"Set risk-reward ratio: {old_value:.1f} -> {risk_reward_ratio:.1f}")
        
        return {
            "success": True,
            "parameter": "risk_reward_ratio",
            "old_value": old_value,
            "new_value": risk_reward_ratio,
            "timestamp": datetime.now().isoformat()
        }
