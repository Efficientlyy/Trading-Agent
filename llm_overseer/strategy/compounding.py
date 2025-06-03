#!/usr/bin/env python
"""
Capital allocation strategy module for LLM Strategic Overseer.

This module implements configurable capital allocation logic for trading,
allowing a specified percentage of available USDC to be used for trading.
"""

import os
import sys
import logging
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CapitalAllocationStrategy:
    """
    Capital allocation strategy for trading.
    
    Implements configurable capital allocation logic to determine
    how much of the available USDC should be used for trading.
    """
    
    def __init__(self, config):
        """
        Initialize capital allocation strategy.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Load allocation configuration
        self.enabled = self.config.get("trading.allocation.enabled", True)
        self.allocation_percentage = self.config.get("trading.allocation.percentage", 0.8)  # 80% by default
        self.min_reserve = self.config.get("trading.allocation.min_reserve", 100)  # $100 minimum reserve
        
        # Initialize tracking variables
        self.total_capital = 0.0
        self.allocated_capital = 0.0
        self.reserve_capital = 0.0
        self.last_allocation_date = None
        
        # Load historical data if available
        self.history = []
        self.data_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "allocation_history.json"
        )
        self._load_history()
        
        logger.info(f"Capital allocation strategy initialized with {self.allocation_percentage*100:.0f}% allocation percentage")
    
    def _load_history(self) -> None:
        """Load allocation history from file."""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.history = data.get("history", [])
                    self.total_capital = data.get("total_capital", 0.0)
                    self.allocated_capital = data.get("allocated_capital", 0.0)
                    self.reserve_capital = data.get("reserve_capital", 0.0)
                    
                    last_date = data.get("last_allocation_date")
                    if last_date:
                        self.last_allocation_date = datetime.fromisoformat(last_date)
                    
                    logger.info(f"Loaded allocation history: {len(self.history)} records")
            except Exception as e:
                logger.error(f"Error loading allocation history: {e}")
    
    def _save_history(self) -> None:
        """Save allocation history to file."""
        try:
            data = {
                "history": self.history,
                "total_capital": self.total_capital,
                "allocated_capital": self.allocated_capital,
                "reserve_capital": self.reserve_capital,
                "last_allocation_date": self.last_allocation_date.isoformat() if self.last_allocation_date else None
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved allocation history: {len(self.history)} records")
        except Exception as e:
            logger.error(f"Error saving allocation history: {e}")
    
    def update_capital(self, available_usdc: float) -> None:
        """
        Update available capital.
        
        Args:
            available_usdc: Total available USDC
        """
        old_capital = self.total_capital
        self.total_capital = available_usdc
        
        # Calculate change
        change = available_usdc - old_capital
        
        logger.info(f"Updated capital: ${available_usdc:.2f} (change: ${change:.2f})")
    
    def calculate_allocation(self, available_usdc: float) -> Dict[str, Any]:
        """
        Calculate capital allocation for trading.
        
        Args:
            available_usdc: Total available USDC
            
        Returns:
            Capital allocation calculation results
        """
        # Update total capital
        self.update_capital(available_usdc)
        
        # Calculate allocation
        raw_allocation = available_usdc * self.allocation_percentage
        raw_reserve = available_usdc - raw_allocation
        
        # Ensure minimum reserve
        if raw_reserve < self.min_reserve:
            # Adjust allocation to maintain minimum reserve
            reserve = self.min_reserve
            allocation = max(0, available_usdc - reserve)
        else:
            allocation = raw_allocation
            reserve = raw_reserve
        
        # Calculate allocation percentage (actual)
        actual_percentage = allocation / available_usdc if available_usdc > 0 else 0
        
        return {
            "total_capital": available_usdc,
            "allocation_amount": allocation,
            "reserve_amount": reserve,
            "target_percentage": self.allocation_percentage,
            "actual_percentage": actual_percentage,
            "timestamp": datetime.now().isoformat()
        }
    
    def execute_allocation(self, available_usdc: float) -> Dict[str, Any]:
        """
        Execute capital allocation strategy.
        
        Args:
            available_usdc: Total available USDC
            
        Returns:
            Capital allocation execution results
        """
        # Calculate allocation
        calc = self.calculate_allocation(available_usdc)
        
        # Update tracking variables
        self.allocated_capital = calc["allocation_amount"]
        self.reserve_capital = calc["reserve_amount"]
        self.last_allocation_date = datetime.now()
        
        # Record in history
        self.history.append({
            "date": self.last_allocation_date.isoformat(),
            "total_capital": available_usdc,
            "allocated": self.allocated_capital,
            "reserve": self.reserve_capital,
            "allocation_percentage": self.allocation_percentage
        })
        
        # Save history
        self._save_history()
        
        logger.info(f"Executed allocation: ${self.allocated_capital:.2f} allocated, ${self.reserve_capital:.2f} reserve")
        
        return {
            "allocated": True,
            "date": self.last_allocation_date.isoformat(),
            "total_capital": available_usdc,
            "allocated_amount": self.allocated_capital,
            "reserve_amount": self.reserve_capital,
            "allocation_percentage": self.allocation_percentage
        }
    
    def get_trading_capital(self, available_usdc: float) -> float:
        """
        Get capital amount to use for trading.
        
        Args:
            available_usdc: Total available USDC
            
        Returns:
            Amount to use for trading
        """
        # Calculate allocation
        calc = self.calculate_allocation(available_usdc)
        
        # Return allocation amount
        return calc["allocation_amount"]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get allocation statistics.
        
        Returns:
            Allocation statistics
        """
        return {
            "enabled": self.enabled,
            "allocation_percentage": self.allocation_percentage,
            "min_reserve": self.min_reserve,
            "total_capital": self.total_capital,
            "allocated_capital": self.allocated_capital,
            "reserve_capital": self.reserve_capital,
            "last_allocation_date": self.last_allocation_date.isoformat() if self.last_allocation_date else None,
            "allocation_events": len(self.history)
        }
    
    def set_allocation_percentage(self, percentage: float) -> Dict[str, Any]:
        """
        Set allocation percentage.
        
        Args:
            percentage: Allocation percentage (0.0 to 1.0)
            
        Returns:
            Update result
        """
        if percentage < 0.0 or percentage > 1.0:
            logger.warning(f"Invalid allocation percentage: {percentage}, must be between 0.0 and 1.0")
            return {
                "success": False,
                "error": f"Invalid allocation percentage: {percentage}, must be between 0.0 and 1.0",
                "timestamp": datetime.now().isoformat()
            }
        
        old_value = self.allocation_percentage
        self.allocation_percentage = percentage
        
        logger.info(f"Set allocation percentage to {percentage*100:.0f}%")
        
        return {
            "success": True,
            "parameter": "allocation_percentage",
            "old_value": old_value,
            "new_value": percentage,
            "timestamp": datetime.now().isoformat()
        }
    
    def set_min_reserve(self, min_reserve: float) -> Dict[str, Any]:
        """
        Set minimum reserve amount.
        
        Args:
            min_reserve: Minimum reserve amount
            
        Returns:
            Update result
        """
        if min_reserve < 0.0:
            logger.warning(f"Invalid minimum reserve: {min_reserve}, must be non-negative")
            return {
                "success": False,
                "error": f"Invalid minimum reserve: {min_reserve}, must be non-negative",
                "timestamp": datetime.now().isoformat()
            }
        
        old_value = self.min_reserve
        self.min_reserve = min_reserve
        
        logger.info(f"Set minimum reserve to ${min_reserve:.2f}")
        
        return {
            "success": True,
            "parameter": "min_reserve",
            "old_value": old_value,
            "new_value": min_reserve,
            "timestamp": datetime.now().isoformat()
        }
    
    def enable(self) -> Dict[str, Any]:
        """
        Enable capital allocation strategy.
        
        Returns:
            Update result
        """
        old_value = self.enabled
        self.enabled = True
        
        logger.info("Capital allocation strategy enabled")
        
        return {
            "success": True,
            "parameter": "enabled",
            "old_value": old_value,
            "new_value": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def disable(self) -> Dict[str, Any]:
        """
        Disable capital allocation strategy.
        
        Returns:
            Update result
        """
        old_value = self.enabled
        self.enabled = False
        
        logger.info("Capital allocation strategy disabled")
        
        return {
            "success": True,
            "parameter": "enabled",
            "old_value": old_value,
            "new_value": False,
            "timestamp": datetime.now().isoformat()
        }


# For backward compatibility, keep the old class name as an alias
CompoundingStrategy = CapitalAllocationStrategy
