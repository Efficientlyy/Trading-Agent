#!/usr/bin/env python
"""
Compounding strategy module for LLM Strategic Overseer.

This module implements profit reinvestment logic with configurable
compounding rates for optimizing trading capital growth.
"""

import os
import sys
import logging
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CompoundingStrategy:
    """
    Compounding strategy for profit reinvestment.
    
    Implements configurable profit reinvestment logic to optimize
    trading capital growth over time.
    """
    
    def __init__(self, config):
        """
        Initialize compounding strategy.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Load compounding configuration
        self.enabled = self.config.get("trading.compounding.enabled", True)
        self.reinvestment_rate = self.config.get("trading.compounding.reinvestment_rate", 0.8)  # 80% by default
        self.min_profit_threshold = self.config.get("trading.compounding.min_profit_threshold", 100)  # $100 minimum
        self.frequency = self.config.get("trading.compounding.frequency", "monthly")  # monthly, weekly, or daily
        
        # Initialize tracking variables
        self.initial_capital = 0.0
        self.current_capital = 0.0
        self.total_profit = 0.0
        self.reinvested_profit = 0.0
        self.withdrawn_profit = 0.0
        self.last_compounding_date = None
        
        # Load historical data if available
        self.history = []
        self.data_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "compounding_history.json"
        )
        self._load_history()
        
        logger.info(f"Compounding strategy initialized with {self.reinvestment_rate*100:.0f}% reinvestment rate")
    
    def _load_history(self) -> None:
        """Load compounding history from file."""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.history = data.get("history", [])
                    self.initial_capital = data.get("initial_capital", 0.0)
                    self.current_capital = data.get("current_capital", 0.0)
                    self.total_profit = data.get("total_profit", 0.0)
                    self.reinvested_profit = data.get("reinvested_profit", 0.0)
                    self.withdrawn_profit = data.get("withdrawn_profit", 0.0)
                    
                    last_date = data.get("last_compounding_date")
                    if last_date:
                        self.last_compounding_date = datetime.fromisoformat(last_date)
                    
                    logger.info(f"Loaded compounding history: {len(self.history)} records")
            except Exception as e:
                logger.error(f"Error loading compounding history: {e}")
    
    def _save_history(self) -> None:
        """Save compounding history to file."""
        try:
            data = {
                "history": self.history,
                "initial_capital": self.initial_capital,
                "current_capital": self.current_capital,
                "total_profit": self.total_profit,
                "reinvested_profit": self.reinvested_profit,
                "withdrawn_profit": self.withdrawn_profit,
                "last_compounding_date": self.last_compounding_date.isoformat() if self.last_compounding_date else None
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved compounding history: {len(self.history)} records")
        except Exception as e:
            logger.error(f"Error saving compounding history: {e}")
    
    def initialize_capital(self, initial_capital: float) -> None:
        """
        Initialize trading capital.
        
        Args:
            initial_capital: Initial trading capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self._save_history()
        
        logger.info(f"Initialized capital: ${initial_capital:.2f}")
    
    def update_capital(self, current_capital: float) -> None:
        """
        Update current trading capital.
        
        Args:
            current_capital: Current trading capital
        """
        old_capital = self.current_capital
        self.current_capital = current_capital
        
        # Calculate profit since last update
        profit = current_capital - old_capital
        self.total_profit += profit
        
        logger.info(f"Updated capital: ${current_capital:.2f} (change: ${profit:.2f})")
    
    def should_compound(self, current_date: Optional[datetime] = None) -> bool:
        """
        Check if compounding should be performed.
        
        Args:
            current_date: Current date (defaults to now)
            
        Returns:
            True if compounding should be performed, False otherwise
        """
        if not self.enabled:
            return False
        
        if not current_date:
            current_date = datetime.now()
        
        if not self.last_compounding_date:
            # First time compounding
            self.last_compounding_date = current_date
            return True
        
        # Check if enough time has passed based on frequency
        if self.frequency == "daily":
            return (current_date - self.last_compounding_date).days >= 1
        elif self.frequency == "weekly":
            return (current_date - self.last_compounding_date).days >= 7
        else:  # monthly
            # Check if we're in a new month
            return (current_date.year > self.last_compounding_date.year or 
                    current_date.month > self.last_compounding_date.month)
    
    def calculate_compounding(self, current_capital: float, 
                             current_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate compounding amounts.
        
        Args:
            current_capital: Current trading capital
            current_date: Current date (defaults to now)
            
        Returns:
            Compounding calculation results
        """
        if not current_date:
            current_date = datetime.now()
        
        # Update current capital
        self.update_capital(current_capital)
        
        # Calculate profit
        profit = current_capital - self.initial_capital
        
        # Check if profit meets minimum threshold
        if profit < self.min_profit_threshold:
            return {
                "can_compound": False,
                "reason": f"Profit (${profit:.2f}) below minimum threshold (${self.min_profit_threshold:.2f})",
                "profit": profit,
                "reinvest_amount": 0.0,
                "withdraw_amount": 0.0
            }
        
        # Calculate reinvestment and withdrawal amounts
        reinvest_amount = profit * self.reinvestment_rate
        withdraw_amount = profit - reinvest_amount
        
        return {
            "can_compound": True,
            "profit": profit,
            "reinvest_amount": reinvest_amount,
            "withdraw_amount": withdraw_amount,
            "reinvestment_rate": self.reinvestment_rate,
            "new_capital": self.initial_capital + reinvest_amount
        }
    
    def execute_compounding(self, current_capital: float, 
                           current_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Execute compounding strategy.
        
        Args:
            current_capital: Current trading capital
            current_date: Current date (defaults to now)
            
        Returns:
            Compounding execution results
        """
        if not current_date:
            current_date = datetime.now()
        
        # Check if compounding should be performed
        if not self.should_compound(current_date):
            return {
                "compounded": False,
                "reason": "Compounding not scheduled for this period",
                "next_compounding": self._get_next_compounding_date()
            }
        
        # Calculate compounding amounts
        calc = self.calculate_compounding(current_capital, current_date)
        
        if not calc["can_compound"]:
            return {
                "compounded": False,
                "reason": calc["reason"],
                "next_compounding": self._get_next_compounding_date()
            }
        
        # Execute compounding
        self.reinvested_profit += calc["reinvest_amount"]
        self.withdrawn_profit += calc["withdraw_amount"]
        self.initial_capital += calc["reinvest_amount"]
        self.last_compounding_date = current_date
        
        # Record in history
        self.history.append({
            "date": current_date.isoformat(),
            "capital_before": current_capital,
            "profit": calc["profit"],
            "reinvested": calc["reinvest_amount"],
            "withdrawn": calc["withdraw_amount"],
            "capital_after": calc["new_capital"]
        })
        
        # Save history
        self._save_history()
        
        logger.info(f"Executed compounding: ${calc['reinvest_amount']:.2f} reinvested, ${calc['withdraw_amount']:.2f} withdrawn")
        
        return {
            "compounded": True,
            "date": current_date.isoformat(),
            "capital_before": current_capital,
            "profit": calc["profit"],
            "reinvested": calc["reinvest_amount"],
            "withdrawn": calc["withdraw_amount"],
            "capital_after": calc["new_capital"],
            "next_compounding": self._get_next_compounding_date()
        }
    
    def _get_next_compounding_date(self) -> str:
        """
        Get next scheduled compounding date.
        
        Returns:
            Next compounding date as ISO format string
        """
        if not self.last_compounding_date:
            return datetime.now().isoformat()
        
        next_date = self.last_compounding_date
        
        if self.frequency == "daily":
            next_date += timedelta(days=1)
        elif self.frequency == "weekly":
            next_date += timedelta(days=7)
        else:  # monthly
            # Move to next month
            if next_date.month == 12:
                next_date = next_date.replace(year=next_date.year + 1, month=1)
            else:
                next_date = next_date.replace(month=next_date.month + 1)
        
        return next_date.isoformat()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get compounding statistics.
        
        Returns:
            Compounding statistics
        """
        return {
            "enabled": self.enabled,
            "reinvestment_rate": self.reinvestment_rate,
            "frequency": self.frequency,
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "total_profit": self.total_profit,
            "reinvested_profit": self.reinvested_profit,
            "withdrawn_profit": self.withdrawn_profit,
            "last_compounding_date": self.last_compounding_date.isoformat() if self.last_compounding_date else None,
            "next_compounding_date": self._get_next_compounding_date(),
            "compounding_events": len(self.history)
        }
    
    def set_reinvestment_rate(self, rate: float) -> None:
        """
        Set reinvestment rate.
        
        Args:
            rate: Reinvestment rate (0.0 to 1.0)
        """
        if rate < 0.0 or rate > 1.0:
            logger.warning(f"Invalid reinvestment rate: {rate}, must be between 0.0 and 1.0")
            return
        
        self.reinvestment_rate = rate
        logger.info(f"Set reinvestment rate to {rate*100:.0f}%")
    
    def enable(self) -> None:
        """Enable compounding strategy."""
        self.enabled = True
        logger.info("Compounding strategy enabled")
    
    def disable(self) -> None:
        """Disable compounding strategy."""
        self.enabled = False
        logger.info("Compounding strategy disabled")
    
    def set_frequency(self, frequency: str) -> None:
        """
        Set compounding frequency.
        
        Args:
            frequency: Compounding frequency ("daily", "weekly", or "monthly")
        """
        if frequency not in ["daily", "weekly", "monthly"]:
            logger.warning(f"Invalid frequency: {frequency}, must be 'daily', 'weekly', or 'monthly'")
            return
        
        self.frequency = frequency
        logger.info(f"Set compounding frequency to {frequency}")
