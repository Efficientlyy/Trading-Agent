#!/usr/bin/env python
"""
Paper Trading Toggle Module

This module provides functionality to toggle between paper trading and real trading
modes based on environment configuration. It ensures consistent behavior across
all components of the Trading Agent system.
"""

import os
import logging
from typing import Dict, Any, Optional

# Import environment configuration
from env_config import load_env, is_paper_trading, is_production, print_env_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingModeManager:
    """
    Trading Mode Manager for handling paper vs real trading modes.
    
    This class provides a centralized way to manage trading modes across
    the entire Trading Agent system, ensuring consistent behavior.
    """
    
    def __init__(self):
        """Initialize Trading Mode Manager."""
        # Load environment variables
        load_env()
        
        # Set trading mode based on environment
        self._paper_trading = is_paper_trading()
        self._production = is_production()
        
        # Log current mode
        self._log_current_mode()
    
    def _log_current_mode(self):
        """Log current trading mode."""
        mode = "PAPER TRADING" if self._paper_trading else "REAL TRADING"
        env = "PRODUCTION" if self._production else "DEVELOPMENT"
        
        if self._production and not self._paper_trading:
            logger.warning(f"⚠️ RUNNING IN {mode} MODE IN {env} ENVIRONMENT ⚠️")
            logger.warning("Real funds will be used for trading!")
        else:
            logger.info(f"Running in {mode} mode in {env} environment")
    
    @property
    def is_paper_trading(self) -> bool:
        """
        Check if paper trading is enabled.
        
        Returns:
            True if paper trading is enabled, False otherwise
        """
        return self._paper_trading
    
    @property
    def is_real_trading(self) -> bool:
        """
        Check if real trading is enabled.
        
        Returns:
            True if real trading is enabled, False otherwise
        """
        return not self._paper_trading
    
    @property
    def is_production(self) -> bool:
        """
        Check if running in production environment.
        
        Returns:
            True if running in production, False otherwise
        """
        return self._production
    
    def get_trading_mode_info(self) -> Dict[str, Any]:
        """
        Get information about current trading mode.
        
        Returns:
            Dictionary with trading mode information
        """
        return {
            "paper_trading": self._paper_trading,
            "real_trading": not self._paper_trading,
            "production": self._production,
            "mode": "paper" if self._paper_trading else "real",
            "environment": "production" if self._production else "development"
        }
    
    def validate_real_trading_safety(self) -> Dict[str, Any]:
        """
        Validate safety checks for real trading.
        
        Returns:
            Dictionary with validation results
        """
        # Only perform checks if in real trading mode
        if self._paper_trading:
            return {"safe": True, "message": "Paper trading mode is active, no real funds at risk"}
        
        # Check for safety conditions
        checks = {
            "production_check": self._production,  # Should be in production for real trading
            "api_key_check": bool(os.getenv('MEXC_API_KEY')),
            "api_secret_check": bool(os.getenv('MEXC_API_SECRET')),
        }
        
        # Determine overall safety
        all_checks_passed = all(checks.values())
        
        # Generate message
        if all_checks_passed:
            message = "All safety checks passed for real trading"
        else:
            failed_checks = [k for k, v in checks.items() if not v]
            message = f"Safety checks failed: {', '.join(failed_checks)}"
        
        return {
            "safe": all_checks_passed,
            "checks": checks,
            "message": message
        }
    
    def print_mode_summary(self):
        """Print summary of current trading mode."""
        mode = "PAPER TRADING" if self._paper_trading else "REAL TRADING"
        env = "PRODUCTION" if self._production else "DEVELOPMENT"
        
        print("\n===================================")
        print(f"  TRADING MODE: {mode}")
        print(f"  ENVIRONMENT: {env}")
        print("===================================\n")
        
        if self._production and not self._paper_trading:
            print("⚠️  WARNING: REAL FUNDS WILL BE USED FOR TRADING  ⚠️\n")


# Singleton instance
trading_mode_manager = TradingModeManager()

# Convenience functions
def is_paper_trading() -> bool:
    """Check if paper trading is enabled."""
    return trading_mode_manager.is_paper_trading

def is_real_trading() -> bool:
    """Check if real trading is enabled."""
    return trading_mode_manager.is_real_trading

def get_trading_mode_info() -> Dict[str, Any]:
    """Get information about current trading mode."""
    return trading_mode_manager.get_trading_mode_info()

def validate_real_trading_safety() -> Dict[str, Any]:
    """Validate safety checks for real trading."""
    return trading_mode_manager.validate_real_trading_safety()


if __name__ == "__main__":
    # Test trading mode manager
    manager = TradingModeManager()
    manager.print_mode_summary()
    
    # Print environment summary
    print_env_summary()
    
    # Print safety validation
    safety = validate_real_trading_safety()
    print(f"Safety validation: {safety['message']}")
