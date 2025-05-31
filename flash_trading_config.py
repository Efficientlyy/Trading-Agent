#!/usr/bin/env python
"""
Flash Trading Configuration

This module provides configuration settings for the flash trading system,
focusing on zero-fee trading pairs (BTCUSDC, ETHUSDC) and paper trading.
"""

import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("flash_trading_config")

# Default configuration
DEFAULT_CONFIG = {
    # Trading pairs configuration
    "trading_pairs": [
        {
            "symbol": "BTCUSDC",
            "base_asset": "BTC",
            "quote_asset": "USDC",
            "min_order_size": 0.001,
            "price_precision": 2,
            "quantity_precision": 6,
            "max_position": 0.1,
            "enabled": True,
            "description": "Bitcoin/USDC - Zero fee trading pair"
        },
        {
            "symbol": "ETHUSDC",
            "base_asset": "ETH",
            "quote_asset": "USDC",
            "min_order_size": 0.01,
            "price_precision": 2,
            "quantity_precision": 5,
            "max_position": 1.0,
            "enabled": True,
            "description": "Ethereum/USDC - Zero fee trading pair"
        }
    ],
    
    # Paper trading configuration
    "paper_trading": {
        "enabled": True,
        "initial_balance": {
            "USDC": 10000.0,
            "BTC": 0.0,
            "ETH": 0.0
        },
        "simulate_slippage": True,
        "slippage_bps": 2.0,  # 0.02% slippage
        "simulate_partial_fills": True,
        "partial_fill_probability": 0.2,
        "log_trades": True,
        "persist_state": True,
        "state_file": "paper_trading_state.json"
    },
    
    # Signal generation configuration
    "signal_generation": {
        "imbalance_threshold": 0.2,      # Order book imbalance threshold
        "volatility_threshold": 0.1,      # Price volatility threshold
        "momentum_threshold": 0.05,       # Price momentum threshold
        "min_spread_bps": 1.0,            # Minimum spread in basis points
        "max_spread_bps": 50.0,           # Maximum spread in basis points
        "order_book_depth": 10,           # Order book depth to monitor
        "update_interval_ms": 100,        # Market state update interval
        "signal_interval_ms": 50,         # Signal generation interval
        "use_cached_data": True,          # Use cached market data
        "cache_max_age_ms": 200           # Maximum age of cached data
    },
    
    # Execution configuration
    "execution": {
        "order_type": "LIMIT",            # Order type (LIMIT, MARKET)
        "time_in_force": "IOC",           # Time in force (IOC, GTC, FOK)
        "take_profit_bps": 20.0,          # Take profit in basis points
        "stop_loss_bps": 10.0,            # Stop loss in basis points
        "max_open_orders": 5,             # Maximum number of open orders
        "max_daily_orders": 1000,         # Maximum daily orders
        "retry_failed_orders": True,      # Retry failed orders
        "max_retries": 3,                 # Maximum retry attempts
        "retry_delay_ms": 500             # Delay between retries
    },
    
    # System configuration
    "system": {
        "log_level": "INFO",
        "log_to_file": True,
        "log_file": "flash_trading.log",
        "metrics_enabled": True,
        "metrics_interval_ms": 5000,
        "persist_metrics": True,
        "metrics_file": "flash_trading_metrics.json"
    }
}

class FlashTradingConfig:
    """Configuration manager for flash trading system"""
    
    def __init__(self, config_file=None):
        """Initialize configuration with optional config file"""
        self.config = DEFAULT_CONFIG.copy()
        self.config_file = config_file
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file):
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
            
            # Update configuration with loaded values
            self._update_nested_dict(self.config, loaded_config)
            logger.info(f"Configuration loaded from {config_file}")
            return True
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return False
    
    def save_config(self, config_file=None):
        """Save configuration to file"""
        save_file = config_file or self.config_file
        if not save_file:
            logger.warning("No config file specified for saving")
            return False
        
        try:
            with open(save_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to {save_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def _update_nested_dict(self, d, u):
        """Update nested dictionary with another dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def get_enabled_trading_pairs(self):
        """Get list of enabled trading pairs"""
        return [pair for pair in self.config["trading_pairs"] if pair["enabled"]]
    
    def get_trading_pair_config(self, symbol):
        """Get configuration for a specific trading pair"""
        for pair in self.config["trading_pairs"]:
            if pair["symbol"] == symbol:
                return pair
        return None
    
    def is_paper_trading_enabled(self):
        """Check if paper trading is enabled"""
        return self.config["paper_trading"]["enabled"]
    
    def get_paper_trading_balance(self, asset):
        """Get paper trading balance for an asset"""
        return self.config["paper_trading"]["initial_balance"].get(asset, 0.0)
    
    def get_signal_config(self):
        """Get signal generation configuration"""
        return self.config["signal_generation"]
    
    def get_execution_config(self):
        """Get execution configuration"""
        return self.config["execution"]
    
    def get_system_config(self):
        """Get system configuration"""
        return self.config["system"]
    
    def validate(self):
        """Validate configuration"""
        # Check if at least one trading pair is enabled
        if not self.get_enabled_trading_pairs():
            logger.warning("No trading pairs are enabled")
            return False
        
        # Validate paper trading configuration
        if self.is_paper_trading_enabled():
            if not self.config["paper_trading"]["initial_balance"]:
                logger.warning("Paper trading enabled but no initial balance specified")
                return False
        
        return True

# Example usage
if __name__ == "__main__":
    # Create configuration
    config = FlashTradingConfig()
    
    # Print configuration
    print("Flash Trading Configuration:")
    print(f"Enabled Trading Pairs: {[pair['symbol'] for pair in config.get_enabled_trading_pairs()]}")
    print(f"Paper Trading Enabled: {config.is_paper_trading_enabled()}")
    print(f"Paper Trading Balance: {config.config['paper_trading']['initial_balance']}")
    
    # Save configuration
    config.save_config("flash_trading_config.json")
