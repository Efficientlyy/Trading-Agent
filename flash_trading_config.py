#!/usr/bin/env python
"""
Flash Trading Configuration with Session-Specific Parameters

This module provides configuration management for the flash trading system,
including session-specific parameters for different global trading sessions.
"""

import os
import json
import logging
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("flash_trading_config")

class FlashTradingConfig:
    """Configuration manager for flash trading system with session awareness"""
    
    def __init__(self, config_path=None):
        """Initialize configuration with optional path to config file"""
        # Default configuration
        self.config = self._get_default_config()
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        else:
            # Save default configuration
            self.config_path = config_path or "flash_trading_config.json"
            self._save_config()
            
    def get_signal_generation_config(self):
        """Get signal generation configuration section
        
        Returns:
            dict: Signal generation configuration
        """
        return self.config.get("signal_generation", {})
    
    def _get_default_config(self):
        """Get default configuration with session-specific parameters"""
        return {
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
            
            # Session configuration
            "session_management": {
                "enabled": True,
                "config_file": "trading_session_config.json",
                "auto_update_interval": 300,  # Update session every 5 minutes
                "sessions": {
                    "ASIA": {
                        "start_hour_utc": 0,
                        "end_hour_utc": 8,
                        "description": "Asian Trading Session (00:00-08:00 UTC)"
                    },
                    "EUROPE": {
                        "start_hour_utc": 8,
                        "end_hour_utc": 16,
                        "description": "European Trading Session (08:00-16:00 UTC)"
                    },
                    "US": {
                        "start_hour_utc": 13,
                        "end_hour_utc": 21,
                        "description": "US Trading Session (13:00-21:00 UTC)"
                    }
                }
            },
            
            # Session-specific signal generation parameters
            "session_parameters": {
                "ASIA": {
                    "imbalance_threshold": 0.15,      # Lower threshold for typically lower volume
                    "volatility_threshold": 0.12,     # Higher threshold for typically higher volatility
                    "momentum_threshold": 0.04,       # Lower threshold for momentum
                    "min_spread_bps": 2.0,            # Higher min spread due to lower liquidity
                    "max_spread_bps": 60.0,           # Higher max spread due to lower liquidity
                    "order_book_depth": 10,           # Standard order book depth
                    "signal_interval_ms": 50,         # Standard signal interval
                    "position_size_factor": 0.8,      # Smaller positions due to higher volatility
                    "take_profit_bps": 25.0,          # Higher take profit due to higher volatility
                    "stop_loss_bps": 15.0,            # Higher stop loss due to higher volatility
                    "max_open_orders": 3,             # Fewer open orders due to higher volatility
                    "time_in_force": "IOC"            # Immediate-or-Cancel for higher volatility
                },
                "EUROPE": {
                    "imbalance_threshold": 0.2,       # Standard threshold
                    "volatility_threshold": 0.1,      # Standard threshold
                    "momentum_threshold": 0.05,       # Standard threshold
                    "min_spread_bps": 1.0,            # Standard min spread
                    "max_spread_bps": 50.0,           # Standard max spread
                    "order_book_depth": 10,           # Standard order book depth
                    "signal_interval_ms": 50,         # Standard signal interval
                    "position_size_factor": 1.0,      # Standard position size
                    "take_profit_bps": 20.0,          # Standard take profit
                    "stop_loss_bps": 10.0,            # Standard stop loss
                    "max_open_orders": 5,             # Standard max open orders
                    "time_in_force": "IOC"            # Standard time in force
                },
                "US": {
                    "imbalance_threshold": 0.25,      # Higher threshold for higher liquidity
                    "volatility_threshold": 0.08,     # Lower threshold for typically lower volatility
                    "momentum_threshold": 0.06,       # Higher threshold for stronger trends
                    "min_spread_bps": 0.8,            # Lower min spread due to higher liquidity
                    "max_spread_bps": 40.0,           # Lower max spread due to higher liquidity
                    "order_book_depth": 15,           # Deeper order book due to higher liquidity
                    "signal_interval_ms": 30,         # Faster signal interval for higher liquidity
                    "position_size_factor": 1.2,      # Larger positions due to higher liquidity
                    "take_profit_bps": 15.0,          # Lower take profit due to lower volatility
                    "stop_loss_bps": 8.0,             # Lower stop loss due to lower volatility
                    "max_open_orders": 8,             # More open orders due to higher liquidity
                    "time_in_force": "IOC"            # Standard time in force
                }
            },
            
            # Default signal generation configuration (used as fallback)
            "signal_generation": {
                "imbalance_threshold": 0.2,      # Order book imbalance threshold
                "volatility_threshold": 0.1,     # Price volatility threshold
                "momentum_threshold": 0.05,      # Price momentum threshold
                "min_spread_bps": 1.0,           # Minimum spread in basis points
                "max_spread_bps": 50.0,          # Maximum spread in basis points
                "order_book_depth": 10,          # Order book depth to monitor
                "update_interval_ms": 100,       # Market state update interval
                "signal_interval_ms": 50,        # Signal generation interval
                "use_cached_data": True,         # Use cached market data
                "cache_max_age_ms": 200          # Maximum age of cached data
            },
            
            # Default execution configuration (used as fallback)
            "execution": {
                "order_type": "LIMIT",           # Order type (LIMIT, MARKET)
                "time_in_force": "IOC",          # Time in force (IOC, GTC, FOK)
                "take_profit_bps": 20.0,         # Take profit in basis points
                "stop_loss_bps": 10.0,           # Stop loss in basis points
                "max_open_orders": 5,            # Maximum number of open orders
                "max_daily_orders": 1000,        # Maximum daily orders
                "retry_failed_orders": True,     # Retry failed orders
                "max_retries": 3,                # Maximum retry attempts
                "retry_delay_ms": 500            # Delay between retries
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
    
    def _load_config(self, config_path):
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Update configuration with loaded values
            self._update_nested_dict(self.config, loaded_config)
            self.config_path = config_path
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    def _update_nested_dict(self, d, u):
        """Update nested dictionary with another dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def get_trading_pair_config(self, symbol):
        """Get configuration for a specific trading pair"""
        for pair in self.config["trading_pairs"]:
            if pair["symbol"] == symbol:
                return pair
        return None
    
    def get_enabled_trading_pairs(self):
        """Get list of enabled trading pairs"""
        return [pair for pair in self.config["trading_pairs"] if pair["enabled"]]
    
    def get_session_config(self):
        """Get session management configuration"""
        return self.config["session_management"]
    
    def get_session_parameters(self, session_name):
        """Get parameters for a specific trading session"""
        if session_name in self.config["session_parameters"]:
            return self.config["session_parameters"][session_name].copy()
        return self.config["signal_generation"].copy()  # Fallback to default
    
    def get_signal_config(self, session_name=None):
        """Get signal generation configuration, optionally for a specific session"""
        if session_name and session_name in self.config["session_parameters"]:
            # Start with default signal config
            config = self.config["signal_generation"].copy()
            # Override with session-specific parameters
            config.update(self.config["session_parameters"][session_name])
            return config
        return self.config["signal_generation"].copy()
    
    def get_execution_config(self, session_name=None):
        """Get execution configuration, optionally for a specific session"""
        if session_name and session_name in self.config["session_parameters"]:
            # Start with default execution config
            config = self.config["execution"].copy()
            # Override with session-specific parameters that apply to execution
            session_params = self.config["session_parameters"][session_name]
            for param in ["take_profit_bps", "stop_loss_bps", "max_open_orders", "time_in_force", "position_size_factor"]:
                if param in session_params:
                    config[param] = session_params[param]
            return config
        return self.config["execution"].copy()
    
    def update_config(self, updates):
        """Update configuration with new values"""
        self._update_nested_dict(self.config, updates)
        self._save_config()
    
    def update_session_parameters(self, session_name, parameters):
        """Update parameters for a specific trading session"""
        if session_name not in self.config["session_parameters"]:
            self.config["session_parameters"][session_name] = {}
        
        self._update_nested_dict(self.config["session_parameters"][session_name], parameters)
        self._save_config()
    
    def add_trading_pair(self, symbol, base_asset, quote_asset, min_order_size=0.001, 
                        price_precision=2, quantity_precision=6, max_position=0.1, 
                        enabled=True, description=None):
        """Add a new trading pair"""
        # Check if pair already exists
        for pair in self.config["trading_pairs"]:
            if pair["symbol"] == symbol:
                return False
        
        # Add new pair
        self.config["trading_pairs"].append({
            "symbol": symbol,
            "base_asset": base_asset,
            "quote_asset": quote_asset,
            "min_order_size": min_order_size,
            "price_precision": price_precision,
            "quantity_precision": quantity_precision,
            "max_position": max_position,
            "enabled": enabled,
            "description": description or f"{base_asset}/{quote_asset} trading pair"
        })
        
        self._save_config()
        return True
    
    def enable_trading_pair(self, symbol, enabled=True):
        """Enable or disable a trading pair"""
        for pair in self.config["trading_pairs"]:
            if pair["symbol"] == symbol:
                pair["enabled"] = enabled
                self._save_config()
                return True
        return False
    
    def add_session(self, name, start_hour_utc, end_hour_utc, description=None):
        """Add a new trading session"""
        if name in self.config["session_management"]["sessions"]:
            return False
        
        # Add session to configuration
        self.config["session_management"]["sessions"][name] = {
            "start_hour_utc": start_hour_utc,
            "end_hour_utc": end_hour_utc,
            "description": description or f"{name} Trading Session"
        }
        
        # Initialize session parameters by copying from most similar existing session
        if name not in self.config["session_parameters"]:
            if start_hour_utc >= 0 and start_hour_utc < 8:
                base_session = "ASIA"
            elif start_hour_utc >= 8 and start_hour_utc < 13:
                base_session = "EUROPE"
            else:
                base_session = "US"
            
            if base_session in self.config["session_parameters"]:
                self.config["session_parameters"][name] = self.config["session_parameters"][base_session].copy()
            else:
                self.config["session_parameters"][name] = self.config["signal_generation"].copy()
        
        self._save_config()
        return True
    
    def remove_session(self, name):
        """Remove a trading session"""
        if name in self.config["session_management"]["sessions"]:
            del self.config["session_management"]["sessions"][name]
            if name in self.config["session_parameters"]:
                del self.config["session_parameters"][name]
            self._save_config()
            return True
        return False


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Flash Trading Configuration')
    parser.add_argument('--config', default="flash_trading_config.json", help='Path to config file')
    parser.add_argument('--reset', action='store_true', help='Reset to default configuration')
    
    args = parser.parse_args()
    
    # Create configuration
    config = FlashTradingConfig(None if args.reset else args.config)
    
    # Print configuration summary
    print("Flash Trading Configuration:")
    print(f"Enabled Trading Pairs: {[pair['symbol'] for pair in config.get_enabled_trading_pairs()]}")
    print(f"Paper Trading Enabled: {config.config['paper_trading']['enabled']}")
    print(f"Paper Trading Balance: {config.config['paper_trading']['initial_balance']}")
    
    # Print session information
    print("\nTrading Sessions:")
    for name, session in config.config["session_management"]["sessions"].items():
        print(f"  {name}: {session['start_hour_utc']:02d}:00-{session['end_hour_utc']:02d}:00 UTC - {session['description']}")
    
    # Print session parameters
    print("\nSession Parameters:")
    for session_name, params in config.config["session_parameters"].items():
        print(f"  {session_name}:")
        for param_name, value in params.items():
            print(f"    {param_name}: {value}")
