#!/usr/bin/env python
"""
Configuration management for LLM Strategic Overseer.
Handles loading settings from files, environment variables, and command-line arguments.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for LLM Strategic Overseer."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default path.
        """
        self.config_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.project_root = self.config_dir.parent
        
        # Load environment variables
        load_dotenv(os.path.join(self.project_root.parent, '.env-secure', '.env'))
        
        # Load default configuration
        self.config = self._load_default_config()
        
        # Override with custom configuration if provided
        if config_path:
            self._load_config_from_file(config_path)
        
        # Override with environment variables
        self._load_config_from_env()
        
        # Validate configuration
        self._validate_config()
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from settings.json."""
        default_config_path = os.path.join(self.config_dir, 'settings.json')
        
        if not os.path.exists(default_config_path):
            logger.warning(f"Default config file not found at {default_config_path}. Creating with default values.")
            default_config = self._create_default_config()
            with open(default_config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
        
        try:
            with open(default_config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading default config: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        return {
            "llm": {
                "provider": "openrouter",
                "api_key_env": "OPENROUTER_API_KEY",
                "tiers": {
                    "tier1": {
                        "models": ["llama3-8b", "phi-3-mini", "claude-haiku"],
                        "default_model": "llama3-8b",
                        "max_tokens": 4096,
                        "temperature": 0.7
                    },
                    "tier2": {
                        "models": ["llama3-70b", "claude-sonnet", "mistral-large"],
                        "default_model": "claude-sonnet",
                        "max_tokens": 8192,
                        "temperature": 0.7
                    },
                    "tier3": {
                        "models": ["claude-opus", "gpt-4o"],
                        "default_model": "claude-opus",
                        "max_tokens": 16384,
                        "temperature": 0.7
                    }
                },
                "cache_enabled": True,
                "cache_ttl": 3600,  # 1 hour
                "token_budget": {
                    "daily_limit": 1000000,  # 1M tokens per day
                    "alert_threshold": 0.8   # Alert at 80% of daily limit
                }
            },
            "telegram": {
                "bot_token_env": "TELEGRAM_BOT_TOKEN",
                "allowed_user_ids": [],  # Will be populated from env
                "session_timeout": 3600,  # 1 hour
                "notification_levels": {
                    "1": ["emergency", "critical"],
                    "2": ["emergency", "critical", "trade"],
                    "3": ["emergency", "critical", "trade", "info"]
                },
                "default_notification_level": 2
            },
            "trading": {
                "compounding": {
                    "enabled": True,
                    "reinvestment_rate": 0.8,  # 80% reinvestment
                    "min_profit_threshold": 100,  # Minimum profit before compounding
                    "frequency": "monthly"  # monthly, weekly, or daily
                },
                "risk_management": {
                    "max_position_size": 0.1,  # 10% of account
                    "max_daily_drawdown": 0.05,  # 5% max daily drawdown
                    "stop_loss": 0.02  # 2% stop loss
                }
            },
            "system": {
                "log_level": "INFO",
                "data_retention_days": 90,
                "heartbeat_interval": 60  # seconds
            }
        }
    
    def _load_config_from_file(self, config_path: str) -> None:
        """Load configuration from file and merge with current config."""
        try:
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                self._deep_update(self.config, custom_config)
        except Exception as e:
            logger.error(f"Error loading custom config from {config_path}: {e}")
    
    def _load_config_from_env(self) -> None:
        """Override configuration with environment variables."""
        # Load API keys
        if api_key := os.getenv(self.config["llm"]["api_key_env"]):
            self.config["llm"]["api_key"] = api_key
        
        if bot_token := os.getenv(self.config["telegram"]["bot_token_env"]):
            self.config["telegram"]["bot_token"] = bot_token
        
        # Load allowed user IDs
        if allowed_users := os.getenv("TELEGRAM_ALLOWED_USERS"):
            try:
                self.config["telegram"]["allowed_user_ids"] = [
                    int(user_id.strip()) for user_id in allowed_users.split(",")
                ]
            except Exception as e:
                logger.error(f"Error parsing TELEGRAM_ALLOWED_USERS: {e}")
        
        # Other environment variables can be added here
    
    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively update nested dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
        return d
    
    def _validate_config(self) -> None:
        """Validate configuration and log warnings for missing required values."""
        required_keys = [
            ("llm", "api_key"),
            ("telegram", "bot_token"),
            ("telegram", "allowed_user_ids")
        ]
        
        for key_path in required_keys:
            if not self._get_nested_value(self.config, key_path):
                logger.warning(f"Missing required configuration: {'.'.join(key_path)}")
    
    def _get_nested_value(self, d: Dict[str, Any], key_path: tuple) -> Any:
        """Get value from nested dictionary using key path."""
        current = d
        for key in key_path:
            if key not in current:
                return None
            current = current[key]
        return current
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Key path in dot notation (e.g., "llm.tiers.tier1.default_model")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split(".")
        return self._get_nested_value(self.config, tuple(keys)) or default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Key path in dot notation (e.g., "llm.tiers.tier1.default_model")
            value: Value to set
        """
        keys = key_path.split(".")
        current = self.config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def save(self, config_path: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration. If None, uses default path.
        """
        if not config_path:
            config_path = os.path.join(self.config_dir, 'settings.json')
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
