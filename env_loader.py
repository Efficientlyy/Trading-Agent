#!/usr/bin/env python
"""
Environment Variable Loader for MEXC Trading System

This script loads environment variables from a .env file and makes them available
to all components of the MEXC Trading System. It provides a consistent interface
for accessing credentials and configuration settings.

Usage:
    from env_loader import load_env, get_env

    # Load environment variables
    load_env()

    # Access environment variables
    api_key = get_env('MEXC_API_KEY')
    api_secret = get_env('MEXC_API_SECRET')
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_env_file():
    """
    Find the .env file in the following locations:
    1. .env-secure/.env (preferred secure location)
    2. .env (fallback)
    """
    # Get the project root directory (where this script is located)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent
    
    # Check for .env file in secure directory
    secure_env_path = project_root / '.env-secure' / '.env'
    if secure_env_path.exists():
        return secure_env_path
    
    # Check for .env file in project root
    default_env_path = project_root / '.env'
    if default_env_path.exists():
        return default_env_path
    
    return None

def load_env():
    """
    Load environment variables from .env file
    """
    env_path = find_env_file()
    
    if env_path:
        logger.info(f"Loading environment variables from {env_path}")
        load_dotenv(dotenv_path=env_path)
        return True
    else:
        logger.warning("No .env file found. Using system environment variables.")
        return False

def get_env(key, default=None):
    """
    Get environment variable value
    
    Args:
        key (str): Environment variable name
        default: Default value if environment variable is not set
        
    Returns:
        str: Environment variable value or default
    """
    value = os.environ.get(key, default)
    
    if value is None:
        logger.warning(f"Environment variable {key} not set")
    
    return value

def get_required_env(key):
    """
    Get required environment variable value
    
    Args:
        key (str): Environment variable name
        
    Returns:
        str: Environment variable value
        
    Raises:
        ValueError: If environment variable is not set
    """
    value = os.environ.get(key)
    
    if value is None:
        logger.error(f"Required environment variable {key} not set")
        raise ValueError(f"Required environment variable {key} not set")
    
    return value

def is_paper_trading():
    """
    Check if paper trading mode is enabled
    
    Returns:
        bool: True if paper trading mode is enabled, False otherwise
    """
    mode = get_env('TRADING_MODE', 'paper').lower()
    return mode == 'paper'

def get_trading_pair():
    """
    Get default trading pair
    
    Returns:
        str: Default trading pair
    """
    return get_env('DEFAULT_TRADING_PAIR', 'BTCUSDC')

# Auto-load environment variables when module is imported
if __name__ != "__main__":
    load_env()
