#!/usr/bin/env python
"""
Environment variable loader for Trading-Agent System

This module loads environment variables from a .env file and provides
them to the rest of the system.
"""

import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("env_loader")

def load_environment_variables(env_path=None):
    """Load environment variables from .env file
    
    Args:
        env_path: Path to .env file (optional)
        
    Returns:
        dict: Dictionary of environment variables
    """
    # Default to .env in current directory if not specified
    if env_path is None:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    
    # Load environment variables from .env file
    load_dotenv(env_path)
    
    # Get environment variables
    env_vars = {
        'MEXC_API_KEY': os.getenv('MEXC_API_KEY'),
        'MEXC_SECRET_KEY': os.getenv('MEXC_SECRET_KEY')
    }
    
    # Check if API keys are available
    if env_vars['MEXC_API_KEY'] and env_vars['MEXC_SECRET_KEY']:
        logger.info("MEXC API credentials loaded successfully")
    else:
        logger.warning("MEXC API credentials not found in environment variables")
    
    return env_vars

if __name__ == "__main__":
    # Test loading environment variables
    env_vars = load_environment_variables()
    
    # Print masked credentials for verification
    if env_vars['MEXC_API_KEY']:
        masked_key = env_vars['MEXC_API_KEY'][:4] + "..." + env_vars['MEXC_API_KEY'][-4:]
        logger.info(f"MEXC API Key: {masked_key}")
    
    if env_vars['MEXC_SECRET_KEY']:
        masked_secret = "..." + env_vars['MEXC_SECRET_KEY'][-4:]
        logger.info(f"MEXC Secret Key: {masked_secret}")
