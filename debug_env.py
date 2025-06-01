#!/usr/bin/env python
"""
Debug script to verify API key loading from environment file
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("debug_env")

def debug_env_loading(env_path):
    """Debug environment variable loading"""
    logger.info(f"Attempting to load environment from: {env_path}")
    
    # Check if file exists
    if not os.path.exists(env_path):
        logger.error(f"Environment file not found: {env_path}")
        return False
    
    # Check file permissions
    try:
        with open(env_path, 'r') as f:
            content = f.read()
            logger.info(f"Successfully read environment file. Content length: {len(content)}")
    except Exception as e:
        logger.error(f"Error reading environment file: {str(e)}")
        return False
    
    # Try loading with dotenv
    try:
        load_dotenv(env_path)
        logger.info("Successfully called load_dotenv")
    except Exception as e:
        logger.error(f"Error loading environment with dotenv: {str(e)}")
        return False
    
    # Check if variables were loaded
    api_key = os.environ.get('MEXC_API_KEY')
    api_secret = os.environ.get('MEXC_API_SECRET')
    
    logger.info(f"MEXC_API_KEY present: {api_key is not None}")
    if api_key:
        logger.info(f"MEXC_API_KEY type: {type(api_key)}")
        logger.info(f"MEXC_API_KEY length: {len(api_key)}")
        logger.info(f"MEXC_API_KEY first 4 chars: {api_key[:4]}")
    
    logger.info(f"MEXC_API_SECRET present: {api_secret is not None}")
    if api_secret:
        logger.info(f"MEXC_API_SECRET type: {type(api_secret)}")
        logger.info(f"MEXC_API_SECRET length: {len(api_secret)}")
    
    return api_key is not None and api_secret is not None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python debug_env.py <env_file_path>")
        sys.exit(1)
    
    env_path = sys.argv[1]
    success = debug_env_loading(env_path)
    
    if success:
        logger.info("Environment variables loaded successfully")
        sys.exit(0)
    else:
        logger.error("Failed to load environment variables")
        sys.exit(1)
