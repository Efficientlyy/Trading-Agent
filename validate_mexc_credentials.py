#!/usr/bin/env python
"""
MEXC API Credential Validator

This script validates MEXC API credentials by making a simple authenticated request.
It uses the environment variables loaded from the .env file.

Usage:
    python validate_mexc_credentials.py
"""

import sys
import json
import time
import hmac
import hashlib
import logging
import requests
from urllib.parse import urlencode
from env_loader import load_env, get_required_env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MEXC API endpoints
BASE_URL = "https://api.mexc.com"
SERVER_TIME_ENDPOINT = "/api/v3/time"
ACCOUNT_INFO_ENDPOINT = "/api/v3/account"

def generate_signature(api_secret, params):
    """Generate HMAC SHA256 signature for API authentication."""
    # Convert params to sorted query string using urlencode
    query_string = urlencode(sorted(params.items()))
    
    # Create signature
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return signature

def get_server_time():
    """Get MEXC server time."""
    try:
        response = requests.get(f"{BASE_URL}{SERVER_TIME_ENDPOINT}")
        response.raise_for_status()
        # Return the parsed JSON directly
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get server time: {e}")
        # Return empty dict instead of None for consistency
        return {}

def validate_credentials(api_key, api_secret):
    """Validate MEXC API credentials by making an authenticated request."""
    # Get server time first to ensure timestamp is synchronized
    server_time = get_server_time()
    if not server_time or 'serverTime' not in server_time:
        logger.error("Failed to get server time. Cannot validate credentials.")
        return False
    
    # Prepare parameters for authenticated request
    params = {
        'timestamp': server_time['serverTime'],
        'recvWindow': 5000
    }
    
    # Generate signature
    signature = generate_signature(api_secret, params)
    params['signature'] = signature
    
    # Set headers
    headers = {
        'X-MEXC-APIKEY': api_key
    }
    
    try:
        # Make authenticated request
        url = f"{BASE_URL}{ACCOUNT_INFO_ENDPOINT}"
        logger.info(f"Making authenticated request to {url}")
        logger.info(f"Headers: {headers}")
        logger.info(f"Params: {params}")
        
        response = requests.get(
            url,
            params=params,
            headers=headers
        )
        
        # Parse response to dict
        try:
            account_info = response.json()
        except ValueError:
            account_info = {}
        
        # Check response
        if response.status_code == 200 and account_info:
            logger.info("MEXC API credentials are valid!")
            logger.info(f"Account type: {account_info.get('accountType', 'Unknown')}")
            logger.info(f"Can trade: {account_info.get('canTrade', False)}")
            logger.info(f"Can deposit: {account_info.get('canDeposit', False)}")
            logger.info(f"Can withdraw: {account_info.get('canWithdraw', False)}")
            return True
        else:
            logger.error(f"Failed to validate credentials: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return False

def main():
    """Main function."""
    # Load environment variables
    if not load_env():
        logger.error("Failed to load environment variables.")
        sys.exit(1)
    
    try:
        # Get API credentials
        api_key = get_required_env('MEXC_API_KEY')
        api_secret = get_required_env('MEXC_API_SECRET')
        
        # Validate credentials
        if validate_credentials(api_key, api_secret):
            logger.info("Credential validation successful!")
            sys.exit(0)
        else:
            logger.error("Credential validation failed!")
            sys.exit(1)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
