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
        # Add timeout to prevent hanging
        response = requests.get(f"{BASE_URL}{SERVER_TIME_ENDPOINT}", timeout=5.0)
        response.raise_for_status()
        
        # Parse and validate response
        try:
            data = response.json()
            
            # Validate response structure
            if not isinstance(data, dict):
                logger.error(f"Invalid server time response type: {type(data)}")
                return {"serverTime": int(time.time() * 1000)}
                
            if "serverTime" not in data:
                logger.error("Missing serverTime in response")
                return {"serverTime": int(time.time() * 1000)}
                
            # Validate server time is a reasonable value
            server_time = data.get("serverTime")
            if not isinstance(server_time, (int, float)):
                logger.error(f"Invalid serverTime format: {server_time}")
                return {"serverTime": int(time.time() * 1000)}
                
            # Check if server time is within 24 hours of current time
            current_time = int(time.time() * 1000)
            if abs(server_time - current_time) > 86400000:  # 24 hours in milliseconds
                logger.error(f"Server time appears invalid: {server_time}")
                return {"serverTime": current_time}
                
            return data
        except ValueError as e:
            logger.error(f"Error parsing server time response: {e}")
            return {"serverTime": int(time.time() * 1000)}
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get server time: {e}")
        # Return dict with current time instead of empty dict
        return {"serverTime": int(time.time() * 1000)}

def validate_credentials(api_key, api_secret):
    """Validate MEXC API credentials by making an authenticated request."""
    try:
        # Validate input parameters
        if not api_key or not isinstance(api_key, str) or len(api_key) < 10:
            logger.error(f"Invalid API key format: {api_key}")
            return False
            
        if not api_secret or not isinstance(api_secret, str) or len(api_secret) < 10:
            logger.error("Invalid API secret format")
            return False
        
        # Get server time first to ensure timestamp is synchronized
        server_time = get_server_time()
        if not server_time or not isinstance(server_time, dict) or 'serverTime' not in server_time:
            logger.error("Failed to get server time. Cannot validate credentials.")
            return False
        
        # Validate server time value
        timestamp = server_time.get('serverTime')
        if not isinstance(timestamp, (int, float)):
            logger.error(f"Invalid server time format: {timestamp}")
            return False
        
        # Prepare parameters for authenticated request with validation
        try:
            params = {
                'timestamp': int(timestamp),
                'recvWindow': 5000
            }
        except (ValueError, TypeError) as e:
            logger.error(f"Error preparing request parameters: {e}")
            return False
        
        # Generate signature with validation
        try:
            signature = generate_signature(api_secret, params)
            if not signature or not isinstance(signature, str) or len(signature) != 64:
                logger.error(f"Invalid signature generated: {signature}")
                return False
                
            params['signature'] = signature
        except Exception as e:
            logger.error(f"Error generating signature: {e}")
            return False
        
        # Set headers with validation
        headers = {
            'X-MEXC-APIKEY': api_key
        }
        
        try:
            # Make authenticated request with timeout
            url = f"{BASE_URL}{ACCOUNT_INFO_ENDPOINT}"
            logger.info(f"Making authenticated request to {url}")
            logger.info(f"Headers: {headers}")
            logger.info(f"Params: {params}")
            
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=10.0
            )
            
            # Parse response to dict with validation
            try:
                account_info = response.json()
                
                # Validate response structure
                if not isinstance(account_info, dict):
                    logger.error(f"Invalid account info response type: {type(account_info)}")
                    return False
            except ValueError as e:
                logger.error(f"Error parsing account info response: {e}")
                account_info = {}
            
            # Check response with robust validation
            if response.status_code == 200 and account_info:
                # Validate required fields exist
                required_fields = ["accountType", "canTrade", "canDeposit", "canWithdraw", "balances"]
                missing_fields = [field for field in required_fields if field not in account_info]
                
                if missing_fields:
                    logger.error(f"Missing required fields in account response: {missing_fields}")
                    return False
                
                # Validate balances field
                balances = account_info.get("balances")
                if not isinstance(balances, list):
                    logger.error(f"Invalid balances type: {type(balances)}")
                    return False
                
                logger.info("MEXC API credentials are valid!")
                logger.info(f"Account type: {account_info.get('accountType', 'Unknown')}")
                logger.info(f"Can trade: {account_info.get('canTrade', False)}")
                logger.info(f"Can deposit: {account_info.get('canDeposit', False)}")
                logger.info(f"Can withdraw: {account_info.get('canWithdraw', False)}")
                return True
            else:
                error_msg = response.text if hasattr(response, 'text') else "No error message"
                logger.error(f"Failed to validate credentials: {response.status_code} - {error_msg}")
                return False
        except requests.exceptions.Timeout:
            logger.error("Request timed out while validating credentials")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return False
    except Exception as e:
        logger.error(f"Unexpected error during credential validation: {e}")
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
