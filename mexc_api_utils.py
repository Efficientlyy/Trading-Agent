#!/usr/bin/env python
"""
MEXC API Utilities for Flash Trading System

This module provides utility functions for interacting with the MEXC API,
including proper signature generation and authentication handling.
"""

import hmac
import hashlib
import requests
import time
from urllib.parse import urlencode
import os
from dotenv import load_dotenv

# Load environment variables from .env file
def load_credentials(env_path=None):
    """Load API credentials from environment variables or .env file"""
    if env_path:
        load_dotenv(env_path)
    
    api_key = os.getenv('MEXC_API_KEY')
    secret_key = os.getenv('MEXC_API_SECRET')  # Updated to match .env file
    
    if not api_key or not secret_key:
        raise ValueError("MEXC API credentials not found in environment variables")
    
    return api_key, secret_key

class MexcApiClient:
    """MEXC API Client with proper authentication handling"""
    
    def __init__(self, api_key=None, secret_key=None, env_path=None):
        """Initialize the MEXC API client with credentials"""
        if api_key and secret_key:
            self.api_key = api_key
            self.secret_key = secret_key
        else:
            self.api_key, self.secret_key = load_credentials(env_path)
        
        self.base_url = "https://api.mexc.com"
        self.api_v3 = "/api/v3"
    
    def get_server_time(self):
        """Get server time from MEXC API"""
        url = f"{self.base_url}{self.api_v3}/time"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()['serverTime']
        else:
            raise ConnectionError(f"Failed to get server time: {response.text}")
    
    def generate_signature(self, params):
        """Generate HMAC SHA256 signature for API request
        
        Follows the exact MEXC API signature requirements
        """
        # Convert params to query string
        query_string = urlencode(params)
        
        # Create HMAC SHA256 signature
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def public_request(self, method, endpoint, params=None):
        """Make a public API request (no authentication required)"""
        url = f"{self.base_url}{endpoint}"
        response = requests.request(method, url, params=params)
        return response
    
    def signed_request(self, method, endpoint, params=None):
        """Make a signed API request (authentication required)"""
        url = f"{self.base_url}{endpoint}"
        
        # Prepare parameters
        request_params = params.copy() if params else {}
        
        # Add timestamp
        request_params['timestamp'] = self.get_server_time()
        
        # Generate signature
        signature = self.generate_signature(request_params)
        request_params['signature'] = signature
        
        # Set headers
        headers = {
            'X-MEXC-APIKEY': self.api_key,
        }
        
        # Make request
        if method == 'GET':
            response = requests.get(url, params=request_params, headers=headers)
        elif method == 'POST':
            response = requests.post(url, params=request_params, headers=headers)
        elif method == 'DELETE':
            response = requests.delete(url, params=request_params, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
            
        return response
    
    def test_connectivity(self):
        """Test API connectivity and authentication"""
        # Test public endpoint
        ping_response = self.public_request('GET', f"{self.api_v3}/ping")
        if ping_response.status_code != 200:
            return False, f"Public API test failed: {ping_response.text}"
        
        # Test authenticated endpoint with minimal parameters
        account_response = self.signed_request('GET', f"{self.api_v3}/account")
        if account_response.status_code != 200:
            return False, f"Authentication failed: {account_response.text} (URL: {account_response.url})"
        
        return True, "API connectivity and authentication successful"

# Example usage
if __name__ == "__main__":
    # Load from environment or .env file
    client = MexcApiClient(env_path=".env-secure/.env")
    
    # Test connectivity
    success, message = client.test_connectivity()
    print(message)
    
    if success:
        # Get account information
        account_info = client.signed_request('GET', "/api/v3/account")
        print(f"Account info: {account_info.json()}")
    else:
        # Print debug information
        print("Debug information:")
        # Test with minimal parameters
        timestamp = client.get_server_time()
        params = {'timestamp': timestamp}
        signature = client.generate_signature(params)
        print(f"Timestamp: {timestamp}")
        print(f"Signature: {signature}")
        print(f"API Key: {client.api_key[:5]}...{client.api_key[-5:]}")
        print(f"Secret Key: {client.secret_key[:5]}...{client.secret_key[-5:]}")
