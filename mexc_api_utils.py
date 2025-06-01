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
        try:
            url = f"{self.base_url}{self.api_v3}/time"
            
            # Add timeout to prevent hanging
            response = requests.get(url, timeout=5.0)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Validate response structure
                    if not isinstance(data, dict):
                        print(f"Invalid server time response type: {type(data)}")
                        return int(time.time() * 1000)
                        
                    server_time = data.get('serverTime')
                    if server_time is None:
                        print("Missing serverTime in response")
                        return int(time.time() * 1000)
                        
                    # Validate server time is a reasonable value
                    try:
                        server_time = int(server_time)
                        current_time = int(time.time() * 1000)
                        
                        # Check if server time is within 24 hours of current time
                        if abs(server_time - current_time) > 86400000:  # 24 hours in milliseconds
                            print(f"Server time appears invalid: {server_time}")
                            return current_time
                            
                        return server_time
                    except (ValueError, TypeError):
                        print(f"Invalid serverTime format: {server_time}")
                        return int(time.time() * 1000)
                except ValueError as e:
                    # Return current time if JSON parsing fails
                    print(f"Error parsing server time response: {str(e)}")
                    return int(time.time() * 1000)
            else:
                # Return current time instead of raising exception for robustness
                print(f"Server time request failed with status {response.status_code}")
                return int(time.time() * 1000)
        except Exception as e:
            print(f"Error getting server time: {str(e)}")
            return int(time.time() * 1000)
    
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
        """Make a public API request (no authentication required)
        
        Returns:
            dict: JSON response if successful, empty dict if failed
        """
        url = f"{self.base_url}{endpoint}"
        try:
            # Add timeout to prevent hanging
            response = requests.request(method, url, params=params, timeout=10.0)
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    # Validate response is a dict or list
                    if not isinstance(result, (dict, list)):
                        print(f"Invalid response type from {endpoint}: {type(result)}")
                        return {} if endpoint.endswith('account') else []
                        
                    return result
                except ValueError as e:
                    print(f"JSON parsing error for {endpoint}: {str(e)}")
                    return {}
            else:
                print(f"API request failed with status {response.status_code}: {response.text}")
                return {}
        except Exception as e:
            print(f"Error in public request: {str(e)}")
            return {}
    
    def signed_request(self, method, endpoint, params=None):
        """Make a signed API request (authentication required)
        
        Returns:
            dict: JSON response if successful, empty dict if failed
        """
        try:
            url = f"{self.base_url}{endpoint}"
            
            # Validate method
            if not method or method not in ['GET', 'POST', 'DELETE']:
                print(f"Invalid HTTP method: {method}")
                return {}
                
            # Prepare parameters with validation
            if params is not None and not isinstance(params, dict):
                print(f"Invalid params type: {type(params)}, expected dict")
                params = {}
                
            request_params = params.copy() if params else {}
            
            # Add timestamp with validation
            try:
                timestamp = self.get_server_time()
                if not timestamp or not isinstance(timestamp, int):
                    print(f"Invalid timestamp: {timestamp}")
                    timestamp = int(time.time() * 1000)
                request_params['timestamp'] = timestamp
            except Exception as e:
                print(f"Error getting timestamp: {str(e)}")
                request_params['timestamp'] = int(time.time() * 1000)
            
            # Generate signature with validation
            try:
                signature = self.generate_signature(request_params)
                if not signature or not isinstance(signature, str):
                    print(f"Invalid signature: {signature}")
                    return {}
                request_params['signature'] = signature
            except Exception as e:
                print(f"Error generating signature: {str(e)}")
                return {}
            
            # Set headers with validation
            if not self.api_key or not isinstance(self.api_key, str):
                print(f"Invalid API key: {self.api_key}")
                return {}
                
            headers = {
                'X-MEXC-APIKEY': self.api_key,
            }
            
            # Make request with timeout to prevent hanging
            try:
                if method == 'GET':
                    response = requests.get(url, params=request_params, headers=headers, timeout=10.0)
                elif method == 'POST':
                    response = requests.post(url, params=request_params, headers=headers, timeout=10.0)
                elif method == 'DELETE':
                    response = requests.delete(url, params=request_params, headers=headers, timeout=10.0)
                else:
                    print(f"Unsupported HTTP method: {method}")
                    return {}
                
                # Parse response with validation
                if response.status_code == 200:
                    try:
                        result = response.json()
                        
                        # Validate response is a dict or list
                        if not isinstance(result, (dict, list)):
                            print(f"Invalid response type from {endpoint}: {type(result)}")
                            return {} if endpoint.endswith('account') or endpoint.endswith('order') else []
                            
                        return result
                    except ValueError as e:
                        print(f"JSON parsing error for {endpoint}: {str(e)}")
                        return {}
                else:
                    print(f"API signed request failed with status {response.status_code}: {response.text}")
                    return {}
            except requests.exceptions.Timeout:
                print(f"Request timeout for {endpoint}")
                return {}
            except requests.exceptions.RequestException as e:
                print(f"Request error for {endpoint}: {str(e)}")
                return {}
        except Exception as e:
            print(f"Error in signed request: {str(e)}")
            return {}
    
    def test_connectivity(self):
        """Test API connectivity and authentication"""
        try:
            # Test public endpoint with validation
            ping_response = self.public_request('GET', f"{self.api_v3}/ping")
            if not ping_response and ping_response != {}:  # Empty dict is valid for ping
                return False, "Public API test failed: Empty response"
            
            # Test authenticated endpoint with minimal parameters
            account_response = self.signed_request('GET', f"{self.api_v3}/account")
            
            # Validate account response
            if not account_response:
                return False, "Authentication failed: Empty response"
                
            if not isinstance(account_response, dict):
                return False, f"Authentication failed: Invalid response type: {type(account_response)}"
                
            # Check for required fields in account response
            required_fields = ["makerCommission", "takerCommission", "balances"]
            for field in required_fields:
                if field not in account_response:
                    return False, f"Authentication failed: Missing required field '{field}' in response"
            
            # Validate balances field
            balances = account_response.get("balances")
            if not isinstance(balances, list):
                return False, f"Authentication failed: Invalid balances type: {type(balances)}"
            
            return True, "API connectivity and authentication successful"
        except Exception as e:
            return False, f"API connectivity test failed with error: {str(e)}"

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
        print(f"Account info: {account_info}")
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
