#!/usr/bin/env python
"""
Optimized MEXC API Client for Flash Trading

This module provides an optimized client for interacting with the MEXC API,
with a focus on minimizing latency for flash trading operations.
"""

import hmac
import hashlib
import time
import os
import json
import logging
import asyncio
import aiohttp
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("flash_trading")

class OptimizedMexcClient:
    """Optimized MEXC API Client for ultra-low latency trading"""
    
    def __init__(self, api_key=None, secret_key=None, env_path=None):
        """Initialize the optimized MEXC client"""
        if api_key and secret_key:
            self.api_key = api_key
            self.secret_key = secret_key
        else:
            if env_path:
                load_dotenv(env_path)
            
            self.api_key = os.getenv('MEXC_API_KEY')
            self.secret_key = os.getenv('MEXC_API_SECRET')
            
            if not self.api_key or not self.secret_key:
                raise ValueError("MEXC API credentials not found in environment variables")
        
        self.base_url = "https://api.mexc.com"
        self.api_v3 = "/api/v3"
        
        # Connection pooling for HTTP requests
        self.session = None
        self.async_session = None
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Cache for frequently accessed data
        self.cache = {
            "server_time_offset": 0,  # Offset between local and server time
            "symbols_info": {},       # Symbol information cache
            "order_book": {},         # Order book cache
            "last_update": {}         # Timestamp of last cache update
        }
        
        # Initialize connection pool
        self._init_session()
        
        # Synchronize time with server
        self._sync_server_time()
    
    def _init_session(self):
        """Initialize HTTP session with connection pooling"""
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "DELETE"]
        )
        
        # Create session with connection pooling
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=retry_strategy
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'FlashTradingBot/1.0',
            'Accept': 'application/json',
            'X-MEXC-APIKEY': self.api_key
        })
    
    async def _init_async_session(self):
        """Initialize async HTTP session with connection pooling"""
        if self.async_session is None:
            timeout = aiohttp.ClientTimeout(total=5)
            self.async_session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': 'FlashTradingBot/1.0',
                    'Accept': 'application/json',
                    'X-MEXC-APIKEY': self.api_key
                }
            )
    
    def _sync_server_time(self):
        """Synchronize local time with server time"""
        try:
            local_time = int(time.time() * 1000)
            response = self.session.get(f"{self.base_url}{self.api_v3}/time")
            if response.status_code == 200:
                server_time = response.json()['serverTime']
                self.cache["server_time_offset"] = server_time - local_time
                logger.info(f"Time synchronized with server. Offset: {self.cache['server_time_offset']}ms")
            else:
                logger.warning(f"Failed to sync time with server: {response.text}")
        except Exception as e:
            logger.error(f"Error synchronizing time: {str(e)}")
    
    def get_server_time(self):
        """Get current server time (with local calculation for speed)"""
        return int(time.time() * 1000) + self.cache["server_time_offset"]
    
    def generate_signature(self, params):
        """Generate HMAC SHA256 signature for API request"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def public_request(self, method, endpoint, params=None):
        """Make a public API request (no authentication required)"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                response = self.session.get(url, params=params)
            elif method == 'POST':
                response = self.session.post(url, params=params)
            else:
                response = self.session.request(method, url, params=params)
                
            return response
        except Exception as e:
            logger.error(f"Error in public request: {str(e)}")
            raise
    
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
        
        try:
            if method == 'GET':
                response = self.session.get(url, params=request_params)
            elif method == 'POST':
                response = self.session.post(url, params=request_params)
            elif method == 'DELETE':
                response = self.session.delete(url, params=request_params)
            else:
                response = self.session.request(method, url, params=request_params)
                
            return response
        except Exception as e:
            logger.error(f"Error in signed request: {str(e)}")
            raise
    
    async def async_public_request(self, method, endpoint, params=None):
        """Make an asynchronous public API request"""
        await self._init_async_session()
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                async with self.async_session.get(url, params=params) as response:
                    return await response.json(), response.status
            elif method == 'POST':
                async with self.async_session.post(url, params=params) as response:
                    return await response.json(), response.status
            else:
                async with self.async_session.request(method, url, params=params) as response:
                    return await response.json(), response.status
        except Exception as e:
            logger.error(f"Error in async public request: {str(e)}")
            raise
    
    async def async_signed_request(self, method, endpoint, params=None):
        """Make an asynchronous signed API request"""
        await self._init_async_session()
        url = f"{self.base_url}{endpoint}"
        
        # Prepare parameters
        request_params = params.copy() if params else {}
        
        # Add timestamp
        request_params['timestamp'] = self.get_server_time()
        
        # Generate signature
        signature = self.generate_signature(request_params)
        request_params['signature'] = signature
        
        try:
            if method == 'GET':
                async with self.async_session.get(url, params=request_params) as response:
                    return await response.json(), response.status
            elif method == 'POST':
                async with self.async_session.post(url, params=request_params) as response:
                    return await response.json(), response.status
            elif method == 'DELETE':
                async with self.async_session.delete(url, params=request_params) as response:
                    return await response.json(), response.status
            else:
                async with self.async_session.request(method, url, params=request_params) as response:
                    return await response.json(), response.status
        except Exception as e:
            logger.error(f"Error in async signed request: {str(e)}")
            raise
    
    def get_order_book(self, symbol, limit=5, use_cache=True, max_age_ms=500):
        """Get order book with optional caching for reduced latency"""
        cache_key = f"{symbol}_{limit}"
        current_time = int(time.time() * 1000)
        
        # Check if we have a recent cached version
        if use_cache and cache_key in self.cache["order_book"]:
            last_update = self.cache["last_update"].get(cache_key, 0)
            if current_time - last_update < max_age_ms:
                return self.cache["order_book"][cache_key]
        
        # Fetch fresh data
        response = self.public_request('GET', f"{self.api_v3}/depth", {
            "symbol": symbol,
            "limit": limit
        })
        
        if response.status_code == 200:
            order_book = response.json()
            
            # Cache the result
            if use_cache:
                self.cache["order_book"][cache_key] = order_book
                self.cache["last_update"][cache_key] = current_time
            
            return order_book
        else:
            logger.warning(f"Failed to get order book: {response.text}")
            return None
    
    def place_order(self, symbol, side, type, **kwargs):
        """Place an order with optimized latency"""
        # Prepare order parameters
        params = {
            "symbol": symbol,
            "side": side,
            "type": type,
        }
        
        # Add optional parameters
        for key, value in kwargs.items():
            if value is not None:
                params[key] = value
        
        # Send order request
        response = self.signed_request('POST', f"{self.api_v3}/order", params)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Order placement failed: {response.text}")
            return None
    
    async def async_place_order(self, symbol, side, type, **kwargs):
        """Place an order asynchronously for minimum latency"""
        # Prepare order parameters
        params = {
            "symbol": symbol,
            "side": side,
            "type": type,
        }
        
        # Add optional parameters
        for key, value in kwargs.items():
            if value is not None:
                params[key] = value
        
        # Send order request
        result, status = await self.async_signed_request('POST', f"{self.api_v3}/order", params)
        
        if status == 200:
            return result
        else:
            logger.warning(f"Async order placement failed: {result}")
            return None
    
    def place_test_order(self, symbol, side, type, **kwargs):
        """Test order placement without executing"""
        # Prepare order parameters
        params = {
            "symbol": symbol,
            "side": side,
            "type": type,
        }
        
        # Add optional parameters
        for key, value in kwargs.items():
            if value is not None:
                params[key] = value
        
        # Send test order request
        response = self.signed_request('POST', f"{self.api_v3}/order/test", params)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Test order failed: {response.text}")
            return None
    
    def get_account_info(self):
        """Get account information"""
        response = self.signed_request('GET', f"{self.api_v3}/account")
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to get account info: {response.text}")
            return None
    
    def get_open_orders(self, symbol=None):
        """Get open orders with optional symbol filter"""
        params = {}
        if symbol:
            params["symbol"] = symbol
            
        response = self.signed_request('GET', f"{self.api_v3}/openOrders", params)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to get open orders: {response.text}")
            return None
    
    def cancel_order(self, symbol, order_id=None, client_order_id=None):
        """Cancel an order by ID or client order ID"""
        params = {"symbol": symbol}
        
        if order_id:
            params["orderId"] = order_id
        elif client_order_id:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id must be provided")
            
        response = self.signed_request('DELETE', f"{self.api_v3}/order", params)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to cancel order: {response.text}")
            return None
    
    def cancel_all_orders(self, symbol):
        """Cancel all open orders for a symbol"""
        params = {"symbol": symbol}
        response = self.signed_request('DELETE', f"{self.api_v3}/openOrders", params)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to cancel all orders: {response.text}")
            return None
    
    def close(self):
        """Close connections and clean up resources"""
        if self.session:
            self.session.close()
        
        if self.executor:
            self.executor.shutdown(wait=False)
        
        # Close async session if it exists
        if self.async_session:
            asyncio.create_task(self.async_session.close())

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test optimized MEXC client')
    parser.add_argument('--env', default=".env-secure/.env", help='Path to .env file')
    parser.add_argument('--symbol', default="BTCUSDT", help='Symbol to test with')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark tests')
    
    args = parser.parse_args()
    
    client = OptimizedMexcClient(env_path=args.env)
    
    try:
        # Test basic functionality
        print(f"Testing optimized client with {args.symbol}...")
        
        # Get order book
        order_book = client.get_order_book(args.symbol)
        print(f"Order book (top bid/ask):")
        if order_book and 'bids' in order_book and 'asks' in order_book:
            print(f"  Top bid: {order_book['bids'][0]}")
            print(f"  Top ask: {order_book['asks'][0]}")
        
        # Get account info
        account = client.get_account_info()
        print(f"Account status: canTrade={account.get('canTrade', False)}")
        
        # Run benchmark if requested
        if args.benchmark:
            print("\nRunning latency benchmark...")
            
            # Test order book latency
            start_time = time.time()
            iterations = 10
            
            for _ in range(iterations):
                client.get_order_book(args.symbol, use_cache=False)
            
            elapsed = time.time() - start_time
            print(f"Order book latency: {(elapsed / iterations) * 1000:.2f}ms average ({iterations} iterations)")
            
            # Test order book with caching
            start_time = time.time()
            for _ in range(iterations):
                client.get_order_book(args.symbol, use_cache=True)
            
            elapsed = time.time() - start_time
            print(f"Order book latency (cached): {(elapsed / iterations) * 1000:.2f}ms average ({iterations} iterations)")
            
            # Test test order latency
            start_time = time.time()
            iterations = 3
            
            for _ in range(iterations):
                client.place_test_order(
                    symbol=args.symbol,
                    side="BUY",
                    type="LIMIT",
                    timeInForce="GTC",
                    quantity="0.001",
                    price="1000.00"  # Far from market price
                )
            
            elapsed = time.time() - start_time
            print(f"Test order latency: {(elapsed / iterations) * 1000:.2f}ms average ({iterations} iterations)")
            
    finally:
        client.close()
