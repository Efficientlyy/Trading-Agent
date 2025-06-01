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
        
        # Request counter for metrics
        self.request_count = 0
        self.async_session = None
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Cache for frequently accessed data
        self.cache = {
            "server_time_offset": 0,  # Offset between local and server time
            "symbols_info": {},       # Symbol information cache
            "order_book": {},         # Order book cache
            "ticker": {},             # Ticker cache
            "klines": {},             # Klines cache
            "last_update": {}         # Timestamp of last cache update
        }
        
    def get_request_count(self):
        """Get the total number of API requests made
        
        Returns:
            int: Total number of API requests
        """
        return self.request_count
    
    def __init_connection(self):
        """Initialize connection pool and sync time if not already done"""
        # Add thread synchronization to prevent race conditions
        import threading
        if not hasattr(self, '_init_lock'):
            self._init_lock = threading.Lock()
            
        with self._init_lock:
            if self.session is None:
                # Initialize connection pool
                self._init_session()
                
                # Synchronize time with server
                self._sync_server_time()
                
                # Verify session is properly initialized
                if self.session is None:
                    logger.error("CRITICAL: Session initialization failed")
                    # Create a minimal session as fallback
                    import requests
                    self.session = requests.Session()
    
    def get_ticker_price(self, symbol):
        """Get latest price for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDC')
            
        Returns:
            dict: Ticker data with price information
        """
        self.__init_connection()
        
        # Check cache first
        cache_key = f"ticker_{symbol}"
        if cache_key in self.cache and time.time() - self.cache["last_update"].get(cache_key, 0) < 1.0:
            return self.cache[cache_key]
        
        # Make API request
        endpoint = f"{self.api_v3}/ticker/price"
        params = {"symbol": symbol}
        
        result = self.public_request('GET', endpoint, params)
        
        # Enhanced validation of result
        if not result:
            logger.warning(f"Empty ticker price response for {symbol}")
            return {"symbol": symbol, "price": "0.0"}
            
        # Validate required fields exist
        if not isinstance(result, dict):
            logger.error(f"Invalid ticker price response type for {symbol}: {type(result)}")
            return {"symbol": symbol, "price": "0.0"}
            
        if "price" not in result:
            logger.warning(f"Missing price in ticker response for {symbol}")
            return {"symbol": symbol, "price": "0.0"}
            
        if "symbol" not in result:
            # Add symbol if missing
            result["symbol"] = symbol
        
        # Update cache
        self.cache[cache_key] = result
        self.cache["last_update"][cache_key] = time.time()
        
        return result
    
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
        
        # Set default headers - FIX: Ensure api_key is a string, not the client object
        if self.api_key is not None and isinstance(self.api_key, str):
            api_key_str = self.api_key
        else:
            logger.warning("API key is not a valid string, using empty string as fallback")
            api_key_str = ""
            
        self.session.headers.update({
            'User-Agent': 'FlashTradingBot/1.0',
            'Accept': 'application/json',
            'X-MEXC-APIKEY': api_key_str
        })
    
    async def _init_async_session(self):
        """Initialize async HTTP session with connection pooling"""
        if self.async_session is None:
            timeout = aiohttp.ClientTimeout(total=5)
            
            # Ensure API key is a string
            api_key_str = str(self.api_key) if self.api_key is not None else ""
                
            self.async_session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': 'FlashTradingBot/1.0',
                    'Accept': 'application/json',
                    'X-MEXC-APIKEY': api_key_str
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
    
    def public_request(self, method, endpoint, params=None, max_retries=3):
        """Make a public API request (no authentication required)
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Request parameters
            max_retries: Maximum number of retry attempts
            
        Returns:
            dict: JSON response if successful, empty dict if failed after retries
        """
        # CRITICAL: Ensure session is initialized before making request
        self.__init_connection()
        
        url = f"{self.base_url}{endpoint}"
        
        # Increment request counter
        self.request_count += 1
        
        # Determine expected response type based on endpoint
        is_order_book = 'depth' in endpoint
        is_ticker = 'ticker' in endpoint
        is_account = 'account' in endpoint
        
        # Prepare default empty response based on endpoint type
        default_empty_response = {}
        if is_order_book:
            default_empty_response = {"bids": [], "asks": []}
        elif is_ticker:
            default_empty_response = {"symbol": params.get("symbol", ""), "price": "0.0"} if params else {}
        
        # Retry logic
        retries = 0
        while retries <= max_retries:
            try:
                if method == 'GET':
                    response = self.session.get(url, params=params)
                elif method == 'POST':
                    response = self.session.post(url, params=params)
                else:
                    response = self.session.request(method, url, params=params)
                
                # Check for successful response
                if response.status_code == 200:
                    try:
                        json_response = response.json()
                        if json_response is None:
                            logger.warning(f"API returned null JSON response for {endpoint}")
                            return default_empty_response
                        return json_response  # Return parsed JSON directly
                    except ValueError as e:
                        logger.error(f"Error parsing JSON response: {str(e)}")
                        return default_empty_response
                else:
                    logger.warning(f"API request failed with status {response.status_code}: {response.text}")
                    
                    # Retry on server errors (5xx)
                    if 500 <= response.status_code < 600 and retries < max_retries:
                        retries += 1
                        retry_delay = 0.5 * (2 ** retries)  # Exponential backoff
                        logger.info(f"Retrying in {retry_delay:.2f} seconds (attempt {retries}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    
                    # Return appropriate empty response for client errors
                    return default_empty_response
                    
            except Exception as e:
                logger.error(f"Error in public request: {str(e)}")
                
                # Retry on connection errors
                if retries < max_retries:
                    retries += 1
                    retry_delay = 0.5 * (2 ** retries)  # Exponential backoff
                    logger.info(f"Retrying in {retry_delay:.2f} seconds (attempt {retries}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    # Return appropriate empty response after all retries failed
                    return default_empty_response
    
    def signed_request(self, method, endpoint, params=None, max_retries=3):
        """Make a signed API request (authentication required)
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Request parameters
            max_retries: Maximum number of retry attempts
            
        Returns:
            dict: JSON response if successful, empty dict if failed after retries
        """
        # CRITICAL: Ensure session is initialized before making request
        self.__init_connection()
        
        url = f"{self.base_url}{endpoint}"
        
        # Increment request counter
        self.request_count += 1
        
        # Prepare parameters
        request_params = params.copy() if params else {}
        
        # Retry logic
        retries = 0
        while retries <= max_retries:
            try:
                # Add timestamp (refreshed on each retry)
                request_params['timestamp'] = self.get_server_time()
                
                # Generate signature
                signature = self.generate_signature(request_params)
                request_params['signature'] = signature
                
                if method == 'GET':
                    response = self.session.get(url, params=request_params)
                elif method == 'POST':
                    response = self.session.post(url, params=request_params)
                elif method == 'DELETE':
                    response = self.session.delete(url, params=request_params)
                else:
                    response = self.session.request(method, url, params=request_params)
                
                # Check for successful response
                if response.status_code == 200:
                    return response.json()  # Return parsed JSON directly
                else:
                    logger.warning(f"API signed request failed with status {response.status_code}: {response.text}")
                    
                    # Retry on server errors (5xx)
                    if 500 <= response.status_code < 600 and retries < max_retries:
                        retries += 1
                        retry_delay = 0.5 * (2 ** retries)  # Exponential backoff
                        logger.info(f"Retrying in {retry_delay:.2f} seconds (attempt {retries}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    
                    # Return empty dict for client errors to avoid NoneType errors
                    return {}
                    
            except Exception as e:
                logger.error(f"Error in signed request: {str(e)}")
                
                # Retry on connection errors
                if retries < max_retries:
                    retries += 1
                    retry_delay = 0.5 * (2 ** retries)  # Exponential backoff
                    logger.info(f"Retrying in {retry_delay:.2f} seconds (attempt {retries}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    # Return empty dict after all retries failed
                    return {}
    
    async def async_public_request(self, method, endpoint, params=None, max_retries=3):
        """Make an asynchronous public API request
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Request parameters
            max_retries: Maximum number of retry attempts
            
        Returns:
            dict: JSON response if successful, empty dict if failed after retries
        """
        await self._init_async_session()
        url = f"{self.base_url}{endpoint}"
        
        # Increment request counter
        self.request_count += 1
        
        # Retry logic
        retries = 0
        while retries <= max_retries:
            try:
                # Unified request handling with common response processing
                if method == 'GET':
                    response_obj = self.async_session.get(url, params=params)
                elif method == 'POST':
                    response_obj = self.async_session.post(url, params=params)
                else:
                    response_obj = self.async_session.request(method, url, params=params)
                
                async with response_obj as response:
                    if response.status == 200:
                        try:
                            result = await response.json()
                            
                            # Validate response is a dict or list
                            if not isinstance(result, (dict, list)):
                                logger.error(f"Invalid response type from {endpoint}: {type(result)}")
                                return {} if endpoint.endswith('account') else []
                                
                            return result
                        except (ValueError, TypeError) as e:
                            logger.error(f"JSON parsing error for {endpoint}: {str(e)}")
                            return {} if endpoint.endswith('account') else []
                    else:
                        text = await response.text()
                        logger.warning(f"Async API request failed with status {response.status}: {text}")
                        
                        # Retry on server errors (5xx)
                        if 500 <= response.status < 600 and retries < max_retries:
                            retries += 1
                            retry_delay = 0.5 * (2 ** retries)  # Exponential backoff
                            logger.info(f"Retrying in {retry_delay:.2f} seconds (attempt {retries}/{max_retries})")
                            await asyncio.sleep(retry_delay)
                            continue
                        
                        # Return appropriate empty result based on endpoint
                        # Account endpoints typically return objects, list endpoints return arrays
                        return {} if endpoint.endswith('account') else []
                        
            except Exception as e:
                logger.error(f"Error in async public request to {endpoint}: {str(e)}")
                
                # Retry on connection errors
                if retries < max_retries:
                    retries += 1
                    retry_delay = 0.5 * (2 ** retries)  # Exponential backoff
                    logger.info(f"Retrying in {retry_delay:.2f} seconds (attempt {retries}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                else:
                    # Return appropriate empty result based on endpoint
                    return {} if endpoint.endswith('account') else []
    async def async_signed_request(self, method, endpoint, params=None, max_retries=3):
        """Make an asynchronous signed API request
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Request parameters
            max_retries: Maximum number of retry attempts
            
        Returns:
            dict: JSON response if successful, empty dict if failed after retries
        """
        await self._init_async_session()
        url = f"{self.base_url}{endpoint}"
        
        # Increment request counter
        self.request_count += 1
        
        # Prepare parameters
        request_params = params.copy() if params else {}
        
        # Retry logic
        retries = 0
        while retries <= max_retries:
            try:
                # Add timestamp (refreshed on each retry)
                request_params['timestamp'] = self.get_server_time()
                
                # Generate signature
                signature = self.generate_signature(request_params)
                request_params['signature'] = signature
                
                # Unified request handling with common response processing
                if method == 'GET':
                    response_obj = self.async_session.get(url, params=request_params)
                elif method == 'POST':
                    response_obj = self.async_session.post(url, params=request_params)
                elif method == 'DELETE':
                    response_obj = self.async_session.delete(url, params=request_params)
                else:
                    response_obj = self.async_session.request(method, url, params=request_params)
                
                async with response_obj as response:
                    if response.status == 200:
                        try:
                            result = await response.json()
                            
                            # Validate response is a dict or list
                            if not isinstance(result, (dict, list)):
                                logger.error(f"Invalid response type from {endpoint}: {type(result)}")
                                return {} if endpoint.endswith('account') or endpoint.endswith('order') else []
                                
                            return result
                        except (ValueError, TypeError) as e:
                            logger.error(f"JSON parsing error for {endpoint}: {str(e)}")
                            return {} if endpoint.endswith('account') or endpoint.endswith('order') else []
                    else:
                        text = await response.text()
                        logger.warning(f"Async API signed request failed with status {response.status}: {text}")
                        
                        # Retry on server errors (5xx)
                        if 500 <= response.status < 600 and retries < max_retries:
                            retries += 1
                            retry_delay = 0.5 * (2 ** retries)  # Exponential backoff
                            logger.info(f"Retrying in {retry_delay:.2f} seconds (attempt {retries}/{max_retries})")
                            await asyncio.sleep(retry_delay)
                            continue
                        
                        # Return appropriate empty result based on endpoint
                        return {} if endpoint.endswith('account') or endpoint.endswith('order') else []
            except Exception as e:
                logger.error(f"Error in async signed request to {endpoint}: {str(e)}")
                
                # Retry on connection errors
                if retries < max_retries:
                    retries += 1
                    retry_delay = 0.5 * (2 ** retries)  # Exponential backoff
                    logger.info(f"Retrying in {retry_delay:.2f} seconds (attempt {retries}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                else:
                    # Return appropriate empty result based on endpoint
                    return {} if endpoint.endswith('account') or endpoint.endswith('order') else []
    
    def get_order_book(self, symbol, limit=5, use_cache=True, max_age_ms=500):
        """Get order book with optional caching for reduced latency
        
        Args:
            symbol: Trading pair symbol
            limit: Order book depth
            use_cache: Whether to use cached data
            max_age_ms: Maximum age of cached data in milliseconds
            
        Returns:
            dict: Order book data with bids and asks, or empty dict if failed
        """
        cache_key = f"{symbol}_{limit}"
        current_time = int(time.time() * 1000)
        
        # Check if we have a recent cached version
        if use_cache and cache_key in self.cache["order_book"]:
            last_update = self.cache["last_update"].get(cache_key, 0)
            if current_time - last_update < max_age_ms:
                return self.cache["order_book"][cache_key]
        
        # Fetch fresh data - public_request now returns parsed JSON directly
        order_book = self.public_request('GET', f"{self.api_v3}/depth", {
            "symbol": symbol,
            "limit": limit
        })
        
        # Cache the result if it has the expected structure
        if order_book and 'bids' in order_book and 'asks' in order_book:
            if use_cache:
                self.cache["order_book"][cache_key] = order_book
                self.cache["last_update"][cache_key] = current_time
            
            return order_book
        else:
            logger.warning(f"Failed to get valid order book for {symbol}")
            return {}
    
    def place_order(self, symbol, side, type, **kwargs):
        """Place an order with optimized latency
        
        Args:
            symbol: Trading pair symbol
            side: Order side (BUY or SELL)
            type: Order type (LIMIT, MARKET, etc.)
            **kwargs: Additional order parameters
            
        Returns:
            dict: Order data if successful, empty dict if failed
        """
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
        
        # Send order request - signed_request now returns parsed JSON directly
        order_data = self.signed_request('POST', f"{self.api_v3}/order", params)
        
        # Check if order data is valid
        if order_data and 'orderId' in order_data:
            return order_data
        else:
            logger.warning(f"Order placement failed for {symbol}")
            return {}
    
    async def async_place_order(self, symbol, side, type, **kwargs):
        """Place an order asynchronously for minimum latency
        
        Args:
            symbol: Trading pair symbol
            side: Order side (BUY or SELL)
            type: Order type (LIMIT, MARKET, etc.)
            **kwargs: Additional order parameters
            
        Returns:
            dict: Order data if successful, empty dict if failed
        """
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
        try:
            result, status = await self.async_signed_request('POST', f"{self.api_v3}/order", params)
            
            # Check if order data is valid
            if result and 'orderId' in result:
                return result
            else:
                logger.warning(f"Async order placement failed for {symbol}")
                return {}
        except Exception as e:
            logger.error(f"Error in async order placement: {str(e)}")
            return {}
        
        # Code below is unreachable due to the try/except block above
        # Removed to avoid confusion
    
    def place_test_order(self, symbol, side, type, **kwargs):
        """Test order placement without executing
        
        Args:
            symbol: Trading pair symbol
            side: Order side (BUY or SELL)
            type: Order type (LIMIT, MARKET, etc.)
            **kwargs: Additional order parameters
            
        Returns:
            dict: Success response if successful, empty dict if failed
        """
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
        
        # Send test order request - signed_request now returns parsed JSON directly
        result = self.signed_request('POST', f"{self.api_v3}/order/test", params)
        
        # For test orders, an empty dict is a successful response
        if result is not None:
            return result
        else:
            logger.warning(f"Test order failed for {symbol}")
            return {}
    
    def get_account_info(self):
        """Get account information
        
        Returns:
            dict: Account information if successful, empty dict if failed
        """
        # signed_request now returns parsed JSON directly
        account_data = self.signed_request('GET', f"{self.api_v3}/account")
        
        # Check if account data is valid
        if account_data and 'balances' in account_data:
            return account_data
        else:
            logger.warning("Failed to get valid account information")
            return {}
    
    def get_open_orders(self, symbol=None):
        """Get open orders with optional symbol filter
        
        Args:
            symbol: Optional trading pair symbol to filter orders
            
        Returns:
            list: List of open orders if successful, empty list if failed
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
            
        # signed_request now returns parsed JSON directly
        orders_data = self.signed_request('GET', f"{self.api_v3}/openOrders", params)
        
        # Open orders response should be a list
        if isinstance(orders_data, list):
            return orders_data
        else:
            logger.warning(f"Failed to get valid open orders data")
            return []
    
    def cancel_order(self, symbol, order_id=None, client_order_id=None):
        """Cancel an order by ID or client order ID
        
        Args:
            symbol: Trading pair symbol
            order_id: Order ID to cancel
            client_order_id: Client order ID to cancel
            
        Returns:
            dict: Cancellation result if successful, empty dict if failed
        """
        params = {"symbol": symbol}
        
        if order_id:
            params["orderId"] = order_id
        elif client_order_id:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id must be provided")
            
        # signed_request now returns parsed JSON directly
        cancel_data = self.signed_request('DELETE', f"{self.api_v3}/order", params)
        
        # Check if cancellation data is valid
        if cancel_data and 'orderId' in cancel_data:
            return cancel_data
        else:
            logger.warning(f"Failed to cancel order for {symbol}")
            return {}
    
    def cancel_all_orders(self, symbol):
        """Cancel all open orders for a symbol"""
        params = {"symbol": symbol}
        
        # signed_request now returns parsed JSON directly
        cancel_data = self.signed_request('DELETE', f"{self.api_v3}/openOrders", params)
        
        # Check if cancellation data is valid (should be a list of cancelled orders)
        if isinstance(cancel_data, list):
            return cancel_data
        else:
            logger.warning(f"Failed to cancel all orders for {symbol}")
            return []
    
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
        print(f"Account status: canTrade={account.get('canTrade', False) if account else False}")
        
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
