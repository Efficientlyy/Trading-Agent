#!/usr/bin/env python
"""
MEXC API Test Script
This script tests connectivity to the MEXC API and retrieves market data for BTC/USDC.
"""

import os
import sys
import time
import json
import hmac
import hashlib
import requests
import websocket
import threading
from datetime import datetime
import logging
from urllib.parse import urlencode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MEXC API credentials
API_KEY = "mx0vglZ8S6aN809vmE"
API_SECRET = "092911cfc14e4e7491a74a750eb1884b"

# API endpoints
REST_API_URL = "https://api.mexc.com"
WS_API_URL = "wss://wbs.mexc.com/ws"

# Trading pair
TRADING_PAIR = "BTCUSDC"

class MexcApiClient:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.ws = None
        self.ws_connected = False
        self.ws_thread = None
        
    def _generate_signature(self, params):
        """Generate HMAC SHA256 signature for API request"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
        
    def _get_timestamp(self):
        """Get current timestamp in milliseconds"""
        return int(time.time() * 1000)
        
    def make_request(self, method, endpoint, params=None, signed=False):
        """Make a REST API request to MEXC"""
        url = f"{REST_API_URL}{endpoint}"
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'MEXC-API-Test/1.0'
        }
        
        if signed:
            if params is None:
                params = {}
                
            # Add timestamp for signed requests
            params['timestamp'] = self._get_timestamp()
            params['recvWindow'] = 5000
            
            # Add API key for signed requests
            headers['X-MEXC-APIKEY'] = self.api_key
            
            # Generate signature
            signature = self._generate_signature(params)
            params['signature'] = signature
        
        try:
            if method == 'GET':
                response = self.session.get(url, headers=headers, params=params)
            elif method == 'POST':
                response = self.session.post(url, headers=headers, json=params)
            elif method == 'DELETE':
                response = self.session.delete(url, headers=headers, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {e}")
            return None
            
    def get_server_time(self):
        """Get MEXC server time"""
        return self.make_request('GET', '/api/v3/time')
        
    def get_exchange_info(self):
        """Get exchange information"""
        return self.make_request('GET', '/api/v3/exchangeInfo')
        
    def get_ticker(self, symbol=None):
        """Get ticker information for a symbol or all symbols"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self.make_request('GET', '/api/v3/ticker/24hr', params)
        
    def get_ticker_price(self, symbol=None):
        """Get latest price for a symbol or all symbols"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self.make_request('GET', '/api/v3/ticker/price', params)
        
    def get_order_book(self, symbol, limit=100):
        """Get order book for a symbol"""
        params = {
            'symbol': symbol,
            'limit': limit
        }
        return self.make_request('GET', '/api/v3/depth', params)
        
    def get_recent_trades(self, symbol, limit=500):
        """Get recent trades for a symbol"""
        params = {
            'symbol': symbol,
            'limit': limit
        }
        return self.make_request('GET', '/api/v3/trades', params)
        
    def get_klines(self, symbol, interval, start_time=None, end_time=None, limit=500):
        """Get candlestick data for a symbol"""
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
            
        return self.make_request('GET', '/api/v3/klines', params)
        
    def get_account_information(self):
        """Get account information (requires signed request)"""
        return self.make_request('GET', '/api/v3/account', {}, signed=True)
        
    def connect_websocket(self, callback):
        """Connect to MEXC WebSocket API"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                callback(data)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse WebSocket message: {message}")
                
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
            self.ws_connected = False
            
        def on_open(ws):
            logger.info("WebSocket connection established")
            self.ws_connected = True
            
        def run_websocket():
            self.ws = websocket.WebSocketApp(
                WS_API_URL,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            self.ws.run_forever()
            
        self.ws_thread = threading.Thread(target=run_websocket, daemon=True)
        self.ws_thread.start()
        
        # Wait for connection to establish
        timeout = 5
        start_time = time.time()
        while not self.ws_connected and time.time() - start_time < timeout:
            time.sleep(0.1)
            
        if not self.ws_connected:
            logger.error("Failed to establish WebSocket connection")
            return False
            
        return True
        
    def subscribe_to_ticker(self, symbol):
        """Subscribe to ticker updates for a symbol"""
        if not self.ws_connected:
            logger.error("WebSocket not connected")
            return False
            
        subscription = {
            "method": "SUBSCRIPTION",
            "params": [f"spot@public.ticker.v3.api@{symbol}"]
        }
        self.ws.send(json.dumps(subscription))
        return True
        
    def subscribe_to_kline(self, symbol, interval):
        """Subscribe to kline updates for a symbol"""
        if not self.ws_connected:
            logger.error("WebSocket not connected")
            return False
            
        subscription = {
            "method": "SUBSCRIPTION",
            "params": [f"spot@public.kline.v3.api@{symbol}@{interval}"]
        }
        self.ws.send(json.dumps(subscription))
        return True
        
    def subscribe_to_depth(self, symbol):
        """Subscribe to order book updates for a symbol"""
        if not self.ws_connected:
            logger.error("WebSocket not connected")
            return False
            
        subscription = {
            "method": "SUBSCRIPTION",
            "params": [f"spot@public.increase.depth.v3.api@{symbol}"]
        }
        self.ws.send(json.dumps(subscription))
        return True
        
    def subscribe_to_trades(self, symbol):
        """Subscribe to trade updates for a symbol"""
        if not self.ws_connected:
            logger.error("WebSocket not connected")
            return False
            
        subscription = {
            "method": "SUBSCRIPTION",
            "params": [f"spot@public.deals.v3.api@{symbol}"]
        }
        self.ws.send(json.dumps(subscription))
        return True
        
    def close_websocket(self):
        """Close WebSocket connection"""
        if self.ws:
            self.ws.close()
            self.ws_connected = False

def test_rest_api():
    """Test REST API endpoints"""
    client = MexcApiClient(API_KEY, API_SECRET)
    
    # Test server time
    logger.info("Testing server time...")
    server_time = client.get_server_time()
    if server_time:
        timestamp = server_time.get('serverTime')
        dt = datetime.fromtimestamp(timestamp / 1000)
        logger.info(f"Server time: {dt}")
    else:
        logger.error("Failed to get server time")
        
    # Test ticker price for BTC/USDC
    logger.info(f"Testing ticker price for {TRADING_PAIR}...")
    ticker = client.get_ticker_price(TRADING_PAIR)
    if ticker:
        logger.info(f"Current price: {ticker}")
    else:
        logger.error(f"Failed to get ticker for {TRADING_PAIR}")
        
    # Test order book for BTC/USDC
    logger.info(f"Testing order book for {TRADING_PAIR}...")
    order_book = client.get_order_book(TRADING_PAIR, 10)
    if order_book:
        logger.info(f"Order book: {json.dumps(order_book, indent=2)}")
    else:
        logger.error(f"Failed to get order book for {TRADING_PAIR}")
        
    # Test recent trades for BTC/USDC
    logger.info(f"Testing recent trades for {TRADING_PAIR}...")
    trades = client.get_recent_trades(TRADING_PAIR, 5)
    if trades:
        logger.info(f"Recent trades: {json.dumps(trades, indent=2)}")
    else:
        logger.error(f"Failed to get recent trades for {TRADING_PAIR}")
        
    # Test klines for BTC/USDC
    logger.info(f"Testing klines for {TRADING_PAIR}...")
    klines = client.get_klines(TRADING_PAIR, '1h', limit=5)
    if klines:
        logger.info(f"Klines: {json.dumps(klines, indent=2)}")
    else:
        logger.error(f"Failed to get klines for {TRADING_PAIR}")
        
    # Test account information (signed request)
    logger.info("Testing account information...")
    account = client.get_account_information()
    if account:
        logger.info(f"Account information: {json.dumps(account, indent=2)}")
    else:
        logger.error("Failed to get account information")
        
def test_websocket_api():
    """Test WebSocket API endpoints"""
    client = MexcApiClient(API_KEY, API_SECRET)
    
    # WebSocket message handler
    def handle_ws_message(message):
        logger.info(f"WebSocket message: {json.dumps(message, indent=2)}")
        
    # Connect to WebSocket
    logger.info("Connecting to WebSocket...")
    if client.connect_websocket(handle_ws_message):
        logger.info("WebSocket connected")
        
        # Subscribe to ticker updates
        logger.info(f"Subscribing to ticker updates for {TRADING_PAIR}...")
        client.subscribe_to_ticker(TRADING_PAIR)
        
        # Subscribe to kline updates
        logger.info(f"Subscribing to kline updates for {TRADING_PAIR}...")
        client.subscribe_to_kline(TRADING_PAIR, '1m')
        
        # Subscribe to depth updates
        logger.info(f"Subscribing to depth updates for {TRADING_PAIR}...")
        client.subscribe_to_depth(TRADING_PAIR)
        
        # Subscribe to trade updates
        logger.info(f"Subscribing to trade updates for {TRADING_PAIR}...")
        client.subscribe_to_trades(TRADING_PAIR)
        
        # Wait for some messages
        logger.info("Waiting for WebSocket messages...")
        time.sleep(30)
        
        # Close WebSocket
        logger.info("Closing WebSocket...")
        client.close_websocket()
    else:
        logger.error("Failed to connect to WebSocket")

def main():
    """Main function"""
    logger.info("Starting MEXC API test")
    
    # Test REST API
    test_rest_api()
    
    # Test WebSocket API
    test_websocket_api()
    
    logger.info("MEXC API test completed")

if __name__ == "__main__":
    main()
