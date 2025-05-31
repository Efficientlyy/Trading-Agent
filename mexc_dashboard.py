#!/usr/bin/env python
"""
MEXC Trading Dashboard
A professional dark-themed dashboard for BTC/USDC trading with real-time market data.
"""

import os
import sys
import time
import json
import hmac
import hashlib
import threading
import logging
from datetime import datetime
from urllib.parse import urlencode

import requests
import websocket
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS

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

# Global data storage
ticker_data = {}
order_book_data = {"bids": [], "asks": []}
trades_data = []
klines_data = []
account_data = {
    "balances": {
        "USDC": 10000.0,
        "BTC": 1.0
    }
}

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
            'User-Agent': 'MEXC-Trading-Dashboard/1.0'
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

# Initialize Flask app with correct static folder configuration
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Initialize MEXC API client
mexc_client = MexcApiClient(API_KEY, API_SECRET)

# WebSocket message handler
def handle_ws_message(message):
    global ticker_data, order_book_data, trades_data, klines_data
    
    # Extract channel and symbol
    channel = message.get('c', '')
    symbol = message.get('s', '')
    
    if not channel or not symbol:
        return
    
    # Process ticker updates
    if 'ticker' in channel:
        data = message.get('d', {})
        ticker_data = {
            'symbol': symbol,
            'price': data.get('c', 0),
            'priceChange': data.get('m', 0),
            'priceChangePercent': data.get('p', 0),
            'volume': data.get('v', 0),
            'high': data.get('h', 0),
            'low': data.get('l', 0),
            'timestamp': message.get('t', 0)
        }
    
    # Process order book updates
    elif 'depth' in channel:
        data = message.get('d', {})
        
        # Process bids
        if 'bids' in data:
            for bid in data['bids']:
                price = float(bid.get('p', 0))
                volume = float(bid.get('v', 0))
                
                # Update or remove bid
                if volume > 0:
                    # Add or update bid
                    found = False
                    for i, (p, v) in enumerate(order_book_data['bids']):
                        if p == price:
                            order_book_data['bids'][i] = (price, volume)
                            found = True
                            break
                    
                    if not found:
                        order_book_data['bids'].append((price, volume))
                        # Sort bids in descending order by price
                        order_book_data['bids'].sort(key=lambda x: x[0], reverse=True)
                else:
                    # Remove bid
                    order_book_data['bids'] = [b for b in order_book_data['bids'] if b[0] != price]
        
        # Process asks
        if 'asks' in data:
            for ask in data['asks']:
                price = float(ask.get('p', 0))
                volume = float(ask.get('v', 0))
                
                # Update or remove ask
                if volume > 0:
                    # Add or update ask
                    found = False
                    for i, (p, v) in enumerate(order_book_data['asks']):
                        if p == price:
                            order_book_data['asks'][i] = (price, volume)
                            found = True
                            break
                    
                    if not found:
                        order_book_data['asks'].append((price, volume))
                        # Sort asks in ascending order by price
                        order_book_data['asks'].sort(key=lambda x: x[0])
                else:
                    # Remove ask
                    order_book_data['asks'] = [a for a in order_book_data['asks'] if a[0] != price]
    
    # Process trade updates
    elif 'deals' in channel:
        data = message.get('d', {})
        deals = data.get('deals', [])
        
        for deal in deals:
            trade = {
                'price': float(deal.get('p', 0)),
                'quantity': float(deal.get('v', 0)),
                'time': deal.get('t', 0),
                'isBuyerMaker': deal.get('S', 0) == 1
            }
            
            # Add trade to the beginning of the list
            trades_data.insert(0, trade)
            
            # Keep only the last 100 trades
            if len(trades_data) > 100:
                trades_data.pop()
    
    # Process kline updates
    elif 'kline' in channel:
        data = message.get('d', {})
        kline = data.get('k', {})
        
        if kline:
            candle = {
                'time': kline.get('t', 0),
                'open': float(kline.get('o', 0)),
                'high': float(kline.get('h', 0)),
                'low': float(kline.get('l', 0)),
                'close': float(kline.get('c', 0)),
                'volume': float(kline.get('v', 0)),
                'closed': kline.get('x', False)
            }
            
            # Update or add candle
            found = False
            for i, k in enumerate(klines_data):
                if k['time'] == candle['time']:
                    klines_data[i] = candle
                    found = True
                    break
            
            if not found:
                klines_data.append(candle)
                # Sort klines by time
                klines_data.sort(key=lambda x: x['time'])

# Initialize market data
def initialize_market_data():
    global ticker_data, order_book_data, trades_data, klines_data
    
    # Get ticker data
    ticker = mexc_client.get_ticker_price(TRADING_PAIR)
    if ticker:
        ticker_data = {
            'symbol': TRADING_PAIR,
            'price': float(ticker.get('price', 0)),
            'timestamp': int(time.time() * 1000)
        }
    
    # Get order book data
    order_book = mexc_client.get_order_book(TRADING_PAIR, 100)
    if order_book:
        bids = [(float(price), float(qty)) for price, qty in order_book.get('bids', [])]
        asks = [(float(price), float(qty)) for price, qty in order_book.get('asks', [])]
        
        order_book_data = {
            'bids': bids,
            'asks': asks
        }
    
    # Get recent trades
    trades = mexc_client.get_recent_trades(TRADING_PAIR, 100)
    if trades:
        trades_data = [
            {
                'price': float(trade.get('price', 0)),
                'quantity': float(trade.get('qty', 0)),
                'time': trade.get('time', 0),
                'isBuyerMaker': trade.get('isBuyerMaker', False)
            }
            for trade in trades
        ]
    
    # Get klines data
    klines = mexc_client.get_klines(TRADING_PAIR, '1m', limit=100)
    if klines:
        klines_data = [
            {
                'time': kline[0],
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5]),
                'closed': True
            }
            for kline in klines
        ]

# Connect to WebSocket and subscribe to channels
def start_websocket():
    if mexc_client.connect_websocket(handle_ws_message):
        # Subscribe to channels
        mexc_client.subscribe_to_ticker(TRADING_PAIR)
        mexc_client.subscribe_to_kline(TRADING_PAIR, '1m')
        mexc_client.subscribe_to_depth(TRADING_PAIR)
        mexc_client.subscribe_to_trades(TRADING_PAIR)
        return True
    return False

# API routes
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory(os.path.join(app.static_folder, 'js'), filename)

@app.route('/api/ticker')
def get_ticker():
    return jsonify(ticker_data)

@app.route('/api/orderbook')
def get_orderbook():
    return jsonify(order_book_data)

@app.route('/api/trades')
def get_trades():
    return jsonify(trades_data)

@app.route('/api/klines')
def get_klines():
    interval = request.args.get('interval', '1m')
    limit = int(request.args.get('limit', 100))
    
    # If interval is not 1m, fetch from API
    if interval != '1m' or not klines_data:
        klines = mexc_client.get_klines(TRADING_PAIR, interval, limit=limit)
        if klines:
            result = [
                {
                    'time': kline[0],
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'closed': True
                }
                for kline in klines
            ]
            return jsonify(result)
    
    # Return cached data
    return jsonify(klines_data[-limit:])

@app.route('/api/account')
def get_account():
    return jsonify(account_data)

@app.route('/api/order', methods=['POST'])
def place_order():
    # Paper trading only
    data = request.json
    
    # Extract order details
    symbol = data.get('symbol', TRADING_PAIR)
    side = data.get('side', 'BUY')
    type = data.get('type', 'LIMIT')
    quantity = float(data.get('quantity', 0))
    price = float(data.get('price', 0)) if type == 'LIMIT' else None
    
    # Validate order
    if quantity <= 0:
        return jsonify({'success': False, 'message': 'Invalid quantity'})
    
    if type == 'LIMIT' and (price is None or price <= 0):
        return jsonify({'success': False, 'message': 'Invalid price'})
    
    # Get current price if market order
    if type == 'MARKET':
        ticker = mexc_client.get_ticker_price(symbol)
        if ticker:
            price = float(ticker.get('price', 0))
        else:
            return jsonify({'success': False, 'message': 'Failed to get current price'})
    
    # Calculate order value
    order_value = quantity * price
    
    # Check if enough balance
    if side == 'BUY':
        if account_data['balances']['USDC'] < order_value:
            return jsonify({'success': False, 'message': 'Insufficient USDC balance'})
        
        # Update balances
        account_data['balances']['USDC'] -= order_value
        account_data['balances']['BTC'] += quantity
    else:  # SELL
        if account_data['balances']['BTC'] < quantity:
            return jsonify({'success': False, 'message': 'Insufficient BTC balance'})
        
        # Update balances
        account_data['balances']['BTC'] -= quantity
        account_data['balances']['USDC'] += order_value
    
    return jsonify({
        'success': True,
        'message': 'Order placed successfully',
        'order': {
            'symbol': symbol,
            'side': side,
            'type': type,
            'quantity': quantity,
            'price': price,
            'value': order_value,
            'time': int(time.time() * 1000)
        }
    })

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'})

def main():
    # Initialize market data
    initialize_market_data()
    
    # Start WebSocket connection
    if not start_websocket():
        logger.error("Failed to start WebSocket connection")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)

if __name__ == '__main__':
    main()
