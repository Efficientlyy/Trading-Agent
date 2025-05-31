#!/usr/bin/env python
"""
MEXC Trading System Dashboard - Standalone Chart Implementation

This script creates a Flask server that serves a professional dark-themed
trading dashboard with real BTC/USDC market data from MEXC API.
"""

import os
import json
import time
import hmac
import hashlib
import requests
import threading
import websocket
from datetime import datetime
from flask import Flask, render_template, jsonify, send_from_directory, request
from flask_cors import CORS

# MEXC API credentials
API_KEY = "mx0vglZ8S6aN809vmE"
API_SECRET = "092911cfc14e4e7491a74a750eb1884b"

# Constants
SYMBOL = "BTCUSDC"
BASE_URL = "https://api.mexc.com"
WS_URL = "wss://wbs.mexc.com/ws"

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)

# Paper trading account
account = {
    "balances": {
        "BTC": 1.0,
        "USDC": 10000.0
    }
}

# In-memory cache for market data
cache = {
    "ticker": {"price": 0, "symbol": SYMBOL, "timestamp": 0},
    "orderbook": {"asks": [], "bids": []},
    "trades": [],
    "klines": []
}

# Helper functions for API authentication
def generate_signature(api_secret, params):
    query_string = '&'.join([f"{key}={params[key]}" for key in sorted(params.keys())])
    signature = hmac.new(api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    return signature

def get_timestamp():
    return int(time.time() * 1000)

# MEXC API functions
def get_ticker():
    url = f"{BASE_URL}/api/v3/ticker/price?symbol={SYMBOL}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        cache["ticker"] = {
            "price": float(data["price"]),
            "symbol": SYMBOL,
            "timestamp": get_timestamp()
        }
        return cache["ticker"]
    return None

def get_orderbook(limit=20):
    url = f"{BASE_URL}/api/v3/depth?symbol={SYMBOL}&limit={limit}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        
        # Format order book data
        asks = []
        for ask in data["asks"][:limit]:
            price = float(ask[0])
            amount = float(ask[1])
            asks.append({"price": price, "amount": amount, "total": price * amount})
        
        bids = []
        for bid in data["bids"][:limit]:
            price = float(bid[0])
            amount = float(bid[1])
            bids.append({"price": price, "amount": amount, "total": price * amount})
        
        cache["orderbook"] = {"asks": asks, "bids": bids}
        return cache["orderbook"]
    return None

def get_trades(limit=50):
    url = f"{BASE_URL}/api/v3/trades?symbol={SYMBOL}&limit={limit}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        
        # Format trades data
        trades = []
        for trade in data:
            trades.append({
                "price": float(trade["price"]),
                "quantity": float(trade["qty"]),
                "time": int(trade["time"]),
                "isBuyerMaker": trade["isBuyerMaker"]
            })
        
        cache["trades"] = trades
        return trades
    return None

def get_klines(interval="1m", limit=100):
    url = f"{BASE_URL}/api/v3/klines?symbol={SYMBOL}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        
        # Format klines data
        klines = []
        for kline in data:
            klines.append({
                "time": int(kline[0]),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5])
            })
        
        cache["klines"] = klines
        return klines
    return None

# Background data fetching
def fetch_data_periodically():
    while True:
        try:
            get_ticker()
            get_orderbook()
            get_trades()
            get_klines()
            time.sleep(5)  # Update every 5 seconds
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(5)

# Start background data fetching
data_thread = threading.Thread(target=fetch_data_periodically, daemon=True)
data_thread.start()

# Flask routes
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'standalone.html')

@app.route('/js/<path:path>')
def serve_js(path):
    return send_from_directory(os.path.join(app.static_folder, 'js'), path)

@app.route('/css/<path:path>')
def serve_css(path):
    return send_from_directory(os.path.join(app.static_folder, 'css'), path)

@app.route('/api/ticker')
def api_ticker():
    return jsonify(cache["ticker"])

@app.route('/api/orderbook')
def api_orderbook():
    return jsonify(cache["orderbook"])

@app.route('/api/trades')
def api_trades():
    return jsonify(cache["trades"])

@app.route('/api/klines')
def api_klines():
    interval = request.args.get('interval', '1m')
    return jsonify(cache["klines"])

@app.route('/api/account')
def api_account():
    return jsonify({"balances": account["balances"]})

# Run the app
if __name__ == '__main__':
    # Fetch initial data
    get_ticker()
    get_orderbook()
    get_trades()
    get_klines()
    
    # Start the server
    app.run(host='0.0.0.0', port=8081, debug=True)
