#!/usr/bin/env python
"""
MEXC Trading Dashboard - Production Server

This script serves the MEXC Trading Dashboard with real BTC/USDC market data.
It connects to the MEXC API using the provided credentials and serves a
professional dark-themed dashboard with real-time market data.

Usage:
    python3 mexc_dashboard_production.py

Features:
    - Real-time BTC/USDC market data from MEXC API
    - Professional dark-themed UI
    - Live order book with depth visualization
    - Candlestick chart with timeframe selection
    - Recent trades display
    - Paper trading functionality
"""

import os
import json
import time
import hmac
import hashlib
import requests
import threading
from datetime import datetime
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS

# MEXC API credentials
API_KEY = "mx0vglZ8S6aN809vmE"
API_SECRET = "092911cfc14e4e7491a74a750eb1884b"

# Constants
BASE_URL = "https://api.mexc.com"
SYMBOL = "BTC_USDC"  # Trading pair
PORT = 8080

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)

# In-memory data store
market_data = {
    "ticker": {"price": 0, "timestamp": 0},
    "orderbook": {"bids": [], "asks": []},
    "trades": [],
    "klines": []
}

# Paper trading account
paper_account = {
    "balances": {
        "BTC": 1.0,
        "USDC": 10000.0
    },
    "orders": []
}

# MEXC API Helper Functions
def generate_signature(api_secret, params=None):
    """Generate HMAC SHA256 signature for API authentication."""
    if params is None:
        params = {}
    
    # Convert params to sorted query string
    query_string = '&'.join([f"{key}={params[key]}" for key in sorted(params.keys())])
    
    # Create signature
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return signature

def make_request(endpoint, method="GET", params=None, auth=False):
    """Make request to MEXC API with proper authentication if needed."""
    url = f"{BASE_URL}{endpoint}"
    headers = {}
    
    if params is None:
        params = {}
    
    if auth:
        # Add authentication parameters
        params['api_key'] = API_KEY
        params['timestamp'] = int(time.time() * 1000)
        params['recvWindow'] = 5000
        
        # Generate signature
        signature = generate_signature(API_SECRET, params)
        params['signature'] = signature
        
        # Set headers
        headers['X-MEXC-APIKEY'] = API_KEY
    
    try:
        if method == "GET":
            response = requests.get(url, params=params, headers=headers)
        elif method == "POST":
            response = requests.post(url, json=params, headers=headers)
        elif method == "DELETE":
            response = requests.delete(url, params=params, headers=headers)
        else:
            return {"error": "Unsupported method"}
        
        # Check for errors
        if response.status_code != 200:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
        
        return response.json()
    except Exception as e:
        return {"error": f"Request Error: {str(e)}"}

# Data Fetching Functions
def fetch_ticker():
    """Fetch current ticker data for BTC/USDC."""
    endpoint = "/api/v3/ticker/price"
    params = {"symbol": SYMBOL}
    
    response = make_request(endpoint, params=params)
    if "error" not in response:
        price = float(response.get("price", 0))
        market_data["ticker"] = {
            "price": price,
            "timestamp": int(time.time() * 1000)
        }
        return market_data["ticker"]
    return {"error": response.get("error", "Unknown error")}

def fetch_orderbook(limit=20):
    """Fetch order book data for BTC/USDC."""
    endpoint = "/api/v3/depth"
    params = {"symbol": SYMBOL, "limit": limit}
    
    response = make_request(endpoint, params=params)
    if "error" not in response:
        # Process bids and asks
        bids = []
        asks = []
        
        # Calculate totals for depth visualization
        for i, bid in enumerate(response.get("bids", [])):
            price = float(bid[0])
            amount = float(bid[1])
            total = price * amount
            bids.append({"price": price, "amount": amount, "total": total})
        
        for i, ask in enumerate(response.get("asks", [])):
            price = float(ask[0])
            amount = float(ask[1])
            total = price * amount
            asks.append({"price": price, "amount": amount, "total": total})
        
        market_data["orderbook"] = {
            "bids": bids,
            "asks": asks,
            "timestamp": int(time.time() * 1000)
        }
        return market_data["orderbook"]
    return {"error": response.get("error", "Unknown error")}

def fetch_trades(limit=20):
    """Fetch recent trades for BTC/USDC."""
    endpoint = "/api/v3/trades"
    params = {"symbol": SYMBOL, "limit": limit}
    
    response = make_request(endpoint, params=params)
    if "error" not in response and isinstance(response, list):
        trades = []
        for trade in response:
            trades.append({
                "id": trade.get("id"),
                "price": float(trade.get("price", 0)),
                "amount": float(trade.get("qty", 0)),
                "time": trade.get("time"),
                "is_buyer_maker": trade.get("isBuyerMaker", False)
            })
        
        market_data["trades"] = trades
        return trades
    return {"error": response.get("error", "Unknown error")}

def fetch_klines(interval="1m", limit=100):
    """Fetch candlestick data for BTC/USDC."""
    endpoint = "/api/v3/klines"
    params = {"symbol": SYMBOL, "interval": interval, "limit": limit}
    
    response = make_request(endpoint, params=params)
    if "error" not in response and isinstance(response, list):
        klines = []
        for candle in response:
            klines.append({
                "time": candle[0],  # Open time
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5])
            })
        
        market_data["klines"] = klines
        return klines
    return {"error": response.get("error", "Unknown error")}

# Paper Trading Functions
def place_paper_order(order_type, side, price, amount):
    """Place a paper trading order."""
    current_price = market_data["ticker"]["price"]
    
    # Validate order
    if side not in ["buy", "sell"]:
        return {"error": "Invalid side. Must be 'buy' or 'sell'"}
    
    if order_type not in ["limit", "market"]:
        return {"error": "Invalid order type. Must be 'limit' or 'market'"}
    
    if amount <= 0:
        return {"error": "Amount must be greater than 0"}
    
    if order_type == "limit" and price <= 0:
        return {"error": "Price must be greater than 0 for limit orders"}
    
    # Use current price for market orders
    if order_type == "market":
        price = current_price
    
    # Check if user has enough balance
    if side == "buy":
        cost = price * amount
        if paper_account["balances"]["USDC"] < cost:
            return {"error": "Insufficient USDC balance"}
    else:  # sell
        if paper_account["balances"]["BTC"] < amount:
            return {"error": "Insufficient BTC balance"}
    
    # Create order
    order_id = int(time.time() * 1000)
    order = {
        "id": order_id,
        "type": order_type,
        "side": side,
        "price": price,
        "amount": amount,
        "status": "filled",  # Auto-fill paper orders
        "timestamp": int(time.time() * 1000)
    }
    
    # Update balances
    if side == "buy":
        paper_account["balances"]["USDC"] -= price * amount
        paper_account["balances"]["BTC"] += amount
    else:  # sell
        paper_account["balances"]["BTC"] -= amount
        paper_account["balances"]["USDC"] += price * amount
    
    # Add to order history
    paper_account["orders"].append(order)
    
    return order

# Background data fetching
def background_data_fetcher():
    """Continuously fetch market data in the background."""
    while True:
        try:
            fetch_ticker()
            fetch_orderbook()
            fetch_trades()
            fetch_klines()
            time.sleep(5)  # Update every 5 seconds
        except Exception as e:
            print(f"Error in background fetcher: {str(e)}")
            time.sleep(10)  # Wait longer on error

# Start background data fetching
data_thread = threading.Thread(target=background_data_fetcher, daemon=True)
data_thread.start()

# Flask Routes
@app.route('/')
def index():
    """Serve the main dashboard page."""
    return send_from_directory('static', 'simple-dashboard.html')

@app.route('/api/ticker')
def get_ticker():
    """API endpoint for current ticker data."""
    return jsonify(market_data["ticker"])

@app.route('/api/orderbook')
def get_orderbook():
    """API endpoint for order book data."""
    return jsonify(market_data["orderbook"])

@app.route('/api/trades')
def get_trades():
    """API endpoint for recent trades."""
    return jsonify(market_data["trades"])

@app.route('/api/klines')
def get_klines():
    """API endpoint for candlestick data."""
    interval = request.args.get('interval', '1m')
    return jsonify(market_data["klines"])

@app.route('/api/account')
def get_account():
    """API endpoint for paper trading account info."""
    return jsonify(paper_account["balances"])

@app.route('/api/orders', methods=['GET'])
def get_orders():
    """API endpoint for order history."""
    return jsonify(paper_account["orders"])

@app.route('/api/orders', methods=['POST'])
def post_order():
    """API endpoint for placing paper trading orders."""
    data = request.json
    order_type = data.get('type', 'limit')
    side = data.get('side')
    price = float(data.get('price', 0))
    amount = float(data.get('amount', 0))
    
    result = place_paper_order(order_type, side, price, amount)
    return jsonify(result)

# Main entry point
if __name__ == '__main__':
    # Initial data fetch
    fetch_ticker()
    fetch_orderbook()
    fetch_trades()
    fetch_klines()
    
    # Start server
    print(f"Starting MEXC Trading Dashboard on http://localhost:{PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False)
