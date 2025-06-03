#!/usr/bin/env python
"""
BTC/USDC Trading System Starter
This script runs the necessary components to launch the BTC/USDC trading system
with real API integration or mock mode for development.
"""

import os
import sys
import time
import json
import threading
import requests
import websocket
import datetime
import argparse
from dotenv import load_dotenv
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import logging
import random
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='trading_agent.log',
    filemode='a'
)
# Add console handler
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='BTC/USDC Trading System')
parser.add_argument('--mock', action='store_true', help='Run in mock mode without real API credentials')
args = parser.parse_args()

# Load environment variables
load_dotenv()
MEXC_API_KEY = os.getenv('MEXC_API_KEY')
MEXC_API_SECRET = os.getenv('MEXC_SECRET_KEY')  # Fixed key name to match .env file

# Check if we should use mock mode
use_mock_mode = args.mock or not (MEXC_API_KEY and MEXC_API_SECRET)

if use_mock_mode:
    logger.info("Running in MOCK MODE - no real trades will be executed")
else:
    logger.info(f"Running in REAL MODE with API credentials: {MEXC_API_KEY[:4]}...{MEXC_API_KEY[-4:]}")

# Initialize Flask app for the dashboard API
app = Flask(__name__, static_folder='static')
CORS(app)

# Global data storage
ticker_data = {"price": 0, "volume": 0, "change": 0}
order_book_data = {"bids": [], "asks": []}
trades_data = []
account_data = {
    "balances": {
        "USDC": 10000.0,
        "BTC": 1.0
    }
}
candles_data = []

# MEXC API endpoints
MEXC_REST_API_URL = "https://api.mexc.com"
MEXC_WS_URL = "wss://wbs.mexc.com/ws"

# Trading pair
TRADING_PAIR = "BTCUSDC"

# WebSocket connection
ws = None
ws_connected = False
last_order_book_update = time.time()
last_trades_update = time.time()

# Generate initial candlestick data
def generate_initial_candles():
    global candles_data
    
    now = datetime.datetime.now()
    start_time = now - datetime.timedelta(days=7)
    
    # Start with a base price around current BTC price
    base_price = 66000
    current_price = base_price
    
    candles = []
    for i in range(168):  # 7 days * 24 hours = 168 hours
        timestamp = int((start_time + datetime.timedelta(hours=i)).timestamp() * 1000)
        
        # Generate realistic price movement
        price_change_pct = np.random.normal(0, 0.01)  # Normal distribution with 1% std dev
        current_price = current_price * (1 + price_change_pct)
        
        # Generate OHLC
        open_price = current_price
        high_price = open_price * (1 + abs(np.random.normal(0, 0.005)))
        low_price = open_price * (1 - abs(np.random.normal(0, 0.005)))
        close_price = open_price * (1 + np.random.normal(0, 0.008))
        
        # Generate volume
        volume = abs(np.random.normal(10, 5))
        
        candle = {
            "timestamp": timestamp,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume
        }
        candles.append(candle)
    
    candles_data = candles
    ticker_data["price"] = candles[-1]["close"]
    
    logger.info(f"Generated {len(candles)} initial candles")

# Generate realistic order book
def generate_order_book():
    global order_book_data, ticker_data
    
    current_price = ticker_data["price"]
    bids = []
    asks = []
    
    # Generate 20 bid prices below current price
    for i in range(20):
        price = current_price * (1 - (i * 0.001) - random.uniform(0, 0.0005))
        amount = random.uniform(0.1, 2.0)
        bids.append([str(price), str(amount)])
    
    # Generate 20 ask prices above current price
    for i in range(20):
        price = current_price * (1 + (i * 0.001) + random.uniform(0, 0.0005))
        amount = random.uniform(0.1, 2.0)
        asks.append([str(price), str(amount)])
    
    # Sort bids in descending order and asks in ascending order
    bids.sort(key=lambda x: float(x[0]), reverse=True)
    asks.sort(key=lambda x: float(x[0]))
    
    order_book_data = {
        "bids": bids,
        "asks": asks
    }
    
    logger.info(f"Generated order book with {len(bids)} bids and {len(asks)} asks")

# Generate recent trades
def generate_trades(count=20):
    global trades_data, ticker_data
    
    current_price = ticker_data["price"]
    trades = []
    
    for i in range(count):
        # Random price around current price
        price_variation = random.uniform(-0.002, 0.002)
        price = current_price * (1 + price_variation)
        
        # Random amount
        amount = random.uniform(0.01, 0.5)
        
        # Random timestamp in the last 10 minutes
        timestamp = int(time.time() * 1000) - random.randint(0, 600000)
        
        # Buyer or seller
        is_buyer_maker = random.choice([True, False])
        
        trade = {
            "id": f"t{int(time.time() * 1000)}-{i}",
            "price": str(price),
            "quantity": str(amount),
            "timestamp": timestamp,
            "isBuyerMaker": is_buyer_maker
        }
        trades.append(trade)
    
    # Sort by timestamp (newest first)
    trades.sort(key=lambda x: x["timestamp"], reverse=True)
    trades_data = trades
    
    logger.info(f"Generated {len(trades)} recent trades")

# Fetch real market data from MEXC API
def fetch_real_market_data():
    global ticker_data, order_book_data, trades_data, candles_data
    
    try:
        # Fetch ticker data
        ticker_url = f"{MEXC_REST_API_URL}/api/v3/ticker/24hr?symbol={TRADING_PAIR}"
        ticker_response = requests.get(ticker_url)
        ticker_data_raw = ticker_response.json()
        
        ticker_data = {
            "price": float(ticker_data_raw.get("lastPrice", 0)),
            "volume": float(ticker_data_raw.get("volume", 0)),
            "change": float(ticker_data_raw.get("priceChangePercent", 0))
        }
        
        # Fetch order book
        orderbook_url = f"{MEXC_REST_API_URL}/api/v3/depth?symbol={TRADING_PAIR}&limit=20"
        orderbook_response = requests.get(orderbook_url)
        orderbook_data_raw = orderbook_response.json()
        
        order_book_data = {
            "bids": orderbook_data_raw.get("bids", []),
            "asks": orderbook_data_raw.get("asks", [])
        }
        
        # Fetch recent trades
        trades_url = f"{MEXC_REST_API_URL}/api/v3/trades?symbol={TRADING_PAIR}&limit=20"
        trades_response = requests.get(trades_url)
        trades_data_raw = trades_response.json()
        
        trades_data = []
        for trade in trades_data_raw:
            trades_data.append({
                "id": trade.get("id", ""),
                "price": trade.get("price", ""),
                "quantity": trade.get("qty", ""),
                "timestamp": trade.get("time", 0),
                "isBuyerMaker": trade.get("isBuyerMaker", False)
            })
        
        # Fetch klines (candles)
        klines_url = f"{MEXC_REST_API_URL}/api/v3/klines?symbol={TRADING_PAIR}&interval=1h&limit=168"
        klines_response = requests.get(klines_url)
        klines_data_raw = klines_response.json()
        
        candles_data = []
        for kline in klines_data_raw:
            candles_data.append({
                "timestamp": kline[0],
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5])
            })
        
        logger.info("Successfully fetched real market data from MEXC API")
        return True
    except Exception as e:
        logger.error(f"Error fetching real market data: {str(e)}")
        return False

# API routes
@app.route('/api/v1/ticker', methods=['GET'])
def get_ticker():
    return jsonify(ticker_data)

@app.route('/api/v1/orderbook', methods=['GET'])
def get_orderbook():
    return jsonify(order_book_data)

@app.route('/api/v1/trades', methods=['GET'])
def get_trades():
    return jsonify(trades_data)

@app.route('/api/v1/klines', methods=['GET'])
def get_candles():
    return jsonify(candles_data)

@app.route('/api/v1/account', methods=['GET'])
def get_account():
    return jsonify(account_data)

@app.route('/api/v1/order', methods=['POST'])
def place_order():
    # Implement paper trading logic
    return jsonify({"success": True, "message": "Order placed successfully"})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "mode": "mock" if use_mock_mode else "real"})

@app.route('/ws')
def websocket():
    return jsonify({"message": "WebSocket endpoint available at /ws"})

# Serve the React app for any other route
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

# Update data in background
def update_data():
    global ticker_data, order_book_data, trades_data, last_order_book_update, last_trades_update
    
    while True:
        if use_mock_mode:
            # Update ticker randomly every 1-3 seconds
            current_price = ticker_data["price"]
            price_change = current_price * random.uniform(-0.002, 0.002)
            new_price = current_price + price_change
            
            ticker_data = {
                "price": new_price,
                "volume": random.uniform(100, 1000),
                "change": price_change / current_price * 100
            }
            
            # Update order book every 5 seconds
            if time.time() - last_order_book_update > 5:
                generate_order_book()
                last_order_book_update = time.time()
            
            # Update trades occasionally
            if time.time() - last_trades_update > 10:
                # Add 1-3 new trades
                new_trade_count = random.randint(1, 3)
                generate_trades(new_trade_count)
                last_trades_update = time.time()
        else:
            # Fetch real market data every 10 seconds
            if time.time() - last_order_book_update > 10:
                fetch_real_market_data()
                last_order_book_update = time.time()
                last_trades_update = time.time()
        
        time.sleep(random.uniform(1, 3))

# Main function
def main():
    logger.info(f"Starting BTC/USDC Trading System in {'MOCK' if use_mock_mode else 'REAL'} mode")
    
    if use_mock_mode:
        # Generate initial mock data
        generate_initial_candles()
        generate_order_book()
        generate_trades()
    else:
        # Fetch initial real market data
        success = fetch_real_market_data()
        if not success:
            logger.warning("Failed to fetch initial real market data, falling back to mock data")
            generate_initial_candles()
            generate_order_book()
            generate_trades()
    
    # Start data update thread
    update_thread = threading.Thread(target=update_data, daemon=True)
    update_thread.start()
    
    # Start the Flask app
    logger.info("Starting dashboard server at http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, threaded=True)

if __name__ == "__main__":
    main()
