#!/usr/bin/env python
"""
Multi-Asset Data Service for Trading-Agent System

This module provides a data service for fetching and managing market data
for multiple cryptocurrency assets (BTC, ETH, SOL) from MEXC API.
"""

import os
import json
import time
import hmac
import hashlib
import requests
import threading
import websocket
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("multi_asset_data_service")

# Import environment loader
try:
    from env_loader import load_environment_variables
except ImportError:
    # Define a simple fallback if env_loader is not available
    def load_environment_variables(env_path=None):
        """Simple fallback for loading environment variables"""
        env_vars = {}
        try:
            if env_path is None:
                env_path = '.env'
            
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            env_vars[key] = value
        except Exception as e:
            logger.error(f"Error loading environment variables: {str(e)}")
        
        return env_vars

class MultiAssetDataService:
    """Data service for multiple cryptocurrency assets"""
    
    def __init__(self, supported_assets=None, env_path=None):
        """Initialize multi-asset data service
        
        Args:
            supported_assets: List of supported assets (default: BTC/USDC, ETH/USDC, SOL/USDC)
            env_path: Path to .env file (optional)
        """
        # Default supported assets
        self.supported_assets = supported_assets or ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
        self.current_asset = self.supported_assets[0]
        
        # Load API credentials
        env_vars = load_environment_variables(env_path)
        self.api_key = env_vars.get('MEXC_API_KEY')
        self.api_secret = env_vars.get('MEXC_SECRET_KEY')
        
        # API endpoints
        self.base_url = "https://api.mexc.com"
        self.ws_url = "wss://wbs.mexc.com/ws"
        
        # Initialize cache for each asset
        self.cache = {asset: {
            "ticker": {"price": 0, "symbol": asset, "timestamp": 0},
            "orderbook": {"asks": [], "bids": []},
            "trades": [],
            "klines": {},  # Keyed by interval
            "patterns": []
        } for asset in self.supported_assets}
        
        # WebSocket connections
        self.ws_connections = {}
        self.ws_threads = {}
        
        # Supported intervals for klines
        self.supported_intervals = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        
        logger.info(f"Initialized MultiAssetDataService with {len(self.supported_assets)} assets")
    
    def switch_asset(self, asset):
        """Switch current asset
        
        Args:
            asset: Asset to switch to
            
        Returns:
            bool: True if switch successful, False otherwise
        """
        if asset in self.supported_assets:
            self.current_asset = asset
            logger.info(f"Switched to asset: {asset}")
            return True
        
        logger.warning(f"Asset not supported: {asset}")
        return False
    
    def get_current_asset(self):
        """Get current asset
        
        Returns:
            str: Current asset
        """
        return self.current_asset
    
    def get_supported_assets(self):
        """Get supported assets
        
        Returns:
            list: Supported assets
        """
        return self.supported_assets
    
    def _generate_signature(self, params):
        """Generate signature for authenticated requests
        
        Args:
            params: Request parameters
            
        Returns:
            str: HMAC SHA256 signature
        """
        if not self.api_secret:
            return ""
            
        query_string = '&'.join([f"{key}={params[key]}" for key in sorted(params.keys())])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _get_timestamp(self):
        """Get current timestamp in milliseconds
        
        Returns:
            int: Current timestamp
        """
        return int(time.time() * 1000)
    
    def get_ticker(self, asset=None):
        """Get ticker for specified asset
        
        Args:
            asset: Asset to get ticker for (default: current asset)
            
        Returns:
            dict: Ticker data
        """
        target_asset = asset or self.current_asset
        symbol = target_asset.replace('/', '')
        
        url = f"{self.base_url}/api/v3/ticker/price?symbol={symbol}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                ticker = {
                    "price": float(data["price"]),
                    "symbol": target_asset,
                    "timestamp": self._get_timestamp()
                }
                self.cache[target_asset]["ticker"] = ticker
                return ticker
            else:
                logger.error(f"Error fetching ticker for {target_asset}: {response.status_code}")
                return self.cache[target_asset]["ticker"]
        except Exception as e:
            logger.error(f"Exception fetching ticker for {target_asset}: {str(e)}")
            return self.cache[target_asset]["ticker"]
    
    def get_orderbook(self, asset=None, limit=20):
        """Get orderbook for specified asset
        
        Args:
            asset: Asset to get orderbook for (default: current asset)
            limit: Number of entries to return
            
        Returns:
            dict: Orderbook data
        """
        target_asset = asset or self.current_asset
        symbol = target_asset.replace('/', '')
        
        url = f"{self.base_url}/api/v3/depth?symbol={symbol}&limit={limit}"
        try:
            response = requests.get(url, timeout=10)
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
                
                orderbook = {"asks": asks, "bids": bids}
                self.cache[target_asset]["orderbook"] = orderbook
                return orderbook
            else:
                logger.error(f"Error fetching orderbook for {target_asset}: {response.status_code}")
                return self.cache[target_asset]["orderbook"]
        except Exception as e:
            logger.error(f"Exception fetching orderbook for {target_asset}: {str(e)}")
            return self.cache[target_asset]["orderbook"]
    
    def get_trades(self, asset=None, limit=50):
        """Get recent trades for specified asset
        
        Args:
            asset: Asset to get trades for (default: current asset)
            limit: Number of trades to return
            
        Returns:
            list: Recent trades
        """
        target_asset = asset or self.current_asset
        symbol = target_asset.replace('/', '')
        
        url = f"{self.base_url}/api/v3/trades?symbol={symbol}&limit={limit}"
        try:
            response = requests.get(url, timeout=10)
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
                
                self.cache[target_asset]["trades"] = trades
                return trades
            else:
                logger.error(f"Error fetching trades for {target_asset}: {response.status_code}")
                return self.cache[target_asset]["trades"]
        except Exception as e:
            logger.error(f"Exception fetching trades for {target_asset}: {str(e)}")
            return self.cache[target_asset]["trades"]
    
    def _normalize_interval(self, interval):
        """Normalize interval to MEXC supported format
        
        Args:
            interval: Kline interval (1m, 5m, 15m, 1h, etc.)
            
        Returns:
            str: Normalized interval
        """
        # Map common interval formats to MEXC supported formats
        interval_map = {
            '1h': '60m',
            '2h': '120m',
            '4h': '4h',
            '6h': '6h',
            '12h': '12h',
            '1d': '1d',
            '1w': '1w',
            '1M': '1M'
        }
        
        # Return mapped interval or original if not in map
        return interval_map.get(interval, interval)
    
    def get_klines(self, asset=None, interval="1m", limit=100):
        """Get klines (candlestick data) for specified asset
        
        Args:
            asset: Asset to get klines for (default: current asset)
            interval: Kline interval (1m, 5m, 15m, 1h, etc.)
            limit: Number of klines to return
            
        Returns:
            list: Klines data
        """
        target_asset = asset or self.current_asset
        symbol = target_asset.replace('/', '')
        
        # Normalize interval
        normalized_interval = self._normalize_interval(interval)
        
        # Check if interval is supported
        if normalized_interval not in self.supported_intervals:
            logger.warning(f"Unsupported interval: {interval}, using fallback interval '5m'")
            normalized_interval = '5m'
        
        url = f"{self.base_url}/api/v3/klines?symbol={symbol}&interval={normalized_interval}&limit={limit}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Format klines data
                klines = []
                for kline in data:
                    # MEXC API returns klines in the format:
                    # [open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
                    klines.append({
                        "time": int(kline[0]),
                        "open": float(kline[1]),
                        "high": float(kline[2]),
                        "low": float(kline[3]),
                        "close": float(kline[4]),
                        "volume": float(kline[5]),
                        "close_time": int(kline[6]),
                        "quote_volume": float(kline[7]) if len(kline) > 7 else 0.0,
                        "trades": int(kline[8]) if len(kline) > 8 else 0
                    })
                
                # Cache klines by interval
                if interval not in self.cache[target_asset]["klines"]:
                    self.cache[target_asset]["klines"][interval] = {}
                
                self.cache[target_asset]["klines"][interval] = klines
                return klines
            else:
                logger.error(f"Error fetching klines for {target_asset}: {response.status_code}")
                return self.cache[target_asset]["klines"].get(interval, [])
        except Exception as e:
            logger.error(f"Exception fetching klines for {target_asset}: {str(e)}")
            return self.cache[target_asset]["klines"].get(interval, [])
    
    def initialize_websocket(self, asset=None):
        """Initialize WebSocket connection for real-time data
        
        Args:
            asset: Asset to initialize WebSocket for (default: current asset)
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        target_asset = asset or self.current_asset
        symbol = target_asset.replace('/', '')
        
        # Check if WebSocket already initialized for this asset
        if target_asset in self.ws_connections and self.ws_connections[target_asset]:
            return True
        
        # Define WebSocket callbacks
        def on_message(ws, message):
            try:
                data = json.loads(message)
                
                # Handle different message types
                if "channel" in data:
                    channel = data["channel"]
                    
                    if channel == "push.kline":
                        # Handle kline update
                        kline_data = data["data"]
                        interval = kline_data.get("interval", "1m")
                        
                        # Update cache
                        if interval not in self.cache[target_asset]["klines"]:
                            self.cache[target_asset]["klines"][interval] = []
                        
                        # Find and update existing kline or add new one
                        new_kline = {
                            "time": int(kline_data["t"]),
                            "open": float(kline_data["o"]),
                            "high": float(kline_data["h"]),
                            "low": float(kline_data["l"]),
                            "close": float(kline_data["c"]),
                            "volume": float(kline_data["v"]),
                            "close_time": int(kline_data["t"]) + (60 * 1000)  # Assume 1m interval
                        }
                        
                        # Update or add kline
                        updated = False
                        for i, kline in enumerate(self.cache[target_asset]["klines"][interval]):
                            if kline["time"] == new_kline["time"]:
                                self.cache[target_asset]["klines"][interval][i] = new_kline
                                updated = True
                                break
                        
                        if not updated:
                            self.cache[target_asset]["klines"][interval].append(new_kline)
                            # Keep only the latest 'limit' klines
                            self.cache[target_asset]["klines"][interval] = self.cache[target_asset]["klines"][interval][-100:]
                    
                    elif channel == "push.ticker":
                        # Handle ticker update
                        ticker_data = data["data"]
                        self.cache[target_asset]["ticker"] = {
                            "price": float(ticker_data["c"]),
                            "symbol": target_asset,
                            "timestamp": self._get_timestamp()
                        }
                    
                    elif channel == "push.depth":
                        # Handle orderbook update
                        depth_data = data["data"]
                        
                        # Update asks
                        if "asks" in depth_data:
                            asks = []
                            for ask in depth_data["asks"][:20]:
                                price = float(ask[0])
                                amount = float(ask[1])
                                asks.append({"price": price, "amount": amount, "total": price * amount})
                            self.cache[target_asset]["orderbook"]["asks"] = asks
                        
                        # Update bids
                        if "bids" in depth_data:
                            bids = []
                            for bid in depth_data["bids"][:20]:
                                price = float(bid[0])
                                amount = float(bid[1])
                                bids.append({"price": price, "amount": amount, "total": price * amount})
                            self.cache[target_asset]["orderbook"]["bids"] = bids
                    
                    elif channel == "push.deal":
                        # Handle trade update
                        trade_data = data["data"]
                        new_trade = {
                            "price": float(trade_data["p"]),
                            "quantity": float(trade_data["q"]),
                            "time": int(trade_data["t"]),
                            "isBuyerMaker": trade_data["S"] == "sell"
                        }
                        
                        # Add new trade to cache
                        self.cache[target_asset]["trades"].insert(0, new_trade)
                        # Keep only the latest 50 trades
                        self.cache[target_asset]["trades"] = self.cache[target_asset]["trades"][:50]
            
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {str(e)}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error for {target_asset}: {str(error)}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket closed for {target_asset}")
            # Remove from connections
            if target_asset in self.ws_connections:
                self.ws_connections[target_asset] = None
        
        def on_open(ws):
            logger.info(f"WebSocket opened for {target_asset}")
            
            # Subscribe to channels
            subscribe_messages = [
                {"method": "SUBSCRIPTION", "params": [f"spot@public.kline.v3.api@{symbol}@1m"]},
                {"method": "SUBSCRIPTION", "params": [f"spot@public.ticker.v3.api@{symbol}"]},
                {"method": "SUBSCRIPTION", "params": [f"spot@public.depth.v3.api@{symbol}"]},
                {"method": "SUBSCRIPTION", "params": [f"spot@public.deals.v3.api@{symbol}"]}
            ]
            
            for msg in subscribe_messages:
                ws.send(json.dumps(msg))
        
        # Create WebSocket connection
        try:
            ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # Store connection
            self.ws_connections[target_asset] = ws
            
            # Start WebSocket in a separate thread
            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Store thread
            self.ws_threads[target_asset] = ws_thread
            
            logger.info(f"WebSocket initialized for {target_asset}")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing WebSocket for {target_asset}: {str(e)}")
            return False
    
    def close_websocket(self, asset=None):
        """Close WebSocket connection
        
        Args:
            asset: Asset to close WebSocket for (default: current asset)
            
        Returns:
            bool: True if close successful, False otherwise
        """
        target_asset = asset or self.current_asset
        
        if target_asset in self.ws_connections and self.ws_connections[target_asset]:
            try:
                self.ws_connections[target_asset].close()
                self.ws_connections[target_asset] = None
                logger.info(f"WebSocket closed for {target_asset}")
                return True
            except Exception as e:
                logger.error(f"Error closing WebSocket for {target_asset}: {str(e)}")
                return False
        
        return False
    
    def close_all_websockets(self):
        """Close all WebSocket connections
        
        Returns:
            bool: True if all closes successful, False otherwise
        """
        success = True
        for asset in self.ws_connections:
            if self.ws_connections[asset]:
                try:
                    self.ws_connections[asset].close()
                    self.ws_connections[asset] = None
                    logger.info(f"WebSocket closed for {asset}")
                except Exception as e:
                    logger.error(f"Error closing WebSocket for {asset}: {str(e)}")
                    success = False
        
        return success
    
    def fetch_all_data(self, asset=None):
        """Fetch all data for specified asset
        
        Args:
            asset: Asset to fetch data for (default: current asset)
            
        Returns:
            dict: All data for asset
        """
        target_asset = asset or self.current_asset
        
        # Fetch all data
        self.get_ticker(target_asset)
        self.get_orderbook(target_asset)
        self.get_trades(target_asset)
        
        # Fetch klines for all supported intervals
        for interval in self.supported_intervals:
            self.get_klines(target_asset, interval)
        
        return self.cache[target_asset]
    
    def start_background_fetching(self, interval=5):
        """Start background data fetching
        
        Args:
            interval: Fetch interval in seconds
            
        Returns:
            threading.Thread: Background thread
        """
        def fetch_data_periodically():
            while True:
                try:
                    for asset in self.supported_assets:
                        self.fetch_all_data(asset)
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in background fetching: {str(e)}")
                    time.sleep(interval)
        
        # Start background thread
        thread = threading.Thread(target=fetch_data_periodically)
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started background data fetching with interval {interval}s")
        return thread
    
    def get_patterns(self, asset=None):
        """Get detected patterns for specified asset
        
        Args:
            asset: Asset to get patterns for (default: current asset)
            
        Returns:
            list: Detected patterns
        """
        target_asset = asset or self.current_asset
        
        # In a real implementation, this would fetch patterns from a pattern detection service
        # For now, we'll return the cached patterns
        return self.cache[target_asset]["patterns"]
    
    def set_patterns(self, patterns, asset=None):
        """Set detected patterns for specified asset
        
        Args:
            patterns: Detected patterns
            asset: Asset to set patterns for (default: current asset)
            
        Returns:
            bool: True if set successful, False otherwise
        """
        target_asset = asset or self.current_asset
        
        try:
            self.cache[target_asset]["patterns"] = patterns
            return True
        except Exception as e:
            logger.error(f"Error setting patterns for {target_asset}: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize data service
    data_service = MultiAssetDataService()
    
    # Fetch data for BTC/USDC
    ticker = data_service.get_ticker()
    print(f"BTC/USDC Price: {ticker['price']}")
    
    # Switch to ETH/USDC
    data_service.switch_asset("ETH/USDC")
    
    # Fetch data for ETH/USDC
    ticker = data_service.get_ticker()
    print(f"ETH/USDC Price: {ticker['price']}")
    
    # Initialize WebSocket for real-time updates
    data_service.initialize_websocket()
    
    # Keep script running to receive WebSocket updates
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        data_service.close_all_websockets()
        print("WebSockets closed")
