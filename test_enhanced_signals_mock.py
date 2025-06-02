#!/usr/bin/env python
"""
Mock Exchange Client for Testing Enhanced Flash Trading Signals

This module provides a mock exchange client for testing the Trading-Agent system
without requiring real API credentials or internet connectivity.
"""

import time
import logging
import json
import os
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mock_exchange_client")

class MockExchangeClient:
    """Mock exchange client for testing"""
    
    def __init__(self, simulate_errors=False, error_rate=0.2):
        """Initialize mock exchange client
        
        Args:
            simulate_errors: Whether to simulate errors
            error_rate: Rate of simulated errors (0.0-1.0)
        """
        self.simulate_errors = simulate_errors
        self.error_rate = error_rate
        self.retry_count = 0
        self.error_count = 0
        
        # Mock data storage
        self.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT"]
        self.timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        self.mock_data = {}
        
        # Generate mock data for all symbols and timeframes
        for symbol in self.symbols:
            self.mock_data[symbol] = {}
            for timeframe in self.timeframes:
                self.mock_data[symbol][timeframe] = self._generate_mock_ohlcv(timeframe, 1000)
        
        logger.info(f"Initialized MockExchangeClient with simulate_errors={simulate_errors}, error_rate={error_rate}")
    
    def _generate_mock_ohlcv(self, timeframe: str, num_candles: int) -> pd.DataFrame:
        """Generate mock OHLCV data
        
        Args:
            timeframe: Timeframe (e.g., "1m", "5m", "15m", "1h")
            num_candles: Number of candles to generate
            
        Returns:
            pd.DataFrame: Mock OHLCV data
        """
        # Determine time delta based on timeframe
        if timeframe == "1m":
            delta = timedelta(minutes=1)
        elif timeframe == "5m":
            delta = timedelta(minutes=5)
        elif timeframe == "15m":
            delta = timedelta(minutes=15)
        elif timeframe == "1h":
            delta = timedelta(hours=1)
        elif timeframe == "4h":
            delta = timedelta(hours=4)
        elif timeframe == "1d":
            delta = timedelta(days=1)
        else:
            delta = timedelta(minutes=5)
        
        # Generate timestamps
        end_time = datetime.now()
        timestamps = [end_time - delta * i for i in range(num_candles)]
        timestamps.reverse()
        
        # Generate price data with a trend and some volatility
        base_price = 50000.0  # Base price (e.g., BTC price)
        trend = np.linspace(0, 5, num_candles)  # Upward trend
        noise = np.random.normal(0, 1, num_candles)  # Random noise
        
        # Calculate OHLC prices
        close_prices = base_price + trend * 100 + noise * 50
        open_prices = close_prices - np.random.normal(0, 0.5, num_candles) * 50
        high_prices = np.maximum(open_prices, close_prices) + np.random.normal(0.5, 0.5, num_candles) * 50
        low_prices = np.minimum(open_prices, close_prices) - np.random.normal(0.5, 0.5, num_candles) * 50
        
        # Generate volume data
        volume = np.random.normal(10, 2, num_candles) * 10
        
        # Create DataFrame
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume
        })
        
        return df
    
    def _maybe_simulate_error(self) -> bool:
        """Maybe simulate an error based on error rate
        
        Returns:
            bool: True if error should be simulated, False otherwise
        """
        if self.simulate_errors and np.random.random() < self.error_rate:
            self.error_count += 1
            return True
        return False
    
    def record_retry(self) -> None:
        """Record a retry attempt"""
        self.retry_count += 1
    
    def get_exchange_info(self) -> Dict:
        """Get exchange information
        
        Returns:
            dict: Exchange information
        """
        if self._maybe_simulate_error():
            raise Exception("Simulated exchange error")
        
        return {
            "timezone": "UTC",
            "serverTime": int(time.time() * 1000),
            "symbols": [
                {
                    "symbol": symbol.replace("/", ""),
                    "status": "TRADING",
                    "baseAsset": symbol.split("/")[0],
                    "quoteAsset": symbol.split("/")[1],
                    "filters": []
                }
                for symbol in self.symbols
            ]
        }
    
    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> List:
        """Get klines (candlestick data)
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            interval: Timeframe (e.g., "1m", "5m", "15m", "1h")
            limit: Number of candles to return
            
        Returns:
            list: Klines data
        """
        if self._maybe_simulate_error():
            raise Exception("Simulated exchange error")
        
        # Normalize symbol format
        symbol = symbol.replace("USDC", "USDT")  # Treat USDC as USDT for simplicity
        
        # Check if symbol exists
        if symbol not in self.symbols:
            logger.warning(f"Symbol {symbol} not found in mock data, using BTC/USDT")
            symbol = "BTC/USDT"
        
        # Check if interval exists
        if interval not in self.timeframes:
            logger.warning(f"Interval {interval} not found in mock data, using 5m")
            interval = "5m"
        
        # Get mock data
        df = self.mock_data[symbol][interval].copy()
        
        # Limit to requested number of candles
        df = df.tail(limit)
        
        # Convert to list of lists format expected by API
        klines = []
        for _, row in df.iterrows():
            kline = [
                int(row["timestamp"].timestamp() * 1000),  # Open time
                str(row["open"]),                          # Open
                str(row["high"]),                          # High
                str(row["low"]),                           # Low
                str(row["close"]),                         # Close
                str(row["volume"]),                        # Volume
                int(row["timestamp"].timestamp() * 1000) + int(timedelta(minutes=1).total_seconds() * 1000),  # Close time
                "0",                                       # Quote asset volume
                0,                                         # Number of trades
                "0",                                       # Taker buy base asset volume
                "0",                                       # Taker buy quote asset volume
                "0"                                        # Ignore
            ]
            klines.append(kline)
        
        return klines
    
    def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """Get order book
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            limit: Depth limit
            
        Returns:
            dict: Order book data
        """
        if self._maybe_simulate_error():
            raise Exception("Simulated exchange error")
        
        # Normalize symbol format
        symbol = symbol.replace("USDC", "USDT")  # Treat USDC as USDT for simplicity
        
        # Check if symbol exists
        if symbol not in self.symbols:
            logger.warning(f"Symbol {symbol} not found in mock data, using BTC/USDT")
            symbol = "BTC/USDT"
        
        # Get latest price from mock data
        df = self.mock_data[symbol]["1m"].copy()
        latest_price = float(df.iloc[-1]["close"])
        
        # Generate mock order book
        bids = []
        asks = []
        
        # Generate bids (slightly below latest price)
        for i in range(limit):
            price = latest_price * (1 - 0.0001 * (i + 1))
            volume = np.random.normal(1, 0.2) * 10
            bids.append([str(price), str(volume)])
        
        # Generate asks (slightly above latest price)
        for i in range(limit):
            price = latest_price * (1 + 0.0001 * (i + 1))
            volume = np.random.normal(1, 0.2) * 10
            asks.append([str(price), str(volume)])
        
        return {
            "lastUpdateId": int(time.time() * 1000),
            "bids": bids,
            "asks": asks
        }
    
    def get_ticker(self, symbol: str) -> Dict:
        """Get ticker
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            dict: Ticker data
        """
        if self._maybe_simulate_error():
            raise Exception("Simulated exchange error")
        
        # Normalize symbol format
        symbol = symbol.replace("USDC", "USDT")  # Treat USDC as USDT for simplicity
        
        # Check if symbol exists
        if symbol not in self.symbols:
            logger.warning(f"Symbol {symbol} not found in mock data, using BTC/USDT")
            symbol = "BTC/USDT"
        
        # Get latest price from mock data
        df = self.mock_data[symbol]["1m"].copy()
        latest_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        return {
            "symbol": symbol.replace("/", ""),
            "priceChange": str(latest_row["close"] - prev_row["close"]),
            "priceChangePercent": str((latest_row["close"] - prev_row["close"]) / prev_row["close"] * 100),
            "weightedAvgPrice": str(latest_row["close"]),
            "prevClosePrice": str(prev_row["close"]),
            "lastPrice": str(latest_row["close"]),
            "lastQty": str(latest_row["volume"]),
            "bidPrice": str(latest_row["close"] * 0.9999),
            "askPrice": str(latest_row["close"] * 1.0001),
            "openPrice": str(latest_row["open"]),
            "highPrice": str(latest_row["high"]),
            "lowPrice": str(latest_row["low"]),
            "volume": str(latest_row["volume"]),
            "quoteVolume": str(latest_row["volume"] * latest_row["close"]),
            "openTime": int(latest_row["timestamp"].timestamp() * 1000) - 86400000,
            "closeTime": int(latest_row["timestamp"].timestamp() * 1000),
            "firstId": 0,
            "lastId": 0,
            "count": 0
        }
    
    def submit_order(self, order) -> Dict:
        """Submit an order
        
        Args:
            order: Order object
            
        Returns:
            dict: Order response
        """
        if self._maybe_simulate_error():
            raise Exception("Simulated exchange error")
        
        # Generate mock order ID
        order_id = str(uuid.uuid4())
        
        return {
            "id": order_id,
            "clientOrderId": order_id,
            "symbol": order.symbol.replace("/", ""),
            "status": "FILLED",
            "price": str(order.price) if order.price else "0",
            "origQty": str(order.quantity),
            "executedQty": str(order.quantity),
            "type": order.type.value,
            "side": order.side.value,
            "time": int(time.time() * 1000)
        }
    
    def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """Cancel an order
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            dict: Cancel response
        """
        if self._maybe_simulate_error():
            raise Exception("Simulated exchange error")
        
        return {
            "id": order_id,
            "clientOrderId": order_id,
            "symbol": symbol.replace("/", ""),
            "status": "CANCELED",
            "time": int(time.time() * 1000)
        }
    
    def get_account(self) -> Dict:
        """Get account information
        
        Returns:
            dict: Account information
        """
        if self._maybe_simulate_error():
            raise Exception("Simulated exchange error")
        
        return {
            "makerCommission": 10,
            "takerCommission": 10,
            "buyerCommission": 0,
            "sellerCommission": 0,
            "canTrade": True,
            "canWithdraw": True,
            "canDeposit": True,
            "updateTime": int(time.time() * 1000),
            "accountType": "SPOT",
            "balances": [
                {
                    "asset": "BTC",
                    "free": "1.0",
                    "locked": "0.0"
                },
                {
                    "asset": "ETH",
                    "free": "10.0",
                    "locked": "0.0"
                },
                {
                    "asset": "SOL",
                    "free": "100.0",
                    "locked": "0.0"
                },
                {
                    "asset": "USDT",
                    "free": "10000.0",
                    "locked": "0.0"
                },
                {
                    "asset": "USDC",
                    "free": "10000.0",
                    "locked": "0.0"
                }
            ]
        }
    
    def get_open_orders(self, symbol: str = None) -> List:
        """Get open orders
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            list: Open orders
        """
        if self._maybe_simulate_error():
            raise Exception("Simulated exchange error")
        
        # Return empty list for simplicity
        return []
