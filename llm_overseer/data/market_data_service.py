#!/usr/bin/env python
"""
Market Data Service for LLM Strategic Overseer.

This module provides real-time market data for BTC, ETH, and SOL,
connecting to cryptocurrency exchanges and streaming data to the event bus.
"""

import os
import sys
import json
import logging
import asyncio
import time
import traceback
import hmac
import hashlib
import requests
import websockets
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'market_data_service.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import event bus
from ..core.event_bus import EventBus

class MarketDataService:
    """
    Market Data Service for real-time cryptocurrency data.
    
    This class provides real-time market data for BTC, ETH, and SOL,
    connecting to cryptocurrency exchanges and streaming data to the event bus.
    """
    
    def __init__(self, config, event_bus: Optional[EventBus] = None):
        """
        Initialize Market Data Service.
        
        Args:
            config: Configuration object
            event_bus: Event bus instance (optional, will create new if None)
        """
        self.config = config
        
        # Initialize event bus if not provided
        self.event_bus = event_bus if event_bus else EventBus()
        
        # Initialize exchange settings
        self.exchange = self.config.get("market_data.exchange", "mexc")
        self.api_key = self.config.get("market_data.api_key", "")
        self.api_secret = self.config.get("market_data.api_secret", "")
        
        # Initialize mock mode for testing
        self.mock_mode = self.config.get("market_data.mock_mode", True)
        
        # Initialize data settings
        self.update_interval = self.config.get("market_data.update_interval", 1.0)  # seconds
        self.symbols = self.config.get("trading.symbols", ["BTC/USDC", "ETH/USDC", "SOL/USDC"])
        self.timeframes = self.config.get("market_data.timeframes", ["1m", "5m", "15m", "1h", "4h", "1d"])
        
        # Initialize websocket connections
        self.ws_connections = {}
        self.ws_keepalive_tasks = {}
        
        # Initialize connection status
        self.connected = False
        self.running = False
        
        # Initialize data storage
        self.market_data = {}
        self.klines_data = {}
        
        # Initialize data counters
        self.data_counters = {symbol: 0 for symbol in self.symbols}
        
        logger.info(f"Market Data Service initialized with exchange: {self.exchange}, mock_mode: {self.mock_mode}")
    
    async def start(self) -> bool:
        """
        Start Market Data Service.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            logger.info("Market Data Service already running")
            return True
        
        try:
            self.running = True
            
            if self.mock_mode:
                # Start mock data generation
                logger.info("Starting mock data generation")
                asyncio.create_task(self._generate_mock_data())
            else:
                # Connect to exchange
                logger.info(f"Connecting to {self.exchange} exchange")
                success = await self._connect_to_exchange()
                if not success:
                    self.running = False
                    return False
            
            logger.info("Market Data Service started")
            return True
        except Exception as e:
            logger.error(f"Error starting Market Data Service: {e}")
            logger.error(traceback.format_exc())
            self.running = False
            return False
    
    async def stop(self) -> bool:
        """
        Stop Market Data Service.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.running:
            logger.info("Market Data Service not running")
            return True
        
        try:
            self.running = False
            
            if not self.mock_mode:
                # Disconnect from exchange
                logger.info(f"Disconnecting from {self.exchange} exchange")
                await self._disconnect_from_exchange()
            
            logger.info("Market Data Service stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping Market Data Service: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def _connect_to_exchange(self) -> bool:
        """
        Connect to cryptocurrency exchange.
        
        Returns:
            True if connected successfully, False otherwise
        """
        try:
            if self.exchange == "mexc":
                # Connect to MEXC exchange
                for symbol in self.symbols:
                    # Convert symbol format for MEXC (e.g., BTC/USDC -> BTCUSDC)
                    mexc_symbol = symbol.replace("/", "")
                    
                    # Create websocket connection for ticker data
                    ws_url = f"wss://wbs.mexc.com/ws"
                    self.ws_connections[f"{symbol}_ticker"] = await websockets.connect(ws_url)
                    
                    # Subscribe to ticker data
                    subscribe_msg = {
                        "method": "SUBSCRIPTION",
                        "params": [f"spot@public.ticker.v3.api@{mexc_symbol}"]
                    }
                    await self.ws_connections[f"{symbol}_ticker"].send(json.dumps(subscribe_msg))
                    
                    # Create websocket connection for klines data
                    self.ws_connections[f"{symbol}_klines"] = await websockets.connect(ws_url)
                    
                    # Subscribe to klines data for each timeframe
                    for timeframe in self.timeframes:
                        # Convert timeframe format for MEXC (e.g., 1h -> 60)
                        mexc_timeframe = self._convert_timeframe_to_mexc(timeframe)
                        
                        subscribe_msg = {
                            "method": "SUBSCRIPTION",
                            "params": [f"spot@public.kline.v3.api@{mexc_symbol}@{mexc_timeframe}"]
                        }
                        await self.ws_connections[f"{symbol}_klines"].send(json.dumps(subscribe_msg))
                    
                    # Start message handling tasks
                    asyncio.create_task(self._handle_mexc_messages(f"{symbol}_ticker"))
                    asyncio.create_task(self._handle_mexc_messages(f"{symbol}_klines"))
                    
                    # Start keepalive tasks
                    self.ws_keepalive_tasks[f"{symbol}_ticker"] = asyncio.create_task(
                        self._keepalive_mexc_connection(f"{symbol}_ticker")
                    )
                    self.ws_keepalive_tasks[f"{symbol}_klines"] = asyncio.create_task(
                        self._keepalive_mexc_connection(f"{symbol}_klines")
                    )
                
                self.connected = True
                logger.info("Connected to MEXC exchange")
                return True
            else:
                logger.error(f"Unsupported exchange: {self.exchange}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to exchange: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def _disconnect_from_exchange(self) -> bool:
        """
        Disconnect from cryptocurrency exchange.
        
        Returns:
            True if disconnected successfully, False otherwise
        """
        try:
            # Cancel keepalive tasks
            for task_key, task in self.ws_keepalive_tasks.items():
                task.cancel()
            
            # Close websocket connections
            for ws_key, ws in self.ws_connections.items():
                await ws.close()
            
            self.ws_connections = {}
            self.ws_keepalive_tasks = {}
            self.connected = False
            
            logger.info(f"Disconnected from {self.exchange} exchange")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from exchange: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def _handle_mexc_messages(self, connection_key: str) -> None:
        """
        Handle messages from MEXC websocket connection.
        
        Args:
            connection_key: Connection key
        """
        try:
            ws = self.ws_connections[connection_key]
            
            while self.running and not ws.closed:
                try:
                    message = await ws.recv()
                    data = json.loads(message)
                    
                    # Process ticker data
                    if "_ticker" in connection_key and "d" in data:
                        symbol = connection_key.split("_ticker")[0]
                        await self._process_mexc_ticker(symbol, data["d"])
                    
                    # Process klines data
                    elif "_klines" in connection_key and "d" in data:
                        symbol = connection_key.split("_klines")[0]
                        await self._process_mexc_klines(symbol, data["d"])
                    
                except websockets.exceptions.ConnectionClosed:
                    logger.warning(f"MEXC websocket connection closed: {connection_key}")
                    if self.running:
                        # Reconnect
                        await self._reconnect_mexc_websocket(connection_key)
                    break
                except Exception as e:
                    logger.error(f"Error handling MEXC message: {e}")
                    logger.error(traceback.format_exc())
        except Exception as e:
            logger.error(f"Error in MEXC message handler: {e}")
            logger.error(traceback.format_exc())
    
    async def _keepalive_mexc_connection(self, connection_key: str) -> None:
        """
        Keep MEXC websocket connection alive.
        
        Args:
            connection_key: Connection key
        """
        try:
            ws = self.ws_connections[connection_key]
            
            while self.running and not ws.closed:
                try:
                    # Send ping message every 30 seconds
                    ping_msg = {"method": "PING"}
                    await ws.send(json.dumps(ping_msg))
                    await asyncio.sleep(30)
                except websockets.exceptions.ConnectionClosed:
                    logger.warning(f"MEXC websocket connection closed during keepalive: {connection_key}")
                    if self.running:
                        # Reconnect
                        await self._reconnect_mexc_websocket(connection_key)
                    break
                except Exception as e:
                    logger.error(f"Error in MEXC keepalive: {e}")
                    logger.error(traceback.format_exc())
                    await asyncio.sleep(5)  # Wait before retry
        except Exception as e:
            logger.error(f"Error in MEXC keepalive handler: {e}")
            logger.error(traceback.format_exc())
    
    async def _reconnect_mexc_websocket(self, connection_key: str) -> bool:
        """
        Reconnect MEXC websocket connection.
        
        Args:
            connection_key: Connection key
            
        Returns:
            True if reconnected successfully, False otherwise
        """
        try:
            # Close existing connection if not already closed
            if connection_key in self.ws_connections:
                try:
                    await self.ws_connections[connection_key].close()
                except:
                    pass
            
            # Extract symbol and connection type
            if "_ticker" in connection_key:
                symbol = connection_key.split("_ticker")[0]
                connection_type = "ticker"
            elif "_klines" in connection_key:
                symbol = connection_key.split("_klines")[0]
                connection_type = "klines"
            else:
                logger.error(f"Unknown connection key format: {connection_key}")
                return False
            
            # Convert symbol format for MEXC
            mexc_symbol = symbol.replace("/", "")
            
            # Create new websocket connection
            ws_url = f"wss://wbs.mexc.com/ws"
            self.ws_connections[connection_key] = await websockets.connect(ws_url)
            
            # Resubscribe to data
            if connection_type == "ticker":
                subscribe_msg = {
                    "method": "SUBSCRIPTION",
                    "params": [f"spot@public.ticker.v3.api@{mexc_symbol}"]
                }
                await self.ws_connections[connection_key].send(json.dumps(subscribe_msg))
            elif connection_type == "klines":
                for timeframe in self.timeframes:
                    # Convert timeframe format for MEXC
                    mexc_timeframe = self._convert_timeframe_to_mexc(timeframe)
                    
                    subscribe_msg = {
                        "method": "SUBSCRIPTION",
                        "params": [f"spot@public.kline.v3.api@{mexc_symbol}@{mexc_timeframe}"]
                    }
                    await self.ws_connections[connection_key].send(json.dumps(subscribe_msg))
            
            # Restart message handling task
            asyncio.create_task(self._handle_mexc_messages(connection_key))
            
            # Restart keepalive task
            if connection_key in self.ws_keepalive_tasks:
                self.ws_keepalive_tasks[connection_key].cancel()
            
            self.ws_keepalive_tasks[connection_key] = asyncio.create_task(
                self._keepalive_mexc_connection(connection_key)
            )
            
            logger.info(f"Reconnected MEXC websocket: {connection_key}")
            return True
        except Exception as e:
            logger.error(f"Error reconnecting MEXC websocket: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def _process_mexc_ticker(self, symbol: str, data: Dict[str, Any]) -> None:
        """
        Process MEXC ticker data.
        
        Args:
            symbol: Trading symbol
            data: Ticker data
        """
        try:
            # Extract relevant fields
            price = float(data.get("c", 0))
            volume_24h = float(data.get("v", 0))
            high_24h = float(data.get("h", 0))
            low_24h = float(data.get("l", 0))
            change_24h = float(data.get("r", 0))
            
            # Create market data object
            market_data = {
                "success": True,
                "symbol": symbol,
                "price": price,
                "volume_24h": volume_24h,
                "high_24h": high_24h,
                "low_24h": low_24h,
                "change_24h": change_24h,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update data store
            self.market_data[symbol] = market_data
            
            # Increment data counter
            self.data_counters[symbol] += 1
            
            # Publish market data event
            await self.event_bus.publish(
                "trading.market_data",
                market_data,
                "normal"
            )
            
            logger.debug(f"Processed MEXC ticker for {symbol}: {price}")
        except Exception as e:
            logger.error(f"Error processing MEXC ticker: {e}")
            logger.error(traceback.format_exc())
    
    async def _process_mexc_klines(self, symbol: str, data: Dict[str, Any]) -> None:
        """
        Process MEXC klines data.
        
        Args:
            symbol: Trading symbol
            data: Klines data
        """
        try:
            # Extract timeframe
            mexc_timeframe = data.get("i", "1")
            timeframe = self._convert_mexc_to_timeframe(mexc_timeframe)
            
            # Extract kline data
            kline = {
                "timestamp": datetime.fromtimestamp(int(data.get("t", 0)) / 1000).isoformat(),
                "open": float(data.get("o", 0)),
                "high": float(data.get("h", 0)),
                "low": float(data.get("l", 0)),
                "close": float(data.get("c", 0)),
                "volume": float(data.get("v", 0))
            }
            
            # Initialize symbol and timeframe data if not exists
            if symbol not in self.klines_data:
                self.klines_data[symbol] = {}
            
            if timeframe not in self.klines_data[symbol]:
                self.klines_data[symbol][timeframe] = []
            
            # Update klines data
            # Check if we already have this timestamp
            existing_index = None
            for i, existing_kline in enumerate(self.klines_data[symbol][timeframe]):
                if existing_kline["timestamp"] == kline["timestamp"]:
                    existing_index = i
                    break
            
            if existing_index is not None:
                # Update existing kline
                self.klines_data[symbol][timeframe][existing_index] = kline
            else:
                # Add new kline
                self.klines_data[symbol][timeframe].append(kline)
                # Sort by timestamp
                self.klines_data[symbol][timeframe].sort(key=lambda x: x["timestamp"])
                # Limit to 1000 klines
                if len(self.klines_data[symbol][timeframe]) > 1000:
                    self.klines_data[symbol][timeframe] = self.klines_data[symbol][timeframe][-1000:]
            
            # Publish klines data event
            await self.event_bus.publish(
                "trading.klines",
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "klines": self.klines_data[symbol][timeframe],
                    "timestamp": datetime.now().isoformat()
                },
                "normal"
            )
            
            logger.debug(f"Processed MEXC klines for {symbol} {timeframe}")
        except Exception as e:
            logger.error(f"Error processing MEXC klines: {e}")
            logger.error(traceback.format_exc())
    
    def _convert_timeframe_to_mexc(self, timeframe: str) -> str:
        """
        Convert timeframe to MEXC format.
        
        Args:
            timeframe: Timeframe (e.g., 1m, 5m, 15m, 1h, 4h, 1d)
            
        Returns:
            MEXC timeframe (e.g., 1, 5, 15, 60, 240, 1440)
        """
        timeframe_map = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "4h": "240",
            "1d": "1440",
            "1w": "10080"
        }
        
        return timeframe_map.get(timeframe, "1")
    
    def _convert_mexc_to_timeframe(self, mexc_timeframe: str) -> str:
        """
        Convert MEXC timeframe to standard format.
        
        Args:
            mexc_timeframe: MEXC timeframe (e.g., 1, 5, 15, 60, 240, 1440)
            
        Returns:
            Standard timeframe (e.g., 1m, 5m, 15m, 1h, 4h, 1d)
        """
        timeframe_map = {
            "1": "1m",
            "5": "5m",
            "15": "15m",
            "30": "30m",
            "60": "1h",
            "240": "4h",
            "1440": "1d",
            "10080": "1w"
        }
        
        return timeframe_map.get(mexc_timeframe, "1m")
    
    async def _generate_mock_data(self) -> None:
        """Generate mock data for testing."""
        try:
            # Initial price points for each symbol
            base_prices = {
                "BTC/USDC": 106739.83,
                "ETH/USDC": 3456.78,
                "SOL/USDC": 123.45
            }
            
            # Volatility for each symbol
            volatility = {
                "BTC/USDC": 100.0,
                "ETH/USDC": 10.0,
                "SOL/USDC": 1.0
            }
            
            # Initialize klines data for each symbol and timeframe
            for symbol in self.symbols:
                if symbol not in self.klines_data:
                    self.klines_data[symbol] = {}
                
                for timeframe in self.timeframes:
                    if timeframe not in self.klines_data[symbol]:
                        self.klines_data[symbol][timeframe] = []
                        
                        # Generate historical klines
                        now = datetime.now()
                        
                        # Determine time delta based on timeframe
                        if timeframe == "1m":
                            delta = timedelta(minutes=1)
                            num_klines = 200
                        elif timeframe == "5m":
                            delta = timedelta(minutes=5)
                            num_klines = 200
                        elif timeframe == "15m":
                            delta = timedelta(minutes=15)
                            num_klines = 200
                        elif timeframe == "1h":
                            delta = timedelta(hours=1)
                            num_klines = 200
                        elif timeframe == "4h":
                            delta = timedelta(hours=4)
                            num_klines = 200
                        elif timeframe == "1d":
                            delta = timedelta(days=1)
                            num_klines = 100
                        else:
                            delta = timedelta(minutes=1)
                            num_klines = 200
                        
                        # Generate klines
                        price = base_prices[symbol]
                        for i in range(num_klines):
                            timestamp = now - (num_klines - i) * delta
                            
                            # Generate random price movement
                            price_change = (volatility[symbol] * (0.5 - pd.np.random.random())) / 10
                            price += price_change
                            
                            # Generate random high/low/open/close
                            high = price * (1 + pd.np.random.random() * 0.01)
                            low = price * (1 - pd.np.random.random() * 0.01)
                            open_price = price * (1 + (0.5 - pd.np.random.random()) * 0.005)
                            close_price = price * (1 + (0.5 - pd.np.random.random()) * 0.005)
                            
                            # Ensure high is highest and low is lowest
                            high = max(high, open_price, close_price)
                            low = min(low, open_price, close_price)
                            
                            # Generate random volume
                            volume = 100 + pd.np.random.random() * 900
                            
                            # Create kline
                            kline = {
                                "timestamp": timestamp.isoformat(),
                                "open": open_price,
                                "high": high,
                                "low": low,
                                "close": close_price,
                                "volume": volume
                            }
                            
                            self.klines_data[symbol][timeframe].append(kline)
            
            # Generate real-time updates
            while self.running:
                for symbol in self.symbols:
                    # Generate market data
                    price = base_prices[symbol] + (volatility[symbol] * pd.np.sin(time.time() / 100))
                    
                    # Add random noise
                    price += volatility[symbol] * (0.5 - pd.np.random.random()) / 10
                    
                    # Ensure price doesn't go negative
                    price = max(price, 0.01)
                    
                    # Update base price
                    base_prices[symbol] = price
                    
                    # Create market data
                    market_data = {
                        "success": True,
                        "symbol": symbol,
                        "price": price,
                        "volume_24h": 1000 + (pd.np.random.random() * 9000),
                        "high_24h": price * 1.05,
                        "low_24h": price * 0.95,
                        "change_24h": (0.5 - pd.np.random.random()) * 5,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Update data store
                    self.market_data[symbol] = market_data
                    
                    # Increment data counter
                    self.data_counters[symbol] += 1
                    
                    # Publish market data event
                    await self.event_bus.publish(
                        "trading.market_data",
                        market_data,
                        "normal"
                    )
                    
                    # Update klines data for each timeframe
                    now = datetime.now()
                    for timeframe in self.timeframes:
                        # Determine if it's time to update this timeframe
                        update_timeframe = False
                        
                        if timeframe == "1m" and now.second == 0:
                            update_timeframe = True
                        elif timeframe == "5m" and now.minute % 5 == 0 and now.second == 0:
                            update_timeframe = True
                        elif timeframe == "15m" and now.minute % 15 == 0 and now.second == 0:
                            update_timeframe = True
                        elif timeframe == "1h" and now.minute == 0 and now.second == 0:
                            update_timeframe = True
                        elif timeframe == "4h" and now.hour % 4 == 0 and now.minute == 0 and now.second == 0:
                            update_timeframe = True
                        elif timeframe == "1d" and now.hour == 0 and now.minute == 0 and now.second == 0:
                            update_timeframe = True
                        
                        # For testing, force update every 5 seconds
                        update_timeframe = True
                        
                        if update_timeframe:
                            # Generate random high/low/open/close
                            high = price * (1 + pd.np.random.random() * 0.01)
                            low = price * (1 - pd.np.random.random() * 0.01)
                            open_price = price * (1 + (0.5 - pd.np.random.random()) * 0.005)
                            close_price = price
                            
                            # Ensure high is highest and low is lowest
                            high = max(high, open_price, close_price)
                            low = min(low, open_price, close_price)
                            
                            # Generate random volume
                            volume = 100 + pd.np.random.random() * 900
                            
                            # Create kline
                            kline = {
                                "timestamp": now.isoformat(),
                                "open": open_price,
                                "high": high,
                                "low": low,
                                "close": close_price,
                                "volume": volume
                            }
                            
                            # Check if we already have this timestamp
                            existing_index = None
                            for i, existing_kline in enumerate(self.klines_data[symbol][timeframe]):
                                existing_time = datetime.fromisoformat(existing_kline["timestamp"])
                                if existing_time.year == now.year and existing_time.month == now.month and existing_time.day == now.day and existing_time.hour == now.hour:
                                    if timeframe == "1m" and existing_time.minute == now.minute:
                                        existing_index = i
                                        break
                                    elif timeframe == "5m" and existing_time.minute // 5 == now.minute // 5:
                                        existing_index = i
                                        break
                                    elif timeframe == "15m" and existing_time.minute // 15 == now.minute // 15:
                                        existing_index = i
                                        break
                                    elif timeframe == "1h":
                                        existing_index = i
                                        break
                            
                            if existing_index is not None:
                                # Update existing kline
                                self.klines_data[symbol][timeframe][existing_index] = kline
                            else:
                                # Add new kline
                                self.klines_data[symbol][timeframe].append(kline)
                                # Sort by timestamp
                                self.klines_data[symbol][timeframe].sort(key=lambda x: x["timestamp"])
                                # Limit to 1000 klines
                                if len(self.klines_data[symbol][timeframe]) > 1000:
                                    self.klines_data[symbol][timeframe] = self.klines_data[symbol][timeframe][-1000:]
                            
                            # Publish klines data event
                            await self.event_bus.publish(
                                "trading.klines",
                                {
                                    "symbol": symbol,
                                    "timeframe": timeframe,
                                    "klines": self.klines_data[symbol][timeframe],
                                    "timestamp": datetime.now().isoformat()
                                },
                                "normal"
                            )
                
                # Sleep until next update
                await asyncio.sleep(self.update_interval)
        except Exception as e:
            logger.error(f"Error generating mock data: {e}")
            logger.error(traceback.format_exc())
            
            if self.running:
                # Restart mock data generation after a delay
                await asyncio.sleep(5)
                asyncio.create_task(self._generate_mock_data())

# Example usage
async def main():
    from ..config.config import Config
    from ..core.event_bus import EventBus
    
    # Create configuration
    config = Config()
    
    # Create event bus
    event_bus = EventBus()
    
    # Create market data service
    market_data_service = MarketDataService(config, event_bus)
    
    # Start market data service
    await market_data_service.start()
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        # Stop market data service
        await market_data_service.stop()

if __name__ == "__main__":
    asyncio.run(main())
