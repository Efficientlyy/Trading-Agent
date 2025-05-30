import asyncio
import json
import websockets
import aiohttp
import time
import hmac
import hashlib
import datetime
from collections import defaultdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MEXC_Market_Data")

# API credentials
API_KEY = "mx0vglZ8S6aN809vmE"
SECRET_KEY = "092911cfc14e4e7491a74a750eb1884b"

class DataSourceType(Enum):
    WEBSOCKET = "websocket"
    REST = "rest"
    HYBRID = "hybrid"  # Uses WebSocket for trades and REST for order book

class OrderBook:
    """Order book implementation with bid/ask management"""
    def __init__(self, symbol):
        self.symbol = symbol
        self.bids = {}  # price -> quantity
        self.asks = {}  # price -> quantity
        self.last_update_id = 0
        self.last_updated = time.time()
    
    def update_bid(self, price, quantity):
        price = float(price)
        quantity = float(quantity)
        if quantity > 0:
            self.bids[price] = quantity
        else:
            if price in self.bids:
                del self.bids[price]
    
    def update_ask(self, price, quantity):
        price = float(price)
        quantity = float(quantity)
        if quantity > 0:
            self.asks[price] = quantity
        else:
            if price in self.asks:
                del self.asks[price]
    
    def best_bid(self):
        if not self.bids:
            return None
        return max(self.bids.keys())
    
    def best_ask(self):
        if not self.asks:
            return None
        return min(self.asks.keys())
    
    def spread(self):
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        if best_bid is None or best_ask is None:
            return None
        return best_ask - best_bid
    
    def mid_price(self):
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2
    
    def market_depth(self, depth=10):
        """Calculate total volume at specified depth levels"""
        bid_volume = 0
        ask_volume = 0
        
        # Get sorted bids and asks
        sorted_bids = sorted(self.bids.items(), reverse=True)[:depth]
        sorted_asks = sorted(self.asks.items())[:depth]
        
        # Sum bid and ask volumes
        for price, qty in sorted_bids:
            bid_volume += qty
            
        for price, qty in sorted_asks:
            ask_volume += qty
            
        return bid_volume, ask_volume
    
    def calculate_imbalance(self, depth=10):
        """Calculate order book imbalance - ratio of bid volume to total volume"""
        bid_volume, ask_volume = self.market_depth(depth)
        total_volume = bid_volume + ask_volume
        
        if total_volume == 0:
            return 0
            
        # Calculate imbalance between -1 and 1
        # Positive means more bids (buying pressure)
        # Negative means more asks (selling pressure)
        return (bid_volume - ask_volume) / total_volume
    
    def is_stale(self, max_age_seconds=30):
        """Check if the order book data is stale"""
        return (time.time() - self.last_updated) > max_age_seconds
    
    def __str__(self):
        return f"OrderBook({self.symbol}) with {len(self.bids)} bids and {len(self.asks)} asks"


class Trade:
    """Represents a single trade"""
    def __init__(self, trade_id, symbol, price, quantity, timestamp, is_buyer_maker):
        self.id = trade_id
        self.symbol = symbol
        self.price = float(price)
        self.quantity = float(quantity)
        self.timestamp = int(timestamp)
        self.is_buyer_maker = is_buyer_maker
    
    def __str__(self):
        side = "SELL" if self.is_buyer_maker else "BUY"
        return f"Trade({self.id}) {side} {self.quantity} {self.symbol} @ {self.price}"


class MarketDataProcessor:
    """Flexible market data processor with WebSocket and REST fallback"""
    def __init__(self, data_source_type=DataSourceType.HYBRID):
        self.data_source_type = data_source_type
        self.order_books = {}  # symbol -> OrderBook
        self.recent_trades = defaultdict(list)  # symbol -> list of Trade objects
        self.websocket = None
        self.rest_refresh_interval = 5  # seconds between REST API calls for order book
        self.ws_url = "wss://wbs.mexc.com/ws"
        self.rest_url = "https://api.mexc.com"
        self.running = False
        self.initialized = {}  # symbol -> bool
        self.last_rest_update = {}  # symbol -> timestamp
    
    async def start(self, symbols):
        """Start market data processing for specified symbols"""
        self.running = True
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        
        # Initialize for each symbol
        for symbol in self.symbols:
            self.order_books[symbol] = OrderBook(symbol)
            self.recent_trades[symbol] = []
            self.initialized[symbol] = False
            self.last_rest_update[symbol] = 0
        
        if self.data_source_type in [DataSourceType.WEBSOCKET, DataSourceType.HYBRID]:
            # Start WebSocket connection
            ws_task = asyncio.create_task(self.run_websocket())
        
        if self.data_source_type in [DataSourceType.REST, DataSourceType.HYBRID]:
            # Start REST polling
            rest_task = asyncio.create_task(self.run_rest_polling())
        
        # Keep the tasks running
        if self.data_source_type == DataSourceType.WEBSOCKET:
            await ws_task
        elif self.data_source_type == DataSourceType.REST:
            await rest_task
        else:  # HYBRID
            await asyncio.gather(ws_task, rest_task)
    
    async def run_websocket(self):
        """Run the WebSocket connection and handle messages"""
        try:
            logger.info(f"Connecting to MEXC WebSocket at {self.ws_url}")
            async with websockets.connect(self.ws_url) as websocket:
                self.websocket = websocket
                logger.info("WebSocket connection established")
                
                # Subscribe to trades for all symbols
                for symbol in self.symbols:
                    # Try order book subscription (might be blocked)
                    depth_subscription = {
                        "method": "SUBSCRIPTION",
                        "params": [f"spot@public.depth.v3.api@{symbol}"]
                    }
                    await websocket.send(json.dumps(depth_subscription))
                    logger.info(f"Sent order book subscription for {symbol}")
                    
                    # Trade subscription
                    trades_subscription = {
                        "method": "SUBSCRIPTION",
                        "params": [f"spot@public.deals.v3.api@{symbol}"]
                    }
                    await websocket.send(json.dumps(trades_subscription))
                    logger.info(f"Sent trade subscription for {symbol}")
                
                # Process messages
                while self.running:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=10)
                        await self.process_websocket_message(response)
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        try:
                            await websocket.ping()
                        except:
                            logger.error("WebSocket ping failed, reconnecting...")
                            break
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        
        # If we get here, the connection was closed or failed
        # Wait a bit before reconnecting
        if self.running:
            logger.info("WebSocket disconnected, reconnecting in 5 seconds...")
            await asyncio.sleep(5)
            asyncio.create_task(self.run_websocket())
    
    async def process_websocket_message(self, message):
        """Process a message received from the WebSocket"""
        try:
            data = json.loads(message)
            
            # Check for subscription response
            if "code" in data:
                if "Blocked" in data.get("msg", ""):
                    logger.warning(f"Subscription blocked: {data}")
                    # Mark this subscription as needing REST fallback
                    if "depth" in data.get("msg", ""):
                        logger.info("Will use REST API fallback for order book data")
                else:
                    logger.info(f"Subscription response: {data}")
                return
            
            # Process market data
            if "c" in data and "s" in data:
                channel = data["c"]
                symbol = data["s"]
                
                if symbol in self.symbols:
                    if "depth" in channel and "data" in data:
                        # Order book update
                        await self.process_orderbook_ws_update(symbol, data["data"])
                    
                    elif "deals" in channel and "data" in data:
                        # Trade update
                        await self.process_trades_update(symbol, data["data"])
        except Exception as e:
            logger.error(f"Error parsing WebSocket message: {e}")
    
    async def process_orderbook_ws_update(self, symbol, data):
        """Process order book update from WebSocket"""
        try:
            order_book = self.order_books[symbol]
            
            # Check if this is the first update (snapshot)
            is_snapshot = not self.initialized.get(symbol, False)
            
            if is_snapshot:
                logger.info(f"Processing order book snapshot for {symbol}")
                order_book = OrderBook(symbol)
                self.initialized[symbol] = True
            
            # Process bids
            if "bids" in data and isinstance(data["bids"], list):
                for bid in data["bids"]:
                    if len(bid) >= 2:
                        order_book.update_bid(bid[0], bid[1])
            
            # Process asks
            if "asks" in data and isinstance(data["asks"], list):
                for ask in data["asks"]:
                    if len(ask) >= 2:
                        order_book.update_ask(ask[0], ask[1])
            
            # Update last update id and timestamp
            if "lastUpdateId" in data:
                order_book.last_update_id = data["lastUpdateId"]
            
            order_book.last_updated = time.time()
            self.order_books[symbol] = order_book
            
            if is_snapshot or (order_book.last_update_id % 10 == 0):
                self.log_orderbook_metrics(symbol)
        except Exception as e:
            logger.error(f"Error processing order book update: {e}")
    
    async def process_trades_update(self, symbol, trades_data):
        """Process trades update from WebSocket"""
        try:
            if not isinstance(trades_data, list):
                return
            
            for trade_data in trades_data:
                trade_id = trade_data.get("i", 0)
                price = trade_data.get("p", 0)
                quantity = trade_data.get("v", 0)
                timestamp = trade_data.get("t", int(time.time() * 1000))
                is_buyer_maker = trade_data.get("m", False)
                
                trade = Trade(trade_id, symbol, price, quantity, timestamp, is_buyer_maker)
                
                # Add to recent trades, keeping only the most recent 100
                self.recent_trades[symbol].append(trade)
                if len(self.recent_trades[symbol]) > 100:
                    self.recent_trades[symbol] = self.recent_trades[symbol][-100:]
                
                # Log the trade
                if len(self.recent_trades[symbol]) % 5 == 1:  # Log every 5th trade to reduce output
                    logger.info(f"Trade: {trade}")
        except Exception as e:
            logger.error(f"Error processing trades update: {e}")
    
    async def run_rest_polling(self):
        """Poll the REST API for order book data"""
        while self.running:
            for symbol in self.symbols:
                # Check if we need to update via REST
                # Only update if: 
                # 1. We're using REST as the primary source for order books, or
                # 2. The order book is stale or not initialized via WebSocket
                if (self.data_source_type == DataSourceType.REST or 
                    (self.data_source_type == DataSourceType.HYBRID and 
                     (time.time() - self.last_rest_update.get(symbol, 0) > self.rest_refresh_interval))):
                    try:
                        await self.fetch_orderbook_rest(symbol)
                    except Exception as e:
                        logger.error(f"Error fetching order book via REST: {e}")
                
                # Also fetch recent trades if needed
                if self.data_source_type == DataSourceType.REST:
                    try:
                        await self.fetch_trades_rest(symbol)
                    except Exception as e:
                        logger.error(f"Error fetching trades via REST: {e}")
            
            # Wait before next polling cycle
            await asyncio.sleep(self.rest_refresh_interval)
    
    async def fetch_orderbook_rest(self, symbol, limit=100):
        """Fetch order book data via REST API"""
        url = f"{self.rest_url}/api/v3/depth?symbol={symbol}&limit={limit}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"User-Agent": "MEXC-Market-Data-Processor"}) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    order_book = OrderBook(symbol)
                    
                    # Process bids
                    if "bids" in data and isinstance(data["bids"], list):
                        for bid in data["bids"]:
                            if len(bid) >= 2:
                                order_book.update_bid(bid[0], bid[1])
                    
                    # Process asks
                    if "asks" in data and isinstance(data["asks"], list):
                        for ask in data["asks"]:
                            if len(ask) >= 2:
                                order_book.update_ask(ask[0], ask[1])
                    
                    # Update timestamp
                    order_book.last_updated = time.time()
                    self.last_rest_update[symbol] = time.time()
                    
                    # Only update if we don't have WebSocket data or if REST data is newer
                    if symbol not in self.order_books or order_book.last_updated > self.order_books[symbol].last_updated:
                        self.order_books[symbol] = order_book
                        logger.info(f"Updated order book for {symbol} via REST API")
                        self.log_orderbook_metrics(symbol)
                else:
                    logger.error(f"Failed to fetch order book: HTTP {response.status}")
    
    async def fetch_trades_rest(self, symbol, limit=100):
        """Fetch recent trades via REST API"""
        url = f"{self.rest_url}/api/v3/trades?symbol={symbol}&limit={limit}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"User-Agent": "MEXC-Market-Data-Processor"}) as response:
                if response.status == 200:
                    trades_data = await response.json()
                    
                    new_trades = []
                    for trade_data in trades_data:
                        trade_id = trade_data.get("id", 0)
                        price = trade_data.get("price", 0)
                        quantity = trade_data.get("qty", 0)
                        timestamp = trade_data.get("time", int(time.time() * 1000))
                        is_buyer_maker = trade_data.get("isBuyerMaker", False)
                        
                        trade = Trade(trade_id, symbol, price, quantity, timestamp, is_buyer_maker)
                        new_trades.append(trade)
                    
                    # Add new trades to recent trades
                    existing_ids = {t.id for t in self.recent_trades[symbol]}
                    for trade in new_trades:
                        if trade.id not in existing_ids:
                            self.recent_trades[symbol].append(trade)
                    
                    # Sort by timestamp and keep only the most recent 100
                    self.recent_trades[symbol] = sorted(
                        self.recent_trades[symbol], 
                        key=lambda t: t.timestamp, 
                        reverse=True
                    )[:100]
                    
                    logger.info(f"Fetched {len(new_trades)} trades for {symbol} via REST API")
                else:
                    logger.error(f"Failed to fetch trades: HTTP {response.status}")
    
    def log_orderbook_metrics(self, symbol):
        """Log order book metrics for a symbol"""
        order_book = self.order_books.get(symbol)
        if not order_book:
            return
        
        logger.info(f"Order book for {symbol}: {len(order_book.bids)} bids, {len(order_book.asks)} asks")
        
        if order_book.spread() is not None:
            logger.info(f"Spread: {order_book.spread()}")
        
        if order_book.mid_price() is not None:
            logger.info(f"Mid price: {order_book.mid_price()}")
        
        imbalance = order_book.calculate_imbalance(10)
        logger.info(f"Order book imbalance (depth 10): {imbalance:.4f}")
        
        if imbalance > 0.2:
            logger.info("Significant buying pressure detected")
        elif imbalance < -0.2:
            logger.info("Significant selling pressure detected")
    
    def get_order_book(self, symbol):
        """Get the current order book for a symbol"""
        return self.order_books.get(symbol)
    
    def get_recent_trades(self, symbol, limit=20):
        """Get recent trades for a symbol"""
        trades = self.recent_trades.get(symbol, [])
        return trades[:limit]
    
    async def stop(self):
        """Stop market data processing"""
        self.running = False
        if self.websocket:
            await self.websocket.close()


async def main():
    # Test with different data source types
    for data_source in [DataSourceType.HYBRID, DataSourceType.REST]:
        logger.info(f"\n\nTesting with {data_source.value} data source\n")
        
        processor = MarketDataProcessor(data_source)
        symbols = ["BTCUSDT"]
        
        # Start the processor
        processor_task = asyncio.create_task(processor.start(symbols))
        
        # Run for 30 seconds
        for i in range(6):
            await asyncio.sleep(5)
            
            # Print current state
            for symbol in symbols:
                order_book = processor.get_order_book(symbol)
                trades = processor.get_recent_trades(symbol, 3)
                
                logger.info(f"\nCurrent state for {symbol}:")
                if order_book:
                    logger.info(f"Order book: {len(order_book.bids)} bids, {len(order_book.asks)} asks")
                    if order_book.mid_price():
                        logger.info(f"Mid price: {order_book.mid_price()}")
                
                logger.info(f"Recent trades: {len(trades)}")
                for trade in trades[:3]:
                    logger.info(f"  {trade}")
        
        # Stop the processor
        await processor.stop()
        
        # Wait for processor to finish
        try:
            await asyncio.wait_for(processor_task, timeout=5)
        except asyncio.TimeoutError:
            logger.warning("Processor task did not complete in time")


if __name__ == "__main__":
    print("MEXC Market Data Processor - Flexible Implementation")
    print("===================================================")
    asyncio.run(main())
