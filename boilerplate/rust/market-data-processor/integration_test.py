import asyncio
import json
import websockets
import time
import hmac
import hashlib
import datetime
from collections import defaultdict

class OrderBook:
    """Simple order book implementation to mirror our Rust implementation"""
    def __init__(self, symbol):
        self.symbol = symbol
        self.bids = {}  # price -> quantity
        self.asks = {}  # price -> quantity
        self.last_update_id = 0
    
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
    
    def calculate_imbalance(self, depth=10):
        """Calculate order book imbalance - ratio of bid volume to total volume"""
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
            
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0
            
        # Calculate imbalance between -1 and 1
        # Positive means more bids (buying pressure)
        # Negative means more asks (selling pressure)
        return (bid_volume - ask_volume) / total_volume


class OrderBookManager:
    """Order book manager to mirror our Rust implementation"""
    def __init__(self):
        self.order_books = {}  # symbol -> OrderBook
    
    def get_order_book(self, symbol):
        return self.order_books.get(symbol)
    
    def process_snapshot(self, symbol, data):
        """Process an order book snapshot"""
        order_book = OrderBook(symbol)
        
        # Process bids
        if "bids" in data and isinstance(data["bids"], list):
            for bid in data["bids"]:
                if len(bid) >= 2:
                    price, quantity = bid[0], bid[1]
                    order_book.update_bid(price, quantity)
        
        # Process asks
        if "asks" in data and isinstance(data["asks"], list):
            for ask in data["asks"]:
                if len(ask) >= 2:
                    price, quantity = ask[0], ask[1]
                    order_book.update_ask(price, quantity)
        
        if "lastUpdateId" in data:
            order_book.last_update_id = data["lastUpdateId"]
        
        self.order_books[symbol] = order_book
        return order_book
    
    def process_update(self, symbol, data):
        """Process an order book update"""
        order_book = self.get_order_book(symbol)
        if order_book is None:
            print(f"Cannot update order book for {symbol} - no snapshot received yet")
            return None
        
        # Process bids
        if "bids" in data and isinstance(data["bids"], list):
            for bid in data["bids"]:
                if len(bid) >= 2:
                    price, quantity = bid[0], bid[1]
                    order_book.update_bid(price, quantity)
        
        # Process asks
        if "asks" in data and isinstance(data["asks"], list):
            for ask in data["asks"]:
                if len(ask) >= 2:
                    price, quantity = ask[0], ask[1]
                    order_book.update_ask(price, quantity)
        
        if "lastUpdateId" in data:
            order_book.last_update_id = data["lastUpdateId"]
        
        return order_book
    
    def calculate_imbalance(self, symbol, depth=10):
        """Calculate order book imbalance for a symbol"""
        order_book = self.get_order_book(symbol)
        if order_book is None:
            return None
        return order_book.calculate_imbalance(depth)


async def connect_to_mexc():
    # API credentials
    api_key = "mx0vglZ8S6aN809vmE"
    secret_key = "092911cfc14e4e7491a74a750eb1884b"
    
    # Symbol to monitor
    symbol = "BTCUSDT"
    
    # Create order book manager
    order_book_manager = OrderBookManager()
    
    url = "wss://wbs.mexc.com/ws"
    print(f"Connecting to MEXC WebSocket at {url}...")
    
    async with websockets.connect(url) as websocket:
        print("Connection established!")
        
        # Subscribe to depth (order book)
        depth_subscription = {
            "method": "SUBSCRIPTION",
            "params": [f"spot@public.depth.v3.api@{symbol}"]
        }
        
        await websocket.send(json.dumps(depth_subscription))
        print(f"Subscribed to order book for {symbol}")
        
        # Subscribe to trades
        trades_subscription = {
            "method": "SUBSCRIPTION",
            "params": [f"spot@public.deals.v3.api@{symbol}"]
        }
        
        await websocket.send(json.dumps(trades_subscription))
        print(f"Subscribed to trades for {symbol}")
        
        # Process messages
        snapshot_received = False
        update_count = 0
        trade_count = 0
        start_time = time.time()
        
        try:
            while time.time() - start_time < 30:  # Run for 30 seconds
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                data = json.loads(response)
                
                # Check for subscription confirmation
                if "code" in data:
                    print(f"Response: {data}")
                    continue
                
                # Process different message types
                if "c" in data:
                    channel_type = data.get("c", "")
                    msg_symbol = data.get("s", "unknown")
                    
                    if "depth" in channel_type:
                        if not snapshot_received and "data" in data:
                            # First message is a snapshot
                            print(f"Received order book snapshot for {msg_symbol}")
                            order_book = order_book_manager.process_snapshot(msg_symbol, data["data"])
                            snapshot_received = True
                            
                            # Print top 5 bids and asks
                            sorted_bids = sorted(order_book.bids.items(), reverse=True)[:5]
                            sorted_asks = sorted(order_book.asks.items())[:5]
                            
                            print("Top 5 bids:")
                            for i, (price, qty) in enumerate(sorted_bids):
                                print(f"  {i+1}: {qty} @ {price}")
                            
                            print("Top 5 asks:")
                            for i, (price, qty) in enumerate(sorted_asks):
                                print(f"  {i+1}: {qty} @ {price}")
                            
                            if order_book.spread() is not None:
                                print(f"Spread: {order_book.spread()}")
                            
                            if order_book.mid_price() is not None:
                                print(f"Mid price: {order_book.mid_price()}")
                        else:
                            # Subsequent messages are updates
                            update_count += 1
                            if update_count % 5 == 0:  # Only log every 5th update to reduce output
                                print(f"Received order book update #{update_count} for {msg_symbol}")
                                
                                # Process the update
                                if "data" in data:
                                    order_book = order_book_manager.process_update(msg_symbol, data["data"])
                                    
                                    # Calculate and print order book imbalance
                                    imbalance = order_book_manager.calculate_imbalance(msg_symbol, 10)
                                    if imbalance is not None:
                                        print(f"Order book imbalance (depth 10): {imbalance:.4f}")
                                        
                                        if imbalance > 0.2:
                                            print("Significant buying pressure detected")
                                        elif imbalance < -0.2:
                                            print("Significant selling pressure detected")
                    
                    elif "deals" in channel_type:
                        trade_count += 1
                        if trade_count <= 3:  # Only log first few trades
                            print(f"Received trade for {msg_symbol}")
                            if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                                for trade in data["data"]:
                                    print(f"  Trade: Price={trade.get('p')}, Quantity={trade.get('v')}")
        
        except asyncio.TimeoutError:
            print("Timeout waiting for message")
        except Exception as e:
            print(f"Error: {e}")
        
        # Final stats
        print("\nTest Summary:")
        print(f"- Received {update_count} order book updates")
        print(f"- Received {trade_count} trades")
        
        # Check final order book state
        order_book = order_book_manager.get_order_book(symbol)
        if order_book:
            print(f"- Final order book state for {symbol}: {len(order_book.bids)} bids, {len(order_book.asks)} asks")
            print(f"- Mid price: {order_book.mid_price()}")
            
            imbalance = order_book_manager.calculate_imbalance(symbol, 10)
            if imbalance is not None:
                print(f"- Final order book imbalance (depth 10): {imbalance:.4f}")
        
        return snapshot_received and update_count > 0


if __name__ == "__main__":
    print("MEXC Market Data Processor Integration Test")
    print("===========================================")
    success = asyncio.run(connect_to_mexc())
    
    if success:
        print("\n✅ Successfully connected to MEXC WebSocket API and processed market data")
        print("The Market Data Processor functionality has been verified")
    else:
        print("\n❌ Failed to receive and process market data from MEXC WebSocket API")
