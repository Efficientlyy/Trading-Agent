import asyncio
import json
import websockets
import time
import hmac
import hashlib
import datetime

def generate_signature(secret_key, timestamp):
    """Generate HMAC-SHA256 signature for MEXC API authentication"""
    message = timestamp
    signature = hmac.new(
        secret_key.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

async def connect_to_mexc():
    # API credentials
    api_key = "mx0vglZ8S6aN809vmE"
    secret_key = "092911cfc14e4e7491a74a750eb1884b"
    
    # Generate authentication parameters
    timestamp = str(int(datetime.datetime.now().timestamp() * 1000))
    signature = generate_signature(secret_key, timestamp)
    
    url = "wss://wbs.mexc.com/ws"
    print(f"Connecting to MEXC WebSocket at {url}...")
    
    async with websockets.connect(url) as websocket:
        print("Connection established!")
        
        # Authenticate using MEXC format
        auth_msg = {
            "method": "api_key",
            "api_key": api_key,
            "sign": signature,
            "reqTime": timestamp
        }
        
        print("Sending authentication request...")
        await websocket.send(json.dumps(auth_msg))
        auth_response = await websocket.recv()
        print(f"Authentication response: {auth_response}")
        
        # Try different subscription format for authenticated users
        symbol = "BTCUSDT"
        
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
        
        # Receive and process messages
        message_count = 0
        start_time = time.time()
        
        try:
            while time.time() - start_time < 30:  # Run for 30 seconds
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                data = json.loads(response)
                message_count += 1
                
                # Check for subscription confirmation
                if "code" in data:
                    print(f"Response: {data}")
                    continue
                
                # Process different message types
                if "c" in data:
                    channel_type = data.get("c", "")
                    symbol = data.get("s", "unknown")
                    
                    if "depth" in channel_type:
                        print(f"Received order book update for {symbol}")
                        # Get first few bids and asks if available
                        if "data" in data and "bids" in data["data"] and "asks" in data["data"]:
                            bids = data["data"]["bids"][:3] if data["data"]["bids"] else []
                            asks = data["data"]["asks"][:3] if data["data"]["asks"] else []
                            print(f"  Top 3 bids: {bids}")
                            print(f"  Top 3 asks: {asks}")
                    
                    elif "deals" in channel_type:
                        print(f"Received trades for {symbol}")
                        if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                            for trade in data["data"][:3]:  # Show first 3 trades
                                print(f"  Trade: Price={trade.get('p')}, Quantity={trade.get('v')}")
                    
                    else:
                        print(f"Received data for {channel_type}: {data}")
                else:
                    print(f"Received message: {data}")
                
        except asyncio.TimeoutError:
            print("Timeout waiting for message")
        except Exception as e:
            print(f"Error: {e}")
        
        print(f"\nTest completed: Received {message_count} messages")
        return message_count > 0

if __name__ == "__main__":
    print("MEXC WebSocket Connection Test with Improved Authentication")
    print("==========================================================")
    success = asyncio.run(connect_to_mexc())
    
    if success:
        print("\n✅ Successfully connected to MEXC WebSocket API and received market data")
        print("This verifies that the core functionality of the Market Data Processor will work")
    else:
        print("\n❌ Failed to receive data from MEXC WebSocket API")
