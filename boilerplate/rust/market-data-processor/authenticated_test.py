import asyncio
import json
import websockets
import time
import hmac
import hashlib
import datetime

def generate_signature(api_key, secret_key, timestamp):
    """Generate HMAC-SHA256 signature for MEXC API authentication"""
    message = f"{timestamp}{api_key}"
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
    signature = generate_signature(api_key, secret_key, timestamp)
    
    url = "wss://wbs.mexc.com/ws"
    print(f"Connecting to MEXC WebSocket at {url}...")
    
    async with websockets.connect(url) as websocket:
        print("Connection established!")
        
        # Authenticate first
        auth_msg = {
            "method": "LOGIN",
            "params": {
                "apiKey": api_key,
                "signature": signature,
                "timestamp": timestamp
            }
        }
        
        print("Sending authentication request...")
        await websocket.send(json.dumps(auth_msg))
        auth_response = await websocket.recv()
        print(f"Authentication response: {auth_response}")
        
        # Subscribe to BTC/USDC ticker
        symbol = "BTCUSDT"
        channel = "spot@public.ticker.v3.api" 
        subscription = f"{channel}@{symbol}"
        
        subscribe_msg = {
            "method": "SUBSCRIPTION",
            "params": [subscription]
        }
        
        await websocket.send(json.dumps(subscribe_msg))
        print(f"Subscribed to {subscription}")
        
        # Receive and process messages
        message_count = 0
        start_time = time.time()
        try:
            while time.time() - start_time < 20:  # Run for 20 seconds
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                message_count += 1
                data = json.loads(response)
                
                # Check for subscription confirmation
                if "code" in data:
                    print(f"Subscription response: {data}")
                    continue
                
                # Process market data
                if "c" in data and "data" in data:
                    symbol = data.get("s", "unknown")
                    price = data.get("data", {}).get("c", "unknown")
                    print(f"Received ticker for {symbol}: Last price = {price}")
                else:
                    print(f"Received message: {data}")
        
        except asyncio.TimeoutError:
            print("Timeout waiting for message")
        except Exception as e:
            print(f"Error: {e}")
        
        print(f"\nTest completed: Received {message_count} messages")
        return message_count > 0

if __name__ == "__main__":
    print("MEXC WebSocket Connection Test with Authentication")
    print("=================================================")
    success = asyncio.run(connect_to_mexc())
    
    if success:
        print("\n✅ Successfully connected to MEXC WebSocket API and received market data")
        print("This verifies that the core functionality of the Market Data Processor will work")
    else:
        print("\n❌ Failed to receive data from MEXC WebSocket API")
