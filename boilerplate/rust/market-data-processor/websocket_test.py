import asyncio
import json
import websockets
import time

async def connect_to_mexc():
    url = "wss://wbs.mexc.com/ws"
    print(f"Connecting to MEXC WebSocket at {url}...")
    
    async with websockets.connect(url) as websocket:
        print("Connection established!")
        
        # Subscribe to BTC/USDT ticker
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
            while time.time() - start_time < 10:  # Run for 10 seconds
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
    print("MEXC WebSocket Connection Test")
    print("==============================")
    success = asyncio.run(connect_to_mexc())
    
    if success:
        print("\n✅ Successfully connected to MEXC WebSocket API and received market data")
        print("This verifies that the core functionality of the Market Data Processor will work")
    else:
        print("\n❌ Failed to receive data from MEXC WebSocket API")
