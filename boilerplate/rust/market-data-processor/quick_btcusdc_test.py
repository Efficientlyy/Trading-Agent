import asyncio
import json
import hmac
import hashlib
import datetime
import websockets
import sys
import requests

# New whitelisted API credentials
API_KEY = "mx0vglTbKSqTso4bzf"
API_SECRET = "63c248e899524b4499f13f428ad01e24"

def generate_signature(secret_key, timestamp):
    """Generate HMAC-SHA256 signature for MEXC API authentication"""
    message = timestamp
    signature = hmac.new(
        secret_key.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

async def test_btcusdc_orderbook():
    """Test BTCUSDC order book subscription with new API credentials"""
    print("\n==== MEXC API Connection Test with New Credentials ====")
    
    # Get public IP for verification
    try:
        ip_response = requests.get("https://api.ipify.org")
        public_ip = ip_response.text
        print(f"Public IP: {public_ip} (This IP should be whitelisted)")
    except Exception as e:
        print(f"Unable to determine public IP: {e}")
        public_ip = "unknown"
    
    # WebSocket connection parameters
    ws_url = "wss://wbs.mexc.com/ws"
    symbol = "BTCUSDC"  # Testing specifically with BTCUSDC
    
    print(f"Connecting to MEXC WebSocket at {ws_url}...")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("WebSocket connection established")
            
            # Generate timestamp for authentication
            timestamp = str(int(datetime.datetime.now().timestamp() * 1000))
            signature = generate_signature(API_SECRET, timestamp)
            
            # Create authentication message
            auth_msg = {
                "method": "api_key",
                "api_key": API_KEY,
                "sign": signature,
                "reqTime": timestamp
            }
            
            print("Sending authentication request...")
            await websocket.send(json.dumps(auth_msg))
            
            # Wait for auth response
            auth_response = await websocket.recv()
            print(f"Authentication response: {auth_response}")
            
            # Subscribe to order book for BTCUSDC
            depth_subscription = {
                "method": "SUBSCRIPTION",
                "params": ["spot@public.depth.v3.api@BTCUSDC"]
            }
            
            print(f"Subscribing to order book for {symbol}...")
            await websocket.send(json.dumps(depth_subscription))
            
            # Track message statistics
            snapshot_received = False
            update_count = 0
            start_time = datetime.datetime.now()
            test_duration = 30  # seconds
            
            print(f"Waiting for order book messages (test will run for {test_duration} seconds)...")
            
            # Process messages for the specified duration
            while (datetime.datetime.now() - start_time).total_seconds() < test_duration:
                try:
                    # Set a timeout to periodically check elapsed time
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    
                    # Parse the response
                    data = json.loads(response)
                    
                    # Check if this is a subscription response
                    if "code" in data:
                        print(f"Subscription response: {data}")
                        
                        # Check for "Blocked" error
                        if "Blocked" in str(data.get("msg", "")):
                            print(f"❌ CRITICAL: Subscription blocked for IP {public_ip}")
                            print("The new API credentials and IP whitelisting did not resolve the issue")
                            return False
                    
                    # Check if this is an order book message
                    elif "c" in data and "data" in data and "s" in data and data["s"] == symbol:
                        channel_type = data["c"]
                        
                        if "depth" in channel_type:
                            if not snapshot_received:
                                snapshot_received = True
                                print(f"✅ Received order book snapshot for {symbol}")
                                
                                # Show some details from the snapshot
                                bid_count = len(data["data"]["bids"]) if "bids" in data["data"] else 0
                                ask_count = len(data["data"]["asks"]) if "asks" in data["data"] else 0
                                print(f"Order book contains {bid_count} bids and {ask_count} asks")
                                
                                # Print top 3 bids/asks
                                if bid_count > 0:
                                    print("Top 3 bids:")
                                    for i, bid in enumerate(data["data"]["bids"][:3]):
                                        print(f"  {i+1}: {bid[1]} @ {bid[0]}")
                                
                                if ask_count > 0:
                                    print("Top 3 asks:")
                                    for i, ask in enumerate(data["data"]["asks"][:3]):
                                        print(f"  {i+1}: {ask[1]} @ {ask[0]}")
                            else:
                                update_count += 1
                                if update_count % 5 == 0:  # Log every 5th update to reduce output
                                    print(f"Received update #{update_count} for {symbol}")
                
                except asyncio.TimeoutError:
                    # This is normal - just checking if test duration has elapsed
                    pass
                except Exception as e:
                    print(f"Error processing message: {e}")
            
            # Print test results
            elapsed_seconds = (datetime.datetime.now() - start_time).total_seconds()
            print("\n==== Test Results ====")
            print(f"Test duration: {elapsed_seconds:.2f} seconds")
            print(f"Snapshot received: {snapshot_received}")
            print(f"Updates received: {update_count}")
            
            if snapshot_received or update_count > 0:
                print(f"\n✅ SUCCESS: The new API credentials work for {symbol} order book!")
                print(f"IP {public_ip} is successfully whitelisted")
                return True
            else:
                print(f"\n❌ FAILED: No order book data received for {symbol}")
                print("The API credentials or IP whitelisting may still have issues")
                return False
    
    except Exception as e:
        print(f"Connection error: {e}")
        return False

# Run the test
if __name__ == "__main__":
    result = asyncio.run(test_btcusdc_orderbook())
    sys.exit(0 if result else 1)
