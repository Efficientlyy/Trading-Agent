import requests
import hmac
import hashlib
import time
import json
from datetime import datetime
import sys

# New API credentials
API_KEY = "mx0vglTbKSqTso4bzf"
API_SECRET = "63c248e899524b4499f13f428ad01e24"

def get_signature(api_secret, params_str):
    """Generate HMAC-SHA256 signature for MEXC API authentication"""
    signature = hmac.new(
        api_secret.encode('utf-8'),
        params_str.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

def timestamp():
    """Get current timestamp in milliseconds"""
    return int(time.time() * 1000)

class ApiPerformanceTracker:
    """Track API performance and rate limits"""
    def __init__(self):
        self.requests = []
        self.errors = []
        self.latencies = []
    
    def add_request(self, endpoint, status_code, latency_ms):
        """Record a request with its latency"""
        self.requests.append({
            'endpoint': endpoint,
            'status_code': status_code,
            'timestamp': timestamp(),
            'latency_ms': latency_ms
        })
        self.latencies.append(latency_ms)
    
    def add_error(self, endpoint, error_msg):
        """Record an error"""
        self.errors.append({
            'endpoint': endpoint,
            'error': error_msg,
            'timestamp': timestamp()
        })
    
    def print_summary(self):
        """Print API performance summary"""
        print("\n==== API Performance Summary ====")
        print(f"Total requests: {len(self.requests)}")
        print(f"Total errors: {len(self.errors)}")
        
        if self.latencies:
            avg_latency = sum(self.latencies) / len(self.latencies)
            min_latency = min(self.latencies)
            max_latency = max(self.latencies)
            print(f"Latency (ms): Avg={avg_latency:.2f}, Min={min_latency:.2f}, Max={max_latency:.2f}")
        
        # Calculate requests per minute
        if self.requests:
            start_time = min(req['timestamp'] for req in self.requests)
            end_time = max(req['timestamp'] for req in self.requests)
            duration_mins = (end_time - start_time) / 60000  # Convert ms to minutes
            if duration_mins > 0:
                rpm = len(self.requests) / duration_mins
                print(f"Request rate: {rpm:.2f} requests per minute")

class MexcRestApiTester:
    """Test MEXC REST API endpoints"""
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.mexc.com"
        self.tracker = ApiPerformanceTracker()
    
    def make_request(self, endpoint, method="GET", params=None, signed=False):
        """Make a request to MEXC REST API"""
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MEXC-APIKEY": self.api_key}
        
        # Add timestamp for signed requests
        if signed and params is None:
            params = {}
        
        if signed:
            params['timestamp'] = timestamp()
            # Convert params to query string
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            # Generate signature
            signature = get_signature(self.api_secret, query_string)
            params['signature'] = signature
        
        start_time = time.time()
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=params)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Record request
            self.tracker.add_request(endpoint, response.status_code, latency_ms)
            
            # Parse and return response
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                self.tracker.add_error(endpoint, error_msg)
                return {"error": error_msg}
                
        except Exception as e:
            # Calculate latency even for errors
            latency_ms = (time.time() - start_time) * 1000
            
            # Record error
            error_msg = str(e)
            self.tracker.add_error(endpoint, error_msg)
            
            return {"error": error_msg}
    
    def test_public_endpoints(self):
        """Test public API endpoints"""
        print("\n==== Testing Public API Endpoints ====")
        
        # Test order book endpoint
        print("\nTesting Order Book Endpoint (BTCUSDC)...")
        result = self.make_request("/api/v3/depth", params={"symbol": "BTCUSDC", "limit": 10})
        self.print_result("Order Book", result)
        
        # Test ticker endpoint
        print("\nTesting Ticker Endpoint (BTCUSDC)...")
        result = self.make_request("/api/v3/ticker/24hr", params={"symbol": "BTCUSDC"})
        self.print_result("Ticker", result)
        
        # Test recent trades endpoint
        print("\nTesting Recent Trades Endpoint (BTCUSDC)...")
        result = self.make_request("/api/v3/trades", params={"symbol": "BTCUSDC", "limit": 10})
        self.print_result("Recent Trades", result)
        
        # Test kline/candlestick endpoint
        print("\nTesting Kline/Candlestick Endpoint (BTCUSDC)...")
        result = self.make_request("/api/v3/klines", params={
            "symbol": "BTCUSDC", 
            "interval": "1m",
            "limit": 5
        })
        self.print_result("Klines", result)
    
    def test_signed_endpoints(self):
        """Test signed API endpoints"""
        print("\n==== Testing Signed API Endpoints ====")
        
        # Test account information endpoint
        print("\nTesting Account Information Endpoint...")
        result = self.make_request("/api/v3/account", signed=True)
        self.print_result("Account Information", result)
        
        # Don't test actual order placement
        print("\nSkipping order placement test to avoid actual trades")
    
    def measure_polling_performance(self, symbol="BTCUSDC", iterations=10, delay_ms=1000):
        """Measure performance of frequent polling for order book data"""
        print(f"\n==== Testing Order Book Polling Performance ({iterations} iterations, {delay_ms}ms delay) ====")
        
        changes_detected = 0
        last_data = None
        
        for i in range(iterations):
            print(f"Polling iteration {i+1}/{iterations}...")
            start_time = time.time()
            
            # Get order book data
            result = self.make_request("/api/v3/depth", params={"symbol": symbol, "limit": 100})
            
            # Check for changes
            if last_data is not None:
                # Simple check: have the asks or bids changed?
                if (result.get('asks', []) != last_data.get('asks', []) or 
                    result.get('bids', []) != last_data.get('bids', [])):
                    changes_detected += 1
            
            last_data = result
            
            # Calculate time spent and sleep for the remainder of the delay
            elapsed_ms = (time.time() - start_time) * 1000
            sleep_ms = max(0, delay_ms - elapsed_ms)
            
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000)  # Convert ms to seconds
        
        print(f"\nPolling complete. Changes detected in {changes_detected} of {iterations} polls.")
        print(f"Change rate: {(changes_detected / iterations) * 100:.2f}%")
    
    def print_result(self, endpoint_name, result):
        """Print API result summary"""
        if "error" in result:
            print(f"❌ {endpoint_name} API call failed: {result['error']}")
            return
        
        print(f"✅ {endpoint_name} API call successful")
        
        # Print some sample data based on the endpoint
        if "bids" in result and "asks" in result:
            # Order book data
            bid_count = len(result.get("bids", []))
            ask_count = len(result.get("asks", []))
            print(f"Order book contains {bid_count} bids and {ask_count} asks")
            
            if bid_count > 0:
                print("Top 3 bids:")
                for i, bid in enumerate(result["bids"][:3]):
                    print(f"  {i+1}: {bid[1]} @ {bid[0]}")
            
            if ask_count > 0:
                print("Top 3 asks:")
                for i, ask in enumerate(result["asks"][:3]):
                    print(f"  {i+1}: {ask[1]} @ {ask[0]}")
        
        elif isinstance(result, list) and len(result) > 0:
            # Trades or klines
            print(f"Received {len(result)} records")
            print("Sample data:")
            for i, item in enumerate(result[:3]):
                print(f"  {i+1}: {item}")
        
        elif "lastPrice" in result:
            # Ticker data
            print(f"Last price: {result.get('lastPrice')}")
            print(f"24h change: {result.get('priceChange')} ({result.get('priceChangePercent')}%)")
            print(f"24h volume: {result.get('volume')} {result.get('symbol', '').replace('BTC', '')}")
        
        elif "balances" in result:
            # Account data
            balances = [b for b in result.get("balances", []) if float(b.get("free", 0)) > 0 or float(b.get("locked", 0)) > 0]
            print(f"Account has {len(balances)} non-zero balances")
            for i, balance in enumerate(balances[:5]):
                print(f"  {i+1}: {balance.get('asset')}: Free={balance.get('free')}, Locked={balance.get('locked')}")
            
            if len(balances) > 5:
                print(f"  ... and {len(balances) - 5} more")

def main():
    print("===== MEXC REST API Test with New Credentials =====")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API Key: {API_KEY}")
    
    # Get public IP for documentation
    try:
        ip_response = requests.get("https://api.ipify.org")
        public_ip = ip_response.text
        print(f"Public IP: {public_ip}")
    except Exception as e:
        print(f"Unable to determine public IP: {e}")
    
    # Create tester and run tests
    tester = MexcRestApiTester(API_KEY, API_SECRET)
    
    # Test public endpoints
    tester.test_public_endpoints()
    
    # Test signed endpoints
    tester.test_signed_endpoints()
    
    # Test polling performance
    tester.measure_polling_performance(iterations=5, delay_ms=1000)
    
    # Print performance summary
    tester.tracker.print_summary()
    
    print("\n===== Test Complete =====")
    
    # Determine overall success
    success = len(tester.tracker.errors) == 0
    print("Overall result:", "✅ SUCCESS" if success else "❌ FAILED")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
