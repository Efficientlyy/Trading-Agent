#!/usr/bin/env python
"""
Benchmarking Utilities for Flash Trading System

This module provides tools for measuring and profiling the performance
of the flash trading system, focusing on latency, throughput, and resource usage.
"""

import time
import statistics
import json
import os
import psutil
import threading
import logging
from datetime import datetime
from mexc_api_utils import MexcApiClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_results.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("benchmark")

class PerformanceBenchmark:
    """Performance benchmarking tools for flash trading system"""
    
    def __init__(self, api_client=None, env_path=None):
        """Initialize the benchmark utility"""
        if api_client:
            self.api_client = api_client
        else:
            self.api_client = MexcApiClient(env_path=env_path)
        
        self.results_dir = "benchmark_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def measure_api_latency(self, endpoint, method="GET", params=None, iterations=10):
        """Measure API request latency for a specific endpoint"""
        latencies = []
        errors = 0
        
        logger.info(f"Measuring latency for {method} {endpoint} ({iterations} iterations)")
        
        for i in range(iterations):
            start_time = time.time()
            
            try:
                if endpoint.startswith("/api/v3/ping") or endpoint.startswith("/api/v3/time"):
                    response = self.api_client.public_request(method, endpoint, params)
                else:
                    response = self.api_client.signed_request(method, endpoint, params)
                
                if response.status_code != 200:
                    logger.warning(f"Request failed: {response.status_code} - {response.text}")
                    errors += 1
                    continue
                
            except Exception as e:
                logger.error(f"Request error: {str(e)}")
                errors += 1
                continue
                
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Add small delay between requests to avoid rate limiting
            time.sleep(0.2)
        
        if not latencies:
            return {
                "endpoint": endpoint,
                "method": method,
                "success": False,
                "error_rate": 1.0,
                "message": "All requests failed"
            }
        
        return {
            "endpoint": endpoint,
            "method": method,
            "success": True,
            "iterations": iterations,
            "errors": errors,
            "error_rate": errors / iterations,
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "avg_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20 else None,
            "std_dev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0
        }
    
    def measure_system_resources(self, duration=10, interval=0.5):
        """Measure system resource usage over a specified duration"""
        cpu_usage = []
        memory_usage = []
        
        logger.info(f"Measuring system resources for {duration} seconds")
        
        end_time = time.time() + duration
        while time.time() < end_time:
            cpu_usage.append(psutil.cpu_percent(interval=None))
            memory_usage.append(psutil.virtual_memory().percent)
            time.sleep(interval)
        
        return {
            "duration_seconds": duration,
            "samples": len(cpu_usage),
            "cpu_usage_avg": statistics.mean(cpu_usage),
            "cpu_usage_max": max(cpu_usage),
            "memory_usage_avg": statistics.mean(memory_usage),
            "memory_usage_max": max(memory_usage)
        }
    
    def simulate_order_workflow(self, symbol="BTCUSDT", iterations=5):
        """Simulate and benchmark a complete order workflow"""
        logger.info(f"Simulating order workflow for {symbol} ({iterations} iterations)")
        
        # Get current market price
        ticker_response = self.api_client.public_request("GET", "/api/v3/ticker/price", {"symbol": symbol})
        if ticker_response.status_code != 200:
            logger.error(f"Failed to get ticker price: {ticker_response.text}")
            return {
                "success": False,
                "message": f"Failed to get ticker price: {ticker_response.status_code}"
            }
        
        current_price = float(ticker_response.json()["price"])
        
        # Set a price far from current market price to ensure order won't execute
        # For buy orders: 20% below current price
        test_price = round(current_price * 0.8, 2)
        
        workflow_times = []
        errors = 0
        
        for i in range(iterations):
            start_time = time.time()
            
            try:
                # 1. Place a test limit order (will not execute due to price)
                order_params = {
                    "symbol": symbol,
                    "side": "BUY",
                    "type": "LIMIT",
                    "timeInForce": "GTC",
                    "quantity": "0.001",
                    "price": str(test_price),
                    "newClientOrderId": f"benchmark_test_{int(time.time())}"
                }
                
                place_response = self.api_client.signed_request("POST", "/api/v3/order/test", order_params)
                if place_response.status_code != 200:
                    logger.warning(f"Test order failed: {place_response.status_code} - {place_response.text}")
                    errors += 1
                    continue
                
            except Exception as e:
                logger.error(f"Order workflow error: {str(e)}")
                errors += 1
                continue
                
            end_time = time.time()
            workflow_time_ms = (end_time - start_time) * 1000
            workflow_times.append(workflow_time_ms)
            
            # Add delay between iterations
            time.sleep(1)
        
        if not workflow_times:
            return {
                "symbol": symbol,
                "success": False,
                "error_rate": 1.0,
                "message": "All workflow simulations failed"
            }
        
        return {
            "symbol": symbol,
            "success": True,
            "iterations": iterations,
            "errors": errors,
            "error_rate": errors / iterations,
            "min_workflow_ms": min(workflow_times),
            "max_workflow_ms": max(workflow_times),
            "avg_workflow_ms": statistics.mean(workflow_times),
            "median_workflow_ms": statistics.median(workflow_times)
        }
    
    def run_comprehensive_benchmark(self):
        """Run a comprehensive benchmark suite and save results"""
        results = {
            "timestamp": self.timestamp,
            "api_latency": {},
            "system_resources": {},
            "order_workflow": {}
        }
        
        # 1. Measure API latency for key endpoints
        endpoints = [
            ("/api/v3/ping", "GET", None),
            ("/api/v3/time", "GET", None),
            ("/api/v3/exchangeInfo", "GET", {"symbol": "BTCUSDT"}),
            ("/api/v3/depth", "GET", {"symbol": "BTCUSDT", "limit": 5}),
            ("/api/v3/ticker/24hr", "GET", {"symbol": "BTCUSDT"}),
            ("/api/v3/account", "GET", None)
        ]
        
        for endpoint, method, params in endpoints:
            endpoint_key = endpoint.split("/")[-1]
            results["api_latency"][endpoint_key] = self.measure_api_latency(endpoint, method, params)
        
        # 2. Measure system resources during API calls
        def make_api_calls():
            for _ in range(5):
                self.api_client.public_request("GET", "/api/v3/ticker/price", {"symbol": "BTCUSDT"})
                time.sleep(0.2)
                self.api_client.public_request("GET", "/api/v3/depth", {"symbol": "BTCUSDT", "limit": 10})
                time.sleep(0.2)
        
        # Start a thread to make API calls while measuring resources
        api_thread = threading.Thread(target=make_api_calls)
        api_thread.start()
        
        # Measure resources while API calls are happening
        results["system_resources"]["during_api_calls"] = self.measure_system_resources(duration=5)
        
        # Wait for API thread to complete
        api_thread.join()
        
        # 3. Simulate order workflow
        results["order_workflow"]["BTCUSDT"] = self.simulate_order_workflow(symbol="BTCUSDT", iterations=3)
        
        # Save results to file
        results_file = os.path.join(self.results_dir, f"benchmark_{self.timestamp}.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {results_file}")
        return results

# Example usage
if __name__ == "__main__":
    benchmark = PerformanceBenchmark(env_path=".env-secure/.env")
    results = benchmark.run_comprehensive_benchmark()
    
    # Print summary
    print("\n=== BENCHMARK SUMMARY ===")
    
    print("\nAPI Latency (ms):")
    for endpoint, data in results["api_latency"].items():
        if data["success"]:
            print(f"  {endpoint}: avg={data['avg_latency_ms']:.2f}, min={data['min_latency_ms']:.2f}, max={data['max_latency_ms']:.2f}")
        else:
            print(f"  {endpoint}: FAILED")
    
    print("\nSystem Resources:")
    res = results["system_resources"]["during_api_calls"]
    print(f"  CPU: avg={res['cpu_usage_avg']:.2f}%, max={res['cpu_usage_max']:.2f}%")
    print(f"  Memory: avg={res['memory_usage_avg']:.2f}%, max={res['memory_usage_max']:.2f}%")
    
    print("\nOrder Workflow (ms):")
    for symbol, data in results["order_workflow"].items():
        if data["success"]:
            print(f"  {symbol}: avg={data['avg_workflow_ms']:.2f}, min={data['min_workflow_ms']:.2f}, max={data['max_workflow_ms']:.2f}")
        else:
            print(f"  {symbol}: FAILED")
    
    print(f"\nDetailed results saved to {os.path.join(benchmark.results_dir, f'benchmark_{benchmark.timestamp}.json')}")
