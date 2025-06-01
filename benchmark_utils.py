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
        
        # Validate inputs
        if not endpoint or not isinstance(endpoint, str):
            logger.error(f"Invalid endpoint: {endpoint}")
            return {
                "endpoint": str(endpoint),
                "method": method,
                "success": False,
                "error_rate": 1.0,
                "message": "Invalid endpoint parameter"
            }
            
        if not method or method not in ["GET", "POST", "DELETE"]:
            logger.error(f"Invalid HTTP method: {method}")
            return {
                "endpoint": endpoint,
                "method": str(method),
                "success": False,
                "error_rate": 1.0,
                "message": "Invalid HTTP method parameter"
            }
            
        if params is not None and not isinstance(params, dict):
            logger.error(f"Invalid params type: {type(params)}")
            params = {}
            
        if not isinstance(iterations, int) or iterations <= 0:
            logger.error(f"Invalid iterations value: {iterations}")
            iterations = 1
        
        logger.info(f"Measuring latency for {method} {endpoint} ({iterations} iterations)")
        
        for i in range(iterations):
            start_time = time.time()
            
            try:
                # Make request with validation of endpoint type
                if endpoint.startswith("/api/v3/ping") or endpoint.startswith("/api/v3/time"):
                    response_data = self.api_client.public_request(method, endpoint, params)
                else:
                    response_data = self.api_client.signed_request(method, endpoint, params)
                
                # Validate response
                if response_data is None:
                    logger.warning(f"Request failed: Response is None")
                    errors += 1
                    continue
                    
                if isinstance(response_data, dict) and not response_data:
                    # Empty dict might be valid for some endpoints like ping
                    if not endpoint.endswith("ping"):
                        logger.warning(f"Request returned empty dict")
                        
                elif isinstance(response_data, list) and not response_data:
                    # Empty list might be valid for some endpoints
                    logger.warning(f"Request returned empty list")
                
            except Exception as e:
                logger.error(f"Request error: {str(e)}")
                errors += 1
                continue
                
            end_time = time.time()
            
            try:
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
            except Exception as e:
                logger.error(f"Error calculating latency: {str(e)}")
                errors += 1
                continue
            
            # Add small delay between requests to avoid rate limiting
            try:
                time.sleep(0.2)
            except Exception as e:
                logger.error(f"Error during sleep: {str(e)}")
                # Continue anyway
        
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
        try:
            # Validate inputs
            if not symbol or not isinstance(symbol, str):
                logger.error(f"Invalid symbol: {symbol}")
                return {
                    "symbol": str(symbol),
                    "success": False,
                    "message": "Invalid symbol parameter"
                }
                
            if not isinstance(iterations, int) or iterations <= 0:
                logger.error(f"Invalid iterations value: {iterations}")
                iterations = 1
                
            logger.info(f"Simulating order workflow for {symbol} ({iterations} iterations)")
            
            # Get current market price with robust validation
            try:
                ticker_data = self.api_client.public_request("GET", "/api/v3/ticker/price", {"symbol": symbol})
                
                # Validate ticker response
                if ticker_data is None:
                    logger.error("Ticker request failed: Response is None")
                    return {
                        "symbol": symbol,
                        "success": False,
                        "message": "Failed to get ticker price: Response is None"
                    }
                    
                if not isinstance(ticker_data, dict):
                    logger.error(f"Invalid ticker response type: {type(ticker_data)}")
                    return {
                        "symbol": symbol,
                        "success": False,
                        "message": f"Failed to get ticker price: Invalid response type {type(ticker_data)}"
                    }
                
                if "price" not in ticker_data:
                    logger.error(f"Missing price in ticker data: {ticker_data}")
                    return {
                        "symbol": symbol,
                        "success": False,
                        "message": "Failed to get ticker price: Missing price field in response"
                    }
                
                # Validate price format
                price_str = ticker_data.get("price")
                if not price_str or not isinstance(price_str, str):
                    logger.error(f"Invalid price format: {price_str}")
                    return {
                        "symbol": symbol,
                        "success": False,
                        "message": f"Failed to get ticker price: Invalid price format {price_str}"
                    }
                
                try:
                    current_price = float(price_str)
                    if current_price <= 0:
                        logger.error(f"Invalid price value: {current_price}")
                        return {
                            "symbol": symbol,
                            "success": False,
                            "message": f"Failed to get ticker price: Invalid price value {current_price}"
                        }
                except (ValueError, TypeError) as e:
                    logger.error(f"Error parsing price: {str(e)}")
                    return {
                        "symbol": symbol,
                        "success": False,
                        "message": f"Failed to parse price: {str(e)}"
                    }
            except Exception as e:
                logger.error(f"Error getting ticker price: {str(e)}")
                return {
                    "symbol": symbol,
                    "success": False,
                    "message": f"Failed to get ticker price: {str(e)}"
                }
            
            # Set a price far from current market price to ensure order won't execute
            # For buy orders: 20% below current price
            try:
                test_price = round(current_price * 0.8, 2)
                if test_price <= 0:
                    logger.error(f"Calculated test price is invalid: {test_price}")
                    test_price = round(current_price * 0.9, 2)  # Try a different calculation
            except Exception as e:
                logger.error(f"Error calculating test price: {str(e)}")
                return {
                    "symbol": symbol,
                    "success": False,
                    "message": f"Failed to calculate test price: {str(e)}"
                }
            
            workflow_times = []
            errors = 0
            
            for i in range(iterations):
                start_time = time.time()
                
                try:
                    # 1. Place a test limit order (will not execute due to price) with validation
                    try:
                        order_params = {
                            "symbol": symbol,
                            "side": "BUY",
                            "type": "LIMIT",
                            "timeInForce": "GTC",
                            "quantity": "0.001",
                            "price": str(test_price),
                            "newClientOrderId": f"benchmark_test_{int(time.time())}"
                        }
                        
                        # Validate order parameters
                        if not all(key in order_params for key in ["symbol", "side", "type", "quantity", "price"]):
                            logger.error(f"Missing required order parameters")
                            errors += 1
                            continue
                            
                        place_response = self.api_client.signed_request("POST", "/api/v3/order/test", order_params)
                        
                        # Validate response
                        if place_response is None:
                            logger.warning(f"Test order failed: Response is None")
                            errors += 1
                            continue
                            
                        # Empty dict is a successful response for test orders
                        if not isinstance(place_response, dict):
                            logger.warning(f"Test order returned invalid response type: {type(place_response)}")
                            errors += 1
                            continue
                    except Exception as e:
                        logger.error(f"Error preparing or placing test order: {str(e)}")
                        errors += 1
                        continue
                    
                except Exception as e:
                    logger.error(f"Order workflow error: {str(e)}")
                    errors += 1
                    continue
                    
                try:
                    end_time = time.time()
                    workflow_time_ms = (end_time - start_time) * 1000
                    workflow_times.append(workflow_time_ms)
                except Exception as e:
                    logger.error(f"Error calculating workflow time: {str(e)}")
                    errors += 1
                    continue
                
                # Add delay between iterations with validation
                try:
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error during sleep: {str(e)}")
                    # Continue anyway
            
            if not workflow_times:
                return {
                    "symbol": symbol,
                    "success": False,
                    "error_rate": 1.0,
                    "message": "All workflow simulations failed"
                }
        except Exception as e:
            logger.error(f"Unexpected error in order workflow simulation: {str(e)}")
            return {
                "symbol": symbol,
                "success": False,
                "error_rate": 1.0,
                "message": f"Unexpected error: {str(e)}"
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
        try:
            # Initialize results with validation
            results = {
                "timestamp": self.timestamp,
                "api_latency": {},
                "system_resources": {},
                "order_workflow": {}
            }
            
            # 1. Measure API latency for key endpoints with validation
            try:
                endpoints = [
                    ("/api/v3/ping", "GET", None),
                    ("/api/v3/time", "GET", None),
                    ("/api/v3/exchangeInfo", "GET", {"symbol": "BTCUSDT"}),
                    ("/api/v3/depth", "GET", {"symbol": "BTCUSDT", "limit": 5}),
                    ("/api/v3/ticker/24hr", "GET", {"symbol": "BTCUSDT"}),
                    ("/api/v3/account", "GET", None)
                ]
                
                for endpoint, method, params in endpoints:
                    try:
                        # Validate endpoint format
                        if not endpoint or not isinstance(endpoint, str):
                            logger.error(f"Invalid endpoint format: {endpoint}")
                            continue
                            
                        # Extract endpoint key with validation
                        try:
                            endpoint_parts = endpoint.split("/")
                            if len(endpoint_parts) < 2:
                                endpoint_key = endpoint.replace("/", "_")
                            else:
                                endpoint_key = endpoint_parts[-1]
                                
                            if not endpoint_key:
                                endpoint_key = "unknown_endpoint"
                        except Exception as e:
                            logger.error(f"Error extracting endpoint key: {str(e)}")
                            endpoint_key = "error_endpoint"
                        
                        # Measure latency with validation
                        latency_result = self.measure_api_latency(endpoint, method, params)
                        
                        # Validate result before storing
                        if not isinstance(latency_result, dict):
                            logger.error(f"Invalid latency result type: {type(latency_result)}")
                            results["api_latency"][endpoint_key] = {
                                "endpoint": endpoint,
                                "method": method,
                                "success": False,
                                "message": f"Invalid result type: {type(latency_result)}"
                            }
                        else:
                            results["api_latency"][endpoint_key] = latency_result
                    except Exception as e:
                        logger.error(f"Error measuring latency for {endpoint}: {str(e)}")
                        results["api_latency"][endpoint_key if 'endpoint_key' in locals() else "error_endpoint"] = {
                            "endpoint": endpoint,
                            "method": method,
                            "success": False,
                            "message": f"Error: {str(e)}"
                        }
            except Exception as e:
                logger.error(f"Error in API latency measurement section: {str(e)}")
                results["api_latency"]["error"] = {
                    "success": False,
                    "message": f"Section error: {str(e)}"
                }
        
            # 2. Measure system resources during API calls with validation
            try:
                def make_api_calls():
                    try:
                        for _ in range(5):
                            try:
                                self.api_client.public_request("GET", "/api/v3/ticker/price", {"symbol": "BTCUSDT"})
                                time.sleep(0.2)
                                self.api_client.public_request("GET", "/api/v3/depth", {"symbol": "BTCUSDT", "limit": 10})
                                time.sleep(0.2)
                            except Exception as e:
                                logger.error(f"Error in API call during resource measurement: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error in make_api_calls function: {str(e)}")
        
                # Start a thread to make API calls while measuring resources with validation
                try:
                    api_thread = threading.Thread(target=make_api_calls)
                    api_thread.start()
                    
                    # Measure resources while API calls are happening with validation
                    try:
                        results["system_resources"]["during_api_calls"] = self.measure_system_resources(duration=5)
                    except Exception as e:
                        logger.error(f"Error measuring system resources: {str(e)}")
                        results["system_resources"]["during_api_calls"] = {
                            "success": False,
                            "message": f"Error: {str(e)}"
                        }
                    
                    # Wait for API thread to complete with validation
                    try:
                        api_thread.join(timeout=10)  # Add timeout to prevent hanging
                    except Exception as e:
                        logger.error(f"Error joining API thread: {str(e)}")
                except Exception as e:
                    logger.error(f"Error in system resources measurement section: {str(e)}")
                    results["system_resources"]["error"] = {
                        "success": False,
                        "message": f"Section error: {str(e)}"
                    }
        

                    # 3. Simulate order workflow with validation
            try:
                # Validate symbol before simulation
                symbol = "BTCUSDT"
                try:
                    results["order_workflow"][symbol] = self.simulate_order_workflow(symbol=symbol, iterations=3)
                except Exception as e:
                    logger.error(f"Error simulating order workflow for {symbol}: {str(e)}")
                    results["order_workflow"][symbol] = {
                        "symbol": symbol,
                        "success": False,
                        "message": f"Error: {str(e)}"
                    }
            except Exception as e:
                logger.error(f"Error in order workflow simulation section: {str(e)}")
                results["order_workflow"]["error"] = {
                    "success": False,
                    "message": f"Section error: {str(e)}"
                }
            # Save results to file with validation
            try:
                # Validate results directory exists
                if not os.path.exists(self.results_dir):
                    os.makedirs(self.results_dir, exist_ok=True)
                    
                results_file = os.path.join(self.results_dir, f"benchmark_{self.timestamp}.json")
                
                # Save with error handling
                try:
                    with open(results_file, "w") as f:
                        json.dump(results, f, indent=2)
                    
                    logger.info(f"Benchmark results saved to {results_file}")
                except (IOError, OSError) as e:
                    logger.error(f"Error saving results to file: {str(e)}")
                    # Try alternative location
                    alt_file = f"benchmark_results_{self.timestamp}.json"
                    try:
                        with open(alt_file, "w") as f:
                            json.dump(results, f, indent=2)
                        logger.info(f"Benchmark results saved to alternative location: {alt_file}")
                    except Exception as e2:
                        logger.error(f"Failed to save results to alternative location: {str(e2)}")
            except Exception as e:
                logger.error(f"Error in results saving section: {str(e)}")
                
            return results
        except Exception as e:
            logger.error(f"Unexpected error in comprehensive benchmark: {str(e)}")
            return {
                "timestamp": self.timestamp,
                "success": False,
                "message": f"Benchmark failed with error: {str(e)}"
            }

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
