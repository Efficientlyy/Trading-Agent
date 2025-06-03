#!/usr/bin/env python
"""
Latency benchmarking for the HFT execution engine.

This script measures the end-to-end latency of the HFT execution engine,
including tick processing, order book updates, and signal generation.
"""

import os
import sys
import time
import statistics
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_integration.hft_engine_wrapper import HFTEngineWrapper

class LatencyBenchmark:
    """Benchmark for measuring HFT engine latency."""
    
    def __init__(self, symbol: str = "BTCUSDC"):
        """Initialize the benchmark."""
        self.symbol = symbol
        self.engine = HFTEngineWrapper(symbol, use_mock=True)
        self.tick_latencies = []
        self.orderbook_latencies = []
        self.signal_latencies = []
    
    def run_tick_benchmark(self, iterations: int = 1000) -> List[float]:
        """
        Benchmark tick processing latency.
        
        Args:
            iterations: Number of iterations to run
            
        Returns:
            List of latencies in microseconds
        """
        latencies = []
        
        for i in range(iterations):
            price = 50000 + np.random.normal(0, 100)
            volume = abs(np.random.normal(0, 1))
            bid = price - 5
            ask = price + 5
            
            start_time = time.time_ns() // 1000  # microseconds
            self.engine.process_tick(price, volume, bid, ask)
            end_time = time.time_ns() // 1000  # microseconds
            
            latency = end_time - start_time
            latencies.append(latency)
        
        self.tick_latencies = latencies
        return latencies
    
    def run_orderbook_benchmark(self, iterations: int = 1000) -> List[float]:
        """
        Benchmark order book update latency.
        
        Args:
            iterations: Number of iterations to run
            
        Returns:
            List of latencies in microseconds
        """
        latencies = []
        
        for i in range(iterations):
            price = 50000 + np.random.normal(0, 100)
            
            # Generate random order book with 20 levels
            bids = [(price - 5 - j + np.random.normal(0, 0.5), 
                    abs(np.random.normal(1, 0.5))) for j in range(20)]
            asks = [(price + 5 + j + np.random.normal(0, 0.5), 
                    abs(np.random.normal(1, 0.5))) for j in range(20)]
            
            start_time = time.time_ns() // 1000  # microseconds
            self.engine.update_orderbook(bids, asks)
            end_time = time.time_ns() // 1000  # microseconds
            
            latency = end_time - start_time
            latencies.append(latency)
        
        self.orderbook_latencies = latencies
        return latencies
    
    def run_signal_benchmark(self, iterations: int = 1000) -> List[float]:
        """
        Benchmark signal generation latency.
        
        Args:
            iterations: Number of iterations to run
            
        Returns:
            List of latencies in microseconds
        """
        # First, populate with some data
        for i in range(100):
            price = 50000 + i * 10
            self.engine.process_tick(price, 0.1, price - 5, price + 5)
            
            bids = [(price - 5 - j, 1.0 / (j + 1)) for j in range(20)]
            asks = [(price + 5 + j, 1.0 / (j + 1)) for j in range(20)]
            self.engine.update_orderbook(bids, asks)
        
        # Now benchmark signal generation
        latencies = []
        
        for i in range(iterations):
            start_time = time.time_ns() // 1000  # microseconds
            self.engine.get_trading_signal()
            end_time = time.time_ns() // 1000  # microseconds
            
            latency = end_time - start_time
            latencies.append(latency)
        
        self.signal_latencies = latencies
        return latencies
    
    def run_full_benchmark(self, iterations: int = 1000) -> Dict[str, List[float]]:
        """
        Run all benchmarks.
        
        Args:
            iterations: Number of iterations for each benchmark
            
        Returns:
            Dictionary of benchmark results
        """
        print(f"Running tick processing benchmark ({iterations} iterations)...")
        tick_latencies = self.run_tick_benchmark(iterations)
        
        print(f"Running order book update benchmark ({iterations} iterations)...")
        orderbook_latencies = self.run_orderbook_benchmark(iterations)
        
        print(f"Running signal generation benchmark ({iterations} iterations)...")
        signal_latencies = self.run_signal_benchmark(iterations)
        
        return {
            "tick": tick_latencies,
            "orderbook": orderbook_latencies,
            "signal": signal_latencies
        }
    
    def print_results(self):
        """Print benchmark results."""
        print("\n=== HFT Engine Latency Benchmark Results ===\n")
        
        if self.tick_latencies:
            print("Tick Processing Latency (microseconds):")
            print(f"  Min: {min(self.tick_latencies):.2f}")
            print(f"  Max: {max(self.tick_latencies):.2f}")
            print(f"  Mean: {statistics.mean(self.tick_latencies):.2f}")
            print(f"  Median: {statistics.median(self.tick_latencies):.2f}")
            print(f"  95th percentile: {np.percentile(self.tick_latencies, 95):.2f}")
            print(f"  99th percentile: {np.percentile(self.tick_latencies, 99):.2f}")
        
        if self.orderbook_latencies:
            print("\nOrder Book Update Latency (microseconds):")
            print(f"  Min: {min(self.orderbook_latencies):.2f}")
            print(f"  Max: {max(self.orderbook_latencies):.2f}")
            print(f"  Mean: {statistics.mean(self.orderbook_latencies):.2f}")
            print(f"  Median: {statistics.median(self.orderbook_latencies):.2f}")
            print(f"  95th percentile: {np.percentile(self.orderbook_latencies, 95):.2f}")
            print(f"  99th percentile: {np.percentile(self.orderbook_latencies, 99):.2f}")
        
        if self.signal_latencies:
            print("\nSignal Generation Latency (microseconds):")
            print(f"  Min: {min(self.signal_latencies):.2f}")
            print(f"  Max: {max(self.signal_latencies):.2f}")
            print(f"  Mean: {statistics.mean(self.signal_latencies):.2f}")
            print(f"  Median: {statistics.median(self.signal_latencies):.2f}")
            print(f"  95th percentile: {np.percentile(self.signal_latencies, 95):.2f}")
            print(f"  99th percentile: {np.percentile(self.signal_latencies, 99):.2f}")
        
        if self.tick_latencies and self.orderbook_latencies and self.signal_latencies:
            # Calculate end-to-end latency (tick + orderbook + signal)
            print("\nEstimated End-to-End Latency (microseconds):")
            e2e_median = (statistics.median(self.tick_latencies) + 
                         statistics.median(self.orderbook_latencies) + 
                         statistics.median(self.signal_latencies))
            e2e_p95 = (np.percentile(self.tick_latencies, 95) + 
                      np.percentile(self.orderbook_latencies, 95) + 
                      np.percentile(self.signal_latencies, 95))
            e2e_p99 = (np.percentile(self.tick_latencies, 99) + 
                      np.percentile(self.orderbook_latencies, 99) + 
                      np.percentile(self.signal_latencies, 99))
            
            print(f"  Median: {e2e_median:.2f}")
            print(f"  95th percentile: {e2e_p95:.2f}")
            print(f"  99th percentile: {e2e_p99:.2f}")
            print(f"  Median (ms): {e2e_median/1000:.3f}")
            print(f"  95th percentile (ms): {e2e_p95/1000:.3f}")
            print(f"  99th percentile (ms): {e2e_p99/1000:.3f}")
    
    def plot_results(self, save_path: str = None):
        """
        Plot benchmark results.
        
        Args:
            save_path: Path to save the plot, or None to display
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot histograms
        if self.tick_latencies:
            axs[0, 0].hist(self.tick_latencies, bins=50, alpha=0.7)
            axs[0, 0].set_title('Tick Processing Latency')
            axs[0, 0].set_xlabel('Microseconds')
            axs[0, 0].set_ylabel('Frequency')
        
        if self.orderbook_latencies:
            axs[0, 1].hist(self.orderbook_latencies, bins=50, alpha=0.7)
            axs[0, 1].set_title('Order Book Update Latency')
            axs[0, 1].set_xlabel('Microseconds')
            axs[0, 1].set_ylabel('Frequency')
        
        if self.signal_latencies:
            axs[1, 0].hist(self.signal_latencies, bins=50, alpha=0.7)
            axs[1, 0].set_title('Signal Generation Latency')
            axs[1, 0].set_xlabel('Microseconds')
            axs[1, 0].set_ylabel('Frequency')
        
        # Plot comparison boxplot
        if self.tick_latencies and self.orderbook_latencies and self.signal_latencies:
            data = [self.tick_latencies, self.orderbook_latencies, self.signal_latencies]
            axs[1, 1].boxplot(data, labels=['Tick', 'OrderBook', 'Signal'])
            axs[1, 1].set_title('Latency Comparison')
            axs[1, 1].set_ylabel('Microseconds')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # Run benchmark
    benchmark = LatencyBenchmark()
    benchmark.run_full_benchmark(iterations=1000)
    benchmark.print_results()
    benchmark.plot_results(save_path="latency_benchmark_results.png")
