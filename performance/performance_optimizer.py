#!/usr/bin/env python
"""
Performance Optimization Module for Trading-Agent System

This module provides performance optimization capabilities for high-frequency trading,
including efficient data processing, batch operations, and resource management.
"""

import os
import sys
import time
import logging
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from collections import deque

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import error handling
from error_handling.error_manager import handle_error, ErrorCategory, ErrorSeverity, safe_execute

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("performance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("performance_optimizer")

class PerformanceMetrics:
    """Performance metrics tracker"""
    
    def __init__(self, window_size: int = 100):
        """Initialize performance metrics
        
        Args:
            window_size: Size of rolling window for metrics
        """
        self.window_size = window_size
        self.execution_times = {}
        self.api_latencies = {}
        self.memory_usage = deque(maxlen=window_size)
        self.cpu_usage = deque(maxlen=window_size)
        self.last_gc_time = time.time()
        
        logger.info(f"Initialized PerformanceMetrics with window_size={window_size}")
    
    def record_execution_time(self, operation: str, execution_time: float):
        """Record execution time
        
        Args:
            operation: Operation name
            execution_time: Execution time in seconds
        """
        if operation not in self.execution_times:
            self.execution_times[operation] = deque(maxlen=self.window_size)
        
        self.execution_times[operation].append(execution_time)
    
    def record_api_latency(self, endpoint: str, latency: float):
        """Record API latency
        
        Args:
            endpoint: API endpoint
            latency: Latency in seconds
        """
        if endpoint not in self.api_latencies:
            self.api_latencies[endpoint] = deque(maxlen=self.window_size)
        
        self.api_latencies[endpoint].append(latency)
    
    def record_memory_usage(self, memory_mb: float):
        """Record memory usage
        
        Args:
            memory_mb: Memory usage in MB
        """
        self.memory_usage.append(memory_mb)
    
    def record_cpu_usage(self, cpu_percent: float):
        """Record CPU usage
        
        Args:
            cpu_percent: CPU usage percentage
        """
        self.cpu_usage.append(cpu_percent)
    
    def get_average_execution_time(self, operation: str) -> float:
        """Get average execution time
        
        Args:
            operation: Operation name
            
        Returns:
            Average execution time in seconds
        """
        if operation not in self.execution_times or not self.execution_times[operation]:
            return 0.0
        
        return sum(self.execution_times[operation]) / len(self.execution_times[operation])
    
    def get_average_api_latency(self, endpoint: str) -> float:
        """Get average API latency
        
        Args:
            endpoint: API endpoint
            
        Returns:
            Average latency in seconds
        """
        if endpoint not in self.api_latencies or not self.api_latencies[endpoint]:
            return 0.0
        
        return sum(self.api_latencies[endpoint]) / len(self.api_latencies[endpoint])
    
    def get_average_memory_usage(self) -> float:
        """Get average memory usage
        
        Returns:
            Average memory usage in MB
        """
        if not self.memory_usage:
            return 0.0
        
        return sum(self.memory_usage) / len(self.memory_usage)
    
    def get_average_cpu_usage(self) -> float:
        """Get average CPU usage
        
        Returns:
            Average CPU usage percentage
        """
        if not self.cpu_usage:
            return 0.0
        
        return sum(self.cpu_usage) / len(self.cpu_usage)
    
    def get_metrics_summary(self) -> Dict:
        """Get metrics summary
        
        Returns:
            Dictionary with metrics summary
        """
        execution_times_summary = {}
        for operation, times in self.execution_times.items():
            if times:
                execution_times_summary[operation] = {
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "count": len(times)
                }
        
        api_latencies_summary = {}
        for endpoint, latencies in self.api_latencies.items():
            if latencies:
                api_latencies_summary[endpoint] = {
                    "avg": sum(latencies) / len(latencies),
                    "min": min(latencies),
                    "max": max(latencies),
                    "count": len(latencies)
                }
        
        return {
            "execution_times": execution_times_summary,
            "api_latencies": api_latencies_summary,
            "memory_usage": {
                "avg": self.get_average_memory_usage(),
                "current": self.memory_usage[-1] if self.memory_usage else 0.0
            },
            "cpu_usage": {
                "avg": self.get_average_cpu_usage(),
                "current": self.cpu_usage[-1] if self.cpu_usage else 0.0
            }
        }
    
    def should_trigger_gc(self, threshold_mb: float = 1000.0, min_interval: float = 60.0) -> bool:
        """Check if garbage collection should be triggered
        
        Args:
            threshold_mb: Memory threshold in MB
            min_interval: Minimum interval between GC in seconds
            
        Returns:
            True if GC should be triggered, False otherwise
        """
        if not self.memory_usage:
            return False
        
        current_memory = self.memory_usage[-1]
        time_since_last_gc = time.time() - self.last_gc_time
        
        return current_memory > threshold_mb and time_since_last_gc > min_interval

class DataAggregator:
    """Data aggregator for efficient processing"""
    
    def __init__(self, max_buffer_size: int = 1000):
        """Initialize data aggregator
        
        Args:
            max_buffer_size: Maximum buffer size
        """
        self.max_buffer_size = max_buffer_size
        self.data_buffers = {}
        self.last_aggregation = {}
        self.aggregation_functions = {}
        
        logger.info(f"Initialized DataAggregator with max_buffer_size={max_buffer_size}")
    
    def add_data(self, key: str, data: Any):
        """Add data to buffer
        
        Args:
            key: Data key
            data: Data to add
        """
        if key not in self.data_buffers:
            self.data_buffers[key] = deque(maxlen=self.max_buffer_size)
            self.last_aggregation[key] = time.time()
        
        self.data_buffers[key].append(data)
    
    def register_aggregation_function(self, key: str, function: Callable):
        """Register aggregation function
        
        Args:
            key: Data key
            function: Aggregation function
        """
        self.aggregation_functions[key] = function
        logger.info(f"Registered aggregation function for {key}")
    
    def aggregate_data(self, key: str, force: bool = False) -> Optional[Any]:
        """Aggregate data
        
        Args:
            key: Data key
            force: Whether to force aggregation
            
        Returns:
            Aggregated data
        """
        if key not in self.data_buffers or not self.data_buffers[key]:
            return None
        
        if key not in self.aggregation_functions:
            logger.warning(f"No aggregation function registered for {key}")
            return None
        
        # Check if buffer is full or force aggregation
        if force or len(self.data_buffers[key]) >= self.max_buffer_size:
            try:
                # Convert to list to avoid modifying during iteration
                data_list = list(self.data_buffers[key])
                
                # Aggregate data
                result = self.aggregation_functions[key](data_list)
                
                # Update last aggregation time
                self.last_aggregation[key] = time.time()
                
                # Clear buffer
                self.data_buffers[key].clear()
                
                return result
            
            except Exception as e:
                handle_error(e, ErrorCategory.DATA, context={"operation": "aggregate_data", "key": key})
                return None
        
        return None
    
    def get_buffer_size(self, key: str) -> int:
        """Get buffer size
        
        Args:
            key: Data key
            
        Returns:
            Buffer size
        """
        if key not in self.data_buffers:
            return 0
        
        return len(self.data_buffers[key])
    
    def get_time_since_last_aggregation(self, key: str) -> float:
        """Get time since last aggregation
        
        Args:
            key: Data key
            
        Returns:
            Time since last aggregation in seconds
        """
        if key not in self.last_aggregation:
            return float('inf')
        
        return time.time() - self.last_aggregation[key]
    
    def should_aggregate(self, key: str, buffer_threshold: float = 0.8, time_threshold: float = 60.0) -> bool:
        """Check if data should be aggregated
        
        Args:
            key: Data key
            buffer_threshold: Buffer threshold (0.0-1.0)
            time_threshold: Time threshold in seconds
            
        Returns:
            True if data should be aggregated, False otherwise
        """
        if key not in self.data_buffers:
            return False
        
        buffer_size = len(self.data_buffers[key])
        buffer_ratio = buffer_size / self.max_buffer_size
        time_since_last = self.get_time_since_last_aggregation(key)
        
        return buffer_ratio >= buffer_threshold or time_since_last >= time_threshold

class BatchProcessor:
    """Batch processor for efficient operations"""
    
    def __init__(self, batch_size: int = 100, max_delay: float = 1.0):
        """Initialize batch processor
        
        Args:
            batch_size: Batch size
            max_delay: Maximum delay in seconds
        """
        self.batch_size = batch_size
        self.max_delay = max_delay
        self.batch_queues = {}
        self.batch_processors = {}
        self.last_batch_time = {}
        self.batch_locks = {}
        self.running = True
        
        # Start background thread for batch processing
        self.background_thread = threading.Thread(target=self._background_processor)
        self.background_thread.daemon = True
        self.background_thread.start()
        
        logger.info(f"Initialized BatchProcessor with batch_size={batch_size}, max_delay={max_delay}s")
    
    def register_batch_processor(self, key: str, processor: Callable):
        """Register batch processor
        
        Args:
            key: Batch key
            processor: Batch processor function
        """
        self.batch_processors[key] = processor
        self.batch_queues[key] = deque()
        self.last_batch_time[key] = time.time()
        self.batch_locks[key] = threading.Lock()
        
        logger.info(f"Registered batch processor for {key}")
    
    def add_to_batch(self, key: str, item: Any):
        """Add item to batch
        
        Args:
            key: Batch key
            item: Item to add
        """
        if key not in self.batch_queues:
            logger.warning(f"No batch queue for {key}")
            return
        
        with self.batch_locks[key]:
            self.batch_queues[key].append(item)
    
    def process_batch(self, key: str, force: bool = False) -> bool:
        """Process batch
        
        Args:
            key: Batch key
            force: Whether to force processing
            
        Returns:
            True if batch was processed, False otherwise
        """
        if key not in self.batch_queues or key not in self.batch_processors:
            return False
        
        with self.batch_locks[key]:
            # Check if batch should be processed
            if not force and len(self.batch_queues[key]) < self.batch_size and time.time() - self.last_batch_time[key] < self.max_delay:
                return False
            
            if not self.batch_queues[key]:
                return False
            
            try:
                # Get items from queue
                items = list(self.batch_queues[key])
                
                # Clear queue
                self.batch_queues[key].clear()
                
                # Update last batch time
                self.last_batch_time[key] = time.time()
            
            except Exception as e:
                handle_error(e, ErrorCategory.DATA, context={"operation": "process_batch", "key": key})
                return False
        
        try:
            # Process batch
            self.batch_processors[key](items)
            
            return True
        
        except Exception as e:
            handle_error(e, ErrorCategory.DATA, context={"operation": "process_batch", "key": key})
            return False
    
    def _background_processor(self):
        """Background thread for batch processing"""
        while self.running:
            for key in list(self.batch_queues.keys()):
                try:
                    # Check if batch should be processed
                    time_since_last = time.time() - self.last_batch_time.get(key, 0)
                    
                    if time_since_last >= self.max_delay:
                        self.process_batch(key, force=True)
                
                except Exception as e:
                    handle_error(e, ErrorCategory.SYSTEM, context={"operation": "background_processor", "key": key})
            
            # Sleep to avoid high CPU usage
            time.sleep(0.1)
    
    def get_batch_size(self, key: str) -> int:
        """Get batch size
        
        Args:
            key: Batch key
            
        Returns:
            Batch size
        """
        if key not in self.batch_queues:
            return 0
        
        return len(self.batch_queues[key])
    
    def get_time_since_last_batch(self, key: str) -> float:
        """Get time since last batch
        
        Args:
            key: Batch key
            
        Returns:
            Time since last batch in seconds
        """
        if key not in self.last_batch_time:
            return float('inf')
        
        return time.time() - self.last_batch_time[key]
    
    def shutdown(self):
        """Shutdown batch processor"""
        self.running = False
        
        # Process remaining batches
        for key in list(self.batch_queues.keys()):
            self.process_batch(key, force=True)
        
        logger.info("BatchProcessor shutdown")

class PerformanceOptimizer:
    """Performance optimizer for Trading-Agent system"""
    
    def __init__(self):
        """Initialize performance optimizer"""
        self.metrics = PerformanceMetrics()
        self.data_aggregator = DataAggregator()
        self.batch_processor = BatchProcessor()
        
        # Register common aggregation functions
        self._register_common_aggregations()
        
        # Register common batch processors
        self._register_common_batch_processors()
        
        logger.info("Initialized PerformanceOptimizer")
    
    def _register_common_aggregations(self):
        """Register common aggregation functions"""
        # Register OHLCV aggregation
        self.data_aggregator.register_aggregation_function(
            "ohlcv",
            lambda data_list: {
                "open": data_list[0]["open"],
                "high": max(item["high"] for item in data_list),
                "low": min(item["low"] for item in data_list),
                "close": data_list[-1]["close"],
                "volume": sum(item["volume"] for item in data_list),
                "timestamp": data_list[-1]["timestamp"]
            }
        )
        
        # Register ticker aggregation
        self.data_aggregator.register_aggregation_function(
            "ticker",
            lambda data_list: data_list[-1]  # Just use the latest ticker
        )
        
        # Register trade aggregation
        self.data_aggregator.register_aggregation_function(
            "trades",
            lambda data_list: {
                "count": len(data_list),
                "volume": sum(item["amount"] for item in data_list),
                "value": sum(item["amount"] * item["price"] for item in data_list),
                "avg_price": sum(item["price"] for item in data_list) / len(data_list),
                "first": data_list[0],
                "last": data_list[-1]
            }
        )
    
    def _register_common_batch_processors(self):
        """Register common batch processors"""
        # Register order batch processor
        self.batch_processor.register_batch_processor(
            "orders",
            lambda items: logger.info(f"Processing {len(items)} orders")
        )
        
        # Register signal batch processor
        self.batch_processor.register_batch_processor(
            "signals",
            lambda items: logger.info(f"Processing {len(items)} signals")
        )
        
        # Register log batch processor
        self.batch_processor.register_batch_processor(
            "logs",
            lambda items: logger.info(f"Processing {len(items)} log entries")
        )
    
    def time_operation(self, operation: str):
        """Decorator for timing operations
        
        Args:
            operation: Operation name
            
        Returns:
            Decorator function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    execution_time = time.time() - start_time
                    self.metrics.record_execution_time(operation, execution_time)
                    
                    if execution_time > 1.0:
                        logger.warning(f"{operation} took {execution_time:.2f}s to execute")
                    
                    return result
                
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.metrics.record_execution_time(operation, execution_time)
                    
                    raise e
            
            return wrapper
        
        return decorator
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        start_mem = df.memory_usage().sum() / 1024**2
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int']).columns:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)
        
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = df[col].astype(np.float32)
        
        # Optimize object columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # If column has low cardinality
                df[col] = df[col].astype('category')
        
        end_mem = df.memory_usage().sum() / 1024**2
        reduction = (start_mem - end_mem) / start_mem
        
        logger.info(f"DataFrame memory reduced from {start_mem:.2f}MB to {end_mem:.2f}MB ({reduction:.2%})")
        
        return df
    
    def optimize_dict_list(self, data_list: List[Dict]) -> List[Dict]:
        """Optimize list of dictionaries
        
        Args:
            data_list: List of dictionaries to optimize
            
        Returns:
            Optimized list of dictionaries
        """
        if not data_list:
            return data_list
        
        # Convert to DataFrame for optimization
        df = pd.DataFrame(data_list)
        
        # Optimize DataFrame
        df = self.optimize_dataframe(df)
        
        # Convert back to list of dictionaries
        return df.to_dict('records')
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary
        
        Returns:
            Dictionary with performance summary
        """
        return {
            "metrics": self.metrics.get_metrics_summary(),
            "data_aggregator": {
                key: {
                    "buffer_size": self.data_aggregator.get_buffer_size(key),
                    "time_since_last_aggregation": self.data_aggregator.get_time_since_last_aggregation(key)
                }
                for key in self.data_aggregator.data_buffers.keys()
            },
            "batch_processor": {
                key: {
                    "batch_size": self.batch_processor.get_batch_size(key),
                    "time_since_last_batch": self.batch_processor.get_time_since_last_batch(key)
                }
                for key in self.batch_processor.batch_queues.keys()
            }
        }
    
    def shutdown(self):
        """Shutdown performance optimizer"""
        self.batch_processor.shutdown()
        logger.info("PerformanceOptimizer shutdown")

# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()

def time_operation(operation: str):
    """Global decorator for timing operations
    
    Args:
        operation: Operation name
        
    Returns:
        Decorator function
    """
    return performance_optimizer.time_operation(operation)

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Global function for optimizing DataFrame memory usage
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Optimized DataFrame
    """
    return performance_optimizer.optimize_dataframe(df)

def optimize_dict_list(data_list: List[Dict]) -> List[Dict]:
    """Global function for optimizing list of dictionaries
    
    Args:
        data_list: List of dictionaries to optimize
        
    Returns:
        Optimized list of dictionaries
    """
    return performance_optimizer.optimize_dict_list(data_list)

def get_performance_summary() -> Dict:
    """Global function for getting performance summary
    
    Returns:
        Dictionary with performance summary
    """
    return performance_optimizer.get_performance_summary()

def shutdown():
    """Global function for shutting down performance optimizer"""
    performance_optimizer.shutdown()

if __name__ == "__main__":
    # Example usage
    @time_operation("example_operation")
    def example_function(n):
        result = 0
        for i in range(n):
            result += i
        return result
    
    # Run example function
    example_function(1000000)
    
    # Get performance summary
    summary = get_performance_summary()
    print(f"Performance summary: {summary}")
    
    # Shutdown
    shutdown()
