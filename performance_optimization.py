#!/usr/bin/env python
"""
Performance Optimization for High-Frequency Trading Visualization

This module provides optimizations for the Trading-Agent visualization system
to handle high-frequency trading data efficiently, including data streaming,
efficient caching, and UI performance enhancements.
"""

import os
import json
import time
import threading
import queue
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable
import logging
from collections import deque

# Import error handling and logging utilities
try:
    from error_handling_and_logging import LoggerFactory, log_execution_time, PerformanceMonitor
except ImportError:
    # Fallback if error_handling_and_logging is not available
    from logging import getLogger as LoggerFactory
    
    def log_execution_time(logger=None, level='DEBUG'):
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    class PerformanceMonitor:
        def __init__(self, logger=None):
            pass
        
        def start(self, operation_name):
            pass
        
        def end(self, operation_name, log_level='DEBUG'):
            return 0

# Configure logging
logger = LoggerFactory.get_logger(
    'performance_optimization',
    log_level='INFO',
    log_file='performance_optimization.log'
)

class DataStreamManager:
    """Manager for efficient data streaming"""
    
    def __init__(self, buffer_size=1000):
        """Initialize data stream manager
        
        Args:
            buffer_size: Size of data buffer
        """
        self.buffer_size = buffer_size
        self.data_buffers = {}
        self.subscribers = {}
        self.lock = threading.RLock()
        self.performance_monitor = PerformanceMonitor(logger=logger)
        
        logger.info(f"Initialized DataStreamManager with buffer size {buffer_size}")
    
    def create_stream(self, stream_id):
        """Create data stream
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            bool: True if created, False if already exists
        """
        with self.lock:
            if stream_id in self.data_buffers:
                logger.warning(f"Stream already exists: {stream_id}")
                return False
            
            self.data_buffers[stream_id] = deque(maxlen=self.buffer_size)
            self.subscribers[stream_id] = []
            
            logger.info(f"Created stream: {stream_id}")
            return True
    
    def delete_stream(self, stream_id):
        """Delete data stream
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            bool: True if deleted, False if not found
        """
        with self.lock:
            if stream_id not in self.data_buffers:
                logger.warning(f"Stream not found: {stream_id}")
                return False
            
            del self.data_buffers[stream_id]
            del self.subscribers[stream_id]
            
            logger.info(f"Deleted stream: {stream_id}")
            return True
    
    def push_data(self, stream_id, data):
        """Push data to stream
        
        Args:
            stream_id: Stream identifier
            data: Data to push
            
        Returns:
            bool: True if pushed, False if stream not found
        """
        self.performance_monitor.start(f"push_data_{stream_id}")
        
        with self.lock:
            if stream_id not in self.data_buffers:
                logger.warning(f"Stream not found: {stream_id}")
                self.performance_monitor.end(f"push_data_{stream_id}")
                return False
            
            # Add data to buffer
            self.data_buffers[stream_id].append(data)
            
            # Notify subscribers
            for callback in self.subscribers[stream_id]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {str(e)}")
        
        self.performance_monitor.end(f"push_data_{stream_id}")
        return True
    
    def get_data(self, stream_id, limit=None):
        """Get data from stream
        
        Args:
            stream_id: Stream identifier
            limit: Maximum number of items to return
            
        Returns:
            list: Data items
        """
        self.performance_monitor.start(f"get_data_{stream_id}")
        
        with self.lock:
            if stream_id not in self.data_buffers:
                logger.warning(f"Stream not found: {stream_id}")
                self.performance_monitor.end(f"get_data_{stream_id}")
                return []
            
            if limit is None:
                data = list(self.data_buffers[stream_id])
            else:
                data = list(self.data_buffers[stream_id])[-limit:]
        
        self.performance_monitor.end(f"get_data_{stream_id}")
        return data
    
    def subscribe(self, stream_id, callback):
        """Subscribe to stream
        
        Args:
            stream_id: Stream identifier
            callback: Callback function
            
        Returns:
            bool: True if subscribed, False if stream not found
        """
        with self.lock:
            if stream_id not in self.subscribers:
                logger.warning(f"Stream not found: {stream_id}")
                return False
            
            self.subscribers[stream_id].append(callback)
            
            logger.info(f"Subscribed to stream: {stream_id}")
            return True
    
    def unsubscribe(self, stream_id, callback):
        """Unsubscribe from stream
        
        Args:
            stream_id: Stream identifier
            callback: Callback function
            
        Returns:
            bool: True if unsubscribed, False if not found
        """
        with self.lock:
            if stream_id not in self.subscribers:
                logger.warning(f"Stream not found: {stream_id}")
                return False
            
            try:
                self.subscribers[stream_id].remove(callback)
                logger.info(f"Unsubscribed from stream: {stream_id}")
                return True
            except ValueError:
                logger.warning(f"Callback not found in stream: {stream_id}")
                return False

class OptimizedDataCache:
    """Optimized data cache for high-frequency trading"""
    
    def __init__(self, max_items=1000, ttl=60):
        """Initialize optimized data cache
        
        Args:
            max_items: Maximum number of items in cache
            ttl: Time to live in seconds
        """
        self.max_items = max_items
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
        self.lock = threading.RLock()
        self.performance_monitor = PerformanceMonitor(logger=logger)
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"Initialized OptimizedDataCache with max_items={max_items}, ttl={ttl}")
    
    def _cleanup_loop(self):
        """Cleanup expired items periodically"""
        while True:
            try:
                self._cleanup()
                time.sleep(self.ttl / 2)  # Run cleanup at half the TTL
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}")
                time.sleep(5)  # Sleep on error
    
    def _cleanup(self):
        """Cleanup expired items"""
        self.performance_monitor.start("cache_cleanup")
        
        with self.lock:
            current_time = time.time()
            expired_keys = []
            
            # Find expired keys
            for key, timestamp in self.timestamps.items():
                if current_time - timestamp > self.ttl:
                    expired_keys.append(key)
            
            # Remove expired items
            for key in expired_keys:
                del self.cache[key]
                del self.timestamps[key]
            
            # If still over max_items, remove oldest
            if len(self.cache) > self.max_items:
                # Sort keys by timestamp
                sorted_keys = sorted(self.timestamps.items(), key=lambda x: x[1])
                
                # Remove oldest items
                for key, _ in sorted_keys[:len(self.cache) - self.max_items]:
                    del self.cache[key]
                    del self.timestamps[key]
        
        self.performance_monitor.end("cache_cleanup")
    
    def get(self, key, default=None):
        """Get item from cache
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Any: Cached value or default
        """
        self.performance_monitor.start("cache_get")
        
        with self.lock:
            if key in self.cache:
                # Update timestamp
                self.timestamps[key] = time.time()
                value = self.cache[key]
                self.performance_monitor.end("cache_get")
                return value
        
        self.performance_monitor.end("cache_get")
        return default
    
    def set(self, key, value):
        """Set item in cache
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            None
        """
        self.performance_monitor.start("cache_set")
        
        with self.lock:
            self.cache[key] = value
            self.timestamps[key] = time.time()
        
        self.performance_monitor.end("cache_set")
    
    def delete(self, key):
        """Delete item from cache
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if deleted, False if not found
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]
                return True
            return False
    
    def clear(self):
        """Clear cache
        
        Returns:
            None
        """
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def get_stats(self):
        """Get cache statistics
        
        Returns:
            dict: Cache statistics
        """
        with self.lock:
            return {
                'size': len(self.cache),
                'max_items': self.max_items,
                'ttl': self.ttl,
                'oldest_item_age': time.time() - min(self.timestamps.values()) if self.timestamps else 0,
                'newest_item_age': time.time() - max(self.timestamps.values()) if self.timestamps else 0
            }

class BatchProcessor:
    """Batch processor for efficient data processing"""
    
    def __init__(self, processor_func, batch_size=100, max_delay=1.0):
        """Initialize batch processor
        
        Args:
            processor_func: Function to process batches
            batch_size: Maximum batch size
            max_delay: Maximum delay before processing
        """
        self.processor_func = processor_func
        self.batch_size = batch_size
        self.max_delay = max_delay
        self.queue = queue.Queue()
        self.lock = threading.RLock()
        self.last_process_time = time.time()
        self.processing = False
        self.performance_monitor = PerformanceMonitor(logger=logger)
        
        # Start processor thread
        self.processor_thread = threading.Thread(target=self._processor_loop, daemon=True)
        self.processor_thread.start()
        
        logger.info(f"Initialized BatchProcessor with batch_size={batch_size}, max_delay={max_delay}")
    
    def _processor_loop(self):
        """Process batches periodically"""
        while True:
            try:
                self._process_batch()
                time.sleep(0.01)  # Small sleep to prevent CPU hogging
            except Exception as e:
                logger.error(f"Error in processor loop: {str(e)}")
                time.sleep(1)  # Sleep on error
    
    def _process_batch(self):
        """Process a batch of items"""
        current_time = time.time()
        time_since_last = current_time - self.last_process_time
        
        # Process if queue size reaches batch_size or max_delay elapsed
        if self.queue.qsize() >= self.batch_size or (self.queue.qsize() > 0 and time_since_last >= self.max_delay):
            with self.lock:
                if self.processing:
                    return
                self.processing = True
            
            try:
                self.performance_monitor.start("batch_processing")
                
                # Get items from queue
                batch = []
                while len(batch) < self.batch_size and not self.queue.empty():
                    batch.append(self.queue.get())
                
                # Process batch
                if batch:
                    self.processor_func(batch)
                    self.last_process_time = time.time()
                
                self.performance_monitor.end("batch_processing")
            
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
            
            finally:
                with self.lock:
                    self.processing = False
    
    def add_item(self, item):
        """Add item to batch
        
        Args:
            item: Item to add
            
        Returns:
            None
        """
        self.queue.put(item)
    
    def get_stats(self):
        """Get processor statistics
        
        Returns:
            dict: Processor statistics
        """
        return {
            'queue_size': self.queue.qsize(),
            'batch_size': self.batch_size,
            'max_delay': self.max_delay,
            'time_since_last_process': time.time() - self.last_process_time,
            'processing': self.processing
        }

class DataAggregator:
    """Data aggregator for efficient data aggregation"""
    
    def __init__(self, window_size=100):
        """Initialize data aggregator
        
        Args:
            window_size: Size of aggregation window
        """
        self.window_size = window_size
        self.data_windows = {}
        self.lock = threading.RLock()
        self.performance_monitor = PerformanceMonitor(logger=logger)
        
        logger.info(f"Initialized DataAggregator with window_size={window_size}")
    
    def add_data_point(self, key, data_point):
        """Add data point to aggregation window
        
        Args:
            key: Aggregation key
            data_point: Data point to add
            
        Returns:
            None
        """
        self.performance_monitor.start(f"add_data_point_{key}")
        
        with self.lock:
            if key not in self.data_windows:
                self.data_windows[key] = deque(maxlen=self.window_size)
            
            self.data_windows[key].append(data_point)
        
        self.performance_monitor.end(f"add_data_point_{key}")
    
    def get_window(self, key):
        """Get aggregation window
        
        Args:
            key: Aggregation key
            
        Returns:
            list: Aggregation window
        """
        with self.lock:
            if key not in self.data_windows:
                return []
            
            return list(self.data_windows[key])
    
    def get_aggregated_value(self, key, aggregation_func):
        """Get aggregated value
        
        Args:
            key: Aggregation key
            aggregation_func: Aggregation function
            
        Returns:
            Any: Aggregated value
        """
        self.performance_monitor.start(f"get_aggregated_value_{key}")
        
        with self.lock:
            if key not in self.data_windows or not self.data_windows[key]:
                self.performance_monitor.end(f"get_aggregated_value_{key}")
                return None
            
            try:
                result = aggregation_func(self.data_windows[key])
                self.performance_monitor.end(f"get_aggregated_value_{key}")
                return result
            except Exception as e:
                logger.error(f"Error in aggregation function: {str(e)}")
                self.performance_monitor.end(f"get_aggregated_value_{key}")
                return None
    
    def clear_window(self, key):
        """Clear aggregation window
        
        Args:
            key: Aggregation key
            
        Returns:
            bool: True if cleared, False if not found
        """
        with self.lock:
            if key in self.data_windows:
                self.data_windows[key].clear()
                return True
            return False
    
    def get_stats(self):
        """Get aggregator statistics
        
        Returns:
            dict: Aggregator statistics
        """
        with self.lock:
            return {
                'num_windows': len(self.data_windows),
                'window_size': self.window_size,
                'window_keys': list(self.data_windows.keys()),
                'window_lengths': {key: len(window) for key, window in self.data_windows.items()}
            }

class OptimizedChartDataProcessor:
    """Optimized chart data processor for high-frequency trading"""
    
    def __init__(self, max_points=1000, downsampling_threshold=5000):
        """Initialize optimized chart data processor
        
        Args:
            max_points: Maximum number of data points
            downsampling_threshold: Threshold for downsampling
        """
        self.max_points = max_points
        self.downsampling_threshold = downsampling_threshold
        self.performance_monitor = PerformanceMonitor(logger=logger)
        
        logger.info(f"Initialized OptimizedChartDataProcessor with max_points={max_points}, downsampling_threshold={downsampling_threshold}")
    
    @log_execution_time(logger=logger)
    def process_klines(self, klines):
        """Process klines data for chart
        
        Args:
            klines: Klines data
            
        Returns:
            list: Processed klines data
        """
        if not klines:
            return []
        
        # Convert to DataFrame for efficient processing
        df = pd.DataFrame(klines)
        
        # Apply downsampling if needed
        if len(df) > self.downsampling_threshold:
            df = self._downsample_klines(df)
        
        # Limit to max_points
        if len(df) > self.max_points:
            df = df.iloc[-self.max_points:]
        
        # Convert back to list of dicts
        return df.to_dict('records')
    
    def _downsample_klines(self, df):
        """Downsample klines data
        
        Args:
            df: Klines DataFrame
            
        Returns:
            DataFrame: Downsampled klines
        """
        self.performance_monitor.start("downsample_klines")
        
        # Calculate target number of points
        target_points = self.max_points
        
        # Calculate sampling factor
        sampling_factor = max(1, len(df) // target_points)
        
        # Apply OHLC downsampling
        result = pd.DataFrame()
        
        # Group by sampling factor
        groups = df.groupby(np.arange(len(df)) // sampling_factor)
        
        # Aggregate OHLC data
        result['time'] = groups['time'].first()
        result['open'] = groups['open'].first()
        result['high'] = groups['high'].max()
        result['low'] = groups['low'].min()
        result['close'] = groups['close'].last()
        result['volume'] = groups['volume'].sum()
        
        # Add other columns if present
        for col in df.columns:
            if col not in result.columns and col not in ['time', 'open', 'high', 'low', 'close', 'volume']:
                result[col] = groups[col].last()
        
        self.performance_monitor.end("downsample_klines")
        
        return result
    
    @log_execution_time(logger=logger)
    def process_indicator_data(self, indicator_data, klines):
        """Process indicator data for chart
        
        Args:
            indicator_data: Indicator data
            klines: Klines data
            
        Returns:
            dict: Processed indicator data
        """
        if not indicator_data or not klines:
            return {}
        
        self.performance_monitor.start("process_indicator_data")
        
        processed_data = {}
        
        for indicator_name, data in indicator_data.items():
            # Skip if data is invalid
            if not data or 'name' not in data:
                continue
            
            # Process based on indicator type
            if indicator_name == 'RSI':
                processed_data[indicator_name] = self._process_rsi_data(data, klines)
            elif indicator_name == 'MACD':
                processed_data[indicator_name] = self._process_macd_data(data, klines)
            elif indicator_name == 'BollingerBands':
                processed_data[indicator_name] = self._process_bollinger_data(data, klines)
            elif indicator_name == 'Volume':
                processed_data[indicator_name] = self._process_volume_data(data, klines)
            else:
                # Generic processing for other indicators
                processed_data[indicator_name] = data
        
        self.performance_monitor.end("process_indicator_data")
        
        return processed_data
    
    def _process_rsi_data(self, data, klines):
        """Process RSI data
        
        Args:
            data: RSI data
            klines: Klines data
            
        Returns:
            dict: Processed RSI data
        """
        # Ensure data length matches klines
        values = data.get('values', [])
        
        if len(values) > len(klines):
            values = values[-len(klines):]
        elif len(values) < len(klines):
            # Pad with None
            values = [None] * (len(klines) - len(values)) + values
        
        return {
            'name': data.get('name', 'RSI'),
            'values': values,
            'overbought': data.get('overbought', 70),
            'oversold': data.get('oversold', 30)
        }
    
    def _process_macd_data(self, data, klines):
        """Process MACD data
        
        Args:
            data: MACD data
            klines: Klines data
            
        Returns:
            dict: Processed MACD data
        """
        # Ensure data length matches klines
        macd = data.get('macd', [])
        signal = data.get('signal', [])
        histogram = data.get('histogram', [])
        
        if len(macd) > len(klines):
            macd = macd[-len(klines):]
        elif len(macd) < len(klines):
            # Pad with None
            macd = [None] * (len(klines) - len(macd)) + macd
        
        if len(signal) > len(klines):
            signal = signal[-len(klines):]
        elif len(signal) < len(klines):
            # Pad with None
            signal = [None] * (len(klines) - len(signal)) + signal
        
        if len(histogram) > len(klines):
            histogram = histogram[-len(klines):]
        elif len(histogram) < len(klines):
            # Pad with None
            histogram = [None] * (len(klines) - len(histogram)) + histogram
        
        return {
            'name': data.get('name', 'MACD'),
            'macd': macd,
            'signal': signal,
            'histogram': histogram
        }
    
    def _process_bollinger_data(self, data, klines):
        """Process Bollinger Bands data
        
        Args:
            data: Bollinger Bands data
            klines: Klines data
            
        Returns:
            dict: Processed Bollinger Bands data
        """
        # Ensure data length matches klines
        middle = data.get('middle', [])
        upper = data.get('upper', [])
        lower = data.get('lower', [])
        
        if len(middle) > len(klines):
            middle = middle[-len(klines):]
        elif len(middle) < len(klines):
            # Pad with None
            middle = [None] * (len(klines) - len(middle)) + middle
        
        if len(upper) > len(klines):
            upper = upper[-len(klines):]
        elif len(upper) < len(klines):
            # Pad with None
            upper = [None] * (len(klines) - len(upper)) + upper
        
        if len(lower) > len(klines):
            lower = lower[-len(klines):]
        elif len(lower) < len(klines):
            # Pad with None
            lower = [None] * (len(klines) - len(lower)) + lower
        
        return {
            'name': data.get('name', 'BollingerBands'),
            'middle': middle,
            'upper': upper,
            'lower': lower
        }
    
    def _process_volume_data(self, data, klines):
        """Process Volume data
        
        Args:
            data: Volume data
            klines: Klines data
            
        Returns:
            dict: Processed Volume data
        """
        # Ensure data length matches klines
        values = data.get('values', [])
        ma = data.get('ma', [])
        
        if len(values) > len(klines):
            values = values[-len(klines):]
        elif len(values) < len(klines):
            # Pad with None
            values = [None] * (len(klines) - len(values)) + values
        
        if len(ma) > len(klines):
            ma = ma[-len(klines):]
        elif len(ma) < len(klines):
            # Pad with None
            ma = [None] * (len(klines) - len(ma)) + ma
        
        return {
            'name': data.get('name', 'Volume'),
            'values': values,
            'ma': ma
        }

class UIUpdateOptimizer:
    """Optimizer for UI updates in high-frequency trading"""
    
    def __init__(self, min_update_interval=0.1, batch_updates=True):
        """Initialize UI update optimizer
        
        Args:
            min_update_interval: Minimum interval between updates
            batch_updates: Whether to batch updates
        """
        self.min_update_interval = min_update_interval
        self.batch_updates = batch_updates
        self.last_update_time = {}
        self.pending_updates = {}
        self.lock = threading.RLock()
        self.performance_monitor = PerformanceMonitor(logger=logger)
        
        logger.info(f"Initialized UIUpdateOptimizer with min_update_interval={min_update_interval}, batch_updates={batch_updates}")
    
    def should_update(self, component_id):
        """Check if component should be updated
        
        Args:
            component_id: Component identifier
            
        Returns:
            bool: True if should update, False otherwise
        """
        current_time = time.time()
        
        with self.lock:
            if component_id not in self.last_update_time:
                self.last_update_time[component_id] = 0
            
            time_since_last = current_time - self.last_update_time[component_id]
            
            if time_since_last >= self.min_update_interval:
                self.last_update_time[component_id] = current_time
                return True
            
            return False
    
    def queue_update(self, component_id, update_data):
        """Queue update for component
        
        Args:
            component_id: Component identifier
            update_data: Update data
            
        Returns:
            bool: True if queued, False if should update immediately
        """
        if not self.batch_updates:
            return False
        
        current_time = time.time()
        
        with self.lock:
            if component_id not in self.last_update_time:
                self.last_update_time[component_id] = 0
            
            time_since_last = current_time - self.last_update_time[component_id]
            
            if time_since_last >= self.min_update_interval:
                self.last_update_time[component_id] = current_time
                return False
            
            if component_id not in self.pending_updates:
                self.pending_updates[component_id] = []
            
            self.pending_updates[component_id].append(update_data)
            return True
    
    def get_pending_updates(self, component_id):
        """Get pending updates for component
        
        Args:
            component_id: Component identifier
            
        Returns:
            list: Pending updates
        """
        with self.lock:
            if component_id not in self.pending_updates:
                return []
            
            updates = self.pending_updates[component_id]
            self.pending_updates[component_id] = []
            return updates
    
    def get_stats(self):
        """Get optimizer statistics
        
        Returns:
            dict: Optimizer statistics
        """
        with self.lock:
            return {
                'min_update_interval': self.min_update_interval,
                'batch_updates': self.batch_updates,
                'components': list(self.last_update_time.keys()),
                'last_update_times': {k: time.time() - v for k, v in self.last_update_time.items()},
                'pending_updates': {k: len(v) for k, v in self.pending_updates.items()}
            }

class WebSocketOptimizer:
    """Optimizer for WebSocket communication in high-frequency trading"""
    
    def __init__(self, compression=True, batch_size=10, max_delay=0.1):
        """Initialize WebSocket optimizer
        
        Args:
            compression: Whether to compress messages
            batch_size: Maximum batch size
            max_delay: Maximum delay before sending
        """
        self.compression = compression
        self.batch_size = batch_size
        self.max_delay = max_delay
        self.message_queues = {}
        self.locks = {}
        self.last_send_times = {}
        self.performance_monitor = PerformanceMonitor(logger=logger)
        
        logger.info(f"Initialized WebSocketOptimizer with compression={compression}, batch_size={batch_size}, max_delay={max_delay}")
    
    def initialize_connection(self, connection_id):
        """Initialize connection
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            bool: True if initialized, False if already exists
        """
        with self._get_lock(connection_id):
            if connection_id in self.message_queues:
                return False
            
            self.message_queues[connection_id] = []
            self.last_send_times[connection_id] = 0
            return True
    
    def _get_lock(self, connection_id):
        """Get lock for connection
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            threading.RLock: Lock
        """
        if connection_id not in self.locks:
            self.locks[connection_id] = threading.RLock()
        
        return self.locks[connection_id]
    
    def queue_message(self, connection_id, message):
        """Queue message for sending
        
        Args:
            connection_id: Connection identifier
            message: Message to send
            
        Returns:
            bool: True if queued, False if should send immediately
        """
        current_time = time.time()
        
        with self._get_lock(connection_id):
            if connection_id not in self.message_queues:
                self.initialize_connection(connection_id)
            
            # Add message to queue
            self.message_queues[connection_id].append(message)
            
            # Check if should send immediately
            queue_size = len(self.message_queues[connection_id])
            time_since_last = current_time - self.last_send_times.get(connection_id, 0)
            
            if queue_size >= self.batch_size or time_since_last >= self.max_delay:
                return False
            
            return True
    
    def get_pending_messages(self, connection_id):
        """Get pending messages for connection
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            list: Pending messages
        """
        with self._get_lock(connection_id):
            if connection_id not in self.message_queues:
                return []
            
            messages = self.message_queues[connection_id]
            self.message_queues[connection_id] = []
            self.last_send_times[connection_id] = time.time()
            
            return messages
    
    def optimize_message(self, message):
        """Optimize message for sending
        
        Args:
            message: Message to optimize
            
        Returns:
            Any: Optimized message
        """
        self.performance_monitor.start("optimize_message")
        
        # Convert to JSON if not already
        if not isinstance(message, str):
            message = json.dumps(message)
        
        # Compress if enabled
        if self.compression:
            import zlib
            message = zlib.compress(message.encode('utf-8'))
        
        self.performance_monitor.end("optimize_message")
        return message
    
    def deoptimize_message(self, message):
        """Deoptimize received message
        
        Args:
            message: Message to deoptimize
            
        Returns:
            Any: Deoptimized message
        """
        self.performance_monitor.start("deoptimize_message")
        
        # Decompress if compressed
        if self.compression and isinstance(message, bytes):
            import zlib
            try:
                message = zlib.decompress(message).decode('utf-8')
            except:
                # If decompression fails, assume it's not compressed
                if isinstance(message, bytes):
                    message = message.decode('utf-8')
        elif isinstance(message, bytes):
            message = message.decode('utf-8')
        
        # Parse JSON if string
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except:
                # If parsing fails, return as is
                pass
        
        self.performance_monitor.end("deoptimize_message")
        return message
    
    def get_stats(self):
        """Get optimizer statistics
        
        Returns:
            dict: Optimizer statistics
        """
        stats = {
            'compression': self.compression,
            'batch_size': self.batch_size,
            'max_delay': self.max_delay,
            'connections': {}
        }
        
        for connection_id in self.message_queues:
            with self._get_lock(connection_id):
                stats['connections'][connection_id] = {
                    'queue_size': len(self.message_queues[connection_id]),
                    'time_since_last_send': time.time() - self.last_send_times.get(connection_id, 0)
                }
        
        return stats

# Example usage
if __name__ == "__main__":
    # Create data stream manager
    stream_manager = DataStreamManager()
    
    # Create streams
    stream_manager.create_stream("klines_btc_usdc")
    stream_manager.create_stream("trades_btc_usdc")
    
    # Create optimized data cache
    data_cache = OptimizedDataCache()
    
    # Create batch processor
    def process_batch(batch):
        print(f"Processing batch of {len(batch)} items")
    
    batch_processor = BatchProcessor(process_batch)
    
    # Create data aggregator
    data_aggregator = DataAggregator()
    
    # Create chart data processor
    chart_processor = OptimizedChartDataProcessor()
    
    # Create UI update optimizer
    ui_optimizer = UIUpdateOptimizer()
    
    # Create WebSocket optimizer
    ws_optimizer = WebSocketOptimizer()
    
    # Example data
    kline_data = {
        "time": 1622505600000,
        "open": 35000.0,
        "high": 36000.0,
        "low": 34500.0,
        "close": 35500.0,
        "volume": 100.0
    }
    
    # Push data to stream
    stream_manager.push_data("klines_btc_usdc", kline_data)
    
    # Cache data
    data_cache.set("latest_kline_btc_usdc", kline_data)
    
    # Add item to batch processor
    batch_processor.add_item(kline_data)
    
    # Add data point to aggregator
    data_aggregator.add_data_point("btc_usdc_price", kline_data["close"])
    
    # Get aggregated value
    avg_price = data_aggregator.get_aggregated_value(
        "btc_usdc_price",
        lambda window: sum(p for p in window) / len(window)
    )
    
    print(f"Average price: {avg_price}")
    
    # Check if UI should update
    if ui_optimizer.should_update("price_chart"):
        print("Updating price chart")
    
    # Queue WebSocket message
    ws_optimizer.queue_message("client1", {"type": "kline", "data": kline_data})
    
    # Get pending messages
    pending_messages = ws_optimizer.get_pending_messages("client1")
    
    print(f"Pending messages: {len(pending_messages)}")
    
    # Get statistics
    print(f"Stream manager stats: {stream_manager.get_data('klines_btc_usdc')}")
    print(f"Data cache stats: {data_cache.get_stats()}")
    print(f"Batch processor stats: {batch_processor.get_stats()}")
    print(f"Data aggregator stats: {data_aggregator.get_stats()}")
    print(f"UI optimizer stats: {ui_optimizer.get_stats()}")
    print(f"WebSocket optimizer stats: {ws_optimizer.get_stats()}")
